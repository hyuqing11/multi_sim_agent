from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent as create_langgraph_react_agent

from backend.agents.llm import get_model, settings


def _call_web_search(query: str) -> str:
    """Helper to reuse the global web search tool without circular imports."""
    if not query:
        return "Empty query provided to web search."

    try:
        from backend.agents.library.chatbot import web_search as chatbot_web_search
    except Exception as exc:  # pragma: no cover - defensive guard
        return f"Web search unavailable: {exc}"

    try:
        if hasattr(chatbot_web_search, "invoke"):
            return chatbot_web_search.invoke(query)
        if callable(chatbot_web_search):
            return chatbot_web_search(query)
        return "Web search tool is not callable."
    except Exception as exc:  # pragma: no cover - to protect orchestrator flow
        return f"Web search failed for '{query}': {exc}"


def _parse_incar_parameters(incar_path: Path) -> dict[str, Any]:
    """Parse INCAR parameters into a structured dictionary."""
    parameters: dict[str, Any] = {}
    if not incar_path.exists():
        return parameters

    content = incar_path.read_text()
    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.split("#")[0].strip()

        try:
            if value.upper() in {".TRUE.", "T"}:
                parameters[key] = True
            elif value.upper() in {".FALSE.", "F"}:
                parameters[key] = False
            elif "." in value:
                parameters[key] = float(value)
            else:
                parameters[key] = int(value)
        except ValueError:
            parameters[key] = value

    return parameters


def read_vasp_output_tool(
        filepath: str,
        max_lines: int = 500,
        search_term: Optional[str] = None
) -> str:
    """
    Read VASP output files intelligently.

    Args:
        filepath: Path to file (OUTCAR, stdout, stderr, etc.)
        max_lines: Maximum lines to return (from end of file)
        search_term: Optional search term to filter relevant sections

    Returns:
        Relevant content from the file
    """
    try:
        file_path = Path(filepath)
        if not file_path.exists():
            return f"Error: File {filepath} not found"

        content = file_path.read_text(encoding="utf-8", errors="ignore")
        lines = content.split("\n")

        if search_term:
            # Find lines containing search term with context
            matches = []
            for i, line in enumerate(lines):
                if search_term.lower() in line.lower():
                    # Include 5 lines before and after
                    start = max(0, i - 5)
                    end = min(len(lines), i + 6)
                    context = "\n".join(lines[start:end])
                    matches.append(f"--- Match at line {i} ---\n{context}\n")

            if matches:
                return "\n".join(matches[:10])  # Return first 10 matches
            else:
                return f"No matches found for '{search_term}' in {filepath}"

        # Return last N lines (most recent output)
        relevant_lines = lines[-max_lines:]

        # Also include any ERROR/WARNING sections
        error_sections = []
        for i, line in enumerate(lines):
            if any(keyword in line.upper() for keyword in ["ERROR", "WARNING", "FATAL", "FAILED"]):
                start = max(0, i - 3)
                end = min(len(lines), i + 4)
                error_sections.append("\n".join(lines[start:end]))

        result = []
        if error_sections:
            result.append("=== ERROR/WARNING SECTIONS ===")
            result.extend(error_sections[:5])  # First 5 error sections
            result.append("\n=== RECENT OUTPUT (last 200 lines) ===")
            result.append("\n".join(relevant_lines[-200:]))
        else:
            result.append("=== RECENT OUTPUT ===")
            result.append("\n".join(relevant_lines))

        return "\n".join(result)

    except Exception as e:
        return f"Error reading file {filepath}: {e}"


def search_vasp_documentation_tool(query: str) -> str:
    """
    Search VASP documentation and forums for solutions.

    Uses web search to find:
    - Official VASP wiki
    - VASP forum discussions
    - Materials science Stack Exchange
    - Research papers with solutions

    Args:
        query: Search query (error message or problem description)

    Returns:
        Relevant information from search results
    """
    search_query = f"VASP {query} site:vasp.at"
    return _call_web_search(search_query)


def read_current_parameters_tool(working_dir: str) -> str:
    """
    Read current VASP input parameters from INCAR.

    Args:
        working_dir: Working directory containing INCAR

    Returns:
        Current INCAR parameters as JSON
    """
    try:
        incar_path = Path(working_dir) / "INCAR"
        if not incar_path.exists():
            return "Error: INCAR file not found"

        params = _parse_incar_parameters(incar_path)
        return json.dumps(params, indent=2)

    except Exception as e:
        return f"Error reading INCAR: {e}"


def check_vasp_wiki_tool(error_keyword: str) -> str:
    """
    Check VASP wiki for specific error messages.

    Args:
        error_keyword: Error keyword or message

    Returns:
        Information from VASP wiki if available
    """
    wiki_query = f"{error_keyword} site:vasp.at/wiki"
    return _call_web_search(wiki_query)


def search_vasp_forum_tool(query: str) -> str:
    """
    Search VASP forum for similar issues and solutions.

    Args:
        query: Problem description or error message

    Returns:
        Forum discussions and solutions
    """
    forum_query = f"{query} site:vasp.at/forum OR site:matsci.org"
    return _call_web_search(forum_query)


# ============================================================================
# CREATE THE DEBUGGING AGENT
# ============================================================================

def create_vasp_debug_agent(use_internet: bool = True):
    """
    Create a ReAct agent that can dynamically debug VASP errors.

    The agent has access to:
    - File reading tools (OUTCAR, stderr, etc.)
    - Internet search (VASP docs, forums, Stack Overflow)
    - Current parameter inspection

    It can reason about errors and propose fixes dynamically.
    """

    # Define tools
    tools = [
        Tool(
            name="read_vasp_output",
            func=read_vasp_output_tool,
            description=(
                "Read VASP output files (OUTCAR, stdout, stderr). "
                "Args: filepath (str), max_lines (int, default 500), "
                "search_term (str, optional to find specific errors). "
                "Use this to examine what went wrong."
            )
        ),
        Tool(
            name="read_current_incar",
            func=read_current_parameters_tool,
            description=(
                "Read current VASP parameters from INCAR file. "
                "Args: working_dir (str). "
                "Use this to see what parameters were used."
            )
        ),
    ]

    if use_internet:
        tools.extend(
            [
                Tool(
                    name="search_vasp_docs",
                    func=search_vasp_documentation_tool,
                    description=(
                        "Search VASP documentation and online resources for error solutions. "
                        "Args: query (str) - the error message or problem. "
                        "Use this to find official documentation about errors."
                    ),
                ),
                Tool(
                    name="check_vasp_wiki",
                    func=check_vasp_wiki_tool,
                    description=(
                        "Check VASP wiki for specific error messages. "
                        "Args: error_keyword (str). "
                        "Use for known VASP errors."
                    ),
                ),
                Tool(
                    name="search_vasp_forum",
                    func=search_vasp_forum_tool,
                    description=(
                        "Search VASP forum for similar problems and solutions. "
                        "Args: query (str). "
                        "Use to find community solutions."
                    ),
                ),
            ]
        )

    # System prompt for the debugging agent
    system_prompt = """You are an expert VASP (Vienna Ab initio Simulation Package) debugger.

Your goal: Analyze VASP calculation errors and propose specific parameter fixes.

## Your Process:

1. **Examine the output files**
   - Use read_vasp_output to check OUTCAR for errors
   - Look for ERROR, WARNING, FATAL messages
   - Check the last iterations for convergence issues
   - Examine stderr if OUTCAR is incomplete

2. **Understand current settings**
   - Use read_current_incar to see what parameters were used
   - Identify potentially problematic settings

3. **Search for solutions**
   - Use search_vasp_docs for error messages
   - Check check_vasp_wiki for known issues
   - Use search_vasp_forum for community solutions
   - Search for the specific error text you found

4. **Reason about the problem**
   - Combine information from files and searches
   - Consider the system type (metal, insulator, molecule, etc.)
   - Think about common causes for this error

5. **Propose specific fixes**
   - Provide exact parameter changes (INCAR modifications)
   - Explain WHY each change helps
   - Prioritize fixes (try simple fixes first)
   - Consider alternative approaches if primary fix fails

## Output Format:

Return a JSON object with:
```json
{
  "error_diagnosis": "Description of what went wrong",
  "error_type": "Category (e.g., 'convergence', 'memory', 'setup')",
  "severity": "critical|high|medium|low",
  "parameter_adjustments": {
    "PARAM_NAME": "new_value"
  },
  "explanation": "Why these changes will help",
  "alternative_solutions": ["Other approaches to try"],
  "should_retry": true/false,
  "confidence": "high|medium|low"
}
```

Be specific, practical, and cite sources when available.
"""

    # Use LangGraph's create_react_agent for better control
    llm = get_model(settings.DEFAULT_MODEL)

    agent = create_langgraph_react_agent(
        llm,
        tools,
        state_modifier=system_prompt
    )

    return agent


async def dynamic_vasp_error_analysis(
        working_dir: str,
        current_params: dict[str, Any],
        use_internet: bool = True
) -> dict[str, Any]:
    """
    Dynamically analyze VASP errors using LLM + internet search.

    This is an alternative to the pattern-matching approach.
    More flexible but slower.

    Args:
        working_dir: Path to calculation directory
        current_params: Current INCAR parameters
        use_internet: Whether to use internet search

    Returns:
        Analysis with error diagnosis and parameter fixes
    """

    agent = create_vasp_debug_agent(use_internet=use_internet)

    working_path = Path(working_dir)
    if not current_params:
        incar_params = _parse_incar_parameters(working_path / "INCAR")
        if incar_params:
            current_params = incar_params

    # Prepare the query for the agent
    try:
        params_blob = json.dumps(current_params, indent=2)
    except TypeError:
        safe_params = {k: str(v) for k, v in current_params.items()}
        params_blob = json.dumps(safe_params, indent=2)

    query = f"""Analyze the VASP calculation in: {working_dir}

The calculation did not complete successfully. Please:

1. Read the OUTCAR file to identify errors
2. Check stderr if needed
3. Review the current INCAR parameters
4. Search online for solutions if the error is unfamiliar
5. Propose specific parameter changes to fix the issue

Current INCAR parameters:
{params_blob}

Provide your analysis in JSON format.
"""

    # Invoke the agent
    try:
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=query)]
        })

        # Extract the agent's response
        final_message = result["messages"][-1]
        content = final_message.content

        # Try to parse JSON from the response
        # LLMs often wrap JSON in ```json blocks
        json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        if json_match:
            analysis = json.loads(json_match.group(1))
        else:
            # Try to parse the entire content
            try:
                analysis = json.loads(content)
            except:
                # Fallback: create structured response from text
                analysis = {
                    "error_diagnosis": content,
                    "parameter_adjustments": {},
                    "should_retry": True,
                    "confidence": "medium"
                }

        return {
            "success": True,
            "analysis": analysis,
            "raw_response": content,
            "agent_messages": [str(m) for m in result["messages"]]
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "analysis": {
                "error_diagnosis": f"Agent failed: {e}",
                "should_retry": False
            }
        }


async def hybrid_vasp_error_analysis(
        working_dir: str,
        current_params: dict[str, Any]
) -> dict[str, Any]:
    """
    Hybrid approach: Try pattern matching first, fall back to LLM if needed.

    This gives best of both worlds:
    - Fast for common errors (pattern matching)
    - Intelligent for rare/complex errors (LLM + search)

    Args:
        working_dir: Calculation directory
        current_params: Current INCAR parameters

    Returns:
        Combined analysis
    """

    # Try pattern-based detection first (fast path)
    from .vasp_errors import analyze_vasp_errors

    pattern_analysis = analyze_vasp_errors(working_dir, current_params)

    recovery_plan = pattern_analysis.get("recovery_plan") or {}
    adjustments = recovery_plan.get("adjustments") or {}
    if "parameter_adjustments" not in recovery_plan:
        recovery_plan["parameter_adjustments"] = dict(adjustments)
    pattern_analysis["recovery_plan"] = recovery_plan

    # Check if we have high confidence in the pattern-based solution
    errors = pattern_analysis.get("errors", [])
    severity = pattern_analysis.get("severity", "none")

    # Use pattern-based solution for common, well-understood errors
    if errors and severity in ["medium", "low"]:
        # We know how to handle this
        return {
            "method": "pattern_matching",
            "confidence": "high",
            **pattern_analysis
        }

    # For critical/high severity or no detected errors, use dynamic analysis
    if severity in ["critical", "high"] or not errors:
        print("🔍 Using dynamic LLM analysis for complex error...")

        dynamic_result = await dynamic_vasp_error_analysis(
            working_dir,
            current_params,
            use_internet=True
        )

        if dynamic_result["success"]:
            analysis = dynamic_result["analysis"]

            # Merge with pattern-based findings if any
            if errors:
                analysis["pattern_detected_errors"] = [str(e) for e in errors]

            parameter_adjustments = analysis.get("parameter_adjustments", {})
            recovery_plan = {
                "adjustments": parameter_adjustments,
                "parameter_adjustments": parameter_adjustments,
                "explanation": analysis.get("explanation", ""),
                "alternatives": analysis.get("alternative_solutions", []),
            }
            return {
                "method": "dynamic_llm",
                "confidence": analysis.get("confidence", "medium"),
                "error_diagnosis": analysis.get("error_diagnosis"),
                "recovery_plan": recovery_plan,
                "should_retry": analysis.get("should_retry", True),
                "severity": analysis.get("severity", "unknown"),
                "raw_llm_response": dynamic_result.get("raw_response", "")
            }

    # Fallback to pattern-based
    return {
        "method": "pattern_matching",
        "confidence": "medium",
        **pattern_analysis
    }
