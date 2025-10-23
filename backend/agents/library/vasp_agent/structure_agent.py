"""
A robust, stateful LangGraph agent for the structure generation and validation workflow.

This agent guides the LLM through a 3-step process:
1. Generate (using execute_ase_script)
2. Quick Validate (using quick_validate_structure)
3. Full Review (using get_llm_validation_and_hint)

It uses a dynamic system prompt and a state-parsing node to ensure the
LLM completes each step in sequence.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Annotated, List, Literal, TypedDict

try:  # Optional dependency: prefer Anthropic when available
    from langchain_anthropic import ChatAnthropic  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    ChatAnthropic = None

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from backend.agents.llm import get_model, settings
from backend.agents.library.vasp_agent.utils import ENGINE_VASP
from .input_agent import prepare_structure_artifacts, summarize_structure
from .tools import execute_ase_script
from .structures_validation import TOOLS as VALIDATION_TOOLS
from .structures_validation import set_active_run_dir

# Load environment variables
load_dotenv()


# --- 1. Define Agent State ---

class StructureAgentState(TypedDict, total=False):
    """Defines the state for the structure generation and validation graph."""

    messages: Annotated[List[BaseMessage], add_messages]
    thread_id: str
    run_dir: str
    engine: str

    # The original user request
    query: str

    # State tracking for the workflow
    structure_filename: str
    generation_script_filename: str
    structure_source_path: str
    structure_artifact_filename: str
    structure_artifact_path: str
    structure_summary: str
    structure_prompt_blob: str
    structure_generation_error: str

    # We will store the raw JSON string output from validation
    quick_validation_result: str
    llm_validation_result: str

    # Plan data and parameters from planner
    plan_data: dict
    dft_parameters: dict


# --- 2. Define Tools ---

# Combine all tools for the LLM
all_tools = [execute_ase_script, *VALIDATION_TOOLS]


def _build_llm():
    """Prefer Anthropic if installed, otherwise fall back to configured default."""

    if ChatAnthropic is not None:
        env_model = os.getenv("STRUCTURE_AGENT_MODEL", "claude-sonnet-4-5-20250929")
        model_name = settings.resolve_anthropic_model_name(env_model) or env_model
        anthropic_kwargs = {
            "model": model_name,
            "temperature": 0,
            "max_tokens": 4096,
        }
        if settings.anthropic_betas:
            anthropic_kwargs["betas"] = settings.anthropic_betas
        return ChatAnthropic(**anthropic_kwargs)
    return get_model(settings.DEFAULT_MODEL)


llm_with_tools = _build_llm().bind_tools(all_tools)


# --- 3. Define Graph Nodes ---

def get_system_prompt(state: StructureAgentState) -> str:
    """Creates a dynamic system prompt based on the current workflow stage."""

    run_dir = state.get("run_dir", os.getcwd())
    query = state.get("query", "No query provided.")
    engine = state.get("engine").lower()

    if engine == "lammps":
        structure_requirement = "Save the final structure using `ase.io.write()` in LAMMPS data format (e.g., `structure_lammps.data`)."
    else:
        structure_requirement = "Save the final structure using `ase.io.write()` in VASP format (e.g., `POSCAR`)."

    # Build plan context if available
    plan_context = ""
    plan_data = state.get("plan_data", {})
    if isinstance(plan_data, dict) and plan_data.get("workflow_plan"):
        plan_context = f"\n\n**Workflow Plan from Planner:**\n{plan_data.get('workflow_plan')}\n"
        plan_context += "Consider this plan when generating the structure to ensure it aligns with the overall workflow objectives.\n"

    # Base prompt with technical requirements
    base_prompt = f"""
You are a materials generation assistant that writes Python code using the Atomic Simulation Environment (ASE) library to generate the structure file only (no INCAR, KPOINTS, or other input files).
Your goal is to fulfill the user's request by generating a structure and then validating it.
The working directory for all scripts and files is: {run_dir}
{plan_context}
**Technical requirements for scripts (IMPORTANT):**
- You MUST use the `execute_ase_script` tool to run code.
- Include all necessary ASE imports (e.g., `from ase.build import bulk`).
- {structure_requirement}
- **Crucially**, you MUST print this exact confirmation line upon success: `STRUCTURE_SAVED:<filename.ext>`
- Do **not** create INCAR, KPOINTS, POTCAR, or any other simulation input files.

**Error Handling:**
If a tool returns a message that starts with "ERROR:", you must:
1.  Carefully read the error message and the script that caused it.
2.  Identify the mistake in your code.
3.  Rewrite the corrected script and call the `execute_ase_script` tool again.
4.  Do not apologize. Just provide the corrected script.
"""

    # --- Workflow Stage Logic ---

    # Check if the last message was a tool error
    last_message = state["messages"][-1] if state.get("messages") else None
    if isinstance(last_message, ToolMessage) and last_message.content.strip().startswith("ERROR:"):
        return f"{base_prompt}\n\n**CURRENT TASK:** Your last attempt failed. Review the error message below and provide a corrected script."

    # Stage 4: Done
    if state.get("llm_validation_result"):
        return f"{base_prompt}\n\n**Workflow complete.**\n**CURRENT TASK:** All generation and validation steps are finished. Review the full history, especially the `quick_validation_result` and `llm_validation_result` tool messages, and provide a final summary to the user. Prefix your message with `FINAL ANSWER:`."

    # Stage 3: LLM Validate
    if state.get("quick_validation_result"):
        sf = state.get("structure_filename")
        gsf = state.get("generation_script_filename")
        return (
            f"{base_prompt}\n\n**Stage 3: Full Review**\n"
            "Quick validation is complete. Your task is to perform the final, detailed review.\n"
            "**CURRENT TASK:** Call the `get_llm_validation_and_hint` tool. You MUST provide the following arguments:\n"
            f"- `original_request`: \"{query}\"\n"
            f"- `run_dir`: \"{run_dir}\"\n"
            f"- `structure_filename`: \"{sf}\"\n"
            f"- `generation_script_filename`: \"{gsf}\""
            f"- `engine`: \"{engine}\""
        )

    # Stage 2: Quick Validate
    if state.get("structure_filename"):
        sf = state.get("structure_filename")
        return (
            f"{base_prompt}\n\n**Stage 2: Quick Validation**\n"
            f"Structure generation was successful. The file is '{sf}'.\n"
            "**CURRENT TASK:** Call the `quick_validate_structure` tool with:\n"
            f"- `structure_filename`: \"{sf}\"\n- `run_dir`: \"{run_dir}\""
            f"- `engine`: \"{engine}\""
        )

    # Stage 1: Generate
    return (
        f"{base_prompt}\n\n**Stage 1: Generate Structure**\n"
        "**CURRENT TASK:** Write and execute a Python script to generate the structure based on the user's request."
        " Use the `execute_ase_script` tool when you are ready.\n"
        f"User Request: {query}"
    )

def call_model(state: StructureAgentState):
    """The main model node. Generates a response based on the dynamic prompt."""

    # 1. Get the dynamic system prompt
    system_prompt = get_system_prompt(state)

    # 2. Prepare messages
    messages = list(state["messages"])
    # Replace or insert the system message
    if messages and isinstance(messages[0], SystemMessage):
        messages[0] = SystemMessage(content=system_prompt)
    else:
        messages.insert(0, SystemMessage(content=system_prompt))

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def _find_tool_call_for_message(messages: List[BaseMessage], tool_message: ToolMessage) -> tuple[AIMessage, dict] | tuple[None, None]:
    """Find the AIMessage and tool_call that corresponds to a ToolMessage."""
    for msg in reversed(messages[:-1]):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                if call["id"] == tool_message.tool_call_id:
                    return msg, call
    return None, None


def _parse_execute_ase_script_output(content: str, run_dir: str, engine: str) -> dict:
    """Parse the output from execute_ase_script tool."""
    updates = {}

    structure_match = re.search(r"STRUCTURE_SAVED:\s*([^\s]+)", content)
    script_match = re.search(r"Successfully executed script '([^']+)'", content)

    if structure_match:
        structure_filename = structure_match.group(1).strip().strip("'\"")
        updates["structure_filename"] = structure_filename

        try:
            prepared = prepare_structure_artifacts(Path(run_dir), structure_filename, engine)
            updates.update({
                "structure_source_path": str(prepared.source_path),
                "structure_artifact_path": str(prepared.artifact_path),
                "structure_artifact_filename": prepared.artifact_filename,
                "structure_summary": summarize_structure(prepared.atoms),
                "structure_prompt_blob": prepared.artifact_content,
            })
            set_active_run_dir(str(Path(run_dir).resolve()))
        except Exception as exc:
            updates["structure_generation_error"] = str(exc)

    if script_match:
        updates["generation_script_filename"] = script_match.group(1).strip().strip("'\"")

    return updates

def parse_structure_tool_output(state: StructureAgentState) -> dict:
    """
    Parses the output of the last tool call and updates the state.
    This is the key to moving the workflow forward.
    """
    if not state.get("messages"):
        return {}

    last_message = state["messages"][-1]
    if not isinstance(last_message, ToolMessage):
        # Not a tool message, nothing to parse
        return {}

    # Find the AIMessage that called this tool
    ai_message, tool_call = _find_tool_call_for_message(state["messages"], last_message)
    if not ai_message or not tool_call:
        return {}


    tool_name = tool_call["name"]
    content = last_message.content

    updates = {}

    # Check for errors first
    if content.strip().startswith("ERROR:"):
        # Don't update state, let the model see the error and retry
        updates["structure_generation_error"] = content
        print(f"[ERROR] Tool error detected:{content[:100]}")
        return updates

    # --- Workflow State Updates ---

    if tool_name == "execute_ase_script":
        script_updates = _parse_execute_ase_script_output(
            content,
            state.get("run_dir", os.getcwd()),
            (state.get("engine") or ENGINE_VASP).lower(),
        )
        updates.update(script_updates)
        if "structure_filename" in script_updates:
            print(f"✓ Structure generated: {script_updates['structure_filename']}")

    elif tool_name == "quick_validate_structure":
        updates["quick_validation_result"] = content
        print(f"✓ Quick validation complete")
    elif tool_name == "get_llm_validation_and_hint":
        updates["llm_validation_result"] = content
        print(f"✓ LLM validation complete")

    return updates


def agent_entrypoint(state: StructureAgentState) -> dict:
    """
    Initializes the agent state at the beginning of the run.
    Sets the run directory and query.
    """
    # Validate messages exist to prevent IndexError
    if not state.get("messages"):
        run_dir = state.get("run_dir") or f"./workspace/run_{uuid.uuid4().hex[:8]}"
        query = state.get("query") or ""
        os.makedirs(run_dir, exist_ok=True)
        abs_run_dir = os.path.abspath(run_dir)
        set_active_run_dir(abs_run_dir)
        return {"run_dir": abs_run_dir, "query": query}

    # 1. Get run_dir and query from the first HumanMessage
    first_message = state["messages"][0]
    content = first_message.content

    # We expect the initial message to be a JSON string for setup
    try:
        initial_data = json.loads(content)
        run_dir = initial_data["run_dir"]
        query = initial_data["query"]
    except (json.JSONDecodeError, KeyError):
        # Fallback for plain text query
        run_dir = state.get("run_dir") or f"./workspace/run_{uuid.uuid4().hex[:8]}"
        query = content

    # 2. Ensure run_dir exists and set the global for the tools
    os.makedirs(run_dir, exist_ok=True)
    abs_run_dir = os.path.abspath(run_dir)
    set_active_run_dir(abs_run_dir)  # This comes from structures_validation.py

    # 3. Update state
    engine = (state.get("engine") or ENGINE_VASP).lower()
    # We replace the setup message with a clean query message
    clean_query_message = HumanMessage(
        content=f"User Request: {query}",
        additional_kwargs={"internal": True},
    )

    return {
        "run_dir": abs_run_dir,
        "query": query,
        "engine": engine,
        "messages": [clean_query_message],
        "structure_filename": "",
        "generation_script_filename": "",
        "structure_source_path": "",
        "structure_artifact_filename": "",
        "structure_artifact_path": "",
        "structure_summary": "",
        "structure_generation_error": "",
        "structure_prompt_blob": "",
        "quick_validation_result": "",
        "llm_validation_result": "",
    }


# --- 4. Build the Graph ---

workflow = StateGraph(StructureAgentState)

# Add nodes
workflow.add_node("agent_entrypoint", agent_entrypoint)
workflow.add_node("model", call_model)
workflow.add_node("tools", ToolNode(all_tools))
workflow.add_node("parser", parse_structure_tool_output)

# Define edges
workflow.set_entry_point("agent_entrypoint")
workflow.add_edge("agent_entrypoint", "model")

# The model decides to call a tool or end the graph
workflow.add_conditional_edges(
    "model",
    tools_condition,
    {
        "tools": "tools",  # If tool call, go to tools
        END: END  # If no tool call (FINAL ANSWER), end
    }
)


workflow.add_edge("tools", "parser")

workflow.add_edge("parser", "model")


structure_agent = workflow.compile()

# --- 5. Test Harness ---

if __name__ == "__main__":
    import shlex

    # --- Setup for the run ---
    run_dir = "./workspace/yh2_test_run"
    query = "Construct a 3x3x3 supercell of YH2"
    thread_id = f"thread-{uuid.uuid4().hex[:8]}"

    # Clean the workspace for a fresh run
    if os.path.exists(run_dir):
        import shutil

        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    print(f"--- Starting Agent Run ---")
    print(f"Thread ID: {thread_id}")
    print(f"Run Dir:   {run_dir}")
    print(f"Query:     {query}\n")

    # The initial message must be a JSON string for the entrypoint node
    initial_input = {
        "messages": [
            HumanMessage(
                content=json.dumps({
                    "run_dir": run_dir,
                    "query": query
                })
            )
        ],
        "thread_id": thread_id,
    }

    # Stream the events
    try:
        for event in structure_agent.stream(initial_input, config={"recursion_limit": 25}):
            for key, value in event.items():
                print(f"--- Event: {key} ---")
                if key == "messages":
                    # Print the newest message
                    msg = value[-1]
                    print(f"[{msg.__class__.__name__}]")
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        print("Tool Call(s):")
                        for call in msg.tool_calls:
                            print(f"  - {call['name']}")
                            print(f"    Args: {json.dumps(call['args'], indent=2)}")
                    else:
                        print(msg.content)
                else:
                    print(f"Updated state: {key} = {value}")
            print("\n" + "=" * 40 + "\n")
    except Exception as e:
        print(f"\n--- AGENT FAILED ---")
        print(f"Error: {e}")

    print("--- Agent Run Complete ---")