"""High-level orchestrator that chains literature, planning, and execution agents."""

from __future__ import annotations
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
import json
from pathlib import Path
from typing import Any, Optional
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel
import os
from backend.agents.library.literature_agent import literature_agent
from backend.agents.library.materials_planner import materials_planner_agent
from backend.agents.library.vasp_agent import vasp_pipeline_agent
from backend.agents.library.hpc_agent import create_hpc_agent
from backend.agents.llm import get_model, settings
from backend.utils.workspace import async_get_workspace_path
import re

try:
    from backend.agents.library.chatbot import web_search as chatbot_web_search
except Exception:  # pragma: no cover - fallback if chatbot module unavailable
    chatbot_web_search = None

try:
    from backend.agents.library.chatbot import fetch_open_access_full_text as chatbot_fetch_full_text
except Exception:  # pragma: no cover - fallback if fetch tool is unavailable
    chatbot_fetch_full_text = None

hpc_agent = create_hpc_agent()


class PlanExtraction(BaseModel):
    workflow_plan: str
    recommended_parameters: dict[str, Any] = {}
    agent_actions: dict[str, list[str]] = {}


class SupervisorReport(BaseModel):
    status: str
    summary: str
    highlights: list[str] = []
    recommended_actions: list[str] = []


class StepGroup(BaseModel):
    """Represents a group of workflow steps that should be executed together."""
    agent: str  # Which agent handles this group: "vasp", "hpc", "analysis"
    steps: list[str]  # List of step descriptions
    description: str  # Overall description of what this group does
    parameters: dict[str, Any] = {}  # Specific parameters for this step group


class SupervisorStepDecision(BaseModel):
    """Supervisor decision about next steps to execute."""
    next_agent: str  # "planner", "vasp", "hpc", "analysis", or "FINISH"
    current_step_group: Optional[StepGroup] = None
    reasoning: str  # Why this decision was made


class OrchestratorState(MessagesState, total=False):
    # Core identifiers
    thread_id: str
    working_directory: str

    # Workflow tracking
    stage: str  # "planning", "structure_gen", "job_submission", "analysis", "complete"
    next_agent: str
    workflow_plan: str
    workflow_steps: list[dict[str, Any]]  # Parsed workflow steps from plan
    current_step_index: int  # Which step group we're executing
    current_step_group: dict[str, Any]  # Current step(s) being executed
    completed_steps: list[str]  # List of completed step descriptions

    latest_user_request: str
    last_planned_request: str

    # Planner outputs
    literature_context: str
    literature_summary: str
    literature_payload: dict[str, Any]
    literature_status: str
    literature_error: str
    last_literature_request: str
    literature_summary_path: str
    literature_payload_path: str
    literature_formatted_path: str
    literature_validation_notes: str
    literature_contradiction_detected: bool
    literature_self_check_notes: str
    literature_secondary_attempt: bool
    literature_secondary_query: str
    literature_source_verification_notes: str
    literature_source_urls: list[str]
    literature_sources_path: str
    dft_parameters: dict[str, Any]
    plan_data: dict[str, Any]

    # VASP pipeline outputs
    structure_ready: bool
    structure_filename: str
    input_directories: dict[str, str]  # folder_name -> path
    structure_summary: str
    vasp_run_status: str
    vasp_last_message: str

    # HPC agent outputs
    job_submitted: bool
    job_id: str
    job_status: str
    job_output_path: str
    hpc_observations: list[str]

    # Analysis results
    convergence_status: str
    convergence_details: dict[str, Any]
    supervisor_summary: str
    supervisor_recommendations: list[str]
    error_details: dict[str, Any]  # Everything in one place!
    error_history: list[dict[str, Any]]  # Full error objects!

    # Retry management
    retry_count: int
    max_retries: int

    # Parameter adjustments
    parameter_adjustments: dict[str, Any]



async def _ensure_workspace(state: OrchestratorState, config: RunnableConfig) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    configurable = (
        config.get("configurable", {})
        if isinstance(config, dict)
        else getattr(config, "configurable", {})
    )
    thread_id = state.get("thread_id") or configurable.get("thread_id")

    if thread_id and not state.get("thread_id"):
        updates["thread_id"] = thread_id

    if thread_id and not state.get("working_directory"):
        workspace_path = await async_get_workspace_path(thread_id)
        updates["working_directory"] = str(workspace_path)

    if "retry_count" not in state:
        updates.setdefault("retry_count", 0)
    if "max_retries" not in state:
        updates.setdefault("max_retries", 1)
    if "convergence_status" not in state:
        updates.setdefault("convergence_status", "unknown")
    if "convergence_details" not in state:
        updates.setdefault("convergence_details", {})
    if "error_history" not in state:
        updates.setdefault("error_history", [])
    else:
        updates["error_history"] = _coerce_error_history(state.get("error_history"))

    # Initialize workflow step tracking
    if "workflow_steps" not in state:
        updates.setdefault("workflow_steps", [])
    if "current_step_index" not in state:
        updates.setdefault("current_step_index", 0)
    if "completed_steps" not in state:
        updates.setdefault("completed_steps", [])


    if not state.get("latest_user_request"):
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                latest_content = msg.content.strip()
                if latest_content:
                    updates["latest_user_request"] = latest_content
                break

    return updates


def _message_to_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part) for part in content)
    return str(content)



def _parse_vasp_convergence(outcar_path: Path) -> dict[str, Any]:
    """Parse VASP OUTCAR for convergence information."""
    try:
        content = outcar_path.read_text(encoding="utf-8")

        results = {
            "electronic_converged": False,
            "ionic_converged": False,
            "final_energy": None,
            "energy_change": None,
            "forces_rms": None,
            "warnings": [],
        }

        # Check for electronic convergence
        if "reached required accuracy" in content.lower():
            results["electronic_converged"] = True

        # Check for ionic convergence
        ionic_match = re.search(r"reached required accuracy.*ionic", content, re.IGNORECASE)
        if ionic_match:
            results["ionic_converged"] = True

        # Extract final energy
        energy_matches = re.findall(r"free  energy   TOTEN  =\s+([-\d.]+)", content)
        if energy_matches:
            results["final_energy"] = float(energy_matches[-1])
            if len(energy_matches) >= 2:
                results["energy_change"] = abs(float(energy_matches[-1]) - float(energy_matches[-2]))

        # Extract forces
        forces_match = re.search(r"FORCES: max atom, RMS\s+([\d.]+)\s+([\d.]+)", content)
        if forces_match:
            results["forces_rms"] = float(forces_match.group(2))

        # Check for warnings
        warning_patterns = [
            "WARNING",
            "ZBRENT: fatal error",
            "EDDDAV: call to ZHEGV failed",
            "FEXCP",
        ]
        for pattern in warning_patterns:
            if pattern in content:
                results["warnings"].append(pattern)

        return results

    except Exception as e:
        return {"error": f"Failed to parse OUTCAR: {e}"}


def _determine_convergence_status(convergence_data: dict[str, Any]) -> str:
    """Determine overall convergence status."""
    if "error" in convergence_data:
        return "error"

    electronic = convergence_data.get("electronic_converged", False)
    ionic = convergence_data.get("ionic_converged", False)
    warnings = convergence_data.get("warnings", [])

    if warnings:
        return "error"

    if electronic and ionic:
        return "converged"
    elif electronic:
        return "partially_converged"  # Electronic converged but not ionic
    else:
        return "not_converged"


def _parse_job_status(observations: list[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    job_status: Optional[str] = None
    job_id: Optional[str] = None
    output_path: Optional[str] = None
    for obs in observations:
        if "JOB_RESULT" in obs:
            if "SUCCESS" in obs:
                job_status = "success"
            elif "FAILED" in obs or "ERROR" in obs:
                job_status = "failed"
        if "JOB_ID" in obs:
            parts = obs.split("JOB_ID:")
            if len(parts) > 1:
                job_id = parts[1].splitlines()[0].strip()
        if "OUT_FILE" in obs:
            parts = obs.split("OUT_FILE:")
            if len(parts) > 1:
                output_path = parts[1].splitlines()[0].strip()
    return job_status, job_id, output_path


def _suggest_parameter_adjustments(
        convergence_data: dict[str, Any],
        current_params: dict[str, Any]
) -> dict[str, Any]:
    """Suggest parameter adjustments based on convergence issues."""
    suggestions = {}

    if not convergence_data.get("electronic_converged"):
        # Increase electronic convergence steps
        suggestions["NELM"] = current_params.get("NELM", 60) + 20
        suggestions["ALGO"] = "Fast"  # Try different algorithm

    if not convergence_data.get("ionic_converged"):
        # Adjust ionic convergence criteria
        current_ediffg = current_params.get("EDIFFG", -0.02)
        suggestions["EDIFFG"] = current_ediffg * 1.5  # Relax criteria
        suggestions["NSW"] = current_params.get("NSW", 100) + 50

    forces_rms = convergence_data.get("forces_rms")
    if forces_rms and forces_rms > 0.1:
        suggestions["EDIFFG"] = -0.05  # More relaxed force convergence

    if "ZBRENT" in convergence_data.get("warnings", []):
        suggestions["ISMEAR"] = 0  # Switch to Gaussian smearing
        suggestions["SIGMA"] = 0.05

    return suggestions


def _safe_json_dumps(data: Any) -> str:
    try:
        return json.dumps(data, indent=2, default=str)
    except TypeError:
        return str(data)


def _coerce_error_history(entries: Optional[list[Any]]) -> list[dict[str, Any]]:
    """Normalize legacy error history entries into structured records."""
    history: list[dict[str, Any]] = []
    if not entries:
        return history
    for item in entries:
        if isinstance(item, dict):
            history.append(item)
        else:
            history.append({"message": str(item)})
    return history



def _last_human_content(messages: list[Any]) -> Optional[str]:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
    return None


def _find_output_files(working_dir: str, patterns: list[str]) -> list[Path]:
    """Find output files matching patterns in working directory."""
    work_path = Path(working_dir)
    found_files = []

    for pattern in patterns:
        found_files.extend(work_path.rglob(pattern))

    return found_files


def _with_thread_suffix(config: RunnableConfig, thread_id: Optional[str], suffix: str) -> RunnableConfig:
    if not thread_id:
        return config
    new_thread_id = f"{thread_id}:{suffix}"
    if isinstance(config, dict):
        new_config = dict(config)
        configurable = dict(new_config.get("configurable", {}))
        configurable["thread_id"] = new_thread_id
        new_config["configurable"] = configurable
        return new_config
    return {"configurable": {"thread_id": new_thread_id}}


class OrchestratorAgent:
    def __init__(self) -> None:
        self.graph = self._build_graph()
    async def _gather_literature_context(
            self,
            query: str,
            state: OrchestratorState,
            config: RunnableConfig,
            allow_retry: bool = True,
    ) -> dict[str, Any]:
        """Run the literature agent and persist its findings for planning."""

        if not query:
            return {}

        literature_state = {
            "messages": [HumanMessage(content=query)],
        }
        literature_config = _with_thread_suffix(config, state.get("thread_id"), "literature")

        try:
            result = await literature_agent.ainvoke(literature_state, config=literature_config)
        except Exception as exc:  # pragma: no cover - defensive guard against missing API keys
            return {
                "literature_status": "error",
                "literature_error": str(exc),
                "literature_context": f"Literature lookup failed: {exc}",
                "last_literature_request": query,
            }

        messages = result.get("messages", [])
        summary_text = ""
        if messages:
            summary_text = (_message_to_text(messages[-1]) or "").strip()

        payload_data: Optional[dict[str, Any]] = None
        formatted_answer: Optional[str] = None
        for msg in messages:
            text = _message_to_text(msg)
            if not text:
                continue
            stripped = text.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    payload_candidate = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload_candidate, dict):
                    payload_data = payload_candidate
                    formatted_answer = payload_candidate.get("formatted_answer")
                    break

        planner_context = summary_text

        status = "unknown"
        if payload_data and isinstance(payload_data, dict):
            status = payload_data.get("status", "success")

        suspicion_reasons: list[str] = []
        payload_status = str(payload_data.get("status", "")) if isinstance(payload_data, dict) else ""
        if not payload_data:
            suspicion_reasons.append("No payload returned")
        elif payload_status.lower() not in {"success", "cached"}:
            suspicion_reasons.append(f"Unexpected status: {payload_status}")
        elif not payload_data.get("search_results"):
            suspicion_reasons.append("No search results present")



        workspace_dir = state.get("working_directory")


        if workspace_dir:
            workspace_root = Path(workspace_dir)
            workspace_root.mkdir(parents=True, exist_ok=True)
            if summary_text:
                summary_path = str(Path(workspace_root, "literature_summary.txt"))
                Path(summary_path).write_text(summary_text)
            if payload_data is not None:
                payload_path = str(Path(workspace_root, "literature_payload.json"))
                Path(payload_path).write_text(json.dumps(payload_data, indent=2))
            if formatted_answer:
                formatted_path = str(Path(workspace_root, "literature_formatted_answer.txt"))
                Path(formatted_path).write_text(formatted_answer)

        updates: dict[str, Any] = {
            "literature_context": planner_context,
            "literature_summary": summary_text,
            "literature_payload": payload_data or {},
            "literature_status": status,
            "last_literature_request": query,
        }
        return updates

    async def _planner_node(self, state: OrchestratorState, config: RunnableConfig) -> dict[str, Any]:
        updates = await _ensure_workspace(state, config)
        merged_state = {**state, **updates}

        # Track the latest user intent driving the workflow
        messages = list(merged_state.get("messages", []))
        latest_request = merged_state.get("latest_user_request") or _last_human_content(messages) or ""
        normalized_request = latest_request.strip()

        last_literature_request = (merged_state.get("last_literature_request") or "").strip()
        literature_status = merged_state.get("literature_status")
        has_literature = bool(merged_state.get("literature_context"))

        needs_literature = bool(normalized_request) and (
            normalized_request != last_literature_request or not has_literature
        )

        if literature_status == "error" and normalized_request == last_literature_request and has_literature:
            # Preserve previous fallback context without re-querying the failing service
            needs_literature = False

        literature_updates: dict[str, Any] = {}
        if needs_literature:
            literature_updates = await self._gather_literature_context(normalized_request, merged_state, config)
            merged_state.update(literature_updates)
            updates.update(literature_updates)


        previous_plan = merged_state.get("workflow_plan")
        last_planned_request = merged_state.get("last_planned_request") or ""
        request_changed = bool(normalized_request and normalized_request != last_planned_request)
        needs_revision = bool(
            previous_plan
            and (
                request_changed
                or (merged_state.get("retry_count") or 0) > 0
                or bool(merged_state.get("parameter_adjustments"))
                or bool(merged_state.get("literature_contradiction_detected"))
            )
        )
        planner_stage = "plan" if needs_revision else "research"

        if previous_plan:
            # Plan revision mode
            user_content_parts = []

            user_content_parts.append(f"## Current Plan\n\n{previous_plan}")

            # Add comprehensive error feedback if retrying
            if merged_state.get("retry_count", 0) > 0:
                error_details = merged_state.get("error_details", {})

                if error_details:
                    user_content_parts.append(f"\n## Previous Calculation Analysis")

                    # Overall status
                    user_content_parts.append(
                        f"\n**Overall Status:** {error_details.get('convergence_status', 'unknown')}")
                    user_content_parts.append(f"**Severity:** {error_details.get('overall_severity', 'unknown')}")

                    # Convergence details
                    conv = error_details.get("convergence", {})
                    if conv:
                        user_content_parts.append(f"\n**Convergence:**")
                        user_content_parts.append(f"- Electronic: {'✓' if conv.get('electronic_converged') else '✗'}")
                        user_content_parts.append(f"- Ionic: {'✓' if conv.get('ionic_converged') else '✗'}")
                        if conv.get('final_energy'):
                            user_content_parts.append(f"- Final Energy: {conv['final_energy']} eV")
                        if conv.get('forces_rms'):
                            user_content_parts.append(f"- Forces RMS: {conv['forces_rms']} eV/Å")

                    # VASP errors
                    vasp_errors = error_details.get("vasp_errors", [])
                    if vasp_errors:
                        user_content_parts.append(f"\n**VASP Errors Found:** {len(vasp_errors)} error(s)")
                        for err in vasp_errors[:3]:  # Show top 3
                            user_content_parts.append(f"- {err.get('type', 'Error')}: {err.get('message', '')}")

                    # LLM diagnosis if available
                    if error_details.get("llm_diagnosis"):
                        user_content_parts.append(f"\n**Analysis:**\n{error_details['llm_diagnosis']}")

                    # Recovery recommendations
                    recovery = error_details.get("recovery_plan", {})
                    if recovery.get("parameter_adjustments"):
                        user_content_parts.append(f"\n**Recommended Parameter Adjustments:**")
                        user_content_parts.append("```json")
                        user_content_parts.append(json.dumps(recovery["parameter_adjustments"], indent=2))
                        user_content_parts.append("```")

                        if recovery.get("explanation"):
                            user_content_parts.append(f"\n**Why:** {recovery['explanation']}")

                    user_content_parts.append(f"\n**Task:** Revise the plan to address the issues described above.")

            if latest_request:
                user_content_parts.append(f"\n## Additional User Feedback\n\n{latest_request}")

            planner_prompt = "\n".join(user_content_parts)
        else:
            planner_prompt = normalized_request

        planner_input = {
            "messages": [HumanMessage(content=planner_prompt)],
            "stage": planner_stage,
        }

        if merged_state.get("literature_context"):
            planner_input["literature_context"] = merged_state["literature_context"]

        planner_config = _with_thread_suffix(config, merged_state.get("thread_id"), "planner")
        result = await materials_planner_agent.ainvoke(planner_input, config=planner_config)

        planner_messages = result.get("messages", [])
        plan_text = _message_to_text(planner_messages[-1]) if planner_messages else ""
        plan_data = await self._recover_plan_data(plan_text, merged_state, planner_config)

        workflow_plan = ""
        if plan_data and isinstance(plan_data, dict):
            workflow_plan = plan_data.get("workflow_plan", plan_text)
        else:
            workflow_plan = plan_text

        # Persist plan artifacts for downstream agents and user transparency
        workspace_dir = merged_state.get("working_directory")
        if workspace_dir and plan_text:
            Path(workspace_dir).mkdir(parents=True, exist_ok=True)
            Path(workspace_dir, "workflow_plan.txt").write_text(plan_text)
            if plan_data:
                Path(workspace_dir, "workflow_plan.json").write_text(_safe_json_dumps(plan_data))

        planner_updates: dict[str, Any] = {
            **updates,
            "messages": [AIMessage(content=plan_text)] if plan_text else planner_messages,
            "workflow_plan": workflow_plan,
            "latest_user_request": normalized_request or merged_state.get("latest_user_request"),
            "last_planned_request": normalized_request or merged_state.get("last_planned_request"),
            "literature_context": result.get("literature_context") or merged_state.get("literature_context"),
            "stage": "planning",
        }

        if plan_data is not None:
            planner_updates["plan_data"] = plan_data
        elif merged_state.get("plan_data") is not None:
            planner_updates["plan_data"] = merged_state["plan_data"]

        return planner_updates

    async def _recover_plan_data(
            self,
            plan_text: str,
            state: OrchestratorState,
            config: RunnableConfig,
    ) -> Optional[dict[str, Any]]:
        """Convert planner output into structured JSON when possible."""
        if not plan_text:
            return None

        try:
            return json.loads(plan_text)
        except (json.JSONDecodeError, TypeError):
            pass

        model = get_model(settings.DEFAULT_MODEL)
        extraction_messages = [
            SystemMessage(
                content=(
                    "You convert narrative DFT workflow plans into structured JSON."
                    " Return keys `workflow_plan`, `recommended_parameters`, and `agent_actions`"
                    " if present."
                )
            ),
            HumanMessage(content=f"Plan text:\n{plan_text}\n"),
        ]

        extraction_config = _with_thread_suffix(config, state.get("thread_id"), "plan_extract")
        try:
            extraction = await model.with_structured_output(
                PlanExtraction,
                method="function_calling",
            ).ainvoke(extraction_messages, config=extraction_config)
        except Exception:
            return None
        return extraction.dict()

    async def _run_vasp_pipeline(
            self,
            state: OrchestratorState,
            config: RunnableConfig,
    ) -> dict[str, Any]:
        updates = await _ensure_workspace(state, config)
        merged_state = {**state, **updates}

        user_query = merged_state.get("latest_user_request") or ""
        if not user_query:
            return {
                **updates,
                "messages": [AIMessage(content="Unable to run VASP pipeline: missing user request context.")],
            }

        plan_data = merged_state.get("plan_data")
        plan_summary = ""
        if isinstance(plan_data, dict):
            plan_summary = plan_data.get("workflow_plan", "")
        if not plan_summary:
            plan_summary = merged_state.get("workflow_plan") or merged_state.get("dft_plan") or ""

        # Get current step group for specific instructions
        current_step_group = merged_state.get("current_step_group")

        # Build step-specific instruction
        if current_step_group:
            step_description = current_step_group.get("description", "")
            step_list = current_step_group.get("steps", [])
            step_params = current_step_group.get("parameters", {})

            instruction = (
                "You are coordinating the VASP pipeline for a DFT workflow.\n"
                f"\n**Current Task:** {step_description}"
                f"\n**Specific Steps to Complete:**"
            )
            for i, step in enumerate(step_list, 1):
                instruction += f"\n  {i}. {step}"

            instruction += f"\n\n**Original User Request:** {user_query}"
            if plan_summary:
                instruction += f"\n**Overall Workflow Plan:** {plan_summary}"

            if step_params:
                instruction += f"\n\n**Step-Specific Parameters:**\n{json.dumps(step_params, indent=2)}"

            instruction += "\n\nGenerate the structure and input files according to these specific steps."
        else:
            # Fallback to original instruction if no step group
            instruction = (
                "You are coordinating the VASP pipeline for a DFT workflow."
                f"\nUser request: {user_query}"
                f"\nPlanned workflow: {plan_summary or 'Follow standard steps for DFT relaxation.'}"
                "Generate the structure and input files accordingly."
            )

        vasp_state: dict[str, Any] = {
            "messages": [HumanMessage(content=instruction,additional_kwargs = {"internal": True})],
            "thread_id": merged_state.get("thread_id"),
            "query": user_query,
        }

        # Pass structured plan data and parameters to pipeline agent
        if plan_data:
            vasp_state["plan_data"] = plan_data

        # Extract and pass DFT parameters from plan_data or state
        dft_params = None
        if isinstance(plan_data, dict):
            dft_params = plan_data.get("recommended_parameters", {})
        if not dft_params:
            dft_params = merged_state.get("dft_parameters", {})
        if dft_params:
            vasp_state["dft_parameters"] = dft_params

        run_dir = merged_state.get("working_directory")
        if run_dir:
            vasp_state["run_directory"] = run_dir
            vasp_state["working_directory"] = run_dir

        vasp_config = _with_thread_suffix(config, merged_state.get("thread_id"), "vasp")
        try:
            result = await vasp_pipeline_agent.ainvoke(vasp_state, config=vasp_config)
        except Exception as exc:
            error_history = _coerce_error_history(merged_state.get("error_history"))
            error_history.append(
                {
                    "source": "vasp_pipeline",
                    "message": f"VASP pipeline error: {exc}",
                    "type": "exception",
                }
            )
            return {
                **updates,
                "vasp_run_status": "error",
                "structure_ready": False,
                "error_history": error_history,
                "messages": [AIMessage(content=f"VASP pipeline error: {exc}")],
            }

        messages = [_message_to_text(msg) for msg in result.get("messages", [])]
        structure_ready = bool(result.get("structure_prompt_blob")) or bool(result.get("input_directories"))

        pipeline_updates: dict[str, Any] = {
            **updates,
            "vasp_run_status": "success" if structure_ready else "error",
            "structure_ready": structure_ready,
            "vasp_last_message": messages[-1] if messages else merged_state.get("vasp_last_message"),
            "input_directories": result.get("input_directories") or merged_state.get("input_directories") or {},
            "structure_summary": result.get("structure_summary") or merged_state.get("structure_summary"),
            "messages": result.get("messages", []),
        }

        for key in (
                "structure_filename",
                "structure_artifact_path",
                "structure_artifact_filename",
                "structure_source_path",
        ):
            if result.get(key):
                pipeline_updates[key] = result[key]

        if not structure_ready and result.get("structure_generation_error"):
            current_history = pipeline_updates.get("error_history") or merged_state.get("error_history")
            error_history = _coerce_error_history(current_history)
            error_history.append(
                {
                    "source": "vasp_pipeline",
                    "message": result["structure_generation_error"],
                    "type": "structure_generation_error",
                }
            )
            pipeline_updates["error_history"] = error_history

        workspace_dir = merged_state.get("working_directory")
        if workspace_dir and messages:
            Path(workspace_dir).mkdir(parents=True, exist_ok=True)
            log_path = Path(workspace_dir, "vasp_pipeline_run.txt")
            log_lines = "\n".join(messages[-3:]) if messages else ""
            if log_lines:
                log_path.write_text(log_lines)

        # Track step completion if using workflow steps
        if structure_ready and current_step_group:
            step_description = current_step_group.get("description", "VASP pipeline step")
            completed = list(merged_state.get("completed_steps", []))
            completed.append(step_description)
            pipeline_updates["completed_steps"] = completed

            # Advance to next step
            current_index = merged_state.get("current_step_index", 0)
            pipeline_updates["current_step_index"] = current_index + 1

        return pipeline_updates

    async def _run_hpc_workflow(
            self,
            state: OrchestratorState,
            config: RunnableConfig,
    ) -> dict[str, Any]:
        updates = await _ensure_workspace(state, config)
        merged_state = {**state, **updates}

        input_dirs = merged_state.get("input_directories", {})
        if not input_dirs:
            return {
                **updates,
                "messages": [AIMessage(content="ERROR: No VASP input directories available for submission.")],
            }

        calc_folder = list(input_dirs.values())[0]
        user_query = merged_state.get("latest_user_request") or ""

        # Get current step group for specific instructions
        current_step_group = merged_state.get("current_step_group")
        plan_data = merged_state.get("plan_data")

        # Build step-specific instruction
        if current_step_group:
            step_description = current_step_group.get("description", "")
            step_list = current_step_group.get("steps", [])
            step_params = current_step_group.get("parameters", {})

            instruction = (
                "Submit and monitor the VASP job on the HPC cluster.\n"
                f"\n**Current Task:** {step_description}"
                f"\n**Specific Steps:**"
            )
            for i, step in enumerate(step_list, 1):
                instruction += f"\n  {i}. {step}"

            instruction += f"\n\n**Working Directory:** {calc_folder}"
            instruction += f"\n**Original User Request:** {user_query}"

            if step_params:
                instruction += f"\n\n**Recommended Job Configuration:**"
                if 'walltime' in step_params:
                    instruction += f"\n- Walltime: {step_params['walltime']}"
                if 'nodes' in step_params:
                    instruction += f"\n- Nodes: {step_params['nodes']}"
                if 'cores' in step_params:
                    instruction += f"\n- Cores: {step_params['cores']}"

            instruction += "\n\nSubmit the job using the VASP input files in the working directory."
        else:
            # Fallback to original instruction
            instruction = (
                "Submit the VASP job to the HPC cluster."
                f"\nUser request: {user_query}"
                f"\nVASP input directories: {calc_folder}"
                "\nUse the provided directories as the working inputs."
            )

        hpc_state: dict[str, Any] = {
            "messages": [HumanMessage(content=instruction,additional_kwargs = {"internal": True}),],
            "thread_id": merged_state.get("thread_id"),
            "working_directory": merged_state.get("working_directory"),
            "retry_count": merged_state.get("retry_count", 0) or 0,
            "max_retries": merged_state.get("max_retries", 1) or 1,
        }

        # Pass structured plan data and parameters to HPC agent
        if plan_data:
            hpc_state["plan_data"] = plan_data

        # Pass DFT parameters
        dft_params = merged_state.get("dft_parameters", {})
        if dft_params:
            hpc_state["dft_parameters"] = dft_params

        # Pass current step group
        if current_step_group:
            hpc_state["current_step_group"] = current_step_group

        hpc_config = _with_thread_suffix(config, merged_state.get("thread_id"), "hpc")
        try:
            result = await hpc_agent.ainvoke(hpc_state, config=hpc_config)
        except Exception as exc:
            error_history = _coerce_error_history(merged_state.get("error_history"))
            error_history.append(
                {
                    "source": "hpc_agent",
                    "message": f"HPC agent error: {exc}",
                    "type": "exception",
                }
            )
            return {
                **updates,
                "job_status": "error",
                "hpc_observations": [f"HPC agent error: {exc}"],
                "error_history": error_history,
                "messages": [AIMessage(content=f"HPC agent error: {exc}")],
            }

        raw_messages = result.get("messages", [])
        observations = [_message_to_text(msg) for msg in raw_messages]
        job_status, job_id, output_path = _parse_job_status(observations)

        hpc_updates: dict[str, Any] = {
            **updates,
            "job_submitted": bool(job_id or observations),
            "job_status": job_status or merged_state.get("job_status"),
            "job_id": job_id or merged_state.get("job_id"),
            "job_output_path": output_path or merged_state.get("job_output_path"),
            "hpc_observations": observations,
            "retry_count": result.get("retry_count", merged_state.get("retry_count")),
            "max_retries": result.get("max_retries", merged_state.get("max_retries")),
            "messages": raw_messages,
        }

        workspace_dir = merged_state.get("working_directory")
        if workspace_dir and observations:
            Path(workspace_dir).mkdir(parents=True, exist_ok=True)
            Path(workspace_dir, "hpc_observations.txt").write_text("\n\n".join(observations))

        # Track step completion if job submitted successfully
        current_step_group = merged_state.get("current_step_group")
        if job_id and current_step_group:
            step_description = current_step_group.get("description", "HPC submission step")
            completed = list(merged_state.get("completed_steps", []))
            completed.append(step_description)
            hpc_updates["completed_steps"] = completed

            # Advance to next step
            current_index = merged_state.get("current_step_index", 0)
            hpc_updates["current_step_index"] = current_index + 1

        return hpc_updates

    async def analyze_results_node(self, state: OrchestratorState, config: RunnableConfig) -> dict[str, Any]:
        """
        Analyze HPC job results using HYBRID approach.

        Returns unified error_details covering ALL issues:
        - Convergence problems
        - VASP errors
        - Analysis metadata
        - Recovery recommendations
        """
        working_dir = state.get("working_directory", "")
        if not working_dir:
            error_details = {
                "has_errors": True,
                "overall_severity": "critical",
                "convergence": {},
                "vasp_errors": [],
                "analysis": {
                    "method": "none",
                    "confidence": "none",
                    "error": "No working directory set"
                },
                "convergence_status": "error",
            }
            history = _coerce_error_history(state.get("error_history"))
            history.append(error_details)
            return {
                "messages": [AIMessage(content="Error: No working directory set")],
                "error_details": error_details,
                "error_history": history,
                "convergence_status": "error",
                "convergence_details": {},
            }
        outcar_files = _find_output_files(working_dir, ["OUTCAR"])
        if not outcar_files:
            error_details = {
                "has_errors": True,
                "overall_severity": "critical",
                "convergence": {},
                "vasp_errors": [],
                "analysis": {
                    "method": "file_check",
                    "confidence": "high",
                    "error": "No OUTCAR file found - job may have crashed"
                },
                "convergence_status": "error",
            }
            history = _coerce_error_history(state.get("error_history"))
            history.append(error_details)
            return {
                "messages": [AIMessage(content="Error: No OUTCAR file found. Job may have failed.")],
                "error_details": error_details,
                "error_history": history,
                "convergence_status": "error",
                "convergence_details": {},
            }

        latest_outcar = max(outcar_files, key=lambda p: p.stat().st_mtime)
        convergence_data = _parse_vasp_convergence(latest_outcar)
        convergence_status = _determine_convergence_status(convergence_data)
        current_params = state.get("dft_parameters", {})
        retry_count = state.get("retry_count", 0)

        from .vasp_debug_agent import hybrid_vasp_error_analysis
        print(f"\n Analyzing errors (retry count: {retry_count})...")

        error_analysis = await hybrid_vasp_error_analysis(working_dir, current_params)

        analysis_method = error_analysis.get("method", "unknown")
        confidence = error_analysis.get("confidence", "unknown")

        print(f"   Method used: {analysis_method}")
        print(f"   Confidence: {confidence}")

        has_convergence_issues = convergence_status != "converged"
        has_vasp_errors = bool(error_analysis.get("errors"))
        vasp_error_severity = error_analysis.get("severity", "none")

        # Overall severity is the worst of convergence and VASP errors
        if vasp_error_severity == "critical":
            overall_severity = "critical"
        elif has_convergence_issues and vasp_error_severity in ["high", "medium"]:
            overall_severity = "high"
        elif has_convergence_issues or vasp_error_severity == "medium":
            overall_severity = "medium"
        elif vasp_error_severity == "low":
            overall_severity = "low"
        else:
            overall_severity = "none"

        # Build unified error details
        error_details = {
            "has_errors": has_convergence_issues or has_vasp_errors,
            "overall_severity": overall_severity,
            "convergence_status": convergence_status,

            # Convergence information
            "convergence": {
                "electronic_converged": convergence_data.get("electronic_converged", False),
                "ionic_converged": convergence_data.get("ionic_converged", False),
                "final_energy": convergence_data.get("final_energy"),
                "energy_change": convergence_data.get("energy_change"),
                "forces_rms": convergence_data.get("forces_rms"),
                "warnings": convergence_data.get("warnings", [])
            },

            # VASP errors
            "vasp_errors": [],

            # Analysis metadata
            "analysis": {
                "method": analysis_method,
                "confidence": confidence,
                "timestamp": Path(latest_outcar).stat().st_mtime,
                "outcar_path": str(latest_outcar)
            },

            # Recovery plan
            "recovery_plan": error_analysis.get("recovery_plan", {}),
        }

        # Add VASP errors if any
        if analysis_method == "pattern_matching":
            if error_analysis.get("errors"):
                error_details["vasp_errors"] = [
                    {
                        "type": err.error_type,
                        "severity": err.severity.value,
                        "message": err.message,
                        "found_in": err.found_in,
                        "line_number": err.line_number,
                        "context": err.context
                    }
                    for err in error_analysis["errors"]
                ]
        elif analysis_method == "dynamic_llm":
            error_details["llm_diagnosis"] = error_analysis.get("error_diagnosis", "")
            if error_analysis.get("pattern_detected_errors"):
                error_details["vasp_errors"] = [
                    {"message": err} for err in error_analysis["pattern_detected_errors"]
                ]


        analysis_sections = []

        analysis_sections.append("## Job Analysis Results")
        analysis_sections.append(f"\n**Overall Status:** {convergence_status}")
        analysis_sections.append(f"**Severity:** {overall_severity}")
        analysis_sections.append(f"**Analysis Method:** {analysis_method} (confidence: {confidence})")

        # Convergence details
        analysis_sections.append(f"\n### Convergence")
        conv = error_details["convergence"]
        analysis_sections.append(f"- Electronic: {'✓' if conv['electronic_converged'] else '✗'}")
        analysis_sections.append(f"- Ionic: {'✓' if conv['ionic_converged'] else '✗'}")
        analysis_sections.append(
            f"- Final Energy: {conv['final_energy']} eV" if conv['final_energy'] else "- Final Energy: N/A")
        analysis_sections.append(
            f"- Energy Change: {conv['energy_change']} eV" if conv['energy_change'] else "- Energy Change: N/A")
        analysis_sections.append(
            f"- Forces RMS: {conv['forces_rms']} eV/Å" if conv['forces_rms'] else "- Forces RMS: N/A")

        # VASP errors
        if error_details["vasp_errors"]:
            analysis_sections.append(f"\n### VASP Errors ({len(error_details['vasp_errors'])})")
            for err in error_details["vasp_errors"][:5]:  # Show first 5
                analysis_sections.append(f"- **{err.get('type', 'Error')}** [{err.get('severity', 'unknown')}]")
                analysis_sections.append(f"  {err['message']}")
                if err.get('found_in'):
                    analysis_sections.append(f"  Found in: {err['found_in']}")
        else:
            analysis_sections.append(f"\n**VASP Errors:** None detected ✓")

        # LLM diagnosis if available
        if error_details.get("llm_diagnosis"):
            analysis_sections.append(f"\n### LLM Analysis")
            analysis_sections.append(error_details["llm_diagnosis"])

        # Recovery plan
        recovery = error_details["recovery_plan"]
        if recovery.get("parameter_adjustments"):
            analysis_sections.append(f"\n### Recovery Plan")
            if recovery.get("explanation"):
                analysis_sections.append(f"\n{recovery['explanation']}")

            analysis_sections.append(f"\n**Parameter Adjustments:**")
            analysis_sections.append("```json")
            analysis_sections.append(json.dumps(recovery["parameter_adjustments"], indent=2))
            analysis_sections.append("```")

            if recovery.get("alternatives"):
                analysis_sections.append(f"\n**Alternatives:**")
                for alt in recovery["alternatives"]:
                    analysis_sections.append(f"- {alt}")

        analysis_sections.append(f"\n**Output File:** {latest_outcar}")

        analysis_msg = "\n".join(analysis_sections)


        recovery = error_details.get("recovery_plan") or {}
        parameter_adjustments = recovery.get("parameter_adjustments")
        if parameter_adjustments is None:
            parameter_adjustments = recovery.get("adjustments") or {}
            if parameter_adjustments:
                recovery["parameter_adjustments"] = parameter_adjustments
                error_details["recovery_plan"] = recovery

        should_retry = error_analysis.get("should_retry", True)

        updates = {
            "messages": [AIMessage(content=analysis_msg)],
            "error_details": error_details,
            "convergence_status": convergence_status,
            "convergence_details": convergence_data,
        }

        history = _coerce_error_history(state.get("error_history"))
        history.append(error_details)
        updates["error_history"] = history

        if parameter_adjustments:
            updates["parameter_adjustments"] = parameter_adjustments

        if error_details["has_errors"] and should_retry and parameter_adjustments:
            updates["retry_count"] = retry_count + 1
        elif not should_retry:
            updates.setdefault("retry_count", retry_count)

        if overall_severity == "critical" and not should_retry:
            updates["messages"][0].content += "\n\n  **CRITICAL ERROR**: Manual intervention recommended."

        if not error_details.get("has_errors"):
            updates.pop("parameter_adjustments", None)

        return updates

    async def _parse_workflow_steps(self, state: OrchestratorState) -> list[dict[str, Any]]:
        """Parse and group workflow steps from plan_data using LLM."""
        plan_data = state.get("plan_data", {})

        if not isinstance(plan_data, dict):
            return []

        # Check if we already have parsed steps
        if state.get("workflow_steps"):
            return state["workflow_steps"]

        # Extract workflow plan text
        workflow_text = plan_data.get("workflow_plan", "")
        agent_actions = plan_data.get("agent_actions", {})

        if not workflow_text and not agent_actions:
            return []

        # Use LLM to intelligently group steps
        system_prompt = """You are a workflow parser for DFT calculations.

Your task is to analyze a workflow plan and group related steps that should be executed together by the same agent.

## Agent Capabilities:
- **vasp**: Generates structures AND creates input files (INCAR, KPOINTS, etc.) - these should be grouped together
- **hpc**: Submits jobs to HPC cluster and monitors execution
- **analysis**: Analyzes results and convergence

## Grouping Rules:
1. "Generate structure" + "Create input files" = ONE vasp step group (they need each other)
2. Each distinct calculation type (relaxation, static, bands, DOS) = separate step groups
3. Each HPC submission = separate step group
4. Analysis after each calculation = separate step group

## Examples:
Plan: "1. Create YH2 structure, 2. Generate relaxation inputs, 3. Submit relaxation job"
Groups: [
  {agent: "vasp", steps: ["Create YH2 structure", "Generate relaxation inputs"], description: "Prepare structure and relaxation inputs"},
  {agent: "hpc", steps: ["Submit relaxation job"], description: "Run relaxation calculation"}
]

Plan: "1. Relax structure, 2. Static calculation, 3. Band structure"
Groups: [
  {agent: "vasp", steps: ["Prepare relaxation"], description: "Structure and relaxation inputs"},
  {agent: "hpc", steps: ["Run relaxation"], description: "Execute relaxation"},
  {agent: "vasp", steps: ["Prepare static calc"], description: "Static calculation inputs"},
  {agent: "hpc", steps: ["Run static"], description: "Execute static calculation"},
  {agent: "vasp", steps: ["Prepare bands"], description: "Band structure inputs"},
  {agent: "hpc", steps: ["Run bands"], description: "Execute band calculation"}
]

Output a list of step groups."""

        llm = get_model(settings.DEFAULT_MODEL)

        # Build context for LLM
        context = f"Workflow Plan:\n{workflow_text}\n\n"
        if agent_actions:
            context += f"Agent Actions:\n{json.dumps(agent_actions, indent=2)}\n"

        try:
            # Try to get structured output
            class StepGroupList(BaseModel):
                groups: list[StepGroup]

            result = await llm.with_structured_output(
                StepGroupList,
                method="function_calling"
            ).ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=context)
            ])

            return [group.dict() for group in result.groups]
        except Exception:
            # Fallback: create simple step groups
            return [{
                "agent": "vasp",
                "steps": ["Generate structure and input files"],
                "description": "Prepare VASP calculation",
                "parameters": {}
            }]

    async def _supervisor_node(self, state: OrchestratorState, config: RunnableConfig) -> dict[str, Any]:
        updates = await _ensure_workspace(state, config)
        merged_state = {**state, **updates}

        # Parse workflow steps if we have a plan but no steps yet
        has_plan = bool(merged_state.get("workflow_plan"))
        has_steps = bool(merged_state.get("workflow_steps"))

        if has_plan and not has_steps:
            workflow_steps = await self._parse_workflow_steps(merged_state)
            updates["workflow_steps"] = workflow_steps
            merged_state["workflow_steps"] = workflow_steps

        system_prompt = """You are an intelligent workflow supervisor for DFT calculations.

## Available Agents
1. **planner** – Creates workflow plan based on literature
2. **vasp** – VASP pipeline (structure + input generation)
3. **hpc** – HPC job submission and monitoring
4. **analysis** – Result analysis and convergence checking
5. **FINISH** – Workflow complete

## Your Task
Analyze the current state and determine:
1. Which agent should execute next
2. Which step group (if any) should be executed
3. Provide clear reasoning

## Step-by-Step Execution
If workflow_steps exist:
- Execute one step group at a time
- Track completed_steps to know what's done
- Move through step groups sequentially
- Mark completion after each successful execution

## Decision Logic
- No plan → planner
- Plan exists, current step is VASP task → vasp
- VASP complete, current step is HPC task → hpc
- Job complete → analysis
- All steps done → FINISH"""

        stage = merged_state.get("stage", "init")
        structure_ready = merged_state.get("structure_ready", False)
        job_submitted = merged_state.get("job_submitted", False)
        convergence = merged_state.get("convergence_status", "unknown")
        retry_count = merged_state.get("retry_count", 0)
        max_retries = merged_state.get("max_retries", 1)

        workflow_steps = merged_state.get("workflow_steps", [])
        current_step_index = merged_state.get("current_step_index", 0)
        completed_steps = merged_state.get("completed_steps", [])

        latest_request = merged_state.get("latest_user_request", "")
        last_planned = merged_state.get("last_planned_request", "")
        request_changed = latest_request and latest_request != last_planned

        context_summary = f"""## Current Workflow State

**Stage:** {stage}
**Working Directory:** {merged_state.get("working_directory", "not set")}

**Progress:**
- Plan created: {"✓" if has_plan else "✗"}
- Structure generated: {"✓" if structure_ready else "✗"}
- Job submitted: {"✓" if job_submitted else "✗"}
- Convergence: {convergence}

**Retry Status:** {retry_count}/{max_retries}
**Request changed:** {"Yes (needs re-planning)" if request_changed else "No"}

**Workflow Steps:** {len(workflow_steps)} step groups total
**Current Step Index:** {current_step_index}
**Completed Steps:** {len(completed_steps)}

"""

        # Show current and next step groups
        if workflow_steps:
            if current_step_index < len(workflow_steps):
                current_group = workflow_steps[current_step_index]
                context_summary += f"\n**Current Step Group:**\n"
                context_summary += f"- Agent: {current_group.get('agent')}\n"
                context_summary += f"- Description: {current_group.get('description')}\n"
                context_summary += f"- Steps: {current_group.get('steps')}\n"

            if current_step_index + 1 < len(workflow_steps):
                next_group = workflow_steps[current_step_index + 1]
                context_summary += f"\n**Next Step Group:**\n"
                context_summary += f"- Agent: {next_group.get('agent')}\n"
                context_summary += f"- Description: {next_group.get('description')}\n"

        if merged_state.get("parameter_adjustments"):
            context_summary += f"\n**Pending Adjustments:** {merged_state['parameter_adjustments']}"

        # Get last few messages for context
        recent_messages = merged_state.get("messages", [])[-5:]

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=context_summary),
            *recent_messages,
            HumanMessage(content="Based on the workflow state above, decide the next agent and step group.")
        ]

        llm = get_model(settings.DEFAULT_MODEL)

        # Get structured decision
        try:
            decision = await llm.with_structured_output(
                SupervisorStepDecision,
                method="function_calling"
            ).ainvoke(messages)
            next_agent = decision.next_agent.lower()
            current_step_group = decision.current_step_group.dict() if decision.current_step_group else None
            reasoning = decision.reasoning
        except Exception as e:
            # Fallback to simple logic
            print(f"Warning: Structured output failed: {e}, using fallback logic")
            needs_replanning = request_changed

            # Simple fallback logic
            if not has_plan or needs_replanning:
                next_agent = "planner"
                current_step_group = None
                reasoning = "No plan exists or request changed"
            elif workflow_steps and current_step_index < len(workflow_steps):
                current_step_group = workflow_steps[current_step_index]
                next_agent = current_step_group.get("agent", "vasp")
                reasoning = f"Executing step group {current_step_index + 1}/{len(workflow_steps)}"
            elif convergence == "unknown" and job_submitted:
                next_agent = "analysis"
                current_step_group = None
                reasoning = "Analyzing job results"
            elif current_step_index >= len(workflow_steps):
                next_agent = "finish"
                current_step_group = None
                reasoning = "All workflow steps completed"
            else:
                next_agent = "vasp"
                current_step_group = None
                reasoning = "Default: starting VASP pipeline"

        # Determine stage from agent
        agent_to_stage = {
            "planner": "planning",
            "vasp": "structure_gen",
            "hpc": "job_submission",
            "analysis": "analysis",
            "finish": "complete",
        }
        new_stage = agent_to_stage.get(next_agent, stage)

        # If we have a current step group, store it and get step-specific parameters
        if current_step_group:
            updates["current_step_group"] = current_step_group

            # Extract step-specific parameters if available
            step_params = current_step_group.get("parameters", {})
            if step_params:
                # Merge with existing dft_parameters
                existing_params = merged_state.get("dft_parameters", {})
                updates["dft_parameters"] = {**existing_params, **step_params}

        display_names = {
            "planner": "planner",
            "vasp": "vasp_pipeline",
            "hpc": "hpc_agent",
            "analysis": "analysis",
            "finish": "FINISH",
        }

        supervisor_msg = f"🎯 **Supervisor Decision:** Routing to **{display_names.get(next_agent, next_agent)}**\n"
        supervisor_msg += f"**Stage:** {new_stage}\n"
        if current_step_group:
            supervisor_msg += f"**Current Step Group:** {current_step_group.get('description', 'N/A')}\n"
            supervisor_msg += f"**Steps:** {', '.join(current_step_group.get('steps', []))}\n"
        supervisor_msg += f"**Reasoning:** {reasoning}"

        return {
            **updates,
            "messages": [AIMessage(content=supervisor_msg)],
            "next_agent": next_agent,
            "stage": new_stage,
        }

    def _route_from_supervisor(self, state: OrchestratorState) -> str:
        next_agent = (state.get("next_agent") or "").lower()
        if next_agent in {"planner", "vasp", "hpc", "analysis"}:
            return next_agent
        if next_agent == "finish":
            return "end"
        return "planner"

    def _build_graph(self):
        workflow = StateGraph(OrchestratorState)
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("vasp", self._run_vasp_pipeline)
        workflow.add_node("hpc", self._run_hpc_workflow)
        workflow.add_node("analysis", self.analyze_results_node)

        workflow.add_edge(START, "supervisor")
        workflow.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "planner": "planner",
                "vasp": "vasp",
                "hpc": "hpc",
                "analysis": "analysis",
                "end": END,
            },
        )

        workflow.add_edge("planner", "supervisor")
        workflow.add_edge("vasp", "supervisor")
        workflow.add_edge("hpc", "supervisor")
        workflow.add_edge("analysis", "supervisor")

        return workflow.compile(checkpointer=MemorySaver())


def create_orchestrator_agent():
    return OrchestratorAgent().graph


orchestrator_agent = create_orchestrator_agent()


async def run_dft_workflow(
        user_request: str,
        thread_id: Optional[str] = None,
        working_directory: Optional[str] = None,
) -> dict[str, Any]:
    """
    Run a complete DFT workflow with the supervisor agent.

    Args:
        user_request: User's calculation request
        thread_id: Optional thread ID for state persistence
        working_directory: Optional working directory path

    Returns:
        Final workflow state
    """
    import uuid
    if not thread_id:
        thread_id = str(uuid.uuid4())

    initial_state = {
        "messages": [HumanMessage(content=user_request)],
        "thread_id": thread_id,
    }

    if working_directory:
        initial_state["working_directory"] = working_directory

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # Stream events
    async for event in orchestrator_agent.astream(initial_state, config):
        for node_name, node_output in event.items():
            print(f"\n{'=' * 60}")
            print(f"Node: {node_name}")
            print(f"{'=' * 60}")

            if "messages" in node_output and node_output["messages"]:
                last_msg = node_output["messages"][-1]
                if hasattr(last_msg, "content"):
                    print(last_msg.content[:500])

            if "stage" in node_output:
                print(f"Stage: {node_output['stage']}")

            if "convergence_status" in node_output:
                print(f"Convergence: {node_output['convergence_status']}")

    # Get final state
    final_state = await orchestrator_agent.aget_state(config)
    return final_state.values


async def main():
    query = 'lattice constant of Pt'
    print("Starting DFT Workflow Supervisor")
    print("=" * 60)

    final_state = await run_dft_workflow(query)
    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE")
    print("=" * 60)
    print(f"Final Stage: {final_state.get('stage')}")
    print(f"Convergence: {final_state.get('convergence_status')}")
    print(f"Retries: {final_state.get('retry_count')}/{final_state.get('max_retries')}")
    print(f"Output: {final_state.get('job_output_path')}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())