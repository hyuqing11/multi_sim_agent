"""Two-stage VASP workflow that chains structure and input agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Literal

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import BaseModel
from backend.agents.library.vasp_agent.utils import ENGINE_VASP, ENGINE_LAMMPS, _normalize_engine,detect_engine
from backend.utils.workspace import async_get_workspace_path
from backend.agents.library.vasp_agent.input_agent import input_agent
from backend.agents.library.vasp_agent.structure_agent import structure_agent
from backend.agents.llm import get_model, settings

class VaspPipelineState(MessagesState, total=False):
    stage: str
    thread_id: str
    working_directory: str
    run_directory: str
    run_dir: str
    engine: str
    query: str
    structure_filename: str
    structure_source_path: str
    structure_artifact_filename: str
    structure_artifact_path: str
    structure_summary: str
    structure_prompt_blob: str
    input_directories: Dict[str, str]
    structure_generation_error: str
    plan_data: Dict[str, Any]
    dft_parameters: Dict[str, Any]



def _with_thread_suffix(config: Optional[RunnableConfig], thread_id: str | None, suffix: str) -> RunnableConfig:
    if not thread_id:
        return config or {}
    new_thread_id = f"{thread_id}:{suffix}"
    base_config = config or {}
    if isinstance(base_config, dict):
        new_config = dict(base_config)
        configurable = dict(new_config.get("configurable", {}))
        configurable["thread_id"] = new_thread_id
        new_config["configurable"] = configurable
        return new_config
    return {"configurable": {"thread_id": new_thread_id}}

def _filter_supervisor_messages(messages):
    """Remove supervisor-generated instruction messages before passing to worker agents."""
    return [
        msg for msg in messages
        if not getattr(msg, "additional_kwargs", {}).get("supervisor_instruction")
    ]
async def _ensure_workspace(state: VaspPipelineState, config: RunnableConfig | None) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}
    config_obj = config or {}
    configurable = (
        config_obj.get("configurable", {})
        if isinstance(config_obj, dict)
        else getattr(config_obj, "configurable", {})
    )
    thread_id = state.get("thread_id") or configurable.get("thread_id")
    if thread_id and state.get("thread_id") != thread_id:
        updates["thread_id"] = thread_id

    workspace_path = None
    if thread_id and not state.get("working_directory"):
        workspace_path = await async_get_workspace_path(thread_id)
        str_workspace = str(workspace_path)
        updates["working_directory"] = str_workspace
        updates.setdefault("run_directory", str_workspace)
        updates.setdefault("run_dir", str_workspace)

    run_dir = (
        updates.get("run_directory")
        or state.get("run_directory")
        or updates.get("run_dir")
        or state.get("run_dir")
    )
    if not run_dir:
        fallback_id = thread_id or "default"
        workspace_path = workspace_path or await async_get_workspace_path(fallback_id)
        str_workspace = str(workspace_path)
        updates.setdefault("working_directory", str_workspace)
        run_dir = str_workspace
        updates["run_directory"] = run_dir
        updates.setdefault("run_dir", run_dir)

    resolved_run_dir = str(Path(run_dir).resolve())
    Path(resolved_run_dir).mkdir(parents=True, exist_ok=True)
    updates["run_directory"] = resolved_run_dir
    updates["run_dir"] = resolved_run_dir

    return updates


async def _structure_node(state: VaspPipelineState, config: RunnableConfig) -> Dict[str, Any]:
    workspace_updates = await _ensure_workspace(state, config)
    merged_state: Dict[str, Any] = {**state, **workspace_updates}
    merged_state.setdefault("input_directories", {})
    if not merged_state.get("engine"):
        merged_state["engine"] = detect_engine(merged_state.get("query", ""))

    structure_state: Dict[str, Any] = {
        #"messages": _filter_supervisor_messages(merged_state.get("messages", [])),
        "messages": [
            msg for msg in _filter_supervisor_messages(merged_state.get("messages", []))
            if not getattr(msg, "additional_kwargs", {}).get("internal")  # Add this filter
        ],
        "thread_id": merged_state.get("thread_id"),
        "run_dir": merged_state.get("run_dir"),
        "query": merged_state.get("query"),
        "engine": merged_state.get("engine"),
    }

    # Ensure at least one message exists for the structure agent
    # If all messages were filtered out, create a message from the query
    if not structure_state["messages"] and merged_state.get("query"):
        from langchain_core.messages import HumanMessage
        structure_state["messages"] = [
            HumanMessage(content=merged_state["query"])
        ]

    # Pass plan data and parameters to structure agent
    if merged_state.get("plan_data"):
        structure_state["plan_data"] = merged_state["plan_data"]
    if merged_state.get("dft_parameters"):
        structure_state["dft_parameters"] = merged_state["dft_parameters"]

    structure_config = _with_thread_suffix(config, merged_state.get("thread_id"), "structure")
    structure_result = await structure_agent.ainvoke(structure_state, config=structure_config)

    updated_state = {**merged_state, **{k: v for k, v in structure_result.items() if k != "messages"}}
    if structure_result.get("run_dir"):
        resolved = str(Path(structure_result["run_dir"]).resolve())
        updated_state["run_dir"] = resolved
        updated_state["run_directory"] = resolved
        updated_state["working_directory"] = updated_state.get("working_directory") or resolved

    messages = structure_result.get("messages", [])
    clean_messages = [
        msg for msg in messages
        if not getattr(msg, "additional_kwargs", {}).get("internal")
    ]

    if updated_state.get("structure_prompt_blob"):
        if not updated_state.get("engine"):
            updated_state["engine"] = detect_engine(updated_state.get("query", ""))
        updated_state["stage"] = "inputs"
    else:
        updated_state["stage"] = "structure"

    updated_state["messages"] = clean_messages
    return updated_state


async def _input_node(state: VaspPipelineState, config: RunnableConfig) -> Dict[str, Any]:
    workspace_updates = await _ensure_workspace(state, config)
    merged_state = {**state, **workspace_updates}
    src_artifact = merged_state.get("structure_artifact_path")
    if src_artifact and Path(src_artifact).exists():
        run_dir = Path(merged_state.get("run_directory"))
        dst_artifact = run_dir / merged_state.get("structure_artifact_filename", "POSCAR")

        if not dst_artifact.exists():
            dst_artifact.write_text(Path(src_artifact).read_text())

            # Update the path to point to the new location
            merged_state["structure_artifact_path"] = str(dst_artifact)

    input_state: Dict[str, Any] = {
        #"messages": _filter_supervisor_messages(merged_state.get("messages", [])),
        "messages": [
            msg for msg in _filter_supervisor_messages(merged_state.get("messages", []))
            if not getattr(msg, "additional_kwargs", {}).get("internal")  # Add this filter
        ],
        "thread_id": merged_state.get("thread_id"),
        "run_directory": merged_state.get("run_directory"),
        "working_directory": merged_state.get("working_directory"),
        "query": merged_state.get("query"),
        "engine": merged_state.get("engine"),
        "structure_source_path": merged_state.get("structure_source_path"),
        "structure_artifact_path": merged_state.get("structure_artifact_path"),
        "structure_artifact_filename": merged_state.get("structure_artifact_filename"),
        "structure_summary": merged_state.get("structure_summary"),
        "structure_prompt_blob": merged_state.get("structure_prompt_blob"),
    }

    # Ensure at least one message exists for the input agent
    # If all messages were filtered out, create a message from the query
    if not input_state["messages"] and merged_state.get("query"):
        from langchain_core.messages import HumanMessage
        input_state["messages"] = [
            HumanMessage(content=merged_state["query"])
        ]

    # Pass plan data and parameters to input agent
    if merged_state.get("plan_data"):
        input_state["plan_data"] = merged_state["plan_data"]
    if merged_state.get("dft_parameters"):
        input_state["dft_parameters"] = merged_state["dft_parameters"]



    input_config = _with_thread_suffix(config, merged_state.get("thread_id"), "inputs")
    input_result = await input_agent.ainvoke(input_state, config=input_config)

    clean_messages = [
        msg for msg in input_result.get("messages", [])
        if not getattr(msg, "additional_kwargs", {}).get("internal")
    ]

    updated_state = {**merged_state, **{k: v for k, v in input_result.items() if k != "messages"}}

    updated_state["structure_prompt_blob"] = merged_state.get("structure_prompt_blob")
    updated_state["structure_artifact_path"] = merged_state.get("structure_artifact_path")
    updated_state["messages"] = clean_messages


    updated_state["stage"] = "done"
    return updated_state


class SupervisorDecision(BaseModel):
    next: Literal["structure", "inputs", "FINISH"]
    member_message: Optional[str] = None
    rationale: Optional[str] = None


def _format_supervisor_context(state: VaspPipelineState) -> str:
    has_structure = bool(state.get("structure_prompt_blob"))
    has_inputs = bool(state.get("input_directories"))
    parts = [
        f"Stage: {state.get('stage', 'structure')}",
        f"Engine: {state.get('engine') or 'unknown'}",
        f"Structure ready: {has_structure}",
        f"Inputs prepared: {has_inputs}",
    ]
    if state.get("structure_summary"):
        parts.append(f"Structure summary available: yes")
    return "\n".join(parts)


def make_supervisor_node(members: list[str]):
    allowed = set(members)

    async def supervisor_node(state: VaspPipelineState, config: Optional[RunnableConfig] = None) -> Command:
        if state.get("stage") == "done":
            return Command(goto=END, update={"stage": "done"})

        if not state.get("query"):
            messages = state.get("messages", [])
            if messages:
                query = messages[0].content if hasattr(messages[0], 'content') else ""
            else:
                query = ""
        else:
            query = state.get("query")



        if not state.get("engine"):
            engine = detect_engine(query)
        else:
            engine = _normalize_engine(state.get("engine"))



        structure_ready = bool(state.get("structure_prompt_blob"))
        structure_error = bool(state.get("structure_generation_error"))
        inputs_ready = bool(state.get("input_directories"))

        capabilities = (
            "You manage two workers:\n"
            "- structure: prepares atomic structures using ASE tools\n"
            "- inputs: generates VASP/LAMMPS input files (INCAR, KPOINTS, etc.) once a structure is available\n"
            "\n"
            "Decision rules:\n"
            "1. If no structure exists yet → route to 'structure'\n"
            "2. If structure exists but no inputs → route to 'inputs'\n"
            "3. If both are done:\n"
            "   - If user is asking for a completely NEW structure/simulation → route to 'structure'\n"
            "   - If user wants to MODIFY the structure parameters → route to 'structure'\n"
            "   - If user wants to MODIFY the input files (INCAR, KPOINTS, etc.) → route to 'inputs'\n"
            "   - If user is just asking questions or wants clarification → respond with FINISH\n"
            "\n"
            "Respond with 'FINISH' only when workflow is truly complete AND no further action is needed."
        )

        context = _format_supervisor_context(state)
        query = state.get("query", "")

        system_prompt = (
            "You are the supervisor for a materials workflow. "
            "Select the next worker to execute based on the current state."
        )

        decision_messages = [
            SystemMessage(content=system_prompt),
            SystemMessage(content=capabilities),
            SystemMessage(content=f"User request: {query}"),
            SystemMessage(content=f"Current state summary:\n{context}"),
            SystemMessage(
                content=(
                    "Reply with which worker should run next (`structure` or `inputs`) or `FINISH` if done. "
                    "Also craft a concise instruction message for that worker if action is required."
                )
            ),
        ]

        llm = get_model(settings.DEFAULT_MODEL)
        supervisor_config = _with_thread_suffix(config, state.get("thread_id"), "supervisor")
        decision = await llm.with_structured_output(
            SupervisorDecision,
            method="function_calling",
        ).ainvoke(
            decision_messages,
            config=supervisor_config,
        )

        goto = decision.next
        # Enforce structural dependency: cannot move to inputs before structure exists.
        if goto == "inputs" and not state.get("structure_prompt_blob"):
            goto = "structure"

        if goto == "structure" and structure_ready and not structure_error:
            goto = "inputs"

        if goto != "FINISH" and inputs_ready and not structure_error:
            goto = "FINISH"
        if goto =="FINISH":
            return Command(goto=END, update={"stage": "done"})
        if state.get("stage") == "done" and goto != "FINISH":
            return Command(goto=END, update={"stage": "done"})

        update: Dict[str, Any] = {}
        update["engine"] = engine
        print(f"🔍 SUPERVISOR: Setting update['engine'] = '{engine}'")
        if not state.get("query") and query:
            update["query"] = query


        if goto in allowed:
            update["stage"] = goto
        elif goto == "FINISH":
            update["stage"] = "done"

        if decision.member_message and goto in allowed:
            update_messages = list(state.get("messages", [])) + [
                HumanMessage(
                    content=decision.member_message.strip(),
                    additional_kwargs={"internal": True, "supervisor_instruction": True}
                )
            ]
            update["messages"] = update_messages

        if goto == "FINISH":
            return Command(goto=END, update=update)

        return Command(goto=goto, update=update)

    return supervisor_node


def _route(state: VaspPipelineState) -> str:
    stage = state.get("stage") or "structure"
    if stage == "structure":
        return "structure"
    if stage == "inputs":
        return "inputs"
    if stage == "done":
        return "end"
    return "end"

pipeline_graph = StateGraph(VaspPipelineState)
pipeline_graph.add_node("supervisor", make_supervisor_node(["structure", "inputs"]))
pipeline_graph.add_node("structure", _structure_node)
pipeline_graph.add_node("inputs", _input_node)
pipeline_graph.add_edge(START, "supervisor")
pipeline_graph.add_conditional_edges(
    "supervisor",
    _route,
    {
        "structure": "structure",
        "inputs": "inputs",
        "end": END,
    },
)
pipeline_graph.add_edge("structure", "supervisor")
pipeline_graph.add_edge("inputs", "supervisor")

vasp_pipeline_agent = pipeline_graph.compile()


async def test_pipeline():
    import uuid

    test_state = {
        "messages": [
            HumanMessage(content="Create 3x3x3 Bulk YH2 supercell and input files for structure relaxation")],
        "thread_id": f"test-{uuid.uuid4().hex[:8]}",
    }

    print("Testing pipeline...\n")

    # Stream and print all events
    async for event in vasp_pipeline_agent.astream(test_state, config={"recursion_limit": 50}):
        for key, value in event.items():
            print(f"\n=== Event: {key} ===")

            if key == "inputs":  # Input agent node
                print(f"Input directories: {value.get('input_directories', {})}")

                # Check last message
                messages = value.get('messages', [])
                if messages:
                    last_msg = messages[-1]
                    print(f"Last message type: {type(last_msg).__name__}")
                    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                        print(f"Tool calls made:")
                        for tc in last_msg.tool_calls:
                            print(f"  - {tc['name']}")
                    else:
                        print(f"Content: {str(last_msg.content)[:200]}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_pipeline())

#__all__ = ["vasp_pipeline_agent"]