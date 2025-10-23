"""Materials planning agent focused on DFT workflow design."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from backend.agents.llm import get_model, settings
from backend.utils.workspace import async_get_workspace_path


class PlannerState(MessagesState, total=False):
    thread_id: str
    working_directory: str
    stage: str
    literature_context: str


async def _ensure_thread_context(state: PlannerState, config: RunnableConfig) -> PlannerState:
    configurable = {}
    if isinstance(config, dict):
        configurable = config.get("configurable", {})
    else:
        configurable = getattr(config, "configurable", {})
    thread_id = state.get("thread_id") or configurable.get("thread_id")
    if not thread_id or state.get("working_directory"):
        return state

    workspace = await async_get_workspace_path(thread_id)
    new_state: PlannerState = {**state, "thread_id": thread_id, "working_directory": str(workspace)}
    return new_state


class MaterialsPlannerAgent:
    def __init__(self) -> None:
        self.graph = self._build_graph()

    async def _llm_node(self, state: PlannerState, config: RunnableConfig) -> dict[str, Any]:
        state = await _ensure_thread_context(state, config)
        configurable = {}
        if isinstance(config, dict):
            configurable = config.get("configurable", {})
        else:
            configurable = getattr(config, "configurable", {})
        model_name = configurable.get("model", settings.DEFAULT_MODEL)
        llm = get_model(model_name)
        stage = state.get("stage", "research")
        literature_context = state.get("literature_context")
        llm_with_tools = llm
        messages = list(state.get("messages", []))

        context_clause = (
            "\n\nAvailable literature context:\n" + literature_context.strip()
            if isinstance(literature_context, str) and literature_context.strip()
            else ""
        )

        base_guidance = (
            "You are a DFT workflow planner.\n"
            "- Using the literature findings provided by the orchestrator, craft a detailed plan to compute the requested material property via DFT.\n"
            "- Include **numbered steps** that downstream agents can execute.\n"
            "- Cite the CROW task ID or note if data came from cache.\n"
            "- Highlight critical parameters (functional, k-points, cutoffs, convergence criteria).\n\n"
            "Agent capabilities to consider:\n"
            "1. **vasp_pipeline agent**\n"
            "- Can generate an initial structure file and corresponding INCAR, POSCAR, and POTCAR files.\n"
            "- Supports convergence testing and multiple runs using the same structure.\n"
            "- The INCAR, KPOINTS, POSCAR, and POTCAR files for these runs will be located in `input_directories`.\n"
            "2. **hpc_agent**\n"
            "- Can generate HPC submission scripts, submit jobs to the HPC cluster, and monitor job success.\n"
            "- If a job fails, it automatically retries up to **three times**.\n"
        )

        if stage == "plan":
            revision_guidance = (
                "You are revising an existing DFT workflow plan based on new feedback or results.\n"
                "- Preserve sound steps from the prior plan when still applicable.\n"
                "- Emphasize updates that address the latest issues or goals.\n\n"
            )
        else:
            revision_guidance = ""

        directive = f"{revision_guidance}{base_guidance}"

        system_prompt = SystemMessage(
            content=(
                f"{directive}\n"
                "- Note assumptions if information is missing.\n"
                "- Avoid fabricating literature data; only reference provided context."
                f"{context_clause}"
            )
        )

        if messages and isinstance(messages[0], SystemMessage):
            messages[0] = system_prompt
        else:
            messages.insert(0, system_prompt)

        response = await llm_with_tools.ainvoke(messages)

        updated: PlannerState = {
            **state,
            "messages": [response],
            "stage": "done",
        }
        return updated

    def _build_graph(self):
        workflow = StateGraph(PlannerState)
        workflow.add_node("llm", self._llm_node)
        workflow.add_edge(START, "llm")
        workflow.add_edge("llm", END)
        return workflow.compile(checkpointer=MemorySaver())


def create_materials_planner_agent():
    return MaterialsPlannerAgent().graph


materials_planner_agent = create_materials_planner_agent()
