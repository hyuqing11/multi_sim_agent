from dataclasses import dataclass
from typing import Callable, Optional

from langgraph.graph.state import CompiledStateGraph

from backend.core import AgentInfo

DEFAULT_AGENT = "chatbot"


@dataclass
class AgentConfig:
    description: str
    factory: Callable[[], CompiledStateGraph]
    _cached_graph: Optional[CompiledStateGraph] = None

    def get_graph(self) -> CompiledStateGraph:
        """Lazy-load the graph when first accessed."""
        if self._cached_graph is None:
            self._cached_graph = self.factory()
        return self._cached_graph


def _create_chatbot():
    from backend.agents.library.chatbot import chatbot

    return chatbot


def _create_vasp_structure_agent():
    from backend.agents.library.vasp_agent import structure_agent

    return structure_agent


def _create_vasp_input_agent():
    from backend.agents.library.vasp_agent import input_agent

    return input_agent


def _create_vasp_pipeline_agent():
    from backend.agents.library.vasp_agent import vasp_pipeline_agent

    return vasp_pipeline_agent


def _create_hpc_agent():
    from backend.agents.library.hpc_agent import create_hpc_agent

    return create_hpc_agent()


def _create_literature_agent():
    from backend.agents.library.literature_agent import create_literature_agent

    return create_literature_agent()


def _create_materials_planner():
    from backend.agents.library.materials_planner import create_materials_planner_agent

    return create_materials_planner_agent()


def _create_orchestrator():
    from backend.agents.library.orchestrator import create_orchestrator_agent

    return create_orchestrator_agent()


agent_configs: dict[str, AgentConfig] = {
    "chatbot": AgentConfig(
        description="Assistant for DFT workflows, structure generation, QE input creation, SLURM job management, and materials science calculations",
        factory=_create_chatbot,
    ),
    "vasp_structure_agent": AgentConfig(
        description="Dedicated structure-generation agent for VASP workflows",
        factory=_create_vasp_structure_agent,
    ),
    "vasp_input_agent": AgentConfig(
        description="Input-file preparation agent for VASP or LAMMPS workflows",
        factory=_create_vasp_input_agent,
    ),
    "vasp_pipeline_agent": AgentConfig(
        description="Two-stage VASP pipeline that chains structure and input agents",
        factory=_create_vasp_pipeline_agent,
    ),
    "hpc_agent": AgentConfig(
        description="Autonomous HPC job submission agent with MCP-backed tool control",
        factory=_create_hpc_agent,
    ),
    "literature_agent": AgentConfig(
        description="CROW-backed literature research agent for computational materials questions",
        factory=_create_literature_agent,
    ),
    "materials_planner": AgentConfig(
        description="DFT planning agent that transforms literature findings into actionable workflows",
        factory=_create_materials_planner,
    ),
    "materials_orchestrator": AgentConfig(
        description="High-level orchestrator chaining literature search and DFT planning",
        factory=_create_orchestrator,
    ),
}


def get_agent(agent_id: str) -> CompiledStateGraph:
    """Get an agent by ID.

    Args:
        agent_id: The unique identifier for the agent

    Returns:
        The compiled state graph for the agent

    Raises:
        ValueError: If agent_id is not found
    """
    if agent_id not in agent_configs:
        available = ", ".join(agent_configs.keys())
        raise ValueError(
            f"Unknown agent: '{agent_id}'. Available agents: {available}"
        )
    return agent_configs[agent_id].get_graph()


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=config.description)
        for agent_id, config in agent_configs.items()
    ]
