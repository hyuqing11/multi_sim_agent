"""Literature agent graph that wraps the FutureHouse / CROW workflow.

The original implementation exposed an imperative `LiteratureAgent` class that
could not be plugged into the shared LangGraph framework. This rewrite exposes
the same capability through a LangGraph state machine with a tool node so the
agent can participate in the unified agent registry.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import time
from futurehouse_client import FutureHouseClient, JobNames
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from backend.agents.llm import get_model, settings
from backend.utils.workspace import async_get_workspace_path


CACHE_FILE = Path(__file__).with_name("literature_cache.json")
_SIM_THRESHOLD = 0.8
_REQUEST_TIMEOUT_S = 15  # safety timeout for individual API calls
_POLL_INTERVAL_S = 50
_DEFAULT_WAIT_S = 5000


class LiteratureState(MessagesState, total=False):
    """Agent state carrying messages plus optional workspace context."""

    working_directory: str
    thread_id: str
    stage: str


_embedder_lock = asyncio.Lock()
_embedder: Optional[OpenAIEmbeddings] = None


async def _get_embedder() -> OpenAIEmbeddings:
    """Create the embedding client lazily to avoid import costs on startup."""

    global _embedder
    if _embedder is not None:
        return _embedder

    async with _embedder_lock:
        if _embedder is None:
            _embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    return _embedder


def _normalize_query(query: str) -> str:
    text = query.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[-_/]", " ", text)
    return text


class LiteratureService:
    """Handles cache, embeddings, and FutureHouse requests."""

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.cache: Dict[str, Any] = {}
        if cache_path.exists():
            try:
                self.cache = json.loads(cache_path.read_text())
            except json.JSONDecodeError:
                self.cache = {}
        self._index: list[dict[str, Any]] = []
        for raw_query in self.cache:
            self._index.append({"norm": _normalize_query(raw_query), "raw": raw_query, "emb": None})

        api_key = os.getenv("CROW_API_KEY")
        if not api_key:
            raise RuntimeError("CROW_API_KEY environment variable is required for literature agent")
        self.client = FutureHouseClient(api_key=api_key)

    def _save_cache(self) -> None:
        self.cache_path.write_text(json.dumps(self.cache, indent=2))

    async def _lookup_cache(self, query: str) -> Optional[dict[str, Any]]:
        if query in self.cache:
            return self.cache[query]

        norm = _normalize_query(query)
        for raw_query, value in self.cache.items():
            if _normalize_query(raw_query) == norm:
                return value

        embedder = await _get_embedder()
        query_emb = embedder.embed_query(norm)

        best_sim, best_query = -1.0, None
        for item in self._index:
            if item["emb"] is None:
                item["emb"] = embedder.embed_query(item["norm"])
            denom = np.linalg.norm(query_emb) * np.linalg.norm(item["emb"])
            similarity = float(np.dot(query_emb, item["emb"]) / denom) if denom else 0.0
            if similarity > best_sim:
                best_sim, best_query = similarity, item["raw"]

        if best_sim >= _SIM_THRESHOLD and best_query and best_query in self.cache:
            return self.cache[best_query]
        return None

    def _success_payload(self, task_id: str, task_status: Any, query: str) -> dict[str, Any]:
        return {
            "status": "success",
            "task_id": task_id,
            "formatted_answer": getattr(task_status, "formatted_answer", None),
            "json": task_status.model_dump_json() if hasattr(task_status, "model_dump_json") else None,
            "has_successful_answer": getattr(task_status, "has_successful_answer", True),
            "search_results": getattr(task_status, "search_results", []),
            "query": query,
        }

    async def query(self, query: str, max_wait_time: int = _DEFAULT_WAIT_S) -> dict[str, Any]:
        if not query or not isinstance(query, str):
            return {"status": "error", "message": "Invalid question format."}

        cached = await self._lookup_cache(query)
        if cached:
            return {"status": "cached", "data": cached}

        llm = ChatOpenAI(model="gpt-4.1", temperature=0.0)
        instruction = (
            "You are an assistant that rewrites vague research questions into detailed CROW queries.\n\n"
            "User question: {query}\n\n"
            "Your task:\n"
            "- Reconstruct the question so it explicitly requests:\n"
            "  1) Previous results from both experiments and simulations related to the property.\n"
            "  2) How to compute this property using DFT, including validated and recommended DFT parameters.\n"
            "- Output the final query in natural question form.\n"
            "- Do not add commentary or explanation, only the rewritten query.\n"
        )
        rewritten = llm.invoke(instruction.format(query=query)).content

        loop = asyncio.get_running_loop()
        try:
            task_id = await asyncio.wait_for(
                loop.run_in_executor(
                    None, self.client.create_task, {"name": JobNames.FALCON, "query": rewritten}
                ),
                timeout=_REQUEST_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            return {"status": "error", "message": "Timed out creating CROW task"}
        except Exception as exc:
            return {"status": "error", "message": f"Failed to create CROW task: {exc}"}

        try:
            task_status = await asyncio.wait_for(
                loop.run_in_executor(None, self.client.get_task, task_id),
                timeout=_REQUEST_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            return {
                "status": "error",
                "message": "Timed out retrieving initial task status",
                "task_id": task_id,
            }
        except Exception as exc:
            return {
                "status": "error",
                "message": f"Failed to fetch initial task status: {exc}",
                "task_id": task_id,
            }
        effective_wait = max_wait_time

        if task_status.status == "success":
            payload = self._success_payload(task_id, task_status, query)
            self.cache[query] = payload
            self._save_cache()
            return payload

        max_attempts = max(1, effective_wait // _POLL_INTERVAL_S)
        for _ in range(max_attempts):
            await asyncio.sleep(_POLL_INTERVAL_S)
            try:
                task_status = await asyncio.wait_for(
                    loop.run_in_executor(None, self.client.get_task, task_id),
                    timeout=_REQUEST_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                return {
                    "status": "error",
                    "message": f"Failed to poll task status: {exc}",
                    "task_id": task_id,
                }
            if task_status.status == "success":
                payload = self._success_payload(task_id, task_status, query)
                self.cache[query] = payload
                self._save_cache()
                return payload
            if task_status.status in {"FAILED", "ERROR", "error"}:
                return {"status": "error", "message": f"Query failed: {task_status.status}", "task_id": task_id}

        return {
            "status": "timeout",
            "message": f"No result within {effective_wait}s (task still running)",
            "task_id": task_id,
        }


_service_lock = asyncio.Lock()
_service: Optional[LiteratureService] = None


async def _get_service() -> LiteratureService:
    global _service
    if _service is not None:
        return _service

    async with _service_lock:
        if _service is None:
            _service = LiteratureService(CACHE_FILE)
    return _service


@tool("query_literature")
async def query_literature_tool(query: str) -> str:
    """Search the CROW literature agent for detailed DFT-related information."""

    service = await _get_service()
    try:
        result = await service.query(query)
    except Exception as exc:  # pragma: no cover - safety net
        return f"ERROR: Literature query failed: {exc}"

    return json.dumps(result, indent=2)


LITERATURE_TOOLS = [query_literature_tool]


async def _ensure_thread_context(state: LiteratureState, config: RunnableConfig) -> LiteratureState:
    configurable = {}
    if isinstance(config, dict):
        configurable = config.get("configurable", {})
    else:
        configurable = getattr(config, "configurable", {})
    thread_id = state.get("thread_id") or configurable.get("thread_id")
    if not thread_id or state.get("working_directory"):
        return state

    workspace = await async_get_workspace_path(thread_id)
    new_state: LiteratureState = {**state, "thread_id": thread_id, "working_directory": str(workspace)}
    return new_state


class LiteratureAgentGraph:
    def __init__(self) -> None:
        self.tools = LITERATURE_TOOLS
        self.graph = self._build_graph()

    async def _llm_node(self, state: LiteratureState, config: RunnableConfig) -> dict[str, Any]:
        state = await _ensure_thread_context(state, config)
        configurable = {}
        if isinstance(config, dict):
            configurable = config.get("configurable", {})
        else:
            configurable = getattr(config, "configurable", {})
        model_name = configurable.get("model", settings.DEFAULT_MODEL)
        llm = get_model(model_name)
        stage = state.get("stage", "research")
        if stage == "research":
            llm_with_tools = llm.bind_tools(self.tools)
        else:
            llm_with_tools = llm

        if stage == "research":
            system_content = (
                "You are a literature research assistant focused on computational materials science.\n"
                "- Use the query_literature tool to search for prior results, recommended DFT parameters, and related work.\n"
                "- Do not fabricate data; rely on the tool.\n"
            )
        else:
            system_content = (
                "You have already gathered literature findings (see prior tool output).\n"
                "Summarize key experimental/simulation results, recommended DFT settings, and cite the task ID or cache status." 
                " Provide a concise narrative that can feed into DFT planning."
            )

        system_message = SystemMessage(content=system_content)

        messages = list(state.get("messages", []))
        if messages and isinstance(messages[0], SystemMessage):
            messages[0] = system_message
        else:
            messages.insert(0, system_message)

        response = await llm_with_tools.ainvoke(messages)
        updated: LiteratureState = {**state, "messages": [response]}
        if getattr(response, "tool_calls", None):
            updated["stage"] = "research"
        return updated

    async def _tool_node(self, state: LiteratureState, config: RunnableConfig) -> dict[str, Any]:
        state = await _ensure_thread_context(state, config)
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", [])
        observations: list[ToolMessage] = []
        tools_by_name = {tool.name: tool for tool in self.tools}

        for call in tool_calls:
            tool = tools_by_name.get(call["name"])
            if not tool:
                observations.append(
                    ToolMessage(content=f"ERROR: Tool {call['name']} not found", tool_call_id=call["id"])
                )
                continue
            try:
                result = await tool.ainvoke(call["args"])
                observations.append(ToolMessage(content=result, tool_call_id=call["id"]))
            except Exception as exc:  # pragma: no cover - defensive guard
                observations.append(
                    ToolMessage(content=f"ERROR: Tool {call['name']} failed: {exc}", tool_call_id=call["id"])
                )

        updated: LiteratureState = {**state, "messages": observations, "stage": "summary"}
        return updated

    def _build_graph(self):
        workflow = StateGraph(LiteratureState)
        workflow.add_node("llm", self._llm_node)
        workflow.add_node("tools", self._tool_node)
        workflow.add_edge(START, "llm")
        workflow.add_conditional_edges(
            "llm",
            self._should_continue,
            {"tools": "tools", "end": END},
        )
        workflow.add_edge("tools", "llm")
        return workflow.compile(checkpointer=MemorySaver())

    def _should_continue(self, state: LiteratureState) -> str:
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", []):
            return "tools"
        return "end"


def create_literature_agent():
    """Factory returning the compiled literature agent graph."""

    return LiteratureAgentGraph().graph


literature_agent = create_literature_agent()
