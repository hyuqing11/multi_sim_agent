"""Helper utilities for preparing VASP/LAMMPS inputs."""

from __future__ import annotations

import inspect
import json
import os
import re
import shutil
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from ase.atoms import Atoms
from ase.io import read, write
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode

from backend.agents.llm import get_model, settings
from backend.utils.workspace import async_get_workspace_path

from backend.agents.library.vasp_agent.input_tools import (
    create_folder,
    list_potcar_sets,
    select_potcar_source,
    write_lammps_input,
    write_vasp_incar,
    write_vasp_kpoints,
)
from backend.agents.library.vasp_agent.prompt import lammps_input_prompt, vasp_input_prompt
from backend.agents.library.vasp_agent.stage_prompts import COLLABORATION_REMINDER, INPUT_STAGE_PROMPT
from backend.agents.library.vasp_agent.utils import ENGINE_VASP, ENGINE_LAMMPS

from backend.agents.library.vasp_agent.utils import _normalize_engine


load_dotenv()


def find_structure_filename(messages: Sequence[BaseMessage], fallback: Optional[str] = None) -> Optional[str]:
    """Extract the latest ``STRUCTURE_SAVED`` marker from tool output messages."""

    for message in reversed(messages):
        content = getattr(message, "content", "")
        if isinstance(content, str) and "STRUCTURE_SAVED:" in content:
            snippet = content.split("STRUCTURE_SAVED:", 1)[1]
            candidate = snippet.strip().splitlines()[0].strip().strip("'\"")
            if candidate:
                return candidate
    return fallback


@dataclass
class PreparedStructure:
    atoms: Atoms
    artifact_path: Path
    artifact_filename: str
    artifact_content: str
    source_path: Path


def prepare_structure_artifacts(run_dir: Path, structure_filename: str, engine: str) -> PreparedStructure:
    """Read a saved structure and emit canonical artefacts for downstream tools."""

    source_path = run_dir / structure_filename
    atoms = read(source_path)

    artifact_filename = "POSCAR"
    format_key = "vasp"

    if engine == ENGINE_LAMMPS:
        artifact_filename = "structure_lammps.data"
        format_key = "lammps-data"

    artifact_path = run_dir / artifact_filename
    write(artifact_path, atoms, format=format_key)
    try:
        artifact_content = artifact_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        artifact_content = artifact_path.read_text(encoding="latin-1")

    return PreparedStructure(
        atoms=atoms,
        artifact_path=artifact_path,
        artifact_filename=artifact_filename,
        artifact_content=artifact_content,
        source_path=source_path,
    )


def summarize_structure(atoms: Atoms) -> str:
    """
    Provide a short structured summary describing an Atoms object.

    Args:
        atoms: ASE Atoms object to summarize

    Returns:
        Formatted string with composition, count, cell, and PBC info
    """

    composition = atoms.get_chemical_formula(mode="hill")
    count = len(atoms)
    cell = atoms.cell if atoms.cell is not None else "<no cell>"
    pbc = atoms.pbc.tolist() if hasattr(atoms, "pbc") else []
    return (
        f"- Composition: {composition}\n"
        f"- Number of atoms: {count}\n"
        f"- Cell (Å): {cell}\n"
        f"- Periodic boundary conditions: {pbc}"
    ).format(composition=composition, count=count, cell=cell, pbc=pbc)


class InputAgentState(MessagesState, total=False):
    thread_id: str
    working_directory: str
    run_directory: str
    query: str
    engine: str
    structure_source_path: str
    structure_artifact_path: str
    structure_artifact_filename: str
    structure_summary: str
    structure_prompt_blob: str
    input_directories: Dict[str, str]
    potcar_source_path: str
    potcar_source_filename: str
    potcar_source_kind: str
    potcar_candidate_file: str
    last_tool_output: str
    plan_data: Dict[str, Any]
    dft_parameters: Dict[str, Any]


# Tools available per engine. `create_folder` is shared to keep run directories consistent.
input_tools = [
    create_folder,
    list_potcar_sets,
    select_potcar_source,
    write_vasp_incar,
    write_vasp_kpoints,
    write_lammps_input,
]

POTCAR_JSON_KEYS = (
    "potcar_source_path",
    "potcar_path",
    "potential_path",
    "vasp_potcar_path",
    "pseudopotential_path",
    "potcar_file",
    "vasp_potential_path",
    "potcar",
)

POTCAR_ENV_KEYS = (
    "POTCAR_GGA_PATH",
    "POTCAR_PBE_PATH",
    "POTCAR_LDA_PATH",
    "POTCAR_PATH",
)



async def _ensure_context(state: InputAgentState, config: RunnableConfig) -> tuple[InputAgentState, Dict[str, Any]]:
    updates: Dict[str, Any] = {}

    configurable = getattr(config, "configurable", {}) if not isinstance(config, dict) else config.get("configurable", {})
    thread_id = state.get("thread_id") or configurable.get("thread_id")
    if thread_id and state.get("thread_id") != thread_id:
        updates["thread_id"] = thread_id

    if thread_id and not state.get("working_directory"):
        workspace = await async_get_workspace_path(thread_id)
        updates["working_directory"] = str(workspace)
        updates["run_directory"] = str(workspace)

    run_dir = updates.get("run_directory") or state.get("run_directory") or state.get("working_directory")
    if run_dir and state.get("run_directory") != run_dir:
        updates["run_directory"] = run_dir

    if "query" not in state:
        updates["query"] = _last_human_content(state.get("messages", []))


    if "input_directories" not in state:
        updates.setdefault("input_directories", {})

    merged = {**state, **updates}

    required_keys = [
        "structure_artifact_path",
        "structure_artifact_filename",
        "structure_prompt_blob",
        "structure_summary",
    ]
    missing = [key for key in required_keys if not merged.get(key)]
    if missing:
        raise ValueError(
            "Input agent requires structure context before starting: missing "
            + ", ".join(missing)
        )

    return merged, updates


def _last_human_content(messages: Sequence[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""

def _coerce_to_path_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (list, tuple)):
        for item in value:
            candidate = _coerce_to_path_string(item)
            if candidate:
                return candidate
        return None
    if isinstance(value, dict):
        for candidate_key in ("path", "file", "location", "value"):
            candidate = _coerce_to_path_string(value.get(candidate_key))
            if candidate:
                return candidate
        return None
    return str(value)

def _extract_potcar_path(payload: Dict[str, Any]) -> Optional[str]:
    for key in POTCAR_JSON_KEYS:
        if key not in payload:
            continue
        candidate = _coerce_to_path_string(payload.get(key))
        if candidate:
            return candidate
    return None


def _build_system_prompt(state: InputAgentState) -> str:
    engine = _normalize_engine(state.get("engine"))
    return INPUT_STAGE_PROMPT.format(
        reminder=COLLABORATION_REMINDER,
        structure_path=state.get("structure_source_path", "unknown"),
        summary=state.get("structure_summary", "No summary available."),
        engine=engine.upper(),
        run_dir=state.get("run_directory", "<unknown>"),
    ).strip()


def _engine_specific_prompt(state: InputAgentState) -> str:
    run_dir = state.get("run_directory", "<unknown>")
    query = state.get("query", "")
    engine = _normalize_engine(state.get("engine"))

    # Build parameter context from plan_data or dft_parameters
    parameter_context = ""
    dft_params = state.get("dft_parameters", {})
    plan_data = state.get("plan_data", {})

    # Extract parameters from plan_data if available
    if isinstance(plan_data, dict) and plan_data.get("recommended_parameters"):
        dft_params = plan_data.get("recommended_parameters", dft_params)

    if dft_params:
        parameter_context = "\n\n## Recommended DFT Parameters from Planner:\n"
        parameter_context += "The planning agent has analyzed the scientific literature and recommends the following parameters:\n"
        parameter_context += json.dumps(dft_params, indent=2)
        parameter_context += "\n\nIMPORTANT: Use these recommended parameters as a starting point, but adjust based on the specific structure and calculation type."

    if engine == ENGINE_LAMMPS:
        return lammps_input_prompt.format(
            structure_summary=state.get("structure_summary", "Structure summary unavailable."),
            structure_filename=state.get("structure_artifact_filename", "structure_lammps.data"),
            run_dir=run_dir,
            query=query,
        ) + parameter_context

    return vasp_input_prompt.format(
        structure_content=state.get("structure_prompt_blob", ""),
        run_dir=run_dir,
        query=query,
    ) + parameter_context





async def call_model(state: InputAgentState, config: RunnableConfig) -> Dict[str, Any]:
    updates = {}
    messages = list(state.get("messages", []))
    if "run_directory" not in state and "run_dir" in state:
        updates["run_directory"] = state["run_dir"]

    if "working_directory" not in state and "run_directory" in {**state, **updates}:
        updates["working_directory"] = updates.get("run_directory") or state.get("run_directory")

    required = ["structure_artifact_path", "structure_artifact_filename",
                "structure_prompt_blob", "structure_summary"]
    already_initialized = all(state.get(k) for k in required)
    if not already_initialized:
        is_first_call = len(messages) == 1 and isinstance(messages[0], HumanMessage)
        if is_first_call:
            try:
                initial_data = json.loads(messages[0].content)
                # Extract structure context
                for key in [
                    "structure_artifact_path",
                    "structure_artifact_filename",
                    "structure_prompt_blob",
                    "structure_summary",
                    "structure_source_path",
                    "run_dir",
                    "run_directory",
                    "working_directory",
                    "engine",
                    "potcar_source_path",
                    "potcar_source_filename",
                    "potcar_source_kind",
                    "potcar_candidate_file",
                ]:
                    if key in initial_data and not state.get(key):
                        value = initial_data[key]
                        if key in {"run_dir", "run_directory", "working_directory", "structure_artifact_path", "structure_source_path", "potcar_source_path", "potcar_candidate_file"} and value is not None:
                            value = str(value)
                        if key == "potcar_source_path" and value is not None:
                            value = str(Path(value).expanduser())
                        if key == "potcar_candidate_file" and value is not None:
                            value = str(Path(value).expanduser())
                        if key == "structure_prompt_blob" and value is not None:
                            value = str(value)
                        if key == "engine":
                            value = _normalize_engine(value)
                        updates[key] = value
                        if key == "potcar_source_path" and value:
                            resolved_path = Path(value)
                            if resolved_path.is_dir():
                                updates.setdefault("potcar_source_kind", "directory")
                                updates.setdefault("potcar_source_filename", "POTCAR")
                            else:
                                updates.setdefault("potcar_source_kind", "file")
                                updates.setdefault("potcar_source_filename", resolved_path.name)

                if not updates.get("potcar_source_filename") and initial_data.get("potcar_source_filename"):
                    updates["potcar_source_filename"] = initial_data["potcar_source_filename"]
                if not updates.get("potcar_source_kind") and initial_data.get("potcar_source_kind"):
                    updates["potcar_source_kind"] = initial_data["potcar_source_kind"]
                if not updates.get("potcar_candidate_file") and initial_data.get("potcar_candidate_file"):
                    updates["potcar_candidate_file"] = str(Path(initial_data["potcar_candidate_file"]).expanduser())

                if not updates.get("potcar_source_path") and not state.get("potcar_source_path"):
                    potcar_candidate = _extract_potcar_path(initial_data)
                    if potcar_candidate:
                        potcar_path = str(Path(potcar_candidate).expanduser())
                        updates["potcar_source_path"] = potcar_path
                        resolved_path = Path(potcar_path)
                        if resolved_path.is_dir():
                            updates["potcar_source_kind"] = "directory"
                            updates["potcar_source_filename"] = "POTCAR"
                        else:
                            updates["potcar_source_kind"] = "file"
                            updates["potcar_source_filename"] = resolved_path.name

                # Extract query and replace message
                if "query" in initial_data:
                    updates["query"] = initial_data["query"]
                    messages = [HumanMessage(content=initial_data["query"],
                                             additional_kwargs={"internal": True})]

            except json.JSONDecodeError:
                # Not JSON, treat as plain query
                if "query" not in state:
                    updates["query"] = messages[0].content

        # Set run_directory
        merged = {**state, **updates}
        if "run_directory" not in merged:
            if "run_dir" in merged:
                updates["run_directory"] = merged["run_dir"]
        merged = {**state, **updates}
        engine = state.get("engine") or _normalize_engine(merged["engine"]) or ENGINE_VASP
        if engine != merged.get("engine"):
            updates["engine"] = engine
            merged = {**state, **updates}

        missing = [k for k in required if not merged.get(k)]

        if missing:
            error_msg = f"Input agent requires: {', '.join(missing)}"
            return {
                **updates,
                "messages": [AIMessage(content=f"ERROR: {error_msg}")]
            }

        # Initialize other fields
        if "input_directories" not in merged:
            updates["input_directories"] = {}

        merged = {**state, **updates}
        engine_for_log = _normalize_engine(merged.get("engine"))
        print(f"✓ Input agent initialized: engine={engine_for_log}")

    current_state = {**state, **updates}
    normalized_engine = _normalize_engine(current_state.get("engine"))
    if normalized_engine != current_state.get("engine"):
        updates["engine"] = normalized_engine
        current_state = {**state, **updates}
    system_prompt = _build_system_prompt(current_state)

    # Prepare messages with system prompt
    if messages and isinstance(messages[0], SystemMessage):
        messages[0] = SystemMessage(content=system_prompt)
    else:
        messages.insert(0, SystemMessage(content=system_prompt))

    # Add engine-specific prompt
    engine_prompt = SystemMessage(content=_engine_specific_prompt(current_state))
    if len(messages) >= 2 and isinstance(messages[1], SystemMessage):
        messages[1] = engine_prompt
    else:
        messages.insert(1, engine_prompt)


    model_name = getattr(config, "configurable", {}).get("model") if not isinstance(config, dict) else config.get(
        "configurable", {}).get("model")
    llm = get_model(model_name or settings.DEFAULT_MODEL).bind_tools(input_tools)
    response = await llm.ainvoke(messages, config=config)

    return {**updates, "messages": [response]}


def _tool_message_to_text(message: ToolMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                part = item.get("text") or item.get("content")
                if part:
                    parts.append(str(part))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _tool_name_for_message(messages: Sequence[BaseMessage], tool_message: ToolMessage) -> Optional[str]:
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        if not getattr(msg, "tool_calls", None):
            continue
        for call in msg.tool_calls:
            if call.get("id") == tool_message.tool_call_id:
                return call.get("name")
    return None


def _extract_folder_path(text: str) -> Optional[Path]:
    if not text:
        return None
    raw = text.strip()
    if not raw:
        return None

    # Try JSON payloads first
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
            for key in ("folder_path", "path", "result", "output"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return Path(value.strip()).expanduser()
        except json.JSONDecodeError:
            pass

    if "Created folder" in raw:
        raw = raw.split("Created folder", 1)[1]
        raw = raw.lstrip(": ")

    candidate = raw.strip().strip("'\"")
    if not candidate:
        return None
    return Path(candidate).expanduser()


def _copy_text_file(src: Path, dst: Path) -> None:
    try:
        content = src.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = src.read_text(encoding="latin-1")
    dst.write_text(content, encoding="utf-8")


def _resolve_potcar_source(state: InputAgentState) -> Optional[Path]:
    engine = _normalize_engine(state.get("engine"))
    if engine != ENGINE_VASP:
        return None

    raw_value: Any = state.get("potcar_source_path")
    if not raw_value:
        for env_key in POTCAR_ENV_KEYS:
            env_value = os.environ.get(env_key)
            if env_value:
                raw_value = env_value
                break
    if not raw_value:
        return None

    candidate_strings: list[str] = []

    def _collect_candidate_strings(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            parts = value.split(os.pathsep) if os.pathsep in value else [value]
            for part in parts:
                cleaned = part.strip()
                if cleaned:
                    candidate_strings.append(cleaned)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                _collect_candidate_strings(item)
            return
        if isinstance(value, dict):
            for candidate_key in ("path", "file", "location", "value"):
                if candidate_key in value:
                    _collect_candidate_strings(value[candidate_key])
            return
        candidate_strings.append(str(value))

    _collect_candidate_strings(raw_value)

    seen: set[str] = set()
    candidates: list[Path] = []

    def _append_candidate(path_str: str) -> None:
        expanded_env = os.path.expandvars(path_str)
        normalized = str(Path(expanded_env).expanduser())
        if normalized in seen:
            return
        seen.add(normalized)
        candidates.append(Path(normalized))

    for candidate_str in candidate_strings:
        if not candidate_str:
            continue
        _append_candidate(candidate_str)
        candidate_path = Path(os.path.expandvars(candidate_str))
        if not candidate_path.is_absolute():
            for base_key in ("run_directory", "working_directory"):
                base_dir = state.get(base_key)
                if base_dir:
                    derived = str(Path(base_dir) / candidate_str)
                    _append_candidate(derived)

    alt_filename = state.get("potcar_source_filename")
    base_dirs: list[Path] = []

    for candidate in candidates:
        expanded = candidate
        if expanded.is_file():
            return expanded
        if expanded.is_dir():
            base_dirs.append(expanded)
            direct = expanded / "POTCAR"
            if direct.is_file():
                return direct
            if alt_filename:
                alt_candidate = expanded / alt_filename
                if alt_candidate.is_file():
                    return alt_candidate

    if not base_dirs:
        return None

    return _build_combined_potcar(base_dirs, state, alt_filename)


def _get_species_order(structure_path: Path) -> list[str]:
    try:
        atoms = read(structure_path)
    except Exception:
        return []
    order: list[str] = []
    for symbol in atoms.get_chemical_symbols():
        if symbol not in order:
            order.append(symbol)
    return order


def _extract_potcar_labels_from_messages(messages: Sequence[BaseMessage], species_order: Sequence[str]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    if not messages:
        return labels

    species_lookup = {element.lower(): element for element in species_order}
    path_token_pattern = re.compile(r'/([A-Za-z0-9_]+)/POTCAR', re.IGNORECASE)

    for message in reversed(messages):
        text = getattr(message, "content", "")
        if not isinstance(text, str):
            continue

        for token in path_token_pattern.findall(text):
            normalized = token.strip()
            if not normalized:
                continue
            base_match = re.match(r'([A-Z][a-z]?)(?:[_-].*)?$', normalized)
            if base_match:
                element_key = base_match.group(1).lower()
                element = species_lookup.get(element_key)
                if element and element not in labels:
                    labels[element] = normalized

        lower_text = text.lower()
        for element in species_order:
            if element in labels:
                continue
            pattern = re.compile(rf'{element.lower()}\s*[:=-]\s*(?P<label>[a-z0-9_./-]+)')
            match = pattern.search(lower_text)
            if match:
                candidate = match.group("label").split()[0]
                candidate = candidate.strip().strip(",.;")
                if candidate:
                    labels[element] = candidate

    return labels


def _derive_potcar_labels(state: InputAgentState, species_order: Sequence[str]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for element in species_order:
        env_key = f"POTCAR_LABEL_{element.upper()}"
        env_value = os.environ.get(env_key)
        if not env_value:
            env_value = os.environ.get(f"POTCAR_LABEL_{element.capitalize()}")
        if env_value:
            labels[element] = env_value.strip()

    message_labels = _extract_potcar_labels_from_messages(state.get("messages", []), species_order)
    for element, label in message_labels.items():
        labels.setdefault(element, label)
    return labels


def _find_potcar_file(base_dirs: Sequence[Path], label: str, element: str) -> Optional[Path]:
    if not label:
        return None

    label_path = Path(label)
    if label_path.is_absolute():
        if label_path.is_file():
            return label_path
        if label_path.is_dir():
            potcar_file = label_path / "POTCAR"
            if potcar_file.is_file():
                return potcar_file
        return None

    variant_strings = [label]
    normalized = label.replace('\\', '/').strip()
    if normalized and normalized not in variant_strings:
        variant_strings.append(normalized)
    if normalized.lower() != normalized:
        variant_strings.append(normalized.lower())
    if normalized.upper() != normalized:
        variant_strings.append(normalized.upper())
    if '/' not in normalized:
        variant_strings.append(f"{element}/{normalized}")
    variant_strings.append(element)

    checked: set[str] = set()
    for variant in variant_strings:
        if not variant:
            continue
        variant = variant.strip()
        if variant in checked:
            continue
        checked.add(variant)
        relative_path = Path(variant)
        for base_dir in base_dirs:
            target = base_dir / relative_path
            if target.is_file() and target.name.upper().startswith("POTCAR"):
                return target
            if target.is_dir():
                potcar_candidate = target / "POTCAR"
                if potcar_candidate.is_file():
                    return potcar_candidate

    element_upper = element.upper()
    candidate_prefixes = {element_upper}
    if "normalized" in locals():
        candidate_prefixes.add(normalized.upper())
    suffixes = {"_SV", "_PV", "_GW", "_US", "_PBE", "_LDA"}

    for base_dir in base_dirs:
        if not base_dir.exists() or not base_dir.is_dir():
            continue
        try:
            for child in base_dir.iterdir():
                if not child.is_dir():
                    continue
                name_upper = child.name.upper()
                if name_upper in candidate_prefixes:
                    candidate_file = child / "POTCAR"
                    if candidate_file.is_file():
                        return candidate_file
                if any(name_upper.startswith(f"{element_upper}") or name_upper.endswith(suffix)
                       for suffix in suffixes):
                    candidate_file = child / "POTCAR"
                    if candidate_file.is_file():
                        return candidate_file
        except OSError:
            continue
    return None


def _build_combined_potcar(base_dirs: Sequence[Path], state: InputAgentState, alt_filename: Optional[str]) -> Optional[Path]:
    structure_path_str = state.get("structure_artifact_path")
    if not structure_path_str:
        return None

    structure_path = Path(structure_path_str)
    species_order = _get_species_order(structure_path)
    if not species_order:
        return None

    labels = _derive_potcar_labels(state, species_order)
    potcar_files: list[Path] = []

    for element in species_order:
        label = labels.get(element, element)
        potcar_file = _find_potcar_file(base_dirs, label, element)
        if not potcar_file and label != element:
            potcar_file = _find_potcar_file(base_dirs, element, element)
        if not potcar_file and alt_filename:
            potcar_file = _find_potcar_file(base_dirs, alt_filename, element)
        if not potcar_file:
            return None
        potcar_files.append(potcar_file)

    run_directory = state.get("run_directory") or state.get("working_directory")
    if not run_directory:
        return None

    run_dir_path = Path(run_directory).resolve()
    run_dir_path.mkdir(parents=True, exist_ok=True)
    combined_path = run_dir_path / "POTCAR"

    try:
        with combined_path.open("wb") as combined_fh:
            for potcar_file in potcar_files:
                with Path(potcar_file).open("rb") as src_fh:
                    shutil.copyfileobj(src_fh, combined_fh)
        return combined_path
    except Exception:
        return None



def _copy_potcar_to_folder(potcar_source: Path, folder_path: Path) -> bool:

    destination = folder_path / "POTCAR"
    if destination.exists():
        return False
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(potcar_source, destination)
        return True
    except shutil.SameFileError:
        return False
    except Exception:
        return False



def _normalize_potcar_selection(path_str: str) -> Optional[Dict[str, str]]:
    if not path_str:
        return None
    expanded = Path(os.path.expandvars(path_str)).expanduser()
    if not expanded.exists():
        return None
    if expanded.is_dir():
        resolved_dir = expanded.resolve()
        payload: Dict[str, str] = {
            "potcar_source_path": str(resolved_dir),
            "potcar_source_filename": "POTCAR",
            "potcar_source_kind": "directory",
        }
        potcar_file = resolved_dir / "POTCAR"
        if potcar_file.is_file():
            payload["potcar_candidate_file"] = str(potcar_file.resolve())
        return payload
    if expanded.is_file():
        resolved = expanded.resolve()
        return {
            "potcar_source_path": str(resolved),
            "potcar_source_filename": resolved.name,
            "potcar_source_kind": "file",
        }
    return None



def _parse_potcar_selection_text(text: str) -> Optional[Dict[str, str]]:
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        candidate = _coerce_to_path_string(text)
        if candidate:
            return _normalize_potcar_selection(candidate)
        return None

    if isinstance(payload, dict):
        candidate = _extract_potcar_path(payload)
        if not candidate:
            candidate = _coerce_to_path_string(payload)
        if candidate:
            return _normalize_potcar_selection(candidate)
    elif isinstance(payload, list):
        for item in payload:
            candidate = _coerce_to_path_string(item)
            if candidate:
                selection = _normalize_potcar_selection(candidate)
                if selection:
                    return selection
    return None



def _potcar_updates_from_messages(messages: Sequence[BaseMessage]) -> Dict[str, str]:
    if not messages:
        return {}
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        tool_name = _tool_name_for_message(messages, message)
        if tool_name != "select_potcar_source":
            continue
        text = _tool_message_to_text(message)
        selection = _parse_potcar_selection_text(text)
        if selection:
            return selection
    return {}





def copy_structure_to_folders(state: InputAgentState) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}

    messages = state.get("messages", [])
    selection_updates = _potcar_updates_from_messages(messages)
    if selection_updates:
        updates.update(selection_updates)

    working_state = {**state, **updates}

    run_directory = working_state.get("run_directory") or working_state.get("working_directory")
    structure_path = working_state.get("structure_artifact_path")
    structure_name = working_state.get("structure_artifact_filename")
    if not run_directory or not structure_path or not structure_name:
        return updates

    engine = _normalize_engine(working_state.get("engine"))
    potcar_source = _resolve_potcar_source(working_state) if engine == ENGINE_VASP else None
    if potcar_source:
        potcar_source = potcar_source.resolve()
        normalized_potcar = str(potcar_source)
        if working_state.get("potcar_source_path") != normalized_potcar:
            updates["potcar_source_path"] = normalized_potcar
            updates["potcar_source_filename"] = potcar_source.name
            updates["potcar_source_kind"] = "file"
            updates["potcar_candidate_file"] = normalized_potcar
            working_state = {**working_state, **updates}

    run_dir_path = Path(run_directory).resolve()
    existing_directories = dict(working_state.get("input_directories", {}))
    known_paths = {Path(p).resolve() for p in existing_directories.values()}

    src_path = Path(structure_path).resolve()
    if not src_path.exists():
        return updates

    if potcar_source:
        for existing_path_str in existing_directories.values():
            existing_path = Path(existing_path_str).resolve()
            if not existing_path.exists() or not existing_path.is_dir():
                continue
            if run_dir_path not in existing_path.parents and existing_path != run_dir_path:
                continue
            _copy_potcar_to_folder(potcar_source, existing_path)

    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue

        tool_name = _tool_name_for_message(messages, message)
        if tool_name != "create_folder":
            continue

        text_content = _tool_message_to_text(message)
        folder_path = _extract_folder_path(text_content)
        if not folder_path:
            continue

        folder_path = folder_path.resolve()
        if not folder_path.exists() or not folder_path.is_dir():
            continue
        if run_dir_path not in folder_path.parents and folder_path != run_dir_path:
            continue
        if folder_path in known_paths:
            continue

        folder_name = folder_path.name
        dst_path = folder_path / structure_name

        try:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            _copy_text_file(src_path, dst_path)
        except Exception:
            continue

        if potcar_source:
            _copy_potcar_to_folder(potcar_source, folder_path)

        existing_directories[folder_name] = str(folder_path)
        known_paths.add(folder_path)

    if existing_directories != working_state.get("input_directories"):
        updates["input_directories"] = existing_directories

    return updates



input_workflow = StateGraph(InputAgentState)
input_workflow.add_node("model", call_model)
input_workflow.add_node("tools", ToolNode(input_tools))  # ← Built-in!
input_workflow.add_node("post_process", copy_structure_to_folders)  # ← Our custom logic

input_workflow.set_entry_point("model")
input_workflow.add_conditional_edges("model", tools_condition, ["tools", END])
input_workflow.add_edge("tools", "post_process")
input_workflow.add_edge("post_process", "model")

input_agent = input_workflow.compile()