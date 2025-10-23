"""Tool definitions for generating VASP and LAMMPS input files."""

from __future__ import annotations

from pathlib import Path
import json
import os
from typing import Any, Dict, Optional

from langchain_core.tools import tool


@tool
def create_folder(run_dir: str, folder_name: str) -> str:
    """Create a calculation subfolder inside ``run_dir``."""

    folder_path = (Path(run_dir) / folder_name).resolve()
    folder_path.mkdir(parents=True, exist_ok=True)
    return str(folder_path)


@tool
def write_lammps_input(folder_path: str, script_content: str) -> str:
    """Persist a LAMMPS input script inside ``folder_path``."""

    script_path = Path(folder_path) / "input.lammps"
    script_path.write_text(script_content.strip() + "\n", encoding="utf-8")
    return str(script_path)


@tool
def write_vasp_kpoints(folder_path: str, kpoints_content: str) -> str:
    """Persist a VASP KPOINTS file inside ``folder_path``."""

    kpoints_path = Path(folder_path) / "KPOINTS"
    kpoints_path.write_text(kpoints_content.strip() + "\n", encoding="utf-8")
    return str(kpoints_path)


@tool
def write_vasp_incar(folder_path: str, incar_content: str) -> str:
    """Persist a VASP INCAR file inside ``folder_path``."""

    incar_path = Path(folder_path) / "INCAR"
    incar_path.write_text(incar_content.strip() + "\n", encoding="utf-8")
    return str(incar_path)

POTCAR_ENV_KEYS = (
    "POTCAR_GGA_PATH",
    "POTCAR_PBE_PATH",
    "POTCAR_LDA_PATH",
    "POTCAR_PATH",
)


@tool
def list_potcar_sets(base_path: Optional[str] = None, depth: int = 2, include_files: bool = False) -> str:
    """List available POTCAR directories starting from the provided base path or defaults."""

    def _collect_roots() -> list[Path]:
        roots: list[Path] = []
        if base_path:
            roots.append(Path(os.path.expandvars(base_path)).expanduser())
            return roots
        seen: set[str] = set()
        for env_key in POTCAR_ENV_KEYS:
            value = os.environ.get(env_key)
            if not value:
                continue
            for part in value.split(os.pathsep):
                part = part.strip()
                if not part:
                    continue
                resolved = Path(os.path.expandvars(part)).expanduser()
                key = str(resolved.resolve()) if resolved.exists() else str(resolved)
                if key not in seen:
                    seen.add(key)
                    roots.append(resolved)
        return roots

    def _describe_path(path: Path, current_depth: int) -> Dict[str, Any]:
        description: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if not path.exists():
            return description
        description["type"] = "dir" if path.is_dir() else "file"
        if path.is_dir() and current_depth > 0:
            children: list[Dict[str, Any]] = []
            for child in sorted(path.iterdir(), key=lambda p: p.name):
                if child.name.startswith('.'):
                    continue
                if child.is_dir():
                    children.append(_describe_path(child, current_depth - 1))
                elif include_files:
                    children.append({"path": str(child), "type": "file"})
            if children:
                description["children"] = children
        return description

    roots = _collect_roots()
    if not roots:
        return json.dumps({"roots": [], "message": "No POTCAR paths configured."})

    payload = {"roots": [_describe_path(root, depth) for root in roots]}
    return json.dumps(payload, indent=2)


@tool
def select_potcar_source(path: str, allow_directory: bool = True) -> str:
    """Resolve and validate a POTCAR library path or specific file."""

    expanded = Path(os.path.expandvars(path)).expanduser()
    if not expanded.exists():
        raise ValueError(f"Path does not exist: {expanded}")

    if expanded.is_file():
        resolved = expanded.resolve()
        payload: Dict[str, Any] = {
            "potcar_source_path": str(resolved),
            "potcar_source_filename": resolved.name,
            "potcar_source_kind": "file",
        }
        return json.dumps(payload)

    resolved_dir = expanded.resolve()
    potcar_file = resolved_dir / "POTCAR"
    if not allow_directory and not potcar_file.is_file():
        raise ValueError(
            f"No POTCAR file found inside directory: {resolved_dir}. "
            "Pass allow_directory=True to use the directory as a library root."
        )

    payload = {
        "potcar_source_path": str(resolved_dir),
        "potcar_source_filename": "POTCAR",
        "potcar_source_kind": "directory",
    }
    if potcar_file.is_file():
        payload["potcar_candidate_file"] = str(potcar_file.resolve())
    return json.dumps(payload)

