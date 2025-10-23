"""ASE-based tooling utilities for the VASP agent."""

from __future__ import annotations

import subprocess
import sys
import uuid
from pathlib import Path
from typing import Final
import os
from langchain_core.tools import tool


_ASE_TIMEOUT_S: Final[int] = 120
_ACTIVE_STRUCTURE_RUN_DIR: str | None = None

@tool
def execute_ase_script(script_content: str, run_dir: str) -> str:
    """Execute an ASE script in the active run directory, normalizing paths."""

    global _ACTIVE_STRUCTURE_RUN_DIR

    base_dir = _ACTIVE_STRUCTURE_RUN_DIR or os.getcwd()
    provided = (run_dir or "").strip()
    if provided:
        if os.path.isabs(provided):
            effective_run_dir = provided
        else:
            effective_run_dir = os.path.abspath(os.path.join(base_dir, provided))
    else:
        effective_run_dir = base_dir

    debug_arg = provided or "<default>"
    print(f"[DEBUG] run_dir_arg: {debug_arg} | effective: {effective_run_dir}\n")

    if not isinstance(script_content, str) or len(script_content.strip()) < 5:
        return "ERROR: script_content must be a non-empty python script string"

    os.makedirs(effective_run_dir, exist_ok=True)
    script_path = os.path.join(effective_run_dir, f"ase_script_{uuid.uuid4().hex[:8]}.py")

    workdir_code = f"import os\nos.chdir(r'''{effective_run_dir}''')\n\n"
    full_script_content = workdir_code + script_content

    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(full_script_content)

        script_filename = os.path.basename(script_path)
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=effective_run_dir,
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode != 0:
            return (
                f"ERROR: ASE script '{script_filename}' failed.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            )

        if "STRUCTURE_SAVED:" not in stdout:
            return (
                f"ERROR: Script '{script_filename}' ran but did not print 'STRUCTURE_SAVED:'.\nOUTPUT:\n{stdout}"
            )

        return f"Successfully executed script '{script_filename}'. Output:\n{stdout}"
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip()
        return (
            "ERROR: The script failed to execute.\n--- SCRIPT CONTENT ---\n"
            f"{script_content}\n--- ERROR ---\n{stderr}"
        )
    except Exception as e:
        return f"ERROR: An unexpected error occurred during script execution: {str(e)}"