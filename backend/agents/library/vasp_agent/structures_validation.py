from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from backend.agents.llm import get_model, settings

from .utils import generate_structure_image

_ACTIVE_RUN_DIR: Optional[str] = None


def set_active_run_dir(path: Optional[str]) -> None:
    global _ACTIVE_RUN_DIR
    _ACTIVE_RUN_DIR = os.path.abspath(path) if path else None


def _resolve_run_dir(run_dir: Optional[str]) -> str:
    base = _ACTIVE_RUN_DIR or os.getcwd()
    if run_dir:
        candidate = run_dir.strip()
        if os.path.isabs(candidate):
            return candidate
        return os.path.abspath(os.path.join(base, candidate))
    return base


@tool
def quick_validate_structure(structure_filename: str, run_dir: str,engine:str="vasp") -> str:
    """
    structure_filename: the name of structure file
    run_dir: the directory to run the structure on
    """
    run_dir = _resolve_run_dir(run_dir)
    filepath = os.path.join(run_dir, structure_filename)
    # Determine the correct format
    if engine.lower() == "lammps":
        read_format = "lammps-data"
    else:
        read_format = "vasp"
    validation_script = textwrap.dedent(
        f"""
    import json
    from ase.io import read
    import numpy as np

    try:
        atoms = read(r'{filepath}',format='{read_format}')
        issues = []

        if len(atoms) == 0:
            issues.append("Structure has no atoms")

        # Use MIC only when periodic
        mic_flag = bool(any(getattr(atoms, "pbc", (False, False, False))))
        if len(atoms) > 1:
            distances = atoms.get_all_distances(mic=mic_flag)
            np.fill_diagonal(distances, np.inf)
            min_dist = float(distances.min())
            if min_dist < 0.8:  # a bit safer threshold than 0.5 Å
                issues.append(f"Atoms too close: {{min_dist:.3f}} Å (expected > 0.8 Å)")

        # Check cell volume if periodic
        if mic_flag and getattr(atoms, "cell", None) is not None and atoms.cell.volume < 0.1:
            issues.append("Periodic system has invalid cell volume")

        result = {{
            "status": "pass" if not issues else "warning",
            "num_atoms": int(len(atoms)),
            "formula": atoms.get_chemical_formula(),
            "issues": issues,
            "quick_validation": True
        }}
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({{"status": "error", "message": str(e)}}))
    """
    ).strip()

    try:
        result = subprocess.run(
            [sys.executable, "-c", validation_script],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=run_dir,
        )
        out = (result.stdout or "").strip()
        if out:
            return out
        # bubble up stderr as JSON if stdout empty
        err = (result.stderr or "").strip()
        return json.dumps(
            {
                "status": "error",
                "message": err or "No output from quick validator",
                "returncode": result.returncode,
            }
        )
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def _coerce_text(raw_content) -> str:
    """Return a best-effort string from Anthropic content blocks."""
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        parts: List[str] = []
        for block in raw_content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts).strip()
    return str(raw_content)


def _image_blocks(image_paths: Dict[str, str]) -> List[dict]:
    """
    Convert local PNG files to Anthropic-compatible image blocks (base64).
    image_paths: dict like {"x": "/path/x.png", "y": "...", "z": "..."}
    """
    blocks = []
    for axis, path in sorted(image_paths.items()):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        blocks.append({
            "type": "text",
            "text": f"View along {axis.upper()}-axis:"
        })
        blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": b64
            }
        })
    return blocks


@tool
def get_llm_validation_and_hint(original_request, run_dir: str, structure_filename: str,
                                generation_script_filename: str,engine:str="vasp") -> dict:
    """
    Uses an LLM to perform full validation including analysis of the actual structure file content and images.
    original_request: The initial user prompt for the structure.
    run_dir: the directory to run the structure on
    structure_filename: The filename of the generated structure.
    generation_script_filename: The filename of the Python script used to create the structure.
    """
    run_dir = _resolve_run_dir(run_dir)
    script_path = os.path.join(run_dir, generation_script_filename)
    structure_path = os.path.join(run_dir, structure_filename)

    if not os.path.exists(script_path):
        return json.dumps({"overall_assessment": "Error: Script file not found.",
                           "identified_issues_detail": [f"File '{generation_script_filename}' does not exist."],
                           "script_modification_hints": []})
    if not os.path.exists(structure_path):
        return json.dumps({"overall_assessment": "Error: Structure file not found.",
                           "identified_issues_detail": [f"File '{structure_filename}' does not exist."],
                           "script_modification_hints": []})
    with open(script_path, 'r', encoding='utf-8') as f:
        generating_script_content = f.read()
    with open(structure_path, 'r', encoding='utf-8', errors='ignore') as f:
        structure_content = f.read()
    image_paths = generate_structure_image(structure_path,engine=engine)
    validation_prompt = f"""
    You are an expert materials scientist and computational modeling specialist.
    Your task is to critically review an unrelaxed atomic structure generated by a Python script. This structure is intended as an initial input for DFT relaxation. Therefore, the purpose is to create a reasonable starting geometry, not a perfect relaxed structure. If this is a grain boundary, interface, or surface structure, be aware that atomic clashes and close contacts (<1.0 Å) are NORMAL and EXPECTED in unrelaxed interfaces, and should not be considered as major issues needed to be fixed.
    **Input Provided to You:**
    1.  **Original User Request for Structure:** A textual description of the desired atomic structure.
    Example: "{original_request}"
    2.**Generating Script Content:** The Python script used to create the structure.
        Example:
        ```python
        {generating_script_content}
        ```
    3.  **Structure File Content:** The raw content of the structure file generated with this exact script.
    4.  **Structure Images (Visual Aid):** Images of the generated structure viewed along the X, Y, and Z axes. These are provided as a supplementary visual reference.
    **Your Task & Output Format:**

    Based on a holistic analysis of ALL provided information (request, script, file content, and images), you MUST output a valid JSON object with the following keys:

    1.  `"overall_assessment"`: (String) A brief (2-3 sentences) overall assessment of the structure's suitability for DFT, its adherence to the original request, and its physical/chemical soundness. Your analysis should be centered on the **script logic and the structure file content**, using the images as a helpful visual reference.
    2.  `"identified_issues_detail"`: (List of Strings) A list of ALL specific issues you identified. Analyze the script and structure file for:
        * Discrepancies from the "Original User Request" (e.g., wrong composition, incorrect lattice, missing defects, wrong surface termination).
        * Gross physical or chemical unreasonableness (e.g., severe atomic clashes that relaxation might not fix, fundamentally wrong bonding indicative of incorrect script logic).
        * Stoichiometry errors.
        * For slabs/surfaces: insufficient vacuum, incorrect layer stacking.
        * Any other obvious issues visible in the file content or images that would cause DFT problems.
        If no critical issues are found, this should be an empty list.
    3.  `"script_modification_hints"`: (List of Strings) Actionable suggestions on how the *provided script* could be modified to address the identified issues. Base these suggestions on your analysis of both the script and the resulting structure. If specialized library documentation is provided, use that library's specific syntax. If the structure is a good starting point, provide an empty list.

    Ensure your output is ONLY the valid JSON object described above. Do not include any other text, explanations, or markdown formatting outside the JSON structure.
    {structure_content}
    """

    content_blocks = [{"type": "text", "text": validation_prompt}]
    if image_paths:
        content_blocks.extend(_image_blocks(image_paths))

    llm = get_model(settings.DEFAULT_MODEL)
    response = llm.invoke([HumanMessage(content=content_blocks)])
    raw_text = _coerce_text(response.content)
    first_brace = raw_text.find("{")
    last_brace = raw_text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_string = raw_text[first_brace: last_brace + 1]
        try:
            llm_feedback = json.loads(json_string)
            print("LLM full validation feedback and script hints received successfully.")
            if not all(k in llm_feedback for k in
                       ["overall_assessment", "identified_issues_detail", "script_modification_hints"]):
                print("LLM feedback JSON is missing one or more expected keys.")
                fallback = {
                    "overall_assessment": llm_feedback.get(
                        "overall_assessment",
                        "LLM assessment incomplete (missing keys)."
                    ),
                    "identified_issues_detail": llm_feedback.get(
                        "identified_issues_detail",
                        ["LLM feedback structure error: missing 'identified_issues_detail'."]
                    ),
                    "script_modification_hints": llm_feedback.get("script_modification_hints", [])
                }
                return json.dumps(fallback)
            return json.dumps(llm_feedback)
        except json.JSONDecodeError as e_json:
            print(
                f"Failed to decode JSON from LLM response substring. Error: {e_json}. Substring: '{json_string[:200]}...'")
            error_msg = f"LLM response could not be parsed as JSON: {e_json}"
    else:
        print(
            f"Could not find valid JSON object delimiters '{{' and '}}' in LLM response. Raw text: {raw_text[:500]}...")
        error_msg = "LLM response did not contain a recognizable JSON object."

    return json.dumps(
        {
            "overall_assessment": "Error: Failed to get valid structured feedback from LLM.",
            "identified_issues_detail": [error_msg],
            "script_modification_hints": [],
        }
    )


TOOLS = [get_llm_validation_and_hint, quick_validate_structure]
