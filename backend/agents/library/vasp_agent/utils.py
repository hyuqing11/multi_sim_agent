from ase.io import read as ase_read, write as ase_write
import matplotlib
import os
from typing import Optional
matplotlib.use('Agg')


def center_structure_for_visualization(atoms):
    try:
        working = atoms.copy()
        working.center()
        if any(working.pbc):
            working.wrap()
        return working
    except Exception as e:
        print(f"Could not center structure: {e}")
        return atoms


def generate_structure_image(structure_file_path: str,engine='vasp'):
    if engine.lower() == "lammps":
        read_format = "lammps-data"
    else:
        read_format = "vasp"
    try:
        atoms = ase_read(structure_file_path, format=read_format)
    except Exception as exc:
        print(f"Failed to read structure file '{structure_file_path}': {exc}")
        return {}

    atoms = center_structure_for_visualization(atoms)

    base_name = os.path.splitext(os.path.basename(structure_file_path))[0]
    image_paths = {}
    output_dir = os.path.dirname(structure_file_path)

    rotations = {
        'x': ('90x,0y,0z', 'View along X-axis'),
        'y': ('0x,90y,0z', 'View along Y-axis'),
        'z': ('0x,0y,0z', 'View along Z-axis'),
        'iso': ('45x,45y,0z', 'Isometric View')
    }

    print(f"Generating structure view images for {structure_file_path} using ASE renderer...")
    for axis, (rotation, label) in rotations.items():
        try:
            output_path = os.path.join(output_dir, f"{base_name}_{axis}.png")
            ase_write(output_path, atoms, format='png', rotation=rotation, radii=0.85)
            image_paths[axis] = output_path
            print(f"Saved {label}: {output_path}")
        except Exception as e:
            print(f"Failed to generate {axis}-axis view: {e}")

    if image_paths:
        print(f"Successfully generated {len(image_paths)} images.")
    else:
        print("No images were generated.")

    return image_paths


ENGINE_VASP = "vasp"
ENGINE_LAMMPS = "lammps"


def detect_engine(query: str) -> str:
    """Pick an engine based on the free-form user query."""

    normalized = (query or "").lower()
    if ENGINE_LAMMPS in normalized:
        return ENGINE_LAMMPS
    if ENGINE_VASP in normalized:
        return ENGINE_VASP
    return ENGINE_VASP

def _normalize_engine(engine: Optional[str]) -> str:
    """Normalize an engine identifier to one of the supported values."""

    if not engine:
        return ENGINE_VASP
    lowered = engine.lower().strip()
    if lowered in {ENGINE_VASP, ENGINE_LAMMPS}:
        return lowered
    if "lammps" in lowered:
        return ENGINE_LAMMPS
    return ENGINE_VASP
