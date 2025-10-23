"""
Pymatgen-based Tools for Materials Analysis

Tools for materials database integration, crystal analysis, and electronic structure
analysis using Pymatgen and Materials Project API.
"""

import gzip
import json
import shutil
import tarfile
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Tuple

import requests
from ase.io import read
from langchain_core.tools import tool
from mp_api.client import MPRester
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from backend.settings import settings

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:142.0) Gecko/20100101 Firefox/142.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


# Load parsed pseudopotential data
def load_pseudopotential_metadata():
    """Load pseudopotential metadata from JSON file."""
    # Try multiple possible locations for the database file
    possible_paths = [
        "data/pseudos_metadata.json",
        Path(__file__).parent.parent.parent.parent / "data" / "pseudos_metadata.json",
        Path.cwd() / "data" / "pseudos_metadata.json",
    ]

    for path in possible_paths:
        if Path(path).exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Silently load pseudopotential database
                    return data
            except Exception as e:
                print(
                    f"Warning: Could not load pseudopotential database from {path}: {e}"
                )
                continue

    # Return empty metadata if file not found - let the system handle missing elements dynamically
    return {}


PP_METADATA = load_pseudopotential_metadata()


@tool
def search_materials_project(
    formula: Annotated[str, "Chemical formula like 'YH2', 'TiO2', 'LiFePO4'. Use this when you know the exact compound."] = "",
    elements: Annotated[str, "Elements string or list. Leave empty if using formula or chemsys."] = "",
    material_ids: Annotated[str, "Materials Project IDs like 'mp-23'. Leave empty if using formula or chemsys."] = "",
    chemsys: Annotated[str, "Chemical system like 'Y-H' or 'Li-Fe-O'. Leave empty if using formula."] = "",
    properties: Annotated[List[str] | None, "List of properties to retrieve. Leave as None for defaults."] = None,
    limit: Annotated[int, "Maximum number of results to return"] = 10,
    band_gap_min: Annotated[float | None, "Minimum band gap in eV"] = None,
    band_gap_max: Annotated[float | None, "Maximum band gap in eV"] = None,
    formation_energy_per_atom_min: Annotated[float | None, "Minimum formation energy per atom in eV/atom"] = None,
    formation_energy_per_atom_max: Annotated[float | None, "Maximum formation energy per atom in eV/atom"] = None,
    energy_above_hull_min: Annotated[float | None, "Minimum energy above hull in eV/atom"] = None,
    energy_above_hull_max: Annotated[float | None, "Maximum energy above hull in eV/atom"] = None,
    is_metal: Annotated[bool | None, "Filter by metallic (True) or non-metallic (False)"] = None,
    num_elements_min: Annotated[int | None, "Minimum number of elements"] = None,
    num_elements_max: Annotated[int | None, "Maximum number of elements"] = None,
    _thread_id: Annotated[str | None, "Thread ID for workspace isolation (auto-injected)"] = None,
) -> str:
    """Search Materials Project database for materials.

    IMPORTANT: Provide exactly ONE of: formula, elements, material_ids, or chemsys.
    Leave the other three as empty strings.

    Args:
        formula: Chemical formula (e.g., 'LiFePO4', 'TiO2', 'YH2'). Use this for searching specific compounds.
        elements: Elements to include (e.g, "LiFePO" or sequence like ["Li","Fe","P","O"]).
            Use with `num_elements_min/max` to bound the number of elements in the composition.
        material_ids: One or more MP material IDs (eg:["mp-23"])
        chemsys: Hyphen-joined chemical system (e.g., "Li-Fe-P-O", "Y-H").
        properties: List of properties to retrieve. Leave empty for default properties.
        limit: Maximum number of results (default: 10)
        band_gap_min: Minimum band gap in eV
        band_gap_max: Maximum band gap in eV
        formation_energy_per_atom_min: Minimum formation energy per atom in eV/atom
        formation_energy_per_atom_max: Maximum formation energy per atom in eV/atom
        energy_above_hull_min: Minimum energy above hull in eV/atom
        energy_above_hull_max: Maximum energy above hull in eV/atom
        is_metal: If set, restrict to metallic (True) or nonmetallic (False) entries
        num_elements_min: Minimum number of elements in composition
        num_elements_max: Maximum number of elements in composition

    Returns:
        Search results with material properties including lattice parameters, structure files, and energetics

    Examples:
        - search_materials_project(formula="YH2")
        - search_materials_project(chemsys="Y-H", limit=5)
        - search_materials_project(formula="TiO2", band_gap_min=2.0)
    """
    try:
        # Use API key from settings if not provided
        if settings.MP_API_KEY:
            api_key = settings.MP_API_KEY.get_secret_value()
        else:
            return "Error: Materials Project API key not configured. Please set MP_API_KEY in environment variables."

        if properties is None or len(properties) == 0:
            properties = [
                "material_id",
                "formula_pretty",
                "structure",
                "formation_energy_per_atom",
                "band_gap",
                "density",
                "symmetry",
                "energy_above_hull",
                "is_metal",
            ]

        # Convert empty strings to None for consistency
        formula = formula if formula else None
        elements = elements if elements else None
        material_ids = material_ids if material_ids else None
        chemsys = chemsys if chemsys else None

        selectors = [s is not None for s in (formula, elements, material_ids, chemsys)]
        if sum(selectors) != 1:
            raise ValueError("Provide exactly one of: formula | elements | material_ids | chemsys")

        query_kwargs = {"fields": properties}
        if formula is not None:
            query_kwargs["formula"] = formula
        elif elements is not None:
            query_kwargs["elements"] = list(elements)
            if num_elements_min is not None or num_elements_max is not None:
                num_elements = []
                if num_elements_min is not None:
                    num_elements.append(num_elements_min)
                else:
                    num_elements.append(1)
                if num_elements_max is not None:
                    num_elements.append(num_elements_max)
                else:
                    num_elements.append(num_elements_min or 10)
                query_kwargs['num_elements'] = num_elements
        elif material_ids is not None:
            query_kwargs["material_ids"] = list(material_ids)
        elif chemsys is not None:
            query_kwargs["chemsys"] = chemsys

        # Add filters if provided
        if band_gap_min is not None or band_gap_max is not None:
            band_gap = []
            if band_gap_min is not None:
                band_gap.append(band_gap_min)
            else:
                band_gap.append(None)
            if band_gap_max is not None:
                band_gap.append(band_gap_max)
            else:
                band_gap.append(None)
            query_kwargs["band_gap"] = tuple(band_gap)
            
        if formation_energy_per_atom_min is not None or formation_energy_per_atom_max is not None:
            formation_energy_per_atom = []
            if formation_energy_per_atom_min is not None:
                formation_energy_per_atom.append(formation_energy_per_atom_min)
            else:
                formation_energy_per_atom.append(None)
            if formation_energy_per_atom_max is not None:
                formation_energy_per_atom.append(formation_energy_per_atom_max)
            else:
                formation_energy_per_atom.append(None)
            query_kwargs["formation_energy_per_atom"] = tuple(formation_energy_per_atom)
            
        if energy_above_hull_min is not None or energy_above_hull_max is not None:
            energy_above_hull = []
            if energy_above_hull_min is not None:
                energy_above_hull.append(energy_above_hull_min)
            else:
                energy_above_hull.append(None)
            if energy_above_hull_max is not None:
                energy_above_hull.append(energy_above_hull_max)
            else:
                energy_above_hull.append(None)
            query_kwargs["energy_above_hull"] = tuple(energy_above_hull)
            
        if is_metal is not None:
            query_kwargs["is_metal"] = is_metal

        with MPRester(api_key=api_key) as mpr:
            docs = mpr.materials.summary.search(**query_kwargs)[:limit]

        if not docs:
            return f"No materials found for formula: {formula}"



        # Create output directory
        output_dir = Path("materials_project_data")
        output_dir.mkdir(exist_ok=True)

        results = []
        for i, doc in enumerate(docs):
            # Handle both dict and object responses
            if isinstance(doc, dict):
                material_id = doc.get("material_id", f"unknown_{i}")
                formula = doc.get("formula_pretty", "unknown")
                formation_energy = doc.get("formation_energy_per_atom")
                band_gap = doc.get("band_gap")
                density = doc.get("density")
                symmetry = doc.get("symmetry")
                structure = doc.get("structure")
            else:
                material_id = getattr(doc, "material_id", f"unknown_{i}")
                formula = getattr(doc, "formula_pretty", "unknown")
                formation_energy = getattr(doc, "formation_energy_per_atom", None)
                band_gap = getattr(doc, "band_gap", None)
                density = getattr(doc, "density", None)
                symmetry = getattr(doc, "symmetry", None)
                structure = getattr(doc, "structure", None)

            result = {
                "material_id": material_id,
                "formula": formula,
            }

            # Add available properties
            if formation_energy is not None:
                result["formation_energy_per_atom"] = float(formation_energy)

            if band_gap is not None:
                result["band_gap"] = float(band_gap)

            if density is not None:
                result["density"] = float(density)

            if hasattr(doc, "is_metal"):
                result["is_metal"] = bool(doc.is_metal)

            if hasattr(doc, "symmetry"):
                result["spacegroup"] = doc.symmetry.symbol
                result["spacegroup_number"] = doc.symmetry.number

            # Save structure if available
            if structure is not None:
                try:
                    structure_file = output_dir / f"{material_id}_{formula}.cif"
                    if hasattr(structure, "to"):
                        # It's a Structure object
                        structure.to(filename=str(structure_file))
                    else:
                        struct_obj = Structure.from_dict(structure)
                        struct_obj.to(filename=str(structure_file))
                    result["structure_file"] = str(structure_file)
                except Exception as e:
                    result["structure_error"] = str(e)

            results.append(result)

        # Save search results
        tag = (
            f"formula_{formula}" if formula is not None else
            f"elements_{'-'.join(elements)}" if elements is not None else
            f"ids_{len(material_ids)}" if material_ids is not None else
            f"chemsys_{chemsys}"
        ).replace(" ", "_")
        results_file = output_dir / f"search_{tag}_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        summary = f"Found {len(results)} materials for {formula}:\\n"
        for result in results:
            metal_str = f" [{'metal' if result.get('is_metal') else 'nonmetal'}]" if 'is_metal' in result else ""
            summary += f"- {result['material_id']}: {result['formula']}{metal_str}"

            if "formation_energy_per_atom" in result:
                summary += f" (ΔHf: {result['formation_energy_per_atom']:.3f} eV/atom)"
            if "band_gap" in result:
                summary += f" (Eg: {result['band_gap']:.2f} eV)"
            summary += "\\n"

        summary += f"\\nResults saved to: {results_file}"
        return summary

    except Exception as e:
        return f"Error searching Materials Project: {str(e)}"


@tool
def analyze_crystal_structure(
    structure_file: str, tolerance: float = 0.01, angle_tolerance: float = 5.0
) -> str:
    """Analyze crystal structure using Pymatgen.

    Args:
        structure_file: Path to structure file
        tolerance: Tolerance for symmetry analysis
        angle_tolerance: Angle tolerance for symmetry analysis

    Returns:
        String with crystal structure analysis
    """
    try:
        # Read structure using ASE then convert to Pymatgen
        atoms = read(structure_file)
        adaptor = AseAtomsAdaptor()
        structure = adaptor.get_structure(atoms)

        # Perform symmetry analysis
        sga = SpacegroupAnalyzer(
            structure, symprec=tolerance, angle_tolerance=angle_tolerance
        )

        # Get crystal system and space group
        crystal_system = sga.get_crystal_system()
        space_group = sga.get_space_group_symbol()
        space_group_number = sga.get_space_group_number()

        # Get conventional structure
        conventional_structure = sga.get_conventional_standard_structure()

        # Analyze local environment
        try:
            cnn = CrystalNN()
            coordination_envs = []
            for i, site in enumerate(structure):
                cn_info = cnn.get_cn(structure, i, use_weights=True)
                coordination_envs.append(
                    {
                        "site_index": i,
                        "element": str(site.specie),
                        "coordination_number": cn_info,
                    }
                )
        except Exception:
            coordination_envs = "Could not analyze coordination environments"

        # Generate output
        input_path = Path(structure_file)
        output_dir = input_path.parent / "analysis"
        output_dir.mkdir(exist_ok=True)

        # Save conventional structure
        conv_file = output_dir / f"{input_path.stem}_conventional.cif"
        conventional_structure.to(filename=str(conv_file))

        # Create analysis report
        analysis = {
            "original_file": structure_file,
            "formula": structure.formula,
            "crystal_system": crystal_system,
            "space_group_symbol": space_group,
            "space_group_number": space_group_number,
            "lattice_parameters": {
                "a": float(structure.lattice.a),
                "b": float(structure.lattice.b),
                "c": float(structure.lattice.c),
                "alpha": float(structure.lattice.alpha),
                "beta": float(structure.lattice.beta),
                "gamma": float(structure.lattice.gamma),
            },
            "volume": float(structure.volume),
            "density": float(structure.density),
            "num_sites": len(structure),
            "coordination_environments": coordination_envs,
            "conventional_structure_file": str(conv_file),
        }

        analysis_file = output_dir / f"{input_path.stem}_analysis.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2, default=str)

        # Create summary string
        summary = f"Crystal Structure Analysis for {structure.formula}:\\n"
        summary += f"Crystal System: {crystal_system}\\n"
        summary += f"Space Group: {space_group} (#{space_group_number})\\n"
        summary += f"Lattice Parameters: a={structure.lattice.a:.3f}, b={structure.lattice.b:.3f}, c={structure.lattice.c:.3f} Å\\n"
        summary += f"Angles: α={structure.lattice.alpha:.1f}°, β={structure.lattice.beta:.1f}°, γ={structure.lattice.gamma:.1f}°\\n"
        summary += f"Volume: {structure.volume:.2f} Å³\\n"
        summary += f"Density: {structure.density:.2f} g/cm³\\n"
        summary += f"Number of sites: {len(structure)}\\n"
        summary += f"\\nAnalysis saved to: {analysis_file}"
        summary += f"\\nConventional structure saved to: {conv_file}"

        return summary

    except Exception as e:
        return f"Error analyzing crystal structure: {str(e)}"


@tool
def find_pseudopotentials(
    elements: List[str],
    pp_type: str = "PAW",
    pp_library: str = "pslibrary",
    functional: str = "PBE",
    quality: str = "high",
    relativistic: Optional[bool] = None,
) -> dict:
    """Find, download, and extract pseudopotentials from PP metadata.

    Args:
        elements: List of chemical elements (symbols, e.g. ["Si", "O"])
        pp_type: Pseudopotential type ("PAW", "US", "ONCV")
        pp_library: Pseudopotential library ("pslibrary", "gbrv", "sg15", "dojo", "SSSP_Efficiency")
        functional: Exchange-correlation functional ("PBE", "PBEsol", "PZ")
        quality: Quality level ("high", "medium", "low")
        relativistic: Whether to use relativistic pseudopotentials (auto-detect for heavy elements if None)

    Returns:
        dict with working_dir, found pseudopotentials (local paths), and parameters
    """
    try:
        output_dir = Path(f"{settings.ROOT_PATH}/WORKSPACE/pseudos")
        output_dir.mkdir(exist_ok=True)

        found_pps = {}
        missing_pps = []

        # Load pseudopotential metadata
        pp_metadata = load_pseudopotential_metadata()

        for elem in elements:
            # Search parsed JSON
            match = None
            for entry in pp_metadata:
                if (
                    entry.get("element", "").capitalize() == elem.capitalize()
                    and entry.get("pp_type", "").lower() == pp_type.lower()
                    and entry.get("generator", "").lower() == pp_library.lower()
                    and entry.get("xc", "").lower() == functional.lower()
                ):
                    match = entry
                    break

            if not match or not match.get("download_url"):
                missing_pps.append(elem)
                continue

            url = match["download_url"]
            local_file = output_dir / Path(url).name

            # Download the pseudopotential if not already present
            if not local_file.exists():
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    with open(local_file, "w") as f:
                        f.write(response.text)
                else:
                    missing_pps.append(elem)
                    continue

            # Extract if compressed
            final_file = local_file
            if local_file.suffix == ".gz":
                extracted_file = output_dir / local_file.stem
                with (
                    gzip.open(local_file, "rb") as f_in,
                    open(extracted_file, "wb") as f_out,
                ):
                    shutil.copyfileobj(f_in, f_out)
                final_file = extracted_file

            elif local_file.suffixes[-2:] in [
                [".tar", ".gz"],
                [".tar", ".bz2"],
                [".tar", ".xz"],
            ]:
                with tarfile.open(local_file, "r:*") as tar:
                    tar.extractall(output_dir)
                # pick extracted UPFs
                extracted_files = list(output_dir.glob("*.UPF")) + list(
                    output_dir.glob("*.upf")
                )
                if extracted_files:
                    final_file = extracted_files[0]

            found_pps[elem] = str(final_file)

        return {
            "working_dir": str(output_dir.resolve()),
            "found_pseudopotentials": found_pps,
            "missing": missing_pps,
        }

    except Exception as e:
        return {"error": str(e)}


@tool
def calculate_formation_energy(
    structure_file: str, reference_energies: Dict[str, float], total_energy: float
) -> str:
    """Calculate formation energy of a compound.

    Args:
        structure_file: Path to structure file
        reference_energies: Dict of element -> energy per atom (eV)
        total_energy: Total energy of the compound (eV)

    Returns:
        Formation energy calculation results
    """
    try:
        # Read structure
        atoms = read(structure_file)

        # Get composition
        symbols = atoms.get_chemical_symbols()
        composition = {}
        for symbol in symbols:
            composition[symbol] = composition.get(symbol, 0) + 1

        # Calculate formation energy
        # ΔHf = E_compound - Σ(n_i * E_i^ref)
        reference_sum = 0.0
        missing_refs = []

        for element, count in composition.items():
            if element in reference_energies:
                reference_sum += count * reference_energies[element]
            else:
                missing_refs.append(element)

        if missing_refs:
            return f"Error: Missing reference energies for elements: {', '.join(missing_refs)}"

        formation_energy = total_energy - reference_sum
        formation_energy_per_atom = formation_energy / len(atoms)

        # Generate output
        input_path = Path(structure_file)
        output_dir = input_path.parent / "formation_energies"
        output_dir.mkdir(exist_ok=True)

        # Create formation energy data
        fe_data = {
            "structure_file": structure_file,
            "formula": atoms.get_chemical_formula(),
            "composition": composition,
            "total_energy": total_energy,
            "reference_energies": reference_energies,
            "reference_sum": reference_sum,
            "formation_energy": formation_energy,
            "formation_energy_per_atom": formation_energy_per_atom,
            "num_atoms": len(atoms),
        }

        fe_file = output_dir / f"{input_path.stem}_formation_energy.json"
        with open(fe_file, "w") as f:
            json.dump(fe_data, f, indent=2)

        # Create summary
        summary = f"Formation Energy Calculation for {atoms.get_chemical_formula()}:\\n"
        summary += f"Total Energy: {total_energy:.4f} eV\\n"
        summary += f"Reference Sum: {reference_sum:.4f} eV\\n"
        summary += f"Formation Energy: {formation_energy:.4f} eV\\n"
        summary += (
            f"Formation Energy per atom: {formation_energy_per_atom:.4f} eV/atom\\n"
        )
        summary += f"\\nComposition: {composition}\\n"
        summary += f"Results saved to: {fe_file}"

        return summary

    except Exception as e:
        return f"Error calculating formation energy: {str(e)}"
