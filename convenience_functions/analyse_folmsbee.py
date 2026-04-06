"""Analyse Folmsbee/Hutchison conformer benchmark energies."""

from __future__ import annotations

import multiprocessing as mp
import re
from pathlib import Path
from typing import Iterable

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
import seaborn as sns
from openff.interchange import Interchange
from openff.toolkit import ForceField, Molecule
from openff.units import unit as off_unit
from openeye import oechem
import openmm
import openmm.unit as omm_unit
from openmm.app import Simulation
from scipy import stats
from statannotations.Annotator import Annotator
from tqdm import tqdm

from presto.find_torsions import get_rot_torsions_by_rot_bond
from presto.sample import (
    _add_torsion_restraint_forces,
    _get_integrator,
    _remove_torsion_restraint_forces,
    _update_torsion_restraints,
)

plt.style.use("ggplot")

# ---------------------------------------------------------------------------
# Worker-process globals
# ---------------------------------------------------------------------------

_WORKER_FFS: dict[str, ForceField] | None = None
_WORKER_MLP_NAMES: list[str] | None = None
_WORKER_MOLECULE_DIRS: dict[str, str] | None = None
_WORKER_TORSION_K: float | None = None
_WORKER_MM_STEPS: int | None = None
_WORKER_PER_MOL_ROOT: str | None = None
_WORKER_SINGLE_POINT_MLP: bool | None = None

_PLOT_RMSE_MAX = 4.0


# ---------------------------------------------------------------------------
# Geometry / math helpers
# ---------------------------------------------------------------------------

def _compute_dihedral_radians(
    coordinates_nm: np.ndarray,
    atoms: tuple[int, int, int, int],
) -> float:
    p0, p1, p2, p3 = (coordinates_nm[idx] for idx in atoms)
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = np.linalg.norm(b1)
    if b1_norm == 0:
        return 0.0
    b1 = b1 / b1_norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def _openeye_rmsd(
    template_molecule: Molecule,
    reference_xyz: np.ndarray,
    target_xyz: np.ndarray,
) -> float:
    """Compute RMSD using OpenEye's implementation with fixed atom order."""
    ref_mol = Molecule(template_molecule)
    ref_mol.conformers.clear()
    ref_mol.add_conformer(off_unit.Quantity(reference_xyz, off_unit.angstrom))

    tgt_mol = Molecule(template_molecule)
    tgt_mol.conformers.clear()
    tgt_mol.add_conformer(off_unit.Quantity(target_xyz, off_unit.angstrom))

    ref_oe = ref_mol.to_openeye()
    tgt_oe = tgt_mol.to_openeye()
    return float(
        oechem.OERMSD(
            ref_oe,
            tgt_oe,
            False,  # no automorphism search; preserve atom order
            False,  # include hydrogens
            True,   # overlay before RMSD calculation
        )
    )


# ---------------------------------------------------------------------------
# Label / ID helpers
# ---------------------------------------------------------------------------

def _method_label(method_name: str) -> str:
    if method_name.endswith("/combined_force_field.offxml"):
        return Path(method_name).parent.name
    if "/" in method_name:
        return Path(method_name).name
    return method_name


def _method_id(method_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", _method_label(method_name))


def _component_id(component_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", component_name.strip() or "unknown")


def _overall_force_field_label(method_name: str) -> str:
    lower = method_name.lower()
    if method_name.endswith("/combined_force_field.offxml"):
        return "bespoke"
    if "esp" in lower:
        return "espaloma"
    if "bespokefit" in lower:
        return "bespokefit_1"
    if "openff_unconstrained-2.3.0" in lower:
        return "sage"
    return _method_label(method_name)


def _safe_relative(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values - np.mean(values)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _get_r_sq(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(stats.linregress(x, y).rvalue ** 2)


def _get_kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return float(stats.kendalltau(x, y).statistic)


def _get_rmse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x - y) ** 2)))


def _get_mae(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(x - y)))


def _calculate_stats(reference: np.ndarray, method: np.ndarray) -> dict[str, float]:
    ref_rel = _safe_relative(reference)
    met_rel = _safe_relative(method)
    return {
        "r2": _get_r_sq(ref_rel, met_rel),
        "kendall_tau": _get_kendall_tau(ref_rel, met_rel),
        "rmse": _get_rmse(ref_rel, met_rel),
        "mae": _get_mae(ref_rel, met_rel),
    }


# ---------------------------------------------------------------------------
# File / data loading helpers
# ---------------------------------------------------------------------------

def _resolve_molecule_geometry_dirs(folmsbee_repo_dir: Path) -> dict[str, Path]:
    geometry_root = folmsbee_repo_dir / "geometries"
    molecule_dirs: dict[str, Path] = {}
    for job_subdir in ("Neutral_jobs", "CHG_jobs"):
        subdir = geometry_root / job_subdir
        if not subdir.exists():
            continue
        for molecule_dir in subdir.iterdir():
            if molecule_dir.is_dir():
                molecule_dirs[molecule_dir.name] = molecule_dir
    return molecule_dirs


def _load_force_fields(force_field_paths: Iterable[str]) -> dict[str, ForceField]:
    return {
        ff_path: ForceField(str(ff_path), load_plugins=True)
        for ff_path in force_field_paths
    }


def _collect_presto_molecules(presto_output_dir: Path) -> set[str]:
    return {
        path.name
        for path in presto_output_dir.iterdir()
        if path.is_dir() and (path / "bespoke_force_field.offxml").exists()
    }


def _write_minimized_conformer_sdf(
    template_molecule: Molecule,
    xyz_angstrom: np.ndarray,
    output_path: Path,
) -> None:
    mol = Molecule(template_molecule)
    mol.conformers.clear()
    mol.add_conformer(off_unit.Quantity(xyz_angstrom, off_unit.angstrom))
    mol.to_file(str(output_path), file_format="sdf")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cached_csv(path: Path) -> pd.DataFrame | None:
    """Return a DataFrame from *path* or None if it does not exist."""
    if path.exists():
        return pd.read_csv(path)
    return None


def _cached_columns_complete(
    df: pd.DataFrame,
    required_columns: list[str],
    geometries: list[str],
) -> bool:
    """True if *df* has all *required_columns* and one row per geometry."""
    if not all(col in df.columns for col in required_columns):
        return False
    geom_counts = df["geom"].value_counts()
    return all(geom_counts.get(g, 0) == 1 for g in geometries)


def _sdfs_complete(directory: Path, geometry_stems: list[str]) -> bool:
    return all((directory / f"{stem}.sdf").exists() for stem in geometry_stems)


# ---------------------------------------------------------------------------
# MM evaluation
# ---------------------------------------------------------------------------

def _build_mm_simulation(
    reference_molecule: Molecule,
    force_field: ForceField,
) -> tuple[Simulation, list[tuple[int, int, int, int]], list[int], int | None, dict[str, list[int]]]:
    """Build an OpenMM Simulation for MM energy evaluation.

    Returns
    -------
    simulation, torsion_atoms, restraint_force_indices, restraint_force_group,
    component_force_groups
    """
    omm_system = Interchange.from_smirnoff(
        force_field,
        reference_molecule.to_topology(),
    ).to_openmm()
    integrator = _get_integrator(300 * omm_unit.kelvin, 1.0 * omm_unit.femtoseconds)
    simulation = Simulation(
        reference_molecule.to_topology().to_openmm(),
        omm_system,
        integrator,
    )

    torsion_atoms = [
        tuple(t) for t in get_rot_torsions_by_rot_bond(reference_molecule).values()
    ]
    restraint_force_indices: list[int] = []
    restraint_force_group: int | None = None
    if torsion_atoms:
        restraint_force_indices, restraint_force_group = _add_torsion_restraint_forces(
            simulation, torsion_atoms, 0.0  # placeholder K; updated per-conformer
        )

    restraint_index_set = set(restraint_force_indices)
    available_groups = [g for g in range(32) if g != restraint_force_group]
    non_restraint_forces = [
        (idx, force)
        for idx, force in enumerate(omm_system.getForces())
        if idx not in restraint_index_set
    ]
    if len(non_restraint_forces) > len(available_groups):
        raise RuntimeError(
            f"Too many OpenMM forces ({len(non_restraint_forces)}) to assign unique force groups"
        )

    component_force_groups: dict[str, list[int]] = {}
    for (_, force), group in zip(non_restraint_forces, available_groups):
        force.setForceGroup(group)
        name = (force.getName().strip() if force.getName() else "") or force.__class__.__name__
        component_force_groups.setdefault(name, []).append(group)

    simulation.context.reinitialize(preserveState=True)
    return simulation, torsion_atoms, restraint_force_indices, restraint_force_group, component_force_groups


def _evaluate_mm(
    reference_molecule: Molecule,
    conformer_positions: list,
    force_field: ForceField,
    torsion_restraint_force_constant: float,
    mm_minimization_steps: int,
) -> tuple[list[float], list[np.ndarray], list[dict[str, float]]]:
    """Evaluate (and optionally minimise) MM energies for a list of conformers.

    Returns (energies_kcalmol, minimised_xyz_angstrom, component_energies).
    """
    simulation, torsion_atoms, restraint_force_indices, restraint_force_group, component_force_groups = (
        _build_mm_simulation(reference_molecule, force_field)
    )

    energies: list[float] = []
    minimized_xyz: list[np.ndarray] = []
    component_energies: list[dict[str, float]] = []

    non_restraint_mask = (
        sum(1 << g for g in range(32) if g != restraint_force_group)
        if restraint_force_group is not None
        else None
    )

    try:
        for positions in conformer_positions:
            simulation.context.setPositions(positions)

            if torsion_atoms:
                coords_nm = positions.value_in_unit(omm_unit.nanometer)
                current_angles = [
                    _compute_dihedral_radians(coords_nm, atoms) for atoms in torsion_atoms
                ]
                _update_torsion_restraints(
                    simulation,
                    restraint_force_indices,
                    current_angles,
                    torsion_restraint_force_constant,
                )

            if mm_minimization_steps != -1:
                simulation.minimizeEnergy(maxIterations=mm_minimization_steps)

            state = simulation.context.getState(
                getEnergy=True,
                getPositions=True,
                groups=non_restraint_mask if non_restraint_mask is not None else -1,
            )
            energy = state.getPotentialEnergy().value_in_unit(omm_unit.kilocalorie_per_mole)
            xyz = state.getPositions(asNumpy=True).value_in_unit(omm_unit.angstrom)

            per_component: dict[str, float] = {}
            for comp_name, groups in component_force_groups.items():
                mask = sum(1 << g for g in groups)
                comp_state = simulation.context.getState(getEnergy=True, groups=mask)
                per_component[comp_name] = comp_state.getPotentialEnergy().value_in_unit(
                    omm_unit.kilocalorie_per_mole
                )

            energies.append(float(energy))
            minimized_xyz.append(np.asarray(xyz))
            component_energies.append(per_component)
    finally:
        if restraint_force_indices:
            _remove_torsion_restraint_forces(simulation, restraint_force_indices)

    return energies, minimized_xyz, component_energies


# ---------------------------------------------------------------------------
# MLP evaluation
# ---------------------------------------------------------------------------

def _build_mlp_simulation(
    reference_molecule: Molecule,
    mlp_name: str,
    torsion_restraint_force_constant: float,
) -> tuple[Simulation, list[tuple[int, int, int, int]], list[int], int | None]:
    """Build an OpenMM Simulation backed by an MLPotential.

    Returns (simulation, torsion_atoms, restraint_force_indices, restraint_force_group).
    """
    from openmmml import MLPotential

    potential = MLPotential(mlp_name)
    omm_topology = reference_molecule.to_topology().to_openmm()
    # Pass molecular total charge explicitly so charged systems are handled
    # correctly by MLPs that support/use charge-aware inference.
    total_charge = reference_molecule.total_charge.m_as(off_unit.e)
    omm_system = potential.createSystem(omm_topology, charge=total_charge)

    integrator = _get_integrator(300 * omm_unit.kelvin, 1.0 * omm_unit.femtoseconds)
    simulation = Simulation(omm_topology, omm_system, integrator)

    torsion_atoms = [
        tuple(t) for t in get_rot_torsions_by_rot_bond(reference_molecule).values()
    ]
    restraint_force_indices: list[int] = []
    restraint_force_group: int | None = None
    if torsion_atoms:
        restraint_force_indices, restraint_force_group = _add_torsion_restraint_forces(
            simulation, torsion_atoms, torsion_restraint_force_constant
        )
        simulation.context.reinitialize(preserveState=True)

    return simulation, torsion_atoms, restraint_force_indices, restraint_force_group


def _evaluate_mlp(
    reference_molecule: Molecule,
    conformer_positions: list,
    mlp_name: str,
    torsion_restraint_force_constant: float,
    minimization_steps: int,
    single_point: bool,
) -> tuple[list[float], list[np.ndarray]]:
    """Evaluate MLP energies (single-point or with minimisation).

    Returns (energies_kcalmol, xyz_angstrom).
    """
    simulation, torsion_atoms, restraint_force_indices, restraint_force_group = (
        _build_mlp_simulation(reference_molecule, mlp_name, torsion_restraint_force_constant)
    )

    energies: list[float] = []
    result_xyz: list[np.ndarray] = []

    non_restraint_mask = (
        sum(1 << g for g in range(32) if g != restraint_force_group)
        if restraint_force_group is not None
        else None
    )

    try:
        for positions in conformer_positions:
            simulation.context.setPositions(positions)

            if not single_point and torsion_atoms:
                coords_nm = positions.value_in_unit(omm_unit.nanometer)
                current_angles = [
                    _compute_dihedral_radians(coords_nm, atoms) for atoms in torsion_atoms
                ]
                _update_torsion_restraints(
                    simulation,
                    restraint_force_indices,
                    current_angles,
                    torsion_restraint_force_constant,
                )

            if not single_point and minimization_steps != -1:
                simulation.minimizeEnergy(maxIterations=minimization_steps)

            state = simulation.context.getState(
                getEnergy=True,
                getPositions=True,
                groups=non_restraint_mask if non_restraint_mask is not None else -1,
            )
            energy = state.getPotentialEnergy().value_in_unit(omm_unit.kilocalorie_per_mole)
            xyz = state.getPositions(asNumpy=True).value_in_unit(omm_unit.angstrom)

            energies.append(float(energy))
            result_xyz.append(np.asarray(xyz))
    finally:
        if restraint_force_indices:
            _remove_torsion_restraint_forces(simulation, restraint_force_indices)

    return energies, result_xyz


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _initialise_worker(
    force_field_paths: list[str],
    mlp_names: list[str],
    molecule_geometry_dirs: dict[str, str],
    torsion_restraint_force_constant: float,
    mm_minimization_steps: int,
    per_molecule_root: str,
    single_point_mlp: bool,
) -> None:
    global _WORKER_FFS, _WORKER_MLP_NAMES, _WORKER_MOLECULE_DIRS
    global _WORKER_TORSION_K, _WORKER_MM_STEPS, _WORKER_PER_MOL_ROOT
    global _WORKER_SINGLE_POINT_MLP
    _WORKER_FFS = _load_force_fields(force_field_paths)
    _WORKER_MLP_NAMES = mlp_names
    _WORKER_MOLECULE_DIRS = molecule_geometry_dirs
    _WORKER_TORSION_K = torsion_restraint_force_constant
    _WORKER_MM_STEPS = mm_minimization_steps
    _WORKER_PER_MOL_ROOT = per_molecule_root
    _WORKER_SINGLE_POINT_MLP = single_point_mlp


def _load_molecule_geometries(
    molecule_name: str,
    geometries: list[str],
    geometry_dir_str: str,
) -> tuple[
    Molecule | None,
    list[str],
    list,
    list[np.ndarray],
    list[Molecule],
]:
    """Load SDF files for all geometries of a molecule.

    Returns (reference_molecule, loaded_geometries, conformer_positions,
             input_xyz_ang, template_molecules).
    """
    reference_molecule: Molecule | None = None
    loaded_geometries: list[str] = []
    conformer_positions: list = []
    input_xyz_ang: list[np.ndarray] = []
    templates: list[Molecule] = []

    for geometry in geometries:
        sdf = Path(geometry_dir_str) / Path(geometry).with_suffix(".openff.sdf")
        if not sdf.exists():
            continue
        try:
            mol = Molecule.from_file(str(sdf))
        except Exception:
            continue
        if not mol.conformers:
            continue

        if reference_molecule is None:
            reference_molecule = Molecule(mol)

        loaded_geometries.append(geometry)
        templates.append(Molecule(mol))
        conformer_positions.append(mol.conformers[0].to_openmm())
        input_xyz_ang.append(mol.conformers[0].m_as(off_unit.angstrom))

    return reference_molecule, loaded_geometries, conformer_positions, input_xyz_ang, templates


def _process_molecule_worker(
    task: tuple[str, list[str]],
) -> tuple[str, dict[str, dict]]:
    """Compute MM and MLP energies for one molecule.

    Returns (molecule_name, {geometry: {column: value}}).
    """
    molecule_name, geometries = task
    assert (
        _WORKER_FFS is not None
        and _WORKER_MLP_NAMES is not None
        and _WORKER_MOLECULE_DIRS is not None
        and _WORKER_TORSION_K is not None
        and _WORKER_MM_STEPS is not None
        and _WORKER_PER_MOL_ROOT is not None
        and _WORKER_SINGLE_POINT_MLP is not None
    ), "Worker globals not initialised"

    molecule_dir = Path(_WORKER_PER_MOL_ROOT) / molecule_name
    molecule_dir.mkdir(parents=True, exist_ok=True)
    minimized_root = molecule_dir / "minimised"
    minimized_root.mkdir(parents=True, exist_ok=True)

    mm_results_path = molecule_dir / "mm_results.csv"
    mlp_results_path = molecule_dir / "mlp_results.csv"

    geometry_stems = [Path(g).stem for g in geometries]

    # ---- Load cached CSVs once -------------------------------------------
    mm_df_cached = _load_cached_csv(mm_results_path)
    mlp_df_cached = _load_cached_csv(mlp_results_path)

    # ---- Determine which FFs / MLPs need computation ---------------------
    def _ff_needs_compute(ff_path: str) -> bool:
        ff_id = _method_id(ff_path)
        required = [ff_path, f"{ff_path}__rmsd_angstrom"]
        if mm_df_cached is None or not _cached_columns_complete(mm_df_cached, required, geometries):
            return True
        return not _sdfs_complete(minimized_root / ff_id, geometry_stems)

    def _mlp_needs_compute(mlp_name: str) -> bool:
        required = [mlp_name]
        if not _WORKER_SINGLE_POINT_MLP:
            required.append(f"{mlp_name}__rmsd_angstrom")
        if mlp_df_cached is None or not _cached_columns_complete(mlp_df_cached, required, geometries):
            return True
        if not _WORKER_SINGLE_POINT_MLP:
            mlp_id = _method_id(mlp_name)
            return not _sdfs_complete(minimized_root / mlp_id, geometry_stems)
        return False

    ffs_to_compute = [ff for ff in _WORKER_FFS if _ff_needs_compute(ff)]
    mlps_to_compute = [mlp for mlp in _WORKER_MLP_NAMES if _mlp_needs_compute(mlp)]

    # If nothing needs computing, read everything from cache and return.
    if not ffs_to_compute and not mlps_to_compute:
        return molecule_name, _read_results_from_cache(
            geometries, _WORKER_FFS, _WORKER_MLP_NAMES, mm_df_cached, mlp_df_cached
        )

    # ---- Load geometries from disk ---------------------------------------
    geometry_dir_str = _WORKER_MOLECULE_DIRS.get(molecule_name)
    if geometry_dir_str is None:
        return molecule_name, {}

    reference_molecule, loaded_geoms, conformer_positions, input_xyz_ang, templates = (
        _load_molecule_geometries(molecule_name, geometries, geometry_dir_str)
    )
    if reference_molecule is None:
        return molecule_name, {}

    loaded_stems = [Path(g).stem for g in loaded_geoms]

    # ---- MM evaluation ---------------------------------------------------
    # Start from cached rows if they exist; we'll update only missing columns.
    mm_rows: dict[str, dict] = {}
    if mm_df_cached is not None:
        for _, row in mm_df_cached.iterrows():
            mm_rows[str(row["geom"])] = dict(row)
    for geom in loaded_geoms:
        mm_rows.setdefault(geom, {"geom": geom})

    for ff_path in ffs_to_compute:
        ff_id = _method_id(ff_path)
        ff_dir = minimized_root / ff_id
        ff_dir.mkdir(parents=True, exist_ok=True)

        try:
            energies, min_xyz, comp_energies = _evaluate_mm(
                reference_molecule=reference_molecule,
                conformer_positions=conformer_positions,
                force_field=_WORKER_FFS[ff_path],
                torsion_restraint_force_constant=_WORKER_TORSION_K,
                mm_minimization_steps=_WORKER_MM_STEPS,
            )
        except Exception as exc:
            raise RuntimeError(
                f"MM evaluation failed: molecule={molecule_name}, ff={ff_path}"
            ) from exc

        for i, geom in enumerate(loaded_geoms):
            energy = float(energies[i]) if i < len(energies) else np.nan
            # Negate: reference data uses atomisation energies (more negative = more stable)
            mm_rows[geom][ff_path] = -energy if np.isfinite(energy) else np.nan

            rmsd = np.nan
            if i < len(min_xyz) and min_xyz[i] is not None:
                rmsd = _openeye_rmsd(
                    templates[i],
                    input_xyz_ang[i],
                    min_xyz[i],
                )
                _write_minimized_conformer_sdf(
                    templates[i], min_xyz[i], ff_dir / f"{loaded_stems[i]}.sdf"
                )
            mm_rows[geom][f"{ff_path}__rmsd_angstrom"] = rmsd

            if i < len(comp_energies):
                for comp_name, comp_energy in comp_energies[i].items():
                    col = f"{ff_path}__component__{_component_id(comp_name)}"
                    mm_rows[geom][col] = -comp_energy

    pd.DataFrame(list(mm_rows.values())).to_csv(mm_results_path, index=False)

    # ---- MLP evaluation --------------------------------------------------
    mlp_rows: dict[str, dict] = {}
    if mlp_df_cached is not None:
        for _, row in mlp_df_cached.iterrows():
            mlp_rows[str(row["geom"])] = dict(row)
    for geom in loaded_geoms:
        mlp_rows.setdefault(geom, {"geom": geom})

    for mlp_name in mlps_to_compute:
        mlp_id = _method_id(mlp_name)
        mlp_dir = minimized_root / mlp_id
        mlp_dir.mkdir(parents=True, exist_ok=True)

        try:
            energies, result_xyz = _evaluate_mlp(
                reference_molecule=reference_molecule,
                conformer_positions=conformer_positions,
                mlp_name=mlp_name,
                torsion_restraint_force_constant=_WORKER_TORSION_K,
                minimization_steps=_WORKER_MM_STEPS,
                single_point=_WORKER_SINGLE_POINT_MLP,
            )
        except Exception as exc:
            raise RuntimeError(
                f"MLP evaluation failed: molecule={molecule_name}, mlp={mlp_name}"
            ) from exc

        for i, geom in enumerate(loaded_geoms):
            energy = float(energies[i]) if i < len(energies) else np.nan
            mlp_rows[geom][mlp_name] = -energy if np.isfinite(energy) else np.nan

            if not _WORKER_SINGLE_POINT_MLP and i < len(result_xyz) and result_xyz[i] is not None:
                rmsd = _openeye_rmsd(
                    templates[i],
                    input_xyz_ang[i],
                    result_xyz[i],
                )
                mlp_rows[geom][f"{mlp_name}__rmsd_angstrom"] = rmsd
                _write_minimized_conformer_sdf(
                    templates[i], result_xyz[i], mlp_dir / f"{loaded_stems[i]}.sdf"
                )

    pd.DataFrame(list(mlp_rows.values())).to_csv(mlp_results_path, index=False)

    # ---- Merge and return ------------------------------------------------
    all_ff_paths = list(_WORKER_FFS.keys())
    mm_df_fresh = _load_cached_csv(mm_results_path)
    mlp_df_fresh = _load_cached_csv(mlp_results_path)
    return molecule_name, _read_results_from_cache(
        geometries, all_ff_paths, _WORKER_MLP_NAMES, mm_df_fresh, mlp_df_fresh
    )


def _read_results_from_cache(
    geometries: list[str],
    ff_paths: Iterable[str],
    mlp_names: Iterable[str],
    mm_df: pd.DataFrame | None,
    mlp_df: pd.DataFrame | None,
) -> dict[str, dict]:
    """Merge MM and MLP cached CSVs into a per-geometry result dict."""
    results: dict[str, dict] = {g: {} for g in geometries}

    for df in (mm_df, mlp_df):
        if df is None:
            continue
        for _, row in df.iterrows():
            geom = str(row["geom"])
            if geom in results:
                results[geom].update(
                    {k: v for k, v in row.items() if k != "geom"}
                )

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_correlation(
    molecule_df: pd.DataFrame,
    molecule_name: str,
    method_names: list[str],
    reference_method: str,
    output_path: Path,
) -> None:
    valid_df = molecule_df.dropna(subset=[reference_method])
    if len(valid_df) < 2:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    all_values: list[float] = []

    for method_name in method_names:
        filtered = valid_df.dropna(subset=[method_name]).copy()
        if len(filtered) < 2:
            continue

        ref_rel = _safe_relative(filtered[reference_method].to_numpy(float))
        met_rel = _safe_relative(filtered[method_name].to_numpy(float))
        all_values.extend(ref_rel.tolist())
        all_values.extend(met_rel.tolist())

        st = _calculate_stats(
            filtered[reference_method].to_numpy(float),
            filtered[method_name].to_numpy(float),
        )
        label = (
            f"{_method_label(method_name)} "
            f"(R²={st['r2']:.2f}, τ={st['kendall_tau']:.2f}, "
            f"RMSE={st['rmse']:.2f}, MAE={st['mae']:.2f})"
        )
        scatter = ax.scatter(ref_rel, met_rel, label=label)

        if method_name.endswith("/combined_force_field.offxml") and "geom" in filtered:
            colour = scatter.get_facecolor()[0]
            for x, y, lbl in zip(ref_rel, met_rel, filtered["geom"]):
                ax.text(x, y, str(lbl), fontsize=6, color=colour, ha="left", va="bottom", alpha=0.9)

    if all_values:
        lo, hi = min(all_values), max(all_values)
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="k")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        ax.text(0.5, 0.5, "No methods with ≥2 valid conformers",
                transform=ax.transAxes, ha="center", va="center")

    ax.set_xlabel(f"{reference_method} rel. conformer AE / kcal mol$^{{-1}}$")
    ax.set_ylabel("Method rel. conformer AE / kcal mol$^{-1}$")
    ax.set_title(molecule_name)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_overall_metric_distribution(
    per_molecule_stats: pd.DataFrame,
    metric_column: str,
    output_path: Path,
) -> None:
    if metric_column not in per_molecule_stats.columns or len(per_molecule_stats) == 0:
        return

    plot_df = per_molecule_stats.dropna(subset=[metric_column]).copy()
    if "rmse_kcal_mol" in per_molecule_stats.columns:
        plot_df = plot_df[plot_df["rmse_kcal_mol"] <= _PLOT_RMSE_MAX].copy()
    if len(plot_df) == 0:
        return

    preferred_order = ["bespoke", "espaloma", "bespokefit_1", "sage"]
    available = list(plot_df["force_field"].unique())
    order = [l for l in preferred_order if l in available] + [
        l for l in available if l not in preferred_order
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(data=plot_df, ax=ax, x="force_field", y=metric_column, order=order, palette="Set2")
    sns.swarmplot(data=plot_df, ax=ax, x="force_field", y=metric_column, order=order, color="k", alpha=0.5, size=1.25)

    candidate_pairs = [
        ("bespoke", "sage"),
        ("bespoke", "bespokefit_1"),
        ("bespoke", "espaloma"),
        ("bespokefit_1", "sage"),
        ("espaloma", "sage"),
    ]
    pivot = plot_df.pivot(index="molecule", columns="force_field", values=metric_column)
    valid_pairs, p_values = [], []
    for a, b in candidate_pairs:
        if a not in pivot.columns or b not in pivot.columns:
            continue
        pair_df = pivot[[a, b]].dropna()
        if len(pair_df) == 0:
            continue
        n = len(pair_df)
        n_pos = int(np.sum(pair_df[a].to_numpy() < pair_df[b].to_numpy()))
        valid_pairs.append((a, b))
        p_values.append(float(binomtest(n_pos, n, 0.5).pvalue))

    if valid_pairs:
        annotator = Annotator(ax, valid_pairs, data=plot_df, x="force_field", y=metric_column, order=order)
        annotator.configure(test=None, text_format="star", loc="inside")
        annotator.set_pvalues(p_values)
        annotator.annotate()

    ax.set_xlabel("force_field")
    ax.set_ylabel(metric_column)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_rmse_vs_nonbonded_range(
    per_molecule_stats: pd.DataFrame,
    output_path: Path,
) -> None:
    required = ["rmse_kcal_mol", "nonbonded_range_kcal_mol", "force_field"]
    fig, ax = plt.subplots(figsize=(8, 6))

    if len(per_molecule_stats) == 0 or not all(c in per_molecule_stats.columns for c in required):
        ax.text(0.5, 0.5, "No MM component ranges available", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    plot_df = (
        per_molecule_stats
        .dropna(subset=required)
        .pipe(lambda df: df[df["rmse_kcal_mol"] <= _PLOT_RMSE_MAX])
    )
    if len(plot_df) == 0:
        ax.text(0.5, 0.5, "No data after filtering", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    sns.scatterplot(data=plot_df, x="nonbonded_range_kcal_mol", y="rmse_kcal_mol",
                    hue="force_field", style="force_field", s=60, alpha=0.85, ax=ax)
    ax.set_xlabel("Non-bonded energy range across conformers / kcal mol$^{-1}$")
    ax.set_ylabel("RMSE / kcal mol$^{-1}$")
    ax.set_title("Per-molecule RMSE vs non-bonded energy range")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_relative_component_energies(
    molecule_df: pd.DataFrame,
    method_names: list[str],
    output_path: Path,
) -> None:
    if "geom" not in molecule_df.columns:
        return

    component_cols_by_method = {
        method: sorted(
            c for c in molecule_df.columns if c.startswith(f"{method}__component__")
        )
        for method in method_names
    }
    component_cols_by_method = {k: v for k, v in component_cols_by_method.items() if v}
    if not component_cols_by_method:
        return

    n = len(component_cols_by_method)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n))
    if n == 1:
        axes = [axes]

    geom_labels = molecule_df["geom"].astype(str).tolist()
    x = np.arange(len(geom_labels), dtype=int)

    for ax, (method, comp_cols) in zip(axes, component_cols_by_method.items()):
        for col in comp_cols:
            vals = molecule_df[col].to_numpy(float)
            mask = np.isfinite(vals)
            if mask.sum() < 1:
                continue
            rel = vals.copy()
            rel[mask] -= np.mean(rel[mask])
            comp_label = col.split("__component__", 1)[1]
            ax.plot(x[mask], rel[mask], marker="o", linestyle="-",
                    linewidth=1.1, markersize=3, label=comp_label)

        overall = molecule_df[method].to_numpy(float)
        mask = np.isfinite(overall)
        if mask.sum() >= 1:
            rel = overall.copy()
            rel[mask] -= np.min(rel[mask])
            ax.plot(x[mask], rel[mask], linestyle="-", linewidth=2.2,
                    color="dimgray", alpha=0.8, label="overall_relative_min0")

        ax.axhline(0.0, linestyle="--", color="k", linewidth=0.8)
        ax.set_ylabel("Relative component E / kcal mol$^{-1}$")
        ax.set_title(_method_label(method))
        ax.legend(fontsize=7, ncol=2)

    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(geom_labels, rotation=90, fontsize=7)
    axes[-1].set_xlabel("Conformer")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyse_folmsbee(
    folmsbee_repo_dir: Path,
    presto_output_dir: Path,
    output_dir: Path,
    force_field_paths: list[str],
    precomputed_methods: list[str],
    mlp_names: list[str] | None = None,
    single_point_mlp: bool = True,
    reference_method: str = "dlpno",
    torsion_restraint_force_constant: float = 10_000.0,
    mm_minimization_steps: int = 0,
    n_processes: int | None = None,
) -> None:
    """Evaluate and compare conformer energies from the Folmsbee/Hutchison benchmark.

    Parameters
    ----------
    folmsbee_repo_dir:
        Root of the Folmsbee benchmark repository (contains ``data-final.csv``
        and ``geometries/``).
    presto_output_dir:
        Directory of PRESTO bespoke FF outputs; only molecules with a
        ``bespoke_force_field.offxml`` here are included.
    output_dir:
        Root output directory for results, per-molecule data, and plots.
    force_field_paths:
        SMIRNOFF force field file paths to evaluate with OpenMM.
    precomputed_methods:
        Column names already present in ``data-final.csv`` to include as
        reference methods (e.g. ``["ani2x", "gfn2"]``).
    mlp_names:
        OpenMM-ML potential names (e.g. ``["ani2x", "mace-off"]``) to evaluate.
        If ``None`` or empty, no MLP evaluation is performed.
    single_point_mlp:
        If ``True``, evaluate MLPs as single-point energies on the input
        geometries.  If ``False``, minimise under the MLP (with torsion
        restraints) as for MM force fields.
    reference_method:
        Column name in ``data-final.csv`` to use as the QM reference.
    torsion_restraint_force_constant:
        Force constant (kcal mol⁻¹ rad⁻²) for torsion restraints during
        minimisation.
    mm_minimization_steps:
        Maximum OpenMM minimisation steps (``0`` = converge to tolerance,
        ``-1`` = single-point, no minimisation).
    n_processes:
        Worker processes for parallelism.  Defaults to ``cpu_count - 1``.
    """
    mlp_names = mlp_names or []

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    per_molecule_root = output_dir / "per_molecule"
    plot_dir.mkdir(parents=True, exist_ok=True)
    per_molecule_root.mkdir(parents=True, exist_ok=True)

    reference_csv = folmsbee_repo_dir / "data-final.csv"
    if not reference_csv.exists():
        raise FileNotFoundError(f"Reference CSV not found: {reference_csv}")

    reference_df = pd.read_csv(reference_csv)
    required_columns = ["name", "geom", reference_method, *precomputed_methods]
    missing = [c for c in required_columns if c not in reference_df.columns]
    if missing:
        raise ValueError(f"Missing columns in {reference_csv}: {missing}")

    # Pre-allocate columns for FF/MLP results so downstream merges are safe.
    all_eval_methods = [*force_field_paths, *mlp_names]
    for col in [*all_eval_methods, *[f"{m}__rmsd_angstrom" for m in all_eval_methods]]:
        if col not in reference_df.columns:
            reference_df[col] = np.nan

    molecule_geometry_dirs = {
        k: str(v) for k, v in _resolve_molecule_geometry_dirs(folmsbee_repo_dir).items()
    }
    molecule_names_to_include = _collect_presto_molecules(presto_output_dir)
    logger.info(f"Loaded {len(molecule_names_to_include)} molecules from PRESTO output")

    included_df = reference_df[reference_df["name"].isin(molecule_names_to_include)]
    tasks = [
        (str(name), [str(g) for g in rows["geom"].tolist()])
        for name, rows in included_df.groupby("name")
    ]

    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)
    logger.info(f"Using {n_processes} process(es)")

    init_args = (
        force_field_paths,
        mlp_names,
        molecule_geometry_dirs,
        torsion_restraint_force_constant,
        mm_minimization_steps,
        str(per_molecule_root),
        single_point_mlp,
    )

    pool = None
    if mlp_names:
        # MLP/OpenMM-ML stacks can retain substantial GPU/host memory across
        # tasks. Use spawned workers and recycle each worker after one task to
        # keep memory bounded.
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(
            processes=n_processes,
            initializer=_initialise_worker,
            initargs=init_args,
            maxtasksperchild=1,
        )
        results_iter: Iterable = pool.imap(_process_molecule_worker, tasks)
    elif n_processes == 1:
        _initialise_worker(*init_args)
        results_iter = map(_process_molecule_worker, tasks)
    else:
        pool = mp.Pool(
            processes=n_processes,
            initializer=_initialise_worker,
            initargs=init_args,
        )
        results_iter = pool.imap(_process_molecule_worker, tasks)

    try:
        for molecule_name, geom_results in tqdm(
            results_iter, total=len(tasks), desc="Evaluating conformers", unit="mol"
        ):
            mol_mask = reference_df["name"] == molecule_name
            for geometry, values in geom_results.items():
                row_mask = mol_mask & (reference_df["geom"] == geometry)
                if not row_mask.any():
                    continue
                idx = reference_df[row_mask].index
                for key, value in values.items():
                    if key != "geom":
                        reference_df.loc[idx, key] = value
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # ---- Per-molecule stats and plots ------------------------------------
    method_names = [*precomputed_methods, *force_field_paths, *mlp_names]
    per_molecule_stats: list[dict] = []
    included_results_df = reference_df[reference_df["name"].isin(molecule_names_to_include)].copy()

    for molecule_name, molecule_df in tqdm(
        list(included_results_df.groupby("name")),
        desc="Computing per-molecule stats/plots",
        unit="mol",
    ):
        per_mol_dir = per_molecule_root / molecule_name
        per_mol_dir.mkdir(parents=True, exist_ok=True)
        molecule_rows: list[dict] = []

        for method_name in method_names:
            filtered = molecule_df.dropna(subset=[reference_method, method_name])
            if len(filtered) < 2:
                continue
            st = _calculate_stats(
                filtered[reference_method].to_numpy(float),
                filtered[method_name].to_numpy(float),
            )
            row: dict = {
                "molecule": molecule_name,
                "method": method_name,
                "n_conformers": len(filtered),
                **{k: st[k] for k in ("r2", "kendall_tau")},
                "rmse_kcal_mol": st["rmse"],
                "mae_kcal_mol": st["mae"],
            }

            rmsd_col = f"{method_name}__rmsd_angstrom"
            if rmsd_col in molecule_df.columns:
                rmsd_vals = molecule_df[rmsd_col].dropna().to_numpy(float)
                if len(rmsd_vals) > 0:
                    row["mean_rmsd_angstrom"] = float(np.mean(rmsd_vals))
                    row["rms_rmsd_angstrom"] = float(np.sqrt(np.mean(rmsd_vals**2)))

            # Non-bonded energy range (MM only; MLPs don't have components)
            nonbonded_tokens = ("vdw", "nonbonded", "lennard", "lj", "buckingham")
            nb_cols = [
                c for c in molecule_df.columns
                if c.startswith(f"{method_name}__component__")
                and any(t in c.lower() for t in nonbonded_tokens)
            ]
            if nb_cols:
                nb_matrix = molecule_df[nb_cols].to_numpy(float)
                finite_rows = np.all(np.isfinite(nb_matrix), axis=1)
                if finite_rows.sum() > 1:
                    nb_total = np.sum(nb_matrix[finite_rows], axis=1)
                    row["nonbonded_range_kcal_mol"] = float(np.ptp(nb_total))

            per_molecule_stats.append(row)
            molecule_rows.append(row)

        plot_input_df = molecule_df.dropna(subset=[reference_method])
        if len(plot_input_df) >= 2:
            _plot_correlation(
                molecule_df=plot_input_df,
                molecule_name=molecule_name,
                method_names=method_names,
                reference_method=reference_method,
                output_path=per_mol_dir / "correlation.png",
            )

        _plot_relative_component_energies(
            molecule_df=molecule_df,
            method_names=method_names,
            output_path=per_mol_dir / "relative_component_energies.png",
        )

        pd.DataFrame(molecule_rows).to_csv(per_mol_dir / "per_molecule_stats.csv", index=False)

    # ---- Aggregate stats ------------------------------------------------
    aggregate_rows: list[dict] = []
    for method_name in tqdm(method_names, desc="Computing aggregate stats", unit="method"):
        per_mol_ref, per_mol_method = [], []
        for _, molecule_df in included_results_df.groupby("name"):
            filtered = molecule_df.dropna(subset=[reference_method, method_name])
            if len(filtered) < 2:
                continue
            per_mol_ref.append(_safe_relative(filtered[reference_method].to_numpy(float)))
            per_mol_method.append(_safe_relative(filtered[method_name].to_numpy(float)))

        if not per_mol_ref:
            continue

        ref_all = np.concatenate(per_mol_ref)
        met_all = np.concatenate(per_mol_method)
        st = _calculate_stats(ref_all, met_all)
        agg_row: dict = {
            "method": method_name,
            "n_conformers": len(ref_all),
            "n_molecules": len(per_mol_ref),
            **{k: st[k] for k in ("r2", "kendall_tau")},
            "rmse_kcal_mol": st["rmse"],
            "mae_kcal_mol": st["mae"],
        }
        rmsd_col = f"{method_name}__rmsd_angstrom"
        if rmsd_col in included_results_df.columns:
            rmsd_vals = included_results_df[rmsd_col].dropna().to_numpy(float)
            if len(rmsd_vals) > 0:
                agg_row["mean_rmsd_angstrom"] = float(np.mean(rmsd_vals))
                agg_row["rms_rmsd_angstrom"] = float(np.sqrt(np.mean(rmsd_vals**2)))
        aggregate_rows.append(agg_row)

    aggregate_stats_df = pd.DataFrame(aggregate_rows)
    per_molecule_stats_df = pd.DataFrame(per_molecule_stats)

    if len(per_molecule_stats_df) > 0:
        per_molecule_stats_df["force_field"] = per_molecule_stats_df["method"].map(
            _overall_force_field_label
        )

    # ---- Save outputs ---------------------------------------------------
    reference_df.to_csv(output_dir / "results.csv", index=False)
    per_molecule_stats_df.to_csv(output_dir / "per_molecule_stats.csv", index=False)
    aggregate_stats_df.to_csv(output_dir / "aggregate_stats.csv", index=False)

    _plot_overall_metric_distribution(per_molecule_stats_df, "rmse_kcal_mol",
                                      plot_dir / "overall_rmse_distribution.png")
    _plot_overall_metric_distribution(per_molecule_stats_df, "mae_kcal_mol",
                                      plot_dir / "overall_mae_distribution.png")
    _plot_overall_metric_distribution(per_molecule_stats_df, "mean_rmsd_angstrom",
                                      plot_dir / "overall_rmsd_distribution.png")
    _plot_rmse_vs_nonbonded_range(per_molecule_stats_df, plot_dir / "rmse_vs_nonbonded_range.png")

    logger.info(f"Saved results to {output_dir}")
