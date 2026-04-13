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
from rdkit import Chem
from rdkit.Chem import Draw

from convenience_functions._stats import (
    bootstrap_ci as _bootstrap_ci,
    format_value_with_ci as _format_value_with_ci,
    js_distance as _js_distance,
    rmse as _get_rmse,
    rms as _rms,
)

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
_JSD_TEMPERATURE_K = 500.0

_METHOD_DISPLAY_NAMES = {
    "aimnet2": "AIMNet2",
    "presto": "presto",
    "espaloma": "espaloma 0.4.0",
    "openff23": "OpenFF 2.3",
    "ani2x": "ANI2x",
    "mp2": "MP2",
    "wb97": r"$\omega$ B97",
}

_SUMMARY_METHOD_ORDER = ["aimnet2", "presto", "espaloma", "openff23", "ani2x", "mp2", "wb97"]


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


def _safe_relative_min(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values - np.min(values)


def _method_key(method_name: str) -> str | None:
    lower = method_name.lower()
    if method_name.endswith("/combined_force_field.offxml"):
        return "presto"
    if "esp04" in lower or "espaloma" in lower:
        return "espaloma"
    if "openff_unconstrained-2.3.0" in lower or "sage" in lower:
        return "openff23"
    if lower in {"aimnet2", "mp2", "wb97"}:
        return lower
    if lower in {"ani2", "ani2x"}:
        return "ani2x"
    return None


def _method_display_name(method_name: str) -> str:
    key = _method_key(method_name)
    if key is None:
        return _method_label(method_name)
    return _METHOD_DISPLAY_NAMES[key]


def _find_method_column(method_names: Iterable[str], method_key: str) -> str | None:
    for method_name in method_names:
        if _method_key(method_name) == method_key:
            return method_name
    return None


def _reference_windows(reference_df: pd.DataFrame, reference_method: str) -> pd.DataFrame:
    required_cols = ["name", reference_method]
    missing = [col for col in required_cols if col not in reference_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for reference window calculation: {missing}")

    window_df = (
        reference_df.dropna(subset=required_cols)
        .groupby("name", as_index=False)[reference_method]
        .agg(ref_min="min", ref_max="max")
    )
    window_df["reference_energy_window_kcal_mol"] = window_df["ref_max"] - window_df["ref_min"]
    return window_df.rename(columns={"name": "molecule"})


def _calculate_core_metrics(reference: np.ndarray, method: np.ndarray) -> dict[str, float]:
    ref_rel = _safe_relative(reference)
    met_rel = _safe_relative(method)
    ref_rel_js = _safe_relative_min(reference)
    met_rel_js = _safe_relative_min(method)
    return {
        "rmse": _get_rmse(ref_rel, met_rel),
        "jsd": _js_distance(met_rel_js, ref_rel_js, temperature=_JSD_TEMPERATURE_K),
    }


def _write_exclusion_report(path: Path, title: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = [title, "", f"n_excluded={len(rows)}"]
    if rows:
        text.append("")
        text.extend(rows)
    path.write_text("\n".join(text))


def _write_if_missing(path: Path, payload: str) -> bool:
    if path.exists():
        logger.info(f"Skipping existing output: {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
    return True


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


def _load_smiles_by_molecule(folmsbee_repo_dir: Path) -> dict[str, str]:
    smiles_file = folmsbee_repo_dir / "SMILES" / "molecules.smi"
    if not smiles_file.exists():
        raise FileNotFoundError(f"SMILES file not found: {smiles_file}")

    smiles_map: dict[str, str] = {}
    for line in smiles_file.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 2:
            raise ValueError(f"Malformed molecules.smi line: {line}")
        smiles_map[parts[1]] = parts[0]
    return smiles_map


def _compile_smarts(smarts_patterns: list[str]) -> list[tuple[str, Chem.Mol]]:
    compiled: list[tuple[str, Chem.Mol]] = []
    for smarts in smarts_patterns:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            raise ValueError(f"Invalid SMARTS pattern: {smarts}")
        compiled.append((smarts, patt))
    return compiled


def _smarts_matches(smiles: str, compiled_patterns: list[tuple[str, Chem.Mol]]) -> list[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return [smarts for smarts, patt in compiled_patterns if mol.HasSubstructMatch(patt)]


def _build_filter_reports(
    reference_df: pd.DataFrame,
    molecule_names_to_include: set[str],
    smiles_by_molecule: dict[str, str],
    exclude_smarts: list[str],
    min_conformers_per_molecule: int,
    min_reference_energy_window: float,
    reference_method: str,
) -> tuple[set[str], list[str], list[str], list[str]]:
    candidate_molecules = sorted(molecule_names_to_include)
    ref_subset = reference_df[reference_df["name"].isin(candidate_molecules)].copy()

    conformer_counts = ref_subset.groupby("name", as_index=False).size().rename(columns={"size": "n_conformers"})
    low_conformer_rows = conformer_counts[conformer_counts["n_conformers"] < min_conformers_per_molecule]
    low_conformer_molecules = set(low_conformer_rows["name"].tolist())

    ref_windows = _reference_windows(ref_subset, reference_method)
    low_window_rows = ref_windows[ref_windows["reference_energy_window_kcal_mol"] < min_reference_energy_window]
    low_window_molecules = set(low_window_rows["molecule"].tolist())

    smarts_hits_rows: list[str] = []
    smarts_molecules: set[str] = set()
    compiled_smarts = _compile_smarts(exclude_smarts)
    if compiled_smarts:
        for molecule in candidate_molecules:
            smiles = smiles_by_molecule.get(molecule)
            if not smiles:
                raise ValueError(
                    "Missing SMILES for molecule while SMARTS filtering is enabled: "
                    f"{molecule}"
                )
            matches = _smarts_matches(smiles, compiled_smarts)
            if matches:
                smarts_molecules.add(molecule)
                smarts_hits_rows.append(f"{molecule}\t{smiles}\t{'; '.join(matches)}")

    kept = set(candidate_molecules) - smarts_molecules - low_conformer_molecules - low_window_molecules

    low_conformer_report = [
        f"{row['name']}\tn_conformers={int(row['n_conformers'])}\tthreshold={min_conformers_per_molecule}"
        for _, row in low_conformer_rows.sort_values("name").iterrows()
    ]
    low_window_report = [
        f"{row['molecule']}\twindow_kcal_mol={float(row['reference_energy_window_kcal_mol']):.6f}\tthreshold={min_reference_energy_window:.6f}"
        for _, row in low_window_rows.sort_values("molecule").iterrows()
    ]

    return kept, smarts_hits_rows, low_conformer_report, low_window_report


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


def _partition_cached_results_and_tasks(
    tasks: list[tuple[str, list[str]]],
    per_molecule_root: Path,
    force_field_paths: list[str],
    mlp_names: list[str],
    single_point_mlp: bool,
) -> tuple[dict[str, dict[str, dict]], list[tuple[str, list[str]]]]:
    """Split tasks into fully cached molecules and molecules needing recompute."""
    cached_results: dict[str, dict[str, dict]] = {}
    tasks_to_compute: list[tuple[str, list[str]]] = []

    mm_required_cols = [col for ff in force_field_paths for col in (ff, f"{ff}__rmsd_angstrom")]
    mlp_required_cols = list(mlp_names)
    if not single_point_mlp:
        mlp_required_cols.extend(f"{mlp}__rmsd_angstrom" for mlp in mlp_names)

    for molecule_name, geometries in tasks:
        molecule_dir = per_molecule_root / molecule_name
        mm_df = _load_cached_csv(molecule_dir / "mm_results.csv")
        mlp_df = _load_cached_csv(molecule_dir / "mlp_results.csv")

        mm_complete = True
        if mm_required_cols:
            mm_complete = mm_df is not None and _cached_columns_complete(mm_df, mm_required_cols, geometries)

        mlp_complete = True
        if mlp_required_cols:
            mlp_complete = mlp_df is not None and _cached_columns_complete(mlp_df, mlp_required_cols, geometries)

        if not (mm_complete and mlp_complete):
            tasks_to_compute.append((molecule_name, geometries))
            continue

        cached_results[molecule_name] = _read_results_from_cache(
            geometries=geometries,
            ff_paths=force_field_paths,
            mlp_names=mlp_names,
            mm_df=mm_df,
            mlp_df=mlp_df,
        )

    return cached_results, tasks_to_compute


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
    if output_path.exists():
        logger.info(f"Skipping existing output: {output_path}")
        return
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
    if output_path.exists():
        logger.info(f"Skipping existing output: {output_path}")
        return
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
    if output_path.exists():
        logger.info(f"Skipping existing output: {output_path}")
        return
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
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_relative_component_energies(
    molecule_df: pd.DataFrame,
    method_names: list[str],
    output_path: Path,
) -> None:
    if output_path.exists():
        logger.info(f"Skipping existing output: {output_path}")
        return
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


def _build_ff_metric_dataframe(
    results_df: pd.DataFrame,
    per_molecule_stats_df: pd.DataFrame,
    method_names: list[str],
    reference_method: str,
    smiles_by_molecule: dict[str, str],
    shared_molecule_methods: list[str] = ["presto", "aimnet2", "espaloma", "openff23"]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rmse_lookup = (
        per_molecule_stats_df[["molecule", "method", "rmse_kcal_mol"]]
        .dropna(subset=["molecule", "method", "rmse_kcal_mol"])
        .drop_duplicates(subset=["molecule", "method"], keep="last")
        .set_index(["molecule", "method"])["rmse_kcal_mol"]
    )
    rmsd_lookup = (
        per_molecule_stats_df[["molecule", "method", "rms_rmsd_angstrom"]]
        .dropna(subset=["molecule", "method", "rms_rmsd_angstrom"])
        .drop_duplicates(subset=["molecule", "method"], keep="last")
        .set_index(["molecule", "method"])["rms_rmsd_angstrom"]
        if "rms_rmsd_angstrom" in per_molecule_stats_df.columns
        else pd.Series(dtype=float)
    )

    methods_enforce_shared_molecules = [m for m in method_names if _method_key(m) in shared_molecule_methods]
    molecules_by_method: dict[str, set[str]] = {}
    for method_name in methods_enforce_shared_molecules:
        valid = results_df[["name", method_name, reference_method]].dropna()
        counts = valid.groupby("name").size()
        molecules_by_method[method_name] = set(counts[counts >= 2].index.tolist())

    shared_molecules = (
        set.intersection(*molecules_by_method.values()) if molecules_by_method else set()
    )

    rows: list[dict] = []
    presto_vs_aimnet2_rows: list[dict] = []
    for molecule, group in results_df.groupby("name", sort=False):
        if molecule not in shared_molecules:
            continue

        for method_name in method_names:
            method_key = _method_key(method_name)
            if method_key is None:
                continue
            valid = group[[method_name, reference_method]].dropna()
            if len(valid) < 2:
                continue

            metrics = _calculate_core_metrics(
                reference=valid[reference_method].to_numpy(float),
                method=valid[method_name].to_numpy(float),
            )
            rmse_precomputed = rmse_lookup.get((molecule, method_name), np.nan)
            rmsd_precomputed = rmsd_lookup.get((molecule, method_name), np.nan)
            rows.append(
                {
                    "molecule": molecule,
                    "smiles": smiles_by_molecule.get(molecule),
                    "force_field": method_key,
                    "force_field_col": method_name,
                    "rmse": float(rmse_precomputed) if np.isfinite(rmse_precomputed) else metrics["rmse"],
                    "jsd": metrics["jsd"],
                    "rmsd": float(rmsd_precomputed) if np.isfinite(rmsd_precomputed) else np.nan,
                }
            )

        presto_col = _find_method_column(method_names, "presto")
        aimnet2_col = _find_method_column(method_names, "aimnet2")
        if presto_col is None or aimnet2_col is None:
            continue
        valid = group[[presto_col, aimnet2_col]].dropna()
        if len(valid) < 2:
            continue
        metrics = _calculate_core_metrics(
            reference=valid[aimnet2_col].to_numpy(float),
            method=valid[presto_col].to_numpy(float),
        )
        presto_vs_aimnet2_rows.append(
            {
                "molecule": molecule,
                "smiles": smiles_by_molecule.get(molecule),
                "rmse": metrics["rmse"],
                "jsd": metrics["jsd"],
                "rmsd": np.nan,
            }
        )

    ff_metric_df = pd.DataFrame(rows)
    presto_vs_aimnet2_df = pd.DataFrame(presto_vs_aimnet2_rows)

    if not presto_vs_aimnet2_df.empty:
        presto_vs_aimnet2_df = presto_vs_aimnet2_df[
            presto_vs_aimnet2_df["molecule"].isin(shared_molecules)
        ].copy()

    return ff_metric_df, presto_vs_aimnet2_df


def _save_figure_if_missing(fig: plt.Figure, output_path: Path) -> bool:
    if output_path.exists():
        logger.info(f"Skipping existing output: {output_path}")
        plt.close(fig)
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def _paired_metric_dataframe(
    ff_metric_df: pd.DataFrame,
    metric_keys: list[str],
    method_keys: list[str],
) -> tuple[pd.DataFrame, dict[str, int], int]:
    subset = ff_metric_df[ff_metric_df["force_field"].isin(method_keys)].copy()
    count_by_method = (
        subset.dropna(subset=["molecule", *metric_keys])
        .groupby("force_field")
        .agg(n_molecules=("molecule", "nunique"))
        .to_dict()["n_molecules"]
    )

    pivoted: dict[str, pd.DataFrame] = {}
    molecule_sets: list[set[str]] = []
    for metric_key in metric_keys:
        metric_pivot = subset.pivot_table(
            index="molecule",
            columns="force_field",
            values=metric_key,
            aggfunc="first",
        )
        missing_methods = [m for m in method_keys if m not in metric_pivot.columns]
        if missing_methods:
            raise ValueError(f"Missing methods for paired metric '{metric_key}': {missing_methods}")
        metric_pivot = metric_pivot[method_keys].dropna()
        pivoted[metric_key] = metric_pivot
        molecule_sets.append(set(metric_pivot.index.tolist()))

    overlap = set.intersection(*molecule_sets) if molecule_sets else set()
    if not overlap:
        raise ValueError("No overlapping molecules available for paired analysis")

    frames: list[pd.DataFrame] = []
    ordered_overlap = sorted(overlap)
    for metric_key in metric_keys:
        frame = pivoted[metric_key].loc[ordered_overlap].reset_index().melt(
            id_vars="molecule",
            var_name="force_field",
            value_name=metric_key,
        )
        frames.append(frame)

    paired_df = frames[0]
    for frame in frames[1:]:
        paired_df = paired_df.merge(frame, on=["molecule", "force_field"], how="inner")
    return paired_df, {k: int(v) for k, v in count_by_method.items()}, len(overlap)


def _plot_violin_with_significance(
    ff_metric_df: pd.DataFrame,
    output_path: Path,
    method_order: list[str],
) -> None:
    metric_column = "rmse"
    plot_df = ff_metric_df.dropna(subset=[metric_column]).copy()
    plot_df["force_field_display"] = plot_df["force_field"].map(_METHOD_DISPLAY_NAMES)
    order = [_METHOD_DISPLAY_NAMES[m] for m in method_order if m in plot_df["force_field"].unique()]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=plot_df, x="force_field_display", y=metric_column, order=order, ax=ax, palette="Set2")
    sns.swarmplot(
        data=plot_df,
        x="force_field_display",
        y=metric_column,
        order=order,
        color="k",
        alpha=0.45,
        size=1.75,
        ax=ax,
    )

    candidate_pairs = [
        ("presto", "espaloma"),
        ("presto", "openff23"),
        ("presto", "aimnet2"),
    ]
    pivot = plot_df.pivot(index="molecule", columns="force_field", values=metric_column)
    valid_pairs: list[tuple[str, str]] = []
    p_values: list[float] = []
    for left, right in candidate_pairs:
        if left not in pivot.columns or right not in pivot.columns:
            continue
        pair_df = pivot[[left, right]].dropna()
        if len(pair_df) == 0:
            continue
        n = len(pair_df)
        n_pos = int(np.sum(pair_df[left].to_numpy() < pair_df[right].to_numpy()))
        valid_pairs.append((_METHOD_DISPLAY_NAMES[left], _METHOD_DISPLAY_NAMES[right]))
        p_values.append(float(binomtest(n_pos, n, 0.5).pvalue))

    if valid_pairs:
        annotator = Annotator(
            ax,
            valid_pairs,
            data=plot_df,
            x="force_field_display",
            y=metric_column,
            order=order,
        )
        annotator.configure(test=None, text_format="star", loc="inside")
        annotator.set_pvalues(p_values)
        annotator.annotate()

    ax.set_xlabel("Method")
    ax.set_ylabel(r"RMSE to QM / kcal mol$^{-1}$")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    _save_figure_if_missing(fig, output_path)


def _plot_paired_metrics_vs_qm(
    paired_df: pd.DataFrame,
    method_order: list[str],
    output_path: Path,
) -> None:
    metric_specs = [
        ("rmse", r"RMSE to QM / kcal mol$^{-1}$"),
        ("jsd", "JS Distance to QM (500 K)"),
    ]
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(14, 4.5), sharex=True)

    display_order = [_METHOD_DISPLAY_NAMES[m] for m in method_order]
    paired_df = paired_df.copy()
    paired_df["force_field_display"] = paired_df["force_field"].map(_METHOD_DISPLAY_NAMES)

    x_positions = {name: i for i, name in enumerate(display_order)}
    for ax, (metric_key, ylabel) in zip(axes, metric_specs):
        for molecule, group in paired_df.groupby("molecule"):
            x = [x_positions[_METHOD_DISPLAY_NAMES[k]] for k in method_order]
            y = [
                group.loc[group["force_field"] == k, metric_key].iloc[0]
                for k in method_order
            ]
            ax.plot(x, y, color="0.7", linewidth=0.7, alpha=0.35, zorder=1)

        for method_key in method_order:
            y = paired_df.loc[paired_df["force_field"] == method_key, metric_key].to_numpy(float)
            x = np.full_like(y, fill_value=x_positions[_METHOD_DISPLAY_NAMES[method_key]], dtype=float)
            ax.scatter(x, y, s=10, alpha=0.6, edgecolors="black", linewidths=0.2, zorder=2)

        ax.set_xticks(range(len(display_order)))
        ax.set_xticklabels(display_order, rotation=15)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Method")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    _save_figure_if_missing(fig, output_path)


def _save_summary_table_latex(
    ff_metric_df: pd.DataFrame,
    presto_vs_aimnet2_df: pd.DataFrame,
    method_order: list[str],
    output_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method_key in method_order:
        sub = ff_metric_df[ff_metric_df["force_field"] == method_key]
        if sub.empty:
            continue
        rmse_vals = sub["rmse"].to_numpy(float)
        jsd_vals = sub["jsd"].to_numpy(float)
        rmsd_vals = sub["rmsd"].dropna().to_numpy(float)
        row = {
            "Comparison": f"{_METHOD_DISPLAY_NAMES[method_key]} / QM",
            "N": int(sub["molecule"].nunique()),
            "RMS RMSE / kcal mol$^{-1}$": _format_value_with_ci(
                _rms(rmse_vals),
                _bootstrap_ci(rmse_vals, _rms),
            ),
            "RMS JS Distance (500 K)": _format_value_with_ci(
                _rms(jsd_vals),
                _bootstrap_ci(jsd_vals, _rms),
            ),
            "RMS RMSD / $\\AA$": (
                _format_value_with_ci(_rms(rmsd_vals), _bootstrap_ci(rmsd_vals, _rms))
                if len(rmsd_vals) > 0
                else "NA"
            ),
        }
        rows.append(row)

    if len(presto_vs_aimnet2_df) > 0:
        rows.append(
            {
                "Comparison": "presto / AIMNet2",
                "N": int(presto_vs_aimnet2_df["molecule"].nunique()),
                "RMS RMSE / kcal mol$^{-1}$": _format_value_with_ci(
                    _rms(presto_vs_aimnet2_df["rmse"].to_numpy(float)),
                    _bootstrap_ci(presto_vs_aimnet2_df["rmse"].to_numpy(float), _rms),
                ),
                "RMS JS Distance (500 K)": _format_value_with_ci(
                    _rms(presto_vs_aimnet2_df["jsd"].to_numpy(float)),
                    _bootstrap_ci(presto_vs_aimnet2_df["jsd"].to_numpy(float), _rms),
                ),
                "RMS RMSD / $\\AA$": "NA",
            }
        )

    summary_df = pd.DataFrame(rows)
    sortable = summary_df[summary_df["Comparison"] != "presto / AIMNet2"].copy()
    sortable["_sort_key"] = sortable["RMS RMSE / kcal mol$^{-1}$"].str.extract(r"\$([0-9.]+)")[0].astype(float)
    sortable = sortable.sort_values("_sort_key", ascending=False).drop(columns=["_sort_key"])
    remainder = summary_df[summary_df["Comparison"] == "presto / AIMNet2"]
    summary_df = pd.concat([sortable, remainder], ignore_index=True)

    if not output_path.exists():
        output_path.write_text(summary_df.to_latex(index=False, escape=False))
    else:
        logger.info(f"Skipping existing output: {output_path}")

    return summary_df


def _save_outlier_tables_and_grid(
    ff_metric_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 20,
) -> None:
    presto_df = ff_metric_df[ff_metric_df["force_field"] == "presto"][["molecule", "smiles", "rmse"]].rename(
        columns={"rmse": "presto_rmse_to_qm_kcal_mol"}
    )
    openff_df = ff_metric_df[ff_metric_df["force_field"] == "openff23"][["molecule", "rmse"]].rename(
        columns={"rmse": "openff23_rmse_to_qm_kcal_mol"}
    )
    merged = presto_df.merge(openff_df, on="molecule", how="inner")
    merged["delta_presto_minus_openff23_kcal_mol"] = (
        merged["presto_rmse_to_qm_kcal_mol"] - merged["openff23_rmse_to_qm_kcal_mol"]
    )

    worst_presto = merged.sort_values("presto_rmse_to_qm_kcal_mol", ascending=False).head(top_n)
    worst_worsening = merged.sort_values("delta_presto_minus_openff23_kcal_mol", ascending=False).head(top_n)

    _write_if_missing(
        output_dir / "worst_presto_outliers_top20.txt",
        worst_presto[["molecule", "smiles", "presto_rmse_to_qm_kcal_mol"]].to_string(index=False),
    )
    _write_if_missing(
        output_dir / "largest_presto_worsening_vs_openff23_top20.txt",
        worst_worsening[
            [
                "molecule",
                "smiles",
                "presto_rmse_to_qm_kcal_mol",
                "openff23_rmse_to_qm_kcal_mol",
                "delta_presto_minus_openff23_kcal_mol",
            ]
        ].to_string(index=False),
    )

    grid_path_png = output_dir / "worst_presto_outliers_grid.png"
    if grid_path_png.exists() or (output_dir / "worst_presto_outliers_grid.svg").exists():
        logger.info("Skipping existing outlier grid image")
        return

    mols: list[Chem.Mol] = []
    legends: list[str] = []
    for _, row in worst_presto.dropna(subset=["smiles"]).iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None:
            continue
        mols.append(mol)
        legends.append(
            f"{row['molecule']}\nRMSE: {float(row['presto_rmse_to_qm_kcal_mol']):.2f} kcal/mol"
        )

    if not mols:
        raise ValueError("No valid molecules available for worst PRESTO outlier grid image")

    grid_img = Draw.MolsToGridImage(mols, legends=legends, molsPerRow=4, subImgSize=(300, 250))
    if not hasattr(grid_img, "save"):
        raise TypeError(
            f"Expected Draw.MolsToGridImage to return a PIL-like image with .save(), got {type(grid_img)}"
        )
    grid_img.save(str(grid_path_png))


def _plot_paired_rmse_vs_reference_window(
    ff_metric_df: pd.DataFrame,
    ref_window_df: pd.DataFrame,
    left_method: str,
    right_method: str,
    output_path: Path,
) -> None:
    if output_path.exists():
        logger.info(f"Skipping existing output: {output_path}")
        return

    pair_df = (
        ff_metric_df[ff_metric_df["force_field"].isin([left_method, right_method])]
        [["molecule", "force_field", "rmse"]]
        .pivot_table(index="molecule", columns="force_field", values="rmse", aggfunc="first")
    )
    missing = [m for m in [left_method, right_method] if m not in pair_df.columns]
    if missing:
        raise ValueError(f"Cannot create paired RMSE-vs-window plot; missing methods: {missing}")

    pair_df = pair_df[[left_method, right_method]].dropna().reset_index()
    if len(pair_df) == 0:
        raise ValueError(
            f"No overlap for paired RMSE-vs-window plot: {left_method} vs {right_method}"
        )

    plot_df = pair_df.merge(
        ref_window_df[["molecule", "reference_energy_window_kcal_mol"]],
        on="molecule",
        how="inner",
    ).dropna(subset=["reference_energy_window_kcal_mol", left_method, right_method])
    if len(plot_df) == 0:
        raise ValueError(
            "No overlap between paired RMSE data and reference energy-window values"
        )

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    colors = np.where(
        plot_df[left_method] < plot_df[right_method],
        "#2ca02c",
        np.where(plot_df[left_method] > plot_df[right_method], "#d62728", "#7f7f7f"),
    )
    for i, row in plot_df.iterrows():
        x = float(row["reference_energy_window_kcal_mol"])
        ax.plot([x, x], [float(row[right_method]), float(row[left_method])], color=colors[i], alpha=0.55, linewidth=1.0)

    ax.scatter(
        plot_df["reference_energy_window_kcal_mol"],
        plot_df[right_method],
        color="#ff7f0e",
        edgecolors="black",
        linewidths=0.25,
        s=12,
        alpha=0.85,
        label=_METHOD_DISPLAY_NAMES[right_method],
    )
    ax.scatter(
        plot_df["reference_energy_window_kcal_mol"],
        plot_df[left_method],
        color="#1f77b4",
        edgecolors="black",
        linewidths=0.25,
        s=12,
        alpha=0.85,
        label=_METHOD_DISPLAY_NAMES[left_method],
    )

    ax.set_xscale("log")
    ax.set_xlabel("Reference Energy Window (max-min) / kcal mol$^{-1}$")
    ax.set_ylabel("RMSE to QM / kcal mol$^{-1}$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True, loc="upper left")
    fig.tight_layout()
    _save_figure_if_missing(fig, output_path)


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
    exclude_smarts: list[str] | None = None,
    min_conformers_per_molecule: int = 5,
    min_reference_energy_window: float = 0.0,
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
    exclude_smarts = exclude_smarts or []

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
    presto_molecule_names = _collect_presto_molecules(presto_output_dir)
    logger.info(f"Loaded {len(presto_molecule_names)} molecules from PRESTO output")

    smiles_by_molecule = _load_smiles_by_molecule(folmsbee_repo_dir)
    (
        molecule_names_to_include,
        smarts_excluded_rows,
        low_conformer_rows,
        low_window_rows,
    ) = _build_filter_reports(
        reference_df=reference_df,
        molecule_names_to_include=presto_molecule_names,
        smiles_by_molecule=smiles_by_molecule,
        exclude_smarts=exclude_smarts,
        min_conformers_per_molecule=min_conformers_per_molecule,
        min_reference_energy_window=min_reference_energy_window,
        reference_method=reference_method,
    )

    _write_exclusion_report(
        output_dir / "excluded_by_smarts.txt",
        "Excluded molecules by SMARTS match",
        smarts_excluded_rows,
    )
    _write_exclusion_report(
        output_dir / "excluded_by_min_conformers.txt",
        "Excluded molecules by minimum conformer threshold",
        low_conformer_rows,
    )
    _write_exclusion_report(
        output_dir / "excluded_by_min_reference_energy_window.txt",
        "Excluded molecules by minimum reference energy-window threshold",
        low_window_rows,
    )

    if not molecule_names_to_include:
        raise ValueError("All molecules were excluded by pre-analysis filters")

    logger.info(f"Molecules kept after filtering: {len(molecule_names_to_include)}")

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

    cached_results, tasks_to_compute = _partition_cached_results_and_tasks(
        tasks=tasks,
        per_molecule_root=per_molecule_root,
        force_field_paths=force_field_paths,
        mlp_names=mlp_names,
        single_point_mlp=single_point_mlp,
    )

    if cached_results:
        logger.info(
            f"Loaded cached per-molecule CSVs for {len(cached_results)}/{len(tasks)} molecules"
        )
        for molecule_name, geom_results in tqdm(
            cached_results.items(),
            total=len(cached_results),
            desc="Reading cached conformers",
            unit="mol",
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

    if tasks_to_compute:
        logger.info(
            f"Computing per-molecule results for {len(tasks_to_compute)}/{len(tasks)} molecules with missing cache"
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
            results_iter: Iterable = pool.imap(_process_molecule_worker, tasks_to_compute)
        elif n_processes == 1:
            _initialise_worker(*init_args)
            results_iter = map(_process_molecule_worker, tasks_to_compute)
        else:
            pool = mp.Pool(
                processes=n_processes,
                initializer=_initialise_worker,
                initargs=init_args,
            )
            results_iter = pool.imap(_process_molecule_worker, tasks_to_compute)

        try:
            for molecule_name, geom_results in tqdm(
                results_iter,
                total=len(tasks_to_compute),
                desc="Evaluating conformers",
                unit="mol",
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
    else:
        logger.info("All molecules have complete cached CSVs; skipping MM/MLP evaluation")

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
            core = _calculate_core_metrics(
                reference=filtered[reference_method].to_numpy(float),
                method=filtered[method_name].to_numpy(float),
            )
            row: dict = {
                "molecule": molecule_name,
                "method": method_name,
                "n_conformers": len(filtered),
                **{k: st[k] for k in ("r2", "kendall_tau")},
                "rmse_kcal_mol": core["rmse"],
                "js_distance_500k": core["jsd"],
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
        core = _calculate_core_metrics(reference=ref_all, method=met_all)
        agg_row: dict = {
            "method": method_name,
            "n_conformers": len(ref_all),
            "n_molecules": len(per_mol_ref),
            **{k: st[k] for k in ("r2", "kendall_tau")},
            "rmse_kcal_mol": core["rmse"],
            "js_distance_500k": core["jsd"],
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

    # ---- Detailed post-analysis artifacts -------------------------------
    ff_metric_df, presto_vs_aimnet2_df = _build_ff_metric_dataframe(
        results_df=included_results_df,
        per_molecule_stats_df=per_molecule_stats_df,
        method_names=method_names,
        reference_method=reference_method,
        smiles_by_molecule=smiles_by_molecule,
    )

    detailed_method_order = [
        m
        for m in _SUMMARY_METHOD_ORDER
        if m in set(ff_metric_df["force_field"].unique())
    ]
    missing_required = [m for m in _SUMMARY_METHOD_ORDER if m not in detailed_method_order]
    if missing_required:
        raise ValueError(
            "Missing required methods for detailed analysis: "
            + ", ".join(missing_required)
        )

    method_count_lines = ["Molecule counts by method"]
    for method_key in detailed_method_order:
        n_method = int(
            ff_metric_df[ff_metric_df["force_field"] == method_key]["molecule"].nunique()
        )
        method_count_lines.append(f"{_METHOD_DISPLAY_NAMES[method_key]}\tN={n_method}")
    _write_if_missing(output_dir / "method_molecule_counts.txt", "\n".join(method_count_lines))

    _plot_violin_with_significance(
        ff_metric_df=ff_metric_df,
        output_path=plot_dir / "overall_rmse_violin_with_significance.png",
        method_order=detailed_method_order,
    )

    paired_df, paired_counts, paired_overlap_n = _paired_metric_dataframe(
        ff_metric_df=ff_metric_df,
        metric_keys=["rmse", "jsd"],
        method_keys=detailed_method_order,
    )
    _write_if_missing(
        output_dir / "paired_method_molecule_counts.txt",
        "\n".join(
            [
                "Molecule counts for paired analysis",
                f"global_exact_overlap_N={paired_overlap_n}",
            ]
            + [
                f"{_METHOD_DISPLAY_NAMES[k]}\tavailable_N={paired_counts.get(k, 0)}\tpaired_N={paired_overlap_n}"
                for k in detailed_method_order
            ]
        ),
    )

    _plot_paired_metrics_vs_qm(
        paired_df=paired_df,
        method_order=detailed_method_order,
        output_path=plot_dir / "paired_stats_vs_qm_dlpno.png",
    )

    _save_summary_table_latex(
        ff_metric_df=ff_metric_df,
        presto_vs_aimnet2_df=presto_vs_aimnet2_df,
        method_order=detailed_method_order,
        output_path=output_dir / "summary_metrics_vs_qm_with_presto_vs_aimnet2.tex",
    )

    _save_outlier_tables_and_grid(
        ff_metric_df=ff_metric_df,
        output_dir=plot_dir,
        top_n=20,
    )

    ref_window_df = _reference_windows(included_results_df, reference_method)
    for right_method, out_name in [
        ("openff23", "paired_rmse_vs_reference_energy_window_presto_vs_openff23.png"),
        ("espaloma", "paired_rmse_vs_reference_energy_window_presto_vs_espaloma.png"),
        ("aimnet2", "paired_rmse_vs_reference_energy_window_presto_vs_aimnet2.png"),
    ]:
        if "presto" not in detailed_method_order or right_method not in detailed_method_order:
            continue
        _plot_paired_rmse_vs_reference_window(
            ff_metric_df=ff_metric_df,
            ref_window_df=ref_window_df,
            left_method="presto",
            right_method=right_method,
            output_path=plot_dir / out_name,
        )

    logger.info(f"Saved results to {output_dir}")
