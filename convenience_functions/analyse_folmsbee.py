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


_WORKER_FFS: dict[str, ForceField] | None = None
_WORKER_MOLECULE_DIRS: dict[str, str] | None = None
_WORKER_TORSION_K: float | None = None
_WORKER_MM_STEPS: int | None = None
_WORKER_PER_MOL_ROOT: str | None = None


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
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.arctan2(y, x))


def _method_label(method_name: str) -> str:
    if method_name.endswith("/combined_force_field.offxml"):
        return Path(method_name).parent.name
    if "/" in method_name:
        return Path(method_name).name
    return method_name


def _method_id(method_name: str) -> str:
    base = _method_label(method_name)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", base)


def _overall_force_field_label(method_name: str) -> str:
    method_lower = method_name.lower()
    if method_name.endswith("/combined_force_field.offxml"):
        return "bespoke"
    if "esp" in method_lower:
        return "espaloma"
    if "bespokefit" in method_lower:
        return "bespokefit_1"
    if "openff_unconstrained-2.3.0" in method_lower:
        return "sage"
    return _method_label(method_name)


def _safe_relative(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values - np.mean(values)


def _get_r_sq(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    _, _, r_value, _, _ = stats.linregress(x, y)
    return float(r_value**2)


def _get_kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    tau, _ = stats.kendalltau(x, y)
    return float(tau)


def _get_rmse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x - y) ** 2)))


def _get_mae(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(x - y)))


def _kabsch_rmsd(reference_xyz: np.ndarray, target_xyz: np.ndarray) -> float:
    ref = reference_xyz - np.mean(reference_xyz, axis=0)
    tgt = target_xyz - np.mean(target_xyz, axis=0)
    h = ref.T @ tgt
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    aligned = tgt @ rot
    diff = ref - aligned
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


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


def _evaluate_with_reused_system(
    reference_molecule: Molecule,
    conformer_positions: list,
    force_field: ForceField,
    torsion_restraint_force_constant: float,
    mm_minimization_steps: int,
) -> tuple[list[float], list[np.ndarray]]:
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
    force_indices: list[int] = []
    restraint_force_group: int | None = None
    if torsion_atoms:
        force_indices, restraint_force_group = _add_torsion_restraint_forces(
            simulation,
            torsion_atoms,
            torsion_restraint_force_constant,
        )

    energies: list[float] = []
    minimized_xyz_ang: list[np.ndarray] = []
    try:
        for coordinates in conformer_positions:
            simulation.context.setPositions(coordinates)
            if torsion_atoms:
                coordinates_nm = coordinates.value_in_unit(omm_unit.nanometer)
                current_angles = [
                    _compute_dihedral_radians(coordinates_nm, atoms)
                    for atoms in torsion_atoms
                ]
                _update_torsion_restraints(
                    simulation,
                    force_indices,
                    current_angles,
                    torsion_restraint_force_constant,
                )

            simulation.minimizeEnergy(maxIterations=mm_minimization_steps)

            if restraint_force_group is not None:
                groups_mask = sum(
                    1 << group for group in range(32) if group != restraint_force_group
                )
                state = simulation.context.getState(
                    getEnergy=True,
                    getPositions=True,
                    groups=groups_mask,
                )
            else:
                state = simulation.context.getState(getEnergy=True, getPositions=True)

            energy = state.getPotentialEnergy().value_in_unit(
                omm_unit.kilocalorie_per_mole
            )
            xyz = state.getPositions(asNumpy=True).value_in_unit(omm_unit.angstrom)
            energies.append(float(energy))
            minimized_xyz_ang.append(np.asarray(xyz))
    finally:
        if force_indices:
            _remove_torsion_restraint_forces(simulation, force_indices)

    return energies, minimized_xyz_ang


def _write_minimized_conformer_sdf(
    template_molecule: Molecule,
    xyz_angstrom: np.ndarray,
    output_path: Path,
) -> None:
    mol = Molecule(template_molecule)
    mol.conformers.clear()
    mol.add_conformer(off_unit.Quantity(xyz_angstrom, off_unit.angstrom))
    mol.to_file(str(output_path), file_format="sdf")


def _initialise_worker(
    force_field_paths: list[str],
    molecule_geometry_dirs: dict[str, str],
    torsion_restraint_force_constant: float,
    mm_minimization_steps: int,
    per_molecule_root: str,
) -> None:
    global _WORKER_FFS
    global _WORKER_MOLECULE_DIRS
    global _WORKER_TORSION_K
    global _WORKER_MM_STEPS
    global _WORKER_PER_MOL_ROOT
    _WORKER_FFS = _load_force_fields(force_field_paths)
    _WORKER_MOLECULE_DIRS = molecule_geometry_dirs
    _WORKER_TORSION_K = torsion_restraint_force_constant
    _WORKER_MM_STEPS = mm_minimization_steps
    _WORKER_PER_MOL_ROOT = per_molecule_root


def _process_molecule_worker(
    task: tuple[str, list[str]],
) -> tuple[str, dict[str, dict[str, float]]]:
    molecule_name, geometries = task
    if (
        _WORKER_FFS is None
        or _WORKER_MOLECULE_DIRS is None
        or _WORKER_TORSION_K is None
        or _WORKER_MM_STEPS is None
        or _WORKER_PER_MOL_ROOT is None
    ):
        raise RuntimeError("Worker globals not initialised")

    ff_ids = {ff_path: _method_id(ff_path) for ff_path in _WORKER_FFS}
    molecule_dir = Path(_WORKER_PER_MOL_ROOT) / molecule_name
    minimized_root = molecule_dir / "minimised"
    mm_results_path = molecule_dir / "mm_results.csv"

    cached_ok = mm_results_path.exists()
    if cached_ok:
        for ff_id in ff_ids.values():
            for geometry in geometries:
                geom_stem = Path(geometry).stem
                if not (minimized_root / ff_id / f"{geom_stem}.sdf").exists():
                    cached_ok = False
                    break
            if not cached_ok:
                break

    if cached_ok:
        mm_df = pd.read_csv(mm_results_path)
        results: dict[str, dict[str, float]] = {}
        for geometry in geometries:
            row = mm_df[mm_df["geom"] == geometry]
            geom_result: dict[str, float] = {}
            if len(row) == 1:
                row = row.iloc[0]
                for ff_path in _WORKER_FFS:
                    geom_result[ff_path] = float(row.get(ff_path, np.nan))
                    geom_result[f"{ff_path}__rmsd_angstrom"] = float(
                        row.get(f"{ff_path}__rmsd_angstrom", np.nan)
                    )
            else:
                for ff_path in _WORKER_FFS:
                    geom_result[ff_path] = np.nan
                    geom_result[f"{ff_path}__rmsd_angstrom"] = np.nan
            results[geometry] = geom_result
        return molecule_name, results

    geometry_dir_str = _WORKER_MOLECULE_DIRS.get(molecule_name)
    if geometry_dir_str is None:
        return molecule_name, {}

    loaded_geometries: list[str] = []
    loaded_conformer_positions: list = []
    loaded_input_xyz_ang: list[np.ndarray] = []
    loaded_templates: list[Molecule] = []
    reference_molecule: Molecule | None = None

    for geometry in geometries:
        openff_sdf = Path(geometry_dir_str) / Path(geometry).with_suffix(".openff.sdf")
        if not openff_sdf.exists():
            continue
        try:
            molecule = Molecule.from_file(str(openff_sdf))
        except Exception:
            continue
        if len(molecule.conformers) == 0:
            continue

        if reference_molecule is None:
            reference_molecule = Molecule(molecule)

        loaded_geometries.append(geometry)
        loaded_templates.append(Molecule(molecule))
        loaded_conformer_positions.append(molecule.conformers[0].to_openmm())
        loaded_input_xyz_ang.append(
            molecule.conformers[0].m_as(off_unit.angstrom)
        )

    if reference_molecule is None:
        return molecule_name, {}

    molecule_dir.mkdir(parents=True, exist_ok=True)
    minimized_root.mkdir(parents=True, exist_ok=True)

    mm_rows: list[dict[str, float | str]] = [{"geom": g} for g in loaded_geometries]
    results: dict[str, dict[str, float]] = {
        g: {} for g in loaded_geometries
    }

    for ff_path, force_field in _WORKER_FFS.items():
        ff_id = ff_ids[ff_path]
        ff_dir = minimized_root / ff_id
        ff_dir.mkdir(parents=True, exist_ok=True)
        try:
            energies, minimized_xyz_ang = _evaluate_with_reused_system(
                reference_molecule=reference_molecule,
                conformer_positions=loaded_conformer_positions,
                force_field=force_field,
                torsion_restraint_force_constant=_WORKER_TORSION_K,
                mm_minimization_steps=_WORKER_MM_STEPS,
            )
        except Exception:
            energies = [np.nan for _ in loaded_geometries]
            minimized_xyz_ang = [None for _ in loaded_geometries]

        for index, geometry in enumerate(loaded_geometries):
            energy = float(energies[index]) if index < len(energies) else np.nan
            mm_rows[index][ff_path] = energy * -1.0 if np.isfinite(energy) else np.nan

            rmsd_value = np.nan
            if index < len(minimized_xyz_ang) and minimized_xyz_ang[index] is not None:
                rmsd_value = _kabsch_rmsd(
                    loaded_input_xyz_ang[index],
                    minimized_xyz_ang[index],
                )
                output_sdf = ff_dir / f"{Path(geometry).stem}.sdf"
                _write_minimized_conformer_sdf(
                    loaded_templates[index],
                    minimized_xyz_ang[index],
                    output_sdf,
                )

            mm_rows[index][f"{ff_path}__rmsd_angstrom"] = rmsd_value
            results[geometry][ff_path] = mm_rows[index][ff_path]  # type: ignore[index]
            results[geometry][f"{ff_path}__rmsd_angstrom"] = rmsd_value

    pd.DataFrame(mm_rows).to_csv(mm_results_path, index=False)
    return molecule_name, results


def _collect_presto_molecules(presto_output_dir: Path) -> set[str]:
    molecule_names: set[str] = set()
    for path in presto_output_dir.iterdir():
        if path.is_dir() and (path / "bespoke_force_field.offxml").exists():
            molecule_names.add(path.name)
    return molecule_names


def _calculate_stats(reference: np.ndarray, method: np.ndarray) -> dict[str, float]:
    reference_rel = _safe_relative(reference)
    method_rel = _safe_relative(method)
    return {
        "r2": _get_r_sq(reference_rel, method_rel),
        "kendall_tau": _get_kendall_tau(reference_rel, method_rel),
        "rmse": _get_rmse(reference_rel, method_rel),
        "mae": _get_mae(reference_rel, method_rel),
    }


def _plot_correlation(
    molecule_df: pd.DataFrame,
    molecule_name: str,
    method_names: list[str],
    reference_method: str,
    output_path: Path,
) -> None:
    reference = molecule_df[reference_method].to_numpy(dtype=float)
    reference_rel = _safe_relative(reference)
    fig, ax = plt.subplots(figsize=(7, 6))
    min_value = float(np.min(reference_rel))
    max_value = float(np.max(reference_rel))

    for method_name in method_names:
        method_values = molecule_df[method_name].to_numpy(dtype=float)
        if np.isnan(method_values).any() or np.isnan(reference_rel).any():
            continue
        method_rel = _safe_relative(method_values)
        min_value = min(min_value, float(np.min(method_rel)))
        max_value = max(max_value, float(np.max(method_rel)))
        stats_values = _calculate_stats(reference, method_values)
        label = (
            f"{_method_label(method_name)} "
            f"(R²={stats_values['r2']:.2f}, "
            f"τ={stats_values['kendall_tau']:.2f}, "
            f"RMSE={stats_values['rmse']:.2f}, "
            f"MAE={stats_values['mae']:.2f})"
        )
        ax.scatter(reference_rel, method_rel, label=label)

    ax.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="k")
    ax.set_xlabel(f"{reference_method} rel. conformer AE / kcal mol$^{{-1}}$")
    ax.set_ylabel("Method rel. conformer AE / kcal mol$^{-1}$")
    ax.set_title(molecule_name)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_overall_metric_distribution(
    per_molecule_stats: pd.DataFrame,
    metric_column: str,
    output_path: Path,
) -> None:
    if len(per_molecule_stats) == 0:
        return

    if metric_column not in per_molecule_stats.columns:
        return

    plot_df = per_molecule_stats.dropna(subset=[metric_column]).copy()
    if len(plot_df) == 0:
        return

    preferred_order = ["bespoke", "espaloma", "bespokefit_1", "sage"]
    available = list(plot_df["force_field"].unique())
    order = [label for label in preferred_order if label in available]
    order.extend([label for label in available if label not in order])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(
        data=plot_df,
        ax=ax,
        x="force_field",
        y=metric_column,
        order=order,
        palette="Set2",
    )
    sns.swarmplot(
        data=plot_df,
        ax=ax,
        x="force_field",
        y=metric_column,
        order=order,
        color="k",
        alpha=0.5,
    )

    candidate_pairs = [
        ("bespoke", "sage"),
        ("bespoke", "bespokefit_1"),
        ("bespoke", "espaloma"),
        ("bespokefit_1", "sage"),
        ("espaloma", "sage"),
    ]
    valid_pairs: list[tuple[str, str]] = []
    p_values: list[float] = []

    pivot = plot_df.pivot(index="molecule", columns="force_field", values=metric_column)
    for pair in candidate_pairs:
        if pair[0] not in pivot.columns or pair[1] not in pivot.columns:
            continue
        pair_df = pivot[[pair[0], pair[1]]].dropna()
        if len(pair_df) == 0:
            continue
        data1 = pair_df[pair[0]].to_numpy(dtype=float)
        data2 = pair_df[pair[1]].to_numpy(dtype=float)
        n = len(data1)
        n_positive = int(np.sum(data1 < data2))
        p_value = float(binomtest(n_positive, n, 0.5).pvalue)
        valid_pairs.append(pair)
        p_values.append(p_value)

    if valid_pairs:
        annotator = Annotator(
            ax,
            valid_pairs,
            data=plot_df,
            x="force_field",
            y=metric_column,
            order=order,
        )
        annotator.configure(test=None, text_format="star", loc="inside")
        annotator.set_pvalues(p_values)
        annotator.annotate()

    ax.set_xlabel("force_field")
    ax.set_ylabel(metric_column)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyse_folmsbee(
    folmsbee_repo_dir: Path,
    presto_output_dir: Path,
    output_dir: Path,
    force_field_paths: list[str],
    precomputed_methods: list[str],
    reference_method: str = "dlpno",
    torsion_restraint_force_constant: float = 10_000.0,
    mm_minimization_steps: int = 0,
    n_processes: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    per_molecule_root = output_dir / "per_molecule"
    plot_dir.mkdir(parents=True, exist_ok=True)
    per_molecule_root.mkdir(parents=True, exist_ok=True)

    reference_csv = folmsbee_repo_dir / "data-final.csv"
    if not reference_csv.exists():
        raise FileNotFoundError(f"Could not find reference CSV: {reference_csv}")

    reference_df = pd.read_csv(reference_csv)
    required_columns = ["name", "geom", reference_method, *precomputed_methods]
    missing_columns = [c for c in required_columns if c not in reference_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {reference_csv}: {missing_columns}")

    rmsd_columns = [f"{ff}__rmsd_angstrom" for ff in force_field_paths]
    for column in [*force_field_paths, *rmsd_columns]:
        if column not in reference_df.columns:
            reference_df[column] = np.nan

    molecule_geometry_dirs = _resolve_molecule_geometry_dirs(folmsbee_repo_dir)
    molecule_geometry_dirs_str = {k: str(v) for k, v in molecule_geometry_dirs.items()}
    molecule_names_to_include = _collect_presto_molecules(presto_output_dir)
    logger.info(f"Loaded {len(molecule_names_to_include)} molecules from PRESTO output")

    included_df = reference_df[reference_df["name"].isin(molecule_names_to_include)]
    grouped_relevant_rows = list(included_df.groupby("name"))
    tasks = [
        (str(name), [str(g) for g in rows["geom"].tolist()])
        for name, rows in grouped_relevant_rows
    ]

    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)
    logger.info(f"Using {n_processes} process(es) for restrained minimisation")

    results_iter: Iterable[tuple[str, dict[str, dict[str, float]]]]
    if n_processes == 1:
        _initialise_worker(
            force_field_paths,
            molecule_geometry_dirs_str,
            torsion_restraint_force_constant,
            mm_minimization_steps,
            str(per_molecule_root),
        )
        results_iter = map(_process_molecule_worker, tasks)
    else:
        pool = mp.Pool(
            processes=n_processes,
            initializer=_initialise_worker,
            initargs=(
                force_field_paths,
                molecule_geometry_dirs_str,
                torsion_restraint_force_constant,
                mm_minimization_steps,
                str(per_molecule_root),
            ),
        )
        results_iter = pool.imap(_process_molecule_worker, tasks)

    try:
        for molecule_name, geom_results in tqdm(
            results_iter,
            total=len(tasks),
            desc="Evaluating molecule conformers",
            unit="mol",
        ):
            molecule_mask = reference_df["name"] == molecule_name
            for geometry, values in geom_results.items():
                row_mask = molecule_mask & (reference_df["geom"] == geometry)
                if not row_mask.any():
                    continue
                target_index = reference_df[row_mask].index
                for key, value in values.items():
                    reference_df.loc[target_index, key] = value
    finally:
        if n_processes != 1:
            pool.close()
            pool.join()

    method_names = [*precomputed_methods, *force_field_paths]
    per_molecule_stats: list[dict[str, float | str | int]] = []
    included_results_df = reference_df[
        reference_df["name"].isin(molecule_names_to_include)
    ].copy()

    for molecule_name, molecule_df in tqdm(
        list(included_results_df.groupby("name")),
        desc="Computing per-molecule stats/plots",
        unit="mol",
    ):
        per_mol_dir = per_molecule_root / molecule_name
        per_mol_dir.mkdir(parents=True, exist_ok=True)
        molecule_rows: list[dict[str, float | str | int]] = []

        for method_name in method_names:
            filtered_df = molecule_df.dropna(subset=[reference_method, method_name])
            if len(filtered_df) < 2:
                continue
            method_stats = _calculate_stats(
                filtered_df[reference_method].to_numpy(dtype=float),
                filtered_df[method_name].to_numpy(dtype=float),
            )
            row: dict[str, float | str | int] = {
                "molecule": molecule_name,
                "method": method_name,
                "n_conformers": len(filtered_df),
                "r2": method_stats["r2"],
                "kendall_tau": method_stats["kendall_tau"],
                "rmse_kcal_mol": method_stats["rmse"],
                "mae_kcal_mol": method_stats["mae"],
            }
            rmsd_col = f"{method_name}__rmsd_angstrom"
            if rmsd_col in molecule_df.columns:
                rmsd_values = molecule_df[rmsd_col].dropna().to_numpy(dtype=float)
                if len(rmsd_values) > 0:
                    row["mean_rmsd_angstrom"] = float(np.mean(rmsd_values))
                    row["rms_rmsd_angstrom"] = float(
                        np.sqrt(np.mean(rmsd_values**2))
                    )
            per_molecule_stats.append(row)
            molecule_rows.append(row)

        complete_plot_df = molecule_df.dropna(subset=[reference_method, *method_names])
        if len(complete_plot_df) >= 2:
            _plot_correlation(
                molecule_df=complete_plot_df,
                molecule_name=molecule_name,
                method_names=method_names,
                reference_method=reference_method,
                output_path=per_mol_dir / "correlation.png",
            )

        pd.DataFrame(molecule_rows).to_csv(
            per_mol_dir / "per_molecule_stats.csv",
            index=False,
        )

    aggregate_rows: list[dict[str, float | str | int]] = []
    for method_name in tqdm(method_names, desc="Computing aggregate stats", unit="method"):
        per_mol_ref: list[np.ndarray] = []
        per_mol_method: list[np.ndarray] = []

        for _, molecule_df in included_results_df.groupby("name"):
            filtered_df = molecule_df.dropna(subset=[reference_method, method_name])
            if len(filtered_df) < 2:
                continue
            per_mol_ref.append(_safe_relative(filtered_df[reference_method].to_numpy(float)))
            per_mol_method.append(_safe_relative(filtered_df[method_name].to_numpy(float)))

        if len(per_mol_ref) == 0:
            continue

        ref_concat = np.concatenate(per_mol_ref)
        method_concat = np.concatenate(per_mol_method)
        stats_values = _calculate_stats(ref_concat, method_concat)

        row = {
            "method": method_name,
            "n_conformers": len(ref_concat),
            "n_molecules": len(per_mol_ref),
            "r2": stats_values["r2"],
            "kendall_tau": stats_values["kendall_tau"],
            "rmse_kcal_mol": stats_values["rmse"],
            "mae_kcal_mol": stats_values["mae"],
        }

        rmsd_col = f"{method_name}__rmsd_angstrom"
        if rmsd_col in included_results_df.columns:
            rmsd_values = included_results_df[rmsd_col].dropna().to_numpy(dtype=float)
            if len(rmsd_values) > 0:
                row["mean_rmsd_angstrom"] = float(np.mean(rmsd_values))
                row["rms_rmsd_angstrom"] = float(np.sqrt(np.mean(rmsd_values**2)))

        aggregate_rows.append(row)

    aggregate_stats_df = pd.DataFrame(aggregate_rows)
    per_molecule_stats_df = pd.DataFrame(per_molecule_stats)

    if len(per_molecule_stats_df) > 0:
        per_molecule_stats_df["force_field"] = per_molecule_stats_df["method"].map(
            _overall_force_field_label
        )

    reference_df.to_csv(output_dir / "results.csv", index=False)
    per_molecule_stats_df.to_csv(output_dir / "per_molecule_stats.csv", index=False)
    aggregate_stats_df.to_csv(output_dir / "aggregate_stats.csv", index=False)

    _plot_overall_metric_distribution(
        per_molecule_stats_df,
        "rmse_kcal_mol",
        plot_dir / "overall_rmse_distribution.png",
    )
    _plot_overall_metric_distribution(
        per_molecule_stats_df,
        "mae_kcal_mol",
        plot_dir / "overall_mae_distribution.png",
    )
    _plot_overall_metric_distribution(
        per_molecule_stats_df,
        "mean_rmsd_angstrom",
        plot_dir / "overall_rmsd_distribution.png",
    )
    logger.info(f"Saved results to {output_dir}")

    logger.info(f"Saved results to {output_dir}")
