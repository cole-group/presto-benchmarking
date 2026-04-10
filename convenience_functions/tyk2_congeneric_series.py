"""Utilities for TYK2 congeneric-series retraining and analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from presto.analyse import read_errors
from presto.outputs import OutputStage, OutputType, StageKind
from presto.settings import PreComputedDatasetSettings, WorkflowSettings


_VALENCE_TYPES = ("Bonds", "Angles", "ProperTorsions", "ImproperTorsions")


@dataclass(frozen=True)
class _RunSpec:
    max_extend_distance: int
    run_id: str
    fit_dir: Path

def _get_path_manager_for_run(
    run_dir: Path,
    fallback_config_path: Path | None = None,
) -> object:
    workflow_settings_path = run_dir / "workflow_settings.yaml"
    settings_path: Path | None = workflow_settings_path if workflow_settings_path.exists() else None
    if settings_path is None:
        settings_path = fallback_config_path

    if settings_path is None or not settings_path.exists():
        raise FileNotFoundError(
            "Could not construct WorkflowPathManager for run because no workflow settings "
            f"were found at {workflow_settings_path} and no existing fallback config was provided."
        )

    settings = WorkflowSettings.from_yaml(settings_path)
    settings.output_dir = run_dir.resolve()
    return settings.get_path_manager()


def _iter_scatter_hdf5_paths(path_manager: object) -> list[tuple[int, Path]]:
    hdf5_paths: list[tuple[int, Path]] = []
    stage = OutputStage(StageKind.TRAINING, 1)
    for mol_idx in range(path_manager.n_mols):
        path = path_manager.get_output_path_for_mol(stage, OutputType.SCATTER, mol_idx)
        hdf5_paths.append((mol_idx, path))

    missing = [path for _, path in hdf5_paths if not path.exists()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(
            "Expected per-molecule training scatter files for iteration 1, but some are missing. "
            f"Examples: {preview}"
        )

    if not hdf5_paths:
        raise FileNotFoundError(
            "No scatter outputs found from WorkflowPathManager for training iteration 1."
        )
    return hdf5_paths


def _read_stage_errors(path_manager: object) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    per_molecule_rows: list[dict[str, float | int]] = []
    all_energy_per_atom: list[np.ndarray] = []
    all_force_component: list[np.ndarray] = []

    for molecule_index, hdf5_path in _iter_scatter_hdf5_paths(path_manager):
        # Reuse PRESTO's canonical scatter reader to avoid duplicating HDF5 parsing logic.
        errors = read_errors({0: hdf5_path})
        if 0 not in errors["energy_differences"]:
            raise KeyError(f"Missing energy_differences in: {hdf5_path}")
        if 0 not in errors["forces_differences"]:
            raise KeyError(f"Missing forces_differences in: {hdf5_path}")
        if "n_atoms" not in errors:
            raise KeyError(f"Missing n_atoms in: {hdf5_path}")

        n_atoms = int(errors["n_atoms"])
        if n_atoms <= 0:
            raise ValueError(f"n_atoms must be positive in: {hdf5_path}")

        energy_differences = np.asarray(errors["energy_differences"][0], dtype=float)
        forces_differences = np.asarray(errors["forces_differences"][0], dtype=float)

        energy_differences_per_atom = energy_differences / float(n_atoms)
        force_differences_reshaped = forces_differences.reshape(-1)

        per_molecule_rows.append(
            {
                "molecule_index": molecule_index,
                "n_atoms": n_atoms,
                "energy_rmse_per_atom_kcal_mol": float(np.sqrt(np.mean(energy_differences_per_atom**2))),
                "force_rmse_kcal_mol_angstrom": float(np.sqrt(np.mean(force_differences_reshaped**2))),
            }
        )

        all_energy_per_atom.append(energy_differences_per_atom)
        all_force_component.append(force_differences_reshaped)

    per_molecule_df = pd.DataFrame(per_molecule_rows).sort_values("molecule_index")
    return (
        per_molecule_df,
        np.concatenate(all_energy_per_atom),
        np.concatenate(all_force_component),
    )


def _format_extend_label(max_extend_distance: int) -> str:
    return "-1 (bespoke)" if max_extend_distance == -1 else str(max_extend_distance)


def _plot_metric_vs_max_extend(
    per_run_df: pd.DataFrame,
    value_column: str,
    ylabel: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    order = sorted(
        per_run_df["max_extend_distance"].astype(int).unique(),
        key=lambda value: (value != -1, value),
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    sns.stripplot(
        data=per_run_df,
        x="max_extend_distance",
        y=value_column,
        order=order,
        color="black",
        alpha=0.55,
        size=4,
        jitter=0.12,
        ax=ax,
    )
    sns.pointplot(
        data=per_run_df,
        x="max_extend_distance",
        y=value_column,
        order=order,
        estimator=np.mean,
        errorbar=("ci", 95),
        color="tab:blue",
        markers="o",
        linestyles="-",
        ax=ax,
    )

    ax.set_xlabel("Maximum Extend Distance")
    ax.set_ylabel(ylabel)
    ax.set_xticklabels([_format_extend_label(int(x)) for x in order])
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def prepare_tyk2_congeneric_retrain_configs(
    base_config_path: Path,
    initial_run_dir: Path,
    output_dir: Path,
    max_extend_distances: list[int],
) -> list[Path]:
    """Create retraining configs that reuse data from an initial congeneric run."""
    if not max_extend_distances:
        raise ValueError("max_extend_distances must be non-empty")

    initial_path_manager = _get_path_manager_for_run(
        run_dir=initial_run_dir,
        fallback_config_path=base_config_path,
    )

    initial_force_field = initial_path_manager.get_output_path(
        OutputStage(StageKind.INITIAL_STATISTICS),
        OutputType.OFFXML,
    )
    if not initial_force_field.exists():
        raise FileNotFoundError(
            "Initial bespoke force field not found. Expected: "
            f"{initial_force_field}"
        )

    training_dataset_paths = [
        initial_path_manager.get_output_path_for_mol(
            OutputStage(StageKind.TRAINING, 1),
            OutputType.ENERGIES_AND_FORCES,
            mol_idx,
        )
        for mol_idx in range(initial_path_manager.n_mols)
    ]
    testing_dataset_paths = [
        initial_path_manager.get_output_path_for_mol(
            OutputStage(StageKind.TESTING),
            OutputType.ENERGIES_AND_FORCES,
            mol_idx,
        )
        for mol_idx in range(initial_path_manager.n_mols)
    ]

    missing_training = [path for path in training_dataset_paths if not path.exists()]
    missing_testing = [path for path in testing_dataset_paths if not path.exists()]
    if missing_training or missing_testing:
        missing_preview = [*missing_training[:2], *missing_testing[:2]]
        raise FileNotFoundError(
            "Missing precomputed dataset directories in initial run outputs. "
            f"Examples: {', '.join(str(path) for path in missing_preview)}"
        )

    if len(training_dataset_paths) != len(testing_dataset_paths):
        raise ValueError(
            "Training and testing dataset path counts differ for initial run: "
            f"{len(training_dataset_paths)} != {len(testing_dataset_paths)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    for max_extend_distance in max_extend_distances:
        settings = WorkflowSettings.from_yaml(base_config_path)
        settings.n_iterations = 1
        settings.parameterisation_settings.initial_force_field = str(initial_force_field.resolve())
        settings.parameterisation_settings.msm_settings = None

        for valence_type in _VALENCE_TYPES:
            generation_settings = settings.parameterisation_settings.type_generation_settings.get(
                valence_type
            )
            if generation_settings is None:
                raise ValueError(
                    f"Missing type_generation_settings entry for {valence_type}"
                )
            generation_settings.max_extend_distance = max_extend_distance

        settings.training_sampling_settings = PreComputedDatasetSettings(dataset_paths=training_dataset_paths)
        settings.testing_sampling_settings = PreComputedDatasetSettings(dataset_paths=testing_dataset_paths)

        config_out = output_dir / f"max_extend_{max_extend_distance}.yaml"
        settings.to_yaml(config_out)
        generated.append(config_out)

    return generated


def analyse_tyk2_congeneric_retrains(
    initial_run_dir: Path,
    retrain_root_dir: Path,
    output_dir: Path,
    max_extend_distances: list[int],
    repeats: int,
) -> None:
    """Analyse MLP test-set errors across congeneric retrains."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if not max_extend_distances:
        raise ValueError("max_extend_distances must be non-empty")

    run_specs: list[_RunSpec] = [_RunSpec(max_extend_distance=-1, run_id="initial", fit_dir=initial_run_dir)]
    for max_extend_distance in max_extend_distances:
        for repeat in range(1, repeats + 1):
            fit_dir = retrain_root_dir / f"max_extend_{max_extend_distance}" / f"run_{repeat:02d}"
            if not fit_dir.exists():
                raise FileNotFoundError(f"Missing retrain fit directory: {fit_dir}")
            run_specs.append(
                _RunSpec(
                    max_extend_distance=max_extend_distance,
                    run_id=f"run_{repeat:02d}",
                    fit_dir=fit_dir,
                )
            )

    per_run_rows: list[dict[str, str | int | float]] = []
    per_molecule_rows: list[dict[str, str | int | float]] = []

    initial_settings_path = initial_run_dir / "workflow_settings.yaml"
    fallback_settings_path = initial_settings_path if initial_settings_path.exists() else None

    for spec in run_specs:
        if spec.max_extend_distance != -1:
            candidate = (
                retrain_root_dir
                / "configs"
                / f"max_extend_{spec.max_extend_distance}.yaml"
            )
            fallback_settings_path = candidate if candidate.exists() else fallback_settings_path

        path_manager = _get_path_manager_for_run(
            run_dir=spec.fit_dir,
            fallback_config_path=fallback_settings_path,
        )
        per_molecule_df, energy_per_atom, force_component = _read_stage_errors(path_manager)

        per_run_rows.append(
            {
                "max_extend_distance": spec.max_extend_distance,
                "run_id": spec.run_id,
                "fit_dir": str(spec.fit_dir),
                "energy_rmse_per_atom_kcal_mol": float(np.sqrt(np.mean(energy_per_atom**2))),
                "force_rmse_kcal_mol_angstrom": float(np.sqrt(np.mean(force_component**2))),
            }
        )

        per_molecule_df = per_molecule_df.copy()
        per_molecule_df["max_extend_distance"] = spec.max_extend_distance
        per_molecule_df["run_id"] = spec.run_id
        per_molecule_df["fit_dir"] = str(spec.fit_dir)
        per_molecule_rows.extend(per_molecule_df.to_dict(orient="records"))

    per_run_df = pd.DataFrame(per_run_rows)
    per_molecule_df = pd.DataFrame(per_molecule_rows)

    summary_df = (
        per_run_df.groupby("max_extend_distance", as_index=False)
        .agg(
            n_runs=("run_id", "count"),
            energy_rmse_per_atom_mean=("energy_rmse_per_atom_kcal_mol", "mean"),
            energy_rmse_per_atom_std=("energy_rmse_per_atom_kcal_mol", "std"),
            force_rmse_mean=("force_rmse_kcal_mol_angstrom", "mean"),
            force_rmse_std=("force_rmse_kcal_mol_angstrom", "std"),
        )
        .sort_values("max_extend_distance", key=lambda s: s.map(lambda v: (v != -1, v)))
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    per_run_df.to_csv(output_dir / "retrain_error_per_run.csv", index=False)
    per_molecule_df.to_csv(output_dir / "retrain_error_per_molecule.csv", index=False)
    summary_df.to_csv(output_dir / "retrain_error_summary.csv", index=False)

    _plot_metric_vs_max_extend(
        per_run_df=per_run_df,
        value_column="energy_rmse_per_atom_kcal_mol",
        ylabel=r"Per-Atom Energy RMSE / kcal mol$^{-1}$ atom$^{-1}$",
        output_path=output_dir / "energy_error_vs_max_extend_distance.png",
    )
    _plot_metric_vs_max_extend(
        per_run_df=per_run_df,
        value_column="force_rmse_kcal_mol_angstrom",
        ylabel=r"Force RMSE / kcal mol$^{-1}$ $\AA^{-1}$",
        output_path=output_dir / "force_error_vs_max_extend_distance.png",
    )
