"""Bulk analysis of presto fit outputs across many fit directories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from convenience_functions.yammbs_torsion_analysis import (
    _bootstrap_ci,
    _format_header_with_units,
    _format_value_with_ci,
    _get_rms,
)

plt.style.use("ggplot")


@dataclass(frozen=True)
class IterationSpec:
    label: str
    stage_dir: str
    stage_index: int


ITERATION_SPECS: tuple[IterationSpec, ...] = (
    IterationSpec(label="Initial", stage_dir="initial_statistics", stage_index=0),
    IterationSpec(label="Iteration 1", stage_dir="training_iteration_1", stage_index=1),
    IterationSpec(label="Iteration 2", stage_dir="training_iteration_2", stage_index=2),
)


def _get_fit_dirs(presto_output_dir: Path) -> list[Path]:
    if not presto_output_dir.exists():
        raise FileNotFoundError(f"presto output directory not found: {presto_output_dir}")

    fit_dirs = sorted(path for path in presto_output_dir.iterdir() if path.is_dir() and path.name.isdigit())
    if not fit_dirs:
        raise FileNotFoundError(
            "No fit directories found under output directory. Expected numeric directories such as 0, 1, 2. "
            f"Path checked: {presto_output_dir}"
        )

    return fit_dirs


def _read_per_atom_energy_errors(hdf5_path: Path) -> np.ndarray:
    if not hdf5_path.exists():
        raise FileNotFoundError(f"Missing expected HDF5 output file: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as handle:
        if "energy_differences" not in handle:
            raise KeyError(f"Dataset 'energy_differences' missing in file: {hdf5_path}")

        if "n_atoms" not in handle.attrs:
            raise KeyError(f"Attribute 'n_atoms' missing in file: {hdf5_path}")

        n_atoms = int(handle.attrs["n_atoms"])
        assert n_atoms > 0, f"Number of atoms must be positive in file: {hdf5_path}"

        energy_differences = np.asarray(handle["energy_differences"][:], dtype=float)

    return energy_differences / float(n_atoms)


def _get_fit_stage_rmse(fit_dir: Path, iteration: IterationSpec) -> float:
    stage_path = fit_dir / iteration.stage_dir
    if not stage_path.exists():
        raise FileNotFoundError(f"Missing expected stage directory: {stage_path}")

    hdf5_paths = sorted(stage_path.glob("energies_and_forces_mol*.hdf5"))
    if not hdf5_paths:
        # Fail fast when expected output is missing.
        raise FileNotFoundError(
            "No validation error files found for fit iteration. "
            f"Expected one or more 'energies_and_forces_mol*.hdf5' in {stage_path}"
        )

    per_atom_errors = np.concatenate([_read_per_atom_energy_errors(path) for path in hdf5_paths])
    if per_atom_errors.size == 0:
        raise ValueError(f"No per-atom energy errors found in stage: {stage_path}")

    return float(np.sqrt(np.mean(per_atom_errors**2)))


def compute_per_fit_rmse_dataframe(presto_output_dir: Path) -> pd.DataFrame:
    """Compute fit-level per-atom energy RMSE for initial/iteration 1/iteration 2."""
    fit_dirs = _get_fit_dirs(presto_output_dir)

    rows: list[dict[str, str | int | float]] = []
    for fit_dir in fit_dirs:
        for iteration in ITERATION_SPECS:
            rmse = _get_fit_stage_rmse(fit_dir=fit_dir, iteration=iteration)
            rows.append(
                {
                    "fit_id": fit_dir.name,
                    "iteration": iteration.label,
                    "iteration_index": iteration.stage_index,
                    "per_atom_energy_rmse": rmse,
                }
            )

    return pd.DataFrame(rows)


def create_bootstrapped_summary_table(
    rmse_df: pd.DataFrame,
    n_bootstrap: int = 10_000,
    confidence_level: float = 95.0,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Create a summary table with bootstrap CIs from fit-level RMSE data."""
    required_columns = {
        "fit_id",
        "iteration",
        "iteration_index",
        "per_atom_energy_rmse",
    }
    missing_cols = required_columns.difference(rmse_df.columns)
    if missing_cols:
        raise ValueError(f"Input dataframe missing required columns: {sorted(missing_cols)}")

    rows: list[dict[str, str | int | float]] = []
    rms_column = _format_header_with_units(
        "RMS Per-Atom Energy RMSE",
        r"kcal mol$^{-1}$ atom$^{-1}$",
    )

    for iteration in ITERATION_SPECS:
        values = rmse_df.loc[
            rmse_df["iteration_index"] == iteration.stage_index,
            "per_atom_energy_rmse",
        ].to_numpy(dtype=float)

        if values.size == 0:
            raise ValueError(f"No RMSE values found for {iteration.label}")

        point_estimate = _get_rms(values)
        ci_low, ci_high = _bootstrap_ci(
            values,
            statistic=_get_rms,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_seed=random_seed,
        )

        rows.append(
            {
                "Iteration": iteration.label,
                "RMS Per-Atom Energy RMSE": point_estimate,
                "CI Lower": ci_low,
                "CI Upper": ci_high,
                "n_fits": int(values.size),
                rms_column: _format_value_with_ci(
                    point_estimate,
                    (ci_low, ci_high),
                    digits=4,
                ),
                "_iteration_index": iteration.stage_index,
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("_iteration_index")
    return summary_df.drop(columns=["_iteration_index"]).reset_index(drop=True)


def plot_fit_rmse_paired(
    rmse_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot fit-level RMSE in a paired box/scatter style."""
    import pingouin

    output_path.parent.mkdir(parents=True, exist_ok=True)

    iteration_order = [spec.label for spec in ITERATION_SPECS]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    pingouin.plot_paired(
        data=rmse_df,
        dv="per_atom_energy_rmse",
        within="iteration",
        subject="fit_id",
        order=iteration_order,
        ax=ax,
        pointplot_kwargs={"alpha": 0.2},
    )

    ax.set_ylabel(r"Per-Atom Energy RMSE / kcal mol$^{-1}$ atom$^{-1}$")
    ax.set_xlabel("presto fitting iteration")
    ax.set_title("Validation Per-Atom Energy RMSE Across presto fits")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_summary_table_latex(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Save the summary dataframe as a LaTeX table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rms_column = _format_header_with_units(
        "RMS Per-Atom Energy RMSE",
        r"kcal mol$^{-1}$ atom$^{-1}$",
    )
    columns = [
        "Iteration",
        rms_column,
        r"$n_{\mathrm{fits}}$",
    ]
    latex_df = summary_df.rename(columns={"n_fits": r"$n_{\mathrm{fits}}$"})
    latex_df[columns].to_latex(
        output_path,
        index=False,
        escape=False,
    )


def analyse_presto_fits(
    presto_output_dir: Path,
    output_dir: Path,
    n_bootstrap: int = 10_000,
    confidence_level: float = 95.0,
    random_seed: int = 0,
) -> None:
    """Run bulk presto fit RMSE analysis and write plot + tables."""
    output_dir.mkdir(parents=True, exist_ok=True)

    rmse_df = compute_per_fit_rmse_dataframe(presto_output_dir=presto_output_dir)
    summary_df = create_bootstrapped_summary_table(
        rmse_df=rmse_df,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_seed=random_seed,
    )

    rmse_df.to_csv(output_dir / "presto_fit_validation_energy_rmse_per_fit.csv", index=False)
    summary_df.to_csv(output_dir / "presto_fit_validation_energy_rmse_summary.csv", index=False)
    save_summary_table_latex(summary_df, output_dir / "presto_fit_validation_energy_rmse_summary.tex")
    plot_fit_rmse_paired(
        rmse_df=rmse_df,
        output_path=output_dir / "presto_fit_validation_energy_rmse.png",
    )
