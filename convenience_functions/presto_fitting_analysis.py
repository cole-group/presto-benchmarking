"""Bulk analysis of presto fit outputs across many fit directories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from convenience_functions._plotting_defaults import (
    get_config_display_name,
    get_dataset_display_name,
)
from convenience_functions._stats import (
    bootstrap_ci as _bootstrap_ci,
    format_header_with_units as _format_header_with_units,
    format_value_with_ci as _format_value_with_ci,
    rms as _get_rms,
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

ITERATION_2_INDEX = 2
ITERATION_2_RMSE_FLAG_THRESHOLD = 1.0 # kcal mol^-1 atom^-1, large RMSE threshold for flagging fits for further investigation.

VALIDATION_FIT_AGGREGATE_OUTPUT_NAME = "presto_fit_validation_error_aggregate.csv"
VALIDATION_FIT_AGGREGATE_LATEX_OUTPUT_NAME = "presto_fit_validation_error_aggregate.tex"


def _get_fit_dirs(presto_output_dir: Path) -> list[Path]:
    """Return a list of fit directories under the presto output directory. This
    is a bit sloppy"""
    if not presto_output_dir.exists():
        raise FileNotFoundError(f"presto output directory not found: {presto_output_dir}")

    fit_dirs = sorted(path for path in presto_output_dir.iterdir() if path.is_dir() and "fail" not in path.name.lower())
    if not fit_dirs:
        raise FileNotFoundError(
            "No fit directories found under output directory. Expected numeric directories such as 0, 1, 2. "
            f"Path checked: {presto_output_dir}"
        )

    return fit_dirs


def _read_validation_errors(hdf5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not hdf5_path.exists():
        raise FileNotFoundError(f"Missing expected HDF5 output file: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as handle:
        if "energy_differences" not in handle:
            raise KeyError(f"Dataset 'energy_differences' missing in file: {hdf5_path}")

        if "forces_differences" not in handle:
            raise KeyError(f"Dataset 'forces_differences' missing in file: {hdf5_path}")

        if "n_atoms" not in handle.attrs:
            raise KeyError(f"Attribute 'n_atoms' missing in file: {hdf5_path}")

        n_atoms = int(handle.attrs["n_atoms"])
        assert n_atoms > 0, f"Number of atoms must be positive in file: {hdf5_path}"

        energy_differences = np.asarray(handle["energy_differences"][:], dtype=float)
        force_differences = np.asarray(handle["forces_differences"][:], dtype=float)

    return energy_differences / float(n_atoms), force_differences.reshape(-1)


def _get_fit_stage_rmse(fit_dir: Path, iteration: IterationSpec) -> tuple[float, float]:
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

    per_atom_error_arrays: list[np.ndarray] = []
    force_error_arrays: list[np.ndarray] = []
    for path in hdf5_paths:
        per_atom_errors, force_errors = _read_validation_errors(path)
        per_atom_error_arrays.append(per_atom_errors)
        force_error_arrays.append(force_errors)

    per_atom_errors = np.concatenate(per_atom_error_arrays)
    force_errors = np.concatenate(force_error_arrays)
    if per_atom_errors.size == 0:
        raise ValueError(f"No per-atom energy errors found in stage: {stage_path}")
    if force_errors.size == 0:
        raise ValueError(f"No force errors found in stage: {stage_path}")

    return float(np.sqrt(np.mean(per_atom_errors**2))), float(np.sqrt(np.mean(force_errors**2)))


def compute_per_fit_rmse_dataframe(presto_output_dir: Path) -> pd.DataFrame:
    """Compute fit-level energy and force RMSE for initial/iteration 1/iteration 2."""
    fit_dirs = _get_fit_dirs(presto_output_dir)

    rows: list[dict[str, str | int | float]] = []
    for fit_dir in fit_dirs:
        for iteration in ITERATION_SPECS:
            energy_rmse, force_rmse = _get_fit_stage_rmse(fit_dir=fit_dir, iteration=iteration)
            rows.append(
                {
                    "fit_id": fit_dir.name,
                    "iteration": iteration.label,
                    "iteration_index": iteration.stage_index,
                    "per_atom_energy_rmse": energy_rmse,
                    "force_rmse": force_rmse,
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
        "force_rmse",
    }
    missing_cols = required_columns.difference(rmse_df.columns)
    if missing_cols:
        raise ValueError(f"Input dataframe missing required columns: {sorted(missing_cols)}")

    rows: list[dict[str, str | int | float]] = []
    rms_column = _format_header_with_units(
        "RMS Per-Atom Energy RMSE",
        r"kcal mol$^{-1}$ atom$^{-1}$",
    )
    force_rms_column = _format_header_with_units(
        "RMS Force RMSE",
        r"kcal mol$^{-1}$ \AA$^{-1}$",
    )

    for iteration in ITERATION_SPECS:
        values = rmse_df.loc[
            rmse_df["iteration_index"] == iteration.stage_index,
            "per_atom_energy_rmse",
        ].to_numpy(dtype=float)
        force_values = rmse_df.loc[
            rmse_df["iteration_index"] == iteration.stage_index,
            "force_rmse",
        ].to_numpy(dtype=float)

        if values.size == 0:
            raise ValueError(f"No RMSE values found for {iteration.label}")
        if force_values.size == 0:
            raise ValueError(f"No force RMSE values found for {iteration.label}")

        point_estimate = _get_rms(values)
        ci_low, ci_high = _bootstrap_ci(
            values,
            statistic=_get_rms,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_seed=random_seed,
        )
        force_ci_low, force_ci_high = _bootstrap_ci(
            force_values,
            statistic=_get_rms,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_seed=random_seed,
        )
        force_point_estimate = _get_rms(force_values)

        rows.append(
            {
                "Iteration": iteration.label,
                "RMS Per-Atom Energy RMSE": point_estimate,
                "CI Lower": ci_low,
                "CI Upper": ci_high,
                "RMS Force RMSE": force_point_estimate,
                "Force CI Lower": force_ci_low,
                "Force CI Upper": force_ci_high,
                "n_fits": int(values.size),
                rms_column: _format_value_with_ci(
                    point_estimate,
                    (ci_low, ci_high),
                    digits=4,
                ),
                force_rms_column: _format_value_with_ci(
                    force_point_estimate,
                    (force_ci_low, force_ci_high),
                    digits=4,
                ),
                "_iteration_index": iteration.stage_index,
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("_iteration_index")
    return summary_df.drop(columns=["_iteration_index"]).reset_index(drop=True)


def create_iteration_2_flagged_fits_dataframe(
    rmse_df: pd.DataFrame,
    rmse_threshold: float = ITERATION_2_RMSE_FLAG_THRESHOLD,
) -> pd.DataFrame:
    """Return fits where iteration 2 per-atom energy RMSE exceeds the threshold."""
    required_columns = {
        "fit_id",
        "iteration_index",
        "per_atom_energy_rmse",
    }
    missing_cols = required_columns.difference(rmse_df.columns)
    if missing_cols:
        raise ValueError(f"Input dataframe missing required columns: {sorted(missing_cols)}")

    flagged_df = (
        rmse_df.loc[
            (rmse_df["iteration_index"] == ITERATION_2_INDEX)
            & (rmse_df["per_atom_energy_rmse"] > rmse_threshold),
            ["fit_id", "per_atom_energy_rmse"],
        ]
        .sort_values("per_atom_energy_rmse", ascending=False)
        .reset_index(drop=True)
    )
    return flagged_df


def _filter_flagged_fits_with_warning(
    rmse_df: pd.DataFrame,
    rmse_threshold: float = ITERATION_2_RMSE_FLAG_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exclude fits above threshold and emit a warning describing what was removed."""
    flagged_df = create_iteration_2_flagged_fits_dataframe(
        rmse_df=rmse_df,
        rmse_threshold=rmse_threshold,
    )
    if flagged_df.empty:
        return rmse_df, flagged_df

    excluded_fit_ids = flagged_df["fit_id"].astype(str).tolist()
    warnings.warn(
        "Excluding "
        f"{len(excluded_fit_ids)} fit(s) with iteration 2 per-atom energy RMSE > {rmse_threshold:.3f}: "
        f"{', '.join(excluded_fit_ids)}",
        UserWarning,
        stacklevel=2,
    )

    filtered_rmse_df = rmse_df.loc[~rmse_df["fit_id"].isin(excluded_fit_ids)].reset_index(drop=True)
    if filtered_rmse_df.empty:
        raise ValueError(
            "All fits were excluded by the iteration 2 RMSE threshold; no data remains for analysis. "
            f"Threshold: {rmse_threshold:.3f}"
        )

    return filtered_rmse_df, flagged_df


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
    force_rms_column = _format_header_with_units(
        "RMS Force RMSE",
        r"kcal mol$^{-1}$ \AA$^{-1}$",
    )
    columns = [
        "Iteration",
        rms_column,
        force_rms_column,
        r"$n_{\mathrm{fits}}$",
    ]
    latex_df = summary_df.rename(columns={"n_fits": r"$n_{\mathrm{fits}}$"})
    latex_df[columns].to_latex(
        output_path,
        index=False,
        escape=False,
    )


def _infer_dataset_config_from_summary_path(summary_csv_path: Path) -> tuple[str, str, str]:
    """Infer dataset, dataset type, and config from benchmark analysis summary path."""
    parts = summary_csv_path.parts
    if "benchmarking" not in parts:
        raise ValueError(f"Could not infer dataset/config from path: {summary_csv_path}")

    benchmark_index = parts.index("benchmarking")
    if len(parts) < benchmark_index + 6:
        raise ValueError(f"Could not infer dataset/config from path: {summary_csv_path}")

    dataset_name = parts[benchmark_index + 1]
    if parts[benchmark_index + 2] != "analysis":
        raise ValueError(f"Expected 'analysis' in summary path: {summary_csv_path}")

    dataset_type = parts[benchmark_index + 3]
    config_name = parts[benchmark_index + 4]
    return dataset_name, dataset_type, config_name


def aggregate_validation_fit_error_summaries(
    summary_csv_paths: list[Path],
    output_dir: Path,
) -> pd.DataFrame:
    """Aggregate validation fit RMS metrics across datasets/configs and write CSV/LaTeX."""
    if not summary_csv_paths:
        raise ValueError("No validation fit summary CSV paths were provided.")

    output_dir.mkdir(parents=True, exist_ok=True)

    required_columns = {
        "Iteration",
        "RMS Per-Atom Energy RMSE",
        "CI Lower",
        "CI Upper",
        "RMS Force RMSE",
        "Force CI Lower",
        "Force CI Upper",
        "n_fits",
    }
    iteration_order = {spec.label: spec.stage_index for spec in ITERATION_SPECS}

    rows: list[dict[str, str | int | float]] = []
    for summary_csv_path in summary_csv_paths:
        summary_df = pd.read_csv(summary_csv_path)

        missing_columns = required_columns.difference(summary_df.columns)
        if missing_columns:
            raise ValueError(
                f"Summary CSV missing required columns {sorted(missing_columns)}: {summary_csv_path}"
            )

        dataset_name, dataset_type, config_name = _infer_dataset_config_from_summary_path(summary_csv_path)
        if dataset_type != "validation":
            warnings.warn(
                f"Including non-validation summary in aggregate table: {summary_csv_path}",
                UserWarning,
                stacklevel=2,
            )

        dataset_display = get_dataset_display_name(dataset_name)
        config_display = get_config_display_name(config_name)

        for _, row in summary_df.iterrows():
            rows.append(
                {
                    "Dataset": dataset_display,
                    "dataset_name": dataset_name,
                    "Config": config_display,
                    "config_name": config_name,
                    "Iteration": str(row["Iteration"]),
                    "iteration_index": iteration_order[str(row["Iteration"])],
                    "RMS Per-Atom Energy RMSE": float(row["RMS Per-Atom Energy RMSE"]),
                    "CI Lower": float(row["CI Lower"]),
                    "CI Upper": float(row["CI Upper"]),
                    "RMS Force RMSE": float(row["RMS Force RMSE"]),
                    "Force CI Lower": float(row["Force CI Lower"]),
                    "Force CI Upper": float(row["Force CI Upper"]),
                    "n_fits": int(row["n_fits"]),
                }
            )

    aggregate_df = pd.DataFrame(rows).sort_values(
        by=["Dataset", "Config", "iteration_index"],
    )

    aggregate_df.to_csv(output_dir / VALIDATION_FIT_AGGREGATE_OUTPUT_NAME, index=False)

    force_header = _format_header_with_units(
        "RMS Force RMSE (95% CI)",
        r"kcal mol$^{-1}$ \AA$^{-1}$",
    )
    energy_header = _format_header_with_units(
        "RMS Per-Atom Energy RMSE (95% CI)",
        r"kcal mol$^{-1}$ atom$^{-1}$",
    )

    latex_df = aggregate_df.copy()
    latex_df[energy_header] = latex_df.apply(
        lambda row: _format_value_with_ci(
            value=float(row["RMS Per-Atom Energy RMSE"]),
            ci=(float(row["CI Lower"]), float(row["CI Upper"])),
            digits=3,
        ),
        axis=1,
    )
    latex_df[force_header] = latex_df.apply(
        lambda row: _format_value_with_ci(
            value=float(row["RMS Force RMSE"]),
            ci=(float(row["Force CI Lower"]), float(row["Force CI Upper"])),
            digits=1,
        ),
        axis=1,
    )

    latex_columns = [
        "Dataset",
        "Config",
        "Iteration",
        energy_header,
        force_header,
    ]
    latex_df[latex_columns].to_latex(
        output_dir / VALIDATION_FIT_AGGREGATE_LATEX_OUTPUT_NAME,
        index=False,
        escape=False,
    )

    return aggregate_df


def analyse_presto_fits(
    presto_output_dir: Path,
    output_dir: Path,
    n_bootstrap: int = 10_000,
    confidence_level: float = 95.0,
    random_seed: int = 0,
    iteration_2_rmse_threshold: float = ITERATION_2_RMSE_FLAG_THRESHOLD,
) -> None:
    """Run bulk presto fit RMSE analysis and write plot + tables."""
    output_dir.mkdir(parents=True, exist_ok=True)

    rmse_df = compute_per_fit_rmse_dataframe(presto_output_dir=presto_output_dir)
    filtered_rmse_df, flagged_iteration_2_fits_df = _filter_flagged_fits_with_warning(
        rmse_df=rmse_df,
        rmse_threshold=iteration_2_rmse_threshold,
    )
    summary_df = create_bootstrapped_summary_table(
        rmse_df=filtered_rmse_df,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_seed=random_seed,
    )

    filtered_rmse_df.to_csv(output_dir / "presto_fit_validation_energy_rmse_per_fit.csv", index=False)
    summary_df.to_csv(output_dir / "presto_fit_validation_energy_rmse_summary.csv", index=False)
    flagged_iteration_2_fits_df.to_csv(
        output_dir / "presto_fit_validation_energy_rmse_iteration_2_gt_threshold.csv",
        index=False,
    )
    save_summary_table_latex(summary_df, output_dir / "presto_fit_validation_energy_rmse_summary.tex")
    plot_fit_rmse_paired(
        rmse_df=filtered_rmse_df,
        output_path=output_dir / "presto_fit_validation_energy_rmse.png",
    )
