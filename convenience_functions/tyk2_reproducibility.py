"""Utilities for analysing TYK2 reproducibility parameter variability."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from openff.toolkit import ForceField
import seaborn as sns
import torch

plt.style.use("ggplot")


_OFFXML_PARAM_KEYS: dict[str, list[str]] = {
    "Bonds": ["k", "length"],
    "Angles": ["k", "angle"],
    "ProperTorsions": ["k1", "k2", "k3", "k4"],
    "ImproperTorsions": ["k1"],
}

_TENSOR_UNIT_LABELS: dict[str, dict[str, str]] = {
    "Bonds": {"k": r"kcal mol$^{-1}$ A$^{-2}$", "length": r"A"},
    "Angles": {"k": r"kcal mol$^{-1}$ rad$^{-2}$", "angle": r"rad"},
    "ProperTorsions": {
        "k": r"kcal mol$^{-1}$",
        "periodicity": "",
        "phase": r"rad",
    },
    "ImproperTorsions": {
        "k": r"kcal mol$^{-1}$",
        "periodicity": "",
        "phase": r"rad",
    },
    "LinearBonds": {
        "k1": r"kcal mol$^{-1}$ A$^{-1}$",
        "k2": r"kcal mol$^{-1}$ A$^{-2}$",
        "b1": r"A",
        "b2": r"A",
    },
    "LinearAngles": {
        "k1": r"kcal mol$^{-1}$ rad$^{-2}$",
        "k2": r"kcal mol$^{-1}$ rad$^{-2}$",
        "angle1": r"rad",
        "angle2": r"rad",
    },
}

_EXCLUDED_TENSOR_TERMS: set[tuple[str, str]] = {
    ("Electrostatics", "*"),
    ("ProperTorsions", "phase"),
    ("ProperTorsions", "periodicity"),
}

_EPOCH_PATTERN = re.compile(r"ff_epoch_(\d+)\.pt")


@dataclass(frozen=True)
class _RunArtifacts:
    run_id: str
    run_dir: Path
    offxml_path: Path
    checkpoints_by_iteration: dict[int, dict[int, Path]]


def _natural_run_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (10**9, path.name)


def _parse_checkpoint_epoch(path: Path) -> int:
    match = _EPOCH_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Could not parse checkpoint epoch from file name: {path.name}")
    return int(match.group(1))


def _sample_checkpoints(checkpoint_dir: Path, sample_every_n_epochs: int) -> dict[int, Path]:
    all_paths = sorted(
        checkpoint_dir.glob("ff_epoch_*.pt"),
        key=_parse_checkpoint_epoch,
    )
    if not all_paths:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    first_path = all_paths[0]
    last_path = all_paths[-1]

    sampled = {
        _parse_checkpoint_epoch(path): path
        for path in all_paths
        if _parse_checkpoint_epoch(path) % sample_every_n_epochs == 0
    }
    sampled[_parse_checkpoint_epoch(first_path)] = first_path
    sampled[_parse_checkpoint_epoch(last_path)] = last_path
    return dict(sorted(sampled.items()))


def _discover_run_artifacts(
    output_root_dir: Path,
    sample_every_n_epochs: int,
    selected_runs: set[str] | None,
) -> list[_RunArtifacts]:
    run_dirs = [
        path
        for path in sorted(output_root_dir.glob("run_*"), key=_natural_run_key)
        if path.is_dir()
    ]
    if selected_runs is not None:
        run_dirs = [path for path in run_dirs if path.name in selected_runs]

    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {output_root_dir}")

    artifacts: list[_RunArtifacts] = []
    for run_dir in run_dirs:
        offxml_path = run_dir / "bespoke_force_field.offxml"
        if not offxml_path.exists():
            raise FileNotFoundError(f"Missing final OFFXML for run {run_dir.name}: {offxml_path}")

        checkpoints_by_iteration: dict[int, dict[int, Path]] = {}
        for iteration in (1, 2):
            checkpoint_dir = run_dir / f"training_iteration_{iteration}" / "ff_trajectory"
            if not checkpoint_dir.exists():
                raise FileNotFoundError(
                    f"Missing checkpoint directory for run {run_dir.name}, iteration {iteration}: {checkpoint_dir}"
                )
            checkpoints_by_iteration[iteration] = _sample_checkpoints(
                checkpoint_dir=checkpoint_dir,
                sample_every_n_epochs=sample_every_n_epochs,
            )

        artifacts.append(
            _RunArtifacts(
                run_id=run_dir.name,
                run_dir=run_dir,
                offxml_path=offxml_path,
                checkpoints_by_iteration=checkpoints_by_iteration,
            )
        )

    return artifacts


def _extract_scalar_quantity(value: Any) -> tuple[float, str]:
    if hasattr(value, "units"):
        units = value.units
        return float(value / units), str(units)
    return float(value), ""


def _collect_offxml_records(artifacts: list[_RunArtifacts]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for artifact in artifacts:
        ff = ForceField(artifact.offxml_path)
        for handler_name, parameter_keys in _OFFXML_PARAM_KEYS.items():
            handler = ff.get_parameter_handler(handler_name)
            for param_index, parameter in enumerate(handler.parameters):
                parameter_id = parameter.id if parameter.id is not None else f"idx_{param_index:04d}"
                # Restrict OFFXML analysis to fitted bespoke terms.
                if "bespoke" not in str(parameter_id).lower():
                    continue
                for param_name in parameter_keys:
                    if not hasattr(parameter, param_name):
                        continue

                    param_value = getattr(parameter, param_name)
                    value, unit = _extract_scalar_quantity(param_value)

                    rows.append(
                        {
                            "run_id": artifact.run_id,
                            "valence_type": handler_name,
                            "parameter_name": param_name,
                            "parameter_id": parameter_id,
                            "value": value,
                            "unit": unit,
                        }
                    )

    if not rows:
        raise ValueError(
            "No bespoke OFFXML parameters were extracted from the selected runs"
        )

    df = pd.DataFrame(rows)
    counts = (
        df.groupby(["valence_type", "parameter_name", "parameter_id"], as_index=False)[
            "run_id"
        ]
        .nunique()
        .rename(columns={"run_id": "n_runs"})
    )
    expected_runs = df["run_id"].nunique()
    missing = counts[counts["n_runs"] != expected_runs]
    if not missing.empty:
        bad = missing.iloc[0]
        raise ValueError(
            "Parameter identity mismatch across runs. Example: "
            f"{bad['valence_type']} {bad['parameter_name']} {bad['parameter_id']} present in {bad['n_runs']} "
            f"of {expected_runs} runs"
        )

    return df


def _compute_offxml_variability_summary(offxml_df: pd.DataFrame) -> pd.DataFrame:
    df = offxml_df.copy()
    tol = 1.0e-12
    per_parameter_df = (
        df.groupby(["valence_type", "parameter_name", "parameter_id"], as_index=False)
        .agg(
            unit=("unit", "first"),
            std_dev=("value", lambda s: float(np.std(s.to_numpy(dtype=float), ddof=0))),
            mean_abs_value_scale=(
                "value",
                lambda s: float(np.mean(np.abs(s.to_numpy(dtype=float)))),
            ),
        )
    )
    per_parameter_df["is_near_zero_scale"] = (
        per_parameter_df["mean_abs_value_scale"].abs() <= tol
    )
    per_parameter_df["normalised_std_dev"] = np.where(
        per_parameter_df["is_near_zero_scale"],
        np.nan,
        per_parameter_df["std_dev"] / per_parameter_df["mean_abs_value_scale"],
    )

    summary_rows: list[dict[str, Any]] = []
    grouped = per_parameter_df.groupby(["valence_type", "parameter_name"], sort=True)
    for (valence_type, parameter_name), sub_df in grouped:
        valid = sub_df["normalised_std_dev"].dropna().to_numpy(dtype=float)
        rms_normalised_std_dev = float(np.sqrt(np.mean(valid**2))) if valid.size else np.nan

        std_dev = sub_df["std_dev"].to_numpy(dtype=float)
        rms_std_dev = float(np.sqrt(np.mean(std_dev**2)))

        summary_rows.append(
            {
                "valence_type": valence_type,
                "parameter_name": parameter_name,
                "unit": sub_df["unit"].iloc[0],
                "n_parameter_ids": int(sub_df["parameter_id"].nunique()),
                "rms_std_dev": rms_std_dev,
                "rms_normalised_std_dev": rms_normalised_std_dev,
            }
        )

    return pd.DataFrame(summary_rows)


def _save_offxml_latex_table(summary_df: pd.DataFrame, output_path: Path) -> None:
    table_df = summary_df[
        [
            "valence_type",
            "parameter_name",
            "unit",
            "n_parameter_ids",
            "rms_normalised_std_dev",
            "rms_std_dev",
        ]
    ].rename(
        columns={
            "valence_type": "Valence",
            "parameter_name": "Parameter",
            "unit": "Unit",
            "n_parameter_ids": "N parameter IDs",
            "rms_normalised_std_dev": "RMS normalised std dev",
            "rms_std_dev": "RMS std dev",
        }
    )

    metric_columns = ["RMS normalised std dev", "RMS std dev"]
    for column in metric_columns:
        table_df[column] = table_df[column].map(
            lambda x: "" if pd.isna(x) else f"{x:.2f}"
        )

    latex_str = table_df.to_latex(index=False)
    output_path.write_text(latex_str)


def _plot_offxml_parameter_distributions(
    offxml_df: pd.DataFrame,
    output_path: Path,
    shift_mean_to_zero: bool,
) -> None:
    plot_df = offxml_df.copy()
    y_col = "value"
    y_label_prefix = "Value"

    if shift_mean_to_zero:
        mean_df = (
            plot_df.groupby(["valence_type", "parameter_name", "parameter_id"], as_index=False)[
                "value"
            ]
            .mean()
            .rename(columns={"value": "mean_value"})
        )
        plot_df = plot_df.merge(
            mean_df,
            on=["valence_type", "parameter_name", "parameter_id"],
            how="left",
        )
        plot_df["value_shifted"] = plot_df["value"] - plot_df["mean_value"]
        y_col = "value_shifted"
        y_label_prefix = "Shifted value"

    plot_df["x_label"] = plot_df["parameter_id"].astype(str)

    groups = list(plot_df.groupby(["valence_type", "parameter_name"], sort=True))
    n_panels = len(groups)
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(max(12.0, ncols * 8.0), max(4.0, nrows * 4.0)),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for ax, ((valence_type, parameter_name), sub_df) in zip(axes_flat, groups, strict=False):
        ordered_labels = [
            label
            for _, label in sorted(
                {(pid, str(pid)) for pid in sub_df["parameter_id"]},
                key=lambda item: item[0],
            )
        ]

        sns.boxplot(
            data=sub_df,
            x="x_label",
            y=y_col,
            order=ordered_labels,
            ax=ax,
            color="lightgray",
            fliersize=0,
        )
        sns.stripplot(
            data=sub_df,
            x="x_label",
            y=y_col,
            order=ordered_labels,
            hue="run_id",
            ax=ax,
            dodge=False,
            alpha=0.7,
            size=3.5,
        )

        unit = sub_df["unit"].iloc[0]
        tick_label_size = 5.5 if valence_type == "ProperTorsions" else 6.5
        ax.set_title(f"{valence_type} {parameter_name}")
        ax.set_xlabel("Parameter ID")
        ax.set_ylabel(f"{y_label_prefix}\n({unit})" if unit else y_label_prefix)
        ax.tick_params(axis="x", rotation=90, labelsize=tick_label_size)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            unique_pairs = []
            seen_labels: set[str] = set()
            for handle, label in zip(handles, labels, strict=False):
                if label in seen_labels:
                    continue
                seen_labels.add(label)
                unique_pairs.append((handle, label))
            ax.legend(
                [h for h, _ in unique_pairs],
                [label_text for _, label_text in unique_pairs],
                title="Run",
                fontsize="x-small",
                title_fontsize="x-small",
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
            )

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _extract_tensor_force_field(checkpoint_object: Any) -> Any:
    if hasattr(checkpoint_object, "potentials_by_type"):
        return checkpoint_object

    if isinstance(checkpoint_object, (tuple, list)):
        for value in checkpoint_object:
            if hasattr(value, "potentials_by_type"):
                return value

    if isinstance(checkpoint_object, dict):
        for value in checkpoint_object.values():
            if hasattr(value, "potentials_by_type"):
                return value

    raise TypeError(
        "Could not locate TensorForceField-like object in checkpoint payload. "
        f"Top-level type: {type(checkpoint_object)!r}"
    )


def _normalise_tensor_potential_type(potential_key: Any) -> str:
    text = str(potential_key)
    if "IMPROPER" in text.upper() and "TORSION" in text.upper():
        return "ImproperTorsions"
    if "BONDS" in text.upper() and "LINEAR" not in text.upper():
        return "Bonds"
    if "ANGLES" in text.upper() and "LINEAR" not in text.upper():
        return "Angles"
    if "PROPER" in text.upper() and "TORSION" in text.upper():
        return "ProperTorsions"
    if "LINEAR_BONDS" in text.upper():
        return "LinearBonds"
    if "LINEAR_ANGLES" in text.upper():
        return "LinearAngles"
    if "ELECTROSTATICS" in text.upper():
        return "Electrostatics"
    return text


def _should_exclude_tensor_parameter(valence_type: str, parameter_name: str) -> bool:
    return (valence_type, "*") in _EXCLUDED_TENSOR_TERMS or (
        valence_type,
        parameter_name,
    ) in _EXCLUDED_TENSOR_TERMS


def _collect_tensor_rows(artifacts: list[_RunArtifacts]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for artifact in artifacts:
        epoch_offset = 0

        for iteration in (1, 2):
            iteration_epochs = sorted(artifact.checkpoints_by_iteration[iteration])
            if not iteration_epochs:
                continue

            for local_epoch in iteration_epochs:
                checkpoint_path = artifact.checkpoints_by_iteration[iteration][local_epoch]
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=torch.device("cpu"),
                    weights_only=False,
                )
                ff = _extract_tensor_force_field(checkpoint)

                global_epoch = local_epoch + epoch_offset

                for potential_key, potential in ff.potentials_by_type.items():
                    valence_type = _normalise_tensor_potential_type(potential_key)
                    parameter_cols = list(potential.parameter_cols)
                    parameters = potential.parameters.detach().cpu().numpy()
                    parameter_keys = list(potential.parameter_keys)
                    periodicity_col_idx = (
                        parameter_cols.index("periodicity")
                        if "periodicity" in parameter_cols
                        else None
                    )

                    if parameters.ndim != 2:
                        raise ValueError(
                            "Expected 2D parameter tensor for potential "
                            f"{valence_type}, got shape {parameters.shape}"
                        )

                    for parameter_index in range(parameters.shape[0]):
                        parameter_id = str(parameter_keys[parameter_index].id)
                        for col_index, parameter_name in enumerate(parameter_cols):
                            parameter_name_for_plot = str(parameter_name)
                            if (
                                parameter_name_for_plot == "k"
                                and valence_type in {"ProperTorsions", "ImproperTorsions"}
                                and periodicity_col_idx is not None
                            ):
                                periodicity_value = parameters[parameter_index, periodicity_col_idx]
                                periodicity_int = int(round(float(periodicity_value)))
                                if np.isfinite(periodicity_value) and periodicity_int >= 1:
                                    parameter_name_for_plot = f"k{periodicity_int}"

                            if _should_exclude_tensor_parameter(
                                valence_type,
                                parameter_name_for_plot,
                            ):
                                continue

                            unit = _TENSOR_UNIT_LABELS.get(valence_type, {}).get(parameter_name, "")
                            rows.append(
                                {
                                    "run_id": artifact.run_id,
                                    "iteration": iteration,
                                    "local_epoch": int(local_epoch),
                                    "global_epoch": int(global_epoch),
                                    "valence_type": valence_type,
                                    "parameter_name": parameter_name_for_plot,
                                    "parameter_id": parameter_id,
                                    "value": float(parameters[parameter_index, col_index]),
                                    "unit": unit,
                                }
                            )

            epoch_offset += max(iteration_epochs) + 1

    if not rows:
        raise ValueError("No tensor parameter values were extracted from checkpoints")

    df = pd.DataFrame(rows)

    baseline = (
        df[(df["iteration"] == 1) & (df["local_epoch"] == 0)]
        [["run_id", "valence_type", "parameter_name", "parameter_id", "value"]]
        .rename(columns={"value": "baseline_value"})
    )

    df = df.merge(
        baseline,
        on=["run_id", "valence_type", "parameter_name", "parameter_id"],
        how="left",
    )

    missing_baseline = df["baseline_value"].isna().sum()
    if missing_baseline:
        df = df.dropna(subset=["baseline_value"]).copy()

    df["signed_change"] = df["value"] - df["baseline_value"]

    per_parameter_change = (
        df.groupby(["valence_type", "parameter_name", "parameter_id"], as_index=False)[
            "signed_change"
        ]
        .apply(lambda s: float(np.max(np.abs(s.to_numpy(dtype=float)))))
        .rename(columns={"signed_change": "max_abs_change"})
    )
    changed_parameter_ids = set(
        per_parameter_change[per_parameter_change["max_abs_change"] > 1.0e-12][
            ["valence_type", "parameter_name", "parameter_id"]
        ].itertuples(index=False, name=None)
    )

    if changed_parameter_ids:
        changed_df = pd.DataFrame(
            list(changed_parameter_ids),
            columns=["valence_type", "parameter_name", "parameter_id"],
        )
        df = df.merge(
            changed_df,
            on=["valence_type", "parameter_name", "parameter_id"],
            how="inner",
        )

    return df


def _plot_mean_change_with_std(
    tensor_df: pd.DataFrame,
    output_path: Path,
    change_column: str,
    line_label: str,
    y_label_prefix: str,
) -> None:
    stats = (
        tensor_df.groupby(["valence_type", "parameter_name", "global_epoch"], as_index=False)[
            change_column
        ]
        .agg(["mean", "std"]) 
        .reset_index()
        .rename(columns={"mean": "mean_change", "std": "std_change"})
    )
    stats["std_change"] = stats["std_change"].fillna(0.0)

    groups = list(stats.groupby(["valence_type", "parameter_name"], sort=True))
    n_panels = len(groups)
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(max(12.0, ncols * 8.0), max(4.0, nrows * 4.0)),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for ax, ((valence_type, parameter_name), sub_df) in zip(axes_flat, groups, strict=False):
        sub_df = sub_df.sort_values("global_epoch")
        epochs = sub_df["global_epoch"].to_numpy(dtype=float)
        mean_change = sub_df["mean_change"].to_numpy(dtype=float)
        std_change = sub_df["std_change"].to_numpy(dtype=float)

        ax.plot(epochs, mean_change, color="tab:blue", label=line_label)
        ax.fill_between(
            epochs,
            mean_change - std_change,
            mean_change + std_change,
            color="tab:blue",
            alpha=0.2,
            linewidth=0.0,
            label="±1 SD",
        )

        transition_epoch = 1000.0
        ax.axvline(
            transition_epoch,
            color="black",
            linestyle=":",
            linewidth=1.0,
            label="Iteration 2 starts",
        )
        y_min, y_max = ax.get_ylim()
        y_text = y_min + 0.95 * (y_max - y_min)
        ax.text(
            transition_epoch,
            y_text,
            "Iteration 2",
            rotation=90,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize="x-small",
        )

        unit_series = tensor_df[
            (tensor_df["valence_type"] == valence_type)
            & (tensor_df["parameter_name"] == parameter_name)
        ]["unit"]
        unit = unit_series.iloc[0] if not unit_series.empty else ""

        ax.set_title(f"{valence_type} {parameter_name}")
        ax.set_xlabel("Merged epoch")
        ax.set_ylabel(f"{y_label_prefix} ({unit})" if unit else y_label_prefix)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize="x-small",
        )

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_individual_parameter_trajectories(
    tensor_df: pd.DataFrame,
    output_path: Path,
    y_column: str,
    y_label_prefix: str,
) -> None:
    groups = list(tensor_df.groupby(["valence_type", "parameter_name"], sort=True))
    n_panels = len(groups)
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(max(12.0, ncols * 8.0), max(4.0, nrows * 4.0)),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for ax, ((valence_type, parameter_name), sub_df) in zip(axes_flat, groups, strict=False):
        labeled_runs: set[str] = set()
        for (run_id, _parameter_id), trajectory in sub_df.groupby(
            ["run_id", "parameter_id"], sort=False
        ):
            trajectory = trajectory.sort_values("global_epoch")
            label = run_id if run_id not in labeled_runs else None
            labeled_runs.add(run_id)
            ax.plot(
                trajectory["global_epoch"].to_numpy(dtype=float),
                trajectory[y_column].to_numpy(dtype=float),
                alpha=0.35,
                linewidth=1.0,
                label=label,
            )

        transition_epoch = 1000.0
        ax.axvline(
            transition_epoch,
            color="black",
            linestyle=":",
            linewidth=1.0,
            label="Iteration 2 starts",
        )
        y_min, y_max = ax.get_ylim()
        y_text = y_min + 0.95 * (y_max - y_min)
        ax.text(
            transition_epoch,
            y_text,
            "Iteration 2",
            rotation=90,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize="x-small",
        )

        unit = sub_df["unit"].iloc[0]
        ax.set_title(f"{valence_type} {parameter_name}")
        ax.set_xlabel("Merged epoch")
        ax.set_ylabel(f"{y_label_prefix} ({unit})" if unit else y_label_prefix)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize="x-small",
            ncol=2,
        )

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyse_tyk2_reproducibility_parameter_variability(
    output_root_dir: Path,
    analysis_output_dir: Path,
    sample_every_n_epochs: int = 50,
    run_ids: list[str] | None = None,
) -> None:
    """Analyse parameter variability across TYK2 reproducibility runs.

    Parameters
    ----------
    output_root_dir
        Root directory containing run_XX subdirectories.
    analysis_output_dir
        Directory where summary files and plots are written.
    sample_every_n_epochs
        Sampling cadence for tensor checkpoints; first and last are always included.
    run_ids
        Optional list of run directory names to include.
    """
    if sample_every_n_epochs < 1:
        raise ValueError("sample_every_n_epochs must be >= 1")

    selected_runs = set(run_ids) if run_ids else None
    artifacts = _discover_run_artifacts(
        output_root_dir=output_root_dir,
        sample_every_n_epochs=sample_every_n_epochs,
        selected_runs=selected_runs,
    )

    analysis_output_dir.mkdir(parents=True, exist_ok=True)

    offxml_df = _collect_offxml_records(artifacts)
    offxml_df.to_csv(analysis_output_dir / "offxml_parameter_values.csv", index=False)

    offxml_summary_df = _compute_offxml_variability_summary(offxml_df)
    offxml_summary_df.to_csv(
        analysis_output_dir / "offxml_variability_summary.csv", index=False
    )
    _save_offxml_latex_table(
        offxml_summary_df,
        analysis_output_dir / "offxml_variability_summary.tex",
    )
    _plot_offxml_parameter_distributions(
        offxml_df,
        analysis_output_dir / "offxml_parameter_values_boxplot.png",
        shift_mean_to_zero=False,
    )
    _plot_offxml_parameter_distributions(
        offxml_df,
        analysis_output_dir / "offxml_parameter_values_boxplot_shifted.png",
        shift_mean_to_zero=True,
    )

    tensor_df = _collect_tensor_rows(artifacts)
    tensor_df["absolute_change"] = tensor_df["signed_change"].abs()
    tensor_df.to_csv(analysis_output_dir / "tensor_parameter_trajectories.csv", index=False)

    _plot_mean_change_with_std(
        tensor_df,
        analysis_output_dir / "tensor_mean_signed_change_vs_epoch.png",
        change_column="signed_change",
        line_label="Mean signed change",
        y_label_prefix="Signed change",
    )
    _plot_mean_change_with_std(
        tensor_df,
        analysis_output_dir / "tensor_mean_absolute_change_vs_epoch.png",
        change_column="absolute_change",
        line_label="Mean absolute change",
        y_label_prefix="Absolute change",
    )
    _plot_individual_parameter_trajectories(
        tensor_df,
        analysis_output_dir / "tensor_individual_trajectories.png",
        y_column="signed_change",
        y_label_prefix="Signed change",
    )
    _plot_individual_parameter_trajectories(
        tensor_df,
        analysis_output_dir / "tensor_individual_trajectories_unshifted.png",
        y_column="value",
        y_label_prefix="Parameter value",
    )
