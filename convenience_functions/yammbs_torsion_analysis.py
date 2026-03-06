"""Analyse torsion drive data using yammbs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from itertools import combinations
from matplotlib import pyplot

from convenience_functions._plotting_defaults import (
    FORCE_FIELD_DISPLAY_MAP,
    METRIC_CDF_XLIM,
    METRIC_LABELS,
    METRIC_UNITS,
)
from yammbs.torsion import TorsionStore
from yammbs.torsion.inputs import QCArchiveTorsionDataset
from yammbs.torsion.outputs import MetricCollection

pyplot.style.use("ggplot")

PAIRED_METRIC_CONFIGS = [
    {"key": "rmsd", "label": METRIC_LABELS["rmsd"], "units": METRIC_UNITS["rmsd"]},
    {"key": "rmse", "label": METRIC_LABELS["rmse"], "units": METRIC_UNITS["rmse"]},
    {
        "key": "js_distance",
        "label": METRIC_LABELS["js_distance"],
        "units": METRIC_UNITS["js_distance"],
    },
]


def analyse_torsion_scans(
    qcarchive_torsion_data: Path,
    database_file: Path,
    output_metrics: Path,
    output_minimized: Path,
    plot_dir: Path,
    base_force_fields: list[str],
    extra_force_fields: list[str],
    method: Literal[
        "openmm_torsion_atoms_frozen", "openmm_torsion_restrained"
    ] = "openmm_torsion_restrained",
    n_processes: int | None = None,
) -> None:
    """Run yammbs torsion analysis across selected force fields."""
    force_fields = base_force_fields + extra_force_fields

    with open(qcarchive_torsion_data) as file_handle:
        dataset = QCArchiveTorsionDataset.model_validate_json(file_handle.read())

    if database_file.exists():
        store = TorsionStore(database_file)
    else:
        database_file.parent.mkdir(parents=True, exist_ok=True)
        store = TorsionStore.from_torsion_dataset(
            dataset,
            database_name=database_file,
        )

    processes = n_processes if n_processes is not None else (os.cpu_count() or 1)
    for force_field in force_fields:
        store.optimize_mm(
            force_field=force_field,
            n_processes=processes,
            method=method,
        )

    output_minimized.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    with open(output_minimized, "w") as file_handle:
        file_handle.write(store.get_outputs().model_dump_json())

    metrics = store.get_metrics(force_fields=force_fields)
    with open(output_metrics, "w") as file_handle:
        file_handle.write(metrics.model_dump_json())

    plot_cdfs(output_metrics=output_metrics, plot_dir=plot_dir)
    plot_rms_stats(output_metrics=output_metrics, plot_dir=plot_dir)
    plot_mean_error_distribution(output_metrics=output_metrics, plot_dir=plot_dir)
    plot_rms_js_distance(output_metrics=output_metrics, plot_dir=plot_dir)
    plot_paired_stats(
        output_metrics=output_metrics, plot_dir=plot_dir, show_significance=True
    )
    plot_paired_stats(
        output_metrics=output_metrics, plot_dir=plot_dir, show_significance=False
    )


def _get_rms(array: np.ndarray) -> float:
    return float(np.sqrt(np.mean(array**2)))


def plot_cdfs(output_metrics: Path, plot_dir: Path) -> None:
    """Plot CDFs for rmsd, rmse, and Jensen-Shannon distance."""
    metrics = MetricCollection.parse_file(output_metrics)

    x_ranges = METRIC_CDF_XLIM
    units = METRIC_UNITS

    force_fields = list(metrics.metrics.keys())

    rmsds = {
        force_field: {
            key: val.rmsd for key, val in metrics.metrics[force_field].items()
        }
        for force_field in force_fields
    }
    rmses = {
        force_field: {
            key: val.rmse for key, val in metrics.metrics[force_field].items()
        }
        for force_field in force_fields
    }
    js_dists = {
        force_field: {
            key: val.js_distance[0] for key, val in metrics.metrics[force_field].items()
        }
        for force_field in force_fields
    }

    js_div_temp = list(list(metrics.metrics.values())[0].values())[0].js_distance[1]
    data = {
        "rmsd": rmsds,
        "rmse": rmses,
        "js_distance": js_dists,
    }

    for metric_name in ["rmsd", "rmse", "js_distance"]:
        figure, axis = pyplot.subplots()

        for force_field in force_fields:
            sorted_data = np.sort([*data[metric_name][force_field].values()])
            axis.plot(
                sorted_data,
                np.arange(1, len(sorted_data) + 1) / len(sorted_data),
                "-",
                label=force_field,
            )

        x_label = (
            metric_name.upper() + " / " + units[metric_name]
            if metric_name != "js_distance"
            else f"Jensen-Shannon Distance at {js_div_temp} K"
        )
        axis.set_xlabel(x_label)
        axis.set_ylabel("CDF")
        axis.set_xlim(x_ranges[metric_name])
        axis.set_ylim((-0.05, 1.05))
        axis.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        figure.savefig(plot_dir / f"{metric_name}.png", dpi=300, bbox_inches="tight")
        pyplot.close(figure)


def plot_rms_stats(output_metrics: Path, plot_dir: Path) -> None:
    """Plot RMS values for RMSD and RMSE."""
    metrics = MetricCollection.parse_file(output_metrics)
    force_fields = list(metrics.metrics.keys())

    units = METRIC_UNITS

    rms_rmses = {
        force_field: _get_rms(
            np.array([val.rmse for val in metrics.metrics[force_field].values()])
        )
        for force_field in force_fields
    }

    rms_rmsds = {
        force_field: _get_rms(
            np.array([val.rmsd for val in metrics.metrics[force_field].values()])
        )
        for force_field in force_fields
    }

    for metric_name, data in zip(["rmsd", "rmse"], [rms_rmsds, rms_rmses]):
        figure, axis = pyplot.subplots()
        axis.bar(data.keys(), data.values(), color=pyplot.cm.tab10.colors)
        axis.set_ylabel(metric_name.upper() + " / " + units[metric_name])
        pyplot.xticks(rotation=90)

        figure.tight_layout()
        figure.savefig(
            plot_dir / f"{metric_name}_rms.png",
            dpi=300,
            bbox_inches="tight",
        )
        pyplot.close(figure)


def plot_rms_js_distance(output_metrics: Path, plot_dir: Path) -> None:
    """Plot RMS Jensen-Shannon distance for each force field."""
    metrics = MetricCollection.parse_file(output_metrics)
    force_fields = list(metrics.metrics.keys())

    rms_js_distance = {
        force_field: _get_rms(
            np.array(
                [val.js_distance[0] for val in metrics.metrics[force_field].values()]
            )
        )
        for force_field in force_fields
    }

    js_div_temp = list(list(metrics.metrics.values())[0].values())[0].js_distance[1]

    figure, axis = pyplot.subplots()
    axis.bar(
        rms_js_distance.keys(), rms_js_distance.values(), color=pyplot.cm.tab10.colors
    )
    axis.set_ylabel(f"Mean Jensen-Shannon Distance at {js_div_temp} K")
    pyplot.xticks(rotation=90)

    figure.tight_layout()
    figure.savefig(plot_dir / "mean_js_distance.png", dpi=300, bbox_inches="tight")
    pyplot.close(figure)


def plot_paired_stats(
    output_metrics: Path,
    plot_dir: Path,
    ff_order: list[str] | None = None,
    ff_display_names: dict[str, str] | None = None,
    show_significance: bool = True,
) -> None:
    """Plot paired per-molecule statistics for RMSD, RMSE, and JSD on a single figure.

    Parameters
    ----------
    output_metrics:
        Path to the ``metrics.json`` produced by :func:`analyse_torsion_scans`.
    plot_dir:
        Directory to write the output plot.
    ff_order:
        Desired left-to-right ordering of force fields (raw labels).  Any FF
        not found in the data is ignored.  If *None* the FFs are sorted by
        ascending mean RMSE so the lowest-error one appears leftmost.
    ff_display_names:
        Mapping from raw FF label to human-readable display name.  Unmapped
        labels are shown as-is.
    show_significance:
        When *True* (default) annotate pairwise significance brackets using
        the sign test with Holm-Bonferroni correction.
    """
    import pingouin
    from statannotations.Annotator import Annotator
    from convenience_functions._stats import (
        holm_bonferroni,
        pvalue_to_stars,
        sign_test_pvalue,
    )

    metrics = MetricCollection.parse_file(output_metrics)
    all_ffs = list(metrics.metrics.keys())
    ff_display = (
        ff_display_names if ff_display_names is not None else FORCE_FIELD_DISPLAY_MAP
    )

    # Build per-metric arrays
    data_arrays: dict[str, dict[str, np.ndarray]] = {}
    for cfg in PAIRED_METRIC_CONFIGS:
        key = cfg["key"]
        if key != "js_distance":
            data_arrays[key] = {
                ff: np.array(
                    [getattr(val, key) for val in metrics.metrics[ff].values()]
                )
                for ff in all_ffs
            }
        else:
            data_arrays[key] = {
                ff: np.array(
                    [val.js_distance[0] for val in metrics.metrics[ff].values()]
                )
                for ff in all_ffs
            }

    # Determine display order
    if ff_order is not None:
        ordered_ffs = [ff for ff in ff_order if ff in all_ffs]
    else:
        ordered_ffs = sorted(all_ffs, key=lambda ff: np.mean(data_arrays["rmse"][ff]))

    display_names = [ff_display.get(ff, ff) for ff in ordered_ffs]

    pair_indices = list(combinations(range(len(ordered_ffs)), 2))
    pairs_display = [(display_names[i], display_names[j]) for i, j in pair_indices]

    fig, axes = pyplot.subplots(1, 3, figsize=(14, 4.5))

    for ax, cfg in zip(axes, PAIRED_METRIC_CONFIGS):
        key = cfg["key"]
        arr = data_arrays[key]

        entries = []
        for ff, disp in zip(ordered_ffs, display_names):
            for idx, val in enumerate(arr[ff]):
                entries.append({"subject": idx, key: val, "force_field": disp})
        df = pd.DataFrame(entries)

        pingouin.plot_paired(
            data=df,
            dv=key,
            within="force_field",
            subject="subject",
            order=display_names,
            ax=ax,
            pointplot_kwargs={"alpha": 0.2},
        )

        if show_significance:
            raw_pvals = [
                sign_test_pvalue(arr[ordered_ffs[i]], arr[ordered_ffs[j]])
                for i, j in pair_indices
            ]
            corrected = holm_bonferroni(raw_pvals)
            star_labels = [pvalue_to_stars(p) for p in corrected]

            annotator = Annotator(
                ax, pairs_display, data=df, x="force_field", y=key, order=display_names
            )
            annotator.configure(test=None, verbose=0)
            annotator.set_custom_annotations(star_labels)
            annotator.annotate()

        ax.set_ylabel(f"{cfg['label']} / {cfg['units']}")
        ax.set_xlabel("")
        pyplot.setp(ax.get_xticklabels(), rotation=20, ha="center")

    fig.tight_layout()
    suffix = "" if show_significance else "_no_sig"
    fig.savefig(plot_dir / f"paired_stats{suffix}.png", dpi=300, bbox_inches="tight")
    pyplot.close(fig)


def plot_mean_error_distribution(output_metrics: Path, plot_dir: Path) -> None:
    import seaborn as sns

    metrics = MetricCollection.parse_file(output_metrics)
    force_fields = list(metrics.metrics.keys())

    mean_errors = {
        force_field: np.array(
            [val.mean_error for val in metrics.metrics[force_field].values()]
        )
        for force_field in force_fields
    }

    figure, axis = pyplot.subplots(figsize=(10, 4))
    for force_field in force_fields:
        sns.kdeplot(
            data=mean_errors[force_field],
            label=force_field,
            ax=axis,
        )

    axis.set_xlabel(r"Mean Error / kcal mol$^{-1}$")
    axis.set_ylabel("Density")
    axis.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    figure.tight_layout()
    figure.savefig(
        plot_dir / "mean_error_distribution.png", dpi=300, bbox_inches="tight"
    )
    pyplot.close(figure)
