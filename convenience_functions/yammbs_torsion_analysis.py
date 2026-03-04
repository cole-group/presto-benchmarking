"""Analyse torsion drive data using yammbs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
from matplotlib import pyplot

from yammbs.torsion import TorsionStore
from yammbs.torsion.inputs import QCArchiveTorsionDataset
from yammbs.torsion.outputs import MetricCollection

pyplot.style.use("ggplot")


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


def _get_rms(array: np.ndarray) -> float:
    return float(np.sqrt(np.mean(array**2)))


def plot_cdfs(output_metrics: Path, plot_dir: Path) -> None:
    """Plot CDFs for rmsd, rmse, and Jensen-Shannon distance."""
    metrics = MetricCollection.parse_file(output_metrics)

    x_ranges = {
        "rmsd": (0, 0.14),
        "rmse": (-0.3, 5.0),
        "js_distance": (None, None),
    }
    units = {
        "rmsd": r"$\mathrm{\AA}$",
        "rmse": r"kcal mol$^{-1}$",
        "js_distance": "",
    }

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

    units = {
        "rmsd": r"$\mathrm{\AA}$",
        "rmse": r"kcal mol$^{-1}$",
    }

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


def plot_mean_error_distribution(output_metrics: Path, plot_dir: Path) -> None:
    """Plot KDE distribution of mean errors for each force field."""
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
