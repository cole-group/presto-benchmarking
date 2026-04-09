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
from convenience_functions._stats import (
    bootstrap_ci as _bootstrap_ci,
    format_header_with_units as _format_header_with_units,
    format_value_with_ci as _format_value_with_ci,
    rms as _get_rms,
)
from yammbs.torsion.analysis import JSDistanceCollection, RMSECollection, RMSD, _normalize
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
    torsion_plot_ids: list[int] | None = None,
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
    plot_requested_torsion_scans(
        store=store,
        force_fields=force_fields,
        torsion_plot_ids=torsion_plot_ids or [],
        output_dir=plot_dir / "torsion_id_scans",
    )
    save_summary_table_latex(
        output_metrics=output_metrics,
        output_table=plot_dir / "summary_metrics.tex",
        database_file=database_file,
    )


def _get_force_field_display(force_field: str, ff_display_names: dict[str, str]) -> str:
    return ff_display_names.get(force_field, force_field)


def _find_force_field_key(
    available_force_fields: list[str],
    ff_display_names: dict[str, str],
    candidates: list[str],
) -> str | None:
    candidate_set = {candidate.lower() for candidate in candidates}

    for force_field in available_force_fields:
        if force_field.lower() in candidate_set:
            return force_field

    for force_field in available_force_fields:
        display_name = ff_display_names.get(force_field, force_field)
        if display_name.lower() in candidate_set:
            return force_field

    return None


def _get_mm_vs_mm_metric_arrays(
    database_file: Path,
    target_force_field: str,
    reference_force_field: str,
    js_temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-torsion RMSD, RMSE, and JS(target || reference) from MM outputs."""
    from openff.toolkit import Molecule

    store = TorsionStore(database_file)
    torsion_ids = store.get_torsion_ids()

    rmsd_values: list[float] = []
    rmse_values: list[float] = []
    js_values: list[float] = []

    for torsion_id in torsion_ids:
        reference_points = store.get_mm_points_by_torsion_id(
            torsion_id=torsion_id,
            force_field=reference_force_field,
        )
        target_points = store.get_mm_points_by_torsion_id(
            torsion_id=torsion_id,
            force_field=target_force_field,
        )

        reference_energies = store.get_mm_energies_by_torsion_id(
            torsion_id=torsion_id,
            force_field=reference_force_field,
        )
        target_energies = store.get_mm_energies_by_torsion_id(
            torsion_id=torsion_id,
            force_field=target_force_field,
        )

        if (
            len(reference_points) == 0
            or len(target_points) == 0
            or len(reference_energies) == 0
            or len(target_energies) == 0
        ):
            raise ValueError(f"Missing MM data for torsion ID {torsion_id}")

        molecule = Molecule.from_mapped_smiles(
            store.get_smiles_by_torsion_id(torsion_id),
            allow_undefined_stereo=True,
        )

        rmsd_values.append(
            float(
                RMSD.from_data(
                    torsion_id=torsion_id,
                    molecule=molecule,
                    qm_points=reference_points,
                    mm_points=target_points,
                ).rmsd
            )
        )

        reference_norm, target_norm = _normalize(reference_energies, target_energies)
        reference_array = np.fromiter(reference_norm.values(), dtype=float)
        target_array = np.fromiter(target_norm.values(), dtype=float)

        if len(reference_array) == 0 or len(target_array) == 0:
            raise ValueError(f"Missing normalized energy data for torsion ID {torsion_id}")

        rmse_values.append(
            float(
                RMSECollection.get_item_type().from_data(
                    torsion_id=torsion_id,
                    qm_energies=reference_array,
                    mm_energies=target_array,
                ).rmse
            )
        )
        js_values.append(
            float(
                JSDistanceCollection.get_item_type().from_data(
                    torsion_id=torsion_id,
                    qm_energies=reference_array,
                    mm_energies=target_array,
                    temperature=js_temperature,
                ).js_distance
            )
        )

    return np.array(rmsd_values), np.array(rmse_values), np.array(js_values)


def create_summary_table(
    output_metrics: Path,
    ff_display_names: dict[str, str] | None = None,
    database_file: Path | None = None,
    n_bootstrap: int = 10_000,
    confidence_level: float = 95.0,
    random_seed: int = 0,
    presto_reference_candidates: tuple[str, ...] = ("AceFF 2.0", "aceff20", "aceff"),
    presto_target_candidates: tuple[str, ...] = ("presto",),
) -> pd.DataFrame:
    """Create a summary dataframe with bootstrap CIs for selected metrics."""
    metrics = MetricCollection.parse_file(output_metrics)
    force_fields = list(metrics.metrics.keys())
    ff_display = (
        ff_display_names if ff_display_names is not None else FORCE_FIELD_DISPLAY_MAP
    )

    force_field_column = "Force Field / Reference"
    rms_rmsd_column = _format_header_with_units("RMS RMSD", METRIC_UNITS["rmsd"])
    rms_rmse_column = _format_header_with_units("RMS RMSE", METRIC_UNITS["rmse"])
    rms_js_column = _format_header_with_units(
        "RMS JS Distance",
        METRIC_UNITS["js_distance"],
    )

    rows: list[dict[str, str | float]] = []
    for force_field in force_fields:
        ff_metrics = metrics.metrics[force_field]

        rmsd_values = np.array([val.rmsd for val in ff_metrics.values()])
        rmse_values = np.array([val.rmse for val in ff_metrics.values()])
        js_values = np.array([val.js_distance[0] for val in ff_metrics.values()])

        rms_rmsd = _get_rms(rmsd_values)
        rms_rmse = _get_rms(rmse_values)
        rms_js_distance = _get_rms(js_values)

        rms_rmsd_ci = _bootstrap_ci(
            rmsd_values,
            statistic=_get_rms,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_seed=random_seed,
        )
        rms_rmse_ci = _bootstrap_ci(
            rmse_values,
            statistic=_get_rms,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_seed=random_seed,
        )
        rms_js_ci = _bootstrap_ci(
            js_values,
            statistic=_get_rms,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_seed=random_seed,
        )

        rows.append(
            {
                force_field_column: (
                    f"{_get_force_field_display(force_field, ff_display)} / QM"
                ),
                rms_rmsd_column: _format_value_with_ci(rms_rmsd, rms_rmsd_ci),
                rms_rmse_column: _format_value_with_ci(rms_rmse, rms_rmse_ci),
                rms_js_column: _format_value_with_ci(rms_js_distance, rms_js_ci),
                "_sort_rms_rmse": rms_rmse,
            }
        )

    base_rows = sorted(
        rows,
        key=lambda row: float(row["_sort_rms_rmse"]),
        reverse=True,
    )

    if database_file is not None:
        presto_force_field = _find_force_field_key(
            available_force_fields=force_fields,
            ff_display_names=ff_display,
            candidates=list(presto_target_candidates),
        )
        reference_force_field = _find_force_field_key(
            available_force_fields=force_fields,
            ff_display_names=ff_display,
            candidates=list(presto_reference_candidates),
        )

        if presto_force_field is not None and reference_force_field is not None:
            rmsd_values, rmse_values, js_values = _get_mm_vs_mm_metric_arrays(
                database_file=database_file,
                target_force_field=presto_force_field,
                reference_force_field=reference_force_field,
                js_temperature=500.0,
            )

            if len(rmse_values) > 0 and len(rmsd_values) > 0 and len(js_values) > 0:
                rms_rmsd = _get_rms(rmsd_values)
                rms_rmse = _get_rms(rmse_values)
                rms_js_distance = _get_rms(js_values)

                rms_rmsd_ci = _bootstrap_ci(
                    rmsd_values,
                    statistic=_get_rms,
                    n_bootstrap=n_bootstrap,
                    confidence_level=confidence_level,
                    random_seed=random_seed,
                )
                rms_rmse_ci = _bootstrap_ci(
                    rmse_values,
                    statistic=_get_rms,
                    n_bootstrap=n_bootstrap,
                    confidence_level=confidence_level,
                    random_seed=random_seed,
                )
                rms_js_ci = _bootstrap_ci(
                    js_values,
                    statistic=_get_rms,
                    n_bootstrap=n_bootstrap,
                    confidence_level=confidence_level,
                    random_seed=random_seed,
                )

                base_rows.append(
                    {
                        force_field_column: (
                            f"{_get_force_field_display(presto_force_field, ff_display)}"
                            f" / {_get_force_field_display(reference_force_field, ff_display)}"
                        ),
                        rms_rmsd_column: _format_value_with_ci(rms_rmsd, rms_rmsd_ci),
                        rms_rmse_column: _format_value_with_ci(rms_rmse, rms_rmse_ci),
                        rms_js_column: _format_value_with_ci(
                            rms_js_distance, rms_js_ci
                        ),
                    }
                )

    summary_df = pd.DataFrame(base_rows)
    if "_sort_rms_rmse" in summary_df.columns:
        summary_df = summary_df.drop(columns=["_sort_rms_rmse"])

    return summary_df


def save_summary_table_latex(
    output_metrics: Path,
    output_table: Path,
    ff_display_names: dict[str, str] | None = None,
    database_file: Path | None = None,
    n_bootstrap: int = 10_000,
    confidence_level: float = 95.0,
    random_seed: int = 0,
) -> None:
    """Create and save the summary dataframe as a LaTeX table."""
    summary_df = create_summary_table(
        output_metrics=output_metrics,
        ff_display_names=ff_display_names,
        database_file=database_file,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_seed=random_seed,
    )

    output_table.parent.mkdir(parents=True, exist_ok=True)
    latex_table = summary_df.to_latex(index=False, escape=False)
    with open(output_table, "w") as file_handle:
        file_handle.write(latex_table)


def _get_torsion_highlight_image(store: TorsionStore, torsion_id: int) -> np.ndarray:
    """Render an image of the molecule with the scanned torsion highlighted."""
    from openff.toolkit import Molecule
    from rdkit.Chem import AllChem
    from rdkit.Chem import Draw

    smiles = store.get_smiles_by_torsion_id(torsion_id)
    dihedral_indices = store.get_dihedral_indices_by_torsion_id(torsion_id)

    molecule = Molecule.from_mapped_smiles(
        smiles,
        allow_undefined_stereo=True,
    )
    rdkit_molecule = molecule.to_rdkit()
    AllChem.Compute2DCoords(rdkit_molecule)

    highlight_atoms = list(dihedral_indices)
    highlight_bonds = []
    for atom_a, atom_b in zip(highlight_atoms, highlight_atoms[1:]):
        bond = rdkit_molecule.GetBondBetweenAtoms(atom_a, atom_b)
        if bond is None:
            raise ValueError(
                f"Could not find bond between atoms {atom_a} and {atom_b} for torsion ID {torsion_id}."
            )
        highlight_bonds.append(bond.GetIdx())

    image = Draw.MolToImage(
        rdkit_molecule,
        size=(500, 350),
        highlightAtoms=highlight_atoms,
        highlightBonds=highlight_bonds,
    )
    return np.asarray(image)


def plot_requested_torsion_scans(
    store: TorsionStore,
    force_fields: list[str],
    torsion_plot_ids: list[int],
    output_dir: Path,
) -> None:
    """Plot scan profiles for requested torsion IDs with molecule highlights and RMSEs."""
    if len(torsion_plot_ids) == 0:
        return

    from cmcrameri import cm
    from matplotlib.colors import to_hex
    from openff.toolkit import Molecule

    requested_ids = list(dict.fromkeys(torsion_plot_ids))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Use six evenly spaced colors from a color-blind-friendly cmcrameri map.
    palette = [to_hex(cm.batlow(value)) for value in np.linspace(0.1, 0.9, 6)]
    markers = ["o", "^", "s", "D", "v", "P", "X", "<", ">", "*"]
    line_styles = ["-", "--", "-.", ":"]

    def _ff_key_from_raw_name(raw_name: str) -> str:
        if raw_name.endswith("/combined_force_field.offxml"):
            return Path(raw_name).parent.name
        return raw_name

    for torsion_id in requested_ids:
        qm_energies = dict(sorted(store.get_qm_energies_by_torsion_id(torsion_id).items()))
        qm_points = store.get_qm_points_by_torsion_id(torsion_id)

        qm_minimum_angle = min(qm_energies.items(), key=lambda item: item[1])[0]
        torsion_angles = list(qm_energies.keys())
        qm_relative_energies = [
            qm_energies[angle] - qm_energies[qm_minimum_angle]
            for angle in torsion_angles
        ]

        molecule = Molecule.from_mapped_smiles(
            store.get_smiles_by_torsion_id(torsion_id),
            allow_undefined_stereo=True,
        )

        figure = pyplot.figure(figsize=(4.56, 6.0))
        grid = figure.add_gridspec(
            3,
            1,
            height_ratios=[1.32, 1.0, 1.0],
            hspace=0.08,
        )

        molecule_axis = figure.add_subplot(grid[0, 0])
        energy_axis = figure.add_subplot(grid[1, 0])
        geometry_axis = figure.add_subplot(grid[2, 0], sharex=energy_axis)

        molecule_axis.imshow(_get_torsion_highlight_image(store=store, torsion_id=torsion_id))
        molecule_axis.axis("off")

        for index, force_field in enumerate(force_fields):
            mm_energies = dict(
                sorted(
                    store.get_mm_energies_by_torsion_id(
                        torsion_id=torsion_id,
                        force_field=force_field,
                    ).items()
                )
            )
            mm_points = store.get_mm_points_by_torsion_id(
                torsion_id=torsion_id,
                force_field=force_field,
            )

            color = palette[index % len(palette)]
            marker = markers[(index // 10) % len(markers)]
            line_style = line_styles[(index // len(palette)) % len(line_styles)]

            mm_relative_energies = [
                mm_energies[angle] - mm_energies[qm_minimum_angle]
                for angle in torsion_angles
            ]
            ff_name = _get_force_field_display(
                _ff_key_from_raw_name(force_field),
                FORCE_FIELD_DISPLAY_MAP,
            )
            ff_label = ff_name

            energy_axis.plot(
                torsion_angles,
                mm_relative_energies,
                color=color,
                marker=marker,
                linestyle=line_style,
                label=ff_label,
            )

            rmsd_values = []
            for angle in torsion_angles:
                rmsd = RMSD.from_data(
                    torsion_id=torsion_id,
                    molecule=molecule,
                    qm_points={angle: qm_points[angle]},
                    mm_points={angle: mm_points[angle]},
                )
                rmsd_values.append(float(rmsd.rmsd))

            geometry_axis.plot(
                torsion_angles,
                rmsd_values,
                color=color,
                marker=marker,
                linestyle=line_style,
                label=ff_label,
            )

        energy_axis.plot(
            torsion_angles,
            qm_relative_energies,
            color="black",
            linestyle="-",
            marker="o",
            markersize=2.5,
            markeredgewidth=0.6,
            label="QM",
            alpha=1.0,
            linewidth=1.6,
            zorder=10,
        )

        energy_axis.set_ylabel("Energy /\nkcal mol$^{-1}$")
        geometry_axis.set_ylabel("RMSD /\n$\mathrm{\AA}$")
        geometry_axis.set_xlabel("Torsion angle / degrees")
        energy_axis.tick_params(axis="x", which="both", labelbottom=False)
        energy_axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)

        figure.savefig(output_dir / f"{torsion_id}.png", dpi=300, bbox_inches="tight")
        pyplot.close(figure)


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
