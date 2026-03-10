"""Plotting utilities for 2D protein torsion validation.

Slightly modified from Chapin Cavendar's script:
https://raw.githubusercontent.com/openforcefield/protein-param-fit/refs/heads/sage-2.1/validation/torsiondrive/3-plot-validation-torsiondrive-metrics.py
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot
from openmm import unit


def compute_profile_rmse(
    ref_energies: np.ndarray,
    energies: np.ndarray,
    normalize: bool = False,
    shift: bool = True,
) -> float:
    """Compute RMSE between two torsion profiles after optimal superimposition.

    Parameters
    ----------
    ref_energies : np.ndarray
        Reference energy profile
    energies : np.ndarray
        Energy profile to compare
    normalize : bool, optional
        Whether to normalize profiles before comparison (default: False)
    shift : bool, optional
        Whether to shift profiles optimally before comparison (default: True)

    Returns
    -------
    float
        RMSE between profiles
    """
    if shift:
        energies = energies + (ref_energies - energies).mean()

    if normalize:
        ref_energies = ref_energies / (ref_energies.max() - ref_energies.min())
        energies = energies / (energies.max() - energies.min())

    return float(np.sqrt(np.mean(np.square(ref_energies - energies))))


def plot_profile(
    plot_data: pd.DataFrame,
    output_path: Path | str,
    figure_size: Tuple[float, float],
    y_label: str,
    y_interval: int,
    y_range: Tuple[int, int],
    x_label: str,
    x_interval: int,
    legend: bool = False,
) -> None:
    """Plot a 1D torsion profile.

    Parameters
    ----------
    plot_data : pd.DataFrame
        Data to plot
    output_path : Path or str
        Output file path
    figure_size : tuple
        Figure size in inches
    y_label : str
        Y-axis label
    y_interval : int
        Y-axis tick interval
    y_range : tuple
        Y-axis range
    x_label : str
        X-axis label
    x_interval : int
        X-axis tick interval
    legend : bool, optional
        Whether to show legend (default: False)
    """
    figure = pyplot.figure(figsize=figure_size)

    if legend:
        pyplot.plot(plot_data, label=plot_data.columns)
    else:
        pyplot.plot(plot_data)

    pyplot.xlim(-180, 180)
    pyplot.xticks(np.arange(-180, 181, x_interval))
    pyplot.ylim(y_range)
    pyplot.yticks(np.arange(y_range[0], y_range[1] + 1, y_interval))
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)

    if legend:
        figure.legend(loc="outside upper center", ncol=3)

    pyplot.savefig(output_path)
    pyplot.close(figure)


def plot_heatmap(
    plot_data: pd.DataFrame,
    output_path: Path | str,
    figure_size: Tuple[float, float],
    colorbar_label: str,
    colorbar_interval: int,
    colorbar_range: Tuple[int, int],
    x_label: str,
    y_label: str,
    xy_interval: int,
) -> None:
    """Plot a 2D heatmap.

    Parameters
    ----------
    plot_data : pd.DataFrame
        Data to plot
    output_path : Path or str
        Output file path
    figure_size : tuple
        Figure size in inches
    colorbar_label : str
        Colorbar label
    colorbar_interval : int
        Colorbar tick interval
    colorbar_range : tuple
        Colorbar range
    x_label : str
        X-axis label
    y_label : str
        Y-axis label
    xy_interval : int
        XY-axis tick interval
    """
    figure = pyplot.figure(figsize=figure_size)
    pyplot.imshow(
        plot_data, origin="lower", vmin=colorbar_range[0], vmax=colorbar_range[1]
    )

    pyplot.xlim(-0.5, 23.5)
    pyplot.ylim(-0.5, 23.5)
    pyplot.xticks(
        ticks=np.arange(0, 25, xy_interval / 15),
        labels=np.arange(-180, 181, xy_interval),
    )
    pyplot.yticks(
        ticks=np.arange(0, 25, xy_interval / 15),
        labels=np.arange(-180, 181, xy_interval),
    )
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)

    # Include colorbar
    pyplot.colorbar(
        label=colorbar_label,
        ticks=np.arange(colorbar_range[0], colorbar_range[1] + 1, colorbar_interval),
    )

    pyplot.savefig(output_path)
    pyplot.close(figure)


def plot_energy(
    record_name: str,
    energy_type: str,
    energy_df: pd.DataFrame,
    output_dir: Path | str,
    figure_size: Tuple[float, float],
    ff_label: str,
    x_label: str,
    y_label: Optional[str] = None,
    energy_interval: int = 4,
    energy_range: Optional[Tuple[int, int]] = None,
    angle_interval: int = 60,
) -> None:
    """Plot energy profile or heatmap.

    Parameters
    ----------
    record_name : str
        Name of the record
    energy_type : str
        Type of energy to plot
    energy_df : pd.DataFrame
        Energy data
    output_dir : Path or str
        Output directory
    figure_size : tuple
        Figure size
    ff_label : str
        Force field label
    x_label : str
        X-axis label
    y_label : str, optional
        Y-axis label (for 2D plots)
    energy_interval : int, optional
        Energy tick interval (default: 4)
    energy_range : tuple, optional
        Energy range (will be auto-determined if None)
    angle_interval : int, optional
        Angle tick interval (default: 60)
    """
    output_path = Path(
        output_dir,
        f'{ff_label}-{record_name}-{energy_type.replace(" ", "-").lower()}.pdf',
    )

    if energy_type == "QM Energy":
        energy_label = f"{record_name}\n{energy_type} (kcal mol$^{{-1}}$)"
    else:
        energy_label = f"{ff_label} {record_name}\n{energy_type} (kcal mol$^{{-1}}$)"

    if energy_range is None:
        energy_range = (
            int(4 * np.floor(energy_df[energy_type].min() / 4)),
            int(4 * np.ceil(energy_df[energy_type].max() / 4)),
        )

        if (energy_range[1] - energy_range[0]) < 16:
            energy_range = (
                int(2 * np.floor(energy_df[energy_type].min() / 2)),
                int(2 * np.ceil(energy_df[energy_type].max() / 2)),
            )
            energy_interval = 2

    # Check number of dimensions and convert DataFrame for plotting
    if "Y" in energy_df.columns:
        plot_data = energy_df.pivot(index="Y", columns="X", values=energy_type)
        plot_heatmap(
            plot_data,
            output_path,
            figure_size,
            energy_label,
            energy_interval,
            energy_range,
            x_label,
            y_label or "Y",
            angle_interval,
        )
    else:
        plot_data = energy_df.set_index("X")[energy_type]
        plot_profile(
            plot_data,
            output_path,
            figure_size,
            energy_label,
            energy_interval,
            energy_range,
            x_label,
            angle_interval,
        )


def plot_difference(
    record_name_1: str,
    energy_type_1: str,
    record_name_2: str,
    energy_type_2: str,
    qc_data: Dict[str, Any],
    output_dir: Path | str,
    figure_size: Tuple[float, float],
    ff_label: str,
    x_label: str,
    y_label: Optional[str] = None,
    energy_interval: int = 4,
    energy_range: Optional[Tuple[int, int]] = None,
    angle_interval: int = 60,
) -> Tuple[float, float]:
    """Plot difference between two energy profiles and compute RMSE.

    Parameters
    ----------
    record_name_1 : str
        First record name
    energy_type_1 : str
        First energy type
    record_name_2 : str
        Second record name
    energy_type_2 : str
        Second energy type
    qc_data : dict
        QC data dictionary
    output_dir : Path or str
        Output directory
    figure_size : tuple
        Figure size
    ff_label : str
        Force field label
    x_label : str
        X-axis label
    y_label : str, optional
        Y-axis label
    energy_interval : int, optional
        Energy tick interval (default: 4)
    energy_range : tuple, optional
        Energy range (will be auto-determined if None)
    angle_interval : int, optional
        Angle tick interval (default: 60)

    Returns
    -------
    tuple
        (RMSE, normalized RMSE)
    """
    if energy_type_1 == energy_type_2:
        output_path = Path(
            output_dir,
            f"{ff_label}-{record_name_1}-{record_name_2}-"
            f'{energy_type_1.replace(" ", "-").lower()}.pdf',
        )

        if energy_type_1 == "QM Energy":
            energy_label = (
                f"{record_name_1} $-$ {record_name_2}\n{energy_type_1} "
                "(kcal mol$^{{-1}}$)"
            )
        else:
            energy_label = (
                f"{ff_label}\n{record_name_1} $-$ {record_name_2}\n"
                f"{energy_type_1} (kcal mol$^{{-1}}$)"
            )

    elif record_name_1 == record_name_2:
        output_path = Path(
            output_dir,
            f"{ff_label}-{record_name_1}-"
            f'{energy_type_1.replace(" ", "-").lower()}-'
            f'{energy_type_2.replace(" ", "-").lower()}.pdf',
        )

        energy_label = (
            f"{ff_label} {record_name_1}\n{energy_type_1} $-$ {energy_type_2}"
            "\n(kcal mol$^{{-1}}$)"
        )

    else:
        output_path = Path(
            output_dir,
            f"{ff_label}-{record_name_1}-"
            f'{energy_type_1.replace(" ", "-").lower()}-'
            f'{record_name_2}-{energy_type_2.replace(" ", "-").lower()}.pdf',
        )

        energy_label = (
            f"{ff_label}\n{record_name_1} {energy_type_1} $-$ {record_name_2} "
            f"{energy_type_2}\n(kcal mol$^{{-1}}$)"
        )

    energies_1 = qc_data[record_name_1]["energies"]
    energies_2 = qc_data[record_name_2]["energies"]

    if energy_range is None:
        energy_range = (
            int(
                4
                * np.floor(
                    min(
                        energies_1[energy_type_1].min(), energies_2[energy_type_2].min()
                    )
                    / 4
                )
            ),
            int(
                4
                * np.ceil(
                    max(
                        energies_1[energy_type_1].max(), energies_2[energy_type_2].max()
                    )
                    / 4
                )
            ),
        )

        if (energy_range[1] - energy_range[0]) < 16:
            energy_range = (
                int(
                    2
                    * np.floor(
                        min(
                            energies_1[energy_type_1].min(),
                            energies_2[energy_type_2].min(),
                        )
                        / 2
                    )
                ),
                int(
                    2
                    * np.ceil(
                        max(
                            energies_1[energy_type_1].max(),
                            energies_2[energy_type_2].max(),
                        )
                        / 2
                    )
                ),
            )
            energy_interval = 2

    # Check dimensions and compute difference for plotting
    if "Y" in energies_1.columns and "Y" in energies_2.columns:
        energy_series_1 = energies_1.set_index(["X", "Y"])[energy_type_1]
        energy_series_2 = energies_2.set_index(["X", "Y"])[energy_type_2]

        plot_data = energies_1.pivot(
            index="Y", columns="X", values=energy_type_1
        ) - energies_2.pivot(index="Y", columns="X", values=energy_type_2)

        plot_heatmap(
            plot_data,
            output_path,
            figure_size,
            energy_label,
            energy_interval,
            energy_range,
            x_label,
            y_label or "Y",
            angle_interval,
        )
    else:
        energy_series_1 = energies_1.set_index("X")[energy_type_1]
        energy_series_2 = energies_2.set_index("X")[energy_type_2]

        plot_data = energy_series_1 - energy_series_2
        plot_profile(
            plot_data,
            output_path,
            figure_size,
            energy_label,
            energy_interval,
            energy_range,
            x_label,
            angle_interval,
        )

    # Compute RMSE and normalized RMSE
    rmse = compute_profile_rmse(
        energy_series_1.values,
        energy_series_2.values,
        normalize=False,
        shift=True,
    )
    norm_rmse = compute_profile_rmse(
        energy_series_1.values,
        energy_series_2.values,
        normalize=True,
        shift=True,
    )

    return rmse, norm_rmse


def plot_rmse(
    rmse: Dict[str, Dict[str, float]],
    output_path: Path | str,
    figure_size: Tuple[float, float],
    y_label: str,
    rotate_x_labels: bool = False,
) -> None:
    """Plot RMSE values for each target by force field.

    Parameters
    ----------
    rmse : dict
        RMSE values by force field and target
    output_path : Path or str
        Output file path
    figure_size : tuple
        Figure size
    y_label : str
        Y-axis label
    rotate_x_labels : bool, optional
        Whether to rotate x-axis labels (default: False)
    """
    ff_labels = list(rmse.keys())
    x_labels = list(rmse[ff_labels[0]].keys())

    tick_locations = np.arange(len(x_labels))
    bar_width = 0.8 / len(ff_labels)
    bar_location_offset = (len(ff_labels) - 1) / 2
    figure = pyplot.figure(figsize=figure_size)

    for i, ff_label in enumerate(ff_labels):
        bar_locations = tick_locations + (i - bar_location_offset) * bar_width
        bar_heights = [rmse[ff_label][x_label] for x_label in x_labels]

        pyplot.bar(bar_locations, bar_heights, width=bar_width, label=ff_label)

    x_label_rotation = 90.0 if rotate_x_labels else 0.0

    pyplot.xticks(
        tick_locations,
        labels=x_labels,
        fontsize=10,
        rotation=x_label_rotation,
    )
    pyplot.ylim(bottom=0)
    pyplot.ylabel(y_label)
    figure.legend(loc="outside upper center", ncol=2)

    pyplot.savefig(output_path)
    pyplot.close(figure)


def plot_force_field_rmse(
    rmse: Dict[str, Dict[str, float]],
    output_path: Path | str,
    figure_size: Tuple[float, float],
    y_label: str,
    rotate_x_labels: bool = False,
    dark_background: bool = False,
    bootstrap_iterations: int = 2000,
    bootstrap_percentile: float = 0.95,
) -> None:
    """Plot average RMSE by force field with bootstrapped confidence intervals.

    Parameters
    ----------
    rmse : dict
        RMSE values by force field and target
    output_path : Path or str
        Output file path
    figure_size : tuple
        Figure size
    y_label : str
        Y-axis label
    rotate_x_labels : bool, optional
        Whether to rotate x-axis labels (default: False)
    dark_background : bool, optional
        Whether using dark background (default: False)
    bootstrap_iterations : int, optional
        Number of bootstrap iterations (default: 2000)
    bootstrap_percentile : float, optional
        Bootstrap percentile for confidence interval (default: 0.95)
    """
    ff_labels = list(rmse.keys())

    # Bootstrap RMSE values to compute error bars
    plot_data = [
        np.array([rmse[ff_label][rmse_index] for rmse_index in rmse[ff_label]])
        for ff_label in ff_labels
    ]

    bootstrap_samples = {
        ff_label: np.zeros(bootstrap_iterations) for ff_label in ff_labels
    }

    bootstrap_sample_count = len(plot_data[0])

    for bootstrap_index in range(bootstrap_iterations):
        sample_indices = np.random.randint(
            low=0,
            high=bootstrap_sample_count,
            size=bootstrap_sample_count,
        )

        for ff_label, ff_rmse in zip(ff_labels, plot_data):
            bootstrap_samples[ff_label][bootstrap_index] = ff_rmse[
                sample_indices
            ].mean()

    lower_percentile_index = int(
        np.round(bootstrap_iterations * (1 - bootstrap_percentile) / 2)
    )
    upper_percentile_index = int(
        np.round(bootstrap_iterations * (1 + bootstrap_percentile) / 2)
    )

    # Plot RMSEs with error bars
    bar_locations = np.arange(len(ff_labels))
    bar_width = 0.8
    bar_heights = np.zeros(len(ff_labels))
    bar_confidence_intervals = np.zeros((2, len(ff_labels)))

    for i, (ff_label, ff_rmse) in enumerate(zip(ff_labels, plot_data)):
        sorted_samples = np.sort(bootstrap_samples[ff_label])
        bar_heights[i] = ff_rmse.mean()
        bar_confidence_intervals[0][i] = float(
            np.abs(bar_heights[i] - sorted_samples[lower_percentile_index])
        )
        bar_confidence_intervals[1][i] = float(
            np.abs(bar_heights[i] - sorted_samples[upper_percentile_index])
        )

    # Print RMSEs and bootstrapped confidence intervals
    print(y_label.replace("$", "").replace("{", "").replace("}", ""))
    for i, ff_label in enumerate(ff_labels):
        print(
            f"    {ff_label:14s} {bar_heights[i]:6.4f} "
            f"({bar_heights[i] - bar_confidence_intervals[0][i]:6.4f} to "
            f"{bar_heights[i] + bar_confidence_intervals[1][i]:6.4f})"
        )

    figure = pyplot.figure(figsize=figure_size)
    pyplot.bar(
        bar_locations,
        bar_heights,
        color=seaborn.color_palette()[: len(ff_labels)],
        width=bar_width,
        yerr=bar_confidence_intervals,
        ecolor="white" if dark_background else "black",
    )

    x_label_rotation = 90.0 if rotate_x_labels else 0.0
    x_labels = [
        x_label.replace("-AshGC", "\nAshGC").replace("-alpha0", "\nalpha0")
        for x_label in ff_labels
    ]

    pyplot.xticks(bar_locations, labels=x_labels, rotation=x_label_rotation)
    pyplot.ylim(bottom=0)
    pyplot.ylabel(y_label)

    # Set tight layout to prevent clipping of error bars and labels
    figure.tight_layout()

    pyplot.savefig(output_path)
    pyplot.close(figure)


def plot_projection(
    record_name: str,
    energy_df: pd.DataFrame,
    output_dir: Path | str,
    figure_size: Tuple[float, float],
    projection_label: str,
    projection_index: Optional[str] = None,
    energy_interval: int = 4,
    energy_range: Optional[Tuple[int, int]] = None,
    angle_interval: int = 60,
    temperature: float = 310,
) -> None:
    """Plot projection of 2D profile onto 1D or plot 1D profile.

    Parameters
    ----------
    record_name : str
        Record name
    energy_df : pd.DataFrame
        Energy data
    output_dir : Path or str
        Output directory
    figure_size : tuple
        Figure size
    projection_label : str
        Label for the projection axis
    projection_index : str, optional
        Index to project onto ('X' or 'Y')
    energy_interval : int, optional
        Energy tick interval (default: 4)
    energy_range : tuple, optional
        Energy range (will be auto-determined if None)
    angle_interval : int, optional
        Angle tick interval (default: 60)
    temperature : float, optional
        Temperature in Kelvin for PMF calculation (default: 310)
    """
    R = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilocalorie_per_mole / unit.kelvin)
    beta = 1.0 / (R * temperature)

    output_path = Path(output_dir, f"{record_name}-{projection_label.lower()}.pdf")
    x_label = f'{record_name.replace("-rotamer-1", "")} {projection_label} (deg)'

    if "Y" in energy_df.columns:
        energy_label = "PMF (kcal mol$^{{-1}}$)"

        # Project profile onto one dihedral and compute potential of mean force
        dropped_index = "Y" if projection_index == "X" else "X"
        energy_columns = [s for s in energy_df.columns if s != dropped_index]
        probability_distribution = np.exp(-beta * energy_df[energy_columns])
        projection_partition_function = probability_distribution.groupby(
            projection_index, as_index=False
        ).sum()
        pmf = -np.log(projection_partition_function) / beta
        plot_data = pmf.set_index(projection_index)
        plot_data = plot_data - plot_data.min()

    else:
        energy_label = "Energy (kcal mol$^{{-1}}$)"

        # Profile has only one dimension, so plot as is
        plot_data = energy_df.set_index("X").sort_index()

    if energy_range is None:
        energy_range = (
            int(4 * np.floor(plot_data.min().min() / 4)),
            int(4 * np.ceil(plot_data.max().max() / 4)),
        )

        if (energy_range[1] - energy_range[0]) < 16:
            energy_range = (
                int(2 * np.floor(plot_data.min().min() / 2)),
                int(2 * np.ceil(plot_data.max().max() / 2)),
            )
            energy_interval = 2

    plot_profile(
        plot_data,
        output_path,
        figure_size,
        energy_label,
        energy_interval,
        energy_range,
        x_label,
        angle_interval,
        legend=True,
    )


def plot_protein_torsion(
    input_dir: str | Path,
    output_dir: str | Path,
    names_file: Optional[str | Path] = None,
    dark_background: bool = True,
    extension: str = "pdf",
    figure_width: float = 4.25,
    figure_height: Optional[float] = None,
    font_size: Optional[int] = None,
) -> None:
    """Generate validation plots for protein torsion benchmarking.

    Parameters
    ----------
    input_dir : str or Path
        Input directory containing minimisation results JSON files
    output_dir : str or Path
        Output directory for plots
    names_file : str or Path, optional
        Path to JSON file mapping record IDs to human-readable names.
        If None, looks for 'torsiondrive-validation-names.json' in
        input_dir.
    dark_background : bool, optional
        Use dark background style (default: True)
    extension : str, optional
        Output file extension (default: 'pdf')
    figure_width : float, optional
        Figure width in inches (default: 4.25)
    figure_height : float, optional
        Figure height in inches (None = 0.75 * width)
    font_size : int, optional
        Font size in points (None = matplotlib default)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dark_background:
        pyplot.style.use("dark_background")

    seaborn.set_palette(
        seaborn.color_palette(
            [seaborn.color_palette("tab10")[i] for i in [0, 9, 4, 6, 1, 8, 3, 2, 5, 7]]
        )
    )

    if figure_height is None:
        figure_size = (figure_width, figure_width * 0.75)
    else:
        figure_size = (figure_width, figure_height)

    if font_size is not None:
        pyplot.rcParams.update({"font.size": font_size})

    # Load dataset names if available
    if names_file is None:
        names_file = input_dir / "torsiondrive-validation-names.json"
    else:
        names_file = Path(names_file)

    if names_file.exists():
        with open(names_file) as f:
            dataset_names = json.load(f)
    else:
        dataset_names = {}

    # Find all minimisation result files (excluding the names file)
    result_files = [
        f
        for f in input_dir.glob("*.json")
        if f.name != "torsiondrive-validation-names.json"
    ]

    # Initialize nested data structures for all force fields
    qc_data: Dict[str, DefaultDict[str, Dict[str, Any]]] = {}
    rmse_values: Dict[str, Dict[str, float]] = {}
    norm_rmse_values: Dict[str, Dict[str, float]] = {}

    # Load data from all force fields
    for result_file in result_files:
        ff_label = result_file.stem

        qc_data[ff_label] = defaultdict(dict)
        rmse_values[ff_label] = {}
        norm_rmse_values[ff_label] = {}

        with open(result_file) as f:
            qc_data_by_id = json.load(f)

        for record_id, record_data in qc_data_by_id.items():
            record_name = dataset_names.get(record_id, record_id)
            record_smiles = record_data.get("smiles", "")
            record_energies = {
                tuple(json.loads(grid_id)): energy
                for grid_id, energy in record_data["energies"].items()
            }

            # Get number of dimensions
            num_dimensions = len(next(iter(record_energies.keys())))

            # Create DataFrame with wrapped angles
            if num_dimensions == 1:
                energy_df = pd.DataFrame(
                    [
                        {
                            "X": (grid_id[0] if grid_id[0] < 180 else grid_id[0] - 360),
                            "QM Energy": record_energies[grid_id][0],
                            "MM Energy": record_energies[grid_id][1],
                            "MM Target": record_energies[grid_id][2],
                            "MM RMSD": record_energies[grid_id][3],
                        }
                        for grid_id in record_energies
                    ]
                )
            else:
                energy_df = pd.DataFrame(
                    [
                        {
                            "X": (grid_id[0] if grid_id[0] < 180 else grid_id[0] - 360),
                            "Y": (grid_id[1] if grid_id[1] < 180 else grid_id[1] - 360),
                            "QM Energy": record_energies[grid_id][0],
                            "MM Energy": record_energies[grid_id][1],
                            "MM Target": record_energies[grid_id][2],
                            "MM RMSD": record_energies[grid_id][3],
                        }
                        for grid_id in record_energies
                    ]
                )

            # Store data for later use in projections
            qc_data[ff_label][record_name]["record_id"] = record_id
            qc_data[ff_label][record_name]["smiles"] = record_smiles
            qc_data[ff_label][record_name]["energies"] = energy_df

            # Set energy ranges based on residue type
            energy_range = {}
            if "pro" in record_name.lower():
                energy_range["QM Energy"] = (0, 32)
                energy_range["MM Energy"] = (-4, 32)
                energy_range["MM Target"] = (-12, 12)
            else:
                for energy_type in ["QM Energy", "MM Energy", "MM Target"]:
                    energy_range[energy_type] = (-4, 28)

            # Get axis labels
            if "rotamer" in record_name:
                x_label = "Phi (deg)"
                y_label = "Psi (deg)"
            else:
                x_label = "Chi1 (deg)"
                y_label = "Chi2 (deg)"

            # Plot torsion profiles for this record
            for energy_type in ["QM Energy", "MM Energy", "MM Target"]:
                plot_energy(
                    record_name,
                    energy_type,
                    energy_df,
                    output_dir,
                    figure_size,
                    ff_label,
                    x_label,
                    y_label,
                    energy_range=energy_range.get(energy_type),
                )

            # Plot difference heatmap between QM and MM energies
            rmse, norm_rmse = plot_difference(
                record_name,
                "QM Energy",
                record_name,
                "MM Energy",
                qc_data[ff_label],
                output_dir,
                figure_size,
                ff_label,
                x_label,
                y_label,
                energy_interval=3,
                energy_range=(-12, 12),
            )

            # Store RMSE values with cleaned record name for aggregation
            rmse_index = record_name.replace("-rotamer-1", "").lower()
            rmse_index = rmse_index.replace("ace-", "").replace("-nme", "")
            rmse_values[ff_label][rmse_index] = rmse
            norm_rmse_values[ff_label][rmse_index] = norm_rmse

    # Plot QM-MM RMSEs for each validation target by force field
    plot_rmse(
        rmse_values,
        Path(output_dir, f"torsiondrive-target-qm-mm-rmse.{extension}"),
        figure_size,
        "Capped 3-mer backbone\nRMSE (kcal mol$^{-1}$)",
        rotate_x_labels=True,
    )

    plot_rmse(
        norm_rmse_values,
        Path(output_dir, f"torsiondrive-target-qm-mm-norm-rmse.{extension}"),
        figure_size,
        "Capped 3-mer backbone\nNormalized RMSE",
        rotate_x_labels=True,
    )

    # Plot average RMSE by force field with bootstrapping for confidence intervals
    plot_force_field_rmse(
        rmse_values,
        Path(output_dir, f"force-field-qm-mm-rmse.{extension}"),
        figure_size,
        "Capped 3-mer backbone\nRMSE (kcal mol$^{-1}$)",
        dark_background=dark_background,
        rotate_x_labels=True,
    )

    plot_force_field_rmse(
        norm_rmse_values,
        Path(output_dir, f"force-field-qm-mm-norm-rmse.{extension}"),
        figure_size,
        "Capped 3-mer backbone\nNormalized RMSE",
        dark_background=dark_background,
        rotate_x_labels=True,
    )

    # Reorder seaborn colorblind palette for projections so that force fields
    # retain their color from previous plots
    seaborn.set_palette(
        seaborn.color_palette(
            [seaborn.color_palette("tab10")[i] for i in [7, 0, 9, 4, 6, 1, 8, 3, 2, 5]]
        )
    )

    # Plot slices of torsion profiles (projections)
    ff_labels = list(qc_data.keys())
    if ff_labels:
        for record_name in qc_data[ff_labels[0]].keys():
            # Merge DataFrames on X, Y, and QM Energy to ensure grid points are
            # the same between force fields
            energy_df = qc_data[ff_labels[0]][record_name]["energies"]

            if "Y" in energy_df.columns:
                select_columns = ["X", "Y", "QM Energy", "MM Energy"]
                merge_columns = ["X", "Y", "QM Energy"]
            else:
                select_columns = ["X", "QM Energy", "MM Energy"]
                merge_columns = ["X", "QM Energy"]

            energy_df = energy_df[select_columns].rename(
                columns={"MM Energy": ff_labels[0]}
            )

            for ff_label in ff_labels[1:]:
                energy_df_2 = qc_data[ff_label][record_name]["energies"]
                energy_df_2 = energy_df_2[select_columns].rename(
                    columns={"MM Energy": ff_label}
                )
                energy_df = pd.merge(energy_df, energy_df_2, on=merge_columns)

            # Generate projection plots based on record type
            if "rotamer" in record_name:
                plot_projection(
                    record_name=record_name,
                    energy_df=energy_df,
                    output_dir=output_dir,
                    figure_size=figure_size,
                    projection_label="Phi",
                    projection_index="X",
                )

                plot_projection(
                    record_name=record_name,
                    energy_df=energy_df,
                    output_dir=output_dir,
                    figure_size=figure_size,
                    projection_label="Psi",
                    projection_index="Y",
                )
            else:
                plot_projection(
                    record_name=record_name,
                    energy_df=energy_df,
                    output_dir=output_dir,
                    figure_size=figure_size,
                    projection_label="Chi1",
                    projection_index="X",
                )

                if "Y" in energy_df.columns:
                    plot_projection(
                        record_name=record_name,
                        energy_df=energy_df,
                        output_dir=output_dir,
                        figure_size=figure_size,
                        projection_label="Chi2",
                        projection_index="Y",
                    )
