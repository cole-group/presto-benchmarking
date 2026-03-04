"""Command-line interface for presto-benchmark utilities.

This module provides CLI commands for all convenience functions using typer.
"""

from pathlib import Path
from typing import Literal, Optional

import typer


app = typer.Typer()


@app.command("run-presto")
def run_presto_cli(
    config_path: Path = typer.Argument(..., help="Path to PRESTO config file"),
    smiles_path: Path = typer.Argument(..., help="Path to SMILES input file"),
    output_dir: Path = typer.Argument(..., help="Output directory for force field"),
) -> None:
    """Run PRESTO to generate bespoke force field."""
    from convenience_functions.run_presto import run_presto

    run_presto(
        config_path=config_path,
        smiles_path=smiles_path,
        output_dir=output_dir,
    )


@app.command("combine-force-fields")
def combine_force_fields_cli(
    output_file: Path = typer.Argument(..., help="Path to output combined force field"),
    ff_paths_str: str = typer.Argument(
        ...,
        help=(
            "Space-separated list of force field paths "
            "(label inferred from parent directory)"
        ),
    ),
) -> None:
    """Combine multiple force fields into a single file."""
    from convenience_functions.combine_ffs import combine_force_fields

    ff_to_combine_paths = {}
    for ff_path in ff_paths_str.split():
        ff_path_obj = Path(ff_path)
        label = ff_path_obj.parent.name
        ff_to_combine_paths[label] = ff_path_obj

    combine_force_fields(
        ff_to_combine_paths=ff_to_combine_paths,
        output_file=output_file,
    )


@app.command("get-qca-torsion-input")
def get_qca_torsion_input_cli(
    dataset_name: str = typer.Argument(..., help="Name of the QCA dataset to retrieve"),
    json_output_path: Path = typer.Argument(..., help="Path for output JSON file"),
) -> None:
    """Retrieve QCA torsion input data."""
    from convenience_functions.get_qca_input import get_qca_torsion_input

    get_qca_torsion_input(
        dataset_name=dataset_name,
        json_output_path=json_output_path,
    )


@app.command("get-tnet500-input")
def get_tnet500_input_cli(
    json_output_path: Path = typer.Argument(..., help="Path for output JSON file"),
) -> None:
    """Retrieve TNet500 SPICE LOT QCA input data."""
    from convenience_functions.get_qca_input import get_tnet_500_spice_lot_qca_input

    get_tnet_500_spice_lot_qca_input(json_output_path=json_output_path)


@app.command("split-qca-input")
def split_qca_input_cli(
    input_json_path: Path = typer.Argument(..., help="Path to input QCA JSON"),
    test_output_path: Path = typer.Argument(
        ..., help="Path for test set output directory"
    ),
    frac_test: float = typer.Option(0.8, help="Fraction of data to use for test set"),
    seed: int = typer.Option(0, help="Random seed for splitting"),
    validation_output_path: Optional[Path] = typer.Option(
        None, help="Path for validation set output directory"
    ),
) -> None:
    """Split QCA input data into validation and test sets."""
    from convenience_functions.split_qca_input import create_validation_and_test_sets

    create_validation_and_test_sets(
        input_json_path=input_json_path,
        frac_test=frac_test,
        seed=seed,
        validation_output_path=validation_output_path,
        test_output_path=test_output_path,
    )


@app.command("minimise-protein-torsion")
def minimise_protein_torsion_cli(
    input_file: Path = typer.Argument(..., help="Path to QCA data JSON file"),
    force_field_path: Path = typer.Argument(..., help="Path to force field file"),
    force_field_label: str = typer.Argument(..., help="Label for the force field"),
    force_field_type: str = typer.Argument(
        ..., help="Type of force field (e.g., smirnoff-nagl, amber)"
    ),
    output_path: Path = typer.Argument(..., help="Path for output results JSON file"),
) -> None:
    """Minimize protein torsion geometries with specified force field."""
    from convenience_functions.protein_2d_torsions import minimise_protein_torsion

    minimise_protein_torsion(
        input_file=input_file,
        force_field_path=force_field_path,
        force_field_label=force_field_label,
        force_field_type=force_field_type,
        output_path=output_path,
    )


@app.command("minimise-protein-torsion-multi")
def minimise_protein_torsion_multi_cli(
    input_file: Path = typer.Argument(..., help="Path to QCA data JSON file"),
    output_dir: Path = typer.Argument(
        ..., help="Output directory for results JSON files"
    ),
    config_json: Path = typer.Option(
        ...,
        "--config",
        help="Path to JSON file with force field configurations",
    ),
) -> None:
    """Minimize protein torsions with multiple force fields from config."""
    import json
    from tqdm import tqdm
    from convenience_functions.protein_2d_torsions import minimise_protein_torsion

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(config_json) as f:
        ff_configs = json.load(f)

    for ff_label, args in tqdm(
        ff_configs.items(), desc="Minimising with different force fields"
    ):
        minimise_protein_torsion(
            input_file=input_file,
            force_field_path=Path(args["ff_path"]),
            force_field_label=ff_label,
            force_field_type=args["ff_type"],
            output_path=output_dir / f"{ff_label}.json",
        )


@app.command("plot-protein-torsion")
def plot_protein_torsion_cli(
    input_dir: Path = typer.Argument(
        ..., help="Directory containing minimized results JSON files"
    ),
    output_dir: Path = typer.Argument(..., help="Output directory for plots"),
    names_file: Optional[Path] = typer.Option(
        None, help="Path to JSON file mapping IDs to human-readable names"
    ),
    dark_background: bool = typer.Option(True, help="Use dark background style"),
    extension: str = typer.Option("pdf", help="Output file extension"),
    figure_width: float = typer.Option(4.25, help="Figure width in inches"),
    figure_height: Optional[float] = typer.Option(
        None, help="Figure height in inches (default: 0.75 * width)"
    ),
    font_size: Optional[int] = typer.Option(None, help="Font size in points"),
) -> None:
    """Generate validation plots for protein torsion benchmarking."""
    from convenience_functions.protein_2d_torsions import plot_protein_torsion

    plot_protein_torsion(
        input_dir=input_dir,
        output_dir=output_dir,
        names_file=names_file,
        dark_background=dark_background,
        extension=extension,
        figure_width=figure_width,
        figure_height=figure_height,
        font_size=font_size,
    )


@app.command("get-qca-input-proteins")
def get_qca_input_for_proteins_cli(
    dataset_name: str = typer.Argument(..., help="Name of the QCA dataset to retrieve"),
    data_output_path: Path = typer.Argument(..., help="Path for output data JSON file"),
    names_output_path: Path = typer.Argument(
        ..., help="Path for output names JSON file"
    ),
) -> None:
    """Retrieve QCA input data and names for protein torsions."""
    from convenience_functions.protein_2d_torsions import get_qca_input

    get_qca_input(
        dataset_name=dataset_name,
        data_output_path=data_output_path,
        names_output_path=names_output_path,
    )


@app.command("analyse-torsion-scans")
def analyse_torsion_scans_cli(
    qcarchive_torsion_data: Path = typer.Argument(
        ..., help="Input QCArchive torsion dataset JSON"
    ),
    combined_force_field: Path = typer.Argument(
        ..., help="Path to the combined FF to include in analysis"
    ),
    output_dir: Path = typer.Argument(..., help="Output directory for analysis"),
    base_force_fields: list[str] = typer.Option(
        [], "--base-force-field", help="Base yammbs force field labels"
    ),
    extra_force_fields: list[str] = typer.Option(
        [], "--extra-force-field", help="Additional local force field paths"
    ),
    method: Literal[
        "openmm_torsion_atoms_frozen", "openmm_torsion_restrained"
    ] = typer.Option(
        "openmm_torsion_restrained",
        help="MM optimization method",
    ),
    n_processes: Optional[int] = typer.Option(
        None,
        help="Number of parallel processes",
    ),
) -> None:
    """Run yammbs torsion scan analysis and generate metrics/plots."""
    from convenience_functions.yammbs_torsion_analysis import analyse_torsion_scans

    output_dir.mkdir(parents=True, exist_ok=True)

    analyse_torsion_scans(
        qcarchive_torsion_data=qcarchive_torsion_data,
        database_file=output_dir / "torsion-data.sqlite",
        output_metrics=output_dir / "metrics.json",
        output_minimized=output_dir / "minimized.json",
        plot_dir=output_dir / "plots",
        base_force_fields=base_force_fields,
        extra_force_fields=[*extra_force_fields, str(combined_force_field)],
        method=method,
        n_processes=n_processes,
    )


@app.command("plot-ablation-comparison")
def plot_ablation_comparison_cli(
    metrics_json: Path = typer.Argument(..., help="Path to metrics.json from torsion scan analysis"),
    output_dir: Path = typer.Argument(..., help="Output directory for plots"),
) -> None:
    """Generate ablation comparison heatmap and distribution plots."""
    from convenience_functions.ablation_comparison import plot_ablation_comparison

    plot_ablation_comparison(metrics_json=metrics_json, output_dir=output_dir)


if __name__ == "__main__":
    app()
