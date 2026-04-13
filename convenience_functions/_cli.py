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
    base_ff: str = typer.Option(
        "openff_unconstrained-2.3.0.offxml",
        help="Base force field XML file to combine with (default: openff_unconstrained-2.3.0.offxml)",
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
        base_ff=base_ff,
    )


@app.command("get-qca-torsion-input")
def get_qca_torsion_input_cli(
    dataset_name: str = typer.Argument(..., help="Name of the QCA dataset to retrieve"),
    json_output_path: Path = typer.Argument(..., help="Path for output JSON file"),
    exclude_smiles: list[str] = typer.Option(
        [], "--exclude-smiles", help="SMILES of a molecule to exclude (repeatable)"
    ),
    qcarchive_ids: list[int] = typer.Option(
        [],
        "--qcarchive-id",
        help="QCArchive record ID to retain (repeatable)",
    ),
) -> None:
    """Retrieve QCA torsion input data."""
    from convenience_functions.get_qca_input import get_qca_torsion_input

    get_qca_torsion_input(
        dataset_name=dataset_name,
        json_output_path=json_output_path,
        exclude_smiles=exclude_smiles or None,
        include_qcarchive_ids=qcarchive_ids or None,
    )


@app.command("get-tnet500-input")
def get_tnet500_input_cli(
    json_output_path: Path = typer.Argument(..., help="Path for output JSON file"),
) -> None:
    """Retrieve TNet500 SPICE LOT QCA input data."""
    from convenience_functions.get_qca_input import get_tnet_500_spice_lot_qca_input

    get_tnet_500_spice_lot_qca_input(json_output_path=json_output_path)


@app.command("get-folmsbee-input")
def get_folmsbee_input_cli(
    output_path: Path = typer.Argument(
        ..., help="Directory path where conformer-benchmark will be cloned"
    ),
) -> None:
    """Download Folmsbee/Hutchison conformer-benchmark input files."""
    from convenience_functions.get_folmsbee_input import (
        download_folmsbee_from_gh,
    )

    download_folmsbee_from_gh(path=output_path)


@app.command("process-folmsbee-smiles")
def process_folmsbee_smiles_cli(
    molecules_smi: Path = typer.Argument(
        ..., help="Path to Folmsbee molecules.smi file"
    ),
    output_dir: Path = typer.Argument(
        ..., help="Directory to write per-molecule .smi files"
    ),
) -> None:
    """Convert Folmsbee molecules.smi to run_presto-compatible .smi files."""
    from convenience_functions.get_folmsbee_input import (
        write_folmsbee_smiles_files,
    )

    write_folmsbee_smiles_files(
        molecules_smi=molecules_smi,
        output_dir=output_dir,
    )


@app.command("analyse-folmsbee")
def analyse_folmsbee_cli(
    folmsbee_repo_dir: Path = typer.Argument(
        ..., help="Path to Folmsbee conformer-benchmark repository clone"
    ),
    presto_output_dir: Path = typer.Argument(
        ..., help="Path to PRESTO molecule output directory for this config"
    ),
    output_dir: Path = typer.Argument(..., help="Output directory for analysis"),
    force_fields: list[str] = typer.Option(
        [], "--force-field", help="Force field path to evaluate (repeatable)"
    ),
    precomputed_methods: list[str] = typer.Option(
        [],
        "--precomputed-method",
        help="Precomputed method column from data-final.csv (repeatable)",
    ),
    mlp_names: list[str] = typer.Option(
        [],
        "--mlp-name",
        help="OpenMM-ML potential name to evaluate (repeatable)",
    ),
    single_point_mlp: bool = typer.Option(
        True,
        "--single-point-mlp/--minimise-mlp",
        help=(
            "Evaluate MLPs as single points on input geometries (default), "
            "or minimise with restraints when disabled"
        ),
    ),
    reference_method: str = typer.Option(
        "dlpno", help="Reference method column from data-final.csv"
    ),
    torsion_restraint_force_constant: float = typer.Option(
        10_000.0,
        help="Force constant for rotatable torsion restraints in kJ/mol/rad²",
    ),
    mm_minimization_steps: int = typer.Option(
        0,
        help="Max minimization iterations per conformer",
    ),
    n_processes: Optional[int] = typer.Option(
        None,
        help="Number of worker processes (reserved for future parallel execution)",
    ),
    exclude_smarts: list[str] = typer.Option(
        [],
        "--exclude-smarts",
        help="SMARTS pattern to exclude molecules before analysis (repeatable)",
    ),
    min_conformers_per_molecule: int = typer.Option(
        5,
        "--min-conformers-per-molecule",
        min=1,
        help="Exclude molecules with fewer than this many conformers",
    ),
    min_reference_energy_window: float = typer.Option(
        0.0,
        "--min-reference-energy-window",
        min=0.0,
        help=(
            "Exclude molecules whose reference-energy window (max-min in kcal/mol) "
            "is below this threshold"
        ),
    ),
) -> None:
    """Analyse Folmsbee conformer energies with restrained minimization."""
    from convenience_functions.analyse_folmsbee import analyse_folmsbee

    output_dir.mkdir(parents=True, exist_ok=True)

    analyse_folmsbee(
        folmsbee_repo_dir=folmsbee_repo_dir,
        presto_output_dir=presto_output_dir,
        output_dir=output_dir,
        force_field_paths=force_fields,
        precomputed_methods=precomputed_methods,
        mlp_names=mlp_names,
        single_point_mlp=single_point_mlp,
        reference_method=reference_method,
        torsion_restraint_force_constant=torsion_restraint_force_constant,
        mm_minimization_steps=mm_minimization_steps,
        n_processes=n_processes,
        exclude_smarts=exclude_smarts,
        min_conformers_per_molecule=min_conformers_per_molecule,
        min_reference_energy_window=min_reference_energy_window,
    )


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


@app.command("subset-qca-input-by-smiles")
def subset_qca_input_by_smiles_cli(
    input_json_path: Path = typer.Argument(..., help="Path to input QCA JSON"),
    selected_smiles_csv_path: Path = typer.Argument(
        ..., help="Path to existing split smiles.csv"
    ),
    output_json_path: Path = typer.Argument(..., help="Path for output subset JSON"),
) -> None:
    """Subset a QCA JSON dataset by molecule membership in an existing split CSV."""
    from convenience_functions.split_qca_input import save_sub_dataset_by_smiles

    save_sub_dataset_by_smiles(
        input_json_path=input_json_path,
        selected_smiles_csv_path=selected_smiles_csv_path,
        output_json_path=output_json_path,
    )


@app.command("filter-qca-torsions-by-bespoke-scans")
def filter_qca_torsions_by_bespoke_scans_cli(
    input_qca_json_path: Path = typer.Argument(..., help="Path to input QCA JSON"),
    force_field_path: Path = typer.Argument(
        ..., help="Path to force field used for torsion parameter assignment"
    ),
    output_qca_json_path: Path = typer.Argument(
        ..., help="Path for output filtered QCA JSON"
    ),
) -> None:
    """Keep only QCA torsions whose scanned central bond is fully bespoke-parameterized."""
    from convenience_functions.filter_qca_torsions import (
        filter_qca_torsions_to_bespoke_scans,
    )

    filter_qca_torsions_to_bespoke_scans(
        input_qca_json_path=input_qca_json_path,
        force_field_path=force_field_path,
        output_qca_json_path=output_qca_json_path,
    )


@app.command("analyse-smiles-descriptors")
def analyse_smiles_descriptors_cli(
    smiles_csv_path: Path = typer.Argument(..., help="Path to smiles.csv file"),
) -> None:
    """Analyse molecular descriptors for one smiles.csv file."""
    from convenience_functions.smiles_descriptor_analysis import analyse_smiles_file

    analyse_smiles_file(smiles_csv_path=smiles_csv_path)


@app.command("aggregate-smiles-descriptors")
def aggregate_smiles_descriptors_cli(
    output_dir: Path = typer.Argument(
        ..., help="Directory to write aggregate descriptor summary"
    ),
    smiles_csv_paths: list[Path] = typer.Argument(
        ..., help="Paths to smiles.csv files to analyse"
    ),
) -> None:
    """Analyse multiple smiles.csv files and write aggregate statistics."""
    from convenience_functions.smiles_descriptor_analysis import analyse_smiles_files

    analyse_smiles_files(smiles_csv_paths=smiles_csv_paths, output_dir=output_dir)


@app.command("minimise-protein-torsion")
def minimise_protein_torsion_cli(
    input_file: Path = typer.Argument(..., help="Path to QCA data JSON file"),
    force_field_path: Path = typer.Argument(..., help="Path to force field file"),
    force_field_label: str = typer.Argument(..., help="Label for the force field"),
    force_field_type: str = typer.Argument(
        ..., help="Type of force field (e.g., smirnoff-nagl, amber)"
    ),
    output_path: Path = typer.Argument(..., help="Path for output results JSON file"),
    dihedral_protocol: str = typer.Option(
        "restrain",
        help="Dihedral protocol: 'restrain' (strong torsion restraints, default) or 'freeze' (zero atom masses)",
    ),
) -> None:
    """Minimize protein torsion geometries with specified force field."""
    from convenience_functions.protein_2d_torsions import minimise_protein_torsion

    minimise_protein_torsion(
        input_file=input_file,
        force_field_path=force_field_path,
        force_field_label=force_field_label,
        force_field_type=force_field_type,
        output_path=output_path,
        dihedral_protocol=dihedral_protocol,
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
    dihedral_protocol: str = typer.Option(
        "restrain",
        help="Dihedral protocol: 'restrain' (strong torsion restraints, default) or 'freeze' (zero atom masses)",
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
            dihedral_protocol=dihedral_protocol,
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
    dark_background: bool = typer.Option(False, help="Use dark background style"),
    extension: str = typer.Option("pdf", help="Output file extension"),
    figure_width: float = typer.Option(6, help="Figure width in inches"),
    figure_height: Optional[float] = typer.Option(
        None, help="Figure height in inches (default: 0.6 * width)"
    ),
    font_size: Optional[int] = typer.Option(None, help="Font size in points"),
    temperature: float = typer.Option(
        310, help="Temperature in Kelvin for Boltzmann weighting"
    ),
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
        temperature=temperature,
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
    plot_torsion_ids: list[int] = typer.Option(
        [],
        "--plot-torsion-id",
        help="Torsion IDs to generate detailed scan plots for",
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
        torsion_plot_ids=plot_torsion_ids,
    )


@app.command("plot-ablation-comparison")
def plot_ablation_comparison_cli(
    metrics_json: Path = typer.Argument(
        ..., help="Path to metrics.json from torsion scan analysis"
    ),
    output_dir: Path = typer.Argument(..., help="Output directory for plots"),
) -> None:
    """Generate ablation comparison heatmap and distribution plots."""
    from convenience_functions.ablation_comparison import plot_ablation_comparison

    plot_ablation_comparison(metrics_json=metrics_json, output_dir=output_dir)


@app.command("analyse-presto-fits")
def analyse_presto_fits_cli(
    presto_output_dir: Path = typer.Argument(
        ..., help="Directory containing many PRESTO fit subdirectories"
    ),
    output_dir: Path = typer.Argument(..., help="Directory for analysis outputs"),
    n_bootstrap: int = typer.Option(
        10_000,
        help="Number of bootstrap resamples for confidence intervals",
    ),
    confidence_level: float = typer.Option(
        95.0,
        help="Confidence interval level in percent",
    ),
    random_seed: int = typer.Option(
        0,
        help="Random seed for bootstrap resampling (default: 0)",
    ),
) -> None:
    """Analyse PRESTO fits and summarize validation per-atom energy RMSE."""
    from convenience_functions.presto_fitting_analysis import analyse_presto_fits

    analyse_presto_fits(
        presto_output_dir=presto_output_dir,
        output_dir=output_dir,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_seed=random_seed,
    )


@app.command("prepare-tyk2-congeneric-retrain-configs")
def prepare_tyk2_congeneric_retrain_configs_cli(
    base_config_path: Path = typer.Argument(..., help="Path to initial TYK2 congeneric PRESTO config"),
    initial_run_dir: Path = typer.Argument(..., help="Directory of the completed initial TYK2 congeneric run"),
    output_dir: Path = typer.Argument(..., help="Directory for generated retraining config files"),
    max_extend_distances: list[int] = typer.Option(
        [0, 1, 2, 3],
        "--max-extend-distance",
        help="Max extend distance to apply to all valence handlers (repeatable)",
    ),
    include_sage_types: bool = typer.Option(
        True,
        "--include-sage-types/--exclude-sage-types",
        help=(
            "Also generate a retrain config with empty type_generation_settings "
            "to keep only Sage input types"
        ),
    ),
) -> None:
    """Generate PRESTO retraining configs that reuse precomputed data from the initial run."""
    from convenience_functions.tyk2_congeneric_series import (
        prepare_tyk2_congeneric_retrain_configs,
    )

    prepare_tyk2_congeneric_retrain_configs(
        base_config_path=base_config_path,
        initial_run_dir=initial_run_dir,
        output_dir=output_dir,
        max_extend_distances=max_extend_distances,
        include_sage_types=include_sage_types,
    )


@app.command("analyse-tyk2-congeneric-retrains")
def analyse_tyk2_congeneric_retrains_cli(
    initial_run_dir: Path = typer.Argument(..., help="Directory of the completed initial TYK2 congeneric run"),
    retrain_root_dir: Path = typer.Argument(..., help="Root directory containing retrain outputs by max_extend_distance"),
    output_dir: Path = typer.Argument(..., help="Directory for analysis outputs"),
    repeats: int = typer.Option(..., min=1, help="Number of repeated retrains per max extend distance"),
    max_extend_distances: list[int] = typer.Option(
        [0, 1, 2, 3],
        "--max-extend-distance",
        help="Max extend distance values that were retrained (repeatable)",
    ),
) -> None:
    """Analyse TYK2 congeneric retrain errors against the reused MLP test set."""
    from convenience_functions.tyk2_congeneric_series import (
        analyse_tyk2_congeneric_retrains,
    )

    analyse_tyk2_congeneric_retrains(
        initial_run_dir=initial_run_dir,
        retrain_root_dir=retrain_root_dir,
        output_dir=output_dir,
        max_extend_distances=max_extend_distances,
        repeats=repeats,
    )


@app.command("analyse-tyk2-reproducibility")
def analyse_tyk2_reproducibility_cli(
    output_root_dir: Path = typer.Argument(
        ..., help="Directory containing TYK2 reproducibility run_XX outputs"
    ),
    analysis_output_dir: Path = typer.Argument(
        ..., help="Directory for reproducibility analysis outputs"
    ),
    sample_every_n_epochs: int = typer.Option(
        50,
        min=1,
        help="Checkpoint sampling interval in epochs (first and last always included)",
    ),
    run_ids: list[str] = typer.Option(
        [],
        "--run-id",
        help="Specific run directory names to include, e.g. run_01 (repeatable)",
    ),
) -> None:
    """Analyse TYK2 reproducibility parameter variability across repeated runs."""
    from convenience_functions.tyk2_reproducibility import (
        analyse_tyk2_reproducibility_parameter_variability,
    )

    analyse_tyk2_reproducibility_parameter_variability(
        output_root_dir=output_root_dir,
        analysis_output_dir=analysis_output_dir,
        sample_every_n_epochs=sample_every_n_epochs,
        run_ids=run_ids or None,
    )


if __name__ == "__main__":
    app()
