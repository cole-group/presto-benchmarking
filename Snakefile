from pathlib import Path
from typing import Any
import json
import tempfile

configfile: "workflow_config.yaml"

RANDOM_SEED = config["random_seed"]
TNET_500_FRAC_TEST = config["tnet500_frac_test"]
TNET500_REOPT_V4_QCA_DATASET = "TorsionNet500 Re-optimization TorsionDrives v4.0"

QCA_DATASET_NAMES = {
    "jacs_fragments": "OpenFF-benchmark-ligand-fragments-v2.0",
    "jacs_fragments_full_mol_fits": "OpenFF-benchmark-ligand-fragments-v2.0",
    "1mer_backbone": "OpenFF Protein Dipeptide 2-D TorsionDrive v2.0",
    "3mer_backbone": "OpenFF Protein Capped 3-mer Backbones v1.0",
    "1mer_side_chain": "OpenFF Protein Capped 1-mer Sidechains v1.3",
}

PROTEIN_DATASETS = ["1mer_backbone", "3mer_backbone", "1mer_side_chain"]

############ Convenience Functions #############

def smiles_dir_outputs(
    wildcards: Any,
    checkpoint_obj: Any,
    smiles_dir: str,
    output_pattern: str,
    checkpoint_kwargs: dict | None = None,
) -> list[str]:
    """Expand output_pattern over all .smi files in smiles_dir once checkpoint_obj is done."""
    checkpoint_obj.get(**(checkpoint_kwargs or {}))
    molecules = glob_wildcards(f"{smiles_dir}/{{molecule}}.smi").molecule
    return expand(output_pattern, molecule=molecules)


def validation_force_fields(wildcards: Any) -> list[str]:
    """Generic input function for create_combined_force_field.

    Infers the smiles directory from the dataset wildcard and resolves
    the per-molecule force field paths after the relevant checkpoint completes.
    Tries a dataset-specific checkpoint (split_{dataset}_input) first; falls back
    to split_test_only_input for datasets without a dedicated checkpoint.
    """
    dataset = wildcards.dataset
    checkpoint_kwargs: dict = {}
    if dataset == "folmsbee_conformers":
        checkpoint_obj = checkpoints.process_folmsbee_smiles
    else:
        checkpoint_obj = getattr(checkpoints, f"split_{dataset}_input", None)
        if checkpoint_obj is None:
            # Fall back to the generic protein backbone checkpoint (wildcard on dataset)
            checkpoint_obj = checkpoints.split_test_only_input
            checkpoint_kwargs = {"dataset": dataset}

    return smiles_dir_outputs(
        wildcards,
        checkpoint_obj=checkpoint_obj,
        smiles_dir=f"benchmarking/{dataset}/input/{wildcards.dataset_type}/smiles",
        output_pattern=f"benchmarking/{dataset}/output/{wildcards.dataset_type}/{wildcards.config_name}/{{molecule}}/bespoke_force_field.offxml",
        checkpoint_kwargs=checkpoint_kwargs,
    )


def smiles_csv_input(wildcards: Any) -> str:
    """Resolve smiles.csv path for a dataset/split after the relevant checkpoint."""
    dataset = wildcards.dataset

    if dataset == "tnet500":
        checkpoints.split_tnet500_input.get()
        return f"benchmarking/{dataset}/input/{wildcards.dataset_type}/smiles.csv"

    checkpoint_obj = getattr(checkpoints, f"split_{dataset}_input", None)
    checkpoint_kwargs: dict = {}
    if checkpoint_obj is None:
        checkpoint_obj = checkpoints.split_test_only_input
        checkpoint_kwargs = {"dataset": dataset}

    checkpoint_obj.get(**checkpoint_kwargs)
    return f"benchmarking/{dataset}/input/{wildcards.dataset_type}/smiles.csv"


def smiles_descriptor_summary_input(wildcards: Any) -> list[str]:
    """Return smiles descriptor summary dependency for datasets that produce smiles.csv."""
    if wildcards.dataset == "folmsbee_conformers":
        return []

    return [
        f"benchmarking/{wildcards.dataset}/input/{wildcards.dataset_type}/smiles_descriptor_summary.csv"
    ]


def qca_exclude_smiles_opts(dataset_name: str) -> str:
    """Return --exclude-smiles CLI flags for a QCA dataset, from workflow_config.yaml."""
    smiles = config.get("exclude_smiles", {}).get(dataset_name, [])
    return " ".join(f"--exclude-smiles '{s}'" for s in smiles)


def full_molecule_fit_force_field_map(wildcards: Any | None = None) -> dict[str, str]:
    """Return {label: offxml_path} for supplied full-molecule bespoke fits."""
    paths = sorted(Path("input_ff/from_sandbox/ff_files").glob("*/*.offxml"))
    if not paths:
        raise ValueError(
            "No full-molecule fit OFFXML files found under "
            "input_ff/from_sandbox/ff_files/*/*.offxml"
        )
    return {path.parent.name: str(path) for path in paths}


def full_molecule_fit_force_field_labels(wildcards: Any | None = None) -> list[str]:
    """Return sorted labels for supplied full-molecule bespoke fits."""
    return sorted(full_molecule_fit_force_field_map().keys())


def full_molecule_fit_force_field_path(wildcards: Any) -> str:
    """Resolve a full-molecule bespoke fit OFFXML by wildcard label."""
    ff_map = full_molecule_fit_force_field_map()
    if wildcards.ff_label not in ff_map:
        raise ValueError(
            f"Unknown ff_label '{wildcards.ff_label}'. "
            f"Expected one of {sorted(ff_map.keys())}."
        )
    return ff_map[wildcards.ff_label]



def yammbs_target_config(wildcards: Any) -> dict[str, Any]:
    """Return yammbs config for a dataset/split target."""
    return config["yammbs_analysis"]["targets"][wildcards.dataset][wildcards.dataset_type]


def folmsbee_target_config(wildcards: Any) -> dict[str, Any]:
    """Return Folmsbee analysis config for a dataset split."""
    return config["folmsbee_analysis"]["targets"][wildcards.dataset_type]


def protein_torsion_combined_ff(wildcards: Any) -> str:
    """Return path to the combined force field for protein torsion minimisation.

    1mer_side_chain reuses the combined force field produced by 1mer_backbone fits.
    """
    source_dataset = (
        "1mer_backbone" if wildcards.dataset == "1mer_side_chain" else wildcards.dataset
    )
    return (
        f"benchmarking/{source_dataset}/output/"
        f"{wildcards.dataset_type}/{wildcards.config_name}/combined_force_field.offxml"
    )


def tyk2_congeneric_retrain_labels() -> list[str]:
    """Return retrain labels including Sage-only typing baseline."""
    labels = [
        str(distance)
        for distance in config["tyk2_congeneric_series"].get(
            "retrain_max_extend_distances", [0, 1, 2, 3]
        )
    ]
    return ["sage_types", *labels]

############ Workflow Rules #############


rule all:
    input:
        # Folmsbee/Hutchison conformer benchmark input
        "benchmarking/folmsbee_conformers/input/gh_repo",
        # TNet 500 validation force fields for different ablations
        "benchmarking/tnet500/output/validation/default/combined_force_field.offxml",
        "benchmarking/tnet500/output/validation/no_reg/combined_force_field.offxml",
        "benchmarking/tnet500/output/validation/no_min/combined_force_field.offxml",
        "benchmarking/tnet500/output/validation/one_it/combined_force_field.offxml",
        "benchmarking/tnet500/output/validation/no_metad/combined_force_field.offxml",
        # Full TNet 500 workflow
        "benchmarking/tnet500/output/test/default/combined_force_field.offxml",
        # JACS Fragments
        "benchmarking/jacs_fragments/output/test/default/combined_force_field.offxml",
        # Folmsbee conformer analysis
        "benchmarking/folmsbee_conformers/analysis/test/default/aggregate_stats.csv",
        # yammbs torsion analyses
        "benchmarking/tnet500/analysis/test/default/metrics.json",
        "benchmarking/tnet500/analysis/validation/ablations/metrics.json",
        "benchmarking/tnet500_reopt_v4/analysis/test/default/metrics.json",
        "benchmarking/tnet500_reopt_v4/analysis/validation/ablations/metrics.json",
        "benchmarking/jacs_fragments/analysis/test/default/metrics.json",
        # JACS Fragments full-molecule fits transfer benchmark (per supplied FF)
        expand(
            "benchmarking/jacs_fragments_full_mol_fits/analysis/test/{ff_label}/metrics.json",
            ff_label=full_molecule_fit_force_field_labels(),
        ),
        # smiles.csv descriptor analysis aggregate
        "benchmarking/analysis/smiles_descriptors/smiles_descriptor_aggregate_mean_std.csv",
        "benchmarking/analysis/smiles_descriptors/smiles_descriptor_aggregate_mean_std.tex",
        # TYK2 reproducibility parameter variability analysis
        "benchmarking/tyk2_reproducibility/analysis/parameter_variability/offxml_variability_summary.tex",
        # TYK2 congeneric series retrain analysis
        "benchmarking/tyk2_congeneric_series/analysis/retrain_error_summary.csv",


############ General Rules #############

rule run_presto:
    input:
        smiles_file="benchmarking/{dataset}/input/{dataset_type}/smiles/{molecule}.smi",
        config_file="configs/{config_name}.yaml",
    output:
        "benchmarking/{dataset}/output/{dataset_type}/{config_name}/{molecule}/bespoke_force_field.offxml",
    threads: 32 # So that only one job at once runs on my workstation...
    resources:
        mem_mb=8000,
        runtime=120,  # minutes
        slurm_partition="gpu-s_free",
        slurm_extra="--gpus-per-task=1",
    shell:
        "pixi run -e default presto-benchmark run-presto {input.config_file} {input.smiles_file} $(dirname {output[0]})"

checkpoint split_test_only_input:
    """Generic split for datasets where everything goes into the test set (frac-test 1.0)."""
    input:
        "benchmarking/{dataset}/input/{dataset}.json"
    output:
        test_set_dir=directory("benchmarking/{dataset}/input/test"),
        test_set_json="benchmarking/{dataset}/input/test/test.json",
        test_set_smiles=directory("benchmarking/{dataset}/input/test/smiles"),
    shell:
        "pixi run -e default presto-benchmark split-qca-input {input[0]} {output.test_set_dir} "
        "--frac-test 1.0 --seed {RANDOM_SEED}"

rule create_combined_force_field:
    input:
        force_fields=validation_force_fields,
        descriptor_summary=smiles_descriptor_summary_input,
    output:
        "benchmarking/{dataset}/output/{dataset_type}/{config_name}/combined_force_field.offxml",
    shell:
        "pixi run -e default presto-benchmark combine-force-fields {output[0]} '{input.force_fields}'"


rule create_tyk2_reproducibility_smiles:
    output:
        "benchmarking/tyk2_reproducibility/input/tyk2.smi",
    run:
        tyk2_config = config["tyk2_reproducibility"]
        output_path = Path(output[0])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"{tyk2_config['smiles']}\n")


rule run_tyk2_reproducibility:
    input:
        smiles_file=rules.create_tyk2_reproducibility_smiles.output[0],
        config_file=lambda wc: (
            f"configs/{config['tyk2_reproducibility'].get('config_name', 'default')}.yaml"
        ),
    output:
        expand(
            "benchmarking/tyk2_reproducibility/output/run_{repeat:02d}/bespoke_force_field.offxml",
            repeat=range(
                1, config["tyk2_reproducibility"].get("repeats", 10) + 1
            ),
        ),
    params:
        repeats=lambda wc: config["tyk2_reproducibility"].get("repeats", 10),
        pixi_environment=lambda wc: config["tyk2_reproducibility"].get(
            "pixi_environment", "default-save-ff-trajectory"
        ),
    resources:
        mem_mb=8000,
        runtime=240,  # minutes
        slurm_partition="gpu-s_free",
        slurm_extra="--gpus-per-task=1",
    shell:
        "for i in $(seq -w 1 {params.repeats}); do "
        "pixi run -e {params.pixi_environment} presto-benchmark run-presto "
        "{input.config_file} {input.smiles_file} benchmarking/tyk2_reproducibility/output/run_$i; "
        "done"


rule analyse_tyk2_reproducibility:
    input:
        rules.run_tyk2_reproducibility.output
    output:
        "benchmarking/tyk2_reproducibility/analysis/parameter_variability/offxml_parameter_values.csv",
        "benchmarking/tyk2_reproducibility/analysis/parameter_variability/offxml_variability_summary.csv",
        "benchmarking/tyk2_reproducibility/analysis/parameter_variability/offxml_variability_summary.tex",
        "benchmarking/tyk2_reproducibility/analysis/parameter_variability/offxml_parameter_values_boxplot.png",
        "benchmarking/tyk2_reproducibility/analysis/parameter_variability/offxml_parameter_values_boxplot_shifted.png",
        "benchmarking/tyk2_reproducibility/analysis/parameter_variability/tensor_parameter_trajectories.csv",
        "benchmarking/tyk2_reproducibility/analysis/parameter_variability/tensor_mean_signed_change_vs_epoch.png",
        "benchmarking/tyk2_reproducibility/analysis/parameter_variability/tensor_mean_absolute_change_vs_epoch.png",
        "benchmarking/tyk2_reproducibility/analysis/parameter_variability/tensor_individual_trajectories.png",
        "benchmarking/tyk2_reproducibility/analysis/parameter_variability/tensor_individual_trajectories_unshifted.png",
    params:
        sample_every_n_epochs=lambda wc: config["tyk2_reproducibility"].get(
            "sample_every_n_epochs", 50
        ),
    shell:
        "pixi run -e default presto-benchmark analyse-tyk2-reproducibility "
        "benchmarking/tyk2_reproducibility/output "
        "benchmarking/tyk2_reproducibility/analysis/parameter_variability "
        "--sample-every-n-epochs {params.sample_every_n_epochs}"


rule create_tyk2_congeneric_series_smiles:
    output:
        "benchmarking/tyk2_congeneric_series/input/tyk2_congeneric_series.smi",
    run:
        congeneric_config = config["tyk2_congeneric_series"]
        smiles = congeneric_config["smiles"]
        if not isinstance(smiles, list) or not smiles:
            raise ValueError("workflow_config.yaml: tyk2_congeneric_series.smiles must be a non-empty list")

        output_path = Path(output[0])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(smiles) + "\n")


rule run_tyk2_congeneric_series:
    input:
        smiles_file=rules.create_tyk2_congeneric_series_smiles.output[0],
        config_file=lambda wc: (
            f"configs/{config['tyk2_congeneric_series'].get('config_name', 'one_it')}.yaml"
        ),
    output:
        "benchmarking/tyk2_congeneric_series/output/training_iteration_1/bespoke_ff.offxml",
    resources:
        mem_mb=8000,
        runtime=240,  # minutes
        slurm_partition="gpu-s_free",
        slurm_extra="--gpus-per-task=1",
    shell:
        "pixi run -e default presto-benchmark run-presto "
        "{input.config_file} {input.smiles_file} benchmarking/tyk2_congeneric_series/output"


rule create_tyk2_congeneric_series_retrain_configs:
    input:
        initial_run_offxml=rules.run_tyk2_congeneric_series.output[0],
        base_config_file=lambda wc: (
            f"configs/{config['tyk2_congeneric_series'].get('config_name', 'one_it')}.yaml"
        ),
    output:
        expand(
            "benchmarking/tyk2_congeneric_series/retrain_configs/max_extend_{max_extend}.yaml",
            max_extend=tyk2_congeneric_retrain_labels(),
        ),
    params:
        max_extend_opts=lambda wc: " ".join(
            f"--max-extend-distance {distance}"
            for distance in config["tyk2_congeneric_series"].get(
                "retrain_max_extend_distances", [0, 1, 2, 3]
            )
        ),
    shell:
        "pixi run -e default presto-benchmark prepare-tyk2-congeneric-retrain-configs "
        "{input.base_config_file} benchmarking/tyk2_congeneric_series/output "
        "benchmarking/tyk2_congeneric_series/retrain_configs "
        "{params.max_extend_opts} --include-sage-types"


rule run_tyk2_congeneric_series_retrain:
    input:
        smiles_file=rules.create_tyk2_congeneric_series_smiles.output[0],
        config_file=(
            "benchmarking/tyk2_congeneric_series/retrain_configs/max_extend_{max_extend}.yaml"
        ),
    output:
        (
            "benchmarking/tyk2_congeneric_series/retrains/max_extend_{max_extend}/"
            "run_{repeat}/training_iteration_1/bespoke_ff.offxml"
        ),
    resources:
        mem_mb=8000,
        runtime=240,  # minutes
        slurm_partition="gpu-s_free",
        slurm_extra="--gpus-per-task=1",
    shell:
        "pixi run -e default presto-benchmark run-presto "
        "{input.config_file} {input.smiles_file} "
        "benchmarking/tyk2_congeneric_series/retrains/max_extend_{wildcards.max_extend}/run_{wildcards.repeat}"


rule analyse_tyk2_congeneric_series_retrains:
    input:
        initial_run_offxml=rules.run_tyk2_congeneric_series.output[0],
        retrain_outputs=expand(
            (
                "benchmarking/tyk2_congeneric_series/retrains/max_extend_{max_extend}/"
                "run_{repeat}/training_iteration_1/bespoke_ff.offxml"
            ),
            max_extend=tyk2_congeneric_retrain_labels(),
            repeat=range(
                1, 1 + 1
            ),
        ),
    output:
        per_run_csv="benchmarking/tyk2_congeneric_series/analysis/retrain_error_per_run.csv",
        per_molecule_csv="benchmarking/tyk2_congeneric_series/analysis/retrain_error_per_molecule.csv",
        summary_csv="benchmarking/tyk2_congeneric_series/analysis/retrain_error_summary.csv",
        error_plot_png="benchmarking/tyk2_congeneric_series/analysis/error_vs_max_extend_distance.png",
    params:
        max_extend_opts=lambda wc: " ".join(
            f"--max-extend-distance {distance}"
            for distance in config["tyk2_congeneric_series"].get(
                "retrain_max_extend_distances", [0, 1, 2, 3]
            )
        ),
    shell:
        "pixi run -e default presto-benchmark analyse-tyk2-congeneric-retrains "
        "benchmarking/tyk2_congeneric_series/output "
        "benchmarking/tyk2_congeneric_series/retrains "
        "benchmarking/tyk2_congeneric_series/analysis "
        "--repeats 1 "
        "{params.max_extend_opts}"


rule analyse_smiles_descriptors:
    input:
        smiles_csv=smiles_csv_input,
    output:
        descriptor_csv="benchmarking/{dataset}/input/{dataset_type}/smiles_descriptors.csv",
        descriptor_tex="benchmarking/{dataset}/input/{dataset_type}/smiles_descriptors.tex",
        summary_csv="benchmarking/{dataset}/input/{dataset_type}/smiles_descriptor_summary.csv",
        summary_tex="benchmarking/{dataset}/input/{dataset_type}/smiles_descriptor_summary.tex",
        mean_std_csv="benchmarking/{dataset}/input/{dataset_type}/smiles_descriptor_mean_std.csv",
        mean_std_tex="benchmarking/{dataset}/input/{dataset_type}/smiles_descriptor_mean_std.tex",
        plots_dir=directory("benchmarking/{dataset}/input/{dataset_type}/smiles_descriptor_plots"),
    wildcard_constraints:
        dataset="tnet500|jacs_fragments|jacs_fragments_full_mol_fits|1mer_backbone|3mer_backbone|1mer_side_chain",
        dataset_type="test|validation",
    shell:
        "pixi run -e default presto-benchmark analyse-smiles-descriptors {input.smiles_csv}"


rule aggregate_smiles_descriptors:
    input:
        tnet500_test="benchmarking/tnet500/input/test/smiles_descriptor_summary.csv",
        tnet500_validation="benchmarking/tnet500/input/validation/smiles_descriptor_summary.csv",
        jacs_test="benchmarking/jacs_fragments/input/test/smiles_descriptor_summary.csv",
    output:
        aggregate_csv="benchmarking/analysis/smiles_descriptors/smiles_descriptor_aggregate_mean_std.csv",
        aggregate_tex="benchmarking/analysis/smiles_descriptors/smiles_descriptor_aggregate_mean_std.tex",
    params:
        output_dir="benchmarking/analysis/smiles_descriptors",
    shell:
        "pixi run -e default presto-benchmark aggregate-smiles-descriptors {params.output_dir} "
        "benchmarking/tnet500/input/test/smiles.csv "
        "benchmarking/tnet500/input/validation/smiles.csv "
        "benchmarking/jacs_fragments/input/test/smiles.csv"

rule analyse_torsion_scans_yammbs:
    input:
        qca_data_json="benchmarking/{dataset}/input/{dataset_type}/{dataset_type}.json",
        combined_ff="benchmarking/{dataset}/output/{dataset_type}/{config_name}/combined_force_field.offxml",
    output:
        metrics_json="benchmarking/{dataset}/analysis/{dataset_type}/{config_name}/metrics.json",
        minimized_json="benchmarking/{dataset}/analysis/{dataset_type}/{config_name}/minimized.json",
        plot_png="benchmarking/{dataset}/analysis/{dataset_type}/{config_name}/plots/rmse.png",
        paired_stats_png="benchmarking/{dataset}/analysis/{dataset_type}/{config_name}/plots/paired_stats.png",
        paired_stats_no_sig_png="benchmarking/{dataset}/analysis/{dataset_type}/{config_name}/plots/paired_stats_no_sig.png",
        fit_rmse_plot_png="benchmarking/{dataset}/analysis/{dataset_type}/{config_name}/plots/presto_fit_validation_energy_rmse.png",
        fit_rmse_summary_csv="benchmarking/{dataset}/analysis/{dataset_type}/{config_name}/plots/presto_fit_validation_energy_rmse_summary.csv",
        fit_rmse_summary_tex="benchmarking/{dataset}/analysis/{dataset_type}/{config_name}/plots/presto_fit_validation_energy_rmse_summary.tex",
    wildcard_constraints:
        dataset="tnet500|tnet500_reopt_v4|jacs_fragments",
    params:
        analysis_dir=lambda wc: f"benchmarking/{wc.dataset}/analysis/{wc.dataset_type}/{wc.config_name}",
        presto_output_dir=lambda wc: f"benchmarking/{wc.dataset}/output/{wc.dataset_type}/{wc.config_name}",
        base_ff_opts=lambda wc: " ".join(
            f"--base-force-field '{ff}'"
            for ff in yammbs_target_config(wc).get(
                "base_force_fields", config["yammbs_analysis"]["base_force_fields"]
            )
        ),
        extra_ff_opts=lambda wc: " ".join(
            f"--extra-force-field '{ff}'"
            for ff in yammbs_target_config(wc)["extra_force_fields"]
        ),
        torsion_plot_id_opts=lambda wc: " ".join(
            f"--plot-torsion-id {torsion_id}"
            for torsion_id in yammbs_target_config(wc).get("torsion_plot_ids", [])
        ),
    shell:
        "pixi run -e default presto-benchmark analyse-torsion-scans "
        "{input.qca_data_json} {input.combined_ff} {params.analysis_dir} "
        "{params.base_ff_opts} {params.extra_ff_opts} {params.torsion_plot_id_opts} && "
        "pixi run -e default presto-benchmark analyse-presto-fits "
        "{params.presto_output_dir} {params.analysis_dir}/plots "
        "--random-seed {RANDOM_SEED}"


############ Folmsbee Conformers #############

rule get_folmsbee_conformer_input:
    output:
        directory("benchmarking/folmsbee_conformers/input/gh_repo"),
    shell:
        "pixi run -e default presto-benchmark get-folmsbee-input {output[0]}"


checkpoint process_folmsbee_smiles:
    input:
        gh_repo=rules.get_folmsbee_conformer_input.output[0]
    output:
        directory("benchmarking/folmsbee_conformers/input/test/smiles")
    shell:
        "pixi run -e default presto-benchmark process-folmsbee-smiles "
        "{input.gh_repo}/SMILES/molecules.smi {output}"


rule analyse_folmsbee_conformers:
    input:
        gh_repo=rules.get_folmsbee_conformer_input.output[0],
        combined_ff="benchmarking/folmsbee_conformers/output/{dataset_type}/{config_name}/combined_force_field.offxml",
    output:
        results_csv="benchmarking/folmsbee_conformers/analysis/{dataset_type}/{config_name}/results.csv",
        per_molecule_stats_csv="benchmarking/folmsbee_conformers/analysis/{dataset_type}/{config_name}/per_molecule_stats.csv",
        aggregate_stats_csv="benchmarking/folmsbee_conformers/analysis/{dataset_type}/{config_name}/aggregate_stats.csv",
        fit_rmse_plot_png="benchmarking/folmsbee_conformers/analysis/{dataset_type}/{config_name}/plots/presto_fit_validation_energy_rmse.png",
        fit_rmse_summary_csv="benchmarking/folmsbee_conformers/analysis/{dataset_type}/{config_name}/plots/presto_fit_validation_energy_rmse_summary.csv",
        fit_rmse_summary_tex="benchmarking/folmsbee_conformers/analysis/{dataset_type}/{config_name}/plots/presto_fit_validation_energy_rmse_summary.tex",
        plots_dir=directory("benchmarking/folmsbee_conformers/analysis/{dataset_type}/{config_name}/plots"),
    params:
        analysis_dir=lambda wc: f"benchmarking/folmsbee_conformers/analysis/{wc.dataset_type}/{wc.config_name}",
        presto_output_dir=lambda wc: f"benchmarking/folmsbee_conformers/output/{wc.dataset_type}/{wc.config_name}",
        precomputed_method_opts=lambda wc: " ".join(
            f"--precomputed-method '{method}'"
            for method in config["folmsbee_analysis"]["precomputed_methods"]
        ),
        mlp_opts=lambda wc: " ".join(
            f"--mlp-name '{name}'"
            for name in config["folmsbee_analysis"].get("mlp_names", [])
        ),
        mlp_mode_opt=lambda wc: (
            "--single-point-mlp"
            if config["folmsbee_analysis"].get("single_point_mlp", True)
            else "--minimise-mlp"
        ),
        extra_ff_opts=lambda wc: " ".join(
            f"--force-field '{ff}'"
            for ff in folmsbee_target_config(wc).get("extra_force_fields", [])
        ),
        reference_method=lambda wc: config["folmsbee_analysis"]["reference_method"],
        torsion_restraint_force_constant=lambda wc: config["folmsbee_analysis"][
            "torsion_restraint_force_constant"
        ],
        mm_minimization_steps=lambda wc: config["folmsbee_analysis"][
            "mm_minimization_steps"
        ],
        exclude_smarts_opts=lambda wc: " ".join(
            f"--exclude-smarts '{smarts}'"
            for smarts in config["folmsbee_analysis"].get("exclude_smarts", [])
        ),
        min_conformers_per_molecule=lambda wc: config["folmsbee_analysis"].get(
            "min_conformers_per_molecule", 5
        ),
        min_reference_energy_window=lambda wc: config["folmsbee_analysis"].get(
            "min_reference_energy_window", 0.0
        ),
        n_processes_opt=lambda wc: (
            f"--n-processes {config['folmsbee_analysis']['n_processes']}"
            if config["folmsbee_analysis"].get("n_processes") is not None
            else ""
        ),
    shell:
        "pixi run -e default presto-benchmark analyse-folmsbee "
        "{input.gh_repo} {params.presto_output_dir} {params.analysis_dir} "
        "--reference-method '{params.reference_method}' "
        "--torsion-restraint-force-constant {params.torsion_restraint_force_constant} "
        "--mm-minimization-steps {params.mm_minimization_steps} "
        "--min-conformers-per-molecule {params.min_conformers_per_molecule} "
        "--min-reference-energy-window {params.min_reference_energy_window} "
        "--force-field '{input.combined_ff}' "
        "{params.extra_ff_opts} "
        "{params.exclude_smarts_opts} "
        "{params.mlp_opts} "
        "{params.mlp_mode_opt} "
        "{params.precomputed_method_opts} "
        "{params.n_processes_opt} && "
        "pixi run -e default presto-benchmark analyse-presto-fits "
        "{params.presto_output_dir} {params.analysis_dir}/plots "
        "--random-seed {RANDOM_SEED}"


############ TNet 500 #############

rule get_tnet500_input:
    output:
        "benchmarking/tnet500/input/full_dataset.json"
    shell:
        "pixi run -e default presto-benchmark get-tnet500-input {output[0]}"


rule get_tnet500_reopt_v4_input:
    output:
        "benchmarking/tnet500_reopt_v4/input/full_dataset.json"
    shell:
        "pixi run -e default presto-benchmark get-qca-torsion-input "
        "'{TNET500_REOPT_V4_QCA_DATASET}' {output[0]}"


rule subset_tnet500_reopt_v4_to_existing_split:
    input:
        full_dataset_json="benchmarking/tnet500_reopt_v4/input/full_dataset.json",
        split_smiles_csv="benchmarking/tnet500/input/{dataset_type}/smiles.csv",
    output:
        subset_json="benchmarking/tnet500_reopt_v4/input/{dataset_type}/{dataset_type}.json",
    wildcard_constraints:
        dataset_type="test|validation",
    shell:
        "pixi run -e default presto-benchmark subset-qca-input-by-smiles "
        "{input.full_dataset_json} {input.split_smiles_csv} {output.subset_json}"

checkpoint split_tnet500_input:
    input:
        "benchmarking/tnet500/input/full_dataset.json"
    output:
        validation_set_dir=directory("benchmarking/tnet500/input/validation"),
        validation_set_json="benchmarking/tnet500/input/validation/validation.json",
        validation_set_smiles=directory("benchmarking/tnet500/input/validation/smiles"),
        test_set_dir=directory("benchmarking/tnet500/input/test"),
        test_set_json="benchmarking/tnet500/input/test/test.json",
        test_set_smiles=directory("benchmarking/tnet500/input/test/smiles"),
    shell:
        "pixi run -e default presto-benchmark split-qca-input {input[0]} {output.test_set_dir} "
        "--frac-test {TNET_500_FRAC_TEST} --seed {RANDOM_SEED} "
        "--validation-output-path {output.validation_set_dir}"


rule analyse_tnet500_validation_ablations:
    input:
        qca_data_json="benchmarking/tnet500/input/validation/validation.json",
        default_ff="benchmarking/tnet500/output/validation/default/combined_force_field.offxml",
        no_reg_ff="benchmarking/tnet500/output/validation/no_reg/combined_force_field.offxml",
        no_min_ff="benchmarking/tnet500/output/validation/no_min/combined_force_field.offxml",
        one_it_ff="benchmarking/tnet500/output/validation/one_it/combined_force_field.offxml",
        no_metad_ff="benchmarking/tnet500/output/validation/no_metad/combined_force_field.offxml",
    output:
        metrics_json="benchmarking/tnet500/analysis/validation/ablations/metrics.json",
        minimized_json="benchmarking/tnet500/analysis/validation/ablations/minimized.json",
        plot_png="benchmarking/tnet500/analysis/validation/ablations/plots/rmse.png",
        heatmap_png="benchmarking/tnet500/analysis/validation/ablations/plots/heatmap.png",
        distributions_png="benchmarking/tnet500/analysis/validation/ablations/plots/distributions.png",
        fit_rmse_plot_png="benchmarking/tnet500/analysis/validation/ablations/plots/presto_fit_validation_energy_rmse.png",
        fit_rmse_summary_csv="benchmarking/tnet500/analysis/validation/ablations/plots/presto_fit_validation_energy_rmse_summary.csv",
        fit_rmse_summary_tex="benchmarking/tnet500/analysis/validation/ablations/plots/presto_fit_validation_energy_rmse_summary.tex",
    params:
        analysis_dir="benchmarking/tnet500/analysis/validation/ablations",
        presto_output_dir="benchmarking/tnet500/output/validation/default",
        base_ff_opts=" ".join(
            f"--base-force-field '{ff}'"
            for ff in config["yammbs_analysis"]["base_force_fields"]
        ),
        torsion_plot_id_opts=" ".join(
            f"--plot-torsion-id {torsion_id}"
            for torsion_id in config["yammbs_analysis"]["targets"]["tnet500"][
                "validation"
            ].get("torsion_plot_ids", [])
        ),
    shell:
        "pixi run -e default presto-benchmark analyse-torsion-scans "
        "{input.qca_data_json} {input.default_ff} {params.analysis_dir} "
        "{params.base_ff_opts} "
        "--extra-force-field '{input.no_reg_ff}' "
        "--extra-force-field '{input.no_min_ff}' "
        "--extra-force-field '{input.one_it_ff}' "
        "--extra-force-field '{input.no_metad_ff}' "
        "{params.torsion_plot_id_opts} && "
        "pixi run -e default presto-benchmark analyse-presto-fits "
        "{params.presto_output_dir} {params.analysis_dir}/plots "
        "--random-seed {RANDOM_SEED} && "
        "pixi run -e default presto-benchmark plot-ablation-comparison "
        "{output.metrics_json} {params.analysis_dir}/plots"


rule analyse_tnet500_reopt_v4_test_default:
    input:
        qca_data_json="benchmarking/tnet500_reopt_v4/input/test/test.json",
        combined_ff="benchmarking/tnet500/output/test/default/combined_force_field.offxml",
    output:
        metrics_json="benchmarking/tnet500_reopt_v4/analysis/test/default/metrics.json",
        minimized_json="benchmarking/tnet500_reopt_v4/analysis/test/default/minimized.json",
        plot_png="benchmarking/tnet500_reopt_v4/analysis/test/default/plots/rmse.png",
        paired_stats_png="benchmarking/tnet500_reopt_v4/analysis/test/default/plots/paired_stats.png",
        paired_stats_no_sig_png="benchmarking/tnet500_reopt_v4/analysis/test/default/plots/paired_stats_no_sig.png",
        fit_rmse_plot_png="benchmarking/tnet500_reopt_v4/analysis/test/default/plots/presto_fit_validation_energy_rmse.png",
        fit_rmse_summary_csv="benchmarking/tnet500_reopt_v4/analysis/test/default/plots/presto_fit_validation_energy_rmse_summary.csv",
        fit_rmse_summary_tex="benchmarking/tnet500_reopt_v4/analysis/test/default/plots/presto_fit_validation_energy_rmse_summary.tex",
    params:
        analysis_dir="benchmarking/tnet500_reopt_v4/analysis/test/default",
        presto_output_dir="benchmarking/tnet500/output/test/default",
        base_ff_opts=" ".join(
            f"--base-force-field '{ff}'"
            for ff in config["yammbs_analysis"]["base_force_fields"]
        ),
        extra_ff_opts=" ".join(
            f"--extra-force-field '{ff}'"
            for ff in config["yammbs_analysis"]["targets"]["tnet500"]["test"][
                "extra_force_fields"
            ]
        ),
    shell:
        "pixi run -e default presto-benchmark analyse-torsion-scans "
        "{input.qca_data_json} {input.combined_ff} {params.analysis_dir} "
        "{params.base_ff_opts} {params.extra_ff_opts} && "
        "pixi run -e default presto-benchmark analyse-presto-fits "
        "{params.presto_output_dir} {params.analysis_dir}/plots "
        "--random-seed {RANDOM_SEED}"


rule analyse_tnet500_reopt_v4_validation_ablations:
    input:
        qca_data_json="benchmarking/tnet500_reopt_v4/input/validation/validation.json",
        default_ff="benchmarking/tnet500/output/validation/default/combined_force_field.offxml",
        no_reg_ff="benchmarking/tnet500/output/validation/no_reg/combined_force_field.offxml",
        no_min_ff="benchmarking/tnet500/output/validation/no_min/combined_force_field.offxml",
        one_it_ff="benchmarking/tnet500/output/validation/one_it/combined_force_field.offxml",
        no_metad_ff="benchmarking/tnet500/output/validation/no_metad/combined_force_field.offxml",
    output:
        metrics_json="benchmarking/tnet500_reopt_v4/analysis/validation/ablations/metrics.json",
        minimized_json="benchmarking/tnet500_reopt_v4/analysis/validation/ablations/minimized.json",
        plot_png="benchmarking/tnet500_reopt_v4/analysis/validation/ablations/plots/rmse.png",
        heatmap_png="benchmarking/tnet500_reopt_v4/analysis/validation/ablations/plots/heatmap.png",
        distributions_png="benchmarking/tnet500_reopt_v4/analysis/validation/ablations/plots/distributions.png",
        fit_rmse_plot_png="benchmarking/tnet500_reopt_v4/analysis/validation/ablations/plots/presto_fit_validation_energy_rmse.png",
        fit_rmse_summary_csv="benchmarking/tnet500_reopt_v4/analysis/validation/ablations/plots/presto_fit_validation_energy_rmse_summary.csv",
        fit_rmse_summary_tex="benchmarking/tnet500_reopt_v4/analysis/validation/ablations/plots/presto_fit_validation_energy_rmse_summary.tex",
    params:
        analysis_dir="benchmarking/tnet500_reopt_v4/analysis/validation/ablations",
        presto_output_dir="benchmarking/tnet500/output/validation/default",
        base_ff_opts=" ".join(
            f"--base-force-field '{ff}'"
            for ff in config["yammbs_analysis"]["base_force_fields"]
        ),
        torsion_plot_id_opts=" ".join(
            f"--plot-torsion-id {torsion_id}"
            for torsion_id in config["yammbs_analysis"]["targets"]["tnet500"][
                "validation"
            ].get("torsion_plot_ids", [])
        ),
    shell:
        "pixi run -e default presto-benchmark analyse-torsion-scans "
        "{input.qca_data_json} {input.default_ff} {params.analysis_dir} "
        "{params.base_ff_opts} "
        "--extra-force-field '{input.no_reg_ff}' "
        "--extra-force-field '{input.no_min_ff}' "
        "--extra-force-field '{input.one_it_ff}' "
        "--extra-force-field '{input.no_metad_ff}' "
        "{params.torsion_plot_id_opts} && "
        "pixi run -e default presto-benchmark analyse-presto-fits "
        "{params.presto_output_dir} {params.analysis_dir}/plots "
        "--random-seed {RANDOM_SEED} && "
        "pixi run -e default presto-benchmark plot-ablation-comparison "
        "{output.metrics_json} {params.analysis_dir}/plots"


############ JACS Fragments #############

rule get_qca_torsion_input_dataset:
    output:
        "benchmarking/{dataset}/input/{dataset}.json"
    wildcard_constraints:
        dataset="|".join(QCA_DATASET_NAMES.keys()),
    params:
        qca_dataset_name=lambda wc: QCA_DATASET_NAMES[wc.dataset],
        exclude_opts=lambda wc: qca_exclude_smiles_opts(QCA_DATASET_NAMES[wc.dataset]),
    shell:
        "pixi run -e default presto-benchmark get-qca-torsion-input "
        "'{params.qca_dataset_name}' {output[0]} {params.exclude_opts}"


rule filter_jacs_fragments_full_mol_fits_torsions_per_force_field:
    input:
        qca_data_json="benchmarking/jacs_fragments_full_mol_fits/input/test/test.json",
        force_field=full_molecule_fit_force_field_path,
    output:
        filtered_qca_data_json="benchmarking/jacs_fragments_full_mol_fits/input_filtered/test/{ff_label}/test_filtered_bespoke.json",
    wildcard_constraints:
        ff_label="|".join(full_molecule_fit_force_field_labels()),
    shell:
        "pixi run -e default presto-benchmark filter-qca-torsions-by-bespoke-scans "
        "{input.qca_data_json} {input.force_field} {output.filtered_qca_data_json}"


rule analyse_jacs_fragments_full_mol_fits_per_force_field:
    input:
        qca_data_json=rules.filter_jacs_fragments_full_mol_fits_torsions_per_force_field.output.filtered_qca_data_json,
        force_field=full_molecule_fit_force_field_path,
    output:
        metrics_json="benchmarking/jacs_fragments_full_mol_fits/analysis/test/{ff_label}/metrics.json",
        minimized_json="benchmarking/jacs_fragments_full_mol_fits/analysis/test/{ff_label}/minimized.json",
        plot_png="benchmarking/jacs_fragments_full_mol_fits/analysis/test/{ff_label}/plots/rmse.png",
        paired_stats_png="benchmarking/jacs_fragments_full_mol_fits/analysis/test/{ff_label}/plots/paired_stats.png",
        paired_stats_no_sig_png="benchmarking/jacs_fragments_full_mol_fits/analysis/test/{ff_label}/plots/paired_stats_no_sig.png",
    wildcard_constraints:
        ff_label="|".join(full_molecule_fit_force_field_labels()),
    params:
        analysis_dir=lambda wc: f"benchmarking/jacs_fragments_full_mol_fits/analysis/test/{wc.ff_label}",
        base_ff_opts=lambda wc: " ".join(
            f"--base-force-field '{ff}'"
            for ff in config["yammbs_analysis"]["targets"]["jacs_fragments_full_mol_fits"]["test"].get(
                "base_force_fields", config["yammbs_analysis"]["base_force_fields"]
            )
        ),
        extra_ff_opts=lambda wc: " ".join(
            f"--extra-force-field '{ff}'"
            for ff in config["yammbs_analysis"]["targets"]["jacs_fragments_full_mol_fits"]["test"][
                "extra_force_fields"
            ]
        ),
        torsion_plot_id_opts=lambda wc: " ".join(
            f"--plot-torsion-id {torsion_id}"
            for torsion_id in config["yammbs_analysis"]["targets"]["jacs_fragments_full_mol_fits"][
                "test"
            ].get("torsion_plot_ids", [])
        ),
    shell:
        "pixi run -e default presto-benchmark analyse-torsion-scans "
        "{input.qca_data_json} {input.force_field} {params.analysis_dir} "
        "{params.base_ff_opts} {params.extra_ff_opts} {params.torsion_plot_id_opts}"





############ Proteins #############

rule get_qca_input_for_protein_torsions:
    output:
        qca_data_json="benchmarking/{dataset}/input/qca_data.json",
        qca_names_json="benchmarking/{dataset}/input/qca_names.json",
    wildcard_constraints:
        dataset="|".join(PROTEIN_DATASETS),
    params:
        qca_dataset_name=lambda wc: QCA_DATASET_NAMES[wc.dataset],
    shell:
        "pixi run -e default presto-benchmark get-qca-input-proteins "
        "'{params.qca_dataset_name}' "
        "{output.qca_data_json} {output.qca_names_json}"

rule run_protein_torsion_minimisation:
    input:
        qca_data_json="benchmarking/{dataset}/input/qca_data.json",
        combined_ff=protein_torsion_combined_ff,
    output:
        directory("benchmarking/{dataset}/analysis/{dataset_type}/{config_name}/minimised"),
    wildcard_constraints:
        dataset="|".join(PROTEIN_DATASETS),
    params:
        ff_config=config["protein_force_fields"],
    run:
        ff_config = dict(params.ff_config)
        ff_config[wildcards.config_name] = {
            "ff_path": input.combined_ff,
            "ff_type": "smirnoff-nagl",
        }

        # Write force field config to temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(ff_config, f)
            config_path = f.name
        
        shell(
            f"pixi run -e espaloma presto-benchmark minimise-protein-torsion-multi "
            f"{input.qca_data_json} {output[0]} --config {config_path}"
        )

rule plot_protein_torsion_analysis:
    input:
        minimised_dir="benchmarking/{dataset}/analysis/{dataset_type}/{config_name}/minimised",
        qca_names_json="benchmarking/{dataset}/input/qca_names.json",
    output:
        directory("benchmarking/{dataset}/analysis/{dataset_type}/{config_name}/plots"),
    wildcard_constraints:
        dataset="|".join(PROTEIN_DATASETS),
    shell:
        "pixi run -e default presto-benchmark plot-protein-torsion {input.minimised_dir} {output[0]} "
        "--names-file {input.qca_names_json}"
