from pathlib import Path
from typing import Any
import json
import tempfile

configfile: "workflow_config.yaml"

RANDOM_SEED = config["random_seed"]
TNET_500_FRAC_TEST = config["tnet500_frac_test"]

############ Convenience Functions #############

def smiles_dir_outputs(wildcards: Any, checkpoint_obj: Any, smiles_dir: str, output_pattern: str) -> list[str]:
    """Expand output_pattern over all .smi files in smiles_dir once checkpoint_obj is done."""
    checkpoint_obj.get()
    molecules = glob_wildcards(f"{smiles_dir}/{{molecule}}.smi").molecule
    return expand(output_pattern, molecule=molecules)


def validation_force_fields(wildcards: Any) -> list[str]:
    """Generic input function for create_combined_force_field.

    Infers the smiles directory from the dataset wildcard and resolves
    the per-molecule force field paths after the relevant checkpoint completes.
    """
    dataset = wildcards.dataset
    checkpoint_obj = getattr(checkpoints, f"split_{dataset}_input", None)
    if checkpoint_obj is None:
        raise ValueError(f"No checkpoint available for dataset '{dataset}'")
    return smiles_dir_outputs(
        wildcards,
        checkpoint_obj=checkpoint_obj,
        smiles_dir=f"benchmarking/{dataset}/input/{wildcards.dataset_type}/smiles",
        output_pattern=f"benchmarking/{dataset}/output/{wildcards.dataset_type}/{wildcards.config_name}/{{molecule}}/bespoke_force_field.offxml",
    )


def yammbs_target_config(wildcards: Any) -> dict[str, Any]:
    """Return yammbs config for a dataset/split target."""
    return config["yammbs_analysis"]["targets"][wildcards.dataset][wildcards.dataset_type]

############ Workflow Rules #############


rule all:
    input:
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
        # yammbs torsion analyses
        "benchmarking/tnet500/analysis/test/default/metrics.json",
        "benchmarking/tnet500/analysis/validation/ablations/metrics.json",
        "benchmarking/jacs_fragments/analysis/test/default/metrics.json",


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

rule create_combined_force_field:
    input:
        force_fields=validation_force_fields,
    output:
        "benchmarking/{dataset}/output/{dataset_type}/{config_name}/combined_force_field.offxml",
    shell:
        "pixi run -e default presto-benchmark combine-force-fields {output[0]} '{input.force_fields}'"

rule analyse_torsion_scans_yammbs:
    input:
        qca_data_json="benchmarking/{dataset}/input/{dataset_type}/{dataset_type}.json",
        combined_ff="benchmarking/{dataset}/output/{dataset_type}/{config_name}/combined_force_field.offxml",
    output:
        metrics_json="benchmarking/{dataset}/analysis/{dataset_type}/{config_name}/metrics.json",
        minimized_json="benchmarking/{dataset}/analysis/{dataset_type}/{config_name}/minimized.json",
        plot_png="benchmarking/{dataset}/analysis/{dataset_type}/{config_name}/plots/rmse.png",
    params:
        analysis_dir=lambda wc: f"benchmarking/{wc.dataset}/analysis/{wc.dataset_type}/{wc.config_name}",
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
    shell:
        "pixi run -e default presto-benchmark analyse-torsion-scans "
        "{input.qca_data_json} {input.combined_ff} {params.analysis_dir} "
        "{params.base_ff_opts} {params.extra_ff_opts}"


############ TNet 500 #############

rule get_tnet500_input:
    output:
        "benchmarking/tnet500/input/full_dataset.json"
    shell:
        "pixi run -e default presto-benchmark get-tnet500-input {output[0]}"

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
    params:
        analysis_dir="benchmarking/tnet500/analysis/validation/ablations",
        base_ff_opts=" ".join(
            f"--base-force-field '{ff}'"
            for ff in config["yammbs_analysis"]["base_force_fields"]
        ),
    shell:
        "pixi run -e default presto-benchmark analyse-torsion-scans "
        "{input.qca_data_json} {input.default_ff} {params.analysis_dir} "
        "{params.base_ff_opts} "
        "--extra-force-field '{input.no_reg_ff}' "
        "--extra-force-field '{input.no_min_ff}' "
        "--extra-force-field '{input.one_it_ff}' "
        "--extra-force-field '{input.no_metad_ff}' && "
        "pixi run -e default presto-benchmark plot-ablation-comparison "
        "{output.metrics_json} {params.analysis_dir}/plots"


############ JACS Fragments #############

rule get_jacs_fragments_input:
    output:
        "benchmarking/jacs_fragments/input/jacs_fragments.json"
    shell:
        "pixi run -e default presto-benchmark get-qca-torsion-input "
        "'OpenFF-benchmark-ligand-fragments-v2.0' {output[0]}"

checkpoint split_jacs_fragments_input:
    input:
        "benchmarking/jacs_fragments/input/jacs_fragments.json"
    output:
        test_set_dir=directory("benchmarking/jacs_fragments/input/test"),
        test_set_json="benchmarking/jacs_fragments/input/test/test.json",
        test_set_smiles=directory("benchmarking/jacs_fragments/input/test/smiles"),
    shell:
        "pixi run -e default presto-benchmark split-qca-input {input[0]} {output.test_set_dir} "
        "--frac-test 1.0 --seed {RANDOM_SEED}"



############ Proteins #############

rule get_1mer_backbone_input:
    output:
        "benchmarking/1mer_backbone/input/1mer_backbone.json"
    shell:
        "pixi run -e default presto-benchmark get-qca-torsion-input "
        "'OpenFF Protein Dipeptide 2-D TorsionDrive v2.0' {output[0]}"

rule get_1mer_sidechain_input:
    output:
        "benchmarking/1mer_side_chain/input/1mer_side_chain.json"
    shell:
        "pixi run -e default presto-benchmark get-qca-torsion-input "
        "'OpenFF Protein Dipeptide Sidechain TorsionDrive v1.0' {output[0]}"


checkpoint split_1mer_backbone_input:
    """Effectively a dummy rule as we just process everything into the test set."""
    input:
        "benchmarking/1mer_backbone/input/1mer_backbone.json"
    output:
        test_set_dir=directory("benchmarking/1mer_backbone/input/test"),
        test_set_json="benchmarking/1mer_backbone/input/test/test.json",
        test_set_smiles=directory("benchmarking/1mer_backbone/input/test/smiles"),
    shell:
        "pixi run -e default presto-benchmark split-qca-input {input[0]} {output.test_set_dir} "
        "--frac-test 1.0 --seed {RANDOM_SEED}"


rule get_qca_input_for_protein_torsions:
    output:
        qca_data_json="benchmarking/1mer_backbone/input/qca_data.json",
        qca_names_json="benchmarking/1mer_backbone/input/qca_names.json",
    shell:
        "pixi run -e default presto-benchmark get-qca-input-proteins "
        "'OpenFF Protein Dipeptide 2-D TorsionDrive v2.0' "
        "{output.qca_data_json} {output.qca_names_json}"

rule run_protein_torsion_minimisation:
    input:
        qca_data_json="benchmarking/1mer_backbone/input/qca_data.json",
        combined_ff="benchmarking/1mer_backbone/output/test/{config_name}/combined_force_field.offxml",
    output:
        protected(directory("benchmarking/1mer_backbone/analysis/{config_name}/minimised")),
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
        minimised_dir="benchmarking/1mer_backbone/analysis/{config_name}/minimised",
        qca_names_json="benchmarking/1mer_backbone/input/qca_names.json",
    output:
        directory("benchmarking/1mer_backbone/analysis/{config_name}/plots"),
    shell:
        "pixi run -e default presto-benchmark plot-protein-torsion {input.minimised_dir} {output[0]} "
        "--names-file {input.qca_names_json}"
