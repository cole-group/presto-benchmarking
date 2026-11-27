from pathlib import Path
from typing import Any
import json
import tempfile
import shutil

configfile: "workflow_config.yaml"

RANDOM_SEED = config["random_seed"]
TNET_500_FRAC_TEST = config["tnet500_frac_test"]

# Reproduce analyses from the committed combined force fields without the fitting
# stage. Enable with `--config skip_fits=True` (see the snakemake-no-refit task).
# When set, create_combined_force_field accepts the committed *.offxml as-is
# (no per-molecule fits required) and rule all drops the results that genuinely
# need the raw fits (see FIT_DEPENDENT_TARGETS).
SKIP_FITS = str(config.get("skip_fits", "")).lower() in ("true", "1", "yes")
TNET500_REOPT_V4_QCA_DATASET = "TorsionNet500 Re-optimization TorsionDrives v4.0"

QCA_DATASET_NAMES = {
    "jacs_fragments": "OpenFF-benchmark-ligand-fragments-v2.0",
    "jacs_fragments_full_mol_fits": "OpenFF-benchmark-ligand-fragments-v2.0",
    "phosphate_torsion_drives": "OpenFF Lipid Torsion Drives v4.0",
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

    In skip_fits mode the per-molecule fits are not required (the committed
    combined force field is used as-is), so return no inputs; this leaves the
    committed combined_force_field.offxml up to date rather than forcing a rebuild
    from missing fits.
    """
    if SKIP_FITS:
        return []

    dataset = wildcards.dataset
    checkpoint_kwargs: dict = {}
    if dataset == "folmsbee_conformers":
        if wildcards.dataset_type == "test":
            checkpoint_obj = checkpoints.process_folmsbee_smiles
        else:
            checkpoint_obj = checkpoints.subset_folmsbee_smiles
            checkpoint_kwargs = {"dataset_type": wildcards.dataset_type}
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

def get_pixi_env(wc):
    """Set the 0.5.1 presto env for older tnet500 and jacs_fragments datasets, 0.6.0 for everything else."""
    env = "presto051" if wc.dataset in ("tnet500", "jacs_fragments") else "default"
    return env


def presto_fit_source_dir(wildcards: Any) -> str:
    """PRESTO fit output directory feeding the per-atom-energy validation analysis.

    Mirrors the presto_output_dir each bundled analysis rule used to pass to
    analyse-presto-fits: tnet500_reopt_v4 reuses the tnet500 fits, and the
    ablation analyses read the 'default' config's fits rather than a config
    literally called 'ablations'.
    """
    dataset = wildcards.dataset
    dataset_type = wildcards.dataset_type
    config_name = wildcards.config_name
    if dataset == "tnet500_reopt_v4":
        return f"benchmarking/tnet500/output/{dataset_type}/default"
    if config_name == "ablations":
        return f"benchmarking/{dataset}/output/{dataset_type}/default"
    return f"benchmarking/{dataset}/output/{dataset_type}/{config_name}"


def presto_fit_combined_ff(wildcards: Any) -> str:
    """Combined force field in the fit source dir, used as a tracked sentinel for the fits."""
    return f"{presto_fit_source_dir(wildcards)}/combined_force_field.offxml"


def presto_fit_pixi_env(wildcards: Any) -> str:
    """Match the pixi env each original bundled rule used for analyse-presto-fits."""
    if wildcards.dataset == "tnet500_reopt_v4" or wildcards.config_name == "ablations":
        return "default"
    return get_pixi_env(wildcards)

def smiles_csv_input(wildcards: Any) -> str:
    """Resolve smiles.csv path for a dataset/split after the relevant checkpoint."""
    dataset = wildcards.dataset

    if dataset == "folmsbee_conformers":
        if wildcards.dataset_type == "test":
            checkpoints.process_folmsbee_smiles.get()
        else:
            checkpoints.subset_folmsbee_smiles.get(dataset_type=wildcards.dataset_type)
    else:
        checkpoint_obj = getattr(checkpoints, f"split_{dataset}_input", None)
        checkpoint_kwargs: dict = {}
        if checkpoint_obj is None:
            checkpoint_obj = checkpoints.split_test_only_input
            checkpoint_kwargs = {"dataset": dataset}

        checkpoint_obj.get(**checkpoint_kwargs)

    return f"benchmarking/{dataset}/input/{wildcards.dataset_type}/smiles.csv"


def folmsbee_smiles_dir(wildcards: Any) -> str:
    """Per-molecule .smi directory for a folmsbee split.

    Used by analyse_folmsbee_conformers to enumerate molecules when the raw fit
    directories are absent (reproducing from committed force fields). Resolves the
    relevant checkpoint so the directory exists before the analysis runs.
    """
    dataset_type = wildcards.dataset_type
    if dataset_type == "test":
        checkpoints.process_folmsbee_smiles.get()
    else:
        checkpoints.subset_folmsbee_smiles.get(dataset_type=dataset_type)
    return f"benchmarking/folmsbee_conformers/input/{dataset_type}/smiles"


def smiles_descriptor_summary_input(wildcards: Any) -> list[str]:
    """Return smiles descriptor summary dependency for datasets that produce smiles.csv."""
    # In skip_fits mode create_combined_force_field must not rebuild, so give it no
    # inputs that could go stale relative to the committed combined force field.
    if SKIP_FITS or wildcards.dataset == "folmsbee_conformers":
        return []

    return [
        f"benchmarking/{wildcards.dataset}/input/{wildcards.dataset_type}/smiles_descriptor_summary.csv"
    ]


def qca_exclude_smiles_opts(dataset_name: str) -> str:
    """Return --exclude-smiles CLI flags for a QCA dataset, from workflow_config.yaml."""
    smiles = config.get("exclude_smiles", {}).get(dataset_name, [])
    return " ".join(f"--exclude-smiles '{s}'" for s in smiles)


def qca_include_ids_opts(dataset_name: str) -> str:
    """Return --qcarchive-id CLI flags for a QCA dataset, from workflow_config.yaml."""
    include_ids = config.get("include_qcarchive_ids", {}).get(dataset_name, [])
    return " ".join(f"--qcarchive-id {record_id}" for record_id in include_ids)


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


def build_torsion_plot_opts(target_config: dict[str, Any]) -> str:
    """Build --plot-torsion-id CLI flags from a yammbs target config section."""
    torsion_plot_ids: dict = target_config.get("torsion_plot_ids", {})
    base_ffs: list[str] = target_config.get("torsion_plot_base_force_fields", [])
    parts = []
    for tid, extra_ffs in torsion_plot_ids.items():
        all_ffs = base_ffs + (extra_ffs or [])
        if all_ffs:
            parts.append(f"--plot-torsion-id {tid}:{','.join(all_ffs)}")
        else:
            parts.append(f"--plot-torsion-id {tid}")
    return " ".join(parts)


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


# Results that require the raw per-molecule PRESTO fits (loss curves, per-atom
# energies, fit trajectories), which are not committed. Dropped from rule all when
# running with --config skip_fits=True; their committed summary outputs are used
# instead. Recomputing them needs the full fitting stage.
FIT_DEPENDENT_TARGETS = [
    # Congeneric series type specificity (Sec. "presto allows simultaneous training...")
    "benchmarking/tyk2_congeneric_series/analysis/retrain_error_summary.csv",
    # SI presto validation-set per-atom energy RMSEs
    "benchmarking/analysis/presto_fit_validation/presto_fit_validation_error_aggregate.csv",
    "benchmarking/analysis/presto_fit_validation/presto_fit_validation_error_aggregate.tex",
    # The fit robustness study is not here: like the speed comparison it needs one
    # dedicated GPU workstation (`pixi run snakemake-robustness`; see README).
]


rule all:
    # Default target: builds exactly the results reported in presto_paper.tex.
    # Each entry is an analysis endpoint; the force fields, fits, and downloads it
    # depends on are pulled in transitively. Targets are grouped by paper section so
    # this list doubles as an index (see README.md for the full section -> target map).
    # Entries here are reproducible from the committed combined force fields (no raw
    # fits); the fit-dependent results live in FIT_DEPENDENT_TARGETS above.
    input:
        # ===================== Main text =====================
        # TorsionNet500 test set (Sec. "presto reduces torsion scan RMSE...", Table 1, Fig. 2)
        "benchmarking/tnet500/analysis/test/default/metrics.json",
        # Workflow component ablations (Sec. "Metadynamics, minimised samples...", Figs. 3-4)
        "benchmarking/tnet500/analysis/validation/ablations/metrics.json",
        # JACS fragment torsion scans (Sec. "presto produces similar results to QM...", Table 2)
        "benchmarking/jacs_fragments/analysis/test/default/metrics.json",
        # JACS fragment torsion scans split by overall molecular charge
        "benchmarking/jacs_fragments/analysis/test/default/plots_neutral",
        "benchmarking/jacs_fragments/analysis/test/default/plots_charged",
        # Folmsbee relative conformer energies (Sec. "presto improves relative conformer
        # energies...", Fig. 5, Table 3); AIMNet2 reference, config "aimnet2"
        "benchmarking/folmsbee_conformers/analysis/test/aimnet2/aggregate_stats.csv",
        # Relative binding free energies (Sec. "Free Energy Calculations", Fig. 6, Table 4)
        "benchmarking/rbfe_sandbox/results/bootstrap_statistics.csv",
        "benchmarking/rbfe_sandbox/results/panel_ddg.png",

        # ================= Supporting information =================
        # Dataset descriptor summary statistics
        "benchmarking/analysis/smiles_descriptors/smiles_descriptor_aggregate_mean_std.csv",
        "benchmarking/analysis/smiles_descriptors/smiles_descriptor_aggregate_mean_std.tex",
        # TorsionNet500 results with B3LYP-D3(BJ)/DZVP reference (reoptimised v4)
        "benchmarking/tnet500_reopt_v4/analysis/test/default/metrics.json",
        "benchmarking/tnet500_reopt_v4/analysis/validation/ablations/metrics.json",
        # Folmsbee phosphorus compounds without MSM or MLP minimisation, and
        # sulphonamides without MSM but with MLP minimisation
        "benchmarking/folmsbee_conformers/analysis/phosphate/aimnet2_no_msm_no_min/aggregate_stats.csv",
        "benchmarking/folmsbee_conformers/analysis/sulphonamide/aimnet2_no_msm/aggregate_stats.csv",
        # TYK2 cyclopropanecarboxamide edge torsion scans
        "benchmarking/tyk2_cyclopropyl_edges_torsions/analysis/metrics.json",
        # Results needing the raw fits (dropped with --config skip_fits=True)
        *([] if SKIP_FITS else FIT_DEPENDENT_TARGETS),


############ General Rules #############

rule run_presto:
    input:
        smiles_file="benchmarking/{dataset}/input/{dataset_type}/smiles/{molecule}.smi",
        config_file="configs/{config_name}.yaml",
    output:
        "benchmarking/{dataset}/output/{dataset_type}/{config_name}/{molecule}/bespoke_force_field.offxml",
    params:
        pixi_environment=get_pixi_env,
    threads: 32 # So that only one job at once runs on my workstation...
    resources:
        mem_mb=8000,
        runtime=120,  # minutes
        slurm_partition="gpu-s_free",
        slurm_extra="--gpus-per-task=1",
    shell:
        "pixi run -e {params.pixi_environment} presto-benchmark run-presto {input.config_file} {input.smiles_file} $(dirname {output[0]})"

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
            f"configs/{config['tyk2_congeneric_series'].get('config_name', 'one_it_aimnet2_no_msm')}.yaml"
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
        loss_plot_png="benchmarking/tyk2_congeneric_series/analysis/loss_vs_max_extend_distance.png",
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


rule create_1mer_backbone_joint_fit_smiles:
    input:
        smiles_dir=lambda wc: checkpoints.split_test_only_input.get(
            dataset="1mer_backbone"
        ).output.test_set_smiles,
    output:
        "benchmarking/1mer_backbone_joint_fit/input/1mer_backbone.smi",
    run:
        smiles_files = sorted(Path(input.smiles_dir).glob("*.smi"))
        if not smiles_files:
            raise ValueError(
                f"No .smi files found in {input.smiles_dir}; cannot create joint-fit input"
            )

        output_path = Path(output[0])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as handle:
            for smiles_file in smiles_files:
                smiles = smiles_file.read_text().strip()
                if not smiles:
                    raise ValueError(f"Empty SMILES file encountered: {smiles_file}")
                handle.write(f"{smiles}\n")


rule run_1mer_backbone_joint_fit:
    input:
        smiles_file=rules.create_1mer_backbone_joint_fit_smiles.output[0],
        config_file="configs/{config_name}.yaml",
    output:
        "benchmarking/1mer_backbone_joint_fit/output/{config_name}/training_iteration_1/bespoke_ff.offxml",
    resources:
        mem_mb=8000,
        runtime=240,  # minutes
        slurm_partition="gpu-s_free",
        slurm_extra="--gpus-per-task=1",
    shell:
        "pixi run -e default presto-benchmark run-presto "
        "{input.config_file} {input.smiles_file} benchmarking/1mer_backbone_joint_fit/output/{wildcards.config_name}"


rule create_1mer_backbone_joint_fit_force_field:
    input:
        joint_fit_offxml=rules.run_1mer_backbone_joint_fit.output[0],
    output:
        "benchmarking/1mer_backbone_joint_fit/output/{config_name}/combined_force_field.offxml",
    shell:
        "cp {input.joint_fit_offxml} {output[0]}"


rule run_1mer_protein_joint_fit_minimisation:
    input:
        qca_data_json="benchmarking/{dataset}/input/qca_data.json",
        combined_ff="benchmarking/1mer_backbone_joint_fit/output/{config_name}/combined_force_field.offxml",
    output:
        directory("benchmarking/{dataset}/analysis_joint_fit/test/{config_name}/minimised"),
    wildcard_constraints:
        dataset="1mer_backbone|1mer_side_chain",
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


rule plot_1mer_protein_joint_fit_analysis:
    input:
        minimised_dir="benchmarking/{dataset}/analysis_joint_fit/test/{config_name}/minimised",
        qca_names_json="benchmarking/{dataset}/input/qca_names.json",
    output:
        directory("benchmarking/{dataset}/analysis_joint_fit/test/{config_name}/plots"),
    wildcard_constraints:
        dataset="1mer_backbone|1mer_side_chain",
    shell:
        "pixi run -e default presto-benchmark plot-protein-torsion {input.minimised_dir} {output[0]} "
        "--names-file {input.qca_names_json}"


rule benchmark_1mer_protein_joint_fit:
    input:
        backbone_plots="benchmarking/1mer_backbone/analysis_joint_fit/test/{config_name}/plots",
        side_chain_plots="benchmarking/1mer_side_chain/analysis_joint_fit/test/{config_name}/plots",
    output:
        touch("benchmarking/1mer_backbone_joint_fit/analysis/{config_name}/benchmark_complete.txt"),
    shell:
        "mkdir -p $(dirname {output[0]}) && touch {output[0]}"


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
        dataset="tnet500|jacs_fragments|phosphate_torsion_drives|1mer_backbone|3mer_backbone|1mer_side_chain|folmsbee_conformers",
        dataset_type="test|validation",
    shell:
        "pixi run -e default presto-benchmark analyse-smiles-descriptors {input.smiles_csv}"


rule aggregate_smiles_descriptors:
    input:
        tnet500_test="benchmarking/tnet500/input/test/smiles_descriptor_summary.csv",
        tnet500_validation="benchmarking/tnet500/input/validation/smiles_descriptor_summary.csv",
        jacs_test="benchmarking/jacs_fragments/input/test/smiles_descriptor_summary.csv",
        folmsbee_test="benchmarking/folmsbee_conformers/input/test/smiles_descriptor_summary.csv",
        smiles_csvs=[
            "benchmarking/tnet500/input/test/smiles.csv",
            "benchmarking/tnet500/input/validation/smiles.csv",
            "benchmarking/jacs_fragments/input/test/smiles.csv",
            "benchmarking/folmsbee_conformers/input/test/smiles.csv",
        ],
    output:
        aggregate_csv="benchmarking/analysis/smiles_descriptors/smiles_descriptor_aggregate_mean_std.csv",
        aggregate_tex="benchmarking/analysis/smiles_descriptors/smiles_descriptor_aggregate_mean_std.tex",
    params:
        output_dir="benchmarking/analysis/smiles_descriptors",
    shell:
        "pixi run -e default presto-benchmark aggregate-smiles-descriptors {params.output_dir} "
        "{input.smiles_csvs}"


rule analyse_presto_fits:
    """Per-atom-energy validation RMSE analysis of the raw PRESTO fits.

    Split out from the torsion/conformer analysis rules so that those (which need
    only the combined force field) can run without the raw per-molecule fit outputs
    this rule requires.
    """
    input:
        combined_ff=presto_fit_combined_ff,
    output:
        summary_csv="benchmarking/{dataset}/analysis/{dataset_type}/{config_name}/presto_fit_validation/presto_fit_validation_energy_rmse_summary.csv",
        summary_tex="benchmarking/{dataset}/analysis/{dataset_type}/{config_name}/presto_fit_validation/presto_fit_validation_energy_rmse_summary.tex",
        plot_png="benchmarking/{dataset}/analysis/{dataset_type}/{config_name}/presto_fit_validation/presto_fit_validation_energy_rmse.png",
    wildcard_constraints:
        dataset="tnet500|tnet500_reopt_v4|jacs_fragments|phosphate_torsion_drives|folmsbee_conformers",
    params:
        pixi_env=presto_fit_pixi_env,
        presto_output_dir=presto_fit_source_dir,
        output_dir=lambda wc: f"benchmarking/{wc.dataset}/analysis/{wc.dataset_type}/{wc.config_name}/presto_fit_validation",
    shell:
        "pixi run -e {params.pixi_env} presto-benchmark analyse-presto-fits "
        "{params.presto_output_dir} {params.output_dir} --random-seed {RANDOM_SEED}"


rule aggregate_validation_fit_errors:
    input:
        tnet500_test="benchmarking/tnet500/analysis/test/default/presto_fit_validation/presto_fit_validation_energy_rmse_summary.csv",
        jacs_test="benchmarking/jacs_fragments/analysis/test/default/presto_fit_validation/presto_fit_validation_energy_rmse_summary.csv",
        folmsbee_test="benchmarking/folmsbee_conformers/analysis/test/aimnet2/presto_fit_validation/presto_fit_validation_energy_rmse_summary.csv",
        summary_csvs=[
            "benchmarking/tnet500/analysis/test/default/presto_fit_validation/presto_fit_validation_energy_rmse_summary.csv",
            "benchmarking/jacs_fragments/analysis/test/default/presto_fit_validation/presto_fit_validation_energy_rmse_summary.csv",
            "benchmarking/folmsbee_conformers/analysis/test/aimnet2/presto_fit_validation/presto_fit_validation_energy_rmse_summary.csv",
        ],
    output:
        aggregate_csv="benchmarking/analysis/presto_fit_validation/presto_fit_validation_error_aggregate.csv",
        aggregate_tex="benchmarking/analysis/presto_fit_validation/presto_fit_validation_error_aggregate.tex",
    params:
        output_dir="benchmarking/analysis/presto_fit_validation",
    shell:
        "pixi run -e default presto-benchmark aggregate-validation-fit-errors {params.output_dir} "
        "{input.summary_csvs}"

rule analyse_torsion_scans_yammbs:
    input:
        qca_data_json="benchmarking/{dataset}/input/{dataset_type}/{dataset_type}.json",
        combined_ff="benchmarking/{dataset}/output/{dataset_type}/{config_name}/combined_force_field.offxml",
    output:
        metrics_json="benchmarking/{dataset}/analysis/{dataset_type}/{config_name}/metrics.json",
        minimized_json="benchmarking/{dataset}/analysis/{dataset_type}/{config_name}/minimized.json",
        database_file="benchmarking/{dataset}/analysis/{dataset_type}/{config_name}/torsion-data.sqlite",
        plot_png="benchmarking/{dataset}/analysis/{dataset_type}/{config_name}/plots/rmse.png",
        paired_stats_png="benchmarking/{dataset}/analysis/{dataset_type}/{config_name}/plots/paired_stats.png",
        paired_stats_no_sig_png="benchmarking/{dataset}/analysis/{dataset_type}/{config_name}/plots/paired_stats_no_sig.png",
    wildcard_constraints:
        dataset="tnet500|tnet500_reopt_v4|jacs_fragments|phosphate_torsion_drives",
    params:
        pixi_env = get_pixi_env,
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
        torsion_plot_id_opts=lambda wc: build_torsion_plot_opts(yammbs_target_config(wc)),
    resources:
        exclusive=1,
    shell:
        "pixi run -e {params.pixi_env} presto-benchmark analyse-torsion-scans "
        "{input.qca_data_json} {input.combined_ff} {params.analysis_dir} "
        "{params.base_ff_opts} {params.extra_ff_opts} {params.torsion_plot_id_opts}"


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


checkpoint subset_folmsbee_smiles:
    input:
        gh_repo=rules.get_folmsbee_conformer_input.output[0],
        smiles_dir=rules.process_folmsbee_smiles.output[0],
    output:
        directory("benchmarking/folmsbee_conformers/input/{dataset_type}/smiles")
    params:
        reference_method=lambda wc: config["folmsbee_analysis"]["reference_method"],
        min_reference_energy_window=lambda wc: folmsbee_target_config(wc).get(
            "min_reference_energy_window",
            config["folmsbee_analysis"].get("min_reference_energy_window", 0.0),
        ),
        max_molecules=lambda wc: folmsbee_target_config(wc).get("subset_max_molecules", 100),
        selection_strategy=lambda wc: folmsbee_target_config(wc).get(
            "subset_selection_strategy", "random"
        ),
        seed=lambda wc: folmsbee_target_config(wc).get("subset_seed", config["random_seed"]),
        exclude_smarts_opts=lambda wc: " ".join(
            f"--exclude-smarts '{smarts}'"
            for smarts in folmsbee_target_config(wc).get(
                "exclude_smarts",
                config["folmsbee_analysis"].get("exclude_smarts", []),
            )
        ),
        include_smarts_opts=lambda wc: " ".join(
            f"--include-smarts '{smarts}'"
            for smarts in folmsbee_target_config(wc).get("include_smarts", [])
        ),
    shell:
        "pixi run -e default presto-benchmark subset-folmsbee-smiles "
        "{input.gh_repo} {input.smiles_dir} {output} "
        "--reference-method '{params.reference_method}' "
        "--min-reference-energy-window {params.min_reference_energy_window} "
        "--max-molecules {params.max_molecules} "
        "--selection-strategy {params.selection_strategy} "
        "--seed {params.seed} "
        "{params.exclude_smarts_opts} "
        "{params.include_smarts_opts}"


rule create_folmsbee_smiles_csv:
    input:
        smiles_dir=rules.process_folmsbee_smiles.output[0]
    output:
        "benchmarking/folmsbee_conformers/input/test/smiles.csv"
    run:
        smiles_files = sorted(Path(input.smiles_dir).glob("*.smi"))
        if not smiles_files:
            raise ValueError(
                f"No .smi files found in {input.smiles_dir}; cannot create smiles.csv"
            )

        output_path = Path(output[0])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as handle:
            handle.write("smiles\n")
            for smiles_file in smiles_files:
                smiles = smiles_file.read_text().strip()
                if not smiles:
                    raise ValueError(f"Empty SMILES file encountered: {smiles_file}")
                handle.write(f"{smiles}\n")


rule analyse_folmsbee_conformers:
    input:
        gh_repo=rules.get_folmsbee_conformer_input.output[0],
        combined_ff="benchmarking/folmsbee_conformers/output/{dataset_type}/{config_name}/combined_force_field.offxml",
        smiles_dir=folmsbee_smiles_dir,
    output:
        results_csv="benchmarking/folmsbee_conformers/analysis/{dataset_type}/{config_name}/results.csv",
        per_molecule_stats_csv="benchmarking/folmsbee_conformers/analysis/{dataset_type}/{config_name}/per_molecule_stats.csv",
        aggregate_stats_csv="benchmarking/folmsbee_conformers/analysis/{dataset_type}/{config_name}/aggregate_stats.csv",
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
            for smarts in folmsbee_target_config(wc).get(
                "exclude_smarts",
                config["folmsbee_analysis"].get("exclude_smarts", []),
            )
        ),
        min_conformers_per_molecule=lambda wc: config["folmsbee_analysis"].get(
            "min_conformers_per_molecule", 5
        ),
        min_reference_energy_window=lambda wc: folmsbee_target_config(wc).get(
            "min_reference_energy_window",
            config["folmsbee_analysis"].get("min_reference_energy_window", 0.0),
        ),
        n_processes_opt=lambda wc: (
            f"--n-processes {config['folmsbee_analysis']['n_processes']}"
            if config["folmsbee_analysis"].get("n_processes") is not None
            else ""
        ),
    resources:
        exclusive=1,
    shell:
        "pixi run -e default presto-benchmark analyse-folmsbee "
        "{input.gh_repo} {params.presto_output_dir} {params.analysis_dir} "
        "--smiles-dir {input.smiles_dir} "
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
        "{params.n_processes_opt}"


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
        validation_set_smiles_csv="benchmarking/tnet500/input/validation/smiles.csv",
        test_set_dir=directory("benchmarking/tnet500/input/test"),
        test_set_json="benchmarking/tnet500/input/test/test.json",
        test_set_smiles=directory("benchmarking/tnet500/input/test/smiles"),
        test_set_smiles_csv="benchmarking/tnet500/input/test/smiles.csv",
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
        torsion_plot_id_opts=build_torsion_plot_opts(
            config["yammbs_analysis"]["targets"]["tnet500"]["validation"]
        ),
    resources:
        exclusive=1,
    shell:
        "pixi run -e default presto-benchmark analyse-torsion-scans "
        "{input.qca_data_json} {input.default_ff} {params.analysis_dir} "
        "{params.base_ff_opts} "
        "--extra-force-field '{input.no_reg_ff}' "
        "--extra-force-field '{input.no_min_ff}' "
        "--extra-force-field '{input.one_it_ff}' "
        "--extra-force-field '{input.no_metad_ff}' "
        "{params.torsion_plot_id_opts} && "
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
    params:
        analysis_dir="benchmarking/tnet500_reopt_v4/analysis/test/default",
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
    resources:
        exclusive=1,
    shell:
        "pixi run -e default presto-benchmark analyse-torsion-scans "
        "{input.qca_data_json} {input.combined_ff} {params.analysis_dir} "
        "{params.base_ff_opts} {params.extra_ff_opts}"


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
    params:
        analysis_dir="benchmarking/tnet500_reopt_v4/analysis/validation/ablations",
        base_ff_opts=" ".join(
            f"--base-force-field '{ff}'"
            for ff in config["yammbs_analysis"]["base_force_fields"]
        ),
        torsion_plot_id_opts=build_torsion_plot_opts(
            config["yammbs_analysis"]["targets"]["tnet500"]["validation"]
        ),
    resources:
        exclusive=1,
    shell:
        "pixi run -e default presto-benchmark analyse-torsion-scans "
        "{input.qca_data_json} {input.default_ff} {params.analysis_dir} "
        "{params.base_ff_opts} "
        "--extra-force-field '{input.no_reg_ff}' "
        "--extra-force-field '{input.no_min_ff}' "
        "--extra-force-field '{input.one_it_ff}' "
        "--extra-force-field '{input.no_metad_ff}' &&"
        "pixi run -e default presto-benchmark plot-ablation-comparison "
        "{output.metrics_json} {params.analysis_dir}/plots"


############ JACS Fragments #############

rule split_torsion_scans_by_charge:
    """Re-plot an existing torsion analysis split into overall neutral/charged molecules.

    Reuses the torsion-data.sqlite and metrics.json produced by
    analyse_torsion_scans_yammbs and writes plots_neutral/ and plots_charged/
    alongside them, to help localise where errors come from.
    """
    input:
        database_file="benchmarking/{dataset}/analysis/{dataset_type}/{config_name}/torsion-data.sqlite",
        metrics_json="benchmarking/{dataset}/analysis/{dataset_type}/{config_name}/metrics.json",
    output:
        neutral_dir=directory("benchmarking/{dataset}/analysis/{dataset_type}/{config_name}/plots_neutral"),
        charged_dir=directory("benchmarking/{dataset}/analysis/{dataset_type}/{config_name}/plots_charged"),
    wildcard_constraints:
        dataset="tnet500|tnet500_reopt_v4|jacs_fragments|phosphate_torsion_drives",
    params:
        pixi_env=get_pixi_env,
    shell:
        "pixi run -e {params.pixi_env} presto-benchmark plot-torsion-scans-by-charge "
        "{input.database_file} {input.metrics_json} {output.neutral_dir} {output.charged_dir}"


rule get_qca_torsion_input_dataset:
    output:
        "benchmarking/{dataset}/input/{dataset}.json"
    wildcard_constraints:
        dataset="|".join(QCA_DATASET_NAMES.keys()),
    params:
        qca_dataset_name=lambda wc: QCA_DATASET_NAMES[wc.dataset],
        exclude_opts=lambda wc: qca_exclude_smiles_opts(QCA_DATASET_NAMES[wc.dataset]),
        include_id_opts=lambda wc: qca_include_ids_opts(QCA_DATASET_NAMES[wc.dataset]),
    shell:
        "pixi run -e default presto-benchmark get-qca-torsion-input "
        "'{params.qca_dataset_name}' {output[0]} {params.exclude_opts} {params.include_id_opts}"


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
        torsion_plot_id_opts=build_torsion_plot_opts(
            config["yammbs_analysis"]["targets"]["jacs_fragments_full_mol_fits"]["test"]
        ),
    resources:
        exclusive=1,
    shell:
        "pixi run -e no-openeye presto-benchmark analyse-torsion-scans "
        "{input.qca_data_json} {input.force_field} {params.analysis_dir} "
        "{params.base_ff_opts} {params.extra_ff_opts} {params.torsion_plot_id_opts}"




rule get_tyk2_cyclopropyl_edges_torsions_input:
    output:
        "benchmarking/tyk2_cyclopropyl_edges_torsions/input/tyk2_cyclopropyl_edges_torsions.json"
    params:
        qca_dataset_name=config["tyk2_cyclopropyl_edges_torsions"]["qca_dataset"],
        include_id_opts=" ".join(
            f"--qcarchive-id {rid}"
            for rid in config["tyk2_cyclopropyl_edges_torsions"]["qcarchive_ids"]
        ),
        exclude_opts=qca_exclude_smiles_opts(
            config["tyk2_cyclopropyl_edges_torsions"]["qca_dataset"]
        ),
    shell:
        "pixi run -e default presto-benchmark get-qca-torsion-input "
        "'{params.qca_dataset_name}' {output[0]} {params.exclude_opts} {params.include_id_opts}"


rule analyse_tyk2_cyclopropyl_edges_torsions:
    input:
        qca_data_json=rules.get_tyk2_cyclopropyl_edges_torsions_input.output[0],
        primary_ff=config["tyk2_cyclopropyl_edges_torsions"]["force_fields"][0],
        extra_ffs=config["tyk2_cyclopropyl_edges_torsions"]["force_fields"][1:],
    output:
        metrics_json="benchmarking/tyk2_cyclopropyl_edges_torsions/analysis/metrics.json",
        minimized_json="benchmarking/tyk2_cyclopropyl_edges_torsions/analysis/minimized.json",
        plot_png="benchmarking/tyk2_cyclopropyl_edges_torsions/analysis/plots/rmse.png",
        paired_stats_png="benchmarking/tyk2_cyclopropyl_edges_torsions/analysis/plots/paired_stats.png",
        paired_stats_no_sig_png="benchmarking/tyk2_cyclopropyl_edges_torsions/analysis/plots/paired_stats_no_sig.png",
    params:
        analysis_dir="benchmarking/tyk2_cyclopropyl_edges_torsions/analysis",
        extra_ff_opts=" ".join(
            f"--extra-force-field '{ff}'"
            for ff in config["tyk2_cyclopropyl_edges_torsions"]["force_fields"][1:]
        ),
        draw_cmds=(
            " && " + " && ".join(
                f"pixi run -e no-openeye presto-benchmark draw-molecule "
                f"'{smiles}' benchmarking/tyk2_cyclopropyl_edges_torsions/analysis/plots/{name}.png"
                for name, smiles in config["tyk2_cyclopropyl_edges_torsions"]
                .get("reference_fragments", {})
                .items()
            )
            if config["tyk2_cyclopropyl_edges_torsions"].get("reference_fragments")
            else ""
        ),
    resources:
        exclusive=1,
    shell:
        "pixi run -e no-openeye presto-benchmark analyse-torsion-scans "
        "{input.qca_data_json} {input.primary_ff} {params.analysis_dir} "
        "{params.extra_ff_opts} --plot-all-torsions"
        "{params.draw_cmds}"


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


rule analyse_rbfe:
    input:
        raw_data="benchmarking/rbfe_sandbox/raw_data",
        ligands="benchmarking/rbfe_sandbox/ligands",
    output:
        "benchmarking/rbfe_sandbox/results/bootstrap_statistics.csv",
        "benchmarking/rbfe_sandbox/results/statistical_tests.csv",
        "benchmarking/rbfe_sandbox/results/panel_dg.png",
        "benchmarking/rbfe_sandbox/results/panel_ddg.png",
    shell:
        "pixi run -e default presto-benchmark analyse-rbfe "
        "{input.raw_data} {input.ligands} benchmarking/rbfe_sandbox/results"


############ Speed comparison: presto vs OpenFF BespokeFit 0.4.4 #############
# Every rule uses all 32 threads so that, with --cores 32, nothing runs alongside a
# timed job. Run with `pixi run snakemake-speed`. run_00 is a warm-up (fills weight
# and kernel caches) that later repeats wait for, and is excluded from statistics.

SPEED = config["speed_comparison"]
SPEED_DIR = "benchmarking/speed_comparison"
BESPOKEFIT_ENVS = {
    "aimnet2_matched": "bespokefit-ml",
    "aimnet2_stock": "bespokefit-ml",
    "qm": "bespokefit-qm",
}


def speed_repeats(method, warmup):
    first = 0 if warmup else 1
    return [f"{i:02d}" for i in range(first, SPEED["repeats"][method] + 1)]


rule create_speed_comparison_smiles:
    output:
        f"{SPEED_DIR}/input/ligand.smi",
    run:
        Path(output[0]).write_text(SPEED["smiles"] + "\n")


rule time_presto:
    input:
        smiles=rules.create_speed_comparison_smiles.output[0],
        config_file=f"configs/{SPEED['presto_config']}.yaml",
        warmup=lambda wc: [] if wc.repeat == "00" else f"{SPEED_DIR}/presto/run_00/timings.json",
    output:
        timings=f"{SPEED_DIR}/presto/run_{{repeat}}/timings.json",
        # Whole-process wall time, measured as for time_presto_uninstrumented
        wall_time=f"{SPEED_DIR}/presto/run_{{repeat}}/wall_time.json",
    threads: 32
    shell:
        "PRESTO_TIMING_T0=$(date +%s.%N) /usr/bin/time -f '{{\"wall_s\": %e}}' "
        "-o {output.wall_time} pixi run --frozen -e default presto-benchmark "
        "run-presto-timed {input.config_file} {input.smiles} "
        "$(dirname {output.timings})/fit {output.timings}"


rule time_presto_uninstrumented:
    input:
        smiles=rules.create_speed_comparison_smiles.output[0],
        config_file=f"configs/{SPEED['presto_config']}.yaml",
        warmup=f"{SPEED_DIR}/presto/run_00/timings.json",
    output:
        f"{SPEED_DIR}/presto_uninstrumented/run_{{repeat}}/wall_time.json",
    threads: 32
    shell:
        "/usr/bin/time -f '{{\"wall_s\": %e}}' -o {output} pixi run --frozen -e default "
        "presto-benchmark run-presto {input.config_file} {input.smiles} $(dirname {output})/fit"


rule time_bespokefit:
    input:
        smiles=rules.create_speed_comparison_smiles.output[0],
        warmup=lambda wc: (
            []
            if wc.variant == "qm" or wc.repeat == "00"
            else f"{SPEED_DIR}/bespokefit_{wc.variant}/run_00/timings.json"
        ),
    output:
        f"{SPEED_DIR}/bespokefit_{{variant}}/run_{{repeat}}/timings.json",
    wildcard_constraints:
        variant="|".join(BESPOKEFIT_ENVS),
    params:
        pixi_environment=lambda wc: BESPOKEFIT_ENVS[wc.variant],
    threads: 32
    shell:
        "BESPOKE_TIMING_T0=$(date +%s.%N) pixi run --frozen -e {params.pixi_environment} "
        "python convenience_functions/standalone/run_bespokefit_timed.py "
        "\"$(cat {input.smiles})\" {wildcards.variant} $(dirname {output})"


rule aimnet2_parity:
    """Same model and speed? AIMNet2 via presto (GPU and CPU) vs via QCEngine."""
    output:
        conformers=f"{SPEED_DIR}/aimnet2_parity_conformers.sdf",
        arms=expand(
            f"{SPEED_DIR}/aimnet2_parity/{{arm}}.json",
            arm=["presto_cuda", "presto_cpu", "qcengine_openmm-ml", "qcengine_aimnet2"],
        ),
    params:
        n_conformers=SPEED["aimnet2_parity"]["n_conformers"],
        n_calls=SPEED["aimnet2_parity"]["n_calls_per_conformer"],
        out_dir=f"{SPEED_DIR}/aimnet2_parity",
    threads: 32
    shell:
        "pixi run --frozen -e default presto-benchmark aimnet2-parity-presto "
        "\"{SPEED[smiles]}\" {output.conformers} {params.out_dir}/presto_cuda.json "
        "--n-conformers {params.n_conformers} --n-calls-per-conformer {params.n_calls} && "
        "CUDA_VISIBLE_DEVICES='' pixi run --frozen -e default presto-benchmark "
        "aimnet2-parity-presto \"{SPEED[smiles]}\" {output.conformers} "
        "{params.out_dir}/presto_cpu.json --n-conformers {params.n_conformers} "
        "--n-calls-per-conformer {params.n_calls} && "
        "for program in openmm-ml aimnet2; do "
        "pixi run --frozen -e bespokefit-ml python "
        "convenience_functions/standalone/aimnet2_parity_qcengine.py {output.conformers} "
        "$program {params.n_calls} {params.out_dir}/qcengine_$program.json; done"


rule analyse_speed_comparison:
    input:
        presto=expand(
            f"{SPEED_DIR}/presto/run_{{repeat}}/timings.json",
            repeat=speed_repeats("presto", warmup=True),
        ),
        presto_uninstrumented=expand(
            f"{SPEED_DIR}/presto_uninstrumented/run_{{repeat}}/wall_time.json",
            repeat=speed_repeats("presto_uninstrumented", warmup=False),
        ),
        bespokefit=[
            f"{SPEED_DIR}/bespokefit_{variant}/run_{repeat}/timings.json"
            for variant in BESPOKEFIT_ENVS
            for repeat in speed_repeats(f"bespokefit_{variant}", warmup=variant != "qm")
        ],
        parity=rules.aimnet2_parity.output.arms,
    output:
        f"{SPEED_DIR}/analysis/summary.csv",
        f"{SPEED_DIR}/analysis/summary.tex",
        f"{SPEED_DIR}/analysis/presto_stage_breakdown.csv",
        f"{SPEED_DIR}/analysis/presto_stage_breakdown.tex",
        f"{SPEED_DIR}/analysis/aimnet2_parity.csv",
        f"{SPEED_DIR}/analysis/phases.png",
    shell:
        "pixi run --frozen -e default presto-benchmark analyse-speed-comparison {SPEED_DIR}"


############ Fit robustness: repeats x regularisation, with timings #############
# Repeat presto fits of a few diverse molecules at three torsion-regularisation
# strengths, scoring each fit against the molecule's own QM data and timing it.
# Run with `pixi run snakemake-robustness` (see pyproject.toml).
# run_00 is a warm-up fit excluded from all statistics.

ROB = config["fit_robustness"]
ROB_DIR = ROB["dir"]
# Driver override: analyse only finished fits, so a failed fit doesn't block the summary.
ROB_EXISTING_ONLY = str(config.get("robustness_existing_only", "")).lower() in ("true", "1")


def rob_run_dir(molecule, reg, repeat):
    return f"{ROB_DIR}/output/{molecule}/{reg}/run_{repeat:02d}"


def rob_runs(molecule, reg):
    """Run dirs (excluding the run_00 warm-up) that the analyses expect."""
    runs = [rob_run_dir(molecule, reg, i) for i in range(1, ROB["repeats"][reg] + 1)]
    if ROB_EXISTING_ONLY:
        runs = [r for r in runs if Path(f"{r}/bespoke_force_field.offxml").exists()]
    return runs


def rob_ffs(molecule, regs=None):
    return [
        f"{run}/bespoke_force_field.offxml"
        for reg in (regs or ROB["configs"])
        for run in rob_runs(molecule, reg)
    ]


def rob_molecules(qm):
    return [name for name, mol in ROB["molecules"].items() if mol["qm"] == qm]


wildcard_constraints:
    rob_molecule="|".join(ROB["molecules"]),
    rob_reg="|".join(ROB["configs"]),
    rob_repeat=r"\d\d",


rule create_robustness_smiles:
    output:
        f"{ROB_DIR}/input/{{rob_molecule}}.smi",
    run:
        Path(output[0]).write_text(ROB["molecules"][wildcards.rob_molecule]["smiles"] + "\n")


# run_presto's generic benchmarking/{dataset}/output/... pattern also matches these paths
ruleorder: fit_robustness > run_presto


rule fit_robustness:
    input:
        smiles=rules.create_robustness_smiles.output[0],
        config_file=lambda wc: ROB["configs"][wc.rob_reg],
    # protected: finished fits are read-only and snakemake refuses to redo them (the shell's
    # rm -rf would otherwise wipe a finished fit if any rerun trigger fires).
    output:
        ff=protected(f"{ROB_DIR}/output/{{rob_molecule}}/{{rob_reg}}/run_{{rob_repeat}}/bespoke_force_field.offxml"),
        timings=protected(f"{ROB_DIR}/output/{{rob_molecule}}/{{rob_reg}}/run_{{rob_repeat}}/timings.json"),
        wall_time=protected(f"{ROB_DIR}/output/{{rob_molecule}}/{{rob_reg}}/run_{{rob_repeat}}/wall_time.json"),
    params:
        run_dir=lambda wc, output: Path(output.ff).parent,
    threads: workflow.cores  # claim every core: one fit at a time, so timings are clean
    shell:
        # Clear leftovers of a failed attempt: presto writes many undeclared files.
        "rm -rf {params.run_dir} && mkdir -p {params.run_dir} && "
        "PRESTO_TIMING_T0=$(date +%s.%N) /usr/bin/time -f '{{\"wall_s\": %e}}' "
        # timeout: a hung fit (fits take ~20-30 min) would otherwise stall all later fits
        "-o {output.wall_time} timeout 2h pixi run --frozen --offline -e {ROB[pixi_environment]} "
        "presto-benchmark run-presto-timed {input.config_file} {input.smiles} "
        "{params.run_dir} {output.timings} > {params.run_dir}/presto.log 2>&1"


rule robustness_fits:
    """Every fit, plus the warm-up."""
    input:
        f"{rob_run_dir(ROB['warmup_molecule'], 'reg1', 0)}/bespoke_force_field.offxml",
        [ff for molecule in ROB["molecules"] for ff in rob_ffs(molecule)],


rule get_robustness_qca_input:
    output:
        f"{ROB_DIR}/input/{{rob_molecule}}.json",
    params:
        dataset=lambda wc: ROB["molecules"][wc.rob_molecule]["qca_dataset"],
        id_opts=lambda wc: " ".join(
            f"--qcarchive-id {i}" for i in ROB["molecules"][wc.rob_molecule]["qcarchive_ids"]
        ),
    shell:
        "pixi run --frozen --offline -e default presto-benchmark get-qca-torsion-input "
        "'{params.dataset}' {output} {params.id_opts}"


rule analyse_robustness_torsions:
    input:
        qca=rules.get_robustness_qca_input.output[0],
        ffs=lambda wc: rob_ffs(wc.rob_molecule),
    output:
        f"{ROB_DIR}/analysis/{{rob_molecule}}/torsions/metrics.json",
    params:
        out_dir=lambda wc, output: Path(output[0]).parent,
        # The CLI adds its positional force field to the extras itself.
        extra_ff_opts=lambda wc, input: " ".join(
            f"--extra-force-field {ff}" for ff in input.ffs[1:]
        ),
    resources:
        exclusive=1,
    shell:
        # Fresh database: yammbs skips any force field already scored under the same name
        # (its offxml path), so a refit would otherwise silently keep the old fit's scores.
        "rm -f {params.out_dir}/torsion-data.sqlite && "
        "pixi run --frozen --offline -e default presto-benchmark analyse-torsion-scans "
        "{input.qca} {input.ffs[0]} {params.out_dir} "
        "--base-force-field openff-2.3.0 {params.extra_ff_opts}"


rule robustness_torsion_summary_html:
    """yammbs per-torsion HTML report (scan overlays + metrics): Sage and one reg's repeats."""
    input:
        rules.analyse_robustness_torsions.output[0],
    output:
        f"{ROB_DIR}/analysis/{{rob_molecule}}/torsions/summary_{{rob_reg}}.html",
    run:
        from yammbs.torsion import TorsionStore

        store = TorsionStore(str(Path(input[0]).parent / "torsion-data.sqlite"))
        # Every fit the store scored for this reg (force-field names are the offxml paths)
        reg_ffs = [ff for ff in store.get_force_fields() if f"/{wildcards.rob_reg}/" in ff]
        ffs = ["openff-2.3.0", *sorted(reg_ffs)]
        store.get_summary(output[0], force_fields=ffs)


rule robustness_torsion_reports:
    input:
        expand(
            f"{ROB_DIR}/analysis/{{m}}/torsions/summary_{{reg}}.html",
            m=rob_molecules("torsion"),
            reg=ROB["configs"],
        ),


rule create_robustness_folmsbee_smiles:
    """Folmsbee analysis finds conformers by file name, so name it by Folmsbee ID.
    process_folmsbee_smiles writes the conformer .openff.sdf files the analysis loads,
    and a .smi only for molecules whose conformers all built, so copy from there."""
    input:
        processed=rules.process_folmsbee_smiles.output[0],
    output:
        smi_dir=directory(f"{ROB_DIR}/input/folmsbee_{{rob_molecule}}"),
        empty_dir=directory(f"{ROB_DIR}/input/folmsbee_{{rob_molecule}}_no_fits"),
    run:
        smi = f"{ROB['molecules'][wildcards.rob_molecule]['folmsbee_id']}.smi"
        Path(output.smi_dir).mkdir(parents=True)
        Path(output.empty_dir).mkdir(parents=True)
        shutil.copy(Path(input.processed) / smi, Path(output.smi_dir) / smi)


rule analyse_robustness_folmsbee:
    input:
        gh_repo=rules.get_folmsbee_conformer_input.output[0],
        smi_dir=rules.create_robustness_folmsbee_smiles.output.smi_dir,
        empty_dir=rules.create_robustness_folmsbee_smiles.output.empty_dir,
        ffs=lambda wc: rob_ffs(wc.rob_molecule),
    output:
        f"{ROB_DIR}/analysis/{{rob_molecule}}/folmsbee/per_molecule_stats.csv",
    params:
        out_dir=lambda wc, output: Path(output[0]).parent,
        ff_opts=lambda wc, input: " ".join(f"--force-field {ff}" for ff in input.ffs),
        fa=config["folmsbee_analysis"],
    resources:
        exclusive=1,
    shell:
        "pixi run --frozen --offline -e default presto-benchmark analyse-folmsbee "
        "{input.gh_repo} {input.empty_dir} {params.out_dir} "
        "--smiles-dir {input.smi_dir} --skip-detailed-analysis "
        "--reference-method {params.fa[reference_method]} "
        "--torsion-restraint-force-constant {params.fa[torsion_restraint_force_constant]} "
        "--mm-minimization-steps {params.fa[mm_minimization_steps]} "
        "--min-conformers-per-molecule {params.fa[min_conformers_per_molecule]} "
        "--force-field input_ff/openff_unconstrained-2.3.0.offxml {params.ff_opts}"


rule analyse_robustness_parameters:
    input:
        ffs=lambda wc: rob_ffs(wc.rob_molecule, [wc.rob_reg]),
    output:
        f"{ROB_DIR}/analysis/{{rob_molecule}}/parameters/{{rob_reg}}/offxml_variability_summary.csv",
    params:
        out_dir=lambda wc, output: Path(output[0]).parent,
        run_opts=lambda wc, input: " ".join(
            f"--run-id {Path(ff).parent.name}" for ff in input.ffs
        ),
    resources:
        exclusive=1,
    shell:
        "pixi run --frozen --offline -e default presto-benchmark analyse-tyk2-reproducibility "
        "{ROB_DIR}/output/{wildcards.rob_molecule}/{wildcards.rob_reg} {params.out_dir} "
        "{params.run_opts}"


rule summarise_fit_robustness:
    input:
        torsions=expand(
            f"{ROB_DIR}/analysis/{{m}}/torsions/metrics.json", m=rob_molecules("torsion")
        ),
        folmsbee=expand(
            f"{ROB_DIR}/analysis/{{m}}/folmsbee/per_molecule_stats.csv",
            m=rob_molecules("folmsbee"),
        ),
        parameters=expand(
            f"{ROB_DIR}/analysis/{{m}}/parameters/{{reg}}/offxml_variability_summary.csv",
            m=ROB["molecules"],
            reg=ROB["configs"],
        ),
        ffs=[ff for molecule in ROB["molecules"] for ff in rob_ffs(molecule)],
    output:
        f"{ROB_DIR}/analysis/summary/qm_accuracy.csv",
        f"{ROB_DIR}/analysis/summary/parameter_spread.csv",
        f"{ROB_DIR}/analysis/summary/mlp_heldout.csv",
        f"{ROB_DIR}/analysis/summary/timings.csv",
    params:
        molecules=json.dumps({k: v["qm"] for k, v in ROB["molecules"].items()}),
    shell:
        "pixi run --frozen --offline -e default presto-benchmark summarise-fit-robustness "
        "{ROB_DIR} '{params.molecules}' {input.ffs}"
