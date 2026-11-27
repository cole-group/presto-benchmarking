# presto-benchmarking
[`presto`](https://github.com/cole-group/presto) is a tool for fitting bespoke SMIRNOFF force fields for your molecule(s) of interest. This repo contains a `Snakemake` workflow that reproduces all of the results in the `presto` paper:

> Finlay Clark, Thomas Pope, Sarah Maier, Simon Boothroyd, Joshua T. Horton, Kevin Ryczko, Andrea Bortolato, and Daniel J. Cole. *Fast training of bespoke SMIRNOFF-format molecular mechanics force fields using machine learning potentials.* ChemRxiv (2025). https://chemrxiv.org/doi/full/10.26434/chemrxiv.15004169/v2

with the exception of the Tango relative binding free energy (RBFE) calculations. Tango is not publicly available, so its precomputed predictions are committed under `benchmarking/rbfe_sandbox/` and the workflow only re-runs the downstream analysis.

To rerun the full workflow (the default target builds results for every section of the paper), [install `pixi`](https://pixi.prefix.dev/latest/installation/), configure Snakemake to use SLURM (see below) and run:
```bash
git clone https://github.com/fjclark/presto-benchmarking.git
cd presto-benchmarking
pixi install --all
pixi run snakemake --cores all
```
Configuring Snamkemake to use SLURM and running on a cluster is crucial if you intend to reproduce the fitting stage, which requires many GPU jobs.

## Reproducing the paper

The default target (`rule all`) builds results for every section of the paper. To rebuild a single section instead of everything, pass its target explicitly:
```bash
pixi run snakemake --cores all benchmarking/tnet500/analysis/test/default/metrics.json
```

| Paper section | Snakemake target |
| --- | --- |
| TorsionNet500 test set | `benchmarking/tnet500/analysis/test/default/metrics.json` |
| Workflow component ablations (on TorsionNet500) | `benchmarking/tnet500/analysis/validation/ablations/metrics.json` |
| JACS fragment torsion scans | `benchmarking/jacs_fragments/analysis/test/default/metrics.json` |
| JACS fragment scans split by molecular charge | `benchmarking/jacs_fragments/analysis/test/default/plots_neutral` and `.../plots_charged` |
| Folmsbee relative conformer energies | `benchmarking/folmsbee_conformers/analysis/test/aimnet2/aggregate_stats.csv` |
| Congeneric series type specificity | `benchmarking/tyk2_congeneric_series/analysis/retrain_error_summary.csv` |
| Relative binding free energies | `benchmarking/rbfe_sandbox/results/bootstrap_statistics.csv` |
| **SI** Dataset descriptor statistics | `benchmarking/analysis/smiles_descriptors/smiles_descriptor_aggregate_mean_std.{csv,tex}` |
| **SI** presto validation per-atom energy RMSEs | `benchmarking/analysis/presto_fit_validation/presto_fit_validation_error_aggregate.{csv,tex}` |
| **SI** TorsionNet500 with B3LYP-D3(BJ)/DZVP reference | `benchmarking/tnet500_reopt_v4/analysis/test/default/metrics.json` and `.../validation/ablations/metrics.json` |
| **SI** Fit robustness: repeats × regularisation strength, with timings (6 molecules; not in `rule all`; see below) | `benchmarking/fit_robustness/analysis/summary/qm_accuracy.csv` |
| **SI** Folmsbee phosphorus compounds rerun (no MSM, no MLP min) | `benchmarking/folmsbee_conformers/analysis/phosphate/aimnet2_no_msm_no_min/aggregate_stats.csv` and `.../summary_metrics_vs_qm_with_presto_vs_aimnet2.tex` |
| **SI** Folmsbee sulfonamides rerun (no MSM, with MLP min) | `benchmarking/folmsbee_conformers/analysis/sulphonamide/aimnet2_no_msm/aggregate_stats.csv` and `.../summary_metrics_vs_qm_with_presto_vs_aimnet2.tex` |
| **SI** TYK2 cyclopropanecarboxamide edge torsion scans | `benchmarking/tyk2_cyclopropyl_edges_torsions/analysis/metrics.json` |
| Speed: presto vs OpenFF BespokeFit (not in `rule all`; see below) | `benchmarking/speed_comparison/analysis/summary.csv` |

### Reproducing the analyses without rerunning the fits

The combined bespoke force fields are committed (see [Committed outputs](#requirements-and-notes)), so you can reproduce the downstream analyses without the expensive many-GPU-job fitting stage — and without SLURM:
```bash
pixi run snakemake-no-refit --resources exclusive=1
```
This sets `--config skip_fits=True`, which makes `create_combined_force_field` accept the committed `combined_force_field.offxml` files as-is (rather than trying to rebuild them from the per-molecule fits, which are not committed). The `--resources exclusive=1` avoids running more than 1 expensive (but non-fitting) jobs at once, which can consume all your RAM. The torsion-scan, charge-split, conformer-energy (Folmsbee), descriptor, and RBFE analyses all regenerate from the committed force fields. Pass `-n` for a dry run, or a specific target to build one section.

The Folmsbee analyses use the raw fits only to list which molecules were fit; with `skip_fits` that molecule set is taken from the committed input SMILES instead, so they reproduce from the combined force field alone (a GPU is still needed for the AIMNet2 single points).

Three results genuinely depend on the raw per-molecule fits (loss curves, per-atom energies, fit trajectories), so `skip_fits` drops them from the default target and leaves their committed summary outputs in place: the **SI presto validation per-atom energy RMSEs**, **SI fit reproducibility** (TYK2), and the **congeneric-series retrain** analysis. Recomputing those requires the full fitting stage.

### Speed comparison: presto vs OpenFF BespokeFit

`pixi run snakemake-speed` times fits of one TYK2 ligand (`workflow_config.yaml: speed_comparison`) with:

| Method | AIMNet2 / QM | Repeats |
| --- | --- | --- |
| `presto` | AIMNet2 via openmm-ml, GPU (`configs/aimnet2_openff221.yaml`) | 5 (+ warm-up, + 2 without timing hooks, to measure their overhead) |
| `bespokefit_aimnet2_matched` | BespokeFit 0.4.4 with AIMNet2 run exactly as in presto (same openmm-ml/aimnet/torch versions and model, GPU) via `convenience_functions/standalone/openmm_ml_harness.py` | 3 (+ warm-up) |
| `bespokefit_aimnet2_stock` | BespokeFit 0.4.4 with QCEngine's own AIMNet2 harness (pyaimnet2 4-model ensemble, fp64, CPU) | 3 (+ warm-up) |
| `bespokefit_qm` | BespokeFit 0.4.4, psi4 B3LYP-D3BJ/DZVP | 1 |

This rule is expensive (the QM especially) and runs everything serially on one machine, so it is not part of `rule all`.
- **presto** is timed stage by stage by wrapping presto's functions (`convenience_functions/presto_timing.py`), without changing what they do.
- **BespokeFit** is timed per executor stage, plus every QC energy/gradient call (`convenience_functions/standalone/run_bespokefit_timed.py`). Stage times are accurate to about 5 s, the coordinator's update interval.

`rule aimnet2_parity` checks that the AIMNet2 evaluations really match between environments, and compares their per-call cost.

Caveats:
- BespokeFit 0.4.4 needs pydantic 1, so it gets its own envs (`bespokefit-ml`, `bespokefit-qm`) and cannot load `openff_unconstrained-2.3.0` (that needs openff-toolkit >= 0.17). Both methods therefore start from `openff_unconstrained-2.2.1`, with OpenEye AM1-BCC charges.
- BespokeFit fits torsions only. presto also fits bonds, angles and impropers.

### Fit robustness: repeats × regularisation

`pixi run snakemake-robustness` (defined in `pyproject.toml`) repeats presto fits (`configs/aimnet2.yaml`, `default-save-ff-trajectory` env, which also saves per-epoch parameters) of six molecules, listed under `fit_robustness` in `workflow_config.yaml`. The six are the neutral TYK2 ligand, an anion, a cation, a neutral fragment, a flexible anionic lipid, and a flexible Folmsbee molecule. Each is fitted at three torsion regularisation strengths: none (`aimnet2_reg0`, 5 repeats), default (`aimnet2`, 10 repeats) and 10× (`aimnet2_reg10`, 5 repeats). Every fit is timed as in the speed comparison.

Each fit is scored against the molecule's own QM data: QCArchive torsion drives, or Folmsbee DLPNO conformer energies (the full TYK2 ligand has none). Also reported are how much parameters spread and shift and the timings (`benchmarking/fit_robustness/analysis/summary/`).

The study runs about 45 h on one GPU (about 44 h of fits, 1 h of analysis), one fit at a time (as it was used to get timings on a single machine with nothing else running), so it is not part of the cluster workflow. Check that nothing else is using the GPU during the run (e.g. with `nvidia-smi`), or the timings are not comparable. Repeats differ only through MD/metadynamics noise, since every repeat starts from the same conformers.

### Requirements and notes

- **GPU.** The `presto` fits (`run_presto`) and the MLP-based analyses require a GPU and CUDA. Most fitting/analysis rules request a GPU via their SLURM resources.
- **Committed outputs.** Most of `benchmarking/` is git-ignored, but the results needed to inspect the paper's findings without rerunning anything are committed:
  - the **combined bespoke force fields** (`combined_force_field.offxml`) for each paper dataset, under `benchmarking/<dataset>/output/...`. Note these are large (the Folmsbee and JACS force fields in particular).
  - the **lightweight analysis outputs** that back the tables and figures: the torsion-scan `metrics.json` files, Folmsbee `aggregate_stats.csv`, the descriptor and validation-error aggregates, the congeneric/reproducibility summaries, and the RBFE `bootstrap_statistics.csv` / `statistical_tests.csv`.

  Heavy or intermediate artifacts (per-molecule fits, `minimized.json`, `*.sqlite` torsion databases, plots/HTML, and all exploratory/"saved" runs) remain ignored. The exact re-included paths are listed in `.gitignore`.
- **Network downloads.** Benchmark *inputs* are not committed and are downloaded at runtime from QCArchive and from the Folmsbee/Hutchison `conformer-benchmark` GitHub repository.
- **Pixi environments** are selected per rule:
  - `presto051` is auto-selected (via `get_pixi_env`) for the `tnet500` and `jacs_fragments` datasets, which were run with `presto` 0.5.1; everything else uses `default` (`presto` 0.6.0).
  - The Folmsbee set uses the `aimnet2` config (AIMNet2 as the reference MLP), matching the paper.
  - The JACS full-molecule-fit and TYK2 cyclopropyl torsion analyses use the `no-openeye` environment, matching the environment used by SandboxAQ.
- **RBFE.** The free-energy section runs entirely from the committed data in `benchmarking/rbfe_sandbox/`; no GPU or download is needed for it.
- **Charge split.** The `split_torsion_scans_by_charge` rule re-plots an existing torsion analysis (`torsion-data.sqlite` + `metrics.json`) separately for overall-neutral and overall-charged molecules into `plots_neutral/` and `plots_charged/`, each with a `molecule_counts.txt`. It runs from cached data (no fit/download/GPU).

## Running on a SLURM cluster

Create a workflow-specific profile (the `profiles/` directory is git-ignored):
```bash
mkdir -p profiles/default
cat > profiles/default/config.yaml << 'EOF'
executor: slurm
jobs: 100
default-resources:
  mem_mb: 4000
  runtime: 60          # minutes
  slurm_partition: ""  # set your partition, e.g. "gpu"
  slurm_account: ""    # set your account if required
latency-wait: 60
rerun-incomplete: true
EOF
```

Snakemake automatically picks up `profiles/default` as the default profile, so you can simply run:
```bash
pixi run snakemake
```

### Changing the queue/partition for the fitting jobs

The GPU fitting rules (`run_presto`, and likewise `run_tyk2_congeneric_series`, `run_tyk2_congeneric_series_retrain`, `run_1mer_backbone_joint_fit`) hard-code their SLURM partition and GPU request in a `resources:` block, e.g. in the `Snakefile`:
```python
rule run_presto:
    ...
    resources:
        mem_mb=8000,
        runtime=120,             # minutes
        slurm_partition="gpu-s_free",   # <- change to your GPU queue
        slurm_extra="--gpus-per-task=1" # <- adjust GPU request if needed
```
Because a value set on the rule overrides the profile's `default-resources`, to switch queue you can edit the `resources:` block(s) in the `Snakefile` directly (change `slurm_partition`, and `slurm_extra` for the GPU/account flags).
