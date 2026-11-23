"""Summarise the fit-robustness study: repeat presto fits at several torsion
regularisation strengths, per molecule (see the Snakefile "Fit robustness" section).

Run directories are ``<root>/output/<molecule>/<reg>/run_XX``. Every table has one
row per run, and an ``_agg`` version with mean/sd/min/max over repeats.
"""

import json
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw

from convenience_functions._stats import rms
from convenience_functions.presto_fitting_analysis import ITERATION_SPECS, _get_fit_stage_rmse
from convenience_functions.speed_comparison import presto_phases

SAGE = "openff-2.3.0"
SAGE_OFFXML = "input_ff/openff_unconstrained-2.3.0.offxml"
_DISPLAY_NAMES = {"lipid": "Lipid Fragment", "tyk2": "TYK2", "folmsbee_flex": "Folmsbee Flexible"}


def display_name(molecule: str) -> str:
    """Molecule name for tables and plots, e.g. folmsbee_flex -> Folmsbee Flex."""
    return _DISPLAY_NAMES.get(molecule, molecule.replace("_", " ").title())


def _run_ids(ff: str) -> dict[str, str]:
    """``.../output/<molecule>/<reg>/run_XX/bespoke_force_field.offxml`` -> ids."""
    run_dir = Path(ff).parent
    return {"molecule": run_dir.parent.parent.name, "reg": run_dir.parent.name, "run": run_dir.name}


def _agg(df: pd.DataFrame, values: list[str]) -> pd.DataFrame:
    out = df.groupby(["molecule", "reg"])[values].agg(["count", "mean", "std", "min", "max"])
    out.columns = [f"{v}_{stat}" for v, stat in out.columns]
    return out.reset_index()


def torsion_accuracy(metrics_path: Path, ffs: list[str]) -> pd.DataFrame:
    """RMS over the molecule's QM torsion drives of each drive's RMSE/RMSD, per fit (and
    Sage), as in the paper's torsion benchmarks (yammbs_torsion_analysis "RMS RMSE")."""
    metrics = json.loads(metrics_path.read_text())["metrics"]
    rows = []
    for ff in [*ffs, SAGE]:
        per_torsion = pd.DataFrame(metrics[ff]).T
        ids = {"reg": SAGE, "run": SAGE} if ff == SAGE else _run_ids(ff)
        rows.append({**ids, "rmse": rms(per_torsion["rmse"]), "rmsd": rms(per_torsion["rmsd"])})
    return pd.DataFrame(rows)


def folmsbee_accuracy(stats_path: Path, ffs: list[str]) -> pd.DataFrame:
    """Relative conformer energy RMSE and Kendall tau vs DLPNO, per fit (and Sage)."""
    stats = pd.read_csv(stats_path).set_index("method")
    rows = []
    for ff in [*ffs, SAGE_OFFXML]:
        ids = {"reg": SAGE, "run": SAGE} if ff == SAGE_OFFXML else _run_ids(ff)
        row = stats.loc[ff]
        rows.append({**ids, "rmse": row["rmse_kcal_mol"], "kendall_tau": row["kendall_tau"]})
    return pd.DataFrame(rows)


def run_timings(run_dir: Path) -> dict[str, float]:
    run = json.loads((run_dir / "timings.json").read_text())
    wall = json.loads((run_dir / "wall_time.json").read_text())["wall_s"]
    return {"wall_s": wall, "startup_s": run["startup_s"], **presto_phases(run)}


def molecule_size(smiles: str) -> dict[str, int]:
    mol = Chem.MolFromSmiles(smiles)
    return {
        "n_atoms": Chem.AddHs(mol).GetNumAtoms(),
        "n_heavy_atoms": mol.GetNumHeavyAtoms(),
        # On the H-free molecule: with explicit Hs RDKit also counts CH3/NH3+ rotors
        "n_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "formal_charge": Chem.GetFormalCharge(mol),
    }


def summarise_fit_robustness(root: Path, molecules_qm: dict[str, str], ffs: list[str]) -> None:
    out_dir = root / "analysis" / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis = root / "analysis"

    def write(df: pd.DataFrame, name: str, agg_values: list[str] | None = None) -> None:
        df.to_csv(out_dir / f"{name}.csv", index=False)
        if agg_values:
            agg = _agg(df, agg_values)
            agg.to_csv(out_dir / f"{name}_agg.csv", index=False)
            agg.assign(molecule=agg["molecule"].map(display_name)).to_latex(out_dir / f"{name}_agg.tex", index=False, float_format="%.3f")

    # QM accuracy, per run
    qm = []
    for molecule, kind in molecules_qm.items():
        mol_ffs = [ff for ff in ffs if _run_ids(ff)["molecule"] == molecule]
        if kind == "torsion":
            df = torsion_accuracy(analysis / molecule / "torsions" / "metrics.json", mol_ffs)
        elif kind == "folmsbee":
            stats = analysis / molecule / "folmsbee" / "per_molecule_stats.csv"
            df = folmsbee_accuracy(stats, mol_ffs)
        else:
            continue
        qm.append(df.assign(molecule=molecule, reference=kind))
    qm_df = pd.concat(qm, ignore_index=True)
    write(qm_df, "qm_accuracy", ["rmse"])

    # Parameter spread across repeats, and how far torsion k's moved from the start
    spread, shift = [], []
    for molecule in molecules_qm:
        for reg_dir in sorted((analysis / molecule / "parameters").iterdir()):
            ids = {"molecule": molecule, "reg": reg_dir.name}
            spread.append(pd.read_csv(reg_dir / "offxml_variability_summary.csv").assign(**ids))
            traj = pd.read_csv(reg_dir / "tensor_parameter_trajectories.csv")
            final = traj[traj["global_epoch"] == traj.groupby("run_id")["global_epoch"].transform("max")]
            shift.append(
                final.assign(abs_shift=final["signed_change"].abs())
                .groupby(["valence_type", "parameter_name"], as_index=False)["abs_shift"]
                .mean()
                .assign(**ids)
            )
    spread_df = pd.concat(spread, ignore_index=True)
    write(spread_df, "parameter_spread")
    (out_dir / "parameter_spread.tex").write_text(parameter_spread_latex(spread_df))
    write(pd.concat(shift, ignore_index=True), "parameter_shift_from_initial")

    # Held-out energy/force error vs the MLP, and timings, per run
    mlp, timings = [], []
    for ff in ffs:
        run_dir, ids = Path(ff).parent, _run_ids(ff)
        for it in ITERATION_SPECS:
            energy, force = _get_fit_stage_rmse(run_dir, it)
            mlp.append({**ids, "iteration": it.label, "energy_rmse_per_atom": energy, "force_rmse": force})
        timings.append({**ids, **run_timings(run_dir)})
    mlp_df = pd.DataFrame(mlp)
    write(mlp_df, "mlp_heldout")
    write(mlp_df[mlp_df["iteration"] == "Iteration 2"], "mlp_heldout_final", ["energy_rmse_per_atom"])

    sizes = pd.DataFrame(
        [
            {"molecule": m, **molecule_size((root / "input" / f"{m}.smi").read_text().strip())}
            for m in molecules_qm
        ]
    )
    timing_cols = [c for c in timings[0] if c not in ("molecule", "reg", "run")]
    timing_df = pd.DataFrame(timings).merge(sizes, on="molecule")
    write(timing_df, "timings")
    per_molecule = timing_df.groupby("molecule")[timing_cols].agg(["mean", "std"])
    per_molecule.columns = [f"{c}_{s}" for c, s in per_molecule.columns]
    per_molecule = per_molecule.reset_index().merge(sizes, on="molecule")
    per_molecule.to_csv(out_dir / "timings_per_molecule.csv", index=False)
    per_molecule.assign(molecule=per_molecule["molecule"].map(display_name)).to_latex(out_dir / "timings_per_molecule.tex", index=False, float_format="%.0f")

    smiles = {m: (root / "input" / f"{m}.smi").read_text().strip() for m in molecules_qm}
    _plot_qm_spread(qm_df, smiles, out_dir / "qm_accuracy_spread.png")
    _plot_time_vs_size(per_molecule, out_dir / "time_vs_size.png")


def _reg_key(reg: str) -> int:
    return int(reg.removeprefix("reg"))


def parameter_spread_latex(spread: pd.DataFrame) -> str:
    """Booktabs table of RMS normalised SD across repeat fits, per regularisation and
    parameter type, for each molecule and pooled over all molecules."""
    spread = spread.assign(
        parameter=spread["valence_type"] + " " + spread["parameter_name"],
        weighted_sq=spread["n_parameter_ids"] * spread["rms_normalised_std_dev"] ** 2,
    )
    keys = ["reg", "parameter"]
    table = spread.pivot_table(index=keys, columns="molecule", values="rms_normalised_std_dev")
    # Pool over molecules as if all their parameter IDs were in one RMS.
    # shortcut: weights by n_parameter_ids, which also counts any near-zero-scale
    # parameters excluded from each molecule's RMS (none expected for these terms).
    pooled = spread.groupby(keys)[["weighted_sq", "n_parameter_ids"]].sum()
    table["Overall"] = np.sqrt(pooled["weighted_sq"] / pooled["n_parameter_ids"])
    table = table.sort_index(level="reg", key=lambda r: r.map(_reg_key), sort_remaining=False)
    table.columns = [display_name(c) for c in table.columns]
    table.index = table.index.set_names(["Regularisation", "Parameter"])
    return table.to_latex(float_format="%.3f", multirow=True)


def _draw_molecule(ax: plt.Axes, smiles: str) -> None:
    image = Draw.MolToImage(Chem.MolFromSmiles(smiles), size=(500, 350))
    buffer = BytesIO()
    image.save(buffer, format="png")
    buffer.seek(0)
    ax.imshow(plt.imread(buffer))
    ax.axis("off")


def _plot_qm_spread(qm_df: pd.DataFrame, smiles: dict[str, str], path: Path) -> None:
    """Per molecule: its 2D structure above the RMSE vs QM of every fit, by
    regularisation, with Sage as a dashed line."""
    fits = qm_df[qm_df["reg"] != SAGE]
    molecules = list(fits["molecule"].unique())
    regs = sorted(fits["reg"].unique(), key=_reg_key)
    fig, axes = plt.subplots(
        2, len(molecules), figsize=(3 * len(molecules), 6),
        gridspec_kw={"height_ratios": [1, 1.6]}, squeeze=False,
    )
    for ax_mol, ax, molecule in zip(axes[0], axes[1], molecules):
        _draw_molecule(ax_mol, smiles[molecule])
        mol = fits[fits["molecule"] == molecule]
        for x, reg in enumerate(regs):
            y = mol[mol["reg"] == reg]["rmse"]
            ax.scatter([x] * len(y), y, alpha=0.7)
        sage = qm_df[(qm_df["molecule"] == molecule) & (qm_df["reg"] == SAGE)]["rmse"]
        ax.axhline(sage.iloc[0], ls="--", c="grey", label="Sage 2.3.0")
        ax.set_ylim(bottom=0)
        ax.set_xticks(range(len(regs)), regs)
        ax.set_title(display_name(molecule))
        ax.set_ylabel("RMSE vs QM / kcal mol$^{-1}$")
    axes[1][-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_time_vs_size(per_molecule: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.errorbar(
        per_molecule["n_atoms"], per_molecule["wall_s_mean"] / 60,
        yerr=per_molecule["wall_s_std"] / 60, fmt="o",
    )
    for _, row in per_molecule.iterrows():
        ax.annotate(display_name(row["molecule"]), (row["n_atoms"], row["wall_s_mean"] / 60), fontsize=7)
    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Wall time per fit / min")
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)
