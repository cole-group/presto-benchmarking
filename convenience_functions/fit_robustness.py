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
from convenience_functions.tyk2_reproducibility import _LATEX_UNITS, _OFFXML_PARAM_KEYS

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


# Validation data source and level of theory, by molecule (TYK2 has none)
_TORSION_LOT = "B3LYP-D3(BJ)/DZVP"
_JACS = "OpenFF-benchmark-ligand-fragments-v2.0"
_VALIDATION_SOURCES = {
    "lipid": ("OpenFF Lipid Torsion Drives v4.0", _TORSION_LOT),
    "anion": (_JACS, _TORSION_LOT),
    "cation": (_JACS, _TORSION_LOT),
    "neutral_fragment": (_JACS, _TORSION_LOT),
    "folmsbee_flex": (
        "Folmsbee and Hutchison (astex\\_1uml)",
        "DLPNO-CCSD(T)/cc-pVTZ//B3LYP-D3(BJ)",
    ),
}


def molecules_table(root: Path, molecules_qm: dict[str, str], sizes: pd.DataFrame) -> pd.DataFrame:
    """Each molecule's SMILES, net charge, atom count and the source, amount
    and level of theory of its validation data, formatted for LaTeX."""
    analysis = root / "analysis"
    rows = []
    for molecule, kind in molecules_qm.items():
        size = sizes.set_index("molecule").loc[molecule]
        if kind == "torsion":
            n = len(json.loads((root / "input" / f"{molecule}.json").read_text())["qm_torsions"])
            data = f"{n} torsion drives"
        elif kind == "folmsbee":
            stats = pd.read_csv(analysis / molecule / "folmsbee" / "per_molecule_stats.csv")
            data = f"{stats['n_conformers'].iloc[0]} conformer energies"
        else:
            data = "--"
        source, lot = _VALIDATION_SOURCES.get(molecule, ("--", "--"))
        smiles = (root / "input" / f"{molecule}.smi").read_text().strip()
        rows.append(
            {
                "Molecule": display_name(molecule),
                "SMILES": r"\texttt{" + smiles.replace("#", r"\#").replace("%", r"\%") + "}",
                "Net charge": f"${size['formal_charge']:+d}$" if size["formal_charge"] else "0",
                "N atoms": size["n_atoms"],
                "Validation source": source,
                "Validation data": data,
                "Level of theory": lot,
            }
        )
    return pd.DataFrame(rows)


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
    final_values, shift = [], []
    for molecule in molecules_qm:
        for reg_dir in sorted((analysis / molecule / "parameters").iterdir()):
            ids = {"molecule": molecule, "reg": reg_dir.name}
            final_values.append(pd.read_csv(reg_dir / "offxml_parameter_values.csv").assign(**ids))
            traj = pd.read_csv(reg_dir / "tensor_parameter_trajectories.csv")
            final = traj[traj["global_epoch"] == traj.groupby("run_id")["global_epoch"].transform("max")]
            shift.append(
                final.assign(abs_shift=final["signed_change"].abs())
                .groupby(["valence_type", "parameter_name"], as_index=False)["abs_shift"]
                .mean()
                .assign(**ids)
            )
    spread_df = parameter_spread(pd.concat(final_values, ignore_index=True))
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
    molecules = molecules_table(root, molecules_qm, sizes)
    molecules.to_latex(out_dir / "molecules.tex", index=False)
    wall = per_molecule.set_index("molecule").loc[list(molecules_qm)]
    brief = molecules[["Molecule", "SMILES", "Net charge", "N atoms"]].assign(
        **{"Wall time / s": [f"${m:.0f} \\pm {s:.0f}$" for m, s in zip(wall["wall_s_mean"], wall["wall_s_std"])]}
    )
    brief.sort_values("N atoms", kind="stable").to_latex(out_dir / "molecules_brief.tex", index=False)
    _plot_time_vs_size(per_molecule, out_dir / "time_vs_size.png")


def _reg_key(reg: str) -> int:
    return int(reg.removeprefix("reg"))


def sage_scales() -> pd.Series:
    """Mean |value| of each parameter term over all of Sage 2.3.0, in the units of
    the bespoke OFFXML values. Proper torsion terms are matched by periodicity, as
    presto's k1-k4 have periodicities 1-4."""
    from openff.toolkit import ForceField

    ff = ForceField(SAGE_OFFXML)
    units = {"Bonds": ("kcal/mol/angstrom**2", "angstrom"), "Angles": ("kcal/mol/radian**2", "radian")}
    values: dict[tuple[str, str], list[float]] = {}
    for handler, names in _OFFXML_PARAM_KEYS.items():
        for param in ff.get_parameter_handler(handler).parameters:
            if handler in units:
                terms = [(n, getattr(param, n).m_as(u)) for n, u in zip(names, units[handler])]
            elif handler == "ProperTorsions":
                terms = [(f"k{n}", k.m_as("kcal/mol")) for n, k in zip(param.periodicity, param.k)]
            else:  # impropers: presto's single k1 has periodicity 2, as in Sage
                terms = [(f"k{i}", k.m_as("kcal/mol")) for i, k in enumerate(param.k, 1)]
            for name, value in terms:
                values.setdefault((handler, name), []).append(abs(value))
    return pd.Series(
        {key: float(np.mean(values[key])) for key in values if key[1] in _OFFXML_PARAM_KEYS[key[0]]},
        name="sage_mean_abs",
    ).rename_axis(["valence_type", "parameter_name"])


def parameter_spread(final: pd.DataFrame) -> pd.DataFrame:
    """RMS over parameter IDs of each ID's SD of final values across repeat fits, raw and
    normalised, per molecule (and pooled over molecules) and regularisation.

    The normaliser is the mean |value| of that parameter term over all of Sage 2.3.0
    (``sage_scales``), so it is the same for every molecule and regularisation
    strength, and near-zero parameters can't dominate."""
    keys = ["valence_type", "parameter_name", "unit"]
    sd = (
        final.groupby(["molecule", "reg", *keys, "parameter_id"])["value"]
        .agg(lambda v: float(np.std(v, ddof=0)))
        .rename("std_dev")
        .reset_index()
        .join(sage_scales(), on=["valence_type", "parameter_name"])
    )
    sd["normalised_std_dev"] = sd["std_dev"] / sd["sage_mean_abs"]
    sd = pd.concat([sd, sd.assign(molecule="Overall")], ignore_index=True)
    return (
        sd.groupby(["molecule", "reg", *keys, "sage_mean_abs"])[["normalised_std_dev", "std_dev"]]
        .agg(lambda v: float(np.sqrt(np.mean(np.square(v)))))
        .add_prefix("rms_")
        .reset_index()
    )


def parameter_spread_latex(rows: pd.DataFrame) -> str:
    """Booktabs table of ``parameter_spread`` output, molecules by rows and
    regularisation strengths by columns."""
    metrics = {"rms_normalised_std_dev": "RMS normalised std dev", "rms_std_dev": "RMS std dev"}
    keys = ["valence_type", "parameter_name", "unit"]
    table = rows.pivot_table(index=["molecule", *keys], columns="reg", values=list(metrics))
    table = table.swaplevel(axis=1)[
        [(reg, m) for reg in sorted(rows["reg"].unique(), key=_reg_key) for m in metrics]
    ]
    table.columns = table.columns.set_names([None, None])
    table = table.rename(columns=metrics, level=1)
    # Molecules alphabetically then Overall; parameters in _OFFXML_PARAM_KEYS order
    molecules = [m for m in sorted(rows["molecule"].unique()) if m != "Overall"] + ["Overall"]
    order = {(v, n): i for i, (v, n) in enumerate(
        (v, n) for v, names in _OFFXML_PARAM_KEYS.items() for n in names
    )}
    table = table.iloc[
        sorted(
            range(len(table)),
            key=lambda i: (molecules.index(table.index[i][0]), order[table.index[i][1:3]]),
        )
    ]
    table = table.rename(index=lambda m: m if m == "Overall" else display_name(m), level=0)
    table = table.rename(index=_LATEX_UNITS, level="unit")
    # Label torsion k's by periodicity: presto's improper k1 has periodicity 2
    table.index = pd.MultiIndex.from_tuples(
        [
            (m, v, "k2" if (v, p) == ("ImproperTorsions", "k1") else p, u)
            for m, v, p, u in table.index
        ]
    )
    table.index = table.index.set_names(["Molecule", "Valence", "Parameter", "Unit"])
    latex = table.to_latex(
        float_format="%.2f", multirow=True, multicolumn=True, multicolumn_format="c"
    )
    # pandas rules off every multirow group; keep one \midrule between molecules
    lines = [
        r"\midrule" if line.startswith(r"\cline{1-") else line
        for line in latex.splitlines()
        if not line.startswith(r"\cline{") or line.startswith(r"\cline{1-")
    ]
    lines.pop(-3)  # the \midrule just before \bottomrule
    return "\n".join(lines) + "\n"


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
