"""Summarise the presto vs BespokeFit speed comparison.

Reads ``<root>/<method>/run_XX/timings.json`` (``run_00`` is a warm-up and is
excluded) and the AIMNet2 parity JSONs, and writes tables and plots to
``<root>/analysis``.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from convenience_functions._stats import rmse

METHODS = [
    "presto",
    "bespokefit_aimnet2_matched",
    "bespokefit_aimnet2_stock",
    "bespokefit_qm",
]
PHASES = [
    "setup",
    "fragmentation",
    "msm",
    "reference_data",
    "optimisation",
    "diagnostics",
    "other",
]
PHASE_LABELS = {
    "setup": "Setup",
    "fragmentation": "Fragmentation",
    "msm": "MSM (ML Hessian)",
    "reference_data": "Reference data",
    "optimisation": "Parameter optimisation",
    "diagnostics": "Diagnostics (presto only)",
    "other": "Other",
}
# Methods shown in the publication table, with their display names
TABLE_METHODS = {
    "presto": "presto",
    "bespokefit_aimnet2_matched": "BespokeFit (AIMNet2)",
    "bespokefit_qm": "BespokeFit (B3LYP-D3(BJ)/DZVP)",
}


def _runs(method_dir: Path, filename: str) -> list[tuple[Path, dict[str, Any]]]:
    """(run dir, parsed ``filename``) for every non-warm-up run of one method."""
    return [
        (p.parent, json.loads(p.read_text()))
        for p in sorted(method_dir.glob(f"run_*/{filename}"))
        if p.parent.name != "run_00"
    ]


def presto_phases(run: dict[str, Any]) -> dict[str, float]:
    """Map presto spans onto the phases shared with BespokeFit."""
    t = {path: row["total_s"] for path, row in run["summary"].items()}

    def get(path: str) -> float:
        return float(t.get(f"total/{path}", 0.0))

    phases = {
        "setup": get("parameterise") - get("parameterise/msm"),
        "fragmentation": 0.0,
        "msm": get("parameterise/msm"),
        "reference_data": get("training_sampling") + get("outlier_filter"),
        "optimisation": get("train") - get("train/loss_no_grad"),
        "diagnostics": get("test_sampling")
        + get("write_scatter")
        + get("plots")
        + get("train/loss_no_grad"),
    }
    phases["other"] = t["total"] - sum(phases.values())
    return phases


def bespokefit_phases(run: dict[str, Any]) -> dict[str, float]:
    """Map BespokeFit stage timings onto the shared phases.

    Stage durations come from polling the coordinator, so are accurate to ~5 s.
    """

    def stage(name: str) -> float:
        s = run["stages"][name]
        return float(s["first_seen_done_s"] - s["first_seen_running_s"])

    phases = {
        "setup": run["schema_s"],
        "fragmentation": stage("fragmentation"),
        "msm": 0.0,
        "reference_data": stage("qc-generation"),
        "optimisation": stage("optimization"),
        "diagnostics": 0.0,
    }
    phases["other"] = run["total_s"] - sum(phases.values())
    return phases


def presto_mlip_s(run: dict[str, Any]) -> float:
    """Time in MLIP evaluations while generating reference data (training data is
    sampled with MM MD, then ML single points and ML restrained minimisations)."""
    t = {path: row["total_s"] for path, row in run["summary"].items()}
    sampling = "total/training_sampling"
    return (
        t[f"{sampling}/ml_single_points"]
        + t[
            f"{sampling}/torsion_minimisation/ml_restrained_minimisation/minimize_energy"
        ]
    )


def compute_call_stats(run_dir: Path) -> dict[str, float]:
    """Number and summed wall time of QC energy/gradient calls in all workers, and
    the wall time during which at least one call was running (``reference_mlip``)."""
    calls = sorted(
        (c["start"], c["end"])
        for log in run_dir.glob("compute_calls/*.jsonl")
        for c in map(json.loads, log.read_text().splitlines())
    )
    durations = np.array([end - start for start, end in calls])
    covered, cur_start, cur_end = 0.0, *calls[0]
    for start, end in calls[1:]:
        if start > cur_end:
            covered += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    covered += cur_end - cur_start
    return {
        "reference_mlip": covered,
        "n_qc_calls": len(calls),
        "qc_call_time_sum_s": float(durations.sum()),
        "qc_call_time_mean_s": float(durations.mean()),
    }


def phase_table(root: Path) -> pd.DataFrame:
    """One row per (method, run) with phase times, totals and QC call stats."""
    rows = []
    for method in METHODS:
        is_presto = method == "presto"
        for run_dir, run in _runs(root / method, "timings.json"):
            phases = presto_phases(run) if is_presto else bespokefit_phases(run)
            row: dict[str, Any] = {"method": method, "run": run_dir.name, **phases}
            row["total"] = sum(phases.values())
            row["fit_essential_total"] = row["total"] - phases["diagnostics"]
            row["startup"] = run["startup_s"]
            if is_presto:
                row["reference_mlip"] = presto_mlip_s(run)
            else:
                row.update(compute_call_stats(run_dir))
            rows.append(row)
    return pd.DataFrame(rows)


def presto_stage_table(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Mean, SD, CV, min and max over runs for every presto span path."""
    totals = pd.DataFrame(
        [{path: row["total_s"] for path, row in run["summary"].items()} for run in runs]
    )
    counts = runs[0]["summary"]
    table = pd.DataFrame(
        {
            "calls_per_run": [counts[p]["count"] for p in totals.columns],
            "mean_s": totals.mean(),
            "sd_s": totals.std(ddof=1),
            "min_s": totals.min(),
            "max_s": totals.max(),
        }
    )
    table["cv"] = table["sd_s"] / table["mean_s"]
    table["pct_of_total"] = 100 * table["mean_s"] / table.loc["total", "mean_s"]
    return table.sort_index()


def presto_iteration_table(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Mean and SD over runs of each top-level presto stage, per training iteration."""
    rows = []
    for i, run in enumerate(runs):
        for span in run["spans"]:
            if span["path"].count("/") == 1:  # direct children of "total"
                rows.append({"run": i, **span})
    df = pd.DataFrame(rows).groupby(["path", "iteration", "run"])["duration_s"].sum()
    return df.groupby(["path", "iteration"]).agg(["mean", "std"])


def parity_table(parity_dir: Path) -> pd.DataFrame:
    """Agreement with presto's own GPU path, and per-call latency, for each AIMNet2 arm."""
    arms = {
        p.stem: json.loads(p.read_text()) for p in sorted(parity_dir.glob("*.json"))
    }
    ref = arms["presto_cuda"]
    ref_rel = np.array(ref["energies_kcal_mol"]) - np.mean(ref["energies_kcal_mol"])
    rows = []
    for name, arm in arms.items():
        rel = np.array(arm["energies_kcal_mol"]) - np.mean(arm["energies_kcal_mol"])
        times = np.array(arm["call_times_s"])
        rows.append(
            {
                "arm": name,
                "rel_energy_mae_kcal_mol": float(np.abs(rel - ref_rel).mean()),
                "force_rmse_kcal_mol_A": rmse(
                    np.array(arm["forces_kcal_mol_A"]).ravel(),
                    np.array(ref["forces_kcal_mol_A"]).ravel(),
                ),
                "median_call_ms": 1e3 * float(np.median(times)),
                "iqr_call_ms": 1e3
                * float(np.subtract(*np.percentile(times, [75, 25]))),
                "torch_threads": arm["torch_threads"],
            }
        )
    return pd.DataFrame(rows).set_index("arm")


def total_time_latex(summary: pd.DataFrame) -> str:
    """Booktabs table of mean (± SD) total wall time in minutes for TABLE_METHODS."""
    rows = []
    for method, label in TABLE_METHODS.items():
        mean, sd = summary.loc[method, ("total", ["mean", "std"])] / 60
        time = f"{mean:.1f}" if np.isnan(sd) else f"{mean:.1f} $\\pm$ {sd:.1f}"
        rows.append(f"{label} & {time} \\\\")
    return "\n".join(
        [
            r"\begin{tabular}{lr}",
            r"\toprule",
            r"Method & Total Wall Time / min \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )


def plot_phases(summary: pd.DataFrame, path: Path) -> None:
    """Fraction of wall time per phase for TABLE_METHODS, with reference data split into MLIP (or
    QM, for BespokeFit (QM)) calls and everything else."""
    mean = summary.xs("mean", axis=1, level=1).loc[list(TABLE_METHODS)]
    mean["reference_other"] = mean["reference_data"] - mean["reference_mlip"]
    labels = {
        **PHASE_LABELS,
        "reference_mlip": "Reference data: MLIP/QM calls",
        "reference_other": "Reference data: other",
    }
    phases = [p for p in PHASES if p != "reference_data"]
    i = phases.index("msm") + 1
    phases[i:i] = ["reference_mlip", "reference_other"]

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(mean) + 1.5))
    y = np.arange(len(mean))
    left = np.zeros(len(mean))
    for phase in phases:
        frac = (mean[phase] / mean["total"]).to_numpy()
        ax.barh(y, frac, left=left, label=labels[phase])
        left += frac
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of wall time")
    ax.set_yticks(y, [TABLE_METHODS[m] for m in mean.index])
    ax.invert_yaxis()  # first method at the top
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyse_speed_comparison(root: Path) -> None:
    """Write summary tables and plots for the speed comparison under ``root``."""
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    per_run = phase_table(root)
    per_run.to_csv(out / "per_run.csv", index=False)
    numeric = per_run.drop(columns="run").groupby("method")
    summary = numeric.agg(["mean", "std", "min", "max"])
    summary.to_csv(out / "summary.csv")
    (out / "summary.tex").write_text(total_time_latex(summary))
    plot_phases(summary, out / "phases.png")

    presto_runs = [run for _, run in _runs(root / "presto", "timings.json")]
    stages = presto_stage_table(presto_runs)
    stages.to_csv(out / "presto_stage_breakdown.csv")
    stages[["calls_per_run", "mean_s", "sd_s", "pct_of_total"]].round(2).to_latex(
        out / "presto_stage_breakdown.tex"
    )
    presto_iteration_table(presto_runs).to_csv(out / "presto_per_iteration.csv")

    # Instrumentation overhead: whole-process wall time with vs without timing
    overhead = {
        m: [run["wall_s"] for _, run in _runs(root / m, "wall_time.json")]
        for m in ["presto", "presto_uninstrumented"]
    }
    (out / "instrumentation_overhead.json").write_text(json.dumps(overhead, indent=2))

    parity_table(root / "aimnet2_parity").round(4).to_csv(out / "aimnet2_parity.csv")
