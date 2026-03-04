"""Analysis plots for ablation comparison of combined force fields.

Produces:
  - <output_dir>/heatmap.png   – annotated % change heatmap with significance stars
  - <output_dir>/distributions.png – histograms + Q-Q plots of per-torsion differences
"""

from pathlib import Path

import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binomtest, probplot, shapiro

matplotlib.use("Agg")

HEATMAP_METRICS = ["RMSD", "RMSE", "Mean JSD (500 K)"]

# Keys excluded from colorbar scaling (still displayed; large values saturate the scale)
COLORBAR_EXCL = {"openff-2.3.0"}

FORCE_FIELD_DISPLAY_MAP = {
    "default": "Default",
    "no_metad": "No metadynamics",
    "no_min": "No minimised samples",
    "no_reg": "No regularisation",
    "one_it": "One Iteration",
    "openff-2.3.0": "OpenFF 2.3.0",
}

_METRIC_COLUMN = {"RMSD": "rmsd", "RMSE": "rmse", "Mean JSD (500 K)": "js_distance"}
_REFERENCE_LABEL = "default"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ff_key_from_raw_name(raw_name: str) -> str:
    if "/validation/" in raw_name and raw_name.endswith("/combined_force_field.offxml"):
        return Path(raw_name).parent.name
    return raw_name


def _extract_metric(record: dict, metric_name: str) -> float:
    value = record[metric_name]
    return float(value[0]) if metric_name == "js_distance" else float(value)


def _aggregate(values: np.ndarray, metric_label: str) -> float:
    if metric_label in {"RMSD", "RMSE"}:
        return float(np.sqrt(np.mean(values ** 2)))
    return float(np.mean(values))  # Mean JSD (500 K)


def _sign_test_pvalue(ref_vals: np.ndarray, ff_vals: np.ndarray) -> float:
    diffs = ff_vals - ref_vals
    n_non_zero = int(np.sum(diffs != 0))
    if n_non_zero == 0:
        return 1.0
    n_pos = int(np.sum(diffs > 0))
    return float(binomtest(n_pos, n_non_zero, p=0.5, alternative="two-sided").pvalue)


def _holm_bonferroni(p_values: list) -> list:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, orig_idx in enumerate(order):
        corrected = min(p_values[orig_idx] * (m - rank), 1.0)
        running_max = max(running_max, corrected)
        adjusted[orig_idx] = running_max
    return adjusted


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _disp(ff_key: str) -> str:
    return FORCE_FIELD_DISPLAY_MAP.get(ff_key, ff_key)


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_per_ff(metrics_json: Path) -> dict[str, pd.DataFrame]:
    with open(metrics_json) as fh:
        payload = json.load(fh)

    per_ff: dict[str, pd.DataFrame] = {}
    for raw_name, torsion_dict in payload["metrics"].items():
        ff_key = _ff_key_from_raw_name(raw_name)
        per_ff[ff_key] = pd.DataFrame([
            {
                "torsion_id": str(tid),
                "rmsd": _extract_metric(v, "rmsd"),
                "rmse": _extract_metric(v, "rmse"),
                "js_distance": _extract_metric(v, "js_distance"),
            }
            for tid, v in torsion_dict.items()
        ])
    return per_ff


def _build_pct_and_stars(
    per_ff: dict[str, pd.DataFrame],
    all_ff_keys: list[str],
    ablation_keys: list[str],
    merged_dfs: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pct = pd.DataFrame(index=HEATMAP_METRICS, columns=all_ff_keys, dtype=float)
    for ff in all_ff_keys:
        mdf = merged_dfs[ff]
        for metric in HEATMAP_METRICS:
            col = _METRIC_COLUMN[metric]
            ref_v = mdf[f"{col}_ref"].to_numpy(float)
            ff_v  = mdf[f"{col}_ff"].to_numpy(float)
            ref_agg = _aggregate(ref_v, metric)
            ff_agg  = _aggregate(ff_v, metric)
            pct.loc[metric, ff] = (
                np.nan if np.isclose(ref_agg, 0.0)
                else 100.0 * (ff_agg - ref_agg) / ref_agg
            )

    test_keys = [(ff, metric) for ff in all_ff_keys for metric in HEATMAP_METRICS]
    raw_pvals = [
        _sign_test_pvalue(
            merged_dfs[ff][f"{_METRIC_COLUMN[metric]}_ref"].to_numpy(float),
            merged_dfs[ff][f"{_METRIC_COLUMN[metric]}_ff"].to_numpy(float),
        )
        for ff, metric in test_keys
    ]
    adj_pvals = _holm_bonferroni(raw_pvals)
    stars_map = {k: _stars(p) for k, p in zip(test_keys, adj_pvals)}

    annot = pd.DataFrame(index=HEATMAP_METRICS, columns=all_ff_keys, dtype=object)
    for metric in HEATMAP_METRICS:
        for ff in all_ff_keys:
            val = pct.loc[metric, ff]
            txt = "NA" if pd.isna(val) else f"{val:+.1f}%"
            annot.loc[metric, ff] = f"{txt}{stars_map[(ff, metric)]}"

    return pct, annot, stars_map


# ── Public API ─────────────────────────────────────────────────────────────────

def plot_ablation_heatmap(metrics_json: Path, output_dir: Path) -> Path:
    """Save an annotated % change heatmap to ``output_dir/heatmap.png``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    per_ff = _load_per_ff(metrics_json)

    if _REFERENCE_LABEL not in per_ff:
        raise ValueError(
            f"Reference '{_REFERENCE_LABEL}' not found. Available: {sorted(per_ff)}"
        )

    reference_df = per_ff[_REFERENCE_LABEL]
    ablation_keys = sorted(
        ff for ff in per_ff if ff != _REFERENCE_LABEL and ff not in COLORBAR_EXCL
    )
    excl_keys   = sorted(ff for ff in COLORBAR_EXCL if ff in per_ff)
    all_ff_keys = ablation_keys + excl_keys

    merged_dfs = {
        ff: reference_df.merge(per_ff[ff], on="torsion_id", suffixes=("_ref", "_ff"))
        for ff in all_ff_keys
    }

    pct, annot, _ = _build_pct_and_stars(per_ff, all_ff_keys, ablation_keys, merged_dfs)

    disp_map = {ff: _disp(ff) for ff in all_ff_keys}
    plot_data  = pct.astype(float).rename(columns=disp_map)
    plot_annot = annot.rename(columns=disp_map)

    abl_data = pct[ablation_keys].astype(float)
    abs_max  = float(np.nanmax(np.abs(abl_data.values)))

    n_cols = len(all_ff_keys)
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * n_cols), 5.4))

    sns.heatmap(
        plot_data,
        ax=ax,
        cmap="coolwarm",
        center=0,
        vmin=-abs_max,
        vmax=abs_max,
        linewidths=0.5,
        linecolor="white",
        annot=plot_annot,
        fmt="",
        cbar_kws={"label": "% change vs default protocol"},
    )
    ax.set_xlabel("Protocol", fontdict={"weight": "bold"}, size=14)
    ax.set_ylabel("Metric", fontdict={"weight": "bold"}, size=14)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="center")

    legend_text = (
        "Significance (sign test, Holm-Bonferroni corrected):  "
        "* p < 0.05    ** p < 0.01    *** p < 0.001"
    )
    fig.text(0.5, -0.02, legend_text, ha="center", va="top", fontsize=10, color="0.35")

    plt.tight_layout()
    out_path = output_dir / "heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_ablation_distributions(metrics_json: Path, output_dir: Path) -> Path:
    """Save histograms + Q-Q plots of per-torsion differences to ``output_dir/distributions.png``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    per_ff = _load_per_ff(metrics_json)

    if _REFERENCE_LABEL not in per_ff:
        raise ValueError(
            f"Reference '{_REFERENCE_LABEL}' not found. Available: {sorted(per_ff)}"
        )

    reference_df = per_ff[_REFERENCE_LABEL]
    ablation_keys = sorted(
        ff for ff in per_ff if ff != _REFERENCE_LABEL and ff not in COLORBAR_EXCL
    )
    excl_keys   = sorted(ff for ff in COLORBAR_EXCL if ff in per_ff)
    all_ff_keys = ablation_keys + excl_keys

    merged_dfs = {
        ff: reference_df.merge(per_ff[ff], on="torsion_id", suffixes=("_ref", "_ff"))
        for ff in all_ff_keys
    }

    disp_map = {ff: _disp(ff) for ff in all_ff_keys}

    n_ff   = len(all_ff_keys)
    n_met  = len(HEATMAP_METRICS)
    n_rows = n_met * 2

    fig, axes = plt.subplots(
        n_rows, n_ff,
        figsize=(2.6 * n_ff, 2.6 * n_rows),
    )

    for m_idx, metric in enumerate(HEATMAP_METRICS):
        col_name = _METRIC_COLUMN[metric]
        hist_row = m_idx * 2
        qq_row   = m_idx * 2 + 1

        for c_idx, ff in enumerate(all_ff_keys):
            mdf   = merged_dfs[ff]
            diffs = (
                mdf[f"{col_name}_ff"].to_numpy(float)
                - mdf[f"{col_name}_ref"].to_numpy(float)
            )

            sw_stat, sw_p = shapiro(diffs)

            # Histogram
            ax_h = axes[hist_row, c_idx]
            ax_h.hist(diffs, bins=20, color="steelblue", edgecolor="white", linewidth=0.4)
            ax_h.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax_h.axvline(
                diffs.mean(), color="tomato", linewidth=1.0,
                label=f"mean={diffs.mean():.3f}",
            )
            ax_h.tick_params(labelsize=7)
            ax_h.legend(fontsize=6, loc="upper right")
            if c_idx == 0:
                ax_h.set_ylabel(f"{metric}\nCount", fontsize=8)
            if m_idx == 0:
                ax_h.set_title(disp_map[ff], fontsize=9)

            # Q-Q plot
            ax_q = axes[qq_row, c_idx]
            (osm, osr), (slope, intercept, _) = probplot(diffs, dist="norm", fit=True)
            ax_q.scatter(osm, osr, s=8, color="steelblue", alpha=0.6, linewidths=0)
            x_line = np.array([osm.min(), osm.max()])
            ax_q.plot(x_line, slope * x_line + intercept, color="tomato", linewidth=1.0)
            ax_q.tick_params(labelsize=7)
            sw_color = "tomato" if sw_p < 0.05 else "forestgreen"
            ax_q.text(
                0.03, 0.97,
                f"S-W p={sw_p:.3g}",
                transform=ax_q.transAxes,
                fontsize=6, va="top", color=sw_color,
            )
            if c_idx == 0:
                ax_q.set_ylabel(f"{metric}\nSample quantiles", fontsize=8)
            if m_idx == n_met - 1:
                ax_q.set_xlabel("Theoretical quantiles", fontsize=8)

    fig.suptitle(
        "Per-torsion differences: FF − Default\n"
        "Top rows: histogram (red = mean, dashed = zero) | "
        "Bottom rows: Q-Q vs normal — Shapiro-Wilk p shown "
        "(green = normal, red = non-normal at α=0.05)",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    out_path = output_dir / "distributions.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_ablation_comparison(metrics_json: Path, output_dir: Path) -> None:
    """Generate both ablation comparison plots and save them to ``output_dir``."""
    plot_ablation_heatmap(metrics_json, output_dir)
    plot_ablation_distributions(metrics_json, output_dir)
