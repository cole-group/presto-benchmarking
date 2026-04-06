"""Shared plotting constants used across all analysis modules."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Force-field display names
# ---------------------------------------------------------------------------

# Canonical mapping from raw force-field labels / paths to human-readable
# display names.  Used as the default in all plotting functions; any key not
# present is shown as-is.
FORCE_FIELD_DISPLAY_MAP: dict[str, str] = {
    # Ablation protocol short-names (used in ablation_comparison plots)
    "default": "Default",
    "no_metad": "No metadynamics",
    "no_min": "No minimised samples",
    "no_reg": "No regularisation",
    "one_it": "One Iteration",
    # Base / external force fields referenced by label or path
    "openff-2.3.0": "OpenFF 2.3.0",
    "input_ff/esp04.offxml": "espaloma 0.4.0",
    "input_ff/aceff20.offxml": "AceFF 2.0",
    # Presto bespoke FF paths (tnet500 test set)
    "benchmarking/tnet500/output/test/default/combined_force_field.offxml": "presto",
    # JACS Fragments
    "benchmarking/jacs_fragments/output/test/default/combined_force_field.offxml": "presto",
    "input_ff/bespokefit1_sage_jacs_frags.offxml": "OpenFF BespokeFit /\n B3LYP-D3BJ/DZVP",
}

# ---------------------------------------------------------------------------
# Metric display metadata
# ---------------------------------------------------------------------------

# Human-readable axis labels for each raw metric key.
METRIC_LABELS: dict[str, str] = {
    "rmsd": "RMSD",
    "rmse": "RMSE",
    "js_distance": "$\sqrt{\mathrm{JSD}} (500 K)$",
}

# LaTeX unit strings for each raw metric key.
METRIC_UNITS: dict[str, str] = {
    "rmsd": r"$\mathrm{\AA}$",
    "rmse": r"kcal mol$^{-1}$",
    "js_distance": r"$\sqrt{\mathrm{bits}}$",
}

# Suggested x-axis limits for CDF plots (None = auto).
METRIC_CDF_XLIM: dict[str, tuple[float | None, float | None]] = {
    "rmsd": (0, 0.5),
    "rmse": (-0.3, 5.0),
    "js_distance": (None, None),
}
