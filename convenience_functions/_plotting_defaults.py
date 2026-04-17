"""Shared plotting constants used across all analysis modules."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Generic naming helpers
# ---------------------------------------------------------------------------


def humanize_token(token: str) -> str:
    """Return a display-friendly label for a snake_case token."""
    return token.replace("_", " ").title()

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
    "input_ff/bespokefit1_sage_jacs_frags.offxml": "OpenFF\nBespokeFit/\nB3LYP-D3BJ/DZVP",
}

# ---------------------------------------------------------------------------
# Dataset/config display names
# ---------------------------------------------------------------------------

# Canonical dataset display names independent of split (test/validation).
DATASET_DISPLAY_MAP: dict[str, str] = {
    "tnet500": "TorsionNet500",
    "tnet500_reopt_v4": "TorsionNet500 Reopt v4",
    "jacs_fragments": "JACS Fragments",
    "phosphate_torsion_drives": "Phosphate Torsion Drives",
    "folmsbee_conformers": "Folmsbee",
    "1mer_backbone": "Protein 1mer Backbone",
    "3mer_backbone": "Protein 3mer Backbone",
    "1mer_side_chain": "Protein 1mer Side Chain",
}

# Split-specific dataset labels for tables where split should be explicit.
DATASET_SPLIT_DISPLAY_MAP: dict[tuple[str, str], str] = {
    ("tnet500", "test"): "TorsionNet500 Test",
    ("tnet500", "validation"): "TorsionNet500 Validation",
    ("jacs_fragments", "test"): "JACS Fragments Test",
    ("folmsbee_conformers", "test"): "Folmsbee Test",
    ("phosphate_torsion_drives", "test"): "Phosphate Torsion Drives Test",
    ("1mer_backbone", "test"): "Protein 1mer Backbone Test",
    ("3mer_backbone", "test"): "Protein 3mer Backbone Test",
    ("1mer_side_chain", "test"): "Protein 1mer Side Chain Test",
}

CONFIG_DISPLAY_MAP: dict[str, str] = {
    "default": "Default",
    "aimnet2": "AIMNet2",
    "no_reg": "No Reg",
    "no_min": "No Min",
    "one_it": "One It",
    "no_metad": "No Metad",
    "ablations": "Ablations",
}


def get_dataset_display_name(dataset_name: str, dataset_type: str | None = None) -> str:
    """Return canonical dataset display name, optionally including split label."""
    if dataset_type is not None:
        split_label = DATASET_SPLIT_DISPLAY_MAP.get((dataset_name, dataset_type))
        if split_label is not None:
            return split_label

    return DATASET_DISPLAY_MAP.get(dataset_name, humanize_token(dataset_name))


def get_config_display_name(config_name: str) -> str:
    """Return canonical config display name."""
    return CONFIG_DISPLAY_MAP.get(config_name, humanize_token(config_name))

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
