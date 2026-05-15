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

# Stable, colour-blind friendly palette for force-field plots.
# Keep this fixed so a given force field always maps to the same colour.
FORCE_FIELD_COLOR_PALETTE: tuple[str, ...] = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
)

# Explicit color overrides for force fields (raw keys, display names, or
# short keys derived from combined_force_field.offxml parent directory names).
# Fill this in to lock specific force fields to specific colours.
FORCE_FIELD_COLOR_MAP: dict[str, str] = {
    # Explicit mappings for key force fields.
    "AceFF 2.0": "#0072B2",
    "input_ff/aceff20.offxml": "#0072B2",
    "presto": "#E69F00",
    "benchmarking/tnet500/output/test/default/combined_force_field.offxml": "#E69F00",
    "benchmarking/jacs_fragments/output/test/default/combined_force_field.offxml": "#E69F00",
    "espaloma 0.4.0": "#009E73",
    "input_ff/esp04.offxml": "#009E73",
    "OpenFF 2.3.0": "#D55E00",
    "openff-2.3.0": "#D55E00",
    "OpenFF\nBespokeFit/\nB3LYP-D3BJ/DZVP": "#CC79A7",
    "input_ff/bespokefit1_sage_jacs_frags.offxml": "#CC79A7",
    # Ablation short keys (parent dir name from combined_force_field.offxml paths)
    "default": "#0072B2",
    "no_metad": "#009E73",
    "no_min": "#CC79A7",
    "one_it": "#E69F00",
    "no_reg": "#56B4E9",
}


def get_force_field_color_map(force_fields: list[str]) -> dict[str, str]:
    """Return a stable force-field-to-colour mapping for plotting."""
    ordered = sorted(force_fields)
    palette = list(FORCE_FIELD_COLOR_PALETTE)
    mapping: dict[str, str] = {}
    used_colors: set[str] = set()

    # Apply explicit overrides first, but avoid duplicates.
    # For combined_force_field.offxml paths not found by full path, also try
    # the parent directory name as a short key (e.g. "no_metad", "default").
    for force_field in ordered:
        color = FORCE_FIELD_COLOR_MAP.get(force_field)
        if color is None and force_field.endswith("/combined_force_field.offxml"):
            from pathlib import Path
            color = FORCE_FIELD_COLOR_MAP.get(Path(force_field).parent.name)
        if color is not None and color not in used_colors:
            mapping[force_field] = color
            used_colors.add(color)

    remaining_colors = [color for color in palette if color not in used_colors]
    unassigned = [ff for ff in ordered if ff not in mapping]

    for index, force_field in enumerate(unassigned):
        if remaining_colors:
            mapping[force_field] = remaining_colors.pop(0)
        else:
            mapping[force_field] = palette[index % len(palette)]

    return mapping


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

# ---------------------------------------------------------------------------
# TYK2 congeneric retrain display names
# ---------------------------------------------------------------------------

SAGE_TYPES_SENTINEL = -2


def get_max_extend_distance_label(max_extend_distance: int) -> str:
    """Return display label for max_extend_distance settings used in TYK2 analysis."""
    if max_extend_distance == SAGE_TYPES_SENTINEL:
        return "Sage Types"
    if max_extend_distance == -1:
        return "Whole Molecule"
    return str(max_extend_distance)
