"""Analyse molecular descriptors for smiles.csv inputs using RDKit.

This module provides strict validation and descriptor summary utilities for
benchmark dataset `smiles.csv` files.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

from convenience_functions._plotting_defaults import get_dataset_display_name

SMILES_COLUMN = "smiles"
PER_MOLECULE_OUTPUT_NAME = "smiles_descriptors.csv"
SUMMARY_OUTPUT_NAME = "smiles_descriptor_summary.csv"
MEAN_STD_OUTPUT_NAME = "smiles_descriptor_mean_std.csv"
PER_MOLECULE_LATEX_OUTPUT_NAME = "smiles_descriptors.tex"
SUMMARY_LATEX_OUTPUT_NAME = "smiles_descriptor_summary.tex"
MEAN_STD_LATEX_OUTPUT_NAME = "smiles_descriptor_mean_std.tex"
PLOTS_DIR_NAME = "smiles_descriptor_plots"
PLOT_OUTPUT_NAME = "descriptor_distributions.png"
AGGREGATE_OUTPUT_NAME = "smiles_descriptor_aggregate_mean_std.csv"
AGGREGATE_LATEX_OUTPUT_NAME = "smiles_descriptor_aggregate_mean_std.tex"

DESCRIPTOR_COLUMNS = [
    "heavy_atom_count",
    "rotatable_bond_count",
    "formal_charge",
    "exact_mol_wt",
    "hbond_donor_count",
    "hbond_acceptor_count",
    "ring_count",
    "aromatic_ring_count",
    "tpsa",
    "logp",
    "fraction_csp3",
]

SUMMARY_STATS = ["mean", "std", "min", "max"]

DESCRIPTOR_DISPLAY_NAMES = {
    "heavy_atom_count": "Heavy Atom Count",
    "rotatable_bond_count": "Rotatable Bond Count",
    "formal_charge": "Formal Charge",
    "exact_mol_wt": "Exact Molecular Weight",
    "hbond_donor_count": "H-Bond Donor Count",
    "hbond_acceptor_count": "H-Bond Acceptor Count",
    "ring_count": "Ring Count",
    "aromatic_ring_count": "Aromatic Ring Count",
    "tpsa": "Topological Polar Surface Area",
    "logp": "LogP",
    "fraction_csp3": "Fraction Csp3",
}

DESCRIPTOR_DECIMALS = {
    "heavy_atom_count": 0,
    "rotatable_bond_count": 0,
    "formal_charge": 0,
    "hbond_donor_count": 0,
    "hbond_acceptor_count": 0,
    "ring_count": 0,
    "aromatic_ring_count": 0,
    "exact_mol_wt": 2,
    "tpsa": 2,
    "logp": 2,
    "fraction_csp3": 2,
}


def _strip_atom_map_numbers(smiles: str) -> str:
    """Remove atom-map numbers from mapped SMILES strings."""
    return re.sub(r":(\d+)(?=])", "", smiles)


def _compute_net_formal_charge(mol: Chem.Mol) -> int:
    """Return the molecule net formal charge from atom formal charges."""
    return sum(atom.GetFormalCharge() for atom in mol.GetAtoms())


def _build_descriptor_row(smiles: str, canonical_smiles: str, mol: Chem.Mol) -> dict[str, int | float | str]:
    """Compute descriptor values for a single molecule."""
    return {
        "smiles": smiles,
        "canonical_smiles": canonical_smiles,
        "heavy_atom_count": mol.GetNumHeavyAtoms(),
        "rotatable_bond_count": Lipinski.NumRotatableBonds(mol),
        "formal_charge": _compute_net_formal_charge(mol),
        "exact_mol_wt": Descriptors.ExactMolWt(mol),
        "hbond_donor_count": Lipinski.NumHDonors(mol),
        "hbond_acceptor_count": Lipinski.NumHAcceptors(mol),
        "ring_count": rdMolDescriptors.CalcNumRings(mol),
        "aromatic_ring_count": rdMolDescriptors.CalcNumAromaticRings(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "logp": Crippen.MolLogP(mol),
        "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
    }


def _validate_and_parse_molecules(smiles_df: pd.DataFrame) -> tuple[list[Chem.Mol], list[str], list[str]]:
    """Validate all SMILES rows and return parsed molecules and canonical SMILES.

    Raises:
        ValueError: If invalid SMILES strings or duplicate molecules are found.
    """
    if SMILES_COLUMN not in smiles_df.columns:
        raise ValueError(
            f"Expected column '{SMILES_COLUMN}' in smiles.csv, "
            f"found columns: {list(smiles_df.columns)}"
        )

    smiles_values = smiles_df[SMILES_COLUMN].astype(str).tolist()
    if len(smiles_values) == 0:
        raise ValueError("smiles.csv contains no rows.")

    molecules: list[Chem.Mol] = []
    canonical_smiles_values: list[str] = []
    invalid_rows: list[str] = []

    for idx, smiles in enumerate(smiles_values):
        stripped = _strip_atom_map_numbers(smiles)
        mol = Chem.MolFromSmiles(stripped)
        if mol is None:
            invalid_rows.append(f"row={idx + 1}, smiles={smiles!r}")
            continue

        canonical = Chem.MolToSmiles(mol, canonical=True)
        molecules.append(mol)
        canonical_smiles_values.append(canonical)

    if invalid_rows:
        raise ValueError(
            "Invalid SMILES found in smiles.csv; refusing to continue:\n"
            + "\n".join(invalid_rows)
        )

    duplicate_rows: list[str] = []
    first_seen_idx: dict[str, int] = {}
    for idx, canonical in enumerate(canonical_smiles_values):
        if canonical in first_seen_idx:
            duplicate_rows.append(
                f"row={idx + 1} duplicates row={first_seen_idx[canonical] + 1} "
                f"(canonical_smiles={canonical})"
            )
        else:
            first_seen_idx[canonical] = idx

    if duplicate_rows:
        raise ValueError(
            "Duplicate molecules found in smiles.csv based on canonical SMILES; "
            "refusing to continue:\n"
            + "\n".join(duplicate_rows)
        )

    return molecules, smiles_values, canonical_smiles_values


def _build_summary_tables(descriptor_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build descriptor summary tables with mean and standard deviation."""
    summary_df = (
        descriptor_df[DESCRIPTOR_COLUMNS]
        .agg(["count", "mean", "std", "min", "max"])
        .transpose()
        .reset_index()
        .rename(columns={"index": "descriptor"})
    )
    summary_df["mean_pm_std"] = summary_df.apply(
        lambda row: f"{row['mean']:.4f} +/- {row['std']:.4f}", axis=1
    )

    mean_std_df = pd.DataFrame(
        {
            "descriptor": summary_df["descriptor"],
            "mean": summary_df["mean"],
            "std": summary_df["std"],
            "mean_pm_std": summary_df["mean_pm_std"],
        }
    )

    return summary_df, mean_std_df


def _write_dataframe_csv_and_latex(df: pd.DataFrame, csv_path: Path, tex_path: Path) -> None:
    """Write DataFrame to CSV and LaTeX outputs."""
    df.to_csv(csv_path, index=False)
    df.to_latex(tex_path, index=False, float_format="%.4f")


def _save_descriptor_plots(descriptor_df: pd.DataFrame, output_dir: Path) -> None:
    """Save descriptor distribution plots to disk."""
    plot_dir = output_dir / PLOTS_DIR_NAME
    plot_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(4, 3, figsize=(16, 16))
    axes_flat = axes.flatten()

    for idx, col in enumerate(DESCRIPTOR_COLUMNS):
        ax = axes_flat[idx]
        sns.histplot(descriptor_df[col], bins=20, kde=True, ax=ax, color="#1f77b4")
        ax.set_title(col)
        ax.set_xlabel(col)

    # Hide the final empty panel in 4x3 grid.
    axes_flat[-1].axis("off")

    fig.tight_layout()
    fig.savefig(plot_dir / PLOT_OUTPUT_NAME, dpi=200)
    plt.close(fig)


def analyse_smiles_file(smiles_csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Analyse a single smiles.csv file and save outputs beside it.

    Raises:
        ValueError: On invalid or duplicate SMILES entries.
    """
    smiles_df = pd.read_csv(smiles_csv_path)
    molecules, smiles_values, canonical_smiles_values = _validate_and_parse_molecules(
        smiles_df
    )

    rows = [
        _build_descriptor_row(smiles, canonical, mol)
        for smiles, canonical, mol in zip(
            smiles_values, canonical_smiles_values, molecules, strict=True
        )
    ]
    descriptor_df = pd.DataFrame(rows)

    summary_df, mean_std_df = _build_summary_tables(descriptor_df)

    output_dir = smiles_csv_path.parent
    _write_dataframe_csv_and_latex(
        descriptor_df,
        output_dir / PER_MOLECULE_OUTPUT_NAME,
        output_dir / PER_MOLECULE_LATEX_OUTPUT_NAME,
    )
    _write_dataframe_csv_and_latex(
        summary_df,
        output_dir / SUMMARY_OUTPUT_NAME,
        output_dir / SUMMARY_LATEX_OUTPUT_NAME,
    )
    _write_dataframe_csv_and_latex(
        mean_std_df,
        output_dir / MEAN_STD_OUTPUT_NAME,
        output_dir / MEAN_STD_LATEX_OUTPUT_NAME,
    )
    _save_descriptor_plots(descriptor_df, output_dir)

    return descriptor_df, summary_df, mean_std_df


def analyse_smiles_files(smiles_csv_paths: list[Path], output_dir: Path) -> pd.DataFrame:
    """Analyse multiple smiles.csv files and save aggregate mean/std summaries."""
    if not smiles_csv_paths:
        raise ValueError("No smiles.csv paths were provided for aggregate analysis.")

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_order: list[str] = []
    aggregate_rows: list[dict[str, int | float | str]] = []
    for smiles_csv_path in smiles_csv_paths:
        _, summary_df, _ = analyse_smiles_file(smiles_csv_path)

        path_parts = smiles_csv_path.parts
        dataset_label = str(smiles_csv_path)
        if len(path_parts) >= 4:
            dataset_name = path_parts[-4]
            dataset_type = path_parts[-2]
            dataset_label = get_dataset_display_name(dataset_name, dataset_type)

        dataset_order.append(dataset_label)

        descriptor_rows = summary_df.set_index("descriptor")
        for descriptor in DESCRIPTOR_COLUMNS:
            row = descriptor_rows.loc[descriptor]
            aggregate_rows.append(
                {
                    "dataset": dataset_label,
                    "descriptor": descriptor,
                    "mean": float(row["mean"]),
                    "std": float(row["std"]),
                    "min": float(row["min"]),
                    "max": float(row["max"]),
                }
            )

    aggregate_df = pd.DataFrame(aggregate_rows)

    aggregate_df.to_csv(output_dir / AGGREGATE_OUTPUT_NAME, index=False)

    pivot_df = (
        aggregate_df.set_index(["descriptor", "dataset"])[SUMMARY_STATS]
        .unstack("dataset")
        .swaplevel(0, 1, axis=1)
    )

    ordered_columns = pd.MultiIndex.from_product([dataset_order, SUMMARY_STATS])
    pivot_df = pivot_df.reindex(columns=ordered_columns)

    formatted_df = pd.DataFrame(index=pivot_df.index, columns=pivot_df.columns, dtype=object)
    for descriptor in pivot_df.index:
        decimals = DESCRIPTOR_DECIMALS.get(descriptor, 2)
        for dataset_label, stat_name in pivot_df.columns:
            value = pivot_df.loc[descriptor, (dataset_label, stat_name)]
            formatted_df.loc[descriptor, (dataset_label, stat_name)] = (
                "" if pd.isna(value) else f"{float(value):.{decimals}f}"
            )

    formatted_df.index = pd.Index(
        [
            DESCRIPTOR_DISPLAY_NAMES.get(
                descriptor, descriptor.replace("_", " ").title()
            )
            for descriptor in formatted_df.index
        ],
        name="Descriptor",
    )

    n_data_columns = len(formatted_df.columns)
    latex_content = formatted_df.to_latex(
        escape=False,
        multicolumn=True,
        column_format="l" + ("r" * n_data_columns),
    )
    (output_dir / AGGREGATE_LATEX_OUTPUT_NAME).write_text(latex_content)

    return aggregate_df
