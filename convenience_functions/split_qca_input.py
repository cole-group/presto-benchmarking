"""Functionality to split QCArchive inputs into validation and test sets.

The JSON outputs contain all molecules (including duplicates, as multiple torsions
may reference the same molecule). The SMILES outputs (CSV and individual .smi files)
are deduplicated: two entries are considered the same molecule if their canonical
SMILES strings match after stripping atom-map numbers from the mapped SMILES.

Split behaviour
---------------
When ``frac_test=1.0`` the MaxMin splitter is **not** called at all: every entry
is assigned to the test set and the validation set is empty. This avoids a
DeepChem quirk whereby ``train_test_split`` always reserves at least one sample
for its internal "test" bucket — which, because the train/test roles are swapped
here, would silently leak one molecule into the validation set even when the
caller explicitly requests no validation split.
"""

import io
import json
import re
from pathlib import Path

import deepchem as dc
import loguru
import numpy as np
import pandas as pd
from openff.toolkit import Molecule
from rdkit.Chem.Draw import rdMolDraw2D
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Table,
    TableStyle,
)

JSON_FILE = "TNet500_minimal_dataset.json"
SMILES_NAME = "smiles.csv"
FRAC_TEST = 0.80  # We'll use the 20 % = 100 molecules for validation
SEED = 0
VALIDATION_OUTPUT_PATH = Path("validation_set")
TEST_OUTPUT_PATH = Path("test_set")

logger = loguru.logger


# ---------------------------------------------------------------------------
# SMILES utilities
# ---------------------------------------------------------------------------


def strip_atom_map_numbers(mapped_smiles: str) -> str:
    """Remove atom-map number tags from a mapped SMILES string.

    Atom-map numbers appear as colon-suffixed integers inside square brackets,
    e.g. ``[C:1]`` → ``[C]``, ``[NH2:3]`` → ``[NH2]``. Brackets that become
    redundant after stripping (i.e. they wrap a single, uncharged, unisotoped
    organic-subset atom with no chirality) are **not** collapsed here; RDKit
    canonicalisation downstream handles normalisation.

    Args:
        mapped_smiles: A SMILES string that may contain atom-map tags such as
            ``[C:1]``, ``[nH:4]``, etc.

    Returns:
        The same SMILES string with all ``:n`` atom-map tags removed.

    Example:
        >>> strip_atom_map_numbers("[CH3:1][C:2](=[O:3])[NH2:4]")
        '[CH3][C](=[O])[NH2]'
    """
    return re.sub(r":(\d+)(?=])", "", mapped_smiles)


def canonical_smiles_from_mapped(mapped_smiles: str) -> str:
    """Return a canonical (unmapped) SMILES string for duplicate detection.

    Strips atom-map numbers then round-trips the SMILES through the OpenFF
    Toolkit to obtain a canonical representation that is independent of atom
    ordering and map numbering.

    Args:
        mapped_smiles: A mapped SMILES string, e.g. as stored in the QCArchive
            JSON under ``"mapped_smiles"``.

    Returns:
        A canonical SMILES string suitable for equality comparison.

    Raises:
        Exception: Propagates any OpenFF Toolkit parsing errors so that
            malformed entries surface early rather than being silently skipped.
    """
    stripped = strip_atom_map_numbers(mapped_smiles)
    mol = Molecule.from_smiles(stripped, allow_undefined_stereo=True)
    return mol.to_smiles()


def get_unique_ids(smiles_df: pd.DataFrame) -> list[int]:
    """Return the IDs of the first occurrence of each unique molecule.

    Uniqueness is determined by the canonical SMILES obtained after removing
    atom-map numbers from the ``"smiles"`` column. When the same molecule
    appears more than once (e.g. because multiple distinct torsions were
    scanned for it), only the entry with the lowest ``"id"`` value is kept
    for the SMILES outputs.

    .. note::
        The JSON dataset outputs are **not** deduplicated by this function;
        all torsion entries are preserved there regardless of molecular
        identity.

    Args:
        smiles_df: DataFrame with at minimum ``"id"`` and ``"smiles"``
            columns, as returned by :func:`load_smiles`.

    Returns:
        Sorted list of integer IDs corresponding to unique molecules.
    """
    seen_canonical: dict[str, int] = {}  # canonical_smiles -> first id
    unique_ids: list[int] = []

    for _, row in smiles_df.iterrows():
        try:
            canonical = canonical_smiles_from_mapped(row["smiles"])
        except Exception as exc:
            logger.warning(
                f"Could not canonicalise SMILES for id={row['id']} "
                f"(smiles={row['smiles']!r}); skipping duplicate check "
                f"and treating as unique. Error: {exc}"
            )
            unique_ids.append(row["id"])
            continue

        if canonical not in seen_canonical:
            seen_canonical[canonical] = row["id"]
            unique_ids.append(row["id"])

    n_duplicates = len(smiles_df) - len(unique_ids)
    logger.info(
        f"Deduplication complete: {len(unique_ids)} unique molecules from "
        f"{len(smiles_df)} total entries ({n_duplicates} duplicate(s) removed)."
    )
    return sorted(unique_ids)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_smiles(json_file: str) -> pd.DataFrame:
    """Load molecule data from a QCArchive-format JSON file.

    Reads the ``"qm_torsions"`` list and extracts the mapped SMILES string,
    numeric ID, and dihedral atom indices for each entry.

    Args:
        json_file: Path to the JSON file to read.

    Returns:
        DataFrame with columns ``"id"`` (int), ``"smiles"`` (str, mapped),
        and ``"torsion_idx"`` (list of four ints).
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    smiles_list = [entry["mapped_smiles"] for entry in data["qm_torsions"]]
    id_list = [entry["id"] for entry in data["qm_torsions"]]
    torsion_idxs = [entry["dihedral_indices"] for entry in data["qm_torsions"]]

    logger.info(f"Loaded {len(id_list)} torsion entries from '{json_file}'.")
    return pd.DataFrame(
        {"id": id_list, "smiles": smiles_list, "torsion_idx": torsion_idxs}
    )


def save_individual_smiles_files(
    ids: list[int], smiles_df: pd.DataFrame, output_dir: Path
) -> None:
    """Save one ``.smi`` file per molecule for the given subset of IDs.

    Only IDs present in *ids* are written. Each file is named ``<id>.smi``
    and contains the (mapped) SMILES string followed by a newline.

    .. note::
        Pass deduplicated IDs (from :func:`get_unique_ids`) to avoid writing
        redundant files for molecules that appear under multiple torsion entries.

    Args:
        ids: Iterable of integer IDs to include.
        smiles_df: Full molecules DataFrame (columns: ``"id"``, ``"smiles"``,
            ``"torsion_idx"``).
        output_dir: Directory in which to create the ``.smi`` files.
            Created automatically if it does not exist.
    """
    output_dir.mkdir(exist_ok=True)
    subset_df = smiles_df[smiles_df["id"].isin(ids)]
    for _, row in subset_df.iterrows():
        id_ = row["id"]
        smiles = row["smiles"]
        output_file = output_dir / f"{id_}.smi"
        with open(output_file, "w") as f:
            f.write(smiles + "\n")

    logger.info(f"Wrote {len(subset_df)} individual .smi files to '{output_dir}'.")


def save_torsion_img(ids: list[int], smiles_df: pd.DataFrame, filename: Path) -> None:
    """Render a multi-page PDF grid of molecules with highlighted torsion atoms.

    Saves to a ``.pdf`` file (the extension is corrected automatically if a
    ``.png`` path is passed) to avoid RDKit/Cairo buffer-parsing issues that
    arise when compositing large numbers of molecules into a single raster
    image.

    Each molecule is rendered individually at 300×300 px via
    ``MolDraw2DCairo`` — bypassing the broken ``MolsToGridImage`` /
    ``_drawerToImage`` PIL code path — and the resulting PNG bytes are
    embedded into a multi-page A4 PDF using ReportLab.

    Args:
        ids: IDs of the molecules to include.
        smiles_df: Full molecules DataFrame (columns: ``"id"``, ``"smiles"``,
            ``"torsion_idx"``).
        filename: Destination path. Any extension is replaced with ``.pdf``.
    """
    MOLS_PER_ROW = 4
    SUB_IMG_SIZE = (300, 300)

    filename = filename.with_suffix(".pdf")
    subset_df = smiles_df[smiles_df["id"].isin(ids)].reset_index(drop=True)

    logger.info(f"Rendering {len(subset_df)} molecules to PDF '{filename}'...")

    # Render each molecule individually to PNG bytes via MolDraw2DCairo.
    # We do NOT use MolsToGridImage because on Cairo builds it calls
    # _drawerToImage which tries to reparse the Cairo buffer as a PNG and
    # raises "PNG header not recognized".
    mol_pngs: list[bytes] = []
    for _, row in subset_df.iterrows():
        mol = Molecule.from_mapped_smiles(
            row["smiles"], allow_undefined_stereo=True
        ).to_rdkit()
        highlight = list(row["torsion_idx"])

        d2d = rdMolDraw2D.MolDraw2DCairo(*SUB_IMG_SIZE)
        d2d.DrawMolecule(mol, highlightAtoms=highlight)
        d2d.FinishDrawing()
        mol_pngs.append(d2d.GetDrawingText())

    logger.info(f"Molecule rendering complete, building PDF layout...")

    # Build PDF with ReportLab
    page_width, page_height = A4
    margin = 10 * mm
    img_w = (page_width - 2 * margin) / MOLS_PER_ROW
    img_h = img_w  # square cells

    doc = SimpleDocTemplate(
        str(filename),
        pagesize=A4,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
    )

    styles = getSampleStyleSheet()
    caption_style = styles["Normal"]
    caption_style.fontSize = 7
    caption_style.alignment = 1  # centre

    legends = [f"ID: {id_}" for id_ in subset_df["id"]]
    rows = []
    current_row_imgs: list = []
    current_row_caps: list = []

    for i, (png_bytes, legend) in enumerate(zip(mol_pngs, legends)):
        rl_img = RLImage(io.BytesIO(png_bytes), width=img_w, height=img_h)
        current_row_imgs.append(rl_img)
        current_row_caps.append(Paragraph(legend, caption_style))

        if len(current_row_imgs) == MOLS_PER_ROW or i == len(mol_pngs) - 1:
            # Pad the final incomplete row so the table stays rectangular
            while len(current_row_imgs) < MOLS_PER_ROW:
                current_row_imgs.append("")
                current_row_caps.append("")
            rows.append(current_row_imgs)
            rows.append(current_row_caps)
            current_row_imgs = []
            current_row_caps = []

    col_widths = [img_w] * MOLS_PER_ROW
    table = Table(rows, colWidths=col_widths)
    table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ]
        )
    )
    doc.build([table])

    logger.info(f"Saved torsion PDF ({len(subset_df)} molecules) to '{filename}'.")


def save_smiles(ids: list[int], smiles_df: pd.DataFrame, filename: Path) -> None:
    """Save a CSV file containing SMILES and metadata for the given IDs.

    The output CSV has the same columns as *smiles_df* (``"id"``, ``"smiles"``,
    ``"torsion_idx"``), filtered to rows whose ``"id"`` is in *ids*.

    .. note::
        Pass deduplicated IDs (from :func:`get_unique_ids`) when the intent
        is to write one row per unique molecule.

    Args:
        ids: IDs to include in the output file.
        smiles_df: Full molecules DataFrame.
        filename: Destination path for the CSV file.
    """
    subset_df = smiles_df[smiles_df["id"].isin(ids)]
    subset_df.to_csv(filename, index=False)
    logger.info(f"Saved SMILES CSV ({len(subset_df)} rows) to '{filename}'.")


def save_sub_dataset(ids: list[int], json_file: Path, output_file: Path) -> None:
    """Write a JSON file containing all torsion entries for the given IDs.

    The output preserves the full structure of the source JSON (including all
    QM torsion scan data) for every entry whose ``"id"`` is present in *ids*.
    Duplicate molecules are **not** removed here; every torsion entry is kept
    so that downstream QM benchmarking tools receive complete data.

    Args:
        ids: IDs of the torsion entries to include.
        json_file: Path to the source JSON file.
        output_file: Destination path for the filtered JSON file.
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    subset_data: dict = {"qm_torsions": []}
    for entry in data["qm_torsions"]:
        if entry["id"] in ids:
            subset_data["qm_torsions"].append(entry)

    with open(output_file, "w") as f:
        json.dump(subset_data, f, indent=2)
    logger.info(
        f"Saved sub-dataset JSON ({len(subset_data['qm_torsions'])} entries) "
        f"to '{output_file}'."
    )


def save_sub_dataset_by_smiles(
    input_json_path: Path,
    selected_smiles_csv_path: Path,
    output_json_path: Path,
) -> None:
    """Save a JSON sub-dataset by matching canonical SMILES from a split CSV.

    This is used to transfer an existing train/validation split onto a new dataset
    version by preserving molecule membership defined by ``smiles.csv``.
    """
    selected_df = pd.read_csv(selected_smiles_csv_path)
    if "smiles" not in selected_df.columns:
        raise ValueError(
            f"Expected a 'smiles' column in {selected_smiles_csv_path}, "
            f"found columns: {list(selected_df.columns)}"
        )

    selected_canonical_smiles = {
        canonical_smiles_from_mapped(smiles) for smiles in selected_df["smiles"]
    }

    with open(input_json_path, "r") as f:
        data = json.load(f)

    subset_entries = []
    found_canonical_smiles: set[str] = set()
    for entry in data["qm_torsions"]:
        canonical_smiles = canonical_smiles_from_mapped(entry["mapped_smiles"])
        if canonical_smiles in selected_canonical_smiles:
            subset_entries.append(entry)
            found_canonical_smiles.add(canonical_smiles)

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump({"qm_torsions": subset_entries}, f, indent=2)

    missing_smiles = selected_canonical_smiles - found_canonical_smiles
    if missing_smiles:
        logger.warning(
            f"Found {len(found_canonical_smiles)} / {len(selected_canonical_smiles)} canonical "
            f"SMILES from split file in {input_json_path}. "
            f"Missing {len(missing_smiles)} molecule(s) in the source dataset."
        )

    logger.info(
        f"Saved split-by-smiles sub-dataset JSON ({len(subset_entries)} entries) "
        f"to '{output_json_path}'."
    )


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_dataset_maxmin(
    smiles_df: pd.DataFrame, frac_train: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split molecule IDs into test and validation subsets using MaxMin diversity.

    Uses DeepChem's :class:`~deepchem.splits.MaxMinSplitter` to maximise
    chemical diversity within each subset. The splitter operates on the
    full (possibly duplicated) set of torsion entries so that the split
    proportions reflect the requested fractions of *torsion entries*, not
    unique molecules.

    .. note::
        The roles of ``train`` and ``test`` inside DeepChem are intentionally
        swapped here: the large ``train`` split becomes our **test set** and
        the small ``test`` split becomes our **validation set**.

    .. warning::
        This function must **not** be called with ``frac_train=1.0``.
        DeepChem's splitter always reserves at least one sample for its
        internal "test" bucket regardless of ``frac_train``, which would leak
        one molecule into the validation set. The caller (:func:`create_validation_and_test_sets`)
        handles the ``frac_test=1.0`` case by skipping this function entirely.

    Args:
        smiles_df: DataFrame with ``"id"`` and ``"smiles"`` columns.
        frac_train: Fraction of entries to place in the test set (the larger
            split). Must be strictly less than 1.0. The validation set receives
            ``1 - frac_train``.
        seed: Random seed for the MaxMin splitter, ensuring reproducibility.

    Returns:
        A two-tuple ``(test_ids, validation_ids)`` where each element is a
        NumPy array of integer IDs.
    """
    logger.info(
        f"Splitting {len(smiles_df)} entries with MaxMinSplitter "
        f"(frac_test={frac_train:.2f}, seed={seed})."
    )
    splitter = dc.splits.MaxMinSplitter()

    dc_dataset = dc.data.DiskDataset.from_numpy(
        X=smiles_df.id,
        ids=smiles_df.smiles,
    )

    # DeepChem's "train" → our test set (large fraction)
    # DeepChem's "test"  → our validation set (small fraction)
    test_dataset, valid_dataset = splitter.train_test_split(
        dc_dataset,
        frac_train=frac_train,
        seed=seed,
    )
    return test_dataset.X, valid_dataset.X


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def create_validation_and_test_sets(
    input_json_path: Path,
    seed: int,
    test_output_path: Path,
    frac_test: float = 1.0,
    validation_output_path: Path | None = None,
) -> None:
    """Split the input dataset into validation and test sets and save them.

    The JSON outputs preserve **all** torsion entries (including multiple entries
    for the same molecule with different dihedral angles). The SMILES outputs
    (CSV and individual ``.smi`` files) are **deduplicated**: if the same molecule
    appears under several torsion IDs, only the first-encountered entry is written.
    Deduplication is based on canonical SMILES derived by stripping atom-map numbers
    from the mapped SMILES strings.

    Split logic
    ~~~~~~~~~~~
    When ``frac_test=1.0`` the MaxMin splitter is skipped entirely and all
    entries go to the test set. This is necessary because DeepChem's
    ``train_test_split`` always reserves at least one sample for its "test"
    bucket, which would silently leak one molecule into the validation set even
    when no validation split is requested.

    When ``frac_test < 1.0`` the MaxMin splitter is used normally. The caller
    must also supply ``validation_output_path``; if it is ``None`` the
    validation IDs are computed but not saved (a warning is emitted).

    Output files
    ~~~~~~~~~~~~
    Test set (always written):

    - ``test_output_path/test.json`` — all torsion entries for the test split.
    - ``test_output_path/test_set_torsions.pdf`` — PDF grid of all test torsions.
    - ``test_output_path/smiles.csv`` — unique molecules only (CSV).
    - ``test_output_path/smiles/<id>.smi`` — one file per unique molecule.

    Validation set (written only when *validation_output_path* is provided and
    ``frac_test < 1.0``):

    - ``validation_output_path/validation.json`` — all torsion entries for the
      validation split.
    - ``validation_output_path/validation_set_torsions.pdf`` — PDF grid.
    - ``validation_output_path/smiles.csv`` — unique molecules only (CSV).
    - ``validation_output_path/smiles/<id>.smi`` — one file per unique molecule.

    Args:
        input_json_path: Path to the input JSON file containing the full dataset.
        seed: Random seed passed to the MaxMin splitter for reproducibility.
        test_output_path: Directory in which to save test set outputs.
        frac_test: Fraction of torsion entries to assign to the test set.
            Must be in the range ``(0, 1]``. When ``1.0`` (the default), all
            entries go to the test set and no validation set is produced.
        validation_output_path: Directory in which to save validation set
            outputs. Ignored (with a warning) when ``frac_test=1.0``.
            If ``None`` and ``frac_test < 1.0``, validation IDs are computed
            but not saved to disk.
    """
    if not 0 < frac_test <= 1.0:
        raise ValueError(f"frac_test must be in (0, 1], got {frac_test}")

    logger.info(f"Loading dataset from '{input_json_path}'.")
    smiles_df = load_smiles(json_file=str(input_json_path))

    # ------------------------------------------------------------------
    # Split — bypass DeepChem entirely when frac_test=1.0 to avoid the
    # "always reserves ≥1 sample for test bucket" quirk that would leak
    # one molecule into valid_inds.
    # ------------------------------------------------------------------
    if frac_test == 1.0:
        logger.info(
            "frac_test=1.0: assigning all entries to the test set. "
            "The MaxMin splitter is not called (it always reserves ≥1 sample "
            "for its internal test bucket, which would leak into validation)."
        )
        if validation_output_path is not None:
            logger.warning(
                "validation_output_path was provided but frac_test=1.0, "
                "so no validation set will be produced."
            )
        test_inds: np.ndarray = smiles_df["id"].to_numpy()
        valid_inds: np.ndarray = np.array([], dtype=test_inds.dtype)
    else:
        logger.info(
            f"Splitting dataset into test ({frac_test:.0%}) "
            f"and validation ({1 - frac_test:.0%}) subsets."
        )
        test_inds, valid_inds = split_dataset_maxmin(smiles_df, frac_test, seed)

    # ------------------------------------------------------------------
    # Validation set
    # ------------------------------------------------------------------
    unique_valid_ids: list[int] = []
    if len(valid_inds) > 0 and validation_output_path is not None:
        logger.info(f"Processing validation set ({len(valid_inds)} torsion entries).")
        validation_output_path.mkdir(exist_ok=True)

        valid_smiles_df = smiles_df[smiles_df["id"].isin(valid_inds)]
        unique_valid_ids = get_unique_ids(valid_smiles_df)
        logger.info(
            f"Validation set: {len(valid_inds)} torsion entries → "
            f"{len(unique_valid_ids)} unique molecules for SMILES outputs."
        )

        save_torsion_img(
            valid_inds,
            smiles_df,
            validation_output_path / "validation_set_torsions.pdf",
        )
        save_smiles(unique_valid_ids, smiles_df, validation_output_path / "smiles.csv")
        save_individual_smiles_files(
            unique_valid_ids, smiles_df, validation_output_path / "smiles"
        )
        save_sub_dataset(
            valid_inds, input_json_path, validation_output_path / "validation.json"
        )
    elif len(valid_inds) > 0 and validation_output_path is None:
        logger.warning(
            f"{len(valid_inds)} entries were assigned to the validation split "
            "but validation_output_path=None, so they will not be saved."
        )

    # ------------------------------------------------------------------
    # Test set
    # ------------------------------------------------------------------
    logger.info(f"Processing test set ({len(test_inds)} torsion entries).")
    test_output_path.mkdir(exist_ok=True)

    test_smiles_df = smiles_df[smiles_df["id"].isin(test_inds)]
    unique_test_ids = get_unique_ids(test_smiles_df)
    logger.info(
        f"Test set: {len(test_inds)} torsion entries → "
        f"{len(unique_test_ids)} unique molecules for SMILES outputs."
    )

    save_torsion_img(test_inds, smiles_df, test_output_path / "test_set_torsions.pdf")
    save_smiles(unique_test_ids, smiles_df, test_output_path / SMILES_NAME)
    save_individual_smiles_files(
        unique_test_ids, smiles_df, test_output_path / "smiles"
    )
    save_sub_dataset(test_inds, input_json_path, test_output_path / "test.json")

    logger.info(
        f"Done. "
        f"Test set: {len(test_inds)} torsion entries ({len(unique_test_ids)} unique molecules). "
        f"Validation set: {len(valid_inds)} torsion entries "
        f"({len(unique_valid_ids)} unique molecules)."
    )
