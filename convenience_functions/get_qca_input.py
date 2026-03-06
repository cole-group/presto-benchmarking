"""Functionality for downloading inputs from QCArchive."""

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union

import loguru
import qcelemental
from openff.qcsubmit.results import TorsionDriveResultCollection
from openff.qcsubmit.results.filters import SMILESFilter
from qcelemental.models.procedures import TorsionDriveResult
from qcportal import PortalClient
from yammbs.torsion.inputs import QCArchiveTorsionDataset, QCArchiveTorsionProfile

HARTREE2KCALMOL = qcelemental.constants.hartree2kcalmol
BOHR2ANGSTROMS = qcelemental.constants.bohr2angstroms


def get_qca_torsion_input(
    dataset_name: str,
    json_output_path: Path,
    spec_name: str = "default",
    exclude_smiles: list[str] | None = None,
) -> None:
    """Download a torsion dataset from QCArchive and save it as JSON.

    Parameters
    ----------
    exclude_smiles:
        SMILES of molecules to exclude before saving.  Useful for removing
        known-bad molecules from a dataset.
    """
    cache_dir = json_output_path.parent / ".cache"
    cache_dir.mkdir(exist_ok=True, parents=True)

    client = PortalClient("https://api.qcarchive.molssi.org:443", cache_dir=cache_dir)

    dataset = TorsionDriveResultCollection.from_server(
        client=client,
        datasets=dataset_name,
        spec_name=spec_name,
    )

    if exclude_smiles:
        n_before = sum(len(recs) for recs in dataset.entries.values())
        dataset = dataset.filter(SMILESFilter(smiles_to_exclude=exclude_smiles))
        n_after = sum(len(recs) for recs in dataset.entries.values())
        loguru.logger.info(
            f"Excluded {n_before - n_after} records by SMILES; {n_after} records remain."
        )

    dataset = QCArchiveTorsionDataset.from_qcsubmit_collection(dataset)

    with open(json_output_path, "w") as f:
        f.write(dataset.model_dump_json())


def download_tnet_500_spice_lot(output_dir: Path) -> None:
    """Download the TNet500_minimal dataset (SPICE LOT) which was
    computed for the MACE-OFF paper."""
    abs_output_dir = output_dir.resolve()
    with TemporaryDirectory() as temp_dir:
        # Tmp dir to download the zip to
        cmds = [
            "curl",
            "https://zenodo.org/records/11385284/files/TNet500_minimal.zip?download=1",
            "-o",
            "TNet500.zip",
        ]
        subprocess.run(cmds, cwd=temp_dir, check=True)
        # Unzip the file to the output directory
        cmds = ["unzip", "TNet500.zip", "-d", str(abs_output_dir)]
        subprocess.run(cmds, cwd=temp_dir, check=True)


def convert_to_qca_torsion_dataset(
    input_dir: Union[str, Path], selected_smiles: Optional[list[str]] = None
) -> QCArchiveTorsionDataset:
    """Convert a directory of TorsionDriveResult JSON files to a QCArchiveTorsionDataset."""

    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"The directory {input_dir} does not exist.")

    qm_torsions = []

    # Order the files in the directory
    input_files = sorted(input_dir.glob("*.json"))
    assert len(input_files) > 0, f"No JSON files found in directory {input_dir}."

    for i, json_file in enumerate(input_files):
        result = TorsionDriveResult.parse_file(json_file)

        # Check if the smiles string is in the provided list
        smiles = result.initial_molecule[0].extras[
            "canonical_isomeric_explicit_hydrogen_mapped_smiles"
        ]
        if selected_smiles is not None and smiles not in selected_smiles:
            continue

        assert len(result.keywords.dihedrals) == 1, "Exactly one dihedral is expected."
        dihedral_indices = result.keywords.dihedrals[0]

        coordinates, energies = {}, {}
        for angle, molecule in result.final_molecules.items():
            coordinates[angle] = molecule.geometry * BOHR2ANGSTROMS
            energies[angle] = result.final_energies[angle] * HARTREE2KCALMOL

        qm_torsions.append(
            QCArchiveTorsionProfile(
                id=i,
                qcarchive_id=i,
                mapped_smiles=smiles,
                dihedral_indices=dihedral_indices,
                coordinates=coordinates,
                energies=energies,
            )
        )

    return QCArchiveTorsionDataset(qm_torsions=qm_torsions)


def get_tnet_500_spice_lot_qca_input(json_output_path: Path) -> None:
    """Download the TNet500_minimal dataset (SPICE LOT) and convert it to QCArchiveTorsionDataset JSON."""
    output_dir = json_output_path.parent / "tnet500"
    download_tnet_500_spice_lot(output_dir=output_dir)
    dataset = convert_to_qca_torsion_dataset(input_dir=output_dir / "TNet500_minimal")
    with open(json_output_path, "w") as f:
        f.write(dataset.json())
