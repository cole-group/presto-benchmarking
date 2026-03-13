"""Utilities to download Folmsbee/Hutchison conformer-benchmark input.

Paper: https://onlinelibrary.wiley.com/doi/full/10.1002/qua.26381
Repository: https://github.com/ghutchis/conformer-benchmark
"""

from pathlib import Path
from subprocess import run
import loguru
import networkx as nx
import numpy as np
from openbabel import openbabel as ob
from openbabel import pybel
from openff.toolkit import Molecule
from openff.units import unit

logger = loguru.logger

_FOLMSBEE_REPO_URL = "https://github.com/ghutchis/conformer-benchmark.git"
_JOB_SUBDIRS = ("Neutral_jobs", "CHG_jobs")


def _read_smiles_by_id(molecules_smi: Path) -> dict[str, str]:
    smiles_by_id: dict[str, str] = {}
    with open(molecules_smi) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                continue
            smiles_by_id[parts[1]] = parts[0]
    return smiles_by_id


def _mol_file_to_graph(mol_path: Path) -> nx.Graph:
    """Build an element/connectivity graph from a MOL file via OpenBabel.

    This intentionally ignores bond order, which is often unreliable in these
    inputs for aromatic systems.
    """
    pb_mol = next(pybel.readfile("mol", str(mol_path)))
    obmol = pb_mol.OBMol
    graph = nx.Graph()
    for atom in ob.OBMolAtomIter(obmol):
        graph.add_node(atom.GetIdx(), symbol=ob.GetSymbol(atom.GetAtomicNum()))
    for bond in ob.OBMolBondIter(obmol):
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return graph


def _smiles_to_graph(mol: Molecule) -> nx.Graph:
    graph = nx.Graph()
    for atom in mol.atoms:
        graph.add_node(atom.molecule_atom_index, symbol=atom.symbol)
    for bond in mol.bonds:
        graph.add_edge(bond.atom1_index, bond.atom2_index)
    return graph


def _read_xyz_coords(xyz_path: Path) -> np.ndarray:
    """Read XYZ coordinates directly as float64 without unit conversion.

    Values are parsed once from text to IEEE-754 double precision to avoid
    additional precision loss.
    """
    lines = xyz_path.read_text().splitlines()
    n_atoms = int(lines[0].strip())
    coord_lines = lines[2 : 2 + n_atoms]
    if len(coord_lines) != n_atoms:
        raise ValueError(f"Malformed XYZ file (atom count mismatch): {xyz_path}")
    return np.array(
        [
            [float(parts[1]), float(parts[2]), float(parts[3])]
            for parts in (line.split() for line in coord_lines)
        ],
        dtype=np.float64,
    )


def _molecule_from_smiles_with_xyz_via_mol(
    smiles: str,
    mol_path: Path,
    xyz_path: Path,
) -> Molecule:
    """Build an OpenFF molecule from trusted SMILES and mapped XYZ coordinates.
    """
    reference_molecule = Molecule.from_smiles(
        smiles,
        allow_undefined_stereo=True,
    )
    mol_graph = _mol_file_to_graph(mol_path)
    smiles_graph = _smiles_to_graph(reference_molecule)

    if mol_graph.number_of_nodes() != smiles_graph.number_of_nodes():
        raise ValueError(
            f"Atom-count mismatch for {mol_path}: mol has "
            f"{mol_graph.number_of_nodes()} atoms while smiles has "
            f"{smiles_graph.number_of_nodes()}"
        )

    node_match = nx.algorithms.isomorphism.categorical_node_match(
        "symbol",
        None,
    )
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        smiles_graph,
        mol_graph,
        node_match=node_match,
    )
    if not matcher.is_isomorphic():
        raise ValueError(
            "Connectivity graph not isomorphic to trusted SMILES: "
            f"{mol_path}"
        )

    # mapping: reference atom index (0-based) -> mol/xyz atom index (1-based)
    atom_mapping = matcher.mapping

    xyz_coords_angstrom = _read_xyz_coords(xyz_path)
    if xyz_coords_angstrom.shape[0] != reference_molecule.n_atoms:
        raise ValueError(
            f"XYZ atom-count mismatch for {xyz_path}: xyz has "
            f"{xyz_coords_angstrom.shape[0]} atoms while smiles has "
            f"{reference_molecule.n_atoms}"
        )

    reordered_coords = np.zeros((reference_molecule.n_atoms, 3), dtype=float)
    for reference_idx, file_idx in atom_mapping.items():
        reordered_coords[reference_idx] = xyz_coords_angstrom[file_idx - 1]

    output_molecule = Molecule(reference_molecule)
    output_molecule.add_conformer(reordered_coords * unit.angstrom)
    return output_molecule


def download_folmsbee_from_gh(path: Path) -> None:
    """Clone the conformer-benchmark repository into *path*."""
    if (path / ".git").exists():
        return

    if path.exists() and any(path.iterdir()):
        raise RuntimeError(
            "Cannot clone into non-empty directory that is not a git repo: "
            f"{path}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", _FOLMSBEE_REPO_URL, str(path)], check=True)


def write_folmsbee_smiles_files(molecules_smi: Path, output_dir: Path) -> None:
    """Process Folmsbee conformers and write per-molecule .smi for successes.

    For each molecule in Neutral/CHG geometry directories:
    - Build topology from trusted SMILES in ``molecules.smi``
    - Map coordinates from paired ``.mol``/``.xyz`` inputs
    - Write ``*.openff.sdf`` files alongside source geometry files, but only if
      every conformer for that molecule succeeds.
    - Write ``<molecule_id>.smi`` to ``output_dir`` only for fully successful
      molecules.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    smiles_by_id = _read_smiles_by_id(molecules_smi)
    repo_root = molecules_smi.parent.parent
    geometry_root = repo_root / "geometries"

    molecule_dirs: list[Path] = []
    for job_subdir in _JOB_SUBDIRS:
        subdir_path = geometry_root / job_subdir
        if not subdir_path.exists():
            continue
        for molecule_dir in sorted(subdir_path.iterdir()):
            if molecule_dir.is_dir():
                molecule_dirs.append(molecule_dir)

    logger.info(
        f"Found {len(molecule_dirs)} molecule directories under {geometry_root}"
    )

    successful_molecules = 0
    failed_molecules = 0
    successful_conformers = 0
    failures: list[tuple[str, str]] = []

    for molecule_dir in molecule_dirs:
        molecule_id = molecule_dir.name
        molecule_smiles = smiles_by_id.get(molecule_id)
        molecule_errors: list[str] = []
        built_outputs: list[tuple[Path, Molecule]] = []

        if molecule_smiles is None:
            molecule_errors.append("missing trusted SMILES in molecules.smi")
        else:
            try:
                Molecule.from_smiles(
                    molecule_smiles,
                    allow_undefined_stereo=True,
                )
            except Exception as error:
                molecule_errors.append(
                    f"trusted SMILES failed to parse: {error}"
                )

        mol_files = sorted(molecule_dir.glob("*.mol"))
        if not mol_files:
            molecule_errors.append("no .mol files found")

        if not molecule_errors and molecule_smiles is not None:
            for mol_file in mol_files:
                xyz_file = mol_file.with_suffix(".xyz")
                if not xyz_file.exists():
                    molecule_errors.append(
                        f"{mol_file.name}: missing paired .xyz"
                    )
                    continue

                try:
                    out_molecule = _molecule_from_smiles_with_xyz_via_mol(
                        smiles=molecule_smiles,
                        mol_path=mol_file,
                        xyz_path=xyz_file,
                    )
                    out_sdf_path = mol_file.with_suffix(".openff.sdf")
                    built_outputs.append((out_sdf_path, out_molecule))
                except Exception as error:
                    molecule_errors.append(f"{mol_file.name}: {error}")

        if molecule_errors:
            failed_molecules += 1
            for reason in molecule_errors:
                failures.append((molecule_id, reason))
            continue

        for out_sdf_path, out_molecule in built_outputs:
            out_molecule.to_file(str(out_sdf_path), file_format="sdf")
            successful_conformers += 1

        (output_dir / f"{molecule_id}.smi").write_text(f"{molecule_smiles}\n")
        successful_molecules += 1

    logger.info(
        f"Folmsbee processing complete: {successful_molecules} / "
        f"{len(molecule_dirs)} molecules succeeded; {failed_molecules} failed"
    )
    logger.info(
        f"Wrote {successful_conformers} mapped conformer SDF files and "
        f"{successful_molecules} .smi files to {output_dir}"
    )

    if failures:
        logger.warning(
            f"Encountered {len(failures)} molecule-level errors (all shown below):"
        )
        for molecule_id, reason in failures:
            logger.warning(f"{molecule_id}: {reason}")

