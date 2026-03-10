"""Minimization analysis for 2D protein torsion validation.

Slightly modified from Chapin Cavendar's script:
https://raw.githubusercontent.com/openforcefield/protein-param-fit/refs/heads/sage-2.1/validation/torsiondrive/2-run-torsiondrive-mm-minimizations.py
"""

import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Tuple

import numpy as np
import openmm
from openff.qcsubmit.results import TorsionDriveResultCollection
from openff.toolkit import ForceField, Molecule, ToolkitRegistry, RDKitToolkitWrapper
from openff.toolkit.utils import toolkit_registry_manager
from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper
from openff.units import unit as off_unit
from openff.units import unit as offunit
from openmm import app, unit
from openmm.app import ForceField as OMMForceField
from openmmforcefields.generators import (
    GAFFTemplateGenerator,
    EspalomaTemplateGenerator,
)
from openmmml import MLPotential
from qcportal.torsiondrive import TorsiondriveRecord
from rdkit.Chem import rdMolAlign
from rdkit.Geometry import Point3D
from tqdm import tqdm


def _parameterize_molecule(
    mapped_smiles: str,
    force_field: ForceField | OMMForceField | MLPotential,
    force_field_type: str,
) -> openmm.System:
    """Parameterize a particular molecules with a specified force field."""

    offmol = Molecule.from_mapped_smiles(mapped_smiles)

    if force_field_type.lower() == "smirnoff":
        return force_field.create_openmm_system(offmol.to_topology())

    elif force_field_type.lower() == "smirnoff-nagl":
        with toolkit_registry_manager(
            ToolkitRegistry([RDKitToolkitWrapper, NAGLToolkitWrapper])
        ):
            openmm_system = force_field.create_openmm_system(offmol.to_topology())

        return openmm_system

    elif force_field_type.lower() == "smirnoff-am1bcc":
        offmol.assign_partial_charges(partial_charge_method="am1bcc")
        return force_field.create_openmm_system(
            offmol.to_topology(), charge_from_molecules=[offmol]
        )

    elif force_field_type.lower() == "amber":
        # Get residue information for Amber biopolymer force field
        offmol.perceive_residues()

        return force_field.createSystem(
            offmol.to_topology().to_openmm(),
            nonbondedCutoff=0.9 * unit.nanometer,
            switchDistance=0.8 * unit.nanometer,
            constraints=None,
        )

    elif force_field_type.lower() == "gaff":
        return force_field.createSystem(
            offmol.to_topology().to_openmm(),
            nonbondedCutoff=0.9 * unit.nanometer,
            switchDistance=0.8 * unit.nanometer,
            constraints=None,
        )

    elif force_field_type.lower() == "espaloma":
        return force_field.createSystem(
            offmol.to_topology().to_openmm(),
            nonbondedCutoff=0.9 * unit.nanometer,
            switchDistance=0.8 * unit.nanometer,
            constraints=None,
        )

    elif force_field_type.lower() == "aceff20":
        omm_top = offmol.to_topology().to_openmm()
        charge = offmol.total_charge.m_as(off_unit.e)
        return force_field.createSystem(omm_top, charge=charge)

    raise NotImplementedError(
        "Only SMIRNOFF, Amber, GAFF, and espaloma force fields are currently supported."
    )


def _evaluate_energy(
    openmm_system: openmm.System,
    coordinates: unit.Quantity,
    platform_name: str = "Reference",
) -> float:
    """Evaluate the potential energy of a conformer in kcal/mol."""
    integrator = openmm.VerletIntegrator(0.001 * unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName(platform_name)
    openmm_context = openmm.Context(openmm_system, integrator, platform)

    openmm_context.setPositions(coordinates.value_in_unit(unit.nanometers))
    state = openmm_context.getState(getEnergy=True)
    potential_energy = state.getPotentialEnergy()

    return float(potential_energy.value_in_unit(unit.kilocalories_per_mole))


def _minimise_structure(
    openmm_system: openmm.System,
    coordinates: unit.Quantity,
    fixed_indices: Tuple[int, ...],
) -> unit.Quantity:
    """Minimize coordinates with fixed atoms."""
    openmm_system = copy.deepcopy(openmm_system)

    # Constrain the fixed atoms
    for index in fixed_indices:
        openmm_system.setParticleMass(index, 0.0)

    integrator = openmm.VerletIntegrator(0.001 * unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName("Reference")
    openmm_context = openmm.Context(openmm_system, integrator, platform)
    openmm_context.setPositions(coordinates.m_as(offunit.nanometer))

    openmm.LocalEnergyMinimizer.minimize(openmm_context)

    state: openmm.State = openmm_context.getState(getPositions=True)
    return state.getPositions()


def _compute_grid_energies(
    force_field: ForceField | OMMForceField,
    force_field_type: str,
    qc_record: TorsiondriveRecord,
    molecule: Molecule,
) -> Dict[str, Tuple[float, float, float, float]]:
    """Compute QM and MM energies on a 2D torsion grid."""
    grid_energies = qc_record.final_energies

    grid_conformers = {
        grid_id: conformer
        for grid_id, conformer in zip(
            molecule.properties["grid_ids"], molecule.conformers
        )
    }

    grid_ids = sorted(grid_conformers, key=lambda x: x[0])

    # Get indices of driven torsions
    dihedral_indices = [
        indices if indices[2] >= indices[1] else indices[::-1]
        for indices in qc_record.specification.keywords.dihedrals
    ]

    # Get indices of non-driven torsions with constraints
    if "constraints" in qc_record.specification.optimization_specification.keywords:
        constraints = qc_record.specification.optimization_specification.keywords[
            "constraints"
        ]

        for constraint_type in ["freeze", "set"]:
            if constraint_type in constraints:
                for constraint in constraints[constraint_type]:
                    if constraint["type"] == "dihedral":
                        indices = constraint["indices"]
                        dihedral_indices.append(
                            indices if indices[2] >= indices[1] else indices[::-1]
                        )

    # Indices of atoms in driven torsions, which are fixed during minimization
    fixed_indices = tuple(int(i) for i in np.unique(dihedral_indices))

    openmm_system = _parameterize_molecule(
        molecule.to_smiles(isomeric=True, mapped=True),
        force_field,
        force_field_type,
    )

    rd_mol = molecule.to_rdkit()
    rd_conf = rd_mol.GetConformer()

    ref_rd_mol = copy.deepcopy(rd_mol)
    ref_rd_conf = ref_rd_mol.GetConformer()

    # Zero out the contribution of the driven torsion to evaluate a target for
    # a Fourier series
    target_system = copy.deepcopy(openmm_system)
    torsion_forces = [
        force
        for force in target_system.getForces()
        if isinstance(force, openmm.PeriodicTorsionForce)
    ]

    if torsion_forces:
        torsion_force = torsion_forces[0]

        for torsion_index in range(torsion_force.getNumTorsions()):
            i, j, k, l, periodicity, phase, _ = torsion_force.getTorsionParameters(
                torsion_index
            )

            if ((i, j, k, l) if k >= j else (l, k, j, i)) in dihedral_indices:
                torsion_force.setTorsionParameters(
                    torsion_index, i, j, k, l, periodicity, phase, 0.0
                )

    # Evaluate the energy for each grid id
    energies: Dict[Tuple[int, ...], Tuple[float, float, float, float]] = dict()

    lowest_qm_energy = None
    lowest_qm_energy_grid_id = None

    for grid_id in grid_ids:
        coordinates = _minimise_structure(
            openmm_system, grid_conformers[grid_id], fixed_indices
        )

        qm_energy = (
            grid_energies[grid_id] * unit.hartree * unit.AVOGADRO_CONSTANT_NA
        ).value_in_unit(unit.kilocalories_per_mole)

        platform_name = "Reference" if force_field_type != "aceff20" else "CUDA"

        mm_energy = _evaluate_energy(
            openmm_system, coordinates, platform_name=platform_name
        )
        mm_target = _evaluate_energy(
            target_system, coordinates, platform_name=platform_name
        )

        # Get RMSD between MM and QM coordinates from RDKit
        ref_coords = grid_conformers[grid_id].m_as(offunit.nanometer)
        min_coords = coordinates.value_in_unit(unit.nanometer)

        for i in range(rd_mol.GetNumAtoms()):
            ref_rd_conf.SetAtomPosition(
                i, Point3D(ref_coords[i][0], ref_coords[i][1], ref_coords[i][2])
            )
            rd_conf.SetAtomPosition(
                i, Point3D(min_coords[i][0], min_coords[i][1], min_coords[i][2])
            )

        rmsd = rdMolAlign.AlignMol(rd_mol, ref_rd_mol)

        if lowest_qm_energy is None or qm_energy < lowest_qm_energy:
            lowest_qm_energy = qm_energy
            lowest_qm_energy_grid_id = grid_id

        energies[grid_id] = (qm_energy, mm_energy, mm_target, rmsd)

    energies = {
        json.dumps(grid_id): (
            qm_energy - energies[lowest_qm_energy_grid_id][0],
            mm_energy - energies[lowest_qm_energy_grid_id][1],
            (
                qm_energy
                - energies[lowest_qm_energy_grid_id][0]
                - (mm_target - energies[lowest_qm_energy_grid_id][2])
            ),
            rmsd,
        )
        for grid_id, (qm_energy, mm_energy, mm_target, rmsd) in energies.items()
    }

    return energies


def minimise_protein_torsion(
    input_file: str | Path,
    force_field_path: str | Path,
    force_field_label: str,
    force_field_type: str,
    output_path: str | Path,
) -> None:
    """Compute MM minimizations and energies for a protein torsion dataset.

    Parameters
    ----------
    input_file : str or Path
        Path to the input JSON file containing the TorsionDrive dataset in QCA format
    force_field_path : str or Path
        Path to the force field file or name
    force_field_label : str
        Label for the force field (used for identification)
    force_field_type : str
        Type of force field: 'smirnoff', 'smirnoff-nagl', 'smirnoff-am1bcc',
        'amber', 'gaff', 'aceff20', or 'espaloma'
    output_path : str or Path
        Path where the output JSON will be written
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    force_field_type = force_field_type.lower()

    input_file = Path(input_file)

    torsiondrive_dataset = TorsionDriveResultCollection.parse_file(input_file)

    records_and_molecules = torsiondrive_dataset.to_records()

    if force_field_type in {"smirnoff", "smirnoff-nagl", "smirnoff-am1bcc"}:
        force_field = ForceField(
            str(force_field_path), load_plugins=True, allow_cosmetic_attributes=True
        )

        if "Constraints" in force_field.registered_parameter_handlers:
            force_field.deregister_parameter_handler("Constraints")

    elif force_field_type == "amber":
        force_field = OMMForceField(str(force_field_path))

    elif force_field_type == "gaff":
        force_field = OMMForceField()
        gaff_generator = GAFFTemplateGenerator(
            molecules=[mol for _mol_record, mol in records_and_molecules],
            forcefield=str(force_field_path),
        )
        force_field.registerTemplateGenerator(gaff_generator.generator)

    elif force_field_type == "espaloma":
        force_field = OMMForceField()
        espaloma = EspalomaTemplateGenerator(
            molecules=[mol for _mol_record, mol in records_and_molecules],
            forcefield="espaloma-0.3.2",
        )
        force_field.registerTemplateGenerator(espaloma.generator)

    elif force_field_type == "aceff20":
        force_field = MLPotential("aceff-2.0")

    else:
        raise NotImplementedError(
            'force_field_type must be one of: "smirnoff", "smirnoff-nagl", '
            '"smirnoff-am1bcc", "amber", "gaff", "aceff20", or "espaloma"'
        )

    qc_data: DefaultDict[str, Any] = defaultdict(dict)

    for qc_record, offmol in tqdm(records_and_molecules):
        if len(offmol.properties["grid_ids"]) == 0:
            continue

        # Track which SMILES (with the driven dihedral tagged) corresponds to
        # which record id.
        dihedral_indices = np.unique(qc_record.specification.keywords.dihedrals)
        offmol_copy = copy.deepcopy(offmol)
        offmol_copy.properties["atom_map"] = {
            j: i + 1 for i, j in enumerate(dihedral_indices)
        }
        qc_data[qc_record.id]["smiles"] = offmol_copy.to_smiles(mapped=True)

        # Compute QM energy, minimized MM energy, MM target, and RMSD
        qc_data[qc_record.id]["energies"] = _compute_grid_energies(
            force_field,
            force_field_type,
            qc_record=qc_record,
            molecule=offmol,
        )

    with open(output_path, "w") as output_file:
        json.dump(qc_data, output_file)
