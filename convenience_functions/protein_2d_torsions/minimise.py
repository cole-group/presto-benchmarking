"""Minimization analysis for 2D protein torsion validation.

Slightly modified from Chapin Cavendar's script:
https://raw.githubusercontent.com/openforcefield/protein-param-fit/refs/heads/sage-2.1/validation/torsiondrive/2-run-torsiondrive-mm-minimizations.py
"""

import copy
import json
import logging
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Literal, Tuple

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
from qcportal.torsiondrive import TorsiondriveRecord
from rdkit.Chem import rdMolAlign
from rdkit.Geometry import Point3D
from tqdm import tqdm

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message="torch.distributed.reduce_op is deprecated",
    category=UserWarning,
    module="torch.distributed.distributed_c10d",
)

_DEFAULT_TORSION_RESTRAINT_K: float = 10_000
"""Torsion restraint force constant in kcal/(mol·rad²)."""

_DEFAULT_ANGLE_TOLERANCE: float = 1.0
"""Tolerance in degrees for post-minimization dihedral sanity checks."""


class MinimisationError(Exception):
    """Raised when a restrained minimization fails a sanity check."""
    pass


############ Force-field parameterization ############


def _parameterize_molecule(
    mapped_smiles: str,
    force_field: ForceField | OMMForceField,
    force_field_type: str,
) -> openmm.System:
    """Parameterize a molecule and return an OpenMM System."""
    offmol = Molecule.from_mapped_smiles(mapped_smiles)

    if force_field_type.lower() == "smirnoff":
        return force_field.create_openmm_system(offmol.to_topology())

    elif force_field_type.lower() == "smirnoff-nagl":
        with toolkit_registry_manager(
            ToolkitRegistry([RDKitToolkitWrapper, NAGLToolkitWrapper])
        ):
            return force_field.create_openmm_system(offmol.to_topology())

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

    elif force_field_type.lower() in {"gaff", "espaloma"}:
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
        "Only SMIRNOFF, Amber, GAFF, espaloma, and aceff20 force fields are currently supported."
    )


def _make_energy_context(
    openmm_system: openmm.System,
    platform_name: str = "Reference",
) -> openmm.Context:
    """Create a reusable OpenMM Context for energy evaluation."""
    integrator = openmm.VerletIntegrator(0.001 * unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName(platform_name)
    return openmm.Context(openmm_system, integrator, platform)


def _evaluate_energy_with_context(
    context: openmm.Context,
    coordinates: unit.Quantity,
) -> float:
    """Evaluate potential energy (kcal/mol) by reusing an existing Context."""
    context.setPositions(coordinates.value_in_unit(unit.nanometers))
    return float(
        context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilocalories_per_mole)
    )


def _evaluate_energy(
    openmm_system: openmm.System,
    coordinates: unit.Quantity,
    platform_name: str = "Reference",
) -> float:
    """Evaluate the potential energy of a conformer in kcal/mol."""
    context = _make_energy_context(openmm_system, platform_name)
    return _evaluate_energy_with_context(context, coordinates)


############ Torsion restraint utilities ############


def _compute_dihedral_angle(
    coords_nm: np.ndarray,
    indices: tuple[int, int, int, int],
) -> float:
    """Compute a dihedral angle in degrees from coordinates in nm.

    Delegates to ``MDAnalysis.lib.distances.calc_dihedrals``, which is backed
    by compiled C code.  Returns a value in (−180, +180].
    """
    from MDAnalysis.lib.distances import calc_dihedrals

    coords_ang = (coords_nm * 10.0).astype(np.float32)
    angle_rad = calc_dihedrals(
        coords_ang[indices[0]][np.newaxis],
        coords_ang[indices[1]][np.newaxis],
        coords_ang[indices[2]][np.newaxis],
        coords_ang[indices[3]][np.newaxis],
    )[0]
    return float(np.degrees(angle_rad))


def _find_unused_force_group(system: openmm.System) -> int:
    """Return the highest-numbered force group not currently used (0–31)."""
    used = {system.getForce(i).getForceGroup() for i in range(system.getNumForces())}
    for group in range(31, -1, -1):
        if group not in used:
            return group
    raise RuntimeError("All 32 force groups are in use; cannot add restraint.")


def _add_torsion_restraints_to_system(
    system: openmm.System,
    dihedral_indices_list: list[tuple[int, int, int, int]],
    initial_angles_deg: list[float],
    force_group: int,
    k: float = _DEFAULT_TORSION_RESTRAINT_K,
) -> openmm.CustomTorsionForce:
    """Add one ``CustomTorsionForce`` covering all restrained dihedrals in-place.

    Returns the force object so callers can call ``updateParametersInContext``
    to change ``theta0`` without rebuilding the system or context.
    """
    torsion_restraint = openmm.CustomTorsionForce(
        "0.5*k_torsion*dtheta^2; dtheta = atan2(sin(theta-theta0), cos(theta-theta0))"
    )
    torsion_restraint.addGlobalParameter(
        "k_torsion",
        k * unit.kilocalorie_per_mole / unit.radian**2,
    )
    torsion_restraint.addPerTorsionParameter("theta0")
    for indices, angle_deg in zip(dihedral_indices_list, initial_angles_deg):
        torsion_restraint.addTorsion(
            int(indices[0]), int(indices[1]), int(indices[2]), int(indices[3]),
            [angle_deg * np.pi / 180.0],
        )
    torsion_restraint.setForceGroup(force_group)
    system.addForce(torsion_restraint)
    return torsion_restraint


def _update_torsion_restraint_angles(
    context: openmm.Context,
    torsion_force: openmm.CustomTorsionForce,
    new_angles_deg: list[float],
) -> None:
    """Update ``theta0`` for each torsion in *torsion_force* and push to *context*."""
    for i, angle_deg in enumerate(new_angles_deg):
        idx0, idx1, idx2, idx3, _ = torsion_force.getTorsionParameters(i)
        torsion_force.setTorsionParameters(i, idx0, idx1, idx2, idx3, [angle_deg * np.pi / 180.0])
    torsion_force.updateParametersInContext(context)


############ Minimization protocols ############


class _FrozenMinimizer:
    """Reusable minimizer that holds a single OpenMM Context with frozen dihedral atoms.

    Zeroes the masses of *fixed_indices* once and builds the Context once per
    record, avoiding repeated JIT compilation overhead for each grid point.
    """

    def __init__(
        self,
        openmm_system: openmm.System,
        fixed_indices: Tuple[int, ...],
    ) -> None:
        frozen_system = copy.deepcopy(openmm_system)
        for index in fixed_indices:
            frozen_system.setParticleMass(index, 0.0)
        integrator = openmm.VerletIntegrator(0.001 * unit.femtoseconds)
        platform = openmm.Platform.getPlatformByName("Reference")
        self._context = openmm.Context(frozen_system, integrator, platform)

    def minimise(self, coordinates: unit.Quantity) -> unit.Quantity:
        """Set positions, minimize, and return positions."""
        self._context.setPositions(coordinates.m_as(offunit.nanometer))
        openmm.LocalEnergyMinimizer.minimize(self._context)
        return self._context.getState(getPositions=True).getPositions()


class _RestrainedMinimizer:
    """Reusable minimizer that holds a single OpenMM Context with torsion restraints.

    Building an OpenMM Context requires expression compilation for any
    ``CustomTorsionForce`` present.  Creating a new Context for every grid
    point (as would happen with a straight deepcopy-per-point approach) therefore
    multiplies this compilation cost by the number of grid points.  This class
    builds the restrained system and Context **once** and updates ``theta0`` via
    ``updateParametersInContext`` for each subsequent grid point, avoiding the
    repeated compilation overhead.
    """

    def __init__(
        self,
        openmm_system: openmm.System,
        all_dihedral_indices: list[tuple[int, int, int, int]],
        initial_angles_deg: list[float],
    ) -> None:
        self._system = copy.deepcopy(openmm_system)
        self._restraint_group = _find_unused_force_group(self._system)
        self._dihedral_indices = all_dihedral_indices
        self._torsion_force = _add_torsion_restraints_to_system(
            self._system,
            all_dihedral_indices,
            initial_angles_deg,
            self._restraint_group,
        )
        integrator = openmm.VerletIntegrator(0.001 * unit.femtoseconds)
        platform = openmm.Platform.getPlatformByName("Reference")
        self._context = openmm.Context(self._system, integrator, platform)

    def minimise(
        self,
        coordinates: unit.Quantity,
        new_angles_deg: list[float],
    ) -> unit.Quantity:
        """Update restraint targets, set positions, minimize, return positions.

        Raises
        ------
        MinimisationError
            If any dihedral deviates beyond ``_DEFAULT_ANGLE_TOLERANCE`` after
            minimization.
        """
        _update_torsion_restraint_angles(self._context, self._torsion_force, new_angles_deg)
        self._context.setPositions(coordinates.m_as(offunit.nanometer))
        openmm.LocalEnergyMinimizer.minimize(self._context)
        final_positions = self._context.getState(getPositions=True).getPositions()

        # Sanity check: verify each dihedral stayed at its target
        raw = final_positions.value_in_unit(unit.nanometer)
        coords_nm = np.array([[v[0], v[1], v[2]] for v in raw])
        for indices, target_deg in zip(self._dihedral_indices, new_angles_deg):
            final_angle = _compute_dihedral_angle(coords_nm, indices)
            diff = abs((final_angle - target_deg + 180.0) % 360.0 - 180.0)
            if diff > _DEFAULT_ANGLE_TOLERANCE:
                raise MinimisationError(
                    f"Torsion restraint sanity check failed for dihedral {indices}: "
                    f"target={target_deg:.2f}°, final={final_angle:.2f}°, "
                    f"diff={diff:.2f}° exceeds tolerance of {_DEFAULT_ANGLE_TOLERANCE:.2f}°"
                )
        return final_positions


############ Dihedral index parsing and target-system construction ############


def _parse_dihedral_indices(
    qc_record: TorsiondriveRecord,
) -> tuple[
    list[tuple[int, int, int, int]],  # driven dihedrals, original atom order
    list[tuple[int, int, int, int]],  # constraint dihedrals, original atom order
    set[tuple[int, int, int, int]],   # all dihedrals, canonicalized (for torsion zeroing)
    Tuple[int, ...],                  # flat unique atom indices across all dihedrals
]:
    """Extract driven and constraint dihedral indices from a QC record.

    Returns
    -------
    driven_raw:
        Driven dihedrals in their original atom order (one per scan axis).
    constraint_raw:
        Additional dihedrals frozen/set in the optimization spec, original order.
    canonical_set:
        All dihedrals normalized so atom[2] >= atom[1], used for zeroing torsion
        contributions in the target system.
    fixed_atom_indices:
        Flattened, unique atom indices across all dihedrals.
    """
    def _canonicalize(idx: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        return idx if idx[2] >= idx[1] else idx[::-1]

    driven_raw: list[tuple[int, int, int, int]] = [
        tuple(int(x) for x in idx)
        for idx in qc_record.specification.keywords.dihedrals
    ]

    constraint_raw: list[tuple[int, int, int, int]] = []
    opt_keywords = qc_record.specification.optimization_specification.keywords
    if "constraints" in opt_keywords:
        for constraint_type in ["freeze", "set"]:
            for constraint in opt_keywords["constraints"].get(constraint_type, []):
                if constraint["type"] == "dihedral":
                    constraint_raw.append(tuple(int(x) for x in constraint["indices"]))

    canonical_set = {_canonicalize(idx) for idx in driven_raw + constraint_raw}
    fixed_atom_indices = tuple(int(i) for i in np.unique(list(canonical_set)))
    return driven_raw, constraint_raw, canonical_set, fixed_atom_indices


def _build_target_system(
    openmm_system: openmm.System,
    canonical_dihedrals: set[tuple[int, int, int, int]],
) -> openmm.System:
    """Return a copy of *openmm_system* with driven torsion k-values zeroed out.

    This produces the "MM target" system used to isolate the non-torsion MM
    energy, which serves as a target for a Fourier-series fit.
    """
    target_system = copy.deepcopy(openmm_system)
    torsion_forces = [
        f for f in target_system.getForces()
        if isinstance(f, openmm.PeriodicTorsionForce)
    ]
    if torsion_forces:
        torsion_force = torsion_forces[0]
        for idx in range(torsion_force.getNumTorsions()):
            i, j, k, l, periodicity, phase, _ = torsion_force.getTorsionParameters(idx)
            canonical = (i, j, k, l) if k >= j else (l, k, j, i)
            if canonical in canonical_dihedrals:
                torsion_force.setTorsionParameters(idx, i, j, k, l, periodicity, phase, 0.0)
    return target_system


############ Grid energy computation ############


def _compute_grid_energies(
    force_field: ForceField | OMMForceField,
    force_field_type: str,
    qc_record: TorsiondriveRecord,
    molecule: Molecule,
    dihedral_protocol: Literal["restrain", "freeze"] = "freeze",
) -> Dict[str, Tuple[float, float, float, float]]:
    """Compute QM and MM energies on a 2D torsion grid.

    Parameters
    ----------
    force_field:
        The MM force field object.
    force_field_type:
        Force field type string (e.g. ``"smirnoff-nagl"``).
    qc_record:
        The QCArchive TorsionDrive record.
    molecule:
        OpenFF Molecule with conformers and a ``grid_ids`` property.
    dihedral_protocol:
        ``"restrain"`` (default): apply strong harmonic torsion restraints
        during minimization, with target angles taken from the grid point
        (driven dihedrals) or the QM conformer (constraint dihedrals).
        ``"freeze"``: zero the masses of dihedral atoms (legacy behaviour).
    """
    grid_energies = qc_record.final_energies
    grid_conformers = {
        grid_id: conformer
        for grid_id, conformer in zip(
            molecule.properties["grid_ids"], molecule.conformers
        )
    }
    grid_ids = sorted(grid_conformers, key=lambda x: x[0])

    driven_raw, constraint_raw, canonical_dihedrals, fixed_atom_indices = (
        _parse_dihedral_indices(qc_record)
    )

    openmm_system = _parameterize_molecule(
        molecule.to_smiles(isomeric=True, mapped=True),
        force_field,
        force_field_type,
    )
    target_system = _build_target_system(openmm_system, canonical_dihedrals)

    rd_mol = molecule.to_rdkit()
    rd_conf = rd_mol.GetConformer()
    ref_rd_mol = copy.deepcopy(rd_mol)
    ref_rd_conf = ref_rd_mol.GetConformer()

    energies: Dict[Tuple[int, ...], Tuple[float, float, float, float]] = {}
    lowest_qm_energy = None
    lowest_qm_energy_grid_id = None
    platform_name = "Reference" if force_field_type != "aceff20" else "CUDA"

    # Build reusable energy-evaluation contexts once per record
    mm_energy_context = _make_energy_context(openmm_system, platform_name)
    mm_target_context = _make_energy_context(target_system, platform_name)

    if dihedral_protocol == "restrain":
        # Compute constraint-dihedral target angles from the first available
        # QM conformer as a representative initial value (updated per grid point).
        first_conformer = grid_conformers[grid_ids[0]]
        first_coords_nm = first_conformer.m_as(offunit.nanometer)
        initial_angles: list[float] = [
            float(grid_ids[0][j]) for j in range(len(driven_raw))
        ] + [
            _compute_dihedral_angle(first_coords_nm, idx) for idx in constraint_raw
        ]
        all_dihedral_indices = driven_raw + constraint_raw
        restrained_minimizer = _RestrainedMinimizer(
            openmm_system, all_dihedral_indices, initial_angles
        )
    else:
        frozen_minimizer = _FrozenMinimizer(openmm_system, fixed_atom_indices)

    for grid_id in grid_ids:
        qm_conformer = grid_conformers[grid_id]

        if dihedral_protocol == "restrain":
            # Driven dihedrals: target angle from the grid point
            new_angles: list[float] = [float(grid_id[j]) for j in range(len(driven_raw))]
            # Constraint dihedrals: target angle measured from the QM conformer
            qm_coords_nm = qm_conformer.m_as(offunit.nanometer)
            for indices in constraint_raw:
                new_angles.append(_compute_dihedral_angle(qm_coords_nm, indices))

            try:
                coordinates = restrained_minimizer.minimise(qm_conformer, new_angles)
            except MinimisationError as e:
                logger.warning(
                    f"Skipping grid point {grid_id} for record {qc_record.id}: {e}"
                )
                continue
        else:
            coordinates = frozen_minimizer.minimise(qm_conformer)

        qm_energy = (
            grid_energies[grid_id] * unit.hartree * unit.AVOGADRO_CONSTANT_NA
        ).value_in_unit(unit.kilocalories_per_mole)

        mm_energy = _evaluate_energy_with_context(mm_energy_context, coordinates)
        mm_target = _evaluate_energy_with_context(mm_target_context, coordinates)

        # RMSD between QM and MM-minimized coordinates (via RDKit, in nm)
        ref_coords = qm_conformer.m_as(offunit.nanometer)
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

    ref_qm, ref_mm, ref_mm_target, _ = energies[lowest_qm_energy_grid_id]
    return {
        json.dumps(grid_id): (
            qm - ref_qm,
            mm - ref_mm,
            (qm - ref_qm) - (mm_t - ref_mm_target),
            rmsd,
        )
        for grid_id, (qm, mm, mm_t, rmsd) in energies.items()
    }


############ Public API ############


def minimise_protein_torsion(
    input_file: str | Path,
    force_field_path: str | Path,
    force_field_label: str,
    force_field_type: str,
    output_path: str | Path,
    dihedral_protocol: Literal["restrain", "freeze"] = "restrain",
) -> None:
    """Compute MM minimizations and energies for a protein torsion dataset.

    Parameters
    ----------
    input_file:
        Path to the QCA TorsionDrive JSON file.
    force_field_path:
        Path to the force field file (or its registered name).
    force_field_label:
        Label used to identify this force field in outputs.
    force_field_type:
        One of ``"smirnoff"``, ``"smirnoff-nagl"``, ``"smirnoff-am1bcc"``,
        ``"amber"``, ``"gaff"``, ``"espaloma"``, ``"aceff20"``.
    output_path:
        Path for the output JSON file.
    dihedral_protocol:
        ``"restrain"`` (default): apply strong harmonic torsion restraints
        during MM minimization.  ``"freeze"``: zero the masses of dihedral
        atoms (legacy behaviour).
    """
    
    output_path = Path(output_path)
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
            molecules=[mol for _, mol in records_and_molecules],
            forcefield=str(force_field_path),
        )
        force_field.registerTemplateGenerator(gaff_generator.generator)

    elif force_field_type == "espaloma":
        force_field = OMMForceField()
        espaloma = EspalomaTemplateGenerator(
            molecules=[mol for _, mol in records_and_molecules],
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

        dihedral_indices = np.unique(qc_record.specification.keywords.dihedrals)
        offmol_copy = copy.deepcopy(offmol)
        offmol_copy.properties["atom_map"] = {
            j: i + 1 for i, j in enumerate(dihedral_indices)
        }
        qc_data[qc_record.id]["smiles"] = offmol_copy.to_smiles(mapped=True)

        qc_data[qc_record.id]["energies"] = _compute_grid_energies(
            force_field,
            force_field_type,
            qc_record=qc_record,
            molecule=offmol,
            dihedral_protocol=dihedral_protocol,
        )

    with open(output_path, "w") as output_file:
        json.dump(qc_data, output_file)
