"""Filter QCArchive torsion records based on bespoke-assigned scanned torsions."""

from __future__ import annotations

from pathlib import Path

import loguru
from openff.toolkit import ForceField, Molecule
from yammbs.torsion.inputs import QCArchiveTorsionDataset

logger = loguru.logger


def _iter_assigned_parameters(assigned: object) -> list[object]:
    """Normalize OpenFF label assignment payloads into a flat parameter list."""
    if isinstance(assigned, (list, tuple)):
        return list(assigned)
    return [assigned]


def _contains_only_bespoke_scanned_torsions(
    force_field: ForceField,
    mapped_smiles: str,
    dihedral_indices: tuple[int, int, int, int],
) -> tuple[bool, str]:
    """Return keep flag and drop reason for scanned-bond bespoke coverage."""
    molecule = Molecule.from_mapped_smiles(
        mapped_smiles,
        allow_undefined_stereo=True,
    )
    labels = force_field.label_molecules(molecule.to_topology())[0]
    proper_torsion_labels = labels.get("ProperTorsions", {})

    scanned_central_bond = tuple(sorted((int(dihedral_indices[1]), int(dihedral_indices[2]))))
    matched_parameter_ids: list[str] = []

    for atom_indices, assigned in proper_torsion_labels.items():
        torsion_central_bond = tuple(sorted((int(atom_indices[1]), int(atom_indices[2]))))
        if torsion_central_bond != scanned_central_bond:
            continue

        for parameter in _iter_assigned_parameters(assigned):
            parameter_id = getattr(parameter, "id")
            matched_parameter_ids.append(str(parameter_id))

    if not matched_parameter_ids:
        return False, "no_match"

    has_only_bespoke_terms = all(
        "bespoke" in parameter_id.lower() for parameter_id in matched_parameter_ids
    )
    if has_only_bespoke_terms:
        return True, "retained"

    return False, "non_bespoke"


def filter_qca_torsions_to_bespoke_scans(
    input_qca_json_path: Path,
    force_field_path: Path,
    output_qca_json_path: Path,
) -> None:
    """Keep only torsions whose scanned bond is fully covered by bespoke terms."""
    with open(input_qca_json_path) as file_handle:
        input_dataset = QCArchiveTorsionDataset.model_validate_json(file_handle.read())

    force_field = ForceField(str(force_field_path))

    retained = []
    dropped_non_bespoke = 0
    dropped_no_match = 0

    for torsion_profile in input_dataset.qm_torsions:
        keeps_profile, reason = _contains_only_bespoke_scanned_torsions(
            force_field=force_field,
            mapped_smiles=torsion_profile.mapped_smiles,
            dihedral_indices=torsion_profile.dihedral_indices,
        )
        if keeps_profile:
            retained.append(torsion_profile)
            continue

        if reason == "non_bespoke":
            dropped_non_bespoke += 1
        else:
            dropped_no_match += 1

    output_dataset = QCArchiveTorsionDataset(qm_torsions=retained)
    output_qca_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_qca_json_path, "w") as file_handle:
        file_handle.write(output_dataset.model_dump_json())

    logger.info(
        f"Filtered QCA torsions for bespoke scanned bonds: retained {len(retained)}/{len(input_dataset.qm_torsions)}, "
        f"dropped_non_bespoke={dropped_non_bespoke}, dropped_no_match={dropped_no_match}",
    )
