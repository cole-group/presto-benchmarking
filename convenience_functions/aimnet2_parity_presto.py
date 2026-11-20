"""AIMNet2 energies, forces and per-call cost through presto's own code path.

Conformers are generated and written to ``conformers_sdf`` if it doesn't exist,
then always read back from it, so every arm (including the QCEngine ones in
``standalone/aimnet2_parity_qcengine.py``) evaluates identical geometries.
Run with ``CUDA_VISIBLE_DEVICES=""`` to time the CPU-only version; openmm-ml picks
the device itself, so hiding the GPU is the only reliable way to force CPU.
"""

import json
import time
from pathlib import Path

import numpy as np
import openmm.unit as omm_unit
import torch
from openff.toolkit import Molecule
from rdkit import Chem

from presto.sample import _build_ml_simulation

N_WARMUP_CALLS = 10


def aimnet2_parity_presto(
    smiles: str,
    n_conformers: int,
    n_calls_per_conformer: int,
    conformers_sdf: Path,
    out_json: Path,
) -> None:
    """Time AIMNet2 energy+force calls as presto makes them; save results to JSON."""
    if not conformers_sdf.exists():
        mol = Molecule.from_smiles(smiles)
        mol.generate_conformers(n_conformers=n_conformers)
        rdmol = mol.to_rdkit()  # OpenFF's to_file writes only the first conformer
        with Chem.SDWriter(str(conformers_sdf)) as writer:
            for conformer in rdmol.GetConformers():
                writer.write(rdmol, confId=conformer.GetId())

    # Always read back, so every arm sees identical (SDF-rounded) coordinates
    loaded = Molecule.from_file(str(conformers_sdf), "SDF")
    mols = loaded if isinstance(loaded, list) else [loaded]
    mol = mols[0]
    for other in mols[1:]:
        mol.add_conformer(other.conformers[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simulation, _ = _build_ml_simulation(
        mol,
        mol.to_topology().to_openmm(),
        "aimnet2",
        temperature=300 * omm_unit.kelvin,
        timestep=1 * omm_unit.femtosecond,
        device=device,
    )

    def energy_and_forces(conformer: np.ndarray) -> tuple[float, np.ndarray]:
        simulation.context.setPositions(conformer * omm_unit.angstrom)
        state = simulation.context.getState(getEnergy=True, getForces=True)
        energy = state.getPotentialEnergy().value_in_unit(omm_unit.kilocalorie_per_mole)
        forces = state.getForces(asNumpy=True).value_in_unit(
            omm_unit.kilocalorie_per_mole / omm_unit.angstrom
        )
        return energy, forces

    conformers = [c.m_as("angstrom") for c in mol.conformers]
    for _ in range(N_WARMUP_CALLS):
        energy_and_forces(conformers[0])

    energies, forces, call_times = [], [], []
    for conformer in conformers:
        for _ in range(n_calls_per_conformer):
            start = time.perf_counter()
            energy, force = energy_and_forces(conformer)
            call_times.append(time.perf_counter() - start)
        energies.append(energy)
        forces.append(force.tolist())

    out_json.write_text(
        json.dumps(
            {
                "arm": f"presto_{device.type}",
                "energies_kcal_mol": energies,
                "forces_kcal_mol_A": forces,
                "call_times_s": call_times,
                "torch_threads": torch.get_num_threads(),
            }
        )
    )
