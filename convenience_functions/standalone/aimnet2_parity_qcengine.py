"""AIMNet2 energies, forces and per-call cost through QCEngine, as bespokefit calls it.

Run by file path in the ``bespokefit-ml`` env, on the conformers written by
``aimnet2_parity_presto.py``:

    python aimnet2_parity_qcengine.py CONFORMERS.sdf {openmm-ml,aimnet2} N_CALLS OUT.json

``openmm-ml`` is the matched harness (single model, fp32, GPU); ``aimnet2`` is the
stock QCEngine harness (pyaimnet2 4-model ensemble, fp64, CPU).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import qcelemental
import qcengine
import torch
from openff.toolkit import Molecule

sys.path.insert(0, str(Path(__file__).parent))
from openmm_ml_harness import OpenMMMLHarness

METHODS = {"openmm-ml": "aimnet2", "aimnet2": "wb97m-d3"}
N_WARMUP_CALLS = 10
HARTREE_TO_KCAL = qcelemental.constants.conversion_factor("hartree", "kcal/mol")
BOHR_TO_A = qcelemental.constants.conversion_factor("bohr", "angstrom")


def main(
    conformers_sdf: Path, program: str, n_calls_per_conformer: int, out_json: Path
) -> None:
    qcengine.register_program(OpenMMMLHarness())
    mol = Molecule.from_file(str(conformers_sdf), "SDF")
    mols = mol if isinstance(mol, list) else [mol]
    qc_mols = [m.to_qcschema() for m in mols]

    def gradient(
        qc_mol: qcelemental.models.Molecule,
    ) -> qcelemental.models.AtomicResult:
        task = {
            "molecule": qc_mol,
            "driver": "gradient",
            "model": {"method": METHODS[program]},
        }
        return qcengine.compute(task, program, raise_error=True)

    for _ in range(N_WARMUP_CALLS):
        gradient(qc_mols[0])

    energies, forces, call_times = [], [], []
    for qc_mol in qc_mols:
        for _ in range(n_calls_per_conformer):
            start = time.perf_counter()
            result = gradient(qc_mol)
            call_times.append(time.perf_counter() - start)
        energies.append(result.properties.return_energy * HARTREE_TO_KCAL)
        forces.append(
            (-np.array(result.return_result) * HARTREE_TO_KCAL / BOHR_TO_A).tolist()
        )

    out_json.write_text(
        json.dumps(
            {
                "arm": f"qcengine_{program}",
                "energies_kcal_mol": energies,
                "forces_kcal_mol_A": forces,
                "call_times_s": call_times,
                "torch_threads": torch.get_num_threads(),
            }
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4]))
