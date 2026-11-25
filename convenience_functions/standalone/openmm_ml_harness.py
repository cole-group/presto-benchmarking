"""QCEngine harness that evaluates OpenMM-ML potentials exactly as presto does.

Mirrors ``presto.sample._build_ml_simulation`` at the pinned presto commit:
``MLPotential(name).createSystem(topology, charge=..., device="cuda")`` evaluated
through an OpenMM CPU-platform context (the ML model itself runs on the GPU).
Adapted from fjclark/contorsion ``openmm_ml_harness.py``.

Register with ``qcengine.register_program(OpenMMMLHarness())``; then use
``QCSpec(program="openmm-ml", method="aimnet2", basis=None)``.
"""

from typing import Any, ClassVar

import openmm
import qcelemental
from qcelemental.models import AtomicInput, AtomicResult, Provenance
from qcengine.config import TaskConfig
from qcengine.exceptions import InputError
from qcengine.programs.model import ProgramHarness

_BOHR_TO_NM = qcelemental.constants.conversion_factor("bohr", "nanometer")
_KJ_PER_MOL_TO_HARTREE = qcelemental.constants.conversion_factor("kJ/mol", "hartree")


class OpenMMMLHarness(ProgramHarness):  # type: ignore[misc]
    _defaults: ClassVar[dict[str, Any]] = {
        "name": "openmm-ml",
        "scratch": False,
        "thread_safe": False,
        "thread_parallel": False,
        "node_parallel": False,
        "managed_memory": False,
    }
    # One context per (molecule, model, charge), reused across gradient calls
    _contexts: ClassVar[dict[tuple[str, str, int], openmm.Context]] = {}

    @staticmethod
    def found(raise_error: bool = False) -> bool:
        from qcelemental.util import which_import

        return bool(which_import("openmmml", return_bool=True, raise_error=raise_error))

    def get_version(self) -> str:
        from importlib.metadata import version

        return version("openmmml")

    def _get_context(
        self, mapped_smiles: str, method: str, charge: int
    ) -> openmm.Context:
        key = (mapped_smiles, method, charge)
        if key not in self._contexts:
            from openff.toolkit import Molecule
            from openmmml import MLPotential

            topology = (
                Molecule.from_mapped_smiles(mapped_smiles).to_topology().to_openmm()
            )
            system = MLPotential(method).createSystem(
                topology, charge=charge, device="cuda"
            )
            integrator = openmm.VerletIntegrator(0.001)
            platform = openmm.Platform.getPlatformByName("CPU")  # as in presto
            self._contexts[key] = openmm.Context(system, integrator, platform)
        return self._contexts[key]

    def compute(self, input_data: AtomicInput, config: TaskConfig) -> AtomicResult:
        if input_data.driver not in ("energy", "gradient"):
            raise InputError(
                f"openmm-ml supports energy/gradient, not {input_data.driver}"
            )

        molecule = input_data.molecule
        mapped_smiles = molecule.extras[
            "canonical_isomeric_explicit_hydrogen_mapped_smiles"
        ]
        context = self._get_context(
            mapped_smiles, input_data.model.method, round(molecule.molecular_charge)
        )
        context.setPositions(molecule.geometry * _BOHR_TO_NM)
        state = context.getState(getEnergy=True, getForces=True)

        energy = (
            state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
            * _KJ_PER_MOL_TO_HARTREE
        )
        forces = state.getForces(asNumpy=True).value_in_unit(
            openmm.unit.kilojoule_per_mole / openmm.unit.nanometer
        )
        gradient = -forces * _KJ_PER_MOL_TO_HARTREE * _BOHR_TO_NM

        return AtomicResult(
            **{
                **input_data.dict(),
                "properties": {
                    "return_energy": energy,
                    "calcinfo_natom": len(gradient),
                },
                "return_result": energy if input_data.driver == "energy" else gradient,
                "success": True,
                "provenance": Provenance(
                    creator="openmm-ml",
                    version=self.get_version(),
                    routine="createSystem",
                ),
            }
        )
