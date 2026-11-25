"""Time one OpenFF BespokeFit 0.4.4 fit, stage by stage.

Run by file path in a bespokefit pixi env (it must not import presto):

    BESPOKE_TIMING_T0=$(date +%s.%N) pixi run -e bespokefit-ml python \\
        convenience_functions/standalone/run_bespokefit_timed.py SMILES aimnet2_matched OUT

What is timed:
- start-up (pixi activation + imports), reported separately from the total;
- schema build, executor start-up and shutdown;
- each executor stage (fragmentation, qc-generation, optimization), by polling the
  coordinator. Its state only updates every ~5 s
  (BEFLOW_COORDINATOR_MAX_UPDATE_INTERVAL), so stage times are accurate to ~5 s;
- every single QC energy/gradient call, by wrapping the QCEngine harness ``compute``
  before the executor forks its workers. Each worker process appends
  ``{driver, start, end}`` lines to ``OUT/compute_calls/<pid>.jsonl``.
"""

import argparse
import functools
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

# (QCSpec kwargs, QC workers, cores per worker)
VARIANTS: dict[str, tuple[dict[str, Any], int, int]] = {
    # AIMNet2 exactly as in presto (openmm-ml, single model, fp32, GPU)
    "aimnet2_matched": (
        {"program": "openmm-ml", "method": "aimnet2", "basis": None},
        8,
        4,
    ),
    # AIMNet2 as shipped with QCEngine (pyaimnet2, 4-model ensemble, fp64, CPU)
    "aimnet2_stock": (
        {"program": "aimnet2", "method": "wb97m-d3", "basis": None},
        8,
        4,
    ),
    # QM; 8 cores per psi4 worker is bespokefit's own recommendation
    "qm": ({"program": "psi4", "method": "b3lyp-d3bj", "basis": "dzvp"}, 4, 8),
}
# 2.3.0 needs openff-toolkit >=0.17 (NAGLCharges), incompatible with bespokefit 0.4.4
INITIAL_FORCE_FIELD = "openff_unconstrained-2.2.1.offxml"
PSI4_MEMORY_GB_PER_CORE = 3.0


def free_port() -> int:
    """A currently unused TCP port, so no stale redis (and its QC cache) is reused."""
    with socket.socket() as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def log_compute_calls(harness_cls: Any, log_dir: Path) -> None:
    """Wrap ``harness_cls.compute`` to log the wall-clock span of every call."""
    compute = harness_cls.compute

    @functools.wraps(compute)
    def timed_compute(self: Any, input_data: Any, config: Any) -> Any:
        start = time.time()
        result = compute(self, input_data, config)
        record = {"driver": input_data.driver, "start": start, "end": time.time()}
        with open(log_dir / f"{os.getpid()}.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        return result

    harness_cls.compute = timed_compute


def main(smiles: str, variant: str, out_dir: Path) -> None:
    out_dir = out_dir.resolve()  # the executor changes the working directory
    t_start = time.time()
    startup_s = t_start - float(os.environ.get("BESPOKE_TIMING_T0", t_start))
    qc_spec_kwargs, n_qc_workers, cores_per_worker = VARIANTS[variant]

    # Must be set before bespokefit reads its settings and before workers fork
    os.environ.update(
        {
            "BEFLOW_REDIS_PORT": str(free_port()),
            "BEFLOW_GATEWAY_PORT": str(free_port()),
            "BEFLOW_KEEP_TMP_FILES": "1",  # keep ForceBalance output
            "OMP_NUM_THREADS": str(cores_per_worker),
            "MKL_NUM_THREADS": str(cores_per_worker),
        }
    )

    import qcengine
    from openff.bespokefit.executor import BespokeExecutor, BespokeWorkerConfig
    from openff.bespokefit.workflows import BespokeWorkflowFactory
    from openff.qcsubmit.common_structures import QCSpec
    from openff.toolkit import Molecule

    sys.path.insert(0, str(Path(__file__).parent))
    from openmm_ml_harness import OpenMMMLHarness

    qcengine.register_program(OpenMMMLHarness())
    program = qc_spec_kwargs["program"]
    compute_log_dir = out_dir / "compute_calls"
    compute_log_dir.mkdir(parents=True)
    log_compute_calls(type(qcengine.get_program(program)), compute_log_dir)

    timings: dict[str, Any] = {"startup_s": startup_s, "variant": variant}
    t0 = time.time()

    factory = BespokeWorkflowFactory(
        initial_force_field=INITIAL_FORCE_FIELD,
        default_qc_specs=[
            QCSpec(**qc_spec_kwargs, spec_name=variant, spec_description=variant)
        ],
    )
    schema = factory.optimization_schema_from_molecule(Molecule.from_smiles(smiles))
    timings["schema_s"] = time.time() - t0

    executor = BespokeExecutor(
        n_fragmenter_workers=1,
        n_qc_compute_workers=n_qc_workers,
        qc_compute_worker_config=BespokeWorkerConfig(
            n_cores=cores_per_worker, max_memory=PSI4_MEMORY_GB_PER_CORE
        ),
        n_optimizer_workers=1,
        directory=str(out_dir / "executor"),
        launch_redis_if_unavailable=True,
    )
    t = time.time()
    executor.__enter__()
    timings["executor_startup_s"] = time.time() - t

    try:
        submitted = time.time()
        optimization_id = executor.submit(schema)
        # First time each stage was seen running / finished, relative to submission
        stage_times: dict[str, dict[str, float]] = {}
        while True:
            output = executor.retrieve(optimization_id)
            now = time.time() - submitted
            for stage in output.stages:
                stage_times.setdefault(stage.type, {})
                if stage.status != "waiting":
                    stage_times[stage.type].setdefault("first_seen_running_s", now)
                if stage.status in ("success", "errored"):
                    stage_times[stage.type].setdefault("first_seen_done_s", now)
            if output.status in ("success", "errored"):
                break
            time.sleep(1.0)  # the coordinator itself only updates every ~5 s
        timings["fit_s"] = time.time() - submitted
        timings["stages"] = stage_times
        timings["status"] = output.status
        timings["error"] = output.error
    finally:
        t = time.time()
        executor.__exit__(None, None, None)
        timings["executor_shutdown_s"] = time.time() - t

    timings["total_s"] = time.time() - t0
    timings["workers"] = {
        "n_qc_workers": n_qc_workers,
        "cores_per_worker": cores_per_worker,
    }
    timings["qc_spec"] = qc_spec_kwargs
    (out_dir / "timings.json").write_text(json.dumps(timings, indent=2))
    (out_dir / "output.json").write_text(output.json())
    if output.bespoke_force_field is not None:
        output.bespoke_force_field.to_file(str(out_dir / "bespoke_force_field.offxml"))
    if output.status != "success":
        raise RuntimeError(f"BespokeFit failed: {output.error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("smiles")
    parser.add_argument("variant", choices=sorted(VARIANTS))
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    main(args.smiles, args.variant, args.out_dir)
