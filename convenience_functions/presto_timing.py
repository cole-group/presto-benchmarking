"""Per-stage wall-clock timing of a presto fit, without modifying presto.

presto functions are wrapped (monkeypatched) at the module where they are looked
up, so every call is recorded as a nested "span". The wrappers only time calls;
arguments and return values pass through unchanged. Patch targets (the ``patches``
table in ``instrument_presto``) match the pinned presto commit 377a638; a missing
target raises AttributeError rather than silently losing a span.
"""

import functools
import json
import os
import platform
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch


class SpanTimer:
    """Records nested, named wall-clock spans."""

    def __init__(self) -> None:
        self.iteration = 0  # presto training iteration (0 = before training)
        self.spans: list[dict[str, Any]] = []
        self._stack: list[str] = []

    def wrap(
        self, fn: Callable[..., Any], name: str | Callable[..., str], sync: bool = False
    ) -> Callable[..., Any]:
        """Return ``fn`` wrapped so each call is recorded as a span.

        ``name`` may be a callable taking the call's arguments, to label calls
        differently (e.g. train vs test loss). ``sync`` calls
        ``torch.cuda.synchronize()`` around the span; only needed for stage-level
        spans, as the inner calls already block on GPU results.
        """

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            label = name(*args, **kwargs) if callable(name) else name
            path = "/".join([*self._stack, label])
            if sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            self._stack.append(label)
            start = time.perf_counter_ns()
            try:
                return fn(*args, **kwargs)
            finally:
                if sync and torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter_ns()
                self._stack.pop()
                self.spans.append(
                    {
                        "path": path,
                        "iteration": self.iteration,
                        "duration_s": (end - start) / 1e9,
                    }
                )

        return wrapper

    def summary(self) -> dict[str, dict[str, float]]:
        """Aggregate spans by path, adding an ``(other)`` row per parent for the
        time not covered by its children."""
        totals: dict[str, list[float]] = defaultdict(list)
        for span in self.spans:
            totals[span["path"]].append(span["duration_s"])

        rows = {
            path: {"count": len(d), "total_s": sum(d), "mean_s": sum(d) / len(d)}
            for path, d in totals.items()
        }
        for parent in list(rows):
            children = [
                rows[p]["total_s"]
                for p in rows
                if p.startswith(parent + "/") and "/" not in p[len(parent) + 1 :]
            ]
            if children:
                other = rows[parent]["total_s"] - sum(children)
                rows[f"{parent}/(other)"] = {
                    "count": 1,
                    "total_s": other,
                    "mean_s": other,
                }
        return rows


def instrument_presto(timer: SpanTimer, training_settings_type: type) -> None:
    """Wrap the presto functions listed below so each call is timed.

    ``training_settings_type`` identifies the training sampler in presto's
    registry; each call to it starts a new training iteration.

    Note this is all pretty brittle and dependent on the internal structure of presto. 
    """
    import openff.interchange
    import openmm.app
    import presto.convert
    import presto.msm
    import presto.sample
    import presto.train
    import presto.workflow
    import smee.converters

    import convenience_functions.run_presto

    # Lets _minimize_with_frozen_torsions calls be labelled ML or MM.
    current_ml_simulation: list[Any] = [None]

    # Span-name helpers only read the arguments; they accept any call signature so
    # they can never change how presto's functions are called.
    def torsion_min_name(*a: Any, **k: Any) -> str:
        current_ml_simulation[0] = k.get("ml_simulation", a[1] if len(a) > 1 else None)
        return "torsion_minimisation"

    def frozen_min_name(*a: Any, **k: Any) -> str:
        simulation = k.get("simulation", a[0] if a else None)
        kind = "ml" if simulation is current_ml_simulation[0] else "mm"
        return f"{kind}_restrained_minimisation"

    def loss_name(*a: Any, **k: Any) -> str:
        compute_grad = k.get("compute_grad", a[7] if len(a) > 7 else True)
        return "loss_and_grad" if compute_grad else "loss_no_grad"

    # (module, attribute, span name, cuda sync)
    patches: list[tuple[Any, str, str | Callable[..., str], bool]] = [
        # Whole fit (run_presto imports it by name)
        (convenience_functions.run_presto, "get_bespoke_force_field", "total", True),
        # Top-level workflow stages
        (presto.workflow, "parameterise", "parameterise", True),
        (presto.workflow, "filter_dataset_outliers", "outlier_filter", True),
        (presto.workflow, "write_scatter", "write_scatter", True),
        (presto.workflow, "convert_to_smirnoff", "convert_to_smirnoff", True),
        (presto.workflow, "analyse_workflow", "plots", True),
        # Parameterisation
        (presto.convert, "_expand_torsions", "expand_torsions", False),
        (presto.convert, "add_types_to_forcefield", "type_generation", False),
        (presto.convert, "apply_msm_to_molecules", "msm", True),
        (presto.convert, "linearise_harmonics_force_field", "linearise", False),
        (presto.convert, "linearise_harmonics_topology", "linearise", False),
        (smee.converters, "convert_interchange", "smee_convert", False),
        # Modified Seminario (ML Hessian)
        (presto.msm, "_build_ml_simulation", "build_ml_simulation", False),
        (presto.msm, "calculate_hessian", "ml_hessian", False),
        # Sampling
        (presto.sample, "_build_ml_simulation", "build_ml_simulation", False),
        (presto.sample, "_build_mm_simulation", "build_mm_simulation", False),
        (presto.sample, "_run_md", "md", False),
        (presto.sample, "recalculate_energies_and_forces", "ml_single_points", False),
        (presto.sample, "generate_torsion_minimised_dataset", torsion_min_name, False),
        (presto.sample, "_minimize_with_frozen_torsions", frozen_min_name, False),
        # Training
        (presto.train, "compute_overall_loss_and_grad", loss_name, False),
        # Caught wherever they are called; attributed by parent span
        (openmm.app.Simulation, "minimizeEnergy", "minimize_energy", False),
    ]
    for module, attr, name, sync in patches:
        setattr(module, attr, timer.wrap(getattr(module, attr), name, sync))

    # Classmethod: wrap the underlying function and re-bind it to the class
    interchange = openff.interchange.Interchange
    from_smirnoff = interchange.from_smirnoff.__func__  # type: ignore[attr-defined]
    interchange.from_smirnoff = classmethod(  # type: ignore[method-assign,assignment]
        timer.wrap(from_smirnoff, "interchange_from_smirnoff")
    )

    # Sampling/training functions are looked up in registries at call time
    def training_sampling_name(*a: Any, **k: Any) -> str:
        timer.iteration += 1
        return "training_sampling"

    sampling = presto.workflow._SAMPLING_FNS_REGISTRY
    for settings_type, fn in sampling.items():
        name = (
            training_sampling_name
            if settings_type is training_settings_type
            else "test_sampling"
        )
        sampling[settings_type] = timer.wrap(fn, name, sync=True)
    training = presto.workflow._TRAINING_FNS_REGISTRY
    for key, fn in training.items():
        training[key] = timer.wrap(fn, "train", sync=True)


def environment_info() -> dict[str, Any]:
    """Hardware and software versions relevant to timings."""
    from importlib.metadata import version

    def run(cmd: list[str]) -> str:
        try:
            return subprocess.run(
                cmd, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    return {
        "hostname": platform.node(),
        "cpu": run(["sh", "-c", "lscpu | grep 'Model name'"]),
        "n_cpus": os.cpu_count(),
        "cpu_governor": governor.read_text().strip()
        if governor.exists()
        else "unknown",
        "gpu": run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,persistence_mode,clocks.sm,clocks.max.sm",
                "--format=csv,noheader",
            ]
        ),
        "versions": {
            pkg: version(pkg)
            for pkg in [
                "torch",
                "openmm",
                "openmmml",
                "aimnet",
                "presto",
                "openff-toolkit",
            ]
        },
    }


def run_presto_timed(
    config_path: Path, smiles_path: Path, output_dir: Path, timings_path: Path
) -> None:
    """Run presto with every stage timed, writing ``timings_path`` (JSON).

    ``PRESTO_TIMING_T0`` (epoch seconds, set by the shell just before ``pixi run``)
    lets start-up (pixi activation + imports) be reported separately.
    """
    startup_s = time.time() - float(os.environ.get("PRESTO_TIMING_T0", time.time()))

    from presto.settings import WorkflowSettings

    from convenience_functions.run_presto import run_presto

    timer = SpanTimer()
    settings = WorkflowSettings.from_yaml(config_path)
    instrument_presto(timer, type(settings.training_sampling_settings))
    run_presto(config_path, smiles_path, output_dir)

    timings_path.write_text(
        json.dumps(
            {
                "startup_s": startup_s,
                "summary": timer.summary(),
                "spans": timer.spans,
                "environment": environment_info(),
            },
            indent=2,
        )
    )
