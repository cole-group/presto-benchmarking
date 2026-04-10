"""Functionality for running presto to generate bespoke forcefields."""

from pathlib import Path

from presto.settings import WorkflowSettings
from presto.utils._suppress_output import suppress_unwanted_output
from presto.workflow import get_bespoke_force_field


def run_presto(
    config_path: Path,
    smiles_path: Path,
    output_dir: Path,
) -> None:
    """
    Run presto with the given configuration, output directory, and SMILES,
    saving the bespoke force field and other outputs to the specified output directory.
    """

    suppress_unwanted_output()

    with open(smiles_path, "r") as f:
        smiles_entries = [line.strip() for line in f.readlines() if line.strip()]

    if not smiles_entries:
        raise ValueError(f"No SMILES found in input file: {smiles_path}")

    config = WorkflowSettings.from_yaml(config_path)
    config.parameterisation_settings.smiles = (
        smiles_entries[0] if len(smiles_entries) == 1 else smiles_entries
    )
    config.output_dir = output_dir

    bespoke_ff = get_bespoke_force_field(config)

    bespoke_ff.to_file(output_dir / "bespoke_force_field.offxml")
