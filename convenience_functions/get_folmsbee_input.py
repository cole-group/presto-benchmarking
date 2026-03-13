"""Utilities to download Folmsbee/Hutchison conformer-benchmark input.

Paper: https://onlinelibrary.wiley.com/doi/full/10.1002/qua.26381
Repository: https://github.com/ghutchis/conformer-benchmark
"""

from pathlib import Path
from subprocess import run
import loguru

logger = loguru.logger

_FOLMSBEE_REPO_URL = "https://github.com/ghutchis/conformer-benchmark.git"


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
    """Convert whitespace-delimited input into per-molecule ``.smi`` files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with open(molecules_smi, "r") as handle:
        for raw_line in handle:
            parts = raw_line.split()
            if len(parts) < 2:
                continue

            smiles, molecule_id = parts[0], parts[1]
            (output_dir / f"{molecule_id}.smi").write_text(f"{smiles}\n")
            n_written += 1

    logger.info(f"Wrote {n_written} .smi files to {output_dir}")

