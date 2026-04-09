"""Shared statistical and metric utility functions."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests


def rms(values: np.ndarray) -> float:
    """Root-mean-square for a 1D numeric array."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be a 1D array")
    if values.size == 0:
        raise ValueError("values must contain at least one element")
    return float(np.sqrt(np.mean(values**2)))


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    """Root-mean-square error between equal-length arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have identical shapes, got {x.shape} and {y.shape}")
    if x.size == 0:
        raise ValueError("x and y must contain at least one element")
    return float(np.sqrt(np.mean((x - y) ** 2)))


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_bootstrap: int = 10_000,
    confidence_level: float = 95.0,
    random_seed: int = 0,
) -> tuple[float, float]:
    """Return a percentile bootstrap confidence interval for a 1D array."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be a 1D array")
    if values.size == 0:
        raise ValueError("values must contain at least one element")

    rng = np.random.default_rng(random_seed)
    bootstrap_stats = np.empty(n_bootstrap, dtype=float)
    n_values = values.size
    for i in range(n_bootstrap):
        sample_idx = rng.integers(0, n_values, size=n_values)
        bootstrap_stats[i] = statistic(values[sample_idx])

    alpha = (100.0 - confidence_level) / 2.0
    lower, upper = np.percentile(bootstrap_stats, [alpha, 100.0 - alpha])
    return float(lower), float(upper)


def format_value_with_ci(value: float, ci: tuple[float, float], digits: int = 2) -> str:
    """Format estimate and CI using LaTeX super/subscript notation."""
    return f"${value:.{digits}f}_{{{ci[0]:.{digits}f}}}^{{{ci[1]:.{digits}f}}}$"


def format_header_with_units(metric_name: str, units: str) -> str:
    """Build a two-line LaTeX column header with metric and units."""
    return rf"\shortstack{{{metric_name} \\ / {units}}}"


def js_distance(
    rel_a: np.ndarray,
    rel_b: np.ndarray,
    temperature: float,
    k_b_kcal_mol_k: float = 0.0019872041,
) -> float:
    """Boltzmann-weighted Jensen-Shannon distance between relative energies."""
    rel_a = np.asarray(rel_a, dtype=float)
    rel_b = np.asarray(rel_b, dtype=float)
    if rel_a.shape != rel_b.shape:
        raise ValueError(f"rel_a and rel_b must have identical shapes, got {rel_a.shape} and {rel_b.shape}")
    if rel_a.size == 0:
        raise ValueError("rel_a and rel_b must contain at least one element")

    beta = 1.0 / (k_b_kcal_mol_k * temperature)
    pa = np.exp(-beta * rel_a)
    pb = np.exp(-beta * rel_b)
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    return float(jensenshannon(pa, pb))


def sign_test_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sided sign test p-value (ties excluded)."""
    diffs = x - y
    diffs = diffs[diffs != 0]
    n_non_zero = len(diffs)
    if n_non_zero == 0:
        return 1.0
    n_pos = int(np.sum(diffs > 0))
    return float(binomtest(n_pos, n_non_zero, p=0.5, alternative="two-sided").pvalue)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni correction; return adjusted p-values in input order."""
    _, adjusted, _, _ = multipletests(p_values, method="holm")
    return adjusted.tolist()


def pvalue_to_stars(p: float, show_ns: bool = True) -> str:
    """Convert a p-value to a significance star string.

    Parameters
    ----------
    p:
        Adjusted p-value.
    show_ns:
        If *True* (default) return ``"ns"`` for non-significant results.
        If *False* return an empty string (useful for heatmap cell annotations).
    """
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns" if show_ns else ""
