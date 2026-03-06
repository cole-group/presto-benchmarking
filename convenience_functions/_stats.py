"""Shared statistical utility functions for torsion analysis."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests


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
