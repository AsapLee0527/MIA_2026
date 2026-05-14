"""Common helpers shared across cohort-specific dataset adapters."""

from __future__ import annotations

import numpy as np


def fisher_z(corr: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Fisher z-transform applied to a Pearson correlation matrix."""
    corr = np.clip(corr, -1.0 + eps, 1.0 - eps)
    return np.arctanh(corr)


def symmetrize(mat: np.ndarray) -> np.ndarray:
    """Force symmetry: M = (M + M^T) / 2 and zero out the diagonal."""
    mat = 0.5 * (mat + mat.T)
    np.fill_diagonal(mat, 0.0)
    return mat


def normalize_connectome(mat: np.ndarray, kind: str = "zscore") -> np.ndarray:
    """Normalize a connectivity matrix.

    Parameters
    ----------
    mat : np.ndarray
        R x R connectome.
    kind : {"zscore", "minmax", "none"}
        Normalization scheme.
    """
    if kind == "none":
        return mat
    flat = mat[np.triu_indices_from(mat, k=1)]
    if kind == "zscore":
        mu, sd = flat.mean(), flat.std() + 1e-8
        return (mat - mu) / sd
    if kind == "minmax":
        lo, hi = flat.min(), flat.max()
        return (mat - lo) / (hi - lo + 1e-8)
    raise ValueError(f"Unknown normalization scheme: {kind}")
