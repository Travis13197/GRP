"""Intrinsic ensemble geometry features.

Faithful re-implementation of the project's core feature extractor
(test_workflow/polyx_ensemble/analyze_ensemble_geometry.py, v2.0, Kabsch+LW
intrinsic frame). All features derive from the eigenspectrum of the shrunk
intrinsic covariance, plus radius of gyration and pairwise-distance statistics.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .align import kabsch_align_ensemble
from .covariance import ledoit_wolf_shrinkage


def radius_of_gyration(positions: np.ndarray) -> np.ndarray:
    """Per-frame radius of gyration. positions: (n_samples, n_res, 3) -> (n_samples,)"""
    centered = positions - positions.mean(axis=1, keepdims=True)
    return np.sqrt((centered**2).sum(axis=(1, 2)) / positions.shape[1])


def spectral_features_from_eigenvalues(eigenvalues: np.ndarray, n_fit: int = 50) -> Dict[str, float]:
    """Compute spectrum-derived features from a descending eigenvalue array."""
    ev = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0)
    total = ev.sum()
    if total <= 0:
        raise ValueError("total variance is zero")
    p = ev / total
    eps = 1e-10

    pr = float(total**2 / (ev**2).sum()) if (ev**2).sum() > 0 else 0.0
    cumsum = np.cumsum(p)
    eff_rank_95 = int(np.searchsorted(cumsum, 0.95) + 1)
    eff_rank_99 = int(np.searchsorted(cumsum, 0.99) + 1)
    entropy = float(-np.sum(p * np.log(p + eps)))

    # Spectral decay: log-log linear fit lambda_k ~ k^{-alpha} over the first n_fit modes
    k = np.arange(1, len(ev) + 1)
    m = min(n_fit, len(ev))
    A = np.vstack([np.log(k[:m]), np.ones(m)]).T
    alpha, _ = np.linalg.lstsq(A, np.log(ev[:m] + eps), rcond=None)[0]
    spectral_decay = float(-alpha)

    # Anisotropy measures
    a_c = float(p[0])  # top-mode concentration (lower bound 1/D)
    inv = 1.0 / (ev + eps)
    a_g = float(inv[0] / inv.sum()) if inv.sum() > 0 else 0.0

    # Pseudo-volume: geometric mean of nonzero eigenvalues
    nz = ev[ev > 1e-10]
    pseudo_volume = float(np.exp(np.log(nz).mean())) if len(nz) else 0.0

    return {
        "PR": pr,
        "eff_rank_95": eff_rank_95,
        "eff_rank_99": eff_rank_99,
        "entropy": entropy,
        "spectral_decay": spectral_decay,
        "A_C": a_c,
        "A_g": a_g,
        "pseudo_volume": pseudo_volume,
        "total_variance": float(total),
    }


def compute_ensemble_geometry(
    positions: np.ndarray,
    already_aligned: bool = False,
    reference_mode: str = "first",
    n_spectral_fit: int = 50,
    pairwise_subsample: int = 100,
    rng: int | np.random.Generator = 42,
) -> Dict[str, float]:
    """Compute the full intrinsic-geometry feature vector of an ensemble.

    Parameters
    ----------
    positions : (n_samples, n_residues, 3) array (Cα ensemble)
    already_aligned : skip Kabsch alignment if True
    reference_mode : Kabsch reference ('first' | 'mean' | 'medoid')
    n_spectral_fit : modes used in the spectral-decay fit (project default 50)
    pairwise_subsample : frames subsampled for pairwise-distance stats
    rng : seed or Generator for the subsample draw

    Returns
    -------
    dict with keys:
        PR, eff_rank_95, eff_rank_99, entropy, spectral_decay, A_C, A_g,
        pseudo_volume, total_variance, variance_per_dof,
        Rg_mean, Rg_std, mean_pairwise_dist, std_pairwise_dist,
        lw_shrinkage, n_samples, n_residues
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("positions must have shape (n_samples, n_residues, 3)")
    n_samples, n_residues, _ = positions.shape
    n_features = n_residues * 3

    aligned = positions if already_aligned else kabsch_align_ensemble(positions, reference_mode=reference_mode)
    X = aligned.reshape(n_samples, n_features)
    X = X - X.mean(axis=0)

    cov, lambda_star = ledoit_wolf_shrinkage(X)
    eigenvalues = np.linalg.eigvalsh(cov)[::-1]
    feats = spectral_features_from_eigenvalues(eigenvalues, n_fit=n_spectral_fit)
    feats["variance_per_dof"] = feats["total_variance"] / n_features

    rg = radius_of_gyration(aligned)
    feats["Rg_mean"] = float(rg.mean())
    feats["Rg_std"] = float(rg.std())

    gen = np.random.default_rng(rng)
    if n_samples > pairwise_subsample:
        idx = gen.choice(n_samples, size=pairwise_subsample, replace=False)
        sub = X[idx]
    else:
        sub = X
    # pairwise distances (upper triangle)
    sq = (sub**2).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2 * (sub @ sub.T)
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(np.maximum(d2, 0.0))
    triu = d[np.triu_indices_from(d, k=1)]
    feats["mean_pairwise_dist"] = float(triu.mean())
    feats["std_pairwise_dist"] = float(triu.std())

    feats["lw_shrinkage"] = float(lambda_star)
    feats["n_samples"] = int(n_samples)
    feats["n_residues"] = int(n_residues)
    return feats
