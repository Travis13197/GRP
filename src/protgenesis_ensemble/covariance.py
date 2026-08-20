"""Covariance estimation: Ledoit-Wolf shrinkage and randomized low-rank eigensolvers.

The Ledoit-Wolf shrinkage estimator is the project's default for intrinsic
ensemble covariance (n_samples ~ 250-500, n_features = 3N up to ~5600).
The randomized eigendecomposition implements the Phase M2 long-chain route:
sub-quadratic estimation of the top-k eigenspectrum, validated at <0.1% median
error for PR / spectral-decay / entropy on proteins of 214-770 aa.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def ledoit_wolf_shrinkage(X: np.ndarray) -> Tuple[np.ndarray, float]:
    """Ledoit-Wolf optimal shrinkage covariance.

    Parameters
    ----------
    X : (n_samples, n_features) array, already centered (or will be centered here)

    Returns
    -------
    cov : (n_features, n_features) shrunk covariance
    shrinkage : optimal shrinkage intensity lambda*
    """
    from sklearn.covariance import LedoitWolf

    X = np.asarray(X, dtype=np.float64)
    lw = LedoitWolf().fit(X)
    return lw.covariance_, float(lw.shrinkage_)


def randomized_eigendecomposition(
    X: np.ndarray,
    rank: int = 64,
    n_oversamples: int = 10,
    n_iter: int = 2,
    rng: int | np.random.Generator = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomized top-k eigendecomposition of the sample covariance of X.

    Implements the Phase M2 'rand64' route: covariance is never materialized;
    the range finder applies (X Xᵀ) products implicitly, giving
    O(n_samples · n_features · rank) cost instead of O(n_features³).

    Parameters
    ----------
    X : (n_samples, n_features) array (will be centered)
    rank : number of eigenpairs to estimate (project default: 64)
    n_oversamples : oversampling for the random sketch
    n_iter : power iterations for spectral accuracy
    rng : int seed or numpy Generator

    Returns
    -------
    eigenvalues : (rank,) descending
    eigenvectors : (n_features, rank) orthonormal columns
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape
    X = X - X.mean(axis=0)
    gen = np.random.default_rng(rng)

    omega = gen.standard_normal((n_features, rank + n_oversamples))
    # Y = C @ Omega where C = X^T X / (n-1), computed implicitly
    Y = X.T @ (X @ omega) / (n_samples - 1)
    for _ in range(n_iter):
        Y = X.T @ (X @ Y) / (n_samples - 1)
    Q, _ = np.linalg.qr(Y, mode="reduced")

    # Small projected eigenproblem: B = Qᵀ C Q
    BQ = X.T @ (X @ Q) / (n_samples - 1)  # C @ Q
    B = Q.T @ BQ
    B = 0.5 * (B + B.T)  # numerical symmetrization
    evals, evecs_small = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1][:rank]
    eigenvalues = np.maximum(evals[order], 0.0)
    eigenvectors = Q @ evecs_small[:, order]
    return eigenvalues, eigenvectors
