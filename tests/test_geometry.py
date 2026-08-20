"""Tests for ensemble geometry features and randomized eigensolver."""

import numpy as np

from protgenesis_ensemble import compute_ensemble_geometry, randomized_eigendecomposition
from protgenesis_ensemble.geometry import spectral_features_from_eigenvalues
from protgenesis_ensemble.covariance import ledoit_wolf_shrinkage


def _make_ensemble(rng, n_samples=250, n_res=20, intrinsic_dim=3, scale=2.0):
    """Ensemble fluctuating along `intrinsic_dim` orthogonal Cα modes."""
    base = rng.standard_normal((n_res, 3)) * 5
    modes = rng.standard_normal((intrinsic_dim, n_res, 3))
    frames = []
    for _ in range(n_samples):
        coeffs = rng.standard_normal(intrinsic_dim) * scale
        frames.append(base + np.tensordot(coeffs, modes, axes=1))
    return np.stack(frames)


def test_spectral_features_identity_spectrum():
    """Flat spectrum of D modes: PR=D, entropy=log(D), eff_rank_95=ceil(0.95D)."""
    D = 40
    ev = np.ones(D)
    f = spectral_features_from_eigenvalues(ev)
    assert abs(f["PR"] - D) < 1e-9
    assert abs(f["entropy"] - np.log(D)) < 1e-7  # +eps regularization term shifts by ~1e-9/D
    assert f["eff_rank_95"] == int(np.ceil(0.95 * D)) or f["eff_rank_95"] == int(np.floor(0.95 * D)) + 1
    assert abs(f["spectral_decay"]) < 0.2  # flat spectrum -> alpha ~ 0


def test_spectral_features_powerlaw_spectrum():
    """lambda_k = k^-1: spectral_decay should recover ~1."""
    ev = 1.0 / np.arange(1, 101)
    f = spectral_features_from_eigenvalues(ev, n_fit=100)
    assert 0.8 < f["spectral_decay"] < 1.2


def test_compute_ensemble_geometry_lowdim():
    rng = np.random.default_rng(3)
    pos = _make_ensemble(rng, intrinsic_dim=3)
    f = compute_ensemble_geometry(pos, rng=3)
    assert 1.0 <= f["PR"] <= 20  # low intrinsic dimensionality
    assert f["eff_rank_95"] >= 1
    assert f["entropy"] > 0
    assert f["Rg_mean"] > 0
    assert f["variance_per_dof"] > 0
    assert f["n_residues"] == 20


def test_geometry_determinism():
    rng = np.random.default_rng(4)
    pos = _make_ensemble(rng)
    f1 = compute_ensemble_geometry(pos, rng=42)
    f2 = compute_ensemble_geometry(pos, rng=42)
    for k in f1:
        assert f1[k] == f2[k], k


def test_ledoit_wolf_shrinkage_bounds():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((100, 30))
    cov, lam = ledoit_wolf_shrinkage(X)
    assert 0.0 <= lam <= 1.0
    assert np.all(np.linalg.eigvalsh(cov) > 0)  # PD guarantee


def test_randomized_eigendecomposition_accuracy():
    """rand route must match exact eigensolver on the top modes."""
    rng = np.random.default_rng(6)
    # spectrum with clear top-10 structure
    D = 120
    true_ev = np.concatenate([np.linspace(50, 5, 10), np.full(D - 10, 0.5)])
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    C = Q @ np.diag(true_ev) @ Q.T
    L = np.linalg.cholesky(C + 1e-8 * np.eye(D))
    X = rng.standard_normal((2000, D)) @ L.T
    ev_rand, _ = randomized_eigendecomposition(X, rank=10, rng=6)
    ev_exact = np.linalg.eigvalsh(np.cov(X.T))[::-1][:10]
    rel_err = np.abs(ev_rand - ev_exact) / ev_exact
    assert np.median(rel_err) < 0.05
