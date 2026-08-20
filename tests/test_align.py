"""Tests for Kabsch alignment: rotation/translation invariance, ensemble alignment."""

import numpy as np

from protgenesis_ensemble import kabsch_align, kabsch_align_ensemble


def test_kabsch_recovers_rotation_translation():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((30, 3)) * 5
    # random proper rotation
    A = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    t = np.array([3.0, -7.0, 11.0])
    mobile = ref @ Q + t
    aligned, _ = kabsch_align(mobile, ref)
    rmsd = np.sqrt(np.mean((aligned - ref) ** 2))
    assert rmsd < 1e-8


def test_kabsch_invalid_shape():
    try:
        kabsch_align(np.zeros((5, 3)), np.zeros((6, 3)))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_ensemble_alignment_removes_rigid_modes():
    """After alignment, covariance must have ~6 zero eigenvalues for a rigid body."""
    rng = np.random.default_rng(1)
    base = rng.standard_normal((20, 3)) * 4
    frames = []
    for _ in range(50):
        A = rng.standard_normal((3, 3))
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        frames.append(base @ Q + rng.standard_normal(3) * 10)
    positions = np.stack(frames)
    aligned = kabsch_align_ensemble(positions, reference_mode="first")
    X = aligned.reshape(50, -1)
    X = X - X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    evals = np.linalg.eigvalsh(cov)
    # rigid body: aligned ensemble is (numerically) constant -> all variance ~ 0
    assert evals.max() < 1e-10


def test_ensemble_alignment_reduces_rmsd():
    rng = np.random.default_rng(2)
    base = rng.standard_normal((15, 3)) * 3
    frames = []
    for _ in range(20):
        A = rng.standard_normal((3, 3))
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        noise = rng.standard_normal((15, 3)) * 0.1
        frames.append(base @ Q + rng.standard_normal(3) * 5 + noise)
    positions = np.stack(frames)
    aligned = kabsch_align_ensemble(positions)
    var_before = positions.reshape(20, -1).std(axis=0).mean()
    var_after = aligned.reshape(20, -1).std(axis=0).mean()
    assert var_after < var_before
