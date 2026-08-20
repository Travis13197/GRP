"""Kabsch alignment: removal of SE(3) rigid-body modes from conformational ensembles.

Canonical implementation extracted from the ProtGenesis2 Ensemble project
(field_theory/scripts/phase_l1_kabsch_metric.py). After alignment, the ensemble
covariance reflects only intrinsic conformational fluctuation (6 near-zero
eigenvalues correspond to the removed rigid-body modes).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def kabsch_align(mobile: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Optimally superimpose ``mobile`` onto ``reference`` (Kabsch algorithm).

    Parameters
    ----------
    mobile, reference : (n_points, 3) arrays

    Returns
    -------
    aligned : (n_points, 3) rotated/translated copy of ``mobile``
    rotation : (3, 3) optimal rotation matrix
    """
    mobile = np.asarray(mobile, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if mobile.shape != reference.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError("mobile and reference must both have shape (n_points, 3)")

    mob_c = mobile - mobile.mean(axis=0)
    ref_c = reference - reference.mean(axis=0)

    H = mob_c.T @ ref_c
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt

    aligned = mob_c @ R + reference.mean(axis=0)
    return aligned, R


def kabsch_align_ensemble(
    positions: np.ndarray,
    reference: Optional[np.ndarray] = None,
    reference_mode: str = "first",
) -> np.ndarray:
    """Align every frame of an ensemble to a common reference frame.

    Parameters
    ----------
    positions : (n_samples, n_residues, 3) array
    reference : (n_residues, 3) optional explicit reference conformation.
    reference_mode : 'first' (frame 0), 'mean' (iterative mean, 2 rounds),
        or 'medoid' (min sum-RMSD frame). Ignored when ``reference`` is given.

    Returns
    -------
    aligned : (n_samples, n_residues, 3) array
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("positions must have shape (n_samples, n_residues, 3)")
    n_samples = positions.shape[0]

    if reference is not None:
        ref = np.asarray(reference, dtype=np.float64)
    elif reference_mode == "first":
        ref = positions[0]
    elif reference_mode == "mean":
        ref = positions[0].copy()
        for _ in range(2):
            tmp = np.empty_like(positions)
            for i in range(n_samples):
                tmp[i], _ = kabsch_align(positions[i], ref)
            ref = tmp.mean(axis=0)
    elif reference_mode == "medoid":
        rmsd = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                a, _ = kabsch_align(positions[i], positions[j])
                rmsd[i, j] = rmsd[j, i] = np.sqrt(np.mean((a - positions[j]) ** 2))
        ref = positions[int(np.argmin(rmsd.sum(axis=1)))]
    else:
        raise ValueError(f"unknown reference_mode: {reference_mode!r}")

    aligned = np.empty_like(positions)
    for i in range(n_samples):
        aligned[i], _ = kabsch_align(positions[i], ref)
    return aligned
