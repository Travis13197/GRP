"""Law 1 — C_geo geometric perturbation stability metric.

Canonical implementation of the audit-corrected formula (2026-07-17, Phase 9):
the local metric at residue i is the regularized inverse of its 3x3 intrinsic
covariance block,

    g_S(i) = (C_i + eps_i * I_3)^-1,   eps_i = REG_EPS * tr(C_i) / 3

and the perturbation cost of a substitution is the squared regularized
Mahalanobis displacement

    C_geo = d^T g_S d

with |d| set by physicochemical substitution magnitude (volume/charge deltas)
and a random direction (seeded). REG_EPS = 0.01 (project default).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .align import kabsch_align_ensemble

REG_EPS = 0.01

AA_VOLUMES: Dict[str, float] = {
    "A": 88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9,
    "G": 60.1, "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7,
    "M": 162.9, "N": 114.1, "P": 112.7, "Q": 143.8, "R": 173.4,
    "S": 89.0, "T": 116.1, "V": 140.0, "W": 227.8, "Y": 193.6,
}

AA_CHARGES: Dict[str, float] = {
    "R": 1, "K": 1, "D": -1, "E": -1, "H": 0.1,
    "A": 0, "C": 0, "F": 0, "G": 0, "I": 0, "L": 0,
    "M": 0, "N": 0, "P": 0, "Q": 0, "S": 0, "T": 0,
    "V": 0, "W": 0, "Y": 0,
}

AA_HYDROPHOBICITY: Dict[str, float] = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
    "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5,
    "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}


def build_residue_metrics(
    positions: np.ndarray,
    already_aligned: bool = False,
    reg_eps: float = REG_EPS,
) -> np.ndarray:
    """Build per-residue regularized local metrics g_S(i) = (C_i + eps I)^-1.

    Parameters
    ----------
    positions : (n_samples, n_residues, 3) Cα ensemble
    already_aligned : skip Kabsch alignment if True
    reg_eps : relative regularization strength (project default 0.01)

    Returns
    -------
    g_blocks : (n_residues, 3, 3) array
    """
    positions = np.asarray(positions, dtype=np.float64)
    n_samples, n_res, _ = positions.shape
    aligned = positions if already_aligned else kabsch_align_ensemble(positions)

    g_blocks = np.zeros((n_res, 3, 3))
    for i in range(n_res):
        Xi = aligned[:, i, :]  # (n_samples, 3)
        C = np.cov(Xi, rowvar=False)
        eps = reg_eps * np.trace(C) / 3.0
        g_blocks[i] = np.linalg.inv(C + eps * np.eye(3))
    return g_blocks


def mutation_displacement_magnitude(wt_aa: str, mut_aa: str) -> float:
    """Physicochemical displacement magnitude |d| for a substitution.

    |d| = 0.1 + |ΔVolume|/100 + |ΔCharge|*0.05   (project convention)
    """
    dv = abs(AA_VOLUMES.get(mut_aa, 0.0) - AA_VOLUMES.get(wt_aa, 0.0))
    dq = abs(AA_CHARGES.get(mut_aa, 0.0) - AA_CHARGES.get(wt_aa, 0.0))
    return 0.1 + dv / 100.0 + dq * 0.05


def compute_cgeo_mutation(
    g_blocks: np.ndarray,
    pos: int,
    wt_aa: str,
    mut_aa: str,
    direction: Optional[np.ndarray] = None,
    rng: int | np.random.Generator = 42,
) -> float:
    """C_geo of a single substitution at 0-based residue index ``pos``.

    Parameters
    ----------
    g_blocks : (n_residues, 3, 3) from :func:`build_residue_metrics`
    pos : 0-based residue index
    wt_aa, mut_aa : one-letter amino-acid codes
    direction : optional explicit unit displacement direction (3,)
    rng : seed/Generator used when ``direction`` is None

    Returns
    -------
    C_geo : float, d^T g_S d  (squared regularized Mahalanobis displacement)
    """
    g_blocks = np.asarray(g_blocks, dtype=np.float64)
    if not (0 <= pos < g_blocks.shape[0]):
        raise IndexError(f"pos {pos} out of range for {g_blocks.shape[0]} residues")

    mag = mutation_displacement_magnitude(wt_aa, mut_aa)
    if direction is None:
        gen = np.random.default_rng(rng)
        direction = gen.standard_normal(3)
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / (np.linalg.norm(direction) + 1e-10)

    d = mag * direction
    return float(d @ g_blocks[pos] @ d)
