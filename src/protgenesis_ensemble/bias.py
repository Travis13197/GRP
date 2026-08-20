"""Finite-sample bias guidance for rank estimators (Phase CS1 finding).

Project result (CS1, 2026-08-12, TECHNICAL_REFERENCE 6.47; Kabsch intrinsic
frame, n=25 homopolymers, reference N=5000):

    estimator             N* (<10% bias)   bias@250   bias@500
    participation entropy (raw)    250      -6.3%      -3.9%   <- most robust
    eff_rank_95 (raw)              500     -31.2%      -7.0%   <- biased low
    PR (raw)                      1000      -7.8%     -16.2%
    eff_rank_95 (LW)              4000     +33.3%     +23.5%   <- biased high
    PR (LW)                       2000     +63.2%     +12.3%

Recommendation: at the project's working depths (N=250-500) prefer
participation entropy over eff_rank_95 for intrinsic-dimensionality claims;
report eff_rank_95 with its expected finite-N bias.
"""

from __future__ import annotations

import numpy as np

# Median relative bias (vs N=5000 reference) per estimator per sample depth
# (CS1 across PolyX_G/S/E/K/L_n25). Values are fractions (not %).
BIAS_CALIBRATION = {
    "entropy_raw": {100: -0.054, 250: -0.063, 500: -0.039, 1000: -0.013},
    "eff_rank_95_raw": {100: -0.28, 250: -0.312, 500: -0.070, 1000: -0.045},
    "PR_raw": {100: -0.20, 250: -0.078, 500: -0.162, 1000: -0.045},
    "eff_rank_95_lw": {100: 0.55, 250: 0.333, 500: 0.235, 1000: 0.17},
    "PR_lw": {100: 0.42, 250: 0.632, 500: 0.123, 1000: 0.04},
    "entropy_lw": {100: 0.30, 250: 0.294, 500: 0.185, 1000: 0.10},
}

N_STAR = {
    "entropy_raw": 250,
    "eff_rank_95_raw": 500,
    "PR_raw": 1000,
    "eff_rank_95_lw": 4000,
    "PR_lw": 2000,
    "entropy_lw": 2000,
}


def participation_entropy(eigenvalues: np.ndarray) -> float:
    """Participation (spectral) entropy H = -sum p_i log p_i, p = ev/sum(ev)."""
    ev = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 1e-12)
    p = ev / ev.sum()
    return float(-np.sum(p * np.log(p)))


def eff_rank_95(eigenvalues: np.ndarray) -> int:
    """Number of top modes needed to reach 95% of total variance."""
    ev = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0)
    total = ev.sum()
    if total <= 0:
        raise ValueError("total variance is zero")
    p = np.cumsum(ev / total)
    return int(np.searchsorted(p, 0.95) + 1)


def expected_eff_rank_bias(n_samples: int, estimator: str = "eff_rank_95_raw") -> float:
    """Expected median relative bias (fraction) of an estimator at a sample depth.

    Uses the CS1 calibration; extrapolates the nearest table entry for depths
    not measured. Returns 0.0 for N >= 5000 (reference depth).
    """
    if n_samples >= 5000:
        return 0.0
    table = BIAS_CALIBRATION.get(estimator)
    if table is None:
        raise KeyError(f"unknown estimator {estimator!r}; choose from {sorted(BIAS_CALIBRATION)}")
    levels = sorted(table)
    if n_samples <= levels[0]:
        return float(table[levels[0]])
    if n_samples >= levels[-1]:
        return float(table[levels[-1]])
    # 线性插值
    for lo, hi in zip(levels, levels[1:]):
        if lo <= n_samples <= hi:
            t = (n_samples - lo) / (hi - lo)
            return float(table[lo] + t * (table[hi] - table[lo]))
    return float(table[levels[-1]])


def dimension_guidance(n_samples: int) -> dict:
    """Recommend an intrinsic-dimensionality estimator for a sample depth."""
    if n_samples < 250:
        return {
            "recommended": "participation_entropy",
            "reason": "N<250: eff_rank_95 bias exceeds 28% (raw) / 55% (LW); "
                      "entropy is the only estimator within ~6% bias",
            "expected_eff_rank_95_bias_raw": expected_eff_rank_bias(n_samples, "eff_rank_95_raw"),
            "expected_eff_rank_95_bias_lw": expected_eff_rank_bias(n_samples, "eff_rank_95_lw"),
        }
    if n_samples < 500:
        return {
            "recommended": "participation_entropy",
            "reason": "N in [250,500): entropy bias <7%, eff_rank_95 raw bias up to -31%",
            "expected_eff_rank_95_bias_raw": expected_eff_rank_bias(n_samples, "eff_rank_95_raw"),
            "expected_eff_rank_95_bias_lw": expected_eff_rank_bias(n_samples, "eff_rank_95_lw"),
        }
    if n_samples < 4000:
        return {
            "recommended": "participation_entropy",
            "reason": "N in [500,4000): eff_rank_95 raw within 7% but LW variant still "
                      "biased >17%; entropy remains the most robust",
            "expected_eff_rank_95_bias_raw": expected_eff_rank_bias(n_samples, "eff_rank_95_raw"),
            "expected_eff_rank_95_bias_lw": expected_eff_rank_bias(n_samples, "eff_rank_95_lw"),
        }
    return {
        "recommended": "eff_rank_95 (raw)",
        "reason": "N>=4000: LW variant stabilizes (<10% bias); raw is converged at reference",
        "expected_eff_rank_95_bias_raw": 0.0,
        "expected_eff_rank_95_bias_lw": expected_eff_rank_bias(n_samples, "eff_rank_95_lw"),
    }
