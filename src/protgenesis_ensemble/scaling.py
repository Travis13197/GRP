"""Scaling-law fitting and geometric phase-transition (breakpoint) detection.

Implements the project's standard analyses:
- power-law fit  y ~ n^beta  via OLS on log-log (with R^2 and p-value);
- two-segment breakpoint detection with bootstrap 95% CI, as used for the
  PolyX PR/spectral-decay transitions and the Phase M5 polyQ/polyA study.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats


def fit_scaling_law(n: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Fit y ~ n^beta on log-log axes.

    Returns dict with keys: beta, intercept, r2, p_value, pearson_r, n_points.
    """
    n = np.asarray(n, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = (n > 0) & (y > 0) & np.isfinite(n) & np.isfinite(y)
    if mask.sum() < 3:
        raise ValueError("need >=3 finite positive points")
    x = np.log(n[mask])
    t = np.log(y[mask])
    slope, intercept, r, p, _ = stats.linregress(x, t)
    return {
        "beta": float(slope),
        "intercept": float(intercept),
        "r2": float(r**2),
        "p_value": float(p),
        "pearson_r": float(r),
        "n_points": int(mask.sum()),
    }


def _two_segment_sse(x: np.ndarray, y: np.ndarray, cp: float) -> Tuple[float, float, float]:
    left = x <= cp
    right = ~left
    if left.sum() < 2 or right.sum() < 2:
        return np.inf, np.nan, np.nan
    s1, i1, *_ = stats.linregress(x[left], y[left])
    s2, i2, *_ = stats.linregress(x[right], y[right])
    pred = np.where(left, s1 * x + i1, s2 * x + i2)
    sse = float(((y - pred) ** 2).sum())
    return sse, float(s1), float(s2)


def detect_transition(
    n: np.ndarray,
    y: np.ndarray,
    cp_grid: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000,
    rng: int | np.random.Generator = 42,
) -> Dict[str, object]:
    """Detect a breakpoint n_cp in y(n) by two-segment OLS (minimum SSE).

    Parameters
    ----------
    n, y : arrays (chain lengths, geometric feature)
    cp_grid : candidate breakpoints (default: integer range 25th-75th pct of n)
    n_bootstrap : bootstrap resamples for the 95% CI
    rng : seed/Generator

    Returns
    -------
    dict with keys: n_cp, ci95 (tuple), slope_before, slope_after,
    sse, converged (bool: CI within grid range)
    """
    n = np.asarray(n, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(n) & np.isfinite(y)
    n, y = n[mask], y[mask]
    if len(n) < 8:
        raise ValueError("need >=8 points for breakpoint detection")

    order = np.argsort(n)
    n, y = n[order], y[order]

    if cp_grid is None:
        lo, hi = np.percentile(n, [25, 75])
        cp_grid = np.arange(np.ceil(lo), np.floor(hi) + 1)
    cp_grid = np.asarray(cp_grid, dtype=np.float64)

    def best_cp(nn: np.ndarray, yy: np.ndarray) -> Tuple[float, float, float, float]:
        best = (np.inf, np.nan, np.nan, np.nan)
        for cp in cp_grid:
            sse, s1, s2 = _two_segment_sse(nn, yy, cp)
            if sse < best[0]:
                best = (sse, cp, s1, s2)
        return best

    sse, cp, s1, s2 = best_cp(n, y)

    gen = np.random.default_rng(rng)
    boots = []
    for _ in range(n_bootstrap):
        idx = gen.integers(0, len(n), len(n))
        nb, yb = n[idx], y[idx]
        ob = np.argsort(nb)
        try:
            _, cpb, _, _ = best_cp(nb[ob], yb[ob])
        except Exception:
            continue
        if np.isfinite(cpb):
            boots.append(cpb)
    if len(boots) >= 30:
        ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        converged = bool(ci[0] > cp_grid.min() and ci[1] < cp_grid.max())
    else:
        ci = (float("nan"), float("nan"))
        converged = False

    return {
        "n_cp": float(cp),
        "ci95": ci,
        "slope_before": float(s1),
        "slope_after": float(s2),
        "sse": float(sse),
        "converged": converged,
    }
