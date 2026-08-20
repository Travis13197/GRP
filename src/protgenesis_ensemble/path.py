"""Path geometry for low-action / allosteric channel analysis (Phase Bio3/3b).

Given a path's efficiency ``eff = D / L`` (end-to-end Wasserstein-2 distance
divided by total path length), the circular-arc model yields the arc angle
``theta`` (eff = 2 sin(theta/2) / theta) and, when the path length ``L`` is
known, the absolute curvature radius ``R = L / theta``.

Project findings (TECHNICAL_REFERENCE 6.52/6.54): two-state allosteric paths
outside Law 3's validity domain have theta in [4.28, 5.16] rad (> 180 deg,
"curved higher-dimensional channels"); the radius scales with chain length
(Spearman rho ~ 0.90, robust to path bin discretization) while the angle is
roughly universal.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def arc_angle_from_efficiency(efficiency: float) -> float:
    """Invert ``eff = 2 sin(theta/2) / theta`` for theta in (0, 2*pi].

    theta -> 0 corresponds to a straight path (eff -> 1); theta = pi is a
    semicircle (eff = 2/pi); theta -> 2*pi is a full circle (eff -> 0).
    """
    eff = float(efficiency)
    if eff <= 0:
        return float(2 * np.pi)
    if eff >= 1.0:
        return 0.0
    return float(brentq(lambda t: 2 * np.sin(t / 2) / t - eff, 1e-6, 2 * np.pi))


def curvature_radius(path_length: float, efficiency: float) -> float:
    """Absolute curvature radius R = L / theta for a circular arc."""
    theta = arc_angle_from_efficiency(efficiency)
    if theta <= 0 or not np.isfinite(path_length):
        return float("nan")
    return float(path_length / theta)


def detour_ratio(efficiency: float) -> float:
    """L/D = 1/eff: how many times longer the path is than the straight chord."""
    eff = float(efficiency)
    return float(1.0 / eff) if eff > 0 else float("inf")


def implied_chord(theta: float, radius: float) -> float:
    """Chord D = 2 R sin(theta/2) consistent with the arc model."""
    return float(2 * radius * np.sin(theta / 2))
