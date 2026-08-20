"""Tests for path geometry utilities (Phase Bio3/3b)."""

import numpy as np

from protgenesis_ensemble.path import (
    arc_angle_from_efficiency,
    curvature_radius,
    detour_ratio,
    implied_chord,
)


def test_straight_path_angle_zero():
    assert arc_angle_from_efficiency(1.0) == 0.0


def test_semicircle_angle_pi():
    assert np.isclose(arc_angle_from_efficiency(2 / np.pi), np.pi, atol=1e-6)


def test_monotonic_decreasing():
    # lower efficiency -> larger arc angle
    a = arc_angle_from_efficiency(0.9)
    b = arc_angle_from_efficiency(0.5)
    c = arc_angle_from_efficiency(0.2)
    assert a < b < c


def test_radius_consistency():
    # R = L/theta; chord = 2R sin(theta/2); efficiency = chord/L
    L, eff = 100.0, 0.3
    R = curvature_radius(L, eff)
    theta = arc_angle_from_efficiency(eff)
    D = implied_chord(theta, R)
    assert np.isclose(D / L, eff, atol=1e-6)


def test_detour_ratio():
    assert np.isclose(detour_ratio(0.25), 4.0)


def test_zero_efficiency_full_circle():
    assert np.isclose(arc_angle_from_efficiency(0.0), 2 * np.pi)
