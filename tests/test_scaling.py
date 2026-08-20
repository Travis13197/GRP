"""Tests for scaling-law fitting and breakpoint detection."""

import numpy as np

from protgenesis_ensemble import detect_transition, fit_scaling_law


def test_fit_scaling_law_recovers_exponent():
    n = np.arange(5, 51)
    y = 2.5 * n ** (-0.24)  # noise-free power law
    f = fit_scaling_law(n, y)
    assert abs(f["beta"] - (-0.24)) < 1e-6
    assert f["r2"] > 0.999
    assert f["p_value"] < 1e-10


def test_fit_scaling_law_noisy():
    rng = np.random.default_rng(11)
    n = np.arange(5, 51)
    y = 3.0 * n ** 0.5 * np.exp(rng.standard_normal(len(n)) * 0.05)
    f = fit_scaling_law(n, y)
    assert 0.4 < f["beta"] < 0.6
    assert f["p_value"] < 0.001


def test_detect_transition_finds_true_breakpoint():
    rng = np.random.default_rng(12)
    n = np.arange(4, 51)
    y = np.where(n <= 25, 0.05 * n + 1.0, 0.30 * n - 5.25)
    y = y + rng.standard_normal(len(n)) * 0.05
    res = detect_transition(n, y, n_bootstrap=200, rng=12)
    assert abs(res["n_cp"] - 25) <= 3
    assert res["slope_after"] > res["slope_before"]
    lo, hi = res["ci95"]
    assert lo <= res["n_cp"] <= hi


def test_detect_transition_deterministic():
    rng = np.random.default_rng(13)
    n = np.arange(4, 51)
    y = np.where(n <= 20, 0.1 * n, 0.4 * n - 6.0) + rng.standard_normal(len(n)) * 0.05
    r1 = detect_transition(n, y, n_bootstrap=100, rng=99)
    r2 = detect_transition(n, y, n_bootstrap=100, rng=99)
    assert r1["n_cp"] == r2["n_cp"]
    assert r1["ci95"] == r2["ci95"]


def test_fit_scaling_law_too_few_points():
    try:
        fit_scaling_law(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    except ValueError:
        return
    raise AssertionError("expected ValueError")
