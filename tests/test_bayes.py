"""Tests for inverse-Wishart Bayesian metric posterior (Phase CS2)."""

import numpy as np

from protgenesis_ensemble import (
    cgeo_posterior_ci,
    inverse_wishart_metric_samples,
    stiffness_posterior_ci,
)


def test_posterior_converges_to_sample_inverse_large_n():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 3))
    S = np.cov(X, rowvar=False)
    g = inverse_wishart_metric_samples(S, 2000, n_post=200, rng=1)
    g_point = np.linalg.inv(S)
    # 大 N 下后验中位接近样本协方差逆
    assert np.allclose(np.median(g, axis=0), g_point, rtol=0.15)


def test_stiffness_posterior_positive():
    rng = np.random.default_rng(1)
    S = np.eye(3) * 0.5
    g = inverse_wishart_metric_samples(S, 250, n_post=200, rng=2)
    st = stiffness_posterior_ci(g)
    assert st["median"] > 0
    assert st["ci_lo"] < st["median"] < st["ci_hi"]


def test_cgeo_posterior_contains_point_estimate():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((250, 3))
    S = np.cov(X, rowvar=False)
    g = inverse_wishart_metric_samples(S, 250, n_post=500, rng=4)
    d = rng.standard_normal(3)
    ci = cgeo_posterior_ci(d, g)
    point = float(d @ np.linalg.inv(S) @ d)
    assert ci["ci_lo"] < point < ci["ci_hi"]


def test_cgeo_positive_definite():
    rng = np.random.default_rng(5)
    S = np.diag([1.0, 2.0, 3.0])
    g = inverse_wishart_metric_samples(S, 100, n_post=100, rng=6)
    ci = cgeo_posterior_ci(np.array([1.0, 1.0, 1.0]), g)
    assert ci["median"] > 0
