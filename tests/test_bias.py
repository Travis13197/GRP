"""Tests for finite-sample bias guidance (Phase CS1)."""

import numpy as np

from protgenesis_ensemble import (
    dimension_guidance,
    eff_rank_95,
    expected_eff_rank_bias,
    participation_entropy,
)


def test_participation_entropy_uniform():
    ev = np.ones(10)
    assert np.isclose(participation_entropy(ev), np.log(10))


def test_participation_entropy_concentrated():
    ev = np.array([1.0, 0.0, 0.0])
    assert np.isclose(participation_entropy(ev), 0.0)


def test_eff_rank_95_uniform():
    ev = np.ones(20)
    assert eff_rank_95(ev) == 19  # ceil(20*0.95)


def test_eff_rank_95_single_mode():
    ev = np.array([1.0, 0.0, 0.0, 0.0])
    assert eff_rank_95(ev) == 1


def test_expected_bias_negative_for_raw_low_n():
    # CS1: eff_rank_95 raw biased low at N=250
    assert expected_eff_rank_bias(250, "eff_rank_95_raw") < 0


def test_expected_bias_positive_for_lw_low_n():
    # CS1: LW variant biased high at N=250
    assert expected_eff_rank_bias(250, "eff_rank_95_lw") > 0


def test_expected_bias_zero_at_reference():
    assert expected_eff_rank_bias(5000, "eff_rank_95_raw") == 0.0


def test_unknown_estimator_raises():
    try:
        expected_eff_rank_bias(250, "nope")
    except KeyError:
        return
    raise AssertionError("expected KeyError")


def test_dimension_guidance_prefers_entropy_at_working_depth():
    g = dimension_guidance(250)
    assert g["recommended"] == "participation_entropy"
    assert g["expected_eff_rank_95_bias_raw"] < 0
    assert g["expected_eff_rank_95_bias_lw"] > 0


def test_dimension_guidance_allows_effrank_at_large_n():
    g = dimension_guidance(5000)
    assert g["recommended"] == "eff_rank_95 (raw)"
