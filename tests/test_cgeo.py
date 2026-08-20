"""Tests for Law 1 C_geo metric."""

import numpy as np

from protgenesis_ensemble import (
    build_residue_metrics,
    compute_cgeo_mutation,
    mutation_displacement_magnitude,
)


def _make_ensemble(rng, n_samples=300, n_res=15):
    base = rng.standard_normal((n_res, 3)) * 5
    modes = rng.standard_normal((4, n_res, 3))
    frames = []
    for _ in range(n_samples):
        coeffs = rng.standard_normal(4) * 1.5
        frames.append(base + np.tensordot(coeffs, modes, axes=1))
    return np.stack(frames)


def test_build_residue_metrics_symmetric_pd():
    rng = np.random.default_rng(7)
    pos = _make_ensemble(rng)
    g = build_residue_metrics(pos)
    assert g.shape == (15, 3, 3)
    for i in range(15):
        assert np.allclose(g[i], g[i].T, atol=1e-10)
        assert np.all(np.linalg.eigvalsh(g[i]) > 0)


def test_cgeo_nonnegative_and_deterministic():
    rng = np.random.default_rng(8)
    pos = _make_ensemble(rng)
    g = build_residue_metrics(pos)
    c1 = compute_cgeo_mutation(g, pos=5, wt_aa="G", mut_aa="W", rng=42)
    c2 = compute_cgeo_mutation(g, pos=5, wt_aa="G", mut_aa="W", rng=42)
    assert c1 == c2
    assert c1 > 0


def test_cgeo_larger_for_stiffer_residues():
    """A residue with smaller fluctuation must yield larger C_geo for the same d."""
    rng = np.random.default_rng(9)
    base = rng.standard_normal((10, 3)) * 5
    frames = []
    for _ in range(300):
        noise = rng.standard_normal((10, 3))
        noise[0] *= 3.0   # residue 0: floppy
        noise[1] *= 0.1   # residue 1: stiff
        frames.append(base + noise)
    pos = np.stack(frames)
    g = build_residue_metrics(pos)
    d = np.array([1.0, 0.0, 0.0])
    c_floppy = compute_cgeo_mutation(g, 0, "A", "V", direction=d)
    c_stiff = compute_cgeo_mutation(g, 1, "A", "V", direction=d)
    assert c_stiff > c_floppy


def test_displacement_magnitude_physicochemical():
    # identity mutation -> baseline 0.1
    assert mutation_displacement_magnitude("A", "A") == 0.1
    # G->W (huge volume jump) > A->V (small jump)
    assert mutation_displacement_magnitude("G", "W") > mutation_displacement_magnitude("A", "V")
    # charge swap adds magnitude (K->E charge jump 2, vs K->R no charge jump)
    assert mutation_displacement_magnitude("K", "E") > mutation_displacement_magnitude("K", "R")


def test_cgeo_out_of_range():
    rng = np.random.default_rng(10)
    g = build_residue_metrics(_make_ensemble(rng))
    try:
        compute_cgeo_mutation(g, pos=999, wt_aa="A", mut_aa="V")
    except IndexError:
        return
    raise AssertionError("expected IndexError")
