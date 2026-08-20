"""Tests for seed engineering and NPZ I/O."""

import pathlib

import numpy as np

from protgenesis_ensemble import load_bioemu_npz, load_ensemble_dir, seed_stream, set_global_seed


def test_set_global_seed_determinism():
    set_global_seed(123)
    a = np.random.standard_normal(10)
    set_global_seed(123)
    b = np.random.standard_normal(10)
    assert np.array_equal(a, b)


def test_seed_stream_stable_and_distinct():
    s1 = seed_stream(42, "PolyX_PolyG_30", "pairwise")
    s2 = seed_stream(42, "PolyX_PolyG_30", "pairwise")
    s3 = seed_stream(42, "PolyX_PolyG_31", "pairwise")
    assert s1 == s2
    assert s1 != s3
    assert 0 <= s1 < 2**32


def test_npz_roundtrip(tmp_path: pathlib.Path):
    pos = np.random.default_rng(0).standard_normal((7, 11, 3))
    np.savez(tmp_path / "batch_000.npz", pos=pos)
    loaded = load_bioemu_npz(tmp_path / "batch_000.npz")
    assert loaded.shape == (7, 11, 3)
    assert np.allclose(loaded, pos)


def test_npz_legacy_key(tmp_path: pathlib.Path):
    pos = np.random.default_rng(1).standard_normal((5, 8, 3))
    np.savez(tmp_path / "batch_000.npz", positions=pos)
    loaded = load_bioemu_npz(tmp_path / "batch_000.npz")
    assert np.allclose(loaded, pos)


def test_load_ensemble_dir_concatenates(tmp_path: pathlib.Path):
    rng = np.random.default_rng(2)
    np.savez(tmp_path / "batch_000.npz", pos=rng.standard_normal((10, 6, 3)))
    np.savez(tmp_path / "batch_001.npz", pos=rng.standard_normal((15, 6, 3)))
    all_pos = load_ensemble_dir(tmp_path)
    assert all_pos.shape == (25, 6, 3)


def test_load_ensemble_dir_missing(tmp_path: pathlib.Path):
    try:
        load_ensemble_dir(tmp_path)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")
