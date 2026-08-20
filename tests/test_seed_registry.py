"""Tests for the persistent seed registry."""

import json
import pathlib

import pytest

from protgenesis_ensemble import get_seed, load_seed_registry, register_seed


def test_register_and_get(tmp_path: pathlib.Path):
    reg_path = tmp_path / "seeds.json"
    entry = register_seed("phase_o4_sampling.hras_d108g", 42, "HRAS D108G", reg_path)
    assert entry["seed"] == 42
    assert entry["stage"] == "phase_o4_sampling.hras_d108g"
    assert get_seed("phase_o4_sampling.hras_d108g", reg_path) == 42
    assert reg_path.exists()


def test_duplicate_stage_rejected(tmp_path: pathlib.Path):
    reg_path = tmp_path / "seeds.json"
    register_seed("stage_a", 1, "", reg_path)
    with pytest.raises(ValueError):
        register_seed("stage_a", 2, "", reg_path)


def test_allow_overwrite(tmp_path: pathlib.Path):
    reg_path = tmp_path / "seeds.json"
    register_seed("stage_a", 1, "", reg_path)
    register_seed("stage_a", 99, "updated", reg_path, allow_overwrite=True)
    assert get_seed("stage_a", reg_path) == 99


def test_missing_stage_raises(tmp_path: pathlib.Path):
    reg_path = tmp_path / "seeds.json"
    with pytest.raises(KeyError):
        get_seed("nope", reg_path)


def test_load_missing_registry_returns_empty(tmp_path: pathlib.Path):
    assert load_seed_registry(tmp_path / "absent.json") == {}


def test_registry_persists_json(tmp_path: pathlib.Path):
    reg_path = tmp_path / "seeds.json"
    register_seed("stage_x", 7, "desc", reg_path)
    raw = json.loads(reg_path.read_text(encoding="utf-8"))
    assert raw["stage_x"]["seed"] == 7
    assert "registered_at" in raw["stage_x"]
