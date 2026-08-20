"""Tests for geometry database schema validation and integrity checks."""

import json
import pathlib

import pytest

from protgenesis_ensemble import (
    load_geometry_db,
    validate_geometry_db,
    verify_sha256,
)


def _valid_db(n: int = 2) -> dict:
    return {
        "project": "protgenesis-ensemble",
        "schema_version": "0.1.0",
        "generated_by": "export_geometry_db.py",
        "conventions": {"unit": "nm"},
        "n_records": n,
        "records": [
            {
                "uid": f"polyx_ensemble:PolyX_PolyG_{i + 1}",
                "seq_id": f"PolyX_PolyG_{i + 1}",
                "category": "polyX",
                "n_residues": i + 1,
                "n_samples": 250,
                "features": {
                    "PR": 3.0 + i,
                    "spectral_decay": 1.5,
                    "length": float(i + 1),
                    "n_samples": 250.0,
                },
                "source": "polyx_ensemble",
                "sampler": "bioemu",
            }
            for i in range(n)
        ],
    }


def _write(path: pathlib.Path, obj) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_valid_db_passes(tmp_path: pathlib.Path):
    p = tmp_path / "db.json"
    _write(p, _valid_db())
    report = validate_geometry_db(p)
    assert report["ok"] is True
    assert report["n_records"] == 2
    assert report["errors"] == []


def test_n_records_mismatch_fails(tmp_path: pathlib.Path):
    db = _valid_db()
    db["n_records"] = 99
    p = tmp_path / "db.json"
    _write(p, db)
    report = validate_geometry_db(p)
    assert report["ok"] is False
    assert any("n_records" in e for e in report["errors"])


def test_duplicate_uid_fails(tmp_path: pathlib.Path):
    db = _valid_db()
    db["records"][1]["uid"] = db["records"][0]["uid"]
    p = tmp_path / "db.json"
    _write(p, db)
    report = validate_geometry_db(p)
    assert report["ok"] is False
    assert any("duplicate uid" in e for e in report["errors"])


def test_cross_source_same_seq_id_is_valid(tmp_path: pathlib.Path):
    """seq_id may repeat across sources (composite identity is uid)."""
    db = _valid_db(1)
    second = json.loads(json.dumps(db["records"][0]))
    second["uid"] = "phase9_systemwide:PolyX_PolyG_1"
    second["source"] = "phase9_systemwide"
    db["records"].append(second)
    db["n_records"] = 2
    p = tmp_path / "db.json"
    _write(p, db)
    report = validate_geometry_db(p)
    assert report["ok"] is True, report["errors"]


def test_missing_record_keys_fails(tmp_path: pathlib.Path):
    db = _valid_db()
    del db["records"][0]["sampler"]
    p = tmp_path / "db.json"
    _write(p, db)
    report = validate_geometry_db(p)
    assert report["ok"] is False
    assert any("missing keys" in e and "sampler" in e for e in report["errors"])


def test_nonfinite_feature_fails(tmp_path: pathlib.Path):
    db = _valid_db()
    db["records"][0]["features"]["PR"] = float("nan")
    p = tmp_path / "db.json"
    _write(p, db)
    report = validate_geometry_db(p)
    assert report["ok"] is False
    assert any("non-finite" in e for e in report["errors"])


def test_non_numeric_feature_fails(tmp_path: pathlib.Path):
    db = _valid_db()
    db["records"][0]["features"]["PR"] = "three"
    p = tmp_path / "db.json"
    _write(p, db)
    report = validate_geometry_db(p)
    assert report["ok"] is False
    assert any("not numeric" in e for e in report["errors"])


def test_sha256_sidecar(tmp_path: pathlib.Path):
    import hashlib

    db_path = tmp_path / "db.json"
    db_path.write_bytes(b'{"records": []}')
    digest = hashlib.sha256(db_path.read_bytes()).hexdigest()
    sidecar = tmp_path / "db.json.sha256"
    sidecar.write_text(f"{digest}  db.json\n", encoding="utf-8")
    ok, msg = verify_sha256(db_path, sidecar)
    assert ok is True
    assert digest in msg

    sidecar_bad = tmp_path / "bad.sha256"
    sidecar_bad.write_text("0" * 64 + "  db.json\n", encoding="utf-8")
    ok2, _ = verify_sha256(db_path, sidecar_bad)
    assert ok2 is False


def test_load_geometry_db(tmp_path: pathlib.Path):
    p = tmp_path / "db.json"
    _write(p, _valid_db(1))
    db = load_geometry_db(p)
    assert db["records"][0]["seq_id"] == "PolyX_PolyG_1"


def test_real_published_database():
    """Smoke test against the published v0.2.0 database + its SHA-256 sidecar."""
    root = pathlib.Path(__file__).resolve().parents[1]
    db_path = root / "database" / "geometry_db_v0.2.0.json"
    sidecar = root / "database" / "geometry_db_v0.2.0.sha256"
    if not db_path.exists():
        pytest.skip("published geometry database not present")
    report = validate_geometry_db(db_path, sha256_path=sidecar)
    assert report["ok"] is True, report["errors"][:5]
    assert report["n_records"] > 1000
    assert report["sha256_ok"] is True
    assert report["warnings"] == []
