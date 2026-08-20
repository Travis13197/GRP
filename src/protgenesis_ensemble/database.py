"""Geometry database schema validation and integrity checks.

The published geometry database (``database/geometry_db_v0.1.0.json``) is a
JSON document with the following schema (see ``export_geometry_db.py``)::

    top-level:
        project        : str
        schema_version : str
        generated_by   : str
        conventions    : dict
        n_records      : int  (== len(records))
        records        : list[record]

    record (v0.2.0):
        uid         : str  (composite key ``f"{source}:{seq_id}"``; unique)
        seq_id      : str  (display name; may repeat across sources by design)
        category    : str
        n_residues  : int
        n_samples   : int
        features    : dict[str, float]   (all finite)
        source      : str
        sampler     : str

``validate_geometry_db`` checks the schema, per-record field presence,
numeric/finiteness of features, duplicate ``uid``, ``n_records``
consistency and (optionally) the SHA-256 sidecar.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
from typing import Optional, Union

PathLike = Union[str, pathlib.Path]

TOP_LEVEL_REQUIRED = (
    "project",
    "schema_version",
    "generated_by",
    "conventions",
    "n_records",
    "records",
)
RECORD_REQUIRED = (
    "uid",
    "seq_id",
    "category",
    "n_residues",
    "n_samples",
    "features",
    "source",
    "sampler",
)


def load_geometry_db(path: PathLike) -> dict:
    """Load a geometry database JSON document."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def sha256_of(path: PathLike) -> str:
    """Streaming SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(db_path: PathLike, sha256_path: PathLike) -> tuple[bool, str]:
    """Compare a database file against its ``.sha256`` sidecar.

    The sidecar is the standard ``<hexdigest>  <filename>`` format.
    """
    expected: Optional[str] = None
    with open(sha256_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                expected = parts[0].lower()
                break
    if expected is None:
        return False, "no hash found in sidecar file"
    actual = sha256_of(db_path)
    return expected == actual, f"expected={expected} actual={actual}"


def validate_geometry_db(
    db_path: PathLike,
    sha256_path: Optional[PathLike] = None,
    require_finite: bool = True,
) -> dict:
    """Validate a geometry database file.

    Returns a report dict::

        {
            "ok": bool,
            "n_records": int,
            "errors": [str, ...],
            "warnings": [str, ...],
            # only when sha256_path is given:
            "sha256_ok": bool,
            "sha256_message": str,
        }
    """
    db = load_geometry_db(db_path)
    errors: list[str] = []
    warnings: list[str] = []

    missing = [k for k in TOP_LEVEL_REQUIRED if k not in db]
    if missing:
        errors.append(f"missing top-level keys: {missing}")

    records = db.get("records")
    if not isinstance(records, list):
        errors.append("'records' must be a list")
        records = []
    if "n_records" in db and db["n_records"] != len(records):
        errors.append(f"n_records={db['n_records']} != len(records)={len(records)}")

    seen: set = set()
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            errors.append(f"record[{i}] is not an object")
            continue
        rm = [k for k in RECORD_REQUIRED if k not in rec]
        if rm:
            errors.append(f"record[{i}] missing keys: {rm}")
        uid = rec.get("uid")
        if not isinstance(uid, str) or not uid:
            errors.append(f"record[{i}] uid missing/empty")
        elif uid in seen:
            errors.append(f"duplicate uid: {uid}")
        seen.add(uid)
        if not isinstance(rec.get("seq_id"), str) or not rec.get("seq_id"):
            errors.append(f"record[{i}] seq_id missing/empty")

        feats = rec.get("features")
        if not isinstance(feats, dict):
            if "features" in rec:
                errors.append(f"record[{i}] features is not an object")
            continue
        for k, v in feats.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                errors.append(f"record[{i}] features[{k!r}] not numeric: {v!r}")
            elif require_finite and not math.isfinite(float(v)):
                errors.append(f"record[{i}] features[{k!r}] non-finite: {v!r}")

        # cross-field consistency (warnings only: schema permits legacy drift)
        if "n_samples" in rec and isinstance(feats.get("n_samples"), (int, float)):
            if float(rec["n_samples"]) != float(feats["n_samples"]):
                warnings.append(
                    f"record[{i}] {uid}: rec.n_samples={rec['n_samples']} "
                    f"!= features.n_samples={feats['n_samples']}"
                )
        if "n_residues" in rec and isinstance(feats.get("length"), (int, float)):
            if float(rec["n_residues"]) != float(feats["length"]):
                warnings.append(
                    f"record[{i}] {uid}: rec.n_residues={rec['n_residues']} "
                    f"!= features.length={feats['length']}"
                )

    report = {
        "ok": not errors,
        "n_records": len(records),
        "errors": errors,
        "warnings": warnings,
    }
    if sha256_path is not None:
        ok, msg = verify_sha256(db_path, sha256_path)
        report["sha256_ok"] = ok
        report["sha256_message"] = msg
        report["ok"] = report["ok"] and ok
    return report
