"""Deterministic seed engineering and seed registry.

Project convention: every stochastic pipeline stage receives an explicit seed;
``set_global_seed`` synchronizes ``random``, ``numpy`` and (when installed)
``torch`` CPU+CUDA. ``seed_stream`` derives independent per-item seeds from a
master seed and arbitrary string keys (stable across processes and machines),
enabling deterministic resume of sharded workloads.

``register_seed`` / ``get_seed`` persist per-stage seeds to a JSON registry
(default ``seed_registry.json`` in the working directory), satisfying the
project's reproducibility requirement that every stochastic stage be auditable.
"""

from __future__ import annotations

import hashlib
import json
import random
import pathlib
from datetime import datetime, timezone
from typing import Optional

import numpy as np

SEED_REGISTRY_DEFAULT = "seed_registry.json"
PathLike = str | pathlib.Path


def set_global_seed(seed: int, torch_deterministic: bool = False) -> int:
    """Seed python ``random`` and ``numpy``; seed torch when importable.

    Parameters
    ----------
    seed : master seed
    torch_deterministic : if True, also enable torch deterministic algorithms
        (may slow down CUDA ops; requires torch)

    Returns
    -------
    the seed (for chaining/logging)
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch_deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
    return seed


def seed_stream(master_seed: int, *keys: object) -> int:
    """Derive a stable 32-bit child seed from a master seed and string keys.

    Example
    -------
    >>> s = seed_stream(42, "PolyX_PolyG_30", "pairwise_subsample")
    >>> rng = np.random.default_rng(s)

    The same (master_seed, keys) tuple always yields the same child seed,
    independent of process, platform, or PYTHONHASHSEED.
    """
    h = hashlib.sha256()
    h.update(str(int(master_seed)).encode("utf-8"))
    for k in keys:
        h.update(b"|")
        h.update(str(k).encode("utf-8"))
    return int.from_bytes(h.digest()[:4], "little")


def load_seed_registry(registry_path: Optional[PathLike] = None) -> dict:
    """Load the seed registry JSON (``{}`` when missing)."""
    p = pathlib.Path(registry_path) if registry_path else pathlib.Path(SEED_REGISTRY_DEFAULT)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def register_seed(
    stage: str,
    seed: int,
    description: str = "",
    registry_path: Optional[PathLike] = None,
    allow_overwrite: bool = False,
) -> dict:
    """Register the deterministic seed used by a pipeline stage.

    Stage keys should be namespaced, e.g. ``"phase_o4_sampling.hras_d108g"``.
    Raises ``ValueError`` when the stage is already registered unless
    ``allow_overwrite=True``. Returns the created entry.
    """
    reg = load_seed_registry(registry_path)
    if stage in reg and not allow_overwrite:
        raise ValueError(
            f"stage {stage!r} already registered (seed={reg[stage]['seed']}); "
            "pass allow_overwrite=True to replace"
        )
    entry = {
        "stage": stage,
        "seed": int(seed),
        "description": description,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    reg[stage] = entry
    p = pathlib.Path(registry_path) if registry_path else pathlib.Path(SEED_REGISTRY_DEFAULT)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(reg, fh, indent=2, ensure_ascii=False)
    return entry


def get_seed(stage: str, registry_path: Optional[PathLike] = None) -> int:
    """Return the registered seed for a stage (``KeyError`` when absent)."""
    reg = load_seed_registry(registry_path)
    if stage not in reg:
        raise KeyError(f"stage {stage!r} not registered")
    return int(reg[stage]["seed"])
