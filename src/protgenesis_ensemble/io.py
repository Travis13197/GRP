"""BioEmu NPZ I/O and ensemble schema validation.

BioEmu sampler output convention (project-verified): each sequence directory
contains ``batch_*.npz`` files with key ``pos`` of shape
``(n_samples, n_residues, 3)`` (Cα coordinates, Å). The legacy key
``positions`` is accepted as a fallback.
"""

from __future__ import annotations

import pathlib
from typing import Union

import numpy as np

PathLike = Union[str, pathlib.Path]


def load_bioemu_npz(path: PathLike) -> np.ndarray:
    """Load a single BioEmu NPZ file.

    Returns
    -------
    positions : (n_samples, n_residues, 3) float64 array

    Raises
    ------
    KeyError : if neither 'pos' nor 'positions' is present.
    """
    path = pathlib.Path(path)
    with np.load(path, allow_pickle=False) as data:
        for key in ("pos", "positions"):
            if key in data:
                pos = np.asarray(data[key], dtype=np.float64)
                break
        else:
            raise KeyError(f"{path}: no 'pos' or 'positions' key; found {list(data.keys())}")
    if pos.ndim != 3 or pos.shape[2] != 3:
        raise ValueError(f"{path}: expected (n_samples, n_residues, 3), got {pos.shape}")
    return pos


def load_ensemble_dir(seq_dir: PathLike, pattern: str = "batch_*.npz") -> np.ndarray:
    """Load and concatenate all NPZ batches of one sequence directory.

    Parameters
    ----------
    seq_dir : directory containing batch NPZ files
    pattern : glob pattern (default 'batch_*.npz')

    Returns
    -------
    positions : (n_total_samples, n_residues, 3) float64 array

    Raises
    ------
    FileNotFoundError : if no matching NPZ files exist.
    """
    seq_dir = pathlib.Path(seq_dir)
    files = sorted(seq_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no {pattern!r} files in {seq_dir}")
    parts = [load_bioemu_npz(f) for f in files]
    n_res = {p.shape[1] for p in parts}
    if len(n_res) != 1:
        raise ValueError(f"inconsistent residue counts across batches: {n_res}")
    return np.concatenate(parts, axis=0)
