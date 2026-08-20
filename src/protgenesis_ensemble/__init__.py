"""protgenesis-ensemble: ensemble geometric field theory for protein conformational ensembles.

Public API
----------
io          : BioEmu NPZ loading and schema validation
align       : Kabsch alignment (SE(3) removal)
covariance  : Ledoit-Wolf shrinkage, randomized low-rank eigendecomposition
geometry    : intrinsic ensemble geometry features (PR, eff_rank, spectral_decay, ...)
cgeo        : Law 1 C_geo stability metric (regularized Mahalanobis)
scaling     : scaling-law fitting and breakpoint (phase-transition) detection
seeds       : deterministic seed engineering
database    : geometry database schema validation and SHA-256 integrity checks
bias        : finite-sample bias guidance for rank estimators (CS1)
bayes       : inverse-Wishart Bayesian posterior for g_S / C_geo (CS2)
path        : path geometry for low-action / allosteric channels (Bio3/3b)
"""

from .align import kabsch_align, kabsch_align_ensemble
from .bayes import cgeo_posterior_ci, inverse_wishart_metric_samples, stiffness_posterior_ci
from .bias import (
    dimension_guidance,
    eff_rank_95,
    expected_eff_rank_bias,
    participation_entropy,
)
from .cgeo import (
    AA_CHARGES,
    AA_HYDROPHOBICITY,
    AA_VOLUMES,
    build_residue_metrics,
    compute_cgeo_mutation,
    mutation_displacement_magnitude,
)
from .covariance import ledoit_wolf_shrinkage, randomized_eigendecomposition
from .geometry import compute_ensemble_geometry
from .database import load_geometry_db, validate_geometry_db, verify_sha256
from .io import load_ensemble_dir, load_bioemu_npz
from .path import (
    arc_angle_from_efficiency,
    curvature_radius,
    detour_ratio,
    implied_chord,
)
from .scaling import detect_transition, fit_scaling_law
from .seeds import get_seed, load_seed_registry, register_seed, seed_stream, set_global_seed

__version__ = "0.4.0"

__all__ = [
    "kabsch_align",
    "kabsch_align_ensemble",
    "ledoit_wolf_shrinkage",
    "randomized_eigendecomposition",
    "compute_ensemble_geometry",
    "build_residue_metrics",
    "compute_cgeo_mutation",
    "mutation_displacement_magnitude",
    "AA_VOLUMES",
    "AA_CHARGES",
    "AA_HYDROPHOBICITY",
    "load_ensemble_dir",
    "load_bioemu_npz",
    "load_geometry_db",
    "validate_geometry_db",
    "verify_sha256",
    "fit_scaling_law",
    "detect_transition",
    "set_global_seed",
    "seed_stream",
    "register_seed",
    "get_seed",
    "load_seed_registry",
    "participation_entropy",
    "eff_rank_95",
    "expected_eff_rank_bias",
    "dimension_guidance",
    "inverse_wishart_metric_samples",
    "cgeo_posterior_ci",
    "stiffness_posterior_ci",
    "arc_angle_from_efficiency",
    "curvature_radius",
    "detour_ratio",
    "implied_chord",
    "__version__",
]
