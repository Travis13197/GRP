# An Effective Geometric Field Theory of Protein Space

**GRP — General Relativity of Protein.** A representation-aware statistical framework that treats sequence, structure, conformational ensemble and trajectory as connected layers of one protein state space.

> *"Biological constraints tell protein space how to curve, and curved space geometry tells proteins how to change."*

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

Proteins do not function as single static structures, but through coupled changes across sequence, structure, conformational ensembles and trajectories. These four levels are usually studied separately. **GRP** takes their *coupled state space* as primary:

| Layer | Role in GRP |
|-------|-------------|
| Sequence | indexes and constrains the space |
| Structure | a state within it |
| Ensemble | a probability measure over accessible states |
| Trajectory | a path through those states |

The conceptual shift is **from objects to distributions, from static configurations to organized possibilities, and from a science of points to a science of paths.** AI-based molecular representations (BioEmu for conformations; ProstT5, ProtT5 and ESM-C for learned embeddings) transform discrete biological observations into computational objects whose covariance structure, effective metrics and path geometry can be quantified.

GRP adopts a **methodological rather than literal** analogy with general relativity: it asks what remains valid when coordinates, models, scales or representations change, and how such structures can be described mathematically. It does **not** identify learned protein geometry with physical spacetime, inverse covariance with a molecular force-constant matrix, geometric cost with free energy, or transport length with a physical action.

The framework is described in the BioRxiv manuscript:

> **Liu, C.** *An Effective Geometric Field Theory of Protein Space.* bioRxiv (2026). ([`docs/paper_GRP2.0.pdf`](docs/paper_GRP2.0.pdf))

---

## The five results

### 1. Protein states define local effective geometry across protein space

Conformational ensembles (1,323 quality-controlled sequence-defined systems: PolyX 750, heteropolymer 239, hydrophobic-gradient 188, linker 94, intrinsically disordered 44, DMS-associated 8) are summarized by their **intrinsic covariance** after rigid-body (Kabsch) alignment. Normalized spectral summaries — participation ratio, anisotropy, spectral decay — reveal reproducible chain-length scaling (participation-ratio pooled log-log Pearson *r* = 0.78, *P* = 7.5×10⁻⁴³, *n* = 208; spectral-decay *β* = −0.66, *R²* = 0.95). These descriptors are statistical summaries of the declared representation, not intrinsic manifold dimensions.

### 2. Covariance-normalized perturbation geometry captures mutational constraint

The regularized local metric **G_S = Ĉ_S,LW⁻¹** (Ledoit–Wolf optimal shrinkage) defines a Mahalanobis-type perturbation cost

**C_geo(𝒫|S) = δzᵀ G_S δz**

that down-weights high-variance background directions. C_geo is interpreted as a **statistical atypicality measure** — not a force, elastic energy, mutation free energy or Hessian cost. Across eight proteins and 84,361 mutations it shows a modest negative association with experimental fitness (mean Spearman *ρ* = −0.1475), and matched Cα/full-atom costs correspond weakly (*r* = 0.219, *n* = 376). Directional-consistency and coefficient-of-variation statistics are retained as *transformation diagnostics*, not independent biological validation.

### 3. Biological constraints define a predictive effective-geometric field

Non-redundant biological source variables (chain length, composition contrasts, physicochemical descriptors, contextual variables) predict ensemble-geometry observables through a local response operator **K_r(t₀)**. 73.6% of per-amino-acid responses reached *R²* > 0.1; the estimated response-coupling matrix is reproducibly organized (bootstrap adjusted Rand index 0.708; 18 leave-one-amino-acid-out analyses ARI = 1.0). The result supports a **partially transferable, representation-dependent** response field — not a coordinate-free causal law.

### 4. Structured protein-state paths show reduced mean Gaussian transport

Ordered state paths (chain-length progressions, mutation sequences, generated candidates) are mapped into a common comparison space and summarized by the mean adjacent-state Gaussian transport W̅₂,G. Structured paths show lower transport than matched null paths (primary: 11.75 vs 22.90, Cohen's *d* = −1.54, *P* = 8.4×10⁻⁶⁸; heteropolymer 3.07 vs 3.46; direct-NPZ 7.62 vs 9.29). This establishes a **low-transport path relation**, not a complete minimum-action principle.

### 5. Unified effective-geometric organization

The five analyses define distinct but connected mathematical constructions — ensemble → geometry, ensemble → metric, (ensemble, perturbation) → cost, source ⇢ geometry, path → transport. Together they form a multiscale effective statistical framework; the mappings do **not** constitute a deterministic causal chain.

---

## Repository layout

```
.
├── src/protgenesis_ensemble/   # canonical, installable Python package (core geometry)
├── tests/                      # unit tests (deterministic, seed-aware)
├── database/                   # ensemble-geometry database (1,904 records + SHA-256)
├── analysis/                   # manuscript analysis scripts, organized by result
│   ├── common/                 #   Kabsch/Ledoit-Wolf metric, statistical hygiene
│   ├── r1_ensemble_geometry/
│   ├── r2_perturbation_cost/
│   ├── r3_source_field/
│   ├── r4_path_transport/
│   ├── r5_cross_representation/
│   └── sampling/               #   FASTA/A3M preparation
├── data/
│   ├── dms/                    #   deep-mutational-scanning fitness (8 proteins)
│   └── sequences/              #   wild-type FASTA definitions
├── figures/
│   ├── main/                   #   Figures 1–4 (PDF/PNG/SVG)
│   └── supplementary/          #   Figures S1–S9 (PDF/PNG/SVG)
├── config/                     #   conda environment specifications
├── docs/                       #   manuscript PDF + figure legends
├── README.md
├── AI_CODER.md                 #   full setup / run / reproduction guide for AI agents
├── pyproject.toml
└── LICENSE
```

---

## Installation

The core geometric machinery is a pure-Python package with NumPy/SciPy/scikit-learn only.

```bash
git clone https://github.com/<your-org>/grp.git
cd grp
pip install -e .
# with test dependencies
pip install -e ".[test]"
```

Run the test suite and database schema check:

```bash
python -m pytest tests/ -q
python -c "import protgenesis_ensemble as pe; assert pe.__version__"
```

---

## Quick start

```python
import numpy as np
from protgenesis_ensemble import (
    kabsch_align_ensemble, compute_ensemble_geometry,
    build_residue_metrics, compute_cgeo_mutation, set_global_seed,
)

set_global_seed(42)

# positions: (n_samples, n_residues, 3) Cα ensemble, e.g. from BioEmu NPZ ('pos')
aligned = kabsch_align_ensemble(positions, reference_mode="first")
geom = compute_ensemble_geometry(aligned)
print(geom["PR"], geom["spectral_decay"], geom["entropy"])

# regularized per-residue metric, then a single-substitution cost
g_blocks = build_residue_metrics(aligned)          # (n_residues, 3, 3)
c_geo = compute_cgeo_mutation(g_blocks, pos=42, wt_aa="G", mut_aa="W", rng=42)
```

Load BioEmu data directly:

```python
from protgenesis_ensemble import load_ensemble_dir
positions = load_ensemble_dir("output/PolyX_PolyG_30")  # reads batch_*.npz ('pos')
```

See [`AI_CODER.md`](AI_CODER.md) for the complete end-to-end reproduction of every manuscript result.

---

## Data and reproducibility

The repository ships the **minimal inputs** required to reproduce the manuscript analyses:

- **DMS fitness** (`data/dms/`): processed deep-mutational-scanning tables for the eight proteins (BLAT, GFP, HRAS, HSP90, P53, PTEN, SPIKE_RBD, UBE4B; 84,361 mutations). Raw ProteinGym sources are publicly available and documented in `AI_CODER.md`.
- **Sequence definitions** (`data/sequences/`): wild-type FASTA for the DMS proteins.
- **Geometry database** (`database/geometry_db_v0.2.0.json`): 1,904 pre-computed ensemble-geometry records (PolyX 354 + heteropolymer 271 + system-wide 1,279), with SHA-256 sidecar.

**Generated conformations are intentionally not committed.** Conformational ensembles (BioEmu NPZ, ~4.4 GB for the full atlas) and learned embeddings are regenerated with the commands in `AI_CODER.md`. The full pipeline requires:

- **BioEmu** (GPU) for conformational sampling;
- **ProstT5 / ProtT5 / ESM-C** for learned representations;
- the **CPU-only** package in this repository for all geometric analysis.

---

## Figures

Final manuscript figures are provided in `figures/` (PDF, PNG and SVG):

| Figure | Content |
|--------|---------|
| Fig. 1 | Protein ensembles define local effective geometry |
| Fig. 2 | Covariance normalization makes perturbations comparable across backgrounds |
| Fig. 3 | Biological source variables predict an effective geometric response field |
| Fig. 4 | Structured protein-state paths show reduced mean Gaussian transport |
| Figs. S1–S9 | Per-amino-acid spectra, DMS per protein, fingerprints, field/path support, UMAP/t-SNE atlas |

Figure legends are in [`docs/figure_legends.md`](docs/figure_legends.md).

---

## Citation

If you use this code or data, please cite:

> Liu, C. **An Effective Geometric Field Theory of Protein Space.** bioRxiv (2026).

---

## License

MIT — see [LICENSE](LICENSE).
