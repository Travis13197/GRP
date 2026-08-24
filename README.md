# An Effective Geometric Field Theory of Protein Space

**GRP — General Relativity of Protein.** A representation-aware statistical framework that treats sequence, structure, conformational ensemble and trajectory as connected layers of one protein state space.

> *"Biological constraints tell protein space how to curve, and curved space geometry tells proteins how to change"*

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## The idea

Proteins are usually studied as a sequence of static objects. GRP starts from a different premise: **what matters is not the object, but the space of possibilities around it.**

Four levels that are usually treated separately — sequence, structure, ensemble, trajectory — are really four views of one thing:

| Level | What it is |
|-------|------------|
| Sequence | constrains *which* states can exist |
| Structure | *one point* in that space |
| Ensemble | the *distribution* over accessible states |
| Trajectory | a *path* through those states |

The conceptual shift is therefore **from objects to distributions, from static configurations to organized possibilities, from a science of points to a science of paths.** A distribution has shape — spread, orientation, soft and stiff directions — and the working hypothesis is that biological constraints leave a *measurable, law-like* trace in that shape.

## The mirror-world hypothesis

AI representations (BioEmu conformations; ProstT5, ProtT5, ESM-C embeddings) are **not** the biological world — they are a *mirror* of it. Biological constraints imprint statistical traces in learned representations, and the mathematics of those traces can be read back as hypotheses about the original system.

This correspondence is deliberately **lossy, task-dependent and scale-limited**: it is not an identity between representation and reality. AI here is a scientific instrument that makes otherwise-invisible collective structure measurable. Its value is not to predict a fixed endpoint, but to *expose candidate variables and relations* that can later be translated into physical quantities and tested.

## The method — a methodological, not literal, analogy

General relativity shifted attention from coordinates to the relations that remain meaningful under coordinate change. GRP asks the same question for protein space: **what remains valid when coordinates, model, scale, or representation change?**

The machinery is deliberately simple, so every step can be named and questioned:

| Construction | Question it answers |
|--------------|---------------------|
| ensemble → geometry | How is a distribution organized (participation ratio, spectral decay, anisotropy)? |
| ensemble → metric | Which directions are soft vs. stiff (regularized inverse covariance G_S)? |
| perturbation → cost | How surprising is a displacement, given the background (C_geo = δzᵀ G_S δz)? |
| source → geometry | Can biological descriptors predict that organization (a local response operator)? |
| path → transport | How far apart are successive ensembles along an ordered path (Wasserstein)? |

Each arrow is a *construction*, not a law. Each is written in a declared representation and tested against matched nulls.

## What it shows

Across roughly 1,300 ensembles and 8 proteins / 84k mutations, the picture is coherent: aligned conformational covariance is reproducibly organized by chain length; a mutation is more comparable once normalized by the local background covariance; biological descriptors predict geometry; and ordered paths move through state space more coherently than matched shuffles. None of these is claimed as a physical law — each is a representation-dependent, statistically tested relation. Full numerical detail lives in [`docs/figure_legends.md`](docs/figure_legends.md) and the per-figure notes.

## What this is — and is not
This is a *language*, not a theory of forces: a bold but preliminary attempt to mathematize biology at the level of distributions rather than points.

It deliberately does **not** claim that learned geometry is physical spacetime, that inverse covariance is a molecular force-constant matrix, that C_geo is free energy, that transport is a physical action, or that any relation is coordinate-free or causal.

What it tries to offer is a representation-aware vocabulary connecting sequence variation, ensemble heterogeneity, mutational response and ordered paths — one whose value lies in generating testable hypotheses rather than assigning literal physical meaning. Whether these relations transfer to biological measurement is an open, empirical question. The mirror is lossy on purpose: the goal is not to replace reality with representation, but to make the *structure that persists across representations* visible and mathematically tractable.

The full argument is in the BioRxiv manuscript (Not Yet):

> **Liu, C.** *An Effective Geometric Field Theory of Protein Space.* bioRxiv (2026). ([`docs/paper_GRP2.0.pdf`](docs/paper_GRP2.0.pdf))

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

Final manuscript figures are provided in `figures/out/` (PDF, PNG, SVG and JPG) and are **fully reproducible from committed inputs** via the bundled workflow:

```bash
pip install numpy scipy pandas scikit-learn matplotlib seaborn statsmodels umap-learn
python figures/scripts/make_figures.py          # regenerate Figures 1-4 + S1-S9
```

| Figure | Content |
|--------|---------|
| Fig. 1 | Protein ensembles define local effective geometry |
| Fig. 2 | Covariance normalization makes perturbations comparable across backgrounds |
| Fig. 3 | Biological source variables predict an effective geometric response field |
| Fig. 4 | Structured protein-state paths show reduced mean Gaussian transport |
| Figs. S1–S9 | Per-amino-acid spectra, DMS per protein, fingerprints, field/path support, UMAP/t-SNE atlas |

Each figure has a dedicated, fully annotated document:

- [Figure 1](figures/Figure1.md) · [Figure 2](figures/Figure2.md) · [Figure 3](figures/Figure3.md) · [Figure 4](figures/Figure4.md) · [Supplementary figures](figures/Supplementary_Figures.md)

The reproduction workflow is defined in [`.github/workflows/figures.yml`](.github/workflows/figures.yml); figure inputs and provenance are described in [`figures/README.md`](figures/README.md), and panel-by-panel legends are in [`docs/figure_legends.md`](docs/figure_legends.md).

---

## Citation

If you use this code or data, please cite:

> Liu, C. **An Effective Geometric Field Theory of Protein Space.** bioRxiv (2026).

> Liu, C. **Universal physical principles govern the deterministic genesis of protein structure.** bioRxiv. (2026). doi: https://doi.org/10.64898/2026.02.20.706798.

---

## License

MIT — see [LICENSE](LICENSE).
