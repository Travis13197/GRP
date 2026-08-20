# AI CODER — Project Guide, Environment, and Reproduction

This document is written for an autonomous coding agent (an "AI coder"). Its purpose is to let a fresh agent, starting from only this repository, (1) understand the project, (2) install all dependencies, (3) understand the analysis pipeline, and (4) reproduce the manuscript results without inventing data.

> **Read this file in full before running anything.** Reproducibility is the contract: every number in the manuscript must be traceable to a script in `analysis/` and an input in `data/` or `database/`.

---

## 1. What this project is

This is the codebase for the BioRxiv manuscript **"An Effective Geometric Field Theory of Protein Space"** (GRP, "General Relativity of Protein"), by Chuanyang Liu. See [`README.md`](README.md) and [`docs/paper_GRP2.0.pdf`](docs/paper_GRP2.0.pdf).

**Core claim (in one paragraph).** Proteins are described not as single structures but as *ensembles* (probability measures) over conformational or learned-representation states. After rigid-body alignment, an ensemble's **intrinsic covariance** defines a local effective geometry; a **Ledoit-Wolf-regularized Mahalanobis metric** turns a perturbation into a covariance-normalized cost `C_geo`; biological source variables predict geometry observables through a local response operator; and ordered state paths are compared through distributional (Gaussian Bures-Wasserstein) transport.

**Vocabulary you must use consistently** (matches the final manuscript, not earlier drafts):

| Term | Meaning | Do NOT say |
|------|---------|-----------|
| GRP | General Relativity of Protein (methodological analogy, not literal spacetime) | "3 laws of protein folding" |
| `C_geo` | covariance-normalized perturbation cost; a statistical atypicality measure | "force", "energy", "free energy", "Hessian cost" |
| intrinsic covariance | Kabsch-aligned, Ledoit-Wolf-shrunk covariance | "manifold dimension", "physical topology" |
| participation ratio / spectral decay / anisotropy | normalized spectral summaries | "intrinsic dimension" |
| W2G | mean adjacent-state Gaussian Bures-Wasserstein transport | "physical action" |
| low-transport relation | empirical, representation-dependent path result | "minimum-action principle" |
| effective geometric field | representation-indexed predictive source-to-geometry map | "coordinate-free field law" |

The manuscript is deliberately conservative: it reports empirical, representation-dependent, matched-null relations, not universal physical laws. Keep that tone in any code comments, docs, or summaries you write.

---

## 2. Repository layout

```
.
├── src/protgenesis_ensemble/   # canonical package (pure Python, CPU)
│   ├── io.py                   #   BioEmu NPZ loading + schema validation
│   ├── align.py                #   Kabsch alignment (SE(3) removal)
│   ├── covariance.py           #   Ledoit-Wolf shrinkage, randomized low-rank eigh
│   ├── geometry.py             #   PR, eff_rank, spectral_decay, anisotropy, entropy
│   ├── cgeo.py                 #   C_geo metric (regularized Mahalanobis)
│   ├── scaling.py              #   log-log scaling fits + breakpoint detection
│   ├── path.py                 #   path/curvature helpers
│   ├── seeds.py                #   deterministic seed registry
│   ├── bias.py                 #   finite-sample bias guidance (CS1)
│   ├── bayes.py                #   inverse-Wishart posterior for g_S / C_geo (CS2)
│   └── database.py             #   geometry_db schema + SHA-256 integrity
├── tests/                      # unit tests; runnable with `pytest` (no GPU)
├── database/                   # geometry_db_v0.2.0.json + .sha256 (1,904 records)
├── analysis/                   # manuscript scripts, organized by result
├── data/                       # committed inputs (DMS, sequences)
├── figures/                    # final manuscript figures
├── config/                     # conda env specs
└── docs/                       # paper PDF + figure legends
```

**What is committed vs. regenerated:**

| Artifact | Status | Size / source |
|----------|--------|---------------|
| `data/dms/*.tsv` | committed | 8 proteins, 84,361 mutations (public DMS) |
| `data/sequences/*.fasta` | committed | wild-type sequences |
| `database/geometry_db_v0.2.0.json` | committed | 1,904 pre-computed geometry records |
| BioEmu conformational NPZ | not committed | ~4.4 GB; regenerate with BioEmu (GPU) |
| full-atom / side-chain XTC/PDB | not committed | regenerate with hpacker + OpenMM |
| learned embeddings (ProstT5/ProtT5/ESM-C) | not committed | regenerate from model checkpoints |

**Never fabricate the non-committed data.** If a script requires missing ensembles, generate them first (Section 6), or report the blocker explicitly.

---

## 3. Coordinate and unit conventions

- Cα Cartesian ensembles: shape `(n_samples, n_residues, 3)`, units **nm** (BioEmu NPZ key `pos`; adjacent Cα about 0.38 nm). Do not mix with Angstrom.
- Covariance: `(3N, 3N)` after Kabsch alignment; Ledoit-Wolf optimal shrinkage.
- Per-residue metric: `(3, 3)` block `g_i = (C_i + eps I)^-1`, `eps = 0.01 * tr(C_i)/3`.
- `C_geo(P|S) = dz^T G_S dz` where `G_S = C_S,LW^-1`.
- Stochastic entry points accept an explicit `rng`/`seed`; `set_global_seed(42)` pins `random`, `numpy`, and (when present) `torch`.

---

## 4. Environment setup

### 4.1 CPU environment (always required)

```bash
conda env create -f config/environment_cpu.yml
conda activate grp-cpu
pip install -e ".[test]"
python -m pytest tests/ -q
python -c "import protgenesis_ensemble as pe; print(pe.__version__)"   # 0.4.0
```

This environment alone is sufficient for the package, its tests, the geometry database, and the DMS association analysis (Result 2) using the committed `data/dms/`.

### 4.2 BioEmu environment (GPU; conformational sampling)

```bash
conda env create -f config/environment_bioemu.yml
conda activate bioemu
python -c "import bioemu, torch; assert torch.cuda.is_available()"
```

BioEmu model weights are downloaded on first use (or set `HF_HUB_OFFLINE=1` if cached). Sampling command (project convention):

```bash
python -m bioemu.sample <sequence>.a3m 250 <out_dir> --batch_size_100 5
```

This writes `batch_*.npz` (one sample per file, key `pos`). For proteins with no useful MSA, use the minimum-A3M bypass: `>seq_id` + the sequence + `>pseudo_hit_1` + the sequence (see `analysis/sampling/phase_o4_prepare.py`).

### 4.3 Protein language-model environment (embeddings; GPU optional)

```bash
conda env create -f config/environment_plm.yml
conda activate grp-plm
```

Used for ProstT5 / ProtT5 / ESM-C embeddings (learned-representation and cross-representation analyses). Model versions, layers, tokenization and pooling are pinned in `analysis/r5_cross_representation/`.

---

## 5. Data model

### 5.1 DMS (deep mutational scanning)

`data/dms/` holds one TSV per protein with mutation and experimental fitness columns. The eight proteins are BLAT, GFP, HRAS, HSP90, P53, PTEN, SPIKE_RBD, UBE4B. Raw sources are ProteinGym `DMS_ProteinGym_substitutions` files (public); the committed TSVs are the quality-controlled inputs for the manuscript `rho(C_geo, DMS)` estimates.

### 5.2 Geometry database

`database/geometry_db_v0.2.0.json` is the aggregated ensemble-geometry result (1,904 records). Schema is enforced by `protgenesis_ensemble.validate_geometry_db`. Every record has a composite `uid` (`source:seq_id`), `n_residues`, `n_samples`, and a `features` dict of finite scalars.

```python
from protgenesis_ensemble import validate_geometry_db
report = validate_geometry_db("database/geometry_db_v0.2.0.json",
                              "database/geometry_db_v0.2.0.sha256")
assert report["ok"]
```

---

## 6. The five results and their scripts

The manuscript has five results. Each maps to scripts in `analysis/` and inputs in `data/`/`database/` (or regenerated ensembles).

### Result 1 — local effective geometry

Pipeline: BioEmu NPZ -> Kabsch align -> Ledoit-Wolf covariance -> spectral summaries -> chain-length scaling.

- `analysis/common/phase_l1_kabsch_metric.py` — Kabsch + LW intrinsic metric (reused everywhere).
- `analysis/r1_ensemble_geometry/phase1_conformation_geometry.py` — per-system geometry.
- `analysis/r1_ensemble_geometry/phase9_comprehensive_geometry.py` — system-wide atlas.
- `analysis/r1_ensemble_geometry/plan2_fss_scaling_analysis.py` — finite-range scaling fits.
- `analysis/r1_ensemble_geometry/phase_m2_longchain_efficient.py` — randomized low-rank eigh for long chains.
- `analysis/r1_ensemble_geometry/phase_l4_convergence_analysis.py` — sampling-convergence checks.

Committed result: `database/geometry_db_v0.2.0.json` (participation ratio, spectral decay, anisotropy across the 1,323 systems).

### Result 2 — covariance-normalized perturbation cost

Pipeline: WT ensemble metric `G_S` -> displacement `dz` -> `C_geo` -> association with DMS.

- `analysis/r2_perturbation_cost/phase_o1_cgeo_v3_deterministic.py` — deterministic `C_geo` v3.
- `analysis/r2_perturbation_cost/phase_o2_baseline_competition.py` — sequence-baseline comparison.
- `analysis/r2_perturbation_cost/phase6_cgeo_dms_calibration.py` — `C_geo` vs DMS across proteins.
- `analysis/r2_perturbation_cost/phase_ensemble_b1_fullatom_cgeo.py` — Cα vs full-atom correspondence.
- `analysis/r2_perturbation_cost/phase_o4_analyze.py` — real-mutant ensemble validation.

Committed inputs: `data/dms/*.tsv`. Committed result: mean `rho = -0.1475` across 8 proteins / 84,361 mutations; cross-resolution `r = 0.219` (n = 376).

### Result 3 — effective geometric field

Pipeline: non-redundant source descriptors -> grouped held-out prediction -> response-coupling matrix.

- `analysis/r3_source_field/phase_ensemble_b2_coupling_matrix.py` — response operator / coupling matrix.
- `analysis/r3_source_field/phase_ensemble_b3_nonlinear_field.py` — nonlinear field checks.
- `analysis/r3_source_field/law2_pattern_descriptors.py` and `law2_pattern_k_refit.py` — descriptors and refit.
- `analysis/r3_source_field/phase_k5_nested_cv_field.py` — grouped nested cross-validation.
- `analysis/r3_source_field/phase_m6_ensemble_gnn.py` — graph-model non-recovery control.

Committed result: 73.6% of responses `R2 > 0.1`; bootstrap ARI 0.708; leave-one-out ARI 1.0; graph model `R2 = -0.1553`.

### Result 4 — low-transport paths

Pipeline: ordered state path -> common-space map (joint PCA) -> Gaussian Bures-Wasserstein transport -> matched null comparison.

- `analysis/r4_path_transport/phase_ensemble_w2_joint_recompute.py` — primary joint-PCA transport.
- `analysis/r4_path_transport/phase_ensemble_npz_direct_w2.py` — direct-NPZ analysis.
- `analysis/r4_path_transport/phase_ensemble_b4_heteropath_action.py` — heteropolymer reproduction.
- `analysis/r4_path_transport/phase_d3_path_analysis.py` — natural-path analysis.
- `analysis/r4_path_transport/phase_f4_cross_system_w2.py` — cross-system transport.

Committed result: primary 11.75 vs 22.90 (d = -1.54, P = 8.4e-68); heteropolymer 3.07 vs 3.46; NPZ 7.62 vs 9.29.

### Result 5 — unified organization (synthesis + cross-representation)

- `analysis/r5_cross_representation/phase_l2_intrinsic_reanalysis.py` — intrinsic-frame reanalysis.
- `analysis/r5_cross_representation/phase_l3_cross_sampler.py` — cross-sampler validation.
- `analysis/r5_cross_representation/phase8_real_embeddings.py` and `phase_x_embedding_laws.py` — learned-representation analyses.
- `analysis/r5_cross_representation/phase_ensemble_f5_cross_validation.py` — grouped cross-validation.

---

## 7. Reproduction recipes

### 7.1 Fast path (CPU only, no BioEmu)

Reproduces the package, geometry database integrity, and DMS association:

```bash
conda activate grp-cpu
pip install -e ".[test]"
python -m pytest tests/ -q
python -c "from protgenesis_ensemble import validate_geometry_db; \
print(validate_geometry_db('database/geometry_db_v0.2.0.json','database/geometry_db_v0.2.0.sha256')['ok'])"
```

Then re-derive `rho(C_geo, DMS)` on the committed DMS tables with `analysis/r2_perturbation_cost/phase6_cgeo_dms_calibration.py`.

> **Path note.** The committed analysis scripts are the original manuscript entry points and expect the original `field_theory/` data layout (absolute or `Path(__file__)`-relative paths, and BioEmu NPZ under a `field_theory/data/...`-style tree). Before running them in a fresh clone, either (a) reproduce the original layout, or (b) edit each script's top-of-file `PROJECT_ROOT` / input paths to point at this repository's `data/` and regenerated ensembles. The package in `src/protgenesis_ensemble/` is path-independent and runs as-is.

### 7.2 Full conformational atlas (BioEmu, GPU)

1. Activate `bioemu`, confirm `torch.cuda.is_available()`.
2. Generate FASTA/A3M for each sequence-defined system (`analysis/sampling/`).
3. Sample each ensemble (`python -m bioemu.sample ...`), then compute geometry with `analysis/r1_ensemble_geometry/phase1_conformation_geometry.py`.
4. Aggregate into `database/geometry_db_*.json` and validate.

### 7.3 Perturbation-cost analyses

1. Build WT metrics from the BioEmu WT ensembles (`build_residue_metrics`).
2. Compute mutation costs and correlate with `data/dms/*.tsv` (Result 2).
3. For real-mutant validation, prepare the 36-variant manifest (`analysis/sampling/phase_o4_prepare.py`), sample 250 conformations per variant, then run `analysis/r2_perturbation_cost/phase_o4_analyze.py`.

### 7.4 Path transport

1. Map ordered path ensembles into a joint-PCA comparison space.
2. Compute `W2G` and matched null paths (Result 4).
3. Report effect size, uncertainty, and Holm-corrected `P` values across the prespecified null constructions.

---

## 8. Statistical hygiene (mandatory)

- Independent proteins/systems are the primary inference unit; conformational frames and transitions within one protein are never treated as independent biological replicates.
- Bootstrap resampling follows the data hierarchy (cluster-resample proteins first, then frames within).
- Outcome-dependent maps (joint PCA, response standardization, model selection) are fit inside the training partition and applied unchanged to held-out systems.
- Use grouped nested cross-validation for any transfer claim.
- Report negative/null results as-is; do not select analyses or thresholds after seeing outcomes.
- A claim of "low transport" requires direction + non-negligible effect size + uncertainty compatible with the prediction + stability across prespecified null paths. A complete action principle is not claimed.
- Transformation diagnostics (directional consistency, coefficient-of-variation) are not independent biological validation.
- All stochastic entry points must accept an explicit seed; the CI pipeline runs deterministically.
- Do not re-label `C_geo` as a physical energy, and do not re-label representation-dependent scaling as universal exponents.

---

## 9. Known caveats to preserve in any output

- `eff_rank_95` is finite-sample biased at 250-500 samples; prefer participation entropy for dimensionality claims (`dimension_guidance`).
- BioEmu minimal-MSA sampling can produce side-chain clashes; Cα backbone distances are healthy (about 0.38 nm) and are the quantities used. Do not over-interpret `_unphysical.xtc` conversion warnings.
- `C_geo` is an equivalent restatement of a stiffness-weighted mutation magnitude, not an independent predictor beyond sequence baselines (Result 2 baseline competition).
- The manuscript is a preliminary, representation-aware framework; it does not establish causal or coordinate-free physical laws.

---

## 10. Acceptance checklist (verify before declaring success)

1. `pytest tests/ -q` passes in `grp-cpu`.
2. `protgenesis_ensemble.__version__ == "0.4.0"` and all `__all__` symbols import.
3. `validate_geometry_db(...)` returns `ok: true` for `database/geometry_db_v0.2.0.json`.
4. Any number you report is traceable to a committed script plus committed input, or to a documented regeneration step that you actually ran.
5. You have not committed BioEmu NPZ, full-atom XTC, embeddings, or log files (see `.gitignore`).
6. Terminology matches Section 1 (no force/energy/action-principle/universal-law claims).
