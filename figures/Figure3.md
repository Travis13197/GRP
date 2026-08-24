# Figure 3 — Biological source terms predict an effective geometric field

**Result 3 of the manuscript.** This figure tests whether non-redundant biological source variables (chain length, composition contrasts, physicochemical and contextual descriptors) predict ensemble-geometry observables through a local response operator **K_r(t₀)**.

## Panels

| Panel | Content |
|-------|---------|
| **a** | Importance of the leading source variables for predicting 27 geometric observables, grouped into chain-length, composition×length and sequence families. |
| **b** | Grouped held-out prediction of the per-amino-acid × observable response; 73.6% of per-amino-acid responses reach R² > 0.1, strongest per-amino-acid mean R² = 0.71, strongest per-observable mean R² = 0.87. |
| **c** | The 36 × 36 response-coupling matrix K clustered into a two-block structure (k = 2); reproducible under bootstrap (mean ARI 0.708) and leave-one-amino-acid-out (ARI 1.00). |
| **d** | Validation of the coupling structure (bootstrap ARI, leave-one-out ARI, silhouette, clustering stability); the structure is not recovered by sequence- or graph-based descriptors. |

## Key statistics

- 73.6% of per-amino-acid responses reached R² > 0.1.
- Coupling-matrix bootstrap adjusted Rand index **0.708**; leave-one-amino-acid-out **ARI = 1.0** (18/18 groups).
- The final graph model achieved R² = −0.1553 (non-recovery within the tested model family, not absolute mathematical irreducibility).

## Reproduction

```bash
python figures/scripts/make_figures.py main
```

The `fig3()` function lives in `figures/scripts/generate_final_figures_all_v7.py`.

## Input data (committed under `figures/data/`)

| Input | Used for |
|-------|----------|
| `l2_intrinsic_geometry.csv` | geometry observables |
| `tables/phase_ensemble_b3_feature_importance_v2.csv` | source-variable importance (panel a) |
| `tables/final_fig3_per_aa_r2.csv` | per-AA × observable R² (panel b) |
| `tables/phase_ensemble_b2_coupling_matrix_v2.csv` | 36×36 coupling matrix K (panel c) |
| `tables/law2_validation_report.json`, `phase_r1_cluster_bootstrap.json` | reproducibility metrics (panel d) |

## Interpretation

These analyses support a **predictive, representation-dependent** relation between biological descriptors and selected geometric responses. They do **not** establish a coordinate-free or causal field law; all associations are predictive and do not imply causal control.
