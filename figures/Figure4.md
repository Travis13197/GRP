# Figure 4 — Structured protein-state paths show reduced mean Gaussian transport

**Result 4 of the manuscript.** This figure compares ordered (sequence-structured) protein-state paths with matched null paths using the mean adjacent-state Gaussian Bures–Wasserstein transport **W̅₂,G**.

## Panels

| Panel | Content |
|-------|---------|
| **a** | A real ordered path PolyG30→PolyG31→PolyG32→PolyG33 in the joint-PCA common space (500 conformations per state); arrows denote adjacent-state Bures W₂. |
| **b** | Primary comparison: mean adjacent-state transport is lower for structured paths (11.75 vs 22.90; Cohen's d = −1.54, p = 8.4×10⁻⁶⁸). |
| **c** | Independent replications: heteropolymer (3.07 vs 3.46; d = −0.20, p = 2.3×10⁻⁶) and direct NPZ (7.62 vs 9.29; p = 4.2×10⁻⁹). |
| **d** | The low-transport relation holds across representations (49%, 11%, 18% reductions) and remains significant under Holm correction. |
| **e** | Real adjacent-state transport along the ordered path, all below the matched-null mean. |

## Key statistics

- Primary: **11.75 vs 22.90** (Cohen's d = −1.54, p = 8.4×10⁻⁶⁸).
- Heteropolymer: 3.07 vs 3.46 (d = −0.20, p = 2.3×10⁻⁶).
- Direct NPZ: 7.62 vs 9.29 (p = 4.2×10⁻⁹).

## Reproduction

```bash
python figures/scripts/make_figures.py main
```

The `fig4()` function lives in `figures/scripts/generate_final_figures_all_v7.py`.

## Input data (committed under `figures/data/`)

| Input | Used for |
|-------|----------|
| `ensembles/PolyX_PolyG_30..33/` | real ordered path (panel a, e) |
| `tables/phase_ensemble_w2_joint_paths.csv` | primary joint-PCA transport (panel b) |
| `tables/phase_ensemble_b4_path_action_v2.csv`, `phase_ensemble_b4_summary_v2.json` | heteropolymer replication (panel c) |
| `tables/phase_ensemble_npz_direct_w2.csv` | direct NPZ analysis (panel c) |

## Interpretation

This establishes a **low-transport path relation**, not a complete minimum-action principle. A broader dimensionless action functional is retained as a theoretical extension and is not independently validated under the same matched-null design.
