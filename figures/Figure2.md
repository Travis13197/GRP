# Figure 2 — Local covariance normalization makes perturbations comparable across protein backgrounds

**Result 2 of the manuscript.** This figure defines the covariance-normalized perturbation cost **C_geo(P|S) = δzᵀ G_S δz** (with the Ledoit-Wolf-regularized metric G_S = Ĉ_S,LW⁻¹) and evaluates its association with cross-resolution scalar costs and experimental mutational fitness.

## Panels

| Panel | Content |
|-------|---------|
| **a** | Real PolyG30 fluctuation ellipse with the G30→G31 perturbation decomposed into soft (PC1) and stiff (PC2) components; dashed contours are the anisotropic C_geo; the inset shows the whitened space where the ellipse becomes isotropic. |
| **b** | Directional consistency of 414 matched adjacent-chain-length perturbations before/after normalization (d = 0.29, p = 9.6×10⁻⁶, win rate 58.7%). |
| **c** | Background-dependent coefficient of variation of perturbation cost (0.417 → 0.224; t = −29.96, Cohen's d = 2.08, win rate 92.0%). |
| **d(i)** | Cross-representation correspondence between Cα- and full-atom-derived costs (Spearman r = 0.219, p = 1.8×10⁻⁵, n = 376). |
| **d(ii)** | Per-protein association between C_geo and DMS fitness across eight proteins / 84,361 mutations (mean ρ = −0.1475; 8/8 negative). |
| **e** | Raw squared displacement is a weak proxy of geometric cost across 56 G→X perturbations (ρ = 0.41, p = 1.6×10⁻³). |

## Key statistics

- C_geo–DMS: mean Spearman **ρ = −0.1475** across 8 proteins / 84,361 mutations.
- Cross-resolution correspondence: **r = 0.219**, p = 1.8×10⁻⁵, n = 376.
- Transformation diagnostics (panels b, c) are retained as diagnostics, **not** independent biological validation (whitening can alter these statistics mechanically).

## Reproduction

```bash
python figures/scripts/make_figures.py main
```

The `fig2()` function lives in `figures/scripts/generate_final_figures_all_v7.py`.

## Input data (committed under `figures/data/`)

| Input | Used for |
|-------|----------|
| `ensembles/PolyX_PolyG_30/`, `..._31/` | real G30→G31 perturbation (panel a) |
| `tables/law1_direct_test.csv` | 414 perturbations (panels b, c) |
| `tables/phase_ensemble_b7_transformation_comparison_v2.csv` | transformation diagnostics |
| `tables/phase_ensemble_b1_ca_vs_fullatom_comparison.csv` | Cα vs full-atom (panel d(i)) |
| `phase_l1_cgeo_kabsch_all_proteins.csv` | C_geo–DMS scatter (panel d(ii)) |
| `tables/phase_x_law1_perturbations.csv` | anisotropy / G→X (panel e) |

## Interpretation

C_geo is a **covariance-normalized squared displacement** (a statistical atypicality measure), **not** a molecular energy, free energy, force, or complete fitness model. The total geometric cost is not treated as a complete fitness predictor.
