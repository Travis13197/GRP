# Figure 1 — Protein ensembles define local effective geometry

**Result 1 of the manuscript.** This figure establishes the central object of the framework: a conformational **ensemble** summarized by its rigid-body-aligned **intrinsic covariance**, and the reproducible chain-length organization of that covariance.

## Panels

| Panel | Content |
|-------|---------|
| **a** | Principal-component atlas of 1,373 sequence-defined conformational ensembles, built from log-scaled, z-standardized intrinsic-covariance, structural and information-theoretic features and coloured by system class. Dashed ellipses mark 2.5σ category clouds; grey arrows are the loadings of the three directionally most informative observables (participation ratio, spectral decay, total variance). |
| **b** | Composition of the quality-controlled atlas (1,323 systems): PolyX (750), heteropolymer (239), hydrophobic-gradient L1 (188), linker (94), intrinsically disordered (44), DMS-associated (8). |
| **c** | Local statistical chart of the real PolyG30 ensemble (500 conformations, 30 Cα) after Kabsch alignment, centring and covariance decomposition. The top two aligned PCs carry 51% and 14% of variance; the inset shows the top-eight mode spectrum. |
| **d(i)** | Normalized eigenvalue spectra for G10, G30, G50 and K50 with the 95%-explained-variance ranks (G10 = 3, G30 = 11, G50 = 18, K50 = 13). |
| **d(ii)** | Intrinsic participation ratio versus chain length for the four residues with the strongest positive scaling (K, G, T, H); pooled log-log Pearson r = 0.78, p = 7.5×10⁻⁴³, n = 208. |
| **e** | Finite-range scaling of intrinsic spectral decay over 4 ≤ n ≤ 60 across nine representative residues; shared fixed-effects slope β = −0.66 (R² = 0.95, 20 amino acids). |

## Key statistics

- 1,323 sequence-defined systems in the quality-controlled atlas.
- PR chain-length scaling: pooled log-log Pearson **r = 0.78**, **p = 7.5×10⁻⁴³**, n = 208.
- Spectral-decay chain-length scaling: **β = −0.66**, **R² = 0.95**.
- 95%-variance ranks are sample-size- and representation-dependent descriptors, **not** intrinsic manifold or physical dimensions.

## Reproduction

```bash
python figures/scripts/make_figures.py main   # regenerates Figure 1-4
```

The `fig1()` function lives in `figures/scripts/generate_final_figures_all_v7.py`.

## Input data (committed under `figures/data/`)

| Input | Used for |
|-------|----------|
| `l2_intrinsic_geometry.csv` | the 1,373-system intrinsic-geometry atlas (panels a, d, e) |
| `tables/final_fig1_atlas.csv` | atlas category counts (panel b) |
| `tables/final_fig1_spectra_G_10/30/50.npz`, `..._K_50.npz` | normalized spectra (panel d(i)) |
| `ensembles/PolyX_PolyG_30/` | real PolyG30 ensemble (panel c) |

## Interpretation

Participation ratio, effective rank and spectral decay are **statistical summaries of the declared representation**, not physical, topological or thermodynamic dimensions. The reported coefficients describe finite-range associations within the analysed representation and sampling domain, not universal scaling exponents.
