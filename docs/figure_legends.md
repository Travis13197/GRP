# Final Figure Legends (2026-08-18, rich version for trimming)

Written against the latest main figures (data-only, all panels square). Values are
recomputed from the source tables and cross-checked with the manuscript.

---

## Figure 1 | Protein ensembles define local effective geometry

**a,** Principal-component atlas of 1,373 sequence-defined conformational ensembles,
obtained from log-scaled, z-standardized intrinsic covariance, structural and
information-theoretic features and coloured by system class. Dashed ellipses mark 2.5σ
category clouds; grey arrows are the loadings of the three directionally most informative
observables (participation ratio, spectral decay, total variance).

**b,** Composition of the quality-controlled atlas (1,323 sequences): PolyX (750),
heteropolymer (239), hydrophobic-gradient L1 (188), linker (94), intrinsically disordered
(44) and DMS-associated (8); the inset shows category shares.

**c,** Local statistical chart of the real PolyG30 ensemble (500 conformations, 30 Cα)
after rigid-body alignment, centring and covariance decomposition. The top two aligned
PCs carry 51% and 14% of variance (λ1/λ2 = 3.6); the dashed ellipse is the 4σ covariance
contour, the arrow marks the leading mode and the inset shows the top-eight mode
spectrum.

**d,** **(i)** Normalized eigenvalue spectra for G10, G30, G50 and K50; filled markers and
dashed cumulative curves locate the mode at which 95% of variance is reached
(G10 = 3, G30 = 11, G50 = 18, K50 = 13). **(ii)** Intrinsic participation ratio versus
chain length for the four residues with the strongest positive scaling (K, G, T, H); thin
lines are per-residue power-law fits and the dashed line is the pooled fit
(log-log Pearson r = 0.78, p = 7.5×10⁻⁴³, n = 208). Participation ratio and effective
rank are statistical covariance summaries, not physical, topological or thermodynamic
dimensions.

**e,** Finite-range scaling of intrinsic spectral decay over 4 ≤ n ≤ 60 across nine
representative residues. Per-residue log-log fits (R² ≥ 0.90; slopes −0.56 to −0.78) are
summarized by a shared fixed-effects slope β = −0.66 (R² = 0.95, 20 amino acids),
establishing the principal reproducible geometric trend. Amino-acid fingerprints,
anisotropy, conformational extent, representation dependence and non-linear embeddings
are in Supplementary Information (Fig. S9).

---

## Figure 2 | Local covariance normalization makes perturbations comparable across protein backgrounds

**a,** Real PolyG30 fluctuation ellipse with the G30→G31 perturbation z = Δ⟨x⟩ decomposed
into soft (PC1) and stiff (PC2) components. Dashed orange contours are the anisotropic
geometric cost C_geo = zᵀ g_S z computed from the real covariance (σ_soft, σ_stiff and z
components annotated); the inset shows the same perturbation in whitened space
(z̃ = g_S^1/2 z, |z̃|² = C_geo), where the ellipse becomes isotropic.

**b,** Directional consistency of 414 matched adjacent-chain-length perturbations before
and after normalization (d = 0.29, p = 9.6×10⁻⁶; win rate 58.7%).

**c,** Background-dependent coefficient of variation of perturbation cost
(0.417 → 0.224; t = −29.96, Cohen's d = 2.08, win rate 92.0%), the dominant
transformation effect.

**d,** Independent validation: **(i)** cross-representation correspondence between
Cα- and full-atom-derived costs (Spearman r = 0.219, p = 1.8×10⁻⁵, n = 376);
**(ii)** per-protein association between C_geo and deep-mutational-scanning fitness
across eight proteins and 84,361 mutations (mean ρ = −0.1475; 8/8 negative).

**e,** Anisotropy underlying the normalization: raw squared displacement is a weak proxy
of geometric cost across 56 G→X perturbations (ρ = 0.41, p = 1.6×10⁻³; 2.2-fold cost
spread at matched chain length), with per-target power laws and the specific cost
C_geo/‖z‖² shown against chain length.

**Interpretation.** C_geo is a covariance-normalized squared displacement, not a
molecular energy, free energy or complete fitness model.

---

## Figure 3 | Biological source terms predict an effective geometric field

**a,** Importance of the leading source variables for predicting 27 geometric observables
(mean absolute importance across observables ± 1 SD; top-1 incidence annotated), grouped
into chain-length, composition×length and sequence families.

**b,** Grouped held-out prediction of the per-amino-acid × observable response; 73.6% of
per-amino-acid responses achieve R² > 0.1 (white dots), with the strongest
per-amino-acid mean R² = 0.71 (K, orange column) and the strongest per-observable mean
R² = 0.87 (effective rank); top and right margins show per-amino-acid fractions and
per-observable means.

**c,** The 36 × 36 response-coupling matrix K, clustered into a two-block structure
(k = 2); the structure is reproducible under bootstrap (mean ARI 0.708, median 0.696,
1,000 resamples) and leave-one-amino-acid-out (ARI 1.00, 18/18 groups), with
silhouette 0.950.

**d,** Validation of the coupling structure: the three reproducibility criteria (bootstrap
ARI, leave-one-out ARI, silhouette), clustering stability across k, and perfect
leave-one-AA-out recovery (18/18 groups = 1.00). Held-out prediction is retained across
all 20 amino acids (mean R² 0.43–0.71) yet the structure is irreducible to simple
sequence descriptors (exclusion chain, Supplementary Fig. S8).

**Interpretation.** All associations are predictive and do not imply causal control.

---

## Figure 4 | Structured protein-state paths show reduced Wasserstein transport

**a,** A real ordered path PolyG30→PolyG31→PolyG32→PolyG33 represented as four empirical
ensembles in the joint PCA common space (500 conformations per state); arrows show
adjacent-state Bures W2.

**b,** Primary comparison: mean adjacent-state transport is lower for structured paths
than for matched null paths (11.75 vs 22.90; Cohen's d = −1.54, p = 8.4×10⁻⁶⁸;
cluster-bootstrap median p = 1.4×10⁻⁶⁶; Δmean = −11.15).

**c,** Independent replications: **(i)** heteropolymer (3.07 vs 3.46; d = −0.20,
p = 2.3×10⁻⁶); **(ii)** direct NPZ (7.62 vs 9.29; p = 4.2×10⁻⁹).

**d,** The low-transport relation holds across all three representations
(structured < null; reductions of 49%, 11% and 18%; primary p = 8.4×10⁻⁶⁸), and all
comparisons remain significant under Holm correction.

**e,** Real adjacent-state Bures W2 along the ordered path (13.9, 9.0, 9.9; path mean
10.9), all below the matched-null mean (22.9 ± 1 SD, grey band); the inset shows the null
W2 distribution with the path mean marked.

**Interpretation.** The broader action-like functional
A[γ] = W̄2[γ] + reorg[γ] + feas[γ] is retained as a theoretical extension
(Supplementary Fig. S7), not a validated physical minimum-action principle.

---

## Supplementary figure one-liners

- **S1** Per-amino-acid intrinsic spectral-decay scaling (20 panels; all 20 negative).
- **S2** Per-protein C_geo–DMS scatter (8 proteins, 84,361 mutations).
- **S3** Amino-acid length-scaling fingerprint heatmap (columns clustered, chemistry strip).
- **S4** Hydrophobicity–charge conditional-response phase map (θ = 121°, CI 40°–154°).
- **S5** Entropy state relation: observed-vs-predicted, standardized coefficients, residuals.
- **S6** K-matrix transformation invariance, real-mutant low-stiffness directions, domain signal.
- **S7** Folded-niche TSI, natural transition paths, direct NPZ transport, scope of the claim.
- **S8** Five-level exclusion chain (V1–V5) for coupling irreducibility.
- **S9** UMAP and t-SNE embeddings of the atlas and the PolyG30 local chart.
