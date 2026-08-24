# Figures — reproduction and documentation

This directory makes every manuscript figure reproducible from committed inputs.

## Layout

```
figures/
├── scripts/                          # generators + driver
│   ├── generate_final_figures_all_v7.py   # Figures 1-4 (main)
│   ├── generate_final_supplementary.py     # Figures S1-S9
│   ├── final_figure_style.py               # shared Nature-grade style
│   └── make_figures.py                     # driver + verification
├── data/                             # committed figure inputs (~20 MB)
│   ├── tables/                       # 26 result tables (CSV/JSON/NPZ)
│   ├── l2_intrinsic_geometry.csv     # 1,373-system intrinsic geometry atlas
│   ├── ensembles/                    # 4 real BioEmu Cα ensembles (PolyG30-33)
│   ├── phase_l1_cgeo_kabsch_all_proteins.csv
│   └── systemwide_enhanced_geometry_v2.csv
├── out/                              # generated outputs (main/, supplementary/)
├── Figure1.md ... Figure4.md          # per-figure documentation
└── Supplementary_Figures.md
```

## Quick start

```bash
pip install numpy scipy pandas scikit-learn matplotlib seaborn statsmodels umap-learn
python figures/scripts/make_figures.py          # main + supplementary
python figures/scripts/make_figures.py main     # Figures 1-4 only
python figures/scripts/make_figures.py supp     # Figures S1-S9 only
```

Outputs are written to `figures/out/main/` and `figures/out/supplementary/` in four formats each (SVG, JPG, PNG, PDF at 300 dpi).

## Per-figure documentation

- [Figure 1](Figure1.md) — local effective geometry
- [Figure 2](Figure2.md) — covariance-normalized perturbation cost
- [Figure 3](Figure3.md) — effective geometric field
- [Figure 4](Figure4.md) — low-transport paths
- [Supplementary figures](Supplementary_Figures.md) — S1–S9

The figure legends (panel-by-panel, with statistics) are in [`docs/figure_legends.md`](../docs/figure_legends.md).

## Provenance

The `figures/data/` tables are the canonical intermediate results produced by the analysis scripts in `analysis/`. The four BioEmu ensembles are the only raw conformations required for figure rendering; the full conformational atlas is regenerated separately (see `AI_CODER.md`).
