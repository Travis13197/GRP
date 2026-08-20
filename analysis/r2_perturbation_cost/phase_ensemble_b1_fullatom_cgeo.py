#!/usr/bin/env python3
"""
B1: 全原子 C_geo 批量计算
===========================
基于已有全原子 NPZ 数据，计算全原子空间中的几何扰动代价 C_geo。
与 Cα 空间的 C_geo 对比，验证第一定律 (几何扰动定律) 在全原子层面。

输入:
  - full_atom_geometry_results.csv (全原子几何特征)
  - full_atom_per_residue.csv (每残基侧链涨落)
  - fullatom_enhanced_geometry.csv (增强几何量)
  - BioEmu NPZ 数据 (全原子坐标)

输出:
  - phase_ensemble_b1_fullatom_cgeo.csv
  - phase_ensemble_b1_ca_vs_fullatom_comparison.csv
"""

import sys, os, json, warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
FIELD_THEORY = Path(__file__).parent.parent
SCRIPTS_DIR = FIELD_THEORY / "scripts"
TABLES_DIR = FIELD_THEORY / "tables"
DATA_DIR = FIELD_THEORY / "data"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
# [v3 2026-07-28] FA side: unified file = old-half 517 (canonical pipeline re-run,
# code-level sc_pct clip [0,100]) + new-half 516 (L8-fixed), dedup GGGGS_50 -> 1032 rows.
# Root cause of v2 corruption: the 2026-06-27 CSV was built from a pre-clip FA results
# file (250 rows sc_pct>100%, PolyS_49/50 spectral_decay 11.27/6.16); that file was later
# overwritten by a filtered re-run (line 986 of full_atom_analysis.py overwrites the CSV).
FA_GEOM = PROJECT_ROOT / "test_workflow/polyx_ensemble/analysis/full_atom/full_atom_geometry_results_unified.csv"
FA_PER_RES = PROJECT_ROOT / "test_workflow/polyx_ensemble/analysis/full_atom/full_atom_per_residue.csv"
FA_ENHANCED = DATA_DIR / "dms/phase9_fullatom/fullatom_enhanced_geometry.csv"
# [v3] CA side: canonical PolyX (354: E,G,K,L,S + linkers) + L1 hydrophobic (188: A,F,I,V)
# identical schema & 250-sample pipeline -> 542 rows covering all 8 perturbation AAs.
CA_GEOM = PROJECT_ROOT / "test_workflow/polyx_ensemble/analysis/ensemble_geometry_results.csv"
CA_GEOM_L1 = PROJECT_ROOT / "test_workflow/polyx_ensemble/analysis/l1_hydrophobic/l1_geometry_results.csv"
POLYX_OUTPUT = PROJECT_ROOT / "test_workflow/polyx_ensemble/output"

print("=" * 70)
print("B1: 全原子 C_geo 批量计算")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ============================================================
# Step 1: Load data
# ============================================================
print("\n[1/5] 加载数据...")

fa_geom = pd.read_csv(FA_GEOM)
fa_enhanced = pd.read_csv(FA_ENHANCED)
ca_geom = pd.read_csv(CA_GEOM)
# [v3] merge L1 hydrophobic Cα geometry (identical schema, disjoint seq_ids)
if CA_GEOM_L1.exists():
    ca_l1 = pd.read_csv(CA_GEOM_L1)
    shared = [c for c in ca_geom.columns if c in ca_l1.columns]
    ca_geom = pd.concat([ca_geom[shared], ca_l1[shared]], ignore_index=True)
    ca_geom = ca_geom.drop_duplicates(subset=['seq_id'], keep='first')
    print(f"  [v3] Cα merged with L1: +{len(ca_l1)} rows")

print(f"  全原子几何: {len(fa_geom)} 序列")
print(f"  全原子增强: {len(fa_enhanced)} 序列")
print(f"  Cα 几何: {len(ca_geom)} 序列")

# Extract AA types - use existing columns if available, otherwise parse
def extract_aa(seq_id):
    """Extract amino acid type from seq_id like PolyX_PolyA_10"""
    parts = str(seq_id).split('_')
    if len(parts) >= 2 and parts[0] == 'PolyX':
        for p in parts:
            if p.startswith('Poly') and len(p) == 5 and p[4] != 'X':
                return p[4]
    if len(parts) >= 2:
        for p in parts:
            if p.startswith('Poly') and len(p) == 5 and p[4] != 'X':
                return p[4]
    return 'unknown'

def extract_n(seq_id):
    parts = str(seq_id).split('_')
    try:
        return int(parts[-1])
    except:
        return 0

# Use existing aa_type if available, otherwise extract
if 'aa_type' in fa_geom.columns:
    fa_geom['aa_type_parsed'] = fa_geom['aa_type'].fillna(fa_geom['seq_id'].apply(extract_aa))
else:
    fa_geom['aa_type_parsed'] = fa_geom['seq_id'].apply(extract_aa)
fa_geom['aa_type'] = fa_geom['aa_type_parsed']

if 'n' in fa_geom.columns:
    fa_geom['n_parsed'] = fa_geom['n'].fillna(fa_geom['seq_id'].apply(extract_n)).astype(int)
else:
    fa_geom['n_parsed'] = fa_geom['seq_id'].apply(extract_n)
fa_geom['n'] = fa_geom['n_parsed']

# For Cα data, extract from seq_id
if 'aa_type' in ca_geom.columns:
    ca_geom['aa_type_parsed'] = ca_geom['aa_type'].fillna(ca_geom['seq_id'].apply(extract_aa))
else:
    ca_geom['aa_type_parsed'] = ca_geom['seq_id'].apply(extract_aa)
ca_geom['aa_type'] = ca_geom['aa_type_parsed']

if 'n' in ca_geom.columns:
    ca_geom['n_parsed'] = ca_geom['n'].fillna(ca_geom['seq_id'].apply(extract_n)).astype(int)
else:
    ca_geom['n_parsed'] = ca_geom['seq_id'].apply(extract_n)
ca_geom['n'] = ca_geom['n_parsed']

# ============================================================
# Step 2: Compute full-atom C_geo via proxy method
# ============================================================
# Since full-atom NPZ data is large, use a proxy: C_geo ≈ (ΔPR)^2/PR_base + (Δspectral_decay)^2
# This is based on the theory that C_geo measures geometric perturbation cost
# in the local tangent space.

print("\n[2/5] 计算全原子 C_geo (代理方法)...")

# PolyG as baseline for each n
g_baseline_fa = fa_geom[fa_geom['aa_type'] == 'G'].copy()
g_baseline_ca = ca_geom[ca_geom['aa_type'] == 'G'].copy()

# Compute C_geo: perturbation cost from PolyG to PolyX
fa_cgeo_records = []
for n in range(4, 51):
    g_row_fa = g_baseline_fa[g_baseline_fa['n'] == n]
    g_row_ca = g_baseline_ca[g_baseline_ca['n'] == n]
    if g_row_fa.empty:
        continue
    
    g_fa = g_row_fa.iloc[0]
    g_ca = g_row_ca.iloc[0] if not g_row_ca.empty else None
    
    for aa in ['A', 'E', 'F', 'I', 'K', 'L', 'S', 'V']:
        x_rows_fa = fa_geom[(fa_geom['aa_type'] == aa) & (fa_geom['n'] == n)]
        if x_rows_fa.empty:
            continue
        x_fa = x_rows_fa.iloc[0]
        
        # C_geo proxy: weighted geometric difference
        # ΔPR normalized by baseline PR
        # Δspectral_decay normalized
        delta_pr = abs(x_fa['PR'] - g_fa['PR']) / max(g_fa['PR'], 0.1)
        delta_sd = abs(x_fa['spectral_decay'] - g_fa['spectral_decay']) / max(g_fa['spectral_decay'], 0.1)
        delta_ac = abs(x_fa['A_C'] - g_fa['A_C'])
        # [v3 fix 2026-07-28] abs() on baseline: FA entropy is NEGATIVE (log-det over
        # rank-deficient covariance, N=250 frames << 3*n_atoms dims, noise eigenvalues
        # dominate). max(g_entropy, 0.01) hit the 0.01 floor for every n -> delta_entropy
        # ~1e5 -> C_geo ~1e10 (the true root cause of the "8e9" v2 corruption).
        delta_entropy = abs(x_fa['entropy'] - g_fa['entropy']) / max(abs(g_fa['entropy']), 0.01)
        delta_var = abs(x_fa['total_variance'] - g_fa['total_variance']) / max(g_fa['total_variance'], 0.01)
        
        # Sidechain-specific contribution
        sc_contrib = x_fa.get('sidechain_contribution_pct', 0) / 100.0
        
        # Full-atom C_geo = weighted sum of geometric differences
        c_geo_fa = (
            0.3 * delta_pr**2 +
            0.25 * delta_sd**2 +
            0.15 * delta_ac**2 +
            0.15 * delta_entropy**2 +
            0.1 * delta_var**2 +
            0.05 * sc_contrib
        )
        
        record = {
            'seq_id': x_fa['seq_id'],
            'aa_type': aa,
            'n': n,
            'C_geo_fullatom': c_geo_fa,
            'PR_fullatom': x_fa['PR'],
            'PR_baseline_G': g_fa['PR'],
            'spectral_decay_fullatom': x_fa['spectral_decay'],
            'spectral_decay_baseline_G': g_fa['spectral_decay'],
            'delta_PR': delta_pr,
            'delta_spectral_decay': delta_sd,
            'sidechain_contribution': sc_contrib,
        }
        fa_cgeo_records.append(record)

df_fa_cgeo = pd.DataFrame(fa_cgeo_records)
print(f"  全原子 C_geo 计算: {len(df_fa_cgeo)} 扰动 (8 AA × 47 n)")

# ============================================================
# Step 3: Compute Cα C_geo for comparison
# ============================================================
print("\n[3/5] 计算 Cα C_geo (对比基准)...")

ca_cgeo_records = []
for n in range(4, 51):
    g_row = g_baseline_ca[g_baseline_ca['n'] == n]
    if g_row.empty:
        continue
    g = g_row.iloc[0]
    
    for aa in ['A', 'E', 'F', 'I', 'K', 'L', 'S', 'V']:
        x_rows = ca_geom[(ca_geom['aa_type'] == aa) & (ca_geom['n'] == n)]
        if x_rows.empty:
            continue
        x = x_rows.iloc[0]
        
        delta_pr = abs(x['PR'] - g['PR']) / max(g['PR'], 0.1)
        delta_sd = abs(x['spectral_decay'] - g['spectral_decay']) / max(g['spectral_decay'], 0.1)
        delta_ac = abs(x['A_C'] - g['A_C'])
        delta_entropy = abs(x['entropy'] - g['entropy']) / max(abs(g['entropy']), 0.01)  # [v3] abs guard, same rationale as FA side
        delta_var = abs(x['total_variance'] - g['total_variance']) / max(g['total_variance'], 0.01)
        
        c_geo_ca = (
            0.35 * delta_pr**2 +
            0.30 * delta_sd**2 +
            0.15 * delta_ac**2 +
            0.15 * delta_entropy**2 +
            0.05 * delta_var**2
        )
        
        ca_cgeo_records.append({
            'seq_id': x['seq_id'],
            'aa_type': aa,
            'n': n,
            'C_geo_Calpha': c_geo_ca,
            'PR_Calpha': x['PR'],
            'spectral_decay_Calpha': x['spectral_decay'],
        })

df_ca_cgeo = pd.DataFrame(ca_cgeo_records)
print(f"  Cα C_geo 计算: {len(df_ca_cgeo)} 扰动")

# ============================================================
# Step 4: Merge and compare
# ============================================================
print("\n[4/5] 合并 Cα vs 全原子 C_geo 对比...")

df_compare = df_fa_cgeo.merge(df_ca_cgeo, on=['aa_type', 'n'], suffixes=('', '_ca'))
df_compare['C_geo_ratio'] = df_compare['C_geo_fullatom'] / (df_compare['C_geo_Calpha'] + 1e-8)
df_compare['C_geo_diff'] = df_compare['C_geo_fullatom'] - df_compare['C_geo_Calpha']

# ============================================================
# Step 5: Statistical analysis
# ============================================================
print("\n[5/5] 统计分析...")

# Per-AA comparison
aa_stats = []
for aa in df_compare['aa_type'].unique():
    aa_df = df_compare[df_compare['aa_type'] == aa]
    if len(aa_df) < 3:
        continue
    
    r, p = spearmanr(aa_df['C_geo_Calpha'], aa_df['C_geo_fullatom'])
    r2, p2 = pearsonr(aa_df['C_geo_Calpha'], aa_df['C_geo_fullatom'])
    
    aa_stats.append({
        'aa_type': aa,
        'n_points': len(aa_df),
        'mean_C_geo_Calpha': aa_df['C_geo_Calpha'].mean(),
        'mean_C_geo_fullatom': aa_df['C_geo_fullatom'].mean(),
        'mean_C_geo_ratio': aa_df['C_geo_ratio'].mean(),
        'spearman_r': r,
        'spearman_p': p,
        'pearson_r': r2,
        'pearson_p': p2,
        'fullatom_more_sensitive': aa_df['C_geo_fullatom'].mean() > aa_df['C_geo_Calpha'].mean(),
    })

df_aa_stats = pd.DataFrame(aa_stats)

# Overall comparison
r_overall, p_overall = spearmanr(df_compare['C_geo_Calpha'], df_compare['C_geo_fullatom'])
print(f"\n  Cα vs 全原子 C_geo Spearman r = {r_overall:.4f} (p = {p_overall:.4e})")
print(f"  全原子 C_geo 均值: {df_compare['C_geo_fullatom'].mean():.4f}")
print(f"  Cα C_geo 均值: {df_compare['C_geo_Calpha'].mean():.4f}")
print(f"  比值均值: {df_compare['C_geo_ratio'].mean():.2f}")

more_sensitive = df_aa_stats['fullatom_more_sensitive'].sum()
print(f"  全原子更敏感: {more_sensitive}/{len(df_aa_stats)} AA")

# Save
df_fa_cgeo.to_csv(TABLES_DIR / 'phase_ensemble_b1_fullatom_cgeo.csv', index=False)
df_compare.to_csv(TABLES_DIR / 'phase_ensemble_b1_ca_vs_fullatom_comparison.csv', index=False)
df_aa_stats.to_csv(TABLES_DIR / 'phase_ensemble_b1_aa_sensitivity.csv', index=False)

print(f"\n{'='*70}")
print("B1 完成!")
print(f"输出: {TABLES_DIR / 'phase_ensemble_b1_fullatom_cgeo.csv'}")
print(f"输出: {TABLES_DIR / 'phase_ensemble_b1_ca_vs_fullatom_comparison.csv'}")
print(f"输出: {TABLES_DIR / 'phase_ensemble_b1_aa_sensitivity.csv'}")
print(f"{'='*70}")