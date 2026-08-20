#!/usr/bin/env python3
"""
G1: 测地线距离 vs 欧氏距离 — 系综几何空间的曲率分析
========================================================
在第一定律框架下，验证局部度量张量 g_S 对构象涨落几何的过滤效应。

核心思想:
  在系综几何特征空间中，局部度量张量 g_S(y) = (C_S(y) + epsilon*I)^{-1}
  编码了空间各点的"拉伸"程度。测地线距离 d_G 是沿此弯曲空间的真实距离，
  而欧氏距离 d_E 是忽略曲率的直线距离。

  测地线曲率 κ_geo = d_G/d_E - 1 量化了空间的弯曲程度:
    κ_geo > 0: 空间在路径方向上被拉伸 (测地线长于欧氏距离)
    κ_geo ≈ 0: 空间近似平坦
    κ_geo < 0: 空间在路径方向上被压缩 (罕见)

方法:
  1. 全局度量张量: g_global = (Cov(X) + εI)^{-1}, 给出全局 Mahalanobis 距离
  2. 局部度量张量: 沿路径各点用 k-NN 局部协方差估计 g_local(y)
  3. 测地线距离: 沿直线路径数值积分 L = ∫ sqrt(dy^T g(y) dy) dt

输入:
  - systemwide_enhanced_geometry_v2.csv (764 序列, 42 几何特征)

输出:
  - phase_ensemble_g1_geodesic_comparison_v2.csv (逐对测地线/欧氏距离)
  - phase_ensemble_g1_summary_v2.json (汇总统计)
  - field_theory/figures/phase_ensemble_g1_*.svg/jpg (可视化)
"""

import sys, os, json, warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.linalg import inv, eigh
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')

# ---- Paths ----
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
FIELD_THEORY = Path(__file__).parent.parent
SCRIPTS_DIR = FIELD_THEORY / "scripts"
TABLES_DIR = FIELD_THEORY / "tables"
DATA_DIR = FIELD_THEORY / "data"
FIGURES_DIR = FIELD_THEORY / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SYSTEMWIDE_CSV = DATA_DIR / "dms/phase9_systemwide/systemwide_enhanced_geometry_v2.csv"

# ---- Configuration ----
POLYX_CATEGORIES = ['PolyX_original', 'L1_hydrophobic', 'linker']
N_INTEGRATION_PTS = 20       # Number of points along each geodesic path
K_NEIGHBORS_LOCAL = 30       # k-NN for local covariance estimate
EPSILON_METRIC = 1e-4        # Regularization for metric tensor inversion
MAX_PAIRS = 8000             # Maximum number of pairs to compute (subsampled)
RANDOM_SEED = 42

# Geometric features to use (exclude metadata columns)
EXCLUDE_COLS = {'seq_id', 'category', 'aa_type', 'n', 'n_samples', 'n_residues'}

print("=" * 70)
print("G1: 测地线距离 vs 欧氏距离 — 系综几何空间曲率分析")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ============================================================
# Step 1: Load and prepare data
# ============================================================
print("\n[1/6] 加载数据...")

df_all = pd.read_csv(SYSTEMWIDE_CSV)
df_polyx = df_all[df_all['category'].isin(POLYX_CATEGORIES)].copy()
print(f"  总序列: {len(df_all)}, PolyX: {len(df_polyx)}")

# Extract geometric feature columns
geo_cols = [c for c in df_polyx.columns if c not in EXCLUDE_COLS]
print(f"  几何特征数: {len(geo_cols)}")

# Handle missing values
df_polyx[geo_cols] = df_polyx[geo_cols].fillna(df_polyx[geo_cols].median())

# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(df_polyx[geo_cols].values)
n_seqs, n_feats = X.shape
print(f"  特征矩阵: {n_seqs} × {n_feats}")

seq_ids = df_polyx['seq_id'].values
aa_types = df_polyx['aa_type'].values
categories = df_polyx['category'].values
n_values = df_polyx['n'].values

# ============================================================
# Step 2: Compute global metric tensor
# ============================================================
print("\n[2/6] 计算全局度量张量 g_global...")

# Remove zero-variance columns
X_std = X.std(axis=0)
nonzero_var_mask = X_std > 1e-10
if not np.all(nonzero_var_mask):
    removed_cols = [geo_cols[i] for i in range(n_feats) if not nonzero_var_mask[i]]
    print(f"  移除 {len(removed_cols)} 零方差列: {removed_cols}")
    X = X[:, nonzero_var_mask]
    geo_cols = [c for i, c in enumerate(geo_cols) if nonzero_var_mask[i]]
    n_feats = len(geo_cols)

# Check for NaN/Inf in X
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Global covariance
C_global = np.cov(X.T)
# Check for NaN in covariance
if np.any(np.isnan(C_global)) or np.any(np.isinf(C_global)):
    C_global = np.nan_to_num(C_global, nan=0.0, posinf=0.0, neginf=0.0)
# Regularize
C_reg = C_global + EPSILON_METRIC * np.eye(n_feats)
g_global = inv(C_reg)

# Verify positive definiteness
eigvals = eigh(g_global, eigvals_only=True)
print(f"  g_global 特征值范围: [{eigvals.min():.2e}, {eigvals.max():.2e}]")
print(f"  g_global 条件数: {eigvals.max() / (eigvals.min() + 1e-15):.2e}")

# Global Mahalanobis distance matrix (for reference)
def mahalanobis_distance(X, metric):
    """Compute pairwise Mahalanobis distances using metric tensor."""
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    # d² = (x_i - x_j)^T M (x_i - x_j) = sum_{a,b} diff_a * M_{a,b} * diff_b
    # Efficient: d² = sum((diff @ M) * diff, axis=2)
    weighted = diff @ metric
    d2 = np.sum(weighted * diff, axis=2)
    d2 = np.maximum(d2, 0)  # Numerical stability
    return np.sqrt(d2)

print("  计算全局 Mahalanobis 距离矩阵...")
D_global_mahalanobis = mahalanobis_distance(X, g_global)

# ============================================================
# Step 3: Generate pair set
# ============================================================
print("\n[3/6] 生成序列对...")

np.random.seed(RANDOM_SEED)

# Build pair list: all pairs within same aa_type, plus cross-aa_type pairs
pairs = []
# Same aa_type (生物相关)
unique_aa = sorted(set(aa_types))
for aa in unique_aa:
    idx = np.where(aa_types == aa)[0]
    if len(idx) < 2:
        continue
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            pairs.append((idx[i], idx[j], 'same_aa', aa))

# Cross aa_type (evolutionary transitions)
for aa1 in unique_aa:
    for aa2 in unique_aa:
        if aa1 >= aa2:
            continue
        idx1 = np.where(aa_types == aa1)[0]
        idx2 = np.where(aa_types == aa2)[0]
        if len(idx1) == 0 or len(idx2) == 0:
            continue
        # Sample up to 50 pairs per cross-aa combination
        n_sample = min(50, len(idx1) * len(idx2))
        for _ in range(n_sample):
            i = np.random.choice(idx1)
            j = np.random.choice(idx2)
            pairs.append((i, j, 'cross_aa', f'{aa1}_to_{aa2}'))

# Same category
unique_cat = sorted(set(categories))
for cat in unique_cat:
    idx = np.where(categories == cat)[0]
    if len(idx) < 2:
        continue
    for _ in range(min(200, len(idx) * (len(idx) - 1) // 2)):
        i, j = np.random.choice(idx, 2, replace=False)
        pairs.append((i, j, 'same_cat', cat))

# Subsample if too many
if len(pairs) > MAX_PAIRS:
    idx = np.random.choice(len(pairs), MAX_PAIRS, replace=False)
    pairs = [pairs[k] for k in idx]

print(f"  总对数: {len(pairs)}")
print(f"    同类AA: {sum(1 for p in pairs if p[2] == 'same_aa')}")
print(f"    跨类AA: {sum(1 for p in pairs if p[2] == 'cross_aa')}")
print(f"    同类类别: {sum(1 for p in pairs if p[2] == 'same_cat')}")

# ============================================================
# Step 4: Compute local metric tensor function
# ============================================================
print("\n[4/6] 准备局部度量张量估计器...")

# Fit k-NN model for fast local covariance computation
nn_model = NearestNeighbors(n_neighbors=K_NEIGHBORS_LOCAL + 1, metric='euclidean')
nn_model.fit(X)

def compute_local_metric(point, k=K_NEIGHBORS_LOCAL):
    """
    Compute local metric tensor at a given point using k-NN covariance.
    g_local(y) = (C_k(y) + epsilon*I)^{-1}
    """
    # Find k nearest neighbors (excluding the point itself if it's in the data)
    dist, idx = nn_model.kneighbors(point.reshape(1, -1), n_neighbors=k + 1)
    nn_X = X[idx[0][1:]]  # Exclude self

    # Local covariance
    C_local = np.cov(nn_X.T)
    C_reg = C_local + EPSILON_METRIC * np.eye(n_feats)
    try:
        g_local = inv(C_reg)
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse
        g_local = np.linalg.pinv(C_reg)
    return g_local

def compute_geodesic_length(x_a, x_b, n_pts=N_INTEGRATION_PTS):
    """
    Compute geodesic distance between points A and B by numerical integration
    along the straight line path in parameter space.

    L[gamma] = integral_0^1 sqrt(g_ij(gamma(t)) * d_gamma^i/dt * d_gamma^j/dt) dt

    where gamma(t) = x_a + t * (x_b - x_a)
    """
    direction = x_b - x_a
    dt = 1.0 / (n_pts - 1)

    total_length = 0.0
    for k in range(n_pts):
        t = k * dt
        point = x_a + t * direction

        # Local metric tensor at this point
        g_local = compute_local_metric(point)

        # Line element: ds² = dy^T g(y) dy
        ds2 = direction @ g_local @ direction
        ds = np.sqrt(max(ds2, 0))

        total_length += ds * dt

    return total_length

# ============================================================
# Step 5: Compute pairwise distances
# ============================================================
print("\n[5/6] 计算逐对测地线距离...")

results = []
n_pairs = len(pairs)

for p_idx, (i, j, pair_type, pair_label) in enumerate(pairs):
    if (p_idx + 1) % 500 == 0 or p_idx == 0:
        print(f"  进度: {p_idx + 1}/{n_pairs} ({100 * (p_idx + 1) / n_pairs:.1f}%)")

    x_a, x_b = X[i], X[j]

    # Euclidean distance
    d_euclidean = np.linalg.norm(x_b - x_a)

    # Global Mahalanobis distance
    d_global = D_global_mahalanobis[i, j]

    # Local geodesic distance (numerical integration)
    d_geodesic = compute_geodesic_length(x_a, x_b)

    # Geodesic curvature
    geo_curvature = d_geodesic / (d_euclidean + 1e-10) - 1.0

    # Delta n
    delta_n = abs(n_values[i] - n_values[j])

    # Same/different aa_type
    same_aa = 1 if aa_types[i] == aa_types[j] else 0

    results.append({
        'seq_id_a': seq_ids[i],
        'seq_id_b': seq_ids[j],
        'aa_type_a': aa_types[i],
        'aa_type_b': aa_types[j],
        'category_a': categories[i],
        'category_b': categories[j],
        'n_a': n_values[i],
        'n_b': n_values[j],
        'delta_n': delta_n,
        'same_aa': same_aa,
        'pair_type': pair_type,
        'pair_label': pair_label,
        'd_euclidean': d_euclidean,
        'd_global_mahalanobis': d_global,
        'd_geodesic_local': d_geodesic,
        'geo_curvature': geo_curvature,
        'ratio_geo_euc': d_geodesic / (d_euclidean + 1e-10),
        'ratio_global_euc': d_global / (d_euclidean + 1e-10),
    })

df_results = pd.DataFrame(results)
print(f"\n  完成 {len(df_results)} 对计算")

# ============================================================
# Step 6: Analysis and visualization
# ============================================================
print("\n[6/6] 分析和可视化...")

# --- 6a: Summary statistics ---
print("\n  --- 测地线曲率统计 ---")
print(f"    均值 κ_geo: {df_results['geo_curvature'].mean():.4f}")
print(f"    中位数 κ_geo: {df_results['geo_curvature'].median():.4f}")
print(f"    标准差 κ_geo: {df_results['geo_curvature'].std():.4f}")
print(f"    范围: [{df_results['geo_curvature'].min():.4f}, {df_results['geo_curvature'].max():.4f}]")
print(f"    κ_geo > 0 比例: {np.mean(df_results['geo_curvature'] > 0):.2%}")

print(f"\n    均值 d_E: {df_results['d_euclidean'].mean():.4f}")
print(f"    均值 d_G (local): {df_results['d_geodesic_local'].mean():.4f}")
print(f"    均值 d_M (global): {df_results['d_global_mahalanobis'].mean():.4f}")

# --- 6b: Correlation with delta_n ---
print("\n  --- κ_geo vs delta_n ---")
r_geo_dn, p_geo_dn = spearmanr(df_results['geo_curvature'], df_results['delta_n'])
r_pearson, p_pearson = pearsonr(df_results['geo_curvature'], df_results['delta_n'])
print(f"    Spearman r = {r_geo_dn:.4f}, p = {p_geo_dn:.4e}")
print(f"    Pearson r = {r_pearson:.4f}, p = {p_pearson:.4e}")

# --- 6c: Same AA vs cross AA ---
print("\n  --- Same AA vs Cross AA ---")
same_aa_data = df_results[df_results['same_aa'] == 1]['geo_curvature']
cross_aa_data = df_results[df_results['same_aa'] == 0]['geo_curvature']
print(f"    Same AA κ_geo: {same_aa_data.mean():.4f} ± {same_aa_data.std():.4f}")
print(f"    Cross AA κ_geo: {cross_aa_data.mean():.4f} ± {cross_aa_data.std():.4f}")
u_stat, u_p = mannwhitneyu(same_aa_data, cross_aa_data, alternative='two-sided')
print(f"    Mann-Whitney U p = {u_p:.4e}")
cohens_d = (same_aa_data.mean() - cross_aa_data.mean()) / (np.sqrt((same_aa_data.var() + cross_aa_data.var()) / 2) + 1e-10)
print(f"    Cohen's d = {cohens_d:.4f}")

# --- 6d: Per-AA analysis ---
print("\n  --- Per-AA 测地线曲率 ---")
aa_geo = df_results.groupby('aa_type_a')['geo_curvature'].agg(['mean', 'std', 'count']).reset_index()
aa_geo = aa_geo.sort_values('mean')
for _, row in aa_geo.iterrows():
    print(f"    {row['aa_type_a']:>12s}: κ={row['mean']:.4f} ± {row['std']:.4f} (n={int(row['count'])})")

# --- 6e: Per-category analysis ---
print("\n  --- Per-Category 测地线曲率 ---")
cat_geo = df_results.groupby('category_a')['geo_curvature'].agg(['mean', 'std', 'count']).reset_index()
cat_geo = cat_geo.sort_values('mean')
for _, row in cat_geo.iterrows():
    print(f"    {row['category_a']:>20s}: κ={row['mean']:.4f} ± {row['std']:.4f} (n={int(row['count'])})")

# --- 6f: d_E vs d_G comparison ---
r_e_g, p_e_g = spearmanr(df_results['d_euclidean'], df_results['d_geodesic_local'])
print(f"\n  d_E vs d_G Spearman r = {r_e_g:.4f}, p = {p_e_g:.4e}")

# ============================================================
# Visualization
# ============================================================
print("\n  生成可视化...")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    # 中文字体
    for font_name in ['Microsoft YaHei', 'SimHei', 'Arial']:
        try:
            matplotlib.rcParams['font.sans-serif'] = [font_name]
            matplotlib.rcParams['axes.unicode_minus'] = False
            break
        except Exception:
            continue

    # ---- Figure 1: Geodesic vs Euclidean scatter ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('G1: 测地线距离 vs 欧氏距离 — 系综几何空间曲率', fontsize=14, fontweight='bold')

    # (a) d_G vs d_E scatter
    ax = axes[0, 0]
    ax.scatter(df_results['d_euclidean'], df_results['d_geodesic_local'],
               c=df_results['delta_n'], cmap='viridis', alpha=0.4, s=8)
    lims = [0, max(df_results['d_euclidean'].max(), df_results['d_geodesic_local'].max()) * 1.05]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='d_G = d_E')
    ax.set_xlabel('Euclidean Distance d_E')
    ax.set_ylabel('Geodesic Distance d_G (local)')
    ax.set_title(f'd_G vs d_E (r={r_e_g:.3f})')
    ax.legend()
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('delta_n')

    # (b) κ_geo vs delta_n
    ax = axes[0, 1]
    ax.scatter(df_results['delta_n'], df_results['geo_curvature'],
               alpha=0.3, s=8, c='steelblue')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('delta_n')
    ax.set_ylabel('Geodesic Curvature κ_geo')
    ax.set_title(f'κ_geo vs delta_n (Spearman r={r_geo_dn:.3f}, p={p_geo_dn:.2e})')

    # (c) κ_geo histogram by same_aa
    ax = axes[0, 2]
    bins = np.linspace(df_results['geo_curvature'].min(), df_results['geo_curvature'].max(), 50)
    ax.hist(same_aa_data, bins=bins, alpha=0.6, label=f'Same AA (μ={same_aa_data.mean():.3f})', color='green')
    ax.hist(cross_aa_data, bins=bins, alpha=0.6, label=f'Cross AA (μ={cross_aa_data.mean():.3f})', color='orange')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('κ_geo')
    ax.set_ylabel('Count')
    ax.set_title(f'Same vs Cross AA (p={u_p:.2e})')
    ax.legend()

    # (d) Per-AA κ_geo bar chart
    ax = axes[1, 0]
    colors = plt.cm.RdYlBu_r((aa_geo['mean'] - aa_geo['mean'].min()) /
                              (aa_geo['mean'].max() - aa_geo['mean'].min() + 1e-10))
    ax.barh(range(len(aa_geo)), aa_geo['mean'].values, xerr=aa_geo['std'].values,
            color=colors, edgecolor='black', alpha=0.8)
    ax.set_yticks(range(len(aa_geo)))
    ax.set_yticklabels(aa_geo['aa_type_a'].values)
    ax.set_xlabel('Mean κ_geo')
    ax.set_title('Per-AA Geodesic Curvature')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)

    # (e) d_G/d_E ratio distribution
    ax = axes[1, 1]
    ax.hist(df_results['ratio_geo_euc'], bins=60, alpha=0.7, color='teal',
            label=f'Local (μ={df_results["ratio_geo_euc"].mean():.3f})')
    ax.hist(df_results['ratio_global_euc'], bins=60, alpha=0.5, color='salmon',
            label=f'Global (μ={df_results["ratio_global_euc"].mean():.3f})')
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('d_G / d_E Ratio')
    ax.set_ylabel('Count')
    ax.set_title('Local vs Global Metric Ratio')
    ax.legend()

    # (f) Per-category κ_geo
    ax = axes[1, 2]
    cat_colors = plt.cm.Set2(np.linspace(0, 1, len(cat_geo)))
    ax.bar(range(len(cat_geo)), cat_geo['mean'].values, yerr=cat_geo['std'].values,
           color=cat_colors, edgecolor='black', alpha=0.8)
    ax.set_xticks(range(len(cat_geo)))
    ax.set_xticklabels([c.replace('_', '\n') for c in cat_geo['category_a'].values],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean κ_geo')
    ax.set_title('Per-Category Geodesic Curvature')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    for fmt in ['svg', 'jpg']:
        fig.savefig(FIGURES_DIR / f'phase_ensemble_g1_geodesic_curvature.{fmt}',
                    dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    → phase_ensemble_g1_geodesic_curvature_v2.svg/jpg")

    # ---- Figure 2: Geodesic curvature heatmap by AA pair ----
    fig, ax = plt.subplots(figsize=(12, 10))

    # Build AA×AA matrix of mean κ_geo
    aa_list = sorted(set(df_results['aa_type_a'].unique()) | set(df_results['aa_type_b'].unique()))
    aa_to_idx = {aa: i for i, aa in enumerate(aa_list)}
    n_aa = len(aa_list)
    kappa_matrix = np.full((n_aa, n_aa), np.nan)
    count_matrix = np.zeros((n_aa, n_aa))

    for _, row in df_results.iterrows():
        i, j = aa_to_idx[row['aa_type_a']], aa_to_idx[row['aa_type_b']]
        if np.isnan(kappa_matrix[i, j]):
            kappa_matrix[i, j] = row['geo_curvature']
            kappa_matrix[j, i] = row['geo_curvature']
            count_matrix[i, j] = 1
            count_matrix[j, i] = 1
        else:
            kappa_matrix[i, j] = (kappa_matrix[i, j] * count_matrix[i, j] + row['geo_curvature']) / (count_matrix[i, j] + 1)
            kappa_matrix[j, i] = kappa_matrix[i, j]
            count_matrix[i, j] += 1
            count_matrix[j, i] += 1

    im = ax.imshow(kappa_matrix, cmap='RdBu_r', aspect='auto', vmin=-np.nanmax(np.abs(kappa_matrix)),
                   vmax=np.nanmax(np.abs(kappa_matrix)))
    ax.set_xticks(range(n_aa))
    ax.set_yticks(range(n_aa))
    ax.set_xticklabels(aa_list, rotation=45, ha='right')
    ax.set_yticklabels(aa_list)
    ax.set_title('Mean Geodesic Curvature κ_geo by AA Pair')
    plt.colorbar(im, ax=ax, label='κ_geo')

    for i in range(n_aa):
        for j in range(n_aa):
            if not np.isnan(kappa_matrix[i, j]):
                ax.text(j, i, f'{kappa_matrix[i, j]:.2f}', ha='center', va='center',
                        fontsize=7, color='black' if abs(kappa_matrix[i, j]) < 0.5 else 'white')

    plt.tight_layout()
    for fmt in ['svg', 'jpg']:
        fig.savefig(FIGURES_DIR / f'phase_ensemble_g1_aa_pair_heatmap.{fmt}',
                    dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    → phase_ensemble_g1_aa_pair_heatmap_v2.svg/jpg")

except Exception as e:
    print(f"  ⚠ 可视化生成失败: {e}")

# ============================================================
# Save results
# ============================================================
print("\n  保存结果...")

# CSV
df_results.to_csv(TABLES_DIR / 'phase_ensemble_g1_geodesic_comparison_v2.csv', index=False)
print(f"    → phase_ensemble_g1_geodesic_comparison_v2.csv ({len(df_results)} rows)")

# Summary JSON
summary = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_sequences': n_seqs,
    'n_pairs': len(df_results),
    'n_features': n_feats,
    'feature_columns': geo_cols,
    'global_metric_condition_number': float(eigvals.max() / (eigvals.min() + 1e-15)),
    'kappa_geo_mean': float(df_results['geo_curvature'].mean()),
    'kappa_geo_median': float(df_results['geo_curvature'].median()),
    'kappa_geo_std': float(df_results['geo_curvature'].std()),
    'kappa_geo_positive_frac': float(np.mean(df_results['geo_curvature'] > 0)),
    'd_euclidean_mean': float(df_results['d_euclidean'].mean()),
    'd_geodesic_mean': float(df_results['d_geodesic_local'].mean()),
    'd_global_mean': float(df_results['d_global_mahalanobis'].mean()),
    'kappa_vs_delta_n_spearman_r': float(r_geo_dn),
    'kappa_vs_delta_n_spearman_p': float(p_geo_dn),
    'kappa_vs_delta_n_pearson_r': float(r_pearson),
    'kappa_vs_delta_n_pearson_p': float(p_pearson),
    'same_aa_kappa_mean': float(same_aa_data.mean()),
    'cross_aa_kappa_mean': float(cross_aa_data.mean()),
    'same_vs_cross_mannwhitney_p': float(u_p),
    'same_vs_cross_cohens_d': float(cohens_d),
    'd_e_vs_d_g_spearman_r': float(r_e_g),
    'd_e_vs_d_g_spearman_p': float(p_e_g),
    'per_aa': {row['aa_type_a']: {'mean': float(row['mean']), 'std': float(row['std']), 'n': int(row['count'])}
               for _, row in aa_geo.iterrows()},
    'per_category': {row['category_a']: {'mean': float(row['mean']), 'std': float(row['std']), 'n': int(row['count'])}
                     for _, row in cat_geo.iterrows()},
}

with open(TABLES_DIR / 'phase_ensemble_g1_summary_v2.json', 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"    → phase_ensemble_g1_summary_v2.json")

# ============================================================
# Key Findings
# ============================================================
print(f"\n{'='*70}")
print("G1 完成! 测地线距离 vs 欧氏距离分析")
print(f"{'='*70}")
print(f"\n关键发现:")
print(f"  1. 测地线曲率 κ_geo = {df_results['geo_curvature'].mean():.4f} ± {df_results['geo_curvature'].std():.4f}")
print(f"     → 空间{'弯曲' if abs(df_results['geo_curvature'].mean()) > 0.01 else '近似平坦'}")
print(f"  2. κ_geo 与 delta_n 相关性: Spearman r={r_geo_dn:.4f} (p={p_geo_dn:.2e})")
print(f"     → 序列长度差异{'显著影响' if p_geo_dn < 0.05 else '无显著影响'}空间曲率")
print(f"  3. Same AA vs Cross AA: p={u_p:.4e}, d={cohens_d:.4f}")
print(f"     → AA类型{'显著影响' if u_p < 0.05 else '无显著影响'}测地线曲率")
print(f"  4. d_E vs d_G 相关性: Spearman r={r_e_g:.4f}")
print(f"     → 欧氏距离与测地线距离{'高度一致' if r_e_g > 0.9 else '存在差异'}")
print(f"\n输出文件:")
print(f"  CSV: {TABLES_DIR / 'phase_ensemble_g1_geodesic_comparison_v2.csv'}")
print(f"  JSON: {TABLES_DIR / 'phase_ensemble_g1_summary_v2.json'}")
print(f"  图: {FIGURES_DIR / 'phase_ensemble_g1_geodesic_curvature.*'}")
print(f"  图: {FIGURES_DIR / 'phase_ensemble_g1_aa_pair_heatmap.*'}")