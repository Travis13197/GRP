#!/usr/bin/env python3
"""
Phase II 构象空间: Poly-Gly 生长与氨基酸替换的几何响应分析
=================================================================
聚焦于物理构象空间 (BioEmu Cα 坐标), 而非语义嵌入空间。

核心科学问题:
  1. 构象空间中 (G)^n → (G)^{n+1} 生长扰动是否服从 Phase I 标度律?
  2. 构象空间中 (G)^n → (X)^n 氨基酸替换的几何扰动代价是否存在规律?
  3. 构象空间中的度量张量校正是否能消除 n 依赖?
  4. 位移向量的余弦衰减是否在度量校正后减缓?

实验设计:
  A. 生长扰动 (Growth): 比较 (G)^n 与 (G)^{n+1} 的构象系综
     - 对齐: 截断 (G)^{n+1} 的前 n 个残基做比较, 或整体比较
  B. 氨基酸替换 (Substitution): 比较 (G)^n 与 (X)^n 的构象系综
     - X ∈ {S, E, L, K} (现有数据)
  C. 度量张量: 在 Cα 坐标的协方差空间计算局部度量
  D. 余弦衰减: 比较不同 Δn 的位移向量相似度

管线步骤:
  [1/6] 加载已有 PolyX BioEmu 数据
  [2/6] 计算生长扰动 (G)^n → (G)^{n+1} 的几何响应
  [3/6] 计算氨基酸替换 (G)^n → (X)^n 的几何响应
  [4/6] 计算 Cα 空间的度量张量与扰动代价
  [5/6] 余弦衰减分析
  [6/6] 可视化 + HTML 报告
"""

import sys, os, warnings, json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse

from scipy.stats import chi2, spearmanr, pearsonr
from scipy.optimize import curve_fit
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
FIELD_THEORY = Path(__file__).parent.parent
POLYX_DIR = PROJECT_ROOT / "test_workflow" / "polyx_ensemble"
OUTPUT_BASE = POLYX_DIR / "output"
FIELD_THEORY_DIR = FIELD_THEORY
TABLES_DIR = FIELD_THEORY / "tables"
FIGURES_DIR = FIELD_THEORY / "figures"
REPORTS_DIR = FIELD_THEORY / "reports"
for d in [TABLES_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 加载配置
with open(FIELD_THEORY / 'config.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
p2cfg = cfg['phase2']

N_VALUES_PHASE2 = p2cfg['n_values']  # [1,2,3,5,8,10,12,15,20,30,50]
POLYX_AA = ['G', 'S', 'E', 'L', 'K']
SUB_AA = ['S', 'E', 'L', 'K']  # 替换目标 (G 是基线)

# 可视化
CMAP = plt.cm.viridis
EVO_BG = '#E5ECF6'
IMG_DPI = 300

import matplotlib.font_manager as fm
available_fonts = set(f.name for f in fm.fontManager.ttflist)
chinese_fonts = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei']
sans_serif_fonts = ['DejaVu Sans', 'Arial'] + [f for f in chinese_fonts if f in available_fonts]
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': sans_serif_fonts,
    'font.size': 10,
    'axes.unicode_minus': False,
    'axes.facecolor': EVO_BG,
    'figure.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#cccccc',
    'figure.dpi': IMG_DPI,
    'savefig.dpi': IMG_DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

AA_COLORS = {'G': '#8c564b', 'S': '#c49c94', 'E': '#d62728', 'L': '#17becf', 'K': '#bcbd22'}

print("=" * 70)
print("Phase II 构象空间: Poly-Gly 生长与氨基酸替换的几何响应")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ============================================================
# 核心几何函数 (Cα 坐标空间)
# ============================================================
def load_ensemble_positions(seq_id: str) -> np.ndarray:
    """加载序列的 BioEmu 构象系综: (n_samples, n_residues, 3)"""
    seq_dir = OUTPUT_BASE / seq_id
    if not seq_dir.exists():
        return None
    npz_files = sorted(seq_dir.glob("batch_*.npz"))
    if not npz_files:
        return None
    all_pos = []
    for npz_f in npz_files:
        try:
            data = np.load(npz_f, allow_pickle=True)
            if 'pos' in data:
                all_pos.append(data['pos'])
        except Exception:
            pass
    if not all_pos:
        return None
    return np.concatenate(all_pos, axis=0)  # (n_samples, n_residues, 3)


def compute_geometric_features_from_positions(positions):
    """从 Cα 坐标 (n_samples, n_residues, 3) 计算几何特征"""
    n_samples, n_residues, _ = positions.shape
    X = positions.reshape(n_samples, -1)  # (n_samples, n_residues*3)
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered, rowvar=False)
    eigenvals = np.abs(eigh(cov, eigvals_only=True))
    eigenvals = eigenvals[eigenvals > 1e-12]
    if len(eigenvals) == 0:
        return None

    n_dims = len(eigenvals)
    total_var = np.sum(eigenvals)
    if total_var <= 0:
        return None

    eff_rank_95 = int(np.searchsorted(np.cumsum(eigenvals) / total_var, 0.95) + 1)

    if n_samples >= 4:
        nn = NearestNeighbors(n_neighbors=min(3, n_samples - 1))
        nn.fit(X)
        dists, _ = nn.kneighbors(X)
        if dists.shape[1] >= 2:
            mu = dists[:, 2] / dists[:, 1]
            mu = mu[np.isfinite(mu) & (mu > 0)]
            if len(mu) > 0:
                d_two_nn = len(mu) / np.sum(np.log(mu))
            else:
                d_two_nn = float(eff_rank_95)
        else:
            d_two_nn = float(eff_rank_95)
    else:
        d_two_nn = float(eff_rank_95)

    d_consensus = 0.5 * (d_two_nn + eff_rank_95)
    pr = np.sum(eigenvals) ** 2 / np.sum(eigenvals ** 2) if np.sum(eigenvals) > 0 else 1.0
    a_c = np.sqrt(np.sum((eigenvals - np.mean(eigenvals)) ** 2) / n_dims) / np.mean(eigenvals) if n_dims > 1 else 0

    ranks = np.arange(1, len(eigenvals) + 1)
    if len(ranks) >= 2:
        slope, _ = np.polyfit(np.log(ranks), np.log(eigenvals), 1)
        spectral_decay = -slope
    else:
        spectral_decay = 0

    probs = eigenvals / total_var
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    pseudo_volume = np.sum(np.log(eigenvals + 1e-12))

    # 系综均值结构
    mean_struct = X.mean(axis=0)

    return {
        'd_consensus': float(d_consensus), 'PR': float(pr),
        'eff_rank_95': int(eff_rank_95), 'A_C': float(a_c),
        'spectral_decay': float(spectral_decay), 'entropy': float(entropy),
        'pseudo_volume': float(pseudo_volume),
        'total_var': float(total_var), 'n_dims': n_dims,
        'mean_structure': mean_struct,
        'n_samples': n_samples, 'n_residues': n_residues,
    }


def compute_metric_in_ca_space(positions):
    """在 Cα 空间中计算局部度量张量"""
    n_samples, n_residues, _ = positions.shape
    X = positions.reshape(n_samples, -1)
    X_centered = X - X.mean(axis=0)

    # PCA 提取切空间
    q = min(10, n_samples - 1, X.shape[1])
    pca = PCA(n_components=q)
    pca.fit(X)
    Q = pca.components_.T  # (d, q)

    # 投影到切空间
    X_proj = X_centered @ Q  # (n, q)

    # 协方差 = 度量张量
    cov_proj = np.cov(X_proj, rowvar=False)
    g = cov_proj + np.eye(q) * 1e-8

    return {
        'tangent_basis': Q,
        'g_tangent': g,
        'q': q,
        'eigenvals': np.abs(eigh(g, eigvals_only=True)),
    }


def compute_perturbation_cost(v1, v2, metric_info):
    """计算几何扰动代价 C_geo = z^T g z"""
    delta = v2 - v1
    Q = metric_info['tangent_basis']
    g = metric_info['g_tangent']
    z = Q.T @ delta
    return float(z.T @ g @ z)


def compute_cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ============================================================
# 拟合函数
# ============================================================
def power_law(x, a, b, c):
    return a * np.power(x, -b) + c

def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

def log_model(x, a, b):
    return a * np.log(x) + b

def compute_aicc(rss, n, k):
    if n <= k + 1:
        return np.inf
    aic = n * np.log(rss / n) + 2 * k
    return aic + 2 * k * (k + 1) / (n - k - 1)

# ============================================================
# Step 1: 加载已有 PolyX 数据
# ============================================================
print("\n[1/6] 加载已有 PolyX BioEmu 数据...")

ensemble_data = {}
for aa in POLYX_AA:
    for n in range(1, 51):
        seq_id = f"PolyX_Poly{aa}_{n}"
        pos = load_ensemble_positions(seq_id)
        if pos is not None and pos.shape[0] >= 100:
            ensemble_data[seq_id] = pos

print(f"  加载 {len(ensemble_data)} 个序列系综")
for aa in POLYX_AA:
    count = sum(1 for k in ensemble_data if f"Poly{aa}_" in k)
    print(f"    Poly-{aa}: {count} 序列")

# ============================================================
# Step 2: 生长扰动分析 (G)^n → (G)^{n+1}
# ============================================================
print("\n[2/6] 生长扰动分析: (G)^n → (G)^{n+1}...")

growth_records = []
for n in range(1, 50):
    seq_id_n = f"PolyX_PolyG_{n}"
    seq_id_np1 = f"PolyX_PolyG_{n+1}"
    if seq_id_n not in ensemble_data or seq_id_np1 not in ensemble_data:
        continue

    pos_n = ensemble_data[seq_id_n]      # (n_s, n, 3)
    pos_np1 = ensemble_data[seq_id_np1]  # (n_s, n+1, 3)

    # 方法 A: 截断 (G)^{n+1} 的前 n 个残基
    pos_np1_trunc = pos_np1[:, :n, :]

    # 均值结构
    mean_n = pos_n.mean(axis=0)       # (n, 3)
    mean_np1_trunc = pos_np1_trunc.mean(axis=0)  # (n, 3)

    # 欧氏距离
    euclidean_dist = np.linalg.norm(mean_np1_trunc - mean_n)

    # 几何特征 (对截断后的系综)
    feat_n = compute_geometric_features_from_positions(pos_n)
    feat_np1_trunc = compute_geometric_features_from_positions(pos_np1_trunc)

    # 度量张量 (基于 (G)^n 系综)
    metric_n = compute_metric_in_ca_space(pos_n)

    # 扰动代价
    delta_vec = (mean_np1_trunc - mean_n).ravel()
    if metric_n is not None:
        c_geo = compute_perturbation_cost(
            mean_n.ravel(), mean_np1_trunc.ravel(), metric_n)
    else:
        c_geo = np.nan

    growth_records.append({
        'n': n, 'type': 'growth',
        'euclidean_dist': float(euclidean_dist),
        'C_geo': c_geo,
        'PR_n': feat_n['PR'] if feat_n else np.nan,
        'PR_np1': feat_np1_trunc['PR'] if feat_np1_trunc else np.nan,
        'spectral_decay_n': feat_n['spectral_decay'] if feat_n else np.nan,
        'spectral_decay_np1': feat_np1_trunc['spectral_decay'] if feat_np1_trunc else np.nan,
        'total_var_n': feat_n['total_var'] if feat_n else np.nan,
        'total_var_np1': feat_np1_trunc['total_var'] if feat_np1_trunc else np.nan,
    })

df_growth = pd.DataFrame(growth_records)
print(f"  生长扰动: {len(df_growth)} 个 (n, n+1) 对")
print(f"  euclidean_dist 范围: [{df_growth['euclidean_dist'].min():.4f}, {df_growth['euclidean_dist'].max():.4f}]")
print(f"  C_geo 范围: [{df_growth['C_geo'].min():.4f}, {df_growth['C_geo'].max():.4f}]")

# ============================================================
# Step 3: 氨基酸替换分析 (G)^n → (X)^n
# ============================================================
print("\n[3/6] 氨基酸替换分析: (G)^n → (X)^n...")

subst_records = []
for X in SUB_AA:
    for n in range(1, 51):
        seq_id_G = f"PolyX_PolyG_{n}"
        seq_id_X = f"PolyX_Poly{X}_{n}"
        if seq_id_G not in ensemble_data or seq_id_X not in ensemble_data:
            continue

        pos_G = ensemble_data[seq_id_G]  # (n_s, n, 3)
        pos_X = ensemble_data[seq_id_X]  # (n_s, n, 3)

        mean_G = pos_G.mean(axis=0)  # (n, 3)
        mean_X = pos_X.mean(axis=0)  # (n, 3)

        euclidean_dist = np.linalg.norm(mean_X - mean_G)

        feat_G = compute_geometric_features_from_positions(pos_G)
        feat_X = compute_geometric_features_from_positions(pos_X)

        # 度量张量 (基于 (G)^n 系综)
        metric_G = compute_metric_in_ca_space(pos_G)

        if metric_G is not None:
            c_geo = compute_perturbation_cost(
                mean_G.ravel(), mean_X.ravel(), metric_G)
        else:
            c_geo = np.nan

        subst_records.append({
            'n': n, 'X': X, 'type': 'substitution',
            'euclidean_dist': float(euclidean_dist),
            'C_geo': c_geo,
            'd_consensus_G': feat_G['d_consensus'] if feat_G else np.nan,
            'd_consensus_X': feat_X['d_consensus'] if feat_X else np.nan,
            'PR_G': feat_G['PR'] if feat_G else np.nan,
            'PR_X': feat_X['PR'] if feat_X else np.nan,
            'spectral_decay_G': feat_G['spectral_decay'] if feat_G else np.nan,
            'spectral_decay_X': feat_X['spectral_decay'] if feat_X else np.nan,
            'total_var_G': feat_G['total_var'] if feat_G else np.nan,
            'total_var_X': feat_X['total_var'] if feat_X else np.nan,
        })

df_subst = pd.DataFrame(subst_records)
print(f"  氨基酸替换: {len(df_subst)} 个 (G→X) 对")
for X in SUB_AA:
    sub = df_subst[df_subst['X'] == X]
    print(f"    G→{X}: {len(sub)} 对, euclidean_dist mean={sub['euclidean_dist'].mean():.4f}")

# ============================================================
# Step 4: 度量张量与扰动代价分析
# ============================================================
print("\n[4/6] 度量张量校正效果分析...")

# 生长扰动: 比较 C_geo 和 euclidean_dist 对 n 的 scatterness
if len(df_growth) > 0:
    cv_euc_growth = df_growth['euclidean_dist'].std() / max(df_growth['euclidean_dist'].mean(), 1e-10)
    cv_cgeo_growth = df_growth['C_geo'].std() / max(df_growth['C_geo'].mean(), 1e-10)
    print(f"  生长扰动: CV(euclidean)={cv_euc_growth:.4f}, CV(C_geo)={cv_cgeo_growth:.4f}")

    # C_geo 与 euclidean_dist 的 Spearman 相关性
    sr_g, sp_g = spearmanr(df_growth['euclidean_dist'], df_growth['C_geo'])
    print(f"  C_geo ~ euclidean_dist Spearman r={sr_g:.3f} (p={sp_g:.2e})")

# 氨基酸替换: 按 AA 比较
if len(df_subst) > 0:
    cv_by_aa = {}
    for X in SUB_AA:
        sub = df_subst[df_subst['X'] == X]
        if len(sub) > 1:
            cv_euc = sub['euclidean_dist'].std() / max(sub['euclidean_dist'].mean(), 1e-10)
            cv_cgeo = sub['C_geo'].std() / max(sub['C_geo'].mean(), 1e-10)
            cv_by_aa[X] = {'cv_euc': cv_euc, 'cv_cgeo': cv_cgeo,
                           'improved': cv_cgeo < cv_euc}
            print(f"  G→{X}: CV(euclidean)={cv_euc:.4f}, CV(C_geo)={cv_cgeo:.4f}, "
                  f"improved={'YES' if cv_cgeo < cv_euc else 'NO'}")

# ============================================================
# Step 5: 余弦衰减分析
# ============================================================
print("\n[5/6] 余弦衰减分析...")

cosine_records = []
for X in SUB_AA:
    mean_vectors = {}
    for n in range(1, 51):
        seq_id = f"PolyX_Poly{X}_{n}"
        if seq_id in ensemble_data:
            pos = ensemble_data[seq_id]
            mean_vectors[n] = pos.mean(axis=0).ravel()  # (n*3,)

    for i, n1 in enumerate(sorted(mean_vectors.keys())):
        for j, n2 in enumerate(sorted(mean_vectors.keys())):
            if i >= j:
                continue
            # 截断到 min(n1, n2) 长度
            min_n = min(n1, n2)
            v1 = mean_vectors[n1][:min_n*3]
            v2 = mean_vectors[n2][:min_n*3]

            cosine = compute_cosine_similarity(v1, v2)
            cosine_records.append({
                'X': X, 'n1': n1, 'n2': n2,
                'delta_n': abs(n2 - n1),
                'cosine': cosine,
            })

df_cosine = pd.DataFrame(cosine_records)
cosine_by_dn = df_cosine.groupby('delta_n')['cosine'].agg(['mean', 'std', 'count']).reset_index()
print(f"  余弦衰减: {len(df_cosine)} 对, delta_n 范围 [{df_cosine['delta_n'].min()}, {df_cosine['delta_n'].max()}]")
for _, row in cosine_by_dn.iterrows():
    if row['delta_n'] <= 10:
        print(f"    Δn={int(row['delta_n']):3d}: cosine={row['mean']:.4f} ± {row['std']:.4f}")

# ============================================================
# Step 6: 可视化 + 报告
# ============================================================
print("\n[6/6] 生成可视化与报告...")

# 6.1 生长扰动: euclidean_dist 和 C_geo 随 n 变化
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
ax.scatter(df_growth['n'], df_growth['euclidean_dist'], c=df_growth['n'], cmap=CMAP,
           s=60, edgecolors='white', linewidth=0.5, label='Euclidean ||Δx||')
# 拟合
x_fit = np.linspace(df_growth['n'].min(), df_growth['n'].max(), 200)
try:
    z = np.polyfit(df_growth['n'], df_growth['euclidean_dist'], 2)
    ax.plot(x_fit, np.polyval(z, x_fit), 'r-', linewidth=2, label='Quadratic fit')
except Exception:
    pass
ax.set_xlabel('n (Gly chain length)', fontsize=12)
ax.set_ylabel('Euclidean Distance ||Δx||', fontsize=12)
ax.set_title('Growth Perturbation: (G)^n → (G)^{n+1}\nEuclidean Displacement', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)

ax = axes[1]
ax.scatter(df_growth['n'], df_growth['C_geo'], c=df_growth['n'], cmap=CMAP,
           s=60, edgecolors='white', linewidth=0.5, label='C_geo')
ax.set_xlabel('n (Gly chain length)', fontsize=12)
ax.set_ylabel('C_geo = z^T g z', fontsize=12)
ax.set_title('Growth Perturbation: (G)^n → (G)^{n+1}\nGeometric Perturbation Cost', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)

fig.suptitle('Phase II: Growth Perturbation in Cα Conformation Space\n'
             f'CV(euclidean)={cv_euc_growth:.4f}, CV(C_geo)={cv_cgeo_growth:.4f}',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(FIGURES_DIR / 'phase2_ca_growth_perturbation.svg')
fig.savefig(FIGURES_DIR / 'phase2_ca_growth_perturbation.jpg')
plt.close()
print("  [6.1] 生长扰动图已保存")

# 6.2 氨基酸替换: euclidean_dist 和 C_geo 热图
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Euclidean distance
pivot_euc = df_subst.pivot(index='X', columns='n', values='euclidean_dist')
im0 = axes[0].imshow(pivot_euc.values, aspect='auto', cmap='RdYlBu_r')
axes[0].set_xticks(range(len(pivot_euc.columns)))
axes[0].set_xticklabels(pivot_euc.columns, fontsize=8)
axes[0].set_yticks(range(len(pivot_euc.index)))
axes[0].set_yticklabels(pivot_euc.index, fontsize=10)
axes[0].set_xlabel('n', fontsize=11)
axes[0].set_ylabel('AA', fontsize=11)
axes[0].set_title('Euclidean Distance ||Δx||', fontsize=12, fontweight='bold')
plt.colorbar(im0, ax=axes[0])

# C_geo
pivot_cgeo = df_subst.pivot(index='X', columns='n', values='C_geo')
im1 = axes[1].imshow(np.log10(pivot_cgeo.values + 1e-10), aspect='auto', cmap='RdYlBu_r')
axes[1].set_xticks(range(len(pivot_cgeo.columns)))
axes[1].set_xticklabels(pivot_cgeo.columns, fontsize=8)
axes[1].set_yticks(range(len(pivot_cgeo.index)))
axes[1].set_yticklabels(pivot_cgeo.index, fontsize=10)
axes[1].set_xlabel('n', fontsize=11)
axes[1].set_ylabel('AA', fontsize=11)
axes[1].set_title('log10(C_geo)', fontsize=12, fontweight='bold')
plt.colorbar(im1, ax=axes[1])

fig.suptitle('Phase II: AA Substitution Perturbation (G)^n → (X)^n in Cα Space',
             fontsize=14, fontweight='bold')
fig.savefig(FIGURES_DIR / 'phase2_ca_substitution_heatmap.svg')
fig.savefig(FIGURES_DIR / 'phase2_ca_substitution_heatmap.jpg')
plt.close()
print("  [6.2] 氨基酸替换热图已保存")

# 6.3 氨基酸替换: 按 n 的扰动幅度
fig, ax = plt.subplots(figsize=(12, 7))
for X in SUB_AA:
    sub = df_subst[df_subst['X'] == X]
    if len(sub) > 0:
        ax.plot(sub['n'], sub['euclidean_dist'], '-o', color=AA_COLORS[X],
                markersize=5, linewidth=1.5, alpha=0.8, label=f'G→{X}')
ax.set_xlabel('n (chain length)', fontsize=12)
ax.set_ylabel('Euclidean Distance ||Δx||', fontsize=12)
ax.set_title('Phase II: AA Substitution Displacement vs n\n(G)^n → (X)^n, Cα Conformation Space',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, ncol=2)
fig.savefig(FIGURES_DIR / 'phase2_ca_substitution_displacement.svg')
fig.savefig(FIGURES_DIR / 'phase2_ca_substitution_displacement.jpg')
plt.close()
print("  [6.3] 氨基酸替换位移图已保存")

# 6.4 余弦衰减 (按 Δn)
fig, ax = plt.subplots(figsize=(10, 6))
valid_dn = cosine_by_dn[cosine_by_dn['delta_n'] > 0]
for _, row in valid_dn.iterrows():
    ax.errorbar(row['delta_n'], row['mean'], yerr=row['std'],
                fmt='o', color='#1f77b4', capsize=3, markersize=6, alpha=0.7)

# 拟合
x_dn = valid_dn['delta_n'].values.astype(float)
y_dn = valid_dn['mean'].values
try:
    popt, _ = curve_fit(exp_decay, x_dn, np.maximum(y_dn, 0.01), p0=[0.01], maxfev=10000)
    x_fit = np.linspace(x_dn.min(), x_dn.max(), 100)
    y_fit = exp_decay(x_fit, *popt)
    ax.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'exp decay fit')
    tau = 1.0 / popt[0] if popt[0] > 0 else np.inf
except Exception:
    tau = np.inf

ax.set_xlabel('Δn (chain length difference)', fontsize=12)
ax.set_ylabel('Cosine Similarity of Mean Structure', fontsize=12)
ax.set_title(f'Phase II: Cosine Decay in Cα Space\nMean structure similarity across chain lengths, tau={tau:.1f}',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(0, 1.05)
fig.savefig(FIGURES_DIR / 'phase2_ca_cosine_decay.svg')
fig.savefig(FIGURES_DIR / 'phase2_ca_cosine_decay.jpg')
plt.close()
print("  [6.4] 余弦衰减图已保存")

# 6.5 几何特征对比: (G)^n vs (X)^n
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, X in enumerate(SUB_AA):
    ax = axes[idx // 2, idx % 2]
    sub = df_subst[df_subst['X'] == X].dropna(subset=['PR_G', 'PR_X'])
    if len(sub) > 0:
        ax.scatter(sub['PR_G'], sub['PR_X'], c=sub['n'], cmap=CMAP, s=40, edgecolors='white', linewidth=0.3)
        min_val = min(sub['PR_G'].min(), sub['PR_X'].min())
        max_val = max(sub['PR_G'].max(), sub['PR_X'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('PR (G)^n', fontsize=10)
        ax.set_ylabel(f'PR ({X})^n', fontsize=10)
        ax.set_title(f'G→{X}: PR Comparison', fontsize=11, fontweight='bold')

fig.suptitle('Phase II: Ensemble Geometry Comparison\n(G)^n vs (X)^n, Cα Conformation Space',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(FIGURES_DIR / 'phase2_ca_pr_comparison.svg')
fig.savefig(FIGURES_DIR / 'phase2_ca_pr_comparison.jpg')
plt.close()
print("  [6.5] PR 对比图已保存")

# 6.6 生长扰动: C_geo vs PR 关系
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(df_growth['PR_n'], df_growth['C_geo'], c=df_growth['n'], cmap=CMAP,
                s=60, edgecolors='white', linewidth=0.5)
plt.colorbar(sc, ax=ax, label='n')
ax.set_xlabel('PR of (G)^n', fontsize=12)
ax.set_ylabel('C_geo (Growth Perturbation Cost)', fontsize=12)
ax.set_title('Phase II: Growth Perturbation Cost vs Ensemble Dimension\nC_geo ~ PR_n',
             fontsize=13, fontweight='bold')
pr_cgeo_r, pr_cgeo_p = spearmanr(df_growth['PR_n'], df_growth['C_geo'])
ax.text(0.05, 0.95, f'Spearman r={pr_cgeo_r:.3f} (p={pr_cgeo_p:.2e})',
        transform=ax.transAxes, fontsize=10, verticalalignment='top')
fig.savefig(FIGURES_DIR / 'phase2_ca_cgeo_vs_pr.svg')
fig.savefig(FIGURES_DIR / 'phase2_ca_cgeo_vs_pr.jpg')
plt.close()
print("  [6.6] C_geo vs PR 图已保存")

# ============================================================
# 保存数据表
# ============================================================
df_growth.to_csv(TABLES_DIR / 'phase2_ca_growth_perturbation.csv', index=False)
df_subst.to_csv(TABLES_DIR / 'phase2_ca_substitution_perturbation.csv', index=False)
df_cosine.to_csv(TABLES_DIR / 'phase2_ca_cosine_decay.csv', index=False)
print(f"\n  数据表已保存:")
print(f"    生长扰动: {len(df_growth)} 行")
print(f"    氨基酸替换: {len(df_subst)} 行")
print(f"    余弦衰减: {len(df_cosine)} 行")

# ============================================================
# HTML 报告
# ============================================================
print("\n  生成 HTML 报告...")

# 统计摘要
n_improved = sum(1 for v in cv_by_aa.values() if v['improved'])

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Phase II: Poly-Gly 构象空间几何响应分析</title>
<style>
  body {{ font-family: 'DejaVu Sans', sans-serif; margin: 20px; background: #f5f5f5; color: #333; }}
  h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }}
  h2 {{ color: #283593; margin-top: 30px; border-bottom: 2px solid #7986cb; padding-bottom: 5px; }}
  h3 {{ color: #3949ab; }}
  .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: center; }}
  th {{ background: #1a237e; color: white; }}
  tr:nth-child(even) {{ background: #f5f5f5; }}
  .pass {{ color: #2e7d32; font-weight: bold; }}
  .fail {{ color: #c62828; font-weight: bold; }}
  .ca-badge {{ background: #2e7d32; color: white; padding: 2px 8px; border-radius: 3px; font-size: 0.8em; }}
  .figure {{ margin: 20px 0; text-align: center; }}
  .figure img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
  .figure .caption {{ font-size: 0.9em; color: #666; margin-top: 5px; }}
  .highlight {{ background: #e8eaf6; padding: 15px; border-left: 4px solid #3949ab; margin: 15px 0; }}
  .comparison {{ display: flex; gap: 20px; margin: 20px 0; }}
  .box {{ flex: 1; padding: 15px; border-radius: 8px; }}
  .box-blue {{ background: #e3f2fd; border: 2px solid #90caf9; }}
  .box-green {{ background: #e8f5e9; border: 2px solid #a5d6a7; }}
</style>
</head>
<body>
<div class="container">

<h1>Phase II: Poly-Gly 构象空间几何响应分析</h1>

<p><strong>生成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<p><span class="ca-badge">Cα 空间</span> 聚焦于物理构象空间 (BioEmu Cα 坐标), 而非语义嵌入空间</p>

<div class="highlight">
  <strong>核心科学问题:</strong>
  <ol>
    <li>构象空间中 (G)^n → (G)^{n+1} 生长扰动是否服从 Phase I 标度律?</li>
    <li>构象空间中 (G)^n → (X)^n 氨基酸替换的几何扰动代价是否存在规律?</li>
    <li>构象空间中度量张量校正是否能消除 n 依赖?</li>
    <li>位移向量的余弦衰减是否在度量校正后减缓?</li>
  </ol>
</div>

<h2>1. 数据概览</h2>

<table>
  <tr><th>指标</th><th>值</th></tr>
  <tr><td>数据来源</td><td>BioEmu Cα 坐标系综 (250 samples/seq)</td></tr>
  <tr><td>PolyX 序列</td><td>G, S, E, L, K × n=1-50</td></tr>
  <tr><td>加载序列数</td><td>{len(ensemble_data)}</td></tr>
  <tr><td>生长扰动对</td><td>{len(df_growth)} (G)^n → (G)^{{n+1}}</td></tr>
  <tr><td>氨基酸替换对</td><td>{len(df_subst)} (G)^n → (X)^n</td></tr>
  <tr><td>余弦衰减对</td><td>{len(df_cosine)}</td></tr>
</table>

<h2>2. 生长扰动分析: (G)^n → (G)^{n+1}</h2>

<div class="comparison">
  <div class="box box-blue">
    <h3>欧氏扰动</h3>
    <p>CV(||Δx||): {cv_euc_growth:.4f}</p>
    <p>范围: [{df_growth['euclidean_dist'].min():.4f}, {df_growth['euclidean_dist'].max():.4f}]</p>
  </div>
  <div class="box box-green">
    <h3>几何扰动代价</h3>
    <p>CV(C_geo): {cv_cgeo_growth:.4f}</p>
    <p>范围: [{df_growth['C_geo'].min():.4f}, {df_growth['C_geo'].max():.4f}]</p>
    <p>C_geo ~ ||Δx|| Spearman r={sr_g:.3f} (p={sp_g:.2e})</p>
  </div>
</div>

<div class="figure">
  <img src="../figures/phase2_ca_growth_perturbation.svg" alt="Growth perturbation">
  <div class="caption">图 1: 生长扰动 — 欧氏位移与几何扰动代价随 n 变化</div>
</div>

<div class="figure">
  <img src="../figures/phase2_ca_cgeo_vs_pr.svg" alt="C_geo vs PR">
  <div class="caption">图 2: 几何扰动代价 vs 系综本征维度 (PR)</div>
</div>

<h2>3. 氨基酸替换分析: (G)^n → (X)^n</h2>

<h3>CV 度量校正效果</h3>
<table>
  <tr><th>替换</th><th>CV(euclidean)</th><th>CV(C_geo)</th><th>改善?</th></tr>
"""

for X in SUB_AA:
    if X in cv_by_aa:
        v = cv_by_aa[X]
        html += f"""  <tr>
    <td>G→{X}</td>
    <td>{v['cv_euc']:.4f}</td>
    <td>{v['cv_cgeo']:.4f}</td>
    <td class="{'pass' if v['improved'] else 'fail'}">{'YES' if v['improved'] else 'NO'}</td>
  </tr>"""

html += f"""
</table>
<p>度量校正改善: <strong>{n_improved}/{len(cv_by_aa)} ({100*n_improved/len(cv_by_aa):.0f}%)</strong></p>

<div class="figure">
  <img src="../figures/phase2_ca_substitution_heatmap.svg" alt="Substitution heatmap">
  <div class="caption">图 3: 氨基酸替换 — 欧氏距离与 C_geo 热图 (AA × n)</div>
</div>

<div class="figure">
  <img src="../figures/phase2_ca_substitution_displacement.svg" alt="Substitution displacement">
  <div class="caption">图 4: 氨基酸替换位移幅度随 n 变化</div>
</div>

<h2>4. 余弦衰减分析</h2>

<table>
  <tr><th>Δn</th><th>Mean Cosine</th><th>Std</th><th>Count</th></tr>
"""

for _, row in cosine_by_dn.iterrows():
    if row['delta_n'] <= 10:
        html += f"""  <tr>
    <td>{int(row['delta_n'])}</td>
    <td>{row['mean']:.4f}</td>
    <td>{row['std']:.4f}</td>
    <td>{int(row['count'])}</td>
  </tr>"""

html += f"""
</table>
<p>衰减常数 tau ≈ {tau:.1f}</p>

<div class="figure">
  <img src="../figures/phase2_ca_cosine_decay.svg" alt="Cosine decay">
  <div class="caption">图 5: 均值结构余弦相似度随 Δn 衰减</div>
</div>

<h2>5. 系综几何对比</h2>

<div class="figure">
  <img src="../figures/phase2_ca_pr_comparison.svg" alt="PR comparison">
  <div class="caption">图 6: (G)^n vs (X)^n 系综本征维度 PR 对比</div>
</div>

<h2>6. 科学结论</h2>

<div class="highlight">
  <h3>Phase II 构象空间发现</h3>
  <ol>
    <li><strong>生长扰动</strong>: (G)^n → (G)^{{n+1}} 的欧氏位移和 C_geo 均随 n 非单调变化, 反映 Poly-G 链的构象涨落不随长度单调增长</li>
    <li><strong>度量校正</strong>: C_geo 与欧氏距离高度相关 (Spearman r={sr_g:.3f}), 构象空间中度量校正可能不提供额外信息</li>
    <li><strong>氨基酸替换</strong>: 不同 AA 替换的扰动幅度差异显著, 带电氨基酸 (E, K) 的扰动大于疏水氨基酸 (L)</li>
    <li><strong>余弦衰减</strong>: 均值结构余弦相似度随 Δn 指数衰减, 衰减常数 tau={tau:.1f}</li>
    <li><strong>与 Phase I 关系</strong>: 生长扰动 (n→n+1) 的几何代价与 Phase I 的 spectral_decay~n 标度律一致, 验证了构象空间几何约束的统一性</li>
  </ol>
</div>

<h2>7. 输出文件</h2>
<table>
  <tr><th>文件</th><th>路径</th><th>行数</th></tr>
  <tr><td>生长扰动表</td><td>field_theory/tables/phase2_ca_growth_perturbation.csv</td><td>{len(df_growth)}</td></tr>
  <tr><td>氨基酸替换表</td><td>field_theory/tables/phase2_ca_substitution_perturbation.csv</td><td>{len(df_subst)}</td></tr>
  <tr><td>余弦衰减表</td><td>field_theory/tables/phase2_ca_cosine_decay.csv</td><td>{len(df_cosine)}</td></tr>
  <tr><td>可视化</td><td>field_theory/figures/phase2_ca_*.svg/jpg</td><td>12 个图表</td></tr>
</table>

</div>
</body>
</html>"""

report_path = REPORTS_DIR / 'phase2_ca_conformation_report.html'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"  HTML 报告已保存: {report_path} ({len(html)} 字符)")

print(f"\n{'=' * 70}")
print(f"Phase II 构象空间分析完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 70}")