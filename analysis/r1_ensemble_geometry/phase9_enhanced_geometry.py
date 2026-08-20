#!/usr/bin/env python3
"""
Phase IX 增强: P53有序域可视化 + DMS蛋白质有序域推广 + 增强几何量设计
======================================================================
三层任务:
  1. P53 全序列 vs 有序结构域对比可视化 (散点图 + 协方差谱)
  2. 有序域C_geo推广到 PTEN / HSP90 (含IDRs的蛋白质)
  3. 系统级增强几何量度量方案设计 (12种新指标)

蛋白质结构域定义:
  P53:   DBD 94-292, OD 323-356 (idr: 1-93, 293-322, 357-393)
  PTEN:  Phosphatase 14-185, C2 190-350 (idr: 1-13, 351-403)
  HSP90: N-domain 1-220, Middle 273-560 (idr: 221-272 linker, 561-709 C-term)

用法 (WSL2):
  source /home/liuchuanyang13/anaconda3/etc/profile.d/conda.sh
  conda activate bioemu
  cd /mnt/b/2026/Exploration/ProtGenesis2_Ensemble
  python field_theory/scripts/phase9_enhanced_geometry.py
"""

import sys, os, json, logging, time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
FIELD_THEORY = PROJECT_ROOT / "field_theory"
DMS_DIR = FIELD_THEORY / "data" / "dms"
BIOEMU_OUTPUT = DMS_DIR / "results" / "bioemu"
DMS_EXPANSION_DIR = DMS_DIR / "phase9_dms_expansion"
P53_ORDERED_DIR = DMS_DIR / "phase9_p53_ordered"
OUTPUT_DIR = DMS_DIR / "phase9_enhanced"
FIGURES_DIR = FIELD_THEORY / "figures"

for d in [OUTPUT_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('enhanced_geo')

# ================================================================
# 蛋白质结构域定义 (1-indexed)
# ================================================================
PROTEIN_DOMAINS = {
    'P53': {
        'full_length': 393,
        'ordered_ranges': [(94, 292), (323, 356)],  # DBD + OD
        'idr_ranges': [(1, 93), (293, 322), (357, 393)],
        'ordered_names': ['DBD', 'OD'],
        'idr_names': ['TAD+Pro-rich', 'Linker', 'REG'],
    },
    'PTEN': {
        'full_length': 403,
        'ordered_ranges': [(14, 350)],  # Phosphatase + C2 (conservative)
        'idr_ranges': [(1, 13), (351, 403)],
        'ordered_names': ['Phosphatase+C2'],
        'idr_names': ['N-tail', 'C-tail'],
    },
    'HSP90': {
        'full_length': 709,
        'ordered_ranges': [(1, 220), (273, 560)],  # N-domain + Middle
        'idr_ranges': [(221, 272), (561, 709)],
        'ordered_names': ['N-domain', 'Middle'],
        'idr_names': ['Charged-linker', 'C-terminal'],
    },
}

# 物化属性
AA_VOLUMES = {
    'A': 88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
    'G': 60.1, 'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
    'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
    'S': 89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6,
}
AA_CHARGES = {
    'R': 1, 'K': 1, 'D': -1, 'E': -1, 'H': 0.1,
    'A': 0, 'C': 0, 'F': 0, 'G': 0, 'I': 0, 'L': 0,
    'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'S': 0, 'T': 0,
    'V': 0, 'W': 0, 'Y': 0,
}


# ================================================================
# Part 1: 增强几何量度量
# ================================================================

def compute_enhanced_geometry(coords: np.ndarray) -> Dict:
    """
    增强几何量计算 — 12种新指标补充当前简单分析
    
    层级1: 基础统计 (已有)
    层级2: 局部涨落 (新增)
    层级3: 非高斯性 (新增)
    层级4: 拓扑特征 (新增)
    层级5: 信息几何 (新增)
    """
    if coords.ndim == 3:
        n_samples, n_residues, n_coords = coords.shape
    else:
        n_samples = coords.shape[0]
        n_residues = coords.shape[1] // 3
        coords = coords.reshape(n_samples, n_residues, 3)

    X_flat = coords.reshape(n_samples, n_residues * 3)
    mean = X_flat.mean(axis=0)
    centered = X_flat - mean
    cov = np.cov(centered, rowvar=False)
    D = cov.shape[0]

    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)
    total_var = eigenvalues.sum()
    normalized = eigenvalues / total_var if total_var > 0 else eigenvalues

    # === 层级1: 基础统计 (已有) ===
    PR = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum() if (eigenvalues ** 2).sum() > 0 else 0
    A_C = normalized[0]
    cumsum = np.cumsum(normalized)
    eff_rank_95 = int(np.searchsorted(cumsum, 0.95) + 1)
    eff_rank_99 = int(np.searchsorted(cumsum, 0.99) + 1)

    k = np.arange(1, min(51, len(eigenvalues) + 1))
    log_k, log_eig = np.log(k), np.log(eigenvalues[:len(k)] + 1e-10)
    A = np.vstack([log_k, np.ones(len(k))]).T
    alpha, _ = np.linalg.lstsq(A, log_eig, rcond=None)[0]
    spectral_decay = -alpha

    entropy = -np.sum(normalized * np.log(normalized + 1e-10))
    nonzero = eigenvalues[eigenvalues > 1e-10]
    log_det = np.log(nonzero).sum() if len(nonzero) > 0 else -np.inf
    pseudo_volume = np.exp(log_det / len(nonzero)) if len(nonzero) > 0 else 0

    # === 层级2: 局部涨落 (新增) ===
    # 2.1 逐残基RMSF (Root Mean Square Fluctuation)
    rmsf = np.zeros(n_residues)
    for i in range(n_residues):
        i0, i1 = i * 3, (i + 1) * 3
        rmsf[i] = np.sqrt(np.mean(np.sum((coords[:, i, :] - coords[:, i, :].mean(axis=0)) ** 2, axis=1)))
    mean_rmsf = float(np.mean(rmsf))
    max_rmsf = float(np.max(rmsf))
    rmsf_cv = float(np.std(rmsf) / mean_rmsf) if mean_rmsf > 0 else 0  # 涨落异质性

    # 2.2 局部刚度 (local stiffness) — 相邻残基涨落相关性
    local_stiffness = 0.0
    if n_residues >= 2:
        rmsf_diff = np.abs(np.diff(rmsf))
        local_stiffness = float(1.0 / (1.0 + np.mean(rmsf_diff)))

    # 2.3 涨落范围比 (fluctuation range ratio)
    fluct_range_ratio = float(np.percentile(rmsf, 90) / np.percentile(rmsf, 10)) if np.percentile(rmsf, 10) > 0 else 1.0

    # === 层级3: 非高斯性 (新增) ===
    # 3.1 多变量偏度 (Multivariate skewness via Mardia's test)
    inv_cov = np.linalg.pinv(cov + 1e-10 * np.eye(D))
    mardia_skewness = 0.0
    for i in range(n_samples):
        for j in range(n_samples):
            mardia_skewness += (centered[i] @ inv_cov @ centered[j]) ** 3
    mardia_skewness /= (n_samples ** 2)

    # 3.2 多变量峰度 (Mardia's kurtosis)
    mardia_kurtosis = 0.0
    for i in range(n_samples):
        mardia_kurtosis += (centered[i] @ inv_cov @ centered[i]) ** 2
    mardia_kurtosis /= n_samples

    # 3.3 PCA投影的非高斯性 (每个PC的skewness/kurtosis)
    pca = PCA(n_components=min(10, n_samples - 1))
    pc_scores = pca.fit_transform(centered)
    pc_skewness = np.zeros(min(10, pc_scores.shape[1]))
    pc_kurtosis = np.zeros(min(10, pc_scores.shape[1]))
    for i in range(len(pc_skewness)):
        pc_skewness[i] = np.mean((pc_scores[:, i] - pc_scores[:, i].mean()) ** 3) / (pc_scores[:, i].std() ** 3 + 1e-10)
        pc_kurtosis[i] = np.mean((pc_scores[:, i] - pc_scores[:, i].mean()) ** 4) / (pc_scores[:, i].std() ** 4 + 1e-10) - 3
    mean_pc_skewness = float(np.mean(np.abs(pc_skewness)))
    mean_pc_kurtosis = float(np.mean(pc_kurtosis))

    # === 层级4: 拓扑/几何特征 (新增) ===
    # 4.1 关联维数 (Correlation dimension via Grassberger-Procaccia)
    pairwise_dist = squareform(pdist(X_flat))
    if n_samples >= 20:
        eps_values = np.logspace(-2, 1, 20) * np.median(pairwise_dist[pairwise_dist > 0])
        C_eps = np.array([np.mean(pairwise_dist < eps) for eps in eps_values])
        valid = (C_eps > 0) & (C_eps < 1)
        if valid.sum() >= 3:
            log_eps = np.log(eps_values[valid])
            log_C = np.log(C_eps[valid])
            A = np.vstack([log_eps, np.ones(len(log_eps))]).T
            corr_dim, _ = np.linalg.lstsq(A, log_C, rcond=None)[0]
        else:
            corr_dim = np.nan
    else:
        corr_dim = np.nan

    # 4.2 局部密度方差 (ensemble heterogeneity)
    nn = NearestNeighbors(n_neighbors=min(10, n_samples - 1))
    nn.fit(X_flat)
    dist_to_knn, _ = nn.kneighbors(X_flat)
    mean_knn_dist = float(np.mean(dist_to_knn[:, -1]))
    cv_knn_dist = float(np.std(dist_to_knn[:, -1]) / mean_knn_dist) if mean_knn_dist > 0 else 0

    # 4.3 曲率代理 (通过Christoffel符号范数 — 从协方差矩阵的导数估计)
    # 简化为: 协方差矩阵的条件数作为局部曲率代理
    if len(eigenvalues) > 1:
        eig_max = eigenvalues[0]
        eig_min = eigenvalues[eigenvalues > 1e-10][-1] if np.any(eigenvalues > 1e-10) else 1e-10
        condition_number = float(eig_max / eig_min)
    else:
        condition_number = 1.0

    # 4.4 谱空隙 (spectral gap) — 最大特征值间隙
    eig_gaps = np.diff(eigenvalues[:min(20, len(eigenvalues))])
    spectral_gap = float(np.max(np.abs(eig_gaps))) if len(eig_gaps) > 0 else 0.0
    spectral_gap_ratio = float(spectral_gap / eigenvalues[0]) if eigenvalues[0] > 0 else 0.0

    # === 层级5: 信息几何 (新增) ===
    # 5.1 Fisher信息近似 (从协方差矩阵的逆)
    fisher_trace = float(np.trace(inv_cov))
    fisher_logdet = float(np.linalg.slogdet(cov + 1e-10 * np.eye(D))[1])

    # 5.2 互信息估计 (使用PCA前3个成分的离散化)
    if n_samples >= 10:
        pca3 = PCA(n_components=min(3, n_samples - 1))
        pc3 = pca3.fit_transform(centered)
        mi_matrix = np.zeros((pc3.shape[1], pc3.shape[1]))
        from sklearn.metrics import mutual_info_score
        for i in range(pc3.shape[1]):
            for j in range(i + 1, pc3.shape[1]):
                try:
                    qi = pd.qcut(pc3[:, i], q=min(5, n_samples), duplicates='drop', labels=False)
                    qj = pd.qcut(pc3[:, j], q=min(5, n_samples), duplicates='drop', labels=False)
                    mi_matrix[i, j] = mutual_info_score(qi, qj)
                    mi_matrix[j, i] = mi_matrix[i, j]
                except:
                    pass
        mean_mi = float(np.mean(mi_matrix[mi_matrix > 0])) if np.any(mi_matrix > 0) else 0.0
    else:
        mean_mi = 0.0

    return {
        # 层级1: 基础
        'n_samples': n_samples, 'n_residues': n_residues,
        'PR': float(PR), 'A_C': float(A_C),
        'eff_rank_95': eff_rank_95, 'eff_rank_99': eff_rank_99,
        'spectral_decay': float(spectral_decay),
        'entropy': float(entropy), 'total_variance': float(total_var),
        'pseudo_volume': float(pseudo_volume),
        # 层级2: 局部涨落
        'mean_rmsf': mean_rmsf, 'max_rmsf': max_rmsf,
        'rmsf_cv': rmsf_cv, 'local_stiffness': local_stiffness,
        'fluct_range_ratio': fluct_range_ratio,
        # 层级3: 非高斯性
        'mardia_skewness': float(mardia_skewness),
        'mardia_kurtosis': float(mardia_kurtosis),
        'mean_pc_skewness': mean_pc_skewness,
        'mean_pc_kurtosis': mean_pc_kurtosis,
        # 层级4: 拓扑
        'corr_dim': float(corr_dim) if not np.isnan(corr_dim) else np.nan,
        'mean_knn_dist': mean_knn_dist, 'cv_knn_dist': cv_knn_dist,
        'condition_number': condition_number,
        'spectral_gap': spectral_gap,
        'spectral_gap_ratio': spectral_gap_ratio,
        # 层级5: 信息几何
        'fisher_trace': fisher_trace, 'fisher_logdet': fisher_logdet,
        'mean_mi_pc3': mean_mi,
        # 原始数据
        'mean_pos': mean, 'cov_matrix': cov,
        'eigenvalues': eigenvalues, 'rmsf': rmsf,
    }


# ================================================================
# Part 2: 有序域C_geo推广
# ================================================================

def load_bioemu_ensemble(protein_name: str) -> Optional[np.ndarray]:
    """加载BioEmu NPZ WT系综"""
    npz_dir = BIOEMU_OUTPUT / f"{protein_name.lower()}_wt"
    if not npz_dir.exists():
        return None
    npz_files = sorted(npz_dir.glob("batch_*.npz"))
    all_pos = []
    for f in npz_files:
        try:
            data = np.load(f, allow_pickle=True)
            if 'pos' in data:
                all_pos.append(data['pos'])
        except:
            pass
    if not all_pos:
        return None
    positions = np.concatenate(all_pos, axis=0)
    if len(positions) > 250:
        positions = positions[:250]
    return positions


def extract_ordered_coords(coords: np.ndarray, ordered_ranges: List[Tuple[int, int]]) -> np.ndarray:
    """从全序列坐标中提取有序结构域坐标"""
    parts = []
    for start_0, end_0 in ordered_ranges:
        parts.append(coords[:, start_0:end_0, :])
    return np.concatenate(parts, axis=1)


def compute_cgeo_domain(single_mutants: pd.DataFrame, protein_name: str,
                         wt_geom: Dict, ordered_ranges: List[Tuple[int, int]],
                         seed: int = 42) -> pd.DataFrame:
    """有序域C_geo计算"""
    prot_df = single_mutants[single_mutants['protein'] == protein_name].copy()

    # 构建有序域位置映射
    ordered_positions = []
    for s, e in ordered_ranges:
        ordered_positions.extend(range(s, e + 1))
    pos_set = set(ordered_positions)

    # 过滤
    positions_orig = prot_df['position'].values.astype(int)
    mask = np.array([p in pos_set for p in positions_orig])
    prot_df = prot_df[mask].copy()

    if len(prot_df) < 10:
        return pd.DataFrame()

    # 重新映射位置
    pos_map = {p: i for i, p in enumerate(ordered_positions)}
    positions_remapped = np.array([pos_map[p] for p in prot_df['position'].values.astype(int)])

    n_ordered = len(ordered_positions)
    cov = wt_geom['cov_matrix']
    D = cov.shape[0]
    cov_blocks = np.zeros((n_ordered, 3, 3))
    for i in range(n_ordered):
        i0, i1 = i * 3, (i + 1) * 3
        if i1 <= D:
            cov_blocks[i] = cov[i0:i1, i0:i1]

    wt_aas = prot_df['wt_aa'].values
    mut_aas = prot_df['mut_aa'].values
    vol_diffs = np.array([AA_VOLUMES.get(m, 0) - AA_VOLUMES.get(w, 0) for w, m in zip(wt_aas, mut_aas)])
    chg_diffs = np.array([AA_CHARGES.get(m, 0) - AA_CHARGES.get(w, 0) for w, m in zip(wt_aas, mut_aas)])
    mags = 0.1 + np.abs(vol_diffs) / 100 + np.abs(chg_diffs) * 0.05

    np.random.seed(seed)
    directions = np.random.randn(len(prot_df), 3)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-10

    cgeo = np.zeros(len(prot_df))
    for i in range(len(prot_df)):
        pos = positions_remapped[i]
        if pos < 0 or pos >= n_ordered:
            continue
        cgeo[i] = directions[i] @ cov_blocks[pos] @ directions[i] * mags[i]

    return pd.DataFrame({
        'protein': f'{protein_name}_ordered',
        'mutant': prot_df['mutant'].values,
        'position_orig': prot_df['position'].values.astype(int),
        'wt_aa': wt_aas, 'mut_aa': mut_aas,
        'DMS_score': prot_df['DMS_score'].values,
        'C_geo_raw': cgeo,
    })


# ================================================================
# Part 3: P53 可视化
# ================================================================

def create_p53_comparison_visualization():
    """生成P53全序列 vs 有序结构域对比图"""
    logger.info("\n" + "=" * 60)
    logger.info("P53 全序列 vs 有序结构域 对比可视化")
    logger.info("=" * 60)

    # 加载数据
    cgeo_full = pd.read_csv(P53_ORDERED_DIR / "p53_full_cgeo_seed42.csv")
    cgeo_ordered = pd.read_csv(P53_ORDERED_DIR / "p53_ordered_cgeo_seed42.csv")

    ensemble = load_bioemu_ensemble("P53")
    if ensemble is None:
        logger.error("无法加载P53系综")
        return

    # 全序列几何
    geom_full = compute_enhanced_geometry(ensemble)
    # 有序域几何
    ordered_coords = extract_ordered_coords(ensemble, [(93, 292), (322, 356)])
    geom_ordered = compute_enhanced_geometry(ordered_coords)

    fig = plt.figure(figsize=(24, 18))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    colors = {'full': '#e74c3c', 'ordered': '#2ecc71'}
    markers = {'full': 'o', 'ordered': 's'}

    # --- A: C_geo~DMS 散点 (全序列) ---
    ax = fig.add_subplot(gs[0, 0])
    valid_f = cgeo_full.dropna(subset=['DMS_score', 'C_geo_raw']).sample(min(5000, len(cgeo_full)), random_state=42)
    ax.scatter(valid_f['C_geo_raw'], valid_f['DMS_score'], c=colors['full'], alpha=0.15, s=5, edgecolors='none')
    sr_f, sp_f = spearmanr(valid_f['C_geo_raw'], valid_f['DMS_score'])
    ax.set_title(f'Full P53 (393 aa)\nr={sr_f:.4f}, p={sp_f:.2e}', fontsize=11, fontweight='bold')
    ax.set_xlabel('C_geo', fontsize=9)
    ax.set_ylabel('DMS score', fontsize=9)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # --- B: C_geo~DMS 散点 (有序域) ---
    ax = fig.add_subplot(gs[0, 1])
    valid_o = cgeo_ordered.dropna(subset=['DMS_score', 'C_geo_raw']).sample(min(5000, len(cgeo_ordered)), random_state=42)
    ax.scatter(valid_o['C_geo_raw'], valid_o['DMS_score'], c=colors['ordered'], alpha=0.15, s=5, edgecolors='none')
    sr_o, sp_o = spearmanr(valid_o['C_geo_raw'], valid_o['DMS_score'])
    ax.set_title(f'Ordered DBD+OD (233 aa)\nr={sr_o:.4f}, p={sp_o:.2e}', fontsize=11, fontweight='bold')
    ax.set_xlabel('C_geo', fontsize=9)
    ax.set_ylabel('DMS score', fontsize=9)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # --- C: 协方差谱对比 ---
    ax = fig.add_subplot(gs[0, 2:])
    eig_f = geom_full['eigenvalues']
    eig_o = geom_ordered['eigenvalues']
    ranks_f = np.arange(1, min(51, len(eig_f) + 1))
    ranks_o = np.arange(1, min(51, len(eig_o) + 1))
    ax.loglog(ranks_f, eig_f[:len(ranks_f)], 'o-', color=colors['full'], markersize=3, linewidth=1.5, alpha=0.7, label=f'Full (PR={geom_full["PR"]:.1f})')
    ax.loglog(ranks_o, eig_o[:len(ranks_o)], 's-', color=colors['ordered'], markersize=3, linewidth=1.5, alpha=0.7, label=f'Ordered (PR={geom_ordered["PR"]:.1f})')
    ax.set_xlabel('Rank', fontsize=10)
    ax.set_ylabel('Eigenvalue', fontsize=10)
    ax.set_title('Covariance Spectrum: Full vs Ordered', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- D: RMSF profile ---
    ax = fig.add_subplot(gs[1, 0:2])
    rmsf_f = geom_full['rmsf']
    x = np.arange(1, len(rmsf_f) + 1)
    ax.fill_between(x, 0, rmsf_f, alpha=0.3, color='gray', label='RMSF')
    # 标记结构域
    domain_colors = {'TAD+Pro-rich': '#ff9999', 'DBD': '#99ff99', 'Linker': '#ffcc99', 'OD': '#99ff99', 'REG': '#ff9999'}
    for dname, (s, e) in [('TAD+Pro-rich', (1, 93)), ('DBD', (94, 292)), ('Linker', (293, 322)),
                            ('OD', (323, 356)), ('REG', (357, 393))]:
        clr = domain_colors.get(dname, 'gray')
        ax.axvspan(s, e, alpha=0.15, color=clr)
        ax.text((s + e) / 2, ax.get_ylim()[1] * 0.95, dname, ha='center', fontsize=7, rotation=90, va='top')
    ax.set_xlabel('Residue position', fontsize=10)
    ax.set_ylabel('RMSF (Å)', fontsize=10)
    ax.set_title('P53 RMSF Profile with Domain Annotation', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # --- E: 增强几何量雷达图 ---
    ax = fig.add_subplot(gs[1, 2:])
    metrics_radar = ['PR', 'A_C', 'eff_rank_95', 'spectral_decay', 'entropy', 'mean_rmsf',
                     'local_stiffness', 'corr_dim', 'rmsf_cv', 'mean_pc_kurtosis']
    values_f = []
    values_o = []
    labels = []
    for m in metrics_radar:
        vf = geom_full.get(m, 0)
        vo = geom_ordered.get(m, 0)
        if vf is None or np.isnan(vf):
            vf = 0
        if vo is None or np.isnan(vo):
            vo = 0
        values_f.append(vf)
        values_o.append(vo)
        labels.append(m)

    # 归一化
    all_vals = np.array(values_f + values_o)
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) > 0 and np.max(np.abs(all_vals)) > 0:
        vmax = np.max(np.abs(all_vals))
        values_f = [v / vmax if vmax > 0 else 0 for v in values_f]
        values_o = [v / vmax if vmax > 0 else 0 for v in values_o]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values_f += values_f[:1]
    values_o += values_o[:1]
    angles += angles[:1]
    ax.fill(angles, values_f, alpha=0.2, color=colors['full'])
    ax.plot(angles, values_f, 'o-', color=colors['full'], linewidth=1.5, markersize=3, label='Full')
    ax.fill(angles, values_o, alpha=0.2, color=colors['ordered'])
    ax.plot(angles, values_o, 's-', color=colors['ordered'], linewidth=1.5, markersize=3, label='Ordered')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_title('Enhanced Geometry Radar (normalized)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')

    # --- F: 全序列 C_geo 分布 ---
    ax = fig.add_subplot(gs[2, 0:2])
    ax.hist(valid_f['C_geo_raw'].clip(0, valid_f['C_geo_raw'].quantile(0.99)),
            bins=80, alpha=0.5, color=colors['full'], label='Full', density=True)
    ax.hist(valid_o['C_geo_raw'].clip(0, valid_o['C_geo_raw'].quantile(0.99)),
            bins=80, alpha=0.5, color=colors['ordered'], label='Ordered', density=True)
    ax.set_xlabel('C_geo', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('C_geo Distribution: Full vs Ordered', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- G: 按结构域 Spearman r 条形图 ---
    ax = fig.add_subplot(gs[2, 2:])
    domains_data = {
        'TAD1': (1, 42), 'TAD2': (43, 63), 'Pro-rich': (64, 92),
        'DBD': (94, 292), 'Linker': (293, 322), 'OD': (323, 356), 'REG': (357, 393)
    }
    domain_r = []
    domain_names = []
    domain_colors_list = []
    for dname, (s, e) in domains_data.items():
        df_d = cgeo_ordered[(cgeo_ordered['position_orig'] >= s) & (cgeo_ordered['position_orig'] <= e)]
        if len(df_d) >= 10:
            sr, sp = spearmanr(df_d['C_geo_raw'], df_d['DMS_score'])
            domain_r.append(sr)
            domain_names.append(dname)
            domain_colors_list.append('#2ecc71' if dname in ['DBD', 'OD'] else '#e74c3c')

    bars = ax.barh(domain_names, domain_r, color=domain_colors_list, alpha=0.7)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Spearman r (C_geo~DMS)', fontsize=10)
    ax.set_title('C_geo~DMS by Domain', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    # 添加p值标注
    for i, (dname, (s, e)) in enumerate(domains_data.items()):
        df_d = cgeo_ordered[(cgeo_ordered['position_orig'] >= s) & (cgeo_ordered['position_orig'] <= e)]
        if len(df_d) >= 10:
            _, sp = spearmanr(df_d['C_geo_raw'], df_d['DMS_score'])
            sig = '***' if sp < 0.001 else '**' if sp < 0.01 else '*' if sp < 0.05 else 'ns'
            ax.text(domain_r[i] + (0.01 if domain_r[i] >= 0 else -0.04), i, sig, va='center', fontsize=8)

    plt.suptitle('P53: Full Sequence vs Ordered Domains (DBD+OD) Comparison',
                 fontsize=14, fontweight='bold', y=0.98)
    for fmt in ['svg', 'jpg', 'png']:
        fig.savefig(FIGURES_DIR / f'phase9_p53_full_vs_ordered.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("  保存: phase9_p53_full_vs_ordered.svg/.jpg/.png")


# ================================================================
# Part 4: 有序域推广 + 增强几何量
# ================================================================

def apply_ordered_domains_to_all():
    """将有序域C_geo推广到所有有IDRs的蛋白质 + 增强几何量"""
    logger.info("\n" + "=" * 60)
    logger.info("有序域C_geo推广 + 增强几何量计算")
    logger.info("=" * 60)

    single_path = DMS_EXPANSION_DIR / "phase9_dms_single_mutants.csv"
    wt_path = DMS_EXPANSION_DIR / "wt_sequences.json"

    single_master = pd.read_csv(single_path)
    with open(wt_path) as f:
        wt_sequences = json.load(f)

    all_results = []
    all_geom = []

    for protein_name in ['P53', 'PTEN', 'HSP90']:
        if protein_name not in PROTEIN_DOMAINS:
            continue

        domain_info = PROTEIN_DOMAINS[protein_name]
        ordered_ranges_0idx = [(s - 1, e) for s, e in domain_info['ordered_ranges']]

        logger.info(f"\n--- {protein_name} ({domain_info['full_length']} aa) ---")

        # 加载系综
        ensemble = load_bioemu_ensemble(protein_name)
        if ensemble is None:
            logger.warning(f"  [{protein_name}] 无BioEmu数据")
            continue

        # 全序列几何
        geom_full = compute_enhanced_geometry(ensemble)
        geom_full['protein'] = f'{protein_name}_full'
        geom_full['domain_type'] = 'full'
        all_geom.append({k: v for k, v in geom_full.items()
                         if k not in ['mean_pos', 'cov_matrix', 'eigenvalues', 'rmsf']})

        # 有序域几何
        ordered_coords = extract_ordered_coords(ensemble, ordered_ranges_0idx)
        geom_ordered = compute_enhanced_geometry(ordered_coords)
        geom_ordered['protein'] = f'{protein_name}_ordered'
        geom_ordered['domain_type'] = 'ordered'
        all_geom.append({k: v for k, v in geom_ordered.items()
                         if k not in ['mean_pos', 'cov_matrix', 'eigenvalues', 'rmsf']})

        logger.info(f"  全序列: PR={geom_full['PR']:.2f}, A_C={geom_full['A_C']:.4f}, "
                    f"mean_rmsf={geom_full['mean_rmsf']:.3f}, "
                    f"corr_dim={geom_full.get('corr_dim', np.nan):.2f}, "
                    f"mardia_kurt={geom_full['mardia_kurtosis']:.1f}")
        logger.info(f"  有序域: PR={geom_ordered['PR']:.2f}, A_C={geom_ordered['A_C']:.4f}, "
                    f"mean_rmsf={geom_ordered['mean_rmsf']:.3f}, "
                    f"corr_dim={geom_ordered.get('corr_dim', np.nan):.2f}, "
                    f"mardia_kurt={geom_ordered['mardia_kurtosis']:.1f}")

        # C_geo 全序列
        wt_seq = wt_sequences[protein_name]
        cgeo_full = compute_cgeo_domain(single_master, protein_name, geom_full,
                                         [(0, domain_info['full_length'] - 1)], seed=42)
        # C_geo 有序域
        cgeo_ordered = compute_cgeo_domain(single_master, protein_name, geom_ordered,
                                            ordered_ranges_0idx, seed=42)

        for label, cgeo_df in [('full', cgeo_full), ('ordered', cgeo_ordered)]:
            if len(cgeo_df) < 10:
                continue
            valid = cgeo_df.dropna(subset=['DMS_score', 'C_geo_raw'])
            sr, sp = spearmanr(valid['C_geo_raw'], valid['DMS_score'])
            pr_val, pp_val = pearsonr(valid['C_geo_raw'], valid['DMS_score'])
            sig = "✅" if sp < 0.05 else "❌"

            all_results.append({
                'protein': protein_name,
                'domain_type': label,
                'n_residues': len(valid) if label == 'full' else ordered_coords.shape[1],
                'n_variants': len(valid),
                'spearman_r': sr, 'spearman_p': sp,
                'pearson_r': pr_val, 'pearson_p': pp_val,
                'significant': sp < 0.05,
            })
            logger.info(f"  {sig} {label:8s}: r={sr:+.4f} (p={sp:.2e}), n={len(valid)}")

    # 保存
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(OUTPUT_DIR / "ordered_domain_all_proteins.csv", index=False)

    df_geom = pd.DataFrame(all_geom)
    df_geom.to_csv(OUTPUT_DIR / "enhanced_geometry_all_proteins.csv", index=False)

    # 汇总
    logger.info("\n" + "=" * 60)
    logger.info("有序域 C_geo~DMS 汇总")
    logger.info("=" * 60)
    for protein_name in ['P53', 'PTEN', 'HSP90']:
        for dtype in ['full', 'ordered']:
            row = df_results[(df_results['protein'] == protein_name) & (df_results['domain_type'] == dtype)]
            if len(row) > 0:
                r = row.iloc[0]
                sig = "✅" if r['significant'] else "❌"
                logger.info(f"  {sig} {protein_name:6s} {dtype:8s}: r={r['spearman_r']:+.4f} (p={r['spearman_p']:.2e})")

    # 增强几何量对比
    logger.info("\n" + "=" * 60)
    logger.info("增强几何量对比 (全序列 vs 有序域)")
    logger.info("=" * 60)
    key_metrics = ['PR', 'A_C', 'eff_rank_95', 'spectral_decay', 'entropy',
                   'mean_rmsf', 'rmsf_cv', 'local_stiffness', 'fluct_range_ratio',
                   'mean_pc_kurtosis', 'corr_dim', 'condition_number',
                   'spectral_gap_ratio', 'mean_mi_pc3']
    for protein_name in ['P53', 'PTEN', 'HSP90']:
        logger.info(f"\n  {protein_name}:")
        logger.info(f"  {'Metric':25s} {'Full':>12s} {'Ordered':>12s} {'Δ':>12s}")
        for m in key_metrics:
            f_row = df_geom[(df_geom['protein'] == f'{protein_name}_full')]
            o_row = df_geom[(df_geom['protein'] == f'{protein_name}_ordered')]
            if len(f_row) > 0 and len(o_row) > 0 and m in f_row.columns:
                vf = f_row.iloc[0][m]
                vo = o_row.iloc[0][m]
                if pd.notna(vf) and pd.notna(vo):
                    delta = vo - vf
                    logger.info(f"  {m:25s} {vf:12.4f} {vo:12.4f} {delta:+12.4f}")

    return df_results, df_geom


# ================================================================
# Main
# ================================================================

def main():
    logger.info("=" * 70)
    logger.info("Phase IX 增强: P53可视化 + 有序域推广 + 增强几何量")
    logger.info("=" * 70)

    # Part 1: P53 可视化
    create_p53_comparison_visualization()

    # Part 2: 有序域推广 + 增强几何量
    df_results, df_geom = apply_ordered_domains_to_all()

    logger.info(f"\n所有结果保存在: {OUTPUT_DIR}")
    logger.info(f"可视化保存在: {FIGURES_DIR}")
    logger.info("完成!")

    return 0


if __name__ == "__main__":
    sys.exit(main())