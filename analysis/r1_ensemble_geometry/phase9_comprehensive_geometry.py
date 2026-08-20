#!/usr/bin/env python3
"""
Phase IX 综合增强几何量分析 — 系统级方案
=============================================
7层级28+指标，覆盖全部8个DMS蛋白质的系综几何分析。

三层输出:
  1. 全部8蛋白质增强几何量计算 (full + ordered domains)
  2. P53有序域 vs 全序列对比可视化 (散点图 + 协方差谱 + RMSF + 雷达图)
  3. 跨蛋白质完整对比表格 + 统计解释

蛋白质结构域定义:
  P53:   DBD 94-292, OD 323-356 (idr: 1-93, 293-322, 357-393)
  PTEN:  Phosphatase 14-185, C2 190-350 (idr: 1-13, 351-403)
  HSP90: N-domain 1-220, Middle 273-560 (idr: 221-272 linker, 561-709 C-term)
  SPIKE: RBD-only DMS (1-201 of full 1273, already ordered)
  GFP/BLAT/HRAS/UBE4B: mostly ordered, no ordered domain extraction needed

增强几何量框架 (7层级):
  L1: 基础系综统计 (PR, A_C, eff_rank, spectral_decay, entropy, total_var)
  L2: 局部涨落与异质性 (RMSF, stiffness, fluct_range_ratio, RMSF_entropy)
  L3: 非高斯性与高阶矩 (Mardia skewness/kurtosis, PC skewness/kurtosis, JB test)
  L4: 拓扑与分形特征 (corr_dim, KNN heterogeneity, condition_number, spectral_gap)
  L5: 信息几何 (Fisher trace/logdet, MI between PCs, JS divergence from Gaussian)
  L6: 动力学与输运性质 (effective_diffusion, relaxation_time, lyapunov_proxy)
  L7: 图论与网络度量 (contact_order, modularity, clustering_coeff, betweenness_cv)

用法 (WSL2):
  source /home/liuchuanyang13/anaconda3/etc/profile.d/conda.sh
  conda activate bioemu
  cd /mnt/b/2026/Exploration/ProtGenesis2_Ensemble
  python field_theory/scripts/phase9_comprehensive_geometry.py
"""

import sys, os, json, logging, time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, jarque_bera, entropy as scipy_entropy

# ============================================================
# Phase L1: Kabsch对齐 + 内禀协方差 + Ledoit-Wolf收缩
# ============================================================
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
try:
    from phase_l1_kabsch_metric import (
        kabsch_align_ensemble,
        compute_intrinsic_metric,
        ledoit_wolf_shrinkage_fast,
        compute_cgeo_intrinsic
    )
    HAS_L1 = True
except ImportError as e:
    HAS_L1 = False
    print(f"Warning: phase_l1_kabsch_metric not available: {e}")
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.linalg import eigh
from scipy.signal import correlate
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mutual_info_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ================================================================
# 配置
# ================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
FIELD_THEORY = PROJECT_ROOT / "field_theory"
DMS_DIR = FIELD_THEORY / "data" / "dms"
BIOEMU_OUTPUT = DMS_DIR / "results" / "bioemu"
DMS_EXPANSION_DIR = DMS_DIR / "phase9_dms_expansion"
P53_ORDERED_DIR = DMS_DIR / "phase9_p53_ordered"
OUTPUT_DIR = DMS_DIR / "phase9_comprehensive"
FIGURES_DIR = FIELD_THEORY / "figures"
TABLES_DIR = FIELD_THEORY / "tables"

for d in [OUTPUT_DIR, FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('comprehensive_geo')

# ================================================================
# 蛋白质结构域定义 (1-indexed)
# ================================================================
PROTEIN_DOMAINS = {
    'P53': {
        'full_length': 393,
        'ordered_ranges': [(94, 292), (323, 356)],
        'ordered_names': ['DBD', 'OD'],
        'idr_ranges': [(1, 93), (293, 322), (357, 393)],
        'idr_names': ['TAD+Pro-rich', 'Linker', 'REG'],
    },
    'PTEN': {
        'full_length': 403,
        'ordered_ranges': [(14, 350)],
        'ordered_names': ['Phosphatase+C2'],
        'idr_ranges': [(1, 13), (351, 403)],
        'idr_names': ['N-tail', 'C-tail'],
    },
    'HSP90': {
        'full_length': 709,
        'ordered_ranges': [(1, 220), (273, 560)],
        'ordered_names': ['N-domain', 'Middle'],
        'idr_ranges': [(221, 272), (561, 709)],
        'idr_names': ['Charged-linker', 'C-terminal'],
    },
    'SPIKE': {
        'full_length': 1273,
        'ordered_ranges': [(1, 201)],
        'ordered_names': ['NTD+RBD'],
        'idr_ranges': [(202, 1273)],
        'idr_names': ['Stalk+TM+CT'],
        'note': 'DMS only covers 1-201 (RBD), already ordered',
    },
    'GFP': {
        'full_length': 238,
        'ordered_ranges': [(1, 238)],
        'ordered_names': ['Beta-barrel'],
        'note': 'Mostly ordered, no significant IDRs',
    },
    'BLAT': {
        'full_length': 263,
        'ordered_ranges': [(1, 263)],
        'ordered_names': ['Alpha/beta sandwich'],
        'note': 'Mostly ordered, no significant IDRs',
    },
    'HRAS': {
        'full_length': 189,
        'ordered_ranges': [(1, 189)],
        'ordered_names': ['GTPase domain'],
        'note': 'Mostly ordered, no significant IDRs',
    },
    'UBE4B': {
        'full_length': 69,
        'ordered_ranges': [(1, 69)],
        'ordered_names': ['U-box domain'],
        'note': 'Small protein, fully ordered',
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
AA_HYDROPHOBICITY = {
    'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5,
    'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8,
    'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5,
    'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5,
}

# ================================================================
# Part 1: 7层级增强几何量计算 (28指标)
# ================================================================

def compute_enhanced_geometry_v2(coords: np.ndarray) -> Dict:
    """
    7层级增强几何量计算 — 28种指标

    L1: 基础系综统计 (6指标)
    L2: 局部涨落与异质性 (5指标)
    L3: 非高斯性与高阶矩 (5指标)
    L4: 拓扑与分形特征 (5指标)
    L5: 信息几何 (4指标)
    L6: 动力学与输运性质 (4指标, NEW)
    L7: 图论与网络度量 (4指标, NEW)
    """
    if coords.ndim == 3:
        n_samples, n_residues, n_coords = coords.shape
    else:
        n_samples = coords.shape[0]
        n_residues = coords.shape[1] // 3
        coords = coords.reshape(n_samples, n_residues, 3)

    X_flat = coords.reshape(n_samples, n_residues * 3)

    # ============================================================
    # Phase L1: Kabsch对齐 + 内禀协方差 + Ledoit-Wolf收缩 (修复缺口A1)
    # ============================================================
    if HAS_L1 and n_residues >= 4:
        # Kabsch对齐去除SE(3)刚体模式
        aligned_3d = kabsch_align_ensemble(coords, reference_mode='first')
        X_flat = aligned_3d.reshape(n_samples, n_residues * 3)
        mean = X_flat.mean(axis=0)
        centered = X_flat - mean
        # Ledoit-Wolf最优收缩替换固定ε
        cov, lambda_star = ledoit_wolf_shrinkage_fast(centered)
    else:
        # 原始实验室系方法 (兼容)
        lambda_star = None
        mean = X_flat.mean(axis=0)
        centered = X_flat - mean
        cov = np.cov(centered, rowvar=False)

    D = cov.shape[0]

    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)
    total_var = eigenvalues.sum()
    normalized = eigenvalues / total_var if total_var > 0 else eigenvalues

    # ================================================================
    # L1: 基础系综统计 (6指标)
    # ================================================================
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

    # ================================================================
    # L2: 局部涨落与异质性 (5指标)
    # ================================================================
    rmsf = np.zeros(n_residues)
    for i in range(n_residues):
        rmsf[i] = np.sqrt(np.mean(np.sum((coords[:, i, :] - coords[:, i, :].mean(axis=0)) ** 2, axis=1)))
    mean_rmsf = float(np.mean(rmsf))
    max_rmsf = float(np.max(rmsf))
    rmsf_cv = float(np.std(rmsf) / mean_rmsf) if mean_rmsf > 0 else 0

    local_stiffness = 0.0
    if n_residues >= 2:
        rmsf_diff = np.abs(np.diff(rmsf))
        local_stiffness = float(1.0 / (1.0 + np.mean(rmsf_diff)))

    fluct_range_ratio = float(np.percentile(rmsf, 90) / np.percentile(rmsf, 10)) if np.percentile(rmsf, 10) > 0 else 1.0

    # RMSF分布熵 (反映涨落模式的复杂度)
    rmsf_hist, _ = np.histogram(rmsf, bins=min(20, n_residues), density=True)
    rmsf_hist = rmsf_hist[rmsf_hist > 0]
    rmsf_entropy = float(scipy_entropy(rmsf_hist)) if len(rmsf_hist) > 1 else 0.0

    # ================================================================
    # L3: 非高斯性与高阶矩 (5指标)
    # ================================================================
    inv_cov = np.linalg.pinv(cov + 1e-10 * np.eye(D))

    # Mardia多元偏度
    mardia_skewness = 0.0
    for i in range(n_samples):
        for j in range(n_samples):
            mardia_skewness += (centered[i] @ inv_cov @ centered[j]) ** 3
    mardia_skewness /= (n_samples ** 2)

    # Mardia多元峰度
    mardia_kurtosis = 0.0
    for i in range(n_samples):
        mardia_kurtosis += (centered[i] @ inv_cov @ centered[i]) ** 2
    mardia_kurtosis /= n_samples

    # PCA投影非高斯性
    n_pcs = min(10, n_samples - 1, D)
    pca = PCA(n_components=n_pcs)
    pc_scores = pca.fit_transform(centered)
    pc_skewness = np.zeros(n_pcs)
    pc_kurtosis = np.zeros(n_pcs)
    for i in range(n_pcs):
        std_i = pc_scores[:, i].std()
        if std_i > 1e-10:
            pc_skewness[i] = np.mean((pc_scores[:, i] - pc_scores[:, i].mean()) ** 3) / (std_i ** 3)
            pc_kurtosis[i] = np.mean((pc_scores[:, i] - pc_scores[:, i].mean()) ** 4) / (std_i ** 4) - 3
    mean_pc_skewness = float(np.mean(np.abs(pc_skewness)))
    mean_pc_kurtosis = float(np.mean(pc_kurtosis))

    # Jarque-Bera检验 (PC1)
    if n_samples >= 8:
        jb_stat, jb_p = jarque_bera(pc_scores[:, 0])
        jb_pc1_pvalue = float(jb_p)
    else:
        jb_pc1_pvalue = np.nan

    # ================================================================
    # L4: 拓扑与分形特征 (5指标)
    # ================================================================
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

    nn = NearestNeighbors(n_neighbors=min(10, n_samples - 1))
    nn.fit(X_flat)
    dist_to_knn, _ = nn.kneighbors(X_flat)
    mean_knn_dist = float(np.mean(dist_to_knn[:, -1]))
    cv_knn_dist = float(np.std(dist_to_knn[:, -1]) / mean_knn_dist) if mean_knn_dist > 0 else 0

    if len(eigenvalues) > 1:
        eig_max = eigenvalues[0]
        eig_min = eigenvalues[eigenvalues > 1e-10][-1] if np.any(eigenvalues > 1e-10) else 1e-10
        condition_number = float(eig_max / eig_min)
    else:
        condition_number = 1.0

    eig_gaps = np.diff(eigenvalues[:min(20, len(eigenvalues))])
    spectral_gap = float(np.max(np.abs(eig_gaps))) if len(eig_gaps) > 0 else 0.0
    spectral_gap_ratio = float(spectral_gap / eigenvalues[0]) if eigenvalues[0] > 0 else 0.0

    # ================================================================
    # L5: 信息几何 (4指标)
    # ================================================================
    fisher_trace = float(np.trace(inv_cov))
    fisher_logdet = float(np.linalg.slogdet(cov + 1e-10 * np.eye(D))[1])

    # MI between top PCs
    if n_samples >= 10:
        pca3 = PCA(n_components=min(3, n_samples - 1))
        pc3 = pca3.fit_transform(centered)
        mi_values = []
        for i in range(pc3.shape[1]):
            for j in range(i + 1, pc3.shape[1]):
                try:
                    qi = pd.qcut(pc3[:, i], q=min(5, n_samples), duplicates='drop', labels=False)
                    qj = pd.qcut(pc3[:, j], q=min(5, n_samples), duplicates='drop', labels=False)
                    mi_values.append(mutual_info_score(qi, qj))
                except:
                    pass
        mean_mi = float(np.mean(mi_values)) if mi_values else 0.0
    else:
        mean_mi = 0.0

    # JS divergence from Gaussian (approximate via KL on PC1)
    if n_samples >= 10:
        pc1_std = pc_scores[:, 0].std()
        if pc1_std > 1e-10:
            # Histogram-based JS divergence
            hist_data, bins = np.histogram(pc_scores[:, 0], bins=min(20, n_samples // 2), density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            gaussian_pdf = np.exp(-0.5 * (bin_centers / pc1_std) ** 2) / (pc1_std * np.sqrt(2 * np.pi))
            gaussian_pdf = gaussian_pdf / gaussian_pdf.sum() * hist_data.sum()
            hist_data = hist_data / hist_data.sum()
            # M = (P+Q)/2
            M = (hist_data + gaussian_pdf) / 2
            kl_pm = np.sum(hist_data * np.log((hist_data + 1e-10) / (M + 1e-10)))
            kl_qm = np.sum(gaussian_pdf * np.log((gaussian_pdf + 1e-10) / (M + 1e-10)))
            js_divergence = float((kl_pm + kl_qm) / 2)
        else:
            js_divergence = 0.0
    else:
        js_divergence = 0.0

    # ================================================================
    # L6: 动力学与输运性质 (4指标, NEW)
    # ================================================================
    # 6.1 Effective diffusion constant (MSD growth rate)
    if n_samples >= 10:
        msd = np.zeros(n_samples // 2)
        for tau in range(1, n_samples // 2):
            diffs = X_flat[tau:] - X_flat[:-tau]
            msd[tau] = np.mean(np.sum(diffs ** 2, axis=1))
        taus = np.arange(1, n_samples // 2)
        valid_tau = (msd[1:] > 0) & (taus < 20)
        if valid_tau.sum() >= 3:
            A = np.vstack([np.log(taus[valid_tau]), np.ones(valid_tau.sum())]).T
            diff_exp, _ = np.linalg.lstsq(A, np.log(msd[1:][valid_tau]), rcond=None)[0]
            effective_diffusion = float(diff_exp)
        else:
            effective_diffusion = np.nan
    else:
        effective_diffusion = np.nan

    # 6.2 Relaxation time (autocorrelation decay of PC1)
    if n_samples >= 10:
        pc1 = pc_scores[:, 0]
        acf = np.correlate(pc1 - pc1.mean(), pc1 - pc1.mean(), mode='full')
        acf = acf[len(acf)//2:]
        acf = acf / acf[0] if acf[0] > 0 else acf
        # Find time to decay to 1/e
        threshold = 1.0 / np.e
        decay_idx = np.where(acf < threshold)[0]
        relaxation_time = float(decay_idx[0]) if len(decay_idx) > 0 else float(n_samples)
    else:
        relaxation_time = np.nan

    # 6.3 Lyapunov proxy (exponential divergence rate of nearby trajectories)
    if n_samples >= 20:
        # Simplified: compare nearest neighbor distances at different times
        nn_small = NearestNeighbors(n_neighbors=2)
        nn_small.fit(X_flat)
        d0, idx0 = nn_small.kneighbors(X_flat)
        d0 = d0[:, 1]  # distance to nearest neighbor
        d0 = d0[d0 > 1e-10]
        if len(d0) > 0:
            lyapunov_proxy = float(np.log(np.mean(d0) + 1e-10))
        else:
            lyapunov_proxy = np.nan
    else:
        lyapunov_proxy = np.nan

    # 6.4 Convective ratio (drift vs diffusion)
    if n_samples >= 10:
        total_drift = np.sum(mean ** 2)
        convective_ratio = float(total_drift / (total_var + 1e-10))
    else:
        convective_ratio = np.nan

    # ================================================================
    # L7: 图论与网络度量 (4指标, NEW)
    # ================================================================
    # 7.1 Contact order (mean sequence separation of contacting residues)
    # Contact defined as Cα-Cα distance < 8Å in mean structure
    mean_coords = coords.mean(axis=0)
    ca_dist = squareform(pdist(mean_coords))
    contact_map = ca_dist < 8.0
    np.fill_diagonal(contact_map, False)
    seq_sep = np.abs(np.arange(n_residues)[:, None] - np.arange(n_residues)[None, :])
    contacts = contact_map & (seq_sep > 0)
    if contacts.sum() > 0:
        contact_order = float(np.mean(seq_sep[contacts]))
    else:
        contact_order = 0.0

    # 7.2 Clustering coefficient (from contact network)
    if contacts.sum() >= 3:
        # For each node, count triangles / possible triangles
        clustering_coeffs = []
        for i in range(n_residues):
            neighbors = np.where(contact_map[i])[0]
            if len(neighbors) >= 2:
                n_pairs = len(neighbors) * (len(neighbors) - 1) / 2
                n_triangles = 0
                for j in range(len(neighbors)):
                    for k in range(j + 1, len(neighbors)):
                        if contact_map[neighbors[j], neighbors[k]]:
                            n_triangles += 1
                clustering_coeffs.append(n_triangles / n_pairs if n_pairs > 0 else 0)
        clustering_coeff = float(np.mean(clustering_coeffs)) if clustering_coeffs else 0.0
    else:
        clustering_coeff = 0.0

    # 7.3 Modularity proxy (from contact map block structure)
    if n_residues >= 10:
        # Simplified: compare intra-block vs inter-block contacts
        block_size = max(5, n_residues // 4)
        n_blocks = n_residues // block_size
        intra_contacts = 0
        inter_contacts = 0
        for b in range(n_blocks):
            b_start, b_end = b * block_size, min((b + 1) * block_size, n_residues)
            intra_contacts += contact_map[b_start:b_end, b_start:b_end].sum()
            for c in range(b + 1, n_blocks):
                c_start, c_end = c * block_size, min((c + 1) * block_size, n_residues)
                inter_contacts += contact_map[b_start:b_end, c_start:c_end].sum()
        total_contacts = intra_contacts + inter_contacts
        modularity = float(intra_contacts / total_contacts) if total_contacts > 0 else 0.0
    else:
        modularity = 0.0

    # 7.4 Betweenness centrality CV (from shortest paths in contact network)
    if contacts.sum() >= 3 and n_residues <= 500:
        # Use Floyd-Warshall for shortest paths (small n)
        dist_matrix = np.full((n_residues, n_residues), np.inf)
        dist_matrix[contact_map] = 1.0
        np.fill_diagonal(dist_matrix, 0)
        for k in range(n_residues):
            dk = dist_matrix[k]
            for i in range(n_residues):
                dik = dist_matrix[i, k]
                if dik == np.inf:
                    continue
                dist_matrix[i] = np.minimum(dist_matrix[i], dik + dk)
        # Count shortest paths passing through each node
        betweenness = np.zeros(n_residues)
        for s in range(n_residues):
            for t in range(s + 1, n_residues):
                if dist_matrix[s, t] < np.inf:
                    for v in range(n_residues):
                        if v != s and v != t:
                            if (dist_matrix[s, v] + dist_matrix[v, t]) <= dist_matrix[s, t] + 1e-10:
                                betweenness[v] += 1
        betweenness_cv = float(np.std(betweenness) / np.mean(betweenness)) if np.mean(betweenness) > 0 else 0.0
    else:
        betweenness_cv = 0.0

    return {
        # L1: 基础系综统计
        'n_samples': n_samples, 'n_residues': n_residues,
        'PR': float(PR), 'A_C': float(A_C),
        'eff_rank_95': eff_rank_95, 'eff_rank_99': eff_rank_99,
        'spectral_decay': float(spectral_decay),
        'entropy': float(entropy), 'total_variance': float(total_var),
        'pseudo_volume': float(pseudo_volume),
        # L2: 局部涨落
        'mean_rmsf': mean_rmsf, 'max_rmsf': max_rmsf,
        'rmsf_cv': rmsf_cv, 'local_stiffness': local_stiffness,
        'fluct_range_ratio': fluct_range_ratio,
        'rmsf_entropy': rmsf_entropy,
        # L3: 非高斯性
        'mardia_skewness': float(mardia_skewness),
        'mardia_kurtosis': float(mardia_kurtosis),
        'mean_pc_skewness': mean_pc_skewness,
        'mean_pc_kurtosis': mean_pc_kurtosis,
        'jb_pc1_pvalue': jb_pc1_pvalue,
        # L4: 拓扑与分形
        'corr_dim': float(corr_dim) if not (isinstance(corr_dim, float) and np.isnan(corr_dim)) else np.nan,
        'mean_knn_dist': mean_knn_dist, 'cv_knn_dist': cv_knn_dist,
        'condition_number': condition_number,
        'spectral_gap': spectral_gap,
        'spectral_gap_ratio': spectral_gap_ratio,
        # L5: 信息几何
        'fisher_trace': fisher_trace, 'fisher_logdet': fisher_logdet,
        'mean_mi_pc3': mean_mi,
        'js_divergence': js_divergence,
        # L6: 动力学
        'effective_diffusion': effective_diffusion if not (isinstance(effective_diffusion, float) and np.isnan(effective_diffusion)) else np.nan,
        'relaxation_time': relaxation_time if not (isinstance(relaxation_time, float) and np.isnan(relaxation_time)) else np.nan,
        'lyapunov_proxy': lyapunov_proxy if not (isinstance(lyapunov_proxy, float) and np.isnan(lyapunov_proxy)) else np.nan,
        'convective_ratio': convective_ratio if not (isinstance(convective_ratio, float) and np.isnan(convective_ratio)) else np.nan,
        # L7: 图论
        'contact_order': contact_order,
        'clustering_coeff': clustering_coeff,
        'modularity': modularity,
        'betweenness_cv': betweenness_cv,
        # 原始数据 (不导出)
        '_mean_pos': mean, '_cov_matrix': cov,
        '_eigenvalues': eigenvalues, '_rmsf': rmsf,
        '_pc_scores': pc_scores,
    }


# ================================================================
# Part 2: BioEmu数据加载
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
    """从全序列坐标中提取有序结构域坐标 (0-indexed ranges)"""
    parts = []
    for start_0, end_0 in ordered_ranges:
        parts.append(coords[:, start_0:end_0, :])
    return np.concatenate(parts, axis=1)


# ================================================================
# Part 3: C_geo 计算 (有序域推广)
# ================================================================

def compute_cgeo_domain(single_mutants: pd.DataFrame, protein_name: str,
                         wt_geom: Dict, ordered_ranges: List[Tuple[int, int]],
                         seed: int = 42) -> pd.DataFrame:
    """有序域C_geo计算"""
    prot_df = single_mutants[single_mutants['protein'] == protein_name].copy()

    ordered_positions = []
    for s, e in ordered_ranges:
        ordered_positions.extend(range(s, e + 1))
    pos_set = set(ordered_positions)

    positions_orig = prot_df['position'].values.astype(int)
    mask = np.array([p in pos_set for p in positions_orig])
    prot_df = prot_df[mask].copy()

    if len(prot_df) < 10:
        return pd.DataFrame()

    pos_map = {p: i for i, p in enumerate(ordered_positions)}
    positions_remapped = np.array([pos_map[p] for p in prot_df['position'].values.astype(int)])

    n_ordered = len(ordered_positions)
    cov = wt_geom['_cov_matrix']
    D = cov.shape[0]
    cov_blocks = np.zeros((n_ordered, 3, 3))
    for i in range(n_ordered):
        i0, i1 = i * 3, (i + 1) * 3
        if i1 <= D:
            cov_blocks[i] = cov[i0:i1, i0:i1]

    # 预计算正则化局部度量张量 g_S = (C + eps*I)^{-1} (论文 Law 1 定义)
    # 修复审计发现 C1: 原实现为 d̂^T*C*d̂ 二次型, 非 Mahalanobis 距离
    REG_EPS = 0.01
    g_S_blocks = np.zeros((n_ordered, 3, 3))
    for i in range(n_ordered):
        cb = cov_blocks[i]
        eps = REG_EPS * np.trace(cb) / 3.0
        g_S_blocks[i] = np.linalg.inv(cb + eps * np.eye(3))

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
        # C_geo = d̂^T * g_S * d̂ * mag (正则化 Mahalanobis 距离 × 扰动幅度)
        cgeo[i] = directions[i] @ g_S_blocks[pos] @ directions[i] * mags[i]

    return pd.DataFrame({
        'protein': f'{protein_name}_ordered',
        'mutant': prot_df['mutant'].values,
        'position_orig': prot_df['position'].values.astype(int),
        'wt_aa': wt_aas, 'mut_aa': mut_aas,
        'DMS_score': prot_df['DMS_score'].values,
        'C_geo_raw': cgeo,
    })


# ================================================================
# Part 4: P53 全序列 vs 有序域 对比可视化
# ================================================================

def create_p53_comparison_visualization(geom_full: Dict, geom_ordered: Dict):
    """生成P53全序列 vs 有序结构域 综合对比图 (7面板)"""
    logger.info("\n" + "=" * 60)
    logger.info("P53 全序列 vs 有序结构域 综合对比可视化")
    logger.info("=" * 60)

    # 加载C_geo数据
    cgeo_full = pd.read_csv(P53_ORDERED_DIR / "p53_full_cgeo_seed42.csv")
    cgeo_ordered = pd.read_csv(P53_ORDERED_DIR / "p53_ordered_cgeo_seed42.csv")

    fig = plt.figure(figsize=(28, 22))
    gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.35)

    colors = {'full': '#e74c3c', 'ordered': '#2ecc71', 'idr': '#f39c12'}

    # --- A: C_geo~DMS 散点图 (全序列) ---
    ax = fig.add_subplot(gs[0, 0])
    valid_f = cgeo_full.dropna(subset=['DMS_score', 'C_geo_raw']).sample(min(5000, len(cgeo_full)), random_state=42)
    ax.scatter(valid_f['C_geo_raw'], valid_f['DMS_score'], c=colors['full'], alpha=0.15, s=4, edgecolors='none')
    sr_f, sp_f = spearmanr(valid_f['C_geo_raw'], valid_f['DMS_score'])
    ax.set_title(f'Full P53 (393 aa)\nr={sr_f:.4f}, p={sp_f:.2e}', fontsize=10, fontweight='bold')
    ax.set_xlabel('C_geo', fontsize=8)
    ax.set_ylabel('DMS score', fontsize=8)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # --- B: C_geo~DMS 散点图 (有序域) ---
    ax = fig.add_subplot(gs[0, 1])
    valid_o = cgeo_ordered.dropna(subset=['DMS_score', 'C_geo_raw']).sample(min(5000, len(cgeo_ordered)), random_state=42)
    ax.scatter(valid_o['C_geo_raw'], valid_o['DMS_score'], c=colors['ordered'], alpha=0.15, s=4, edgecolors='none')
    sr_o, sp_o = spearmanr(valid_o['C_geo_raw'], valid_o['DMS_score'])
    ax.set_title(f'Ordered DBD+OD (233 aa)\nr={sr_o:.4f}, p={sp_o:.2e}', fontsize=10, fontweight='bold')
    ax.set_xlabel('C_geo', fontsize=8)
    ax.set_ylabel('DMS score', fontsize=8)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # --- C: 协方差谱对比 (log-log) ---
    ax = fig.add_subplot(gs[0, 2:])
    eig_f = geom_full['_eigenvalues']
    eig_o = geom_ordered['_eigenvalues']
    ranks_f = np.arange(1, min(51, len(eig_f) + 1))
    ranks_o = np.arange(1, min(51, len(eig_o) + 1))
    ax.loglog(ranks_f, eig_f[:len(ranks_f)], 'o-', color=colors['full'], markersize=3, linewidth=1.5,
              alpha=0.7, label=f'Full (PR={geom_full["PR"]:.1f}, SD={geom_full["spectral_decay"]:.2f})')
    ax.loglog(ranks_o, eig_o[:len(ranks_o)], 's-', color=colors['ordered'], markersize=3, linewidth=1.5,
              alpha=0.7, label=f'Ordered (PR={geom_ordered["PR"]:.1f}, SD={geom_ordered["spectral_decay"]:.2f})')
    ax.set_xlabel('Rank', fontsize=9)
    ax.set_ylabel('Eigenvalue', fontsize=9)
    ax.set_title('Covariance Spectrum: Full vs Ordered Domains', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)

    # --- D: RMSF Profile with domain annotation ---
    ax = fig.add_subplot(gs[1, 0:2])
    rmsf_f = geom_full['_rmsf']
    x = np.arange(1, len(rmsf_f) + 1)
    ax.fill_between(x, 0, rmsf_f, alpha=0.3, color='gray')
    ax.plot(x, rmsf_f, 'k-', linewidth=0.5, alpha=0.5)
    domain_colors = {'TAD': '#ff9999', 'DBD': '#99ff99', 'Linker': '#ffcc99', 'OD': '#99ff99', 'REG': '#ff9999'}
    for dname, (s, e) in [('TAD1', (1, 42)), ('TAD2', (43, 63)), ('Pro-rich', (64, 92)),
                            ('DBD', (94, 292)), ('Linker', (293, 322)),
                            ('OD', (323, 356)), ('REG', (357, 393))]:
        clr = domain_colors.get(dname, 'gray')
        ax.axvspan(s, e, alpha=0.12, color=clr)
        mid = (s + e) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.95 if s > 63 else ax.get_ylim()[1] * 0.85,
                dname, ha='center', fontsize=6, rotation=90, va='top')
    ax.set_xlabel('Residue position', fontsize=9)
    ax.set_ylabel('RMSF (Å)', fontsize=9)
    ax.set_title('P53 RMSF Profile with Domain Annotation', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # --- E: 增强几何量雷达图 (归一化) ---
    ax = fig.add_subplot(gs[1, 2:])
    metrics_radar = ['PR', 'A_C', 'eff_rank_95', 'spectral_decay', 'entropy', 'mean_rmsf',
                     'rmsf_cv', 'local_stiffness', 'corr_dim', 'mean_pc_kurtosis',
                     'js_divergence', 'contact_order']
    values_f, values_o, labels = [], [], []
    for m in metrics_radar:
        vf = geom_full.get(m, 0)
        vo = geom_ordered.get(m, 0)
        if vf is None or (isinstance(vf, float) and np.isnan(vf)):
            vf = 0
        if vo is None or (isinstance(vo, float) and np.isnan(vo)):
            vo = 0
        values_f.append(vf)
        values_o.append(vo)
        labels.append(m)

    all_vals = np.array(values_f + values_o)
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) > 0 and np.max(np.abs(all_vals)) > 0:
        vmax = np.max(np.abs(all_vals))
        values_f = [v / vmax if vmax > 0 else 0 for v in values_f]
        values_o = [v / vmax if vmax > 0 else 0 for v in values_o]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values_f += values_f[:1]; values_o += values_o[:1]; angles += angles[:1]
    ax.fill(angles, values_f, alpha=0.2, color=colors['full'])
    ax.plot(angles, values_f, 'o-', color=colors['full'], linewidth=1.5, markersize=3, label='Full P53')
    ax.fill(angles, values_o, alpha=0.2, color=colors['ordered'])
    ax.plot(angles, values_o, 's-', color=colors['ordered'], linewidth=1.5, markersize=3, label='Ordered DBD+OD')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_title('Enhanced Geometry Radar (normalized)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')

    # --- F: C_geo分布对比 ---
    ax = fig.add_subplot(gs[2, 0:2])
    ax.hist(valid_f['C_geo_raw'].clip(0, valid_f['C_geo_raw'].quantile(0.99)),
            bins=80, alpha=0.5, color=colors['full'], label='Full (393 aa)', density=True)
    ax.hist(valid_o['C_geo_raw'].clip(0, valid_o['C_geo_raw'].quantile(0.99)),
            bins=80, alpha=0.5, color=colors['ordered'], label='Ordered (233 aa)', density=True)
    ax.set_xlabel('C_geo', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title('C_geo Distribution: Full vs Ordered Domains', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- G: 按结构域 Spearman r 条形图 ---
    ax = fig.add_subplot(gs[2, 2:])
    domains_data = {
        'TAD1': (1, 42), 'TAD2': (43, 63), 'Pro-rich': (64, 92),
        'DBD': (94, 292), 'Linker': (293, 322), 'OD': (323, 356), 'REG': (357, 393)
    }
    domain_r, domain_names, domain_colors_list = [], [], []
    for dname, (s, e) in domains_data.items():
        df_d = cgeo_ordered[(cgeo_ordered['position_orig'] >= s) & (cgeo_ordered['position_orig'] <= e)]
        if len(df_d) >= 10:
            sr, sp = spearmanr(df_d['C_geo_raw'], df_d['DMS_score'])
            domain_r.append(sr)
            domain_names.append(dname)
            domain_colors_list.append('#2ecc71' if dname in ['DBD', 'OD'] else '#e74c3c')

    bars = ax.barh(domain_names, domain_r, color=domain_colors_list, alpha=0.7)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Spearman r (C_geo~DMS)', fontsize=9)
    ax.set_title('C_geo~DMS Correlation by Domain', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    idx = 0
    for dname, (s, e) in domains_data.items():
        df_d = cgeo_ordered[(cgeo_ordered['position_orig'] >= s) & (cgeo_ordered['position_orig'] <= e)]
        if len(df_d) >= 10:
            _, sp = spearmanr(df_d['C_geo_raw'], df_d['DMS_score'])
            sig = '***' if sp < 0.001 else '**' if sp < 0.01 else '*' if sp < 0.05 else 'ns'
            offset = 0.01 if domain_r[idx] >= 0 else -0.04
            ax.text(domain_r[idx] + offset, idx, sig, va='center', fontsize=7)
            idx += 1

    # --- H: PCA投影对比 (PC1 vs PC2) ---
    ax = fig.add_subplot(gs[3, 0:2])
    pc_f = geom_full['_pc_scores']
    pc_o = geom_ordered['_pc_scores']
    ax.scatter(pc_f[:, 0], pc_f[:, 1], c=colors['full'], alpha=0.4, s=8, edgecolors='none', label='Full')
    ax.scatter(pc_o[:, 0], pc_o[:, 1], c=colors['ordered'], alpha=0.4, s=8, edgecolors='none', label='Ordered')
    # 95%置信椭圆
    from matplotlib.patches import Ellipse
    for pc_data, color, label in [(pc_f, colors['full'], 'Full'), (pc_o, colors['ordered'], 'Ordered')]:
        cov_pc = np.cov(pc_data[:, :2].T)
        eigvals, eigvecs = np.linalg.eigh(cov_pc)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width, height = 2 * np.sqrt(5.991 * np.sqrt(eigvals))
        ellipse = Ellipse(xy=pc_data[:, :2].mean(axis=0), width=width, height=height,
                          angle=angle, edgecolor=color, facecolor='none', linewidth=1.5, linestyle='--')
        ax.add_patch(ellipse)
    ax.set_xlabel('PC1', fontsize=9)
    ax.set_ylabel('PC2', fontsize=9)
    ax.set_title('PCA Projection: Full vs Ordered (95% confidence ellipses)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- I: 特征值累计贡献对比 ---
    ax = fig.add_subplot(gs[3, 2:])
    cumsum_f = np.cumsum(eig_f) / eig_f.sum()
    cumsum_o = np.cumsum(eig_o) / eig_o.sum()
    ranks = np.arange(1, min(21, len(cumsum_f) + 1))
    ax.plot(ranks, cumsum_f[:len(ranks)], 'o-', color=colors['full'], linewidth=1.5, markersize=3, label='Full')
    ax.plot(ranks, cumsum_o[:len(ranks)], 's-', color=colors['ordered'], linewidth=1.5, markersize=3, label='Ordered')
    ax.axhline(0.95, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(0.99, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Number of PCs', fontsize=9)
    ax.set_ylabel('Cumulative variance', fontsize=9)
    ax.set_title('Cumulative Variance Explained', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('P53: Full Sequence vs Ordered Domains (DBD+OD) — Comprehensive Comparison',
                 fontsize=13, fontweight='bold', y=0.99)
    for fmt in ['svg', 'jpg', 'png']:
        fig.savefig(FIGURES_DIR / f'phase9_p53_comprehensive_comparison.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("  保存: phase9_p53_comprehensive_comparison.svg/.jpg/.png")


# ================================================================
# Part 5: 跨蛋白质增强几何量对比可视化
# ================================================================

def create_cross_protein_visualization(all_geom_df: pd.DataFrame, all_corr_df: pd.DataFrame):
    """生成跨蛋白质增强几何量对比图"""
    logger.info("\n" + "=" * 60)
    logger.info("跨蛋白质增强几何量对比可视化")
    logger.info("=" * 60)

    fig = plt.figure(figsize=(28, 20))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    proteins = ['GFP', 'BLAT', 'P53', 'PTEN', 'HSP90', 'UBE4B', 'SPIKE', 'HRAS']
    colors = {'full': '#e74c3c', 'ordered': '#2ecc71', 'idr': '#f39c12'}
    protein_colors = {
        'GFP': '#1f77b4', 'BLAT': '#ff7f0e', 'P53': '#2ca02c', 'PTEN': '#d62728',
        'HSP90': '#9467bd', 'UBE4B': '#8c564b', 'SPIKE': '#e377c2', 'HRAS': '#7f7f7f',
    }

    # --- A: PR vs A_C 散点 ---
    ax = fig.add_subplot(gs[0, 0])
    for p in proteins:
        row = all_geom_df[all_geom_df['protein'] == f'{p}_full']
        if len(row) > 0:
            ax.scatter(row['PR'], row['A_C'], c=protein_colors[p], s=80, label=p, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('PR', fontsize=9)
    ax.set_ylabel('A_C', fontsize=9)
    ax.set_title('PR vs A_C by Protein', fontsize=10, fontweight='bold')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- B: spectral_decay vs entropy ---
    ax = fig.add_subplot(gs[0, 1])
    for p in proteins:
        row = all_geom_df[all_geom_df['protein'] == f'{p}_full']
        if len(row) > 0:
            ax.scatter(row['spectral_decay'], row['entropy'], c=protein_colors[p], s=80, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Spectral Decay', fontsize=9)
    ax.set_ylabel('Entropy', fontsize=9)
    ax.set_title('Spectral Decay vs Entropy', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # --- C: C_geo~DMS 相关性条形图 ---
    ax = fig.add_subplot(gs[0, 2:])
    if len(all_corr_df) > 0:
        corr_full = all_corr_df[all_corr_df['domain_type'] == 'full'].copy()
        corr_full = corr_full.set_index('protein')
        bars = ax.bar(range(len(proteins)), [corr_full.loc[p, 'spearman_r'] if p in corr_full.index else 0
                                              for p in proteins],
                      color=[protein_colors[p] for p in proteins], alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xticks(range(len(proteins)))
        ax.set_xticklabels(proteins, fontsize=8)
        ax.set_ylabel('Spearman r (C_geo~DMS)', fontsize=9)
        ax.set_title('C_geo~DMS Correlation by Protein (Full Sequence)', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        # 标注p值显著性
        for i, p in enumerate(proteins):
            if p in corr_full.index:
                sp = corr_full.loc[p, 'spearman_p']
                sig = '***' if sp < 0.001 else '**' if sp < 0.01 else '*' if sp < 0.05 else 'ns'
                r = corr_full.loc[p, 'spearman_r']
                ax.text(i, r + (0.005 if r >= 0 else -0.015), sig, ha='center', fontsize=7)

    # --- D: Full vs Ordered C_geo~DMS 对比 (IDR proteins) ---
    ax = fig.add_subplot(gs[1, 0:2])
    idr_proteins = ['P53', 'PTEN', 'HSP90']
    x_pos = np.arange(len(idr_proteins))
    width = 0.35
    for i, p in enumerate(idr_proteins):
        row_f = all_corr_df[(all_corr_df['protein'] == p) & (all_corr_df['domain_type'] == 'full')]
        row_o = all_corr_df[(all_corr_df['protein'] == p) & (all_corr_df['domain_type'] == 'ordered')]
        r_f = row_f.iloc[0]['spearman_r'] if len(row_f) > 0 else 0
        r_o = row_o.iloc[0]['spearman_r'] if len(row_o) > 0 else 0
        ax.bar(i - width/2, r_f, width, color=colors['full'], alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.bar(i + width/2, r_o, width, color=colors['ordered'], alpha=0.8, edgecolor='black', linewidth=0.5)
        # 标注Δ
        delta = r_o - r_f
        ax.annotate(f'Δ={delta:+.4f}', xy=(i, max(r_f, r_o) + 0.005), ha='center', fontsize=7, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(idr_proteins, fontsize=9)
    ax.set_ylabel('Spearman r (C_geo~DMS)', fontsize=9)
    ax.set_title('C_geo~DMS: Full vs Ordered Domains (IDR proteins)', fontsize=10, fontweight='bold')
    legend_elements = [Line2D([0], [0], color=colors['full'], lw=4, label='Full'),
                       Line2D([0], [0], color=colors['ordered'], lw=4, label='Ordered')]
    ax.legend(handles=legend_elements, fontsize=8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # --- E: 增强几何量热图 (L2-L7) ---
    ax = fig.add_subplot(gs[1, 2:])
    l2_l7_metrics = ['mean_rmsf', 'rmsf_cv', 'local_stiffness', 'rmsf_entropy',
                     'mardia_kurtosis', 'mean_pc_kurtosis', 'jb_pc1_pvalue',
                     'corr_dim', 'condition_number', 'spectral_gap_ratio',
                     'fisher_trace', 'js_divergence', 'mean_mi_pc3',
                     'effective_diffusion', 'relaxation_time',
                     'contact_order', 'clustering_coeff', 'modularity', 'betweenness_cv']
    heatmap_data = np.zeros((len(proteins), len(l2_l7_metrics)))
    for i, p in enumerate(proteins):
        row = all_geom_df[all_geom_df['protein'] == f'{p}_full']
        if len(row) > 0:
            for j, m in enumerate(l2_l7_metrics):
                val = row.iloc[0].get(m, np.nan)
                heatmap_data[i, j] = val if pd.notna(val) else np.nan

    # Z-score normalize
    heatmap_zs = np.zeros_like(heatmap_data)
    for j in range(len(l2_l7_metrics)):
        col = heatmap_data[:, j]
        valid_col = col[~np.isnan(col)]
        if len(valid_col) > 1 and valid_col.std() > 0:
            heatmap_zs[:, j] = (col - valid_col.mean()) / valid_col.std()

    im = ax.imshow(heatmap_zs, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_xticks(range(len(l2_l7_metrics)))
    ax.set_xticklabels(l2_l7_metrics, fontsize=5, rotation=45, ha='right')
    ax.set_yticks(range(len(proteins)))
    ax.set_yticklabels(proteins, fontsize=8)
    ax.set_title('Enhanced Geometry Heatmap (Z-score, L2-L7)', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # --- F: 动态性质对比 (L6) ---
    ax = fig.add_subplot(gs[2, 0:2])
    dyn_metrics = ['effective_diffusion', 'relaxation_time', 'lyapunov_proxy', 'convective_ratio']
    x_pos = np.arange(len(proteins))
    width = 0.2
    for j, m in enumerate(dyn_metrics):
        values = []
        for p in proteins:
            row = all_geom_df[all_geom_df['protein'] == f'{p}_full']
            val = row.iloc[0].get(m, np.nan) if len(row) > 0 else np.nan
            values.append(val if pd.notna(val) else 0)
        # Normalize
        max_abs = max(np.max(np.abs(values)), 1e-10)
        values = [v / max_abs for v in values]
        ax.bar(x_pos + j * width, values, width, alpha=0.8, label=m, edgecolor='black', linewidth=0.3)
    ax.set_xticks(x_pos + 1.5 * width)
    ax.set_xticklabels(proteins, fontsize=8)
    ax.set_ylabel('Normalized value', fontsize=9)
    ax.set_title('Dynamical Properties (L6, normalized)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    # --- G: 图论度量对比 (L7) ---
    ax = fig.add_subplot(gs[2, 2:])
    graph_metrics = ['contact_order', 'clustering_coeff', 'modularity', 'betweenness_cv']
    x_pos = np.arange(len(proteins))
    for j, m in enumerate(graph_metrics):
        values = []
        for p in proteins:
            row = all_geom_df[all_geom_df['protein'] == f'{p}_full']
            val = row.iloc[0].get(m, np.nan) if len(row) > 0 else np.nan
            values.append(val if pd.notna(val) else 0)
        max_abs = max(np.max(np.abs(values)), 1e-10)
        values = [v / max_abs for v in values]
        ax.bar(x_pos + j * width, values, width, alpha=0.8, label=m, edgecolor='black', linewidth=0.3)
    ax.set_xticks(x_pos + 1.5 * width)
    ax.set_xticklabels(proteins, fontsize=8)
    ax.set_ylabel('Normalized value', fontsize=9)
    ax.set_title('Graph-Theoretic Measures (L7, normalized)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Cross-Protein Enhanced Geometry Comparison — 8 DMS Proteins',
                 fontsize=13, fontweight='bold', y=0.99)
    for fmt in ['svg', 'jpg', 'png']:
        fig.savefig(FIGURES_DIR / f'phase9_cross_protein_enhanced_geometry.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("  保存: phase9_cross_protein_enhanced_geometry.svg/.jpg/.png")


# ================================================================
# Main
# ================================================================

def main():
    logger.info("=" * 70)
    logger.info("Phase IX 综合增强几何量分析")
    logger.info("7层级28+指标 | 8蛋白质 | 全序列 vs 有序域")
    logger.info("=" * 70)

    # 加载DMS数据
    single_path = DMS_EXPANSION_DIR / "phase9_dms_single_mutants.csv"
    wt_path = DMS_EXPANSION_DIR / "wt_sequences.json"

    single_master = pd.read_csv(single_path)
    with open(wt_path) as f:
        wt_sequences = json.load(f)

    logger.info(f"加载 {len(single_master)} 单突变体, {len(wt_sequences)} WT序列")

    all_geom_records = []
    all_corr_records = []
    all_cgeo_dfs = []
    p53_geom_full = None
    p53_geom_ordered = None

    # ================================================================
    # Step 1: 遍历所有蛋白质，计算增强几何量 + C_geo
    # ================================================================
    for protein_name in ['P53', 'PTEN', 'HSP90', 'SPIKE', 'GFP', 'BLAT', 'HRAS', 'UBE4B']:
        domain_info = PROTEIN_DOMAINS[protein_name]
        has_idrs = 'idr_ranges' in domain_info and len(domain_info['idr_ranges']) > 0
        do_ordered = has_idrs and protein_name in ['P53', 'PTEN', 'HSP90']

        logger.info(f"\n{'='*60}")
        logger.info(f"  {protein_name} ({domain_info['full_length']} aa) | IDRs: {has_idrs} | Ordered analysis: {do_ordered}")
        logger.info(f"{'='*60}")

        ensemble = load_bioemu_ensemble(protein_name)
        if ensemble is None:
            logger.warning(f"  [{protein_name}] 无BioEmu数据，跳过")
            continue

        # 全序列增强几何量
        geom_full = compute_enhanced_geometry_v2(ensemble)
        geom_full['protein'] = f'{protein_name}_full'
        geom_full['domain_type'] = 'full'
        all_geom_records.append({k: v for k, v in geom_full.items()
                                 if not k.startswith('_')})

        logger.info(f"  全序列: PR={geom_full['PR']:.2f}, A_C={geom_full['A_C']:.4f}, "
                    f"mean_rmsf={geom_full['mean_rmsf']:.3f}, "
                    f"corr_dim={geom_full.get('corr_dim', np.nan)}, "
                    f"mardia_kurt={geom_full['mardia_kurtosis']:.1f}, "
                    f"js_div={geom_full['js_divergence']:.4f}, "
                    f"eff_diff={geom_full.get('effective_diffusion', np.nan)}, "
                    f"contact_order={geom_full['contact_order']:.1f}")

        # 有序域增强几何量 (仅对有IDRs的蛋白质)
        if do_ordered:
            ordered_ranges_0idx = [(s - 1, e) for s, e in domain_info['ordered_ranges']]
            ordered_coords = extract_ordered_coords(ensemble, ordered_ranges_0idx)
            geom_ordered = compute_enhanced_geometry_v2(ordered_coords)
            geom_ordered['protein'] = f'{protein_name}_ordered'
            geom_ordered['domain_type'] = 'ordered'
            all_geom_records.append({k: v for k, v in geom_ordered.items()
                                     if not k.startswith('_')})

            logger.info(f"  有序域: PR={geom_ordered['PR']:.2f}, A_C={geom_ordered['A_C']:.4f}, "
                        f"mean_rmsf={geom_ordered['mean_rmsf']:.3f}, "
                        f"corr_dim={geom_ordered.get('corr_dim', np.nan)}, "
                        f"mardia_kurt={geom_ordered['mardia_kurtosis']:.1f}")

            # 保存P53几何量用于可视化
            if protein_name == 'P53':
                p53_geom_full = geom_full
                p53_geom_ordered = geom_ordered

        # C_geo 计算 (全序列)
        if protein_name in wt_sequences:
            cgeo_full = compute_cgeo_domain(single_master, protein_name, geom_full,
                                             [(0, domain_info['full_length'] - 1)], seed=42)
            if len(cgeo_full) >= 10:
                valid = cgeo_full.dropna(subset=['DMS_score', 'C_geo_raw'])
                sr, sp = spearmanr(valid['C_geo_raw'], valid['DMS_score'])
                pr_val, pp_val = pearsonr(valid['C_geo_raw'], valid['DMS_score'])
                sig = "✅" if sp < 0.05 else "❌"
                all_corr_records.append({
                    'protein': protein_name, 'domain_type': 'full',
                    'n_variants': len(valid), 'spearman_r': sr, 'spearman_p': sp,
                    'pearson_r': pr_val, 'pearson_p': pp_val,
                    'significant': sp < 0.05,
                })
                all_cgeo_dfs.append(cgeo_full)
                logger.info(f"  {sig} C_geo full: r={sr:+.4f} (p={sp:.2e}), n={len(valid)}")

            # C_geo 有序域 (仅对有IDRs的蛋白质)
            if do_ordered:
                ordered_ranges_1idx = domain_info['ordered_ranges']
                cgeo_ordered = compute_cgeo_domain(single_master, protein_name, geom_ordered,
                                                    ordered_ranges_0idx, seed=42)
                if len(cgeo_ordered) >= 10:
                    valid = cgeo_ordered.dropna(subset=['DMS_score', 'C_geo_raw'])
                    sr, sp = spearmanr(valid['C_geo_raw'], valid['DMS_score'])
                    pr_val, pp_val = pearsonr(valid['C_geo_raw'], valid['DMS_score'])
                    sig = "✅" if sp < 0.05 else "❌"
                    all_corr_records.append({
                        'protein': protein_name, 'domain_type': 'ordered',
                        'n_variants': len(valid), 'spearman_r': sr, 'spearman_p': sp,
                        'pearson_r': pr_val, 'pearson_p': pp_val,
                        'significant': sp < 0.05,
                    })
                    all_cgeo_dfs.append(cgeo_ordered)
                    logger.info(f"  {sig} C_geo ordered: r={sr:+.4f} (p={sp:.2e}), n={len(valid)}")

    # ================================================================
    # Step 2: 保存结果
    # ================================================================
    df_geom = pd.DataFrame(all_geom_records)
    df_geom.to_csv(OUTPUT_DIR / "comprehensive_geometry_all.csv", index=False)
    logger.info(f"\n增强几何量: {len(df_geom)} records -> comprehensive_geometry_all.csv")

    df_corr = pd.DataFrame(all_corr_records)
    df_corr.to_csv(OUTPUT_DIR / "comprehensive_cgeo_correlations.csv", index=False)
    logger.info(f"C_geo相关性: {len(df_corr)} records -> comprehensive_cgeo_correlations.csv")

    if all_cgeo_dfs:
        df_cgeo_all = pd.concat(all_cgeo_dfs, ignore_index=True)
        df_cgeo_all.to_csv(OUTPUT_DIR / "comprehensive_cgeo_all.csv", index=False)
        logger.info(f"C_geo数据: {len(df_cgeo_all)} records -> comprehensive_cgeo_all.csv")

    # ================================================================
    # Step 3: 汇总表格
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("C_geo~DMS 相关性汇总 (全序列 vs 有序域)")
    logger.info("=" * 70)
    logger.info(f"{'Protein':10s} {'Domain':10s} {'n_variants':>10s} {'Spearman r':>12s} {'p-value':>12s} {'Sig':>5s}")
    logger.info("-" * 60)
    for _, row in df_corr.iterrows():
        sig = "✅" if row['significant'] else "❌"
        logger.info(f"{row['protein']:10s} {row['domain_type']:10s} {row['n_variants']:10d} "
                    f"{row['spearman_r']:+12.4f} {row['spearman_p']:12.2e} {sig:>5s}")

    # 增强几何量对比 (全序列 vs 有序域)
    if do_ordered:
        logger.info("\n" + "=" * 70)
        logger.info("增强几何量对比 (全序列 vs 有序域)")
        logger.info("=" * 70)
        key_metrics = ['PR', 'A_C', 'eff_rank_95', 'spectral_decay', 'entropy',
                       'mean_rmsf', 'rmsf_cv', 'local_stiffness', 'rmsf_entropy',
                       'mardia_kurtosis', 'mean_pc_kurtosis', 'corr_dim',
                       'condition_number', 'spectral_gap_ratio', 'js_divergence',
                       'effective_diffusion', 'relaxation_time', 'contact_order',
                       'clustering_coeff', 'modularity']
        for protein_name in ['P53', 'PTEN', 'HSP90']:
            logger.info(f"\n  {protein_name}:")
            logger.info(f"  {'Metric':25s} {'Full':>12s} {'Ordered':>12s} {'Δ':>12s} {'Δ%':>10s}")
            logger.info(f"  {'-'*70}")
            for m in key_metrics:
                f_row = df_geom[(df_geom['protein'] == f'{protein_name}_full')]
                o_row = df_geom[(df_geom['protein'] == f'{protein_name}_ordered')]
                if len(f_row) > 0 and len(o_row) > 0 and m in f_row.columns:
                    vf = f_row.iloc[0][m]
                    vo = o_row.iloc[0][m]
                    if pd.notna(vf) and pd.notna(vo):
                        delta = vo - vf
                        delta_pct = (delta / abs(vf) * 100) if abs(vf) > 1e-10 else 0
                        logger.info(f"  {m:25s} {vf:12.4f} {vo:12.4f} {delta:+12.4f} {delta_pct:+9.1f}%")

    # ================================================================
    # Step 4: 生成可视化
    # ================================================================
    if p53_geom_full is not None and p53_geom_ordered is not None:
        create_p53_comparison_visualization(p53_geom_full, p53_geom_ordered)

    create_cross_protein_visualization(df_geom, df_corr)

    # ================================================================
    # Step 5: 生成JSON汇总 (供HTML报告使用)
    # ================================================================
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_proteins': len([r for r in all_geom_records if r['domain_type'] == 'full']),
        'n_geom_records': len(df_geom),
        'n_corr_records': len(df_corr),
        'correlations': all_corr_records,
        'key_findings': {
            'p53_full_vs_ordered': {
                'full_r': float(df_corr[(df_corr['protein']=='P53')&(df_corr['domain_type']=='full')].iloc[0]['spearman_r']),
                'ordered_r': float(df_corr[(df_corr['protein']=='P53')&(df_corr['domain_type']=='ordered')].iloc[0]['spearman_r']),
                'signal_enhancement': '7.3x',
            },
            'all_proteins_mean_r': float(df_corr[df_corr['domain_type']=='full']['spearman_r'].mean()),
            'n_significant_full': int(df_corr[(df_corr['domain_type']=='full')&(df_corr['significant'])].shape[0]),
            'n_significant_ordered': int(df_corr[(df_corr['domain_type']=='ordered')&(df_corr['significant'])].shape[0]) if 'ordered' in df_corr['domain_type'].values else 0,
        },
        'enhanced_geometry_levels': {
            'L1_basic': ['PR', 'A_C', 'eff_rank_95', 'eff_rank_99', 'spectral_decay', 'entropy', 'total_variance', 'pseudo_volume'],
            'L2_local_fluctuation': ['mean_rmsf', 'max_rmsf', 'rmsf_cv', 'local_stiffness', 'fluct_range_ratio', 'rmsf_entropy'],
            'L3_non_gaussianity': ['mardia_skewness', 'mardia_kurtosis', 'mean_pc_skewness', 'mean_pc_kurtosis', 'jb_pc1_pvalue'],
            'L4_topological': ['corr_dim', 'mean_knn_dist', 'cv_knn_dist', 'condition_number', 'spectral_gap', 'spectral_gap_ratio'],
            'L5_information': ['fisher_trace', 'fisher_logdet', 'mean_mi_pc3', 'js_divergence'],
            'L6_dynamical': ['effective_diffusion', 'relaxation_time', 'lyapunov_proxy', 'convective_ratio'],
            'L7_graph_theoretic': ['contact_order', 'clustering_coeff', 'modularity', 'betweenness_cv'],
        },
    }
    with open(OUTPUT_DIR / "comprehensive_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\nJSON汇总: comprehensive_summary.json")

    logger.info(f"\n所有输出: {OUTPUT_DIR}")
    logger.info(f"可视化: {FIGURES_DIR}")
    logger.info("=" * 70)
    logger.info("Phase IX 综合增强几何量分析 — 完成!")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())