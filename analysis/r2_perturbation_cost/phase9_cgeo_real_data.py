#!/usr/bin/env python3
"""
Phase IX Fix: C_geo 重新计算 — 使用真实BioEmu NPZ数据
===========================================================
将原 phase9_dms_expansion.py 中的合成系综替换为真实BioEmu NPZ数据，
重新计算全部8个蛋白质的C_geo和DMS相关性。

核心变更:
  1. 用真实BioEmu NPZ替换 generate_synthetic_ensemble()
  2. C_geo 使用正则化 Mahalanobis 度量 (2026-07-17 审计修复 C1):
     C_geo = d^T * (C + eps*I)^{-1} * d, eps = 0.01*trace(C)/3
     (原二次型 d^T*C*d 物理方向相反, 已废弃)
  3. 对比新旧结果

数据:
  - BioEmu NPZ: field_theory/data/dms/results/bioemu/{protein}_wt/batch_*.npz
  - DMS单突变体: field_theory/data/dms/phase9_dms_expansion/phase9_dms_single_mutants.csv
  - WT序列: field_theory/data/dms/phase9_dms_expansion/wt_sequences.json

用法 (WSL2):
  source /home/liuchuanyang13/anaconda3/etc/profile.d/conda.sh
  conda activate bioemu
  cd /mnt/b/2026/Exploration/ProtGenesis2_Ensemble
  python field_theory/scripts/phase9_cgeo_real_data.py
"""

import sys
import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
FIELD_THEORY = PROJECT_ROOT / "field_theory"
DMS_DIR = FIELD_THEORY / "data" / "dms"
BIOEMU_OUTPUT = DMS_DIR / "results" / "bioemu"
DMS_EXPANSION_DIR = DMS_DIR / "phase9_dms_expansion"
OUTPUT_DIR = DMS_DIR / "phase9_cgeo_real"

for d in [OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('cgeo_real')

# ============================================================
# 氨基酸物化属性
# ============================================================
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


def load_bioemu_ensemble(protein_name: str) -> Optional[np.ndarray]:
    """从真实BioEmu NPZ文件加载WT系综"""
    npz_dir = BIOEMU_OUTPUT / f"{protein_name.lower()}_wt"
    if not npz_dir.exists():
        logger.warning(f"  [{protein_name}] BioEmu目录不存在: {npz_dir}")
        return None

    npz_files = sorted(npz_dir.glob("batch_*.npz"))
    if not npz_files:
        logger.warning(f"  [{protein_name}] 无NPZ文件")
        return None

    all_pos = []
    for f in npz_files:
        try:
            data = np.load(f, allow_pickle=True)
            if 'pos' in data:
                all_pos.append(data['pos'])
        except Exception as e:
            logger.warning(f"  [{protein_name}] 读取失败: {f.name}: {e}")

    if not all_pos:
        return None

    positions = np.concatenate(all_pos, axis=0)
    if len(positions) > 250:
        positions = positions[:250]

    logger.info(f"  [{protein_name}] 加载 {len(positions)} 构象, "
                f"形状 {positions.shape}")
    return positions


def compute_ensemble_geometry(coords: np.ndarray) -> Dict:
    """计算系综几何特征 (与phase9_dms_expansion.py相同)"""
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

    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)

    total_var = eigenvalues.sum()
    if total_var <= 0:
        return {'PR': 0, 'A_C': 0, 'eff_rank_95': 1, 'spectral_decay': 0,
                'entropy': 0, 'total_variance': 0, 'pseudo_volume': 0,
                'mean_pos': np.zeros(n_residues * 3), 'cov_matrix': cov,
                'n_samples': n_samples, 'n_residues': n_residues}

    normalized = eigenvalues / total_var

    PR = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum() if (eigenvalues ** 2).sum() > 0 else 0
    A_C = normalized[0]
    cumsum = np.cumsum(normalized)
    eff_rank_95 = int(np.searchsorted(cumsum, 0.95) + 1)

    k = np.arange(1, min(51, len(eigenvalues) + 1))
    log_k = np.log(k)
    log_eig = np.log(eigenvalues[:len(k)] + 1e-10)
    A = np.vstack([log_k, np.ones(len(k))]).T
    alpha, _ = np.linalg.lstsq(A, log_eig, rcond=None)[0]
    spectral_decay = -alpha

    entropy = -np.sum(normalized * np.log(normalized + 1e-10))

    nonzero = eigenvalues[eigenvalues > 1e-10]
    log_det = np.log(nonzero).sum() if len(nonzero) > 0 else -np.inf
    pseudo_volume = np.exp(log_det / len(nonzero)) if len(nonzero) > 0 else 0

    return {
        'n_samples': n_samples,
        'n_residues': n_residues,
        'PR': float(PR),
        'A_C': float(A_C),
        'eff_rank_95': eff_rank_95,
        'spectral_decay': float(spectral_decay),
        'entropy': float(entropy),
        'total_variance': float(total_var),
        'pseudo_volume': float(pseudo_volume),
        'mean_pos': mean,
        'cov_matrix': cov,
    }


def compute_cgeo_for_protein(protein_name: str, single_mutants: pd.DataFrame,
                              wt_seq: str, wt_geom: Dict) -> pd.DataFrame:
    """计算单个蛋白质的C_geo (与phase9_dms_expansion.py相同算法)"""
    prot_df = single_mutants[single_mutants['protein'] == protein_name].copy()
    if len(prot_df) == 0:
        return pd.DataFrame()

    n_res = len(wt_seq)
    mean_pos = wt_geom['mean_pos'].reshape(n_res, 3)
    cov = wt_geom['cov_matrix']
    D = cov.shape[0]
    np.random.seed(42)

    # 预计算每个残基位置的3x3协方差子矩阵
    cov_blocks = np.zeros((n_res, 3, 3))
    for i in range(n_res):
        i0, i1 = i * 3, (i + 1) * 3
        if i1 <= D:
            cov_blocks[i] = cov[i0:i1, i0:i1]

    # 预计算正则化局部度量张量 g_S = (C + eps*I)^{-1} (论文 Law 1 定义)
    # 修复审计发现 C1: 原实现为 d^T*C*d 二次型, 非 Mahalanobis 距离
    # eps = 0.01 * trace(C_block)/q, q=3 (单残基 3D 坐标块)
    REG_EPS = 0.01
    g_S_blocks = np.zeros((n_res, 3, 3))
    for i in range(n_res):
        cb = cov_blocks[i]
        eps = REG_EPS * np.trace(cb) / 3.0
        g_S_blocks[i] = np.linalg.inv(cb + eps * np.eye(3))

    # 预计算物化属性差异
    positions = prot_df['position'].values.astype(int) - 1
    wt_aas = prot_df['wt_aa'].values
    mut_aas = prot_df['mut_aa'].values

    vol_diffs = np.array([AA_VOLUMES.get(m, 0) - AA_VOLUMES.get(w, 0)
                          for w, m in zip(wt_aas, mut_aas)])
    chg_diffs = np.array([AA_CHARGES.get(m, 0) - AA_CHARGES.get(w, 0)
                          for w, m in zip(wt_aas, mut_aas)])
    hydro_diffs = np.array([AA_HYDROPHOBICITY.get(m, 0) - AA_HYDROPHOBICITY.get(w, 0)
                            for w, m in zip(wt_aas, mut_aas)])

    mags = 0.1 + np.abs(vol_diffs) / 100 + np.abs(chg_diffs) * 0.05
    directions = np.random.randn(len(prot_df), 3)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-10

    n_mut = len(prot_df)
    cgeo_raw = np.zeros(n_mut)

    for i in range(n_mut):
        pos = positions[i]
        if pos < 0 or pos >= n_res:
            continue
        d = mags[i] * directions[i]
        g_S = g_S_blocks[pos]  # 3x3 regularized local metric (C+eps*I)^{-1}
        # C_geo_raw = d^T * g_S * d (正则化 Mahalanobis 距离, 论文 Eq. C_geo)
        cgeo_raw[i] = d @ g_S @ d

    cgeo_metric = cgeo_raw.copy()

    result_data = {
        'protein': protein_name,
        'mutant': prot_df['mutant'].values,
        'position': prot_df['position'].values.astype(int),
        'wt_aa': wt_aas,
        'mut_aa': mut_aas,
        'DMS_score': prot_df['DMS_score'].values,
        'DMS_score_bin': prot_df.get('DMS_score_bin', np.full(n_mut, np.nan)).values,
        'C_geo_raw': cgeo_raw,
        'C_geo_metric': cgeo_metric,
        'C_geo_euclidean': np.sum(mags.reshape(-1, 1) * directions, axis=1) ** 2,
        'C_geo_local': np.sum(mags.reshape(-1, 1) * directions, axis=1) ** 2,
        'aa_volume_diff': vol_diffs,
        'aa_charge_diff': chg_diffs,
        'aa_hydrophobicity_diff': hydro_diffs,
        'seq_length': n_res,
    }

    return pd.DataFrame(result_data)


def main():
    logger.info("=" * 60)
    logger.info("Phase IX: C_geo 重新计算 — 使用真实BioEmu NPZ数据")
    logger.info("=" * 60)

    # 加载DMS数据
    single_path = DMS_EXPANSION_DIR / "phase9_dms_single_mutants.csv"
    wt_path = DMS_EXPANSION_DIR / "wt_sequences.json"

    if not single_path.exists():
        logger.error(f"单突变体表不存在: {single_path}")
        return 1
    if not wt_path.exists():
        logger.error(f"WT序列不存在: {wt_path}")
        return 1

    single_master = pd.read_csv(single_path)
    with open(wt_path) as f:
        wt_sequences = json.load(f)

    logger.info(f"加载 {len(single_master)} 单突变体")
    logger.info(f"WT序列: {list(wt_sequences.keys())}")

    # 检查哪些蛋白质有真实BioEmu数据
    available_proteins = []
    for p in wt_sequences:
        npz_dir = BIOEMU_OUTPUT / f"{p.lower()}_wt"
        npz_files = list(npz_dir.glob("batch_*.npz")) if npz_dir.exists() else []
        status = "✅" if len(npz_files) >= 5 else "⚠️" if len(npz_files) > 0 else "❌"
        logger.info(f"  {status} {p}: {len(npz_files)} NPZ files")
        if len(npz_files) >= 5:
            available_proteins.append(p)

    if not available_proteins:
        logger.error("没有可用的BioEmu数据！")
        return 1

    logger.info(f"\n可用蛋白质: {available_proteins}")

    # 加载真实BioEmu系综并计算几何
    all_geom = {}
    for protein_name in available_proteins:
        wt_seq = wt_sequences.get(protein_name)
        if wt_seq is None:
            continue

        ensemble = load_bioemu_ensemble(protein_name)
        if ensemble is None:
            logger.warning(f"  [{protein_name}] 无法加载系综，跳过")
            continue

        geom = compute_ensemble_geometry(ensemble)
        all_geom[protein_name] = geom
        logger.info(f"  [{protein_name}] PR={geom['PR']:.2f}, A_C={geom['A_C']:.4f}, "
                    f"SD={geom['spectral_decay']:.2f}, eff_rank={geom['eff_rank_95']}")

    # 计算C_geo
    all_cgeo = []
    all_correlations = []

    for protein_name in available_proteins:
        if protein_name not in all_geom:
            continue
        wt_seq = wt_sequences[protein_name]
        geom = all_geom[protein_name]

        logger.info(f"\n  计算 {protein_name} C_geo ({len(wt_seq)} aa)...")
        t0 = time.time()

        cgeo_df = compute_cgeo_for_protein(
            protein_name, single_master, wt_seq, geom
        )

        if len(cgeo_df) == 0:
            logger.warning(f"  [{protein_name}] 无有效突变体")
            continue

        # 相关性分析
        valid = cgeo_df.dropna(subset=['DMS_score', 'C_geo_raw'])
        if len(valid) < 10:
            logger.warning(f"  [{protein_name}] 有效数据点不足: {len(valid)}")
            continue

        sr_raw, sp_raw = spearmanr(valid['C_geo_raw'], valid['DMS_score'])
        sr_metric, sp_metric = spearmanr(valid['C_geo_metric'], valid['DMS_score'])
        sr_euclid, sp_euclid = spearmanr(valid['C_geo_euclidean'], valid['DMS_score'])
        sr_local, sp_local = spearmanr(valid['C_geo_local'], valid['DMS_score'])

        pr_raw, pp_raw = pearsonr(valid['C_geo_raw'], valid['DMS_score'])

        sig = "✅" if sp_raw < 0.05 else "❌"
        logger.info(f"  [{protein_name}] {sig} C_geo_raw~DMS: "
                    f"Spearman r={sr_raw:.4f} (p={sp_raw:.2e}), "
                    f"Pearson r={pr_raw:.4f} (p={pp_raw:.2e}), "
                    f"n={len(valid)}, {time.time()-t0:.1f}s")

        all_cgeo.append(cgeo_df)
        all_correlations.append({
            'protein': protein_name,
            'n_residues': len(wt_seq),
            'n_variants': len(valid),
            'C_geo_raw_spearman_r': sr_raw,
            'C_geo_raw_spearman_p': sp_raw,
            'C_geo_raw_pearson_r': pr_raw,
            'C_geo_raw_pearson_p': pp_raw,
            'C_geo_metric_spearman_r': sr_metric,
            'C_geo_metric_spearman_p': sp_metric,
            'C_geo_euclidean_spearman_r': sr_euclid,
            'C_geo_euclidean_spearman_p': sp_euclid,
            'C_geo_local_spearman_r': sr_local,
            'C_geo_local_spearman_p': sp_local,
            'PR': geom['PR'],
            'A_C': geom['A_C'],
            'spectral_decay': geom['spectral_decay'],
            'entropy': geom['entropy'],
            'data_source': 'real_bioemu',
        })

    # 保存结果
    if all_cgeo:
        cgeo_master = pd.concat(all_cgeo, ignore_index=True)
        cgeo_path = OUTPUT_DIR / "phase9_cgeo_real_results.csv"
        cgeo_master.to_csv(cgeo_path, index=False)
        logger.info(f"\nC_geo结果: {len(cgeo_master)} 突变体 -> {cgeo_path}")

    if all_correlations:
        corr_df = pd.DataFrame(all_correlations)
        corr_path = OUTPUT_DIR / "phase9_cgeo_real_correlations.csv"
        corr_df.to_csv(corr_path, index=False)

        # 汇总
        logger.info("\n" + "=" * 60)
        logger.info("C_geo~DMS 相关性汇总 (真实BioEmu数据)")
        logger.info("=" * 60)

        mean_sr = corr_df['C_geo_raw_spearman_r'].mean()
        n_sig = (corr_df['C_geo_raw_spearman_p'] < 0.05).sum()
        logger.info(f"Mean Spearman r = {mean_sr:.4f}")
        logger.info(f"Significant (p<0.05): {n_sig}/{len(corr_df)}")
        logger.info(f"数据来源: 真实BioEmu NPZ")

        for _, row in corr_df.iterrows():
            sig = "✅" if row['C_geo_raw_spearman_p'] < 0.05 else "❌"
            logger.info(f"  {sig} {row['protein']}: r={row['C_geo_raw_spearman_r']:.4f} "
                        f"(p={row['C_geo_raw_spearman_p']:.2e}), n={int(row['n_variants'])}")

        logger.info(f"\n所有结果保存在: {OUTPUT_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())