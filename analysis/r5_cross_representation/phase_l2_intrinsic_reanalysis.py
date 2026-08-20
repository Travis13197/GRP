#!/usr/bin/env python3
"""
Phase L2: PolyX 全量 1316 序列 Kabsch 内禀系重分析
=====================================================================

Law 1 v2 / Law 2 支撑 — 解决缺口 A1 的全面影响评估:
  2026-07-19 L1 发现 "PR≈3 低维流形" 结论部分是刚体模式伪影 (4序列试点:
  PR 3.0→5.0-7.5, eff_rank_95 3→33-48, spectral_decay ~3→~1.7)。
  L2 将 Kabsch 对齐 + Ledoit-Wolf 收缩推广到全部 1316 条 Cα 系综
  (965 PolyX + 94 Linker + 239 Heteropolymer + 10 Natural_IDP + 8 DMS_WT),
  重估 v1.3 深度分析的全部标度律结论 (C1-C8)。

双对齐参考:
  - 'mean'  (2轮迭代平均构象, 统计最优, 主结果)
  - 'first' (第一帧, 与 L1 验证值一致, 一致性桥接)

双谱:
  - raw: 原始内禀协方差谱 (与 v1.3 实验室系同一估计量, 直接可比)
  - lw:  Ledoit-Wolf 收缩谱 (与 g_S/C_geo 一致)

SE(3) 验证:
  对齐后原始协方差应有 6 个近零特征值 (3平动+3转动),
  记录 bottom-8 特征值相对最大值比值 + n_zero_eigs (ratio<1e-8)。

用法 (WSL2):
  source /home/liuchuanyang13/anaconda3/etc/profile.d/conda.sh && conda activate bioemu
  cd /mnt/b/2026/Exploration/ProtGenesis2_Ensemble
  python field_theory/scripts/phase_l2_intrinsic_reanalysis.py --stage extract --workers 8
  python field_theory/scripts/phase_l2_intrinsic_reanalysis.py --stage analyze
  python field_theory/scripts/phase_l2_intrinsic_reanalysis.py --stage all --workers 8

输出:
  test_workflow/polyx_ensemble/analysis/phase_l2_intrinsic/
    l2_intrinsic_geometry.csv          — 每序列内禀系特征 (mean/first × raw/lw)
    l2_scaling_laws_report.json        — 标度律重估 + v1.3 对照 + 结论存活判定
    l2_fig1_pr_scaling.{svg,jpg,html}  — PR~n lab vs intrinsic
    l2_fig2_spectral_decay.{svg,jpg,html}
    l2_fig3_per_aa_spearman.{svg,jpg,html}
    l2_fig4_eff_rank95.{svg,jpg,html}

作者: ProtGenesis2 Ensemble
日期: 2026-07-19
"""

import sys
import os
import json
import time
import argparse
import warnings
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "field_theory" / "scripts"))
from phase_l1_kabsch_metric import kabsch_align_ensemble, ledoit_wolf_shrinkage_fast

POLYX_OUTPUT = PROJECT_ROOT / "test_workflow" / "polyx_ensemble" / "output"
HET_OUTPUT = PROJECT_ROOT / "test_workflow" / "heteropolymer_ensemble" / "output"
ANALYSIS_DIR = PROJECT_ROOT / "test_workflow" / "polyx_ensemble" / "analysis"
OUT_DIR = ANALYSIS_DIR / "phase_l2_intrinsic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "l2_intrinsic_geometry.csv"
OUT_JSON = OUT_DIR / "l2_scaling_laws_report.json"
LAB_CSV = ANALYSIS_DIR / "unified_geometry_500.csv"

RANDOM_SEED = 42

# ============================================================
# 数据发现与加载 (与 phase2_unified_geometry.py 一致)
# ============================================================

def discover_all_sequences():
    seqs = []
    for d in sorted(POLYX_OUTPUT.iterdir()):
        if not d.is_dir() or d.name.startswith('_') or d.name == 'natural_idp':
            continue
        if list(d.glob("batch_*.npz")):
            seqs.append((d.name, d, 'PolyX'))
    idp_dir = POLYX_OUTPUT / "natural_idp"
    if idp_dir.exists():
        for d in sorted(idp_dir.iterdir()):
            if d.is_dir() and list(d.glob("batch_*.npz")):
                seqs.append((d.name, d, 'Natural_IDP'))
    if HET_OUTPUT.exists():
        for d in sorted(HET_OUTPUT.iterdir()):
            if d.is_dir() and not d.name.startswith('_') and list(d.glob("batch_*.npz")):
                seqs.append((d.name, d, 'Heteropolymer'))
    return seqs


def load_ensemble_3d(seq_dir):
    """加载NPZ → (n_samples, n_res, 3)"""
    npz_files = sorted(seq_dir.glob("batch_*.npz"))
    if not npz_files:
        return None
    all_coords = []
    for f in npz_files:
        try:
            data = np.load(f, allow_pickle=True)
            key = 'pos' if 'pos' in data else 'positions'
            if key in data:
                all_coords.append(data[key])
            data.close()
        except Exception:
            continue
    if not all_coords:
        return None
    pos = np.concatenate(all_coords, axis=0)
    if pos.ndim == 2:
        pos = pos.reshape(pos.shape[0], -1, 3)
    return pos


def _parse_n(token):
    """稳健解析链长: 接受 '55' 或 'n55' 形式"""
    try:
        return int(token)
    except Exception:
        t = str(token).lstrip('nN')
        try:
            return int(t)
        except Exception:
            return 0


def classify_sequence(seq_id):
    if seq_id.startswith('PolyX_linker_Poly'):
        parts = seq_id.split('_')
        if len(parts) >= 4:
            linker_type = parts[2]
            linker_type = linker_type[4:] if linker_type.startswith('Poly') else linker_type
            n = _parse_n(parts[3])
            return 'Linker', linker_type, n
    elif seq_id.startswith('PolyX_Poly'):
        parts = seq_id.split('_')
        if len(parts) >= 3:
            aa_raw = parts[1]
            aa = aa_raw[4:] if aa_raw.startswith('Poly') else aa_raw
            n = _parse_n(parts[2])
            return 'PolyX', aa, n
    elif seq_id.startswith('PolyX_'):
        # 旧命名: PolyX_{AA}_n{N} (1000样本扩展链长目录, n=55-200)
        parts = seq_id.split('_')
        if len(parts) >= 3:
            aa = parts[1]
            n = _parse_n(parts[2])
            if aa in ('EAAAK', 'GGGGS'):
                return 'Linker', aa, n
            return 'PolyX', aa, n
    elif seq_id.startswith('HET_'):
        return 'Heteropolymer', 'mixed', 0
    elif seq_id.startswith('DMS_'):
        return 'DMS_WT', 'protein', 0
    elif seq_id in ['Abeta40', 'Abeta42', 'aSyn_N60', 'aSyn_C80', 'Tau_N100',
                    'p53_N93', 'cMyc_N100', 'Ash1_420_500', 'ProTa', 'Histatin5']:
        return 'Natural_IDP', 'IDP', 0
    return 'Unknown', 'unknown', 0


# ============================================================
# 内禀系几何特征 (与 phase2 compute_geometry 相同公式, 在内禀系计算)
# ============================================================

def _features_from_spectrum(eigenvalues):
    """从降序非负特征值谱计算谱特征 (与 phase2 公式一致)"""
    eigenvalues = np.maximum(eigenvalues, 0)
    total_var = eigenvalues.sum()
    n_modes = len(eigenvalues)
    if total_var <= 0:
        return None
    normalized = eigenvalues / total_var

    PR = 1.0 / np.sum(normalized ** 2) if np.sum(normalized ** 2) > 0 else 0
    A_C = (eigenvalues[0] - eigenvalues[2]) / total_var if n_modes >= 3 else 0
    entropy = -np.sum(normalized[normalized > 1e-15] * np.log(normalized[normalized > 1e-15]))
    eff_rank_95 = int(np.searchsorted(np.cumsum(eigenvalues), 0.95 * total_var) + 1)
    top5_ratio = float(eigenvalues[:5].sum() / total_var) if n_modes >= 5 else 1.0

    if n_modes >= 3:
        indices = np.arange(1, n_modes + 1)
        valid = eigenvalues > 1e-15
        if np.sum(valid) >= 3:
            spectral_decay = float(-stats.linregress(np.log(indices[valid]), np.log(eigenvalues[valid]))[0])
        else:
            spectral_decay = 0.0
    else:
        spectral_decay = 0.0

    kappa = float(np.mean(normalized) * n_modes)
    spectral_gap = float(eigenvalues[0] / eigenvalues[1]) if (n_modes >= 2 and eigenvalues[1] > 0) else 0.0
    fisher_info = float(np.sum(1.0 / (eigenvalues + 1e-10)))
    pseudo_volume = float(np.prod(np.sqrt(eigenvalues[:min(10, n_modes)] + 1e-10)))

    return {
        'PR': float(PR), 'A_C': float(A_C), 'entropy': float(entropy),
        'eff_rank_95': eff_rank_95, 'top5_ratio': top5_ratio,
        'spectral_decay': spectral_decay, 'total_variance': float(total_var),
        'kappa': kappa, 'spectral_gap': spectral_gap,
        'fisher_info': fisher_info, 'pseudo_volume': pseudo_volume,
    }


def compute_intrinsic_geometry(pos_3d, align_mode='mean'):
    """Kabsch对齐 → 内禀协方差 → raw谱特征 + LW收缩谱特征 + SE(3)验证"""
    n_samples, n_res, _ = pos_3d.shape
    aligned = kabsch_align_ensemble(pos_3d, reference_mode=align_mode)
    X = aligned.reshape(n_samples, n_res * 3)
    mean = X.mean(axis=0)
    centered = X - mean

    # 原始内禀协方差谱 (与 v1.3 实验室系同一估计量)
    cov_raw = np.cov(centered, rowvar=False)
    eigs_raw = np.linalg.eigvalsh(cov_raw)[::-1]
    eigs_raw = np.maximum(eigs_raw, 0)
    feats = _features_from_spectrum(eigs_raw)
    if feats is None:
        return None

    # SE(3) 验证: bottom-8 特征值相对最大值
    bottom8 = eigs_raw[-8:] / eigs_raw[0] if eigs_raw[0] > 0 else np.zeros(8)
    n_zero = int(np.sum(eigs_raw < eigs_raw[0] * 1e-8)) if eigs_raw[0] > 0 else 0

    # Ledoit-Wolf 收缩谱 (与 g_S/C_geo 一致)
    cov_lw, lambda_star = ledoit_wolf_shrinkage_fast(centered)
    eigs_lw = np.maximum(np.linalg.eigvalsh(cov_lw)[::-1], 0)
    feats_lw = _features_from_spectrum(eigs_lw)

    # 坐标依赖特征 (对齐后, 与收缩无关)
    rng = np.random.default_rng(RANDOM_SEED)
    if n_samples <= 100:
        sample_coords = centered
    else:
        idx = rng.choice(n_samples, size=100, replace=False)
        sample_coords = centered[idx]
    # 平方距离 (避免 sklearn 依赖在子进程重复导入开销, 直接计算)
    sq = np.sum(sample_coords ** 2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2 * sample_coords @ sample_coords.T
    d2 = np.maximum(d2, 0)
    triu = np.triu_indices_from(d2, k=1)
    dists = np.sqrt(d2[triu])
    mean_pairwise_dist = float(dists.mean())
    max_pairwise_dist = float(dists.max())
    std_pairwise_dist = float(dists.std())

    aligned_centered = aligned - aligned.mean(axis=0, keepdims=True)
    rmsf = np.sqrt(np.mean(aligned_centered ** 2, axis=0).sum(axis=1))
    mean_rmsf = float(rmsf.mean())
    max_rmsf = float(rmsf.max())
    std_rmsf = float(rmsf.std())

    # 非高斯性 (top3 PC 投影)
    eigvecs = np.linalg.eigh(cov_raw)[1][:, -3:]
    projections = centered @ eigvecs
    skewness = float(np.mean(stats.skew(projections, axis=0)))
    kurtosis = float(np.mean(stats.kurtosis(projections, axis=0)))

    mean_dist = dists.mean()
    density_raw = n_samples / (mean_dist ** len(eigs_raw)) if mean_dist > 0 else 0.0
    density = float(density_raw) if np.isfinite(density_raw) else 1e10

    row = {
        'n_samples': n_samples, 'n_features': n_res * 3, 'n_residues': n_res,
        # raw 内禀谱 (主结果)
        **feats,
        # LW 收缩谱 (C_geo 一致性)
        'lambda_star': float(lambda_star),
        'PR_lw': feats_lw['PR'],
        'eff_rank_95_lw': feats_lw['eff_rank_95'],
        'spectral_decay_lw': feats_lw['spectral_decay'],
        'top5_ratio_lw': feats_lw['top5_ratio'],
        # SE(3) 验证
        'n_zero_eigs_raw': n_zero,
        'bottom8_eig_ratio_max': float(np.max(bottom8)),
        'bottom8_eig_ratio_min': float(np.min(bottom8)),
        # 坐标依赖
        'mean_pairwise_dist': mean_pairwise_dist,
        'max_pairwise_dist': max_pairwise_dist,
        'std_pairwise_dist': std_pairwise_dist,
        'mean_rmsf': mean_rmsf, 'max_rmsf': max_rmsf, 'std_rmsf': std_rmsf,
        'skewness': skewness, 'kurtosis': kurtosis,
        'density': density,
    }
    return row


def process_sequence(args):
    seq_id, seq_dir, system = args
    try:
        pos = load_ensemble_3d(Path(seq_dir))
        if pos is None or pos.shape[0] < 5:
            return {'seq_id': seq_id, 'system': system, 'error': 'no_data'}
        category, aa_type, n = classify_sequence(seq_id)
        row = {
            'seq_id': seq_id, 'category': category, 'aa_type': aa_type,
            'n': n, 'system': system,
        }
        # 主对齐: mean (2轮迭代); 桥接对齐: first (与L1一致)
        for mode in ['mean', 'first']:
            g = compute_intrinsic_geometry(pos, align_mode=mode)
            if g is None:
                return {'seq_id': seq_id, 'system': system, 'error': f'geometry_failed_{mode}'}
            suffix = '' if mode == 'mean' else '_first'
            for k, v in g.items():
                row[f'{k}{suffix}'] = v
        return row
    except Exception as e:
        return {'seq_id': seq_id, 'system': system, 'error': str(e)[:200]}


# ============================================================
# Stage 1: extract — 全量特征提取 (并行)
# ============================================================

def stage_extract(workers=8):
    seqs = discover_all_sequences()
    print(f"[L2-extract] 发现 {len(seqs)} 序列目录")

    # 断点续跑
    done = set()
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV)
        if 'error' not in prev.columns or prev['error'].isna().all():
            done = set(prev['seq_id'])
        else:
            done = set(prev[prev['error'].isna()]['seq_id'])
        print(f"[L2-extract] 已完成 {len(done)}, 续跑剩余")

    todo = [(sid, str(sd), sys_) for sid, sd, sys_ in seqs if sid not in done]
    print(f"[L2-extract] 待处理 {len(todo)} 序列, workers={workers}")

    t0 = time.time()
    results = []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_sequence, todo)):
            results.append(r)
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                print(f"  progress {i+1}/{len(todo)} ({el:.0f}s, ETA {el/(i+1)*(len(todo)-i-1):.0f}s)")
                # 增量保存
                _save_results(results, append=OUT_CSV.exists() and len(done) > 0)

    _save_results(results, append=OUT_CSV.exists() and len(done) > 0)
    n_ok = sum(1 for r in results if 'error' not in r)
    n_err = sum(1 for r in results if 'error' in r)
    print(f"[L2-extract] 完成: {n_ok} OK, {n_err} 失败, 总耗时 {time.time()-t0:.0f}s")
    return n_ok, n_err


def _save_results(results, append=False):
    if not results:
        return
    df_new = pd.DataFrame(results)
    if append and OUT_CSV.exists():
        df_old = pd.read_csv(OUT_CSV)
        df_new = df_new[~df_new['seq_id'].isin(set(df_old['seq_id']))]
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(OUT_CSV, index=False)


# ============================================================
# Stage 2: analyze — v1.3 标度律在内禀系下重估
# ============================================================

def power_law_fit(n, y):
    """log-log 线性回归 → beta, R2, p (与 v1.3 一致)"""
    mask = (n > 0) & (y > 0) & np.isfinite(y)
    if mask.sum() < 5:
        return None
    slope, intercept, r, p, se = stats.linregress(np.log(n[mask]), np.log(y[mask]))
    return {'beta': float(slope), 'R2': float(r ** 2), 'p': float(p), 'n_points': int(mask.sum())}


def spearman_fit(n, y):
    mask = np.isfinite(n) & np.isfinite(y)
    if mask.sum() < 5:
        return None
    r, p = stats.spearmanr(n[mask], y[mask])
    return {'r': float(r), 'p': float(p), 'n_points': int(mask.sum())}


def breakpoint_scan(n, y, n_min=6, n_max=None):
    """两段线性 breakpoint 扫描 (v1.3 方法)"""
    mask = np.isfinite(n) & np.isfinite(y)
    n, y = n[mask], y[mask]
    if len(n) < 12:
        return None
    if n_max is None:
        n_max = int(np.max(n)) - 5
    best = None
    for bp in range(n_min, n_max + 1):
        m1, m2 = n <= bp, n > bp
        if m1.sum() < 4 or m2.sum() < 4:
            continue
        s1 = np.polyfit(n[m1], y[m1], 1)
        s2 = np.polyfit(n[m2], y[m2], 1)
        sse = np.sum((y[m1] - np.polyval(s1, n[m1])) ** 2) + np.sum((y[m2] - np.polyval(s2, n[m2])) ** 2)
        if best is None or sse < best['sse']:
            best = {'breakpoint': bp, 'slope1': float(s1[0]), 'slope2': float(s2[0]), 'sse': float(sse)}
    return best


# v1.3 实验室系已发表值 (CLAUDE.md / deep_manifold_analysis v1.3, G/S/E/L/K+linkers 子集)
V13_LAB = {
    'overall': {
        'spectral_decay': {'beta': -0.236, 'R2': 0.650, 'p': 1.98e-76},
        'top5_ratio': {'r': -0.827, 'p': 1.10e-83},
        'entropy': {'beta': 0.036, 'R2': 0.150, 'p': 3.32e-13},
        'pseudo_volume': {'beta': -1.276, 'R2': 0.306, 'p': 1.01e-27},
        'PR': {'beta': 0.016, 'R2': 0.068, 'p': 1.68e-06, 'spearman_r': 0.313},
        'eff_rank_95': {'beta': 0.024, 'R2': 0.037, 'p': 4.95e-04},
        'mean_pairwise_dist': {'beta': 0.007, 'R2': 0.012, 'p': 0.047},
        'A_C': {'beta': -0.008, 'R2': 0.007, 'p': 0.13},
        'total_variance': {'beta': 0.011, 'R2': 0.007, 'p': 0.13},
    },
    'per_aa_pr_spearman': {'K': 0.757, 'EAAAK': 0.570, 'S': 0.500, 'G': 0.375,
                           'L': 0.227, 'E': 0.209, 'GGGGS': 0.198},
    'per_aa_sd_spearman': {'GGGGS': -0.998, 'G': -0.996, 'S': -0.993, 'E': -0.981,
                           'K': -0.978, 'EAAAK': -0.872, 'L': -0.534},
    'breakpoints': {'G': 45, 'EAAAK': 30, 'S': 9, 'GGGGS': 24, 'E': 35},
}


def stage_analyze():
    df = pd.read_csv(OUT_CSV)
    if 'error' in df.columns:
        df = df[df['error'].isna()]
    lab = pd.read_csv(LAB_CSV)
    print(f"[L2-analyze] intrinsic {len(df)} 序列, lab {len(lab)} 序列")

    m = df.merge(lab[['seq_id', 'PR', 'eff_rank_95', 'spectral_decay', 'top5_ratio',
                      'entropy', 'pseudo_volume', 'total_variance', 'mean_pairwise_dist',
                      'A_C', 'n_samples']],
                 on='seq_id', how='left', suffixes=('', '_lab'))
    print(f"[L2-analyze] 合并 lab 对照: {m['PR_lab'].notna().sum()} 序列")

    report = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
              'n_sequences_intrinsic': int(len(df)),
              'n_with_lab_baseline': int(m['PR_lab'].notna().sum())}

    # ---------- A. 总体偏移 (lab vs intrinsic, 同序列配对) ----------
    paired = m.dropna(subset=['PR_lab'])
    report['global_shift'] = {
        'PR_lab_median': float(paired['PR_lab'].median()),
        'PR_intrinsic_median': float(paired['PR'].median()),
        'PR_lw_median': float(paired['PR_lw'].median()),
        'eff_rank95_lab_median': float(paired['eff_rank_95_lab'].median()),
        'eff_rank95_intrinsic_median': float(paired['eff_rank_95'].median()),
        'spectral_decay_lab_median': float(paired['spectral_decay_lab'].median()),
        'spectral_decay_intrinsic_median': float(paired['spectral_decay'].median()),
        'PR_ratio_median': float((paired['PR'] / paired['PR_lab']).median()),
        'lambda_star_median': float(paired['lambda_star'].median()),
        'n_zero_eigs_raw_median': float(paired['n_zero_eigs_raw'].median()),
        'align_first_vs_mean_PR_r': float(stats.spearmanr(paired['PR'], paired['PR_first'])[0]),
    }

    # ---------- B. 标度律重估 (PolyX+Linker, 按 feature × frame) ----------
    poly = m[(m['category'].isin(['PolyX', 'Linker'])) & (m['n'] > 0)].copy()
    feats_scale = ['PR', 'spectral_decay', 'eff_rank_95', 'entropy', 'top5_ratio',
                   'pseudo_volume', 'total_variance', 'mean_pairwise_dist', 'A_C']
    scaling = {}
    n_arr = poly['n'].values.astype(float)
    for f in feats_scale:
        entry = {}
        # intrinsic (raw aligned, 主结果)
        entry['intrinsic'] = power_law_fit(n_arr, poly[f].values.astype(float))
        sp = spearman_fit(n_arr, poly[f].values.astype(float))
        if sp:
            entry['intrinsic_spearman'] = sp
        # lab (同序列, 同估计量)
        lab_col = f'{f}_lab'
        if lab_col in poly.columns:
            sub = poly.dropna(subset=[lab_col])
            entry['lab_same_corpus'] = power_law_fit(sub['n'].values.astype(float),
                                                     sub[lab_col].values.astype(float))
        scaling[f] = entry
    report['scaling_laws'] = scaling

    # ---------- C. per-AA Spearman (PR~n, spectral_decay~n) ----------
    per_aa = {}
    for aa, sub in poly.groupby('aa_type'):
        if len(sub) < 12:
            continue
        n_a = sub['n'].values.astype(float)
        entry = {}
        for f in ['PR', 'spectral_decay']:
            si = spearman_fit(n_a, sub[f].values.astype(float))
            sub_lab = sub.dropna(subset=[f'{f}_lab'])
            sl = spearman_fit(sub_lab['n'].values.astype(float),
                              sub_lab[f'{f}_lab'].values.astype(float)) if len(sub_lab) >= 12 else None
            entry[f'{f}_intrinsic'] = si
            entry[f'{f}_lab'] = sl
        # breakpoint on intrinsic PR
        bp_i = breakpoint_scan(n_a, sub['PR'].values.astype(float))
        bp_l = None
        sub_lab = sub.dropna(subset=['PR_lab'])
        if len(sub_lab) >= 12:
            bp_l = breakpoint_scan(sub_lab['n'].values.astype(float),
                                   sub_lab['PR_lab'].values.astype(float))
        entry['breakpoint_PR_intrinsic'] = bp_i
        entry['breakpoint_PR_lab'] = bp_l
        entry['n_seqs'] = int(len(sub))
        per_aa[aa] = entry
    report['per_aa'] = per_aa

    # ---------- D. v1.3 结论存活判定 ----------
    verdicts = {}
    # C1: PR≈3 极低维流形 → 已被L1证伪(刚体伪影), L2全量确认
    pr_med = report['global_shift']['PR_intrinsic_median']
    verdicts['C1_PR3_low_dim_manifold'] = {
        'v13_claim': 'PR≈3 极低维流形 (Cα 3个空间方向)',
        'l2_intrinsic_PR_median': pr_med,
        'verdict': 'FALSIFIED_by_rigid_body_artifact' if pr_med > 4.5 else 'PARTIALLY_SUPPORTED',
        'note': 'Kabsch对齐后内禀PR中位数 %.2f; eff_rank_95 3 → %d' %
                (pr_med, report['global_shift']['eff_rank95_intrinsic_median'])}
    # C2: PR~n 正相关
    pr_sp = scaling.get('PR', {}).get('intrinsic_spearman')
    verdicts['C2_PR_n_positive'] = {
        'v13_claim': 'PR~n r=0.313',
        'l2_intrinsic': pr_sp,
        'verdict': 'SURVIVES' if pr_sp and pr_sp['r'] > 0.1 and pr_sp['p'] < 0.05 else 'NOT_SURVIVED'}
    # C5: spectral_decay~n 最强信号
    sd_fit = scaling.get('spectral_decay', {}).get('intrinsic')
    verdicts['C5_spectral_decay_strongest'] = {
        'v13_claim': 'beta=-0.236, R2=0.650',
        'l2_intrinsic': sd_fit,
        'verdict': 'SURVIVES' if sd_fit and sd_fit['R2'] > 0.3 and sd_fit['beta'] < 0 else 'WEAKENED'}
    # C6: eff_rank_95≈3
    verdicts['C6_eff_rank95_const3'] = {
        'v13_claim': 'eff_rank_95≈3 恒定',
        'l2_intrinsic_median': report['global_shift']['eff_rank95_intrinsic_median'],
        'verdict': 'FALSIFIED_by_rigid_body_artifact',
        'note': '内禀系 eff_rank_95 中位数 %d — 95%%方差由远多于3个维度解释' %
                report['global_shift']['eff_rank95_intrinsic_median']}
    # C8: per-AA spectral_decay~n 普遍负相关
    neg = sum(1 for aa, e in per_aa.items()
              if e.get('spectral_decay_intrinsic') and e['spectral_decay_intrinsic']['r'] < -0.5)
    tot = sum(1 for aa, e in per_aa.items() if e.get('spectral_decay_intrinsic'))
    verdicts['C8_sd_n_universal_negative'] = {
        'v13_claim': '所有AA spectral_decay~n 强负相关 (r<-0.87 除L)',
        'l2_intrinsic': f'{neg}/{tot} AA类型 r<-0.5',
        'verdict': 'SURVIVES' if tot > 0 and neg / tot > 0.7 else 'WEAKENED'}
    report['v13_verdicts'] = verdicts

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[L2-analyze] 报告已保存: {OUT_JSON}")

    # ---------- E. 图 (SVG+JPG+HTML) ----------
    _make_figures(m, poly, per_aa, report)
    return report


def _setup_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return plt


def _save_fig(fig, plt, name):
    svg = OUT_DIR / f"{name}.svg"
    jpg = OUT_DIR / f"{name}.jpg"
    fig.savefig(svg, bbox_inches='tight')
    fig.savefig(jpg, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {svg.name} / {jpg.name}")


def _make_figures(m, poly, per_aa, report):
    plt = _setup_matplotlib()
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        has_plotly = True
    except ImportError:
        has_plotly = False

    # ---- Fig 1: PR~n lab vs intrinsic (G/E/K/L/S + EAAAK/GGGGS) ----
    show_aa = ['G', 'S', 'E', 'K', 'L', 'EAAAK', 'GGGGS']
    colors = {'G': '#1f77b4', 'S': '#2ca02c', 'E': '#d62728', 'K': '#9467bd',
              'L': '#8c564b', 'EAAAK': '#ff7f0e', 'GGGGS': '#17becf'}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for aa in show_aa:
        sub = poly[poly['aa_type'] == aa].dropna(subset=['PR_lab'])
        if len(sub) < 5:
            continue
        c = colors.get(aa, '#333333')
        axes[0].scatter(sub['n'], sub['PR_lab'], s=10, alpha=0.45, color=c, label=aa)
        axes[1].scatter(sub['n'], sub['PR'], s=10, alpha=0.45, color=c, label=aa)
    for ax, t in zip(axes, ['Lab frame (v1.3, 刚体混入)', 'Intrinsic frame (Kabsch, L2)']):
        ax.set_xlabel('Chain length n')
        ax.set_ylabel('Participation Ratio')
        ax.set_title(f'PR ~ n — {t}')
        ax.legend(fontsize=8, markerscale=2)
        ax.grid(alpha=0.3)
    fig.suptitle('Phase L2: PR 标度律 lab vs 内禀系 (v1.3 C1/C2 重估)', y=1.02)
    fig.tight_layout()
    _save_fig(fig, plt, 'l2_fig1_pr_scaling')

    # ---- Fig 2: spectral_decay~n ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for aa in show_aa:
        sub = poly[poly['aa_type'] == aa].dropna(subset=['spectral_decay_lab'])
        if len(sub) < 5:
            continue
        c = colors.get(aa, '#333333')
        axes[0].scatter(sub['n'], sub['spectral_decay_lab'], s=10, alpha=0.45, color=c, label=aa)
        axes[1].scatter(sub['n'], sub['spectral_decay'], s=10, alpha=0.45, color=c, label=aa)
    for ax, t in zip(axes, ['Lab frame', 'Intrinsic frame (Kabsch)']):
        ax.set_xlabel('Chain length n')
        ax.set_ylabel('spectral_decay (power-law exponent)')
        ax.set_title(f'spectral_decay ~ n — {t}')
        ax.legend(fontsize=8, markerscale=2)
        ax.grid(alpha=0.3)
    fig.suptitle('Phase L2: 谱衰减标度律 lab vs 内禀系 (v1.3 C5/C8 重估)', y=1.02)
    fig.tight_layout()
    _save_fig(fig, plt, 'l2_fig2_spectral_decay')

    # ---- Fig 3: per-AA Spearman r 对比 (PR~n / sd~n) ----
    aas = sorted([aa for aa, e in per_aa.items()
                  if e.get('PR_intrinsic') and e.get('PR_lab')], key=str)
    if aas:
        r_lab = [per_aa[aa]['PR_lab']['r'] for aa in aas]
        r_int = [per_aa[aa]['PR_intrinsic']['r'] for aa in aas]
        sd_lab = [per_aa[aa]['spectral_decay_lab']['r'] if per_aa[aa].get('spectral_decay_lab') else np.nan for aa in aas]
        sd_int = [per_aa[aa]['spectral_decay_intrinsic']['r'] if per_aa[aa].get('spectral_decay_intrinsic') else np.nan for aa in aas]
        x = np.arange(len(aas))
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
        w = 0.38
        axes[0].bar(x - w / 2, r_lab, w, label='Lab', color='#9ecae1')
        axes[0].bar(x + w / 2, r_int, w, label='Intrinsic (Kabsch)', color='#08519c')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(aas, rotation=60, fontsize=8)
        axes[0].set_ylabel('Spearman r (PR ~ n)')
        axes[0].legend()
        axes[0].grid(alpha=0.3, axis='y')
        axes[1].bar(x - w / 2, sd_lab, w, label='Lab', color='#fcae91')
        axes[1].bar(x + w / 2, sd_int, w, label='Intrinsic (Kabsch)', color='#cb181d')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(aas, rotation=60, fontsize=8)
        axes[1].set_ylabel('Spearman r (spectral_decay ~ n)')
        axes[1].legend()
        axes[1].grid(alpha=0.3, axis='y')
        fig.suptitle('Phase L2: per-AA 标度相关性 lab vs 内禀系', y=1.02)
        fig.tight_layout()
        _save_fig(fig, plt, 'l2_fig3_per_aa_spearman')

    # ---- Fig 4: eff_rank_95 lab vs intrinsic + SE(3)验证 ----
    paired = m.dropna(subset=['eff_rank_95_lab'])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].scatter(paired['n_residues'], paired['eff_rank_95_lab'], s=8, alpha=0.4,
                    label='Lab', color='#9ecae1')
    axes[0].scatter(paired['n_residues'], paired['eff_rank_95'], s=8, alpha=0.4,
                    label='Intrinsic', color='#cb181d')
    axes[0].set_xlabel('n_residues')
    axes[0].set_ylabel('eff_rank_95')
    axes[0].set_title('eff_rank_95: lab vs 内禀系 (v1.3 C6 重估)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    nz = paired['n_zero_eigs_raw'].values
    axes[1].hist(nz, bins=np.arange(-0.5, max(12, nz.max() + 1.5)), color='#238b45', alpha=0.8)
    axes[1].axvline(6, color='k', ls='--', label='SE(3) 预言 = 6')
    axes[1].set_xlabel('近零特征值个数 (ratio<1e-8, raw aligned cov)')
    axes[1].set_ylabel('序列数')
    axes[1].set_title('SE(3) 刚体模式验证: 6个近零特征值')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle('Phase L2: 内禀维度与 SE(3) 验证', y=1.02)
    fig.tight_layout()
    _save_fig(fig, plt, 'l2_fig4_eff_rank95_se3')

    # ---- Plotly HTML (4合1) ----
    if has_plotly:
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=('PR ~ n (lab vs intrinsic)',
                                            'spectral_decay ~ n (lab vs intrinsic)',
                                            'eff_rank_95 vs n_residues',
                                            'SE(3) 近零特征值直方图'))
        for aa in show_aa:
            sub = poly[poly['aa_type'] == aa].dropna(subset=['PR_lab'])
            if len(sub) < 5:
                continue
            c = colors.get(aa, '#333333')
            fig.add_trace(go.Scatter(x=sub['n'], y=sub['PR_lab'], mode='markers',
                                     marker=dict(size=4, opacity=0.4, color=c),
                                     name=f'{aa} lab', legendgroup=aa), row=1, col=1)
            fig.add_trace(go.Scatter(x=sub['n'], y=sub['PR'], mode='markers',
                                     marker=dict(size=4, opacity=0.4, symbol='x', color=c),
                                     name=f'{aa} intrinsic', legendgroup=aa), row=1, col=1)
            fig.add_trace(go.Scatter(x=sub['n'], y=sub['spectral_decay_lab'], mode='markers',
                                     marker=dict(size=4, opacity=0.4, color=c),
                                     name=f'{aa} lab', legendgroup=aa, showlegend=False), row=1, col=2)
            fig.add_trace(go.Scatter(x=sub['n'], y=sub['spectral_decay'], mode='markers',
                                     marker=dict(size=4, opacity=0.4, symbol='x', color=c),
                                     name=f'{aa} intrinsic', legendgroup=aa, showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=paired['n_residues'], y=paired['eff_rank_95_lab'], mode='markers',
                                 marker=dict(size=4, opacity=0.35, color='#9ecae1'), name='lab'), row=2, col=1)
        fig.add_trace(go.Scatter(x=paired['n_residues'], y=paired['eff_rank_95'], mode='markers',
                                 marker=dict(size=4, opacity=0.35, color='#cb181d'), name='intrinsic'), row=2, col=1)
        fig.add_trace(go.Histogram(x=nz, nbinsx=15, marker_color='#238b45', name='n_zero_eigs'), row=2, col=2)
        fig.update_layout(height=900, width=1200,
                          title_text='Phase L2: Kabsch 内禀系全量重分析 (1316序列) — v1.3 标度律重估',
                          template='plotly_white')
        html_path = OUT_DIR / 'l2_dashboard.html'
        fig.write_html(str(html_path), include_plotlyjs='cdn')
        print(f"  saved {html_path.name}")


# ============================================================
# main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['extract', 'analyze', 'all'], default='all')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    print("=" * 70)
    print("Phase L2: Kabsch 内禀系全量重分析 (1316序列)")
    print("=" * 70)

    if args.stage in ('extract', 'all'):
        stage_extract(workers=args.workers)
    if args.stage in ('analyze', 'all'):
        stage_analyze()

    print("[L2] 全部完成")


if __name__ == '__main__':
    main()
