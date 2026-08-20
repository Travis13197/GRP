#!/usr/bin/env python3
"""
Phase L3: Cross-Sampler Validation (BioEmu vs OpenMM)
=====================================================================

Law 1 v2 / Law 2 外部效度验证 — 解决缺口 A3 (单一采样器循环论证)

设计:
  - 20条代表性PolyX序列: G/S/E/K/L × n=10/20/30/50
  - BioEmu: 500 samples @ 300K (已有数据, 直接加载)
  - OpenMM: 5ns NVT @ 300K (amber99sb, tip3p, 1000 frames)
  - OpenMM: 5ns NVT @ 350K (amber99sb, tip3p, 1000 frames)
  - 几何量: Kabsch+LW 内禀谱特征 (PR, eff_rank_95, spectral_decay, C_geo)
  - SE(3)验证: bottom-8 特征值/最大值, n_zero_eigs

输出:
  test_workflow/phase_l3_cross_sampler/
    l3_cross_sampler_results.csv      — 每序列×每采样器几何特征
    l3_cross_sampler_report.json      — 一致性/差异性统计
    l3_fig1_pr_comparison.{svg,jpg}   — PR跨采样器对比
    l3_fig2_spectral_decay.{svg,jpg}  — spectral_decay跨采样器对比
    l3_fig3_cgeo_comparison.{svg,jpg} — C_geo跨采样器对比
    l3_fig4_se3_validation.{svg,jpg}  — SE(3)验证对比

用法 (WSL2):
  source /home/liuchuanyang13/anaconda3/etc/profile.d/conda.sh && conda activate bioemu
  cd /mnt/b/2026/Exploration/ProtGenesis2_Ensemble
  python field_theory/scripts/phase_l3_cross_sampler.py --stage extract --workers 8
  python field_theory/scripts/phase_l3_cross_sampler.py --stage analyze
  python field_theory/scripts/phase_l3_cross_sampler.py --stage all --workers 8

作者: ProtGenesis2 Ensemble
日期: 2026-07-20
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
OUT_DIR = PROJECT_ROOT / "test_workflow" / "phase_l3_cross_sampler"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "l3_cross_sampler_results.csv"
OUT_JSON = OUT_DIR / "l3_cross_sampler_report.json"

RANDOM_SEED = 42

# ============================================================
# 序列选择: 5 AA × 4链长 = 20条
# ============================================================

AA_LIST = ['G', 'S', 'E', 'K', 'L']
N_LIST = [10, 20, 30, 50]

def get_l3_sequences():
    """返回20条L3验证序列的 (seq_id, aa, n, seq)"""
    seqs = []
    for aa in AA_LIST:
        for n in N_LIST:
            seq_id = f"PolyX_Poly{aa}_{n}"
            seq = aa * n
            seqs.append((seq_id, aa, n, seq))
    return seqs


# ============================================================
# BioEmu 数据加载
# ============================================================

def load_bioemu_ensemble(seq_id):
    """加载BioEmu NPZ → (n_samples, n_res, 3)"""
    seq_dir = POLYX_OUTPUT / seq_id
    if not seq_dir.exists():
        return None
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


# ============================================================
# OpenMM MD 采样 (占位 — 需WSL2 GPU执行)
# ============================================================

def run_openmm_md(seq_id, sequence, temperature_k=300, duration_ns=5, output_dir=None):
    """
    运行OpenMM MD模拟 (占位函数 — 实际执行需WSL2 GPU环境)

    参数:
        seq_id: 序列ID
        sequence: 氨基酸序列
        temperature_k: 温度 (K)
        duration_ns: 模拟时长 (ns)
        output_dir: 输出目录

    返回:
        success: bool
        message: str
        output_path: str or None
    """
    if output_dir is None:
        output_dir = OUT_DIR / "openmm_md" / f"{seq_id}_{temperature_k}K"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否已完成
    xtc_path = output_dir / "trajectory.xtc"
    if xtc_path.exists():
        return True, f"Already done: {xtc_path}", str(xtc_path)

    # 构建初始结构 (从BioEmu topology.pdb或生成扩展链)
    # 注意: 此处为占位实现, 实际执行需:
    # 1. 从BioEmu输出加载topology.pdb作为初始结构
    # 2. 使用hpacker重建侧链
    # 3. 使用OpenMM amber99sb + tip3p进行NVT MD
    # 4. 保存轨迹为XTC格式

    # 生成占位标记文件
    marker_path = output_dir / "MARKER_PENDING_WSL2_EXECUTION.txt"
    with open(marker_path, 'w') as f:
        f.write(f"OpenMM MD pending WSL2 execution\n")
        f.write(f"seq_id: {seq_id}\n")
        f.write(f"sequence: {sequence}\n")
        f.write(f"temperature: {temperature_k} K\n")
        f.write(f"duration: {duration_ns} ns\n")
        f.write(f"forcefield: amber99sb + tip3p\n")
        f.write(f"frames_to_save: 1000\n")

    return False, f"Pending WSL2 execution: {marker_path}", None


def load_openmm_trajectory(seq_id, temperature_k=300):
    """
    加载OpenMM轨迹 → (n_frames, n_res, 3) Cα坐标 (Angstrom)

    数据来源: phase_l3_openmm_worker.py 输出的 ca_positions.npy
    """
    output_dir = OUT_DIR / "openmm_md" / f"{seq_id}_{temperature_k}K"
    ca_npy = output_dir / "ca_positions.npy"

    if not ca_npy.exists():
        return None

    pos = np.load(ca_npy)  # (n_frames, n_res, 3) in Angstrom
    return pos


# ============================================================
# 内禀系几何特征 (与L2一致)
# ============================================================

def _features_from_spectrum(eigenvalues):
    """从降序非负特征值谱计算谱特征"""
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

    # 原始内禀协方差谱
    cov_raw = np.cov(centered, rowvar=False)
    eigs_raw = np.linalg.eigvalsh(cov_raw)[::-1]
    eigs_raw = np.maximum(eigs_raw, 0)
    feats = _features_from_spectrum(eigs_raw)
    if feats is None:
        return None

    # SE(3) 验证
    bottom8 = eigs_raw[-8:] / eigs_raw[0] if eigs_raw[0] > 0 else np.zeros(8)
    n_zero = int(np.sum(eigs_raw < eigs_raw[0] * 1e-8)) if eigs_raw[0] > 0 else 0

    # Ledoit-Wolf 收缩谱
    cov_lw, lambda_star = ledoit_wolf_shrinkage_fast(centered)
    eigs_lw = np.maximum(np.linalg.eigvalsh(cov_lw)[::-1], 0)
    feats_lw = _features_from_spectrum(eigs_lw)

    # 坐标依赖特征
    rng = np.random.default_rng(RANDOM_SEED)
    if n_samples <= 100:
        sample_coords = centered
    else:
        idx = rng.choice(n_samples, size=100, replace=False)
        sample_coords = centered[idx]
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

    # 非高斯性
    eigvecs = np.linalg.eigh(cov_raw)[1][:, -3:]
    projections = centered @ eigvecs
    skewness = float(np.mean(stats.skew(projections, axis=0)))
    kurtosis = float(np.mean(stats.kurtosis(projections, axis=0)))

    mean_dist = dists.mean()
    density_raw = n_samples / (mean_dist ** len(eigs_raw)) if mean_dist > 0 else 0.0
    density = float(density_raw) if np.isfinite(density_raw) else 1e10

    row = {
        'n_samples': n_samples, 'n_features': n_res * 3, 'n_residues': n_res,
        **feats,
        'lambda_star': float(lambda_star),
        'PR_lw': feats_lw['PR'],
        'eff_rank_95_lw': feats_lw['eff_rank_95'],
        'spectral_decay_lw': feats_lw['spectral_decay'],
        'top5_ratio_lw': feats_lw['top5_ratio'],
        'n_zero_eigs_raw': n_zero,
        'bottom8_eig_ratio_max': float(np.max(bottom8)),
        'bottom8_eig_ratio_min': float(np.min(bottom8)),
        'mean_pairwise_dist': mean_pairwise_dist,
        'max_pairwise_dist': max_pairwise_dist,
        'std_pairwise_dist': std_pairwise_dist,
        'mean_rmsf': mean_rmsf, 'max_rmsf': max_rmsf, 'std_rmsf': std_rmsf,
        'skewness': skewness, 'kurtosis': kurtosis,
        'density': density,
    }
    return row


# ============================================================
# 处理单个序列×采样器
# ============================================================

def process_sequence_sampler(args):
    """处理单个序列的单个采样器"""
    seq_id, aa, n, seq, sampler = args
    try:
        if sampler == 'bioemu_300K':
            pos = load_bioemu_ensemble(seq_id)
            if pos is None or pos.shape[0] < 5:
                return {'seq_id': seq_id, 'sampler': sampler, 'error': 'no_data'}
        elif sampler == 'openmm_300K':
            pos = load_openmm_trajectory(seq_id, temperature_k=300)
            if pos is None or pos.shape[0] < 5:
                return {'seq_id': seq_id, 'sampler': sampler, 'error': 'no_data'}
        elif sampler == 'openmm_350K':
            pos = load_openmm_trajectory(seq_id, temperature_k=350)
            if pos is None or pos.shape[0] < 5:
                return {'seq_id': seq_id, 'sampler': sampler, 'error': 'no_data'}
        else:
            return {'seq_id': seq_id, 'sampler': sampler, 'error': f'unknown_sampler'}

        # 计算内禀几何
        g = compute_intrinsic_geometry(pos, align_mode='mean')
        if g is None:
            return {'seq_id': seq_id, 'sampler': sampler, 'error': 'geometry_failed'}

        row = {
            'seq_id': seq_id, 'aa_type': aa, 'n': n,
            'sampler': sampler,
            **g
        }
        return row

    except Exception as e:
        return {'seq_id': seq_id, 'sampler': sampler, 'error': str(e)[:200]}


# ============================================================
# Stage 1: extract — 全量特征提取
# ============================================================

def stage_extract(workers=8):
    """提取所有序列×采样器的几何特征"""
    l3_seqs = get_l3_sequences()
    samplers = ['bioemu_300K', 'openmm_300K', 'openmm_350K']

    print(f"[L3-extract] 序列: {len(l3_seqs)}, 采样器: {len(samplers)}")

    # 断点续跑
    done = set()
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV)
        if 'error' not in prev.columns or prev['error'].isna().all():
            done = set(zip(prev['seq_id'], prev['sampler']))
        else:
            done = set(zip(prev[prev['error'].isna()]['seq_id'], prev[prev['error'].isna()]['sampler']))
        print(f"[L3-extract] 已完成 {len(done)}, 续跑剩余")

    todo = []
    for seq_id, aa, n, seq in l3_seqs:
        for sampler in samplers:
            if (seq_id, sampler) not in done:
                todo.append((seq_id, aa, n, seq, sampler))

    print(f"[L3-extract] 待处理 {len(todo)} 任务, workers={workers}")

    t0 = time.time()
    results = []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_sequence_sampler, todo)):
            results.append(r)
            if (i + 1) % 10 == 0:
                el = time.time() - t0
                print(f"  progress {i+1}/{len(todo)} ({el:.0f}s)")
                _save_results(results, append=OUT_CSV.exists() and len(done) > 0)

    _save_results(results, append=OUT_CSV.exists() and len(done) > 0)
    n_ok = sum(1 for r in results if 'error' not in r)
    n_err = sum(1 for r in results if 'error' in r)
    print(f"[L3-extract] 完成: {n_ok} OK, {n_err} 失败, 总耗时 {time.time()-t0:.0f}s")
    return n_ok, n_err


def _save_results(results, append=False):
    if not results:
        return
    df_new = pd.DataFrame(results)
    if append and OUT_CSV.exists():
        df_old = pd.read_csv(OUT_CSV)
        # 移除旧CSV中与新结果重复的(seq_id, sampler)行 (替换no_data)
        new_keys = set(zip(df_new['seq_id'], df_new['sampler']))
        old_keys = set(zip(df_old['seq_id'], df_old['sampler']))
        overlapping = new_keys & old_keys
        if overlapping:
            df_old = df_old[~df_old.set_index(['seq_id', 'sampler']).index.isin(overlapping)]
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(OUT_CSV, index=False)


# ============================================================
# Stage 2: analyze — 跨采样器一致性/差异性分析
# ============================================================

def stage_analyze():
    """分析跨采样器一致性"""
    df = pd.read_csv(OUT_CSV)
    if 'error' in df.columns:
        df = df[df['error'].isna()]

    print(f"[L3-analyze] 加载 {len(df)} 条记录")

    # 按seq_id分组, 对比不同采样器
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_records': int(len(df)),
        'n_sequences': int(df['seq_id'].nunique()),
    }

    # 检查哪些序列有完整的三采样器数据
    sampler_counts = df.groupby('seq_id')['sampler'].nunique()
    complete_seqs = sampler_counts[sampler_counts == 3].index.tolist()
    report['n_complete_3samplers'] = len(complete_seqs)
    report['complete_sequences'] = complete_seqs

    # 跨采样器相关性 (以bioemu_300K为基准)
    metrics = ['PR', 'eff_rank_95', 'spectral_decay', 'total_variance',
               'mean_pairwise_dist', 'mean_rmsf', 'lambda_star']

    cross_sampler_stats = {}
    for metric in metrics:
        pivot = df.pivot(index='seq_id', columns='sampler', values=metric)
        if 'bioemu_300K' not in pivot.columns:
            continue
        stats_dict = {}
        for col in pivot.columns:
            if col == 'bioemu_300K':
                continue
            valid = pivot[['bioemu_300K', col]].dropna()
            if len(valid) >= 3:
                r, p = stats.spearmanr(valid['bioemu_300K'], valid[col])
                stats_dict[f'spearman_r_{col}'] = float(r)
                stats_dict[f'spearman_p_{col}'] = float(p)
                stats_dict[f'n_valid_{col}'] = int(len(valid))
        cross_sampler_stats[metric] = stats_dict

    report['cross_sampler_correlations'] = cross_sampler_stats

    # 保存报告
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[L3-analyze] 报告已保存: {OUT_JSON}")

    # 生成可视化
    _make_figures(df, report)

    return report


def _make_figures(df, report):
    """生成跨采样器对比图"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 设置中文字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    # Fig 1: PR 跨采样器对比
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Phase L3: Cross-Sampler Validation — PR Comparison', fontsize=14)

    metrics = ['PR', 'eff_rank_95', 'spectral_decay', 'mean_rmsf']
    titles = ['Participation Ratio (PR)', 'Effective Rank 95%', 'Spectral Decay', 'Mean RMSF (Å)']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        pivot = df.pivot(index='seq_id', columns='sampler', values=metric)

        if 'bioemu_300K' not in pivot.columns:
            ax.text(0.5, 0.5, 'No bioemu_300K data', ha='center', va='center')
            ax.set_title(title)
            continue

        # 绘制每个采样器 vs bioemu_300K
        colors = {'openmm_300K': 'blue', 'openmm_350K': 'red'}
        for col in pivot.columns:
            if col == 'bioemu_300K':
                continue
            valid = pivot[['bioemu_300K', col]].dropna()
            if len(valid) > 0:
                ax.scatter(valid['bioemu_300K'], valid[col],
                          label=col, alpha=0.7, c=colors.get(col, 'gray'))
                # 对角线
                lims = [valid.min().min(), valid.max().max()]
                ax.plot(lims, lims, 'k--', alpha=0.3)

        ax.set_xlabel('BioEmu 300K')
        ax.set_ylabel('OpenMM')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['svg', 'jpg']:
        fig_path = OUT_DIR / f"l3_fig1_pr_comparison.{ext}"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[L3-fig] 图1已保存: l3_fig1_pr_comparison.svg/jpg")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase L3: Cross-Sampler Validation")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["extract", "analyze", "all"],
                        help="执行阶段")
    parser.add_argument("--workers", type=int, default=8,
                        help="并行workers数")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase L3: Cross-Sampler Validation (BioEmu vs OpenMM)")
    print("=" * 60)

    if args.stage in ["extract", "all"]:
        stage_extract(workers=args.workers)

    if args.stage in ["analyze", "all"]:
        stage_analyze()

    print("=" * 60)
    print("Phase L3 完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
