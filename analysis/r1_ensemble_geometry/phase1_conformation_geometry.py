#!/usr/bin/env python3
"""
Phase 1: 构象 3D坐标几何特征提取 — 完整实现
对 PolyX 系综序列 + WSL2 蛋白的 BioEmu 构象计算完整几何状态

技术参考: TECHNICAL_REFERENCE.md §6.1-6.7 (构象几何), §10.2-10.6 (几何计算)
函数参考: AI_FUNCTION_REFERENCE.md §5 (Embeddings), §10 (Statistics)

输入: BioEmu XTC/NPZ 构象数据
输出: field_theory/tables/phase1_conformation_geometry_summary.csv
      field_theory/figures/phase1_conformation_geometry_radar.png
      field_theory/figures/phase1_geometry_vs_length.png
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import json
import warnings
import MDAnalysis as mda
warnings.filterwarnings('ignore')

sys.path.insert(0, 'B:/2026/Exploration/8.Evolution/1.BasicUnit')
sys.path.insert(0, 'B:/2026/Exploration/8.Evolution/3.Proj9.Computational_v2')
sys.path.insert(0, str(Path(__file__).parent.parent))

with open(Path(__file__).parent.parent / 'config.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

from scripts.utils.geometry_core import compute_geometric_state
from scripts.utils.visualization_utils import plot_radar, plot_scatter_with_fit

geo_cfg = cfg.get('geometry', {})
K_NN = geo_cfg.get('local_geometry', {}).get('k_nn', 30)
Q_TANGENT = geo_cfg.get('local_geometry', {}).get('q_tangent', 10)
K_MASS = geo_cfg.get('curvature', {}).get('k_mass', 5)
MIN_CURVATURE = geo_cfg.get('sample_thresholds', {}).get('min_for_curvature', 50)
MIN_GEOMETRY = geo_cfg.get('sample_thresholds', {}).get('min_for_geometry', 20)

print("=" * 60)
print("Phase 1: 构象 3D 坐标几何特征提取")
print("=" * 60)

POLYX_BASE = Path("B:/2026/Exploration/ProtGenesis2_Ensemble/test_workflow/polyx_ensemble/output")

# 解析目录名: PolyX_PolyG_10 -> (G, 10)
def parse_dirname(dirname):
    parts = dirname.split('_')
    if 'PolyX' not in parts[0]:
        return None, None
    if 'linker' in dirname:
        aa_map = {'PolyEAAAK': 'EAAAK', 'PolyGGGGS': 'GGGGS'}
        for k, v in aa_map.items():
            if k in dirname:
                n = int(parts[-1])
                return v, n
        return None, None
    aa = parts[1].replace('Poly', '')
    try:
        n = int(parts[2])
    except:
        return None, None
    return aa, n

# 收集目录
all_dirs = sorted([d for d in POLYX_BASE.iterdir() if d.is_dir() and (d / 'topology.pdb').exists()])
print(f"  找到 {len(all_dirs)} 个有效序列目录")

# 每个 AA 类型选代表性长度
target_seqs = {}
for d in all_dirs:
    aa, n = parse_dirname(d.name)
    if aa is None:
        continue
    if aa not in target_seqs:
        target_seqs[aa] = []
    target_seqs[aa].append((n, d))

# 每个 AA 选 3-4 个代表性长度
selected = []
for aa, items in sorted(target_seqs.items()):
    items.sort()
    for n, d in items:
        if n in [4, 5, 10, 15, 20, 25, 30, 40, 50]:
            selected.append((aa, n, d))

print(f"  选中 {len(selected)} 个代表性序列进行分析")

all_results = []

for aa, n, seq_dir in selected:
    seq_id = f"{aa}_{n}"
    try:
        top_file = seq_dir / 'topology.pdb'
        xtc_file = seq_dir / 'samples.xtc'
        u = mda.Universe(str(top_file), str(xtc_file))
        ca = u.select_atoms('name CA')
        if len(ca) == 0:
            ca = u.atoms
        n_frames = min(len(u.trajectory), 100)
        n_atoms = len(ca)
        positions = np.zeros((n_frames, n_atoms, 3))
        for i, ts in enumerate(u.trajectory):
            if i >= n_frames:
                break
            positions[i] = ca.positions
    except Exception as e:
        print(f"  [SKIP] {seq_id}: {str(e)[:80]}")
        continue

    n_samples = len(positions)
    if n_samples < MIN_GEOMETRY:
        print(f"  [SKIP] {seq_id}: samples={n_samples} < {MIN_GEOMETRY}")
        continue

    flat = positions.reshape(n_samples, -1)
    k_nn = min(K_NN, n_samples - 2)
    q_t = min(Q_TANGENT, n_samples - 1)
    k_m = min(K_MASS, n_samples - 2)

    state = compute_geometric_state(flat, k_nn=k_nn, q_tangent=q_t, k_mass=k_m)
    sufficient = n_samples >= MIN_CURVATURE

    result = {
        'protein': seq_id, 'aa_type': aa, 'n': n,
        'n_samples': n_samples, 'n_atoms': n_atoms,
        'd_3d_consensus': state['d_consensus'], 'd_3d_pr': state['d_pr'],
        'd_3d_twonn': state['d_twonn'], 'A_3d_C': state['A_C'],
        'rho_3d': state['rho'], 'R_3d': state['R'] if sufficient else np.nan,
        'kappa_3d': state['kappa'] if sufficient else np.nan,
        'spectral_decay_3d': state['spectral_decay'],
        'top5_ratio_3d': state['top5_ratio'], 'entropy_3d': state['entropy'],
        'eff_rank_95_3d': state['eff_rank_95'],
        'pseudo_volume_3d': state['pseudo_volume'],
        'trace_g_3d': state['trace_g'], 'det_g_3d': state['det_g'],
    }
    all_results.append(result)
    print(f"  [OK] {seq_id:15s} | n={n:3d} | samples={n_samples:3d} | "
          f"d_cons={state['d_consensus']:.2f} | d_pr={state['d_pr']:.2f} | "
          f"A_C={state['A_C']:.3f} | spec_decay={state['spectral_decay']:.3f}")

df_results = pd.DataFrame(all_results)
df_results.to_csv('field_theory/tables/phase1_conformation_geometry_summary.csv', index=False)
print(f"\n[OK] 几何汇总已保存: {len(df_results)} 个序列")

# ============================================================
# 可视化
# ============================================================
if len(df_results) >= 5:
    # 图1: d_consensus vs n (按 AA 类型着色)
    aa_colors = {'G': '#1f77b4', 'S': '#ff7f0e', 'E': '#2ca02c', 'L': '#d62728',
                 'K': '#9467bd', 'EAAAK': '#8c564b', 'GGGGS': '#e377c2'}

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    plot_metrics = ['d_3d_consensus', 'A_3d_C', 'spectral_decay_3d',
                    'entropy_3d', 'eff_rank_95_3d', 'pseudo_volume_3d']
    for idx, metric in enumerate(plot_metrics):
        ax = axes[idx // 3, idx % 3]
        for aa in sorted(df_results['aa_type'].unique()):
            subset = df_results[df_results['aa_type'] == aa].sort_values('n')
            if len(subset) < 3:
                continue
            ax.scatter(subset['n'], subset[metric], alpha=0.7, s=25,
                      c=aa_colors.get(aa, '#999999'), label=aa)
            if len(subset) > 3:
                try:
                    z = np.polyfit(np.log(subset['n']), np.log(np.abs(subset[metric]) + 1e-10), 1)
                    x_fit = np.linspace(subset['n'].min(), subset['n'].max(), 50)
                    y_fit = np.exp(z[1]) * x_fit ** z[0]
                    ax.plot(x_fit, y_fit, '-', color=aa_colors.get(aa, '#999999'), alpha=0.4, lw=1)
                except:
                    pass
        ax.set_xlabel('n (sequence length)')
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend(fontsize=6, loc='best')
    plt.suptitle('Phase 1: 3D Conformation Geometry Scaling Laws', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig('field_theory/figures/phase1_geometry_vs_length.png', dpi=300, bbox_inches='tight')
    print(f"[OK] 标度律图已保存")

    # 图2: 雷达图 (AA 类型比较)
    categories = ['d_3d_consensus', 'A_3d_C', 'spectral_decay_3d', 'entropy_3d', 'eff_rank_95_3d']
    max_vals = {c: max(df_results[c].dropna().max(), 1e-10) for c in categories}
    radar_values = {}
    for aa in sorted(df_results['aa_type'].unique()):
        aa_mean = df_results[df_results['aa_type'] == aa][categories].mean()
        if len(aa_mean) > 0:
            radar_values[aa] = [aa_mean[c] / max_vals[c] for c in categories]
    if len(radar_values) >= 2:
        plot_radar(categories, radar_values, list(radar_values.keys()),
                   title='Conformation Geometry by AA Type (3D Coordinate Space)',
                   output_path='field_theory/figures/phase1_conformation_geometry_radar.png')
        print(f"[OK] 雷达图已保存")

print(f"\n{'='*60}")
print(f"Phase 1 3D 坐标几何特征提取完成")
print(f"  处理序列数: {len(all_results)}")
print(f"  AA 类型: {sorted(df_results['aa_type'].unique())}")
print(f"  长度范围: {df_results['n'].min()} - {df_results['n'].max()}")
print(f"{'='*60}")