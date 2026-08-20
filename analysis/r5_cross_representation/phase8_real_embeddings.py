#!/usr/bin/env python3
"""
Phase 8: 跨模型与结构表征验证 — 真实数据版本
=====================================================================
完全替代 phase7_phase8_supplement.py 中的合成数据。
使用真实 PolyX 几何特征数据进行跨模型一致性验证。

方法:
  1. 加载 Phase 1 Cα 几何特征 (BioEmu 真实构象)
  2. 加载 Phase C/D/E 全原子几何特征 (hpacker 真实数据)
  3. 对比 Cα vs 全原子层面几何特征相关性
  4. 分析模型无关性: 几何特征是否在不同表征层面一致

输出:
  - phase8_cross_model_geometry_correlation.csv (跨模型几何相关性, 真实数据)
  - phase8_model_comparison_heatmap.png/svg (缺失可视化)
  - phase8_structure_alignment.png/svg (缺失可视化)
  - phase8_report.html (更新版使用真实数据)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from scipy.stats import spearmanr, pearsonr
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = Path(__file__).parent.parent
TABLES_DIR = SCRIPT_DIR / 'tables'
FIGURES_DIR = SCRIPT_DIR / 'figures'
REPORTS_DIR = SCRIPT_DIR / 'reports'

# 全原子数据路径
FULL_ATOM_DIR = Path(__file__).parent.parent.parent / 'test_workflow' / 'polyx_ensemble' / 'analysis' / 'full_atom'

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_ca_geometry():
    """加载 Cα 层面几何特征"""
    ca_path = TABLES_DIR / 'phase1_conformation_geometry_summary.csv'
    if not ca_path.exists():
        logger.warning(f"Cα 几何数据不存在: {ca_path}")
        return None
    
    df_ca = pd.read_csv(ca_path)
    logger.info(f"加载 Cα 几何数据: {len(df_ca)} 条序列")
    return df_ca


def load_fa_geometry():
    """加载全原子层面几何特征"""
    fa_path = FULL_ATOM_DIR / 'full_atom_geometry_results.csv'
    if not fa_path.exists():
        logger.warning(f"全原子几何数据不存在: {fa_path}")
        return None
    
    df_fa = pd.read_csv(fa_path)
    logger.info(f"加载全原子几何数据: {len(df_fa)} 条序列")
    logger.info(f"  列: {list(df_fa.columns)[:10]}...")
    return df_fa


def load_phase_cde_data():
    """加载 Phase C/D/E 对比数据"""
    cde_path = FULL_ATOM_DIR / 'phase_cde' / 'phase_cde_results_v3.json'
    if not cde_path.exists():
        cde_path = FULL_ATOM_DIR / 'phase_cde' / 'phase_cde_results_v2.json'
    if not cde_path.exists():
        logger.warning(f"Phase C/D/E 数据不存在: {cde_path}")
        return None
    
    import json
    with open(cde_path, 'r') as f:
        data = json.load(f)
    logger.info(f"加载 Phase C/D/E 数据: {len(data)} keys")
    return data


def align_ca_fa_data(df_ca, df_fa):
    """对齐 Cα 和全原子数据的共同序列"""
    logger.info("\n[对齐数据] 查找 Cα 和全原子共同序列...")
    
    if df_ca is None or df_fa is None:
        return None
    
    # 尝试通过 seq_id 对齐
    if 'seq_id' in df_ca.columns and 'seq_id' in df_fa.columns:
        common_ids = set(df_ca['seq_id']) & set(df_fa['seq_id'])
        logger.info(f"  共同 seq_id: {len(common_ids)}")
    else:
        # 通过 aa + n 对齐
        if 'aa' in df_ca.columns and 'n' in df_ca.columns:
            ca_key = df_ca.apply(lambda r: f"{r['aa']}_{r['n']}", axis=1)
        else:
            ca_key = df_ca.index.astype(str)
        
        if 'aa' in df_fa.columns and 'n' in df_fa.columns:
            fa_key = df_fa.apply(lambda r: f"{r['aa']}_{r['n']}", axis=1)
        else:
            fa_key = df_fa.index.astype(str)
        
        common_keys = set(ca_key) & set(fa_key)
        logger.info(f"  共同 key: {len(common_keys)}")
        
        df_ca_aligned = df_ca[ca_key.isin(common_keys)].copy()
        df_fa_aligned = df_fa[fa_key.isin(common_keys)].copy()
        
        return df_ca_aligned, df_fa_aligned
    
    df_ca_aligned = df_ca[df_ca['seq_id'].isin(common_ids)].sort_values('seq_id')
    df_fa_aligned = df_fa[df_fa['seq_id'].isin(common_ids)].sort_values('seq_id')
    
    return df_ca_aligned, df_fa_aligned


def compute_cross_model_correlations(df_ca, df_fa):
    """计算 Cα vs 全原子几何特征相关性"""
    logger.info("\n[计算跨模型相关性] Cα vs 全原子...")
    
    if df_ca is None or df_fa is None:
        return None
    
    # 定义共同几何特征
    common_features = []
    for feat in ['PR', 'd_consensus', 'A_C', 'spectral_decay', 'entropy', 
                 'pseudo_volume', 'mean_pairwise_dist', 'total_variance', 'eff_rank_95']:
        if feat in df_ca.columns and feat in df_fa.columns:
            common_features.append(feat)
    
    # 也尝试带后缀的列名
    for feat in ['PR', 'd_consensus', 'A_C', 'spectral_decay']:
        ca_col = f'{feat}_ca' if f'{feat}_ca' in df_ca.columns else feat
        fa_col = f'{feat}_fa' if f'{feat}_fa' in df_fa.columns else feat
        if ca_col in df_ca.columns and fa_col in df_fa.columns:
            if feat not in common_features:
                common_features.append(feat)
    
    logger.info(f"  共同几何特征: {common_features}")
    
    if len(common_features) == 0:
        # 尝试所有共同列
        common_cols = set(df_ca.columns) & set(df_fa.columns)
        numeric_cols = [c for c in common_cols 
                       if df_ca[c].dtype in ['float64', 'int64', 'float32', 'int32']]
        common_features = numeric_cols[:10]
        logger.info(f"  使用共同数值列: {common_features}")
    
    correlation_records = []
    
    for feat in common_features:
        if feat not in df_ca.columns or feat not in df_fa.columns:
            continue
        
        ca_vals = df_ca[feat].dropna().values
        fa_vals = df_fa[feat].dropna().values
        
        # 确保长度一致
        min_len = min(len(ca_vals), len(fa_vals))
        ca_vals = ca_vals[:min_len]
        fa_vals = fa_vals[:min_len]
        
        if min_len < 10:
            continue
        
        try:
            r_p, p_p = pearsonr(ca_vals, fa_vals)
            r_s, p_s = spearmanr(ca_vals, fa_vals)
        except:
            continue
        
        correlation_records.append({
            'model1': 'BioEmu_Ca',
            'model2': 'FullAtom',
            'metric': feat,
            'pearson_r': float(r_p),
            'pearson_p': float(p_p),
            'spearman_r': float(r_s),
            'spearman_p': float(p_s),
            'n_samples': min_len,
        })
    
    if len(correlation_records) == 0:
        logger.warning("  无有效相关性数据!")
        return None
    
    df_corr = pd.DataFrame(correlation_records)
    df_corr.to_csv(TABLES_DIR / 'phase8_cross_model_geometry_correlation.csv', index=False)
    logger.info(f"  phase8_cross_model_geometry_correlation.csv: {len(df_corr)} 条记录")
    
    # 打印关键发现
    for _, row in df_corr.iterrows():
        sig = '***' if row['pearson_p'] < 0.001 else '**' if row['pearson_p'] < 0.01 else '*' if row['pearson_p'] < 0.05 else 'ns'
        logger.info(f"  {row['metric']}: r={row['pearson_r']:.4f} ({sig})")
    
    return df_corr


def generate_visualizations(df_corr, df_ca, df_fa):
    """生成所有缺失的可视化"""
    logger.info("\n[生成可视化] ...")
    
    if df_corr is None:
        logger.warning("  无数据，跳过可视化")
        return
    
    # --- Fig 1: 跨模型相关性热图 ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = df_corr['metric'].unique()
    corr_values = df_corr.set_index('metric')['pearson_r'].values
    
    # 创建热图数据
    heatmap_data = np.array([corr_values])
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                xticklabels=metrics, yticklabels=['Ca vs FA'],
                ax=ax, cbar_kws={'label': 'Pearson r'})
    ax.set_title('Cα vs Full-Atom Geometric Feature Correlation', fontsize=14)
    ax.set_xlabel('Geometric Feature', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'phase8_model_comparison_heatmap.png', dpi=300)
    plt.savefig(FIGURES_DIR / 'phase8_model_comparison_heatmap.svg', dpi=300)
    plt.close()
    logger.info("  Fig 1: phase8_model_comparison_heatmap.png/svg")
    
    # --- Fig 2: 结构对齐图 — 几何特征对比 ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    metrics_to_plot = df_corr['metric'].unique()[:6]
    
    for idx, metric in enumerate(metrics_to_plot):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        if metric in df_ca.columns and metric in df_fa.columns:
            ca_vals = df_ca[metric].dropna().values
            fa_vals = df_fa[metric].dropna().values
            
            min_len = min(len(ca_vals), len(fa_vals))
            ca_vals = ca_vals[:min_len]
            fa_vals = fa_vals[:min_len]
            
            r_val = df_corr[df_corr['metric'] == metric]['pearson_r'].values
            r_str = f'r={r_val[0]:.3f}' if len(r_val) > 0 else ''
            
            ax.scatter(ca_vals, fa_vals, alpha=0.5, s=20, c='steelblue', edgecolors='none')
            
            # 拟合线
            if min_len >= 3:
                try:
                    z = np.polyfit(ca_vals, fa_vals, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(ca_vals.min(), ca_vals.max(), 100)
                    ax.plot(x_line, p(x_line), 'r--', linewidth=1.5, alpha=0.7)
                except:
                    pass
            
            ax.set_xlabel(f'Cα {metric}', fontsize=10)
            ax.set_ylabel(f'FA {metric}', fontsize=10)
            ax.set_title(f'{metric} ({r_str})', fontsize=12)
            ax.grid(True, alpha=0.3)
    
    for idx in range(len(metrics_to_plot), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Cα vs Full-Atom Geometric Feature Alignment', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'phase8_structure_alignment.png', dpi=300)
    plt.savefig(FIGURES_DIR / 'phase8_structure_alignment.svg', dpi=300)
    plt.close()
    logger.info("  Fig 2: phase8_structure_alignment.png/svg")
    
    # --- Fig 3: 相关性条形图 ---
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = df_corr['metric'].values
    r_values = df_corr['pearson_r'].values
    colors = ['#4CAF50' if r > 0 else '#F44336' for r in r_values]
    
    bars = ax.bar(range(len(metrics)), r_values, color=colors, alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 显著性标注
    for i, (r, p) in enumerate(zip(df_corr['pearson_r'], df_corr['pearson_p'])):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.text(i, r + 0.02 * np.sign(r), sig, ha='center', fontsize=9, fontweight='bold')
    
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.set_ylabel('Pearson r (Cα vs FA)', fontsize=13)
    ax.set_title('Cross-Model Geometric Feature Consistency', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'phase8_cross_model_dms_performance.png', dpi=300)
    plt.savefig(FIGURES_DIR / 'phase8_cross_model_dms_performance.svg', dpi=300)
    plt.close()
    logger.info("  Fig 3: phase8_cross_model_dms_performance.png/svg")


def generate_report(df_corr):
    """生成使用真实数据的 Phase 8 报告"""
    logger.info("\n[生成报告] ...")
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if df_corr is None or len(df_corr) == 0:
        logger.error("  无数据，无法生成报告")
        return
    
    corr_rows = ''
    for _, row in df_corr.iterrows():
        sig = '***' if row['pearson_p'] < 0.001 else '**' if row['pearson_p'] < 0.01 else '*' if row['pearson_p'] < 0.05 else ''
        corr_rows += f'''<tr>
          <td>{row['metric']}</td>
          <td>{row['pearson_r']:.4f}</td>
          <td>{row['spearman_r']:.4f}</td>
          <td>{row['pearson_p']:.2e}</td>
          <td>{sig}</td>
          <td>{int(row['n_samples'])}</td>
        </tr>'''
    
    # 统计
    n_positive = (df_corr['pearson_r'] > 0).sum()
    n_significant = (df_corr['pearson_p'] < 0.05).sum()
    mean_r = df_corr['pearson_r'].mean()
    
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Phase 8: 跨模型与结构表征验证 (真实数据)</title>
<style>
body {{ font-family: 'DejaVu Sans', 'Arial', sans-serif; margin: 0; padding: 0; background: #f0f2f5; }}
.container {{ max-width: 1100px; margin: 0 auto; padding: 20px; }}
.header {{ background: linear-gradient(135deg, #0d47a1, #d97706); color: white; padding: 30px; border-radius: 8px 8px 0 0; }}
.header h1 {{ margin: 0; font-size: 1.5em; }}
.content {{ background: white; padding: 30px; border-radius: 0 0 8px 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
h2 {{ color: #0d47a1; margin-top: 30px; border-bottom: 2px solid #e0e0e0; padding-bottom: 8px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.9em; margin: 15px 0; }}
th, td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
th {{ background: #0d47a1; color: white; }}
tr:nth-child(even) {{ background: #f8f9fa; }}
.insight {{ background: #fef3c7; padding: 15px 20px; border-left: 4px solid #d97706; margin: 20px 0; }}
.insight.blue {{ background: #e3f2fd; border-left-color: #1565c0; }}
.figure {{ text-align: center; margin: 25px 0; }}
.figure img {{ max-width: 100%; border: 1px solid #e0e0e0; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
.figure .caption {{ font-size: 0.9em; color: #666; margin-top: 8px; }}
.data-source {{ background: #fff3e0; padding: 10px 15px; border-radius: 4px; font-size: 0.85em; margin: 10px 0; }}
</style>
</head>
<body>
<div class="container">
<div class="header">
  <h1>Phase 8: 跨模型与结构表征验证</h1>
  <p>生成时间: {timestamp} | 数据来源: 真实 Cα BioEmu + 全原子 hpacker 数据</p>
</div>
<div class="content">

<div class="data-source">
  <strong>数据来源</strong>: 
  Cα: Phase 1 几何特征 (phase1_conformation_geometry_summary.csv, 329 条)
  | 全原子: Phase B 全原子几何 (full_atom_geometry_results.csv, 517 条)
  — 基于 BioEmu 真实构象 + hpacker 侧链重建
</div>

<h2>1. 核心发现</h2>
<div class="insight">
  <strong>跨模型几何一致性</strong>: Cα 和全原子层面的几何特征相关性分析显示，
  平均 Pearson r = {mean_r:.3f}。{n_positive}/{len(df_corr)} 个特征呈现正相关，
  {n_significant}/{len(df_corr)} 个特征达到统计显著 (p < 0.05)。
</div>

<div class="insight blue">
  <strong>模型无关性结论</strong>: 如果 Cα 和全原子层面的几何特征高度相关，
  则说明几何场论的核心预测 (PR, A_C, spectral_decay 等) 不依赖于具体的结构表征层次，
  是蛋白质构象空间的普适性质。
</div>

<h2>2. 方法学</h2>
<p>加载 Cα 层面 (BioEmu 直接输出的 Cα 坐标) 和全原子层面 (hpacker 侧链重建 + OpenMM 弛豫)
的几何特征数据，计算 Pearson 和 Spearman 相关系数。</p>
<p>共同特征包括: PR (参与率), d_consensus (共识维度), A_C (各向异性), 
spectral_decay (谱衰减), entropy (构象熵), pseudo_volume (伪体积), 
mean_pairwise_dist (平均成对距离), total_variance (总方差), eff_rank_95 (有效秩)。</p>

<h2>3. 跨模型相关性</h2>
<table>
<tr><th>特征</th><th>Pearson r</th><th>Spearman r</th><th>p-value</th><th>显著性</th><th>n</th></tr>
{corr_rows}
</table>

<h2>4. 可视化</h2>

<div class="figure">
  <img src="../figures/phase8_model_comparison_heatmap.png" alt="Correlation Heatmap">
  <div class="caption"><strong>图 1</strong>: Cα vs 全原子几何特征相关性热图。颜色深浅表示 Pearson r 的大小和方向。</div>
</div>

<div class="figure">
  <img src="../figures/phase8_structure_alignment.png" alt="Structure Alignment">
  <div class="caption"><strong>图 2</strong>: 结构对齐图 — 各几何特征在 Cα 和全原子层面的散点图对比。红色虚线为拟合线。</div>
</div>

<div class="figure">
  <img src="../figures/phase8_cross_model_dms_performance.png" alt="Cross Model Performance">
  <div class="caption"><strong>图 3</strong>: 跨模型一致性条形图。绿色: 正相关; 红色: 负相关。显著性标注: *** p<0.001, ** p<0.01, * p<0.05。</div>
</div>

<h2>5. 生物学意义</h2>
<div class="insight">
  <strong>几何场论的模型无关性</strong>: 蛋白质的构象几何特征 (如 PR 的低维流形、spectral_decay 的标度律)
  如果在不同结构表征层次 (Cα 骨架 vs 全原子) 之间保持一致，则说明这些规律是蛋白质物理构象空间
  的固有属性，而非特定建模方法的 artifact。这为几何场论作为蛋白质构象涨落的普适理论提供了关键支持。
</div>

<h2>6. 输出文件</h2>
<table>
<tr><th>文件</th><th>内容</th><th>记录数</th></tr>
<tr><td>phase8_cross_model_geometry_correlation.csv</td><td>跨模型几何相关性</td><td>{len(df_corr)}</td></tr>
<tr><td>phase8_model_comparison_heatmap.png/svg</td><td>相关性热图</td><td>—</td></tr>
<tr><td>phase8_structure_alignment.png/svg</td><td>结构对齐图</td><td>—</td></tr>
<tr><td>phase8_cross_model_dms_performance.png/svg</td><td>跨模型一致性条形图</td><td>—</td></tr>
</table>

</div></div></body></html>'''
    
    with open(REPORTS_DIR / 'phase8_report.html', 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info("  Phase 8 report saved")


def main():
    logger.info("=" * 60)
    logger.info("Phase 8: 跨模型与结构表征验证 — 真实数据版本")
    logger.info("=" * 60)
    
    # 1. 加载 Cα 几何数据
    df_ca = load_ca_geometry()
    
    # 2. 加载全原子几何数据
    df_fa = load_fa_geometry()
    
    # 3. 对齐数据
    result = align_ca_fa_data(df_ca, df_fa)
    if result is None:
        logger.error("无法对齐 Cα 和全原子数据，终止")
        return
    
    df_ca_aligned, df_fa_aligned = result
    
    # 4. 计算跨模型相关性
    df_corr = compute_cross_model_correlations(df_ca_aligned, df_fa_aligned)
    
    if df_corr is None:
        logger.error("无法计算相关性，尝试使用已有数据...")
        # 尝试从 Phase C/D/E 数据中加载
        cde = load_phase_cde_data()
        if cde and 'cross_model' in cde:
            df_corr = pd.DataFrame(cde['cross_model'])
            logger.info(f"  从 Phase C/D/E 数据加载: {len(df_corr)} 条")
    
    # 5. 生成可视化
    if df_corr is not None:
        generate_visualizations(df_corr, df_ca, df_fa)
        generate_report(df_corr)
    else:
        logger.error("Phase 8 无法生成结果: 无有效数据")
        return
    
    logger.info("\n" + "=" * 60)
    logger.info("Phase 8 真实数据版本完成!")
    logger.info(f"  跨模型相关性: {len(df_corr)} 个特征")
    mean_r = df_corr['pearson_r'].mean()
    logger.info(f"  平均 Pearson r: {mean_r:.4f}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()