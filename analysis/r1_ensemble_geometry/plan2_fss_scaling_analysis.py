#!/usr/bin/env python3
"""
计划二: 有限尺寸标度分析 (Finite-Size Scaling Analysis)
=====================================================
对PolyX长链数据 (n=4-50 + n=55-100) 进行FSS分析:
  1. 标度坍缩 (Scaling Collapse)
  2. 有效指数 (Effective Exponent)
  3. FSS拟合 (非线性最小二乘)
  4. 交叉验证外推

用法:
  python field_theory/scripts/plan2_fss_scaling_analysis.py
  (需要先运行长链BioEmu采样, 并运行analyze_ensemble_geometry.py)
"""

import sys, os, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
FIG_DIR = PROJECT_ROOT / "field_theory" / "figures"
TABLE_DIR = PROJECT_ROOT / "field_theory" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════════

AA_COLORS = {'G': '#2196F3', 'S': '#4CAF50', 'E': '#FF9800', 'K': '#F44336', 'L': '#9C27B0'}
AA_ORDER = ['G', 'S', 'E', 'K', 'L']

# 几何特征列表
GEOMETRIC_FEATURES = [
    'PR', 'spectral_decay', 'entropy', 'A_C', 'top5_ratio',
    'eff_rank_95', 'total_variance', 'mean_pairwise_dist'
]

# ══════════════════════════════════════════════════════════════════════════════
# 1. 加载数据
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    """加载现有分析结果 (n=4-50) + 长链数据 (n=55-100, 如果存在)"""
    
    # 现有数据
    csv_path = PROJECT_ROOT / "test_workflow" / "polyx_ensemble" / "analysis" / "ensemble_geometry_results.csv"
    
    if not csv_path.exists():
        print(f"WARNING: {csv_path} not found, using placeholder")
        return _generate_placeholder_data()
    
    df = pd.read_csv(csv_path)
    print(f"Loaded: {len(df)} sequences from {csv_path}")
    
    # 提取PolyX数据 (G, S, E, K, L)
    polyx_mask = df['seq_id'].str.contains('PolyX_Poly[GSEKL]', na=False)
    df = df[polyx_mask].copy()
    
    # 提取n和AA
    df['n'] = df['seq_id'].str.extract(r'_(\d+)$').astype(int)
    df['aa'] = df['seq_id'].str.extract(r'Poly([GSEKL])_')
    
    print(f"  PolyX sequences: {len(df)}")
    print(f"  n range: {df['n'].min()}-{df['n'].max()}")
    print(f"  AA types: {sorted(df['aa'].unique())}")
    
    return df


def _generate_placeholder_data():
    """生成占位数据 (用于测试脚本结构)"""
    np.random.seed(42)
    data = []
    for aa in AA_ORDER:
        for n in [4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50]:
            # 模拟标度律: spectral_decay ~ n^(-0.236) + noise
            spectral_decay = n**(-0.236) * np.exp(np.random.normal(0, 0.05))
            PR = 3 + 0.016 * n + np.random.normal(0, 0.1)
            entropy = np.log(n) * 0.036 + np.random.normal(0, 0.05)
            A_C = 0.5 + np.random.normal(0, 0.02)
            
            data.append({
                'seq_id': f'PolyX_Poly{aa}_{n}',
                'aa': aa, 'n': n,
                'spectral_decay': spectral_decay,
                'PR': PR,
                'entropy': entropy,
                'A_C': A_C,
            })
    return pd.DataFrame(data)


# ══════════════════════════════════════════════════════════════════════════════
# 2. 有效指数分析 (Effective Exponent)
# ══════════════════════════════════════════════════════════════════════════════

def compute_effective_exponent(df, feature, aa):
    """计算局部标度指数 α_eff(n) = d(log Y)/d(log n)"""
    df_aa = df[df['aa'] == aa].sort_values('n').copy()
    
    if feature not in df_aa.columns:
        return None
    
    vals = df_aa[feature].values
    ns = df_aa['n'].values
    
    # 移除NaN和零值
    valid = (vals > 1e-10) & (~np.isnan(vals))
    vals = vals[valid]
    ns = ns[valid]
    
    if len(vals) < 5:
        return None
    
    # 对数差分: α_eff(n_i) = (log Y_{i+1} - log Y_{i-1}) / (log n_{i+1} - log n_{i-1})
    log_n = np.log(ns)
    log_Y = np.log(vals)
    
    alpha_eff = np.zeros(len(ns))
    alpha_eff[0] = (log_Y[1] - log_Y[0]) / (log_n[1] - log_n[0])
    alpha_eff[-1] = (log_Y[-1] - log_Y[-2]) / (log_n[-1] - log_n[-2])
    
    for i in range(1, len(ns) - 1):
        alpha_eff[i] = (log_Y[i+1] - log_Y[i-1]) / (log_n[i+1] - log_n[i-1])
    
    return ns, alpha_eff


def plot_effective_exponents(df):
    """绘制有效指数图"""
    features = ['spectral_decay', 'PR', 'entropy', 'A_C']
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    for idx, feature in enumerate(features):
        ax = axes[idx // 2, idx % 2]
        
        for aa in AA_ORDER:
            result = compute_effective_exponent(df, feature, aa)
            if result is not None:
                ns, alpha = result
                ax.plot(ns, alpha, 'o-', color=AA_COLORS[aa], label=aa, markersize=4, alpha=0.7)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('n (chain length)', fontsize=12)
        ax.set_ylabel(f'α_eff = d(log {feature})/d(log n)', fontsize=12)
        ax.set_title(f'Effective Exponent: {feature}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(FIG_DIR / "plan2_fss_effective_exponent.svg", dpi=300)
    fig.savefig(FIG_DIR / "plan2_fss_effective_exponent.jpg", dpi=300)
    plt.close()
    print("Saved: plan2_fss_effective_exponent.svg")


# ══════════════════════════════════════════════════════════════════════════════
# 3. FSS拟合 (Finite-Size Scaling Fit)
# ══════════════════════════════════════════════════════════════════════════════

def fss_model(n, Y_inf, A, Delta):
    """FSS模型: Y(n) = Y_inf + A * n^(-Delta)"""
    return Y_inf + A * n**(-Delta)


def fss_model_with_correction(n, Y_inf, A, Delta, B, omega):
    """FSS模型含修正: Y(n) = Y_inf + A * n^(-Delta) + B * n^(-Delta-omega)"""
    return Y_inf + A * n**(-Delta) + B * n**(-Delta - omega)


def fit_fss(df, feature, aa, correction=False):
    """对特定AA的特定特征进行FSS拟合"""
    df_aa = df[df['aa'] == aa].sort_values('n').copy()
    
    if feature not in df_aa.columns:
        return None
    
    vals = df_aa[feature].values
    ns = df_aa['n'].values
    
    valid = (~np.isnan(vals)) & (vals > 1e-10)
    vals = vals[valid]
    ns = ns[valid]
    
    if len(vals) < 8:  # 至少需要8个数据点用于FSS拟合
        return None
    
    try:
        # 初始猜测
        Y_inf_guess = vals[-1]  # 最大n的值作为热力学极限的猜测
        # 对于spectral_decay (负值), 使用最小值
        if np.all(vals < 0):
            Y_inf_guess = vals[-1] * 0.9
        
        # 基础模型: Y = Y_inf + A * n^(-Delta)
        popt, pcov = curve_fit(
            fss_model, ns, vals,
            p0=[Y_inf_guess, (vals[0] - Y_inf_guess) * ns[0], 0.5],
            maxfev=10000
        )
        
        Y_inf, A, Delta = popt
        Y_pred = fss_model(ns, *popt)
        residuals = vals - Y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((vals - np.mean(vals))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return {
            'aa': aa, 'feature': feature,
            'Y_inf': Y_inf, 'A': A, 'Delta': Delta,
            'R2': r2, 'n_points': len(vals),
            'ns': ns.tolist(), 'vals': vals.tolist(),
            'Y_pred': Y_pred.tolist()
        }
    except Exception as e:
        print(f"  FSS fit failed for {feature}/{aa}: {str(e)[:80]}")
        return None


def plot_fss_fits(df, feature='spectral_decay'):
    """绘制FSS拟合图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    fss_results = {}
    
    for idx, aa in enumerate(AA_ORDER):
        ax = axes[idx]
        df_aa = df[df['aa'] == aa].sort_values('n')
        
        if feature in df_aa.columns:
            vals = df_aa[feature].values
            ns = df_aa['n'].values
            valid = ~np.isnan(vals)
            ax.scatter(ns[valid], vals[valid], color=AA_COLORS[aa], s=30, zorder=5)
        
        # FSS拟合
        result = fit_fss(df, feature, aa)
        if result:
            fss_results[aa] = result
            n_fine = np.linspace(min(ns[valid]), max(ns[valid]) * 1.5, 200)
            Y_fine = fss_model(n_fine, result['Y_inf'], result['A'], result['Delta'])
            ax.plot(n_fine, Y_fine, '-', color=AA_COLORS[aa], linewidth=2, alpha=0.7)
            ax.axhline(y=result['Y_inf'], color=AA_COLORS[aa], linestyle=':', alpha=0.5)
            ax.text(0.95, 0.05, 
                    f"Y_∞={result['Y_inf']:.4f}\nΔ={result['Delta']:.3f}\nR²={result['R2']:.3f}",
                    transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        ax.set_title(f'{aa} ({feature})', fontsize=12, fontweight='bold')
        ax.set_xlabel('n', fontsize=10)
        ax.set_ylabel(feature, fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # 第6个面板: 标度坍缩
    ax = axes[5]
    for aa in AA_ORDER:
        if aa in fss_results:
            r = fss_results[aa]
            ns = np.array(r['ns'])
            vals = np.array(r['vals'])
            Y_scaled = (vals - r['Y_inf']) / abs(r['A'])
            n_scaled = ns
            ax.scatter(n_scaled, Y_scaled, color=AA_COLORS[aa], s=20, alpha=0.6, label=aa)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('n', fontsize=10)
    ax.set_ylabel(f'(Y-Y_∞)/A', fontsize=10)
    ax.set_title(f'Scaling Collapse: {feature}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"plan2_fss_{feature}_fit.svg", dpi=300)
    fig.savefig(FIG_DIR / f"plan2_fss_{feature}_fit.jpg", dpi=300)
    plt.close()
    print(f"Saved: plan2_fss_{feature}_fit.svg")
    
    return fss_results


# ══════════════════════════════════════════════════════════════════════════════
# 4. 标度坍缩分析 (Scaling Collapse)
# ══════════════════════════════════════════════════════════════════════════════

def scaling_collapse_analysis(df, feature='spectral_decay'):
    """对所有AA进行标度坍缩分析"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    collapse_data = {}
    
    for aa in AA_ORDER:
        result = fit_fss(df, feature, aa)
        if result is None:
            continue
        
        ns = np.array(result['ns'])
        vals = np.array(result['vals'])
        
        # 标度: y_scaled = (Y - Y_inf) / A, x_scaled = n
        Y_scaled = (vals - result['Y_inf']) / abs(result['A'])
        
        collapse_data[aa] = {
            'ns': ns.tolist(),
            'Y_scaled': Y_scaled.tolist(),
            'Y_inf': result['Y_inf'],
            'Delta': result['Delta'],
            'A': result['A']
        }
        
        ax.scatter(ns, Y_scaled, color=AA_COLORS[aa], s=30, alpha=0.6, label=f'{aa} (Δ={result["Delta"]:.3f})')
    
    # 理论预测: Y_scaled = n^(-Delta), 如果标度坍缩成功
    n_fine = np.logspace(np.log10(4), np.log10(200), 100)
    for Delta_guess in [0.1, 0.2, 0.3, 0.5]:
        ax.plot(n_fine, n_fine**(-Delta_guess), '--', color='gray', alpha=0.3, linewidth=1)
        ax.text(n_fine[-1], n_fine[-1]**(-Delta_guess), f'Δ={Delta_guess}', fontsize=7, alpha=0.5)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('n (chain length)', fontsize=12)
    ax.set_ylabel(f'({feature} - Y_∞) / A', fontsize=12)
    ax.set_title(f'Scaling Collapse: {feature}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"plan2_fss_scaling_collapse_{feature}.svg", dpi=300)
    fig.savefig(FIG_DIR / f"plan2_fss_scaling_collapse_{feature}.jpg", dpi=300)
    plt.close()
    print(f"Saved: plan2_fss_scaling_collapse_{feature}.svg")
    
    return collapse_data


# ══════════════════════════════════════════════════════════════════════════════
# 5. 交叉验证外推
# ══════════════════════════════════════════════════════════════════════════════

def cross_validation_extrapolation(df, feature='spectral_decay'):
    """留出最大n数据点, 检验FSS外推准确性"""
    results = {}
    
    for aa in AA_ORDER:
        df_aa = df[df['aa'] == aa].sort_values('n').copy()
        if feature not in df_aa.columns:
            continue
        
        vals = df_aa[feature].values
        ns = df_aa['n'].values
        valid = ~np.isnan(vals)
        vals = vals[valid]
        ns = ns[valid]
        
        if len(vals) < 10:
            continue
        
        # 留出最大2个n
        train_mask = np.ones(len(vals), dtype=bool)
        train_mask[-2:] = False
        
        try:
            popt, _ = curve_fit(fss_model, ns[train_mask], vals[train_mask],
                               p0=[vals[-1], vals[0] - vals[-1], 0.5], maxfev=10000)
            
            Y_pred_test = fss_model(ns[~train_mask], *popt)
            Y_true_test = vals[~train_mask]
            
            # 外推误差
            mae = np.mean(np.abs(Y_pred_test - Y_true_test))
            rel_error = np.mean(np.abs(Y_pred_test - Y_true_test) / np.abs(Y_true_test))
            
            results[aa] = {
                'Y_inf': popt[0], 'Delta': popt[1], 'A': popt[2],
                'MAE': mae, 'rel_error': rel_error,
                'n_train': ns[train_mask].tolist(),
                'n_test': ns[~train_mask].tolist(),
                'Y_true_test': Y_true_test.tolist(),
                'Y_pred_test': Y_pred_test.tolist()
            }
        except Exception as e:
            print(f"  CV extrapolation failed for {feature}/{aa}: {str(e)[:80]}")
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 6. 主函数
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("计划二: 有限尺寸标度分析 (FSS)")
    print("=" * 60)
    
    # 加载数据
    df = load_data()
    
    # 特征列表
    features = ['spectral_decay', 'PR', 'entropy', 'A_C']
    
    all_results = {}
    
    for feature in features:
        if feature not in df.columns:
            print(f"\nSKIP: {feature} not in dataframe")
            continue
        
        print(f"\n{'='*60}")
        print(f"FSS Analysis: {feature}")
        print(f"{'='*60}")
        
        # 2.1 有效指数
        print(f"\n  [2.1] Effective Exponents")
        for aa in AA_ORDER:
            result = compute_effective_exponent(df, feature, aa)
            if result:
                ns, alpha = result
                print(f"    {aa}: α_eff(n={ns[0]})={alpha[0]:.4f} → α_eff(n={ns[-1]})={alpha[-1]:.4f}")
        
        # 2.2 FSS拟合
        print(f"\n  [2.2] FSS Fits")
        fss_results = {}
        for aa in AA_ORDER:
            result = fit_fss(df, feature, aa)
            if result:
                fss_results[aa] = result
                print(f"    {aa}: Y_∞={result['Y_inf']:.4e}, Δ={result['Delta']:.3f}, R²={result['R2']:.3f}")
        
        # 2.3 交叉验证
        print(f"\n  [2.3] Cross-Validation Extrapolation")
        cv_results = cross_validation_extrapolation(df, feature)
        for aa, cv in cv_results.items():
            print(f"    {aa}: MAE={cv['MAE']:.4e}, rel_error={cv['rel_error']:.3f}")
        
        all_results[feature] = {
            'fss_fits': fss_results,
            'cv_extrapolation': cv_results
        }
    
    # 生成图表
    print(f"\n{'='*60}")
    print("生成图表")
    print(f"{'='*60}")
    
    plot_effective_exponents(df)
    
    for feature in features:
        if feature in df.columns:
            plot_fss_fits(df, feature)
            scaling_collapse_analysis(df, feature)
    
    # 保存结果
    results_path = TABLE_DIR / "plan2_fss_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved: {results_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()