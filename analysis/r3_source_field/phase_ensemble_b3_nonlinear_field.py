#!/usr/bin/env python3
"""
B3: 非线性场方程 F_theta(T^bio) 独立分析
=============================================
基于第二定律的非线性推广: Y^geom = F_theta(T^bio) + epsilon

在线性耦合矩阵 K (B2) 的基础上，系统性地检验非线性场方程:
  - 哪些几何量具有强非线性响应？
  - 非线性增益是否跨系统一致？
  - 特征重要性分析 (哪些 T^bio 分量驱动非线性？)

方法:
  1. Linear (Ridge CV) — 基准
  2. Polynomial (degree=2, Ridge CV) — 二次非线性
  3. GBRT (Gradient Boosting) — 通用非线性
  4. Kernel Ridge (RBF) — 核方法
  5. SVR (RBF) — 支持向量回归

输入:
  - systemwide_enhanced_geometry_v2.csv (1279序列)

输出:
  - phase_ensemble_b3_nonlinear_results_v2.csv
  - phase_ensemble_b3_feature_importance_v2.csv
  - phase_ensemble_b3_nonlinear_summary_v2.json
"""

import sys, os, json, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
FIELD_THEORY = Path(__file__).parent.parent
SCRIPTS_DIR = FIELD_THEORY / "scripts"
TABLES_DIR = FIELD_THEORY / "tables"
DATA_DIR = FIELD_THEORY / "data"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
SYSTEMWIDE = DATA_DIR / "dms/phase9_systemwide/systemwide_enhanced_geometry_v2.csv"

print("=" * 70)
print("B3: 非线性场方程 F_theta(T^bio) 独立分析")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ============================================================
# AA properties
# ============================================================
AA_PROPERTIES = {
    'G': {'hydrophobicity': -0.4, 'charge': 0, 'mw': 75.07, 'volume': 60.1, 'flexibility': 1.0},
    'A': {'hydrophobicity': 1.8, 'charge': 0, 'mw': 89.09, 'volume': 88.6, 'flexibility': 0.8},
    'S': {'hydrophobicity': -0.8, 'charge': 0, 'mw': 105.09, 'volume': 89.0, 'flexibility': 0.7},
    'E': {'hydrophobicity': -3.5, 'charge': -1, 'mw': 147.13, 'volume': 138.4, 'flexibility': 0.6},
    'L': {'hydrophobicity': 3.8, 'charge': 0, 'mw': 131.17, 'volume': 166.7, 'flexibility': 0.5},
    'K': {'hydrophobicity': -3.9, 'charge': 1, 'mw': 146.19, 'volume': 168.6, 'flexibility': 0.7},
    'V': {'hydrophobicity': 4.2, 'charge': 0, 'mw': 117.15, 'volume': 140.0, 'flexibility': 0.5},
    'I': {'hydrophobicity': 4.5, 'charge': 0, 'mw': 131.17, 'volume': 166.7, 'flexibility': 0.5},
    'F': {'hydrophobicity': 2.8, 'charge': 0, 'mw': 165.19, 'volume': 189.9, 'flexibility': 0.45},
}

CATEGORY_PROPERTIES = {
    'PolyX_original': {'is_homopolymer': 1, 'is_linker': 0, 'is_heteropolymer': 0, 'is_dms': 0, 'complexity': 0},
    'L1_hydrophobic': {'is_homopolymer': 1, 'is_linker': 0, 'is_heteropolymer': 0, 'is_dms': 0, 'complexity': 0},
    'linker': {'is_homopolymer': 0, 'is_linker': 1, 'is_heteropolymer': 0, 'is_dms': 0, 'complexity': 1},
    'HET_BLOCK': {'is_homopolymer': 0, 'is_linker': 0, 'is_heteropolymer': 1, 'is_dms': 0, 'complexity': 2},
    'HET_ALT': {'is_homopolymer': 0, 'is_linker': 0, 'is_heteropolymer': 1, 'is_dms': 0, 'complexity': 2},
    'HET_COMP': {'is_homopolymer': 0, 'is_linker': 0, 'is_heteropolymer': 1, 'is_dms': 0, 'complexity': 3},
    'HET_KAPPA': {'is_homopolymer': 0, 'is_linker': 0, 'is_heteropolymer': 1, 'is_dms': 0, 'complexity': 3},
    'HET_IDP': {'is_homopolymer': 0, 'is_linker': 0, 'is_heteropolymer': 1, 'is_dms': 0, 'complexity': 4},
    'DMS_protein': {'is_homopolymer': 0, 'is_linker': 0, 'is_heteropolymer': 0, 'is_dms': 1, 'complexity': 5},
}

# ============================================================
# Step 1: Load and prepare data
# ============================================================
print("\n[1/5] 加载数据...")

df = pd.read_csv(SYSTEMWIDE)
print(f"  系统级增强几何量: {len(df)} 序列, {len(df.columns)} 列")

# Build T^bio
def build_bio_features(data_df):
    n = len(data_df)
    features = {}
    features['n'] = data_df['n'].values
    features['log_n'] = np.log(data_df['n'].values + 1)
    features['sqrt_n'] = np.sqrt(data_df['n'].values)

    aa_types = data_df['aa_type'].fillna('unknown').values
    for prop in ['hydrophobicity', 'charge', 'mw', 'volume', 'flexibility']:
        values = np.array([AA_PROPERTIES.get(aa, {'hydrophobicity': 0, 'charge': 0, 'mw': 100, 'volume': 100, 'flexibility': 0.6})[prop] for aa in aa_types])
        features[prop] = values
        features[f'{prop}_x_n'] = values * data_df['n'].values

    features['abs_charge'] = np.abs(features['charge'])
    features['charge_x_n'] = features['charge'] * data_df['n'].values

    categories = data_df['category'].fillna('unknown').values
    for cat_key, cat_props in CATEGORY_PROPERTIES.items():
        mask = np.array([c == cat_key for c in categories], dtype=float)
        features[f'cat_{cat_key}'] = mask
        for prop_name, prop_val in cat_props.items():
            if prop_name not in features:
                features[prop_name] = np.zeros(n)
            features[prop_name] += mask * prop_val

    features['complexity_x_n'] = features['complexity'] * data_df['n'].values
    features['hydrophobicity_x_charge'] = features['hydrophobicity'] * features['charge']
    features['hydrophobicity_abs'] = np.abs(features['hydrophobicity'])

    T_bio = pd.DataFrame(features, index=data_df.index)
    T_bio = T_bio.loc[:, T_bio.std() > 1e-8]
    return T_bio

T_bio = build_bio_features(df)
print(f"  T^bio: {T_bio.shape[1]} 生物源项")

Y_COLUMNS = [
    'PR', 'A_C', 'eff_rank_95', 'spectral_decay', 'entropy',
    'total_variance', 'pseudo_volume', 'mean_rmsf', 'max_rmsf',
    'local_stiffness', 'fluct_range_ratio', 'rmsf_entropy',
    'mardia_skewness', 'mardia_kurtosis', 'corr_dim', 'mean_knn_dist',
    'condition_number', 'spectral_gap', 'spectral_gap_ratio',
    'fisher_trace', 'fisher_logdet', 'effective_diffusion', 'relaxation_time',
    'contact_order', 'clustering_coeff', 'modularity', 'betweenness_cv',
]
available_y = [c for c in Y_COLUMNS if c in df.columns]
print(f"  Y^geom: {len(available_y)} 几何观测量")

Y_geom = df[available_y].copy()
valid_mask = ~(T_bio.isna().any(axis=1) | Y_geom.isna().any(axis=1))
T_valid = T_bio[valid_mask].values
Y_valid = Y_geom[valid_mask].values
print(f"  有效序列: {valid_mask.sum()}/{len(df)}")

# Standardize
scaler_T = StandardScaler()
scaler_Y = StandardScaler()
T_scaled = scaler_T.fit_transform(T_valid)
Y_scaled = scaler_Y.fit_transform(Y_valid)

# ============================================================
# Step 2: Multi-method nonlinear regression
# ============================================================
print("\n[2/5] 多方法非线性回归 (5 methods × 36 Y)...")

cv = KFold(n_splits=5, shuffle=True, random_state=42)

methods = {
    'Linear': lambda: RidgeCV(alphas=np.logspace(-3, 3, 10)),
    'Poly2': lambda: RidgeCV(alphas=np.logspace(-3, 3, 10)),
    'GBRT': lambda: GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
    'KernelRidge': lambda: KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1),
    'SVR': lambda: SVR(kernel='rbf', C=1.0, epsilon=0.1),
}

# Pre-compute polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
T_poly = poly.fit_transform(T_scaled)

results = []
for i, y_name in enumerate(available_y):
    y = Y_scaled[:, i]
    
    # Linear
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 10))
    r2_linear = cross_val_score(ridge, T_scaled, y, cv=cv, scoring='r2').mean()
    
    # Poly2
    ridge_poly = RidgeCV(alphas=np.logspace(-3, 3, 10))
    r2_poly2 = cross_val_score(ridge_poly, T_poly, y, cv=cv, scoring='r2').mean()
    
    # GBRT
    gbrt = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    r2_gbrt = cross_val_score(gbrt, T_scaled, y, cv=cv, scoring='r2').mean()
    
    # KernelRidge (RBF)
    try:
        kr = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1)
        r2_kr = cross_val_score(kr, T_scaled, y, cv=cv, scoring='r2').mean()
    except:
        r2_kr = np.nan
    
    # SVR (RBF)
    try:
        svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        r2_svr = cross_val_score(svr, T_scaled, y, cv=cv, scoring='r2').mean()
    except:
        r2_svr = np.nan
    
    r2_values = {
        'Linear': r2_linear, 'Poly2': r2_poly2, 'GBRT': r2_gbrt,
        'KernelRidge': r2_kr, 'SVR': r2_svr
    }
    
    # Best method and nonlinear gain
    best_method = max(r2_values, key=lambda k: r2_values[k] if not np.isnan(r2_values[k]) else -np.inf)
    best_r2 = r2_values[best_method]
    nonlinear_gain = best_r2 - r2_linear
    
    # Nonlinearity classification
    if nonlinear_gain > 0.05:
        nonlinearity = 'strong'
    elif nonlinear_gain > 0.01:
        nonlinearity = 'moderate'
    else:
        nonlinearity = 'weak'
    
    results.append({
        'Y_observable': y_name,
        'R2_Linear_CV': r2_linear,
        'R2_Poly2_CV': r2_poly2,
        'R2_GBRT_CV': r2_gbrt,
        'R2_KernelRidge_CV': r2_kr,
        'R2_SVR_CV': r2_svr,
        'Best_Method': best_method,
        'Best_R2_CV': best_r2,
        'Nonlinear_Gain': nonlinear_gain,
        'Nonlinearity': nonlinearity,
    })

df_results = pd.DataFrame(results).sort_values('Nonlinear_Gain', ascending=False)
print(f"\n  非线性增益统计:")
print(f"    强非线性 (gain>0.05): {(df_results['Nonlinear_Gain'] > 0.05).sum()} 个 Y")
print(f"    中等非线性 (0.01<gain<0.05): {((df_results['Nonlinear_Gain'] > 0.01) & (df_results['Nonlinear_Gain'] <= 0.05)).sum()} 个 Y")
print(f"    弱非线性 (gain<0.01): {(df_results['Nonlinear_Gain'] <= 0.01).sum()} 个 Y")
print(f"    最佳方法分布: {df_results['Best_Method'].value_counts().to_dict()}")

# ============================================================
# Step 3: Feature importance analysis (GBRT)
# ============================================================
print("\n[3/5] 特征重要性分析 (GBRT)...")

# Fit GBRT on full data for each Y to get feature importance
feature_importance = {}
for i, y_name in enumerate(available_y):
    gbrt = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    gbrt.fit(T_scaled, Y_scaled[:, i])
    feature_importance[y_name] = dict(zip(T_bio.columns, gbrt.feature_importances_))

# Build importance matrix
df_importance = pd.DataFrame(feature_importance).T
df_importance.index.name = 'Y_observable'

# Top features per Y
print("\n  Top 3 T^bio 特征 (按GBRT重要性):")
for y_name in available_y[:10]:
    top3 = df_importance.loc[y_name].sort_values(ascending=False).head(3)
    print(f"    {y_name:25s}: {', '.join([f'{k}({v:.3f})' for k, v in top3.items()])}")

# ============================================================
# Step 4: Per-system nonlinearity analysis
# ============================================================
print("\n[4/5] 分系统非线性分析...")

system_results = []
for cat in df['category'].unique():
    cat_mask = df['category'] == cat
    cat_df = df[cat_mask]
    if len(cat_df) < 10:
        continue
    
    cat_T = build_bio_features(cat_df)
    cat_Y = cat_df[available_y].copy()
    cat_valid = ~(cat_T.isna().any(axis=1) | cat_Y.isna().any(axis=1))
    if cat_valid.sum() < 5:
        continue
    
    cat_T_val = cat_T[cat_valid].values
    cat_Y_val = cat_Y[cat_valid].values
    
    cat_scaler_T = StandardScaler()
    cat_scaler_Y = StandardScaler()
    cat_T_s = cat_scaler_T.fit_transform(cat_T_val)
    cat_Y_s = cat_scaler_Y.fit_transform(cat_Y_val)
    
    n_linear_better = 0
    n_nonlinear_better = 0
    per_y_gains = []
    
    for i, y_name in enumerate(available_y):
        y = cat_Y_s[:, i]
        
        # Linear
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 10))
        r2_lin = cross_val_score(ridge, cat_T_s, y, cv=min(5, len(cat_T_s)), scoring='r2').mean()
        
        # GBRT
        gbrt = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        r2_gbrt = cross_val_score(gbrt, cat_T_s, y, cv=min(5, len(cat_T_s)), scoring='r2').mean()
        
        if r2_gbrt > r2_lin:
            n_nonlinear_better += 1
        else:
            n_linear_better += 1
        
        per_y_gains.append(r2_gbrt - r2_lin)
    
    system_results.append({
        'category': cat,
        'n_sequences': len(cat_df),
        'n_linear_better': n_linear_better,
        'n_nonlinear_better': n_nonlinear_better,
        'nonlinear_ratio': n_nonlinear_better / (n_linear_better + n_nonlinear_better + 1e-8),
        'mean_nonlinear_gain': np.mean(per_y_gains),
        'median_nonlinear_gain': np.median(per_y_gains),
    })

df_system = pd.DataFrame(system_results).sort_values('nonlinear_ratio', ascending=False)
print(f"\n  分系统非线性比例:")
for _, row in df_system.iterrows():
    print(f"    {row['category']:20s}: {row['nonlinear_ratio']:.2%} ({row['n_nonlinear_better']}/{row['n_linear_better'] + row['n_nonlinear_better']})")

# ============================================================
# Step 5: Save results
# ============================================================
print("\n[5/5] 保存结果...")

df_results.to_csv(TABLES_DIR / 'phase_ensemble_b3_nonlinear_results_v2.csv', index=False)
df_importance.to_csv(TABLES_DIR / 'phase_ensemble_b3_feature_importance_v2.csv')
df_system.to_csv(TABLES_DIR / 'phase_ensemble_b3_per_system_nonlinear_v2.csv', index=False)

# Summary
summary = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_sequences': int(valid_mask.sum()),
    'n_T_features': T_bio.shape[1],
    'n_Y_observables': len(available_y),
    'n_strong_nonlinear': int((df_results['Nonlinear_Gain'] > 0.05).sum()),
    'n_moderate_nonlinear': int(((df_results['Nonlinear_Gain'] > 0.01) & (df_results['Nonlinear_Gain'] <= 0.05)).sum()),
    'n_weak_nonlinear': int((df_results['Nonlinear_Gain'] <= 0.01).sum()),
    'best_method_distribution': {str(k): int(v) for k, v in df_results['Best_Method'].value_counts().items()},
    'mean_nonlinear_gain': float(df_results['Nonlinear_Gain'].mean()),
    'median_nonlinear_gain': float(df_results['Nonlinear_Gain'].median()),
    'max_nonlinear_gain_Y': df_results.iloc[0]['Y_observable'],
    'max_nonlinear_gain': float(df_results.iloc[0]['Nonlinear_Gain']),
    'systems_analyzed': len(df_system),
    'most_nonlinear_system': df_system.iloc[0]['category'] if len(df_system) > 0 else None,
    'most_nonlinear_system_ratio': float(df_system.iloc[0]['nonlinear_ratio']) if len(df_system) > 0 else None,
    'top_T_features': {y: df_importance.loc[y].sort_values(ascending=False).head(3).to_dict() for y in available_y[:5]},
}

with open(TABLES_DIR / 'phase_ensemble_b3_nonlinear_summary_v2.json', 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\n{'='*70}")
print("B3 完成! 非线性场方程 F_theta(T^bio) 分析")
print(f"  非线性结果: {TABLES_DIR / 'phase_ensemble_b3_nonlinear_results_v2.csv'}")
print(f"  特征重要性: {TABLES_DIR / 'phase_ensemble_b3_feature_importance_v2.csv'}")
print(f"  分系统分析: {TABLES_DIR / 'phase_ensemble_b3_per_system_nonlinear_v2.csv'}")
print(f"\n  关键发现:")
print(f"    强非线性 Y: {(df_results['Nonlinear_Gain'] > 0.05).sum()}/{len(df_results)}")
print(f"    GBRT 最佳: {(df_results['Best_Method'] == 'GBRT').sum()}/{len(df_results)}")
print(f"    平均非线性增益: {df_results['Nonlinear_Gain'].mean():.4f}")
print(f"{'='*70}")