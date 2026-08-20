#!/usr/bin/env python3
"""
B2: 耦合矩阵 K 系统估计 (1279序列)
=====================================
基于第二定律: Y^geom = B_0 + K · T^bio + ε

耦合矩阵 K 将生物源项 T^bio (疏水性、电荷、链长、类别等) 映射到
几何观测量 Y^geom (PR, spectral_decay, A_C, entropy, 28+指标)。

输入:
  - systemwide_enhanced_geometry_v2.csv (1279序列, 40+列增强几何量)
  - ensemble_geometry_results.csv (517 PolyX Cα 几何)
  - het_geometry_results.csv (239 异聚物几何)

输出:
  - phase_ensemble_b2_coupling_matrix_v2.csv (K 矩阵, Y_dim × T_dim)
  - phase_ensemble_b2_coupling_stats_v2.csv (每个 Y 的 R², p 值)
  - phase_ensemble_b2_coupling_heatmap_v2.csv (用于热图可视化)
  - phase_ensemble_b2_nonlinear_comparison_v2.csv (线性 vs 非线性对比)
"""

import sys, os, json, warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
FIELD_THEORY = Path(__file__).parent.parent
SCRIPTS_DIR = FIELD_THEORY / "scripts"
TABLES_DIR = FIELD_THEORY / "tables"
DATA_DIR = FIELD_THEORY / "data"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
SYSTEMWIDE = DATA_DIR / "dms/phase9_systemwide/systemwide_enhanced_geometry_v2.csv"
CA_GEOM = PROJECT_ROOT / "test_workflow/polyx_ensemble/analysis/ensemble_geometry_results.csv"
HET_GEOM = PROJECT_ROOT / "test_workflow/heteropolymer_ensemble/analysis/het_geometry/het_geometry_results.csv"

print("=" * 70)
print("B2: 耦合矩阵 K 系统估计 (第二定律)")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ============================================================
# Amino acid properties
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
    'R': {'hydrophobicity': -4.5, 'charge': 1, 'mw': 174.20, 'volume': 173.4, 'flexibility': 0.65},
    'D': {'hydrophobicity': -3.5, 'charge': -1, 'mw': 133.10, 'volume': 111.1, 'flexibility': 0.65},
    'N': {'hydrophobicity': -3.5, 'charge': 0, 'mw': 132.12, 'volume': 114.1, 'flexibility': 0.65},
    'Q': {'hydrophobicity': -3.5, 'charge': 0, 'mw': 146.15, 'volume': 143.8, 'flexibility': 0.6},
    'H': {'hydrophobicity': -3.2, 'charge': 0.5, 'mw': 155.16, 'volume': 153.2, 'flexibility': 0.65},
    'P': {'hydrophobicity': -1.6, 'charge': 0, 'mw': 115.13, 'volume': 112.7, 'flexibility': 0.55},
    'T': {'hydrophobicity': -0.7, 'charge': 0, 'mw': 119.12, 'volume': 116.1, 'flexibility': 0.65},
    'W': {'hydrophobicity': -0.9, 'charge': 0, 'mw': 204.23, 'volume': 227.8, 'flexibility': 0.55},
    'Y': {'hydrophobicity': -1.3, 'charge': 0, 'mw': 181.19, 'volume': 193.6, 'flexibility': 0.55},
    'C': {'hydrophobicity': 2.5, 'charge': 0, 'mw': 121.16, 'volume': 108.5, 'flexibility': 0.65},
    'M': {'hydrophobicity': 1.9, 'charge': 0, 'mw': 149.21, 'volume': 162.9, 'flexibility': 0.55},
}

# Category-level properties
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
print("\n[1/6] 加载数据...")

df = pd.read_csv(SYSTEMWIDE)
print(f"  系统级增强几何量: {len(df)} 序列, {len(df.columns)} 列")

# Build biological source terms T^bio
def build_bio_features(df):
    """Build T^bio feature matrix from sequence metadata"""
    n = len(df)
    features = {}
    
    # Basic features
    features['n'] = df['n'].values  # Chain length
    features['log_n'] = np.log(df['n'].values + 1)
    features['sqrt_n'] = np.sqrt(df['n'].values)
    
    # AA-type derived features
    aa_types = df['aa_type'].fillna('unknown').values
    for prop in ['hydrophobicity', 'charge', 'mw', 'volume', 'flexibility']:
        values = np.array([AA_PROPERTIES.get(aa, {'hydrophobicity': 0, 'charge': 0, 'mw': 100, 'volume': 100, 'flexibility': 0.6})[prop] for aa in aa_types])
        features[prop] = values
        # Interaction with n
        features[f'{prop}_x_n'] = values * df['n'].values
        features[f'{prop}_x_log_n'] = values * np.log(df['n'].values + 1)
    
    # Charge-related
    features['abs_charge'] = np.abs(features['charge'])
    features['charge_x_n'] = features['charge'] * df['n'].values
    
    # Category features
    categories = df['category'].fillna('unknown').values
    for cat_key, cat_props in CATEGORY_PROPERTIES.items():
        mask = np.array([c == cat_key for c in categories], dtype=float)
        features[f'cat_{cat_key}'] = mask
        for prop_name, prop_val in cat_props.items():
            if prop_name not in features:
                features[prop_name] = np.zeros(n)
            features[prop_name] += mask * prop_val
    
    # Interaction features
    features['complexity_x_n'] = features['complexity'] * df['n'].values
    features['hydrophobicity_x_charge'] = features['hydrophobicity'] * features['charge']
    features['hydrophobicity_abs'] = np.abs(features['hydrophobicity'])
    
    # Build DataFrame
    T_bio = pd.DataFrame(features, index=df.index)
    
    # Remove constant columns
    T_bio = T_bio.loc[:, T_bio.std() > 1e-8]
    
    return T_bio

print("\n[2/6] 构建生物源项 T^bio...")
T_bio = build_bio_features(df)
print(f"  T^bio 维度: {T_bio.shape[1]} 个生物源项 (共 {len(T_bio)} 序列)")

# Geometric observables Y^geom
Y_COLUMNS = [
    'PR', 'A_C', 'eff_rank_95', 'eff_rank_99', 'spectral_decay', 'entropy',
    'total_variance', 'pseudo_volume', 'mean_rmsf', 'max_rmsf', 'rmsf_cv',
    'local_stiffness', 'fluct_range_ratio', 'rmsf_entropy',
    'mardia_skewness', 'mardia_kurtosis', 'mean_pc_skewness', 'mean_pc_kurtosis',
    'corr_dim', 'mean_knn_dist', 'cv_knn_dist',
    'condition_number', 'spectral_gap', 'spectral_gap_ratio',
    'fisher_trace', 'fisher_logdet', 'mean_mi_pc3', 'js_divergence',
    'effective_diffusion', 'relaxation_time', 'lyapunov_proxy', 'convective_ratio',
    'contact_order', 'clustering_coeff', 'modularity', 'betweenness_cv'
]

# Filter to available columns
available_y = [c for c in Y_COLUMNS if c in df.columns]
print(f"  Y^geom 维度: {len(available_y)} 个几何观测量")

Y_geom = df[available_y].copy()

# Remove rows with NaN
valid_mask = ~(T_bio.isna().any(axis=1) | Y_geom.isna().any(axis=1))
T_bio_valid = T_bio[valid_mask].values
Y_geom_valid = Y_geom[valid_mask].values
print(f"  有效序列: {valid_mask.sum()}/{len(df)}")

# ============================================================
# Step 3: Linear coupling matrix K (Ridge regression)
# ============================================================
print("\n[3/6] 线性耦合矩阵 K 估计 (Ridge 回归)...")

# Standardize
scaler_T = StandardScaler()
scaler_Y = StandardScaler()
T_scaled = scaler_T.fit_transform(T_bio_valid)
Y_scaled = scaler_Y.fit_transform(Y_geom_valid)

# Ridge CV to find optimal alpha
ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 20))
ridge_cv.fit(T_scaled, Y_scaled)
print(f"  最优 alpha: {ridge_cv.alpha_:.4f}")

# Fit with optimal alpha
ridge = Ridge(alpha=ridge_cv.alpha_)
ridge.fit(T_scaled, Y_scaled)

# K matrix: (n_Y, n_T)
K_matrix = ridge.coef_  # Shape: (n_Y_obs, n_T)
K_df = pd.DataFrame(
    K_matrix,
    index=available_y,
    columns=T_bio.columns.tolist()
)

# R² per Y
Y_pred = ridge.predict(T_scaled)
r2_per_y = {}
for i, y_name in enumerate(available_y):
    r2_per_y[y_name] = r2_score(Y_scaled[:, i], Y_pred[:, i])

# Cross-validated R²
cv_scores = {}
for i, y_name in enumerate(available_y):
    scores = cross_val_score(Ridge(alpha=ridge_cv.alpha_), T_scaled, Y_scaled[:, i], cv=5, scoring='r2')
    cv_scores[y_name] = (scores.mean(), scores.std())

# Build statistics
coupling_stats = []
for y_name in available_y:
    coupling_stats.append({
        'Y_observable': y_name,
        'R2_linear': r2_per_y[y_name],
        'R2_cv_mean': cv_scores[y_name][0],
        'R2_cv_std': cv_scores[y_name][1],
        'top_T_feature': T_bio.columns[np.argmax(np.abs(K_matrix[available_y.index(y_name)]))],
        'top_K_value': np.max(np.abs(K_matrix[available_y.index(y_name)])),
    })

df_coupling_stats = pd.DataFrame(coupling_stats)
df_coupling_stats = df_coupling_stats.sort_values('R2_linear', ascending=False)

print(f"\n  线性 K 矩阵 R² 统计:")
print(f"    最高 R²: {df_coupling_stats['R2_linear'].max():.4f} ({df_coupling_stats.iloc[0]['Y_observable']})")
print(f"    中位 R²: {df_coupling_stats['R2_linear'].median():.4f}")
print(f"    显著 R² (R²>0.1): {(df_coupling_stats['R2_linear'] > 0.1).sum()}/{len(df_coupling_stats)}")

# ============================================================
# Step 4: Nonlinear field equation (GBRT)
# ============================================================
print("\n[4/6] 非线性场方程 F_θ(T^bio) (Gradient Boosting)...")

nonlinear_r2 = {}
for i, y_name in enumerate(available_y):
    gbr = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    cv_scores_nl = cross_val_score(gbr, T_scaled, Y_scaled[:, i], cv=5, scoring='r2')
    nonlinear_r2[y_name] = (cv_scores_nl.mean(), cv_scores_nl.std())

# Polynomial features (degree=2)
poly = PolynomialFeatures(degree=2, include_bias=False)
T_poly = poly.fit_transform(T_scaled)
poly_r2 = {}
for i, y_name in enumerate(available_y):
    ridge_poly = RidgeCV(alphas=np.logspace(-3, 3, 10))
    cv_scores_poly = cross_val_score(ridge_poly, T_poly, Y_scaled[:, i], cv=5, scoring='r2')
    poly_r2[y_name] = (cv_scores_poly.mean(), cv_scores_poly.std())

# Build comparison
nonlinear_comparison = []
for y_name in available_y:
    nonlinear_comparison.append({
        'Y_observable': y_name,
        'R2_linear': r2_per_y[y_name],
        'R2_linear_cv': cv_scores[y_name][0],
        'R2_poly2_cv': poly_r2[y_name][0],
        'R2_gbrt_cv': nonlinear_r2[y_name][0],
        'delta_poly2': poly_r2[y_name][0] - cv_scores[y_name][0],
        'delta_gbrt': nonlinear_r2[y_name][0] - cv_scores[y_name][0],
        'best_method': 'GBRT' if nonlinear_r2[y_name][0] > max(cv_scores[y_name][0], poly_r2[y_name][0])
                       else ('Poly2' if poly_r2[y_name][0] > cv_scores[y_name][0] else 'Linear'),
        'best_R2': max(cv_scores[y_name][0], poly_r2[y_name][0], nonlinear_r2[y_name][0]),
    })

df_nonlinear_comparison = pd.DataFrame(nonlinear_comparison)
df_nonlinear_comparison = df_nonlinear_comparison.sort_values('best_R2', ascending=False)

print(f"\n  非线性 vs 线性对比:")
print(f"    线性中位 R²(cv): {np.median([cv_scores[y][0] for y in available_y]):.4f}")
print(f"    多项式中位 R²(cv): {np.median([poly_r2[y][0] for y in available_y]):.4f}")
print(f"    GBRT中位 R²(cv): {np.median([nonlinear_r2[y][0] for y in available_y]):.4f}")
n_gbrt_better = sum(1 for y in available_y if nonlinear_r2[y][0] > cv_scores[y][0])
print(f"    GBRT优于线性: {n_gbrt_better}/{len(available_y)}")

# ============================================================
# Step 5: Per-category coupling analysis
# ============================================================
print("\n[5/6] 分系统耦合分析...")

category_coupling = []
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
    
    cat_T_val = cat_T[cat_valid]
    cat_Y_val = cat_Y[cat_valid]
    
    # Ridge
    cat_scaler_T = StandardScaler()
    cat_scaler_Y = StandardScaler()
    cat_T_s = cat_scaler_T.fit_transform(cat_T_val)
    cat_Y_s = cat_scaler_Y.fit_transform(cat_Y_val)
    
    cat_ridge = RidgeCV(alphas=np.logspace(-3, 3, 10))
    cat_ridge.fit(cat_T_s, cat_Y_s)
    cat_Y_pred = cat_ridge.predict(cat_T_s)
    
    for i, y_name in enumerate(available_y):
        r2 = r2_score(cat_Y_s[:, i], cat_Y_pred[:, i])
        category_coupling.append({
            'category': cat,
            'Y_observable': y_name,
            'R2': r2,
            'n_sequences': cat_valid.sum(),
        })

df_cat_coupling = pd.DataFrame(category_coupling)

# ============================================================
# Step 6: Save results
# ============================================================
print("\n[6/6] 保存结果...")

K_df.to_csv(TABLES_DIR / 'phase_ensemble_b2_coupling_matrix_v2.csv')
df_coupling_stats.to_csv(TABLES_DIR / 'phase_ensemble_b2_coupling_stats_v2.csv', index=False)
df_nonlinear_comparison.to_csv(TABLES_DIR / 'phase_ensemble_b2_nonlinear_comparison_v2.csv', index=False)
df_cat_coupling.to_csv(TABLES_DIR / 'phase_ensemble_b2_per_category_coupling_v2.csv', index=False)

# Save heatmap data
K_heatmap = K_df.copy()
K_heatmap.to_csv(TABLES_DIR / 'phase_ensemble_b2_coupling_heatmap_v2.csv')

# Summary JSON
summary = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_sequences': int(valid_mask.sum()),
    'n_T_features': T_bio.shape[1],
    'n_Y_observables': len(available_y),
    'T_features': T_bio.columns.tolist(),
    'Y_observables': available_y,
    'linear_median_R2': float(df_coupling_stats['R2_linear'].median()),
    'linear_max_R2': float(df_coupling_stats['R2_linear'].max()),
    'linear_max_Y': df_coupling_stats.iloc[0]['Y_observable'],
    'nonlinear_median_R2': float(df_nonlinear_comparison['best_R2'].median()),
    'nonlinear_max_R2': float(df_nonlinear_comparison['best_R2'].max()),
    'gbrt_better_count': n_gbrt_better,
    'gbrt_better_total': len(available_y),
    'n_linear_significant': int((df_coupling_stats['R2_linear'] > 0.1).sum()),
    'category_count': len(df_cat_coupling['category'].unique()),
}

with open(TABLES_DIR / 'phase_ensemble_b2_summary_v2.json', 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# Top 10 most predictable Y
print(f"\n{'='*70}")
print("B2 完成! 耦合矩阵 K 系统估计")
print(f"  线性 K 矩阵: {TABLES_DIR / 'phase_ensemble_b2_coupling_matrix_v2.csv'}")
print(f"  耦合统计: {TABLES_DIR / 'phase_ensemble_b2_coupling_stats_v2.csv'}")
print(f"  非线性对比: {TABLES_DIR / 'phase_ensemble_b2_nonlinear_comparison_v2.csv'}")
print(f"  分系统耦合: {TABLES_DIR / 'phase_ensemble_b2_per_category_coupling_v2.csv'}")
print(f"\n  Top 10 最可预测几何量:")
for _, row in df_coupling_stats.head(10).iterrows():
    print(f"    {row['Y_observable']:25s} R²={row['R2_linear']:.4f} (top: {row['top_T_feature']})")
print(f"{'='*70}")