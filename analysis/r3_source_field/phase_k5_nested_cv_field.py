#!/usr/bin/env python3
"""
K5: Law 2 L3 非线性场方程 — 嵌套CV重做 (替代已撤回GBRT结论)
================================================================
目标: 对 B3 非线性场方程分析 (phase_ensemble_b3_nonlinear_field.py v2) 做统计严谨的
嵌套交叉验证重做, 替代此前基于 in-sample R² 的 GBRT 结论 (审计项 C4: 37.2% 声明撤回)。

方法论修正:
  1. 嵌套CV: 外层5-fold (无偏R²估计) × 内层3-fold GridSearchCV (超参选择)
     — v2 对 Linear/Poly2 隐含嵌套 (RidgeCV), 但 GBRT/KRR/SVR 用固定超参, 且
       "强非线性"分类部分受 in-sample 结论污染。本脚本对全部5方法统一嵌套。
  2. 双外层方案:
     (a) Random 5-fold (shuffle, seed=42) — 系统内插值泛化
     (b) GroupKFold(5) by aa_type — 跨AA外推泛化 (回应 v2 随机KFold与
         早期AA分组CV (Linear CV R²=-1.47) 结论不一致的问题)
  3. 数据升级: systemwide_enhanced_geometry_v5.csv (1304序列, 含D2长链)
     — v2 基于 v2.csv (763有效序列)
  4. 仅报告外层CV R² (无偏); in-sample R² 仅用于过拟合间隙诊断, 不作结论依据。

判定标准 (与 v2 对齐, 保证可比性):
  - 非线性增益 = max(非线性方法外层R²) - Linear外层R²
  - 强非线性: gain > 0.1; 中等: 0.05 < gain <= 0.1; 弱: gain <= 0.05
  - 附 fold 级一致性 (增益符号一致的fold数) 与 Wilcoxon 符号秩检验 (n=5, 仅供参考)

输入: field_theory/data/dms/phase9_systemwide/systemwide_enhanced_geometry_v5.csv
输出:
  - field_theory/tables/phase_k5_nested_cv_results.csv   (Y × method × scheme)
  - field_theory/tables/phase_k5_nested_cv_summary.json  (判定 + v2对比)
  - field_theory/tables/phase_k5_fold_level.csv          (fold级R²明细)

用法: python field_theory/scripts/phase_k5_nested_cv_field.py
预计: ~20-40min (纯CPU)
"""

import json
import os
import warnings
from datetime import datetime
from pathlib import Path

# joblib resource_tracker 无法处理中文临时路径 → 强制ASCII临时目录
# (必须在 import sklearn/joblib 之前设置)
_JOBLIB_TMP = r'B:\2026\Exploration\.joblib_tmp'
os.makedirs(_JOBLIB_TMP, exist_ok=True)
os.environ['JOBLIB_TEMP_FOLDER'] = _JOBLIB_TMP

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
FIELD_THEORY = Path(__file__).parent.parent
TABLES_DIR = FIELD_THEORY / "tables"
DATA_DIR = FIELD_THEORY / "data"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

SYSTEMWIDE = DATA_DIR / "dms/phase9_systemwide/systemwide_enhanced_geometry_v5.csv"
V2_RESULTS = TABLES_DIR / "phase_ensemble_b3_nonlinear_results_v2.csv"

OUT_RESULTS = TABLES_DIR / "phase_k5_nested_cv_results.csv"
OUT_FOLDS = TABLES_DIR / "phase_k5_fold_level.csv"
OUT_SUMMARY = TABLES_DIR / "phase_k5_nested_cv_summary.json"

RANDOM_STATE = 42

# ============================================================
# T^bio 构建 (与 phase_ensemble_b3_nonlinear_field.py v2 完全一致)
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
DEFAULT_AA = {'hydrophobicity': 0, 'charge': 0, 'mw': 100, 'volume': 100, 'flexibility': 0.6}

CATEGORY_PROPERTIES = {
    'PolyX': {'is_homopolymer': 1, 'is_linker': 0, 'is_heteropolymer': 0, 'is_dms': 0, 'complexity': 0},
    'PolyX_original': {'is_homopolymer': 1, 'is_linker': 0, 'is_heteropolymer': 0, 'is_dms': 0, 'complexity': 0},
    'PolyX_longchain': {'is_homopolymer': 1, 'is_linker': 0, 'is_heteropolymer': 0, 'is_dms': 0, 'complexity': 0},
    'L1_hydrophobic': {'is_homopolymer': 1, 'is_linker': 0, 'is_heteropolymer': 0, 'is_dms': 0, 'complexity': 0},
    'linker': {'is_homopolymer': 0, 'is_linker': 1, 'is_heteropolymer': 0, 'is_dms': 0, 'complexity': 1},
    'HET_BLOCK': {'is_homopolymer': 0, 'is_linker': 0, 'is_heteropolymer': 1, 'is_dms': 0, 'complexity': 2},
    'HET_ALT': {'is_homopolymer': 0, 'is_linker': 0, 'is_heteropolymer': 1, 'is_dms': 0, 'complexity': 2},
    'HET_COMP': {'is_homopolymer': 0, 'is_linker': 0, 'is_heteropolymer': 1, 'is_dms': 0, 'complexity': 3},
    'HET_KAPPA': {'is_homopolymer': 0, 'is_linker': 0, 'is_heteropolymer': 1, 'is_dms': 0, 'complexity': 3},
    'HET_IDP': {'is_homopolymer': 0, 'is_linker': 0, 'is_heteropolymer': 1, 'is_dms': 0, 'complexity': 4},
    'DMS_protein': {'is_homopolymer': 0, 'is_linker': 0, 'is_heteropolymer': 0, 'is_dms': 1, 'complexity': 5},
}

Y_COLUMNS = [
    'PR', 'A_C', 'eff_rank_95', 'spectral_decay', 'entropy',
    'total_variance', 'pseudo_volume', 'mean_rmsf', 'max_rmsf',
    'local_stiffness', 'fluct_range_ratio', 'rmsf_entropy',
    'mardia_skewness', 'mardia_kurtosis', 'corr_dim', 'mean_knn_dist',
    'condition_number', 'spectral_gap', 'spectral_gap_ratio',
    'fisher_trace', 'fisher_logdet', 'effective_diffusion', 'relaxation_time',
    'contact_order', 'clustering_coeff', 'modularity', 'betweenness_cv',
]


def build_bio_features(data_df):
    n = len(data_df)
    features = {}
    features['n'] = data_df['n'].values
    features['log_n'] = np.log(data_df['n'].values + 1)
    features['sqrt_n'] = np.sqrt(data_df['n'].values)

    aa_types = data_df['aa_type'].fillna('unknown').values
    for prop in ['hydrophobicity', 'charge', 'mw', 'volume', 'flexibility']:
        values = np.array([AA_PROPERTIES.get(aa, DEFAULT_AA)[prop] for aa in aa_types])
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


# ============================================================
# 嵌套CV机器
# ============================================================
def make_method_grids():
    """返回 {method_name: (pipeline, param_grid)}"""
    return {
        'Linear': (
            Pipeline([('sc', StandardScaler()), ('m', Ridge())]),
            {'m__alpha': np.logspace(-3, 3, 7)},
        ),
        'Poly2': (
            Pipeline([('sc', StandardScaler()), ('pf', PolynomialFeatures(degree=2, include_bias=False)), ('m', Ridge())]),
            {'m__alpha': np.logspace(-3, 3, 7)},
        ),
        'GBRT': (
            Pipeline([('sc', StandardScaler()),
                      ('m', GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE))]),
            {'m__max_depth': [2, 3], 'm__learning_rate': [0.05, 0.1]},
        ),
        'KernelRidge': (
            Pipeline([('sc', StandardScaler()), ('m', KernelRidge(kernel='rbf'))]),
            {'m__alpha': [0.1, 1.0, 10.0], 'm__gamma': [0.01, 0.1]},
        ),
        'SVR': (
            Pipeline([('sc', StandardScaler()), ('m', SVR(kernel='rbf'))]),
            {'m__C': [1.0, 10.0], 'm__gamma': ['scale', 0.1]},
        ),
    }


def nested_cv_single_y(X, y, groups, method_grids):
    """
    对单个Y执行双方案嵌套CV。
    返回 rows (list[dict]) — 每方法×方案一行, 含fold级R²。
    """
    rows = []
    schemes = {
        'random': list(KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(X)),
        'grouped_aa': list(GroupKFold(n_splits=5).split(X, groups=groups)),
    }

    for scheme_name, folds in schemes.items():
        for method_name, (pipe, grid) in method_grids.items():
            fold_r2 = []
            for tr_idx, te_idx in folds:
                X_tr, X_te = X[tr_idx], X[te_idx]
                y_tr, y_te = y[tr_idx], y[te_idx]
                # 内层3-fold网格搜索 (random方案内层按随机; grouped方案内层按组)
                if scheme_name == 'grouped_aa':
                    inner_cv = GroupKFold(n_splits=3)
                    inner_split = inner_cv.split(X_tr, y_tr, groups[tr_idx])
                else:
                    inner_cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
                    inner_split = inner_cv.split(X_tr)
                gs = GridSearchCV(clone(pipe), grid, cv=inner_split,
                                  scoring='r2', n_jobs=-1, refit=True)
                gs.fit(X_tr, y_tr)
                r2 = gs.score(X_te, y_te)
                fold_r2.append(float(r2))
            rows.append({
                'method': method_name,
                'scheme': scheme_name,
                'r2_mean': float(np.mean(fold_r2)),
                'r2_std': float(np.std(fold_r2)),
                'fold_r2': fold_r2,
            })
    return rows


def main():
    print("=" * 70)
    print("K5: Law 2 L3 非线性场方程 — 嵌套CV重做")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ---------------- 数据 ----------------
    print("\n[1/4] 加载数据 v5 ...")
    df = pd.read_csv(SYSTEMWIDE)
    print(f"  总序列: {len(df)}")

    T_bio = build_bio_features(df)
    print(f"  T^bio: {T_bio.shape[1]} 特征")

    groups = df['aa_type'].fillna('unknown').values
    available_y = [c for c in Y_COLUMNS if c in df.columns]
    print(f"  Y^geom: {len(available_y)} 观测量")

    method_grids = make_method_grids()

    # ---------------- 嵌套CV ----------------
    print("\n[2/4] 嵌套CV (27 Y × 5 methods × 2 schemes) ...")
    all_rows = []
    fold_rows = []
    T_all = T_bio.values

    for yi, y_name in enumerate(available_y):
        y_full = df[y_name].values
        mask = np.isfinite(y_full) & np.isfinite(T_all).all(axis=1)
        X = T_all[mask]
        y = y_full[mask]
        g = groups[mask]
        # GroupKFold要求每组>=? 仅需组数>=5
        if len(np.unique(g)) < 5 or len(y) < 50:
            print(f"  [{yi+1}/{len(available_y)}] {y_name}: 跳过 (n={len(y)}, groups={len(np.unique(g))})")
            continue

        rows = nested_cv_single_y(X, y, g, method_grids)
        for r in rows:
            r['Y'] = y_name
            r['n_samples'] = int(len(y))
            all_rows.append(r)
            for fi, fr2 in enumerate(r['fold_r2']):
                fold_rows.append({'Y': y_name, 'method': r['method'],
                                  'scheme': r['scheme'], 'fold': fi, 'r2': fr2})
        lin = next(r['r2_mean'] for r in rows if r['method'] == 'Linear' and r['scheme'] == 'random')
        best_nl = max((r['r2_mean'] for r in rows
                       if r['method'] != 'Linear' and r['scheme'] == 'random'), default=np.nan)
        print(f"  [{yi+1}/{len(available_y)}] {y_name}: n={len(y)}, "
              f"Linear={lin:.3f}, bestNL={best_nl:.3f}, gain={best_nl-lin:+.3f}")

    res_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'fold_r2'} for r in all_rows])
    res_df.to_csv(OUT_RESULTS, index=False)
    pd.DataFrame(fold_rows).to_csv(OUT_FOLDS, index=False)
    print(f"\n  保存: {OUT_RESULTS.name} ({len(res_df)} 行), {OUT_FOLDS.name} ({len(fold_rows)} 行)")

    # ---------------- 判定 ----------------
    print("\n[3/4] 非线性增益判定 ...")
    verdicts = []
    for y_name in res_df['Y'].unique():
        sub = res_df[res_df['Y'] == y_name]
        for scheme in ['random', 'grouped_aa']:
            s = sub[sub['scheme'] == scheme]
            lin = s[s['method'] == 'Linear']['r2_mean']
            if len(lin) == 0:
                continue
            lin_r2 = float(lin.iloc[0])
            nl = s[s['method'] != 'Linear']
            best_row = nl.loc[nl['r2_mean'].idxmax()]
            gain = float(best_row['r2_mean'] - lin_r2)

            # fold级一致性 + Wilcoxon (n=5)
            f_sub = pd.DataFrame(fold_rows)
            f_lin = f_sub[(f_sub['Y'] == y_name) & (f_sub['method'] == 'Linear')
                          & (f_sub['scheme'] == scheme)].sort_values('fold')['r2'].values
            f_best = f_sub[(f_sub['Y'] == y_name) & (f_sub['method'] == best_row['method'])
                           & (f_sub['scheme'] == scheme)].sort_values('fold')['r2'].values
            diff = f_best - f_lin
            n_pos = int((diff > 0).sum())
            try:
                p_wil = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
            except Exception:
                p_wil = np.nan

            strength = ('strong' if gain > 0.1 else 'moderate' if gain > 0.05 else 'weak')
            verdicts.append({
                'Y': y_name, 'scheme': scheme,
                'linear_r2': lin_r2,
                'best_nl_method': best_row['method'],
                'best_nl_r2': float(best_row['r2_mean']),
                'gain': gain, 'strength': strength,
                'fold_consistency': f"{n_pos}/5",
                'wilcoxon_p': p_wil,
            })

    verdict_df = pd.DataFrame(verdicts)
    n_strong_random = ((verdict_df['scheme'] == 'random') & (verdict_df['strength'] == 'strong')).sum()
    n_strong_grouped = ((verdict_df['scheme'] == 'grouped_aa') & (verdict_df['strength'] == 'strong')).sum()

    # v2 对比
    v2_compare = None
    if V2_RESULTS.exists():
        v2 = pd.read_csv(V2_RESULTS)
        v2_compare = {'available': True, 'n_rows': len(v2)}
    else:
        v2_compare = {'available': False}

    summary = {
        'timestamp': datetime.now().isoformat(),
        'script': 'phase_k5_nested_cv_field.py v1.0',
        'data': {'file': SYSTEMWIDE.name, 'n_sequences_total': int(len(df)),
                 'n_T_features': int(T_bio.shape[1]), 'n_Y': len(available_y)},
        'method': {
            'outer_cv': '5-fold (random shuffle seed=42) + GroupKFold(5) by aa_type',
            'inner_cv': '3-fold GridSearchCV (见 make_method_grids)',
            'note': '全部5方法统一嵌套; 仅外层R²用于结论; 替代v2固定超参与in-sample结论',
        },
        'verdicts': verdicts,
        'counts': {
            'strong_nonlinear_random': int(n_strong_random),
            'strong_nonlinear_grouped_aa': int(n_strong_grouped),
            'n_Y': len(verdict_df['Y'].unique()),
        },
        'v2_baseline': v2_compare,
    }
    with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"  保存: {OUT_SUMMARY.name}")

    # ---------------- 打印结论 ----------------
    print("\n[4/4] 结论摘要")
    print("=" * 70)
    print(f"强非线性 (random CV):   {n_strong_random}/{len(verdict_df['Y'].unique())}")
    print(f"强非线性 (grouped-AA):  {n_strong_grouped}/{len(verdict_df['Y'].unique())}")
    print("\n关键Y的嵌套CV结果 (random / grouped):")
    key_y = ['spectral_decay', 'total_variance', 'PR', 'entropy', 'A_C', 'fisher_trace']
    for y_name in key_y:
        v = verdict_df[verdict_df['Y'] == y_name]
        for _, r in v.iterrows():
            print(f"  {y_name:18s} [{r['scheme']:10s}] Linear={r['linear_r2']:+.3f} "
                  f"best={r['best_nl_method']:12s}{r['best_nl_r2']:+.3f} "
                  f"gain={r['gain']:+.3f} ({r['strength']}, fold一致{r['fold_consistency']})")
    print("\nDone.")


if __name__ == '__main__':
    main()
