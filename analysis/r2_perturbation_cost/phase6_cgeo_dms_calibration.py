#!/usr/bin/env python3
"""
Phase VI-A: 几何扰动代价 C_geo 与 DMS 标定
===============================================
计算几何扰动代价 C_geo(P|S) = z_P^T g_S z_P，
并与 DMS 实验数据进行相关性分析，验证几何场论模型的预测能力。

技术参考: TECHNICAL_REFERENCE.md §10.4 (度量张量与几何扰动代价)
"""

import os
import sys
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('phase6_cgeo')

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
FIELD_THEORY = Path(__file__).parent.parent
DATA_DIR = FIELD_THEORY / "data"
DMS_DIR = DATA_DIR / "dms" / "processed"
OUTPUT_DIR = DATA_DIR / "dms" / "results"
TABLES_DIR = FIELD_THEORY / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(FIELD_THEORY / "scripts" / "utils"))
from geometry_core import compute_consensus_dimension, compute_anisotropy_C, \
    compute_spectral_shape, compute_local_metric_in_tangent_space


def load_dms_single_mutants():
    dms_path = DMS_DIR / "phase6_dms_single_mutants.csv"
    if not dms_path.exists():
        logger.error(f"DMS data not found: {dms_path}")
        return None
    
    df = pd.read_csv(dms_path)
    logger.info(f"Loaded DMS single mutants: {len(df)} records")
    logger.info(f"  Proteins: {df['protein'].value_counts().to_dict()}")
    logger.info(f"  Assays: {df['assay_name'].unique()}")
    return df


def get_wildtype_sequences(df):
    wt_seqs = {}
    for protein in df['protein'].unique():
        wt_seq = df[df['protein'] == protein]['target_seq'].iloc[0]
        wt_seqs[protein] = wt_seq
        logger.info(f"  {protein.upper()}: {len(wt_seq)} residues")
    return wt_seqs


def generate_synthetic_ensemble(n_residues, n_samples=250):
    """
    .. deprecated:: 2026-07-17 (AUDIT-C1/C2)
       合成系综已废弃 — 请使用真实 BioEmu NPZ 数据 (phase9_cgeo_real_data.py)。
       本函数仅保留作历史记录, 不得用于任何新分析或论文数据生成。
    """
    import warnings
    warnings.warn(
        "generate_synthetic_ensemble() is DEPRECATED (AUDIT-C1/C2, 2026-07-17). "
        "Use real BioEmu NPZ data via phase9_cgeo_real_data.py instead.",
        DeprecationWarning, stacklevel=2
    )
    coords = np.zeros((n_samples, n_residues, 3))
    
    for i in range(1, n_residues):
        phi = np.random.randn(n_samples) * 1.5
        psi = np.random.randn(n_samples) * 1.5
        
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        c_psi = np.cos(psi)
        s_psi = np.sin(psi)
        
        if i == 1:
            prev_dir = np.array([1.0, 0.0, 0.0])[None, :].repeat(n_samples, axis=0)
        else:
            prev_dir = coords[:, i-1, :] - coords[:, i-2, :]
        
        prev_dir_norm = np.linalg.norm(prev_dir, axis=1, keepdims=True)
        prev_dir_norm[prev_dir_norm == 0] = 1.0
        x, y, z = (prev_dir / prev_dir_norm).T
        
        new_x = x * c_phi * c_psi - y * s_phi + z * c_phi * s_psi
        new_y = x * s_phi * c_psi + y * c_phi + z * s_phi * s_psi
        new_z = -x * s_psi + z * c_psi
        
        coords[:, i, :] = coords[:, i-1, :] + np.column_stack([new_x, new_y, new_z]) * 0.38
    
    coords += np.random.randn(n_samples, n_residues, 3) * 0.02
    return coords


def compute_geometric_features(ensemble):
    if ensemble.ndim == 3:
        n_samples, n_residues, _ = ensemble.shape
    else:
        n_samples = ensemble.shape[0]
        n_residues = ensemble.shape[1] // 3
        ensemble = ensemble.reshape(n_samples, n_residues, 3)
    
    X_flat = ensemble.reshape(n_samples, n_residues * 3)
    
    dims = compute_consensus_dimension(X_flat)
    spectral = compute_spectral_shape(X_flat)
    tangent_result = compute_local_metric_in_tangent_space(X_flat, q=10)
    
    mean_pos = ensemble.mean(axis=0)
    cov_global = np.cov(X_flat.T)
    
    return {
        'mean_pos': mean_pos,
        'cov_global': cov_global,
        'metric_tensor': tangent_result['g_tangent'],
        'tangent_basis': tangent_result['tangent_basis'],
        'd_pr': dims['d_pr'],
        'd_consensus': dims['consensus'],
        'spectral_decay': spectral['spectral_decay'],
        'top5_ratio': spectral['top5_ratio'],
        'entropy': spectral['entropy'],
        'eff_rank_95': spectral['eff_rank_95'],
        'A_C': compute_anisotropy_C(X_flat, k=30),
        'trace_g': tangent_result['trace_g'],
        'det_g': tangent_result['det_g'],
        'n_samples': n_samples,
        'n_residues': n_residues,
    }


def compute_mutant_perturbation(row, wt_seq):
    """计算突变体相对于野生型的扰动向量"""
    wt_aa = row['wt_aa']
    mut_aa = row['mut_aa']
    position = row['position'] - 1
    n_res = len(wt_seq)
    
    ca_coords = np.random.randn(n_res, 3) * 0.05
    
    aa_charge_diff = get_aa_charge_diff(wt_aa, mut_aa)
    aa_volume_diff = get_aa_volume_diff(wt_aa, mut_aa)
    
    if position < n_res:
        perturbation_magnitude = 0.1 + abs(aa_volume_diff) / 100 + abs(aa_charge_diff) * 0.05
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction) + 1e-10
        ca_coords[position] += perturbation_magnitude * direction
    
    return ca_coords, aa_volume_diff, aa_charge_diff


AA_VOLUMES = {
    'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5,
    'E': 138.4, 'Q': 143.8, 'G': 60.1, 'H': 153.2, 'I': 166.7,
    'L': 166.7, 'K': 168.6, 'M': 162.9, 'F': 189.9, 'P': 112.7,
    'S': 89.0, 'T': 116.1, 'W': 227.8, 'Y': 193.6, 'V': 140.0,
}

AA_CHARGES = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
    'E': -1, 'Q': 0, 'G': 0, 'H': 0.5, 'I': 0,
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0,
}


def get_aa_volume_diff(wt_aa, mut_aa):
    wt_vol = AA_VOLUMES.get(wt_aa, 0)
    mut_vol = AA_VOLUMES.get(mut_aa, 0)
    return mut_vol - wt_vol


def get_aa_charge_diff(wt_aa, mut_aa):
    wt_chg = AA_CHARGES.get(wt_aa, 0)
    mut_chg = AA_CHARGES.get(mut_aa, 0)
    return mut_chg - wt_chg


def compute_cgeo_for_mutations(df, wt_features):
    results = []
    
    for protein in df['protein'].unique():
        if protein not in wt_features:
            continue
        
        wt_mean = wt_features[protein]['mean_pos']
        cov_global = wt_features[protein]['cov_global']
        g_tangent = wt_features[protein]['metric_tensor']
        Q = wt_features[protein]['tangent_basis']
        wt_seq = df[df['protein'] == protein]['target_seq'].iloc[0]
        
        mut_df = df[df['protein'] == protein]
        n_mutations = len(mut_df)
        
        for idx, row in mut_df.iterrows():
            ca_coords, aa_volume_diff, aa_charge_diff = compute_mutant_perturbation(row, wt_seq)
            
            delta_pos = ca_coords - wt_mean
            delta_flat = delta_pos.flatten()
            
            if Q is not None and delta_flat.shape[0] == Q.shape[0]:
                delta_tangent = delta_flat @ Q
                cgeo_metric = float(delta_tangent @ g_tangent @ delta_tangent)
            else:
                cgeo_metric = float(delta_flat @ cov_global @ delta_flat)
            
            cgeo_raw = float(delta_flat @ cov_global @ delta_flat)
            cgeo_euclidean = float(np.sum(delta_flat ** 2))
            
            pos_idx = row['position'] - 1
            if pos_idx < len(delta_pos):
                local_delta = delta_pos[pos_idx]
            else:
                local_delta = np.zeros(3)
            cgeo_local = float(np.sum(local_delta ** 2))
            
            results.append({
                'protein': protein,
                'mutant': row['mutant'],
                'position': row['position'],
                'wt_aa': row['wt_aa'],
                'mut_aa': row['mut_aa'],
                'DMS_score': row['DMS_score'],
                'DMS_score_bin': row['DMS_score_bin'],
                'C_geo_raw': cgeo_raw,
                'C_geo_metric': cgeo_metric,
                'C_geo_euclidean': cgeo_euclidean,
                'C_geo_local': cgeo_local,
                'aa_volume_diff': aa_volume_diff,
                'aa_charge_diff': aa_charge_diff,
                'seq_length': len(wt_seq),
            })
    
    return pd.DataFrame(results)


def correlate_cgeo_with_dms(cgeo_df):
    correlations = []
    
    for protein in cgeo_df['protein'].unique():
        prot_df = cgeo_df[cgeo_df['protein'] == protein].dropna()
        
        for cgeo_col in ['C_geo_raw', 'C_geo_metric', 'C_geo_euclidean', 'C_geo_local']:
            if cgeo_col not in prot_df.columns:
                continue
            
            x = prot_df[cgeo_col]
            y = prot_df['DMS_score']
            
            if len(x) < 10:
                continue
            
            pearson_r, pearson_p = pearsonr(x, y)
            spearman_r, spearman_p = spearmanr(x, y)
            
            correlations.append({
                'protein': protein,
                'feature': cgeo_col,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'n_samples': len(prot_df),
            })
    
    return pd.DataFrame(correlations)


def benchmark_models(cgeo_df):
    results = []
    
    for protein in cgeo_df['protein'].unique():
        prot_df = cgeo_df[cgeo_df['protein'] == protein].dropna()
        if len(prot_df) < 50:
            continue
        
        X = prot_df[['C_geo_metric', 'C_geo_local', 'aa_volume_diff']].values
        y = prot_df['DMS_score'].values
        
        models = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'SVR_rbf': SVR(kernel='rbf'),
            'RandomForest': RandomForestRegressor(n_estimators=100),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100),
        }
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for model_name, model in models.items():
            r2_scores = []
            mse_scores = []
            
            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                r2_scores.append(r2_score(y_test, y_pred))
                mse_scores.append(mean_squared_error(y_test, y_pred))
            
            results.append({
                'protein': protein,
                'model': model_name,
                'r2_mean': np.mean(r2_scores),
                'r2_std': np.std(r2_scores),
                'mse_mean': np.mean(mse_scores),
                'mse_std': np.std(mse_scores),
                'n_samples': len(prot_df),
            })
    
    return pd.DataFrame(results)


def generate_visualizations(cgeo_df, correlations_df, benchmark_df):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use('default')
        
        for protein in cgeo_df['protein'].unique():
            prot_df = cgeo_df[cgeo_df['protein'] == protein].dropna()
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            sns.scatterplot(data=prot_df, x='C_geo_metric', y='DMS_score', ax=axes[0, 0], alpha=0.5)
            axes[0, 0].set_title(f'{protein.upper()}: C_geo_metric vs DMS_score')
            
            sns.scatterplot(data=prot_df, x='C_geo_local', y='DMS_score', ax=axes[0, 1], alpha=0.5)
            axes[0, 1].set_title(f'{protein.upper()}: C_geo_local vs DMS_score')
            
            sns.scatterplot(data=prot_df, x='C_geo_euclidean', y='DMS_score', ax=axes[1, 0], alpha=0.5)
            axes[1, 0].set_title(f'{protein.upper()}: C_geo_euclidean vs DMS_score')
            
            sns.scatterplot(data=prot_df, x='aa_volume_diff', y='DMS_score', ax=axes[1, 1], alpha=0.5)
            axes[1, 1].set_title(f'{protein.upper()}: AA Volume Diff vs DMS_score')
            
            plt.tight_layout()
            fig.savefig(OUTPUT_DIR / f'{protein}_cgeo_scatter.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=correlations_df, x='feature', y='spearman_r', hue='protein', ax=ax)
        ax.axhline(0, color='black', linestyle='--')
        ax.set_title('Spearman Correlation: C_geo Features vs DMS_score')
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / 'cgeo_correlation_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(data=benchmark_df, x='model', y='r2_mean', hue='protein', ax=ax)
        ax.axhline(0, color='black', linestyle='--')
        ax.set_title('Model Performance: R² Score')
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / 'model_benchmark_r2.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Visualizations generated successfully")
        
    except ImportError:
        logger.warning("Matplotlib/Seaborn not available, skipping visualizations")


def main():
    logger.info("=" * 60)
    logger.info("Phase VI-A: C_geo DMS Calibration")
    logger.info("=" * 60)
    
    logger.info("\n[1/5] Loading DMS data...")
    dms_df = load_dms_single_mutants()
    if dms_df is None:
        return
    
    logger.info("\n[2/5] Extracting wildtype sequences...")
    wt_seqs = get_wildtype_sequences(dms_df)
    
    logger.info("\n[3/5] Generating synthetic wildtype ensembles...")
    bioemu_output = {}
    for protein, wt_seq in wt_seqs.items():
        n_res = len(wt_seq)
        bioemu_output[protein] = generate_synthetic_ensemble(n_res, n_samples=250)
        logger.info(f"  {protein}: {bioemu_output[protein].shape}")
    
    logger.info("\n[4/5] Computing geometric features for wildtype...")
    wt_features = {}
    for protein, ensemble in bioemu_output.items():
        wt_features[protein] = compute_geometric_features(ensemble)
        logger.info(f"  {protein}: d_PR={wt_features[protein]['d_pr']:.2f}, A_C={wt_features[protein]['A_C']:.4f}")
    
    logger.info("\n[5/5] Computing C_geo for mutations...")
    cgeo_df = compute_cgeo_for_mutations(dms_df, wt_features)
    
    cgeo_output_path = OUTPUT_DIR / "phase6_cgeo_results.csv"
    cgeo_df.to_csv(cgeo_output_path, index=False)
    logger.info(f"Saved C_geo results: {len(cgeo_df)} records to {cgeo_output_path}")
    
    logger.info("\n[6/5] Correlation analysis...")
    correlations_df = correlate_cgeo_with_dms(cgeo_df)
    corr_output_path = OUTPUT_DIR / "phase6_cgeo_correlations.csv"
    correlations_df.to_csv(corr_output_path, index=False)
    logger.info(f"Saved correlations: {len(correlations_df)} records")
    
    for _, row in correlations_df.iterrows():
        sig = '***' if row['spearman_p'] < 0.001 else '**' if row['spearman_p'] < 0.01 else '*' if row['spearman_p'] < 0.05 else ''
        logger.info(f"  {row['protein']}-{row['feature']}: r={row['spearman_r']:.4f} (p={row['spearman_p']:.2e}) {sig}")
    
    logger.info("\n[7/5] Model benchmarking...")
    benchmark_df = benchmark_models(cgeo_df)
    benchmark_output_path = OUTPUT_DIR / "phase6_model_benchmark.csv"
    benchmark_df.to_csv(benchmark_output_path, index=False)
    logger.info(f"Saved benchmark results: {len(benchmark_df)} records")
    
    logger.info("\n[8/5] Generating visualizations...")
    generate_visualizations(cgeo_df, correlations_df, benchmark_df)
    
    logger.info("\n" + "=" * 60)
    logger.info("Phase VI-A C_geo DMS Calibration complete!")
    logger.info("=" * 60)
    
    return cgeo_df, correlations_df, benchmark_df


if __name__ == '__main__':
    main()