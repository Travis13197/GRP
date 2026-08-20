#!/usr/bin/env python3
"""
Phase D3: 第三定律路径多样性分析 (Path Diversity Analysis)
===========================================================
基于 D1 (IDP) + D4 (折叠蛋白) + DMS 数据，分析多种构象转换路径的
低作用量特性，验证第三定律。

路径类型:
  1. 别构路径 (Allosteric): IDP内不同构象状态间的W2距离
  2. 折叠-去折叠路径 (Folding-Unfolding): IDP vs Folded构象对比
  3. 突变累积路径 (Mutation Accumulation): DMS突变体的构象变化
  4. 跨系统路径 (Cross-System): PolyX→HET→IDP→Folded四级过渡

方法:
  - 从NPZ文件直接计算W2距离 (Wasserstein-2)
  - 使用TSI (Topological Similarity Index) 评估路径紧凑性
  - 置换检验 + Bootstrap CI

执行: python field_theory/scripts/phase_d3_path_analysis.py
预计: ~5min (纯分析, 无GPU)
版本: v1.0 | 创建: 2026-07-13
"""

import json, warnings, sys
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import pdist, squareform, cdist
from collections import defaultdict
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "field_theory" / "tables" / "phase_d3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
D1_OUTPUT = BASE_DIR / "test_workflow" / "idp_ensemble_phase_d" / "output"
D2_OUTPUT = BASE_DIR / "test_workflow" / "polyx_ensemble" / "output"  # D2长链数据
D4_OUTPUT = BASE_DIR / "test_workflow" / "folded_proteins_phase_d" / "output"
POLYX_OUTPUT = BASE_DIR / "test_workflow" / "polyx_ensemble" / "output"
HET_OUTPUT = BASE_DIR / "test_workflow" / "heteropolymer_ensemble" / "output"
GEOMETRY_CSV = BASE_DIR / "field_theory" / "data" / "dms" / "phase9_systemwide" / "systemwide_enhanced_geometry_v4.csv"
HET_CSV = BASE_DIR / "test_workflow" / "heteropolymer_ensemble" / "analysis" / "het_geometry" / "het_geometry_results.csv"

# ============================================================
# 数据加载
# ============================================================
def load_ensemble_coords(seq_id, output_dir, max_samples=500):
    """加载系综构象坐标 (C-alpha flattened)"""
    seq_dir = output_dir / seq_id
    if not seq_dir.exists():
        return None
    npz_files = sorted(seq_dir.glob("batch_*.npz"))
    if not npz_files:
        return None
    all_coords = []
    for npz_file in npz_files:
        try:
            data = np.load(npz_file, allow_pickle=True)
            pos = data.get('pos', data.get('positions'))
            if pos is not None:
                flattened = pos.reshape(pos.shape[0], -1)
                all_coords.append(flattened)
        except:
            pass
    if not all_coords:
        return None
    coords = np.vstack(all_coords)
    if len(coords) > max_samples:
        idx = np.random.RandomState(42).choice(len(coords), max_samples, replace=False)
        coords = coords[idx]
    return coords

def compute_w2_distance(coords1, coords2):
    """计算两个系综间的Wasserstein-2距离 (Gaussian approximation)"""
    if coords1 is None or coords2 is None:
        return None
    n1 = coords1.shape[0]
    n2 = coords2.shape[0]
    if n1 < 3 or n2 < 3:
        return None
    # 使用Gaussian W2: W2^2 = ||mu1 - mu2||^2 + Tr(S1 + S2 - 2(S1^1/2 S2 S1^1/2)^1/2)
    mu1 = coords1.mean(axis=0)
    mu2 = coords2.mean(axis=0)
    diff_mean = np.sum((mu1 - mu2) ** 2)
    S1 = np.cov(coords1, rowvar=False)
    S2 = np.cov(coords2, rowvar=False)
    eigenvalues1 = np.linalg.eigvalsh(S1)
    eigenvalues1 = np.maximum(eigenvalues1, 0)
    S1_sqrt = np.diag(np.sqrt(eigenvalues1))
    try:
        inner = S1_sqrt @ S2 @ S1_sqrt
        eigenvalues_inner = np.linalg.eigvalsh(inner)
        eigenvalues_inner = np.maximum(eigenvalues_inner, 0)
        trace_term = np.trace(S1) + np.trace(S2) - 2 * np.sum(np.sqrt(eigenvalues_inner))
    except:
        trace_term = np.trace(S1) + np.trace(S2)
    w2_sq = max(0, diff_mean + trace_term)
    return np.sqrt(w2_sq)

def compute_geometry_from_coords(coords):
    """从坐标计算几何特征"""
    n_samples, n_features = coords.shape
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)
    total_var = eigenvalues.sum()
    if total_var <= 0:
        return None
    normalized = eigenvalues / total_var
    PR = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum() if (eigenvalues ** 2).sum() > 0 else 0
    cumsum = np.cumsum(normalized)
    eff_rank_95 = int(np.searchsorted(cumsum, 0.95) + 1)
    A_C = float(normalized[0])
    entropy = float(-np.sum(normalized * np.log(normalized + 1e-10)))
    k = np.arange(1, len(eigenvalues) + 1)
    n_use = min(50, len(eigenvalues))
    A_mat = np.vstack([np.log(k[:n_use]), np.ones(n_use)]).T
    alpha, _ = np.linalg.lstsq(A_mat, np.log(eigenvalues[:n_use] + 1e-10), rcond=None)[0]
    spectral_decay = float(-alpha)
    variance_per_dof = float(total_var / n_features)
    return {
        'PR': PR, 'A_C': A_C, 'eff_rank_95': eff_rank_95,
        'spectral_decay': spectral_decay, 'entropy': entropy,
        'variance_per_dof': variance_per_dof, 'total_variance': float(total_var),
        'n_samples': n_samples, 'n_features': n_features
    }

# ============================================================
# TSI 计算 (快速置换检验)
# ============================================================
def fast_permutation_tsi(dist_matrix, cat_mask, n_perm=2000, seed=42):
    """快速置换TSI计算"""
    rng = np.random.RandomState(seed)
    n_cat = cat_mask.sum()
    if n_cat < 3:
        return {'TSI': None, 'p': None, 'n': int(n_cat), 'status': 'INSUFFICIENT_DATA'}
    cat_idx = np.where(cat_mask)[0]
    real_dists = dist_matrix[np.ix_(cat_idx, cat_idx)]
    real_mean = np.mean(real_dists[np.triu_indices(len(cat_idx), k=1)])
    null_means = np.zeros(n_perm)
    all_idx = np.arange(len(cat_mask))
    for p in range(n_perm):
        shuffled = rng.permutation(all_idx)
        perm_idx = np.where(cat_mask[shuffled])[0]
        perm_dists = dist_matrix[np.ix_(perm_idx, perm_idx)]
        null_means[p] = np.mean(perm_dists[np.triu_indices(len(perm_idx), k=1)])
    null_mean = np.mean(null_means)
    null_std = np.std(null_means)
    tsi = (real_mean - null_mean) / null_std if null_std > 0 else 0
    p_val = np.mean(np.abs(null_means - null_mean) >= np.abs(real_mean - null_mean))
    status = 'VERIFIED' if (tsi < 0 and p_val < 0.05) else ('PARTIAL' if tsi < 0 else 'NOT_VERIFIED')
    return {'real_mean': float(real_mean), 'null_mean': float(null_mean),
            'null_std': float(null_std), 'TSI': float(tsi), 'p': float(p_val),
            'n': int(n_cat), 'status': status}

def bootstrap_tsi_ci(dist_matrix, cat_mask, n_bootstrap=500, ci=95, seed=42):
    """Bootstrap TSI置信区间"""
    rng = np.random.RandomState(seed)
    tsi_vals = []
    n_cat = cat_mask.sum()
    all_idx = np.arange(len(cat_mask))
    for _ in range(n_bootstrap):
        boot_idx = rng.choice(all_idx, size=len(all_idx), replace=True)
        if np.sum(cat_mask[boot_idx]) < 3:
            continue
        tsi_result = fast_permutation_tsi(dist_matrix, cat_mask[boot_idx], n_perm=500, seed=rng.randint(0, 10000))
        if tsi_result['TSI'] is not None:
            tsi_vals.append(tsi_result['TSI'])
    if len(tsi_vals) < 10:
        return {'ci_low': None, 'ci_high': None, 'n_valid': len(tsi_vals)}
    alpha = (100 - ci) / 100
    return {
        'ci_low': float(np.percentile(tsi_vals, alpha/2 * 100)),
        'ci_high': float(np.percentile(tsi_vals, (1 - alpha/2) * 100)),
        'tsi_mean': float(np.mean(tsi_vals)),
        'tsi_std': float(np.std(tsi_vals)),
        'n_valid': len(tsi_vals)
    }

# ============================================================
# 路径1: 别构路径 (IDP内构象子状态)
# ============================================================
def path1_allosteric():
    """别构路径: IDP内不同构象状态的W2距离"""
    print("\n" + "="*60)
    print("  Path 1: Allosteric Paths (IDP sub-states)")
    print("="*60)
    
    results = {}
    # Load all D1 IDP ensembles
    idp_ensembles = {}
    for d in sorted(D1_OUTPUT.glob("*/")):
        sid = d.name
        coords = load_ensemble_coords(sid, D1_OUTPUT, max_samples=500)
        if coords is not None and coords.shape[0] >= 50:
            # Split into 2 sub-states (first half vs second half)
            mid = coords.shape[0] // 2
            sub1 = coords[:mid]
            sub2 = coords[mid:]
            # Compute W2 between sub-states
            w2 = compute_w2_distance(sub1, sub2)
            # Compute geometry for each sub-state
            geom1 = compute_geometry_from_coords(sub1)
            geom2 = compute_geometry_from_coords(sub2)
            if w2 is not None and geom1 and geom2:
                idp_ensembles[sid] = {
                    'w2': float(w2),
                    'geom1': geom1, 'geom2': geom2,
                    'n_samples': coords.shape[0]
                }
                print(f"  {sid[:25]}: W2={w2:.4f}, PR1={geom1['PR']:.2f}, PR2={geom2['PR']:.2f}")
    
    # Pairwise geometric feature distances (instead of W2, since proteins have different lengths)
    idp_names = sorted(idp_ensembles.keys())
    if len(idp_names) >= 3:
        n = len(idp_names)
        
        # Build distance matrix from geometric features (dimension-independent)
        geom_features = ['PR', 'A_C', 'spectral_decay', 'entropy', 'variance_per_dof']
        # Use the average of geom1 and geom2 for each protein
        idp_geom = {}
        for name, data in idp_ensembles.items():
            g1 = data['geom1']
            g2 = data['geom2']
            avg_geom = {k: (g1.get(k, 0) + g2.get(k, 0)) / 2 for k in geom_features}
            idp_geom[name] = avg_geom
        
        feat_matrix = np.array([[idp_geom[name][k] for k in geom_features] for name in idp_names])
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        feat_scaled = scaler.fit_transform(feat_matrix)
        w2_matrix = squareform(pdist(feat_scaled, metric='euclidean'))
        
        # TSI: Are IDP ensembles geometrically similar?
        cat_mask = np.ones(n, dtype=bool)
        tsi = fast_permutation_tsi(w2_matrix, cat_mask, n_perm=2000)
        boot = bootstrap_tsi_ci(w2_matrix, cat_mask, n_bootstrap=500)
        
        results['allosteric_idp'] = {
            'n_proteins': n,
            'protein_names': idp_names,
            'geom_matrix': w2_matrix.tolist(),
            'tsi': tsi,
            'bootstrap': boot,
            'sub_state_w2': {k: v['w2'] for k, v in idp_ensembles.items()},
            'note': 'Distance matrix uses geometric feature space (PR, A_C, spectral_decay, entropy, variance_per_dof) due to different protein lengths'
        }
        print(f"\n  IDP allosteric TSI (geom space): {tsi['TSI']:.4f}, p={tsi['p']:.4f}, status={tsi['status']}")
        if boot['ci_low'] is not None:
            print(f"  Bootstrap 95%CI: [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]")
    
    return results

# ============================================================
# 路径2: 折叠-去折叠路径 (IDP vs Folded)
# ============================================================
def path2_folding_unfolding():
    """折叠-去折叠路径: IDP vs Folded Protein W2距离"""
    print("\n" + "="*60)
    print("  Path 2: Folding-Unfolding Paths (IDP vs Folded)")
    print("="*60)
    
    results = {}
    
    # Load IDP ensembles and compute geometry
    idp_geom = {}
    for d in sorted(D1_OUTPUT.glob("*/")):
        sid = d.name
        coords = load_ensemble_coords(sid, D1_OUTPUT, max_samples=500)
        if coords is not None and coords.shape[0] >= 50:
            geom = compute_geometry_from_coords(coords)
            if geom:
                idp_geom[sid] = geom
    
    # Load folded protein ensembles and compute geometry
    folded_geom = {}
    for d in sorted(D4_OUTPUT.glob("*/")):
        sid = d.name
        coords = load_ensemble_coords(sid, D4_OUTPUT, max_samples=500)
        if coords is not None and coords.shape[0] >= 50:
            geom = compute_geometry_from_coords(coords)
            if geom:
                folded_geom[sid] = geom
    
    print(f"  IDP: {len(idp_geom)} proteins, Folded: {len(folded_geom)} proteins")
    
    if len(idp_geom) >= 3 and len(folded_geom) >= 3:
        # Build combined distance matrix from geometric features
        geom_features = ['PR', 'A_C', 'spectral_decay', 'entropy', 'variance_per_dof']
        all_names = sorted(idp_geom.keys()) + sorted(folded_geom.keys())
        idp_mask = np.array([name in idp_geom for name in all_names])
        folded_mask = np.array([name in folded_geom for name in all_names])
        all_geom = {**idp_geom, **folded_geom}
        
        feat_matrix = np.array([[all_geom[name].get(k, 0) for k in geom_features] for name in all_names])
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        feat_scaled = scaler.fit_transform(feat_matrix)
        w2_matrix = squareform(pdist(feat_scaled, metric='euclidean'))
        
        # TSI: IDP vs Folded
        # H0: IDP和Folded在构象空间中没有显著差异
        tsi_idp_vs_folded = fast_permutation_tsi(w2_matrix, idp_mask, n_perm=2000)
        tsi_folded_vs_idp = fast_permutation_tsi(w2_matrix, folded_mask, n_perm=2000)
        
        # Cross-category distances
        idp_idx = np.where(idp_mask)[0]
        folded_idx = np.where(folded_mask)[0]
        cross_w2 = w2_matrix[np.ix_(idp_idx, folded_idx)]
        within_idp = w2_matrix[np.ix_(idp_idx, idp_idx)]
        within_folded = w2_matrix[np.ix_(folded_idx, folded_idx)]
        
        results['folding_unfolding'] = {
            'n_idp': len(idp_geom),
            'n_folded': len(folded_geom),
            'idp_names': sorted(idp_geom.keys()),
            'folded_names': sorted(folded_geom.keys()),
            'tsi_idp': tsi_idp_vs_folded,
            'tsi_folded': tsi_folded_vs_idp,
            'mean_cross_dist': float(np.mean(cross_w2[cross_w2 > 0])) if np.any(cross_w2 > 0) else 0,
            'mean_within_idp': float(np.mean(within_idp[within_idp > 0])) if np.any(within_idp > 0) else 0,
            'mean_within_folded': float(np.mean(within_folded[within_folded > 0])) if np.any(within_folded > 0) else 0,
            'note': 'Distance matrix uses geometric feature space (PR, A_C, spectral_decay, entropy, variance_per_dof) due to different protein lengths'
        }
        
        print(f"\n  IDP TSI: {tsi_idp_vs_folded['TSI']:.4f}, p={tsi_idp_vs_folded['p']:.4f}, status={tsi_idp_vs_folded['status']}")
        print(f"  Folded TSI: {tsi_folded_vs_idp['TSI']:.4f}, p={tsi_folded_vs_idp['p']:.4f}, status={tsi_folded_vs_idp['status']}")
        print(f"  Cross dist: {results['folding_unfolding']['mean_cross_dist']:.4f}")
        print(f"  Within IDP: {results['folding_unfolding']['mean_within_idp']:.4f}")
        print(f"  Within Folded: {results['folding_unfolding']['mean_within_folded']:.4f}")
    
    return results

# ============================================================
# 路径3: 四级过渡谱 (PolyX→HET→IDP→Folded)
# ============================================================
def path3_transition_spectrum():
    """四级过渡谱: PolyX→HET→IDP→Folded 构象空间连续过渡"""
    print("\n" + "="*60)
    print("  Path 3: Four-Level Transition Spectrum")
    print("="*60)
    
    results = {}
    
    # Load existing geometry data
    import pandas as pd
    
    # PolyX geometry
    polyx_geom = {}
    if GEOMETRY_CSV.exists():
        df_polyx = pd.read_csv(GEOMETRY_CSV)
        if 'seq_id' in df_polyx.columns:
            for _, row in df_polyx.iterrows():
                if 'PR' in row and pd.notna(row['PR']):
                    polyx_geom[row['seq_id']] = {
                        'PR': float(row['PR']),
                        'A_C': float(row.get('A_C', 0)),
                        'spectral_decay': float(row.get('spectral_decay', 0)),
                        'entropy': float(row.get('entropy', 0))
                    }
    
    # HET geometry
    het_geom = {}
    if HET_CSV.exists():
        df_het = pd.read_csv(HET_CSV)
        if 'seq_id' in df_het.columns:
            for _, row in df_het.iterrows():
                if 'PR' in row and pd.notna(row['PR']):
                    het_geom[row['seq_id']] = {
                        'PR': float(row['PR']),
                        'A_C': float(row.get('A_C', 0)),
                        'spectral_decay': float(row.get('spectral_decay', 0)),
                        'entropy': float(row.get('entropy', 0)),
                        'category': str(row.get('category', 'HET'))
                    }
    
    # IDP geometry (from D1)
    idp_geom = {}
    for d in sorted(D1_OUTPUT.glob("*/")):
        sid = d.name
        coords = load_ensemble_coords(sid, D1_OUTPUT, max_samples=500)
        if coords is not None:
            geom = compute_geometry_from_coords(coords)
            if geom:
                idp_geom[sid] = geom
    
    # Folded geometry (from D4)
    folded_geom = {}
    for d in sorted(D4_OUTPUT.glob("*/")):
        sid = d.name
        coords = load_ensemble_coords(sid, D4_OUTPUT, max_samples=500)
        if coords is not None:
            geom = compute_geometry_from_coords(coords)
            if geom:
                folded_geom[sid] = geom
    
    print(f"  PolyX: {len(polyx_geom)}, HET: {len(het_geom)}, IDP: {len(idp_geom)}, Folded: {len(folded_geom)}")
    
    # Build combined feature matrix for transition spectrum
    all_features = []
    all_labels = []
    all_names = []
    
    for name, g in polyx_geom.items():
        all_features.append([g['PR'], g['A_C'], g['spectral_decay'], g['entropy']])
        all_labels.append('PolyX')
        all_names.append(name)
    
    for name, g in het_geom.items():
        all_features.append([g['PR'], g['A_C'], g['spectral_decay'], g['entropy']])
        all_labels.append('HET')
        all_names.append(name)
    
    for name, g in idp_geom.items():
        all_features.append([g['PR'], g['A_C'], g['spectral_decay'], g['entropy']])
        all_labels.append('IDP')
        all_names.append(name)
    
    for name, g in folded_geom.items():
        all_features.append([g['PR'], g['A_C'], g['spectral_decay'], g['entropy']])
        all_labels.append('Folded')
        all_names.append(name)
    
    if len(all_features) >= 10:
        all_features = np.array(all_features)
        all_labels = np.array(all_labels)
        
        # Standardize
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(all_features)
        
        # Distance matrix
        dist_matrix = squareform(pdist(features_scaled, metric='euclidean'))
        
        # TSI for each category
        categories = ['PolyX', 'HET', 'IDP', 'Folded']
        for cat in categories:
            cat_mask = all_labels == cat
            if cat_mask.sum() >= 3:
                tsi = fast_permutation_tsi(dist_matrix, cat_mask, n_perm=2000)
                boot = bootstrap_tsi_ci(dist_matrix, cat_mask, n_bootstrap=500)
                results[f'transition_{cat}'] = {
                    'n': int(cat_mask.sum()),
                    'tsi': tsi,
                    'bootstrap': boot
                }
                print(f"  {cat}: TSI={tsi['TSI']:.4f}, p={tsi['p']:.4f}, status={tsi['status']}")
        
        # Cross-category distances
        for cat1 in categories:
            for cat2 in categories:
                if cat1 < cat2:
                    m1 = all_labels == cat1
                    m2 = all_labels == cat2
                    if m1.sum() >= 3 and m2.sum() >= 3:
                        cross_dists = dist_matrix[np.ix_(np.where(m1)[0], np.where(m2)[0])]
                        key = f'cross_{cat1}_{cat2}'
                        results[key] = {
                            'mean': float(np.mean(cross_dists)),
                            'std': float(np.std(cross_dists)),
                            'n1': int(m1.sum()), 'n2': int(m2.sum())
                        }
    
    return results

# ============================================================
# 路径4: 长链PolyX构象路径 (基于D2新数据)
# ============================================================
def path4_longchain_paths():
    """长链PolyX构象路径: 利用D2新数据 (n=55-200)"""
    print("\n" + "="*60)
    print("  Path 4: Long-chain PolyX Paths (D2 data)")
    print("="*60)
    
    results = {}
    
    # Load D2 long-chain ensembles
    d2_ensembles = {}
    for d in sorted(D2_OUTPUT.glob("PolyX_*_n*/")):
        sid = d.name
        coords = load_ensemble_coords(sid, D2_OUTPUT, max_samples=500)
        if coords is not None and coords.shape[0] >= 50:
            geom = compute_geometry_from_coords(coords)
            if geom:
                # Extract AA and n
                parts = sid.split('_')
                aa = parts[1] if len(parts) >= 2 else '?'
                n_val = int(parts[2][1:]) if len(parts) >= 3 else 0
                d2_ensembles[sid] = {'coords': coords, 'geom': geom, 'aa': aa, 'n': n_val}
    
    print(f"  D2 ensembles: {len(d2_ensembles)}")
    
    if len(d2_ensembles) >= 5:
        # Group by AA and compute geometric feature distances along chain length
        aa_groups = defaultdict(list)
        for sid, data in d2_ensembles.items():
            aa_groups[data['aa']].append((data['n'], sid, data))
        
        geom_features = ['PR', 'A_C', 'spectral_decay', 'entropy', 'variance_per_dof']
        
        for aa, items in aa_groups.items():
            items.sort(key=lambda x: x[0])
            if len(items) >= 3:
                # Geometric feature distance between consecutive chain lengths (instead of W2)
                geom_dist_chain = []
                n_vals = []
                for i in range(len(items) - 1):
                    n1, sid1, d1 = items[i]
                    n2, sid2, d2 = items[i+1]
                    g1 = np.array([d1['geom'].get(k, 0) for k in geom_features])
                    g2 = np.array([d2['geom'].get(k, 0) for k in geom_features])
                    dist = np.linalg.norm(g1 - g2)
                    geom_dist_chain.append(dist)
                    n_vals.append(f"{n1}→{n2}")
                
                if geom_dist_chain:
                    results[f'longchain_{aa}'] = {
                        'n_pairs': len(geom_dist_chain),
                        'geom_dist_values': [float(d) for d in geom_dist_chain],
                        'n_transitions': n_vals,
                        'mean_geom_dist': float(np.mean(geom_dist_chain)),
                        'geom_dist_gradient': float(np.mean(np.diff(geom_dist_chain))) if len(geom_dist_chain) > 1 else 0,
                        'note': 'Uses geometric feature distance (PR, A_C, spectral_decay, entropy, variance_per_dof) instead of W2 due to different chain lengths'
                    }
                    print(f"  {aa}: {len(geom_dist_chain)} chain transitions, mean geom_dist={np.mean(geom_dist_chain):.4f}")
    
    return results

# ============================================================
# Main
# ============================================================
def main():
    print("="*60)
    print("  Phase D3: Path Diversity Analysis (Law 3)")
    print("="*60)
    
    all_results = {}
    
    # Path 1: Allosteric
    all_results['path1_allosteric'] = path1_allosteric()
    
    # Path 2: Folding-Unfolding
    all_results['path2_folding'] = path2_folding_unfolding()
    
    # Path 3: Transition Spectrum
    all_results['path3_transition'] = path3_transition_spectrum()
    
    # Path 4: Long-chain paths
    all_results['path4_longchain'] = path4_longchain_paths()
    
    # Save results
    output_path = OUTPUT_DIR / "phase_d3_path_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved to: {output_path}")
    
    # Summary
    print("\n" + "="*60)
    print("  D3 PATH ANALYSIS SUMMARY")
    print("="*60)
    
    # Count VERIFIED paths
    verified = 0
    partial = 0
    not_verified = 0
    for key, val in all_results.items():
        if isinstance(val, dict):
            for subkey, subval in val.items():
                if isinstance(subval, dict) and 'tsi' in subval:
                    status = subval['tsi'].get('status', 'N/A')
                    if status == 'VERIFIED':
                        verified += 1
                    elif status == 'PARTIAL':
                        partial += 1
                    elif status == 'NOT_VERIFIED':
                        not_verified += 1
    
    print(f"  VERIFIED paths: {verified}")
    print(f"  PARTIAL paths: {partial}")
    print(f"  NOT_VERIFIED paths: {not_verified}")
    print(f"  Total paths analyzed: {verified + partial + not_verified}")
    print("="*60)

if __name__ == "__main__":
    main()