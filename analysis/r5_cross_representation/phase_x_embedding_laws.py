#!/usr/bin/env python3
"""
Phase X: Embedding空间三定律验证 — 完整管线
===============================================
基于 GPT-Modification-202606262005.md 的三定律数学框架,
在 ProstT5 embedding 空间 (1024d) 中验证 ProtGenesis 三条定律。

管线步骤:
  [1/6] 构建 PolyX embedding ensemble (ProstT5, 缓存)
  [2/6] 计算局部切空间 + 度量张量 g_S (SVD + Ledoit-Wolf)
  [3/6] 第一定律: 几何扰动定律 — 余弦一致性 + CV 比较
  [4/6] 第二定律: 生物约束场方程 — 耦合矩阵 K 估计
  [5/6] 第三定律: 低作用量路径定律 — 路径作用量 + Null 检验
  [6/6] 跨空间验证: Embedding vs Cα 几何量对比

用法:
  # 在 WSL2 中运行 (需要 GPU + ProstT5 模型)
  cd /mnt/b/2026/Exploration/ProtGenesis2_Ensemble
  python field_theory/scripts/phase_x_embedding_laws.py --step all

  # 或分步运行
  python field_theory/scripts/phase_x_embedding_laws.py --step 1  # 仅计算embedding
  python field_theory/scripts/phase_x_embedding_laws.py --step 2  # 仅计算几何
  ...
"""

import sys, os, json, warnings, random, argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import pickle

import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr, pearsonr, ttest_ind
from scipy.optimize import curve_fit
from scipy.linalg import eigh, sqrtm
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ============================================================
# 配置与路径
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
FIELD_THEORY = Path(__file__).parent.parent
SCRIPTS_DIR = FIELD_THEORY / "scripts"
DATA_DIR = FIELD_THEORY / "data" / "phase_x"
FIGURES_DIR = FIELD_THEORY / "figures" / "phase_x"
TABLES_DIR = FIELD_THEORY / "tables"

for d in [DATA_DIR, FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 序列参数
POLYX_AAS = ['G', 'S', 'E', 'L', 'K']
N_VALUES_POLYX = list(range(1, 51))  # n=1-50
N_RANDOM_PER_N = 50  # 每个n的随机序列数 (ensemble)
N_VALUES_LAW2 = [2, 3, 5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50]  # 第二定律用

# 嵌入参数
EMBEDDING_CACHE = DATA_DIR / "polyx_embeddings.npz"
EMBEDDING_META = DATA_DIR / "polyx_embeddings_meta.json"
EMBEDDING_DIM = 1024

# 几何参数
Q_TANGENT = 10       # 切空间维度
PCA_TARGET = 50      # PCA 预降维
EPSILON = 1e-3       # 正则化
USE_SHRINKAGE = True # Ledoit-Wolf 收缩

# 随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# 氨基酸物理化学性质
# ============================================================
AA_PROPERTIES = {
    'G': {'hydrophobicity': -0.4, 'charge': 0, 'volume': 60.1, 'atom_count': 1},
    'S': {'hydrophobicity': -0.8, 'charge': 0, 'volume': 89.0, 'atom_count': 3},
    'E': {'hydrophobicity': -3.5, 'charge': -1, 'volume': 138.4, 'atom_count': 5},
    'L': {'hydrophobicity': 3.8, 'charge': 0, 'volume': 166.7, 'atom_count': 4},
    'K': {'hydrophobicity': -3.9, 'charge': 1, 'volume': 168.6, 'atom_count': 5},
    'A': {'hydrophobicity': 1.8, 'charge': 0, 'volume': 88.6, 'atom_count': 1},
    'V': {'hydrophobicity': 4.2, 'charge': 0, 'volume': 140.0, 'atom_count': 3},
    'I': {'hydrophobicity': 4.5, 'charge': 0, 'volume': 166.7, 'atom_count': 4},
    'F': {'hydrophobicity': 2.8, 'charge': 0, 'volume': 189.9, 'atom_count': 7},
}


def generate_random_polyx(n, aa, count):
    """生成随机 PolyX 序列"""
    seqs = []
    for _ in range(count):
        positions = [aa] * n
        # 随机替换一个位置为其他氨基酸
        pos = random.randint(0, n - 1)
        other_aa = random.choice([a for a in POLYX_AAS if a != aa])
        positions[pos] = other_aa
        seqs.append(''.join(positions))
    return seqs


def generate_random_mixed(n, count):
    """生成随机混合序列 (多种氨基酸)"""
    seqs = []
    for _ in range(count):
        seq = ''.join(random.choice(POLYX_AAS) for _ in range(n))
        seqs.append(seq)
    return seqs


# ============================================================
# Step 1: Embedding 计算
# ============================================================
def step1_compute_embeddings(force_recompute=False):
    """计算并缓存所有 PolyX 序列的 ProstT5 embeddings"""
    print("=" * 60)
    print("Phase X.1: 计算 ProstT5 Embeddings")
    print("=" * 60)
    
    if EMBEDDING_CACHE.exists() and not force_recompute:
        print(f"✅ 嵌入缓存已存在: {EMBEDDING_CACHE}")
        data = np.load(EMBEDDING_CACHE, allow_pickle=True)
        with open(EMBEDDING_META, 'r') as f:
            meta = json.load(f)
        return data['embeddings'], meta
    
    sys.path.insert(0, str(PROJECT_ROOT))
    
    # 使用 Phase 2 验证过的嵌入路径
    _EXPLORATION_ROOT = Path("B:/2026/Exploration")
    if not _EXPLORATION_ROOT.exists():
        _EXPLORATION_ROOT = Path("/mnt/b/2026/Exploration")
    EMBED_MECH = _EXPLORATION_ROOT / "1.How_Protein_space_lookslike/EmbeddingMechanics"
    sys.path.insert(0, str(EMBED_MECH / "src" / "embed"))
    
    from prostt5_embedder import ProstT5Embedder
    
    model_path = "B:/2026/Exploration/8.Evolution/1.BasicUnit/models/ProstT5"
    if not Path(model_path).exists():
        model_path = "/mnt/b/2026/Exploration/8.Evolution/1.BasicUnit/models/ProstT5"
    if not Path(model_path).exists():
        model_path = "Rostlab/ProstT5"
        print(f"  本地模型未找到，使用 HuggingFace: {model_path}")
    
    embedder = ProstT5Embedder(
        model_path=model_path,
        batch_size=16,
        max_length=512,
        device="cuda",
        precision="fp16",
    )
    
    all_sequences = []
    all_ids = []
    all_labels = []
    
    # 1. PolyX 纯序列 (G, S, E, L, K, n=1-50)
    for n in N_VALUES_POLYX:
        for aa in POLYX_AAS:
            seq = aa * n
            all_sequences.append(seq)
            all_ids.append(f"PolyX_Poly{aa}_{n}")
            all_labels.append(f"Poly{aa}_n{n}")
    
    # 2. 随机变异序列 (每个 n 每个 AA 生成 N_RANDOM_PER_N 条)
    for n in N_VALUES_LAW2:
        for aa in POLYX_AAS:
            random_seqs = generate_random_polyx(n, aa, N_RANDOM_PER_N)
            for i, seq in enumerate(random_seqs):
                all_sequences.append(seq)
                all_ids.append(f"Random_Poly{aa}_{n}_{i}")
                all_labels.append(f"Random_Poly{aa}_n{n}")
    
    print(f"总序列数: {len(all_sequences)}")
    print(f"  PolyX 纯序列: {len(N_VALUES_POLYX) * len(POLYX_AAS)}")
    print(f"  随机变异序列: {len(N_VALUES_LAW2) * len(POLYX_AAS) * N_RANDOM_PER_N}")
    
    # 批量嵌入
    embeddings = embedder.embed(all_sequences, pooling="mean")
    embeddings = np.array(embeddings)
    
    # 保存缓存
    np.savez_compressed(EMBEDDING_CACHE, embeddings=embeddings)
    meta = {
        'ids': all_ids,
        'labels': all_labels,
        'sequences': all_sequences,
        'n_sequences': len(all_sequences),
        'n_polyx': len(N_VALUES_POLYX) * len(POLYX_AAS),
        'n_random': len(N_VALUES_LAW2) * len(POLYX_AAS) * N_RANDOM_PER_N,
        'embedding_dim': EMBEDDING_DIM,
        'timestamp': datetime.now().isoformat(),
    }
    with open(EMBEDDING_META, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"✅ 嵌入已缓存: {EMBEDDING_CACHE} ({embeddings.shape})")
    return embeddings, meta


# ============================================================
# Step 2: 局部切空间 + 度量张量
# ============================================================
def compute_local_geometry(embeddings, meta, n):
    """对给定 n 计算所有 AA 的局部几何"""
    results = {}
    
    for aa in POLYX_AAS:
        # 获取该 n 和 AA 的所有随机序列 embedding
        label = f"Random_Poly{aa}_n{n}"
        idxs = [i for i, l in enumerate(meta['labels']) if l == label]
        X = embeddings[idxs]
        
        if X.shape[0] < Q_TANGENT + 2:
            print(f"  ⚠️ Poly{aa}_n{n}: 样本数不足 ({X.shape[0]} < {Q_TANGENT + 2})")
            continue
        
        # PCA 预降维
        if PCA_TARGET < X.shape[1]:
            pca = PCA(n_components=min(PCA_TARGET, X.shape[0] - 1))
            X_pca = pca.fit_transform(X)
        else:
            X_pca = X.copy()
        
        # 中心化
        mu = X_pca.mean(axis=0)
        X_centered = X_pca - mu
        
        # SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        q = min(Q_TANGENT, X.shape[0] - 1, len(S))
        Q = Vt[:q].T  # (pca_dim, q)
        
        # 投影到切空间
        Z = X_centered @ Q  # (n_samples, q)
        
        # 协方差 (Ledoit-Wolf)
        if USE_SHRINKAGE and Z.shape[0] >= 3:
            lw = LedoitWolf().fit(Z)
            C = lw.covariance_
        else:
            C = np.cov(Z.T)
        
        # 度量张量
        g = np.linalg.inv(C + EPSILON * np.eye(q))
        
        # 谱分解
        eigvals = np.linalg.eigvalsh(C)
        eigvals = np.sort(eigvals)[::-1]
        total_var = np.sum(eigvals)
        top5_var = np.sum(eigvals[:5])
        eff_rank_95 = np.searchsorted(np.cumsum(eigvals) / total_var, 0.95) + 1
        
        # 几何量
        spectral_decay = -np.polyfit(np.log(np.arange(1, len(eigvals) + 1)), np.log(eigvals + 1e-12), 1)[0] if len(eigvals) >= 3 else 0
        entropy = -np.sum((eigvals / total_var) * np.log(eigvals / total_var + 1e-12)) if total_var > 0 else 0
        
        results[aa] = {
            'n': n, 'aa': aa,
            'mu': mu, 'Q': Q, 'g': g, 'C': C,
            'q': q, 'n_samples': X.shape[0],
            'eigvals': eigvals,
            'total_variance': float(total_var),
            'top5_ratio': float(top5_var / total_var) if total_var > 0 else 0,
            'eff_rank_95': int(eff_rank_95),
            'spectral_decay': float(spectral_decay),
            'entropy': float(entropy),
            'PR': float(eff_rank_95),
        }
    
    return results


def step2_compute_geometry(embeddings, meta):
    """对所有 n 计算局部几何"""
    print("\n" + "=" * 60)
    print("Phase X.2: 计算局部切空间 + 度量张量")
    print("=" * 60)
    
    geom_cache = DATA_DIR / "phase_x_geometry.pkl"
    if geom_cache.exists():
        print(f"✅ 几何缓存已存在: {geom_cache}")
        with open(geom_cache, 'rb') as f:
            return pickle.load(f)
    
    all_geometry = {}  # {n: {aa: {geom_dict}}}
    
    for n in N_VALUES_LAW2:
        print(f"  n={n}...", end=' ')
        geom = compute_local_geometry(embeddings, meta, n)
        all_geometry[n] = geom
        print(f"完成 ({len(geom)} AA)")
    
    with open(geom_cache, 'wb') as f:
        pickle.dump(all_geometry, f)
    
    print(f"✅ 几何已缓存: {geom_cache}")
    return all_geometry


# ============================================================
# Step 3: 第一定律验证
# ============================================================
def step3_test_law1(embeddings, meta, geometry):
    """第一定律: 几何扰动定律 — 余弦一致性 + CV 比较"""
    print("\n" + "=" * 60)
    print("Phase X.3: 第一定律 — 几何扰动定律验证")
    print("=" * 60)
    
    results = []
    
    # 对每个 n, 计算 PolyG→PolyX 扰动
    for n in N_VALUES_LAW2:
        if n not in geometry or 'G' not in geometry[n]:
            continue
        
        g_geom = geometry[n]['G']
        Q_G = g_geom['Q']
        g_G = g_geom['g']
        mu_G = g_geom['mu']
        
        for target_aa in ['S', 'E', 'L', 'K']:
            if target_aa not in geometry[n]:
                continue
            
            # 获取 PolyG 和 PolyX 的 embedding
            g_id = f"PolyX_PolyG_{n}"
            t_id = f"PolyX_Poly{target_aa}_{n}"
            
            g_idx = meta['ids'].index(g_id) if g_id in meta['ids'] else None
            t_idx = meta['ids'].index(t_id) if t_id in meta['ids'] else None
            
            if g_idx is None or t_idx is None:
                continue
            
            x_G = embeddings[g_idx]
            x_X = embeddings[t_idx]
            
            # 原始位移
            delta_x = x_X - x_G
            raw_norm_sq = float(np.sum(delta_x ** 2))
            
            # 投影到切空间
            # 先 PCA 降维
            if PCA_TARGET < EMBEDDING_DIM:
                # 对 G 的ensemble做PCA
                label = f"Random_PolyG_n{n}"
                idxs = [i for i, l in enumerate(meta['labels']) if l == label]
                X_G_ensemble = embeddings[idxs]
                pca = PCA(n_components=min(PCA_TARGET, X_G_ensemble.shape[0] - 1))
                pca.fit(X_G_ensemble)
                delta_x_pca = pca.transform(delta_x.reshape(1, -1)).flatten()
            else:
                delta_x_pca = delta_x
            
            z_P = Q_G.T @ delta_x_pca
            C_geo = float(z_P @ g_G @ z_P)
            bare_vector = sqrtm(g_G + 1e-6 * np.eye(g_G.shape[0])) @ z_P
            
            results.append({
                'n': n, 'source_aa': 'G', 'target_aa': target_aa,
                'perturbation': f'G→{target_aa}',
                'raw_norm_sq': raw_norm_sq,
                'C_geo': C_geo,
                'z_norm': float(np.linalg.norm(z_P)),
                'bare_vector_norm': float(np.linalg.norm(bare_vector)),
            })
    
    df = pd.DataFrame(results)
    df.to_csv(TABLES_DIR / 'phase_x_law1_perturbations.csv', index=False)
    
    # --- 余弦一致性检验 ---
    print("\n--- 余弦一致性检验 ---")
    consistency_results = []
    
    for pert in df['perturbation'].unique():
        pert_df = df[df['perturbation'] == pert]
        if len(pert_df) < 3:
            continue
        
        # Raw cosine consistency
        raw_vecs = []
        geo_vecs = []
        for _, row in pert_df.iterrows():
            n = row['n']
            if n not in geometry or 'G' not in geometry[n]:
                continue
            g_geom = geometry[n]['G']
            Q = g_geom['Q']
            g = g_geom['g']
            
            g_idx = meta['ids'].index(f"PolyX_PolyG_{n}")
            t_idx = meta['ids'].index(f"PolyX_Poly{row['target_aa']}_{n}")
            delta_x = embeddings[t_idx] - embeddings[g_idx]
            
            if PCA_TARGET < EMBEDDING_DIM:
                label = f"Random_PolyG_n{n}"
                idxs = [i for i, l in enumerate(meta['labels']) if l == label]
                X_G_ensemble = embeddings[idxs]
                pca = PCA(n_components=min(PCA_TARGET, X_G_ensemble.shape[0] - 1))
                pca.fit(X_G_ensemble)
                delta_x_pca = pca.transform(delta_x.reshape(1, -1)).flatten()
            else:
                delta_x_pca = delta_x
            
            raw_vecs.append(delta_x_pca)
            z = Q.T @ delta_x_pca
            v_hat = sqrtm(g + 1e-6 * np.eye(g.shape[0])) @ z
            geo_vecs.append(v_hat)
        
        # 计算两两余弦
        def cosine_consistency(vecs):
            m = len(vecs)
            if m < 2:
                return 0
            cos_sum = 0
            count = 0
            for i in range(m):
                for j in range(i + 1, m):
                    cos = np.dot(vecs[i], vecs[j]) / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]) + 1e-12)
                    cos_sum += cos
                    count += 1
            return cos_sum / count if count > 0 else 0
        
        cons_raw = cosine_consistency(raw_vecs)
        cons_geo = cosine_consistency(geo_vecs)
        
        consistency_results.append({
            'perturbation': pert,
            'n_pairs': len(pert_df),
            'cons_raw': cons_raw,
            'cons_geo': cons_geo,
            'cons_diff': cons_geo - cons_raw,
            'law1_supported': cons_geo > cons_raw,
        })
    
    cons_df = pd.DataFrame(consistency_results)
    cons_df.to_csv(TABLES_DIR / 'phase_x_law1_consistency.csv', index=False)
    
    # --- CV 比较 ---
    print("\n--- CV 比较检验 ---")
    cv_results = []
    for pert in df['perturbation'].unique():
        pert_df = df[df['perturbation'] == pert]
        if len(pert_df) < 3:
            continue
        
        cv_raw = pert_df['raw_norm_sq'].std() / pert_df['raw_norm_sq'].mean()
        cv_geo = pert_df['C_geo'].std() / pert_df['C_geo'].mean()
        
        cv_results.append({
            'perturbation': pert,
            'cv_raw': cv_raw,
            'cv_geo': cv_geo,
            'cv_diff': cv_raw - cv_geo,
            'law1_cv_supported': cv_geo < cv_raw,
        })
    
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(TABLES_DIR / 'phase_x_law1_cv.csv', index=False)
    
    # 打印结果
    n_supported = cons_df['law1_supported'].sum()
    n_total = len(cons_df)
    print(f"\n余弦一致性: {n_supported}/{n_total} 扰动支持第一定律")
    print(cons_df.to_string())
    
    n_cv_supported = cv_df['law1_cv_supported'].sum()
    print(f"\nCV比较: {n_cv_supported}/{len(cv_df)} 扰动支持第一定律")
    print(cv_df.to_string())
    
    return {'consistency': cons_df, 'cv': cv_df, 'perturbations': df}


# ============================================================
# Step 4: 第二定律验证
# ============================================================
def step4_test_law2(geometry):
    """第二定律: 生物约束场方程 — 耦合矩阵 K 估计"""
    print("\n" + "=" * 60)
    print("Phase X.4: 第二定律 — 生物约束场方程验证")
    print("=" * 60)
    
    # 构建数据矩阵
    rows = []
    for n in N_VALUES_LAW2:
        if n not in geometry:
            continue
        for aa in POLYX_AAS:
            if aa not in geometry[n]:
                continue
            g = geometry[n][aa]
            props = AA_PROPERTIES.get(aa, {})
            
            rows.append({
                'n': n, 'aa': aa,
                'hydrophobicity': props.get('hydrophobicity', 0),
                'charge': props.get('charge', 0),
                'volume': props.get('volume', 0),
                'atom_count': props.get('atom_count', 0),
                'PR': g['PR'],
                'spectral_decay': g['spectral_decay'],
                'entropy': g['entropy'],
                'total_variance': g['total_variance'],
                'top5_ratio': g['top5_ratio'],
                'eff_rank_95': g['eff_rank_95'],
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / 'phase_x_law2_data.csv', index=False)
    
    # 生物源项向量 T^bio
    bio_cols = ['n', 'hydrophobicity', 'charge', 'volume', 'atom_count']
    # 几何观测向量 Y^geom
    geom_cols = ['PR', 'spectral_decay', 'entropy', 'total_variance', 'top5_ratio', 'eff_rank_95']
    
    X = df[bio_cols].values
    X_scaled = StandardScaler().fit_transform(X)
    
    # 耦合矩阵 K 估计 (每个几何量分别回归)
    coupling_results = []
    for geom_col in geom_cols:
        y = df[geom_col].values
        
        # 线性回归 (Ridge)
        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        ridge.fit(X_scaled, y)
        
        # 交叉验证 R²
        cv_scores = cross_val_score(ridge, X_scaled, y, cv=5)
        
        coupling_results.append({
            'geometry_feature': geom_col,
            'R2_train': float(ridge.score(X_scaled, y)),
            'R2_cv_mean': float(cv_scores.mean()),
            'R2_cv_std': float(cv_scores.std()),
            'coefficients': {col: float(coef) for col, coef in zip(bio_cols, ridge.coef_)},
            'intercept': float(ridge.intercept_),
        })
    
    coupling_df = pd.DataFrame(coupling_results)
    coupling_df.to_csv(TABLES_DIR / 'phase_x_law2_coupling.csv', index=False)
    
    # 简化版: 对每个 (几何量, 生物源项) 对估计 Spearman 相关
    spearman_results = []
    for geom_col in geom_cols:
        for bio_col in bio_cols:
            r, p = spearmanr(df[bio_col], df[geom_col])
            spearman_results.append({
                'geometry': geom_col,
                'bio_source': bio_col,
                'spearman_r': r,
                'p_value': p,
                'significant': p < 0.05,
            })
    
    spear_df = pd.DataFrame(spearman_results)
    spear_df.to_csv(TABLES_DIR / 'phase_x_law2_spearman.csv', index=False)
    
    # 打印结果
    print(f"\n耦合矩阵 K 估计 (R²):")
    for r in coupling_results:
        sig = "✅" if r['R2_cv_mean'] > 0.1 else "⚠️" if r['R2_cv_mean'] > 0 else "❌"
        print(f"  {sig} {r['geometry_feature']}: R²_cv={r['R2_cv_mean']:.4f} (train={r['R2_train']:.4f})")
    
    n_sig = spear_df['significant'].sum()
    print(f"\nSpearman 相关: {n_sig}/{len(spear_df)} 显著")
    
    return {'coupling': coupling_df, 'spearman': spear_df, 'data': df}


# ============================================================
# Step 5: 第三定律验证
# ============================================================
def step5_test_law3(embeddings, meta, geometry):
    """第三定律: 低作用量路径定律 — 路径作用量 + Null 检验"""
    print("\n" + "=" * 60)
    print("Phase X.5: 第三定律 — 低作用量路径定律验证")
    print("=" * 60)
    
    # 定义生物路径: PolyG_5 → PolyG_10 → PolyG_20 → PolyG_30 → PolyG_50
    bio_path_n = [5, 10, 20, 30, 50]
    
    # 计算生物路径的 W₂ 距离
    bio_W2 = []
    bio_delta = []
    for i in range(len(bio_path_n) - 1):
        n1, n2 = bio_path_n[i], bio_path_n[i + 1]
        if n1 not in geometry or n2 not in geometry or 'G' not in geometry[n1] or 'G' not in geometry[n2]:
            continue
        
        g1 = geometry[n1]['G']
        g2 = geometry[n2]['G']
        
        # W₂ 距离 (高斯近似)
        mu1, C1 = g1['mu'], g1['C']
        mu2, C2 = g2['mu'], g2['C']
        
        # 均值差
        mean_diff = np.sum((mu1 - mu2) ** 2)
        
        # 协方差项
        C1_sqrt = sqrtm(C1 + 1e-8 * np.eye(C1.shape[0]))
        inner = sqrtm(C1_sqrt @ C2 @ C1_sqrt + 1e-8 * np.eye(C1_sqrt.shape[0]))
        trace_term = np.trace(C1 + C2 - 2 * inner)
        W2_sq = mean_diff + max(0, trace_term)
        
        bio_W2.append(np.sqrt(W2_sq))
        bio_delta.append({
            'from_n': n1, 'to_n': n2,
            'delta_PR': g2['PR'] - g1['PR'],
            'delta_spectral_decay': g2['spectral_decay'] - g1['spectral_decay'],
            'delta_entropy': g2['entropy'] - g1['entropy'],
            'W2': np.sqrt(W2_sq),
        })
    
    # 生成 Null 路径 (随机打乱 n 顺序)
    n_null = 100
    null_W2_sums = []
    for _ in range(n_null):
        null_path = random.sample(bio_path_n, len(bio_path_n))
        null_W2 = []
        for i in range(len(null_path) - 1):
            n1, n2 = null_path[i], null_path[i + 1]
            if n1 not in geometry or n2 not in geometry or 'G' not in geometry[n1] or 'G' not in geometry[n2]:
                continue
            g1 = geometry[n1]['G']
            g2 = geometry[n2]['G']
            mu1, C1 = g1['mu'], g1['C']
            mu2, C2 = g2['mu'], g2['C']
            mean_diff = np.sum((mu1 - mu2) ** 2)
            C1_sqrt = sqrtm(C1 + 1e-8 * np.eye(C1.shape[0]))
            inner = sqrtm(C1_sqrt @ C2 @ C1_sqrt + 1e-8 * np.eye(C1_sqrt.shape[0]))
            trace_term = np.trace(C1 + C2 - 2 * inner)
            W2_sq = mean_diff + max(0, trace_term)
            null_W2.append(np.sqrt(W2_sq))
        null_W2_sums.append(np.sum(null_W2))
    
    bio_W2_sum = np.sum(bio_W2)
    null_mean = np.mean(null_W2_sums)
    null_std = np.std(null_W2_sums)
    z_score = (bio_W2_sum - null_mean) / null_std if null_std > 0 else 0
    p_value = np.mean([1 if w <= bio_W2_sum else 0 for w in null_W2_sums])
    
    law3_results = {
        'bio_path': bio_path_n,
        'bio_W2_sum': float(bio_W2_sum),
        'null_W2_mean': float(null_mean),
        'null_W2_std': float(null_std),
        'z_score': float(z_score),
        'p_value': float(p_value),
        'law3_supported': bio_W2_sum < null_mean,
        'bio_delta': bio_delta,
        'n_null': n_null,
    }
    
    with open(DATA_DIR / 'phase_x_law3_results.json', 'w') as f:
        json.dump(law3_results, f, indent=2, default=float)
    
    print(f"\n生物路径 W₂ 总和: {bio_W2_sum:.4f}")
    print(f"Null 路径 W₂ 均值: {null_mean:.4f} ± {null_std:.4f}")
    print(f"Z-score: {z_score:.4f}")
    print(f"Permutation p-value: {p_value:.4f}")
    print(f"第三定律支持: {'✅ 是' if law3_results['law3_supported'] else '❌ 否'}")
    
    return law3_results


# ============================================================
# Step 6: 跨空间验证
# ============================================================
def step6_cross_space_validation(geometry):
    """跨空间验证: Embedding vs Cα 几何量对比"""
    print("\n" + "=" * 60)
    print("Phase X.6: 跨空间验证 (Embedding vs Cα)")
    print("=" * 60)
    
    # 加载 Cα 空间几何数据
    ca_path = FIELD_THEORY / 'data' / 'dms' / 'phase9_systemwide' / 'systemwide_enhanced_geometry.csv'
    if not ca_path.exists():
        print(f"⚠️ Cα 数据未找到: {ca_path}")
        print("  跳过跨空间验证")
        return None
    
    ca_df = pd.read_csv(ca_path)
    
    # 提取 Embedding 空间几何量
    emb_rows = []
    for n in N_VALUES_LAW2:
        if n not in geometry:
            continue
        for aa in POLYX_AAS:
            if aa not in geometry[n]:
                continue
            g = geometry[n][aa]
            emb_rows.append({
                'n': n, 'aa': aa,
                'PR_emb': g['PR'],
                'spectral_decay_emb': g['spectral_decay'],
                'entropy_emb': g['entropy'],
                'total_variance_emb': g['total_variance'],
                'top5_ratio_emb': g['top5_ratio'],
            })
    
    emb_df = pd.DataFrame(emb_rows)
    
    # 匹配 Cα 数据 (PolyX 纯序列)
    ca_polyx = ca_df[ca_df['category'].str.contains('PolyX_Poly', na=False)]
    
    # 简化: 计算每个 AA 的平均几何量
    comparison = []
    for aa in POLYX_AAS:
        emb_aa = emb_df[emb_df['aa'] == aa]
        ca_aa = ca_polyx[ca_polyx['aa'] == aa] if 'aa' in ca_polyx.columns else None
        
        if ca_aa is None or len(ca_aa) == 0:
            continue
        
        for metric in ['PR', 'spectral_decay', 'entropy']:
            emb_col = f'{metric}_emb'
            ca_col = metric if metric in ca_aa.columns else None
            if ca_col is None:
                continue
            
            # 按 n 对齐
            merged = emb_aa[['n', emb_col]].merge(
                ca_aa[['n', ca_col]], on='n', how='inner'
            )
            if len(merged) < 5:
                continue
            
            r, p = pearsonr(merged[emb_col], merged[ca_col])
            comparison.append({
                'aa': aa, 'metric': metric,
                'pearson_r': r, 'p_value': p,
                'n_pairs': len(merged),
                'significant': p < 0.05,
            })
    
    comp_df = pd.DataFrame(comparison)
    comp_df.to_csv(TABLES_DIR / 'phase_x_cross_space.csv', index=False)
    
    print("\n跨空间相关性 (Embedding vs Cα):")
    for _, row in comp_df.iterrows():
        sig = "✅" if row['significant'] else "❌"
        print(f"  {sig} {row['aa']} {row['metric']}: r={row['pearson_r']:.3f}, p={row['p_value']:.4f}")
    
    return comp_df


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Phase X: Embedding空间三定律验证')
    parser.add_argument('--step', type=str, default='all',
                        choices=['1', '2', '3', '4', '5', '6', 'all'],
                        help='执行步骤 (1-6 或 all)')
    parser.add_argument('--force-recompute', action='store_true',
                        help='强制重新计算 embeddings')
    parser.add_argument('--no-monitor', action='store_true',
                        help='不输出监控日志')
    args = parser.parse_args()
    
    start_time = datetime.now()
    print(f"Phase X 启动: {start_time.isoformat()}")
    print(f"  Embedding 维度: {EMBEDDING_DIM}")
    print(f"  切空间维度: {Q_TANGENT}")
    print(f"  PCA 预降维: {PCA_TARGET}")
    print(f"  正则化 ε: {EPSILON}")
    print(f"  Ledoit-Wolf: {USE_SHRINKAGE}")
    print(f"  随机序列/n: {N_RANDOM_PER_N}")
    
    results = {}
    
    # Step 1: Embeddings
    if args.step in ['1', 'all']:
        embeddings, meta = step1_compute_embeddings(force_recompute=args.force_recompute)
        results['embeddings_shape'] = embeddings.shape
        results['n_sequences'] = meta['n_sequences']
    else:
        if not EMBEDDING_CACHE.exists():
            print("❌ Embedding 缓存不存在，请先运行 --step 1")
            return
        data = np.load(EMBEDDING_CACHE, allow_pickle=True)
        embeddings = data['embeddings']
        with open(EMBEDDING_META, 'r') as f:
            meta = json.load(f)
        print(f"✅ 加载缓存: {embeddings.shape}")
    
    # Step 2: 几何计算
    if args.step in ['2', 'all']:
        geometry = step2_compute_geometry(embeddings, meta)
        results['n_n_values'] = len(geometry)
    else:
        geom_cache = DATA_DIR / "phase_x_geometry.pkl"
        if not geom_cache.exists():
            print("❌ 几何缓存不存在，请先运行 --step 2")
            return
        with open(geom_cache, 'rb') as f:
            geometry = pickle.load(f)
        print(f"✅ 加载几何缓存: {len(geometry)} n值")
    
    # Step 3: 第一定律
    if args.step in ['3', 'all']:
        results['law1'] = step3_test_law1(embeddings, meta, geometry)
    
    # Step 4: 第二定律
    if args.step in ['4', 'all']:
        results['law2'] = step4_test_law2(geometry)
    
    # Step 5: 第三定律
    if args.step in ['5', 'all']:
        results['law3'] = step5_test_law3(embeddings, meta, geometry)
    
    # Step 6: 跨空间验证
    if args.step in ['6', 'all']:
        results['cross_space'] = step6_cross_space_validation(geometry)
    
    # 保存完整结果
    elapsed = (datetime.now() - start_time).total_seconds()
    summary = {
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'steps_completed': args.step,
        'results': {k: str(v) if not isinstance(v, (dict, list, pd.DataFrame)) else 'see files' for k, v in results.items()},
    }
    with open(DATA_DIR / 'phase_x_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Phase X 完成! 耗时: {elapsed:.1f}s")
    print(f"输出目录: {DATA_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()