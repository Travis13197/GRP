#!/usr/bin/env python3
"""
B7: 直接 NPZ  Wasserstein-2 距离计算 (第三定律 直接验证)
=============================================================
替换 B4/B5 中的代理 W_2 方法，使用 BioEmu Cα NPZ 数据直接计算
构象系综之间的 Wasserstein-2 距离。

W_2(P, Q)^2 = ||mu_P - mu_Q||^2 + Tr(Sigma_P + Sigma_Q - 2*(Sigma_P^{1/2} Sigma_Q Sigma_P^{1/2})^{1/2})

路径类型:
  1. PolyG  PolyX (同链长 n, 氨基酸转换) 生物路径
  2. 同AA n  n+1 (链长递增) 生物路径
  3. 随机配对 零路径

输入:
  - test_workflow/polyx_ensemble/output/{seq_id}/batch_*.npz (Cα NPZ, key: pos)
  - field_theory/tables/phase_ensemble_b4_path_action.csv (B4 代理结果, 用于对比)
  - field_theory/tables/phase_ensemble_b5_fullatom_path_action.csv (B5 代理结果, 用于对比)

输出:
  - field_theory/tables/phase_ensemble_npz_direct_w2.csv (NPZ 直接 W_2 路径)
  - field_theory/tables/phase_ensemble_npz_direct_summary.json (对比统计汇总)
"""

import sys, os, json, warnings, gc, time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind, spearmanr, pearsonr
from scipy.linalg import sqrtm, eigh
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
FIELD_THEORY = Path(__file__).parent.parent
SCRIPTS_DIR = FIELD_THEORY / "scripts"
TABLES_DIR = FIELD_THEORY / "tables"
DATA_DIR = FIELD_THEORY / "data"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# NPZ data directory
NPZ_OUTPUT_DIR = PROJECT_ROOT / "test_workflow/polyx_ensemble/output"

# Proxy results for comparison
B4_PROXY = TABLES_DIR / "phase_ensemble_b4_path_action.csv"
B5_PROXY = TABLES_DIR / "phase_ensemble_b5_fullatom_path_action.csv"

# Output files
OUTPUT_CSV = TABLES_DIR / "phase_ensemble_npz_direct_w2.csv"
OUTPUT_SUMMARY = TABLES_DIR / "phase_ensemble_npz_direct_summary.json"

# ============================================================
# Configuration
# ============================================================
NPZ_KEY = "pos"                          # Key in NPZ file
N_RESAMPLE = 250                         # Expected samples per sequence
PCA_N_COMPONENTS = 50                    # Top PCs for covariance truncation
REGULARIZATION_EPS = 1e-6                # Regularization for covariance
RANDOM_SEED = 42                         # For null path generation
BATCH_SIZE = 50                          # Batch size for loading
AA_TYPES = ["G", "A", "S", "E", "L", "K", "V", "I", "F"]  # All available AA types
N_RANGE = range(4, 51)                   # Chain length range

print("=" * 70)
print("B7: 直接 NPZ  Wasserstein-2 距离计算 (第三定律  直接验证)")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)


# ============================================================
# Utility Functions
# ============================================================

def extract_aa(seq_id):
    """Extract amino acid type from seq_id"""
    parts = str(seq_id).split("_")
    for p in parts:
        if p.startswith("Poly") and len(p) == 5 and p[4] != "X":
            return p[4]
    return "unknown"


def extract_n(seq_id):
    """Extract chain length from seq_id"""
    parts = str(seq_id).split("_")
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return 0


def robust_sqrtm(A, eps=REGULARIZATION_EPS, max_retries=3):
    """
    Robust matrix square root with regularization.
    Falls back to eigendecomposition if sqrtm fails.
    """
    # Add regularization
    n = A.shape[0]
    A_reg = A + eps * np.eye(n)
    
    for attempt in range(max_retries):
        try:
            return sqrtm(A_reg).real
        except Exception:
            if attempt < max_retries - 1:
                eps *= 10
                A_reg = A + eps * np.eye(n)
            else:
                # Fallback: eigendecomposition
                eigvals, eigvecs = eigh(A_reg)
                eigvals = np.maximum(eigvals, 0)
                return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def compute_wasserstein2(mu_P, Sigma_P, mu_Q, Sigma_Q, eps=REGULARIZATION_EPS):
    """
    Compute W_2^2(P, Q) = ||mu_P - mu_Q||^2 + Bures(Sigma_P, Sigma_Q)
    
    Bures(Sigma_P, Sigma_Q) = Tr(Sigma_P + Sigma_Q - 2*(Sigma_P^{1/2} Sigma_Q Sigma_P^{1/2})^{1/2})
    """
    # Mean term
    mean_diff = np.sum((mu_P - mu_Q) ** 2)
    
    # Covariance term (Bures metric)
    # Compute Sigma_P^{1/2}
    sqrt_P = robust_sqrtm(Sigma_P, eps)
    
    # Compute Sigma_P^{1/2} * Sigma_Q * Sigma_P^{1/2}
    inner = sqrt_P @ Sigma_Q @ sqrt_P
    
    # Compute its square root
    sqrt_inner = robust_sqrtm(inner, eps)
    
    # Bures metric
    trace_P = np.trace(Sigma_P)
    trace_Q = np.trace(Sigma_Q)
    trace_cross = 2.0 * np.trace(sqrt_inner)
    
    bures = trace_P + trace_Q - trace_cross
    bures = max(bures, 0.0)  # Numerical safeguard
    
    w2_sq = mean_diff + bures
    w2 = np.sqrt(max(w2_sq, 0.0))
    
    return w2


def pca_truncate_covariance(pos, n_components=PCA_N_COMPONENTS):
    """
    PCA truncation for large covariance matrices.
    Returns (mu_truncated, Sigma_truncated) in reduced space.
    
    pos: (n_samples, n_atoms, 3) or (n_samples, n_residues, 3)
    Returns: mu (n_components,), Sigma (n_components, n_components)
    """
    n_samples = pos.shape[0]
    original_dim = pos.shape[1] * 3  # Flattened dimension
    
    # Flatten
    X = pos.reshape(n_samples, -1)  # (n_samples, original_dim)
    
    # If original dim is small enough, don't truncate
    if original_dim <= n_components:
        mu = np.mean(X, axis=0)
        Sigma = np.cov(X, rowvar=False)
        return mu, Sigma
    
    # Center
    X_centered = X - np.mean(X, axis=0)
    
    # PCA via SVD (more numerically stable for wide matrices)
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Keep top n_components
    s_trunc = s[:n_components]
    Vt_trunc = Vt[:n_components]  # (n_components, original_dim)
    
    # Projected data
    # X_proj = X_centered @ Vt_trunc.T, but we compute directly
    # Actually, SVD gives: X_centered = U @ diag(s) @ Vt
    # So projection onto top PCs: X_proj = U[:, :n_components] @ diag(s[:n_components])
    X_proj = U[:, :n_components] * s_trunc[np.newaxis, :]  # (n_samples, n_components)
    
    mu = np.mean(X_proj, axis=0)
    Sigma = np.cov(X_proj, rowvar=False)
    
    # Add small regularization
    Sigma += REGULARIZATION_EPS * np.eye(n_components)

    return mu, Sigma


def compute_w2_joint(pos_P, pos_Q, n_components=PCA_N_COMPONENTS):
    """
    数学正确的 W₂ 计算: 两条序列投影到**联合 PCA 共同基底**后再计算 W₂。

    修复审计发现 C3: 原实现对每条序列独立 SVD, 两系综 μ/Σ 处于不同正交基底,
    私有基底间的 W₂ 比较不具备 Wasserstein 距离的内蕴度量性质。

    方法:
      1. 长度对齐: 取前 min(n_res_P, n_res_Q) 个残基 (chain_growth n vs n+1 合理)
      2. 堆叠两系综样本做联合 SVD, 得到共同基底 Vt_joint
      3. 两系综分别投影到同一 Vt_joint, 再计算各自 μ/Σ
      4. 在共同空间内计算 W₂ (compute_wasserstein2)

    pos_P, pos_Q: (n_samples, n_residues, 3)
    Returns: w2 (float) or None if failed
    """
    n_comp = n_components
    # 1. 长度对齐到共同残基数
    n_res = min(pos_P.shape[1], pos_Q.shape[1])
    if n_res < 2:
        return None
    Xp = pos_P[:, :n_res, :].reshape(pos_P.shape[0], -1)  # (n_sP, 3*n_res)
    Xq = pos_Q[:, :n_res, :].reshape(pos_Q.shape[0], -1)  # (n_sQ, 3*n_res)

    dim = Xp.shape[1]
    # 2. 联合堆叠 + 联合中心化 (用合并均值, 保证两系综在同一坐标系)
    Xall = np.vstack([Xp, Xq])
    mean_all = np.mean(Xall, axis=0)
    Xall_c = Xall - mean_all

    if dim <= n_comp:
        # 无需截断: 直接在原始 (共同) 空间计算
        Xp_c = Xp - mean_all
        Xq_c = Xq - mean_all
        mu_P = np.mean(Xp_c, axis=0)
        mu_Q = np.mean(Xq_c, axis=0)
        Sigma_P = np.cov(Xp_c, rowvar=False) + REGULARIZATION_EPS * np.eye(dim)
        Sigma_Q = np.cov(Xq_c, rowvar=False) + REGULARIZATION_EPS * np.eye(dim)
        return compute_wasserstein2(mu_P, Sigma_P, mu_Q, Sigma_Q)

    # 3. 联合 SVD 求共同基底
    U, s, Vt = np.linalg.svd(Xall_c, full_matrices=False)
    Vt_joint = Vt[:n_comp]  # (n_comp, dim) 共同基底

    # 4. 两系综投影到同一共同基底
    Xp_proj = (Xp - mean_all) @ Vt_joint.T  # (n_sP, n_comp)
    Xq_proj = (Xq - mean_all) @ Vt_joint.T  # (n_sQ, n_comp)

    mu_P = np.mean(Xp_proj, axis=0)
    mu_Q = np.mean(Xq_proj, axis=0)
    Sigma_P = np.cov(Xp_proj, rowvar=False) + REGULARIZATION_EPS * np.eye(n_comp)
    Sigma_Q = np.cov(Xq_proj, rowvar=False) + REGULARIZATION_EPS * np.eye(n_comp)

    return compute_wasserstein2(mu_P, Sigma_P, mu_Q, Sigma_Q)


# ============================================================
# Step 1: Load NPZ data for all sequences
# ============================================================
print("\n[1/7] 加载 NPZ 数据...")

def load_sequence_npz(seq_id):
    """
    Load all NPZ batch files for a sequence and concatenate.
    Returns pos array (n_samples, n_residues, 3) or None if no data.
    """
    seq_dir = NPZ_OUTPUT_DIR / seq_id
    if not seq_dir.is_dir():
        return None
    
    # Find all batch NPZ files (exclude special files like .samples.xtc_offsets.npz)
    npz_files = sorted([
        f for f in seq_dir.glob("batch_*.npz")
        if not f.name.startswith(".samples")
    ])
    
    if not npz_files:
        return None
    
    # Load and concatenate
    all_pos = []
    for npz_file in npz_files:
        try:
            data = np.load(npz_file)
            if NPZ_KEY in data:
                all_pos.append(data[NPZ_KEY])
            data.close()
        except Exception as e:
            print(f"    Warning: Failed to load {npz_file}: {e}")
            continue
    
    if not all_pos:
        return None
    
    pos = np.concatenate(all_pos, axis=0)  # (total_samples, n_residues, 3)
    return pos


# Discover available sequences
all_seq_dirs = sorted([d.name for d in NPZ_OUTPUT_DIR.iterdir() if d.is_dir()])
print(f"  Total sequence directories: {len(all_seq_dirs)}")

# Load all sequences with NPZ data
seq_data = {}       # seq_id -> pos array
seq_mu = {}         # seq_id -> mean vector
seq_Sigma = {}      # seq_id -> covariance matrix
seq_n_residues = {} # seq_id -> n_residues
seq_n_samples = {}  # seq_id -> n_samples

loaded_count = 0
skipped_count = 0
error_count = 0

for seq_id in all_seq_dirs:
    if seq_id.startswith("."):
        continue
    
    pos = load_sequence_npz(seq_id)
    if pos is None:
        skipped_count += 1
        continue
    
    n_samples = pos.shape[0]
    n_residues = pos.shape[1]
    
    if n_samples < 3:  # Need at least 3 samples for covariance
        skipped_count += 1
        continue
    
    seq_data[seq_id] = pos
    seq_n_samples[seq_id] = n_samples
    seq_n_residues[seq_id] = n_residues
    loaded_count += 1
    
    if loaded_count % 50 == 0:
        print(f"    Loaded {loaded_count} sequences...")

print(f"\n  Loaded: {loaded_count} sequences with NPZ data")
print(f"  Skipped: {skipped_count} sequences (no NPZ or too few samples)")

# ============================================================
# Step 2: Compute per-sequence statistics (mu, Sigma)
# ============================================================
print("\n[2/7] 计算每个序列的统计量 (mu, Sigma)...")

compute_count = 0
for seq_id, pos in seq_data.items():
    try:
        mu, Sigma = pca_truncate_covariance(pos, PCA_N_COMPONENTS)
        seq_mu[seq_id] = mu
        seq_Sigma[seq_id] = Sigma
        compute_count += 1
    except Exception as e:
        print(f"    Warning: Failed to compute stats for {seq_id}: {e}")
        error_count += 1
        continue

print(f"  Computed stats for {compute_count} sequences")

# Build metadata DataFrame
seq_metadata = []
for seq_id in seq_mu.keys():
    seq_metadata.append({
        "seq_id": seq_id,
        "aa_type": extract_aa(seq_id),
        "n": extract_n(seq_id),
        "n_residues": seq_n_residues.get(seq_id, 0),
        "n_samples": seq_n_samples.get(seq_id, 0),
        "mu_dim": seq_mu[seq_id].shape[0],
        "sigma_dim": seq_Sigma[seq_id].shape[0],
    })
df_metadata = pd.DataFrame(seq_metadata)
print(f"  AA types: {df_metadata['aa_type'].value_counts().to_dict()}")
print(f"  Chain lengths: {df_metadata['n'].min()} - {df_metadata['n'].max()}")
print(f"  Total: {len(df_metadata)} sequences with valid stats")

# Index map for fast lookup
seq_ids = sorted(seq_mu.keys())
seq_id_to_idx = {sid: i for i, sid in enumerate(seq_ids)}

# ============================================================
# Step 3: Define biological paths and null paths
# ============================================================
print("\n[3/7] 定义生物路径和零路径...")

# --- Biological Paths ---

# Path type 1: PolyG -> PolyX, same chain length n
bio_paths_g2x = []
for n in N_RANGE:
    # Find PolyG at this n
    g_seqs = df_metadata[
        (df_metadata["aa_type"] == "G") & (df_metadata["n"] == n)
    ]["seq_id"].tolist()
    if not g_seqs:
        continue
    g_id = g_seqs[0]
    if g_id not in seq_mu:
        continue
    
    for aa in AA_TYPES:
        if aa == "G":
            continue
        x_seqs = df_metadata[
            (df_metadata["aa_type"] == aa) & (df_metadata["n"] == n)
        ]["seq_id"].tolist()
        if not x_seqs:
            continue
        x_id = x_seqs[0]
        if x_id not in seq_mu:
            continue
        
        bio_paths_g2x.append({
            "path_type": f"G_to_{aa}",
            "seq_from": g_id,
            "seq_to": x_id,
            "aa_from": "G",
            "aa_to": aa,
            "n": n,
            "category": "aa_change",
        })

# Path type 2: Same AA, n -> n+1 (chain length increment)
bio_paths_n2n1 = []
for aa in AA_TYPES:
    for n in range(4, 50):
        s1_seqs = df_metadata[
            (df_metadata["aa_type"] == aa) & (df_metadata["n"] == n)
        ]["seq_id"].tolist()
        s2_seqs = df_metadata[
            (df_metadata["aa_type"] == aa) & (df_metadata["n"] == n + 1)
        ]["seq_id"].tolist()
        if not s1_seqs or not s2_seqs:
            continue
        s1_id = s1_seqs[0]
        s2_id = s2_seqs[0]
        if s1_id not in seq_mu or s2_id not in seq_mu:
            continue
        
        bio_paths_n2n1.append({
            "path_type": f"{aa}_n{n}_to_n{n+1}",
            "seq_from": s1_id,
            "seq_to": s2_id,
            "aa_from": aa,
            "aa_to": aa,
            "n": n,
            "category": "chain_growth",
        })

bio_paths = bio_paths_g2x + bio_paths_n2n1
print(f"  生物路径 (G->X): {len(bio_paths_g2x)} 条")
print(f"  生物路径 (n->n+1): {len(bio_paths_n2n1)} 条")
print(f"  生物路径总计: {len(bio_paths)} 条")

# --- Null Paths ---
np.random.seed(RANDOM_SEED)
n_null = min(len(bio_paths) * 3, 500)
null_paths = []

for _ in range(n_null):
    i, j = np.random.choice(len(seq_ids), 2, replace=False)
    s1, s2 = seq_ids[i], seq_ids[j]
    aa1 = extract_aa(s1)
    aa2 = extract_aa(s2)
    n1 = extract_n(s1)
    n2 = extract_n(s2)
    null_paths.append({
        "path_type": "null_random",
        "seq_from": s1,
        "seq_to": s2,
        "aa_from": aa1,
        "aa_to": aa2,
        "n_from": n1,
        "n_to": n2,
        "category": "null",
    })

print(f"  零路径: {len(null_paths)} 条")

# ============================================================
# Step 4: Compute direct W_2 distances
# ============================================================
print("\n[4/7] 计算直接 NPZ W_2 距离...")

def compute_path_w2(paths, label=""):
    """Compute W_2 distance for each path using direct NPZ data."""
    results = []
    n_paths = len(paths)
    n_computed = 0
    n_skipped = 0
    
    for idx, p in enumerate(paths):
        s1, s2 = p["seq_from"], p["seq_to"]
        
        if s1 not in seq_mu or s2 not in seq_mu:
            n_skipped += 1
            continue
        
        try:
            # ⚠️ AUDIT-C3 WARNING: seq_mu/seq_Sigma 由 pca_truncate_covariance 对每条序列
            # 独立 SVD 得到, 两系综处于不同正交基底, 下方 W₂ 比较数学上不具备内蕴度量性质。
            # 正确做法: 改用 compute_w2_joint(pos[s1], pos[s2]) (联合 PCA 共同基底) 重算。
            # 本数值仅作历史对照, 论文引用前须用 compute_w2_joint 重新计算全部路径。
            w2 = compute_wasserstein2(
                seq_mu[s1], seq_Sigma[s1],
                seq_mu[s2], seq_Sigma[s2],
                eps=REGULARIZATION_EPS
            )
            
            # Action = W_2 + penalty terms
            action = w2
            
            # AA change penalty (same as B4/B5 convention)
            if p.get("aa_from", "?") != p.get("aa_to", "?"):
                action += 0.1
            
            # Delta_n penalty
            n_from = p.get("n", p.get("n_from", 0))
            n_to = p.get("n", p.get("n_to", 0))
            delta_n = abs(n_to - n_from)
            if delta_n > 0:
                action += 0.05 * np.log(delta_n + 1)
            
            results.append({
                **p,
                "W2_npz": round(float(w2), 6),
                "action_npz": round(float(action), 6),
                "is_bio": "null" not in p.get("path_type", ""),
                "n_from": n_from,
                "n_to": n_to,
                "delta_n": delta_n,
            })
            n_computed += 1
            
        except Exception as e:
            n_skipped += 1
            if n_skipped <= 5:
                print(f"    Warning: Failed W_2 for {s1} -> {s2}: {e}")
            continue
        
        if (idx + 1) % 100 == 0:
            print(f"    {label}: {idx + 1}/{n_paths} paths computed...")
    
    print(f"    {label}: {n_computed} computed, {n_skipped} skipped")
    return results

# Compute bio paths
t0 = time.time()
bio_results = compute_path_w2(bio_paths, "Bio paths")
t1 = time.time()
print(f"  Bio paths computed in {t1 - t0:.1f}s")

# Compute null paths
null_results = compute_path_w2(null_paths, "Null paths")
t2 = time.time()
print(f"  Null paths computed in {t2 - t1:.1f}s")

# Combine all results
all_results = bio_results + null_results
df_all = pd.DataFrame(all_results)

# ============================================================
# Step 5: Third Law test
# ============================================================
print("\n[5/7] 第三定律检验...")

bio_w2 = df_all[df_all["is_bio"] == True]["action_npz"].values
null_w2 = df_all[df_all["is_bio"] == False]["action_npz"].values

print(f"  生物路径数量: {len(bio_w2)}")
print(f"  零路径数量: {len(null_w2)}")

# Statistical tests
u_stat, u_p = mannwhitneyu(bio_w2, null_w2, alternative="less")
t_stat, t_p = ttest_ind(bio_w2, null_w2, alternative="less")

pooled_std = np.sqrt((np.var(bio_w2) + np.var(null_w2)) / 2)
cohens_d = (np.mean(bio_w2) - np.mean(null_w2)) / (pooled_std + 1e-8)
z_score = (np.mean(bio_w2) - np.mean(null_w2)) / (np.std(null_w2) + 1e-8)

print(f"\n  第三定律检验 (NPZ 直接 W_2):")
print(f"    Bio A[gamma] = {np.mean(bio_w2):.4f} +/- {np.std(bio_w2):.4f}")
print(f"    Null A[gamma] = {np.mean(null_w2):.4f} +/- {np.std(null_w2):.4f}")
print(f"    Mann-Whitney p = {u_p:.4e}")
print(f"    t-test p = {t_p:.4e}")
print(f"    Cohen's d = {cohens_d:.4f}")
print(f"    Z-score = {z_score:.4f}")
print(f"    第三定律: {'VERIFIED' if u_p < 0.05 and np.mean(bio_w2) < np.mean(null_w2) else 'NOT verified'}")

# Per-category analysis
category_stats = []
for cat in df_all[df_all["is_bio"] == True]["category"].unique():
    cat_data = df_all[(df_all["is_bio"] == True) & (df_all["category"] == cat)]["action_npz"].values
    if len(cat_data) < 3:
        continue
    u_cat, p_cat = mannwhitneyu(cat_data, null_w2, alternative="less")
    cat_z = (np.mean(cat_data) - np.mean(null_w2)) / (np.std(null_w2) + 1e-8)
    category_stats.append({
        "category": cat,
        "n_paths": len(cat_data),
        "mean_bio": np.mean(cat_data),
        "std_bio": np.std(cat_data),
        "Z_score": cat_z,
        "MannWhitney_p": p_cat,
    })

df_category = pd.DataFrame(category_stats).sort_values("Z_score")
print(f"\n  按类别分析:")
for _, row in df_category.iterrows():
    sig = "***" if row["MannWhitney_p"] < 0.001 else ("**" if row["MannWhitney_p"] < 0.01 else ("*" if row["MannWhitney_p"] < 0.05 else ""))
    print(f"    {row['category']:20s}: n={row['n_paths']:3d}, A={row['mean_bio']:.4f}, Z={row['Z_score']:.2f} {sig}")

# ============================================================
# Step 6: Compare with proxy results from B4/B5
# ============================================================
print("\n[6/7] 与 B4/B5 代理结果对比...")

comparison_data = []

# Load B4 proxy results
if B4_PROXY.exists():
    df_b4 = pd.read_csv(B4_PROXY)
    print(f"  B4 代理结果: {len(df_b4)} 条路径")
    
    # Match by (seq_from, seq_to)
    b4_lookup = {}
    for _, row in df_b4.iterrows():
        key = (str(row["seq_from"]), str(row["seq_to"]))
        b4_lookup[key] = {
            "W2_proxy": row.get("W2_distance", np.nan),
            "action_proxy": row.get("action_A", np.nan),
        }
    
    # Match with our results
    n_matched = 0
    for _, row in df_all.iterrows():
        key = (str(row["seq_from"]), str(row["seq_to"]))
        if key in b4_lookup:
            proxy = b4_lookup[key]
            comparison_data.append({
                "seq_from": row["seq_from"],
                "seq_to": row["seq_to"],
                "path_type": row["path_type"],
                "category": row.get("category", ""),
                "is_bio": row["is_bio"],
                "W2_npz": row["W2_npz"],
                "W2_proxy": proxy["W2_proxy"],
                "W2_diff": row["W2_npz"] - proxy["W2_proxy"],
                "action_npz": row["action_npz"],
                "action_proxy": proxy["action_proxy"],
                "source": "B4",
            })
            n_matched += 1
    print(f"  B4 匹配: {n_matched} 条路径")
else:
    print(f"  B4 代理结果不存在: {B4_PROXY}")

# Load B5 proxy results
if B5_PROXY.exists():
    df_b5 = pd.read_csv(B5_PROXY)
    print(f"  B5 代理结果: {len(df_b5)} 条路径")
    
    b5_lookup = {}
    for _, row in df_b5.iterrows():
        key = (str(row["seq_from"]), str(row["seq_to"]))
        b5_lookup[key] = {
            "W2_proxy": row.get("W2_distance", np.nan),
            "action_proxy": row.get("action_A", np.nan),
        }
    
    n_matched = 0
    for _, row in df_all.iterrows():
        key = (str(row["seq_from"]), str(row["seq_to"]))
        if key in b5_lookup and key not in {(d["seq_from"], d["seq_to"]) for d in comparison_data}:
            proxy = b5_lookup[key]
            comparison_data.append({
                "seq_from": row["seq_from"],
                "seq_to": row["seq_to"],
                "path_type": row["path_type"],
                "category": row.get("category", ""),
                "is_bio": row["is_bio"],
                "W2_npz": row["W2_npz"],
                "W2_proxy": proxy["W2_proxy"],
                "W2_diff": row["W2_npz"] - proxy["W2_proxy"],
                "action_npz": row["action_npz"],
                "action_proxy": proxy["action_proxy"],
                "source": "B5",
            })
            n_matched += 1
    print(f"  B5 匹配: {n_matched} 条路径")
else:
    print(f"  B5 代理结果不存在: {B5_PROXY}")

# Compute comparison statistics
if comparison_data:
    df_comparison = pd.DataFrame(comparison_data)
    
    # Correlation between NPZ direct and proxy
    w2_npz_vals = df_comparison["W2_npz"].dropna().values
    w2_proxy_vals = df_comparison["W2_proxy"].dropna().values
    
    if len(w2_npz_vals) > 2:
        r_spearman, p_spearman = spearmanr(w2_npz_vals, w2_proxy_vals)
        r_pearson, p_pearson = pearsonr(w2_npz_vals, w2_proxy_vals)
        print(f"\n  NPZ 直接 W_2 vs 代理 W_2 相关性:")
        print(f"    Spearman r = {r_spearman:.4f} (p = {p_spearman:.4e})")
        print(f"    Pearson r = {r_pearson:.4f} (p = {p_pearson:.4e})")
        print(f"    W_2 差异均值: {np.mean(w2_npz_vals - w2_proxy_vals):.4f}")
        print(f"    W_2 差异标准差: {np.std(w2_npz_vals - w2_proxy_vals):.4f}")
    
    # Third Law comparison
    bio_npz = df_comparison[df_comparison["is_bio"] == True]["action_npz"].dropna().values
    null_npz = df_comparison[df_comparison["is_bio"] == False]["action_npz"].dropna().values
    bio_proxy = df_comparison[df_comparison["is_bio"] == True]["action_proxy"].dropna().values
    null_proxy = df_comparison[df_comparison["is_bio"] == False]["action_proxy"].dropna().values
    
    if len(bio_npz) > 0 and len(null_npz) > 0 and len(bio_proxy) > 0 and len(null_proxy) > 0:
        u_npz, p_npz = mannwhitneyu(bio_npz, null_npz, alternative="less")
        u_proxy, p_proxy = mannwhitneyu(bio_proxy, null_proxy, alternative="less")
        
        d_npz = (np.mean(bio_npz) - np.mean(null_npz)) / (np.sqrt((np.var(bio_npz) + np.var(null_npz)) / 2) + 1e-8)
        d_proxy = (np.mean(bio_proxy) - np.mean(null_proxy)) / (np.sqrt((np.var(bio_proxy) + np.var(null_proxy)) / 2) + 1e-8)
        
        print(f"\n  第三定律对比:")
        print(f"                       NPZ 直接         代理")
        print(f"    Bio A[gamma]       {np.mean(bio_npz):.4f}           {np.mean(bio_proxy):.4f}")
        print(f"    Null A[gamma]      {np.mean(null_npz):.4f}           {np.mean(null_proxy):.4f}")
        print(f"    Mann-Whitney p     {p_npz:.4e}        {p_proxy:.4e}")
        print(f"    Cohen's d          {d_npz:.4f}           {d_proxy:.4f}")
        print(f"    一致性: {'一致' if (p_npz < 0.05) == (p_proxy < 0.05) else '不一致!'}")
else:
    df_comparison = pd.DataFrame()  # Empty dataframe

# ============================================================
# Step 7: Save results
# ============================================================
print("\n[7/7] 保存结果...")

# Save main CSV with all path results
# Merge comparison data into the main results
if not df_comparison.empty:
    # Add proxy columns to main results
    cmp_lookup = {}
    for _, row in df_comparison.iterrows():
        key = (str(row["seq_from"]), str(row["seq_to"]))
        cmp_lookup[key] = row
    
    w2_proxy_list = []
    action_proxy_list = []
    w2_diff_list = []
    proxy_source_list = []
    for _, row in df_all.iterrows():
        key = (str(row["seq_from"]), str(row["seq_to"]))
        if key in cmp_lookup:
            w2_proxy_list.append(cmp_lookup[key]["W2_proxy"])
            action_proxy_list.append(cmp_lookup[key]["action_proxy"])
            w2_diff_list.append(cmp_lookup[key]["W2_diff"])
            proxy_source_list.append(cmp_lookup[key]["source"])
        else:
            w2_proxy_list.append(np.nan)
            action_proxy_list.append(np.nan)
            w2_diff_list.append(np.nan)
            proxy_source_list.append("")
    
    df_all["W2_proxy"] = w2_proxy_list
    df_all["action_proxy"] = action_proxy_list
    df_all["W2_diff"] = w2_diff_list
    df_all["proxy_source"] = proxy_source_list

# Reorder columns
output_cols = [
    "path_type", "seq_from", "seq_to", "category", "aa_from", "aa_to",
    "n_from", "n_to", "delta_n", "is_bio",
    "W2_npz", "action_npz",
]
if "W2_proxy" in df_all.columns:
    output_cols += ["W2_proxy", "action_proxy", "W2_diff", "proxy_source"]

df_out = df_all[output_cols].copy()
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"  主结果: {OUTPUT_CSV}")
print(f"    行数: {len(df_out)}")

# Build summary JSON
summary = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "script": "phase_ensemble_npz_direct_w2.py",
    "description": "Direct NPZ-based Wasserstein-2 distance computation for Third Law validation",
    "data": {
        "total_sequences": len(all_seq_dirs),
        "loaded_sequences": loaded_count,
        "sequences_with_stats": compute_count,
        "skipped_sequences": skipped_count,
        "error_sequences": error_count,
        "aa_types": [str(a) for a in sorted(df_metadata["aa_type"].unique())],
        "n_range": [int(df_metadata["n"].min()), int(df_metadata["n"].max())],
        "pca_n_components": PCA_N_COMPONENTS,
        "regularization_eps": REGULARIZATION_EPS,
    },
    "paths": {
        "n_bio_g2x": len(bio_paths_g2x),
        "n_bio_n2n1": len(bio_paths_n2n1),
        "n_bio_total": len(bio_paths),
        "n_null": len(null_paths),
        "n_bio_computed": len(bio_results),
        "n_null_computed": len(null_results),
    },
    "third_law_test": {
        "mean_bio": float(np.mean(bio_w2)),
        "std_bio": float(np.std(bio_w2)),
        "mean_null": float(np.mean(null_w2)),
        "std_null": float(np.std(null_w2)),
        "mannwhitney_p": float(u_p),
        "ttest_p": float(t_p),
        "cohens_d": float(cohens_d),
        "z_score": float(z_score),
        "verified": bool(u_p < 0.05 and np.mean(bio_w2) < np.mean(null_w2)),
    },
    "category_analysis": [
        {
            "category": str(row["category"]),
            "n_paths": int(row["n_paths"]),
            "mean_bio": float(row["mean_bio"]),
            "std_bio": float(row["std_bio"]),
            "Z_score": float(row["Z_score"]),
            "MannWhitney_p": float(row["MannWhitney_p"]),
        }
        for _, row in df_category.iterrows()
    ],
    "comparison_with_proxy": {},
}

if comparison_data:
    df_cmp = pd.DataFrame(comparison_data)
    w2n = df_cmp["W2_npz"].dropna().values
    w2p = df_cmp["W2_proxy"].dropna().values
    if len(w2n) > 2:
        sr, sp = spearmanr(w2n, w2p)
        pr, pp = pearsonr(w2n, w2p)
        summary["comparison_with_proxy"] = {
            "n_matched": len(df_cmp),
            "spearman_r": float(sr),
            "spearman_p": float(sp),
            "pearson_r": float(pr),
            "pearson_p": float(pp),
            "mean_w2_npz": float(np.mean(w2n)),
            "mean_w2_proxy": float(np.mean(w2p)),
            "mean_diff": float(np.mean(w2n - w2p)),
            "std_diff": float(np.std(w2n - w2p)),
            "sources": df_cmp["source"].value_counts().to_dict(),
        }
        
        if len(bio_npz) > 0 and len(null_npz) > 0 and len(bio_proxy) > 0 and len(null_proxy) > 0:
            u_npz, p_npz = mannwhitneyu(bio_npz, null_npz, alternative="less")
            u_proxy, p_proxy = mannwhitneyu(bio_proxy, null_proxy, alternative="less")
            summary["comparison_with_proxy"]["third_law_consistency"] = {
                "npz_verified": bool(p_npz < 0.05 and np.mean(bio_npz) < np.mean(null_npz)),
                "proxy_verified": bool(p_proxy < 0.05 and np.mean(bio_proxy) < np.mean(null_proxy)),
                "consistent": bool((p_npz < 0.05) == (p_proxy < 0.05)),
                "npz_cohens_d": float(d_npz),
                "proxy_cohens_d": float(d_proxy),
            }

with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
print(f"  汇总 JSON: {OUTPUT_SUMMARY}")

# ============================================================
# Final Report
# ============================================================
print(f"\n{'=' * 70}")
print("B7 完成! 直接 NPZ  Wasserstein-2 距离计算")
print(f"{'=' * 70}")
print(f"  输出:")
print(f"    主 CSV:  {OUTPUT_CSV}")
print(f"    汇总 JSON: {OUTPUT_SUMMARY}")
print(f"\n  关键发现:")
print(f"    加载序列: {loaded_count} (stats: {compute_count})")
print(f"    生物路径: {len(bio_results)} 条 (W_2 均值: {np.mean([r['W2_npz'] for r in bio_results]):.4f})")
print(f"    零路径: {len(null_results)} 条 (W_2 均值: {np.mean([r['W2_npz'] for r in null_results]):.4f})")
print(f"    第三定律: {'VERIFIED' if u_p < 0.05 and np.mean(bio_w2) < np.mean(null_w2) else 'NOT verified'}")
print(f"    Cohen's d: {cohens_d:.4f}")
print(f"    Mann-Whitney p: {u_p:.4e}")
if comparison_data:
    print(f"    代理对比: {len(comparison_data)} 条匹配")
    print(f"      Spearman r: {summary['comparison_with_proxy'].get('spearman_r', 'N/A')}")
print(f"{'=' * 70}")