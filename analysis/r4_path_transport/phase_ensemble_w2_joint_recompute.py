#!/usr/bin/env python3
"""
W2 Joint-PCA 重算 (AUDIT-C3 修复)
====================================
将所有 936 条路径的 W₂ 从独立 SVD 基 (数学无效) 重算为联合 PCA 共同基 (数学有效)。

方法: compute_w2_joint() — 两系综堆叠做联合 SVD, 得到共同基底 Vt_joint,
     两系综分别投影到同一基底后计算 W₂ (满足 Wasserstein 内蕴度量性质)。

输入:
  - phase_ensemble_npz_direct_w2.csv (936 条路径定义)
  - test_workflow/polyx_ensemble/output/{seq_id}/batch_*.npz (Cα 坐标)
  - test_workflow/heteropolymer_ensemble/output/{seq_id}/batch_*.npz

输出:
  - field_theory/tables/phase_ensemble_w2_joint_paths.csv (全部 936 条路径 joint W₂)
  - field_theory/tables/phase_ensemble_w2_joint_summary.json (统计汇总 + 与旧值对比)
"""

import sys, os, json, warnings, gc, time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm

warnings.filterwarnings("ignore")

# ============================================================
# Paths
# ============================================================
FIELD_THEORY = Path(__file__).parent.parent
TABLES_DIR = FIELD_THEORY / "tables"
PROJECT_ROOT = FIELD_THEORY.parent

NPZ_DIRS = [
    PROJECT_ROOT / "test_workflow/polyx_ensemble/output",
    PROJECT_ROOT / "test_workflow/heteropolymer_ensemble/output",
]

INPUT_CSV = TABLES_DIR / "phase_ensemble_npz_direct_w2.csv"
OUTPUT_CSV = TABLES_DIR / "phase_ensemble_w2_joint_paths.csv"
OUTPUT_JSON = TABLES_DIR / "phase_ensemble_w2_joint_summary.json"

# ============================================================
# Configuration (与原脚本一致)
# ============================================================
NPZ_KEY = "pos"
PCA_N_COMPONENTS = 50
REGULARIZATION_EPS = 1e-6

print("=" * 70)
print("W2 Joint-PCA 重算 (AUDIT-C3 修复)")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ============================================================
# Core Functions (从原脚本复制, 确保一致性)
# ============================================================

def load_sequence_npz(seq_id):
    """Load all NPZ batch files for a sequence from either output dir."""
    for base_dir in NPZ_DIRS:
        seq_dir = base_dir / seq_id
        if not seq_dir.is_dir():
            continue
        npz_files = sorted([
            f for f in seq_dir.glob("batch_*.npz")
            if not f.name.startswith(".samples")
        ])
        if not npz_files:
            continue
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
        if all_pos:
            return np.concatenate(all_pos, axis=0)
    return None


def robust_sqrtm(A, eps=REGULARIZATION_EPS):
    """Compute matrix square root with regularization fallback."""
    from scipy.linalg import eigh
    for attempt in range(3):
        try:
            A_reg = A + eps * np.eye(A.shape[0])
            eigvals, eigvecs = eigh(A_reg)
            eigvals = np.maximum(eigvals, 0)
            return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        except Exception:
            eps *= 10
    return np.eye(A.shape[0]) * np.sqrt(eps)


def compute_wasserstein2(mu1, Sigma1, mu2, Sigma2, eps=REGULARIZATION_EPS):
    """W_2^2 = ||mu1-mu2||^2 + Tr(Sigma1 + Sigma2 - 2*(Sigma1^{1/2} Sigma2 Sigma1^{1/2})^{1/2})"""
    mean_term = np.sum((mu1 - mu2) ** 2)
    sqrt_Sigma1 = robust_sqrtm(Sigma1, eps)
    middle = sqrt_Sigma1 @ Sigma2 @ sqrt_Sigma1
    sqrt_middle = robust_sqrtm(middle, eps)
    cov_term = np.trace(Sigma1 + Sigma2 - 2 * sqrt_middle)
    w2_sq = mean_term + cov_term
    return np.sqrt(max(w2_sq, 0))


def compute_w2_joint(pos_P, pos_Q, n_components=PCA_N_COMPONENTS):
    """
    数学正确的 W₂ 计算: 两条序列投影到联合 PCA 共同基底后再计算 W₂。

    修复审计发现 C3: 原实现对每条序列独立 SVD, 两系综 μ/Σ 处于不同正交基底,
    私有基底间的 W₂ 比较不具备 Wasserstein 距离的内蕴度量性质。

    方法:
      1. 长度对齐: 取前 min(n_res_P, n_res_Q) 个残基
      2. 堆叠两系综样本做联合 SVD, 得到共同基底 Vt_joint
      3. 两系综分别投影到同一 Vt_joint, 再计算各自 μ/Σ
      4. 在共同空间内计算 W₂
    """
    n_res = min(pos_P.shape[1], pos_Q.shape[1])
    if n_res < 2:
        return None
    Xp = pos_P[:, :n_res, :].reshape(pos_P.shape[0], -1)
    Xq = pos_Q[:, :n_res, :].reshape(pos_Q.shape[0], -1)

    dim = Xp.shape[1]
    Xall = np.vstack([Xp, Xq])
    mean_all = np.mean(Xall, axis=0)
    Xall_c = Xall - mean_all

    if dim <= n_components:
        Xp_c = Xp - mean_all
        Xq_c = Xq - mean_all
        mu_P = np.mean(Xp_c, axis=0)
        mu_Q = np.mean(Xq_c, axis=0)
        Sigma_P = np.cov(Xp_c, rowvar=False) + REGULARIZATION_EPS * np.eye(dim)
        Sigma_Q = np.cov(Xq_c, rowvar=False) + REGULARIZATION_EPS * np.eye(dim)
        return compute_wasserstein2(mu_P, Sigma_P, mu_Q, Sigma_Q)

    U, s, Vt = np.linalg.svd(Xall_c, full_matrices=False)
    Vt_joint = Vt[:n_components]

    Xp_proj = (Xp - mean_all) @ Vt_joint.T
    Xq_proj = (Xq - mean_all) @ Vt_joint.T

    mu_P = np.mean(Xp_proj, axis=0)
    mu_Q = np.mean(Xq_proj, axis=0)
    Sigma_P = np.cov(Xp_proj, rowvar=False) + REGULARIZATION_EPS * np.eye(n_components)
    Sigma_Q = np.cov(Xq_proj, rowvar=False) + REGULARIZATION_EPS * np.eye(n_components)

    return compute_wasserstein2(mu_P, Sigma_P, mu_Q, Sigma_Q)


# ============================================================
# Step 1: Load path definitions
# ============================================================
print("\n[1/5] 加载路径定义...")
df = pd.read_csv(INPUT_CSV)
print(f"  总路径: {len(df)}")
print(f"  chain_growth: {len(df[df['category']=='chain_growth'])}")
print(f"  aa_change: {len(df[df['category']=='aa_change'])}")

# ============================================================
# Step 2: Collect unique sequences and load NPZ data
# ============================================================
print("\n[2/5] 收集并加载 NPZ 数据...")
all_seq_ids = set(df["seq_from"].unique()) | set(df["seq_to"].unique())
print(f"  唯一序列: {len(all_seq_ids)}")

seq_data = {}
n_loaded = 0
n_failed = 0

for sid in sorted(all_seq_ids):
    pos = load_sequence_npz(sid)
    if pos is not None and pos.shape[0] >= 3:
        seq_data[sid] = pos
        n_loaded += 1
    else:
        n_failed += 1
    if n_loaded % 100 == 0:
        print(f"    Loaded {n_loaded}/{len(all_seq_ids)}...")

print(f"  加载成功: {n_loaded}, 失败/跳过: {n_failed}")

# ============================================================
# Step 3: Compute joint W2 for all paths
# ============================================================
print("\n[3/5] 计算 joint-PCA W₂ (936 条路径)...")

w2_joint_vals = []
n_computed = 0
n_skipped = 0
t0 = time.time()

for idx, row in df.iterrows():
    s1, s2 = row["seq_from"], row["seq_to"]

    if s1 not in seq_data or s2 not in seq_data:
        w2_joint_vals.append(np.nan)
        n_skipped += 1
        continue

    try:
        w2 = compute_w2_joint(seq_data[s1], seq_data[s2])
        w2_joint_vals.append(float(w2) if w2 is not None else np.nan)
        n_computed += 1
    except Exception as e:
        w2_joint_vals.append(np.nan)
        n_skipped += 1
        if n_skipped <= 5:
            print(f"    Warning: Failed joint W₂ for {s1} -> {s2}: {e}")

    if (idx + 1) % 100 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (idx + 1) * (len(df) - idx - 1)
        print(f"    {idx+1}/{len(df)} computed ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

t1 = time.time()
print(f"  计算完成: {n_computed} 成功, {n_skipped} 跳过, 总耗时 {t1-t0:.1f}s")

# ============================================================
# Step 4: Save results
# ============================================================
print("\n[4/5] 保存结果...")

df["W2_joint"] = w2_joint_vals

# Compute joint action (same formula as original)
def compute_action(row):
    w2 = row["W2_joint"]
    if np.isnan(w2):
        return np.nan
    action = w2
    if str(row.get("aa_from", "?")) != str(row.get("aa_to", "?")):
        action += 0.1
    delta_n = abs(row.get("n_to", row.get("n", 0)) - row.get("n_from", row.get("n", 0)))
    if delta_n > 0:
        action += 0.05 * np.log(delta_n + 1)
    return action

df["action_joint"] = df.apply(compute_action, axis=1)

# Reorder columns: put W2_joint and action_joint next to original W2_npz
cols = list(df.columns)
# Move W2_joint and action_joint to be after W2_npz and action_npz
for c in ["W2_joint", "action_joint"]:
    cols.remove(c)
insert_idx = cols.index("action_npz") + 1
for i, c in enumerate(["W2_joint", "action_joint"]):
    cols.insert(insert_idx + i, c)
df = df[cols]

df.to_csv(OUTPUT_CSV, index=False)
print(f"  路径数据已保存: {OUTPUT_CSV}")

# ============================================================
# Step 5: Summary statistics and comparison with old values
# ============================================================
print("\n[5/5] 统计汇总...")

summary = {
    "timestamp": datetime.now().isoformat(),
    "script": "phase_ensemble_w2_joint_recompute.py",
    "description": "AUDIT-C3 fix: W2 recomputed with joint PCA basis",
    "n_paths_total": len(df),
    "n_computed": n_computed,
    "n_skipped": n_skipped,
    "computation_time_seconds": round(t1 - t0, 1),
    "categories": {},
    "old_vs_new": {},
}

# Per-category stats
for cat in ["chain_growth", "aa_change"]:
    cat_df = df[df["category"] == cat].dropna(subset=["W2_joint"])
    if len(cat_df) == 0:
        continue
    w2j = cat_df["W2_joint"].values
    w2o = cat_df["W2_npz"].values
    aj = cat_df["action_joint"].values
    ao = cat_df["action_npz"].values

    from scipy.stats import spearmanr, pearsonr
    r_s, p_s = spearmanr(w2o, w2j)
    r_p, p_p = pearsonr(w2o, w2j)

    summary["categories"][cat] = {
        "n": len(cat_df),
        "W2_joint_mean": float(np.mean(w2j)),
        "W2_joint_std": float(np.std(w2j)),
        "W2_old_mean": float(np.mean(w2o)),
        "W2_old_std": float(np.std(w2o)),
        "action_joint_mean": float(np.mean(aj)),
        "action_old_mean": float(np.mean(ao)),
        "W2_old_vs_new_spearman": float(r_s),
        "W2_old_vs_new_pearson": float(r_p),
        "W2_old_vs_new_spearman_p": float(p_s),
    }

    print(f"\n  {cat}:")
    print(f"    n = {len(cat_df)}")
    print(f"    W2_joint: {np.mean(w2j):.4f} +/- {np.std(w2j):.4f}")
    print(f"    W2_old:   {np.mean(w2o):.4f} +/- {np.std(w2o):.4f}")
    print(f"    action_joint: {np.mean(aj):.4f} +/- {np.std(aj):.4f}")
    print(f"    action_old:   {np.mean(ao):.4f} +/- {np.std(ao):.4f}")
    print(f"    old vs new Spearman: r={r_s:.4f} (p={p_s:.2e})")

# Key comparison: chain_growth vs aa_change (the core Law 3 test)
cg = df[(df["category"] == "chain_growth")].dropna(subset=["action_joint"])
ac = df[(df["category"] == "aa_change")].dropna(subset=["action_joint"])

if len(cg) > 0 and len(ac) > 0:
    from scipy.stats import mannwhitneyu
    cg_a = cg["action_joint"].values
    ac_a = ac["action_joint"].values
    u_stat, u_p = mannwhitneyu(cg_a, ac_a, alternative="less")
    pooled_std = np.sqrt((np.var(cg_a) + np.var(ac_a)) / 2)
    cohens_d = (np.mean(cg_a) - np.mean(ac_a)) / (pooled_std + 1e-8)

    summary["law3_core_test"] = {
        "chain_growth_action_mean": float(np.mean(cg_a)),
        "aa_change_action_mean": float(np.mean(ac_a)),
        "mean_diff": float(np.mean(cg_a) - np.mean(ac_a)),
        "mannwhitney_p": float(u_p),
        "cohens_d": float(cohens_d),
        "direction": "bio < null" if np.mean(cg_a) < np.mean(ac_a) else "bio >= null",
    }

    print(f"\n  Law 3 核心检验 (joint W₂):")
    print(f"    chain_growth action: {np.mean(cg_a):.4f}")
    print(f"    aa_change action:    {np.mean(ac_a):.4f}")
    print(f"    Mann-Whitney p = {u_p:.2e}")
    print(f"    Cohen's d = {cohens_d:.4f}")
    print(f"    方向: {'bio < null ✅' if np.mean(cg_a) < np.mean(ac_a) else 'bio >= null ❌'}")

with open(OUTPUT_JSON, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\n  汇总已保存: {OUTPUT_JSON}")

print("\n" + "=" * 70)
print("W2 Joint-PCA 重算完成")
print("=" * 70)
