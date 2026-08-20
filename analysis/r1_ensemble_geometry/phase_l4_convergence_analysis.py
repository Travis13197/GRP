#!/usr/bin/env python3
"""
L4 后分析: K/E 收敛验证 + spectral_decay N_req 精化 + FSS
==============================================================
用 L4 新采样的 K/E (n=10,50) 数据验证 spectral_decay 收敛，
精化 N_req 估计，并分析 FSS (n=150,200) 结果。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path("/mnt/b/2026/Exploration/ProtGenesis2_Ensemble")
L4_DIR = PROJECT_ROOT / "test_workflow" / "phase_l4_extended" / "output"
L4_SUPP_DIR = PROJECT_ROOT / "test_workflow" / "phase_l4_extended" / "output_supplement"
D6_JSON = PROJECT_ROOT / "field_theory" / "tables" / "phase_d6_convergence_results.json"
B_JSON = PROJECT_ROOT / "field_theory" / "tables" / "phase_b_spectral_decay_convergence.json"
OUTPUT_DIR = PROJECT_ROOT / "field_theory" / "tables"
RESULT_PATH = OUTPUT_DIR / "phase_l4_convergence_analysis.json"


def load_ensemble(npz_dir, supp_dir=None):
    """加载主目录 (+ 可选 supplement 目录合并) 的全部构象"""
    all_pos = []
    n_files = 0
    for d in [npz_dir, supp_dir]:
        if d is None or not d.exists():
            continue
        for f in sorted(d.glob("batch_*.npz")):
            data = np.load(f)
            pos = data["pos"] if "pos" in data else data["positions"]
            all_pos.append(pos)
            n_files += 1
    if not all_pos:
        return None, 0
    return np.concatenate(all_pos, axis=0), n_files


def compute_geometry(coords):
    n_samples, n_residues, _ = coords.shape
    X = coords.reshape(n_samples, n_residues * 3)
    mean = X.mean(axis=0)
    centered = X - mean
    cov = np.cov(centered, rowvar=False)
    eigs = np.linalg.eigvalsh(cov)[::-1]
    eigs = np.maximum(eigs, 0)
    total = eigs.sum()
    if total <= 0:
        return {}
    normed = eigs / total
    PR = float((eigs.sum() ** 2) / (eigs ** 2).sum())
    A_C = float(normed[0])
    cumsum = np.cumsum(normed)
    eff_rank_95 = int(np.searchsorted(cumsum, 0.95) + 1)
    k = np.arange(1, min(51, len(eigs) + 1))
    log_k = np.log(k)
    log_eig = np.log(eigs[:len(k)] + 1e-10)
    A = np.vstack([log_k, np.ones(len(k))]).T
    alpha, _ = np.linalg.lstsq(A, log_eig, rcond=None)[0]
    spectral_decay = float(-alpha)
    entropy = float(-np.sum(normed * np.log(normed + 1e-10)))
    return {
        "PR": PR, "A_C": A_C, "spectral_decay": spectral_decay,
        "eff_rank_95": eff_rank_95, "entropy": entropy,
        "n_samples": n_samples, "n_residues": n_residues,
    }


def bootstrap_convergence(coords, n_bootstrap=50, seed=42):
    """Bootstrap 检验 spectral_decay 收敛性"""
    rng = np.random.default_rng(seed)
    n_total = coords.shape[0]
    if n_total < 200:
        return {}
    subset_sizes = [100, 250, 500, 1000, 1500, 2000, 4000]
    subset_sizes = [s for s in subset_sizes if s <= n_total]
    results = {}
    for n_sub in subset_sizes:
        sds = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n_total, n_sub, replace=False)
            geo = compute_geometry(coords[idx])
            if geo:
                sds.append(geo["spectral_decay"])
        if sds:
            results[n_sub] = {
                "mean": float(np.mean(sds)),
                "std": float(np.std(sds)),
                "cv": float(np.std(sds) / (abs(np.mean(sds)) + 1e-10)),
                "n_bootstrap": len(sds),
            }
    return results


def main():
    result = {"timestamp": pd.Timestamp.now().isoformat()}

    print("=" * 60)
    print("L4 后分析: 收敛验证 + spectral_decay N_req 精化")
    print("=" * 60)

    # === Part 1: K/E 新序列几何特征 ===
    sequences = {
        "PolyX_K_n10": ("K", 10),
        "PolyX_K_n50": ("K", 50),
        "PolyX_E_n10": ("E", 10),
        "PolyX_E_n50": ("E", 50),
        "PolyX_G_n150": ("G", 150),
        "PolyX_G_n200": ("G", 200),
        "PolyX_K_n150": ("K", 150),
        "PolyX_K_n200": ("K", 200),
    }

    all_geometry = {}
    for seq_id, (aa, n) in sequences.items():
        seq_dir = L4_DIR / seq_id
        if not seq_dir.exists():
            print(f"\n[SKIP] {seq_id}: 目录不存在")
            continue
        coords, n_batches = load_ensemble(seq_dir, L4_SUPP_DIR / seq_id)
        if coords is None:
            print(f"\n[SKIP] {seq_id}: 无数据")
            continue
        geo = compute_geometry(coords)
        all_geometry[seq_id] = geo
        print(f"\n{seq_id} ({aa}{n}): {geo['n_samples']} samples, {n_batches} batches")
        print(f"  spectral_decay={geo['spectral_decay']:.4f}, PR={geo['PR']:.4f}, "
              f"eff_rank_95={geo['eff_rank_95']}, entropy={geo['entropy']:.4f}")

    result["geometry"] = all_geometry

    # === Part 2: K/E 收敛性 Bootstrap ===
    print("\n" + "=" * 60)
    print("K/E spectral_decay Bootstrap 收敛性检验")
    print("=" * 60)

    ke_convergence = {}
    for seq_id in ["PolyX_K_n10", "PolyX_K_n50", "PolyX_E_n10", "PolyX_E_n50",
                   "PolyX_G_n150", "PolyX_G_n200", "PolyX_K_n150", "PolyX_K_n200"]:
        if seq_id not in all_geometry:
            continue
        seq_dir = L4_DIR / seq_id
        coords, _ = load_ensemble(seq_dir, L4_SUPP_DIR / seq_id)
        if coords is None:
            continue
        conv = bootstrap_convergence(coords, n_bootstrap=30)
        ke_convergence[seq_id] = conv
        print(f"\n{seq_id}:")
        for n_sub, stats_d in sorted(conv.items()):
            cv = stats_d["cv"]
            status = "CONVERGED" if cv < 0.05 else ("MARGINAL" if cv < 0.10 else "NOT_CONVERGED")
            print(f"  N={n_sub}: sd={stats_d['mean']:.4f} ± {stats_d['std']:.4f}, CV={cv:.4f} {status}")

    result["convergence"] = ke_convergence

    # === Part 3: spectral_decay 标度律 (与 D6 n=25 对比) ===
    print("\n" + "=" * 60)
    print("spectral_decay 标度律 (L4 新数据)")
    print("=" * 60)

    scaling_data = []
    for seq_id, geo in all_geometry.items():
        aa = seq_id.split("_")[1]
        n = int(seq_id.split("_n")[1])
        scaling_data.append({"seq_id": seq_id, "aa": aa, "n": n,
                            "spectral_decay": geo["spectral_decay"],
                            "n_samples": geo["n_samples"]})

    if scaling_data:
        sdf = pd.DataFrame(scaling_data)
        for aa in sdf["aa"].unique():
            sub = sdf[sdf["aa"] == aa].sort_values("n")
            if len(sub) >= 2:
                log_n = np.log(sub["n"].values)
                log_sd = np.log(sub["spectral_decay"].values)
                slope, intercept, r, p, se = stats.linregress(log_n, log_sd)
                print(f"\n{aa}: spectral_decay ~ n^{slope:.4f}, R²={r**2:.4f}, p={p:.4e}")
                print(f"  Data: {list(zip(sub['n'].values, sub['spectral_decay'].values))}")
                result.setdefault("scaling", {})[aa] = {
                    "beta": float(slope), "r_squared": float(r**2),
                    "p_value": float(p), "n_points": len(sub),
                }

    # === Part 4: FSS 结果 (n=150,200) ===
    print("\n" + "=" * 60)
    print("FSS (n=150,200) 结果")
    print("=" * 60)

    fss_data = {k: v for k, v in all_geometry.items() if "n150" in k or "n200" in k}
    if fss_data:
        for seq_id, geo in fss_data.items():
            print(f"\n{seq_id}: PR={geo['PR']:.4f}, spectral_decay={geo['spectral_decay']:.4f}, "
                  f"eff_rank_95={geo['eff_rank_95']}")
    else:
        print("FSS 序列尚未完成采样")

    result["fss"] = fss_data

    # === Part 5: N_req 精化 ===
    print("\n" + "=" * 60)
    print("N_req 精化 (K/E)")
    print("=" * 60)

    # 从 D6 (n=25) 获取 N_req 参考
    if D6_JSON.exists():
        with open(D6_JSON) as f:
            d6 = json.load(f)
        print("D6 (n=25) N_req:")
        # D6 结构可能不同，尝试提取
        if isinstance(d6, dict):
            for key in d6:
                if "K" in key or "E" in key:
                    print(f"  {key}: {type(d6[key])}")
    else:
        print("D6 JSON 不存在")

    result["n_req_summary"] = {
        "K_n10": "待 bootstrap 结果确定",
        "K_n50": "待 bootstrap 结果确定",
        "E_n10": "待 bootstrap 结果确定",
        "E_n50": "待 bootstrap 结果确定",
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n结果保存: {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())