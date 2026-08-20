#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase O1: C_geo v3 确定性版 + 种子稳健性 + 偏相关分解 (2026-08-03)
====================================================================
评审发现 S1 的修正实现:
  v2 (旧): C_geo = mag^2 * (d_hat^T g_S d_hat), d_hat ~ seed=42 随机方向 (单次抽签)
  v3 (新): C_geo = mag^2 * tr(g_S)/3          — 随机方向的解析期望, 零 MC 噪声

流程:
  A. 逐蛋白重算 Kabsch+LW 内禀协方差 (复用 phase_l1_kabsch_metric, 与 Phase L1 几何一致)
  B. 复现门: seed=42 随机方向版必须与存储 C_geo_kabsch_lw 逐位一致 (rho>0.999)
  C. v3 确定性版计算 + 每蛋白 Spearman rho + n_eff=8 cluster t
  D. 100 seeds 稳健性: 随机方向版 mean rho 分布
  E. 分解: rho(mag), rho(S), rho(v3), partial Spearman rho(v3, DMS | mag) (秩残差)
  F. 输出增强逐变体表 (供 O2 基线竞赛复用) + 每蛋白均值结构/刚度缓存

判定标准 (预注册, PhaseO_correction_plan_20260803.md §1):
  v3 vs v2: per-protein 符号全同 且 mean |Δrho| < 0.02
  种子稳健:  mean rho 的 seed-SD < 0.01 且 100/100 seeds 负号
  独立贡献:  partial rho(v3, DMS | mag) 在 >=6/8 蛋白显著负
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, ttest_1samp, rankdata

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "field_theory" / "scripts"))
from phase_l1_kabsch_metric import compute_intrinsic_metric, kabsch_align_ensemble  # noqa: E402

L1_CSV = PROJECT_ROOT / "field_theory" / "data" / "phase_l1" / "phase_l1_cgeo_kabsch_vs_phase9_cgeo_real.csv"
DMS_BIOEMU = PROJECT_ROOT / "field_theory" / "data" / "dms" / "results" / "bioemu"
OUT_DIR = PROJECT_ROOT / "field_theory" / "data" / "phase_o"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES = PROJECT_ROOT / "field_theory" / "tables"
FIG_DIR = PROJECT_ROOT / "field_theory" / "figures" / "phase_o"
FIG_DIR.mkdir(parents=True, exist_ok=True)

REG_EPS = 0.01
N_SEEDS = 100
N_SAMPLES = 250  # 与 test_l1_real_data.py 一致 (250 样本复现门已验证: rho=1.0, max_rel=3.3e-14)

# 与 cgeo.py / test_l1_real_data.py 完全一致的查找表与幅度公式
AA_VOLUMES = {"A": 88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9, "G": 60.1,
              "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7, "M": 162.9, "N": 114.1,
              "P": 112.7, "Q": 143.8, "R": 173.4, "S": 89.0, "T": 116.1, "V": 140.0,
              "W": 227.8, "Y": 193.6}
AA_CHARGES = {"R": 1, "K": 1, "D": -1, "E": -1, "H": 0.1}

PROTEIN_DIRS = {  # 候选目录 (按优先级), 自动选择能复现存储值者
    "BLAT": ["blat_wt"], "GFP": ["gfp_wt"], "HRAS": ["hras_wt"],
    "HSP90": ["hsp90_wt"], "P53": ["p53_wt"], "PTEN": ["pten_wt"],
    "SPIKE": ["spike_rbd", "spike_wt"], "UBE4B": ["ube4b_wt"],
}


def load_ensemble(dir_name):
    """加载 batch_*.npz 并拼接 -> (n_samples, n_res, 3)"""
    d = DMS_BIOEMU / dir_name
    if not d.exists():
        return None
    arrs = []
    for f in sorted(d.glob("batch_*.npz")):
        try:
            z = np.load(f)
            key = "pos" if "pos" in z else list(z.keys())[0]
            arrs.append(z[key])
        except Exception:
            continue
    return np.concatenate(arrs, axis=0) if arrs else None


def residue_metrics(cov, n_res):
    """从全协方差提取每残基 3x3 块 -> C_i, g_i=(C_i+eps I)^-1, S_i=tr(g_i)/3, rmsf_i"""
    C = np.zeros((n_res, 3, 3))
    g = np.zeros((n_res, 3, 3))
    for i in range(n_res):
        cb = cov[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3]
        C[i] = cb
        eps = REG_EPS * np.trace(cb) / 3.0
        g[i] = np.linalg.inv(cb + eps * np.eye(3))
    S = np.trace(g, axis1=1, axis2=2) / 3.0          # 刚度 (确定性方向期望)
    rmsf = np.sqrt(np.trace(C, axis1=1, axis2=2))    # 位置涨落
    return C, g, S, rmsf


def map_positions(positions, n_res):
    """位置->0-based 索引自动映射; 返回 (idx, offset_scheme)"""
    pos = positions.astype(int)
    idx = pos - 1
    if idx.max() < n_res and idx.min() >= 0:
        return idx, "pos-1"
    offset = pos.min()
    idx2 = pos - offset
    if idx2.max() < n_res and idx2.min() >= 0:
        return idx2, f"pos-{offset}"
    return None, "FAILED"


def mags_from_table(df):
    dv = np.abs(df["aa_volume_diff"].values.astype(float))
    dq = np.abs(df["aa_charge_diff"].values.astype(float))
    return 0.1 + dv / 100.0 + dq * 0.05


def partial_spearman(x, y, z):
    """偏 Spearman: 秩变换后对 z 的秩残差取 Pearson"""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.vstack([rz, np.ones_like(rz)]).T
    bx = np.linalg.lstsq(A, rx, rcond=None)[0]
    by = np.linalg.lstsq(A, ry, rcond=None)[0]
    ex, ey = rx - A @ bx, ry - A @ by
    r, p = pearsonr(ex, ey)
    return r, p


def main():
    t_start = time.time()
    df = pd.read_csv(L1_CSV)
    # UBE4B 勘误过滤 (899 零占位行, 精确匹配 position>69)
    n0 = len(df)
    df = df[~((df["protein"] == "UBE4B") & (df["position"] > 69))].reset_index(drop=True)
    print(f"[勘误过滤] {n0} -> {len(df)} (剔除 {n0 - len(df)} 行 UBE4B 零占位)")

    results = {}
    seed_mean_rhos = np.zeros(N_SEEDS)
    seed_per_protein = {p: [] for p in PROTEIN_DIRS}
    augmented = []

    for prot, dirs in PROTEIN_DIRS.items():
        sub = df[df["protein"] == prot]
        if len(sub) == 0:
            continue
        # --- 选择能复现存储值的系综目录 ---
        chosen = None
        for dn in dirs:
            pos = load_ensemble(dn)
            if pos is None:
                continue
            pos = pos[:N_SAMPLES]  # 与 Phase L1 管线一致: 前 250 样本
            res = compute_intrinsic_metric(pos, align_reference="first", shrinkage="ledoit_wolf")
            n_res = pos.shape[1]
            idx, scheme = map_positions(sub["position"].values, n_res)
            if idx is None:
                continue
            C, g, S, rmsf = residue_metrics(res["cov"], n_res)
            mag = mags_from_table(sub)
            # 复现门: seed=42 随机方向 (legacy RandomState, 与 test_l1_real_data.py cgeo_from_cov 逐位一致)
            np.random.seed(42)
            direc = np.random.randn(len(sub), 3)
            direc /= np.linalg.norm(direc, axis=1, keepdims=True) + 1e-10
            repro = mag ** 2 * np.einsum("bi,bij,bj->b", direc, g[idx], direc)
            r_rep = spearmanr(repro, sub["C_geo_kabsch_lw"].values)[0]
            max_rel = float(np.max(np.abs(repro - sub["C_geo_kabsch_lw"].values) /
                                   (np.abs(sub["C_geo_kabsch_lw"].values) + 1e-30)))
            if r_rep > 0.9995 and max_rel < 1e-6:
                chosen = dict(dir=dn, pos=pos, res=res, n_res=n_res, idx=idx,
                              scheme=scheme, C=C, g=g, S=S, rmsf=rmsf, mag=mag,
                              repro_rho=float(r_rep))
                break
            print(f"  [{prot}] {dn}: 复现 rho={r_rep:.4f} 不足, 试下一候选")
        if chosen is None:
            print(f"  [{prot}] !! 无法复现存储值, 跳过")
            continue

        idx, g, S, mag = chosen["idx"], chosen["g"], chosen["S"], chosen["mag"]
        y = sub["DMS_score"].values.astype(float)

        # --- v3 确定性版 ---
        S_var = S[idx]
        cgeo_v3 = mag ** 2 * S_var

        rho_v2, p_v2 = spearmanr(sub["C_geo_kabsch_lw"].values, y)
        rho_v3, p_v3 = spearmanr(cgeo_v3, y)
        rho_mag, p_mag = spearmanr(mag, y)
        rho_S, p_S = spearmanr(S_var, y)
        rho_rmsf, p_rmsf = spearmanr(-chosen["rmsf"][idx], y)  # 负号: 涨落大=耐受
        pr_v3_mag, pp_v3_mag = partial_spearman(cgeo_v3, y, mag)
        pr_S_mag, pp_S_mag = partial_spearman(S_var, y, mag)

        # --- 100 seeds 随机方向版 ---
        for si, seed in enumerate(range(1, N_SEEDS + 1)):
            rng = np.random.default_rng(seed)
            direc = rng.standard_normal((len(sub), 3))
            direc /= np.linalg.norm(direc, axis=1, keepdims=True) + 1e-10
            cgeo_s = mag ** 2 * np.einsum("bi,bij,bj->b", direc, g[idx], direc)
            r_s = spearmanr(cgeo_s, y)[0]
            seed_per_protein[prot].append(float(r_s))
            seed_mean_rhos[si] += r_s / 8.0

        results[prot] = dict(
            n_variants=int(len(sub)), n_res=int(chosen["n_res"]),
            ensemble_dir=chosen["dir"], pos_scheme=chosen["scheme"],
            repro_rho=chosen["repro_rho"], lambda_star=float(chosen["res"]["lambda_star"]),
            rho_v2=float(rho_v2), p_v2=float(p_v2),
            rho_v3=float(rho_v3), p_v3=float(p_v3),
            rho_mag=float(rho_mag), p_mag=float(p_mag),
            rho_S=float(rho_S), p_S=float(p_S),
            rho_rmsf=float(rho_rmsf), p_rmsf=float(p_rmsf),
            partial_v3_given_mag=dict(r=float(pr_v3_mag), p=float(pp_v3_mag)),
            partial_S_given_mag=dict(r=float(pr_S_mag), p=float(pp_S_mag)),
            seed_rho_mean=float(np.mean(seed_per_protein[prot])),
            seed_rho_sd=float(np.std(seed_per_protein[prot])),
        )
        print(f"  [{prot}] n={len(sub)} ({chosen['dir']}, {chosen['scheme']}, "
              f"复现rho={chosen['repro_rho']:.5f}, λ*={chosen['res']['lambda_star']:.3f}) "
              f"v2={rho_v2:+.4f} v3={rho_v3:+.4f} | mag={rho_mag:+.4f} S={rho_S:+.4f} "
              f"partial(v3|mag)={pr_v3_mag:+.4f} (p={pp_v3_mag:.1e})")

        sub2 = sub.copy()
        sub2["mag"] = mag
        sub2["S_stiffness"] = S_var
        sub2["rmsf_site"] = chosen["rmsf"][idx]
        sub2["C_geo_v3"] = cgeo_v3
        augmented.append(sub2)
        mean_ca = kabsch_align_ensemble(chosen["pos"], reference_mode="first").mean(axis=0)
        np.save(OUT_DIR / f"{prot.lower()}_mean_ca.npy", mean_ca)
        np.save(OUT_DIR / f"{prot.lower()}_stiffness.npy", S)
        np.save(OUT_DIR / f"{prot.lower()}_rmsf.npy", chosen["rmsf"])

    # ============ 汇总 (n_eff=8 蛋白级 cluster 推断) ============
    prots = list(results.keys())
    v2s = np.array([results[p]["rho_v2"] for p in prots])
    v3s = np.array([results[p]["rho_v3"] for p in prots])
    mags_r = np.array([results[p]["rho_mag"] for p in prots])
    Ss = np.array([results[p]["rho_S"] for p in prots])
    parts = np.array([results[p]["partial_v3_given_mag"]["r"] for p in prots])
    part_ps = np.array([results[p]["partial_v3_given_mag"]["p"] for p in prots])

    t_v3, p_t_v3 = ttest_1samp(v3s, 0)
    summary = dict(
        n_variants=int(len(df)), n_proteins=len(prots),
        mean_rho_v2=float(v2s.mean()), mean_rho_v3=float(v3s.mean()),
        mean_abs_delta=float(np.abs(v3s - v2s).mean()),
        sign_agreement=bool(np.all(np.sign(v3s) == np.sign(v2s))),
        v3_neff8_t=float(t_v3), v3_neff8_p=float(p_t_v3),
        enhancement_vs_lab_baseline=None,  # 实验室系基线在勘误后 = -0.0953
        seed_robustness=dict(
            mean_rho_across_seeds=float(seed_mean_rhos.mean()),
            sd_across_seeds=float(seed_mean_rhos.std()),
            min=float(seed_mean_rhos.min()), max=float(seed_mean_rhos.max()),
            all_negative=bool(np.all(seed_mean_rhos < 0))),
        decomposition=dict(
            mean_rho_mag=float(mags_r.mean()), mean_rho_S=float(Ss.mean()),
            mean_partial_v3_given_mag=float(parts.mean()),
            n_partial_sig_neg=int(np.sum((part_ps < 0.05) & (parts < 0)))),
        verdicts=dict(
            v3_vs_v2_consistency=bool(np.all(np.sign(v3s) == np.sign(v2s)) and np.abs(v3s - v2s).mean() < 0.02),
            seed_robust=bool(seed_mean_rhos.std() < 0.01 and np.all(seed_mean_rhos < 0)),
            independent_geometric_contribution=bool(np.sum((part_ps < 0.05) & (parts < 0)) >= 6),
        ),
    )
    print("\n========== O1 汇总 ==========")
    print(f"mean ρ: v2={summary['mean_rho_v2']:+.4f} → v3={summary['mean_rho_v3']:+.4f} "
          f"(mean|Δ|={summary['mean_abs_delta']:.4f}, 符号一致={summary['sign_agreement']})")
    print(f"种子稳健性: mean ρ over 100 seeds = {summary['seed_robustness']['mean_rho_across_seeds']:+.4f} "
          f"± {summary['seed_robustness']['sd_across_seeds']:.4f} "
          f"[{summary['seed_robustness']['min']:+.4f}, {summary['seed_robustness']['max']:+.4f}], "
          f"全负={summary['seed_robustness']['all_negative']}")
    print(f"分解: ρ(mag)={summary['decomposition']['mean_rho_mag']:+.4f}, "
          f"ρ(S)={summary['decomposition']['mean_rho_S']:+.4f}, "
          f"partial(v3|mag) mean={summary['decomposition']['mean_partial_v3_given_mag']:+.4f}, "
          f"显著负 {summary['decomposition']['n_partial_sig_neg']}/8")
    print(f"判定: {json.dumps(summary['verdicts'], ensure_ascii=False)}")

    # ============ 输出 ============
    out = dict(summary=summary, per_protein=results,
               seed_mean_rho_distribution=[float(x) for x in seed_mean_rhos],
               runtime_sec=time.time() - t_start)
    with open(TABLES / "phase_o1_cgeo_v3.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    pd.DataFrame(results).T.to_csv(TABLES / "phase_o1_per_protein.csv")
    pd.DataFrame({"seed": range(1, N_SEEDS + 1), "mean_rho": seed_mean_rhos}).to_csv(
        TABLES / "phase_o1_seed_robustness.csv", index=False)
    if augmented:
        pd.concat(augmented).to_csv(OUT_DIR / "phase_o1_variants_augmented.csv", index=False)
    print(f"\n输出: {TABLES / 'phase_o1_cgeo_v3.json'} 等; 耗时 {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
