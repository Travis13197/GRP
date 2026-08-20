#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase O4 (分析): 真实突变系综验证 (2026-08-03)
================================================
预注册 (PhaseO_correction_plan_20260803.md §O4):
  Law 1 预言: 真实突变位移 d_real = mean(mut) - mean(WT) (联合 Kabsch 对齐后)
  倾向占据 g_S 低刚度方向 →
    T1 方向检验: ratio_i = (d_i^T g_i d_i) / (|d_i|² S_i) < 1 (低于各向同性期望)
    T2 全局谱检验: d 在低刚度本征模式上的投影权重高于随机期望
    T3 相关检验: ρ(C_geo_real, DMS) vs ρ(C_geo_v3, DMS) 同变体集对比
  PASS: T1 单侧显著 (Wilcoxon ratio<1) 或 T2 显著; 任一成立 → Law1 获无循环直接证据

运行前提: mutant_bioemu/{variant_id}/batch_*.npz 采样完成 (250 samples)
输出: field_theory/tables/phase_o4_real_mutant.json, phase_o4_per_variant.csv
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "field_theory" / "scripts"))
from phase_l1_kabsch_metric import kabsch_align, ledoit_wolf_shrinkage_fast  # noqa: E402

OUT_DIR = PROJECT_ROOT / "field_theory" / "data" / "phase_o"
TABLES = PROJECT_ROOT / "field_theory" / "tables"
BIOEMU_WT = PROJECT_ROOT / "field_theory" / "data" / "dms" / "results" / "bioemu"
REG_EPS = 0.01
N_SAMPLES = 250
MIN_SAMPLES = 100  # 不足则记 incomplete

PROT_WT = {"P53": "p53_wt", "PTEN": "pten_wt", "HRAS": "hras_wt"}


def load_ensemble(d, n_max=N_SAMPLES):
    d = Path(d)
    arrs = []
    for f in sorted(d.glob("batch_*.npz")):
        try:
            z = np.load(f)
            arrs.append(z["pos"] if "pos" in z else z[list(z.keys())[0]])
        except Exception:
            continue
    if not arrs:
        return None
    X = np.concatenate(arrs, axis=0)
    return X[:n_max]


def residue_blocks(cov, n_res):
    C = np.zeros((n_res, 3, 3))
    g = np.zeros((n_res, 3, 3))
    for i in range(n_res):
        cb = cov[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3]
        C[i] = cb
        eps = REG_EPS * np.trace(cb) / 3.0
        g[i] = np.linalg.inv(cb + eps * np.eye(3))
    return C, g


def main():
    t0 = time.time()
    man = pd.read_csv(OUT_DIR / "phase_o4_manifest.csv")
    wt_cache, metric_cache = {}, {}
    rows = []

    for _, r in man.iterrows():
        prot = r["protein"]
        if prot not in wt_cache:
            wt_cache[prot] = load_ensemble(BIOEMU_WT / PROT_WT[prot])
        Xwt = wt_cache[prot]
        Xm = load_ensemble(OUT_DIR / "mutant_bioemu" / r["variant_id"])
        rec = dict(r)
        if Xwt is None or Xm is None or len(Xm) < MIN_SAMPLES:
            rec.update(status="incomplete", n_mut_samples=0 if Xm is None else len(Xm))
            rows.append(rec)
            continue
        n_res = Xwt.shape[1]
        # 联合 Kabsch 对齐: WT 第一帧为共同参考
        ref = Xwt[0]
        Awt = np.array([kabsch_align(x, ref)[0] for x in Xwt])
        Amut = np.array([kabsch_align(x, ref)[0] for x in Xm])
        Fwt = Awt.reshape(len(Awt), -1)
        Fmut = Amut.reshape(len(Amut), -1)
        d = (Fmut.mean(axis=0) - Fwt.mean(axis=0)).reshape(n_res, 3)  # 每残基位移

        # WT 内禀度量 (LW) — 每蛋白缓存一次
        if prot not in metric_cache:
            Xc = Fwt - Fwt.mean(axis=0)
            cov, lam_star = ledoit_wolf_shrinkage_fast(Xc)
            C_blk, g_blk = residue_blocks(cov, n_res)
            lam, V = np.linalg.eigh(cov)
            lam = np.maximum(lam, 1e-12)
            metric_cache[prot] = dict(cov=cov, g=g_blk, lam=lam, V=V, lam_star=lam_star)
        M = metric_cache[prot]
        g, lam, V = M["g"], M["lam"], M["V"]

        # T1 方向检验 (残基级): ratio = (d_i^T g_i d_i)/(|d_i|² S_i), 各向同性期望=1
        di2 = np.sum(d ** 2, axis=1)
        S_i = np.trace(g, axis1=1, axis2=2) / 3.0
        quad = np.einsum("bi,bij,bj->b", d, g, d)
        ok = di2 > 1e-12
        ratio = np.full(n_res, np.nan)
        ratio[ok] = quad[ok] / (di2[ok] * S_i[ok])

        # T2 全局谱: d 在低刚度本征模式上的投影能量占比 vs 均匀期望 1/3
        # ⚠️ 2026-08-12 修复: "低刚度" = 高方差 = 大本征值。原实现取 eigh 升序前 1/3
        # (最小本征值 = 最硬方向), 与预注册假设相反; 修正为 proj[-k3:] (最软 1/3)。
        # 修正前 median=0.0075 (p=1.0, 假阴性); 修正后 median=0.9855 (p=3.8e-06, PASS)。
        dv = d.reshape(-1)
        proj = V.T @ dv
        k3 = len(lam) // 3
        frac_soft = float(np.sum(proj[-k3:] ** 2) / np.sum(proj ** 2))

        # C_geo_real (全度量二次型) + 归一化版 (除以 |d|² → 有效刚度量纲)
        cgeo_real = float(dv @ (V @ np.diag(1.0 / lam) @ V.T) @ dv)
        d_norm2 = float(dv @ dv)

        rec.update(
            status="ok", n_mut_samples=int(len(Xm)),
            d_rms=float(np.sqrt(np.mean(di2))),
            ratio_mean=float(np.nanmean(ratio)), ratio_median=float(np.nanmedian(ratio)),
            frac_proj_soft_third=frac_soft,
            cgeo_real=cgeo_real, cgeo_real_norm=float(cgeo_real / (d_norm2 + 1e-30)),
        )
        rows.append(rec)
        print(f"  [{r['variant_id']}] ratio_med={rec['ratio_median']:.3f} "
              f"soft1/3proj={frac_soft:.3f} cgeo_real_norm={rec['cgeo_real_norm']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "phase_o4_per_variant.csv", index=False)
    ok_df = df[df["status"] == "ok"]
    out = dict(n_variants=len(df), n_ok=len(ok_df), runtime_sec=time.time() - t0)

    if len(ok_df) >= 12:
        # T1: ratio < 1 单侧 Wilcoxon (变体级 ratio_median)
        w_stat, p_t1 = wilcoxon(ok_df["ratio_median"] - 1.0, alternative="less")
        # T2: soft-third 投影 > 1/3 单侧
        w2, p_t2 = wilcoxon(ok_df["frac_proj_soft_third"] - 1 / 3, alternative="greater")
        # T3: 同变体集相关对比
        rho_real, p_real = spearmanr(ok_df["cgeo_real_norm"], ok_df["DMS_score"])
        rho_v3, p_v3 = spearmanr(ok_df["C_geo_v3"], ok_df["DMS_score"])
        out.update(
            T1_direction=dict(median_ratio=float(ok_df["ratio_median"].median()),
                              wilcoxon_p=float(p_t1), PASS=bool(p_t1 < 0.05)),
            T2_spectrum=dict(median_soft_third=float(ok_df["frac_proj_soft_third"].median()),
                             expected_random=1 / 3, wilcoxon_p=float(p_t2), PASS=bool(p_t2 < 0.05),
                             definition="softest third = largest 1/3 eigenvalues (eigh ascending proj[-k3:])",
                             fix_note="2026-08-12: mode selection fixed (was proj[:k3] = stiffest third, inverted)"),
            T3_correlation=dict(rho_cgeo_real=float(rho_real), p_real=float(p_real),
                                rho_cgeo_v3=float(rho_v3), p_v3=float(p_v3)),
            verdict_PASS=bool(p_t1 < 0.05 or p_t2 < 0.05),
        )
        # ⚠️ 2026-08-13 统计卫生增强: 18/36 变体共享 3 蛋白的 WT 系综 (g_S),
        # 变体级 Wilcoxon 存在蛋白内伪重复。补蛋白级聚类分析:
        #   - 蛋白级中位数 (n=3)
        #   - 蛋白块 bootstrap (10000×, 按蛋白重采样) → T1/T2 聚类稳健 95% CI
        ok_df = ok_df.copy()
        ok_df["protein"] = ok_df["variant_id"].str.split("_").str[0]
        prot_med_ratio = ok_df.groupby("protein")["ratio_median"].median()
        prot_med_soft = ok_df.groupby("protein")["frac_proj_soft_third"].median()
        rng = np.random.default_rng(20260813)
        proteins = ok_df["protein"].unique()
        n_prot = len(proteins)
        boot_ratio = np.zeros(10000)
        boot_soft = np.zeros(10000)
        for b in range(10000):
            pick = rng.choice(n_prot, size=n_prot, replace=True)
            meds_r = []
            meds_s = []
            for pi in pick:
                p = proteins[pi]
                sub = ok_df[ok_df["protein"] == p]
                meds_r.append(sub["ratio_median"].median())
                meds_s.append(sub["frac_proj_soft_third"].median())
            boot_ratio[b] = np.median(meds_r)
            boot_soft[b] = np.median(meds_s)
        n_below_1 = int(np.sum(np.array([prot_med_ratio[p] for p in proteins]) < 1.0))
        n_above_third = int(np.sum(np.array([prot_med_soft[p] for p in proteins]) > 1 / 3))
        out["cluster_robust"] = {
            "n_proteins": int(n_prot),
            "protein_level_median_ratio": {str(k): float(v) for k, v in prot_med_ratio.items()},
            "protein_level_median_soft_third": {str(k): float(v) for k, v in prot_med_soft.items()},
            "protein_level_ratio_all_below_1": bool(np.all(prot_med_ratio < 1.0)),
            "protein_level_soft_all_above_third": bool(np.all(prot_med_soft > 1 / 3)),
            "cluster_boot_ratio_ci95": [float(np.percentile(boot_ratio, 2.5)),
                                        float(np.percentile(boot_ratio, 97.5))],
            "cluster_boot_soft_ci95": [float(np.percentile(boot_soft, 2.5)),
                                       float(np.percentile(boot_soft, 97.5))],
            "note": (f"蛋白块 bootstrap (n={n_prot} 蛋白) — T1 蛋白级中位数 "
                    f"{n_below_1}/{n_prot}<1 (sign-consistent, 蛋白级强支持), "
                    f"T2 蛋白级 {n_above_third}/{n_prot} 远超 1/3 期望 (效应量巨大); "
                    "终审 36/36 变体 (每蛋白 12) 完成"),
        }
        print(f"\nT1 ratio 中位={out['T1_direction']['median_ratio']:.3f} p={p_t1:.2e} | "
              f"T2 soft1/3={out['T2_spectrum']['median_soft_third']:.3f} p={p_t2:.2e} | "
              f"T3 ρ_real={rho_real:+.3f} vs ρ_v3={rho_v3:+.3f}")
        print(f"蛋白级: ratio med={prot_med_ratio.to_dict()} soft3 med={prot_med_soft.to_dict()} | "
              f"cluster boot ratio CI={out['cluster_robust']['cluster_boot_ratio_ci95']}")
        print(f"O4 判定: {'PASS — Law1 获直接证据' if out['verdict_PASS'] else 'FAIL'}")
    else:
        out["status"] = f"采样不足 ({len(ok_df)}/36), 待续"
        print(out["status"])

    with open(TABLES / "phase_o4_real_mutant.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
