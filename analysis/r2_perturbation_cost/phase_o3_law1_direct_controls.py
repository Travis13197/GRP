#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase O3: Law 1 直接检验三重对照 (2026-08-03)
================================================
评审发现 S2 的修正实现 (预注册: PhaseO_correction_plan_20260803.md §O3):

  O3-a 错配度量对照: 同一扰动 (n_s -> n_s+1) 分别用
    ① correct  : 本序列 Kabsch+LW 内禀精度 g_S
    ② permuted : 残基块随机置换的协方差 (谱匹配, 块结构破坏)
    ③ crossAA  : 跨 AA 错配 (同 n_s 他类序列的 Kabsch+LW 协方差)
    ④ rotated  : 本征向量随机旋转 (谱完全匹配, 方向随机化)
    ⑤ identity : 单位阵 (=raw 无度量)
    比较 Cons_geo (白化空间个体位移-平均位移余弦一致性) 与 CV_geo
  O3-b 内禀系重跑: Kabsch+LW 下 Cons_geo>Cons_raw win rate, 与实验室系 (stage10 口径) 对照
  O3-c 合成高斯零模型: 匹配 source/target 均值+LW协方差的合成系综过同一管线

判定 (预注册):
  错配对照 PASS: correct 的 Cons_geo 对全部 3 个错配变体 paired Wilcoxon 显著 (Holm)
  内禀系   PASS: win rate >= 80% 且与实验室系同向
  合成零模型 PASS: 真实 win rate - 合成 win rate > 5pp (否则检验无鉴别力)

输出: field_theory/tables/phase_o3_summary.json, phase_o3_per_perturbation.csv
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "field_theory" / "scripts"))
from phase_l1_kabsch_metric import kabsch_align, kabsch_align_ensemble, ledoit_wolf_shrinkage_fast  # noqa: E402

POLYX_DIR = PROJECT_ROOT / "test_workflow" / "polyx_ensemble" / "output"
TABLES = PROJECT_ROOT / "field_theory" / "tables"

AA_TYPES = ["G", "S", "E", "L", "K", "A", "V", "I", "F"]
N_MIN, N_MAX = 4, 50
N_SAMPLES = 250
LAB_REG = 1e-6  # stage10 实验室系口径
MASTER_SEED = 20260803


def load_ensemble(seq_id):
    d = POLYX_DIR / seq_id
    if not d.exists():
        return None
    arrs = []
    for f in sorted(d.glob("batch_*.npz")):
        try:
            z = np.load(f)
            arrs.append(z["pos"] if "pos" in z else z[list(z.keys())[0]])
        except Exception:
            continue
    if not arrs:
        return None
    return np.concatenate(arrs, axis=0)[:N_SAMPLES]


def align_trunc(X, n_min, ref=None):
    """Kabsch 对齐到 ref (缺省=自身第一帧), 截断 n_min, 返回 (n, 3*n_min) 平铺"""
    Xt = X[:, :n_min, :]
    ref3 = Xt[0] if ref is None else ref
    aligned = np.array([kabsch_align(x, ref3)[0] for x in Xt])
    return aligned.reshape(len(Xt), -1)


def cov_of(flat, mode="lw"):
    Xc = flat - flat.mean(axis=0)
    if mode == "lw":
        cov, _ = ledoit_wolf_shrinkage_fast(Xc)
    else:  # lab: stage10 固定正则化
        cov = (Xc.T @ Xc) / (len(Xc) - 1) + LAB_REG * np.eye(Xc.shape[1])
    return cov


def whiten_mat(cov):
    lam, V = np.linalg.eigh(cov)
    lam = np.maximum(lam, 1e-12)
    return V @ np.diag(1.0 / np.sqrt(lam))


def cons_cv(Xc_flat, d, W):
    """白化空间: 个体位移与 d 的余弦一致性均值 + 个体位移幅度 CV"""
    Xw = Xc_flat @ W
    dw = W.T @ d
    dn = np.linalg.norm(dw)
    if dn < 1e-12:
        return np.nan, np.nan
    xn = np.linalg.norm(Xw, axis=1)
    ok = xn > 1e-12
    cons = float(np.mean((Xw[ok] @ dw) / (xn[ok] * dn)))
    mags = xn[ok]
    cv = float(mags.std() / (mags.mean() + 1e-12))
    return cons, cv


def block_permute_cov(cov, n_min, rng):
    """残基 3x3 块级随机置换 (谱近似匹配, 块结构破坏)"""
    perm = rng.permutation(n_min)
    P = np.zeros((3 * n_min, 3 * n_min))
    for i, j in enumerate(perm):
        P[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3] = np.eye(3)
    return P @ cov @ P.T


def rotate_eigvecs_cov(cov, rng):
    """本征值不变, 本征向量 Haar 随机旋转"""
    lam, V = np.linalg.eigh(cov)
    Q, _ = np.linalg.qr(rng.standard_normal(cov.shape))
    V2 = V @ Q
    return (V2 * lam) @ V2.T


def main():
    t0 = time.time()
    rng = np.random.default_rng(MASTER_SEED)

    # ---------- 预加载全部系综 ----------
    ensembles = {}
    for aa in AA_TYPES:
        for n in range(N_MIN, N_MAX + 1):
            X = load_ensemble(f"PolyX_Poly{aa}_{n}")
            if X is not None and X.shape[0] >= 50:
                ensembles[(aa, n)] = X
    print(f"加载系综 {len(ensembles)} 个")

    rows = []
    for ai, aa in enumerate(AA_TYPES):
        partner = AA_TYPES[(ai + 1) % len(AA_TYPES)]
        ns = sorted(n for (a, n) in ensembles if a == aa)
        for n_s, n_t in zip(ns[:-1], ns[1:]):
            if n_t != n_s + 1:
                continue
            n_min = n_s
            Xs, Xt = ensembles[(aa, n_s)], ensembles[(aa, n_t)]
            # --- 内禀系联合对齐: target 对齐到 source 第一帧参考 ---
            Fs = align_trunc(Xs, n_min)
            ref_s = kabsch_align(Xs[0, :n_min], Xs[0, :n_min])[0]
            Ft = np.array([kabsch_align(x[:n_min], ref_s)[0] for x in Xt]).reshape(len(Xt), -1)
            ms, mt = Fs.mean(axis=0), Ft.mean(axis=0)
            d = mt - ms
            Xc = Fs - ms
            cov_s = cov_of(Fs, "lw")
            W_correct = whiten_mat(cov_s)
            cons_corr, cv_corr = cons_cv(Xc, d, W_correct)

            # --- 错配变体 ---
            W_perm = whiten_mat(block_permute_cov(cov_s, n_min, rng))
            cons_perm, cv_perm = cons_cv(Xc, d, W_perm)

            cons_cross, cv_cross = np.nan, np.nan
            Xp = ensembles.get((partner, n_s))
            if Xp is not None:
                Fp = align_trunc(Xp, n_min)
                W_cross = whiten_mat(cov_of(Fp, "lw"))
                cons_cross, cv_cross = cons_cv(Xc, d, W_cross)

            W_rot = whiten_mat(rotate_eigvecs_cov(cov_s, rng))
            cons_rot, cv_rot = cons_cv(Xc, d, W_rot)

            cons_raw, cv_raw = cons_cv(Xc, d, np.eye(3 * n_min))

            # --- 实验室系 (stage10 口径: 无对齐, 固定 reg) ---
            Fs_lab = Xs[:, :n_min, :].reshape(len(Xs), -1)
            Ft_lab = Xt[:, :n_min, :].reshape(len(Xt), -1)
            d_lab = Ft_lab.mean(axis=0) - Fs_lab.mean(axis=0)
            Xc_lab = Fs_lab - Fs_lab.mean(axis=0)
            W_lab = whiten_mat(cov_of(Fs_lab, "lab"))
            cons_geo_lab, cv_geo_lab = cons_cv(Xc_lab, d_lab, W_lab)
            cons_raw_lab, cv_raw_lab = cons_cv(Xc_lab, d_lab, np.eye(3 * n_min))

            # --- O3-c 合成高斯零模型 (匹配内禀系均值+协方差) ---
            cov_t = cov_of(Ft, "lw")
            Ls = np.linalg.cholesky(cov_s + 1e-12 * np.eye(len(cov_s)))
            Lt = np.linalg.cholesky(cov_t + 1e-12 * np.eye(len(cov_t)))
            Zs = ms + rng.standard_normal((N_SAMPLES, len(ms))) @ Ls.T
            Zt = mt + rng.standard_normal((N_SAMPLES, len(mt))) @ Lt.T
            d_syn = Zt.mean(axis=0) - Zs.mean(axis=0)
            Zc = Zs - Zs.mean(axis=0)
            W_syn = whiten_mat(cov_of(Zs, "lw"))
            cons_syn, cv_syn = cons_cv(Zc, d_syn, W_syn)
            cons_syn_raw, cv_syn_raw = cons_cv(Zc, d_syn, np.eye(3 * n_min))

            rows.append(dict(
                aa=aa, n_s=n_s, n_t=n_t, n_min=n_min,
                cons_correct=cons_corr, cv_correct=cv_corr,
                cons_permuted=cons_perm, cv_permuted=cv_perm,
                cons_crossAA=cons_cross, cv_crossAA=cv_cross,
                cons_rotated=cons_rot, cv_rotated=cv_rot,
                cons_raw=cons_raw, cv_raw=cv_raw,
                cons_geo_lab=cons_geo_lab, cv_geo_lab=cv_geo_lab,
                cons_raw_lab=cons_raw_lab, cv_raw_lab=cv_raw_lab,
                cons_syn_geo=cons_syn, cv_syn_geo=cv_syn,
                cons_syn_raw=cons_syn_raw, cv_syn_raw=cv_syn_raw,
            ))
        print(f"  [{aa}] 累计 {len(rows)} 扰动")

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "phase_o3_per_perturbation.csv", index=False)
    n = len(df)

    # ---------- 判定 ----------
    # O3-a: correct vs 3 错配 (paired Wilcoxon, Holm)
    mismatches = ["permuted", "crossAA", "rotated"]
    pvals, deltas, winrates = [], {}, {}
    for v in mismatches:
        c, m = df["cons_correct"].values, df[f"cons_{v}"].values
        ok = ~(np.isnan(c) | np.isnan(m))
        deltas[v] = float(np.mean(c[ok] - m[ok]))
        winrates[v] = float(np.mean(c[ok] > m[ok]))
        pvals.append(wilcoxon(c[ok], m[ok], alternative="greater")[1])
    rej, p_holm, _, _ = multipletests(pvals, method="holm")
    o3a_pass = bool(np.all(rej))

    # O3-b: 内禀系 win rate + 实验室系对照
    win_intr_cons = float(np.mean(df["cons_correct"] > df["cons_raw"]))
    win_intr_cv = float(np.mean(df["cv_correct"] < df["cv_raw"]))
    win_lab_cons = float(np.mean(df["cons_geo_lab"] > df["cons_raw_lab"]))
    win_lab_cv = float(np.mean(df["cv_geo_lab"] < df["cv_raw_lab"]))
    o3b_pass = bool(win_intr_cons >= 0.80 and np.sign(win_intr_cons - 0.5) == np.sign(win_lab_cons - 0.5))

    # O3-c: 合成零模型鉴别力
    win_syn_cons = float(np.mean(df["cons_syn_geo"] > df["cons_syn_raw"]))
    win_syn_cv = float(np.mean(df["cv_syn_geo"] < df["cv_syn_raw"]))
    o3c_pass = bool((win_intr_cons - win_syn_cons) > 0.05)

    summary = dict(
        n_perturbations=int(n),
        means={k: float(df[f"cons_{k}"].mean()) for k in
               ["correct", "permuted", "crossAA", "rotated", "raw", "geo_lab", "raw_lab", "syn_geo", "syn_raw"]},
        o3a_mismatch=dict(
            delta_cons=deltas, win_rate=winrates,
            p_raw=dict(zip(mismatches, [float(p) for p in pvals])),
            p_holm=dict(zip(mismatches, [float(p) for p in p_holm])),
            PASS=o3a_pass),
        o3b_intrinsic=dict(
            win_rate_cons=win_intr_cons, win_rate_cv=win_intr_cv,
            lab_win_rate_cons=win_lab_cons, lab_win_rate_cv=win_lab_cv,
            same_direction=bool(np.sign(win_intr_cons - 0.5) == np.sign(win_lab_cons - 0.5)),
            PASS=o3b_pass),
        o3c_synthetic=dict(
            win_rate_cons_syn=win_syn_cons, win_rate_cv_syn=win_syn_cv,
            gap_pp_cons=float(win_intr_cons - win_syn_cons),
            PASS=o3c_pass),
        verdicts=dict(O3a=o3a_pass, O3b=o3b_pass, O3c=o3c_pass),
        runtime_sec=time.time() - t0,
    )
    with open(TABLES / "phase_o3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========== O3 汇总 ==========")
    print(f"扰动数: {n}")
    print(f"Cons 均值: correct={summary['means']['correct']:.4f} permuted={summary['means']['permuted']:.4f} "
          f"crossAA={summary['means']['crossAA']:.4f} rotated={summary['means']['rotated']:.4f} raw={summary['means']['raw']:.4f}")
    print(f"O3-a 错配: Δcons={ {k: round(v, 4) for k, v in deltas.items()} } "
          f"win={ {k: round(v, 3) for k, v in winrates.items()} } "
          f"p_holm={ {k: f'{p:.1e}' for k, p in zip(mismatches, p_holm)} } → {'PASS' if o3a_pass else 'FAIL'}")
    print(f"O3-b 内禀系: win(cons)={win_intr_cons:.3f} win(cv)={win_intr_cv:.3f} | "
          f"实验室系 win(cons)={win_lab_cons:.3f} win(cv)={win_lab_cv:.3f} → {'PASS' if o3b_pass else 'FAIL'}")
    print(f"O3-c 合成: win(cons)={win_syn_cons:.3f} win(cv)={win_syn_cv:.3f} "
          f"gap={win_intr_cons - win_syn_cons:+.3f} → {'PASS' if o3c_pass else 'FAIL'}")
    print(f"耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
