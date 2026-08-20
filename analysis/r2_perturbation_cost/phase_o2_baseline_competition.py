#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase O2: 基线竞赛 (Baseline Competition) (2026-08-03)
========================================================
评审发现 S1 的修正实现 — C_geo v3 必须与平凡基线竞争:
  M0 mag (理化幅度)          M1 S (系综刚度, 内禀系)     M2 -RMSF (同系综)
  M3 GNM 刚度 (Kirchhoff 伪逆, 均值 Cα 结构, 10Å)        M4 埋藏度 (Cα 配位数)
  M5 Grantham 距离           M6 -BLOSUM62                M7 C_geo v3 (= mag²·S)
推断: 每蛋白 Spearman rho → mean-of-8 + n_eff=8 t; v3 vs 每基线 Δrho 蛋白整群 bootstrap CI.
判定: v3 vs 最优单基线 Δrho 95%CI 不含 0 且 v3 更强 → PASS.
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp, ttest_rel

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "field_theory" / "data" / "phase_o"
TABLES = PROJECT_ROOT / "field_theory" / "tables"
DMS_BIOEMU = PROJECT_ROOT / "field_theory" / "data" / "dms" / "results" / "bioemu"

N_BOOT = 10000
CUTOFF = 10.0  # Å, Cα 接触截断 (GNM/配位数)


def load_first_frame(dir_name):
    """加载系综第一帧 (真实物理构象) — 接触类基线 (GNM/埋藏度) 的代表结构.

    注意: 柔性系综的 Kabsch 对齐均值是坍缩伪影 (所有配对距离 <5Å),
    不能用于接触图计算; 第一帧即 Kabsch 对齐参考, 为真实采样构象.
    """
    d = DMS_BIOEMU / dir_name
    for f in sorted(d.glob("batch_*.npz")):
        z = np.load(f)
        key = "pos" if "pos" in z else list(z.keys())[0]
        return z[key][0] * 10.0  # BioEmu pos 单位为 nm → 转换为 Å
    raise FileNotFoundError(dir_name)

GRANTHAM = {  # Grantham 1974 距离矩阵
    "A": {"R": 112, "N": 111, "D": 126, "C": 195, "Q": 91, "E": 107, "G": 60, "H": 86, "I": 94, "L": 96, "K": 106, "M": 84, "F": 113, "P": 27, "S": 99, "T": 58, "V": 64, "W": 148, "Y": 112},
    "R": {"N": 86, "D": 96, "C": 180, "Q": 43, "E": 54, "G": 125, "H": 29, "I": 97, "L": 102, "K": 26, "M": 91, "F": 97, "P": 103, "S": 110, "T": 71, "V": 96, "W": 101, "Y": 77},
    "N": {"D": 23, "C": 139, "Q": 46, "E": 42, "G": 80, "H": 68, "I": 149, "L": 153, "K": 94, "M": 142, "F": 158, "P": 91, "S": 46, "T": 65, "V": 133, "W": 174, "Y": 143},
    "D": {"C": 154, "Q": 61, "E": 45, "G": 94, "H": 81, "I": 168, "L": 172, "K": 101, "M": 160, "F": 177, "P": 108, "S": 65, "T": 85, "V": 152, "W": 181, "Y": 160},
    "C": {"Q": 154, "E": 170, "G": 159, "H": 174, "I": 198, "L": 198, "K": 202, "M": 196, "F": 205, "P": 169, "S": 112, "T": 149, "V": 192, "W": 215, "Y": 194},
    "Q": {"E": 29, "G": 87, "H": 24, "I": 109, "L": 113, "K": 53, "M": 101, "F": 116, "P": 76, "S": 68, "T": 42, "V": 121, "W": 130, "Y": 99},
    "E": {"G": 98, "H": 40, "I": 134, "L": 138, "K": 56, "M": 126, "F": 140, "P": 93, "S": 80, "T": 65, "V": 121, "W": 152, "Y": 122},
    "G": {"H": 98, "I": 153, "L": 126, "K": 127, "M": 127, "F": 153, "P": 42, "S": 56, "T": 59, "V": 109, "W": 184, "Y": 147},
    "H": {"I": 94, "L": 99, "K": 32, "M": 87, "F": 100, "P": 77, "S": 89, "T": 47, "V": 84, "W": 115, "Y": 83},
    "I": {"L": 5, "K": 135, "M": 10, "F": 21, "P": 95, "S": 142, "T": 89, "V": 21, "W": 61, "Y": 33},
    "L": {"K": 107, "M": 15, "F": 22, "P": 98, "S": 145, "T": 92, "V": 32, "W": 61, "Y": 36},
    "K": {"M": 95, "F": 102, "P": 103, "S": 121, "T": 78, "V": 97, "W": 110, "Y": 85},
    "M": {"F": 28, "P": 87, "S": 135, "T": 81, "V": 21, "W": 67, "Y": 36},
    "F": {"P": 114, "S": 155, "T": 103, "V": 50, "W": 40, "Y": 22},
    "P": {"S": 74, "T": 38, "V": 110, "W": 147, "Y": 110},
    "S": {"T": 58, "V": 124, "W": 177, "Y": 144},
    "T": {"V": 69, "W": 128, "Y": 92},
    "V": {"W": 88, "Y": 55},
    "W": {"Y": 37},
}
BLOSUM62 = {  # BLOSUM62 矩阵 (Henikoff & Henikoff 1992)
    "A": {"A": 4, "R": -1, "N": -2, "D": -2, "C": 0, "Q": -1, "E": -1, "G": 0, "H": -2, "I": -1, "L": -1, "K": -1, "M": -1, "F": -2, "P": -1, "S": 1, "T": 0, "V": 0, "W": -3, "Y": -2},
    "R": {"R": 5, "N": 0, "D": -2, "C": -3, "Q": 1, "E": 0, "G": -2, "H": 0, "I": -3, "L": -2, "K": 2, "M": -1, "F": -3, "P": -2, "S": -1, "T": -1, "V": -3, "W": -3, "Y": -2},
    "N": {"N": 6, "D": 1, "C": -3, "Q": 0, "E": 0, "G": 0, "H": 1, "I": -3, "L": -3, "K": 0, "M": -2, "F": -3, "P": -2, "S": 1, "T": 0, "V": -3, "W": -4, "Y": -2},
    "D": {"D": 6, "C": -3, "Q": 0, "E": 2, "G": -1, "H": -1, "I": -3, "L": -4, "K": -1, "M": -3, "F": -3, "P": -1, "S": 0, "T": -1, "V": -3, "W": -4, "Y": -3},
    "C": {"C": 9, "Q": -3, "E": -4, "G": -3, "H": -3, "I": -1, "L": -1, "K": -3, "M": -1, "F": -2, "P": -3, "S": -1, "T": -1, "V": -1, "W": -2, "Y": -2},
    "Q": {"Q": 5, "E": 2, "G": -2, "H": 0, "I": -3, "L": -2, "K": 1, "M": 0, "F": -3, "P": -1, "S": 0, "T": -1, "V": -2, "W": -2, "Y": -1},
    "E": {"E": 5, "G": -2, "H": 0, "I": -3, "L": -3, "K": 1, "M": -2, "F": -3, "P": -1, "S": 0, "T": -1, "V": -2, "W": -3, "Y": -2},
    "G": {"G": 6, "H": -2, "I": -4, "L": -4, "K": -2, "M": -3, "F": -3, "P": -2, "S": 0, "T": -2, "V": -3, "W": -2, "Y": -3},
    "H": {"H": 8, "I": -3, "L": -3, "K": -1, "M": -2, "F": -1, "P": -2, "S": -1, "T": -2, "V": -3, "W": -2, "Y": 2},
    "I": {"I": 4, "L": 2, "K": -3, "M": 1, "F": 0, "P": -3, "S": -2, "T": -1, "V": 3, "W": -3, "Y": -1},
    "L": {"L": 4, "K": -2, "M": 2, "F": 0, "P": -3, "S": -2, "T": -1, "V": 1, "W": -2, "Y": -1},
    "K": {"K": 5, "M": -1, "F": -3, "P": -1, "S": 0, "T": -1, "V": -2, "W": -3, "Y": -2},
    "M": {"M": 5, "F": 0, "P": -2, "S": -1, "T": -1, "V": 1, "W": -1, "Y": -1},
    "F": {"F": 6, "P": -4, "S": -2, "T": -2, "V": -1, "W": 1, "Y": 3},
    "P": {"P": 7, "S": -1, "T": -1, "V": -2, "W": -4, "Y": -3},
    "S": {"S": 4, "T": 1, "V": -2, "W": -3, "Y": -2},
    "T": {"T": 5, "V": 0, "W": -4, "Y": -2},
    "V": {"V": 4, "W": -3, "Y": -1},
    "W": {"W": 11, "Y": 2},
    "Y": {"Y": 7},
}


def lookup(mat, a, b, default=0.0):
    if a in mat and b in mat[a]:
        return mat[a][b]
    if b in mat and a in mat[b]:
        return mat[b][a]
    return default


def gnm_stiffness(mean_ca, cutoff=CUTOFF):
    """GNM: Kirchhoff 矩阵伪逆对角元 = 均方涨落; 刚度 = 1/(msf+eps)"""
    n = len(mean_ca)
    d2 = np.sum((mean_ca[:, None, :] - mean_ca[None, :, :]) ** 2, axis=-1)
    contact = (d2 < cutoff ** 2) & (d2 > 1e-12)
    K = np.zeros((n, n))
    K[contact] = -1.0
    np.fill_diagonal(K, -K.sum(axis=1))
    msf = np.diag(np.linalg.pinv(K, hermitian=True))
    eps = 0.01 * np.mean(msf[msf > 0]) if np.any(msf > 0) else 1e-6
    return 1.0 / (msf + eps)


def coordination(mean_ca, cutoff=CUTOFF):
    d2 = np.sum((mean_ca[:, None, :] - mean_ca[None, :, :]) ** 2, axis=-1)
    return ((d2 < cutoff ** 2) & (d2 > 1e-12)).sum(axis=1).astype(float)


def map_idx(positions, n_res):
    pos = positions.astype(int)
    if pos.max() - 1 < n_res and pos.min() - 1 >= 0:
        return pos - 1
    off = pos.min()
    if (pos - off).max() < n_res:
        return pos - off
    return None


def main():
    t0 = time.time()
    df = pd.read_csv(OUT_DIR / "phase_o1_variants_augmented.csv")
    o1 = pd.read_csv(TABLES / "phase_o1_per_protein.csv", index_col=0)
    proteins = sorted(df["protein"].unique())
    rng = np.random.default_rng(20260803)

    per_prot = []
    for prot in proteins:
        sub = df[df["protein"] == prot].copy()
        rep_ca = load_first_frame(o1.loc[prot, "ensemble_dir"])  # 第一帧真实构象
        idx = map_idx(sub["position"].values, len(rep_ca))
        y = sub["DMS_score"].values.astype(float)
        gnm = gnm_stiffness(rep_ca)[idx]
        coord = coordination(rep_ca)[idx]
        grantham = np.array([lookup(GRANTHAM, w, m) for w, m in zip(sub["wt_aa"], sub["mut_aa"])])
        nblosum = -np.array([lookup(BLOSUM62, w, m) for w, m in zip(sub["wt_aa"], sub["mut_aa"])])
        predictors = {
            "M0_mag": sub["mag"].values, "M1_S_stiffness": sub["S_stiffness"].values,
            "M2_negRMSF": -sub["rmsf_site"].values, "M3_GNM_stiffness": gnm,
            "M4_burial_coord": coord, "M5_Grantham": grantham, "M6_negBLOSUM62": nblosum,
            "M7_C_geo_v3": sub["C_geo_v3"].values,
        }
        row = {"protein": prot, "n": len(sub)}
        for name, x in predictors.items():
            r, p = spearmanr(x, y)
            row[f"rho_{name}"] = r
        per_prot.append(row)
        print(f"  [{prot}] " + "  ".join(
            f"{k.split('rho_')[1]}={row[k]:+.4f}" for k in row if k.startswith("rho_")))

    perf = pd.DataFrame(per_prot)
    methods = [c[4:] for c in perf.columns if c.startswith("rho_")]
    R = {m: perf[f"rho_{m}"].values for m in methods}

    # --- mean-of-8 + n_eff=8 t ---
    summary = {}
    for m in methods:
        t, p = ttest_1samp(R[m], 0)
        summary[m] = dict(mean_rho=float(R[m].mean()), sd=float(R[m].std()),
                          neff8_t=float(t), neff8_p=float(p),
                          n_negative=int(np.sum(R[m] < 0)))
    # --- v3 vs 每基线: 配对 t + 蛋白整群 bootstrap Δrho CI ---
    deltas = {}
    for m in methods:
        if m == "M7_C_geo_v3":
            continue
        d = R["M7_C_geo_v3"] - R[m]  # 更负 = v3 更强 → 检验 d<0
        t, p = ttest_rel(R["M7_C_geo_v3"], R[m])
        boots = []
        for _ in range(N_BOOT):
            pick = rng.integers(0, len(d), len(d))
            boots.append(d[pick].mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        deltas[f"v3_vs_{m}"] = dict(mean_delta=float(d.mean()), ci95=[float(lo), float(hi)],
                                    paired_t=float(t), paired_p=float(p),
                                    v3_stronger=bool(hi < 0))
        print(f"  Δ(v3-{m}) = {d.mean():+.4f}  CI95[{lo:+.4f},{hi:+.4f}]  paired p={p:.3f}")

    best_baseline = max((m for m in methods if m != "M7_C_geo_v3"),
                        key=lambda m: abs(summary[m]["mean_rho"]))
    verdict = {
        "best_baseline": best_baseline,
        "best_baseline_mean_rho": summary[best_baseline]["mean_rho"],
        "v3_mean_rho": summary["M7_C_geo_v3"]["mean_rho"],
        "v3_beats_best_baseline": deltas[f"v3_vs_{best_baseline}"]["v3_stronger"],
        "note": "v3=mag²·S 为复合指标; S 单项 (M1) 亦列出供分解叙事",
    }
    out = dict(summary=summary, deltas=deltas, verdict=verdict,
               per_protein=perf.to_dict("records"), runtime_sec=time.time() - t0)
    with open(TABLES / "phase_o2_baseline_competition.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    perf.to_csv(TABLES / "phase_o2_per_protein.csv", index=False)
    print(f"\n判定: v3 vs 最优基线 {best_baseline}: "
          f"{'PASS' if verdict['v3_beats_best_baseline'] else 'FAIL (v3 无增量)'}")
    print(f"输出: {TABLES / 'phase_o2_baseline_competition.json'}; 耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
