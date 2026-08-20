#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase O4 (准备): 真实突变系综验证 — 变体选择 + A3M 生成 (2026-08-03)
====================================================================
预注册 (PhaseO_correction_plan_20260803.md §O4):
  P53/PTEN/HRAS 三蛋白, 按 C_geo v3 三分位 × DMS 效应 (有害=最低四分位 / 中性=中位附近)
  分层, 每格 2 个, 每蛋白 12 变体 = 36 变体; 单点突变, position 在构建体内, wt_aa 与
  系综 sequence.fasta 逐位核对 (不一致则弃用该变体).

输出:
  field_theory/data/phase_o/mutant_a3m/{prot}_{wt}{pos}{mut}.a3m   (最小 A3M, 绕开 MSA)
  field_theory/data/phase_o/phase_o4_manifest.csv                  (变体清单+分层标签)
  field_theory/data/phase_o/run_o4_sampling.sh                     (WSL 后台采样脚本)
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "field_theory" / "data" / "phase_o"
A3M_DIR = OUT_DIR / "mutant_a3m"
SAMP_OUT = OUT_DIR / "mutant_bioemu"
A3M_DIR.mkdir(parents=True, exist_ok=True)
SAMP_OUT.mkdir(parents=True, exist_ok=True)
BIOEMU_WT = PROJECT_ROOT / "field_theory" / "data" / "dms" / "results" / "bioemu"

PROTEINS = {"P53": "p53_wt", "PTEN": "pten_wt", "HRAS": "hras_wt"}
SEED = 20260803
PER_CELL = 2  # 3 tertiles x 2 effects x 2 = 12/protein


def read_fasta_seq(p):
    lines = p.read_text().splitlines()
    return "".join(l.strip() for l in lines if not l.startswith(">"))


def main():
    df = pd.read_csv(OUT_DIR / "phase_o1_variants_augmented.csv")
    rng = np.random.default_rng(SEED)
    selected = []

    for prot, wt_dir in PROTEINS.items():
        wt_seq = read_fasta_seq(BIOEMU_WT / wt_dir / "sequence.fasta")
        n_res = len(wt_seq)
        sub = df[df["protein"] == prot].copy()
        # 位置映射 pos-1 (O1 已验证 3/3 蛋白均为 pos-1 方案)
        sub["idx0"] = sub["position"].astype(int) - 1
        sub = sub[(sub["idx0"] >= 0) & (sub["idx0"] < n_res)]
        # wt_aa 核对
        ok = sub.apply(lambda r: wt_seq[int(r["idx0"])] == r["wt_aa"], axis=1)
        n_bad = int((~ok).sum())
        sub = sub[ok].reset_index(drop=True)
        # 分层
        q13, q23 = sub["C_geo_v3"].quantile([1 / 3, 2 / 3]).values
        sub["cgeo_tertile"] = np.where(sub["C_geo_v3"] <= q13, "T1_low",
                                np.where(sub["C_geo_v3"] <= q23, "T2_mid", "T3_high"))
        del_thr = sub["DMS_score"].quantile(0.25)
        med = sub["DMS_score"].median()
        sub["dist_med"] = (sub["DMS_score"] - med).abs()
        neu_thr = sub["dist_med"].quantile(0.25)
        pools = {
            "deleterious": sub[sub["DMS_score"] <= del_thr],
            "neutral": sub[sub["dist_med"] <= neu_thr],
        }
        n_sel = 0
        for eff, pool in pools.items():
            for ter in ["T1_low", "T2_mid", "T3_high"]:
                cell = pool[pool["cgeo_tertile"] == ter]
                if len(cell) < PER_CELL:
                    cell = pool  # 回退: 该效应池内任选
                take = cell.sample(n=min(PER_CELL, len(cell)), random_state=rng.integers(1 << 31))
                for _, r in take.iterrows():
                    vid = f"{prot}_{r['wt_aa']}{int(r['position'])}{r['mut_aa']}"
                    mut_seq = wt_seq[: int(r["idx0"])] + r["mut_aa"] + wt_seq[int(r["idx0"]) + 1:]
                    a3m = A3M_DIR / f"{vid}.a3m"
                    a3m.write_text(f">{vid}\n{mut_seq}\n>pseudo_hit_1\n{mut_seq}\n")
                    selected.append(dict(
                        variant_id=vid, protein=prot, position=int(r["position"]),
                        wt_aa=r["wt_aa"], mut_aa=r["mut_aa"],
                        DMS_score=float(r["DMS_score"]), C_geo_v3=float(r["C_geo_v3"]),
                        mag=float(r["mag"]), S_stiffness=float(r["S_stiffness"]),
                        cgeo_tertile=ter, effect=eff, n_res=n_res,
                        a3m=str(a3m), out_dir=str(SAMP_OUT / vid),
                        wt_check="PASS",
                    ))
                    n_sel += 1
        print(f"[{prot}] n_res={n_res}, wt_aa 核对剔除 {n_bad}, 选出 {n_sel} 变体")

    man = pd.DataFrame(selected)
    man.to_csv(OUT_DIR / "phase_o4_manifest.csv", index=False)
    print(f"共 {len(man)} 变体 → {OUT_DIR / 'phase_o4_manifest.csv'}")

    # ---- WSL 采样脚本 (与主集参数一致: 250 samples, batch_size_100=5) ----
    lines = [
        "#!/bin/bash",
        "# Phase O4 真实突变系综采样 (BioEmu, 250 samples/变体)",
        "set +u  # conda activate 脚本引用未绑定变量, 不能用 set -u",
        "source /home/liuchuanyang13/anaconda3/etc/profile.d/conda.sh",
        "conda activate bioemu",
        "",
    ]
    for r in selected:
        a3m_wsl = r["a3m"].replace("B:\\", "/mnt/b/").replace("\\", "/")
        out_wsl = r["out_dir"].replace("B:\\", "/mnt/b/").replace("\\", "/")
        n = r["n_res"]
        timeout = 7200 if n > 300 else 3600
        lines += [
            f"mkdir -p '{out_wsl}'",
            f"if [ $(ls '{out_wsl}'/batch_*.npz 2>/dev/null | wc -l) -lt 5 ]; then",
            f"  echo '[O4] {r['variant_id']} ({n} aa)'",
            f"  timeout {timeout} python -m bioemu.sample '{a3m_wsl}' 250 '{out_wsl}' --batch_size_100 5 || echo 'TIMEOUT/FAIL: {r['variant_id']}'",
            "fi",
        ]
    sh = OUT_DIR / "run_o4_sampling.sh"
    sh.write_text("\n".join(lines) + "\n")
    print(f"WSL 采样脚本: {sh}")


if __name__ == "__main__":
    main()
