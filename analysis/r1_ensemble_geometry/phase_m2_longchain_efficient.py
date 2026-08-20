#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase M2: 长链高效推断 (Long-Chain Efficient Geometric Inference)
================================================================

目标 (plan.md v1.0 Phase M2 🔴 / Q-C1):
  BRCA1 (1863aa) 级长链的几何特征能否通过 块协方差/低秩近似/精确恒等式 高效获得,
  从而把"几何估计"从采样成本中解耦。回答两个子问题:
    (a) 近似方法在 200-800aa 可验证区间的误差是否 <10%
    (b) N_req 随链长 n 的标度是否亚二次 (α<2), BRCA1 spectral_decay 外推是否有限

预注册判据 (先于任何计算固定, 与 plan.md §3 M2 行一致):
  PRIMARY  : 推荐方法 (rand_lowrank r=64 + flat-tail; block b=40) 在 200-800aa
             23 蛋白上, PR / spectral_decay(top-20) / entropy 三观测量的
             跨蛋白中位相对误差 < 10%
  SECONDARY: N_req(n) 幂律指数 α < 2 (亚二次); BRCA1 (n=1863) spectral_decay
             外推 95% 预测区间下界 > 0 (有限性)
  失败处置 : 披露不可达, 保留 FSS 外推+CI 的诚实声明 (plan.md 风险表 M2 行)

方法学核心 (三种加速路线):
  [1] ustat_pr   — PR 精确恒等式: PR = ||X||_F^4 / ||X^T X||_F^2,
                   ||X^T X||_F^2 = Σ_ij (x_i·x_j)^2 经 N×N Gram 矩阵获得,
                   O(N^2 q) 时间 / O(Nq) 内存, 完全不形成 q×q 协方差, 无近似
  [2] rand_lr    — 随机值域_finder (Halko-Martinsson-Tropp) top-r 特征值
                   + flat-tail 修正 (tr 精确已知, 尾部均摊到 q_eff-r 维)
                   O(N q r) 时间; r=64 时 BRCA1 级 (q=5589) 几何量分钟级可得
  [3] block_diag — 残基块对角协方差 (块大小 b 残基, 3b 维), 谱=各块谱并集
                   O(N q b) 时间; 检验长程关联截断的误差代价

观测量约定 (与 L6/K6 主线一致, Kabsch mean 参考 2 轮迭代对齐后):
  PR            = 1/Σ p_i^2,  p_i = λ_i/Σλ
  spectral_decay= -slope(log p_k ~ log k), 取 top-20 特征值 (与 M1 一致;
                  另报告 sd_all=L6 全谱拟合约定, 仅作参照不纳入判据)
  entropy       = -Σ p_i log p_i (Shannon 谱熵)

验证集 (200-800aa, ≥250 样本, 23 蛋白 — 全部已有系综, 无新 GPU 采样):
  L6 (250):  SMAD3 TAU MYC A4 ASH1 CREB1 FUS EWS MALE CHEY ELNE CAH2 TRY1 CTRA KAD1 UBC
  D1 (1000): TADBP CD28 PTEN MDM2 SNF5 SQSTM
  DMS (500): P53
  参照外推点: L4 FSS (PolyX K/G/E, n=10-200, n_req_1pct 实测 3240→465)

用法 (WSL2):
  source /home/liuchuanyang13/anaconda3/etc/profile.d/conda.sh && conda activate bioemu
  cd /mnt/b/2026/Exploration/ProtGenesis2_Ensemble
  python field_theory/scripts/phase_m2_longchain_efficient.py --validate   # Step1 近似误差
  python field_theory/scripts/phase_m2_longchain_efficient.py --nreq       # Step2 样本复杂度标度
  python field_theory/scripts/phase_m2_longchain_efficient.py --extrapolate# Step3 BRCA1 外推
  python field_theory/scripts/phase_m2_longchain_efficient.py --all        # 全部 + 判定 + 图

输出:
  field_theory/data/phase_m2/phase_m2_validation.csv    (蛋白×方法 相对误差)
  field_theory/data/phase_m2/phase_m2_nreq.csv          (N_req vs n)
  field_theory/data/phase_m2/phase_m2_report.json       (预注册判定)
  field_theory/figures/phase_m2/phase_m2_fig{1,2,3}.{jpg,svg,html}

作者: ProtGenesis2 Ensemble | 日期: 2026-07-23
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
FIELD_THEORY = Path(__file__).parent.parent
M2_DIR = FIELD_THEORY / "data" / "phase_m2"
FIG_DIR = FIELD_THEORY / "figures" / "phase_m2"
VALID_CSV = M2_DIR / "phase_m2_validation.csv"
NREQ_CSV = M2_DIR / "phase_m2_nreq.csv"
REPORT_JSON = M2_DIR / "phase_m2_report.json"
L4_FSS_JSON = FIELD_THEORY / "tables" / "phase_l4_convergence_analysis.json"
L6_POOL_CSV = FIELD_THEORY / "tables" / "phase_l6_geometric_functional.csv"

ENSEMBLE_DIRS = {
    "L6": PROJECT_ROOT / "test_workflow" / "phase_l6_natural50" / "output",
    "D1": PROJECT_ROOT / "test_workflow" / "idp_ensemble_phase_d" / "output",
    "D4": PROJECT_ROOT / "test_workflow" / "folded_proteins_phase_d" / "output",
    "DMS": PROJECT_ROOT / "test_workflow" / "polyx_ensemble" / "output",
}
DMS_NAMES = {"P04637_P53_HUMAN": "DMS_P53_WT", "P42212_GFP_AEQVI": "DMS_GFP_WT"}

# (protein_id, group, source, n_res)
PROTEINS = [
    ("P84022_SMAD3_HUMAN", "IDP", "L6", 425),
    ("P10636_TAU_HUMAN", "IDP", "L6", 758),
    ("P01106_MYC_HUMAN", "IDP", "L6", 454),
    ("P05067_A4_HUMAN", "IDP", "L6", 770),
    ("P34233_ASH1_YEAST", "IDP", "L6", 588),
    ("P16220_CREB1_HUMAN", "IDP", "L6", 327),
    ("P35637_FUS_HUMAN", "IDP", "L6", 526),
    ("Q01844_EWS_HUMAN", "IDP", "L6", 656),
    ("Q13148_TADBP_HUMAN", "IDP", "D1", 414),
    ("P10747_CD28_HUMAN", "IDP", "D1", 220),
    ("P60484_PTEN_HUMAN", "IDP", "D1", 403),
    ("Q00987_MDM2_HUMAN", "IDP", "D1", 491),
    ("Q12824_SNF5_HUMAN", "IDP", "D1", 385),
    ("Q13501_SQSTM_HUMAN", "IDP", "D1", 440),
    ("P04637_P53_HUMAN", "IDP", "DMS", 393),
    ("P0AEX9_MALE_ECOLI", "Folded", "L6", 396),
    ("P0AEZ3_CHEY_ECOLI", "Folded", "L6", 270),
    ("P08246_ELNE_HUMAN", "Folded", "L6", 267),
    ("P00918_CAH2_HUMAN", "Folded", "L6", 260),
    ("P00760_TRY1_BOVIN", "Folded", "L6", 246),
    ("P00766_CTRA_BOVIN", "Folded", "L6", 245),
    ("P69441_KAD1_ECOLI", "Folded", "L6", 214),
    ("P0CG48_UBC_HUMAN", "Folded", "L6", 685),
]

RANDOM_SEED = 42
SD_TOPK = 20                # spectral_decay 拟合取 top-20 (与 M1 约定一致)
RAND_RANKS = [32, 64, 128]  # 随机低秩档位
BLOCK_SIZES = [10, 20, 40, 80]  # 块对角档位 (残基)
PREREG_ERR = 0.10           # 预注册: 中位相对误差阈值
PREREG_ALPHA = 2.0          # 预注册: N_req 标度指数上限 (亚二次)
CV_THRESH = 0.05            # N_req 判定: bootstrap CV < 5%
N_GRID = [50, 100, 250, 500]
N_BOOT = 10
BRCA1_N = 1863              # BRCA1 链长 (P38398, L6 实测物理不可行: 0.22 样本/h)


# ---------------------------------------------------------------------------
# 数据加载与对齐 (与 M1/L2 管线一致)
# ---------------------------------------------------------------------------

def load_ensemble(protein_id, source):
    base = ENSEMBLE_DIRS[source]
    d = base / (DMS_NAMES.get(protein_id, protein_id) if source == "DMS" else protein_id)
    files = sorted(d.glob("batch_*.npz"))
    if not files:
        return None
    arrs = []
    for f in files:
        try:
            data = np.load(f)
            arrs.append(data["pos"] if "pos" in data else data["positions"])
        except Exception:
            pass
    return np.concatenate(arrs, axis=0) if arrs else None


def kabsch_to_ref(P, Q):
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return Pc @ R.T + Q.mean(axis=0)


def kabsch_align_ensemble(pos, n_iter=2):
    ref = pos.mean(axis=0)
    aligned = pos.copy()
    for _ in range(n_iter):
        aligned = np.stack([kabsch_to_ref(p, ref) for p in aligned])
        ref = aligned.mean(axis=0)
    return aligned


def centered_flat(aligned):
    n = aligned.shape[0]
    X = aligned.reshape(n, -1)
    return X - X.mean(axis=0)


# ---------------------------------------------------------------------------
# 观测量 (L6/K6 主线约定: 归一化谱)
# ---------------------------------------------------------------------------

def features_from_spectrum(lam, topk=SD_TOPK):
    """lam: 正特征值 (降序). 返回 PR, sd_topk, sd_all, entropy."""
    lam = lam[lam > 1e-12]
    p = lam / lam.sum()
    pr = 1.0 / np.sum(p ** 2)
    k_all = np.arange(1, len(p) + 1)
    sd_all = -np.polyfit(np.log(k_all), np.log(p), 1)[0] if len(p) >= 3 else np.nan
    kt = np.arange(1, min(topk, len(p)) + 1)
    sd_top = -np.polyfit(np.log(kt), np.log(p[:len(kt)]), 1)[0] if len(kt) >= 3 else np.nan
    ent = -np.sum(p * np.log(p + 1e-15))
    return {"PR": float(pr), "sd": float(sd_top), "sd_all": float(sd_all),
            "entropy": float(ent)}


# ---------------------------------------------------------------------------
# 方法 [0] 参考: 经济 SVD 全谱 (现行管线)
# ---------------------------------------------------------------------------

def ref_features(X):
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    return features_from_spectrum(s ** 2)


# ---------------------------------------------------------------------------
# 方法 [1] ustat_pr: PR 精确恒等式 (无协方差, 无近似)
# ---------------------------------------------------------------------------

def ustat_pr(X):
    """PR = ||X||_F^4 / ||X^T X||_F^2;  ||X^T X||_F^2 = Σ_ij (x_i·x_j)^2 = ||G||_F^2."""
    fro2 = float(np.sum(X ** 2))
    G = X @ X.T
    return fro2 * fro2 / float(np.sum(G ** 2))


# ---------------------------------------------------------------------------
# 方法 [2] rand_lr: 随机值域 finder top-r + flat-tail 修正
# ---------------------------------------------------------------------------

def rand_lowrank_features(X, r=64, over=10, qiter=1, rng=None):
    rng = rng or np.random.default_rng(RANDOM_SEED)
    n, q = X.shape
    l = int(min(r + over, n, q))
    Om = rng.standard_normal((q, l))
    Y = X @ Om
    for _ in range(qiter):
        Y = X @ (X.T @ Y)
    Q, _ = np.linalg.qr(Y, mode="reduced")
    sB = np.linalg.svd(Q.T @ X, compute_uv=False)
    lam_top = sB[:r] ** 2
    # flat-tail 修正: tr 精确已知, 尾部均摊到 q_eff - r 维
    tr = float(np.sum(X ** 2))
    q_eff = int(min(n - 1, q))
    tail = max(tr - float(lam_top.sum()), 0.0)
    n_tail = max(q_eff - r, 1)
    lam_ext = np.concatenate([lam_top, np.full(n_tail, tail / n_tail)])
    return features_from_spectrum(lam_ext)


# ---------------------------------------------------------------------------
# 方法 [3] block_diag: 残基块对角协方差
# ---------------------------------------------------------------------------

def block_diag_features(X, n_res, block_res=40):
    n, q = X.shape
    lam_all = []
    for i0 in range(0, n_res, block_res):
        i1 = min(i0 + block_res, n_res)
        # 残基 i 的 3 个坐标在展平向量中连续存放: [3i, 3i+3)
        cols = np.arange(3 * i0, 3 * i1)
        Xb = X[:, cols]
        if Xb.shape[1] < 3:
            continue
        s = np.linalg.svd(Xb, full_matrices=False, compute_uv=False)
        lam_all.append(s ** 2)
    lam = np.sort(np.concatenate(lam_all))[::-1]
    return features_from_spectrum(lam)


# ---------------------------------------------------------------------------
# Step 1: 近似误差验证 (200-800aa)
# ---------------------------------------------------------------------------

def validate():
    M2_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for protein_id, group, source, n_res in PROTEINS:
        pos = load_ensemble(protein_id, source)
        if pos is None:
            print(f"[validate] {protein_id}: ensemble missing, skip")
            continue
        t0 = time.time()
        aligned = kabsch_align_ensemble(pos)
        X = centered_flat(aligned)
        ref = ref_features(X)
        pr_exact = ustat_pr(X)
        rec = {"protein": protein_id, "group": group, "source": source,
               "n_res": n_res, "n_samples": int(pos.shape[0]),
               "ref_PR": ref["PR"], "ref_sd": ref["sd"], "ref_sd_all": ref["sd_all"],
               "ref_entropy": ref["entropy"],
               "ustat_PR": float(pr_exact),
               "err_ustat_PR": abs(pr_exact - ref["PR"]) / ref["PR"]}
        for r in RAND_RANKS:
            g = rand_lowrank_features(X, r=r, rng=rng)
            for k in ("PR", "sd", "entropy"):
                rec[f"rand{r}_{k}"] = g[k]
                rec[f"err_rand{r}_{k}"] = abs(g[k] - ref[k]) / max(abs(ref[k]), 1e-12)
        for b in BLOCK_SIZES:
            g = block_diag_features(X, n_res, block_res=b)
            for k in ("PR", "sd", "entropy"):
                rec[f"block{b}_{k}"] = g[k]
                rec[f"err_block{b}_{k}"] = abs(g[k] - ref[k]) / max(abs(ref[k]), 1e-12)
        rec["elapsed_s"] = round(time.time() - t0, 1)
        rows.append(rec)
        print(f"[validate] {protein_id} n={n_res} N={pos.shape[0]} "
              f"PR={ref['PR']:.2f} sd={ref['sd']:.3f} "
              f"err(rand64): PR={rec['err_rand64_PR']:.3f} sd={rec['err_rand64_sd']:.3f} "
              f"H={rec['err_rand64_entropy']:.3f} ({rec['elapsed_s']}s)")

    keys = sorted({k for r_ in rows for k in r_})
    with open(VALID_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[validate] {len(rows)} proteins -> {VALID_CSV}")

    # 预注册 PRIMARY 判定
    med = {}
    for method in ("ustat", "rand64", "rand128", "block40"):
        for k in ("PR", "sd", "entropy"):
            col = f"err_{method}_{k}"
            vals = [r[col] for r in rows if col in r and np.isfinite(r[col])]
            if vals:
                med[col] = float(np.median(vals))
    return rows, med


# ---------------------------------------------------------------------------
# Step 2: 样本复杂度 N_req(n) 标度
# ---------------------------------------------------------------------------

def nreq():
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for protein_id, group, source, n_res in PROTEINS:
        pos = load_ensemble(protein_id, source)
        if pos is None:
            continue
        aligned = kabsch_align_ensemble(pos)
        n_max = aligned.shape[0]
        rec = {"protein": protein_id, "group": group, "n_res": n_res, "n_max": int(n_max)}
        for n_sub in N_GRID:
            if n_sub > n_max:
                continue
            prs, sds = [], []
            for _ in range(N_BOOT):
                idx = rng.choice(n_max, size=n_sub, replace=False)
                Xb = centered_flat(aligned[idx])
                prs.append(ustat_pr(Xb))
                g = rand_lowrank_features(Xb, r=64, rng=rng)
                sds.append(g["sd"])
            for name, arr in (("PR", prs), ("sd", sds)):
                arr = np.array(arr)
                rec[f"cv_{name}_{n_sub}"] = float(arr.std(ddof=1) / arr.mean()) \
                    if arr.mean() > 0 else np.nan
        # N_req: 最小 N 使 CV<5% (线性插值)
        for name in ("PR", "sd"):
            ns = [n_ for n_ in N_GRID if f"cv_{name}_{n_}" in rec]
            cvs = [rec[f"cv_{name}_{n_}"] for n_ in ns]
            n_req = np.nan
            for i in range(len(ns)):
                if cvs[i] < CV_THRESH:
                    if i == 0:
                        n_req = ns[0]
                    else:
                        n_req = ns[i - 1] + (ns[i] - ns[i - 1]) * \
                            (cvs[i - 1] - CV_THRESH) / (cvs[i - 1] - cvs[i])
                    break
            rec[f"n_req_{name}"] = float(n_req) if np.isfinite(n_req) else np.nan
        rows.append(rec)
        print(f"[nreq] {protein_id} n={n_res}: N_req(PR)={rec['n_req_PR']:.0f}, "
              f"N_req(sd)={rec['n_req_sd']:.0f}")

    keys = sorted({k for r_ in rows for k in r_})
    with open(NREQ_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[nreq] {len(rows)} proteins -> {NREQ_CSV}")

    # 标度拟合: log N_req ~ α log n
    fits = {}
    for name in ("PR", "sd"):
        xs = np.array([r["n_res"] for r in rows if np.isfinite(r[f"n_req_{name}"])])
        ys = np.array([r[f"n_req_{name}"] for r in rows if np.isfinite(r[f"n_req_{name}"])])
        if len(xs) >= 5:
            b, a = np.polyfit(np.log(xs), np.log(ys), 1)
            pred = a + b * np.log(xs)
            r2 = 1 - np.sum((np.log(ys) - pred) ** 2) / np.sum((np.log(ys) - np.log(ys).mean()) ** 2)
            fits[name] = {"alpha": float(b), "intercept": float(a), "r2": float(r2),
                          "n_points": int(len(xs)),
                          "n_req_1863": float(np.exp(a + b * np.log(BRCA1_N)))}
            print(f"[nreq] N_req({name}) ~ n^{b:.2f} (R²={r2:.2f}), "
                  f"N_req(1863)={fits[name]['n_req_1863']:.0f}")
    return rows, fits


# ---------------------------------------------------------------------------
# Step 3: BRCA1 外推 (sd/PR/entropy 幂律 + 95% PI)
# ---------------------------------------------------------------------------

def _powerlaw_predict(n_arr, y_arr, n_target):
    """log-log 线性回归 + n_target 处 95% 预测区间."""
    from scipy import stats as sstats
    x, y = np.log(np.asarray(n_arr, float)), np.log(np.asarray(y_arr, float))
    m = np.isfinite(x) & np.isfinite(y) & (y > -30)
    x, y = x[m], y[m]
    slope, intercept, r, p, se = sstats.linregress(x, y)
    n_pts = len(x)
    x0 = np.log(n_target)
    y0 = intercept + slope * x0
    # 预测区间 (新观测)
    s_res = np.sqrt(np.sum((y - intercept - slope * x) ** 2) / (n_pts - 2))
    se_pred = s_res * np.sqrt(1 + 1 / n_pts + (x0 - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    t_crit = sstats.t.ppf(0.975, n_pts - 2)
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r ** 2),
            "p": float(p), "n_points": int(n_pts),
            "pred": float(np.exp(y0)),
            "pi_lo": float(np.exp(y0 - t_crit * se_pred)),
            "pi_hi": float(np.exp(y0 + t_crit * se_pred))}


def extrapolate():
    import pandas as pd
    df = pd.read_csv(L6_POOL_CSV)
    out = {}
    for feat in ("spectral_decay", "PR", "entropy"):
        out[feat] = _powerlaw_predict(df["n_residues"].values, df[feat].values, BRCA1_N)
        o = out[feat]
        print(f"[extrapolate] {feat}(1863) = {o['pred']:.3f} "
              f"95%PI=[{o['pi_lo']:.3f}, {o['pi_hi']:.3f}] "
              f"(slope={o['slope']:.3f}, R²={o['r2']:.2f}, n={o['n_points']})")
    # L4 FSS N_req 参照 (PolyX K/G/E, n=10-200, n_req_1pct)
    l4 = json.load(open(L4_FSS_JSON))
    l4_pts = []
    for key, v in l4.get("n_req_extrapolation", {}).items():
        n_res = int(key.split("_n")[1])
        l4_pts.append({"series": key, "n_res": n_res, "n_req_1pct": v["n_req_1pct"]})
    out["l4_fss_points"] = l4_pts
    if l4_pts:
        ns = np.array([p_["n_res"] for p_ in l4_pts])
        ys = np.array([p_["n_req_1pct"] for p_ in l4_pts])
        b, a = np.polyfit(np.log(ns), np.log(ys), 1)
        out["l4_fss_fit"] = {"alpha": float(b),
                             "n_req_1863": float(np.exp(a + b * np.log(BRCA1_N)))}
        print(f"[extrapolate] L4 FSS: N_req(1%) ~ n^{b:.2f}, N_req(1863)={out['l4_fss_fit']['n_req_1863']:.0f}")
    return out


# ---------------------------------------------------------------------------
# 图 (jpg+svg+html, 项目规范)
# ---------------------------------------------------------------------------

def make_figures(valid_rows, med_errs, nreq_rows, nreq_fits, extra, verdict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Fig1: 方法误差箱线图 (PR / sd / entropy 三面板) ----
    methods = [("ustat", "U-stat恒等式"), ("rand32", "低秩r=32"), ("rand64", "低秩r=64"),
               ("rand128", "低秩r=128"), ("block20", "块b=20"), ("block40", "块b=40"),
               ("block80", "块b=80")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, obs, obs_cn in zip(axes, ("PR", "sd", "entropy"),
                               ("参与比 PR", "谱衰减 sd(top-20)", "谱熵 entropy")):
        data, labels = [], []
        for m, mcn in methods:
            col = f"err_{m}_{obs}"
            vals = [r[col] for r in valid_rows if col in r and np.isfinite(r[col])]
            if vals:
                data.append(vals)
                labels.append(f"{mcn}\n中位={np.median(vals):.3f}")
        bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
        for patch, c in zip(bp["boxes"], plt.cm.viridis(np.linspace(0.15, 0.85, len(data)))):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        for i, vals in enumerate(data):
            ax.scatter(np.full(len(vals), i + 1) + np.random.default_rng(i).normal(0, 0.05, len(vals)),
                       vals, s=12, c="k", alpha=0.45, zorder=3)
        ax.axhline(PREREG_ERR, color="red", ls="--", lw=1.5, label=f"预注册阈值 {PREREG_ERR:.0%}")
        ax.set_ylabel("相对误差 relative error")
        ax.set_title(f"{obs_cn}", fontsize=12)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.legend(fontsize=9)
    fig.suptitle("Phase M2 长链高效推断 — 近似方法误差全景 (23 蛋白, 200-800aa)\n"
                 f"PRIMARY 判据: 中位误差<{PREREG_ERR:.0%} → rand64: "
                 f"PR={med_errs.get('err_rand64_PR', float('nan')):.3f}, "
                 f"sd={med_errs.get('err_rand64_sd', float('nan')):.3f}, "
                 f"H={med_errs.get('err_rand64_entropy', float('nan')):.3f} "
                 f"({'PASS' if verdict['primary_pass'] else 'FAIL'})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    for ext in ("jpg", "svg"):
        fig.savefig(FIG_DIR / f"phase_m2_fig1_method_errors.{ext}", dpi=300)
    plt.close(fig)

    # ---- Fig2: N_req ~ n 标度 (log-log) + L4 FSS 叠加 + BRCA1 外推 ----
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for r_ in nreq_rows:
        c = "#d62728" if r_["group"] == "IDP" else "#1f77b4"
        for name, mk in (("PR", "o"), ("sd", "s")):
            v = r_.get(f"n_req_{name}")
            if v and np.isfinite(v):
                ax.scatter(r_["n_res"], v, c=c, marker=mk, s=48, alpha=0.8,
                           edgecolors="k", linewidths=0.4, zorder=3)
    for name, mk, mc in (("PR", "o", "#2ca02c"), ("sd", "s", "#9467bd")):
        f_ = nreq_fits.get(name)
        if f_:
            xs = np.linspace(200, 2000, 100)
            ax.plot(xs, np.exp(f_["intercept"]) * xs ** f_["alpha"], "--", color=mc,
                    label=f"N_req({name}) ~ n^{f_['alpha']:.2f} (R²={f_['r2']:.2f})")
            ax.scatter([BRCA1_N], [f_["n_req_1863"]], marker="*", s=260, color=mc,
                       edgecolors="k", zorder=5)
    for p_ in extra.get("l4_fss_points", []):
        ax.scatter(p_["n_res"], p_["n_req_1pct"], marker="^", s=55, c="#ff7f0e",
                   edgecolors="k", linewidths=0.4, zorder=3)
    if extra.get("l4_fss_fit"):
        xs = np.linspace(8, 2000, 100)
        f_ = extra["l4_fss_fit"]
        ax.plot(xs, np.exp(-3.0) * xs ** f_["alpha"], ":", color="#ff7f0e", alpha=0.8)
    ax.scatter([], [], marker="^", c="#ff7f0e", label="L4 FSS PolyX (n_req 1%阈值)")
    ax.scatter([], [], marker="o", c="#2ca02c", label="N_req(PR) 天然蛋白 (5%阈值)")
    ax.scatter([], [], marker="s", c="#9467bd", label="N_req(sd) 天然蛋白 (5%阈值)")
    ax.axvline(BRCA1_N, color="gray", ls=":", lw=1)
    ax.text(BRCA1_N * 0.92, ax.get_ylim()[0] * 1.6, "BRCA1\n1863aa", rotation=90,
            fontsize=9, ha="right", color="gray")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("链长 n (residues, log)")
    ax.set_ylabel("N_req (log)")
    ax.set_title(f"Phase M2 样本复杂度标度 — α={nreq_fits.get('PR', {}).get('alpha', float('nan')):.2f} "
                 f"(预注册 α<{PREREG_ALPHA:.0f} 亚二次 → {'PASS' if verdict['secondary_pass'] else 'FAIL'})",
                 fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    for ext in ("jpg", "svg"):
        fig.savefig(FIG_DIR / f"phase_m2_fig2_nreq_scaling.{ext}", dpi=300)
    plt.close(fig)

    # ---- Fig3: BRCA1 spectral_decay 外推 (44蛋白幂律 + 95% PI) ----
    import pandas as pd
    df = pd.read_csv(L6_POOL_CSV)
    o = extra["spectral_decay"]
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for g, c in (("IDP", "#d62728"), ("Folded", "#1f77b4")):
        sub = df[df["group"] == g]
        ax.scatter(sub["n_residues"], sub["spectral_decay"], c=c, s=45, alpha=0.8,
                   edgecolors="k", linewidths=0.4, label=f"{g} (n={len(sub)})")
    xs = np.linspace(40, 2200, 200)
    ax.plot(xs, np.exp(o["intercept"]) * xs ** o["slope"], "k-",
            label=f"幂律 sd ~ n^{o['slope']:.3f} (R²={o['r2']:.2f})")
    ax.errorbar([BRCA1_N], [o["pred"]],
                yerr=[[o["pred"] - o["pi_lo"]], [o["pi_hi"] - o["pred"]]],
                fmt="*", ms=20, color="#ff7f0e", ecolor="#ff7f0e", elinewidth=2,
                capsize=6, zorder=5,
                label=f"BRCA1(1863) 外推: {o['pred']:.2f} 95%PI=[{o['pi_lo']:.2f},{o['pi_hi']:.2f}]")
    ax.set_xscale("log")
    ax.set_xlabel("链长 n (residues, log)")
    ax.set_ylabel("spectral_decay")
    ax.set_title(f"Phase M2 BRCA1 有限性检验 — sd 外推 95%PI 下界={o['pi_lo']:.3f}"
                 f" {'>0 有限 PASS' if o['pi_lo'] > 0 else 'FAIL'}", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    for ext in ("jpg", "svg"):
        fig.savefig(FIG_DIR / f"phase_m2_fig3_brca1_extrapolation.{ext}", dpi=300)
    plt.close(fig)

    # ---- HTML 交互版 (plotly) ----
    try:
        import plotly.graph_objects as go
        fig_h = go.Figure()
        for g, c in (("IDP", "#d62728"), ("Folded", "#1f77b4")):
            sub = df[df["group"] == g]
            fig_h.add_trace(go.Scatter(
                x=sub["n_residues"], y=sub["spectral_decay"], mode="markers",
                marker=dict(size=8, color=c), text=sub["protein"], name=g))
        fig_h.add_trace(go.Scatter(x=xs, y=np.exp(o["intercept"]) * xs ** o["slope"],
                                   mode="lines", name=f"sd ~ n^{o['slope']:.3f}",
                                   line=dict(color="black")))
        fig_h.add_trace(go.Scatter(x=[BRCA1_N], y=[o["pred"]], mode="markers",
                                   marker=dict(size=16, symbol="star", color="#ff7f0e"),
                                   error_y=dict(type="data", symmetric=False,
                                                array=[o["pi_hi"] - o["pred"]],
                                                arrayminus=[o["pred"] - o["pi_lo"]]),
                                   name=f"BRCA1 extrap {o['pred']:.2f}"))
        fig_h.update_xaxes(type="log", title="链长 n (log)")
        fig_h.update_yaxes(title="spectral_decay")
        fig_h.update_layout(width=1100, height=650,
                            title="Phase M2 BRCA1 spectral_decay 外推 (交互式)")
        fig_h.write_html(str(FIG_DIR / "phase_m2_fig3_brca1_extrapolation.html"))
    except Exception as e:
        print(f"[figures] plotly skipped: {e}")
    print(f"[figures] -> {FIG_DIR}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase M2: 长链高效推断")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--nreq", action="store_true")
    ap.add_argument("--extrapolate", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        args.validate = args.nreq = args.extrapolate = True
    if not (args.validate or args.nreq or args.extrapolate):
        ap.print_help()
        return

    valid_rows, med_errs, nreq_rows, nreq_fits, extra = [], {}, [], {}, {}
    if args.validate:
        valid_rows, med_errs = validate()
        with open(M2_DIR / "phase_m2_median_errors.json", "w") as f:
            json.dump(med_errs, f, indent=1)
    if args.nreq:
        nreq_rows, nreq_fits = nreq()
    if args.extrapolate:
        extra = extrapolate()

    if args.all and valid_rows and nreq_rows and extra:
        # ---- 预注册判定 ----
        primary = all(med_errs.get(f"err_rand64_{k}", 1.0) < PREREG_ERR
                      for k in ("PR", "sd", "entropy"))
        alpha_pr = nreq_fits.get("PR", {}).get("alpha", np.inf)
        sd_pi_lo = extra["spectral_decay"]["pi_lo"]
        secondary = (alpha_pr < PREREG_ALPHA) and (sd_pi_lo > 0)
        verdict = {
            "primary_pass": bool(primary),
            "secondary_pass": bool(secondary),
            "primary_detail": {k: med_errs.get(f"err_rand64_{k}") for k in ("PR", "sd", "entropy")},
            "secondary_detail": {"alpha_PR": alpha_pr, "sd_1863_pi_lo": sd_pi_lo,
                                 "sd_1863_pred": extra["spectral_decay"]["pred"]},
            "M2_verdict": "PASS" if (primary and secondary) else
                          ("PARTIAL" if (primary or secondary) else "FAIL_DISCLOSE"),
        }
        report = {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "preregistered": {"err_thresh": PREREG_ERR, "alpha_thresh": PREREG_ALPHA,
                                    "cv_thresh": CV_THRESH, "rand_rank_primary": 64,
                                    "seed": RANDOM_SEED, "n_boot": N_BOOT},
                  "median_errors": med_errs, "nreq_fits": nreq_fits,
                  "brca1_extrapolation": extra, "verdict": verdict}
        with open(REPORT_JSON, "w") as f:
            json.dump(report, f, indent=1)
        print(f"\n[verdict] PRIMARY (rand64 中位误差<10%): {'PASS' if primary else 'FAIL'} "
              f"{verdict['primary_detail']}")
        print(f"[verdict] SECONDARY (α<2 & BRCA1 sd有限): {'PASS' if secondary else 'FAIL'} "
              f"α_PR={alpha_pr:.2f}, sd(1863) PI_lo={sd_pi_lo:.3f}")
        print(f"[verdict] M2 = {verdict['M2_verdict']}")
        print(f"[report] -> {REPORT_JSON}")
        make_figures(valid_rows, med_errs, nreq_rows, nreq_fits, extra, verdict)


if __name__ == "__main__":
    main()
