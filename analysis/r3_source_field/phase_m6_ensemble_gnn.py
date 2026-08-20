#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase M6: Ensemble-level K Representation Learning — GNN on Ensemble Graphs
================================================================================
背景:
  Law 2 (T^bio →K→ Y^geom) 的 H0 (耦合不可还原性) 已确立: K 结构不可还原为任何
  **序列级** 描述符 (四级排除链: 物化属性 GBRT R²=-0.431; V2 二阶模式矩+15上下文
  特征 Ridge LOAO R²=-0.147 / GBRT -0.718; 天然IDP迁移 ΔR²=-2.399)。
  M6 是 H0 之后的唯一可行路线: 用 GNN 直接在**系综图** (残基=节点, 涨落相关=边)
  上学习 K 的表示, 检验系综级表示能否压缩 K。

预注册判据 (PRE-REGISTERED — 先于任何计算固定, 判定只使用主配置结果):
  P1 (PRIMARY): 与 V2 完全同构的 leave-one-AA-out (LOAO) CV — Panel A 60 个
      上下文对 (6 focal AAs {D,G,H,K,L,R} × C(5,2)=10), y = cos_sim(K_c1, K_c2)
      (K = GEO5×BIO8 = 40维, Ridge α=50 项目标准估计器);
      GNN 系综嵌入 (64维) → 对特征 |Δz| → Ridge(α=50)。
      全体预测拼接后 OOS R² > 0.3 → SUCCESS (K 可被系综级深度表示压缩, H0 弱化)。
  P2 (失败处置): OOS R² ≤ 0.3 → verdict = H0_STRENGTHENED
      ("K 亦不可被系综级深度模型压缩" 作为理论发现报告)。
  P3 (基线对照): V2 Ridge -0.147, GBRT -0.718, 单AA物化基线 -0.431。
  P4 (阴性对照): 打乱 stratum K 标签 (seed=123) 重跑一次完整 LOAO,
      OOS R² 应 ≈ 0 或为负; 若阴性对照同样高, 主结果不可信。
  P5 (复现性检查): 先复现 V2 Ridge 基线 (15上下文特征 |Δx| → Ridge α=50,
      kappa 列 NaN 自动剔除 → 14 有效特征), 所得 R² 应接近官方值 -0.147
      (容差 ±0.1), 确认管线与 V2 同构。
  P6 (架构调整登记): 仅当主配置 OOS R² ∈ [0.2, 0.35] 时允许有限架构调整
      (layers 2-4, hidden 32/64/128, dropout 0-0.3), 全部尝试登记在 report;
      最终判定只使用主配置 (3×GINEConv, hidden=64, emb=64, dropout=0.1)。

V2 协议镜像要点 (test_workflow/law2_validation/validation_tests.py criterion_v2):
  - 仅 Panel A: 6 focal AAs × 5 contexts (A1..A5) = 30 strata (NPZ 名 'A_{aa}_{c}')
  - stratum K 向量: 直接用 NPZ K_geo5 行 (40维), 并与 estimate_k 重算核对一致
  - 对: 同 AA 内 C(5,2)=10 对 × 6 AA = 60 对; y = cos_sim(K_c1, K_c2)
  - LOAO: 每折留 1 个 AA; StandardScaler 仅在训练 AA 对上拟合
  - OOS R² = 1 - SS_res/SS_tot (全体预测拼接, SS_tot 对全局均值)

M6 设计 (GNN):
  - 图构建 (每序列): Kabsch 对齐系综 (≤250 样本) → 残基为节点;
    节点特征 [rmsf, 局部弯曲涨落(相邻残基), 相对序列位置];
    边: 序列相邻 + top-8 |Pearson相关| (残基涨落幅度时间序列) 非相邻残基对;
    边特征 [相关值, 系综平均距离, 序列间隔]。变长图 (n=12..48), GNN 天然处理。
  - 模型: 3 层 GINEConv (edge_dim=3) + BatchNorm + global mean/max pool
    → 64维序列嵌入 → MLP 回归该序列所属 stratum 的 K_geo5 (40维,
    同 stratum 内 4 条序列共享标签)。Panel A 120 条序列参与。
  - 端到端 CV (公平性): 每折留 1 个 AA → GNN 只在其余 5 AA 的序列上训练
    → 嵌入全部 6 AA 序列 → stratum 嵌入 = 成员序列嵌入均值 → |Δz| (64维)
    → Ridge(α=50) 在训练 AA 的 50 对上拟合 → 预测留出 AA 的 10 对。
  - 训练: Adam lr=1e-3, batch=16, ≤200 epochs, early stop
    (val = 每训练 AA 随机 1 个 context, 即 25 strata 中 5 个, 20%, seed=42,
    patience=25, min_epochs=30), 记录训练曲线。

数据资产 (全部已存在, 不重新采样):
  - 系综: test_workflow/law2_validation/output_full/{seq_id}/batch_*.npz (键 'pos')
  - 元数据: test_workflow/law2_validation/validation_320_metadata.csv
  - 上下文特征: test_workflow/law2_validation/context_features.csv
  - 几何特征: field_theory/tables/law2_full_geometry.csv
  - K 矩阵: test_workflow/law2_validation/K_matrices_context.npz

输出:
  field_theory/data/phase_m6/phase_m6_report.json   (判据+复现+主结果+阴性+登记)
  field_theory/data/phase_m6/m6_pairs.csv           (60对 × aa,c1,c2,y_true,pred)
  field_theory/figures/phase_m6/phase_m6_fig1.{svg,jpg,html}  (预测vs实测)
  field_theory/figures/phase_m6/phase_m6_fig2.{svg,jpg,html}  (训练曲线+对照)
  运行日志经 shell tee 写入 field_theory/data/phase_m6/m6_run.log

用法 (WSL2, bioemu env):
  OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    python -u field_theory/scripts/phase_m6_ensemble_gnn.py \
    2>&1 | tee field_theory/data/phase_m6/m6_run.log
"""

import json
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "field_theory" / "scripts"))
from phase_l1_kabsch_metric import kabsch_align_ensemble  # noqa: E402

L2V_DIR = PROJECT_ROOT / "test_workflow" / "law2_validation"
GEOM_CSV = PROJECT_ROOT / "field_theory" / "tables" / "law2_full_geometry.csv"
META_CSV = L2V_DIR / "validation_320_metadata.csv"
CTX_CSV = L2V_DIR / "context_features.csv"
K_NPZ = L2V_DIR / "K_matrices_context.npz"
ENS_DIR = L2V_DIR / "output_full"

OUT_DIR = PROJECT_ROOT / "field_theory" / "data" / "phase_m6"
FIG_DIR = PROJECT_ROOT / "field_theory" / "figures" / "phase_m6"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_CACHE = OUT_DIR / "m6_graph_cache.pt"

# ---------------------------------------------------------------------------
# 常量 (与 V2 / k_context_estimation.py 完全同构)
# ---------------------------------------------------------------------------
ALPHA = 50.0
SEED = 42
NULL_SEED = 123
MAX_SAMPLES = 250
BIO_FEATURES = ['n', 'hydrophobicity', 'charge', 'volume', 'mw', 'flexibility',
                'helix_propensity', 'sheet_propensity']
GEO5 = ['PR', 'entropy', 'spectral_decay', 'eff_rank_95', 'total_variance']
FOCAL_AAS = ['D', 'G', 'H', 'K', 'L', 'R']
CONTEXTS_A = ['A1', 'A2', 'A3', 'A4', 'A5']
CTX_FEATURES_15 = ['pattern_period', 'block_position', 'local_comp_std', 'focal_fraction',
                   'FCR', 'NCPR', 'abs_NCPR', 'mean_hydro', 'kappa', 'SCD_sqrtL',
                   'SHD', 'gamma1', 'blockiness', 'H_comp', 'H_dipep']
Y_COLUMNS = [  # 与 validation_tests.py / k_context_estimation.py 完全一致的 36 维几何列
    'PR', 'A_C', 'eff_rank_95', 'eff_rank_99', 'spectral_decay', 'entropy',
    'total_variance', 'pseudo_volume', 'mean_rmsf', 'max_rmsf', 'rmsf_cv',
    'local_stiffness', 'fluct_range_ratio', 'rmsf_entropy',
    'mardia_skewness', 'mardia_kurtosis', 'mean_pc_skewness', 'mean_pc_kurtosis',
    'corr_dim', 'mean_knn_dist', 'cv_knn_dist',
    'condition_number', 'spectral_gap', 'spectral_gap_ratio',
    'fisher_trace', 'fisher_logdet', 'mean_mi_pc3', 'js_divergence',
    'effective_diffusion', 'relaxation_time', 'lyapunov_proxy', 'convective_ratio',
    'contact_order', 'clustering_coeff', 'modularity', 'betweenness_cv'
]
V2_REFERENCE_R2 = -0.147          # 官方 law2_validation_report.json: -0.1467
V2_GBRT_R2 = -0.718
SINGLE_AA_BASELINE = -0.431
SUCCESS_THRESHOLD = 0.3
TUNE_BAND = (0.2, 0.35)

MAIN_CONFIG = {'layers': 3, 'hidden': 64, 'emb': 64, 'dropout': 0.1,
               'lr': 1e-3, 'batch': 16, 'max_epochs': 200,
               'patience': 25, 'min_epochs': 30}
TUNING_CONFIGS = [                 # 仅当主配置 R² ∈ [0.2, 0.35] 时运行 (P6 登记)
    {'layers': 2, 'hidden': 64, 'emb': 64, 'dropout': 0.1},
    {'layers': 4, 'hidden': 64, 'emb': 64, 'dropout': 0.1},
    {'layers': 3, 'hidden': 128, 'emb': 64, 'dropout': 0.3},
    {'layers': 3, 'hidden': 32, 'emb': 64, 'dropout': 0.0},
]

AA_PROPERTIES = {
    'G': {'hydrophobicity': -0.4, 'charge': 0, 'volume': 60.1, 'mw': 75.07, 'flexibility': 0.544, 'helix_propensity': 0.57, 'sheet_propensity': 0.75},
    'A': {'hydrophobicity': 1.8, 'charge': 0, 'volume': 88.6, 'mw': 89.09, 'flexibility': 0.437, 'helix_propensity': 1.42, 'sheet_propensity': 0.83},
    'S': {'hydrophobicity': -0.8, 'charge': 0, 'volume': 89.0, 'mw': 105.09, 'flexibility': 0.507, 'helix_propensity': 0.77, 'sheet_propensity': 0.75},
    'V': {'hydrophobicity': 4.2, 'charge': 0, 'volume': 140.0, 'mw': 117.15, 'flexibility': 0.386, 'helix_propensity': 1.06, 'sheet_propensity': 1.70},
    'I': {'hydrophobicity': 4.5, 'charge': 0, 'volume': 166.7, 'mw': 131.18, 'flexibility': 0.402, 'helix_propensity': 1.08, 'sheet_propensity': 1.60},
    'L': {'hydrophobicity': 3.8, 'charge': 0, 'volume': 166.7, 'mw': 131.18, 'flexibility': 0.398, 'helix_propensity': 1.41, 'sheet_propensity': 1.30},
    'F': {'hydrophobicity': 2.8, 'charge': 0, 'volume': 189.9, 'mw': 165.19, 'flexibility': 0.382, 'helix_propensity': 1.13, 'sheet_propensity': 1.38},
    'W': {'hydrophobicity': -0.9, 'charge': 0, 'volume': 227.8, 'mw': 204.23, 'flexibility': 0.314, 'helix_propensity': 1.08, 'sheet_propensity': 1.37},
    'Y': {'hydrophobicity': -1.3, 'charge': 0, 'volume': 193.6, 'mw': 181.19, 'flexibility': 0.393, 'helix_propensity': 0.69, 'sheet_propensity': 1.47},
    'P': {'hydrophobicity': -1.6, 'charge': 0, 'volume': 112.7, 'mw': 115.13, 'flexibility': 0.509, 'helix_propensity': 0.57, 'sheet_propensity': 0.55},
    'C': {'hydrophobicity': 2.5, 'charge': 0, 'volume': 108.5, 'mw': 121.16, 'flexibility': 0.346, 'helix_propensity': 0.70, 'sheet_propensity': 1.19},
    'M': {'hydrophobicity': 1.9, 'charge': 0, 'volume': 162.9, 'mw': 149.21, 'flexibility': 0.433, 'helix_propensity': 1.45, 'sheet_propensity': 1.05},
    'T': {'hydrophobicity': -0.7, 'charge': 0, 'volume': 116.1, 'mw': 119.12, 'flexibility': 0.444, 'helix_propensity': 0.83, 'sheet_propensity': 1.19},
    'N': {'hydrophobicity': -3.5, 'charge': 0, 'volume': 114.1, 'mw': 132.12, 'flexibility': 0.463, 'helix_propensity': 0.67, 'sheet_propensity': 0.89},
    'Q': {'hydrophobicity': -3.5, 'charge': 0, 'volume': 143.8, 'mw': 146.15, 'flexibility': 0.478, 'helix_propensity': 1.11, 'sheet_propensity': 1.10},
    'D': {'hydrophobicity': -3.5, 'charge': -1, 'volume': 111.1, 'mw': 133.10, 'flexibility': 0.508, 'helix_propensity': 1.01, 'sheet_propensity': 0.54},
    'E': {'hydrophobicity': -3.5, 'charge': -1, 'volume': 138.4, 'mw': 147.13, 'flexibility': 0.497, 'helix_propensity': 1.51, 'sheet_propensity': 0.37},
    'H': {'hydrophobicity': -3.2, 'charge': 0.5, 'volume': 153.2, 'mw': 155.16, 'flexibility': 0.432, 'helix_propensity': 1.00, 'sheet_propensity': 0.87},
    'K': {'hydrophobicity': -3.9, 'charge': 1, 'volume': 168.6, 'mw': 146.19, 'flexibility': 0.466, 'helix_propensity': 1.16, 'sheet_propensity': 0.74},
    'R': {'hydrophobicity': -4.5, 'charge': 1, 'volume': 173.4, 'mw': 174.20, 'flexibility': 0.478, 'helix_propensity': 0.98, 'sheet_propensity': 0.93},
}
DEFAULT_PROPS = {'hydrophobicity': 0, 'charge': 0, 'volume': 100, 'mw': 100,
                 'flexibility': 0.4, 'helix_propensity': 0.8, 'sheet_propensity': 0.8}

# 图很小 (≤48 节点), CPU 足够快且逐位确定 (GPU scatter 非确定, 不利于科研复现);
# 如需 GPU 可设环境变量 M6_DEVICE=cuda.
DEVICE = torch.device('cpu')
import os
if os.environ.get('M6_DEVICE', 'cpu').lower() == 'cuda' and torch.cuda.is_available():
    DEVICE = torch.device('cuda')
torch.manual_seed(42)
np.random.seed(42)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _js(obj):
    if isinstance(obj, dict):
        return {k: _js(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_js(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, np.ndarray):
        return _js(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    return obj


def cos_sim(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return np.nan
    return float(np.dot(u, v) / (nu * nv))


def seq_mean_props(seq):
    if not seq:
        return DEFAULT_PROPS.copy()
    props = [AA_PROPERTIES.get(a, DEFAULT_PROPS) for a in seq]
    return {k: float(np.mean([p[k] for p in props])) for k in DEFAULT_PROPS}


def estimate_k(sdf, geo_cols):
    """与 validation_tests.estimate_k / k_context_estimation 同构:
    Ridge(α=50) per-geometry on StandardScaler(BIO8) → K [n_geo × 8]."""
    X = sdf[BIO_FEATURES].values.astype(float)
    Xs = StandardScaler().fit_transform(X)
    k = np.zeros((len(geo_cols), X.shape[1]))
    for gi, g in enumerate(geo_cols):
        y = sdf[g].values.astype(float)
        ok = np.isfinite(y)
        if ok.sum() < 3 or np.std(y[ok]) < 1e-12:
            continue
        m = Ridge(alpha=ALPHA)
        m.fit(Xs[ok], y[ok])
        k[gi] = m.coef_
    return k


# ---------------------------------------------------------------------------
# Step 1: V2 基线复现 (P5) + 60 对定义
# ---------------------------------------------------------------------------
def reproduce_v2(geom, meta, ctx, geo_cols):
    """精确镜像 criterion_v2: 重算 stratum K → 60 对 → LOAO Ridge/GBRT.
    返回 (pairs, strata_K_recomputed, r2_ridge, ridge_loao_preds)."""
    g5_idx = [geo_cols.index(g) for g in GEO5 if g in geo_cols]
    ctx_idx = ctx.set_index('seq_id')

    strata_K = {}
    for aa in FOCAL_AAS:
        for c in CONTEXTS_A:
            ids = meta[(meta['panel'] == 'A') & (meta['focal_aa'] == aa) &
                       (meta['context'] == c)]['seq_id'].tolist()
            rows = []
            meta_idx = meta.set_index('seq_id')
            for sid in ids:
                g = geom[geom['seq_id'] == sid]
                if len(g) == 0:
                    continue
                m = meta_idx.loc[sid]
                row = {'n': float(m['length'])}
                row.update(seq_mean_props(m['sequence']))
                for gc in geo_cols:
                    row[gc] = g.iloc[0][gc]
                rows.append(row)
            if len(rows) < 3:
                continue
            sdf = pd.DataFrame(rows)
            k = estimate_k(sdf, geo_cols)
            strata_K[(aa, c)] = k[g5_idx].flatten()

    def ctx_vec(aa, c):
        ids = meta[(meta['panel'] == 'A') & (meta['focal_aa'] == aa) &
                   (meta['context'] == c)]['seq_id'].tolist()
        feats = []
        for f in CTX_FEATURES_15:
            vals = [ctx_idx.loc[sid, f] for sid in ids if sid in ctx_idx.index]
            feats.append(float(np.nanmean(vals)) if vals else np.nan)
        return np.array(feats, dtype=float)

    pairs = []
    for aa in sorted(set(a for a, _ in strata_K)):
        ctxs = [c for a2, c in strata_K if a2 == aa]
        for i, c1 in enumerate(ctxs):
            for c2 in ctxs[i + 1:]:
                y = cos_sim(strata_K[(aa, c1)], strata_K[(aa, c2)])
                if not np.isfinite(y):
                    continue
                pairs.append({'aa': aa, 'c1': c1, 'c2': c2, 'y': y,
                              'dx': np.abs(ctx_vec(aa, c1) - ctx_vec(aa, c2))})

    Y = np.array([p['y'] for p in pairs])
    X = np.array([p['dx'] for p in pairs])
    aa_arr = np.array([p['aa'] for p in pairs])
    bad_cols = np.any(~np.isfinite(X), axis=0)
    X = X[:, ~bad_cols]

    preds = np.full(len(Y), np.nan)
    for aa in sorted(set(aa_arr)):
        tr, te = aa_arr != aa, aa_arr == aa
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=ALPHA)
        m.fit(sc.transform(X[tr]), Y[tr])
        preds[te] = m.predict(sc.transform(X[te]))
    ss_res = np.sum((Y - preds) ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r2_ridge = float(1 - ss_res / ss_tot)
    return pairs, strata_K, r2_ridge, preds, [CTX_FEATURES_15[i] for i, b in enumerate(bad_cols) if b]


# ---------------------------------------------------------------------------
# Step 2: 图构建 (Kabsch 对齐 → 节点/边特征)
# ---------------------------------------------------------------------------
def build_graph_for_sequence(seq_dir, max_samples=MAX_SAMPLES):
    """单序列系综 → PyG Data (原始未标准化特征)."""
    files = sorted(Path(seq_dir).glob("batch_*.npz"))
    if not files:
        return None
    pos = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        key = 'pos' if 'pos' in z else 'positions'
        pos.append(z[key])
    pos = np.concatenate(pos, axis=0)[:max_samples].astype(np.float64)
    if len(pos) < 50:
        return None

    aligned = kabsch_align_ensemble(pos, reference_mode='first')  # (T,N,3)
    T, N, _ = aligned.shape
    mean_pos = aligned.mean(axis=0)
    fluct = aligned - mean_pos                                   # (T,N,3)

    # 节点特征 1: RMSF
    d_ts = np.sqrt((fluct ** 2).sum(-1))                         # (T,N) 涨落幅度
    rmsf = d_ts.mean(axis=0)

    # 节点特征 2: 局部弯曲涨落 (相邻残基中点偏差)
    bend = np.zeros((T, N))
    if N >= 3:
        b = aligned[:, 1:-1] - 0.5 * (aligned[:, :-2] + aligned[:, 2:])
        bend[:, 1:-1] = np.sqrt((b ** 2).sum(-1))
    bend[:, 0] = np.sqrt(((aligned[:, 0] - aligned[:, 1]) ** 2).sum(-1))
    bend[:, -1] = np.sqrt(((aligned[:, -1] - aligned[:, -2]) ** 2).sum(-1))
    local_stiff = bend.mean(axis=0)

    # 节点特征 3: 相对序列位置
    rel_pos = np.linspace(0.0, 1.0, N)

    x_feat = np.column_stack([rmsf, local_stiff, rel_pos]).astype(np.float32)

    # 残基间涨落相关 (Pearson on 幅度时间序列)
    C = np.corrcoef(d_ts.T)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(C, 0.0)

    # 系综平均距离矩阵
    diff = aligned[:, :, None, :] - aligned[:, None, :, :]       # (T,N,N,3)
    Dmean = np.sqrt((diff ** 2).sum(-1)).mean(axis=0)            # (N,N)

    # 边: 序列相邻 + top-8 |corr| 非相邻
    edges = set()
    for i in range(N - 1):
        edges.add((i, i + 1))
    cand = [(abs(C[i, j]), i, j) for i in range(N) for j in range(i + 2, N)]
    cand.sort(reverse=True)
    for _, i, j in cand[:8]:
        edges.add((i, j))

    ei, ea = [], []
    for (i, j) in sorted(edges):
        e = [C[i, j], Dmean[i, j], float(abs(i - j))]
        ei.append([i, j]); ea.append(e)
        ei.append([j, i]); ea.append(e)
    edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(np.array(ea), dtype=torch.float32)
    return Data(x=torch.tensor(x_feat), edge_index=edge_index, edge_attr=edge_attr,
                n_res=N, n_samples_used=T)


def build_all_graphs(panel_a_meta):
    """全部 Panel A 序列构图 (带磁盘缓存)."""
    if GRAPH_CACHE.exists():
        log(f"Loading graph cache: {GRAPH_CACHE.name}")
        return torch.load(GRAPH_CACHE, weights_only=False)
    graphs = {}
    t0 = time.time()
    for k, (_, r) in enumerate(panel_a_meta.iterrows()):
        sid = r['seq_id']
        g = build_graph_for_sequence(ENS_DIR / sid)
        if g is None:
            log(f"  WARNING: no ensemble for {sid}, skipped")
            continue
        g.stratum = f"A_{r['focal_aa']}_{r['context']}"
        g.seq_id = sid
        graphs[sid] = g
        if (k + 1) % 20 == 0:
            log(f"  graphs built: {k + 1}/{len(panel_a_meta)} ({time.time() - t0:.0f}s)")
    torch.save(graphs, GRAPH_CACHE)
    log(f"Graph cache saved ({len(graphs)} graphs, {time.time() - t0:.0f}s)")
    return graphs


# ---------------------------------------------------------------------------
# Step 3: GNN 模型
# ---------------------------------------------------------------------------
class EnsembleGNN(nn.Module):
    """3×GINEConv + mean/max pool → 64d 嵌入 → MLP 回归 K_geo5 (40d)."""

    def __init__(self, node_dim=3, edge_dim=3, hidden=64, emb=64, out_dim=40,
                 layers=3, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.input_lin = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(nn=mlp, edge_dim=edge_dim))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.emb_lin = nn.Linear(2 * hidden, emb)
        self.head = nn.Sequential(nn.Linear(emb, emb), nn.ReLU(),
                                  nn.Linear(emb, out_dim))

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.input_lin(x)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        z = self.emb_lin(g)          # 嵌入 (pre-activation, 用于对特征)
        out = self.head(F.relu(z))   # K 回归
        return out, z


def standardize_graphs(graphs, train_sids):
    """用训练图统计量标准化全部图的节点/边特征 (防泄漏: 仅训练集统计)."""
    xs = torch.cat([graphs[s].x for s in train_sids], dim=0)
    eas = torch.cat([graphs[s].edge_attr for s in train_sids], dim=0)
    xm, xstd = xs.mean(0), xs.std(0).clamp(min=1e-8)
    em, estd = eas.mean(0), eas.std(0).clamp(min=1e-8)
    out = {}
    for sid, g in graphs.items():
        ng = Data(x=(g.x - xm) / xstd, edge_index=g.edge_index,
                  edge_attr=(g.edge_attr - em) / estd)
        ng.stratum, ng.seq_id = g.stratum, g.seq_id
        out[sid] = ng
    return out


def train_gnn(model, train_list, val_list, cfg, seed=SEED):
    """训练 + early stop. 返回 (model, train_losses, val_losses)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    train_loader = DataLoader(train_list, batch_size=cfg['batch'], shuffle=True)
    val_loader = DataLoader(val_list, batch_size=cfg['batch'], shuffle=False)
    best_val, best_state, wait = np.inf, None, 0
    train_losses, val_losses = [], []
    for epoch in range(cfg['max_epochs']):
        model.train()
        tl, nb = 0.0, 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            out, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = F.mse_loss(out, batch.y)
            loss.backward()
            opt.step()
            tl += loss.item(); nb += 1
        train_losses.append(tl / max(nb, 1))
        model.eval()
        vl, nb = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                out, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                vl += F.mse_loss(out, batch.y).item(); nb += 1
        val_losses.append(vl / max(nb, 1))
        if epoch + 1 < cfg['min_epochs']:
            continue
        if val_losses[-1] < best_val - 1e-5:
            best_val = val_losses[-1]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg['patience']:
                break
    if best_state is None:
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, train_losses, val_losses, best_val


@torch.no_grad()
def embed_all(model, graphs, sids, batch_size=32):
    model.eval()
    clean = [Data(x=graphs[s].x, edge_index=graphs[s].edge_index,
                  edge_attr=graphs[s].edge_attr) for s in sids]
    loader = DataLoader(clean, batch_size=batch_size, shuffle=False)
    zs = []
    for batch in loader:
        batch = batch.to(DEVICE)
        _, z = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        zs.append(z.cpu().numpy())
    return np.vstack(zs)


# ---------------------------------------------------------------------------
# Step 4: 端到端 LOAO CV (P1 主协议)
# ---------------------------------------------------------------------------
def run_loao(graphs_raw, seq2stratum, stratum_labels, strata_by_aa, cfg, seed=SEED,
             tag='main'):
    """完整 LOAO: 每折训练 GNN → 嵌入 → stratum 均值 → |Δz| → Ridge(α=50).
    stratum_labels: dict stratum → K_geo5 (40,) (或打乱后的标签, 阴性对照).
    返回 dict(preds={(aa,c1,c2): y_pred}, curves, folds)."""
    aas = sorted(strata_by_aa.keys())
    all_sids = sorted(graphs_raw.keys())
    preds, curves, folds = {}, [], []
    for fi, aa_hold in enumerate(aas):
        t0 = time.time()
        train_aas = [a for a in aas if a != aa_hold]
        rng = np.random.RandomState(seed)
        # val: 每训练 AA 随机 1 个 context (25 strata 中 5 个 = 20%, 按 stratum)
        val_strata, train_strata = [], []
        for a in train_aas:
            ctxs = sorted(strata_by_aa[a])
            vctx = ctxs[rng.permutation(len(ctxs))[0]]
            for c in ctxs:
                (val_strata if c == vctx else train_strata).append(f"A_{a}_{c}")
        train_sids = [s for s in all_sids if seq2stratum[s] in train_strata]
        val_sids = [s for s in all_sids if seq2stratum[s] in val_strata]

        # 特征标准化 (仅训练图统计) + K 标签标准化 (仅训练 strata)
        graphs = standardize_graphs(graphs_raw, train_sids)
        k_scaler = StandardScaler().fit(
            np.stack([stratum_labels[s] for s in train_strata]))

        def with_y(sid):
            g = graphs[sid]
            g.y = torch.tensor(
                k_scaler.transform(stratum_labels[g.stratum][None, :]),
                dtype=torch.float32)  # shape (1,40) → collate 后为 (batch,40)
            return g

        model = EnsembleGNN(hidden=cfg['hidden'], emb=cfg['emb'],
                            layers=cfg['layers'], dropout=cfg['dropout']).to(DEVICE)
        model, tr_l, vl_l, best_val = train_gnn(model, [with_y(s) for s in train_sids],
                                                [with_y(s) for s in val_sids], cfg, seed)
        curves.append({'fold_aa': aa_hold, 'train': tr_l, 'val': vl_l})

        # 嵌入全部序列 → stratum 嵌入 = 成员均值
        Z = embed_all(model, graphs, all_sids)
        z_df = pd.DataFrame({'seq_id': all_sids,
                             'stratum': [seq2stratum[s] for s in all_sids]})
        z_df['z'] = list(Z)
        stratum_emb = z_df.groupby('stratum')['z'].apply(
            lambda v: np.mean(np.stack(v.values), axis=0)).to_dict()

        # 对特征 |Δz| → Ridge(α=50) (StandardScaler 仅训练对)
        def dz(a, c1, c2):
            return np.abs(stratum_emb[f"A_{a}_{c1}"] - stratum_emb[f"A_{a}_{c2}"])

        tr_pairs = [(a, c1, c2) for a in train_aas
                    for i, c1 in enumerate(CONTEXTS_A) for c2 in CONTEXTS_A[i + 1:]]
        te_pairs = [(aa_hold, c1, c2) for i, c1 in enumerate(CONTEXTS_A)
                    for c2 in CONTEXTS_A[i + 1:]]
        y_lookup = {}
        for a in aas:
            for i, c1 in enumerate(CONTEXTS_A):
                for c2 in CONTEXTS_A[i + 1:]:
                    y_lookup[(a, c1, c2)] = cos_sim(stratum_labels[f"A_{a}_{c1}"],
                                                    stratum_labels[f"A_{a}_{c2}"])
        Xtr = np.stack([dz(*p) for p in tr_pairs])
        ytr = np.array([y_lookup[p] for p in tr_pairs])
        Xte = np.stack([dz(*p) for p in te_pairs])
        sc = StandardScaler().fit(Xtr)
        ridge = Ridge(alpha=ALPHA)
        ridge.fit(sc.transform(Xtr), ytr)
        pte = ridge.predict(sc.transform(Xte))
        for p, yp in zip(te_pairs, pte):
            preds[p] = float(yp)
        folds.append({'held_out_aa': aa_hold, 'epochs': len(tr_l),
                      'best_val_mse': float(best_val),
                      'n_train_seqs': len(train_sids), 'n_val_seqs': len(val_sids),
                      'fold_r2_on_heldout_pairs': float(
                          1 - np.sum((np.array([y_lookup[p] for p in te_pairs]) - pte) ** 2) /
                          np.sum((np.array([y_lookup[p] for p in te_pairs]) -
                                  np.mean([y_lookup[p] for p in te_pairs])) ** 2))})
        log(f"  [{tag}] fold {fi + 1}/6 (hold {aa_hold}): {len(tr_l)} epochs, "
            f"best_val={best_val:.4f}, {time.time() - t0:.0f}s")
    return {'preds': preds, 'curves': curves, 'folds': folds}


def oos_r2(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Step 5: 图件
# ---------------------------------------------------------------------------
def make_fig1(pairs_df, r2_gnn, r2_v2, r2_null):
    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=300)
    aas = sorted(pairs_df['aa'].unique())
    cmap = dict(zip(aas, plt.cm.tab10(np.linspace(0, 1, len(aas)))))
    for aa in aas:
        d = pairs_df[pairs_df['aa'] == aa]
        ax.scatter(d['y_true'], d['y_pred_gnn'], s=42, color=cmap[aa],
                   edgecolor='k', linewidth=0.4, label=f'GNN (AA={aa})', zorder=3)
    ax.scatter(pairs_df['y_true'], pairs_df['y_pred_v2ridge'], s=30, marker='x',
               color='gray', alpha=0.75, label='V2 Ridge (context feats)', zorder=2)
    lim = [min(pairs_df['y_true'].min(), pairs_df['y_pred_gnn'].min()) - 0.08,
           max(pairs_df['y_true'].max(), pairs_df['y_pred_gnn'].max()) + 0.08]
    ax.plot(lim, lim, 'k--', lw=1, alpha=0.6)
    txt = (f"GNN ensemble-repr OOS $R^2$ = {r2_gnn:.3f}\n"
           f"V2 Ridge $R^2$ = {r2_v2:.3f}   GBRT = {V2_GBRT_R2:.3f}\n"
           f"single-AA baseline = {SINGLE_AA_BASELINE:.3f}\n"
           f"null (shuffled K) $R^2$ = {r2_null:.3f}\n"
           f"SUCCESS threshold: $R^2$ > {SUCCESS_THRESHOLD}")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va='top', ha='left',
            fontsize=9, bbox=dict(boxstyle='round', fc='white', ec='0.6', alpha=0.9))
    ax.set_xlabel('Observed stratum-K pair cos-similarity (GEO5 x BIO8)')
    ax.set_ylabel('LOAO-CV predicted cos-similarity')
    ax.set_title('Phase M6 — GNN on ensemble graphs vs V2 sequence-level baseline\n'
                 '(60 Panel A context pairs, leave-one-AA-out CV)')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ['svg', 'jpg']:
        fig.savefig(FIG_DIR / f"phase_m6_fig1.{ext}", dpi=300)
    plt.close(fig)
    # plotly html
    try:
        import plotly.graph_objects as go
        fg = go.Figure()
        for aa in aas:
            d = pairs_df[pairs_df['aa'] == aa]
            fg.add_trace(go.Scatter(
                x=d['y_true'], y=d['y_pred_gnn'], mode='markers', name=f'GNN AA={aa}',
                text=[f"{r.c1}-{r.c2}" for r in d.itertuples()],
                marker=dict(size=9, line=dict(width=0.6, color='black'))))
        fg.add_trace(go.Scatter(x=pairs_df['y_true'], y=pairs_df['y_pred_v2ridge'],
                                mode='markers', name='V2 Ridge',
                                marker=dict(size=7, symbol='x', color='gray')))
        fg.add_trace(go.Scatter(x=lim, y=lim, mode='lines', name='y=x',
                                line=dict(dash='dash', color='black')))
        fg.update_layout(
            title=(f"Phase M6 GNN LOAO — OOS R2={r2_gnn:.3f} | V2 Ridge={r2_v2:.3f} | "
                   f"null={r2_null:.3f} | threshold>0.3"),
            xaxis_title='Observed K cos-similarity',
            yaxis_title='Predicted cos-similarity', width=860, height=720)
        fg.write_html(str(FIG_DIR / "phase_m6_fig1.html"))
    except Exception as e:
        log(f"  plotly fig1 skipped: {e}")


def make_fig2(curves, r2_gnn, r2_null, r2_v2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=300)
    # (a) 训练曲线 (6折均值±std)
    maxlen = max(len(c['train']) for c in curves)
    tr = np.full((len(curves), maxlen), np.nan)
    vl = np.full((len(curves), maxlen), np.nan)
    for i, c in enumerate(curves):
        tr[i, :len(c['train'])] = c['train']
        vl[i, :len(c['val'])] = c['val']
    ep = np.arange(1, maxlen + 1)
    ax1.plot(ep, np.nanmean(tr, 0), label='train MSE (mean over folds)', color='tab:blue')
    ax1.fill_between(ep, np.nanmin(tr, 0), np.nanmax(tr, 0), color='tab:blue', alpha=0.15)
    ax1.plot(ep, np.nanmean(vl, 0), label='val MSE (mean over folds)', color='tab:red')
    ax1.fill_between(ep, np.nanmin(vl, 0), np.nanmax(vl, 0), color='tab:red', alpha=0.15)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('MSE (standardized K)')
    ax1.set_title('(a) GNN training curves — 6 LOAO folds (main config)')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)
    # (b) 对照柱状图
    names = ['GNN\nensemble repr.', 'null\n(shuffled K)', 'V2 Ridge\n(context feats)',
             'V2 GBRT\n(context feats)', 'single-AA\nphysicochem.']
    vals = [r2_gnn, r2_null, r2_v2, V2_GBRT_R2, SINGLE_AA_BASELINE]
    colors = ['tab:blue', 'tab:gray', 'tab:orange', 'tab:orange', 'tab:orange']
    bars = ax2.bar(names, vals, color=colors, edgecolor='k', linewidth=0.5)
    ax2.axhline(SUCCESS_THRESHOLD, color='green', ls='--', lw=1.5,
                label=f'SUCCESS threshold R2={SUCCESS_THRESHOLD}')
    ax2.axhline(0, color='k', lw=0.8)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + (0.02 if v >= 0 else -0.06),
                 f"{v:.3f}", ha='center', fontsize=9)
    ax2.set_ylabel('LOAO-CV OOS R2')
    ax2.set_title('(b) M6 GNN vs pre-registered baselines & null control')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    for ext in ['svg', 'jpg']:
        fig.savefig(FIG_DIR / f"phase_m6_fig2.{ext}", dpi=300)
    plt.close(fig)
    try:
        import plotly.graph_objects as go
        pcolors = ['#1f77b4', '#7f7f7f', '#ff7f0e', '#ff7f0e', '#ff7f0e']
        fg = go.Figure()
        fg.add_trace(go.Bar(x=names, y=vals, marker_color=pcolors,
                            text=[f"{v:.3f}" for v in vals], textposition='outside'))
        fg.add_hline(y=SUCCESS_THRESHOLD, line_dash='dash', line_color='green',
                     annotation_text='SUCCESS threshold 0.3')
        fg.update_layout(title='Phase M6 — GNN vs baselines & null (LOAO OOS R2)',
                         yaxis_title='OOS R2', width=900, height=560)
        fg.write_html(str(FIG_DIR / "phase_m6_fig2.html"))
    except Exception as e:
        log(f"  plotly fig2 skipped: {e}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    log("=" * 72)
    log("Phase M6: Ensemble-level K Representation Learning (GNN on ensemble graphs)")
    log(f"Device: {DEVICE} | seed={SEED} | MAX_SAMPLES={MAX_SAMPLES}")
    log("Pre-registered criteria P1-P6 fixed in docstring BEFORE computation.")

    # ---- 数据加载 ----
    geom = pd.read_csv(GEOM_CSV)
    meta = pd.read_csv(META_CSV)
    ctx = pd.read_csv(CTX_CSV)
    knpz = np.load(K_NPZ, allow_pickle=True)
    geo_cols = [c for c in Y_COLUMNS if c in geom.columns]
    log(f"Loaded: geom {geom.shape}, meta {meta.shape}, ctx {ctx.shape}, "
        f"K strata {knpz['strata'].shape}")

    # ---- P5: V2 基线复现 ----
    log("-" * 72)
    log("[P5] Reproducing V2 Ridge baseline (mirroring criterion_v2) ...")
    pairs, strata_K_re, r2_v2, v2_preds, dropped = reproduce_v2(geom, meta, ctx, geo_cols)
    log(f"  pairs={len(pairs)}, dropped NA features={dropped}")
    log(f"  V2 Ridge LOAO R2 reproduced = {r2_v2:.4f} (official -0.1467, tol +/-0.1)")
    v2_ok = abs(r2_v2 - V2_REFERENCE_R2) <= 0.1
    log(f"  reproduction within tolerance: {v2_ok}")

    # ---- K 一致性核对: NPZ K_geo5 vs estimate_k 重算 ----
    strata_names = list(knpz['strata'])
    K_geo5 = knpz['K_geo5']
    npz_idx = {s: i for i, s in enumerate(strata_names)}
    cos_list = []
    strata_K_npz = {}
    for (aa, c), kvec in strata_K_re.items():
        sname = f"A_{aa}_{c}"
        k_npz = K_geo5[npz_idx[sname]]
        cos_list.append(cos_sim(kvec, k_npz))
        strata_K_npz[(aa, c)] = k_npz
    log(f"  K consistency (NPZ vs re-estimated) mean cos-sim = {np.mean(cos_list):.6f} "
        f"(min {np.min(cos_list):.6f}, 30 strata)")

    # ---- 图构建 ----
    log("-" * 72)
    log("[M6] Building ensemble graphs for Panel A (120 sequences) ...")
    panel_a = meta[meta['panel'] == 'A'].reset_index(drop=True)
    graphs_raw = build_all_graphs(panel_a)
    seq2stratum = {r['seq_id']: f"A_{r['focal_aa']}_{r['context']}"
                   for _, r in panel_a.iterrows() if r['seq_id'] in graphs_raw}
    strata_by_aa = {aa: CONTEXTS_A[:] for aa in FOCAL_AAS}
    n_nodes = [int(g.n_res) for g in graphs_raw.values()]
    n_edges = [g.edge_index.shape[1] // 2 for g in graphs_raw.values()]
    log(f"  graphs={len(graphs_raw)}, nodes/graph {min(n_nodes)}-{max(n_nodes)}, "
        f"undirected edges/graph {min(n_edges)}-{max(n_edges)}")

    stratum_labels = {f"A_{aa}_{c}": strata_K_npz[(aa, c)]
                      for aa in FOCAL_AAS for c in CONTEXTS_A}

    # ---- P1 主配置 LOAO ----
    log("-" * 72)
    log(f"[P1] Main-config end-to-end LOAO CV: {MAIN_CONFIG}")
    main_res = run_loao(graphs_raw, seq2stratum, stratum_labels, strata_by_aa,
                        MAIN_CONFIG, tag='main')
    Y = np.array([p['y'] for p in pairs])
    pair_keys = [(p['aa'], p['c1'], p['c2']) for p in pairs]
    gnn_preds = np.array([main_res['preds'][k] for k in pair_keys])
    r2_gnn = oos_r2(Y, gnn_preds)
    log(f"  >>> M6 GNN OOS R2 (main config) = {r2_gnn:.4f}")

    # ---- P4 阴性对照 ----
    log("-" * 72)
    log(f"[P4] Null control: shuffle stratum K labels (seed={NULL_SEED}) ...")
    keys_sorted = sorted(stratum_labels.keys())
    perm = np.random.RandomState(NULL_SEED).permutation(len(keys_sorted))
    shuffled = {keys_sorted[i]: stratum_labels[keys_sorted[perm[i]]]
                for i in range(len(keys_sorted))}
    null_res = run_loao(graphs_raw, seq2stratum, shuffled, strata_by_aa,
                        MAIN_CONFIG, tag='null')
    null_preds = np.array([null_res['preds'][k] for k in pair_keys])
    r2_null = oos_r2(Y, null_preds)
    log(f"  >>> null-control OOS R2 = {r2_null:.4f} (expected ~0 or negative)")

    # ---- P6 有限架构调整 (仅当主配置 R2 ∈ [0.2, 0.35]) ----
    tuning_attempts = []
    if TUNE_BAND[0] <= r2_gnn <= TUNE_BAND[1]:
        log("-" * 72)
        log(f"[P6] Main R2 in tuning band {TUNE_BAND} — running registered grid ...")
        for ti, tc in enumerate(TUNING_CONFIGS):
            cfg = {**MAIN_CONFIG, **tc}
            log(f"  tuning {ti + 1}/{len(TUNING_CONFIGS)}: {tc}")
            res = run_loao(graphs_raw, seq2stratum, stratum_labels, strata_by_aa,
                           cfg, tag=f'tune{ti + 1}')
            r2_t = oos_r2(Y, np.array([res['preds'][k] for k in pair_keys]))
            tuning_attempts.append({'config': tc, 'oos_r2': r2_t})
            log(f"  tuning {ti + 1} OOS R2 = {r2_t:.4f}")
    else:
        log(f"[P6] Main R2={r2_gnn:.4f} outside tuning band {TUNE_BAND} — no tuning.")

    # ---- 判定 (只用主配置) ----
    verdict = 'SUCCESS' if r2_gnn > SUCCESS_THRESHOLD else 'H0_STRENGTHENED'
    log("=" * 72)
    log(f"VERDICT: {verdict}  (main-config OOS R2={r2_gnn:.4f} vs threshold "
        f">{SUCCESS_THRESHOLD})")

    # ---- 输出: pairs CSV ----
    pairs_df = pd.DataFrame({
        'aa': [p['aa'] for p in pairs],
        'c1': [p['c1'] for p in pairs],
        'c2': [p['c2'] for p in pairs],
        'y_true': Y,
        'y_pred_gnn': gnn_preds,
        'y_pred_v2ridge': v2_preds,
    })
    pairs_df.to_csv(OUT_DIR / "m6_pairs.csv", index=False)
    log(f"Saved: {OUT_DIR / 'm6_pairs.csv'} ({len(pairs_df)} pairs)")

    # ---- 输出: 图件 ----
    make_fig1(pairs_df, r2_gnn, r2_v2, r2_null)
    make_fig2(main_res['curves'], r2_gnn, r2_null, r2_v2)
    log(f"Saved figures: {FIG_DIR}/phase_m6_fig1/fig2 (svg+jpg+html)")

    # ---- 输出: report JSON ----
    report = {
        'stage': 'Phase M6 — Ensemble-level K Representation Learning (GNN on ensemble graphs)',
        'generated_at': datetime.now().isoformat(),
        'runtime_min': (time.time() - t_start) / 60,
        'device': str(DEVICE),
        'preregistered_criteria': {
            'P1_primary': ('LOAO CV identical to V2 on 60 Panel A context pairs; '
                           'y=cos_sim(K_geo5); GNN ensemble embedding |dz| -> Ridge(a=50); '
                           'OOS R2 > 0.3 -> SUCCESS'),
            'P2_failure': ('OOS R2 <= 0.3 -> H0_STRENGTHENED: K not compressible even by '
                           'ensemble-level deep representation (theoretical finding)'),
            'P3_baselines': {'v2_ridge': V2_REFERENCE_R2, 'v2_gbrt': V2_GBRT_R2,
                             'single_aa': SINGLE_AA_BASELINE},
            'P4_null_control': ('shuffle stratum K labels (seed=123), rerun full LOAO once; '
                                'R2 expected ~0 or negative'),
            'P5_reproduction': ('reproduce V2 Ridge LOAO R2 ~ -0.147 (tol +/-0.1) before GNN'),
            'P6_tuning_registry': ('architecture tuning only if main R2 in [0.2,0.35]; all '
                                   'attempts registered; verdict uses MAIN config only'),
        },
        'data': {
            'n_panel_a_sequences': len(graphs_raw),
            'n_strata': len(stratum_labels),
            'n_pairs': len(pairs),
            'max_samples_per_seq': MAX_SAMPLES,
            'nodes_per_graph_range': [min(n_nodes), max(n_nodes)],
            'undirected_edges_per_graph_range': [min(n_edges), max(n_edges)],
            'k_source': 'K_matrices_context.npz K_geo5 (40d), consistency-checked vs estimate_k',
        },
        'k_consistency_npz_vs_reestimated': {
            'mean_cos_sim': float(np.mean(cos_list)),
            'min_cos_sim': float(np.min(cos_list)),
        },
        'v2_baseline_reproduction': {
            'r2_ridge_reproduced': r2_v2,
            'r2_ridge_official': -0.1467,
            'tolerance': 0.1,
            'within_tolerance': bool(v2_ok),
            'n_pairs': len(pairs),
            'dropped_na_features': dropped,
            'y_mean': float(np.mean(Y)), 'y_std': float(np.std(Y)),
        },
        'm6_main_config': MAIN_CONFIG,
        'm6_gnn_oos_r2': r2_gnn,
        'null_control_r2': r2_null,
        'per_fold': main_res['folds'],
        'null_per_fold': null_res['folds'],
        'tuning_attempts': tuning_attempts,
        'verdict': verdict,
        'verdict_basis': 'MAIN config only (P6); tuning results exploratory',
        'outputs': {
            'pairs_csv': str(OUT_DIR / 'm6_pairs.csv'),
            'figures': [str(FIG_DIR / f'phase_m6_fig{i}.{e}')
                        for i in (1, 2) for e in ('svg', 'jpg', 'html')],
            'graph_cache': str(GRAPH_CACHE),
        },
    }
    with open(OUT_DIR / "phase_m6_report.json", 'w', encoding='utf-8') as f:
        json.dump(_js(report), f, indent=2, ensure_ascii=False)
    log(f"Saved: {OUT_DIR / 'phase_m6_report.json'}")
    log(f"Total runtime: {(time.time() - t_start) / 60:.1f} min")
    log("=" * 72)
    log("KEY NUMBERS:")
    log(f"  V2 Ridge reproduction R2 = {r2_v2:.4f} (official -0.1467)")
    log(f"  M6 GNN OOS R2 (main)     = {r2_gnn:.4f}")
    log(f"  Null control R2          = {r2_null:.4f}")
    log(f"  VERDICT                  = {verdict}")


if __name__ == "__main__":
    main()
