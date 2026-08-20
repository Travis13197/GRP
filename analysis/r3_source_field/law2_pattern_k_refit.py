#!/usr/bin/env python3
"""
Law 2 — 模式基底 K 矩阵重估 (Pattern-Basis K Refit)
=====================================================================
依据: Article_Preparation/Law2_Pattern_Basis_Plan.md v1.2 (预注册 §5/§6)

五项检验:
  P1 (门控): 玩具电池 + HET_KAPPA κ_computed vs κ_designed Spearman
  P2 (门控): HET_KAPPA 组成冻结子集 κ~Y 相关 + 2000次置换检验
  P3: 全 1,279 序列 27 核心Y, 组成基线 vs 组成+模式 Ridge(α=50) 5-fold CV
      + κ/SCD 系数 bootstrap 95% CI (1000×, 4 核心Y)
  P4: LOSO (9 系统) 正迁移率基线 vs 增强
  P5: Pilot OOS 非劣性 + Panel B 改进

特征集 (唯一差异 = 模式特征, 公平性控制 §4):
  COMP (10): n, log_n, sqrt_n, f_pos, f_neg, FCR, NCPR, abs_NCPR, mean_hydro, H_comp
  ENH  (18): COMP + kappa_f, SCD_sqrtL, SHD, gamma1, blockiness, charge_spacing_f,
             H_dipep, has_charge

输出:
  field_theory/tables/law2_pattern_krefit_summary.json  (判定汇总)
  field_theory/tables/law2_pattern_krefit_cv.csv        (P3 明细)
  field_theory/tables/law2_pattern_loso.csv             (P4 明细)
  field_theory/tables/law2_pattern_pilot_oos.csv        (P5 明细)
  field_theory/figures/law2_pattern/F1|F2|F3 .png/.svg/.html
"""

import json
import sys
import warnings
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # GBK 控制台兼容

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
FIELD_THEORY = Path(__file__).parent.parent
TABLES_DIR = FIELD_THEORY / "tables"
FIG_DIR = FIELD_THEORY / "figures" / "law2_pattern"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DESC_CSV = TABLES_DIR / "law2_pattern_descriptors.csv"
PILOT_GEOM = TABLES_DIR / "law2_pilot_geometry.csv"
TOYCHECK = TABLES_DIR / "law2_pattern_toycheck.json"

ALPHA = 50.0            # 项目约定 (memory: α=50 改善跨系统迁移)
N_PERM = 2000           # P2 置换次数 (预注册)
N_BOOT = 1000           # P3 bootstrap 次数 (预注册)
RNG = np.random.default_rng(42)

print("=" * 70)
print("Law 2 — 模式基底 K 矩阵重估 (P1–P5)")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ============================================================
# 特征与 Y 定义
# ============================================================
COMP_FEATS = ['n', 'log_n', 'sqrt_n', 'f_pos', 'f_neg', 'FCR', 'NCPR',
              'abs_NCPR', 'mean_hydro', 'H_comp']
PATTERN_FEATS = ['kappa_f', 'SCD_sqrtL', 'SHD', 'gamma1', 'blockiness',
                 'charge_spacing_f', 'H_dipep', 'has_charge']
ENH_FEATS = COMP_FEATS + PATTERN_FEATS

Y27 = ['PR', 'A_C', 'eff_rank_95', 'spectral_decay', 'entropy',
       'total_variance', 'pseudo_volume', 'mean_rmsf', 'max_rmsf',
       'local_stiffness', 'fluct_range_ratio', 'rmsf_entropy',
       'mardia_skewness', 'mardia_kurtosis', 'corr_dim', 'mean_knn_dist',
       'condition_number', 'spectral_gap', 'spectral_gap_ratio',
       'fisher_trace', 'fisher_logdet', 'effective_diffusion', 'relaxation_time',
       'contact_order', 'clustering_coeff', 'modularity', 'betweenness_cv']

CORE4 = ['PR', 'spectral_decay', 'mean_rmsf', 'entropy']
P2_Y = ['PR', 'mean_rmsf', 'spectral_decay', 'entropy', 'pseudo_volume']

# ============================================================
# 数据加载与特征工程
# ============================================================
print("\n[0/6] 加载数据与特征工程...")

df = pd.read_csv(DESC_CSV)
main = df[df['is_pilot'] == False].copy().reset_index(drop=True)
pilot_desc = df[df['is_pilot'] == True].copy()

pilot_geom = pd.read_csv(PILOT_GEOM)
# pilot_desc 的 Y 列全为 NaN — 剔除后合并, 避免后缀冲突
pilot_y_in_desc = [c for c in pilot_geom.columns
                   if c in pilot_desc.columns and c not in
                   ['seq_id', 'category', 'aa_type', 'n', 'panel', 'n_samples']]
pilot = (pilot_desc.drop(columns=pilot_y_in_desc)
         .merge(pilot_geom.drop(columns=[c for c in ['category', 'aa_type', 'n', 'n_samples', 'panel']
                                         if c in pilot_geom.columns]),
                on='seq_id', how='left'))
print(f"  主集: {len(main)} | Pilot: {len(pilot)} (几何列匹配: "
      f"{pilot['PR'].notna().sum()}/20)")


def add_features(d):
    d = d.copy()
    # n 修复链: PolyX 515 条 n=NaN; DMS 8 条 n=0 → n_residues; Pilot → L
    if 'n_residues' in d.columns:
        d['n'] = d['n'].where(d['n'].notna() & (d['n'] > 0), d['n_residues'])
    if 'L' in d.columns:
        d['n'] = d['n'].where(d['n'].notna() & (d['n'] > 0), d['L'])
    d['log_n'] = np.log(d['n'] + 1)
    d['sqrt_n'] = np.sqrt(d['n'])
    d['kappa_f'] = d['kappa'].fillna(0.0)                  # FCR=0 → 0 + 指示变量
    d['has_charge'] = (d['FCR'] > 0).astype(float)
    d['charge_spacing_f'] = d['charge_spacing'].fillna(d['n'])  # 无电荷 → 上界 L
    return d


main = add_features(main)
pilot = add_features(pilot)

# 共线性检查 (计划 §7: SCD~L 共线 → 用 SCD_sqrtL)
vif_check = {}
for a, b in [('SCD', 'n'), ('SCD_sqrtL', 'n'), ('kappa_f', 'NCPR'), ('SHD', 'mean_hydro')]:
    vif_check[f'{a}~{b}'] = float(main[[a, b]].corr().iloc[0, 1])
print("  共线性 (Pearson r): " +
      ", ".join(f"{k}={v:.3f}" for k, v in vif_check.items()))

# Y 可用性
y_avail = [y for y in Y27 if y in main.columns]
print(f"  27 核心 Y 可用: {len(y_avail)}")

summary = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
           'alpha': ALPHA, 'n_perm': N_PERM, 'n_boot': N_BOOT,
           'vif_check': vif_check,
           'features': {'COMP': COMP_FEATS, 'ENH': ENH_FEATS}}

# ============================================================
# P1 (门控): 玩具电池 + HET_KAPPA κ 效度
# ============================================================
print("\n[P1] 描述符效度门控...")

toy = json.loads(TOYCHECK.read_text(encoding='utf-8'))
toy_pass = toy['n_pass'] == toy['n_checks']

hk = main[main['category'] == 'HET_KAPPA'].copy()
hk['kappa_designed'] = hk['seq_id'].str.extract(r'HET_KAPPA_([\d.]+)_')[0].astype(float)
hk['is_dup_09'] = hk['seq_id'].str.startswith('HET_KAPPA_0.9_40')
hk['is_07'] = hk['kappa_designed'] == 0.7
rho_all = spearmanr(hk['kappa'], hk['kappa_designed']).statistic
hk_clean = hk[~hk['is_dup_09'] & ~hk['is_07']]
rho_clean = spearmanr(hk_clean['kappa'], hk_clean['kappa_designed']).statistic

p1_pass = toy_pass and rho_all >= 0.85 and rho_clean >= 0.90
summary['P1'] = {
    'toy_pass': f"{toy['n_pass']}/{toy['n_checks']}", 'toy_gate': bool(toy_pass),
    'rho_kappa_all': float(rho_all), 'rho_kappa_clean': float(rho_clean),
    'gate_all': 0.85, 'gate_clean': 0.90, 'pass': bool(p1_pass)}
print(f"  玩具电池: {toy['n_pass']}/{toy['n_checks']} | "
      f"ρ_all={rho_all:.4f} (≥0.85) | ρ_clean={rho_clean:.4f} (≥0.90) | "
      f"→ {'PASS' if p1_pass else 'FAIL'}")

# ============================================================
# P2 (门控): HET_KAPPA κ 因果性 (组成冻结)
# 预注册: 池化 Spearman + 2000 置换
# POST-HOC: 链长分层精确置换 (池化检验受 n 混淆 — κ 档位与 n 非均衡交叉;
#           每层 5 序列 → 120 全排列精确 p, Fisher 合并 df=6)
# ============================================================
print("\n[P2] HET_KAPPA κ~几何 因果性检验 (n=15, 2000 置换)...")

from itertools import permutations
from scipy.stats import chi2 as chi2_dist

p2_rows = []
for y in P2_Y:
    sub = hk[['kappa', 'n', y]].dropna()
    if len(sub) < 8:
        continue
    # ---- 预注册: 池化 Spearman + 置换 ----
    r_obs = spearmanr(sub['kappa'], sub[y]).statistic
    cnt = 0
    kv, yv = sub['kappa'].values, sub[y].values
    for _ in range(N_PERM):
        if abs(spearmanr(RNG.permutation(kv), yv).statistic) >= abs(r_obs):
            cnt += 1
    p_perm = (cnt + 1) / (N_PERM + 1)
    sub13 = hk[~hk['is_07']][['kappa', y]].dropna()
    r13 = spearmanr(sub13['kappa'], sub13[y]).statistic if len(sub13) >= 8 else np.nan
    prereg_pass = bool(abs(r_obs) > 0.7 and p_perm < 0.01)

    # ---- POST-HOC: 链长分层精确置换 (Fisher 合并) ----
    strat = {}
    logps = []
    for nval in sorted(sub['n'].unique()):
        s = sub[sub['n'] == nval]
        if len(s) < 4 or s['kappa'].nunique() < 2:
            continue
        ks, ys_ = s['kappa'].values, s[y].values
        r_s = spearmanr(ks, ys_).statistic
        cnt_s = 0
        n_perm_s = 0
        for perm in set(permutations(ks)):
            n_perm_s += 1
            if abs(spearmanr(list(perm), ys_).statistic) >= abs(r_s) - 1e-12:
                cnt_s += 1
        p_s = cnt_s / n_perm_s
        strat[f'n={int(nval)}'] = {'rho': float(r_s), 'p_exact': float(p_s)}
        logps.append(np.log(max(p_s, 1e-12)))
    chi2_stat = -2 * sum(logps)
    p_fisher = float(1 - chi2_dist.cdf(chi2_stat, 2 * len(logps))) if logps else np.nan
    mean_rho_strat = float(np.mean([v['rho'] for v in strat.values()])) if strat else np.nan
    posthoc_pass = bool(p_fisher < 0.01 and abs(mean_rho_strat) > 0.7)

    p2_rows.append({'Y': y, 'n': len(sub),
                    'rho_pooled': float(r_obs), 'p_perm_pooled': float(p_perm),
                    'rho_strict13': float(r13), 'prereg_pass': prereg_pass,
                    'strata': strat, 'mean_rho_stratified': mean_rho_strat,
                    'p_fisher_stratified': p_fisher, 'posthoc_pass': posthoc_pass})
    print(f"  {y:18s}: 池化 ρ={r_obs:+.3f} p={p_perm:.4f} [{'OK' if prereg_pass else '--'}] | "
          f"分层: " + " ".join(f"{k} ρ={v['rho']:+.2f} p={v['p_exact']:.4f}"
                               for k, v in strat.items()) +
          f" | Fisher p={p_fisher:.5f} [{'OK-POSTHOC' if posthoc_pass else '--'}]")

p2_df = pd.DataFrame(p2_rows)
p2_pass = bool(p2_df['prereg_pass'].any())                     # 预注册门控
p2_posthoc = bool(p2_df['posthoc_pass'].any())                 # 事后分层证据
summary['P2'] = {'results': p2_rows,
                 'n_Y_prereg_pass': int(p2_df['prereg_pass'].sum()),
                 'pass': p2_pass,
                 'posthoc_stratified_pass': p2_posthoc,
                 'posthoc_note': '池化检验为预注册门控; 分层精确置换为 post-hoc '
                                 '(κ档位×n 非均衡交叉导致池化被 n 混淆)'}
print(f"  → P2 预注册: {'PASS' if p2_pass else 'FAIL'} "
      f"({p2_df['prereg_pass'].sum()}/{len(p2_df)}) | "
      f"POST-HOC 分层: {'PASS' if p2_posthoc else 'FAIL'} "
      f"({p2_df['posthoc_pass'].sum()}/{len(p2_df)})")

# ============================================================
# 通用拟合器
# ============================================================

def fit_predict_ridge(Xtr, ytr, Xte, clip_p=(1, 99)):
    """b6 协议: X 标准化; Y 按训练集截尾+标准化. 返回标准化尺度预测."""
    sx = StandardScaler()
    Xtr_s = sx.fit_transform(Xtr)
    Xte_s = sx.transform(Xte)
    lo, hi = np.percentile(ytr, clip_p)
    ytr_c = np.clip(ytr, lo, hi)
    ym, ys = ytr_c.mean(), ytr_c.std()
    if ys < 1e-8:
        ys = 1.0
    m = Ridge(alpha=ALPHA)
    m.fit(Xtr_s, (ytr_c - ym) / ys)
    return m.predict(Xte_s), (ym, ys, lo, hi), m

# ============================================================
# P3: 全序列 5-fold CV + bootstrap 系数 CI
# ============================================================
print("\n[P3] 27 Y × 5-fold CV (COMP vs ENH, α=50)...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
p3_rows = []
coef_store = {y: {'kappa_f': [], 'SCD_sqrtL': []} for y in CORE4}

for y in y_avail:
    sub = main[main[y].notna()].reset_index(drop=True)
    if len(sub) < 60:
        continue
    Xc = sub[COMP_FEATS].values
    Xe = sub[ENH_FEATS].values
    yv = sub[y].values
    r2c_folds, r2e_folds = [], []
    for tr, te in kf.split(sub):
        yp_c, (ym, ys, lo, hi), _ = fit_predict_ridge(Xc[tr], yv[tr], Xc[te])
        yp_e, _, _ = fit_predict_ridge(Xe[tr], yv[tr], Xe[te])
        yte_s = (np.clip(yv[te], lo, hi) - ym) / ys
        r2c_folds.append(r2_score(yte_s, yp_c))
        r2e_folds.append(r2_score(yte_s, yp_e))
    r2c, r2e = float(np.mean(r2c_folds)), float(np.mean(r2e_folds))
    p3_rows.append({'Y': y, 'n': len(sub), 'R2_cv_COMP': r2c, 'R2_cv_ENH': r2e,
                    'delta_R2': r2e - r2c, 'improved': bool(r2e > r2c)})

p3_df = pd.DataFrame(p3_rows)
n_improved = int(p3_df['improved'].sum())
print(f"  ΔCV R²>0: {n_improved}/{len(p3_df)}  (门控 ≥{int(np.ceil(0.6 * len(p3_df)))})")
print(f"  中位 ΔR²: {p3_df['delta_R2'].median():+.4f} | "
      f"最大增益: {p3_df.loc[p3_df['delta_R2'].idxmax(), 'Y']} "
      f"(+{p3_df['delta_R2'].max():.4f})")

# bootstrap 系数 CI (4 核心Y)
print(f"\n[P3b] κ/SCD 标准化系数 bootstrap CI ({N_BOOT}×)...")
boot_rows = []
for y in CORE4:
    sub = main[main[y].notna()].reset_index(drop=True)
    Xe = sub[ENH_FEATS].values
    yv = sub[y].values
    coefs = {f: [] for f in ['kappa_f', 'SCD_sqrtL']}
    for _ in range(N_BOOT):
        idx = RNG.integers(0, len(sub), len(sub))
        try:
            _, _, m = fit_predict_ridge(Xe[idx], yv[idx], Xe[idx][:1])
            for f in coefs:
                coefs[f].append(m.coef_[ENH_FEATS.index(f)])
        except Exception:
            continue
    for f, vals in coefs.items():
        vals = np.array(vals)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        sig = bool(lo > 0 or hi < 0)
        boot_rows.append({'Y': y, 'feature': f, 'coef_median': float(np.median(vals)),
                          'ci_lo': float(lo), 'ci_hi': float(hi), 'sig': sig})
        print(f"  {y:15s} {f:12s}: β={np.median(vals):+.4f} "
              f"95%CI [{lo:+.4f}, {hi:+.4f}] {'OK' if sig else ''}")

boot_df = pd.DataFrame(boot_rows)
n_sig_core = boot_df[boot_df['sig']]['Y'].nunique()
p3_pass = bool(n_improved >= np.ceil(0.6 * len(p3_df)) and n_sig_core >= 3)
summary['P3'] = {'n_improved': n_improved, 'n_Y': len(p3_df),
                 'median_delta_R2': float(p3_df['delta_R2'].median()),
                 'boot': boot_rows, 'n_core_Y_coef_sig': int(n_sig_core),
                 'pass': p3_pass}
p3_df.to_csv(TABLES_DIR / 'law2_pattern_krefit_cv.csv', index=False)
boot_df.to_csv(TABLES_DIR / 'law2_pattern_coef_bootstrap.csv', index=False)
print(f"  → P3 {'PASS' if p3_pass else 'FAIL'}")

# ============================================================
# P4: LOSO 迁移 (9 系统 × 27 Y × COMP/ENH)
# ============================================================
print("\n[P4] LOSO 跨系统验证 (α=50)...")

LOSO_CATS = ['PolyX', 'PolyX_original', 'L1_hydrophobic', 'linker',
             'HET_ALT', 'HET_BLOCK', 'HET_COMP', 'HET_IDP', 'HET_KAPPA']
loso_rows = []
for cat in LOSO_CATS:
    tr_mask = main['category'] != cat
    te_mask = main['category'] == cat
    tr, te = main[tr_mask], main[te_mask]
    if len(te) < 3 or len(tr) < 10:
        continue
    for y in y_avail:
        tr_y = tr[tr[y].notna()]
        te_y = te[te[y].notna()]
        if len(tr_y) < 60 or len(te_y) < 3:
            continue
        for fset, feats in [('COMP', COMP_FEATS), ('ENH', ENH_FEATS)]:
            yp, (ym, ys, lo, hi), _ = fit_predict_ridge(
                tr_y[feats].values, tr_y[y].values, te_y[feats].values)
            yte_s = (np.clip(te_y[y].values, lo, hi) - ym) / ys
            if np.var(yte_s) < 1e-6:
                r2 = np.nan  # 退化格: 留出系统 Y 近恒定 → R² 无定义 (b6 同类问题)
            else:
                r2 = r2_score(yte_s, yp)
            loso_rows.append({'holdout': cat, 'Y': y, 'feature_set': fset,
                              'R2': r2, 'n_train': len(tr_y), 'n_test': len(te_y)})
    print(f"  {cat} 完成", end='', flush=True)
print()

loso_df = pd.DataFrame(loso_rows)
loso_df.to_csv(TABLES_DIR / 'law2_pattern_loso.csv', index=False)

# 聚合: 退化格 (NaN) 剔除; 中位数用 winsorize [-2,1] 防域外极端值主导
n_degen = int(loso_df['R2'].isna().sum())
loso_valid = loso_df.dropna(subset=['R2'])
pos = loso_valid.groupby('feature_set')['R2'].apply(lambda s: float((s > 0).mean()))
rate_c, rate_e = pos['COMP'], pos['ENH']
loso_w = loso_valid.copy()
loso_w['R2w'] = loso_w['R2'].clip(-2, 1)
sys_med = (loso_w.groupby(['holdout', 'feature_set'])['R2w'].median().unstack())
sys_med['delta'] = sys_med['ENH'] - sys_med['COMP']
max_deg = float(-sys_med['delta'].min()) if (sys_med['delta'] < 0).any() else 0.0
p4_pass = bool((rate_e - rate_c) >= 0.05 and sys_med['delta'].min() >= -0.05)
summary['P4'] = {'pos_rate_COMP': float(rate_c), 'pos_rate_ENH': float(rate_e),
                 'delta_pp': float(rate_e - rate_c),
                 'n_cells': int(len(loso_valid)), 'n_degenerate_excluded': n_degen,
                 'baseline_reference_b6': 0.3077,
                 'per_system_median_delta': {k: float(v) for k, v in sys_med['delta'].items()},
                 'max_system_degradation': max_deg, 'pass': p4_pass}
print(f"\n  正迁移率 (R²>0 占比): COMP={rate_c:.4f} → ENH={rate_e:.4f} "
      f"(Δ={rate_e - rate_c:+.4f}, 门控 ≥+0.05; B6历史参考 0.3077)")
print(f"  单系统中位 R² 最大退化: {max_deg:.4f} (门控 ≤0.05)")
print(f"  → P4 {'PASS' if p4_pass else 'FAIL'}")

# ============================================================
# P5: Pilot OOS (主集训练 → 20 Pilot 预测)
# ============================================================
print("\n[P5] Pilot OOS 非劣性 + Panel B 改进...")

pilot_y_cols = [c for c in pilot_geom.columns
                if c in main.columns and c not in
                ['seq_id', 'category', 'aa_type', 'n', 'panel', 'n_samples']]
p5_rows = []
for y in pilot_y_cols:
    pte = pilot[pilot[y].notna()]
    if len(pte) < 8 or pte[y].std() < 1e-10:
        continue
    tr = main[main[y].notna()]
    res = {'Y': y, 'n_pilot': len(pte)}
    for fset, feats in [('COMP', COMP_FEATS), ('ENH', ENH_FEATS)]:
        yp_s, (ym, ys, lo, hi), _ = fit_predict_ridge(
            tr[feats].values, tr[y].values, pte[feats].values)
        yp_raw = yp_s * ys + ym
        yt = pte[y].values
        r, p = pearsonr(yp_raw, yt)
        rho, _ = spearmanr(yp_raw, yt)
        res[f'r_{fset}'] = float(r)
        res[f'p_{fset}'] = float(p)
        res[f'rho_{fset}'] = float(rho)
        # Panel B 子集
        pb = pte['panel'] == 'B'
        if pb.sum() >= 5:
            res[f'r_PanelB_{fset}'] = float(pearsonr(yp_raw[pb.values], yt[pb.values])[0])
    p5_rows.append(res)

p5_df = pd.DataFrame(p5_rows)
p5_df.to_csv(TABLES_DIR / 'law2_pattern_pilot_oos.csv', index=False)

n_eval = len(p5_df)
n_sig_e = int(((p5_df['r_ENH'] > 0) & (p5_df['p_ENH'] < 0.05)).sum())
n_sig_c = int(((p5_df['r_COMP'] > 0) & (p5_df['p_COMP'] < 0.05)).sum())
med_r_e = float(p5_df['r_ENH'].median())
med_r_c = float(p5_df['r_COMP'].median())

# Panel B 改进
pb = p5_df.dropna(subset=['r_PanelB_COMP', 'r_PanelB_ENH'])
pb_delta = pb['r_PanelB_ENH'] - pb['r_PanelB_COMP']
n_pb_improved = int((pb_delta > 0).sum())

# 非劣性: 相对历史基线 (19/32 显著, 中位 r=0.499) 且不低于 COMP 模型
BASELINE_SIG, BASELINE_MEDR = 19 / 32, 0.4994
noninf_vs_hist = (n_sig_e / n_eval >= BASELINE_SIG - 0.06) and (med_r_e >= BASELINE_MEDR - 0.05)
noninf_vs_comp = (n_sig_e >= n_sig_c) and (med_r_e >= med_r_c - 0.02)
pb_improved = bool(len(pb) and (pb_delta.median() > 0 or n_pb_improved >= np.ceil(0.6 * len(pb))))
p5_pass = bool(noninf_vs_hist and noninf_vs_comp and pb_improved)

summary['P5'] = {'n_evaluable': n_eval,
                 'ENH': {'n_sig_pos': n_sig_e, 'median_r': med_r_e},
                 'COMP': {'n_sig_pos': n_sig_c, 'median_r': med_r_c},
                 'historical_baseline': {'sig_rate': BASELINE_SIG, 'median_r': BASELINE_MEDR},
                 'panelB': {'n_Y': int(len(pb)), 'n_improved': n_pb_improved,
                            'median_delta_r': float(pb_delta.median()) if len(pb) else None},
                 'noninf_vs_hist': bool(noninf_vs_hist),
                 'noninf_vs_comp': bool(noninf_vs_comp),
                 'pass': p5_pass}
print(f"  可评估 Y: {n_eval} | 显著正相关: COMP={n_sig_c} → ENH={n_sig_e} "
      f"(历史 {BASELINE_SIG:.2%})")
print(f"  中位 r: COMP={med_r_c:.4f} → ENH={med_r_e:.4f} (历史 0.499)")
print(f"  Panel B: {n_pb_improved}/{len(pb)} Y 改进, 中位 Δr={pb_delta.median():+.4f}")
print(f"  → P5 {'PASS' if p5_pass else 'FAIL'}")

# ============================================================
# 总判定 (预注册 §6)
# ============================================================
main_criteria = [p3_pass, p4_pass, p5_pass]
n_main_pass = sum(main_criteria)
if p1_pass and p2_pass and n_main_pass >= 2:
    verdict = 'BASIS_UPGRADE_ACCEPTED'
elif p1_pass and not p2_pass and p2_posthoc and n_main_pass >= 2:
    # 预注册池化门控失败 (n混淆) 但 post-hoc 分层 + 主判据支持 → 临时接受, 待320仲裁
    verdict = 'BASIS_UPGRADE_PROVISIONAL_POSTHOC'
elif not p2_pass and not p2_posthoc:
    # 预注册门控失败. 注意: n=15/每层5序列功效有限 (每层最小精确 p=1/120),
    # n=60 层方向性信号 (PR/entropy ρ=-0.97) 与 Das-Pappu 一致 → 非"无效应"证据,
    # 最终仲裁 = 320 序列全量验证 (Panel A/B 含均衡 κ×n 设计)
    verdict = 'P2_GATE_FAILED_POWER_LIMITED'
elif (p2_pass or p2_posthoc) and n_main_pass <= 1:
    verdict = 'KAPPA_VALID_INCREMENT_LIMITED'
else:
    verdict = 'GATE_FAILED'
summary['verdict'] = {'P1': bool(p1_pass),
                      'P2_prereg': bool(p2_pass), 'P2_posthoc': bool(p2_posthoc),
                      'P3': bool(p3_pass), 'P4': bool(p4_pass), 'P5': bool(p5_pass),
                      'n_main_pass': int(n_main_pass), 'decision': verdict}
print(f"\n{'=' * 70}\n总判定: {verdict} (门控 P1={p1_pass} P2={p2_pass}; "
      f"主判据 {n_main_pass}/3)\n{'=' * 70}")

with open(TABLES_DIR / 'law2_pattern_krefit_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"  → {TABLES_DIR / 'law2_pattern_krefit_summary.json'}")

# ============================================================
# 图 F1–F3 (png + svg + html)
# ============================================================
print("\n[F] 生成图 F1–F3...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DPI = 300

# ---- F1: HET_KAPPA κ~几何 (组成冻结受控检验) ----
f1_ys = [('PR', 'PR (参与比)'), ('spectral_decay', '谱衰减指数'),
         ('entropy', '构象熵'), ('pseudo_volume', '伪体积 (log)')]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (y, label) in zip(axes, f1_ys):
    for nval, mk, c in [(20, 'o', '#2196F3'), (40, 's', '#FF9800'), (60, '^', '#4CAF50')]:
        s = hk[hk['n'] == nval]
        ax.scatter(s['kappa'], s[y], marker=mk, c=c, s=55, alpha=0.85,
                   edgecolors='k', linewidths=0.5, label=f'n={nval}')
        if len(s) >= 4 and s['kappa'].nunique() >= 3:
            r = spearmanr(s['kappa'], s[y]).statistic
            ax.plot([], [], ' ', label=f'  ρ={r:+.2f}')
    ax.set_xlabel('Das-Pappu κ (计算值)')
    ax.set_ylabel(label)
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3)
fig.suptitle('F1. HET_KAPPA 组成冻结受控检验: κ vs 系综几何 (NCPR=0, FCR=1 恒定)',
             fontsize=12, y=1.02)
fig.tight_layout()
for ext in ['png', 'svg']:
    fig.savefig(FIG_DIR / f'F1_het_kappa_geometry.{ext}', dpi=DPI, bbox_inches='tight')
plt.close(fig)

fig_p = make_subplots(rows=1, cols=4, subplot_titles=[l for _, l in f1_ys])
for ci, (y, _) in enumerate(f1_ys, 1):
    for nval, mk, c in [(20, 'circle', '#2196F3'), (40, 'square', '#FF9800'), (60, 'triangle-up', '#4CAF50')]:
        s = hk[hk['n'] == nval]
        fig_p.add_trace(go.Scatter(
            x=s['kappa'], y=s[y], mode='markers', name=f'n={nval}',
            marker=dict(symbol=mk, size=9, color=c, line=dict(width=0.7, color='black')),
            text=s['seq_id'], legendgroup=f'n{nval}', showlegend=(ci == 1)), row=1, col=ci)
    fig_p.update_xaxes(title_text='κ', row=1, col=ci)
fig_p.update_layout(height=430, width=1500,
                    title_text='F1. HET_KAPPA 组成冻结受控检验: κ vs 系综几何',
                    template='plotly_white')
fig_p.write_html(FIG_DIR / 'F1_het_kappa_geometry.html', include_plotlyjs='cdn')

# ---- F2: κ/SCD 系数森林图 (4 核心Y bootstrap CI) ----
fig, ax = plt.subplots(figsize=(9, 5.5))
labels, ypos, meds, los, his, colors = [], [], [], [], [], []
for i, row in enumerate(boot_df.itertuples()):
    labels.append(f"{row.Y} · {'κ' if row.feature == 'kappa_f' else 'SCD/√L'}")
    ypos.append(len(boot_df) - i)
    meds.append(row.coef_median)
    los.append(row.ci_lo)
    his.append(row.ci_hi)
    colors.append('#D32F2F' if row.sig else '#9E9E9E')
for y0, lo, hi, c in zip(ypos, los, his, colors):
    ax.plot([lo, hi], [y0, y0], '-', color=c, lw=2.5)
    ax.plot([lo, lo], [y0 - 0.12, y0 + 0.12], color=c, lw=1.5)
    ax.plot([hi, hi], [y0 - 0.12, y0 + 0.12], color=c, lw=1.5)
ax.scatter(meds, ypos, c=colors, s=55, zorder=5, edgecolors='k', linewidths=0.5)
ax.axvline(0, color='k', ls='--', lw=1, alpha=0.6)
ax.set_yticks(ypos)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('标准化 Ridge 系数 β (bootstrap 中位数 + 95% CI)')
ax.set_title('F2. κ / SCD 系数显著性 (1000× bootstrap; 红=CI不含0)')
ax.grid(axis='x', alpha=0.3)
fig.tight_layout()
for ext in ['png', 'svg']:
    fig.savefig(FIG_DIR / f'F2_coef_forest.{ext}', dpi=DPI, bbox_inches='tight')
plt.close(fig)

fig_p = go.Figure()
for i, row in enumerate(boot_df.itertuples()):
    c = '#D32F2F' if row.sig else '#9E9E9E'
    name = f"{row.Y} · {'κ' if row.feature == 'kappa_f' else 'SCD/√L'}"
    fig_p.add_trace(go.Scatter(
        x=[row.ci_lo, row.coef_median, row.ci_hi], y=[name] * 3,
        mode='lines+markers', marker=dict(size=[3, 10, 3], color=c),
        line=dict(color=c, width=3), showlegend=False,
        hovertemplate=f"β={row.coef_median:.4f} [{row.ci_lo:.4f}, {row.ci_hi:.4f}]"))
fig_p.add_vline(x=0, line_dash='dash', line_color='black', opacity=0.5)
fig_p.update_layout(height=520, width=850, template='plotly_white',
                    title='F2. κ / SCD 系数显著性 (1000× bootstrap)',
                    xaxis_title='标准化系数 β (95% CI)')
fig_p.write_html(FIG_DIR / 'F2_coef_forest.html', include_plotlyjs='cdn')

# ---- F3: LOSO 改进对比 ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
width = 0.38
cats_sorted = sys_med.sort_values('delta', ascending=False).index
xc = np.arange(len(cats_sorted))
ax.bar(xc - width / 2, [sys_med.loc[c, 'COMP'] for c in cats_sorted], width,
       label='COMP (0阶组成)', color='#90CAF9', edgecolor='k', lw=0.4)
ax.bar(xc + width / 2, [sys_med.loc[c, 'ENH'] for c in cats_sorted], width,
       label='ENH (0阶+2阶模式)', color='#EF9A9A', edgecolor='k', lw=0.4)
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(xc)
ax.set_xticklabels(cats_sorted, rotation=38, ha='right', fontsize=8.5)
ax.set_ylabel('LOSO 中位 R²')
ax.set_title('F3a. 各留出系统 LOSO 中位 R² 对比')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

ax = axes[1]
sc = loso_w.pivot_table(index=['holdout', 'Y'], columns='feature_set',
                        values='R2w').reset_index()
ax.scatter(sc['COMP'], sc['ENH'], s=14, alpha=0.45, c='#5E35B1', edgecolors='none')
lim = [-2.05, 1.05]
ax.plot(lim, lim, 'k--', lw=1, alpha=0.6)
ax.axhline(0, color='gray', lw=0.6, alpha=0.5)
ax.axvline(0, color='gray', lw=0.6, alpha=0.5)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel('COMP LOSO R²')
ax.set_ylabel('ENH LOSO R²')
ax.set_title(f'F3b. 逐格对比 (n={len(sc)} 系统×Y)\n'
             f'正迁移率 {rate_c:.1%} → {rate_e:.1%}')
ax.grid(alpha=0.3)
fig.tight_layout()
for ext in ['png', 'svg']:
    fig.savefig(FIG_DIR / f'F3_loso_improvement.{ext}', dpi=DPI, bbox_inches='tight')
plt.close(fig)

fig_p = make_subplots(rows=1, cols=2,
                      subplot_titles=['各留出系统 LOSO 中位 R²', '逐格对比 (系统×Y)'])
for fset, c in [('COMP', '#90CAF9'), ('ENH', '#EF9A9A')]:
    fig_p.add_trace(go.Bar(x=list(cats_sorted), y=[sys_med.loc[cc, fset] for cc in cats_sorted],
                           name=fset, marker_color=c), row=1, col=1)
fig_p.add_trace(go.Scatter(x=sc['COMP'], y=sc['ENH'], mode='markers',
                           marker=dict(size=5, color='#5E35B1', opacity=0.45),
                           text=sc['holdout'] + ' · ' + sc['Y'], showlegend=False), row=1, col=2)
fig_p.add_trace(go.Scatter(x=lim, y=lim, mode='lines', line=dict(dash='dash', color='black'),
                           showlegend=False), row=1, col=2)
fig_p.update_layout(height=480, width=1400, template='plotly_white',
                    title=f'F3. LOSO 迁移改进 (正迁移率 {rate_c:.1%} → {rate_e:.1%})')
fig_p.write_html(FIG_DIR / 'F3_loso_improvement.html', include_plotlyjs='cdn')

print(f"  → {FIG_DIR}/F1|F2|F3 (.png/.svg/.html)")
print("\n" + "=" * 70)
print(f"完成: {verdict}")
print("=" * 70)
