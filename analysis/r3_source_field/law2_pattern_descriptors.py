#!/usr/bin/env python3
"""
Law 2 — 模式描述符计算 (Pattern Descriptors: Das-Pappu kappa, Hofmann SCD, ...)
=============================================================================
依据: Article_Preparation/Law2_Pattern_Basis_Plan.md v1.0 (预注册)

对 systemwide_enhanced_geometry_v4.csv 全部 1,279 序列 + 20 Pilot 序列计算:
  0 阶矩: f_pos, f_neg, FCR, NCPR, abs_NCPR, mean_hydro (KD)
  2 阶矩: kappa (Das-Pappu 2013), SCD (Hofmann 2012), SHD (Sawle-Ghosh 2015),
          gamma1 (最近邻电荷相关), blockiness, charge_spacing, H_dipep
  1 阶:   H_comp (20AA 组成熵)

电荷约定: K,R=+1; D,E=-1; 其余=0 (localCIDER 标准; H=0)

输出:
  field_theory/tables/law2_pattern_descriptors.csv       (1,279 + 20 行)
  field_theory/tables/law2_pattern_toycheck.json         (sanity 电池)
  field_theory/tables/law2_pattern_descriptors_qc.json   (解析 QC 日志)
"""

import json
import math
import re
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
FIELD_THEORY = Path(__file__).parent.parent
TABLES_DIR = FIELD_THEORY / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

SYSTEMWIDE = FIELD_THEORY / "data/dms/phase9_systemwide/systemwide_enhanced_geometry_v4.csv"
PILOT_META = PROJECT_ROOT / "test_workflow/law2_validation/validation_320_metadata.csv"

POLYX_OUT = PROJECT_ROOT / "test_workflow/polyx_ensemble/output"
HET_OUT = PROJECT_ROOT / "test_workflow/heteropolymer_ensemble/output"
PILOT_OUT = PROJECT_ROOT / "test_workflow/law2_validation/output_pilot"

print("=" * 70)
print("Law 2 — 模式描述符计算 (kappa / SCD / SHD / ...)")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ============================================================
# 物理化学常数表
# ============================================================
CHARGE = {**{a: 1 for a in 'KR'}, **{a: -1 for a in 'DE'}}
KD_HYDRO = {
    'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9, 'A': 1.8,
    'G': -0.4, 'T': -0.7, 'S': -0.8, 'W': -0.9, 'Y': -1.3, 'P': -1.6,
    'H': -3.2, 'E': -3.5, 'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5,
}
KD_MIN, KD_MAX = -4.5, 4.5  # 归一化到 [0,1]: h = (KD - KD_MIN)/(KD_MAX - KD_MIN)

AA20 = set('ACDEFGHIKLMNPQRSTVWY')

# ============================================================
# 描述符函数
# ============================================================

def charge_array(seq):
    return np.array([CHARGE.get(a, 0) for a in seq], dtype=float)


# ============================================================
# Das-Pappu κ — localCIDER 0.1.21 权威算法 (滑窗 δ/δ_max)
# 依据: localCIDER backend/sequence.py (deltaForm/delta/deltaMax/kappa)
#   σ = NCPR²/FCR (序列级); δ_b = mean_windows (σ − σ_b)², σ_b = NCPR_b²/FCR_b
#   δ = (δ_{b=5} + δ_{b=6})/2; δ_max = 同组成最大电荷分离排列族的最大 δ
#   κ = δ/δ_max; 1.0<κ<1.1 → 1.0; FCR=0 → NaN; 同聚电解质 → 1.0 (约定)
# ============================================================

def _sigma_of_counts(npos, nneg, L):
    if npos + nneg == 0:
        return 0.0
    ncpr = (npos - nneg) / L
    fcr = (npos + nneg) / L
    return ncpr * ncpr / fcr


def _delta_form(q, bloblen):
    """滑窗 δ: localCIDER deltaForm(bloblen)."""
    L = len(q)
    nwin = L - bloblen + 1
    if nwin <= 0:
        return 0.0
    npos = float(np.sum(q > 0))
    nneg = float(np.sum(q < 0))
    sigma = _sigma_of_counts(npos, nneg, L)
    pos = (q > 0).astype(float)
    neg = (q < 0).astype(float)
    kernel = np.ones(bloblen)
    wpos = np.convolve(pos, kernel, mode='valid')
    wneg = np.convolve(neg, kernel, mode='valid')
    bncpr = (wpos - wneg) / bloblen
    bfcr = (wpos + wneg) / bloblen
    with np.errstate(divide='ignore', invalid='ignore'):
        bsig = np.where(bfcr > 0, bncpr * bncpr / np.maximum(bfcr, 1e-12), 0.0)
    return float(np.mean((sigma - bsig) ** 2))


def _delta_pattern(q):
    """δ = (δ_5 + δ_6)/2 (localCIDER delta)."""
    return (_delta_form(q, 5) + _delta_form(q, 6)) / 2


def _delta_max(npos, nneg, nneut):
    """localCIDER deltaMax: 同组成 (n+, n−, n0) 下最大电荷分离排列族的最大 δ."""
    L = npos + nneg + nneut
    if npos + nneg == 0:
        return 0.0
    dmax = 0.0
    if npos == 0 or nneg == 0:
        # 单一电荷类型: 带电块在中性背景上滑动 (或中性块在带电背景上滑动)
        ncharge = max(npos, nneg)
        cv = 1.0 if npos > 0 else -1.0
        if nneut > ncharge:
            for p in range(0, L - ncharge + 1):
                q = np.zeros(L)
                q[p:p + ncharge] = cv
                dmax = max(dmax, _delta_pattern(q))
        else:
            for p in range(0, L - nneut + 1):
                q = np.array([cv] * p + [0.0] * nneut + [cv] * (ncharge - p))
                dmax = max(dmax, _delta_pattern(q))
    elif nneut == 0:
        # Maximum Charge Separation: 小块滑过大块 (双block 族)
        if npos > nneg:
            for p in range(0, L - nneg + 1):
                q = np.array([1.0] * p + [-1.0] * nneg + [1.0] * (npos - p))
                dmax = max(dmax, _delta_pattern(q))
        else:
            for p in range(0, L - npos + 1):
                q = np.array([-1.0] * p + [1.0] * npos + [-1.0] * (nneg - p))
                dmax = max(dmax, _delta_pattern(q))
    elif nneut >= 18:
        # 中性残基首/中/尾分布启发式 (0–6 at ends)
        for s in range(0, 7):
            for e in range(0, 7):
                m = nneut - s - e
                if m < 0:
                    continue
                q = np.array([0.0] * s + [1.0] * npos + [0.0] * m + [-1.0] * nneg + [0.0] * e)
                dmax = max(dmax, _delta_pattern(q))
    else:
        # 一般情形: 穷举 (mid, start) 中性分布的双block 族
        for m in range(0, nneut + 1):
            for s in range(0, nneut - m + 1):
                e = nneut - s - m
                q = np.array([0.0] * s + [1.0] * npos + [0.0] * m + [-1.0] * nneg + [0.0] * e)
                dmax = max(dmax, _delta_pattern(q))
    return dmax


def das_pappu_kappa(seq):
    """Das-Pappu κ (2013, PNAS 110:13392) — localCIDER 0.1.21 滑窗 δ/δ_max 算法.

    返回: κ ∈ [0,1]; FCR=0 → NaN; 同聚电解质 (单电荷无中性) → 1.0 (项目约定).
    """
    q = charge_array(seq)
    L = len(q)
    if L == 0:
        return np.nan
    npos = int(np.sum(q > 0))
    nneg = int(np.sum(q < 0))
    nneut = L - npos - nneg
    if npos + nneg == 0:
        return np.nan  # FCR = 0 → κ 未定义
    d = _delta_pattern(q)
    dm = _delta_max(npos, nneg, nneut)
    if dm <= 0:
        # 同聚电解质 (单一电荷且无中性残基): δ_max ≡ 0, localCIDER 返回 −1
        return 1.0  # 项目约定: 平凡最大分离
    kappa = d / dm
    if 1.0 < kappa < 1.1:
        kappa = 1.0
    return float(kappa)


def hofmann_scd(seq):
    """Hofmann SCD (2012, PCCP 14:13213).

    SCD = (1/L) Σ_{j} Σ_{i>j} q_i q_j √(i-j)
    符号: >0 同号排斥(扩张), <0 异号吸引(塌缩)
    """
    q = charge_array(seq)
    L = len(q)
    if L < 2 or np.all(q == 0):
        return 0.0
    total = 0.0
    for i in range(1, L):
        if q[i] == 0:
            continue
        for j in range(i):
            if q[j] == 0:
                continue
            total += q[i] * q[j] * math.sqrt(i - j)
    return total / L


def sawle_ghosh_shd(seq):
    """SHD: 与 SCD 同核, q → 归一化疏水性 h ∈ [0,1] (Sawle & Ghosh 2015)."""
    h = np.array([(KD_HYDRO.get(a, 0.0) - KD_MIN) / (KD_MAX - KD_MIN) for a in seq])
    L = len(h)
    if L < 2:
        return 0.0
    total = 0.0
    for i in range(1, L):
        for j in range(i):
            total += h[i] * h[j] * math.sqrt(i - j)
    return total / L


def gamma1(seq):
    """最近邻电荷相关 Γ₁ = (1/(L-1)) Σ q_i q_{i+1}."""
    q = charge_array(seq)
    if len(q) < 2:
        return 0.0
    return float(np.mean(q[:-1] * q[1:]))


def blockiness(seq):
    """最大同号连续 block 长度 / L."""
    q = charge_array(seq)
    L = len(q)
    if L == 0:
        return 0.0
    best, cur, sign = 0, 0, 0
    for v in q:
        if v != 0 and v == sign:
            cur += 1
        elif v != 0:
            sign, cur = v, 1
        else:
            sign, cur = 0, 0
        best = max(best, cur)
    return best / L


def charge_spacing(seq):
    """相邻带电残基的平均序列间隔 (≥2 带电残基时定义)."""
    idx = [i for i, a in enumerate(seq) if a in CHARGE]
    if len(idx) < 2:
        return np.nan
    return float(np.mean(np.diff(idx)))


def composition_entropy(seq):
    """20AA 组成香农熵 (bits)."""
    L = len(seq)
    if L == 0:
        return 0.0
    counts = pd.Series(list(seq)).value_counts()
    p = counts / L
    return float(-(p * np.log2(p)).sum())


def dipeptide_entropy(seq):
    """二肽分布香农熵 (bits)."""
    if len(seq) < 2:
        return 0.0
    di = [seq[i:i + 2] for i in range(len(seq) - 1)]
    counts = pd.Series(di).value_counts()
    p = counts / len(di)
    return float(-(p * np.log2(p)).sum())


def compute_descriptors(seq):
    """计算全部描述符, 返回 dict."""
    seq = ''.join(a for a in seq.upper() if a in AA20)
    L = len(seq)
    q = charge_array(seq)
    f_pos = float(np.mean(q > 0)) if L else 0.0
    f_neg = float(np.mean(q < 0)) if L else 0.0
    scd = hofmann_scd(seq)
    return {
        'L': L,
        'f_pos': f_pos,
        'f_neg': f_neg,
        'FCR': f_pos + f_neg,
        'NCPR': f_pos - f_neg,
        'abs_NCPR': abs(f_pos - f_neg),
        'mean_hydro': float(np.mean([KD_HYDRO[a] for a in seq])) if L else 0.0,
        'kappa': das_pappu_kappa(seq),
        'SCD': scd,
        'SCD_sqrtL': scd / math.sqrt(L) if L > 0 else 0.0,
        'SHD': sawle_ghosh_shd(seq),
        'gamma1': gamma1(seq),
        'blockiness': blockiness(seq),
        'charge_spacing': charge_spacing(seq),
        'H_comp': composition_entropy(seq),
        'H_dipep': dipeptide_entropy(seq),
    }

# ============================================================
# Step 0: 玩具序列 sanity 电池 (预注册 §2)
# ============================================================
print("\n[0/4] 玩具序列 sanity 电池...")

toy = {
    'K10E10_diblock': 'K' * 10 + 'E' * 10,
    'KE10_alternating': 'KE' * 10,
    'K20_single': 'K' * 20,
    'A20_neutral': 'A' * 20,
    'KKKEKKKE_mixed': 'KKKEKKKE',
    'K5E5K5E5_4block': 'K' * 5 + 'E' * 5 + 'K' * 5 + 'E' * 5,
}
toy_res = {name: compute_descriptors(s) for name, s in toy.items()}

checks = []
def check(name, cond, detail):
    checks.append({'name': name, 'pass': bool(cond), 'detail': detail})
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")

check('kappa(K10E10)==1', abs(toy_res['K10E10_diblock']['kappa'] - 1.0) < 1e-9,
      f"kappa={toy_res['K10E10_diblock']['kappa']:.4f} (期望=1.0, 自身即δ_max排列)")
check('kappa((KE)10)<0.01', toy_res['KE10_alternating']['kappa'] < 0.01,
      f"kappa={toy_res['KE10_alternating']['kappa']:.6f} (期望≈0)")
check('kappa(K20)==1', abs(toy_res['K20_single']['kappa'] - 1.0) < 1e-9,
      f"kappa={toy_res['K20_single']['kappa']:.4f} (同聚电解质约定)")
check('kappa(A20)==NaN', np.isnan(toy_res['A20_neutral']['kappa']),
      f"kappa={toy_res['A20_neutral']['kappa']} (FCR=0)")
check('kappa(KKKEKKKE)∈[0.05,0.15]',
      0.05 <= toy_res['KKKEKKKE_mixed']['kappa'] <= 0.15,
      f"kappa={toy_res['KKKEKKKE_mixed']['kappa']:.4f} (脚本核验值≈0.103)")
check('kappa(K5E5K5E5)∈[0.22,0.32]',
      0.22 <= toy_res['K5E5K5E5_4block']['kappa'] <= 0.32,
      f"kappa={toy_res['K5E5K5E5_4block']['kappa']:.4f} (脚本核验值≈0.262, 块数敏感)")
check('kappa排序: diblock=single>4block>mixed>alt',
      (toy_res['K10E10_diblock']['kappa'] >= toy_res['K20_single']['kappa'] - 1e-9 and
       toy_res['K20_single']['kappa'] > toy_res['K5E5K5E5_4block']['kappa'] and
       toy_res['K5E5K5E5_4block']['kappa'] > toy_res['KKKEKKKE_mixed']['kappa'] and
       toy_res['KKKEKKKE_mixed']['kappa'] > toy_res['KE10_alternating']['kappa']),
      f"1.0=1.0>{toy_res['K5E5K5E5_4block']['kappa']:.3f}>{toy_res['KKKEKKKE_mixed']['kappa']:.3f}>{toy_res['KE10_alternating']['kappa']:.4f}")
check('SCD(K10E10)<0', toy_res['K10E10_diblock']['SCD'] < 0,
      f"SCD={toy_res['K10E10_diblock']['SCD']:.3f}")
check('SCD排序: diblock<4block<alt<0<single',
      (toy_res['K10E10_diblock']['SCD'] < toy_res['K5E5K5E5_4block']['SCD'] and
       toy_res['K5E5K5E5_4block']['SCD'] < toy_res['KE10_alternating']['SCD'] and
       toy_res['KE10_alternating']['SCD'] < 0 < toy_res['K20_single']['SCD']),
      f"{toy_res['K10E10_diblock']['SCD']:.2f}<{toy_res['K5E5K5E5_4block']['SCD']:.2f}<{toy_res['KE10_alternating']['SCD']:.2f}<0<{toy_res['K20_single']['SCD']:.2f} (√d核长程分离主导, Hofmann2012一致)")
check('SCD(K20)>0', toy_res['K20_single']['SCD'] > 0,
      f"SCD={toy_res['K20_single']['SCD']:.3f}")
check('SCD(A20)==0', abs(toy_res['A20_neutral']['SCD']) < 1e-12,
      f"SCD={toy_res['A20_neutral']['SCD']:.6f}")
check('SCD(KKKEKKKE)>0', toy_res['KKKEKKKE_mixed']['SCD'] > 0,
      f"SCD={toy_res['KKKEKKKE_mixed']['SCD']:.3f}")

toy_summary = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_checks': len(checks),
    'n_pass': sum(c['pass'] for c in checks),
    'checks': checks,
    'toy_values': {k: {kk: (None if (isinstance(vv, float) and np.isnan(vv)) else vv)
                       for kk, vv in v.items()} for k, v in toy_res.items()},
}
with open(TABLES_DIR / 'law2_pattern_toycheck.json', 'w', encoding='utf-8') as f:
    json.dump(toy_summary, f, indent=2, ensure_ascii=False)
print(f"  → {TABLES_DIR / 'law2_pattern_toycheck.json'}: {toy_summary['n_pass']}/{toy_summary['n_checks']} PASS")

# ============================================================
# Step 1: 构建全局 seq_id → sequence 解析器
# ============================================================
print("\n[1/4] 构建序列解析器...")

seq_db = {}

def read_fasta_seq(path):
    """读取 fasta 第一条序列 (跳过注释/描述行)."""
    try:
        lines = Path(path).read_text(encoding='utf-8', errors='ignore').splitlines()
    except Exception:
        return None
    seq = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith('>') or ln.startswith(';'):
            if seq:
                break
            continue
        seq.append(ln)
    s = ''.join(seq)
    s = ''.join(a for a in s.upper() if a in AA20)
    return s if s else None

# ① validation_320_metadata.csv (含 Pilot 全部序列)
if PILOT_META.exists():
    meta = pd.read_csv(PILOT_META)
    for _, r in meta.iterrows():
        if isinstance(r.get('sequence'), str) and r['sequence']:
            seq_db[r['seq_id']] = r['sequence'].upper()
    print(f"  ① validation_320_metadata.csv: {len(seq_db)} 序列")

# ② 扫描三个 output 目录的 sequence.fasta / wt.fasta
n_scan = 0
for base in [POLYX_OUT, HET_OUT, PILOT_OUT, POLYX_OUT / 'natural_idp']:
    if not base.exists():
        continue
    for d in base.iterdir():
        if not d.is_dir() or d.name.startswith('_temp'):
            continue
        for fname in ['sequence.fasta', 'wt.fasta']:
            fp = d / fname
            if fp.exists():
                s = read_fasta_seq(fp)
                if s:
                    seq_db.setdefault(d.name, s)
                    n_scan += 1
                break
print(f"  ② output 目录扫描: +{n_scan} 序列 (累计 {len(seq_db)})")

# ③ DMS master table target_seq → 8 蛋白 wt 序列 (v1.3 补充)
DMS_MASTER = FIELD_THEORY / "data/dms/phase9_dms_expansion/phase9_dms_master_table.csv"
DMS_SEQID_MAP = {'SPIKE': 'spike_rbd'}  # 特例: SPIKE → spike_rbd; 其余 {name.lower()}_wt
if DMS_MASTER.exists():
    dms = pd.read_csv(DMS_MASTER, usecols=['protein', 'target_seq'])
    dms = dms.drop_duplicates(subset='protein')
    n_dms = 0
    for _, r in dms.iterrows():
        sid = DMS_SEQID_MAP.get(r['protein'], f"{str(r['protein']).lower()}_wt")
        if isinstance(r['target_seq'], str) and r['target_seq'] and sid not in seq_db:
            seq_db[sid] = r['target_seq'].upper()
            n_dms += 1
    print(f"  ③ DMS master target_seq: +{n_dms} 序列 (累计 {len(seq_db)})")

# ④ 名称解析回退 (同聚物 / linker)
def parse_from_name(seq_id):
    m = re.match(r'^PolyX_Poly([A-Z])_(\d+)$', seq_id)
    if m:
        return m.group(1) * int(m.group(2))
    m = re.match(r'^PolyX_([A-Z])_n(\d+)$', seq_id)
    if m:
        return m.group(1) * int(m.group(2))
    m = re.match(r'^PolyX_linker_Poly(EAAAK|GGGGS)_(\d+)$', seq_id)
    if m:
        unit, n = m.group(1), int(m.group(2))
        return (unit * (n // 5 + 2))[:n]
    return None

print("  ④ 名称回退解析器就绪 (PolyX_Poly*_n / PolyX_*_n / PolyX_linker_*)")

# ============================================================
# Step 2: 主数据集 (v4, 1,279 序列)
# ============================================================
print("\n[2/4] 主数据集描述符计算...")

df = pd.read_csv(SYSTEMWIDE)
print(f"  v4 几何表: {len(df)} 序列")

qc = {'resolved_fasta': 0, 'resolved_name': 0, 'failed': [],
      'length_mismatch': []}
records = []
for _, row in df.iterrows():
    sid = row['seq_id']
    seq = seq_db.get(sid)
    src = 'fasta'
    if seq is None:
        seq = parse_from_name(sid)
        src = 'name'
    if seq is None:
        qc['failed'].append(sid)
        continue
    if src == 'fasta':
        qc['resolved_fasta'] += 1
    else:
        qc['resolved_name'] += 1
    n_res = row.get('n_residues')
    if pd.notna(n_res) and int(n_res) != len(seq):
        qc['length_mismatch'].append({'seq_id': sid, 'table_n': int(n_res), 'seq_L': len(seq)})
    desc = compute_descriptors(seq)
    desc['seq_id'] = sid
    desc['resolve_src'] = src
    records.append(desc)

df_desc = pd.DataFrame(records)
print(f"  解析成功: {len(df_desc)}/{len(df)} "
      f"(fasta={qc['resolved_fasta']}, name={qc['resolved_name']}, failed={len(qc['failed'])})")
print(f"  长度不一致: {len(qc['length_mismatch'])} 条 (保留但标记)")

# ============================================================
# Step 3: Pilot 20 序列
# ============================================================
print("\n[3/4] Pilot 20 序列描述符计算...")

pilot_records = []
if PILOT_META.exists():
    meta = pd.read_csv(PILOT_META)
    pilot_ids = [d.name for d in PILOT_OUT.iterdir() if d.is_dir()] if PILOT_OUT.exists() else []
    for sid in pilot_ids:
        row = meta[meta['seq_id'] == sid]
        seq = seq_db.get(sid)
        if seq is None and len(row):
            seq = row.iloc[0]['sequence']
        if seq is None:
            qc['failed'].append(f'PILOT:{sid}')
            continue
        desc = compute_descriptors(seq)
        desc['seq_id'] = sid
        desc['resolve_src'] = 'pilot_meta'
        if len(row):
            desc['panel'] = row.iloc[0]['panel']
            desc['context'] = row.iloc[0]['context']
        pilot_records.append(desc)
df_pilot = pd.DataFrame(pilot_records)
print(f"  Pilot: {len(df_pilot)} 序列")

# ============================================================
# Step 4: 合并保存
# ============================================================
print("\n[4/4] 保存...")

df_main = df.merge(df_desc.drop(columns=['L']), on='seq_id', how='left')
df_main['is_pilot'] = False
if len(df_pilot):
    df_pilot['is_pilot'] = True
    df_all = pd.concat([df_main, df_pilot], ignore_index=True, sort=False)
else:
    df_all = df_main

out_csv = TABLES_DIR / 'law2_pattern_descriptors.csv'
df_all.to_csv(out_csv, index=False)
print(f"  → {out_csv} ({len(df_all)} 行, {df_all.shape[1]} 列)")

# QC 摘要
qc['n_main'] = int(len(df_main))
qc['n_pilot'] = int(len(df_pilot))
qc['kappa_defined_main'] = int(df_main['kappa'].notna().sum())
qc['kappa_defined_by_category'] = (df_main.groupby('category')['kappa']
                                   .apply(lambda s: int(s.notna().sum())).to_dict())
with open(TABLES_DIR / 'law2_pattern_descriptors_qc.json', 'w', encoding='utf-8') as f:
    json.dump(qc, f, indent=2, ensure_ascii=False)
print(f"  → {TABLES_DIR / 'law2_pattern_descriptors_qc.json'}")

# HET_KAPPA 子集快览 (P1 门控预览, v1.1: 登记设计缺陷)
het_kappa = df_main[df_main['category'] == 'HET_KAPPA'].copy()
if len(het_kappa):
    het_kappa['kappa_designed'] = het_kappa['seq_id'].str.extract(r'HET_KAPPA_([\d.]+)_')[0].astype(float)
    # 设计缺陷登记: 0.9_40 与 1.0_40 序列重复; 0.7 档 NCPR≠0 (组成不冻结)
    het_kappa['is_dup_09'] = het_kappa['seq_id'].str.startswith('HET_KAPPA_0.9_40')
    het_kappa['is_07_nonfrozen'] = (het_kappa['kappa_designed'] == 0.7)
    rho_all = het_kappa[['kappa', 'kappa_designed']].corr(method='spearman').iloc[0, 1]
    hk_clean = het_kappa[~het_kappa['is_dup_09'] & ~het_kappa['is_07_nonfrozen']]
    rho_clean = hk_clean[['kappa', 'kappa_designed']].corr(method='spearman').iloc[0, 1]
    print(f"\n  HET_KAPPA 子集 (n={len(het_kappa)}):")
    print(f"    Spearman ρ(κ_computed, κ_designed) 全部 = {rho_all:.4f} (P1门控≥0.85)")
    print(f"    排除设计缺陷 (0.9重复+0.7不冻结, n={len(hk_clean)}) = {rho_clean:.4f} (P1门控≥0.90)")
    print(het_kappa[['seq_id', 'kappa_designed', 'kappa', 'NCPR', 'SCD', 'gamma1', 'blockiness',
                     'PR', 'spectral_decay', 'mean_pairwise_dist']]
          .sort_values(['kappa_designed', 'seq_id']).to_string(index=False))

print("\n" + "=" * 70)
print("完成")
print("=" * 70)
