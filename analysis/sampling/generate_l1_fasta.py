#!/usr/bin/env python3
"""
L1 疏水梯度实验 — 序列生成器
=============================
生成 PolyA, PolyV, PolyI, PolyF 纯合序列 (n=4-50) 的 FASTA 文件.

疏水梯度 (按 Kyte-Doolittle 疏水性):
  A (Ala, 1.8) → V (Val, 4.2) → I (Ile, 4.5) → F (Phe, 2.8)
  已有: L (Leu, 3.8), G (Gly, -0.4, 柔性基线), K (Lys, -3.9, 电荷基线)

输出: test_workflow/polyx_ensemble/l1_hydrophobic_sequences.fasta
      test_workflow/polyx_ensemble/l1_manifest.csv

用法:
    python field_theory/scripts/generate_l1_fasta.py
"""

import pathlib
import csv
import sys
import os

# 添加项目根目录
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.resolve()))

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()
OUT_DIR = PROJECT_ROOT / "test_workflow" / "polyx_ensemble"
FASTA_PATH = OUT_DIR / "l1_hydrophobic_sequences.fasta"
MANIFEST_PATH = OUT_DIR / "l1_manifest.csv"

# L1 疏水梯度氨基酸 (按疏水性排序)
L1_AA = ["A", "V", "I", "F"]

# 疏水性参数 (Kyte-Doolittle)
HYDROPHOBICITY = {
    "A": 1.8,   # 丙氨酸
    "V": 4.2,   # 缬氨酸
    "I": 4.5,   # 异亮氨酸
    "F": 2.8,   # 苯丙氨酸
    "L": 3.8,   # 亮氨酸 (已有)
    "G": -0.4,  # 甘氨酸 (基线, 已有)
    "K": -3.9,  # 赖氨酸 (电荷基线, 已有)
    "E": -3.5,  # 谷氨酸 (电荷基线, 已有)
    "S": -0.8,  # 丝氨酸 (极性, 已有)
}

AA_DESCRIPTIONS = {
    "A": "疏水-小侧链",
    "V": "疏水-中等支链",
    "I": "疏水-大β支链",
    "F": "疏水-芳香环",
}


def generate_l1_sequences():
    """生成 L1 疏水梯度序列: (A)n, (V)n, (I)n, (F)n (n=4~50)"""
    records = []
    for aa in L1_AA:
        for n in range(4, 51):  # n=4-50 (BioEmu 最低要求 >=4)
            seq_id = f"PolyX_Poly{aa}_{n}"
            seq = aa * n
            records.append({
                "seq_id": seq_id,
                "sequence": seq,
                "aa": aa,
                "category": "L1_hydrophobic",
                "n": n,
                "hydrophobicity": HYDROPHOBICITY.get(aa, 0),
                "description": AA_DESCRIPTIONS.get(aa, ""),
            })
    return records


def write_fasta(records, path):
    """写入 FASTA 文件"""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(f">{rec['seq_id']}\n")
            f.write(f"{rec['sequence']}\n")
    print(f"FASTA written: {path} ({len(records)} sequences)")


def write_manifest(records, path):
    """写入 manifest CSV"""
    fieldnames = ["seq_id", "sequence", "aa", "category", "n", "hydrophobicity", "description"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Manifest written: {path} ({len(records)} rows)")


def main():
    print("=" * 60)
    print("L1 Hydrophobic Gradient — Sequence Generation")
    print("=" * 60)
    print(f"AAs: {L1_AA}")
    print(f"Hydrophobicity: {', '.join(f'{aa}={HYDROPHOBICITY[aa]}' for aa in L1_AA)}")
    print(f"n range: 4-50 (47 per AA)")
    print(f"Total: {len(L1_AA)} × 47 = {len(L1_AA) * 47} sequences")
    print()

    records = generate_l1_sequences()
    print(f"Generated {len(records)} sequences")
    print(f"  A: {sum(1 for r in records if r['aa'] == 'A')} sequences")
    print(f"  V: {sum(1 for r in records if r['aa'] == 'V')} sequences")
    print(f"  I: {sum(1 for r in records if r['aa'] == 'I')} sequences")
    print(f"  F: {sum(1 for r in records if r['aa'] == 'F')} sequences")

    write_fasta(records, FASTA_PATH)
    write_manifest(records, MANIFEST_PATH)

    print("\nDone. Next steps:")
    print(f"  wsl bash -c \"source /home/liuchuanyang13/anaconda3/etc/profile.d/conda.sh && conda activate bioemu && cd /mnt/b/2026/Exploration/ProtGenesis2_Ensemble && python test_workflow/polyx_ensemble/run_polyx_bioemu.py --fasta test_workflow/polyx_ensemble/l1_hydrophobic_sequences.fasta --output test_workflow/polyx_ensemble/output --num-samples 250 --workers 2\"")


if __name__ == "__main__":
    main()