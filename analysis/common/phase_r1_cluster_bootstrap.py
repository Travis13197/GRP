#!/usr/bin/env python3
"""
Phase 3: R1 聚类 Bootstrap — 修正伪重复的 p 值重估
=========================================================
审计问题: R1 chain_growth 分析中 297 条路径仅来自 ~9 个独立 AA 家族,
导致伪重复 (pseudoreplication), 有效样本量 n_eff ≈ 9 而非 297.

方法: Cluster Block Bootstrap
  - 聚类单元: AA 家族 (A, E, F, G, I, K, L, S, V) = 9 clusters
  - 每次 bootstrap: 从 9 个 cluster 中有放回抽取 9 个, 用其全部路径
  - 重算 bio vs null 的 Mann-Whitney p 值和效应量
  - 10000 次 bootstrap 得到 p 值的分布

输入: field_theory/tables/phase_ensemble_w2_joint_paths.csv
输出: field_theory/tables/phase_r1_cluster_bootstrap.json
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from pathlib import Path
from datetime import datetime

FIELD_THEORY = Path(__file__).parent.parent
INPUT_CSV = FIELD_THEORY / "tables" / "phase_ensemble_w2_joint_paths.csv"
OUTPUT_JSON = FIELD_THEORY / "tables" / "phase_r1_cluster_bootstrap.json"

N_BOOTSTRAP = 10000
RANDOM_SEED = 42

def main():
    print("=" * 60)
    print("Phase 3: R1 聚类 Bootstrap (n_eff correction)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} paths")

    # Focus on chain_growth (R1) vs null_random
    bio = df[df['is_bio'] == True].copy()
    null = df[df['is_bio'] == False].copy()

    # Add cluster label: AA family
    bio['cluster'] = bio['aa_from']
    null['cluster'] = null['aa_from']

    bio_clusters = sorted(bio['cluster'].unique())
    null_clusters = sorted(null['cluster'].unique())
    all_clusters = sorted(set(bio_clusters) | set(null_clusters))

    print(f"\nBio paths: {len(bio)}, clusters: {len(bio_clusters)} ({bio_clusters})")
    print(f"Null paths: {len(null)}, clusters: {len(null_clusters)} ({null_clusters})")
    print(f"Effective sample size: n_eff = {len(all_clusters)} clusters")

    # Original (unclustered) statistics
    bio_vals = bio['W2_joint'].values
    null_vals = null['W2_joint'].values
    orig_u, orig_p = mannwhitneyu(bio_vals, null_vals, alternative='less')
    orig_diff = bio_vals.mean() - null_vals.mean()
    pooled_std = np.sqrt((bio_vals.std()**2 + null_vals.std()**2) / 2)
    orig_d = orig_diff / pooled_std if pooled_std > 0 else 0

    print(f"\n--- Original (unclustered) ---")
    print(f"Bio mean: {bio_vals.mean():.4f}, Null mean: {null_vals.mean():.4f}")
    print(f"Diff: {orig_diff:.4f}, Cohen's d: {orig_d:.4f}")
    print(f"Mann-Whitney p: {orig_p:.3e}")

    # Cluster bootstrap
    rng = np.random.RandomState(RANDOM_SEED)
    boot_p_values = []
    boot_diffs = []
    boot_ds = []

    print(f"\n--- Cluster Bootstrap ({N_BOOTSTRAP} iterations) ---")
    print(f"Resampling {len(all_clusters)} clusters with replacement...")

    for i in range(N_BOOTSTRAP):
        # Resample clusters with replacement
        sampled = rng.choice(all_clusters, size=len(all_clusters), replace=True)

        # Collect all paths from sampled clusters
        boot_bio_parts = []
        boot_null_parts = []
        for c in sampled:
            bc = bio[bio['cluster'] == c]['W2_joint'].values
            nc = null[null['cluster'] == c]['W2_joint'].values
            if len(bc) > 0:
                boot_bio_parts.append(bc)
            if len(nc) > 0:
                boot_null_parts.append(nc)

        if not boot_bio_parts or not boot_null_parts:
            continue

        b_bio = np.concatenate(boot_bio_parts)
        b_null = np.concatenate(boot_null_parts)

        if len(b_bio) < 2 or len(b_null) < 2:
            continue

        try:
            _, p = mannwhitneyu(b_bio, b_null, alternative='less')
            boot_p_values.append(p)

            diff = b_bio.mean() - b_null.mean()
            boot_diffs.append(diff)

            ps = np.sqrt((b_bio.std()**2 + b_null.std()**2) / 2)
            boot_ds.append(diff / ps if ps > 0 else 0)
        except Exception:
            continue

        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{N_BOOTSTRAP} done")

    boot_p = np.array(boot_p_values)
    boot_diffs = np.array(boot_diffs)
    boot_ds = np.array(boot_ds)

    print(f"\nCompleted {len(boot_p)} valid bootstrap iterations")

    # Results
    results = {
        "timestamp": datetime.now().isoformat(),
        "description": "R1 cluster bootstrap for pseudoreplication correction",
        "method": "Cluster Block Bootstrap (AA family as cluster unit)",
        "n_clusters": len(all_clusters),
        "clusters": all_clusters,
        "n_bio_paths": len(bio),
        "n_null_paths": len(null),
        "n_bootstrap": len(boot_p),
        "original": {
            "bio_mean": float(bio_vals.mean()),
            "null_mean": float(null_vals.mean()),
            "mean_diff": float(orig_diff),
            "cohens_d": float(orig_d),
            "mannwhitney_p": float(orig_p),
            "note": "Unclustered — inflated significance due to pseudoreplication"
        },
        "cluster_bootstrap": {
            "p_median": float(np.median(boot_p)),
            "p_mean": float(np.mean(boot_p)),
            "p_ci95_low": float(np.percentile(boot_p, 2.5)),
            "p_ci95_high": float(np.percentile(boot_p, 97.5)),
            "p_fraction_significant_005": float(np.mean(boot_p < 0.05)),
            "p_fraction_significant_001": float(np.mean(boot_p < 0.01)),
            "diff_median": float(np.median(boot_diffs)),
            "diff_ci95_low": float(np.percentile(boot_diffs, 2.5)),
            "diff_ci95_high": float(np.percentile(boot_diffs, 97.5)),
            "cohens_d_median": float(np.median(boot_ds)),
            "cohens_d_ci95_low": float(np.percentile(boot_ds, 2.5)),
            "cohens_d_ci95_high": float(np.percentile(boot_ds, 97.5)),
        },
        "interpretation": {
            "original_p": f"p = {orig_p:.2e} (unclustered, likely inflated)",
            "bootstrap_p_median": f"p = {np.median(boot_p):.2e} (cluster bootstrap median)",
            "bootstrap_p_ci95": f"95% CI: [{np.percentile(boot_p, 2.5):.2e}, {np.percentile(boot_p, 97.5):.2e}]",
            "significance_robust": bool(np.percentile(boot_p, 97.5) < 0.05),
            "n_eff_note": f"n_eff = {len(all_clusters)} AA families (vs nominal n = {len(bio)} bio paths)",
            "conclusion": "Law 3 R1 result is robust to pseudoreplication" if np.percentile(boot_p, 97.5) < 0.05 else "Law 3 R1 result may be partially inflated by pseudoreplication"
        }
    }

    print(f"\n--- Results ---")
    print(f"Original p:        {orig_p:.3e}")
    print(f"Bootstrap p median: {np.median(boot_p):.3e}")
    print(f"Bootstrap p 95% CI: [{np.percentile(boot_p, 2.5):.3e}, {np.percentile(boot_p, 97.5):.3e}]")
    print(f"Fraction p<0.05:   {np.mean(boot_p < 0.05):.4f}")
    print(f"Fraction p<0.01:   {np.mean(boot_p < 0.01):.4f}")
    print(f"Robust: {'YES' if np.percentile(boot_p, 97.5) < 0.05 else 'NO'}")

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
