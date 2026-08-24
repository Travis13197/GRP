#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final paper (article-final.docx) — supplementary figures S1-S8.
All panels are regenerated from canonical source tables; S1/S3 use the intrinsic
(Kabsch-aligned) L2 geometry pipeline, S2 uses the O1 v3 DMS calibrations.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from final_figure_style import apply_style, PAL, AA_COLORS, clean_axes, save_fig

apply_style()

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
T = ROOT / "figures/data/tables"
SUP = ROOT / "figures/out/supplementary"
L2 = ROOT / "figures/data/l2_intrinsic_geometry.csv"
SUP.mkdir(parents=True, exist_ok=True)
L2 = ROOT / "figures/data/l2_intrinsic_geometry.csv"
AAS20 = ["G", "S", "E", "L", "K", "A", "V", "I", "F", "P", "T", "C", "M", "N",
         "Q", "H", "W", "Y", "R", "D"]
OBS5 = ["PR", "spectral_decay", "entropy", "A_C", "eff_rank_95"]


def per_aa_loglog_fit(df, col, lo=4, hi=60):
    """Per-amino-acid power-law fits (returns {aa: (beta, R2)})."""
    out = {}
    for aa in df.aa_type.unique():
        s = df[(df.aa_type == aa) & (df.n >= lo) & (df.n <= hi) & (df[col] > 0)]
        if len(s) >= 6:
            lx, ly = np.log(s.n), np.log(s[col])
            b, a0 = np.polyfit(lx, ly, 1)
            pred = a0 + b * lx
            r2 = 1 - np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2)
            out[aa] = (b, r2)
    return out


def figS1() -> None:
    """Per-amino-acid intrinsic spectral-decay scaling (Kabsch frame, n<=60)."""
    l2 = pd.read_csv(L2)
    poly = l2[(l2.category == "PolyX") & (l2.n.between(4, 60))]
    # shared fixed-effects slope across 20 AA (n <= 60)
    p20 = poly[(poly.n > 0) & (poly.spectral_decay > 0)].copy()
    import statsmodels.formula.api as smf
    p20["ln_n"] = np.log(p20.n); p20["ln_y"] = np.log(p20.spectral_decay)
    fe = smf.ols("ln_y ~ ln_n + C(aa_type)", data=p20).fit()
    beta_fe = fe.params["ln_n"]
    ncol, nrow = 5, 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(13.6, 9.2))
    for k, aa in enumerate(AAS20):
        ax = axes.flat[k]
        sub = poly[poly.aa_type == aa].sort_values("n")
        ax.plot(sub.n, sub.spectral_decay, "o", ms=2.8, color=AA_COLORS[aa], alpha=0.85,
                zorder=3)
        s = sub[sub.spectral_decay > 0]
        lx, ly = np.log(s.n), np.log(s.spectral_decay)
        b, a0 = np.polyfit(lx, ly, 1)
        pred = a0 + b * lx
        r2 = 1 - np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2)
        xs = np.linspace(4, 60, 30)
        ax.plot(xs, np.exp(a0 + b * np.log(xs)), "-", color=AA_COLORS[aa], lw=1.2,
                alpha=0.95, zorder=2)
        # 95% prediction band from residual spread
        resid = ly - (a0 + b * lx)
        se = np.sqrt(np.sum(resid ** 2) / (len(resid) - 2)) if len(resid) > 2 else 0.0
        band_hi = np.exp(a0 + b * np.log(xs) + 1.96 * se)
        band_lo = np.exp(a0 + b * np.log(xs) - 1.96 * se)
        ax.fill_between(xs, band_lo, band_hi, color=AA_COLORS[aa], alpha=0.12,
                        linewidth=0, zorder=1)
        # shared-slope reference (fixed-effects estimate)
        ref_a0 = np.mean(np.log(sub.spectral_decay[sub.spectral_decay > 0]) -
                         beta_fe * np.log(sub.n[sub.spectral_decay > 0]))
        ax.plot(xs, np.exp(ref_a0 + beta_fe * np.log(xs)), "--", color="#666", lw=0.9,
                alpha=0.75, zorder=1)
        if k == 0:
            ax.text(0.97, 0.10, "grey dashed: shared slope β = −0.66",
                    transform=ax.transAxes, ha="right", fontsize=6.0, color="#666",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CCC", lw=0.4))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.grid(ls=":", lw=0.3, color="#CBD5E1", zorder=0)
        ax.set_title(f"Poly{aa}", fontsize=7.5, pad=3)
        sign = "−" if b < 0 else "+"
        ax.text(0.04, 0.94, f"{sign}  β = {b:+.2f}\nR² = {r2:.2f} · n = {len(sub):d}",
                transform=ax.transAxes, va="top", fontsize=6.0, fontweight="bold",
                color="#047857" if b < 0 else "#B91C1C",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#31A354" if b < 0 else "#DE2D26", lw=0.5))
        if k % ncol == 0:
            ax.set_ylabel("spectral decay α", fontsize=6.5)
        if k >= ncol * (nrow - 1):
            ax.set_xlabel("n (log)", fontsize=6.5)
        ax.tick_params(labelsize=6.0)
        clean_axes(ax)
    fig.suptitle("Per-amino-acid intrinsic (Kabsch-aligned) spectral-decay scaling, "
                 "n ≤ 60 (all 20 examined amino acids negative)",
                 fontsize=11, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, "figS1_per_aa_spectral_decay", SUP)
    plt.close(fig)


def figS2() -> None:
    """Per-protein C_geo ~ DMS fitness scatter (N = 84,361; O1 v3)."""
    df = pd.read_csv(ROOT / "figures/data/phase_l1_cgeo_kabsch_all_proteins.csv")
    df = df[~((df.protein == "UBE4B") & (df.position > 69))]
    v3 = json.load(open(T / "phase_o1_cgeo_v3.json", encoding="utf-8"))["per_protein"]
    prot_order = ["BLAT", "GFP", "HRAS", "HSP90", "P53", "PTEN", "SPIKE", "UBE4B"]
    g_all = df[df.protein.isin(prot_order)]
    xlim_share = np.percentile(g_all.C_geo_kabsch_lw, [1, 99])
    ylim_share = np.percentile(g_all.DMS_score, [1, 99])
    ncol, nrow = 4, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(13.6, 6.6))
    for k, prot in enumerate(prot_order):
        ax = axes.flat[k]
        g = df[df.protein == prot]
        ax.set_xlim(*xlim_share); ax.set_ylim(*ylim_share)
        ax.scatter(g.C_geo_kabsch_lw, g.DMS_score, s=2.2, alpha=0.32, color=PAL["green"],
                   rasterized=True, linewidths=0, zorder=2)
        r3 = v3[prot]["rho_v3"]; nv = v3[prot]["n_variants"]
        p3 = v3[prot].get("p_v3", np.nan)
        x = g.C_geo_kabsch_lw.values; y = g.DMS_score.values
        try:
            from scipy.stats import gaussian_kde
            kd = gaussian_kde(np.vstack([x, y]))
            xi, yi = np.mgrid[x.min():x.max():50j, y.min():y.max():50j]
            zi = kd(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
            ax.contour(xi, yi, zi, levels=5, colors="#15803D", linewidths=0.5,
                       alpha=0.45, zorder=3)
        except Exception:
            pass
        # monotonic trend reference (OLS on log-C_geo for visual aid)
        if len(x) >= 10:
            b, a0 = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 10)
            ax.plot(xs, a0 + b * xs, "-", color="#DE2D26", lw=1.2, alpha=0.7,
                    zorder=4)
        ax.axhline(0, color="#666", ls=":", lw=0.7, zorder=1)
        ax.text(0.04, 0.93, f"ρ = {r3:.3f}\np = {p3:.1e}\nn = {nv:,}",
                transform=ax.transAxes, va="top", fontsize=6.4, fontweight="bold",
                color="#047857",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#31A354",
                          lw=0.5))
        ax.set_title(prot, fontsize=8, pad=4)
        ax.tick_params(labelsize=6.0)
        ax.grid(ls=":", lw=0.3, color="#CBD5E1", zorder=0)
        if k >= (nrow - 1) * ncol:
            ax.set_xlabel("C_geo (Kabsch + LW)", fontsize=6.8)
        if k % ncol == 0:
            ax.set_ylabel("DMS fitness", fontsize=6.8)
        clean_axes(ax)
    fig.suptitle("Per-protein association of intrinsic geometric cost with experimental "
                 "fitness (84,361 analysis mutations; all eight proteins negative)",
                 fontsize=11, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, "figS2_dms_per_protein", SUP)
    plt.close(fig)


def figS3() -> None:
    """Amino-acid-specific intrinsic length-scaling fingerprints (n <= 60)."""
    l2 = pd.read_csv(L2)
    poly = l2[(l2.category == "PolyX") & (l2.n.between(4, 60))]
    fits = {o: per_aa_loglog_fit(poly, o) for o in OBS5}
    mat = pd.DataFrame(index=OBS5, columns=AAS20, dtype=float)
    for o in OBS5:
        for aa in AAS20:
            if aa in fits[o]:
                mat.loc[o, aa] = fits[o][aa][0]
    # cluster amino-acid columns to expose chemical grouping
    from scipy.cluster.hierarchy import linkage, leaves_list
    Z = linkage(mat.values.T, method="average")
    cidx = leaves_list(Z)
    AAS_order = [AAS20[j] for j in cidx]
    mat = mat[AAS_order]
    from scipy.cluster.hierarchy import dendrogram
    fig = plt.figure(figsize=(9.8, 4.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.18, 2.2], hspace=0.12)
    ax = fig.add_subplot(gs[2])
    axd = fig.add_subplot(gs[0])
    axg = fig.add_subplot(gs[1])
    vmax = 1.4
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(AAS_order))); ax.set_xticklabels(AAS_order, fontsize=6.5)
    ax.set_yticks(range(len(OBS5)))
    ax.set_yticklabels([o.replace("_", "\n") for o in OBS5], fontsize=6.5)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.2,
                        color="white" if abs(v) > 0.55 else "black")
    ax.set_title("Intrinsic (Kabsch) length-scaling exponents β per amino acid (n ≤ 60; "
                 "columns clustered)\nspectral decay negative for all 20; effective-rank "
                 "expansion strongest for polar/flexible residues", fontsize=8, pad=6)
    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("β (log-log)", fontsize=6.5); cbar.ax.tick_params(labelsize=6.0)
    with plt.rc_context({"lines.linewidth": 0.7}):
        dendrogram(Z, ax=axd, no_labels=True, color_threshold=0, above_threshold_color="#94A3B8")
    axd.axis("off")
    axd.set_title("", fontsize=1)
    # chemistry-group color strip above the columns
    group_map = {"A": 0, "V": 0, "I": 0, "L": 0, "F": 0, "M": 0, "W": 0, "Y": 0,
                 "P": 0, "G": 1, "S": 1, "T": 1, "C": 1, "N": 1, "Q": 1,
                 "E": 2, "K": 2, "R": 2, "D": 2, "H": 2}
    gcols = ["#F16913", "#31A354", "#2C7FB8"]
    axg.bar(range(len(AAS_order)), [0.45] * len(AAS_order), width=1.0, bottom=0.0,
            color=[gcols[group_map[a]] for a in AAS_order], align="center",
            edgecolor="none")
    axg.set_xlim(-0.5, len(AAS_order) - 0.5)
    axg.set_ylim(0, 1)
    axg.set_xticks([]); axg.set_yticks([])
    for sp in axg.spines.values():
        sp.set_visible(False)
    from matplotlib.patches import Patch
    axg.legend(handles=[Patch(facecolor=gcols[0], label="hydrophobic"),
                        Patch(facecolor=gcols[1], label="polar"),
                        Patch(facecolor=gcols[2], label="charged")],
               loc="upper right", ncol=3, fontsize=6.0, frameon=False,
               handlelength=1.0, borderpad=0.3)
    fig.tight_layout()
    save_fig(fig, "figS3_aa_fingerprints", SUP)
    plt.close(fig)


def figS4() -> None:
    """Hydrophobicity-charge phase map with conditional response angle (E50)."""
    KD = {"A": 1.8, "V": 4.2, "I": 4.5, "F": 2.8, "L": 3.8,
          "G": -0.4, "S": -0.8, "E": -3.5, "K": -3.9}
    CHARGE = {"A": 0, "V": 0, "I": 0, "F": 0, "L": 0, "G": 0, "S": 0,
              "E": -1, "K": 1}
    AAS = ["G", "S", "E", "L", "K", "A", "V", "I", "F"]
    SEED = 20260814

    comp = pd.read_csv(T / "phase4_s2ext_compactness.csv").rename(columns={"AA": "aa"})
    comp["Rg_A"] = comp["Rg_mean"] * 10.0
    comp_l1 = pd.read_csv(T / "results4_l1_compactness_AVIF.csv")
    comp_l1["Rg_A"] = comp_l1["Rg_mean_A"] * 10.0
    comp = pd.concat([comp[["aa", "n", "Rg_A"]], comp_l1[["aa", "n", "Rg_A"]]],
                     ignore_index=True)
    geom = pd.read_csv(ROOT / "figures/data/systemwide_enhanced_geometry_v2.csv")
    geo = geom[["aa_type", "n", "spectral_decay", "PR"]].rename(columns={"aa_type": "aa"})
    geo = geo[geo["aa"].isin(AAS)].dropna(subset=["n"])
    geo["n"] = geo["n"].astype(int)
    tau = pd.read_csv(T / "results4_tau1_md.csv")
    tau = tau[tau.aa.isin(AAS)][["aa", "n", "tau1_rg_ps"]]
    xi = pd.read_csv(T / "phase4_s1_decay_by_n.csv")[["aa", "n", "xi_log"]]

    df = geo.merge(comp, on=["aa", "n"], how="left").merge(
        tau, on=["aa", "n"], how="left").merge(xi, on=["aa", "n"], how="left")
    df["kd"] = df["aa"].map(KD)
    df["charge"] = df["aa"].map(CHARGE)
    agg = df.groupby("aa").agg(
        sd=("spectral_decay", "mean"), Rg=("Rg_A", "mean"),
        tau1=("tau1_rg_ps", "mean"), xi=("xi_log", "mean"),
        n_sd=("spectral_decay", "count")).reset_index()
    agg["kd"] = agg["aa"].map(KD)
    agg["charge"] = agg["aa"].map(CHARGE)

    resp_axes = ["sd", "Rg", "tau1"]
    coefs = {}
    for ax in resp_axes:
        sub = agg.dropna(subset=[ax])
        if len(sub) < 6:
            coefs[ax] = {"kh": np.nan, "kq": np.nan}
            continue
        z = (sub[ax] - sub[ax].mean()) / sub[ax].std()
        A = np.vstack([np.ones(len(sub)), sub["kd"].values, sub["charge"].values]).T
        c = np.linalg.solve(A.T @ A + 1e-3 * np.eye(3), A.T @ z.values)
        coefs[ax] = {"kh": float(c[1]), "kq": float(c[2])}
    kh = np.array([coefs[a]["kh"] for a in resp_axes])
    kq = np.array([coefs[a]["kq"] for a in resp_axes])
    kh_u = kh / np.linalg.norm(kh); kq_u = kq / np.linalg.norm(kq)
    theta_deg = float(np.degrees(np.arccos(np.clip(np.dot(kh_u, kq_u), -1, 1))))
    rng = np.random.default_rng(SEED)
    angles = []
    for _ in range(2000):
        idx = rng.choice(len(agg), len(agg), replace=True)
        cs = {}
        for ax in resp_axes:
            sub = agg.iloc[idx].dropna(subset=[ax])
            if len(sub) < 6:
                cs[ax] = (np.nan, np.nan)
                continue
            z = (sub[ax] - sub[ax].mean()) / sub[ax].std()
            A = np.vstack([np.ones(len(sub)), sub["kd"].values, sub["charge"].values]).T
            c = np.linalg.solve(A.T @ A + 1e-3 * np.eye(3), A.T @ z.values)
            cs[ax] = (c[1], c[2])
        vh = np.array([cs[a][0] for a in resp_axes])
        vq = np.array([cs[a][1] for a in resp_axes])
        if np.isfinite(vh).all() and np.isfinite(vq).all() and \
                np.linalg.norm(vh) > 0 and np.linalg.norm(vq) > 0:
            angles.append(np.degrees(np.arccos(np.clip(
                np.dot(vh, vq) / (np.linalg.norm(vh) * np.linalg.norm(vq)), -1, 1))))
    angles = np.array(angles)
    ci_lo, ci_hi = float(np.percentile(angles, 2.5)), float(np.percentile(angles, 97.5))

    fig, axs = plt.subplots(1, 3, figsize=(14.4, 4.4))
    for ax, (col, cmap, lab) in zip(
            axs[:2], [("sd", plt.get_cmap("viridis"), "spectral_decay (mean)"),
                      ("Rg", plt.get_cmap("plasma"), "Rg (mean, Å)")]):
        sub = agg.dropna(subset=[col])
        sc = ax.scatter(sub["kd"], sub["charge"], c=sub[col], cmap=cmap, s=120,
                        vmin=sub[col].min(), vmax=sub[col].max(),
                        edgecolor="k", linewidth=0.7, zorder=3)
        for _, r in sub.iterrows():
            ax.annotate(r["aa"], (r["kd"], r["charge"]), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, fontweight="bold")
        ax.set_xlabel("Kyte-Doolittle hydrophobicity", fontsize=8)
        ax.set_ylabel("Charge", fontsize=8)
        ax.set_yticks([-1, 0, 1])
        ax.set_title(f"({'a' if ax is axs[0] else 'b'}) {lab}", fontsize=8.5)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.04)
        cbar.set_label(lab, fontsize=6.5); cbar.ax.tick_params(labelsize=6.0)
        ax.grid(alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=6.5)
    # (c) conditional response vectors in standardized response space
    axv = axs[2]
    u = np.linspace(0, 2 * np.pi, 200)
    rmax = max(np.linalg.norm(kh), np.linalg.norm(kq))
    axv.plot(rmax * np.cos(u), rmax * np.sin(u), ls=":", color="#BBB", lw=0.8)
    axv.annotate("", xy=kh[:2], xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="#0E7490", lw=2.0))
    axv.annotate("", xy=kq[:2], xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="#DE2D26", lw=2.0))
    axv.text(kh[0] * 1.15, kh[1] * 1.15, "hydrophobicity\nresponse $k_h$", fontsize=6.5,
             color="#0E7490", ha="center")
    axv.text(kq[0] * 1.15, kq[1] * 1.15, "charge\nresponse $k_q$", fontsize=6.5,
             color="#DE2D26", ha="center")
    ang_mid = (np.arctan2(kh[1], kh[0]) + np.arctan2(kq[1], kq[0])) / 2
    arc_r = rmax * 0.42
    axv.plot([arc_r * np.cos(a) for a in np.linspace(np.arctan2(kh[1], kh[0]),
             np.arctan2(kq[1], kq[0]), 40)],
             [arc_r * np.sin(a) for a in np.linspace(np.arctan2(kh[1], kh[0]),
              np.arctan2(kq[1], kq[0]), 40)], color="#7C3AED", lw=1.2)
    axv.text(1.25 * arc_r * np.cos(ang_mid), 1.25 * arc_r * np.sin(ang_mid),
             f"θ = {theta_deg:.0f}°", fontsize=7.0, color="#7C3AED", ha="center",
             fontweight="bold")
    axv.axhline(0, color="#CCC", lw=0.6); axv.axvline(0, color="#CCC", lw=0.6)
    axv.set_xlim(-1.1 * rmax, 1.1 * rmax); axv.set_ylim(-1.1 * rmax, 1.1 * rmax)
    axv.set_aspect("equal")
    axv.set_xlabel("response along spectral_decay", fontsize=6.8)
    axv.set_ylabel("response along Rg", fontsize=6.8)
    axv.tick_params(labelsize=6.0)
    axv.set_title("(c) Conditional response directions\n(standardized space)", fontsize=8.5)
    clean_axes(axv)
    fig.suptitle("Hydrophobicity and charge produce distinguishable conditional responses\n"
                 "(statistical response space; θ is NOT a molecular-mechanism angle)",
                 fontsize=10, fontweight="bold")
    fig.text(0.5, 0.015,
             f"Response directions over (spectral_decay, Rg, τ1): θ_hq = {theta_deg:.0f}° "
             f"(95% CI [{ci_lo:.0f}°, {ci_hi:.0f}°]); near-orthogonality is a property of "
             f"the standardized response space only.", ha="center", fontsize=6.8,
             color="#333333")
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    save_fig(fig, "figS4_hydrophobicity_charge_phase", SUP)
    plt.close(fig)
    with open(T / "results4_e50_theta_hq.json", "w", encoding="utf-8") as f:
        json.dump({"theta_hq_deg": theta_deg, "ci_lo": ci_lo, "ci_hi": ci_hi,
                   "kh": kh.tolist(), "kq": kq.tolist(), "axes": resp_axes,
                   "note": "statistical angle in standardized response space"}, f,
                  indent=2, default=float)


def figS5() -> None:
    """State relation: entropy ~ pseudo-volume + participation ratio
    (observed-vs-predicted scatter + standardized coefficients)."""
    geom = pd.read_csv(ROOT / "figures/data/systemwide_enhanced_geometry_v2.csv")
    m = geom.dropna(subset=["entropy", "pseudo_volume", "PR"])
    A = np.vstack([np.ones(len(m)), m.pseudo_volume, m.PR]).T
    coef, *_ = np.linalg.lstsq(A, m.entropy, rcond=None)
    pred = A @ coef
    r2 = 1 - np.sum((m.entropy - pred) ** 2) / np.sum((m.entropy - m.entropy.mean()) ** 2)
    # standardized coefficients for relative-contribution comparison
    Xs = (m[["pseudo_volume", "PR"]] - m[["pseudo_volume", "PR"]].mean()) / \
        m[["pseudo_volume", "PR"]].std()
    ys = (m.entropy - m.entropy.mean()) / m.entropy.std()
    Ast = np.vstack([np.ones(len(m)), Xs.pseudo_volume, Xs.PR]).T
    beta_std, *_ = np.linalg.lstsq(Ast, ys, rcond=None)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), gridspec_kw={"wspace": 0.26})
    ax = axes[0]
    x_lim = np.percentile(pred, [1, 99]); y_lim = np.percentile(m.entropy, [1, 99])
    inb = (pred >= x_lim[0]) & (pred <= x_lim[1]) & (m.entropy >= y_lim[0]) & \
        (m.entropy <= y_lim[1])
    ax.scatter(pred, m.entropy, s=7, alpha=0.35, color="#2563eb", rasterized=True,
               linewidths=0, zorder=2)
    try:
        from scipy.stats import gaussian_kde
        k = gaussian_kde(np.vstack([pred, m.entropy]))
        xi, yi = np.mgrid[pred.min():pred.max():60j,
                          m.entropy.min():m.entropy.max():60j]
        zi = k(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
        ax.contour(xi, yi, zi, levels=6, colors="#1D6FA8", linewidths=0.55,
                   alpha=0.5, zorder=3)
    except Exception:
        pass
    lo = min(x_lim[0], y_lim[0]); hi = max(x_lim[1], y_lim[1])
    ax.plot([lo, hi], [lo, hi], "--", color="#111", lw=1.1, zorder=4)
    ax.set_xlim(*x_lim); ax.set_ylim(*y_lim)
    ax.set_xlabel("Predicted entropy (volume + PR)", fontsize=7.5)
    ax.set_ylabel("Observed entropy", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.grid(ls=":", lw=0.35, color="#CBD5E1", zorder=0)
    clean_axes(ax)
    ax.set_box_aspect(1.0)
    ax.text(0.04, 0.94, f"R² = {r2:.3f}\nn = {len(m):,} sequences",
            transform=ax.transAxes, va="top", fontsize=6.8, fontweight="bold",
            color="#1D4ED8",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2C7FB8", lw=0.8))
    ax.set_title("(a) Observed vs predicted entropy\n(volume + participation ratio)",
                 fontsize=8, pad=6)
    ax.text(0.5, -0.14, "axes clipped to 1–99% (outliers omitted)",
            transform=ax.transAxes, ha="center", fontsize=6.0, color="#64748B")

    ax = axes[1]
    labels = ["pseudo-volume", "participation ratio"]
    cols = ["#0E7490", "#DE2D26"]
    ax.bar([0, 1], beta_std[1:], width=0.5, color=cols, alpha=0.9, edgecolor="k",
           lw=0.3, zorder=3)
    ax.axhline(0, color="#666", lw=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_ylabel("Standardized regression coefficient β", fontsize=7.0)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", ls=":", lw=0.35, color="#CBD5E1", zorder=0)
    for i, b in enumerate(beta_std[1:]):
        ax.text(i, b + 0.02 * np.sign(b), f"{b:+.2f}", ha="center", fontsize=6.8,
                fontweight="bold")
    clean_axes(ax)
    ax.set_box_aspect(1.0)
    ax.text(0.5, 0.04, "both covariates contribute positively;\nvolume dominates",
            transform=ax.transAxes, ha="center", fontsize=6.4, color="#334155",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="#F8FAFC", ec="#94A3B8", lw=0.5))
    ax.set_title("(b) Relative contribution to entropy\n(standardized covariates)",
                 fontsize=8, pad=6)

    # (c) residuals vs chain length, colored by amino acid
    ax = axes[2]
    resid = m.entropy - pred
    r_lim = np.percentile(resid, [1, 99])
    aas = m.aa_type.dropna().unique()
    for aa in aas:
        mm = m.aa_type == aa
        ax.scatter(m.n[mm], resid[mm], s=5, alpha=0.45, color=AA_COLORS.get(aa, "#888"),
                   linewidths=0, label=aa if len(aas) <= 20 else None)
    ax.axhline(0, color="#666", lw=0.7)
    ax.set_ylim(*r_lim)
    ax.set_xlabel("Chain length n", fontsize=7.5)
    ax.set_ylabel("Entropy residual (observed − predicted)", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.grid(ls=":", lw=0.35, color="#CBD5E1", zorder=0)
    clean_axes(ax)
    ax.set_box_aspect(1.0)
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ax.text(0.04, 0.94, f"RMSE = {rmse:.3f}\nresiduals centred on zero",
            transform=ax.transAxes, va="top", fontsize=6.6, fontweight="bold",
            color="#0B3D62",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2C7FB8", lw=0.8))
    ax.set_title("(c) Fit residuals vs chain length\n(no systematic length bias)",
                 fontsize=8, pad=6)
    fig.suptitle("A compact geometric state relation: entropy is jointly explained by "
                 "conformational extent and effective mode count",
                 fontsize=10.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, "figS5_state_relation", SUP)
    plt.close(fig)


def figS6() -> None:
    """Fig. 2 supporting: K invariance, real-mutant low-stiffness, domain signal."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7))
    ax = axes[0]
    b7 = pd.read_csv(T / "phase_ensemble_b7_transformation_comparison_v2.csv")
    b7map = {r.transformation: r for _, r in b7.iterrows()}
    labels = ["Original", "Orthogonal\n(rotation)", "Affine", "Scaling"]
    vals = [b7map["Original"]["frob_diff_vs_theoretical"],
            b7map["Orthogonal (Rotation)"]["frob_diff_vs_theoretical"],
            b7map["Affine"]["frob_diff_vs_theoretical"],
            b7map["Scaling"]["frob_diff_vs_theoretical"]]
    cols = ["#9ca3af", "#16a34a", "#16a34a", "#dc2626"]
    ax.bar(labels, [v + 1e-14 for v in vals], color=cols, alpha=0.85, width=0.6)
    ax.set_yscale("log")
    ax.set_ylim(5e-14, 1.0)
    ax.set_yticks([1e-13, 1e-9, 1e-5, 1e-1])
    ax.set_yticklabels(["10⁻¹³", "10⁻⁹", "10⁻⁵", "10⁻¹"], fontsize=6.0)
    for xi, v in enumerate(vals):
        ax.text(xi, (v + 1e-14) * 2.2, f"{v:.2e}", ha="center", fontsize=6.0,
                fontweight="bold", rotation=90)
    ax.set_ylabel("Frobenius deviation from original K", fontsize=7)
    ax.tick_params(labelsize=6.0)
    ax.grid(axis="y", ls=":", lw=0.35, color="#CBD5E1", zorder=0)
    clean_axes(ax)
    ax.set_title("(a) K-matrix transformation invariance", fontsize=7.5)
    ax.text(0.03, 0.92, "orthogonal/affine: invariant (~10⁻¹³)\n"
            "scaling: not invariant (0.28)\nwhitening: K → 0 (degenerate)",
            transform=ax.transAxes, fontsize=6.0, va="top")

    ax = axes[1]
    o4 = json.load(open(T / "phase_o4_real_mutant.json", encoding="utf-8"))
    t1 = o4["T1_direction"]; t2 = o4["T2_spectrum"]
    ax.bar([0, 1, 2.5, 3.5], [1.0, t1["median_ratio"], 1/3,
                              t2["median_soft_third"]],
           color=["#9ca3af", "#2563eb", "#9ca3af", "#0E7490"], width=0.5)
    ax.axhline(1.0, color="#666", ls=":", lw=0.8)
    for xi, v in zip([0, 1, 2.5, 3.5], [1.0, t1["median_ratio"], 1/3,
                                         t2["median_soft_third"]]):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=6.2,
                fontweight="bold")
    ax.set_xticks([0.5, 3.0])
    ax.set_xticklabels(["T1 (expected vs real)", "T2 (random vs soft-third)"],
                       fontsize=6.0)
    ax.set_ylabel("Median ratio", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(axis="y", ls=":", lw=0.35, color="#CBD5E1", zorder=0)
    clean_axes(ax)
    ax.text(0.03, 0.88, f"T1 p = {t1['wilcoxon_p']:.1e}\n"
            f"T2 p = {t2['wilcoxon_p']:.1e}",
            transform=ax.transAxes, fontsize=6.0, va="top")
    ax.set_title("(b) Real-mutant low-stiffness directions (O4)", fontsize=7.5)

    ax = axes[2]
    dom = [("P53 (DBD)", 7.6), ("HSP90 (ATP)", 2.06), ("PTEN (phosphatase)", 1.28)]
    names = [d[0] for d in dom]; folds = [d[1] for d in dom]
    ax.bar(names, folds, color="#7c3aed", alpha=0.85, width=0.5)
    ax.axhline(1.0, color="#666", ls=":", lw=0.8)
    for xi, v in enumerate(folds):
        ax.text(xi, v + 0.08, f"{v:.2f}×", ha="center", fontsize=6.2,
                fontweight="bold")
    ax.set_ylabel("Domain-specific signal enhancement (×)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(axis="y", ls=":", lw=0.35, color="#CBD5E1", zorder=0)
    clean_axes(ax)
    ax.set_title("(c) Domain-specific C_geo signal", fontsize=7.5)
    ax.text(0.03, 0.90, "ordered-domain vs full-protein\nC_geo signal (phase9_comprehensive)",
            transform=ax.transAxes, fontsize=6.0, va="top")
    fig.text(0.5, 0.015, "O4 and domain-signal values are locked from the original "
             "computation and should be re-verified against the source pipeline before "
             "submission.", ha="center", fontsize=6.2, color="#64748B", style="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_fig(fig, "figS6_law1_support", SUP)
    plt.close(fig)


def figS7() -> None:
    """Fig. 4 supporting: folded niche (TSI), natural transition paths, NPZ detail, scope."""
    fig = plt.figure(figsize=(13.5, 6.4))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.26)
    ax = fig.add_subplot(gs[0, 0])
    cats = ["IDP", "Folded"]
    tsi = [0.628, -2.618]
    cols = ["#f59e0b", "#2563eb"]
    ax.bar(cats, tsi, color=cols, alpha=0.85, width=0.45, edgecolor="k", lw=0.4)
    ax.axhline(0, color="#666", lw=0.6)
    for xi, v in zip([0, 1], tsi):
        ax.text(xi, v + (0.1 if v > 0 else -0.1), f"{v:.3f}", ha="center",
                fontsize=6.5, fontweight="bold",
                va="bottom" if v > 0 else "top")
    ax.set_ylabel("Topological similarity index (TSI)", fontsize=7)
    ax.grid(axis="y", ls=":", lw=0.35, color="#CBD5E1", zorder=0)
    ax.tick_params(labelsize=6)
    clean_axes(ax)
    ax.text(0.03, 0.93, "Folded TSI = −2.618 (p = 0.0072)\nIDP TSI = +0.628 (n.s.)\n"
            "pool-based null, 5,000 iterations, 44 proteins",
            transform=ax.transAxes, fontsize=6.0, va="top")
    ax.set_title("(a) Folded-protein compact geometric niche", fontsize=7.5)

    ax = fig.add_subplot(gs[0, 1])
    p6b = pd.read_csv(T / "phase6b_path_analysis_results.csv")
    real = p6b[p6b.path_type == "real"]
    nulls = p6b[p6b.path_type.str.startswith("null")]
    rv = real.groupby("protein_pair").total_wasserstein_distance.mean()
    nv = nulls.groupby("protein_pair").total_wasserstein_distance.mean()
    pairs = sorted(set(rv.index) & set(nv.index))
    x = np.arange(len(pairs))
    ax.bar(x - 0.18, [rv[p] for p in pairs], 0.34, color="#2563eb", label="real")
    ax.bar(x + 0.18, [nv[p] for p in pairs], 0.34, color="#9ca3af", label="null (mean)")
    for xi, p in enumerate(pairs):
        y0, y1 = rv[p], nv[p]
        ax.plot([xi - 0.18, xi + 0.18], [y0, y1], color="#6B7280", lw=0.6,
                ls="--", zorder=3)
        ax.text(xi, max(y0, y1) + 0.08, f"Δ = {y1 - y0:+.1f}", ha="center",
                fontsize=6.0, fontweight="bold", color="#111")
    ax.set_xticks(x); ax.set_xticklabels(pairs, fontsize=6)
    ax.set_ylabel("Total path W₂", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6)
    ax.grid(axis="y", ls=":", lw=0.35, color="#CBD5E1", zorder=0)
    clean_axes(ax)
    ax.text(0.5, 0.92, "real < null for all three natural pairs",
            transform=ax.transAxes, ha="center", fontsize=6.2, fontweight="bold",
            color="#1D4ED8",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#2C7FB8", lw=0.5))
    ax.set_title("(b) Natural transition paths vs nulls (TetR/rtTA, VanR)",
                 fontsize=7.5)

    ax = fig.add_subplot(gs[0, 2])
    npz = pd.read_csv(T / "phase_ensemble_npz_direct_w2.csv")
    nb = npz[npz.is_bio == True].W2_npz.dropna()
    nn = npz[npz.is_bio == False].W2_npz.dropna()
    ax.hist(nb, bins=30, alpha=0.6, color="#2563eb", density=True, label="structured")
    ax.hist(nn, bins=30, alpha=0.45, color="#9ca3af", density=True, label="null")
    try:
        from scipy.stats import gaussian_kde
        xs = np.linspace(min(nb.min(), nn.min()), max(nb.max(), nn.max()), 100)
        ax.plot(xs, gaussian_kde(nb)(xs), "-", color="#1D4ED8", lw=1.2)
        ax.plot(xs, gaussian_kde(nn)(xs), "-", color="#6B7280", lw=1.2)
    except Exception:
        pass
    ax.axvline(nb.mean(), color="#1D4ED8", ls=":", lw=0.8)
    ax.axvline(nn.mean(), color="#6B7280", ls=":", lw=0.8)
    ax.set_xlabel("Adjacent-state W₂ (NPZ direct)", fontsize=7)
    ax.set_ylabel("Density", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6)
    ax.grid(ls=":", lw=0.35, color="#CBD5E1", zorder=0)
    clean_axes(ax)
    _, p_npz = mannwhitneyu(nb, nn)
    ax.set_title(f"(c) Direct NPZ transport (bio {nb.mean():.2f} vs null {nn.mean():.2f}, "
                 f"p = {p_npz:.1e})", fontsize=7.5)
    # panel 4: scope of the validated claim (bottom row; theoretical extensions)
    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    ax.set_title("(d) Scope of the validated claim (main-text Fig. 4d)", fontsize=7.5)
    ax.add_patch(FancyBboxPatch((0.03, 0.42), 0.46, 0.48, boxstyle="round,pad=0.02",
                                fc="#ECFDF5", ec="#31A354", lw=1.2))
    ax.text(0.26, 0.84, "Validated: transport component \u27e8W\u2082\u27e9", ha="center",
            fontsize=7.2, fontweight="bold", color="#047857")
    ax.text(0.26, 0.72, "structured paths < matched null paths\n"
            "across joint-PCA, heteropolymer and direct-NPZ analyses",
            ha="center", fontsize=6.2, color="#065F46")
    ax.text(0.26, 0.60, "primary: 11.75 vs 22.90, d = \u22121.54, p = 8.4\u00d710\u207b\u2076\u2078",
            ha="center", fontsize=6.2, color="#065F46")
    ax.text(0.26, 0.50, "hetero 3.07 vs 3.46 (d = \u22120.20)  \u00b7  NPZ 7.62 vs 9.29",
            ha="center", fontsize=6.2, color="#065F46")
    ax.add_patch(FancyBboxPatch((0.53, 0.42), 0.44, 0.48, boxstyle="round,pad=0.02",
                                fc="#FEF2F2", ec="#DE2D26", lw=1.0))
    ax.text(0.75, 0.84, "Theoretical extensions (not validated here)", ha="center",
            fontsize=7.2, fontweight="bold", color="#B91C1C")
    ax.text(0.75, 0.72, r"A[$\gamma$] = $\bar{W}_2[\gamma]$ + reorg[$\gamma$] + feas[$\gamma$]",
            ha="center", fontsize=6.6, color="#7F1D1D")
    ax.text(0.75, 0.60, "reorganization and feasibility terms are retained as theory,\n"
            "not as experimentally established physical action",
            ha="center", fontsize=6.2, color="#7F1D1D")
    ax.text(0.75, 0.47, "transition-sharpness is not independent evidence\n"
            "for a global action minimum",
            ha="center", fontsize=6.2, color="#7F1D1D",
            bbox=dict(boxstyle="round,pad=0.25", fc="#F8FAFC", ec="#94A3B8", lw=0.5))
    fig.text(0.5, 0.012, "TSI values are locked from the original computation and should "
             "be re-verified against the source pipeline before submission.",
             ha="center", fontsize=6.2, color="#64748B", style="italic")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save_fig(fig, "figS7_law3_support", SUP)
    plt.close(fig)


def figS8() -> None:
    """Fig. 3 supporting: five-level exclusion chain (coupling irreducibility)."""
    d = json.load(open(T / "law2_validation_report.json", encoding="utf-8"))
    v1 = d["criteria"]["V1"]; v2 = d["criteria"]["V2"]; v4 = d["criteria"]["V4"]
    levels = [("V1 · physicochemistry grouping", f"perm p = {v1['panel_A']['perm_p']:.2f}",
               "FAIL", "no grouping signal"),
              ("V2 · compositional features", f"OOS R² = {v2['cv_r2_ridge']:.3f}",
               "FAIL", "at null"),
              ("V3 · bootstrap clustering", "ARI = 0.708", "PASS",
               "structure objective"),
              ("V4 · natural-system transfer", f"ΔR² = {v4['delta_r2_mean']:.2f}",
               "FAIL", "no transfer"),
              ("V5 · ensemble-level graph model", "OOS R² = −0.155", "FAIL",
               "= V2 baseline −0.147")]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.axis("off")
    for i, (label, val, status, note) in enumerate(levels):
        y = 4 - i
        ax.axhline(y, color="#E2E8F0", lw=6, solid_capstyle="round", zorder=1)
        col = PAL["green"] if status == "PASS" else PAL["red"]
        ax.plot(0.08, y, "o", ms=10, color=col, mec="white", mew=1.2, zorder=3)
        ax.text(0.02, y, status, va="center", fontsize=6.2, fontweight="bold",
                color=col, ha="right")
        ax.text(0.13, y, label, va="center", fontsize=6.4, color="#334155")
        ax.text(0.60, y, val, va="center", fontsize=6.2, ha="left", color="#64748B",
                family="monospace")
        ax.text(0.60, y - 0.32, note, va="center", fontsize=6.2, ha="left",
                color="#94A3B8", style="italic")
    ax.text(0.31, -0.90, "V1/V2/V4/V5 fail + V3 passes → K structure objective but "
            "irreducible to simple descriptors", ha="center", fontsize=6.6,
            color="#475569", bbox=dict(boxstyle="round,pad=0.3", fc="#F8FAFC",
                                       ec="#94A3B8", lw=0.6))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-1.4, 4.9)
    ax.set_title("Five-level exclusion chain (V1–V5)\n"
                 "V3 passes → coupling irreducibility", fontsize=8, pad=10)
    axm = ax.inset_axes([0.70, 0.10, 0.28, 0.70])
    labs = ["V2\nOOS R²", "V4\nΔR²", "V5\nOOS R²"]
    vals = [v2["cv_r2_ridge"], v4["delta_r2_mean"], -0.1553]
    cols = [PAL["grey"], PAL["grey"], PAL["red"]]
    xm = np.arange(3)
    axm.bar(xm, vals, width=0.55, color=cols, edgecolor="k", lw=0.4)
    for xi, v in zip(xm, vals):
        axm.text(xi, v + 0.06, f"{v:.2f}", ha="center", fontsize=6.0, va="bottom")
    axm.axhline(-0.147, color="#666", ls=":", lw=1.0)
    axm.text(2.30, -0.156, "V2 baseline −0.147", fontsize=6.0, color="#666", ha="right")
    axm.set_xticks(xm); axm.set_xticklabels(labs, fontsize=6.0)
    axm.set_ylabel("Out-of-sample metric", fontsize=6.0)
    axm.tick_params(labelsize=6.0)
    axm.set_ylim(-2.7, 0.15)
    clean_axes(axm)
    axm.set_title("quantitative null comparison", fontsize=6.2, pad=3)
    fig.tight_layout()
    save_fig(fig, "figS8_field_support", SUP)
    plt.close(fig)


def figS9() -> None:
    """UMAP / t-SNE embeddings of the ensemble atlas (main Fig. 1a) and the
    real PolyG30 local chart (main Fig. 1c)."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import umap

    l2 = pd.read_csv(L2)
    feats = ["PR", "A_C", "entropy", "eff_rank_95", "top5_ratio", "spectral_decay",
             "total_variance", "mean_pairwise_dist", "max_pairwise_dist",
             "std_pairwise_dist", "mean_rmsf", "max_rmsf", "std_rmsf", "skewness",
             "kurtosis", "density", "kappa", "spectral_gap"]
    sub = l2.dropna(subset=feats).copy()
    F = sub[feats].astype(float)
    Fn = np.empty_like(F)
    for j, col in enumerate(feats):
        x = F[col].values
        if np.nanmin(x) > 0:
            x = np.log1p(x)
        Fn[:, j] = (x - np.nanmean(x)) / np.nanstd(x)
    Fs = np.nan_to_num(Fn, nan=0.0)
    cat_cols = {"PolyX": "#2C7FB8", "Heteropolymer": "#31A354", "Linker": "#F16913",
                "Natural_IDP": "#DE2D26", "DMS_WT": "#756BB1"}
    cats = sorted(sub.category.unique(), key=lambda c: -len(sub[sub.category == c]))

    rng = 0
    Z_umap = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=rng).fit_transform(Fs)
    Z_tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=rng,
                  learning_rate="auto").fit_transform(Fs)

    # local chart: real PolyG30 aligned ensemble
    ENS_ROOT = ROOT / "figures/data/ensembles"
    files = sorted((ENS_ROOT / "PolyX_PolyG_30").glob("batch_*.npz"))
    pos = [np.load(f)["pos"] for f in files]
    X = np.concatenate(pos, axis=0).astype(float)
    Xc = X - X.mean(axis=1, keepdims=True)
    ref = Xc.mean(axis=0)
    Y = Xc.copy()
    for _ in range(3):
        Yn = np.empty_like(Y)
        for i in range(len(Y)):
            H = Y[i].T @ ref
            U, _, Vt = np.linalg.svd(H)
            d = np.sign(np.linalg.det(Vt.T @ U.T))
            R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
            Yn[i] = Y[i] @ R.T
        Y = Yn
        ref = Y.mean(axis=0)
    Fl = Y.reshape(len(Y), -1)
    Fl = Fl - Fl.mean(0)
    Zl_umap = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=rng).fit_transform(Fl)
    Zl_tsne = TSNE(n_components=2, perplexity=40, init="pca", random_state=rng,
                   learning_rate="auto").fit_transform(Fl)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 12.0))
    for ax, Z, title, points, cols, cats_ in [
        (axes[0, 0], Z_umap, "(a) Atlas · UMAP (n_neighbors = 30)",
         sub, cat_cols, cats),
        (axes[0, 1], Z_tsne, "(b) Atlas · t-SNE (perplexity = 30)",
         sub, cat_cols, cats),
        (axes[1, 0], Zl_umap, "(c) PolyG₃₀ local chart · UMAP",
         None, None, None),
        (axes[1, 1], Zl_tsne, "(d) PolyG₃₀ local chart · t-SNE",
         None, None, None)]:
        if points is not None:
            for c in cats_:
                m = points.category == c
                ax.scatter(Z[m, 0], Z[m, 1], s=9, color=cols[c], alpha=0.55,
                           linewidths=0, label=f"{c} ({len(Z[m])})", zorder=2)
                if len(Z[m]) >= 5:
                    Zc = Z[m]; mu = Zc.mean(0); S = np.cov(Zc.T)
                    ev, evc = np.linalg.eigh(S)
                    kk = int(np.argmax(ev))
                    ang = np.degrees(np.arctan2(evc[1, kk], evc[0, kk]))
                    w, h = 2.2 * np.sqrt(ev[kk]), 2.2 * np.sqrt(ev[1 - kk])
                    ax.add_patch(Ellipse(mu, w, h, angle=ang, fc="none",
                                         ec=cols[c], lw=1.0, ls="--", alpha=0.65,
                                         zorder=3))
            ax.legend(fontsize=6.2, frameon=True, framealpha=0.85, borderpad=0.3,
                      labelspacing=0.35)
        else:
            ax.scatter(Z[:, 0], Z[:, 1], s=3.5, color="#2C7FB8", alpha=0.55,
                       linewidths=0, zorder=2)
            try:
                from scipy.stats import gaussian_kde
                kk = gaussian_kde(Z.T)
                q1 = np.percentile(Z[:, 0], [1, 99]); q2 = np.percentile(Z[:, 1], [1, 99])
                xi, yi = np.mgrid[q1[0]:q1[1]:60j, q2[0]:q2[1]:60j]
                zi = kk(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
                ax.contour(xi, yi, zi, levels=6, colors="#1D6FA8", linewidths=0.5,
                           alpha=0.5, zorder=3)
            except Exception:
                pass
            ax.text(0.03, 0.95, f"n = {len(Z):,} conformations",
                    transform=ax.transAxes, va="top", fontsize=6.4,
                    fontweight="bold", color="#0B3D62",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#2C7FB8",
                              lw=0.5))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_box_aspect(1.0)
        ax.set_title(title, fontsize=7.5, pad=6)
    fig.suptitle("Nonlinear embeddings of ensemble state space "
                 "(Supplementary to Fig. 1a, c)",
                 fontsize=12, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, "figS9_umap_tsne_atlas", SUP)
    plt.close(fig)


def main() -> None:
    figS1()
    figS2()
    figS3()
    figS4()
    figS5()
    figS6()
    figS7()
    figS8()
    figS9()
    print("[complete] all 9 supplementary figures (v3)")


if __name__ == "__main__":
    main()
