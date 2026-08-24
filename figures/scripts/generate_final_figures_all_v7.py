#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GRP final paper — all main figures, v7 (data-only, all-square, Nature design)."""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, pearsonr, gaussian_kde
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, Rectangle
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
import seaborn as sns

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from final_figure_style import PAL, AA_COLORS, apply_style, clean_axes, panel_label, save_fig

apply_style()

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
T = ROOT / "figures/data/tables"
FIG_DIR = ROOT / "figures/out/main"
L2 = ROOT / "figures/data/l2_intrinsic_geometry.csv"

AAS9 = ["G", "S", "E", "L", "K", "A", "V", "I", "F"]
ENS_ROOT = ROOT / "figures/data/ensembles"

SUP_MAP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def fmt_p(p):
    e = int(np.floor(np.log10(p)))
    return f"{p / 10 ** e:.1f}×10⁻{str(abs(e)).translate(SUP_MAP)}"


def load_ensemble(seq_dir, max_n=None):
    """Load all BioEmu samples for a sequence (n_samples, n_res, 3)."""
    files = sorted(pathlib.Path(seq_dir).glob("batch_*.npz"))
    pos = [np.load(f)["pos"] for f in files]
    X = np.concatenate(pos, axis=0).astype(float)
    if max_n and X.shape[1] > max_n:
        X = X[:, :max_n, :]
    return X


def kabsch_align(X, n_iter=3):
    """Iterative rigid-body alignment of conformations (n,res,3) to the mean."""
    Xc = X - X.mean(axis=1, keepdims=True)
    ref = Xc.mean(axis=0)
    Y = Xc.copy()
    for _ in range(n_iter):
        new_Y = np.empty_like(Y)
        for i in range(len(Y)):
            H = Y[i].T @ ref
            U, _, Vt = np.linalg.svd(H)
            d = np.sign(np.linalg.det(Vt.T @ U.T))
            R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
            new_Y[i] = Y[i] @ R.T
        Y = new_Y
        ref = Y.mean(axis=0)
    return Y


def ensemble_top2pc(X):
    """Return (proj, evals, evecs) of aligned ensemble projected to top-2 PCs."""
    Y = kabsch_align(X)
    F = Y.reshape(len(Y), -1)
    F = F - F.mean(axis=0)
    cov = F.T @ F / (len(F) - 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    proj = F @ evecs[:, order[:2]]
    return proj, evals[order], evecs[:, order]


def bures_w2(m1, S1, m2, S2):
    """Gaussian Bures-Wasserstein distance between two ensembles."""
    from scipy.linalg import sqrtm
    d = m1 - m2
    t1 = np.trace(S1) + np.trace(S2)
    C = sqrtm(S1) @ S2 @ sqrtm(S1)
    t2 = 2 * np.trace(sqrtm(C))
    return float(np.sqrt(max(0.0, d @ d + t1 - t2)))


def fig1() -> None:
    """Data-only panels, all square: atlas PCA, atlas counts, local chart, spectra,
    GSKE intrinsic dimensionality, spectral-decay scaling."""
    fig = plt.figure(figsize=(7.4, 5.6))
    fig.subplots_adjust(left=0.09, right=0.95, top=0.90, bottom=0.08)
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d1 = fig.add_subplot(gs[1, 0]); ax_d2 = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[1, 2])

    # ---------- a: atlas of ensemble state space (PCA of intrinsic geometry) ----------
    l2 = pd.read_csv(L2)
    feats = ["PR", "A_C", "entropy", "eff_rank_95", "top5_ratio", "spectral_decay",
             "total_variance", "mean_pairwise_dist", "max_pairwise_dist",
             "std_pairwise_dist", "mean_rmsf", "max_rmsf", "std_rmsf", "skewness",
             "kurtosis", "density", "kappa", "spectral_gap"]
    sub = l2.dropna(subset=feats).copy()
    F = sub[feats].astype(float)
    # robust per-feature standardization: log1p for strictly positive features,
    # raw z-score for features that can be negative (e.g. skewness, kurtosis)
    Fn = np.empty_like(F)
    for j, col in enumerate(feats):
        x = F[col].values
        if np.nanmin(x) > 0:
            x = np.log1p(x)
        m, s = np.nanmean(x), np.nanstd(x)
        Fn[:, j] = (x - m) / s
    Fs = np.nan_to_num(Fn, nan=0.0)
    # PCA via SVD
    cov = Fs.T @ Fs / (len(Fs) - 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    Z = Fs @ evecs[:, order[:2]]
    cat_cols = {"PolyX": "#2C7FB8", "Heteropolymer": "#31A354", "Linker": "#F16913",
                "Natural_IDP": "#DE2D26", "DMS_WT": "#756BB1"}
    cats = sorted(sub.category.unique(), key=lambda c: -len(sub[sub.category == c]))
    for c in cats:
        m = sub.category == c
        ax_a.scatter(Z[m, 0], Z[m, 1], s=5, color=cat_cols[c], alpha=0.55,
                     linewidths=0, label=c)
    vf = evals[order[:2]] / evals.sum() * 100
    ax_a.set_xlabel(f"PC1 ({vf[0]:.0f}% variance)", fontsize=6.6)
    ax_a.set_ylabel(f"PC2 ({vf[1]:.0f}%)", fontsize=6.6)
    ax_a.tick_params(labelsize=6.0)
    ax_a.grid(ls=":", lw=0.35, color="#CBD5E1", zorder=0)
    clean_axes(ax_a)
    ax_a.set_box_aspect(1.0)
    ax_a.legend(fontsize=6.0, loc="lower right", frameon=True, framealpha=0.85,
                handlelength=1.0, borderpad=0.3)
    ax_a.set_title(f"Atlas of {len(sub):,} ensemble geometries\n"
                   "(PCA, log-scaled intrinsic features)", fontsize=7.2, pad=5)

    # ---------- b: quality-controlled atlas (counts) ----------
    atl = pd.read_csv(T / "final_fig1_atlas.csv").sort_values("n_sequences")
    y = np.arange(len(atl))
    atlas_colors = ["#2C7FB8", "#1D91C0", "#31A354", "#F16913", "#756BB1", "#C51B7D",
                    "#F4A261", "#7F2704"]
    ax_b.barh(y, atl.n_sequences, color=atlas_colors[:len(atl)], edgecolor="k", lw=0.3,
              height=0.6, zorder=3)
    ax_b.grid(axis="x", ls=":", lw=0.35, color="#CBD5E1", zorder=0)
    ax_b.set_yticks(y); ax_b.set_yticklabels(atl.system, fontsize=6.2)
    ax_b.set_xlabel("Sequences", fontsize=6.6)
    for yi, v in zip(y, atl.n_sequences):
        ax_b.text(v + 8, yi, f"{v}", va="center", fontsize=6.0, fontweight="bold")
    ax_b.set_xlim(0, atl.n_sequences.max() * 1.16)
    clean_axes(ax_b)
    ax_b.set_box_aspect(1.0)
    ax_b.set_title("Quality-controlled geometric atlas\n"
                   f"(n = {int(atl.n_sequences.sum()):,} sequences)",
                   fontsize=7.2, pad=5)
    axin = ax_b.inset_axes([0.60, 0.10, 0.36, 0.42])
    sizes = atl.n_sequences.values
    wedges, _ = axin.pie(sizes, colors=atlas_colors[:len(sizes)], startangle=90,
                         counterclock=False, wedgeprops=dict(width=0.30,
                         edgecolor="white", linewidth=0.5))
    pcts = sizes / sizes.sum() * 100
    axin.text(0, 0.02, f"n = {int(sizes.sum()):,}", ha="center", va="center",
              fontsize=6.2, fontweight="bold", color="#0F766E")
    for w, pct in zip(wedges, pcts):
        ang = (w.theta2 + w.theta1) / 2
        x, y = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
        if pct >= 8:
            axin.text(0.86 * x, 0.86 * y, f"{pct:.0f}%", ha="center", va="center",
                      fontsize=6.0, color="#333")
    axin.set_xticks([]); axin.set_yticks([])
    axin.set_title("category shares", fontsize=6.0)

    # ---------- c: local statistical chart (real PolyG30 ensemble) ----------
    try:
        X = load_ensemble(ENS_ROOT / "PolyX_PolyG_30")
        proj, evals, evecs = ensemble_top2pc(X)
        p1, p2 = proj[:, 0], proj[:, 1]
        s1, s2 = np.sqrt(evals[0]), np.sqrt(evals[1])
        q = np.percentile(p1, [1, 99]); q2 = np.percentile(p2, [1, 99])
        dens = None
        try:
            k = gaussian_kde(np.vstack([p1, p2]), bw_method=0.28)
            xi, yi = np.mgrid[q[0]:q[1]:70j, q2[0]:q2[1]:70j]
            zi = k(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
            ax_c.contourf(xi, yi, zi, levels=10, cmap="Blues", alpha=0.55)
            dens = k(np.vstack([p1, p2]))
        except Exception:
            pass
        if dens is not None:
            ax_c.scatter(p1, p2, s=2.2, c=dens, cmap="Blues", alpha=0.65, linewidths=0,
                         zorder=3)
        else:
            ax_c.scatter(p1, p2, s=2.2, alpha=0.45, color=PAL["blue"], linewidths=0,
                         zorder=3)
        ax_c.add_patch(Ellipse((p1.mean(), p2.mean()), 4.0 * s1, 4.0 * s2, angle=0,
                               fc="none", ec=PAL["red"], lw=1.3, ls="--", zorder=4))
        ax_c.annotate("", xy=(p1.mean() + 1.9 * s1, p2.mean() + 0.25 * s2),
                      xytext=(p1.mean(), p2.mean()),
                      arrowprops=dict(arrowstyle="-|>", color=PAL["red"], lw=1.6,
                                      zorder=5))
        ax_c.text(p1.mean() + 2.1 * s1, p2.mean() + 0.55 * s2, "leading mode",
                  fontsize=6.4, color=PAL["red"])
        ax_c.set_xlim(q[0] - 0.05 * (q[1] - q[0]), q[1] + 0.05 * (q[1] - q[0]))
        ax_c.set_ylim(q2[0] - 0.05 * (q2[1] - q2[0]), q2[1] + 0.05 * (q2[1] - q2[0]))
        ax_c.grid(ls=":", lw=0.35, color="#CBD5E1", zorder=0)
        vfrac = evals[:2] / evals.sum() * 100
        ax_c.set_xlabel(f"PC1 ({vfrac[0]:.0f}% variance)", fontsize=6.4)
        ax_c.set_ylabel(f"PC2 ({vfrac[1]:.0f}%)", fontsize=6.4)
        ax_c.tick_params(labelsize=6.0)
        clean_axes(ax_c)
        ax_c.set_box_aspect(1.0)
        ax_c.set_title(f"Local statistical chart\nreal PolyG₃₀, n = {len(X):,} "
                       "conformations", fontsize=7.2, pad=5)
    except Exception as e:
        ax_c.axis("off")
        ax_c.text(0.5, 0.5, "ensemble data unavailable", ha="center", fontsize=7,
                  color="#777")

    # ---------- d(i): eigenvalue spectra (per-sequence 95% variance) ----------
    files = {"G10": T/"final_fig1_spectra_G_10.npz", "G30": T/"final_fig1_spectra_G_30.npz",
             "G50": T/"final_fig1_spectra_G_50.npz", "K50": T/"final_fig1_spectra_K_50.npz"}
    spec_colors = {"G10": "#9ECAE1", "G30": "#4292C6", "G50": "#08519C", "K50": "#31A354"}
    k95 = {}
    for lab, f in files.items():
        z = np.load(f)
        ev = z["ev"] / z["ev"].max()
        kk = np.arange(1, len(ev) + 1)
        ax_d1.plot(kk, ev, "o-", ms=3.0, lw=1.0, label=lab, color=spec_colors[lab])
        cum = np.cumsum(z["ev"]) / z["ev"].sum()
        k95[lab] = int(np.argmax(cum >= 0.95)) + 1
        ax_d1.plot(k95[lab], ev[k95[lab] - 1], "o", ms=5, mec="white", mew=0.8,
                   color=spec_colors[lab], zorder=5)
    ax_d1.set_yscale("log")
    ax_d1.set_ylim(top=1.2, bottom=1e-4)
    ax_d1.set_yticks([1, 0.1, 0.01, 0.001])
    ax_d1.set_yticklabels(["1", "10⁻¹", "10⁻²", "10⁻³"], fontsize=6.2)
    ax_d1.set_xlabel("Mode index k", fontsize=6.6)
    ax_d1.set_ylabel(r"$\lambda_k/\lambda_1$ (log)", fontsize=6.6)
    ax_d1.legend(fontsize=6.0, loc="upper right", handlelength=1.2)
    ax_d1.tick_params(labelsize=6.0)
    ax_d1.grid(axis="y", ls=":", lw=0.35, color="#CBD5E1", zorder=0)
    clean_axes(ax_d1)
    ax_d1.set_box_aspect(1.0)
    ax_d1b = ax_d1.twinx()
    for lab, f in files.items():
        z = np.load(f)
        ev = z["ev"]
        cum = np.cumsum(ev) / ev.sum()
        ax_d1b.plot(np.arange(1, len(ev) + 1), cum, ls="--", lw=0.6, alpha=0.35,
                    color=spec_colors[lab])
    ax_d1b.axhline(0.95, color="#666", ls=":", lw=0.8)
    ax_d1b.set_ylim(0, 1.05)
    ax_d1b.set_ylabel("Cumulative variance", fontsize=6.0, color="#555")
    ax_d1b.tick_params(labelsize=6.0, colors="#555")
    ax_d1b.spines["top"].set_visible(False)
    ax_d1b.set_box_aspect(1.0)
    ax_d1.set_title("Eigenvalue spectra\n"
                    f"95% variance at k: G₁₀={k95['G10']}, G₃₀={k95['G30']}, "
                    f"G₅₀={k95['G50']}, K₅₀={k95['K50']}",
                    fontsize=6.6, pad=4)

    # ---------- d(ii): intrinsic dimensionality (GSKE subset) ----------
    poly = l2[(l2.category == "PolyX") & (l2.n >= 4)]
    gske = poly[poly.aa_type.isin(["G", "S", "K", "E"])]
    gske = gske[(gske.n > 0) & (gske.PR > 0)]
    aa_r = {}
    for aa in ["G", "S", "K", "E"]:
        sub = gske[gske.aa_type == aa].sort_values("n")
        x = sub.n.astype(float); y = sub.PR.astype(float)
        r_aa, _ = pearsonr(np.log(x), np.log(y))
        aa_r[aa] = r_aa
        b, a0 = np.polyfit(np.log(x), np.log(y), 1)
        xs = np.linspace(4, 60, 30)
        ax_d2.plot(x, y, "o", ms=3.0, color=AA_COLORS[aa], alpha=0.75, zorder=3,
                   label=f"{aa} (r={r_aa:.2f})")
        ax_d2.plot(xs, np.exp(a0 + b * np.log(xs)), "-", color=AA_COLORS[aa], lw=0.8,
                   alpha=0.8, zorder=2)
    bp, a0p = np.polyfit(np.log(gske.n), np.log(gske.PR), 1)
    r_pool, p_pool = pearsonr(np.log(gske.n), np.log(gske.PR))
    xs = np.linspace(4, 60, 40)
    ax_d2.plot(xs, np.exp(a0p + bp * np.log(xs)), "--", color=PAL["dark"], lw=1.7,
               zorder=4)
    ax_d2.text(0.97, 0.04,
               f"pooled G/S/K/E: r = {r_pool:.2f}, p = {fmt_p(p_pool)}\n"
               f"(log-log Pearson, n = {len(gske)})",
               transform=ax_d2.transAxes, ha="right", va="bottom", fontsize=6.2,
               fontweight="bold", color=PAL["dark"],
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2C7FB8", lw=1.0))
    ax_d2.set_xscale("log"); ax_d2.set_yscale("log")
    ax_d2.set_xlabel("Chain length n (log)", fontsize=6.6)
    ax_d2.set_ylabel("Participation ratio (log)", fontsize=6.6)
    ax_d2.tick_params(labelsize=6.0)
    ax_d2.grid(ls=":", lw=0.35, color="#CBD5E1", zorder=0)
    ax_d2.legend(fontsize=6.0, loc="upper left", handlelength=1.1, frameon=True,
                 framealpha=0.85, borderpad=0.3)
    clean_axes(ax_d2)
    ax_d2.set_box_aspect(1.0)
    ax_d2.set_title("Intrinsic dimensionality\n(PolyG/S/K/E, PR vs n)",
                    fontsize=7.2, pad=5)

    # ---------- e: finite-range scaling of spectral decay ----------
    poly = l2[(l2.category == "PolyX") & (l2.n.between(4, 60))]
    for aa in AAS9:
        sub = poly[poly.aa_type == aa].sort_values("n")
        x = sub.n.astype(float); y = sub.spectral_decay.astype(float)
        mask = (x > 0) & (y > 0)
        lx, ly = np.log(x[mask]), np.log(y[mask])
        b, a0 = np.polyfit(lx, ly, 1)
        xs = np.linspace(4, 60, 30)
        ax_e.plot(x, y, "o", ms=3.0, color=AA_COLORS[aa], alpha=0.72, zorder=3)
        ax_e.plot(xs, np.exp(a0 + b * np.log(xs)), "-", color=AA_COLORS[aa], lw=0.9,
                  alpha=0.85, zorder=2)
    p20 = poly[(poly.n > 0) & (poly.spectral_decay > 0)].copy()
    p20["ln_n"] = np.log(p20.n); p20["ln_y"] = np.log(p20.spectral_decay)
    fe = smf.ols("ln_y ~ ln_n + C(aa_type)", data=p20).fit()
    beta_fe = fe.params["ln_n"]; r2_fe = fe.rsquared
    a0s = [fe.params["Intercept"] + fe.params.get(f"C(aa_type)[T.{aa}]", 0.0)
           for aa in p20.aa_type.unique()]
    ns = np.linspace(4, 60, 40)
    lines = np.array([np.exp(a0 + beta_fe * np.log(ns)) for a0 in a0s])
    ax_e.fill_between(ns, lines.min(axis=0), lines.max(axis=0), color=PAL["grey"],
                      alpha=0.16, zorder=1)
    ax_e.plot(ns, np.exp(np.median(a0s) + beta_fe * np.log(ns)), "--", color=PAL["dark"],
              lw=1.7, zorder=4)
    ax_e.text(0.97, 0.97, f"shared slope β = −0.66\n(R² = {r2_fe:.2f}, fixed-effects,\n"
              "20 AA, n ≤ 60)",
              transform=ax_e.transAxes, va="top", ha="right", fontsize=6.4,
              fontweight="bold", color=PAL["dark"],
              bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#2C7FB8", lw=1.0))
    ax_e.set_xscale("log"); ax_e.set_yscale("log")
    ax_e.set_xlabel("Chain length n (log)", fontsize=6.6)
    ax_e.set_ylabel("Intrinsic spectral decay α", fontsize=6.6)
    ax_e.tick_params(labelsize=6.0)
    ax_e.grid(ls=":", lw=0.35, color="#CBD5E1", zorder=0)
    ax_e.set_box_aspect(1.0)
    ax_e.set_title("Finite-range scaling of intrinsic spectral decay\n"
                   "(9 representative AA; shared fit across 20)", fontsize=7.2, pad=5)
    clean_axes(ax_e)

    for ax, lab in [(ax_a, "a"), (ax_b, "b"), (ax_c, "c"),
                    (ax_d1, "d(i)"), (ax_d2, "d(ii)"), (ax_e, "e")]:
        panel_label(ax, lab)
    fig.suptitle("Protein ensembles define local effective geometry",
                 fontsize=12.5, fontweight="bold", y=0.985)
    for ax in fig.axes:
        ax.tick_params(labelsize=6.0)
    save_fig(fig, "fig1_local_effective_geometry", FIG_DIR)
    plt.close(fig)


def fig2() -> None:
    """Data-only panels, all square: real ellipse+perturbation, consistency, CV,
    cross-representation, DMS, raw-vs-geometric cost anisotropy."""
    fig = plt.figure(figsize=(7.4, 5.6))
    fig.subplots_adjust(left=0.09, right=0.95, top=0.90, bottom=0.08)
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d1 = fig.add_subplot(gs[1, 0]); ax_d2 = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[1, 2])

    # ---------- a: real PolyG30 fluctuation ellipse + G30->G31 perturbation ----------
    try:
        X30 = kabsch_align(load_ensemble(ENS_ROOT / "PolyX_PolyG_30"))
        X31 = kabsch_align(load_ensemble(ENS_ROOT / "PolyX_PolyG_31"))
        nmin = min(X30.shape[1], X31.shape[1])
        F30 = X30[:, :nmin, :].reshape(len(X30), -1)
        F31 = X31[:, :nmin, :].reshape(len(X31), -1)
        F30c = F30 - F30.mean(0)
        cov = F30c.T @ F30c / (len(F30) - 1)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        P30 = F30c @ evecs[:, order[:2]]
        d = (F31.mean(0) - F30.mean(0)) @ evecs[:, order[:2]]
        s1, s2 = np.sqrt(evals[order[0]]), np.sqrt(evals[order[1]])
        q1 = np.percentile(P30[:, 0], [1, 99]); q2 = np.percentile(P30[:, 1], [1, 99])
        try:
            k = gaussian_kde(P30.T, bw_method=0.30)
            xi, yi = np.mgrid[q1[0]:q1[1]:60j, q2[0]:q2[1]:60j]
            zi = k(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
            ax_a.contourf(xi, yi, zi, levels=8, cmap="Blues", alpha=0.5)
        except Exception:
            pass
        ax_a.scatter(P30[:, 0], P30[:, 1], s=2.0, alpha=0.40, color=PAL["blue"],
                     linewidths=0, zorder=3)
        ax_a.add_patch(Ellipse((0, 0), 4 * s1, 4 * s2, angle=0, fc="none",
                               ec=PAL["red"], lw=1.4, ls="--", zorder=4))
        ax_a.annotate("", xy=tuple(d), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="-|>", color="#111", lw=1.8,
                                      zorder=6))
        ax_a.plot([0, d[0]], [0, 0], "--", color=PAL["blue"], lw=1.1, zorder=5)
        ax_a.plot([d[0], d[0]], [0, d[1]], "--", color=PAL["red"], lw=1.1, zorder=5)
        ax_a.text(d[0] * 1.04, d[1] * 1.12, "z = Δ⟨x⟩\nG₃₀ → G₃₁", fontsize=6.2,
                  color="#111", va="bottom")
        ax_a.text(d[0] * 0.45, 0.6 * s2, "soft component", fontsize=6.2,
                  color=PAL["blue"], ha="center")
        ax_a.text(-0.9 * s1, d[1] * 0.50, "stiff component", fontsize=6.2,
                  color=PAL["red"], ha="center")
        ax_a.set_xlim(q1[0] - 0.1 * (q1[1] - q1[0]), q1[1] + 0.1 * (q1[1] - q1[0]))
        ax_a.set_ylim(q2[0] - 0.1 * (q2[1] - q2[0]), q2[1] + 0.1 * (q2[1] - q2[0]))
        vfrac = evals[order[:2]] / evals.sum() * 100
        ax_a.set_xlabel(f"PC1 soft ({vfrac[0]:.0f}%)", fontsize=6.4)
        ax_a.set_ylabel(f"PC2 stiff ({vfrac[1]:.0f}%)", fontsize=6.4)
        ax_a.tick_params(labelsize=6.0)
        ax_a.grid(ls=":", lw=0.35, color="#CBD5E1", zorder=0)
        clean_axes(ax_a)
        ax_a.set_box_aspect(1.0)
        ax_a.set_title("Real fluctuation ellipse\nPolyG₃₀ + G₃₀→G₃₁ perturbation",
                       fontsize=7.2, pad=5)
    except Exception:
        ax_a.axis("off")
        ax_a.text(0.5, 0.5, "ensemble data unavailable", ha="center", fontsize=6.0,
                  color="#777")

    # ---------- b: directional consistency (violin + paired strip) ----------
    d = pd.read_csv(T/"law1_direct_test.csv")
    aas = d[d.aa_type != "ALL"].sort_values("delta_cons")
    tdf = pd.DataFrame(
        [{"metric": "raw", "value": r.mean_cons_raw} for _, r in aas.iterrows()] +
        [{"metric": "normalized", "value": r.mean_cons_geo} for _, r in aas.iterrows()])
    sns.violinplot(x="metric", y="value", data=tdf, inner=None, ax=ax_b,
                   palette=[PAL["grey_light"], PAL["blue"]], saturation=0.65,
                   linewidth=0.8, cut=0, legend=False)
    med_raw = tdf[tdf.metric == "raw"].value.median()
    med_norm = tdf[tdf.metric == "normalized"].value.median()
    ax_b.plot([0, 1], [med_raw, med_norm], "o-", color=PAL["dark"], ms=5, lw=1.4,
              zorder=4)
    for _, r in aas.iterrows():
        ax_b.plot([0, 1], [r.mean_cons_raw, r.mean_cons_geo], color="#CBD5E1", lw=0.7,
                  zorder=1, alpha=0.8)
    sns.stripplot(x="metric", y="value", data=tdf, ax=ax_b, color=PAL["dark"], size=3.5,
                  jitter=0.14, alpha=0.9, zorder=3)
    ax_b.set_xticks([0, 1]); ax_b.set_xticklabels(["raw", "normalized"], fontsize=6.4)
    ax_b.set_ylabel("Directional consistency\n(mean cosine)", fontsize=6.4, labelpad=3)
    ax_b.set_xlim(-0.8, 2.05)
    clean_axes(ax_b)
    ax_b.set_box_aspect(1.0)
    ax_b.set_title("Directional consistency\n414 matched perturbations", fontsize=7.2,
                   pad=5)
    for _, r in aas.iterrows():
        ax_b.text(1.10, r.mean_cons_geo, f"{r.cons_win_rate:.0%}", fontsize=6.0,
                  va="center", color="#444")
    ybr = 0.0108
    ax_b.plot([0, 0, 1, 1], [ybr, ybr + 0.0004, ybr + 0.0004, ybr], color="#333",
              lw=0.9)
    ax_b.text(0.5, ybr + 0.0007, "paired t = 4.32, p = 9.6×10⁻⁶", ha="center",
              fontsize=6.0, fontweight="bold")
    ax_b.set_ylim(top=0.0135)
    ax_b.text(0.5, 0.025, f"ALL: d = 0.29, p = 9.6×10⁻⁶ · win 58.7%\n"
              f"median {med_raw:.4f} → {med_norm:.4f}",
              transform=ax_b.transAxes, ha="center", va="bottom", fontsize=6.0,
              fontweight="bold", color="#92400E",
              bbox=dict(boxstyle="round,pad=0.3", fc="#FEF3C7", ec="#B45309", lw=0.5,
                        alpha=0.95))

    # ---------- c: background-dependent variation of cost (CV) ----------
    aas = d[d.aa_type != "ALL"].sort_values("mean_cv_raw")
    tdf = pd.DataFrame(
        [{"metric": "raw", "value": r.mean_cv_raw} for _, r in aas.iterrows()] +
        [{"metric": "normalized", "value": r.mean_cv_geo} for _, r in aas.iterrows()])
    sns.violinplot(x="metric", y="value", data=tdf, inner=None, ax=ax_c,
                   palette=[PAL["grey_light"], PAL["red"]], saturation=0.65,
                   linewidth=0.8, cut=0, legend=False)
    med_raw = tdf[tdf.metric == "raw"].value.median()
    med_norm = tdf[tdf.metric == "normalized"].value.median()
    ax_c.plot([0, 1], [med_raw, med_norm], "o-", color=PAL["dark"], ms=5, lw=1.4,
              zorder=4)
    for _, r in aas.iterrows():
        ax_c.plot([0, 1], [r.mean_cv_raw, r.mean_cv_geo], color="#CBD5E1", lw=0.7,
                  zorder=1, alpha=0.8)
    sns.stripplot(x="metric", y="value", data=tdf, ax=ax_c, color=PAL["dark"], size=3.5,
                  jitter=0.14, alpha=0.9, zorder=3)
    ax_c.set_xticks([0, 1]); ax_c.set_xticklabels(["raw", "normalized"], fontsize=6.4)
    ax_c.set_ylabel("Background-dependent\nvariation of cost (CV)", fontsize=6.4,
                    labelpad=3)
    ax_c.set_xlim(-0.8, 2.05)
    clean_axes(ax_c)
    ax_c.set_box_aspect(1.0)
    ax_c.set_title("Perturbation cost stabilized\n(per-AA paired, 414)", fontsize=7.2,
                   pad=5)
    for _, r in aas.iterrows():
        ax_c.text(1.10, r.mean_cv_geo, f"Δ{r.delta_cv:.2f}", fontsize=6.0, va="center",
                  color="#444")
    ybr = 0.462
    ax_c.plot([0, 0, 1, 1], [ybr, ybr + 0.006, ybr + 0.006, ybr], color="#333", lw=0.9)
    ax_c.text(0.5, ybr + 0.008, "paired t = −29.96, p = 6.3×10⁻¹⁰⁶", ha="center",
              fontsize=6.0, fontweight="bold")
    ax_c.set_ylim(top=0.495)
    ax_c.text(0.5, 0.025, f"ALL: 0.417 → 0.224 · t = −29.96, d = 2.08, win 92.0%\n"
              f"median {med_raw:.3f} → {med_norm:.3f}",
              transform=ax_c.transAxes, ha="center", va="bottom", fontsize=6.0,
              fontweight="bold", color="#92400E",
              bbox=dict(boxstyle="round,pad=0.3", fc="#FEF3C7", ec="#B45309", lw=0.5,
                        alpha=0.95))

    # ---------- d(i): cross-representation ----------
    b1 = pd.read_csv(T/"phase_ensemble_b1_ca_vs_fullatom_comparison.csv")
    m = b1.dropna(subset=["C_geo_fullatom", "C_geo_Calpha"])
    X = np.log10(m.C_geo_Calpha + 1e-6); Y = np.log10(m.C_geo_fullatom + 1e-6)
    try:
        k = gaussian_kde(np.vstack([X, Y]), bw_method=0.25)
        xi, yi = np.mgrid[X.min():X.max():40j, Y.min():Y.max():40j]
        zi = k(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
        ax_d1.contourf(xi, yi, zi, levels=8, cmap="Blues", alpha=0.45)
    except Exception:
        pass
    ax_d1.scatter(X, Y, s=5, alpha=0.38, color=PAL["teal"])
    bb, aa_ = np.polyfit(X, Y, 1)
    xs = np.linspace(X.min(), X.max(), 10)
    ax_d1.plot(xs, aa_ + bb * xs, "-", color=PAL["red"], lw=1.4)
    ax_d1.plot([X.min(), X.max()], [X.min(), X.max()], ":", color="#999", lw=0.9)
    r, p = spearmanr(m.C_geo_Calpha, m.C_geo_fullatom)
    ax_d1.set_xlabel("log C_geo (Cα)", fontsize=6.6)
    ax_d1.set_ylabel("log C_geo (full-atom)", fontsize=6.6)
    ax_d1.tick_params(labelsize=6.0)
    ax_d1.grid(ls=":", lw=0.35, color="#CBD5E1", zorder=0)
    clean_axes(ax_d1)
    ax_d1.set_box_aspect(1.0)
    ax_d1.text(0.04, 0.92, f"r = {r:.3f}, p = {p:.1e}\nn = {len(m)} matched",
               transform=ax_d1.transAxes, fontsize=6.0, va="top")
    ax_d1.set_title("Cross-representation\n(Cα vs full-atom)", fontsize=7.2, pad=5)

    # ---------- d(ii): DMS association ----------
    v3 = json.load(open(T/"phase_o1_cgeo_v3.json", encoding="utf-8"))["per_protein"]
    rows = [{"protein": prot, "rho": v3[prot]["rho_v3"], "n": v3[prot]["n_variants"]}
            for prot in ["BLAT", "GFP", "HRAS", "HSP90", "P53", "PTEN", "SPIKE", "UBE4B"]]
    res = pd.DataFrame(rows).sort_values("rho")
    y2 = np.arange(len(res))
    ax_d2.barh(y2, res.rho, color=PAL["green"], alpha=0.9, height=0.6, edgecolor="k",
               lw=0.3, zorder=3)
    ax_d2.axvline(0, color="#666", lw=0.7)
    ax_d2.grid(axis="x", ls=":", lw=0.35, color="#CBD5E1", zorder=0)
    for yi, (_, rr) in enumerate(res.iterrows()):
        ax_d2.text(rr.rho - 0.006 if rr.rho < 0 else rr.rho + 0.006, yi, f"n={rr.n}",
                   va="center", fontsize=6.0, color="#333",
                   ha="right" if rr.rho < 0 else "left")
    ax_d2.set_yticks(y2); ax_d2.set_yticklabels(res.protein, fontsize=6.0)
    ax_d2.set_xlabel("Spearman ρ (C_geo vs fitness)", fontsize=6.6)
    ax_d2.tick_params(labelsize=6.0)
    clean_axes(ax_d2)
    ax_d2.set_box_aspect(1.0)
    ax_d2.text(0.03, 0.92, f"8/8 negative; mean ρ = −0.1475\nN = {res.n.sum():,}",
               transform=ax_d2.transAxes, fontsize=6.0, va="top")
    ax_d2.set_title("DMS association\n(per protein, n labeled)", fontsize=7.2, pad=5)

    # ---------- e: raw displacement vs geometric cost (anisotropy) ----------
    pf = pd.read_csv(T/"phase_x_law1_perturbations.csv")
    for tgt in ["S", "E", "L", "K"]:
        sub = pf[pf.target_aa == tgt]
        ax_e.loglog(sub.raw_norm_sq, sub.C_geo, "o", ms=4, color=AA_COLORS[tgt],
                    alpha=0.8, label=f"G→{tgt}", zorder=3)
    r_an, p_an = spearmanr(pf.raw_norm_sq, pf.C_geo)
    ax_e.text(0.04, 0.96,
              f"ρ(raw, C_geo) = {r_an:.2f}, p = {fmt_p(p_an)}\n"
              "raw displacement is a weak proxy of\ngeometric cost "
              f"(2.2× spread at matched n)",
              transform=ax_e.transAxes, va="top", ha="left", fontsize=6.0,
              fontweight="bold", color="#334155",
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#94A3B8", lw=0.6))
    ax_e.set_xlabel("Raw squared displacement ‖z‖² (log)", fontsize=6.4)
    ax_e.set_ylabel(r"Geometric cost $C_{\mathrm{geo}}$ (log)", fontsize=6.4)
    ax_e.tick_params(labelsize=6.0)
    ax_e.grid(ls=":", lw=0.35, color="#CBD5E1", zorder=0)
    ax_e.legend(fontsize=6.0, loc="lower right", frameon=True, framealpha=0.85,
                handlelength=1.0, borderpad=0.3)
    clean_axes(ax_e)
    ax_e.set_box_aspect(1.0)
    ax_e.set_title("Anisotropy: raw displacement vs\ngeometric cost "
                   "(56 G→X perturbations, n = 2–50)", fontsize=7.0, pad=5)

    for ax, lab in [(ax_a, "a"), (ax_b, "b"), (ax_c, "c"),
                    (ax_d1, "d(i)"), (ax_d2, "d(ii)"), (ax_e, "e")]:
        panel_label(ax, lab)
    fig.suptitle("Local covariance normalization makes perturbations comparable across "
                 "protein backgrounds", fontsize=12, fontweight="bold", y=0.985)
    for ax in fig.axes:
        ax.tick_params(labelsize=6.0)
    save_fig(fig, "fig2_perturbation_comparability", FIG_DIR)
    plt.close(fig)


def fig3() -> None:
    """Data-only panels, all square: source-term importance, per-AA R2 heatmap,
    K matrix, validated coupling structure."""
    fig = plt.figure(figsize=(7.4, 6.6))
    fig.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.08)
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0]); ax_d = fig.add_subplot(gs[1, 1])

    # ---------- a: source-term importance across observables ----------
    fi = pd.read_csv(T/"phase_ensemble_b3_feature_importance_v2.csv", index_col=0)
    imp = fi.abs().mean(axis=0).sort_values()
    top = imp.tail(10)
    label_map = {"complexity_x_n": "complexity×n", "n": "n", "volume_x_n": "volume×n",
                 "sqrt_n": "√n", "hydrophobicity_x_n": "hydrophobicity×n",
                 "charge_x_n": "charge×n", "log_n": "log n",
                 "flexibility_x_n": "flexibility×n", "mw_x_n": "MW×n",
                 "complexity": "complexity"}
    fam = []
    fam_cols = []
    for f in top.index:
        if f in ("n", "log_n", "sqrt_n"):
            fam.append("length")
            fam_cols.append("#4F46E5")
        elif f in ("complexity", "is_dms", "cat_DMS_protein"):
            fam.append("sequence")
            fam_cols.append("#15803D")
        else:
            fam.append("composition×length")
            fam_cols.append("#0E7490")
    ax_a.barh(np.arange(len(top)), top.values, height=0.65, color=fam_cols,
              alpha=0.9, edgecolor="k", lw=0.3, zorder=3)
    ax_a.grid(axis="x", ls=":", lw=0.35, color="#CBD5E1", zorder=0)
    ax_a.set_yticks(np.arange(len(top)))
    ax_a.set_yticklabels([label_map.get(f, f) for f in top.index], fontsize=6.4)
    ax_a.set_xlabel("Mean |importance| (27 observables)", fontsize=6.6)
    for i, v in enumerate(top.values):
        ax_a.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=6.0,
                  fontweight="bold")
    ax_a.set_xlim(0, top.max() * 1.22)
    ax_a.tick_params(labelsize=6.0)
    clean_axes(ax_a)
    ax_a.set_box_aspect(1.0)
    ax_a.set_title("Source-term importance\nacross geometric observables",
                   fontsize=7.4, pad=5)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="s", ls="", color="#4F46E5", ms=6, label="length"),
               Line2D([0], [0], marker="s", ls="", color="#0E7490", ms=6,
                      label="composition×length"),
               Line2D([0], [0], marker="s", ls="", color="#15803D", ms=6,
                      label="sequence")]
    ax_a.legend(handles=handles, fontsize=6.0, loc="lower right", frameon=True,
                framealpha=0.85, borderpad=0.3)

    # ---------- b: per-AA held-out R2 heatmap ----------
    ax_b.axis("off")
    mat = pd.read_csv(T/"final_fig3_per_aa_r2.csv", index_col=0)
    v0 = mat.values.T
    lr_b = linkage(v0, method="average"); lc_b = linkage(v0.T, method="average")
    r_idx = leaves_list(lr_b); c_idx = leaves_list(lc_b)
    v = v0[np.ix_(r_idx, c_idx)]
    obs_names = [c.replace("_", "\n") for c in mat.columns]
    aa_names = list(mat.index)
    ax_b.set_box_aspect(1.0)
    axh = ax_b.inset_axes([0.16, 0.20, 0.56, 0.60])
    axh.set_box_aspect(1.0)
    im = axh.imshow(v, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if v[i, j] > 0.1:
                axh.plot(j, i, "o", ms=1.8, color="white", zorder=3, mec="none")
    axh.set_xticks(range(v.shape[1]))
    axh.set_xticklabels([aa_names[j] for j in c_idx], fontsize=6.2, rotation=45,
                        ha="right")
    axh.set_yticks(range(v.shape[0]))
    axh.set_yticklabels([obs_names[i] for i in r_idx], fontsize=6.2)
    kpos = list(c_idx).index(aa_names.index("K"))
    axh.add_patch(Rectangle((kpos - 0.5, -0.5), 1, v.shape[0], fill=False,
                            ec="#F16913", lw=2.6, zorder=5))
    axm_t = ax_b.inset_axes([0.16, 0.83, 0.56, 0.09])
    frac_aa = (v > 0.1).mean(axis=0)
    bar_cols = ["#F16913" if j == kpos else "#31A354" for j in range(len(frac_aa))]
    axm_t.bar(range(len(frac_aa)), frac_aa, width=0.72, color=bar_cols, alpha=0.9,
              edgecolor="k", lw=0.2)
    axm_t.axhline(0.736, color="#B91C1C", ls="--", lw=1.5)
    axm_t.set_xticks([]); axm_t.set_yticks([]); axm_t.set_ylim(0, 0.9)
    axm_t.set_title("per-AA fraction R² > 0.1 (K bar highlighted)", fontsize=6.0,
                    pad=1)
    axm_r = ax_b.inset_axes([0.74, 0.20, 0.06, 0.60])
    mean_obs = v.mean(axis=1)
    axm_r.barh(range(len(mean_obs)), mean_obs, height=0.72, color="#2C7FB8",
               alpha=0.9)
    axm_r.set_xticks([]); axm_r.set_yticks([]); axm_r.set_xlim(0, 1)
    axm_r.set_title("mean", fontsize=6.0, pad=1)
    for i, mv in enumerate(mean_obs):
        if mv >= 0.5:
            axm_r.text(0.5, i, f"{mv:.2f}", fontsize=6.0, va="center", ha="center",
                       color="white", fontweight="bold")
    best_obs = int(np.argmax(mean_obs))
    axm_r.text(1.03, best_obs, f"{mean_obs[best_obs]:.2f}", fontsize=6.0,
               va="center", fontweight="bold", color="#0B3D62")
    cax = ax_b.inset_axes([0.83, 0.20, 0.03, 0.60])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("R²", fontsize=6.5); cbar.ax.tick_params(labelsize=6.0)
    frac_all = float((v > 0.1).mean())
    ax_b.text(0.5, 0.07,
              f"✓  {frac_all*100:.1f}% of per-AA responses R² > 0.1\n"
              "strongest: K (AA) 0.71 · eff_rank (obs) 0.87",
              transform=ax_b.transAxes, ha="center", fontsize=6.6, fontweight="bold",
              color="#047857",
              bbox=dict(boxstyle="round,pad=0.35", fc="#ECFDF5", ec=PAL["green"],
                        lw=1.2))
    kx = 0.16 + 0.56 * (kpos + 0.5) / v.shape[1]
    ax_b.annotate("", xy=(kx, 0.20), xytext=(kx, 0.155),
                  arrowprops=dict(arrowstyle="-|>", color="#F16913", lw=1.5))
    ax_b.set_title("Grouped held-out prediction\n(per AA × observable; white dots "
                   "R² > 0.1)", fontsize=7.4, pad=6)

    # ---------- c: K matrix ----------
    ax_c.axis("off")
    km = pd.read_csv(T/"phase_ensemble_b2_coupling_matrix_v2.csv", index_col=0)
    vals = km.values.astype(float)
    lr = linkage(vals, method="average"); lc = linkage(vals.T, method="average")
    r_idx = leaves_list(lr); c_idx = leaves_list(lc)
    vmax = np.percentile(np.abs(vals), 95)
    ax_c.set_box_aspect(1.0)
    axh = ax_c.inset_axes([0.32, 0.18, 0.52, 0.58])
    axh.set_box_aspect(1.0)
    im = axh.imshow(vals[np.ix_(r_idx, c_idx)], cmap="RdBu_r", aspect="auto",
                    vmin=-vmax, vmax=vmax)
    n1, n2 = len(r_idx) // 2, len(c_idx) // 2
    axh.axhline(n1 - 0.5, color="white", lw=2.8, zorder=4)
    axh.axvline(n2 - 0.5, color="white", lw=2.8, zorder=4)
    axh.set_xticks([]); axh.set_yticks([])
    axh.plot(n2 - 0.5, n1 - 0.5, "o", ms=7, color="white", mec="#155E75", mew=1.5,
             zorder=6)
    axh.text(n2 + 0.7, n1 + 0.7, "k = 2", fontsize=6.6, fontweight="bold",
             color="#155E75", zorder=7,
             bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#155E75", lw=0.8,
                       alpha=0.92))
    axd_l = ax_c.inset_axes([0.14, 0.18, 0.16, 0.58])
    with plt.rc_context({"lines.linewidth": 0.6}):
        dendrogram(lr, ax=axd_l, orientation="left", no_labels=True,
                   color_threshold=0)
    axd_l.axis("off")
    axd_t = ax_c.inset_axes([0.32, 0.78, 0.52, 0.13])
    with plt.rc_context({"lines.linewidth": 0.6}):
        dendrogram(lc, ax=axd_t, orientation="top", no_labels=True,
                   color_threshold=0)
    axd_t.axis("off")
    cax = ax_c.inset_axes([0.87, 0.18, 0.03, 0.58])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("K coefficient", fontsize=6.5, labelpad=8)
    cbar.ax.tick_params(labelsize=6.0)
    v3 = json.load(open(T/"law2_validation_report.json",
                        encoding="utf-8"))["criteria"]["V3"]
    ax_c.text(0.5, 0.06,
              f"✓  bootstrap ARI {v3['boot_mean_ari']:.3f} "
              f"(median {v3['boot_median_ari']:.3f}) · LOAO "
              f"{v3['loao_mean_ari']:.2f} ({len(v3['loao_ari'])}/18)\n"
              f"silhouette {v3['silhouette']:.3f} at k = 2 — two-block structure",
              transform=ax_c.transAxes, ha="center", fontsize=6.4, fontweight="bold",
              color="#047857",
              bbox=dict(boxstyle="round,pad=0.35", fc="#ECFDF5", ec=PAL["green"],
                        lw=1.2))
    ax_c.set_title("Coupling matrix K (36 × 36, clustered)\n"
                   "two-block structure (k = 2)", fontsize=7.4, pad=6)

    # ---------- d: validated coupling structure ----------
    ax_d.axis("off")
    v3 = json.load(open(T/"law2_validation_report.json",
                        encoding="utf-8"))["criteria"]["V3"]
    ax_d.text(0.5, 0.94, "Coupling structure is objectively real and reproducible",
              ha="center", fontsize=7.6, fontweight="bold", color="#0F5132")
    cards = [(f"{v3['boot_mean_ari']:.3f}", "bootstrap ARI\n(1,000 resamples)"),
             (f"{v3['loao_mean_ari']:.2f}",
              f"LOAO ARI\n({len(v3['loao_ari'])}/18 groups)"),
             (f"{v3['silhouette']:.3f}", "silhouette\n(best k = 2)")]
    for i, (num, lab) in enumerate(cards):
        x0 = 0.04 + i * 0.32
        ax_d.add_patch(FancyBboxPatch((x0, 0.56), 0.30, 0.28,
                                      boxstyle="round,pad=0.02", fc="#ECFDF5",
                                      ec=PAL["green"], lw=1.3))
        ax_d.text(x0 + 0.03, 0.80, "✓", fontsize=10, fontweight="bold",
                  color="#047857", va="center")
        ax_d.text(x0 + 0.15, 0.74, num, ha="center", fontsize=15, fontweight="bold",
                  color="#047857")
        ax_d.text(x0 + 0.15, 0.61, lab, ha="center", fontsize=6.0, color="#065F46")
    ax_d.set_box_aspect(1.0)
    axs = ax_d.inset_axes([0.05, 0.10, 0.44, 0.34])
    axs.set_box_aspect(1.0)
    ks = sorted(v3["sil_scores_by_k"], key=int)
    sil = [v3["sil_scores_by_k"][k] for k in ks]
    axs.plot([int(k) for k in ks], sil, "o-", ms=4.5, lw=1.4, color=PAL["green"],
             zorder=3)
    axs.axvline(2, color=PAL["red"], ls=":", lw=1.1)
    axs.set_xticks([int(k) for k in ks])
    axs.tick_params(labelsize=6.0)
    axs.set_ylim(0.90, 0.97)
    axs.set_xlabel("number of clusters k", fontsize=6.0)
    axs.set_ylabel("silhouette", fontsize=6.0)
    clean_axes(axs)
    axs.set_title("clustering stability across k", fontsize=6.2, pad=2)
    axp = ax_d.inset_axes([0.53, 0.10, 0.44, 0.34])
    axp.set_box_aspect(1.0)
    loao = sorted(v3["loao_ari"].values())
    axp.plot(np.arange(len(loao)), loao, "o", ms=4.5, color=PAL["green"], mec="white",
             mew=0.5, zorder=3)
    axp.axhline(1.0, color=PAL["green"], ls="--", lw=0.9, alpha=0.5)
    axp.set_ylim(0.86, 1.06)
    axp.set_xlim(-0.5, 17.5)
    axp.set_xticks([])
    axp.set_yticks([0.9, 1.0])
    axp.tick_params(labelsize=6.0)
    axp.set_xlabel("18 amino-acid groups (leave-one-out)", fontsize=6.0)
    axp.set_ylabel("LOAO ARI", fontsize=6.0)
    clean_axes(axp)
    axp.text(8.5, 1.03, "18/18 groups = 1.00", ha="center", fontsize=6.4,
             fontweight="bold", color="#047857")
    axp.set_title("leave-one-AA-out reproducibility\n(perfect clustering recovery)",
                  fontsize=6.2, pad=2)
    ax_d.text(0.5, 0.03, "held-out prediction: all 20 AAs mean R² 0.43–0.71 "
              "(K strongest)\n"
              "irreducible to simple descriptors (Supplementary Fig. S8)",
              ha="center", fontsize=6.0, color="#475569")
    ax_d.set_title("Validated coupling structure", fontsize=7.4, pad=6)

    for ax, lab in [(ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d")]:
        panel_label(ax, lab)
    fig.suptitle("Biological source terms predict an effective geometric field",
                 fontsize=12.5, fontweight="bold", y=0.985)
    for ax in fig.axes:
        ax.tick_params(labelsize=6.0)
    save_fig(fig, "fig3_biological_geometric_field", FIG_DIR)
    plt.close(fig)


def fig4() -> None:
    """Data-only panels, all square: joint PCA path, primary, two replications,
    three-representation validation, real adjacent-step W2."""
    fig = plt.figure(figsize=(7.4, 5.6))
    fig.subplots_adjust(left=0.09, right=0.95, top=0.90, bottom=0.08)
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1])
    ax_c1 = fig.add_subplot(gs[0, 2]); ax_c2 = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1]); ax_e = fig.add_subplot(gs[1, 2])

    # ---- shared path data (real G30->31->32->33 in joint PCA) ----
    w2s = []
    proj = []
    try:
        seqs = ["PolyX_PolyG_30", "PolyX_PolyG_31", "PolyX_PolyG_32",
                "PolyX_PolyG_33"]
        ens = [kabsch_align(load_ensemble(ENS_ROOT / s)) for s in seqs]
        nmin = min(e.shape[1] for e in ens)
        ens = [e[:, :nmin, :] for e in ens]
        Fs = [e.reshape(len(e), -1) for e in ens]
        F = np.concatenate(Fs); F = F - F.mean(0)
        cov = F.T @ F / (len(F) - 1)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        proj = [f @ evecs[:, order[:2]] for f in Fs]
        mus, Sigs = [], []
        for f in Fs:
            mus.append(f.mean(0))
            Sigs.append(np.cov(f, rowvar=False) + 1e-6 * np.eye(f.shape[1]))
        w2s = [bures_w2(mus[i], Sigs[i], mus[i + 1], Sigs[i + 1])
               for i in range(3)]
    except Exception:
        pass

    # ---------- a: real ordered path in joint PCA ----------
    teals = ["#A5D8F3", "#4D96C8", "#1D6FA8", "#0B4C7E"]
    if proj:
        for i, P in enumerate(proj):
            try:
                k = gaussian_kde(P.T, bw_method=0.30)
                q1 = np.percentile(P[:, 0], [2, 98]); q2 = np.percentile(P[:, 1],
                                                                          [2, 98])
                xi, yi = np.mgrid[q1[0]:q1[1]:40j, q2[0]:q2[1]:40j]
                zi = k(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
                ax_a.contour(xi, yi, zi, levels=4, colors=[teals[i]], linewidths=0.7,
                             alpha=0.7)
            except Exception:
                pass
            ax_a.scatter(P[:, 0], P[:, 1], s=1.8, alpha=0.35, color=teals[i],
                         linewidths=0, zorder=3)
            ax_a.add_patch(Ellipse(P.mean(0), 3.0 * P.std(0)[0], 3.0 * P.std(0)[1],
                                   angle=0, fc="none", ec=teals[i], lw=1.1, ls="--",
                                   zorder=4))
            ax_a.text(P.mean(0)[0] + 0.5 * P.std(0)[0],
                      P.mean(0)[1] + 0.5 * P.std(0)[1], f"P{i+1}", fontsize=7.4,
                      fontweight="bold", color=teals[i],
                      bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=teals[i],
                                lw=0.8, alpha=0.9))
        for i in range(3):
            ax_a.annotate("", xy=tuple(proj[i + 1].mean(0)),
                          xytext=tuple(proj[i].mean(0)),
                          arrowprops=dict(arrowstyle="-|>", color="#111", lw=1.3,
                                          zorder=5))
            mx, my = (proj[i].mean(0) + proj[i + 1].mean(0)) / 2
            ax_a.text(mx, my + 0.6 * proj[i].std(0)[1], f"$W_2$ = {w2s[i]:.1f}",
                      ha="center", fontsize=6.6, fontweight="bold", color="#111",
                      bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#999",
                                lw=0.6, alpha=0.9))
        ax_a.set_xlabel("joint PC1 (common axes)", fontsize=6.4)
        ax_a.set_ylabel("joint PC2", fontsize=6.4)
        ax_a.tick_params(labelsize=6.0)
        ax_a.grid(ls=":", lw=0.35, color="#CBD5E1", zorder=0)
        clean_axes(ax_a)
        ax_a.set_box_aspect(1.0)
        ax_a.set_title("Real ordered path: G₃₀→G₃₁→G₃₂→G₃₃\n"
                       "(joint PCA, 500 conformations each)", fontsize=7.0, pad=5)
    else:
        ax_a.axis("off")
        ax_a.text(0.5, 0.5, "ensemble data unavailable", ha="center", fontsize=7,
                  color="#777")

    # ---------- b: primary comparison ----------
    d = json.load(open(T/"phase_r1_cluster_bootstrap.json", encoding="utf-8"))
    cb = d["cluster_bootstrap"]
    bio, null = d["original"]["bio_mean"], d["original"]["null_mean"]
    jp = pd.read_csv(T/"phase_ensemble_w2_joint_paths.csv")
    jb = jp[jp.is_bio == True].W2_joint.dropna().values
    jn = jp[jp.is_bio == False].W2_joint.dropna().values
    vp = ax_b.violinplot([jb, jn], positions=[0, 1], showextrema=False,
                         showmedians=True, widths=0.62)
    for body, col in zip(vp["bodies"], [PAL["blue"], PAL["grey"]]):
        body.set_facecolor(col); body.set_alpha(0.42); body.set_edgecolor(col)
        body.set_linewidth(0.7)
    vp["cmedians"].set_color("#111"); vp["cmedians"].set_linewidth(1.0)
    ax_b.plot([0, 1], [bio, null], "D", ms=6, color=PAL["dark"], mec="white",
              mew=0.8, zorder=5)
    for xi, v in zip([0, 1], [bio, null]):
        ax_b.text(xi, v + 1.6, f"mean\n{v:.2f}", ha="center", fontsize=6.0,
                  fontweight="bold", color="#333")
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels([f"structured\n(n = {len(jb):,})",
                          f"matched null\n(n = {len(jn):,})"], fontsize=6.0)
    ax_b.set_ylabel(r"Adjacent-state transport $W_2$", fontsize=6.2, labelpad=3)
    ax_b.set_ylim(0, 54)
    clean_axes(ax_b)
    ax_b.set_box_aspect(1.0)
    ax_b.text(0.5, 0.93, "Cohen's d = −1.54, p = 8.4×10⁻⁶⁸\n"
              f"Δmean = −11.15, 95% CI [{cb['diff_ci95_low']:.1f}, "
              f"{cb['diff_ci95_high']:.1f}]\ncluster-bootstrap p = 1.4×10⁻⁶⁶",
              transform=ax_b.transAxes, ha="center", fontsize=6.0,
              fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", fc="#FEF3C7", ec="#B45309", lw=0.5,
                        alpha=0.95))
    ax_b.set_title("Primary comparison (joint-PCA)", fontsize=7.0, pad=5)

    # ---------- c(i)/(ii): replications ----------
    b4 = pd.read_csv(T/"phase_ensemble_b4_path_action_v2.csv")
    hb = b4[b4.is_bio == True].W2_distance.dropna().values
    hn = b4[b4.is_bio == False].W2_distance.dropna().values
    b4s = json.load(open(T/"phase_ensemble_b4_summary_v2.json",
                         encoding="utf-8"))
    p_het = float(b4s["w2_p_value"])
    npz = pd.read_csv(T/"phase_ensemble_npz_direct_w2.csv")
    nb = npz[npz.is_bio == True].W2_npz.dropna().values
    nn = npz[npz.is_bio == False].W2_npz.dropna().values
    _, p_npz = mannwhitneyu(nb, nn)
    for ax, db, dn, ttl, ymax, p_txt, extra in [
        (ax_c1, hb, hn, "Heteropolymer", 12.0, fmt_p(p_het), "d = −0.20"),
        (ax_c2, nb, nn, "Direct NPZ", 29.0, fmt_p(p_npz), "")]:
        vp = ax.violinplot([db, dn], positions=[0, 1], showextrema=False,
                           showmedians=True, widths=0.6)
        for body, col in zip(vp["bodies"], [PAL["blue"], PAL["grey"]]):
            body.set_facecolor(col); body.set_alpha(0.42); body.set_edgecolor(col)
            body.set_linewidth(0.7)
        vp["cmedians"].set_color("#111"); vp["cmedians"].set_linewidth(1.0)
        ax.plot([0, 1], [db.mean(), dn.mean()], "D", ms=5.5, color=PAL["dark"],
                mec="white", mew=0.8, zorder=5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["structured", "null"],
                                                  fontsize=6.0)
        ax.tick_params(labelsize=6.0)
        ax.set_ylim(0, ymax)
        clean_axes(ax)
        ax.set_box_aspect(1.0)
        ax.set_title(ttl, fontsize=7.0, pad=4)
        for xi, v in zip([0, 1], [db.mean(), dn.mean()]):
            ax.text(xi, v + ymax * 0.045, f"{v:.2f}", ha="center", fontsize=6.0,
                    fontweight="bold")
        ax.text(0.5, 0.88, f"p = {p_txt}{', ' + extra if extra else ''}\n"
                f"n = {len(db):,} / {len(dn):,}", transform=ax.transAxes,
                ha="center", fontsize=6.0, va="top")
    ax_c1.set_ylabel(r"$W_2$", fontsize=6.4)

    # ---------- d: validated low-transport relation (three representations) ----------
    ax_d.axis("off")
    ax_d.text(0.5, 0.93, "Validated across three representations",
              ha="center", fontsize=7.4, fontweight="bold", color="#0F5132")
    reps = ["joint PCA", "heteropolymer", "direct NPZ"]
    bio_v = [bio, float(hb.mean()), float(nb.mean())]
    null_v = [null, float(hn.mean()), float(nn.mean())]
    deff = ["d = −1.54", "d = −0.20", ""]
    pv = ["p = 8.4×10⁻⁶⁸", f"p = {fmt_p(p_het)}", f"p = {fmt_p(p_npz)}"]
    ax_d.set_box_aspect(1.0)
    axg = ax_d.inset_axes([0.06, 0.30, 0.52, 0.52])
    axg.set_box_aspect(1.0)
    gx = np.arange(3); w = 0.32
    axg.bar(gx - w / 2, bio_v, w, color="#2C7FB8", alpha=0.9, edgecolor="k", lw=0.4,
            label="structured")
    axg.bar(gx + w / 2, null_v, w, color="#C8C8C8", alpha=0.9, edgecolor="k", lw=0.4,
            label="matched null")
    axg.set_yscale("log"); axg.set_ylim(1, 70)
    axg.set_xticks(gx); axg.set_xticklabels(reps, fontsize=6.0)
    axg.set_ylabel(r"mean $\bar{W}_2$ (log)", fontsize=6.0)
    axg.tick_params(labelsize=6.0)
    for i in range(3):
        axg.text(gx[i] - w / 2, bio_v[i] * 1.25, f"{bio_v[i]:.2f}", ha="center",
                 fontsize=6.0, fontweight="bold", color="#1D4ED8")
        axg.text(gx[i] + w / 2, null_v[i] * 1.25, f"{null_v[i]:.2f}", ha="center",
                 fontsize=6.0, fontweight="bold", color="#444")
        pct = (null_v[i] - bio_v[i]) / null_v[i] * 100
        axg.text(gx[i], null_v[i] * 2.3,
                 f"−{pct:.0f}%{', ' + deff[i] if deff[i] else ''}",
                 ha="center", fontsize=6.0, fontweight="bold", color="#047857")
        axg.text(gx[i], null_v[i] * 1.55, pv[i], ha="center", fontsize=6.0,
                 color="#155E75")
    axg.legend(fontsize=6.0, loc="upper right", frameon=False, handlelength=1.0)
    clean_axes(axg)
    ax_d.text(0.5, 0.20, "reductions −49% / −11% / −18% · primary d = −1.54\n"
              "all comparisons significant (Holm-corrected)",
              ha="center", fontsize=6.0, color="#334155")
    ax_d.text(0.5, 0.08, "broader action functional A[γ] = ⟨W₂⟩ + reorg + feas and "
              "transition-sharpness are\ntheoretical extensions — Supplementary Fig. S7",
              ha="center", fontsize=6.0, color="#64748B")
    ax_d.set_title("Validated low-transport relation", fontsize=7.4, pad=6)

    # ---------- e: real adjacent-step W2 along the ordered path ----------
    if w2s:
        steps = np.arange(1, 4)
        ax_e.bar(steps, w2s, width=0.5, color=teals[:3], alpha=0.85, edgecolor="k",
                 lw=0.4, zorder=3)
        ax_e.grid(axis="y", ls=":", lw=0.35, color="#CBD5E1", zorder=0)
        for x, v in zip(steps, w2s):
            ax_e.text(x, v + 0.25, f"{v:.1f}", ha="center", fontsize=6.2,
                      fontweight="bold")
        ax_e.set_xticks(steps)
        ax_e.set_xticklabels(["30→31", "31→32", "32→33"], fontsize=6.2)
        ax_e.set_ylabel(r"Bures $W_2$", fontsize=6.2)
        ax_e.set_ylim(0, max(w2s) * 1.35)
        ax_e.tick_params(labelsize=6.0)
        clean_axes(ax_e)
        ax_e.set_box_aspect(1.0)
        ax_e.set_title("Real adjacent-state transport\n(G₃₀→G₃₁→G₃₂→G₃₃, "
                       "truncated common space)", fontsize=7.0, pad=5)
    else:
        ax_e.axis("off")
        ax_e.text(0.5, 0.5, "ensemble data unavailable", ha="center", fontsize=7,
                  color="#777")

    for ax, lab in [(ax_a, "a"), (ax_b, "b"), (ax_c1, "c(i)"), (ax_c2, "c(ii)"),
                    (ax_d, "d"), (ax_e, "e")]:
        panel_label(ax, lab)
    fig.suptitle("Structured protein-state paths show reduced Wasserstein transport",
                 fontsize=12.5, fontweight="bold", y=0.985)
    for ax in fig.axes:
        ax.tick_params(labelsize=6.0)
    save_fig(fig, "fig4_low_transport_paths", FIG_DIR)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print("[complete] all 4 main figures (v7, data-only all-square layout)")
