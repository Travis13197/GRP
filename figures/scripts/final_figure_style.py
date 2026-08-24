"""Nature-grade shared figure style for the GRP final paper."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# curated ColorBrewer-inspired palette (colorblind-safe-ish, print-friendly)
PAL = {
    "blue": "#2C7FB8",      # main blue
    "teal": "#1D91C0",
    "green": "#31A354",
    "red": "#DE2D26",
    "purple": "#756BB1",
    "orange": "#F16913",
    "gold": "#F4A261",
    "grey": "#969696",
    "grey_light": "#C8C8C8",
    "dark": "#252525",
    "soft_blue": "#BFD7EA",
    "soft_red": "#F5C6C4",
    "soft_green": "#C7E9C0",
    "paper": "#FFFFFF",
}

# per-AA curated colors (Tab10-like but harmonized)
AA_COLORS = {
    "G": "#2C7FB8", "S": "#6BAED6", "E": "#DE2D26", "L": "#8B1A1A",
    "K": "#31A354", "A": "#F16913", "V": "#F4A261", "I": "#8C6D31",
    "F": "#C51B7D", "P": "#7F2704", "W": "#525252", "Y": "#41AB5D",
    "T": "#FDD0A2", "H": "#A6BDDB", "D": "#FB6A4A", "N": "#74C476",
    "Q": "#BDBDBD", "R": "#238B45", "M": "#C7E9C0", "C": "#FFF7BC",
}


def apply_style():
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "nature"])
    except Exception:
        pass
    plt.rcParams.update({
        "text.usetex": False,
        "pgf.rcfonts": False,
        "font.family": "Arial, DejaVu Sans",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.0,
        "axes.titleweight": "bold",
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.fontsize": 6.0,
        "legend.frameon": False,
        "legend.borderpad": 0.4,
        "legend.handlelength": 1.4,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def clean_axes(ax, spines=("top", "right")):
    for s in spines:
        ax.spines[s].set_visible(False)


def panel_label(ax, lab, x=-0.09, y=1.05, fs=14):
    ax.text(x, y, lab, transform=ax.transAxes, fontsize=fs, fontweight="bold",
            va="bottom", ha="right")


def save_fig(fig, path_stem, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "jpg", "png", "pdf"):
        fig.savefig(out_dir / f"{path_stem}.{ext}", dpi=300, bbox_inches="tight")
    print(f"[saved] {path_stem}.{{svg,jpg,png,pdf}}")
