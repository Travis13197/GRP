#!/usr/bin/env python3
"""Driver: regenerate every manuscript figure from committed inputs.

Usage:
    python figures/scripts/make_figures.py          # main + supplementary
    python figures/scripts/make_figures.py main     # only Figures 1-4
    python figures/scripts/make_figures.py supp     # only Figures S1-S9

This driver runs the two generator scripts in the same directory and then
verifies that all expected output files exist. Inputs are under figures/data/;
outputs are written to figures/out/{main,supplementary}.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent.parent
OUT = REPO / "figures" / "out"

MAIN = {
    "fig1_local_effective_geometry": "main",
    "fig2_perturbation_comparability": "main",
    "fig3_biological_geometric_field": "main",
    "fig4_low_transport_paths": "main",
}
SUPP = {
    "figS1_per_aa_spectral_decay": "supplementary",
    "figS2_dms_per_protein": "supplementary",
    "figS3_aa_fingerprints": "supplementary",
    "figS4_hydrophobicity_charge_phase": "supplementary",
    "figS5_state_relation": "supplementary",
    "figS6_law1_support": "supplementary",
    "figS7_law3_support": "supplementary",
    "figS8_field_support": "supplementary",
    "figS9_umap_tsne_atlas": "supplementary",
}
EXTS = ("svg", "jpg", "png", "pdf")


def run(script: str) -> None:
    print(f"\n=== running {script} ===")
    r = subprocess.run([sys.executable, str(SCRIPTS / script)],
                       cwd=str(REPO), check=False)
    if r.returncode != 0:
        raise SystemExit(f"{script} failed with exit code {r.returncode}")


def verify(targets: dict) -> None:
    missing = []
    for stem, sub in targets.items():
        for ext in EXTS:
            p = OUT / sub / f"{stem}.{ext}"
            if not p.exists():
                missing.append(str(p))
    if missing:
        raise SystemExit("Missing outputs:\n  " + "\n  ".join(missing))
    print(f"verified {len(targets)} figures x {len(EXTS)} formats")


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "main"):
        run("generate_final_figures_all_v7.py")
        verify(MAIN)
    if which in ("all", "supp"):
        run("generate_final_supplementary.py")
        verify(SUPP)
    print("[complete] figures written to", OUT)


if __name__ == "__main__":
    main()
