# Data

Minimal committed inputs required to reproduce the manuscript analyses.

## `dms/`

Deep-mutational-scanning fitness tables for the eight proteins analyzed in Result 2:

| File | Protein | Notes |
|------|---------|-------|
| `blat.tsv`, `gfp.tsv`, `hras.tsv`, `hsp90.tsv`, `p53.tsv`, `pten.tsv`, `spike.tsv`, `ube4b.tsv` | BLAT / GFP / HRAS / HSP90 / P53 / PTEN / SPIKE_RBD / UBE4B | quality-controlled DMS scores |
| `phase6_dms_master_table.csv` | all eight | consolidated master table |
| `phase6_dms_single_mutants.csv` | all eight | single-substitution subset |

Raw sources are the public ProteinGym `DMS_ProteinGym_substitutions` files. The eight proteins and the 84,361-mutation working set are defined by the preprocessing in `../analysis/r2_perturbation_cost/phase6_cgeo_dms_calibration.py`.

## `sequences/`

Wild-type FASTA definitions for the DMS proteins (`*_wt.fasta`) plus the consolidated `wt_sequences.json`.

## Not committed (regenerated)

- BioEmu conformational ensembles (NPZ, ~4.4 GB) — regenerate with the `bioemu` environment.
- Full-atom / side-chain reconstructions (XTC/PDB).
- ProstT5 / ProtT5 / ESM-C learned embeddings.

See [`../AI_CODER.md`](../AI_CODER.md) for regeneration commands.
