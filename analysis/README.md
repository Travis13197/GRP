# Analysis scripts

Manuscript analysis scripts, organized by result. Each subdirectory maps to one of the five results in **"An Effective Geometric Field Theory of Protein Space"**.

| Directory | Result | Primary scripts |
|-----------|--------|-----------------|
| `common/` | shared utilities | `phase_l1_kabsch_metric.py` (Kabsch + Ledoit-Wolf metric), `phase_l12_statistical_hygiene.py`, `phase_r1_cluster_bootstrap.py` |
| `r1_ensemble_geometry/` | Result 1: local effective geometry | `phase1_conformation_geometry.py`, `phase9_comprehensive_geometry.py`, `plan2_fss_scaling_analysis.py` |
| `r2_perturbation_cost/` | Result 2: covariance-normalized perturbation cost | `phase_o1_cgeo_v3_deterministic.py`, `phase6_cgeo_dms_calibration.py`, `phase_o4_analyze.py` |
| `r3_source_field/` | Result 3: effective geometric field | `phase_ensemble_b2_coupling_matrix.py`, `law2_pattern_descriptors.py` |
| `r4_path_transport/` | Result 4: low-transport paths | `phase_ensemble_w2_joint_recompute.py`, `phase_ensemble_npz_direct_w2.py` |
| `r5_cross_representation/` | Result 5: synthesis + cross-representation | `phase_l2_intrinsic_reanalysis.py`, `phase_x_embedding_laws.py` |
| `sampling/` | sequence / A3M preparation | `phase_o4_prepare.py`, `phase_l6_generate_fasta.py` |

The scripts assume the input layout described in [`../AI_CODER.md`](../AI_CODER.md). They are the canonical, cleaned analysis entry points; the full historical workspace (hundreds of exploratory, fix and diagnostic scripts) is intentionally not included here.

Run order and exact data flow for each result are documented in [`../AI_CODER.md`](../AI_CODER.md#6-the-five-results-and-their-scripts).
