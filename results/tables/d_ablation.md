# Ablations

Seed 42, fixed validation split. Keep machine labels when comparing rows. C-A2 E2 is the β=0 reference.
The r=3 row was selected after viewing these validation runs.

| Factor | Run | Machine | K | β | r | μ | Best Val Top-1 (%) | Last 10 Mean (%) | Val Macro-F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| K (edge-guided) | `b_dual_edge_k1_s42_seq` | B-original | 1 | — | — | — | 80.06 | 79.80 | 0.7996 |
| K (edge-guided) | `b_dual_edge_k4_s42_seq` | B-original | 4 | — | — | — | 79.64 | 79.52 | 0.7956 |
| K (edge-guided) | `b_source_edge_k4_s42_seq` | B-original | 4 | — | — | — | 80.36 | 80.20 | 0.8023 |
| K (edge-guided) | `b_dual_edge_k8_s42_seq` | B-original | 8 | — | — | — | 79.62 | 79.28 | 0.7946 |
| K (edge-guided) | `b_source_edge_k8_s42_seq` | B-original | 8 | — | — | — | 79.30 | 79.07 | 0.7924 |
| β (rectangle seam) | `c_e3_k4_s42` | C-main | 4 | 1.0 | — | — | 79.72 | 79.52 | 0.7962 |
| β (rectangle seam) | `c_ctrl_e2_k4_s42` | C-A2 | 4 | 0.0 | — | — | 79.84 | 79.73 | 0.7977 |
| β (rectangle seam) | `c_e3_b05_k4_s42` | C-A2 | 4 | 0.5 | — | — | 79.96 | 79.80 | 0.7980 |
| β (rectangle seam) | `c_e3_b2_k4_s42` | C-A2 | 4 | 2.0 | — | — | 80.16 | 79.81 | 0.8005 |
| r / μ (DP boundary) | `c_e4_k4_s42` | C-main | 4 | 1.0 | 2 | 0.05 | 80.30 | 80.16 | 0.8025 |
| r / μ (DP boundary) | `c_e4_r1_k4_s42` | C-B | 4 | 1.0 | 1 | 0.05 | 80.10 | 79.74 | 0.8001 |
| r / μ (DP boundary) | `c_e4_r3_k4_s42` | C-B | 4 | 1.0 | 3 | 0.05 | 80.52 | 80.41 | 0.8040 |
| r / μ (DP boundary) | `c_e4_mu0_k4_s42` | C-B | 4 | 1.0 | 2 | 0.0 | 79.92 | 79.75 | 0.7985 |
| r / μ (DP boundary) | `c_e4_mu02_k4_s42` | C-B | 4 | 1.0 | 2 | 0.2 | 80.04 | 79.81 | 0.7994 |
