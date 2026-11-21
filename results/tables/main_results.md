# Main results

CIFAR-100, fixed 45,000/5,000 train/validation split, seed 42, 200 epochs. Top-1 values are validation percentages.
Machine labels identify distinct hosts or original environments. A's original run times are sums of epoch durations;
B/C times are complete process wall times. Timing values should be compared within their measurement host.

| Experiment | Run | Machine / GPU | Best Val Top-1 (%) | Last 10 Mean (%) | Val Macro-F1 | Time (min) | Aug. (ms/batch) |
|---|---|---|---:|---:|---:|---:|---:|
| E0 | `e0_final_s42` | A-original / NVIDIA A40 | 77.74 | 77.53 | N/A | 36.10 | N/A |
| E1 | `e1_final_s42` | A-original / NVIDIA A40 | 80.52 | 80.12 | N/A | 54.95 | N/A |
| E1-equivalent | `b_dual_edge_k1_s42_seq` | B-original / NVIDIA A800-SXM4-80GB | 80.06 | 79.80 | 0.7996 | 46.58 | 13.95 |
| E2 | `b_dual_edge_k4_s42_seq` | B-original / NVIDIA A800-SXM4-80GB | 79.64 | 79.52 | 0.7956 | 50.47 | 14.14 |
| E2-S | `b_source_edge_k4_s42_seq` | B-original / NVIDIA A800-SXM4-80GB | 80.36 | 80.20 | 0.8023 | 46.29 | 14.75 |
| E2 | `b_dual_edge_k8_s42_seq` | B-original / NVIDIA A800-SXM4-80GB | 79.62 | 79.28 | 0.7946 | 46.70 | 14.78 |
| E2-S | `b_source_edge_k8_s42_seq` | B-original / NVIDIA A800-SXM4-80GB | 79.30 | 79.07 | 0.7924 | 45.37 | 14.38 |
| E3 | `c_e3_k4_s42` | C-main / NVIDIA GeForce RTX 4090 | 79.72 | 79.52 | 0.7962 (CPU) | 38.39 | 9.17 |
| E4 | `c_e4_k4_s42` | C-main / NVIDIA GeForce RTX 4090 | 80.30 | 80.16 | 0.8025 (CPU) | 46.60 | 19.79 |
| E4-R | `c_e4r_k4_s42` | C-main / NVIDIA GeForce RTX 4090 | 79.46 | 79.24 | 0.7927 (CPU) | 46.76 | 19.34 |
| E2 | `c_ctrl_e2_k4_s42` | C-A2 / NVIDIA GeForce RTX 4090 | 79.84 | 79.73 | 0.7977 | 34.76 | 10.55 |
| E1 | `c_ctrl_e1_s42` | C-A2 / NVIDIA GeForce RTX 4090 | 80.02 | 79.82 | 0.7991 | 33.76 | 8.75 |
| E3 | `c_e3_b05_k4_s42` | C-A2 / NVIDIA GeForce RTX 4090 | 79.96 | 79.80 | 0.7980 | 36.04 | 11.00 |
| E3 | `c_e3_b2_k4_s42` | C-A2 / NVIDIA GeForce RTX 4090 | 80.16 | 79.81 | 0.8005 | 36.02 | 11.06 |
| E4 | `c_e4_r1_k4_s42` | C-B / NVIDIA GeForce RTX 4090 | 80.10 | 79.74 | 0.8001 | 58.06 | 30.67 |
| E4 | `c_e4_r3_k4_s42` | C-B / NVIDIA GeForce RTX 4090 | 80.52 | 80.41 | 0.8040 | 57.57 | 32.05 |
| E4 | `c_e4_mu0_k4_s42` | C-B / NVIDIA GeForce RTX 4090 | 79.92 | 79.75 | 0.7985 | 57.60 | 30.90 |
| E4 | `c_e4_mu02_k4_s42` | C-B / NVIDIA GeForce RTX 4090 | 80.04 | 79.81 | 0.7994 | 57.54 | 31.59 |

C-main E4-R was 79.46% in the training log and 79.44% in CPU checkpoint re-evaluation (one image).
The table uses the training-log Top-1. Official test scores have not been measured in this handoff.
