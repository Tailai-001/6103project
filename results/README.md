# Public experiment evidence

The original [18-run table](tables/main_results.md) and [CSV](tables/main_results.csv) cover A's 2 baselines, B's 5 content-selection experiments and C's 11 main/control/ablation runs. A's raw CSVs remain in the archived A handoff; B/C raw logs and run metadata are under `member_b/` and `member_c/`.

The three additional [same-host confirmation runs](local_audit/local_confirmation.md) are under `local_audit/raw_logs/`, with saved configurations, environment records and validation predictions. The complete project therefore contains **21 formal 200-epoch runs**. They are grouped by host and seed, not pooled into a controlled multi-seed claim.

`member_d/` contains the paired training-image boundary audit. [Boundary tables](tables/d_boundary_quality.md), efficiency tables and `figures/` trace back to these inputs. All currently reported classification metrics are validation metrics; official-test fields remain empty. Datasets and checkpoints are excluded.

Rebuild tables and four plots with `python analysis/reproduce_results.py --out-dir outputs/reproduced` from the root. The command recomputes confirmation accuracy/F1 from public predictions, verifies raw logs and writes an input-hash manifest. [Result interpretation](../RESULTS_D.md) · [Independent audit](../RESULTS_AUDIT.md).
