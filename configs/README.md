# Experiment configurations

All YAML files run through the same root `train.py`. Use the [README](../README.md) for commands and the [protocol](../docs/experiment_protocol.md) for the recorded settings.

| Files | Purpose |
|---|---|
| `main.yaml` | A's 200-epoch E0 default; use `--policy random` for E1 |
| `b_*_s42.yaml` | Five B content-selection runs, K=1/4/8, seed 42 |
| `c_seam_*_k4_s42.yaml` | Main E3, E4 and E4-R runs |
| `c_ctrl_*`, `c_abl_*` | C's same-GPU-model controls and beta/radius/bending ablations |
| `local_confirm_*_s43.yaml` | Three same-host confirmation runs, seed 43, CPU augmentation + MPS model |
| `smoke_cpu.yaml` | One-epoch integration check, batch 4, workers 0; not formal evidence |

The historical B/C and confirmation YAMLs are retained as recorded, including the settings used in saved logs. They are reproducibility configurations, not a claim of globally optimal hyperparameters. E4/E4-R explicitly set radius 2, minimum width 2 and area budget 0.2; E4-R uses its training-shape-calibrated bend penalty 0.225. Recalibrate that control if changing its corresponding geometry.

Copy a configuration to adjust data paths or loader workers. Preserve the supplied split and use a unique run name. `experiment_template.json` is an archived design schema, not an executable trainer input.
