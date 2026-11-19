# 实验配置

`main.yaml` 复用 A 默认配置。`b_*_s42.yaml` 是 B 的五组可执行 200-epoch 配置；dual 的 gamma=.5、source 的 gamma=0，K=1/4/8，单侧和双侧共享 K=1 控制。B 的 last checkpoint 每 10 epoch 保存，最终轮必存；best 选择规则不变。训练命令见 README_B.md。`experiment_template.json` 保留为历史字段草案，训练入口不读取它。

`c_*_s42.yaml` 是 C 的 E3（`seam_aware`）、E4（`seam_dp`）、E4-R（`seam_random`）配置，除 `augmentation` 外与 B 的 `b_dual_edge_k4_s42.yaml` 完全相同。接缝参数为**暂定起点**，需与 D 在验证集上选定；E4 参数一旦改动，须用 `scripts/calibrate_seam_random_mu.py` 重新校准 E4-R 的 `seam_random_mu`。说明见 README_C.md。
