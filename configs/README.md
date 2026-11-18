# 实验配置

`main.yaml` 复用 A 默认配置。`b_*_s42.yaml` 是 B 的五组可执行 200-epoch 配置；dual 的 gamma=.5、source 的 gamma=0，K=1/4/8，单侧和双侧共享 K=1 控制。B 的 last checkpoint 每 10 epoch 保存，最终轮必存；best 选择规则不变。训练命令见 README_B.md。`experiment_template.json` 保留为历史字段草案，训练入口不读取它。
