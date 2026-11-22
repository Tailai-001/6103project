# Member D｜Evaluation & Experimental Analysis

后续独立核查、接缝敏感性分析与本机同机三臂复核见 [补充核查](RESULTS_AUDIT.md)。下文保留 D 原交付的 18 次 seed-42 结果，新增运行另表列出。

共汇总 18 个完整的 200-epoch 正式运行（A 2、B 5、C 11）。模型为 CIFAR 版四阶段 ResNet-18，CIFAR-100 使用固定 45,000/5,000 分层训练／验证划分，所有现有正式结果为 **seed 42**。Best Validation Top-1 取训练日志最大值，同分取最早 epoch。Last-10-Epoch Mean 为第 191–200 轮的平均值。Macro-F1 来自重新加载最佳 checkpoint 后的干净验证结果。

## 三个研究问题

| 问题 | 现有证据 | 可以写进报告的结论 |
|---|---|---|
| E4 是否优于 E1？ | E4 r=2 为 **80.30%**（末 10 轮 **80.16%**）。A 的 E1 为 **80.52% / 80.12%**，C-A2 的 E1 重跑为 **80.02% / 79.82%**。C 的 E1、E4、E4-R 使用相同训练设置和数据，但 C-A2 与 C-main 是不同的 4090 主机。 | 目前无法证明 E4 超过 Original CutMix。r=3 的 **80.52%** 来自第三台主机，是查看已有验证结果后发现的较好配置，需要独立复核。 |
| E4 是否优于 E4-R？ | C-main 上最佳验证 Top-1 **80.30% 对 79.46%**（**+0.84 个百分点**），末 10 轮均值 **80.16% 对 79.24%**（**+0.92 个百分点**），Macro-F1 **0.8025 对 0.7927**。 | 这是目前控制最好的单 seed 对照，支持内容引导 DP 在本次运行中优于随机边界。两组共用基础 E3 矩形和约束，随机臂另校准弯曲惩罚。最终形状仍有差异，无法把全部差距归因于接缝目标。 |
| E3 为什么没有明显提高准确率？ | 相对 B 的 E2 K=4，E3 最佳值只高 **0.08 个百分点**，末 10 轮均为 **79.524%**。与 C-A2 的 E2 比较时方向相反（最佳 **−0.12 个百分点**）。在固定训练图像上，E3 把原始接缝差从 **0.3066** 降到 **0.2470**。 | 改善 RGB 连续性本身未观察到稳定的分类收益。接缝指标衡量局部颜色差，无法衡量语义保真度；更平滑的边界可能削弱 CutMix 的正则化扰动。后者是待验证假设。 |

C-main 的 E4-R 在 CPU 重新评价为 79.44%，GPU 日志为 79.46%，相差一张图。分类总表保留 GPU 日志的最佳值，Macro-F1 注明来自 CPU。A 原始日志缺少 Macro-F1 和单独的增强计时，显示为 N/A。

## 边界质量与计算成本

D 在 NTU 上复用固定训练划分和 C 的默认形状配置审计 2,560 张配对影像（20 批 × 128，诊断时强制混合）。E1–E3 的替换面积逐样本一致，E4 和 E4-R 的基础矩形逐样本一致。内部边界长度按跨 Mask 的四邻接像素对计数。无有效接缝的样本按对应指标记为 N/A。

E4 / E4-R 的平均边界长度为 **60.92 / 66.81** 个像素对（随机臂约长 **9.7%**）、平均替换面积比为 **0.303 / 0.294**、平均拐弯数为 **8.97 / 9.20**、回退率为 **7.76% / 8.43%**。E4 约 **15.7%** 的改形样本新贴到图像左右边缘。观测到的接受路径和退回原矩形的 Mask 均通过几何检查。

用随图片一起裁剪和翻转的有效像素通道识别真实填充位置。E4 约 **6.45%** 的接缝像素对两端都是填充，排除后 E4 的 RGB 均值由 **0.2098** 变为 **0.2242**。进一步排除任意一端含填充的像素对后，E4 / E4-R 为 **0.2147 / 0.2614**，E3 / E2 为 **0.2543 / 0.2910**。两组接缝质量的比较方向仍在。填充会改变所纳入的像素对，三种汇总口径及各自 N/A 数量应一起保留。

C-main 同一台 4090 的同步增强计时显示，E2 / E3 / E4 / E4-R 分别为 **8.47 / 9.17 / 19.79 / 19.34 ms/批**。E4 相比 E2 增加 **11.32 ms/批**，相比 E3 的完整训练进程增加约 **8.21 分钟**。另在 C-A2 上同机测量，E1 为 **8.75 ms/批**，E4 为 **24.78 ms/批**（约 **2.83 倍**）。计时采用 batch 128、float32、10 次预热和 50 次重复，包含增强，数据读取与模型计算在计时之外。C-A2 的增强调用计时与 C-main 的正式训练结果各自对应原主机。

## 报告素材和复现

- [主表 Markdown](results/tables/main_results.md) · [主表 CSV](results/tables/main_results.csv) · [验证曲线](results/figures/d_validation_curves.png)
- [K / β / r / μ 消融表](results/tables/d_ablation.md) · [消融 CSV](results/tables/d_ablation.csv)
- [边界指标表](results/tables/d_boundary_quality.md) · [指标分布](results/figures/d_boundary_distributions.png) · [混合图像与边缘／回退案例](results/figures/d_mixed_examples.png)
- [效率数据](results/tables/d_efficiency.csv) · [效率图](results/figures/d_efficiency.png)
- [逐样本形状记录](results/member_d/boundary_samples.csv) · [审计 JSON](results/member_d/boundary_summary.json) · [评估入口验证记录](results/member_d/evaluator_smoke.json)

```bash
python analysis/compile_results.py
python analysis/boundary_audit.py --data-root /path/to/existing/cifar100
python analysis/plot_boundary.py
python -m pytest tests -q
python evaluate.py --checkpoint /path/to/best.pth --run-id RUN_ID \
  --log /path/to/epochs.csv --data-root /path/to/existing/cifar100 --split validation
```

[统一评估入口](evaluate.py) 输出 Top‑1、Macro‑F1、交叉熵、checkpoint 哈希和固定验证划分哈希。冻结参数后可用 `--split test` 评价官方测试集，入口要求带完整训练日志以核对最佳验证 epoch。本次复用已有实验，正式同主机 E1/E4/E4-R 三臂、多 seed 复核以及官方测试分数仍为空。结论限于已有的单 seed 验证和独立增强审计。
