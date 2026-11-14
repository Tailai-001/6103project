# 增强模块（B/C）

当前没有算法实现。B 负责 `edge_utils.py`、`source_edge.py`、`dual_edge.py` 与共享候选；C 负责 `seam_utils.py`、`seam_aware.py`、`seam_dp.py` 和 `seam_random.py`。文件名均为待实现交付物。

统一接口草案：`cutmix_batch(images, labels, policy, alpha, p, k, gamma, beta=0.0, seam_radius=0, seam_mu=0.0)`。policy 为 `none / random / source_edge / dual_edge / seam_aware / seam_dp / seam_random`；返回值建议为 mixed images、两组 labels、逐样本实际面积权重与 metadata。A/B/C/D 在实现前冻结 schema，增强函数不运行分类模型。

实验对应：E0 none、E1 random、E2 dual_edge、E3 seam_aware、E4 seam_dp；E2-S 为 source_edge 消融，E4-R 为 seam_random 形状对照。旧双侧 E3 已归入新 E2。

B/C 共用 Sobel、图像级归一化、候选与同分规则。E4/E4-R 从同一 E3 矩形出发；Source 从未修改 batch 读取，Mask=1 表示 Source。metadata 应支持返回最终 Mask、基础矩形、实际面积、边界计数、路径/回退状态；按需记录诊断，避免影响计时解释。

接缝代价使用 `[0,1]` RGB 的有效跨 Mask 四邻接像素对及通道平均，不计图像外填充；空/满 Mask 的边界指标为不适用。DP 要约束左右路径不交叉，并对最终 Mask 重新计算完整边界代价。坐标、退化与面积控制见 [实验协议](../docs/experiment_protocol.md)。
