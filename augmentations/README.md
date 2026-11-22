# 增强模块（B/C）

B 已实现 `edge_utils.py`、`candidates.py`、`source_edge.py`、`dual_edge.py`，统一入口为 `cutmix_batch`。完整数学约定、metadata 和 C 接入示例见 [README_B.md](../README_B.md)。`cutmix.py` 原样复用 A E1。

C 已实现 `seam_utils.py`、`seam_aware.py`（E3）、`seam_dp.py`（E4）和 `seam_random.py`（E4-R），全部共用 B 的候选、Sobel 分数与同分规则，并按最终 Mask 重算标签权重。七个 policy 均已接入 `cutmix_batch`，未知 policy 报 `ValueError`。设计、参数、测试和真实数据诊断见 [README_C.md](../README_C.md)。

| 文件 | 内容 |
| --- | --- |
| `seam_utils.py` | 跨 Mask 四邻接像素对的 RGB 接缝代价 `D_seam`（图外不计、空/满 Mask 计数为 0 表示不适用）、图像级归一化 `D_hat`、候选批量计算 |
| `seam_aware.py` | E3：`edge_B − γ·edge_A − β·D_hat`；β=0 与 `dual_edge` 逐位一致，不额外消耗随机数 |
| `seam_dp.py` | E4：每行左右边界局部代价、左右边联合 DP（最小宽度、不交叉、弯曲惩罚 μ）、面积预算与回退状态码、最终 Mask 的完整 `D_seam` 重算；`seam_radius=0` 与 E3 逐位一致 |
| `seam_random.py` | E4-R：同一 E3 矩形与全部约束/回退规则，DP 改用随机代价；`seam_random_mu` 为其单独校准的弯曲惩罚 |

`cutmix_batch` 的 `seam_radius` 默认是 0，E4/E4-R 的配置必须显式设置，否则会退化为 E3。
