# 增强模块（B/C）

B 已实现 `edge_utils.py`、`candidates.py`、`source_edge.py`、`dual_edge.py`，统一入口为 `cutmix_batch`。完整数学约定、metadata 和 C 接入示例见 [README_B.md](../README_B.md)。`cutmix.py` 原样复用 A E1。

C 的 `seam_utils.py`、`seam_aware.py`、`seam_dp.py`、`seam_random.py` 尚未实现，相关 policy 调用明确报错。C 应共用 B 的候选与边缘分数，再构造最终 Mask 并重算权重。
