# 增强模块（B/C）

计划交付 `edge_utils.py`、`source_edge.py`、`dual_edge.py` 与统一增强入口。当前没有算法实现。

统一调用：`cutmix_batch(images, labels, policy, alpha, p, k, gamma)`，policy 为 `none / random / source_edge / dual_edge`。A/B/C 在实现前冻结返回值；建议返回 mixed images、两组 labels 和实际面积权重，而不在增强函数中运行模型。

共用 Sobel、图像归一化、候选矩形与评分代码；E2/E3 用同一随机候选集。坐标、退化行为和面积匹配规则见 [实验协议](../docs/experiment_protocol.md)。
