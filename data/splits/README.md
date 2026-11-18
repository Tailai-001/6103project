# A 固定划分（B 原样复用）

索引来自 CIFAR-100 官方 50,000 张训练图像，split seed=42，分层划分为 45,000/5,000。每类训练 450 张、验证 50 张，完整覆盖且无重叠。B 未重新生成或更改 A 的索引。

| 文件 | SHA-256 |
|---|---|
| train_indices.npy | 24520246ef0988e61bb0fe0f5c66da3a1f944be0569ccb5f0afc2536b92dfc15 |
| val_indices.npy | 2c80060ed0fa8eba98b2023be25aa56794982c9d6bf762dd1eb29205835d67c2 |

B 验证证据见 results/member_b/split_validation.json。文件不含原始图像或个人信息。
