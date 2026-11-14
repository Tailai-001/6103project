# 模型（A）

待实现 CIFAR-100 的 ResNet-18：四个 stage、BasicBlock 数量 `[2,2,2,2]`、3×3 stride 1 stem、无初始 MaxPool、100 类输出。所有 policy 共用模型与初始化规则。

官方 `third_party/CutMix-PyTorch/resnet.py` 的 CIFAR 路径采用另一种三 stage 架构，不能仅设置 `depth=18` 代替。这里尚无本组模型实现。
