# 模型

`resnet.py` 原样复用 A 的 CIFAR-100 四 stage `[2,2,2,2]` ResNet-18，3×3 stride 1 stem、无初始 MaxPool、100 类输出。所有已实现 policy 共用模型与初始化。
