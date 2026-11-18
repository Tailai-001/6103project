# 数据

`dataset.py` 与 `split.py` 原样复用 A。训练只从 CIFAR-100 官方训练集按 A 保存的固定索引划分 45,000/5,000，ToTensor 前训练使用 RandomCrop/Flip，模型标准化在可选混合后执行。验证仅 ToTensor，不做混合。

原始数据存放 `data/cifar100/`，由 `.gitignore` 排除。B 的 GPU 工作副本已下载并核验官方归档 MD5；不提交原始图像。测试集不用于 B 的训练、调参或 checkpoint 选择。固定索引和校验值见 splits/README.md。
