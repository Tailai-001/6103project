# 第三方代码与来源

`CutMix-PyTorch/` 是官方仓库的完整小型工作树快照，**未作修改**，不包含嵌套 `.git`，普通 clone 即可取得源码，不依赖 submodule 初始化。

- 上游：https://github.com/clovaai/CutMix-PyTorch
- 固定提交：[`2d8eb68faff7fe4962776ad51d175c3b01a25734`](https://github.com/clovaai/CutMix-PyTorch/commit/2d8eb68faff7fe4962776ad51d175c3b01a25734)
- 导入日期：2026-11-14
- [MIT LICENSE](CutMix-PyTorch/LICENSE) 和 [NOTICE](CutMix-PyTorch/NOTICE) 保留上游及其子组件的全部声明。
- [来源与逐文件 SHA-256](cutmix_provenance.json)

| 文件 | 为什么保留 |
| --- | --- |
| `train.py` | 原始 CutMix batch 混合、`rand_bbox`、混合损失、训练与 checkpoint 参考 |
| `test.py` | 原始评价入口与模型加载参考 |
| `resnet.py` | 模型结构参考；CIFAR 分支不等于本组要求的 ResNet-18 |
| `pyramidnet.py` | 原论文 CIFAR 主模型，且被上游训练/测试入口导入 |
| `utils.py` | 上游训练依赖的图像处理工具 |
| `README.md`, `img1.PNG` | 上游说明、示意图和原始运行示例 |
| `.gitignore`, `LICENSE`, `NOTICE` | 完整快照、许可与来源信息 |

这份快照是研究参考，不代表已经在本项目验证可运行。适配工作应写到顶层项目模块，标注来源并保留适用许可；不要直接改动快照后仍声称它未修改。

主要适配项：`np.int` 等历史接口、硬编码 CUDA、独立验证集、四 stage CIFAR ResNet-18，以及更清楚的 CutMix 参数命名。当前统一运行命令见根 [README](../README.md)，完整适配来源见 [NOTICE.md](../NOTICE.md)；`python scripts/verify_repository.py` 会核对本快照的逐文件哈希。
