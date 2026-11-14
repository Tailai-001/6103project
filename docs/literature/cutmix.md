# CutMix 原文阅读记录

阅读版本：[arXiv:1905.04899v2](https://arxiv.org/abs/1905.04899v2)，2019-08-07 修订，完整 PDF 14 页，含附录。已阅读全文，并核对方法公式和 Algorithm A1。官方代码版本见 [来源清单](../../third_party/cutmix_provenance.json)。

## 与本项目直接相关的内容

- **§3.1，Eq. (1)–(2)：** 图像局部替换、标签按面积混合；λ 服从对称 Beta 分布，原文 α=1；宽、高共同按 `sqrt(1−λ)` 缩放。
- **Appendix A / Algorithm A1：** batch 内随机配对、中心采样、边界裁剪，之后重算实际面积比例。新实现应保留这个顺序。
- **§4.1.2 / Table 5：** CIFAR 主模型是 PyramidNet-200，训练 300 epochs，batch 64；本文结果不能用作本组 ResNet-18 的实测基线。
- **§4.1.3 / Table 8：** 包含中心高斯选区、固定大小和标签规则等消融。位置偏置需要验证，不能预设获益。
- **§4.2–4.4、Appendix B–E：** 涉及定位、迁移、鲁棒性和 OOD；本组分类实验不能自动支持这些任务的结论。

## 本组由此提出的设计与实现注意事项

下面是本组的推论和待验证设计，不是原文结论：

1. 先固定裁剪后的矩形高宽再平移候选，可在改变位置策略时匹配实际面积分布；见 [实验协议](../experiment_protocol.md)。
2. Sobel 衡量低层边缘，不等同于语义显著性。需要观察背景纹理、高边缘失败案例和性能/开销关系。
3. 官方代码用 CIFAR 的 `train=False` 作为逐轮评估集；本项目需另设验证集。代码中的 `--beta` 才是 CutMix α，`--alpha` 是 PyramidNet 参数。
4. 官方 CIFAR ResNet 与本组四 stage ResNet-18 不同；同时要处理 `np.int` 和硬编码 CUDA 等历史接口。

## 后续相关工作清单

- [SaliencyMix: A Saliency Guided Data Augmentation Strategy for Better Regularization](https://arxiv.org/abs/2006.01791)
- [Puzzle Mix: Exploiting Saliency and Local Statistics for Optimal Mixup](https://proceedings.mlr.press/v119/kim20b.html)

以上两项由原始项目说明提出，已补上原文入口，待 E 与 B/C 进一步精读后完善比较。不要把它们描述成已经完成的实验对照。
