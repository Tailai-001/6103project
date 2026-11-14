# AI6103 Final Project — Lightweight Edge-Guided CutMix

**Track 2: Improving an Existing Paper**  
**基础论文：** *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features*（ICCV 2019）  
**官方代码：** https://github.com/clovaai/CutMix-PyTorch

## 0. 从这里开始

本仓库是 AI6103 五人小组的**项目起始仓库**，目前已整理项目说明、实验协议草案、目录分工和官方代码快照。**本组 E0–E3 训练框架、边缘增强算法及实验结果尚未实现或运行。** 下文的成员任务和验收标准是后续开发计划。

- [原始项目说明（原样归档）](docs/project_brief.md)
- [实验协议与待定事项](docs/experiment_protocol.md)
- [CutMix 原文阅读笔记](docs/literature/cutmix.md)
- [代码来源、版本和适配清单](third_party/README.md)
- [协作方式](CONTRIBUTING.md) · [实验登记表](experiments/registry.csv) · [结果表模板](results/tables/main_results.csv)

基础论文：[arXiv 页面](https://arxiv.org/abs/1905.04899) · [完整原文 PDF（含附录）](https://arxiv.org/pdf/1905.04899)。仓库保留原文链接和阅读笔记，便于组员核查。

```bash
git clone https://github.com/Tailai-001/6103project.git
cd 6103project
```

当前没有项目级 `train.py` / `evaluate.py` 或已验证的训练环境；请先完成 A 的训练框架并冻结实验配置，再发布可复现的训练命令。`third_party/CutMix-PyTorch/train.py` 是保留的历史源码，使用方式和限制见第 4 节。

## 1. 项目概述

### 原论文与研究问题

CutMix 是一种图像分类数据增强方法：从一张图像中随机选取矩形 patch，粘贴到另一张图像，并根据实际混合面积加权两个类别标签。它能改善泛化性能，但**patch 的位置是随机选取的**，没有考虑哪些区域更有视觉信息，也没有考虑被覆盖区域的信息损失。

### 我们的改进方案

以 **CIFAR-100 + CIFAR 适配版 ResNet-18** 为统一实验平台，在原始 CutMix 基础上实现两种轻量方法：

- **E0 — No CutMix：** 标准 ResNet-18，作为无混合增强基线。
- **E1 — Original CutMix：** 按原论文随机选取矩形区域，作为主要对照组。
- **E2 — Source Edge-Guided CutMix：** 用 Sobel 计算 Source 图像的边缘强度；生成 `K` 个面积和形状相同、位置不同的候选矩形，选择平均边缘强度最高的区域进行粘贴。
- **E3 — Dual-Sided Edge-Guided CutMix：** 同时评估 Source 的边缘信息和 Target 被覆盖区域的边缘信息，按 `score = source_score - γ × target_score` 选择候选区域，兼顾粘贴信息与保留信息。两个得分应先在各自图像内采用一致的归一化规则。

**核心验证问题：** E2 是否优于随机 CutMix？E3 是否比 E2 更合理？提升（若有）是否值得额外计算成本？边缘强度是否真正代表有效的分类信息？不预设改进方法一定获得更高准确率。

**统一原则：** E1–E3 保持相同模型、数据划分、训练轮数、优化器、CutMix 概率与面积采样分布；仅改变 patch 的**位置选择策略**。使用粘贴后的**实际矩形面积**计算标签混合权重。代码允许复用 CutMix 官方实现，但必须清楚标注复用与本组修改的部分。

## 2. 五人职责与交付物

### 成员 A — Baseline & Training Infrastructure（训练框架负责人）

**任务：**
1. 准备 CIFAR-100，固定 train/validation/test 划分及随机种子；实现 CIFAR 版 ResNet-18（3×3 stem、stride 1、取消初始 MaxPool）。
2. 整理并适配 CutMix 官方代码，建立**统一训练入口**：配置模型、增强策略、训练参数、seed、输出目录；支持 `none / random / source_edge / dual_edge` 四种 policy。
3. 实现和运行 E0、E1；保存 checkpoint、训练/验证曲线、配置和原始日志。
4. 负责各模块合并、训练框架兼容性和最终一键运行说明，避免不同成员各写一套训练循环。

**交付：** `train.py`、`models/`、`configs/`、E0/E1 训练结果、训练与复现命令。

### 成员 B — Source Edge-Guided Method（单侧边缘算法负责人）

**任务：**
1. 实现 Sobel 边缘图计算（灰度处理、梯度幅值）以及矩形区域平均边缘得分；使用可批量运行的 PyTorch 操作。
2. 实现 E2：沿用 CutMix 的面积/长宽采样，随机生成 `K` 个相同形状的有效候选矩形，选择 Source 边缘得分最高者。
3. 保证矩形坐标、图像粘贴和标签比例正确，检查极小 patch、靠边矩形以及 `K=1` 等情况。
4. 与 A 的统一接口对接，输出 Original CutMix 与 E2 的直观图像对比；参与 E2 训练和异常排查。

**交付：** `augmentations/edge_utils.py`、`augmentations/source_edge.py`、对应单元测试、增强图像示例和 E2 结果。

### 成员 C — Dual-Sided Method & Ablation（双侧算法负责人）

**任务：**
1. 基于 B 提供的边缘得分工具实现 E3：对同一候选区域计算 Source 和 Target 分数，并用 `source_score - γ × target_score` 排序。
2. 明确得分归一化方式及 `γ` 参数；保证 `γ=0` 时退化为 E2，并通过测试验证。
3. 对接统一训练入口，运行 E3，与 E1/E2 对照；记录算法选择的 patch 与典型成功/失败案例。
4. 负责核心消融：候选数 `K=1/4/8`；如资源允许，再测 `γ` 的不同取值或高边缘/低边缘选取策略。所有超参数只用验证集选择。

**交付：** `augmentations/dual_edge.py`、对应单元测试、E3 结果、消融结果及可视化。

### 成员 D — Evaluation & Experimental Analysis（评估与实验管理负责人）

**任务：**
1. 制定并核对 E0–E3 统一实验协议：数据划分、epoch、batch size、优化器、学习率策略、CutMix `α`、应用概率和 seed；检查各组是否公平。
2. 实现独立评价与结果汇总：Top-1 Accuracy（主指标）、Macro-F1（辅助）、train/validation loss、每批增强耗时和训练总耗时；多 seed 时统计均值与标准差。
3. 负责实验登记表，汇总 A/B/C 的配置、日志、checkpoint、失败运行及最终有效结果；检查是否存在测试集调参或 Baseline 被不公平削弱的问题。
4. 输出论文级表格与图表：主实验结果、收敛曲线、效率对比及消融。条件允许时，补充噪声/模糊/遮挡等**推理阶段**鲁棒性测试，并明确其为自建扰动实验。
5. 对改进有效、无效或负面结果都提供证据支持的分析，避免夸大结论。

**交付：** `evaluate.py`、`analysis/`、统一结果表、曲线图、消融与效率分析、实验复现清单。

### 成员 E — Report & Oral Presentation（报告与展示负责人）

**任务：**
1. 负责完整英文报告（**AAAI 双栏，正文最多 6 页，不含 References**）：Abstract、Introduction、Related Work、Method、Experiments、Discussion、Conclusion。
2. 准确介绍 Original CutMix 与 E2/E3 的技术差异；引用相关工作（至少 CutMix、SaliencyMix、Puzzle Mix 等），不声称“首次使用边缘引导图像混合”。
3. 向 A–D 收集算法描述、训练设置、实验日志、表格与图示；核对每个结果、公式、超参数和复现细节，并交由对应技术负责人审阅。
4. 制作并录制 **不超过 5 分钟**的视频：研究问题 → 原始 CutMix → 两项改进 → 核心对照实验 → 结论；采用清楚的可视化，避免展示过密的表格或夸大提升。
5. 汇总成员姓名、学号、NTU 邮箱；检查引用、Academic Integrity Pledge 及报告/视频/代码三项提交材料。最终提交由小组指定一人统一操作。

**交付：** `report/`（LaTeX 源文件、参考文献、最终 PDF）、`presentation/`（脚本、素材、最终视频）、提交检查表。

## 3. 协作约定与验收标准

- **统一增强接口：** `cutmix_batch(images, labels, policy, alpha, p, k, gamma)`；训练循环不因增强策略而分叉。A 负责接口及合并，B/C 负责各自 policy 的实现。
- **边缘算法约定：** B、C 共用 Sobel 和候选区域评分代码；候选矩形的形状和面积在每次选择中保持一致；计算 Label Mix 权重时一律使用实际粘贴面积。
- **测试：** `policy=random` 能复现普通 CutMix；`K=1` 时 Edge-Guided 不进行多候选筛选；`γ=0` 时 E3 与 E2 的评分规则一致；无越界、NaN、设备或数据类型错误。
- **可复现：** 每次实验保留 seed、完整配置、代码版本、日志和结果。Baseline 与改进方法在同等预算下比较；测试集只用于最终评价。
- **最低完成标准：** E1、E2、E3 三组公平的主实验 + 至少一项有效消融 + 统一结果表、方法可视化和可运行的代码仓库。E0、鲁棒性和更多消融用于增强论证，不以牺牲主实验完整性为代价。
- **报告责任：** E 负责最终写作，但 A–D 分别对其模块的技术正确性与实验真实性负责。

## 4. 阅读原论文与官方代码后的补充

### 4.1 原论文事实与本组设定要区分

以下依据 CutMix 原文第 3.1、4.1.2、4.1.3 节及附录 Algorithm A1；[阅读笔记](docs/literature/cutmix.md)给出定位。

- 原始 CutMix 采样 `λ ~ Beta(α, α)`，宽高分别按 `sqrt(1−λ)` 缩放；**不额外独立采样长宽比**。本组所称“沿用面积/长宽采样”应按这一点实现。
- 官方实现从图像内均匀采样中心，再将矩形裁剪到边界；最终标签权重需要用**裁剪后的实际面积**重算。
- 原论文 CIFAR 主实验使用 **PyramidNet-200**，不是本组的 CIFAR 版 ResNet-18。原文训练 300 epochs，batch size 64，初始学习率 0.25；这些是文献设置，**不是本组已经批准的超参数**，原文准确率也不是本组结果。
- 原文已有选区和标签变体消融，如中心高斯采样与固定尺寸；本组应验证轻量边缘评分的收益与代价，避免泛化成“引导选区必然更好”的结论。

### 4.2 选区策略比较中的面积分布问题

“先采样中心再裁剪”与“直接生成完全位于图像内的固定尺寸矩形”会产生不同的实际面积分布。如果 E1 用前者、E2/E3 用后者，就同时改变了位置策略和标签权重分布。

**建议方案（待 A/B/C/D 共同确认，尚未实现）：**

1. 每次先按官方 `rand_bbox` 规则得到一个裁剪后的矩形，固定它的实际高、宽。
2. 把该原始矩形作为第一个候选，其余 `K−1` 个候选只平移同样大小的矩形，并保证完全在图像内；Source 与 Target 使用相同坐标。
3. E1 直接选第一个候选；E2 按 Source 得分选；E3 按双侧得分选。这样三组保留相同的实际面积分布，`K=1` 也便于逐像素对照。
4. 每个样本的 `λ_actual = 1 − patch_area/(H×W)`。零面积矩形按无混合处理。若采用不同方案，需说明面积与边界处理，并增加匹配采样规则的随机对照。

### 4.3 官方代码不能直接当作本组完成版

官方快照固定在提交 [`2d8eb68faff7fe4962776ad51d175c3b01a25734`](https://github.com/clovaai/CutMix-PyTorch/commit/2d8eb68faff7fe4962776ad51d175c3b01a25734)，完整保留在 [third_party/CutMix-PyTorch/](third_party/CutMix-PyTorch/)。重点参考 `train.py` 的 `train` / `rand_bbox`、`test.py`、`resnet.py` 和 `utils.py`；同时保留 `pyramidnet.py`，因为原训练入口会导入它。

适配前要解决：

- `train.py` 的 CIFAR `val_loader` 直接读取 `train=False` 数据，并逐轮选最优 checkpoint。**本组应从官方训练集划分验证集**，测试集只在设置冻结后用于最终评价。
- 上游 `resnet.py` 的 CIFAR 路径是三 stage 的传统 CIFAR ResNet，不能直接把 `depth=18` 当作要求的四 stage ResNet-18。应实现 `[2,2,2,2]` BasicBlock 结构，并采用 3×3、stride 1 stem，无初始 MaxPool。
- 上游存在 `np.int`、硬编码 `.cuda()`、`DataParallel` 和旧式张量操作。需要适配当前依赖与设备，并验证实际运行；快照本身保持原样，便于溯源。
- 上游参数 `--beta` 对应 CutMix 分布参数 **α**；上游 `--alpha` 用于 PyramidNet 宽度。新配置中使用清楚的字段名，避免混淆。
- 官方循环每个 batch 使用共享的采样矩形；本组若改为逐样本选区，要对 E1–E3 采用相同粒度并记录。混合训练时仅对原标签算出的 accuracy 也不能代替干净验证集指标。

### 4.4 边缘得分与效率的实现约定

以下为本组实现建议，均需通过实验验证：

- Sobel 对几何增强后的、模型标准化前的图像计算；若输入已标准化，应先按通道还原。同一候选集在 E2/E3 中复用。
- Source 与 Target 共用**按整张图像**定义的归一化规则，不为每个候选单独缩放；常量图像应稳定输出有限得分，分母加明确的 `eps`。
- 可先比较“每张图像的边缘图除以该图全图均值加 `eps`”这一草案，再由验证集决定。它不是原论文提出或本仓库已经实现的方法。
- 明确坐标为 `(top, left, bottom, right)` 半开区间，对 NCHW 张量使用 `images[..., top:bottom, left:right]`。Source 从未被覆盖的原 batch 读取，避免原地写入污染后续样本。
- `γ=0` 的退化测试需复用相同配对、面积、候选集和 tie-breaking；仅比较不同随机运行的最终 accuracy 不能证明 E3 退化为 E2。
- 增强耗时应包含 Sobel、评分、选区与粘贴；GPU 计时需预热和同步。另记录完整训练总耗时、设备、batch size 与精度，不能只报选区函数耗时。

## 5. 仓库目录与分工

| 路径 | 内容 / 主要负责人 |
| --- | --- |
| `docs/` | 原始说明、实验协议、论文阅读笔记 |
| `models/`, `configs/`, `data/` | A：CIFAR ResNet-18、配置、数据与划分说明 |
| `augmentations/` | B/C：共享接口、Sobel、random/source_edge/dual_edge |
| `tests/` | A/B/C：面积、边界、退化和设备正确性检查 |
| `experiments/` | D：实验登记、完整配置与运行追踪 |
| `analysis/`, `results/` | D：汇总分析、表格、图表与已核实的小型结果 |
| `notebooks/`, `scripts/` | 探索与运行工具；正式训练入口由 A 统一 |
| `report/` | E：英文报告、参考文献、图示和最终 PDF |
| `presentation/` | E：视频脚本、素材和展示文件 |
| `third_party/` | 未修改的官方源码、许可证、NOTICE、文件校验值 |

每个目录已有用途说明；新增算法和训练结果由对应负责人后续提交。大数据、checkpoint、运行日志和视频不放入普通 Git 历史，使用共享存储并在登记表中记录可访问链接。

## 6. 开始主实验前的检查

- [ ] A/D 冻结 train/validation 划分、seed、训练预算、优化器、学习率、精度与 checkpoint 选择规则。
- [ ] A/B/C/D 冻结矩形采样、batch/逐样本粒度、Sobel 输入范围、归一化与混合标签的实现。
- [ ] 完成 `K=1`、`γ=0`、常量图像、零面积与边界测试，再启动主实验。
- [ ] 在相同环境和预算下运行 E1/E2/E3，登记失败运行；资源允许时增加 E0 与多 seed。
- [ ] 只用验证集选择 K、γ 和 checkpoint；最终才评估测试集。
- [ ] 结果表填入真实日志、commit 与配置路径；没有完成的实验保持空白或 `planned`。

补充文献入口：[SaliencyMix 原文](https://arxiv.org/abs/2006.01791)、[Puzzle Mix 会议原文](https://proceedings.mlr.press/v119/kim20b.html)。它们已列入后续阅读清单；当前详细阅读笔记针对基础论文 CutMix。报告格式、页数和视频时限沿用原始项目说明，最终提交前仍需由 E 对照课程官方要求核实。

## 7. 来源与许可

官方源码的 MIT 许可证与第三方 BSD/MIT 声明见 [LICENSE](third_party/CutMix-PyTorch/LICENSE) 和 [NOTICE](third_party/CutMix-PyTorch/NOTICE)，须随复用代码保留。代码快照的来源与 SHA-256 校验值见 [cutmix_provenance.json](third_party/cutmix_provenance.json)。本组新增材料的整体开源许可证尚待成员共同决定；公开仓库不自动改变第三方材料的许可。
