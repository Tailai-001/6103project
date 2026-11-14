# AI6103 Final Project — Lightweight Edge-Guided CutMix

**Track 2: Improving an Existing Paper**  
**基础论文：** *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features*（ICCV 2019）  
**官方代码：** https://github.com/clovaai/CutMix-PyTorch

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
