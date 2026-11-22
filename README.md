# AI6103 Final Project — Seam-Aware CutMix

**Seam-Aware CutMix: Lightweight Content and Boundary Optimization for Image Classification**  
**中文：** 接缝感知的边缘引导图像混合增强（SAEG-CutMix）  
**Track 2：** Improving an Existing Paper  
**基础论文：** [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/abs/1905.04899)（ICCV 2019） · [原文 PDF（含附录）](https://arxiv.org/pdf/1905.04899)  
**官方代码：** [clovaai/CutMix-PyTorch](https://github.com/clovaai/CutMix-PyTorch)

## 0. 从这里开始

本仓库是 AI6103 五人小组项目。当前集成分支已包含 A/B/C 的方法实现、18 个完整 200-epoch 运行，以及 D 的评价与结果分析。实际交付和证据限制见 [D 的结果说明](RESULTS_D.md) 与 [补充核查与同机确认实验](RESULTS_AUDIT.md)。各成员 PR 尚需合并至 main；下文的方法设计与分工不应当作训练结果。

补充同机 seed-43 三臂实验也已完成，各 200 轮：E1 **79.92%**、E4 **79.78%**、E4-R **80.08%**（最佳验证 Top-1，均经独立 checkpoint 复评）。目前支持平均 RGB 接缝质量改善，**不支持稳定的分类提升**；详细结果、末 10 轮均值和限制见 [RESULTS_AUDIT.md](RESULTS_AUDIT.md)。

本版将原先的纯边缘选区方案扩展为“内容评分 + 接缝评分 + 局部边界优化”。**实验编号已更新：原双侧边缘方法归入新 E2，新 E3 表示接缝感知矩形选区，E4 表示动态规划边界优化。** 单侧边缘方法保留为 E2-S 消融，避免旧 E3 与新 E3 混用。

- [实验协议与待定事项](docs/experiment_protocol.md) · [增强模块与接口草案](augmentations/README.md)
- [原始项目说明（原样归档，反映旧方案）](docs/project_brief.md)
- [CutMix 原文阅读笔记](docs/literature/cutmix.md) · [代码来源、版本和适配清单](third_party/README.md)
- [协作方式](CONTRIBUTING.md) · [实验登记表](experiments/registry.csv) · [结果表模板](results/tables/main_results.csv)

```bash
git clone https://github.com/Tailai-001/6103project.git
cd 6103project
```

项目级 `train.py` 复用 A 的统一训练循环，支持 `none/random/source_edge/dual_edge/seam_aware/seam_dp/seam_random`。安装与训练说明见 [README_B.md](README_B.md)、[RESULTS_C.md](RESULTS_C.md)；D 的统一评价入口为 [evaluate.py](evaluate.py)。`third_party/CutMix-PyTorch/train.py` 为保持原样的历史参考源码。

## 1. 原论文、研究问题与预期方案

### 1.1 原始 CutMix 与本组研究问题

CutMix 从 Source 图像粘贴一个随机矩形到 Target 图像，并按实际替换面积混合标签。本组沿用 **CIFAR-100 + 四 stage CIFAR 适配版 ResNet-18**，研究：在考虑区域内容的同时，选择颜色跳变较小的接缝、或局部调整区域边界，是否能改善分类表现？收益是否值得额外计算成本？

思路借鉴图像拼接中的边界兼容性，以及 Seam Carving 的低代价路径搜索。这里把路径优化用于混合区域的硬边界，**不改变图像分辨率，不引入额外神经网络，也不采用羽化、模糊或 Poisson Blending**。边缘强度只是低层结构线索，不等同于语义重要性。

以下定义 `x_A` 为 Target、`x_B` 为 Source，`M=1` 表示使用 Source 像素：

$$
\tilde{x}=(1-M)\odot x_A+M\odot x_B,\qquad
\lambda_{\mathrm{actual}}=1-\frac{\sum_{u,v}M(u,v)}{HW}.
$$

矩形、弯曲区域和随机边界都按最终二值 Mask 的真实面积计算标签与混合损失：

$$
\mathcal L=\lambda_{\mathrm{actual}}\operatorname{CE}(f(\tilde{x}),y_A)
+(1-\lambda_{\mathrm{actual}})\operatorname{CE}(f(\tilde{x}),y_B).
$$

### 1.2 实验编号与方法

| ID | 方法 / policy | 验证的问题 |
| --- | --- | --- |
| E0 | No CutMix / `none` | 无混合增强的普通训练基线 |
| E1 | Original CutMix / `random` | 原始随机矩形基线 |
| E2 | Dual-Sided Edge-Guided / `dual_edge` | Source 信息与 Target 信息保留是否有帮助 |
| E3 | Seam-Aware Patch Selection / `seam_aware` | 在 E2 上增加接缝代价是否有帮助，保持矩形形状 |
| E4 | Seam-Optimized CutMix / `seam_dp` | 在 E3 选定矩形附近，动态规划调整边界是否进一步有帮助 |
| E2-S（消融） | Source Edge-Guided / `source_edge` | 只看 Source 与双侧评分的区别 |
| E4-R（形状对照） | Randomly Perturbed Boundary / `seam_random` | 对同一 E3 矩形随机扰动边界，与优化边界比较 |

**E3 是最低可交付的接缝创新，E4 是核心扩展目标。** 主线至少完成 E0–E3 的公平对照、有效消融和成本分析；要对 E4 的“边界优化有效性”作出结论，必须完成 E4-R 等形状/面积控制。未完成的扩展应明确记录，不能写成已实现贡献。

### 1.3 内容评分与接缝感知矩形选择

B 实现共享 Sobel 边缘图与候选矩形。对于同一候选矩形 `R`，E2 的得分为：

$$
S_{\mathrm{edge}}(R)=\hat E_B(R)-\gamma\hat E_A(R).
$$

其中 `E_A/E_B` 是区域平均边缘强度，帽号表示事先约定的图像级尺度归一化。`γ=0` 退化为单侧 Source 方法。E1–E3 共用面积采样、候选集规则与 Source/Target 同坐标粘贴；**不独立搜索两张图像上的不同粘贴位置**。

C 在 E2 上加入接缝代价：

$$
S_{\mathrm{seam}}(R)=\hat E_B(R)-\gamma\hat E_A(R)-\beta\hat D_{\mathrm{seam}}(R),\qquad
R^*=\arg\max_{R\in\mathcal R}S_{\mathrm{seam}}(R).
$$

令 `B(M)` 是图像内所有跨越 Mask 边界的四邻接像素对 `(p,q)`，其中 `M(p)=1`、`M(q)=0`。接缝代价为：

$$
D_{\mathrm{seam}}(M)=\frac{1}{3|B(M)|}
\sum_{(p,q)\in B(M)}\sum_{c\in\{R,G,B\}}|x_{B,c}(p)-x_{A,c}(q)|.
$$

矩形评分使用对应矩形 Mask。只统计图像内真实的跨接缝邻接，不把图像外填充值当作 Target；无内部接缝的 Mask 单独计数，边界指标记为不适用。RGB 差异在 `[0,1]`、模型标准化前计算；各评分项的尺度规则、`eps` 与参数必须记录，不能逐候选独立归一化。

### 1.4 局部动态规划边界优化（E4）

从 E3 选定的矩形 `R*` 出发，上下边界暂时固定，左右边界在小范围内逐行移动。例如半径 `r=2` 是待验证的起点，**不是已经冻结的参数**；相邻行位移限制为至多 1 像素。

单条边界路径可采用以下递推框架：

$$
DP(y,x)=c(y,x)+\min_{d\in\{-1,0,1\}}\left[DP(y-1,x+d)+\mu|d|\right].
$$

`c(y,x)` 由 Source 内侧与 Target 外侧的兼容性定义，`μ` 惩罚弯曲。具体代价与求解方式由 C 实现并记录；这不是直接执行原论文的图像缩放操作。必须处理：

- **左右边界耦合约束：** 每一行满足图像边界与最小宽度要求；若分别求解，必须约束后求的路径，或采用联合状态，不能任由两条路径相交。
- **最终 Mask 面积：** 预先约定相对基础矩形的面积偏移预算；面积不可行或区域过小时回退到原矩形，并记录回退率。标签始终按最终 Mask 重算。
- **完整接缝：** 上式若只优化局部代理代价，不应宣称获得完整边界代价的全局最优。生成 Mask 后，重新计算含上下边界、弯曲处水平连接在内的真实 `D_seam`。
- **匹配对照：** E4-R 从同一 E3 矩形出发，采用相同位移、连续性、最小宽度、面积与回退规则；报告实际面积和边界长度分布，排查不规则形状本身的作用。

E4 可能改变面积、周长及所含内容，因此只能在控制这些因素后解释其收益，不能继续笼统声称所有方法“只改变矩形位置”。降低颜色跳变也不保证提高准确率，人工接缝可能具有正则化作用。

## 2. 五人职责与交付物

### 成员 A — Baseline & Training Infrastructure（训练框架负责人）

1. 准备 CIFAR-100：从官方训练集固定 train/validation 划分与随机种子，官方测试集只用于最终评价；实现四 stage `[2,2,2,2]` CIFAR ResNet-18，3×3 stride 1 stem，无初始 MaxPool。
2. 建立唯一训练入口，接入第 1.2 节全部 policy；统一 Dataset、优化器、学习率、checkpoint 与日志。**先交付可跑通一个 batch 的骨架和返回值约定**，方便 B/C/D 同步接入。
3. 实现 E0/E1，保存配置、代码版本、环境、日志与 checkpoint；与 D 共同冻结预算和 checkpoint 选择规则。
4. 集成 B/C 模块，确保训练循环统一使用最终 Mask 面积权重；负责设备兼容性、训练调度与最终运行说明。

**A 已交付（见原始 A 目录）：** `train.py`、`models/`、`configs/`、固定划分说明、E0/E1 结果、环境与复现命令。

### 成员 B — Edge-Guided Selection（单／双侧边缘方法负责人）

1. 实现灰度转换、Sobel 梯度幅值、图像级归一化和矩形平均分；采用可批量运行的 PyTorch 操作。
2. 实现共享矩形候选生成，以及 `source_edge` 和 `dual_edge`。E2 以双侧方法为主，E2-S 保留单侧消融；共享坐标、候选与同分规则。
3. 检查实际面积、边界、零/极小区域、K=1、γ=0、常量图像、device/dtype；为 C 提供稳定的候选集与边缘分数接口。
4. 对接 A 的入口，完成 E2/E2-S 与 K=1/4/8 等内容选择消融，输出同一对图像的选框和混合示例。

**B 已完成实现、检查与单 seed 实验（见 README_B.md）：** `augmentations/edge_utils.py`、`source_edge.py`、`dual_edge.py`、共享候选工具、正确性测试、E2 及消融结果。

### 成员 C — Seam Selection & Boundary Optimization（接缝方法负责人）

1. 实现跨 Mask 边界的 RGB 代价、有效边界像素计数与归一化，接入 E3 的 `edge − γ×target − β×seam` 候选排序。
2. 实现受限左右边界动态规划与二值 Mask 生成，明确 `c`、位移半径、μ、最小宽度、面积预算、路径约束与回退行为；验证 r=0 退化到 E3 的原矩形。
3. 实现同一 E3 矩形上的随机边界对照 E4-R；与 B 共用配对和基础候选，不另写训练循环。
4. 先用两张模拟的 32×32 图像与一个矩形独立测试，再交给 A 集成；完成 E3、争取完成 E4/E4-R，记录 β、r、μ 等消融与成功/失败案例。

**交付（待实现）：** `augmentations/seam_utils.py`、`seam_aware.py`、`seam_dp.py`、`seam_random.py`、Mask/路径测试、E3/E4 及对应对照结果。

### 成员 D — Evaluation & Experimental Analysis（评估与实验管理负责人）

1. 与 A 冻结统一协议，核对模型、划分、seed、训练预算、α、p、采样粒度；为各方法安排可比较的验证集参数搜索预算。
2. 实现 Top-1 Accuracy（主指标）、Macro-F1、loss、增强耗时和训练总耗时；增加实际混合图像的边界不连续度、Mask 面积/周长分布、非法路径与回退率。
3. 管理实验登记、完整配置、日志、checkpoint 与失败记录；独立核对 E4/E4-R 的形状/面积约束，分析边界指标与分类效果的关系。
4. 生成主结果、收敛曲线、效率、形状及参数消融图表；多 seed 保留原值并统计均值/标准差。资源允许时补充自建噪声/模糊/遮挡推理测试。
5. 对有效、无效及负面结果都保留证据；不把更低的接缝代价直接解释为更高语义质量或更高分类准确率。

**交付（待实现）：** `evaluate.py`、`analysis/`、结果表与图、形状/边界诊断、实验复现清单。

### 成员 E — Report & Oral Presentation（报告与展示负责人）

1. 从项目开始阅读文献并搭建英文报告，说明从 Image Stitching / Seam Carving 向 CutMix 边界选择迁移的动机、定义与局限。
2. 重点核对 CutMix、SaliencyMix、Puzzle Mix、SmoothMix 与 Seam Carving。**把该方案称为候选研究设计，不宣称绝对首创、已有性能提升或已完成全面查新。**
3. 向 A–D 收集接口、公式、配置、日志、图表及失败案例，并请各负责人审核对应技术描述；明确区分 E3 矩形选区与 E4 形状优化。
4. 沿用原始项目说明的英文 AAAI 双栏、正文最多 6 页（不含 References）与不超过 5 分钟视频要求，最终以课程官方文件为准。展示研究问题、两阶段设计、公平对照、成本与结论。
5. 汇总提交材料、引用与 Academic Integrity Pledge；成员个人信息按课程要求另行汇总，不默认发布到公开仓库。

**交付（待完成）：** `report/`、参考文献、报告 PDF、`presentation/` 的脚本与展示视频、提交检查表。

## 3. 并行协作、统一接口与验收

### 3.1 可以同步开始

| 成员 | 无需等待完整训练即可开展的工作 | 集成前需要对齐 |
| --- | --- | --- |
| A | 模型、划分、E0/E1 最小训练骨架 | 与 B/C 定好增强输入输出；与 D 冻结日志与预算 |
| B | 模拟张量上的 Sobel、候选与单／双侧评分 | 向 C 提供候选形状、坐标和分数约定 |
| C | 两张图像与固定矩形上的接缝代价、DP、随机边界 | 与 B 共用候选；向 A 返回合法 Mask 与面积 |
| D | 协议、指标、空结果表、模拟 Mask 的诊断与计时工具 | 读取 A 的日志 schema，与 C 核对完整边界定义 |
| E | 相关工作、报告结构、方法示意与展示脚本 | 用 A–D 的实际实现和日志更新方法与结论 |

先冻结最小接口和几何约定，再并行开发；通过集成检查后启动统一主实验。C 不必等 B 的完整训练结果，D/E 也不必等主实验结束。这里不规定日历或时间表。

### 3.2 统一增强接口（A/B 已接入，C 待接入）

```python
cutmix_batch(images, labels, policy, alpha, p, k, gamma,
             beta=0.0, seam_radius=0, seam_mu=0.0)
```

- policy 为第 1.2 节各项；`alpha` 是 CutMix Beta 分布参数，`beta` 是**接缝惩罚系数**，不要与官方脚本 `--beta` 的旧命名混用。
- 建议返回 `mixed_images, labels_a, labels_b, lam_actual, metadata`。`lam_actual` 为长度 N 的逐样本权重，metadata 提供最终 Mask、选定基础矩形与必要诊断；具体 schema 由 A/B/C/D 冻结。
- 所有方法共用一个训练循环。Source 从未被修改的原 batch 读取；坐标采用 `(top, left, bottom, right)` 半开区间，NCHW 切片为 `images[..., top:bottom, left:right]`。
- 只在训练阶段混合；验证/测试使用干净输入。同一 Mask 定义同时服务图像替换、标签权重和边界诊断。

### 3.3 正确性与公平性

- **退化：** 同一配对、候选与归一化下，`dual_edge` 的 γ=0 等于 `source_edge`；`seam_aware` 的 β=0 等于 `dual_edge`；`seam_dp` 的 r=0 等于 `seam_aware` 的矩形 Mask。
- **K=1：** 只表示没有多候选筛选；E4 仍可能改变形状，不能把它等同于普通 CutMix。
- **边界：** 常量图像、图像边缘、空/满 Mask、极小区域、无可行路径、左右不交叉、连续性、面积预算均需检查；输出无 NaN、越界及设备/dtype 错误。
- **公平：** E1–E3 匹配实际矩形面积；E4/E4-R 另控制形状、面积与回退规则。共享模型、划分、训练预算及随机种子，只用验证集选择参数和 checkpoint。
- **可追溯：** 每次运行保留配置、seed、commit、依赖、设备、日志与结果；参数搜索与失败运行也登记。单 seed 不伪造标准差，不用原论文数值填本组表格。
- **交付：** E0–E3 + 至少一项有效消融 + 方法可视化 + 准确率/接缝/效率分析 + 可运行仓库；E4/E4-R 为核心扩展，完成程度如实说明。

## 4. 阅读原论文与官方代码后的适配注意事项

### 4.1 原论文事实与本组设定要区分

CutMix 原文第 3.1 节与附录 Algorithm A1 采用 `λ ~ Beta(α,α)`，宽高共同按 `sqrt(1−λ)` 缩放，不另外独立采样长宽比；均匀采样中心后裁剪到图像边界，并按实际面积重算标签。详见[原文阅读笔记](docs/literature/cutmix.md)。

原文 CIFAR 主实验为 PyramidNet-200、300 epochs、batch 64、初始学习率 0.25。这些是文献设置，不是本组 ResNet-18 已批准的配置；原文的选区/标签消融也不能替代本组实验。

### 4.2 面积匹配与形状变化

E1–E3 建议先按官方 `rand_bbox` 得到裁剪后的矩形，固定实际高宽，将原矩形作为候选 0，再平移生成其余 K−1 个有效候选；E1 选候选 0，E2/E3 按各自得分选。这样在相同基础随机样本下保留相同的实际矩形面积。

E4/E4-R 从同一 E3 矩形出发，施加预先冻结的面积偏移预算，记录最终面积、周长与回退率，并分析是否仍存在面积分布偏差。**只重算 λ 能保证标签正确，不能自动保证面积分布公平。** 具体约束和匹配评价见[实验协议](docs/experiment_protocol.md)。

### 4.3 官方代码不能直接当作本组完成版

官方快照固定在提交 [`2d8eb68faff7fe4962776ad51d175c3b01a25734`](https://github.com/clovaai/CutMix-PyTorch/commit/2d8eb68faff7fe4962776ad51d175c3b01a25734)，完整保留在 [third_party/CutMix-PyTorch/](third_party/CutMix-PyTorch/)。参考 `train.py` 的 `train/rand_bbox`、`test.py`、`resnet.py` 和 `utils.py`；`pyramidnet.py` 是原训练入口的导入依赖。

- 上游 CIFAR `val_loader` 直接读取 `train=False` 并逐轮选择 checkpoint。本组必须从官方训练集另划验证集，测试集只用于最终评价。
- 上游 CIFAR ResNet 是三 stage 结构，不能设置 `depth=18` 就当作四 stage `[2,2,2,2]` ResNet-18。
- 上游含 `np.int`、硬编码 `.cuda()`、`DataParallel` 及旧式张量操作；需要验证适配当前依赖和设备。快照本身保持原样。
- 上游 `--beta` 对应 CutMix 分布 α，`--alpha` 用于 PyramidNet 宽度；本组接缝 β 是新增参数，应分别命名。
- 上游按 batch 共享矩形。本组若逐样本评分，需对可比较方法统一采样粒度。混合训练时对原标签计算的 accuracy 不能替代干净验证指标。

### 4.4 Sobel、接缝与效率

Sobel 与 RGB 接缝代价使用几何增强后、模型标准化前的输入；若已标准化需先还原。B/C 共用图像级边缘归一化，常量图像与分母 `eps` 行为一致；接缝项尺度另行明确，不能把所有项未经校准直接相减。

增强计时包含 Sobel、候选生成、评分、选区、DP/随机扰动、Mask 构造与粘贴；GPU 计时需预热和同步。另报完整训练耗时、设备、batch size 和精度。复杂度及低算力可行性需要实测，不预先声称“没有额外开销”。

## 5. 仓库目录与分工

| 路径 | 内容 / 主要负责人 |
| --- | --- |
| `docs/` | 原始说明、当前实验协议、CutMix 阅读记录 |
| `models/`, `configs/`, `data/` | A：模型、配置、数据划分；D 核对协议 |
| `augmentations/` | B：单／双侧边缘与候选；C：接缝评分、DP 与随机边界 |
| `tests/` | A/B/C：增强正确性；D：面积和边界指标核对 |
| `experiments/` | D：配置与运行登记；A 协调统一训练任务 |
| `analysis/`, `results/` | D：分类、边界、面积/形状、效率与消融分析 |
| `notebooks/`, `scripts/` | 探索与工具；正式训练入口由 A 统一 |
| `report/`, `presentation/` | E：文献、报告、脚本、素材与视频 |
| `third_party/` | 未修改的官方源码、LICENSE、NOTICE 与文件校验值 |

目录中的交付文件名是开发计划。数据、checkpoint、大日志和视频使用共享存储并登记链接，不放入普通 Git 历史。

## 6. 主实验前检查

- [ ] A/D 冻结划分、seed、训练预算、优化器、学习率、精度与 checkpoint 规则。
- [ ] A/B/C/D 冻结候选几何、采样粒度、归一化、返回值与 Mask 约定。
- [ ] C/D 冻结 E4/E4-R 的路径代价、位移、最小宽度、面积预算、回退与边界度量。
- [ ] 完成 γ=0、β=0、r=0、K=1、常量图像、空/满 Mask 与非法路径检查。
- [ ] 通过最小训练集成检查；在同等预算下运行 E0–E3，再推进 E4/E4-R。
- [ ] 参数只用验证集选择；结果表、图示、代码版本与原始日志可互相核对。

## 7. 相关工作与创新声明

| 原文入口 | 与本组的关系 |
| --- | --- |
| [CutMix](https://arxiv.org/abs/1905.04899) | 基础的区域替换与面积标签混合 |
| [SaliencyMix](https://arxiv.org/abs/2006.01791) | 信息引导选区的重要相关工作，需精读比较 |
| [Puzzle Mix（ICML 2020）](https://proceedings.mlr.press/v119/kim20b.html) | 显著性、混合 Mask 与最优传输优化；不可声称首次优化空间混合形状 |
| [SmoothMix（CVPR Workshops 2020）](https://openaccess.thecvf.com/content_CVPRW_2020/html/w45/Lee_SmoothMix_A_Simple_Yet_Effective_Data_Augmentation_to_Train_Robust_CVPRW_2020_paper.html) | 通过软边缘混合处理强边界；本组候选方案保持二值硬边界并改变选区/路径 |
| [Seam Carving（Avidan & Shamir, 2007）原文](https://www.cs.princeton.edu/courses/archive/fall07/cos597B/papers/avidan-seam-carving.pdf) | 内容感知缩放中的低代价路径思想；本组迁移其局部路径优化思路，不执行缩放 |

本组探索的具体设计是：**轻量内容与接缝联合选区，以及局部受限的硬边界动态规划。** 它是待实现、待验证的研究方案；当前文档不保证未有相同已发表方法，也不预设准确率提升。E 应继续核对相关工作与引用，所有贡献声明以实际实现和实验为依据。

## 8. 来源与许可

官方源码的 [MIT LICENSE](third_party/CutMix-PyTorch/LICENSE) 与 [NOTICE](third_party/CutMix-PyTorch/NOTICE) 原样保留，复用时须保留适用声明；来源与 SHA-256 见 [cutmix_provenance.json](third_party/cutmix_provenance.json)。原始项目说明继续保存在 `docs/project_brief.md`，不覆盖历史方案。本组新增材料的整体开源许可证尚待成员共同决定。
