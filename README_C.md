# 阶段 C：Seam Selection & Boundary Optimization

C 在 A 的训练框架与 B 的边缘选区（PR #1）之上实现 E3（`seam_aware`）、E4（`seam_dp`）和 E4-R（`seam_random`）。三者共用 B 的候选矩形、Sobel 分数、同分规则与 A 的统一训练入口，不另写训练循环。

## 0. 状态（请先读）

| 项目 | 状态 |
| --- | --- |
| E3 / E4 / E4-R 代码与 `cutmix_batch` 接入 | **已实现**，103 项测试通过（另 1 项 CUDA 测试在无 GPU 环境跳过） |
| 退化检查 β=0→E2、r=0→E3 | **已验证**（逐位一致） |
| 三份配置在真实 CIFAR 上跑通 `train.py` | **已验证**：CPU、`--epochs 1 --max-batches 2` 冒烟，仅证明流程可运行 |
| 真实 CIFAR 训练集上的增强行为诊断 | **已测**：见第 5 节，只描述增强本身（接缝、形状、回退），**不是分类结果** |
| 200-epoch 训练与准确率 | **未做**：需要 A 在 GPU 上运行，本文档没有任何准确率结论 |
| β、r、μ、最小宽度、面积预算 | **暂定**：需与 D 在验证集上选定；E4 参数改动后须重新校准 `seam_random_mu` |

开发环境：macOS CPU、Python 3.11、torch 2.9.1（CPU 版）、torchvision 0.24.1、numpy 1.26.4、PyYAML 6.0.2、pytest 8.4.2，与 A 锁定版本一致（A 使用 CUDA 版）。

## 1. 方法与实现

记号沿用根 README：`x_A` 为 Target、`x_B` 为 Source、`M=1` 表示用 Source 像素；标签始终按最终 Mask 的实际面积计算。

**接缝代价（`seam_utils.py`）。** `D_seam` 为所有跨 Mask 边界的图内四邻接像素对 `(p,q)`（`M(p)=1, M(q)=0`）上 `|x_B(p) − x_A(q)|` 的三通道平均，RGB 在 `[0,1]`、标准化之前计算。图外不计；空/满 Mask 计数为 0，诊断中记为不适用（代价置 0 而非 NaN）。
**尺度：** `D_hat = D_seam / max(s, eps)`，`s` 为两张图各自全部相邻像素平均颜色差的均值（图像级，不逐候选），`eps = 1e-3`（RGB 单位）。`D_hat = 1` 表示接缝与图像自身纹理一样"跳"。

**E3（`seam_aware.py`）。** 在 B 的同一批 K 个等面积候选上打分 `S = Ê_B − γ·Ê_A − β·D_hat`，取 argmax（同分取首个，与 E2 相同）。接缝项不消耗随机数，所以 β=0 时输出与 `dual_edge` 逐位一致，任何 β 下候选与面积分布都与 E2 相同。

**E4（`seam_dp.py`）。** 先完整运行 E3 得到矩形（同一随机数、候选与选择），再：

1. **局部代价**：右边界在第 y 行放在 `X = right + d` 时，`c = |x_B(y, X−1) − x_A(y, X)|`；左边界 `L = left + d` 时 `c = |x_B(y, L) − x_A(y, L−1)|`，用与 E3 相同的尺度归一化。这正是完整 `D_seam` 在每行左右两侧的像素对。
2. **左右边联合 DP**：状态为 `(左偏移 a, 右偏移 b)`，`d ∈ [−r, r]`，相邻行每条边至多移动 1 像素，弯曲惩罚 `μ·(|Δa|+|Δb|)`，每行 `(right+b) − (left+a) ≥ min_width`（≥1，故左右边不交叉）。联合求解没有先后顺序偏差，对上述局部目标是精确最优；同分优先保持原矩形边。
3. **面积预算与回退**（`finalize_boundaries`，E4 与 E4-R 共用）：`|A_final − A_base| ≤ area_budget × A_base`，否则回退到 E3 矩形。状态码 `dp_status`：0 accepted、1 no_patch、2 too_small（原宽度 < min_width）、3 infeasible、4 area。
4. **完整接缝重算**：对最终 Mask（含上下边与拐弯处的水平连接段）重新计算 `D_seam`。DP 只优化左右两侧，不宣称完整接缝全局最优。

`seam_radius = 0` 时 E4 与 E3 逐位一致。

**E4-R（`seam_random.py`）。** 同一 E3 矩形、同一约束集合、同一回退规则，联合 DP 改在 i.i.d. Uniform[0,1) 随机代价上求解，因此路径与图像内容无关，而 too_small/infeasible 判定与 E4 逐样本一致。随机数由克隆的设备 RNG 状态取值后异或固定常数重新播种：可复现、不推进全局 RNG，且与 B 候选平移使用的随机流不同。
**弯曲惩罚单独校准：** 随机代价逐行不相关，同样的 μ 下 E4-R 的拐弯数约为 E4 的 2 倍（第 5 节）。因此 E4-R 用自己的 `seam_random_mu`，由 `scripts/calibrate_seam_random_mu.py` 只按形状统计（平均拐弯数）对齐 E4，不涉及模型、标签或验证集。

## 2. 接口与 metadata

```python
cutmix_batch(images, labels, policy, alpha, p, k, gamma, beta=0.0,
             seam_radius=0, seam_mu=0.0, seam_min_width=2,
             seam_area_budget=0.1, seam_random_mu=None)
```

返回值与 A/B 一致：`mixed_images, labels_a, labels_b, lam_actual[N], metadata`。`seam_min_width`、`seam_area_budget`、`seam_random_mu` 为本阶段新增，`train.py` 从配置的 `augmentation` 段读取。⚠️ `seam_radius` 默认 0，E4/E4-R 配置必须显式设置，否则退化为 E3。

E3 metadata 在 E2 字段之外增加 `edge_scores`、`seam_costs`/`seam_costs_raw`/`seam_counts`（每个候选）和 `seam_cost`/`seam_cost_raw`/`seam_count`（选中框）。E4/E4-R 另含 `dp_status`、`dp_changed`、`left_offsets`/`right_offsets`、`area_base`/`area_final`、`bends`（拐弯总格数）、`seam_cost*`（最终 Mask）与 `seam_cost*_base`（E3 矩形），以及 `seam_radius`、`seam_mu`、`bend_mu`（实际使用的弯曲惩罚）、`min_width`、`area_budget`。

## 3. 配置与运行

| 配置 | policy | 增强参数（暂定） |
| --- | --- | --- |
| `configs/c_seam_aware_k4_s42.yaml` | E3 | K=4、γ=0.5、β=1.0 |
| `configs/c_seam_dp_k4_s42.yaml` | E4 | 同上 + r=2、μ=0.05、min_width=2、area_budget=0.2 |
| `configs/c_seam_random_k4_s42.yaml` | E4-R | 同 E4 + `seam_random_mu=0.225` |

`augmentation` 以外的设置与 B 的 `b_dual_edge_k4_s42.yaml` 完全相同（seed 42、200 epoch、SGD 0.1、cosine、batch 128），E2/E3/E4/E4-R 共享同一训练设定。

```bash
python -m pytest tests -v
python train.py --config configs/c_seam_aware_k4_s42.yaml --run-name unique_e3_run
python train.py --config configs/c_seam_dp_k4_s42.yaml --run-name unique_e4_run
python train.py --config configs/c_seam_random_k4_s42.yaml --run-name unique_e4r_run
# E4 参数改动后重新校准 E4-R（只读训练集，不训练模型）
python scripts/calibrate_seam_random_mu.py --config configs/c_seam_dp_k4_s42.yaml --batches 40 --out results/member_c/seam_random_mu_calibration.json
# 增强行为诊断与示例图
python scripts/member_c_diagnostics.py --batches 20
python scripts/member_c_cases_figure.py results/member_c/cases_cifar.png
```

B 原有的 `python -m unittest discover -s tests` 只运行 B 的 unittest 风格测试；C 的测试为 pytest 风格，完整测试请用 `python -m pytest tests`。

### 3.1 正式训练（GPU）

在有 CUDA 的机器上，从仓库根目录一键运行（参照 B 的 `run_member_b.sh`，全部调用统一 `train.py`）：

```bash
bash scripts/run_member_c.sh                       # 测试 → 数据 → GPU 增强计时 → smoke → 5 epoch pilot → E3/E4/E4-R 各 200 epoch → 汇总
C_ARMS="e4 e4r" bash scripts/run_member_c.sh       # 只跑部分 arm
C_ARMS=e3 sbatch scripts/train_member_c.slurm      # SLURM：每个 arm 一个作业，singleton 保证依次运行
```

- 运行按顺序执行，每个作业开始前写入 `experiments/registry.csv`；已存在的 run 不会被覆盖，重跑请设 `C_RUN_SUFFIX`。正式运行要求训练/增强代码已提交（registry 记录 commit）。
- 输出：`logs/c_{e3,e4,e4r}_k4_s42.csv`、`checkpoints/<run>/best.pth`、`results/member_c/aug_timing.json`（GPU 同步增强耗时）、`results/member_c/summary.{json,csv}`（只含验证集）。
- `train.py` 不支持断点续训，SLURM `--time` 必须覆盖整次运行。

**耗时估算（未实测，pilot 会打印实际值）**：B 在 A800 上 E2 K=4 每 epoch 约 12.8 s（200 epoch ≈ 43 min），其中增强约 14 ms/batch，主要是 kernel 启动开销。按本地 CPU 上增强调用的 aten 算子数（E2 ≈3.6k、E3 ≈4.0k、E4 ≈10.5k、E4-R ≈10.1k / batch）粗略外推，E3 ≈ 45 min、E4 和 E4-R 各 ≈ 1.2 h，三组合计约 3–3.5 h（A800 级 GPU、单进程）。较慢的 GPU 可能需要 1.5–3 倍时间。

## 4. 测试

`tests/test_seam_utils.py`、`test_seam_aware.py`、`test_seam_dp.py`、`test_seam_dp_policy.py`、`test_seam_random.py`，覆盖：

- 接缝代价与逐像素对慢速参考实现一致；周长计数、贴边不计、空/满 Mask、半精度。
- E3：β=0 与 E2 逐位一致；任意 β 候选与面积相同；不消耗随机数；选择等于独立重算分数的 argmax；构造样例中优先选低接缝。
- DP：单边与左右联合 DP 均与暴力枚举一致；宽度宽松时联合 DP 等于两次单边 DP；不交叉与最小宽度；填充行不影响结果；μ=0 时 DP 目标等于最终 Mask 全部横向接缝对之和。
- 面积与回退：被接受的 Mask 满足预算、偏移、连续性且上下边不动；各回退状态的 Mask 等于原矩形。
- E4/E4-R 策略：r=0 与 E3 逐位一致；与 E3 共享矩形和随机数；E4-R 可复现、不受图像内容影响、可行性与 E4 一致；`seam_random_mu` 生效、被记录且不影响 E4；校准脚本冒烟。
- 用 `train.py` 的 `train_one_epoch` 和 CIFAR ResNet-18 在假 batch 上各跑一步 E4、E4-R。

## 5. 真实 CIFAR 训练集上的增强诊断（不是分类结果）

来源：`results/member_c/diagnostics_stdout.txt`（2,560 张训练图，p=1）与 `seam_random_mu_calibration.json`（5,101 张）。只用官方训练集中 A 划分的 45k 训练部分，包含训练时的 crop/flip；未训练模型、未使用验证集或测试集。

**E3 相对 E2（K=4、γ=0.5）**

| β | 选择与 E2 不同 | 原始 RGB 接缝差 |
| --- | --- | --- |
| 0（=E2） | 0% | 0.310 |
| 0.5 | 55% | 0.249 |
| 1.0 | 62% | 0.244 |
| 2.0 | 67% | 0.242 |

**E4（β=1、r=2、μ=0.05、min_width=2）**

| 面积预算 | 接受 | 面积回退 | 完整 D_hat（E3→E4，改形样本） | 新贴图像边缘 |
| --- | --- | --- | --- | --- |
| 0.05 | 47% | 53% | 4.02 → 3.44 | 11% |
| 0.10 | 73% | 27% | 4.04 → 3.44 | 15% |
| 0.20 | 92% | 8% | 4.06 → 3.45 | 16% |

**E4 与 E4-R 形状（预算 0.2，校准集 5,101 张）**

| | 平均拐弯 | 边界长度 | 面积变化 | 接受 |
| --- | --- | --- | --- | --- |
| E4（μ=0.05） | 9.09 | 61.3 | 5.3% | 91.7% |
| E4-R，μ 相同（0.05，诊断集） | 16.75 | 74.7 | 5.3% | 93.5% |
| E4-R，`seam_random_mu=0.225` | 9.10 | 66.7 | 5.7% | 91.6% |

**RandomCrop 黑边**（排除"黑对黑"像素对后的原始接缝差）：E1 0.316、E2 0.326、E3 0.267、E4 0.225、E4-R 0.286。

解读边界：E3 与 E4 确实降低了接缝颜色跳变，E4-R 校准后拐弯数与 E4 对齐。**这些结果不能推出分类准确率提高**；更平滑的接缝也可能削弱人工接缝的正则化作用，必须以训练结果为准。

## 6. 给 A（训练与集成）

1. 三份配置可直接用于 200-epoch 训练；E4-R 的 `seam_random_mu` 只对当前 E4 参数有效。
2. 接缝参数确定后再启动正式 E4/E4-R；若先跑 E3，β 也属于待选参数。
3. 本分支 `member-c-seam` 基于 PR #1（提交 `9562fdf`），对 B/A 共享文件只做了小改动：`augmentations/__init__.py` 接入三个 policy 并新增三个参数；`train.py` 新增三行读取 `seam_min_width`、`seam_area_budget`、`seam_random_mu`；`tests/test_member_b.py` 中原先断言"C 未实现"的一行改为断言未知 policy 报 `ValueError`。`README_B.md` 里"C 的 E3/E4 策略未实现"一句已过时，留待 B 更新。
4. E4/E4-R 每个 batch 需要额外的 DP 计算。CPU 上 E3 约比 E2 多 60% 增强时间（batch 128、K=8 的合成数据粗测）；E4/E4-R 在 GPU 上的耗时尚未测量，需按协议预热并同步后计时。

## 7. 给 D（评价与协议）

1. **参数选择（验证集）**：建议优先 β ∈ {0.25, 0.5, 1}；E4 的 r ∈ {1, 2}、μ ∈ {0, 0.05, 0.2}、面积预算 ∈ {0.1, 0.2}。面积预算 0.1 会让约四分之一样本回退为矩形，削弱 E4 与 E3 的差异。
2. **E4 参数冻结后**重新运行校准脚本，把新 `seam_random_mu` 写入 E4-R 配置，并保留 JSON。
3. **形状对照需如实报告**：校准只对齐平均拐弯数；E4-R 边界仍长约 9%、面积变化略大。请报告两组的 `bends`、`seam_count`（边界长度）、`area_final/area_base`、`dp_status` 分布。回退在单个样本上可能不一致（E4 回退而 E4-R 被接受，或相反）。
4. **贴边现象**：E4 约 16% 的改形样本把边推到图像边缘（边缘处无接缝，代价为 0）。两组比例接近，贴边样本的接缝降幅与其他样本相当，但报告中应说明。
5. **RandomCrop 黑边**：约 99% 的训练图带黑边，E3/E4 中约 7% 的接缝像素对是"黑对黑"（E1 约 3%）。排除后接缝下降幅度基本不变。是否在接缝代价中排除填充像素、或改为在填充前混合，属于共享协议变更，需团队决定；当前实现未处理。
6. 边界诊断请直接读取 metadata 中的 `seam_count == 0` 判定不适用，不要把置 0 的代价当作完美接缝。

## 8. 局限与待定

- 所有结论均为单 seed、增强层面的诊断；没有训练结果，不声称性能提升或统计显著性。
- DP 只对左右两侧的局部代价精确最优，上下边和拐弯连接不在目标内。
- 面积约束为求解后检查并回退，而非在 DP 中强制；若最终参数下回退率仍高，可考虑把累计面积纳入 DP 状态。
- E4-R 用一个参数无法同时对齐拐弯数、边界长度与接受率。

## 9. 来源

CutMix（Yun et al., ICCV 2019）的面积标签与官方 `rand_bbox` 采样经由 A 的 `random_cutmix` 复用；局部路径动态规划思路借鉴 Seam Carving（Avidan & Shamir, 2007），此处用于混合区域的硬边界，不执行图像缩放。`third_party/CutMix-PyTorch/` 未改动。
