# 阶段 B：Edge-Guided Selection

B 在 A 上传的 CIFAR-100 / ResNet-18 框架上实现 E2（dual_edge）和 E2-S（source_edge）。根目录 `train.py`、`data/`、`models/` 是 A 框架的集成入口，优化器、模型、数据增强、归一化及训练循环保持 A 逻辑；训练新增源码和划分 SHA-256 记录。B 配置将 last checkpoint 间隔设为 10 epoch（最终 epoch 必存），减少个人 NAS 目录的写入；best checkpoint 仍在每次验证准确率严格提高时保存，选择规则不变。保存采用同目录临时文件加原子替换，A 默认配置的间隔仍为 1。A 的原始交付目录与 CSV 保留不动。C 的 E3/E4 策略未实现，调用时明确抛错。

## 完成状态与实测结果

B 的 17 项远端 CPU/CUDA 检查全部通过，5 组 200-epoch 正式实验全部完成，5 个 best checkpoint 均已独立复核。

单侧 K=4 最佳验证准确率 80.36%；本环境 K=1 控制为 80.06%。双侧 K=4/8 为 79.64%/79.62%，单侧 K=8 为 79.30%。均为 seed 42 的验证结果，未使用官方 test，不宣称统计显著提升。完整结果、负面结果、排除记录与模型清单见 [RESULTS_B.md](results/member_b/RESULTS_B.md)。

## 运行

建议 Python 3.10+、torch 2.5.1、torchvision 0.20.1、numpy 1.26.4、PyYAML、matplotlib。实际运行环境见结果目录 `requirements-portable.txt`（含私人路径的原始 freeze 仅保留在本地完整交付包）；CUDA 驱动需要匹配。

```bash
export B_ALLOWED_ROOT="$PWD"  # 当前项目须位于你获准写入的目录
mkdir -p tmp cache/pip
export TMPDIR="$PWD/tmp" XDG_CACHE_HOME="$PWD/cache" PIP_CACHE_DIR="$PWD/cache/pip"
export CUDA_CACHE_PATH="$PWD/cache/cuda" TORCH_HOME="$PWD/cache/torch"
export MPLCONFIGDIR="$PWD/cache/matplotlib" TRITON_CACHE_DIR="$PWD/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/inductor" PYTHONDONTWRITEBYTECODE=1
python -B -m venv .venv
source .venv/bin/activate
python -m pip install --no-compile -r requirements.txt
python -m unittest discover -s tests -v
# 只在训练集归档中读取 train；不使用官方 test 作开发和调参
python -c 'from torchvision.datasets import CIFAR100; CIFAR100("data/cifar100", train=True, download=True)'
python train.py --config configs/b_dual_edge_k4_s42.yaml --run-name unique_e2_run
python train.py --config configs/b_source_edge_k4_s42.yaml --run-name unique_e2s_run
python scripts/member_b_diagnostics.py
```

完整 GPU 顺序执行脚本 `scripts/run_member_b.sh` 通过 `B_ALLOWED_ROOT` 明确限制在获准写入的工作目录内运行。它使用目录内 `.venv`，把临时文件、Python/pip/Matplotlib/CUDA 缓存、数据、日志、checkpoint 全部写入此目录；不安装或升级系统环境。先单元测试，再图示和计时、2×2 batch 冒烟、5 epoch pilot，最后 5 组 200 epoch 训练。正式训练推荐顺序执行（本次 GPU 实测并发降低了总吞吐）；可通过 `B_RUN_WORKERS` 设置正式任务并发数，smoke/pilot 与增强计时始终隔离执行。并发任务的 walltime 会包含 GPU 争用，不当作隔离速度指标。可用 `B_RUN_SUFFIX` 为整套运行增加新后缀，避免覆盖已有 run 名。中断后不要重跑整套脚本覆盖证据，使用新的 run 名；当前未提供训练恢复功能。

## 冻结的 B 数学与几何约定

- 输入为浮点 `N×3×H×W` RGB `[0,1]`，在 crop/flip 后、CIFAR 标准化前；label `[N]` 与图像同 device。
- 灰度 `0.299R+0.587G+0.114B`，3×3 Sobel（未除 8），replicate padding，幅值 `sqrt(gx²+gy²)`。每张图除以其幅值均值的 `clamp_min(1e-6)`。常量图严格输出零；不逐候选归一化。半精度和 bfloat16 评分转 float32，float64 保留；最终图像和 lambda 保留输入 dtype。
- 区域均值通过 integral image 批量求和，内部 float64 累积防止大前缀相减吞掉极小边缘值，输出还原评分 dtype。空区域得分零。E2-S 最大化 Source 区域均值，E2 最大化 Source − gamma×Target，双侧默认 gamma=0.5、单侧配置为 0，仅为预先指定的实验值，不声称经验证最优。
- E1 基础采样原样复用 A `random_cutmix`：torch `randperm`，逐样本 torch Bernoulli、numpy Beta、整数宽高、torch 中心采样和裁剪；奇数宽高仍按 A 的两侧 `cut_size//2` 规则。原始代码不改动。
- 候选 0 为基础矩形，其他候选均匀采样能容纳相同实际高宽的左上角，坐标 `(top,left,bottom,right)` 半开。Source/Target 在同坐标粘贴。允许重复候选及自身配对，同分选择最小索引。
- 候选平移使用当前全局 torch 状态的独立克隆，不推进全局 RNG。逐候选使用固定 `N×2` draw shape，按 K-major 顺序采样，避免 CUDA RNG kernel 随总张量大小变化，因此相同 seed/batch 下 K=4 是 K=8 的前缀；K、策略不会改变下一个 batch 的 A 基础采样。CPU/CUDA RNG 实现不同，不要求跨设备位级一致。
- Mask `N×1×H×W` bool，1 为 Source。Source 来自未修改的原 batch。最终逐样本 Target 权重 `1−mask.mean()`，空矩形保持原图。未激活样本的空框可能被平移成另一空框，Mask 仍为空。
- K=1 无筛选，图像、标签、lambda 和 box 应与同 seed 的 A E1 完全相同；K=1 仍计算边缘 metadata，分类等价不代表耗时等于原生 E1，E1 原生耗时另按 random 路径测量；gamma=0 双侧应与单侧一致。C 可复用这些退化测试，而不是另定义采样。

## C 对接接口

```python
from augmentations.candidates import sample_candidates, boxes_to_mask
from augmentations.edge_utils import sobel_edges, rectangle_means, candidate_edge_scores
candidates, base = sample_candidates(images, labels, alpha=1., p=.5, k=4)
source, target, scores = candidate_edge_scores(images, base["permutation"], candidates, gamma=.5)
# candidates: int64 [N,K,4]; source/target/scores: [N,K]
# C 加入接缝评分后 argmax；再用 boxes_to_mask 或最终曲线 Mask 生成图像和权重。
```

`cutmix_batch` 保留 A 的完整签名和五元返回。E2 metadata：`policy`, `mask`, `boxes`（选中框）, `base_boxes`, `permutation`, `candidates`, `selected_indices`, `source_edge_scores`, `target_edge_scores`, `candidate_scores`。评分与候选均在原设备。`boxes_to_mask` 要求合法 int64 框；该低层工具不做逐元素 GPU 同步校验，外部手写框需自行保证范围。

`sample_candidates` 为保证旧 E1 完全一致，调用 A 的 sampler 并丢弃其混合图，再批量评分和构造最终 Mask。因此仍有 A 的逐样本采样同步开销，未声称全部采样向量化或零开销。Sobel、integral image、候选平移、选优与最终混合均批量运行。若未来优化 sampler，应重新验证 E1 并明确 RNG 版本变化。

## 实验与结果原则

使用 A 保存的 45k/5k 分层索引，seed 42、batch 128、200 epoch、SGD(lr=.1, momentum=.9, weight_decay=.0005)、cosine、alpha=1、p=.5，干净验证 Top-1 最大值选择 checkpoint，同分保留最早 checkpoint。五组配置为 dual K=1/4/8 与 source K=4/8，K=1 是共同控制。没有额外 gamma 搜索，不用 test 调参。

A 已上传 E0 77.74%、E1 80.52%，均为单 seed 的最佳验证准确率。它们来自 A40 / torch 2.9.1；B GPU 和 torch 版本可能不同，不能把跨环境耗时直接相除。K=1 控制提供 B 环境的采样一致性参考；小幅分类差异不应宣称统计显著。最终结果必须以 B 原始 CSV 为准，不用 A 指标填充 B 表。

`results/member_b/` 保存可视化、同步 GPU 计时、正确性输出和结果汇总；`experiments/registry.csv` 登记正式 run，失败也保留；每个 checkpoint 目录保存配置、环境、划分与源码 hash。单次 seed 不计算标准差。所有 B 结果均为 validation；最终 test 和多 seed 交由 A/D 统一开展。

独立 checkpoint 复核默认使用单进程验证读取，避免 NAS 临时 IPC 清理告警；训练的 DataLoader 与 A 保持 4 workers。原 workers=4 和最终 workers=0 的全部验证指标一致。
