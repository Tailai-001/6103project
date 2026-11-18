# 阶段 B：交付与实测结果

已完成 Sobel、共享同面积候选、E2-S 单侧、E2 双侧、统一 A 训练循环接入、正确性测试、配对图示、效率测量，以及 5 组完整 200-epoch 实验。C 的接缝和 DP 算法仍由 C 实现。

以下全部为 seed 42、固定 5,000 张干净验证集的结果。不是官方 test 指标，也不是多 seed 均值。

| 方法 | K | γ | 最佳 Val Top-1 (%) | 相对控制 (百分点) | Val Macro-F1 | 最佳 epoch | 完整进程 (min) |
|---|---:|---:|---:|---:|---:|---:|---:|
| K=1 / E1 等价增强控制 | 1 | 0.5 | 80.06 | +0.00 | 0.7996 | 186 | 46.58 |
| E2 双侧 | 4 | 0.5 | 79.64 | -0.42 | 0.7956 | 191 | 50.47 |
| E2-S 单侧 | 4 | 0 | 80.36 | +0.30 | 0.8023 | 197 | 46.29 |
| E2 双侧 | 8 | 0.5 | 79.62 | -0.44 | 0.7946 | 198 | 46.70 |
| E2-S 单侧 | 8 | 0 | 79.30 | -0.76 | 0.7924 | 189 | 45.37 |

## 结论与解释

本次最佳为单侧 K=4（80.36%），比本环境 K=1 控制高 0.30 个百分点。双侧 K=4/8 分别为 79.64%/79.62%，未超过控制。单侧 K=8 为 79.30%，比单侧 K=4 低 1.06 个百分点。增加候选数没有在本次实验中带来更好结果。
K=4 时双侧比单侧低 0.72 个百分点，K=8 时双侧比单侧高 0.32 个百分点，不能把 Target 边缘惩罚概括为普遍有效或无效。所有结果为一个 seed，0.30 个百分点不构成统计显著性证据。γ=.5 是预先指定值，未执行额外 γ 搜索。
边缘强度是低层结构线索。更高 Source 边缘分、保留更多 Target 边缘，都不等于更高语义质量或分类准确率；区域选择还会改变位置分布与正则化多样性。
A 已上传 E0=77.74%、E1=80.52%（A40 / torch 2.9.1+cu126）；B 使用 A800 / torch 2.5.1+cu124。原值完整保留，跨环境只作背景。上述增减均相对 B 自己的 80.06% 控制，不能声称超过 A 已上传的 80.52%。

## 正确性与公平性

- CPU 16 项通过、CUDA 项在本机跳过；远端 17 项全部通过、无跳过，原始输出见 tests.txt。
- K=1 的增强输出与同 seed 的 A E1 逐值一致；γ=0 双侧等于单侧；候选前缀、实际面积、像素替换、dtype/device、空/满/极小区域和常量输入均有检查。
- A 原始数据加载器、ResNet-18、E1 源文件保留；固定索引 SHA-256 完全一致，每类训练 450 / 验证 50，详见 split_validation.json。
- 五组均为 45,000 张训练样本 × 200 epoch，优化器、学习率、split、seed、α=1、p=.5 一致；所有 checkpoint 仅按干净验证 Top-1 选择，同分保留最早。
- 5 个 best checkpoint 已分别重新加载，在 5,000 张验证集复核，准确率与最佳 epoch 均与 CSV 一致，并计算 Macro-F1。

## 效率与诊断

设备 A800，batch=128，float32，p=.5，预热 10 次、同步重复 50 次。计时包含完整增强，不含数据加载、绘图或 classifier；单次隔离微基准不报告伪造的标准差。

| 路径 | K | 增强耗时 (ms/batch) |
|---|---:|---:|
| random | 1 | 12.51 |
| dual_edge | 1 | 13.95 |
| source_edge | 4 | 14.75 |
| dual_edge | 4 | 14.14 |
| source_edge | 8 | 14.38 |
| dual_edge | 8 | 14.78 |

K=1 控制仍计算边缘 metadata，分类等价不代表耗时等于原生 E1。原生 E1 的 random 路径单独计时。
固定 128 张训练图像、p=1 的配对诊断中，各方法平均替换面积均为 0.2975387573，选框位置可变而面积一致。单侧 K=4 的 Source/Target 区域边缘均值约 1.3721/1.0539，双侧约 1.3469/0.9414；这说明评分改变了选区行为，不保证提升分类。
epoch_seconds 在 checkpoint 写入前记录，求和不含全部保存 I/O。表中完整进程 walltime 来自 registry，包含初始化、加载和保存；不与 A40 的 epoch 时间直接比较。
5 组正式顺序任务的完整进程时间合计约 235.40 分钟，未包含环境安装、pilot 和被取消的并发尝试。

## 失败、排除与环境处理

最初 5 个并发尝试每组约 100–112 秒/epoch，单任务 pilot 约 10 秒/epoch，总吞吐下降。它们被早期取消并标记 excluded，日志与源码保留；最终 5 组均使用新后缀 _seq，从 seed 42 重新初始化、完整顺序训练。没有按分类好坏排除运行。
调度调整前后前三个控制组 epoch 的训练/验证 loss、准确率逐值一致，核心代码 hash 一致，见 restart_consistency.json。
原验证器在 NAS 多进程临时文件清理阶段出现 EBUSY 告警，5 个验证结果已成功写入。最终独立复核器默认 num_workers=0，避免 NAS IPC 临时文件清理问题；再次验证的准确率、Macro-F1、loss 与原 workers=4 结果完全一致。原始输出及 comparison 均保留。训练代码与训练时 workers=4 未更改。

## 文件与复现

- final_results.json / results/tables/member_b_validation_results.csv：最终指标、耗时、相对控制差值和 run_id。
- raw_logs/ 与项目 logs/：原始 CSV；完整 stdout、排除记录保留在交付包。
- run_metadata/：每个运行的 resolved config、环境、源文件及 split 校验值。
- checkpoint_manifest.json：远端 10 个 best/last 文件的路径、大小、SHA-256。大型模型不装入源码 ZIP。
- paired_examples.png / sobel_candidates.png / validation_curves.png：GPU 图示与收敛曲线；*_cpu.png 为独立 CPU 示例。
- requirements-portable.txt：可移植的 Linux GPU 依赖版本列表；含私人路径的原始 freeze 仅在本地完整交付包保留，公开仓库使用 portable 版本。
- source_ef7d7ed365408bd8.tar.gz / ai6103_member_b_source_v2.tar.gz：本地完整交付包中的初始及顺序实验源码快照（不公开内部服务器路径），SHA 见 registry；最终独立验证脚本的 SHA 另记录在 checkpoint_validation.json。

在 GPU 个人目录内安装依赖并准备同一数据后，使用新后缀防止覆盖已有记录：

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
B_RUN_WORKERS=1 B_RUN_SUFFIX=repro1 bash scripts/run_member_b.sh
```

已有实验的复核命令：`python scripts/evaluate_member_b_val.py --suffix seq --num-workers 0`。C 对接接口、精确采样和归一化约定见 README_B.md。官方 test、多 seed 与 C 的 E3/E4 仍由团队后续完成。
