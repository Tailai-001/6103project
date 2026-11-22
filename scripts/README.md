# 运行工具

`run_member_b.sh` 完成个人目录内环境隔离、测试、数据准备、可视化、冒烟和 pilot 后调用 `member_b_suite.py` 运行 B 的 5 个正式实验（默认顺序，B_RUN_WORKERS 可设置并发）。`member_b_diagnostics.py` 导出配对图、Sobel/候选图以及完整增强同步计时。所有正式训练均调用 A 统一循环 `train.py`。详见 README_B.md。

`run_member_c.sh` 依次完成测试、数据准备、GPU 增强同步计时（`member_c_aug_timing.py`）、冒烟和 5 epoch pilot，然后由 `member_c_suite.py` 按顺序运行 C 的 E3/E4/E4-R 正式实验，最后用 `summarize_member_c.py` 汇总验证集结果。`train_member_c.slurm` 是 SLURM 模板。所有正式训练均调用统一的 `train.py`。详见 README_C.md 第 3.1 节。
`plot_member_c_curves.py` 从已提交的 CSV 日志生成 `results/member_c/validation_curves.png`。
`evaluate_member_c_val.py` 重新加载 C 的 best checkpoint，复核验证准确率并计算 Macro-F1（只用验证集）。

`run_local_confirmation.py` 在 Apple MPS 上按顺序运行冻结的 seed-43 E1/E4/E4-R 三臂，增强保留 CPU 原始算法，仍调用同一 `train.py`。它记录退出状态与源码哈希，拒绝覆盖现有运行记录；`--smoke` 只做流程预检，不能计入正式结果。详细设置见 [实验计划](../results/local_audit/EXPERIMENT_PLAN.md)。
