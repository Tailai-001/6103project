# 运行工具

`run_member_b.sh` 完成个人目录内环境隔离、测试、数据准备、可视化、冒烟和 pilot 后调用 `member_b_suite.py` 运行 B 的 5 个正式实验（默认顺序，B_RUN_WORKERS 可设置并发）。`member_b_diagnostics.py` 导出配对图、Sobel/候选图以及完整增强同步计时。所有正式训练均调用 A 统一循环 `train.py`。详见 README_B.md。

`run_member_c.sh` 依次完成测试、数据准备、GPU 增强同步计时（`member_c_aug_timing.py`）、冒烟和 5 epoch pilot，然后由 `member_c_suite.py` 按顺序运行 C 的 E3/E4/E4-R 正式实验，最后用 `summarize_member_c.py` 汇总验证集结果。`train_member_c.slurm` 是 SLURM 模板。所有正式训练均调用统一的 `train.py`。详见 README_C.md 第 3.1 节。
