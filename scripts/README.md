# 运行工具

`run_member_b.sh` 完成个人目录内环境隔离、测试、数据准备、可视化、冒烟和 pilot 后调用 `member_b_suite.py` 运行 B 的 5 个正式实验（默认顺序，B_RUN_WORKERS 可设置并发）。`member_b_diagnostics.py` 导出配对图、Sobel/候选图以及完整增强同步计时。所有正式训练均调用 A 统一循环 `train.py`。详见 README_B.md。
