# 实验登记

`registry.csv` 由 B 的 `scripts/member_b_suite.py` 在每个训练任务启动前登记，完成后更新状态；smoke、pilot、formal 分别标注，失败保留 exit code。每个运行使用唯一 run_id，不覆盖已有证据。A 的原始 E0/E1 CSV 保留在 A 上传目录。

GPU 工作副本不含 Git 历史时，git_commit 留空，notes 写明完整源码归档 SHA-256；每个训练环境另记录 Python 源码和固定划分的 SHA-256。运行配置、日志及 checkpoint 路径均相对项目根目录。

B 的结果仅是固定 validation，官方 test 表由 A/D 后续填写；禁止把验证准确率填入 test 字段。多 seed 和 C 的 seam/DP 实验需另行登记。
