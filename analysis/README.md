# 结果分析（D）

[交接说明](../RESULTS_D.md) 链接全部报告表格、图和复现命令。

- `compile_results.py` 从 A/B/C 原始日志、配置、运行登记、checkpoint 复评与计时记录生成主表、消融、收敛曲线和效率图。
- `boundary_audit.py` 沿用 C 的配对样本诊断设计和统一增强接口，追踪真实 RandomCrop 填充，重算最终掩码的 RGB 差异、面积、边界长度、拐弯与回退。诊断范围是固定训练划分。
- `plot_boundary.py` 从审计 CSV/JSON 生成边界表和分布图。无可用内部接缝按 N/A 计。
- 根目录 `evaluate.py` 复用 C 的评价指标计算，默认固定验证集，可显式选择官方测试集。
- `audit_training_evidence.py` 独立核对 18 份正式日志、已存配置与历史源码哈希。
- `paired_boundary_evidence.py` 对 D 的原始逐图 CSV 做配对 batch bootstrap 和填充／贴边／形状敏感性分析；`plot_paired_boundary_evidence.py` 生成对应图。区间不能用于推断分类性能。
- 本地同机三臂复核与新增测试见 [RESULTS_AUDIT.md](../RESULTS_AUDIT.md)。
