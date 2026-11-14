# 实验登记（D）

`registry.csv` 当前只有表头，没有运行记录。每个 seed 单独一行，用唯一 `run_id`。状态建议 `planned / running / completed / failed / excluded`，排除需在 notes 中写明原因。

配置、数据划分与 commit 必须可追溯；日志/checkpoint 可放共享存储并登记 URL。K/γ 的选择只用验证集，测试评价在配置冻结后进行。草案字段参见 [配置模板](../configs/experiment_template.json)。
