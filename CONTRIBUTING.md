# 协作方式

先阅读 [项目首页](README.md) 和 [实验协议](docs/experiment_protocol.md)。A 负责统一训练入口与合并，B/C 共用增强工具，D 核对实验公平性，E 汇总报告。

1. 克隆仓库后，从 `main` 新建自己的分支，如 `feature/source-edge`、`feature/training`、`docs/report`。Codex 自动创建的分支使用 `codex/` 前缀。
2. 保持改动集中，提交前同步 `main`；通过 Pull Request 说明修改、复用来源与验证结果。
3. 修改共享接口、数据划分、候选采样或实验预算时，同时更新协议并由相关负责人审阅。
4. 每次训练先在 `experiments/registry.csv` 登记，保存完整配置、commit、seed、依赖版本和原始日志。失败运行也保留记录。
5. 公开仓库仅提交必要代码和可核验的小型结果。数据、checkpoint 和视频存放共享存储；学生信息和课程提交材料按课程要求另行汇总。

当前实验登记表和结果表都是空模板，目录说明中的代码文件名是待实现交付物。项目级安装和运行命令由 A 在首次验证训练后补充。

只读访问和克隆公开仓库无需加入协作者。需要直接推送的成员由仓库拥有者通过 GitHub Settings → Collaborators 邀请；也可以采用 fork + Pull Request。
