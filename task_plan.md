# Task Plan: RISC 独立工作区与源码基座

## Goal

建立 `/data1/zcy/RISC` 独立 Git 仓库，保留 OpenRSD 与 OV-CapFlow 的源码性成果，写入 RISC-ER 的正式目标和可执行研究路线，并排除数据、模型及运行产物。

## Current Phase

Phase 1 — 仓库权威文件与迁移边界。

## Phases

### Phase 1: 需求、来源与边界

- [x] 确认目标路径、独立 Git 需求和 source-only 迁移授权。
- [x] 核对两个源仓库的 HEAD、远端和 dirty 状态。
- [x] 固化排除政策和 RISC 研究边界。
- **Status:** complete

### Phase 2: 权威文件与首个提交

- [x] 初始化 Git `main` 分支。
- [ ] 写入并验证根目标、设计、计划和持久化工作记录。
- [ ] 创建仅含文档的首个提交。
- **Status:** in_progress

### Phase 3: OpenRSD 与 OV-CapFlow 源码快照

- [ ] 以非破坏性筛选规则复制 OpenRSD 框架底座。
- [ ] 以相同规则复制 OV-CapFlow/RISC-ER 参考。
- [ ] 生成来源、计数和哈希 manifest。
- **Status:** pending

### Phase 4: 迁移验证与第二个提交

- [ ] 验证关键代码/规格存在。
- [ ] 验证 Git 未跟踪被禁止的路径或扩展名。
- [ ] 检查空白、工作树和提交树。
- [ ] 提交 source snapshot。
- **Status:** pending

### Phase 5: 交付与后续研究入口

- [ ] 更新 findings/progress 和完成定义。
- [ ] 记录 OpenRSD 工作日志的完成条目。
- [ ] 向用户交付路径、提交和验证证据。
- **Status:** pending

## Key Questions

1. 迁移是否可以在不含任何数据、权重、日志和可视化的条件下保留全部代码性成果？**已确认：可以。**
2. 将 OpenRSD 的本地未提交代码当作上游发布版本吗？**否；它只作为当前工作树的快照并在 manifest 中明确标记。**
3. 本次是否启动训练或修改源仓库代码？**否。**

## Decisions Made

| Decision | Rationale |
|---|---|
| 新建独立 Git `main` 仓库 | 避免现有两个 dirty 工作树相互污染。 |
| 使用源码快照而非完整复制或 submodule | 同时保留本地代码性成果、上游基线和可审计边界。 |
| 排除运行产物 | 数据、模型和结果规模大且不可由 Git 可靠复现。 |
| 将 B\* 与 RISC-ER 分层 | 防止系统工程增益被误归因给论文方法。 |

## Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| 新目录在首次检查时不存在 | 1 | 用户创建 `/data1/zcy/RISC` 后继续。 |
| 占位词扫描匹配到扫描命令自身 | 1 | 将模式拆为运行时 shell 字符串，再重新扫描。 |
| `rtk git diff --cached --check` 返回状态 2 且没有 Git 诊断 | 1 | 改用 `rtk run 'git diff --cached --check'` 取得原生命令结果。 |
| 原生 staged whitespace 校验发现 `RISC_GOAL.md` 两行尾随空格 | 1 | 移除 Markdown 硬换行空格后重新暂存与验证。 |

## Notes

- 每次阶段变化前重读本文件和 `RISC_GOAL.md`。
- 初始来源快照一经提交不重写；后续刷新用新提交和新 manifest 表达。
