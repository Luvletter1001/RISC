# Task Plan: RISC 独立工作区与源码基座

## Goal

建立 `/data1/zcy/RISC` 独立 Git 仓库，保留 OpenRSD 与 OV-CapFlow 的源码性成果，写入 RISC-ER 的正式目标和可执行研究路线，并排除数据、模型及运行产物。

## Current Phase

Phase 4 — 迁移验证与第二个提交。

## Phases

### Phase 1: 需求、来源与边界

- [x] 确认目标路径、独立 Git 需求和 source-only 迁移授权。
- [x] 核对两个源仓库的 HEAD、远端和 dirty 状态。
- [x] 固化排除政策和 RISC 研究边界。
- **Status:** complete

### Phase 2: 权威文件与首个提交

- [x] 初始化 Git `main` 分支。
- [x] 写入并验证根目标、设计、计划和持久化工作记录。
- [x] 创建仅含文档的首个提交（`6bc4dc6`）。
- **Status:** complete

### Phase 3: OpenRSD 与 OV-CapFlow 源码快照

- [x] 以非破坏性筛选规则复制 OpenRSD 框架底座。
- [x] 以相同规则复制 OV-CapFlow/RISC-ER 参考。
- [x] 生成来源、计数和哈希 manifest。
- **Status:** complete

### Phase 4: 迁移验证与第二个提交

- [x] 验证关键代码/规格存在。
- [x] 验证 Git 已跟踪路径中不存在被禁止的路径或扩展名，三份 allowlisted BPE 词表除外。
- [x] 验证本仓库新写 authority 文件的 whitespace，并记录来源快照的历史诊断。
- [ ] 提交 source snapshot。
- **Status:** in_progress

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
| 初始 source snapshot 不批量格式化 | 保留来源工作树的可追溯字节内容；只对本仓库新写文件强制 whitespace clean。 |

## Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| 新目录在首次检查时不存在 | 1 | 用户创建 `/data1/zcy/RISC` 后继续。 |
| 占位词扫描匹配到扫描命令自身 | 1 | 将模式拆为运行时 shell 字符串，再重新扫描。 |
| `rtk git diff --cached --check` 返回状态 2 且没有 Git 诊断 | 1 | 改用 `rtk run 'git diff --cached --check'` 取得原生命令结果。 |
| 原生 staged whitespace 校验发现 `RISC_GOAL.md` 两行尾随空格 | 1 | 移除 Markdown 硬换行空格后重新暂存与验证。 |
| 首次 OpenRSD 预演仍包含约 15.9 GB | 1 | 定位为实验树的 outputs/reports；排除 `experiments/`，仅二次复制四个实验代码目录。 |
| OpenRSD `results` 绝对符号链接穿透到外部结果树 | 1 | 精确移除目标链接，并将过滤规则由 `results/` 扩展为 `results` 与 `results/`。 |
| 首次绝对链接审计的 `awk` 程序受 shell 引号影响而语法失败 | 1 | 改用 `find -type l -lname '/*'`，无需字段解析。 |
| 完整 source snapshot 的 staged whitespace 检查产生 2,021 行继承诊断 | 1 | 确认示例文件与来源逐字相同；保留快照原样，改用 scoped 检查验证本仓库新写 authority 文件并在 manifest 记录残余。 |
| 更新格式保真政策的首个补丁上下文不匹配 | 1 | 读取当前精确段落后重新应用，不改动源码快照。 |
| 禁入扫描发现三份 `.txt.gz` 文件 | 1 | 证实为 tokenizer 直接依赖的 BPE 词表；保留三条精确 allowlist，忽略/拒绝其余 `.gz`。 |

## Notes

- 每次阶段变化前重读本文件和 `RISC_GOAL.md`。
- 初始来源快照一经提交不重写；后续刷新用新提交和新 manifest 表达。
