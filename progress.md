# Progress Log

## Session: 2026-08-22

### Phase 1: 需求、来源与边界

- **Status:** complete
- **Started:** 2026-08-22 Asia/Shanghai
- Actions taken:
  - 用户确认创建 `/data1/zcy/RISC` 并同意“独立 Git + 筛选代码快照”迁移。
  - 检查目标目录为空且尚未初始化 Git。
  - 读取并记录 OpenRSD、OV-CapFlow 的 Git 基线、远端和 dirty 摘要。
  - 已向 `/data1/zcy/OpenRSD/CODEX_WORKLOG.md` 追加本次迁移的 start 记录。
- Files created/modified:
  - `/data1/zcy/OpenRSD/CODEX_WORKLOG.md`（仅追加工作记录）
  - `task_plan.md`、`findings.md`、`progress.md`（本仓库）

### Phase 2: 权威文件与首个提交

- **Status:** complete
- Actions taken:
  - 初始化 `/data1/zcy/RISC/.git`，分支为 `main`。
  - 起草根目标、迁移设计、排除政策和逐项实施计划。
  - 完成占位词与原生 Git whitespace 校验；首个提交为 `6bc4dc6 docs: establish RISC research authority`。
- Files created/modified:
  - `.gitignore`
  - `README.md`
  - `RISC_GOAL.md`
  - `docs/provenance/MIGRATION_POLICY.md`
  - `docs/superpowers/specs/2026-08-22-risc-repository-migration-design.md`
  - `docs/superpowers/plans/2026-08-22-risc-workspace-establishment.md`

### Phase 3: OpenRSD 与 OV-CapFlow 源码快照

- **Status:** complete
- Actions taken:
  - 重读迁移政策，确认 `rsync` 可用，并准备先做不写入的预演。
  - 预演发现 OpenRSD 会复制约 15.9G；定位到 `experiments/rotation_semantic_attractor` 的运行 outputs/reports，已改为窄迁入其四个代码目录。
  - 复制后审计发现 `framework/openrsd/results` 是外部绝对链接；正在按迁移政策移除并收紧过滤规则。
  - 已复制 OpenRSD 主底座 5,562 个常规文件（124,518,419 bytes）和实验代码 190 个常规文件（1,571,796 bytes）。
  - 已复制 OV-CapFlow 792 个常规文件（19,127,929 bytes），并写入 `SOURCE_SNAPSHOT_MANIFEST.md`。
  - 关键 RISC 源文件、正式规格和实施计划与来源的 `cmp -s` 比对通过；目标中不存在外部绝对符号链接。
- Files created/modified:
  - `task_plan.md`
  - `docs/superpowers/plans/2026-08-22-risc-workspace-establishment.md`
  - `framework/openrsd/`
  - `reference/ov-capflow/`
  - `docs/provenance/SOURCE_SNAPSHOT_MANIFEST.md`

### Phase 4: 迁移验证与第二个提交

- **Status:** complete
- Actions taken:
  - 已完成 snapshot 前 provenance 记录，关键路径、实验代码边界、禁止物和外部链接检查均通过。
  - 完整 staged whitespace 检查报告 2,021 条来源继承诊断；已以 source/target `cmp -s` 复现根因，新写 authority 文件 scoped 检查通过，格式保真决定已写入政策与 manifest。
  - 禁入扫描识别出三份 CATSeg tokenizer BPE `.gz` 词表；已验证 Python 直接依赖，收窄为精确路径/哈希 allowlist，其他 `.gz` 仍忽略。
  - 已重新验证：禁止 tracked path 为零、allowlist 默认拒绝生效、关键 RISC 源文件仍与来源一致。
  - 提交 `f8313c3 chore: snapshot RISC source foundations` 已创建；提交后 Git path policy、scoped whitespace 和关键 SHA 都通过。
  - `git fsck` 返回零；有 25 个中间暂存产生的悬空 blob、无悬空 commit/tree，未执行任何垃圾回收。
  - 二次审计发现 `.codex` 文件和 `CODEX_WORKLOG.md` 绕过初版目录式过滤；已在 `921d972` 中精确删除，并通过加强后的规则复验。
- Files created/modified:
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

### Phase 5: 交付与后续研究入口

- **Status:** complete
- Actions taken:
  - 已向 OpenRSD `CODEX_WORKLOG.md` 追加 start/finish 记录。
  - 已重读 `RISC_GOAL.md` §10，并核对独立 Git、两个源码子树、manifest、严格路径政策、格式边界和源仓库修改范围。
  - 最终提交树审计：`main` 干净；`921d972` 为边界修正；4,997 个 tracked paths；OpenRSD 5,750 文件/136M；OV-CapFlow 792 文件/21M；历史 whitespace 残余 1,751 行；无悬空 commit/tree。
- Files created/modified:
  - `RISC_GOAL.md`
  - `docs/provenance/SOURCE_SNAPSHOT_MANIFEST.md`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

## Test Results

| Test | Input | Expected | Actual | Status |
|---|---|---|---|---|
| 新目录预检 | `rtk ls -la /data1/zcy/RISC` | 空目录 | 空目录 | pass |
| Git 初始化 | `rtk git init -b main` | 空 Git 仓库 | 已创建 `.git` | pass |
| 来源 SHA | `git rev-parse HEAD` | 两个可读取 SHA | OpenRSD `12d3fd8`、OV-CapFlow `e87ae43` | pass |
| 跟踪路径政策 | `git ls-files` + allowlist scan | 无禁止路径 | 仅三份 BPE `.gz` 例外，扫描通过 | pass |
| 新 authority 格式 | scoped `git diff --check` | 无空白错误 | exit 0 | pass |
| Git 对象连通性 | `git fsck` | 无损坏/悬空 commit/tree | exit 0；25 个中间 blob 保留 | pass |
| 最终边界审计 | 严格 `git ls-files` 扫描 | 无 metadata/日志/产物；仅三份 BPE 例外 | `921d972` 后通过 | pass |

## Error Log

| Timestamp | Error | Attempt | Resolution |
|---|---|---:|---|
| 2026-08-22 Asia/Shanghai | 目标目录在用户首次建议时尚未存在 | 1 | 等待用户创建后继续。 |
| 2026-08-22 Asia/Shanghai | 占位词扫描匹配到扫描命令中的关键词 | 1 | 改用运行时拼接的 pattern 后重新扫描。 |
| 2026-08-22 Asia/Shanghai | `rtk git diff --cached --check` 返回状态 2 且无诊断 | 1 | 使用 `rtk run` 调用原生命令；未在验证失败时提交。 |
| 2026-08-22 Asia/Shanghai | 原生 Git 发现 `RISC_GOAL.md` 两行尾随空格 | 1 | 已删除硬换行空格，重新进行 staged 校验。 |
| 2026-08-22 Asia/Shanghai | OpenRSD 初版 rsync 预演仍为约 15.9G | 1 | 排除整个 `experiments/`，随后仅迁入 `configs/src/scripts/tests`。 |
| 2026-08-22 Asia/Shanghai | 目标保留了 OpenRSD `results` 的绝对符号链接 | 1 | 通过 `unlink` 移除精确目标，并升级排除规则。 |
| 2026-08-22 Asia/Shanghai | `awk` 链接审计因引号错误无法执行 | 1 | 使用 `find -lname '/*'` 替代。 |
| 2026-08-22 Asia/Shanghai | 完整 staged whitespace 检查产生 2,021 条来源继承诊断 | 1 | 用 `cmp -s` 和来源 whitespace 查找确认根因；保留 source fidelity，单独验证新 authority 文件。 |
| 2026-08-22 Asia/Shanghai | 格式保真政策的首个补丁上下文不匹配 | 1 | 读取当前段落后以精确上下文重新应用；没有文件被错误修改。 |
| 2026-08-22 Asia/Shanghai | 禁入扫描命中三份 CATSeg `.txt.gz` | 1 | 验证为 tokenizer 运行时 BPE 依赖，建立精确 allowlist，其余 `.gz` 保持禁止。 |
| 2026-08-22 Asia/Shanghai | `.codex` 文件和 `CODEX_WORKLOG.md` 留在 source snapshot | 1 | 扩展 metadata 文件/目录规则，并在最终修正提交中精确移除两条路径。 |

## 5-Question Reboot Check

| Question | Answer |
|---|---|
| Where am I? | 完成：RISC 独立工作区已建立并完成最终审计。 |
| Where am I going? | 工作区已交付；下一项工作从 `RISC_GOAL.md` 的 Stage 1 E12 诊断门开始。 |
| What's the goal? | 独立、可审计、不含运行产物的 RISC 代码底座与研究总纲。 |
| What have I learned? | 见 `findings.md`。 |
| What have I done? | 见本文件及 `task_plan.md`。 |
