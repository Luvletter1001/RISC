# Task Plan: RISC 独立工作区与源码基座

## Goal

建立 `/data1/zcy/RISC` 独立 Git 仓库，保留 OpenRSD 与 OV-CapFlow 的源码性成果，写入 RISC-ER 的正式目标和可执行研究路线，并排除数据、模型及运行产物。

## Current Phase

In progress — CPU-only 封存 OpenRSD A10 N0-O 输入、scene plan 与 per-scene text7 ledger；GPU 仍未授权。

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
- [x] 提交 source snapshot（`f8313c3`）。
- **Status:** complete

### Phase 5: 交付与后续研究入口

- [x] 更新 findings/progress、完成定义和来源 manifest。
- [x] 记录 OpenRSD 工作日志的完成条目。
- [x] 准备向用户交付路径、提交和验证证据。
- **Status:** complete

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
| 再次扫描 metadata 的文件与目录两种形式 | 目录后缀过滤不能捕捉零字节文件，需显式覆盖两种 rsync 匹配形态。 |

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
| `.codex` 零字节文件与 `CODEX_WORKLOG.md` 绕过初版 source-only 边界 | 1 | 删除目标中两条精确路径，过滤器/扫描器同时匹配 metadata 文件和目录，并将 work log 视为运行状态。 |

## Notes

- 每次阶段变化前重读本文件和 `RISC_GOAL.md`。
- 初始来源快照一经提交不重写；后续刷新用新提交和新 manifest 表达。

## Active Research Phase: OpenRSD Final-Readout S0

- [x] 核对最新已提交 M1 设计及 SHA256，确认其晚于弱底座 RISC-ER 规格。
- [x] 确认 A10 head 的语义接入点为 `pred_embed -> rtm_cls_heads[idx]`，回归路径独立。
- [x] 确认复用既有 `OpenRSDHookRecorder`，不新建第二套 orbit 数据格式。
- [x] 更新并提交 RISC 权威目标、S0 规格和逐文件计划。
- [x] 按 TDD 实现默认关闭、zero-alpha 恒等的低秩 final-readout adapter。
- [x] 按 TDD 接入 A10 head，并增加 interface-only S0 config。
- [x] 按 TDD 扩展 full-tensor readout/geometry capture。
- [x] 完成独立复审、聚焦回归、编译和 branch-wide whitespace 验证。
- **Status:** complete

### S0 Stop Boundary

本阶段不启动 GPU、训练、N0-O 或 AP 评测。完成 S0 只授权下一步准备并封存
OpenRSD A10 的真实 N0-O 运行 manifest。

## Active Research Phase: OpenRSD N0-O Input Seal

- [x] 用户批准 CPU-only 运行前封存，明确不启动 GPU、推理或训练。
- [x] 核验 paper-mouth `6605/scale1024/text7/val_using_aux=False` 与历史结果权威。
- [x] 核验 A10 raw E24 checkpoint、support assets、dataset `13833=6605+7228`。
- [x] 定位并核验 160-scene、四折互斥、排除 P0148 的 C4/C8 scene plan。
- [x] 发现历史 P77E 每 batch 重采样 prompt，冻结新的 per-scene SHA-ranked text7 规则。
- [x] 提交输入封存规格与逐文件计划。
- [x] 按 TDD 实现 CPU-only deterministic seal builder。
- [x] 两次独立生成 byte-identical 后发布 tracked manifest/scene/ledger。
- [ ] 完成独立复审、测试、编译、branch-wide whitespace 与工作日志。
- **Status:** in_progress

### N0-O Input-Seal Stop Boundary

完成状态只能是 `SEALED_INPUTS_GPU_NOT_AUTHORIZED`。本阶段不得创建新预测、运行
model forward、占用 GPU、计算新 AP/rotation metric、执行 backward/optimizer 或写 checkpoint。

### Active Errors

| Error | Attempt | Resolution |
|---|---:|---|
| `openrsd` conda Python imported incompatible `~/.local` SciPy/sklearn during head-test collection | 1 | Run dependency-sensitive Python commands with `PYTHONNOUSERSITE=1`; require the next RED to reach the missing S0 interface. |
| Branch-wide diff check found two trailing-space hard breaks in the new S0 design header | 1 | Remove both hard breaks and rerun `git diff main...HEAD --check` before completion. |
| First real N0-O seal build rejected support dtype | 1 | Root cause: all 18 authoritative `text_embeds` arrays are finite float16 and historical runtime casts them with `torch.Tensor` before mapping. Add a float16-to-float32 regression test and amend the pre-prediction contract. |
| Builder fix commit used root-relative pathspecs from `framework/openrsd` | 1 | No files were staged; rerun `git add/commit` from `/data1/zcy/RISC` without repeating the mismatched cwd. |
| First tracked publish command had unmatched nested shell quote | 1 | Parser failed before builder execution; split target-absence check and direct `rtk env ... builder` into separate calls. |
