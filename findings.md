# Findings & Decisions

## Requirements

- 在 `/data1/zcy/RISC` 创建独立 Git 工作区。
- 保留代码性成果和 OpenRSD 框架底座。
- 写入最详细的书面目标/计划。
- 排除数据、权重、日志、缓存和结果大文件；用户已明确同意该迁移方式。

## Research Findings

- `/data1/zcy/OpenRSD` 当前基线为 `12d3fd8b75e8b64ec53fded9cf035a2306d58874`，远端为 `https://ghfast.top/https://github.com/floatingstarZ/OpenRSD.git`，存在 38 项已修改/删除状态和 146 项未跟踪状态（按 `git status --porcelain` 行计）。
- `/data1/zcy/OV-CapFlow` 当前基线为 `e87ae43ad294d9918cd6a36a458bf6f03dabb9fe`，远端为 `https://github.com/VisionXLab/CastDet.git`，存在 2 项已修改/删除状态和 24 项未跟踪状态。
- OpenRSD 具有 `risc_orbit_projection_head.py`、`risc_orbit_projection.py` 和相应诊断配置；OV-CapFlow 具有经确认的 RISC-ER 正式规格和实施计划。
- 历史 OpenRSD 仓库的 `data/` 链接不可作为有效数据来源；后续实验使用 `/data1/zcy/datasets` 的绝对路径。
- 首次 source-only 预演中，OpenRSD 仍有 217,688 个文件、15,871,071,596 bytes；定位后发现主因是 `experiments/rotation_semantic_attractor` 的约 16G `outputs/` 和约 1.6G `reports/`，而可复用 `configs/src/scripts/tests` 合计仅约 3.8M。
- 实际快照包含 OpenRSD 5,752 个常规文件（目标占用 137M）和 OV-CapFlow 792 个常规文件（目标占用 21M）；四个关键 RISC 文件已用 `cmp -s` 与来源逐项一致比较。
- 完整 staged `git diff --cached --check` 有 2,021 行历史诊断；比较显示 `framework/openrsd/CODEX_WORKLOG.md` 与来源逐字相同且来源已存在同类空白。新写 RISC authority 文件的 scoped 检查为零错误。
- 禁入扫描发现三份 1.3M、SHA256 相同的 `.txt.gz` 文件；它们被 `DeCLIP_CATSeg` 的三份 tokenizer Python 文件以相对路径直接读取，是静态 BPE 词表依赖。
- source snapshot 提交为 `f8313c3`；提交后为 4,999 个 tracked paths。`git fsck` 返回零、无悬空 commit/tree，但保留 25 个由中间暂存生成的悬空 blob；未授权垃圾回收，故未清理。
- 二次提交树审计发现 `framework/openrsd/.codex` 是来源中的零字节 metadata 文件，`framework/openrsd/CODEX_WORKLOG.md` 是 701,733-byte 运行日志；二者均不属于代码性成果。
- 最终边界修正提交为 `921d972`：当前 4,997 个 tracked paths，OpenRSD 5,750 个常规文件/136M，OV-CapFlow 792 个常规文件/21M；严格路径政策和结构审计均通过。最终全树历史 whitespace 残余为 1,751 行。

## Technical Decisions

| Decision | Rationale |
|---|---|
| 以当前工作树的 source-only 快照迁移 | 已提交基线不足以保留本地代码成果；完整树会混入不可审计的运行产物。 |
| 采用 `rsync -a` 且不使用 `--delete` | 不修改源树，也不让重复迁移删除目标中的资料。 |
| 初始来源和文档分两次提交 | 让研究 authority 与源码快照可分别审查和追责。 |
| 使用目标根 `RISC_GOAL.md` 作为最高目标 | 用户要求最详细的书面目标文件，需要比历史具体实施计划更上层的边界和门槛。 |
| 实验树采取“全排除 + 四个代码目录窄迁入” | 保留 orbit/RISC 分析代码，避免将 outputs 和 reports 作为源码误入 Git。 |
| 移除目标中的 OpenRSD `results` 绝对链接 | 不复制外部结果数据，也不让新仓库隐式依赖旧结果目录。 |
| 保留来源继承 whitespace，而不做批量格式化 | 这次任务是源码保真迁移；新文件的 scoped 校验仍严格执行，残余已写入 manifest。 |
| 仅允许三份 CATSeg BPE `.gz` 词表 | 保持 vendored tokenizer 的直接文件依赖；通过路径加哈希 allowlist 防止普通压缩包进入 Git。 |
| 从最终快照删除 `.codex` 和 `CODEX_WORKLOG.md` | 它们是工具/运行状态，违反 source-only 政策；框架代码不依赖二者。 |

## Issues Encountered

| Issue | Resolution |
|---|---|
| 直接复制会带入 OV-CapFlow 的约 402G `work_dirs`、13G `pretrained` 等运行资产 | 用显式过滤政策排除。 |
| 两个来源都是 dirty 工作树 | 将 HEAD、远端和 dirty 摘要写入来源 manifest，不把快照伪装为上游发布版本。 |
| 占位词扫描会命中自身的检索表达式 | 通过运行时拼接检索模式，仍然扫描全部政策与计划文件。 |
| RTK 的 `git diff --cached --check` 包装未给出可用状态 | 对 staged whitespace verification 使用 `rtk run` 执行原生 Git；仍保留 RTK 作为命令前缀。 |
| 初稿用 Markdown 两个空格表示硬换行，触发 Git whitespace check | 正式目标改用普通换行，避免尾随空格。 |
| 初版 rsync 没有排除 OpenRSD `experiments/`，预演约 15.9G | 更新迁移政策后执行新的预演；任何实际复制继续等待预演规模合理。 |
| `rsync -a` 将 OpenRSD 的顶层 `results` 绝对链接作为链接保留 | 删除目标中的单个链接，添加不带尾斜杠的排除规则，并在最终审计中禁止外部绝对链接。 |
| 基于 `awk` 字段的绝对链接审计受到嵌套 shell 引号影响 | 使用 `find -lname '/*'` 直接匹配绝对链接目标。 |
| 全量 staged whitespace 检查报告大量继承问题 | 根因是 source/reference 原有文本；选择记录并隔离校验范围，而不是改写数千行历史快照。 |
| 格式保真政策的首次补丁定位失败 | 重新读取当前段落后以精确上下文更新；未改变任何 snapshot 文件。 |
| 初版禁入扫描将所有 `.gz` 一律视为禁止 | 检查 OpenCLIP/EVA 引用后收窄为三条 tokenizer 词表 allowlist。 |
| `git fsck` 报告 25 个悬空 blob | 它们来自本任务的中间暂存；无悬空 commit/tree，且 Git 返回零。为避免未经授权的不可恢复清理，保留对象。 |
| 初版 metadata 过滤器只有带斜杠目录模式 | 源 `.codex` 是文件而非目录；规则和最终扫描改为覆盖名字及名字加 `/`。 |

## Resources

- `RISC_GOAL.md`
- `docs/provenance/MIGRATION_POLICY.md`
- `docs/superpowers/specs/2026-08-22-risc-repository-migration-design.md`
- `reference/ov-capflow/docs/superpowers/specs/2026-08-11-risc-er-formal-design.md`（快照完成后）

## Visual/Browser Findings

本任务未使用浏览器或视觉材料。

## 2026-08-22 — OpenRSD Final-Readout S0

- `[FACT]` 最新已提交设计 `a812837` 固定 OpenRSD A10 E24 为首选强父模型；设计文件 SHA256 为 `a2e988bafc0dbf25688c1e42e6a8b3405bb705517674cde4e32e7bce3a1bfb99`。
- `[FACT]` 当前 RISC source snapshot 未包含 `research/risc-er` worktree 的 query-based 生产实现；该 worktree 还混有大量未提交后续实验，不能整体迁入。
- `[FACT]` A10 的 `pred_embed` 在 `rtm_cls_heads[idx]` 前生成；bbox、angle 和 objectness 使用独立 `reg_feat`，因此存在只改语义 readout 的窄接入点。
- `[FACT]` 既有 `OpenRSDHookRecorder` 已能定位 dense logits、class embedding projection 和 head 输出，但只记录统计量并覆盖重复模块调用。
- `[DECISION]` S0 扩展既有 recorder 捕获多调用 full tensors；核心 head 不新增文件 writer，orbit runner 继续负责 scene/angle metadata 与序列化。
- `[DECISION]` 旧 E12/Q600 诊断降为弱底座历史协议；当前顺序固定为 `S0 -> N0-O -> M1-P -> M1-C`。
- `[FACT]` A10 resolved bbox-head config has `with_objectness=False`; capture completeness must require objectness only for parents that actually register `rtm_obj` and must seal its structural absence for A10.
- `[FACT]` The head's returned `pred_embeds` feed historical `loss_align`/CCL consumers. S0 therefore uses a separate adapted readout variable for classification and returns the unadapted parent embedding to those consumers.
- `[FACT]` Follow-up independent review of `ab12e76` returned PASS with no Critical/Important findings. It independently verified 33 focused tests, preservation of all 10 CUDA RNG states, the low-norm `0.05` residual bound, resolved A10 capture with structural objectness absence, and full-head parent parity.

## 2026-08-22 — OpenRSD N0-O Input Seal

- `[FACT]` A10 raw E24 checkpoint is 1,180,290,410 bytes with SHA256 `097585080a4c95f23370840da546acfdcb85093480454133ad2a34876cc20bd6`; it contains both raw and EMA states, and P77E authority uses raw `state_dict`.
- `[FACT]` DOTA2 `ss_val` contains 13,833 images and annotations: 6,605 non-empty and 7,228 empty. Paper-mouth is only the non-empty 6,605 at scale1024/text7/`val_using_aux=False`.
- `[FACT]` Reusable scene authority `scene_plan_40.json` has SHA256 `0b2c190bfa7231cb19cbf746aaf4612e898f9417263ac383087c2746e69abe35`, 160 unique scenes, four disjoint 40-scene C4/C8 folds, and excludes P0148.
- `[FACT]` Historical P77E `predict()` samples support with `np.random.permutation` on every batch; a path/hash alone cannot satisfy same-scene identity/C4/C8 support equality.
- `[DECISION]` N0-O support is frozen per scene by SHA-ranking each class's source prompt indices and taking seven. Parent/candidate and every view of that scene reconstruct and verify identical mapped `[18,7,256]` float32 bytes.
- `[DECISION]` AP replay and N0-O subset diagnosis remain separate authorities. The 160-scene N0-O subset cannot be called a 6605-image AP result.
- `[FACT]` First published input manifest v1 had a wrong handwritten `dota_mAP` hex identity (`0.7049497863006894`). It is invalid regardless of its correct scene/support hashes and remains preserved only as an auditable failed attempt.
- `[DECISION]` Metric identities are generated by `float.hex()` and round-trip tested. Correct authority is published in a separate v2 directory; v1 is never overwritten or silently repaired.
