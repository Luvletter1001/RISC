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

## Technical Decisions

| Decision | Rationale |
|---|---|
| 以当前工作树的 source-only 快照迁移 | 已提交基线不足以保留本地代码成果；完整树会混入不可审计的运行产物。 |
| 采用 `rsync -a` 且不使用 `--delete` | 不修改源树，也不让重复迁移删除目标中的资料。 |
| 初始来源和文档分两次提交 | 让研究 authority 与源码快照可分别审查和追责。 |
| 使用目标根 `RISC_GOAL.md` 作为最高目标 | 用户要求最详细的书面目标文件，需要比历史具体实施计划更上层的边界和门槛。 |

## Issues Encountered

| Issue | Resolution |
|---|---|
| 直接复制会带入 OV-CapFlow 的约 402G `work_dirs`、13G `pretrained` 等运行资产 | 用显式过滤政策排除。 |
| 两个来源都是 dirty 工作树 | 将 HEAD、远端和 dirty 摘要写入来源 manifest，不把快照伪装为上游发布版本。 |
| 占位词扫描会命中自身的检索表达式 | 通过运行时拼接检索模式，仍然扫描全部政策与计划文件。 |
| RTK 的 `git diff --cached --check` 包装未给出可用状态 | 对 staged whitespace verification 使用 `rtk run` 执行原生 Git；仍保留 RTK 作为命令前缀。 |
| 初稿用 Markdown 两个空格表示硬换行，触发 Git whitespace check | 正式目标改用普通换行，避免尾随空格。 |

## Resources

- `RISC_GOAL.md`
- `docs/provenance/MIGRATION_POLICY.md`
- `docs/superpowers/specs/2026-08-22-risc-repository-migration-design.md`
- `reference/ov-capflow/docs/superpowers/specs/2026-08-11-risc-er-formal-design.md`（快照完成后）

## Visual/Browser Findings

本任务未使用浏览器或视觉材料。
