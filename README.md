# RISC

RISC 是一个独立、可审计的旋转语义干扰（Rotation-Induced Semantic
Confusion）研究工作区。它将 OpenRSD 作为强检测底座，将 OV-CapFlow 中的
RISC-ER 规格与实现证据作为可追溯参考；二者的历史 Git 工作树不会被修改。

## 从这里开始

- [RISC_GOAL.md](RISC_GOAL.md)：正式目标、研究边界、阶段门槛与交付标准。
- [docs/superpowers/specs/2026-08-22-risc-repository-migration-design.md](docs/superpowers/specs/2026-08-22-risc-repository-migration-design.md)：独立仓库与迁移设计。
- [docs/superpowers/plans/2026-08-22-risc-workspace-establishment.md](docs/superpowers/plans/2026-08-22-risc-workspace-establishment.md)：可逐项执行的建库计划。
- [docs/provenance/SOURCE_SNAPSHOT_MANIFEST.md](docs/provenance/SOURCE_SNAPSHOT_MANIFEST.md)：迁移后生成的来源、版本和范围清单。

## 目录边界

```text
framework/openrsd/    OpenRSD 框架代码快照（可运行底座）
reference/ov-capflow/ OV-CapFlow/RISC-ER 代码与文档参考快照
docs/                 目标、规格、计划、证据和来源记录
```

此仓库只跟踪源码、配置、测试、文本设计与文本实验记录。数据集、权重、检查点、
可视化、运行日志、缓存与二进制归档均不迁入。未来实验必须把数据根目录显式指向
`/data1/zcy/datasets`，不得依赖历史工作树的相对 `data/` 链接。
