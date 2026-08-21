# RISC 正式目标与工作总纲

**状态：** 2026-08-22 已建立独立工作区；源码快照与首个可审计提交正在执行。
**研究主线：** 用强检测底座与严格因果对照，研究并缓解旋转导致、且超过普通重复噪声的对象级语义风险。
**目标投稿：** ICLR 2027；任何论文主张均以后续证据门通过为前提，而非既成结论。

---

## 1. 北极星目标

在不改变检测几何路径和单视图输出口径的前提下，建立并检验 RISC-ER：

1. 把旋转诱发的语义干扰定义为相对于 identity-repeat 对照的 **rotation excess semantic risk**；
2. 在 `image rotation × query permutation` 的商空间中，按 GT object identity 而不是固定 query index 对齐；
3. 学习一个有界、轨道零均值、仅作用于分类语义 readout 的低秩残差；
4. 用与候选完全匹配的 control 证明机制改善和 AP 改善，且不把底座增益误归给 RISC。

本仓库必须同时成为：

- 可继续开发的 OpenRSD 代码底座；
- 可复查的 OV-CapFlow/RISC-ER 规格、实现和实验档案；
- 不含训练产物的轻量 Git 历史；
- 未来每个结论可回溯到 config、checkpoint、评测口径、原始产物位置和哈希的证据账本。

## 2. 已冻结的研究边界

### 2.1 方法边界

- **B\*** 是强检测底座，负责通用遥感检测能力；它不是 RISC 的论文贡献。
- **RISC-ER** 是唯一方法贡献：identity-controlled excess risk、对象身份对齐、轨道零均值的低秩语义残差。
- 辅助风险只更新 quotient 参数；bbox、reference point、objectness/quality 路径与共享视觉路径不接收该辅助梯度。
- 推理仅使用单图，不需要配对视图、GT、Hungarian matching 或 contender 构造。
- 禁止把全局 OT、decoder 后 top-k/NMS、重写 bbox regression 或普通数据增强包装成 RISC 创新。

### 2.2 禁止的论文表述

- 不声称 RISC 解决通用密集目标覆盖或 Q600 的全部容量问题。
- 不把 all-18 prompt 训练结果称为严格 open-vocabulary 泛化。
- 不将无权威标注的 P0148 预测称为错误或正确的定量证据。
- 不将不同 output mouth、不同 prompt、不同父权重或不同训练资源的比较写成因果增益。
- 在更新投稿日前的系统文献检索完成前，不使用“first/首次”。

## 3. 当前已知事实、推断与未知量

| 类型 | 内容 | 来源 |
|---|---|---|
| FACT | 当前 E12 在 filtered-6605、scale-1024 口径为 `dota/mAP=0.6340868473`、`AP50=0.6340`。 | OV-CapFlow RISC-ER 正式规格 §2 |
| FACT | 同 mouth 的本地 OpenRSD 复现约 `0.7049593925/0.7050`，差距约 7.10 AP 点。 | 同上 |
| FACT | scene-disjoint N0-RI 在 4,147 个几何稳定身份上观察到 rotation-induced class-flip excess `0.157421`，bootstrap 区间 `[0.130640, 0.184247]`。 | 同上 |
| FACT | 当前 E12 尚没有同 checkpoint、全查询 dump，因此 geometry、semantic、duplicate、ranking 和 Q-capacity headroom 尚未归因。 | 同上 |
| INFERENCE | 与 OpenRSD 的差距可能由父模型、特征分辨率、训练数据/时长、prompt、排序和固定 Q600 共同造成，而非单一 RISC 缺陷。 | 同上 |
| OPEN | POQ 相对 matched no-POQ control 的真实增量、E12 oracle headroom、强遥感父权重的数据 provenance、严格 held-out-class 结果。 | 同上 |

所有新陈述必须标为 `[FACT]`、`[INFERENCE]`、`[OPEN]`、`[DECISION]` 或 `[NON-CLAIM]`，并链接实际证据位置。

## 4. 仓库结构和来源契约

```text
RISC/
├── RISC_GOAL.md                         # 本文件：最高目标与研究门槛
├── framework/openrsd/                   # OpenRSD 当前源码快照，不含运行产物
├── reference/ov-capflow/                # OV-CapFlow/RISC-ER 源码与文档快照
├── docs/provenance/                     # 来源提交、远端、范围、哈希与排除规则
├── docs/superpowers/specs/              # 经确认的仓库/方法设计
├── docs/superpowers/plans/              # 可执行的逐项计划
├── task_plan.md                         # 当前任务阶段状态
├── findings.md                          # 事实、决策和问题台账
└── progress.md                          # 时间顺序的执行与验证记录
```

迁移的固定规则：

1. **OpenRSD** 使用 `/data1/zcy/OpenRSD` 的当前代码工作树，记录基线提交 `12d3fd8b75e8b64ec53fded9cf035a2306d58874` 及其远端；受控复制未提交的源码、配置、测试和 Markdown，以保留代码性成果。
2. **OV-CapFlow** 使用 `/data1/zcy/OV-CapFlow` 的当前代码工作树，记录基线提交 `e87ae43ad294d9918cd6a36a458bf6f03dabb9fe` 及其远端；保留 RISC-ER 方法规格、实施计划和相关代码参考。
3. 不复制 `.git`、数据、权重、预训练文件、checkpoint、预测 dump、可视化、日志、缓存、PDF、压缩包或二进制模型。精确排除规则见 `docs/provenance/MIGRATION_POLICY.md`。
4. 历史源码不被修改；新仓库的提交历史独立，任何后续可运行变更只在本仓库产生。

## 5. 研究实施路线与硬门

| 阶段 | 目标 | 必交证据 | 晋级条件 | 失败时的动作 |
|---|---|---|---|---|
| 0 | 建立可审计代码底座 | Git 初始提交、来源清单、排除扫描、关键文件哈希 | 两个源码快照存在；Git 不跟踪禁止产物 | 修复迁移规则，不开始训练 |
| 1 | 当前 E12 诊断 | raw-13833 Q600 dump、验证器、oracle 分解、mouth/config/checkpoint SHA | 13,833 条记录且每图 600 行；单一官方 metric row 可复现 | 产物、口径或权威性失败即 `INVALID_NO_DECISION` |
| 2 | 作出单一 B\* 决策 | E12 诊断报告和一条 fail-closed 决策 | 仅在 `RISC_ER_ONLY_ELIGIBLE`、`BUILD_BSTAR_GEOMETRY_FIRST`、`BUILD_BSTAR_RANKING_FIRST` 三者之一间选择 | 不同时设计 geometry 与 ranking 改动 |
| 3 | 实现 RISC-ER 与 matched control | unit tests、load audit、step-0 outputs、全行 Q600 mouth audit | zero-init candidate 与 parent/control 在 step 0 完全一致；仅预定参数不同 | 修复实现或对照，不启动完整训练 |
| 4 | DDP smoke 和弱底座因果实验 | 8-GPU 两迭代 smoke、控制/候选等资源记录、机制指标 | ER 有效有限、对象覆盖正、geometry 不降、small-vehicle recall 不实质坍塌 | 分类为 objective/support/optimization/substrate mismatch 并封存端点 |
| 5 | 全端点与多 seed | 同 mouth AP、scene-macro flip/margin/hub excess、效率、三 seed | 单 seed 相对 matched B0 至少 `+0.3` AP 且机制方向一致，才进入三 seed | `+0.3` 以下且无机制支持，关闭该配方 |
| 6 | 强底座 2×2 与 OV 证据 | B0-strong/B1-strong、provenance、strict held-out folds、90°/arbitrary controls | B\* 对照和 RISC 版本除方法开关外完全对称 | B\* 增益单列，不归因给 RISC |
| 7 | 论文证据包 | claim-evidence matrix、负实验台账、配置/checkpoint/eval 哈希、效率报告 | 每项主张均有最小证据；全部 non-claim 合规 | 删除或降级无证据表述 |

## 6. 阶段 1：E12 诊断的不可跳过流程

1. 固定 E12 checkpoint 和 resolved config，写入 SHA256。
2. 仅在确认 GPU 空闲后，导出 raw-13833 的全 Q600 预测；不得用旧 E24 分解替代。
3. 验证 record 数、每图 query 数、类数和唯一 metric 记录。
4. 分解 fixed-box perfect-ranking、GT geometry/semantic/ownership、AP-support FP、per-class、size/density 和 Q-capacity headroom。
5. 从以下互斥决策中写入唯一一项：
   - `RISC_ER_ONLY_ELIGIBLE`
   - `BUILD_BSTAR_GEOMETRY_FIRST`
   - `BUILD_BSTAR_RANKING_FIRST`
   - `INVALID_NO_DECISION`

`BUILD_BSTAR_GEOMETRY_FIRST` 需要四个主要缺口类中至少三个的 geometry miss 为最大 GT-miss 状态，且 tiny/dense 分层同向、fixed-box perfect-ranking headroom `<1.5` AP；`BUILD_BSTAR_RANKING_FIRST` 需要 fixed-box perfect-ranking oracle 至少收回 `1.5` AP。阈值是预注册工程门，不是统计置信界。

## 7. 阶段 3：RISC-ER 的冻结训练协议

每个 orbit update 的三视图角色固定为 `x^{0a}`、`x^{0b}`、`x^θ`：前两个是同像素 identity 深拷贝，只提供 no-grad baseline 和重复噪声；非零旋转视图同时承担标准检测损失与 quotient-only ER 梯度。每个视图独立 Hungarian assignment，再通过写入变换前的 `orbit_instance_ids` 按 GT 身份对齐。

对象风险只由 owner 的错误类能量和几何相关 contender 能量构成；IoU 归属、bbox 与文本 readout 的输入均 stop-gradient。唯一辅助目标为：

\[
\mathcal L_{ER}=\frac{1}{|\mathcal J|}\sum_{j\in\mathcal J}
\left[[R_j^θ-R_j^{0a}]_+-\operatorname{sg}([R_j^{0b}-R_j^{0a}]_+)\right]_+.
\]

低秩 quotient 用二阶/四阶 \(\pi\)-周期谐波、12 点离散群积分、`max_gate` 和 `max_delta_norm_ratio` 上限；零初始化必须令 `z_stable == z` 逐元素成立。candidate/control 必须匹配数据、seed、batch、累计步数、LR、prompt、orbit 采样、检测 loss、评测 mouth、checkpoint 规则和硬件资源。

## 8. 必做消融、指标和报告结构

### 必做对照

- matched orbit control；
- quotient without ER；
- 不扣 identity-repeat 的 raw consistency；
- without contender energy；
- without orbit centering；
- lossless 90° 与 arbitrary-angle 分报；
- label-permuted orbit placebo；
- strict held-out-class folds（仅在有结果后用于 OV 主张）。

### 指标

- 主任务：filtered-6605 与 raw-13833 的 AP50/mAP、per-class AP、small-vehicle AP；
- 机制：scene-macro class-flip excess、true-vs-best-wrong margin drift excess、JS excess、false-hub rate、contender excess、geometry retention；
- 安全：true small-vehicle recall、tiny `<=8 px`、density `>600`、empty-tile foreground、novel/held-out AP；
- 成本：训练 wall time、峰值显存、单视图推理延迟。

所有表格永久区分 filtered-6605 与 raw-13833、canonical 与 text7、control 与 candidate、B\* 与 RISC-ER。每个数值必须关联 config、checkpoint、评测产物、命令和 SHA。

## 9. 日常工作规则

1. 每项实现先在 `docs/superpowers/specs/` 固化边界，再在 `docs/superpowers/plans/` 写成逐项计划。
2. 每次开始、阶段转换、训练启动/停止、或结论改变时，更新 `task_plan.md`、`findings.md` 与 `progress.md`。
3. 实验输出写入仓库外的约定路径，仓库中只提交文本 manifest、命令、哈希和摘要；禁止将 checkpoint/数据/图片加入 Git。
4. 每项代码修改须有聚焦测试；完整训练前必须过 CPU/constructor/one-batch/step-0/DDP smoke 的递进验证。
5. 后续工作在特性分支完成，并以小而可回滚的提交记录；不重写本仓库的初始来源提交。

## 10. 本次建库完成定义

本次任务只有在以下事实均由当次命令验证后才算完成：

- `/data1/zcy/RISC` 是独立 Git 仓库，初始分支为 `main`；
- `framework/openrsd/` 与 `reference/ov-capflow/` 各自含 README、源码、配置和测试参考；
- `docs/provenance/SOURCE_SNAPSHOT_MANIFEST.md` 记录两个来源的绝对路径、基线 SHA、远端、状态概览、迁移时间和文件计数；
- `git ls-files` 不包含禁止的模型、数据、压缩包、PDF、图像、cache 或运行目录；
- 目标、设计、逐项计划、发现和进度文件均已提交；
- 源仓库 Git 状态除 OpenRSD 的工作日志追加外没有被本次迁移改变。
