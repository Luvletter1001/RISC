# OV-CapFlow ICLR Persistent Research Plan

## Goal

在不干扰现有合法任务、允许使用物理 GPU 0–9、保持 Q=600 且无 NMS/全局 top-k/dense inference mouth 的前提下，持续至少 18 小时完成“状态核验 → E24 all-query 根因诊断 → 单变量最小方法 → 严格 DOTA-v2.0 全量验证 → 开放词汇与论文证据闭环”，工程目标为完整 13,833 图验证集上的 mAP ≥ 0.7000 且 AP50 ≥ 0.7000；在达到同口径 mAP 门槛后补足可证实创新性与 ICLR 级完整证据。

## Current Phase

2026-08-03 continuation — 恢复现场、核验 GPU 0–9 与今日实验、选择下一条可证伪因果路线（in_progress）

## Global Guardrails

- 所有 shell 命令以 `rtk` 开头。
- 允许物理 GPU 0–9；按当前空闲资源分配，多卡命令设置显式 `CUDA_VISIBLE_DEVICES`、`NCCL_P2P_DISABLE=1`、`NCCL_IB_DISABLE=1`。
- 不打断任何现有合法任务，不杀无关进程，不覆盖 checkpoint，不复用冲突端口/tmux/work_dir，不做破坏性 Git 操作，不删除文件。
- 所有历史结论在原始配置、日志、checkpoint、评测产物复核前只视为“用户提供的待核验起点”。
- 所有方法实验一次只改变一个主要因果变量；改代码前保存基线行为，改后验证 parent equivalence、shape、gradient 和严格协议。
- 完整 raw 评测是最终裁判；proxy 只做筛选。
- 未经真实文献审计，不使用“首个”定论；全类文本提示训练不等同于 open-vocabulary generalization。
- 未完成至少 18 小时持续值守且未形成严格完整证据闭环时，不把持续目标标记为 complete；若同一硬阻塞连续三轮仍无法推进，再按系统规则报告 blocked。

## Phases

### Phase 0: Protect and Audit Current State

- [x] 完整读取用户提示及内联 AGENTS 约束
- [x] 建立持久计划、发现记录和进度日志
- [x] 检查仓库级 AGENTS、README、计划、总结、未来工作和 `.lab`
- [x] 检查 `resultmd`、`work_dirs`、日志、配置、队列、checkpoint 和磁盘
- [x] 检查 Git 状态并记录用户已有修改
- [x] 检查 GPU 2/3/8/9、tmux、训练进程、日志增长、端口与 T7/E24
- [x] 核验 OpenRSD 近两个月 rotation/DOTA2/false-small-vehicle 记录的实际位置和原始证据
- [x] 明确 E24 all-query dump 后三个最关键科学问题
- **Status:** completed

### Phase 1: E24 Strict All-Query Diagnosis

- [x] 确认 E24 checkpoint 完整且与配置/commit 对应
- [x] 审计全量 raw 推理协议和 dump 规模（期望 13,833 × 600 queries）
- [x] 生成主指标、per-class AP/recall、small-vehicle、base/novel（若 split 已定义）
- [x] 生成 query 分配、matched/unmatched、empty foreground、duplicate、score-IoU、reachability、same-label miss 诊断
- [ ] 生成 size/density/rotation/context 分桶
- [ ] 复核 P0148 false hub、P0682 true-SV-rich 及最小 embedding/prototype/score 干预
- [x] 写六审稿人记录并给出 CONTINUE/PAUSE/PIVOT/STOP
- **Status:** in_progress

### Phase 2: Select One Causal Route

- [ ] 根据 E24 证据在 semantic hub、geometry/ownership、ranking/calibration 中只选一条主路线
- [ ] 用用户可解释的方式确认中心假设、最小干预、预期机制指标和失败标准
- [ ] 完成相关最新文献/先验工作审计，验证 novelty claim 强度
- [ ] 设计 parent-preserving、query-identity-preserving 的最小实验
- **Status:** pending

### Phase 3: Minimal Verified Implementation

- [ ] 复用最近的现有实现模式并建立基线行为测试
- [ ] 先写失败测试，再做最小代码/配置修改
- [ ] parent equivalence、shape、gradient、strict protocol audit
- [ ] smoke test；必要时 HRSC 结构门控
- [ ] 启动唯一单变量 DOTA2 因果实验（tmux + 唯一 work_dir/port）
- [ ] 按 E1/E3/E6 生成审稿记录并执行停止规则
- **Status:** pending

### Phase 4: Promotion Experiments

- [ ] 完整 raw 评测达到晋升阈值
- [ ] 需要时延长训练并检查 E12/E18/E24
- [ ] 多 seed、严格 base/novel、absent prompt、泄漏审计
- [ ] rotation、empty-tile、true/false-SV、效率、消融与反例
- [ ] 对每项 claim 更新强度：观察/相关/干预/因果/多数据集
- **Status:** pending

### Phase 5: ICLR Evidence Package

- [ ] 完成时间线、配置、全部成功/失败结果和严格协议表
- [ ] 完成 Paper Storyline Ledger 与 claim-evidence 矩阵
- [ ] 判断是否达到 70+；未达到时量化差距和最有希望路线
- [ ] 形成用户可审阅的 Introduction/Method 大纲
- [ ] 形成 Figure 1/2/3 和主结果/消融/OV 表草案（只使用真实数据）
- [ ] 完成最终六审稿人预审与复现性审计
- **Status:** pending

## Three Critical E24 Questions

1. 当前 E24 的主要限制是否是背景/上下文 query 被 small-vehicle prototype 捕获，并且 P0148 干预能在不伤害 P0682 true-SV recall 的情况下因果改变该现象？
2. 若非 semantic hub 主导，tiny/dense 漏检主要来自几何不可达、同类语义已命中但 ownership 失败，还是 Q600 内部容量/匹配分配失衡？
3. raw all-query AP 损失中有多少来自 score-IoU/null/existence 排序校准，而不是输出 mouth、query 数或后处理；相应机制指标能否预测完整 raw AP 改善？

## Promotion and Stop Gates

- 晋升：完整 raw mAP/AP50 提升 ≥ 0.005；或提升 ≥ 0.003 且核心机制指标显著改善、主要类别退化不超过 0.001。
- 停止：完整 raw AP 下降 > 0.005，机制指标未改善，SV recall/novel 崩溃，empty foreground/duplicate 恶化，依赖禁用 mouth，或仅单图/proxy 有效。
- 连续两次实验不能区分假设时停止当前配方并回到 Phase 2。

## Decisions Made

| Decision | Rationale |
|---|---|
| 先只读保护性审计，不立即启动训练 | 用户明确要求先确认 T7/E24 与真实资源状态 |
| 历史事实单独标注为待核验 | 避免 OpenRSD 机制未经验证迁移到当前 OV-CapFlow |
| 使用文件化计划和证据账本 | 支持 72 小时恢复、持续监控和可审计结论 |
| 多卡默认禁用 NCCL P2P/IB | 用户硬约束与 A40 通信技能均要求 |
| Phase 0 完成后不启动候选，等待 gated E24 canonical dump | E18无同路线dump；现有scale1280只可做preflight，不能选因果路线 |
| 文献 novelty 暂不写“first” | PaQ-DETR已正式CVPR2026，HSA-DINO/ViTPrompt/VK-Det等构成强近邻 |
| P0148 hub 必须绑定 P0682 true-SV 安全门 | 历史orbit抑制P0148时同时明显压低P0682 SV ratio，且无GT recall证据 |

## Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| 尝试重复创建 `/goal`，系统报告已有 active goal | 1 | 保留现有目标并用 `get_goal` 读取，不重置进度 |
| 根目录未找到物理 `AGENTS.md` | 1 | 采用用户消息中的内联 AGENTS 约束；继续检查隐藏目录内的项目规则 |
| 沙箱 `pgrep` 未显示宿主 GPU 进程命令行 | 1 | 不把它解释为任务消失；改查 tmux pane 元数据、capture-pane 和日志 |
| tmux list-panes/capture-pane 首次被沙箱 socket 权限拒绝 | 1 | 只读提权后成功；未变更会话状态 |
| `.lab/results.tsv` 整段读取因体量过大被截断 | 1 | 改查总行数、最新尾部和目标 run ID，避免重复大读 |
| Orbit CSV首次聚合引用不存在的`sv_ratio`列 | 1 | 读取header后改用`final_sv_ratio`并成功复算 |
| `rtk conda env list` 未解析到 `conda` | 1 | 不使用系统 Python；下一步从已知 Anaconda 根目录定位项目环境并用绝对解释器 |
| `rtk find` 不支持 GNU `-newermt/-printf` | 1 | 不重复该命令；改用 `/usr/bin/find` 绝对路径或现有 ledger 的时间戳筛选 |
| OpenRSD 今日文件扫描把 `.` 与 `data_local` 纳入，输出过大被截断 | 1 | 不重复广域扫描；后续只读明确的 `resultmd/exp_cser_phase1_20260803`、spec/plan/config/log 路径 |
| ODQ supervisor 计划文件名首次少了 `ov-capflow-` 前缀，且与 `&&` 组合导致后续搜索未执行 | 1 | 用 `rg --files docs` 先解析真实路径，再做聚焦读取 |
| `rtk find` 不支持 GNU `-print`，`rtk jq` 在当前 wrapper/环境不可用 | 1 | 不重复这些调用；用 `rtk rg --files` 与项目解释器只读 JSON |
| Swin-B 可构建性首轮脚本读取不存在的 `SwinTransformer.embed_dims` 运行时属性 | 1 | 去掉展示属性，只从冻结 config 读取结构并统计模型参数；成功构建 Swin-T/B，参数量分别 172.845M/232.909M |
| `rtk test ! -e ...` 被 wrapper 错误解析为 `sh: -e: not found`，未执行预期存在性断言 | 1 | 启动前已用 `rtk ls` 明确确认目标不存在；下载只写新 `.part`。后续存在性检查改用 `rtk ls`/项目 Python，不重复 `rtk test` |

## Reboot Rule

重大决策前重读本文件和 `findings.md`；每完成两次搜索/浏览就写入发现；每个阶段、异常、测试和运行状态都更新 `progress.md`。

## 2026-08-03 — Swin-B B0 formal overnight state

- [x] Final-config single-GPU extreme cases and ten-rank three-update DDP smoke pass after the single-field `static_graph=True` runtime amendment.
- [x] Formal r3 launched on physical GPUs 0--9 with immutable launch receipt, exact epoch0 sampler audit, and an independent 60-second process/GPU/fatal monitor.
- [x] Repair and independently re-review the automatic gate supervisor before enabling stop actions; commit `6fcd0982` passed 45 fresh tests and an independent PASS review, then exact-bound enforce watch launched without touching the training recipe.
- [x] Reach E1 and apply the frozen gate: exact raw mAP `0.44958066940307617`, rounded AP50 `0.4500`, complete 13,833-image/Q600 validation, checkpoint/mouth/sampler/decision hashes, status PASS with `+0.0495806694` margin over `0.4000`.
- [x] E1 passed and the exact-bound supervisor continued the unchanged formal run without sending Ctrl-C; E2 was observed live at iter60 with monitor fatal-null.
- [ ] Reach E6 and require both exact mAP `>=0.5800` and delta `>=+0.0200` over T7 E6 exact `0.5536741614341736`; preserve all artifacts and stop only the exactly bound pane if either conjunct fails.
- [ ] Keep conclusion language in three layers: current closed-set system evidence, conditional B0 anchor, and separately initialized future negative-safe strict-OV evidence.
- [x] Freeze and execute the annotation-blind scene-only M0 inventory: 47,294 train tiles / 1,664 scenes split by the one-shot frozen hash into 35,109/5,172/7,013 train/dev/test tiles with zero declared scene/stem/crop overlap; keep `training_use_forbidden=true`.
- [ ] Run M1 PNG-header and streaming content hashes only after B0 releases the shared 65 GB train-image path; M1 still does not authorize N0/N1.

## 2026-08-01 — Holistic QAF source preflight freeze

- The holistic Query Allocation Quality Flow design is frozen; training is restricted to physical GPUs 8 and 9.
- D13-N closed at the registered E12 endpoint: 0.838305831 candidate versus 0.847486973 control, delta -0.009181142.
- Active hypothesis: one query-to-image evidence coupling jointly governs center transport, null validity, matching/training quality, and all-row score calibration.
- Current authorized action: frozen E24 stride-32 source preflight only.
- Hard stop: no production model/config code and no distributed training before all five source gates pass.

## 2026-08-01 — Source-gate decision and branch closure

- [x] Frozen source audit completed with status `FAIL`.
- [x] Immutable report recorded at `.lab/workspace/exp-8-qaf-source-v1/source_report.json`, external SHA256 `d51f88a677af48ddbdc0455f4300452bcaf51e44bbddec7c22176555b034ca33`.
- [x] Exact reachabilities recorded: overall `0.07178882557149407`; base14 `0.0717736256898161`; novel4 `1.0` (`total=1`, faithfully reported); spatial shuffle `0.07262396017554201`; semantic shuffle `0.07183795113643807`.
- [x] Failed gates: `reachability_at_least_25pct`, `beats_prior_e_by_3pp`, `placebo_sensitivity`. Passed gates: `finite_reproducible_provenance`, `novel_base_gap_at_most_10pp`.
- [x] Primary and placebo results are effectively indistinguishable and the primary is far below `0.25` and prior E plus 3 percentage points; Approach A / Query Allocation Quality Flow is closed for this source family.
- [x] Plan 2 will not execute. No production module/config, proxy/raw/full training, scale rescue, or alternate evidence level/threshold/seed/subset is authorized.
- [x] The training resource rule remains physical GPUs 8 and 9 only; this branch launches no training.

## 2026-08-12 — Three-month retrospective and derivative detector study

- [x] Inventory OpenRSD and OV-CapFlow evidence from 2026-05-12 through 2026-08-12.
- [x] Reconstruct experiment families, comparable metrics, failures, pivots, and the OpenRSD → OV-CapFlow transition.
- [x] Separate verified results from plans, live runs, protocol failures, and non-comparable evaluations.
- [x] Extract reusable research and engineering lessons.
- [x] Ground a derivative remote-sensing oriented-detection framework against current primary literature.
- [x] Compare full-, point-, and horizontal-box-supervised routes and recommend a staged design plus falsifiable experiment matrix.
- **Status:** completed (rolling-review snapshot; ongoing experiments remain active research rather than part of this task)
- **Scope:** read-only audit and research design; do not modify training code, interrupt runs, or launch experiments.

## 2026-08-12 — CNN-HBox detector proposal v2

- [x] Reframe the derivative paper around a CNN dense detector rather than a fixed-query architecture.
- [x] Preserve fixed-query lessons as mappings to FPN reachability, location-instance ownership, competition, null/background calibration, and negative safety.
- [x] Audit the local H2RBox-v2 target path and Rotated RTMDet dynamic-assigner assets.
- [x] Compare the revised gap against H2RBox/v2, EIE-Det, BGHR, ABBSPO, Wholly-WOOD, PWOOD/SPWOOD, and dense-assignment literature.
- [x] Specify OrbitAssign, matched ablations, mechanism metrics, dataset order, promotion gates, failure modes, and claim boundaries.
- [x] Produce a traceable 18-paper focused literature package from 26 screened candidates.
- **Status:** completed (design-only; implementation and training intentionally await user approval)
- **Scope:** no training code/config/data/checkpoint/process/GPU changes.

## 2026-08-12 — OrbitAssign v3 FMSA fusion

- [x] Read the complete FMSA-Point assisted draft and preserve it as private user material.
- [x] Audit each FMSA mechanism for direct reuse, adaptation, deferral, or rejection under the CNN-HBox scope.
- [x] Verify MessDet, Fourier Angle Alignment, PointOBB-v3, Point2RBox-v2/v3, and the TIP 2025 HBox-orientation near neighbor through public traceable sources.
- [x] Correct the group-size, Fourier-domain, axial-periodicity, scalar-quality, density, boundary, and teacher–student assumptions that cannot be copied as facts.
- [x] Upgrade OrbitAssign from single-entropy ambiguity handling to ownership/geometry/background factorized evidence routing.
- [x] Specify conflict-local transport, second-order axial concentration, matched controls, diagnostics, gates, failure modes, and claim boundaries.
- [x] Preserve v2 and create a separate v3 plus an incremental literature addendum.
- **Status:** completed (design-only; implementation/training remain unapproved)
- **Scope:** no code/config/data/checkpoint/process/tmux/GPU changes.
