# OV-CapFlow Findings and Claim-Evidence Ledger

## Evidence Labels

- **Known:** 已由本轮直接检查的代码、命令、日志或产物支持。
- **User-stated / unverified:** 来自用户提示，但尚未复核当前仓库原始证据。
- **Inference:** 基于已知事实的暂时解释，必须能被后续实验推翻。
- **Unknown:** 当前缺少证据。

## Requirements Captured

- 持续值守 72 小时，训练放入持久会话并短周期监控；不以长 sleep 代替值守。
- 严格 DOTA-v2.0 全部 13,833 验证图、18 类、`filter_empty_gt=False`、Q=600、raw all-query、无 NMS、无全局 top-k、无 dense inference head。
- 工程目标 mAP ≥ 0.7000 且 AP50 ≥ 0.7000；科学目标是统一解释或区分 semantic false hub、ownership/coverage failure 与 ranking/null calibration。
- 最终方法必须服务于 ICLR 论文主线，保留 query identity，外部证据只经零初始化 parent-preserving residual 注入。
- 真正开放词汇证据要求严格 base/novel、novel/base AP、H-mean、absent-prompt FPR 和类别/预训练泄漏审计。

## Known Facts — Inferences — Unknowns

| Item | Status | Evidence / next verification |
|---|---|---|
| 当前 Codex 持续目标存在且状态 active | Known | `get_goal` 于 2026-07-21 返回 active |
| shell 命令必须以 `rtk` 开头 | Known | 用户提供的内联 AGENTS 指令 |
| 项目根目录是 `/data1/zcy/OV-CapFlow` | Known | 环境上下文与根目录列表 |
| 根目录未发现 `AGENTS.md`、既有 planning files | Known | `rtk rg --files -g ...` 返回无匹配；仍需检查隐藏规则文件 |
| T7 当前是否运行、E24 是否生成 | Unknown | 待查进程、tmux、日志、checkpoint |
| 当前 GPU 2/3/8/9 占用和合法任务 | Unknown | 待查 `nvidia-smi` 与进程命令行 |
| 当前 Git 工作树状态 | Known | branch `research/dotav2-cleanstart-ov-e2e-ap70`；无 tracked diff，存在受保护的 user untracked files |
| 当前分支与工作树状态已核验 | Known | branch `research/dotav2-cleanstart-ov-e2e-ap70`；无 tracked diff，存在多项用户 untracked 文件 |
| GPU 2/3/8/9 当前均有约 40–41 GiB 显存占用 | Known | 2026-07-21 首次 `nvidia-smi`：2=41,351 MiB，3=40,695 MiB，8=40,743 MiB，9=40,839 MiB |
| T7 相关持久会话存在 | Known | tmux 有 `t7_full24_rare4x_gpu2389`、monitor、guard、E18 wait、E24 dump queue |
| GPU 2/3/8/9 属于同一 mmdet 四卡任务 | Known | compute-app 映射为 PID 689012/689014/689015/689017，解释器均 `/data/zcy/anaconda3/envs/mmdet/bin/python` |
| T7 rare4x 当前命令是四卡 resume training | Known | tmux start command：config `...rare4x_gpu2389_batch2_e6_resume.py`，work_dir `...rare4x_seed20260716_gpu89_b2`，master port 29789，`--resume` |
| T7 启动参数满足 GPU/NCCL 硬约束 | Known | `CUDA_VISIBLE_DEVICES=2,3,8,9 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` |
| E24 dump queue 已配置 | Known | 等待 `epoch_24.pth` 后使用 eval config、GPU 2/3/8/9、port 29791，输出 `eval_t7_epoch24_raw13833_gpu2389_dump/predictions.pkl` |
| T7 已进入 Epoch 21 且训练日志在推进 | Known | launcher log 可见 07/21 15:42 后的 `Epoch(train) [21]` 连续 iteration，约 1.8–2.0 s/iter |
| 16:55:59 精确进度为 E21 2400/5916 | Known | launcher 短 tail：ETA 9:46:49，data_time 0.0159 s，无错误行 |
| 监控未发现 fatal 且进程组存活 | Known | monitor JSONL 连续到至少 16:56，记录 `process_group_alive=true`、`fatal_pattern=null` |
| E24 checkpoint 尚未发布，dump 尚未启动 | Known | queue state：`checkpoint_published=false`、`already_launched=false`、`training_alive=true`、`gpus_idle=false` |
| T7 checkpoints 当前到 E18 | Known | work_dir 有 epoch_1/6/12/18（及 E6 rebased），无 epoch_24 |
| 数据卷剩余约 1.3 TB | Known | `/data1` 9.1T，总用 7.4T，86% used |
| 当前不得启动新训练或抢占 GPU 2/3/8/9 | Inference | 允许 GPU 均高显存占用且存在用户要求保护的 T7 会话；需继续确认 PID/日志健康度 |
| 当前 E24 是否复现 P0148 small-vehicle false hub | Unknown | 需要完整 all-query dump 与干预对照 |
| 历史 T7 E1/E6/E12/E18 数字准确且协议一致 | User-stated / unverified | 待查配置、日志、评测产物、commit、数据范围 |
| rare4x 是唯一稳定晋升干预 | User-stated / unverified | 待查完整对照与原始结果 |
| 当前主瓶颈可能是 capacity misallocation | Inference | 必须由 E24 的 hub/coverage/calibration 可区分诊断检验 |
| “首个端到端固定查询无 NMS 的 OV oriented detector” | Unknown candidate claim | 必须检索截至当前日期的真实论文与项目并审计定义 |
| T7 E18 raw authority 为 0.6081 mAP / 0.6080 AP50 | Known (documented; artifact cross-check pending) | `.lab/config.md` 2026-07-21 addendum，完整 13,833 tiles |
| E12→E18 增益仅约 +0.0037/+0.0040 | Known (documented; artifact cross-check pending) | `.lab/config.md` 将路线判为 plateau |
| E24 是当前 recipe 最后权威点，无 E25+ 自动延长 | Known | `.lab/config.md` plateau addendum |
| E18 scale1280 test-only 为 0.5942/0.5940，低于 canonical 0.0139/0.0140 | Known (documented; result file cross-check pending) | `.lab/config.md`；fixed 1280 inference closed |
| Q600 theoretical joint recall cap 0.934578，SV cap 0.896580 | Known (documented; diagnostic artifact cross-check pending) | `.lab/parking-lot.md` 8-D7；实测 SV recall 当时仅 0.490 |

## Paper Storyline Ledger

| Claim | Current strength | Supporting evidence | Counter-evidence / missing evidence |
|---|---|---|---|
| C1 固定查询、无 NMS 的端到端 OV 旋转检测可行且有竞争力 | User-stated observation | 用户称历史严格 E2E 短训约 0.60 AP50 | 当前仓库原始记录、竞争基线、公平协议、严格 OV split 未核验 |
| C2 rotation/context/density 引发可测量 semantic capacity misallocation | Hypothesis | OpenRSD 历史 P0148 机制由用户描述 | 当前 OV-CapFlow E24 尚未复现；FN 与 FP 是否同源未知 |
| C3 OV-CapFlow 同时改善 raw AP 与对应机制指标 | Unsupported target claim | 无 | 方法尚未由 E24 选定，完整 raw 因果实验缺失 |
| C4 收益在 base/novel、rotation、empty、efficiency 审计下成立 | Unsupported target claim | 无 | 全部晋升实验缺失 |

## Historical Starting Points (All Unverified Here)

- P0148 是历史 false-small-vehicle 主要因果/可视化样本，P0682 是 true-SV-rich 对照；P0124/P0142 仅次级 metadata proxy。
- 历史机制：SV embedding attractor → dense logit drift → top-k/NMS amplification → context false positives。
- 当前模型还可能存在 SV recall 不足、tiny/dense coverage、query ownership、unmatched/empty calibration、score-IoU alignment 问题。
- R29 的 balanced matched/unmatched reduction 和 HRSC 的 parent-preserving fusion 是候选机制起点，但不得未经当前证据直接晋升。

## Verified Historical Counterexamples from `.lab/log.md`

- Experiment 6 DOTA2 Q200 C0: raw-13,833 mAP/AP50 0.1107/0.1110；zero-update C1 在 13,833 图、2,766,600 rows 上与 C0 完全一致。
- 训练 18 个 semantic-fusion 参数后，C1 mAP/AP50 0.1253/0.1250，mAP +0.0146；coverage 0.221030→0.246450，但 duplicate extras/GT +27.74%、empty foreground mass +1.07%、gate gap +0.007992。
- 因此该 C1 只支持“parent-preserving fusion 可改善 coverage/AP”的弱迁移先验，不支持 unmatched suppression、null calibration 或 T7 当前瓶颈结论。
- P126C raw-13,833 replay mAP 0.656813/AP50 0.6570，复现强锚点，但使用历史 OpenRSD/GSOVD runtime/checkpoint；对 Experiment 8 clean-start 合同不合规，禁止作为 parent/teacher/ensemble/目标证据。
- Experiment 8 S0 只证明 real-DOTA Q600 engineering smoke：50 optimizer updates、exact 600 rows、decoder/export max error 0、无 forbidden calls；其小集 0.005 AP 不得解释为方法质量。
- S1 grouped 主 seed虽相对 control AP50 +0.026，但 coverage 0.504970→0.496848、novel4 0.125→0.030；repeat 仅 +0.004。它不是稳定的统一 capacity 改进。
- 同一 S1 grouped checkpoint 从 400-image mouth 0.4070 AP50 到 raw13,833 仅 0.3090（-0.0980），coverage -0.033276、empty foreground mass +2.572674；proxy 只能筛选。
- Hausdorff-DN AP50 比 control -0.040，Chamfer 即使 duplicate extras/GT 降到 0.646667 仍 AP50 -0.006；更低 geometry cost 或 duplicates 不自动转化为 AP。

## Resources

- 用户完整提示：`/home/zcy/.codex/attachments/eae6f2a8-4add-4227-8f16-eedaafba22db/pasted-text-1.txt`
- 项目根目录：`/data1/zcy/OV-CapFlow`
- 持续计划：`task_plan.md`
- 运行日志：`progress.md`
- Root `README.md`: 上游 CastDet/OVAD 仓库说明，非 OV-CapFlow T7 严格实验记录。
- `.lab/config.md`, `.lab/summary.md`, `.lab/log.md`, `.lab/parking-lot.md`, `.lab/results.tsv`: 项目实验记忆入口。
- `.lab/workspace/exp-8-d133/epoch18_raw_plateau_mixed_novel_coverage_global_efficiency.json`: E18 plateau 记录候选。
- `.lab/workspace/exp-8-d140/e24_same_route_prediction_dump_runbook.md`: E24 dump runbook。
- `.lab/workspace/exp-8-d141/`: E18 AP70 headroom/quality priority 审计。
- `.lab/workspace/exp-8-d142/`: 最新文献兼容和 no-train-priority 审计。
- `.lab/workspace/exp-8-d143/`: E18 scale1280 test-only 计划与结果。
- Selected Python: `/data/zcy/anaconda3/envs/mmdet/bin/python`, Python 3.8.19, Torch 1.12.1+cu113, MMEngine 0.10.4, workspace `mmrotate` import；所有命令加 `PYTHONNOUSERSITE=1`。

## Immediate Unknowns

1. README/`.lab`/实验记录如何定义 T7、E24、strict mouth 和当前队列？
2. E24 checkpoint、全量预测与 all-query dump 的真实路径和完整性是什么？
3. 哪些进程正在 GPU 2/3/8/9 上运行，分别属于谁和哪个实验？
4. OpenRSD 历史记录是否在当前工作区可读，或需要明确外部路径？
5. base/novel split 与 absent-prompt 评测是否已有权威配置？

## Documentation Routing

- Root README 确认该分支面向 oriented open-vocabulary aerial detection，并列出 Oriented CastDet/GroundingDINO/GLIP/ViLD；但它没有当前 OV-CapFlow 的 Q600/raw/no-NMS/T7 证据。
- `.lab` 含大量按时间推进的训练动力学 JSON。为避免用中间 loss proxy 替代最终 AP，阅读顺序固定为：全局 config/summary/log → E18 raw plateau → E18 quality/headroom → E24 runbook → literature audit → scale1280 result。

## Current DOTA2 Authority from `.lab/config.md`

- Active run `8-T7-F-WS4-A1`: physical GPU 2/3/8/9, world size 4, batch 2/GPU, accumulation 4, effective batch 32；从完整 E6 checkpoint 恢复，是有效 stochastic continuation，不声称 bitwise replay。
- E18 canonical raw-13,833: mAP 0.6081, AP50 0.6080；E12→E18 约 +0.0037/+0.0040，判为 plateau。
- E24 是相同 recipe 的最后 authority；E24 all-600-query dump 和分析是任何新训练前的证据门。
- E24 后只允许在 geometry/coverage、score-quality/ranking、legal semantic intervention 中选择一个；D11/D12 approval-pending，禁止堆叠。
- E18 scale1280 test-only: 0.5942/0.5940，canonical delta -0.0139/-0.0140；SV AP +0.023 但 recall -0.007，只保留 scale-sensitivity 机制注释，关闭 1280 训练路线。
- 更早 HRSC summary 提供结构先验：C1 两 seed 通过 gate，parent-visible tensor max abs diff=0；C2/C3 均失败，禁止无新单变量设计地堆叠。

## Protocol Provenance and Supersession

- Clean-start 设计锁定：train 47,294 tiles / 836,745 boxes / 22,575 empty；raw val 13,833 / 7,228 empty / 18 classes，`filter_empty_gt=False`，scale1024 canonical，Q600 all rows。
- 允许的初始化为 generic GroundingDINO OGC + local BERT；兼容转换 checkpoint 单独保存 provenance。禁止 DOTA/HRSC/DIOR-R/FAIR1M/OpenRSD、P126C/P121/A10、旧 OV-CapFlow 权重、teacher/distill/pseudo/proposal cache/derived query prototype。
- 原始设计中“physical GPU4–7”和“E24 上升则续到 E36/E48”是历史计划；当前 2026-07-20/21 addendum 明确覆盖为 GPU2/3/8/9，E24 last authority，automatic E25+ forbidden。
- `full24e-protocol.md` 是早期 scale800 grouped 路线：E1 0.3443/0.3440；E6 0.4842/0.4840、novel4 0.204，未过 0.506/0.506/0.250 gate 后暂停，不是当前 T7 rare4x work_dir。
- S1 grouped O2O 在冻结主 seed 上 AP50 0.407、相对 control +0.026，但 seed20260716 repeat 仅 +0.004，已有种子敏感风险。不能把 grouped 的训练 loss 动力学当作 E24 AP 证据。

## T7 GPU2389 Continuation Provenance

- rare4x full scale1024 raw E1 mAP/AP50 0.4168/0.4170；E6 0.5537/0.5540。
- world2→world4 迁移保持 batch2×world2×accum8 = batch2×world4×accum4 = effective batch 32；每 rank 5,916 microsteps/epoch。
- 重基准 checkpoint 只修正 iter/max_iters 元数据；909 model tensors 和 2,659 optimizer tensors content digest 与原 E6 checkpoint 完全相同，24 个 world4 sampler epochs duplicate=0/missing=0。
- 第一次 GPU2389 启动漏 CLI `--resume`，20 steps 后停止、无 checkpoint/科学评测，属于 invalid-engineering；修正仅增加 `--resume`，日志明确 `resumed epoch: 6, iter: 35496`。
- world4 改变 rank/worker 后 augmentation、DN noise、accumulation grouping 不再逐样本相同，因此当前是 valid stochastic continuation，不声称 bitwise replay；raw milestone 仍有效。
- E6 rounded novel4/base14 AP 为 0.4315/0.588571；container-crane AP/recall 0.052/0.366、SV 0.246/0.490，指向不同机制，不能用一个未经诊断的模块同时处理。
- rare4x 在 matched scale1024 proxy 上的单变量效应可识别：AP50/novel4/base14 `+0.066/+0.25525/+0.007786`。
- raw E6 相对旧 scale800 的 `+0.070` 同时包含 scale 与 exposure，缺少 full scale1024/no-rare cell，禁止拆分 resolution/rare/interaction 主效应。
- Proxy helipad=0 GT，official proxy macro 实际为17个 supported classes；今后需同时报 supported total 与 aligned 18-class macro，单类机制只能由 raw milestone 验证。
- E1→E6 AP 增益的对称分解：recall 34.78%、AP/recall efficiency 65.22%；这是历史动力学，不替代 E24 canonical 分解。
- Continuation 文档中的历史 “D11 first” 已被后来的 `8-D141-PRIORITY-RECONCILIATION` 明确撤回，不能作为当前启动授权。
- 当前 resume config：batch2/GPU、world4、accum4、`update_count_multiple=4`，`load_from=epoch_6_world4_iterrebased.pth`、`resume=True`，work_dir 与实时 tmux 一致。
- E24 non-resume eval base：val/test 使用 `/data1/zcy/datasets/DOTA2_1024_500/ss_val`、scale1024、`filter_empty_gt=False`；训练 rare4x data root 独立，eval 不读取 proxy mouth。
- Q600、milestone、strict no-NMS/no-top-k 等仍位于更深 inheritance layer，需继续展开核验后才能把 config audit 标记完成。
- `full24e.py` base 锁定 max_epochs=24、checkpoint milestones 1/6/12/18/24、train 47,294/raw val 13,833、两者 `filter_empty_gt=False`；grouped delta 仅训练期 3 groups/1,800 matching queries。
- `backbone.with_cp=False` 与 wrapper `static_graph=False` 是 PyTorch1.12 DDP+accum 工程兼容修复，不改变科学模型/损失；当前 world4 child 将 accum2 覆盖为 accum4。
- `s1_control.py` 只覆盖 subset dataloaders、12e schedule、accum2 与 checkpoint policy；它继承 `q600_base.py`，没有另设推理 mouth。
- `q600_base.py` 明确 `num_queries=600`、18 canonical classes、14 base/4 novel、`classification=text_token_similarity`、`fixed_classifier_width=False`；model=`OVCapFlow`/head=`OVCapFlowHead`、`test_cfg` deleted。
- q600/full24/grouped 配置层无 NMS/top-k/dense/RPN/RoI 字段；仍需读 `c0_native_1e.py` 和代码级 predict/export 才完成隐性路径审计。
- `c0_native_1e.py` 最终 evaluator=`DOTAMetric(metric='mAP', iou_thrs=0.5)`；val/test sampler shuffle=False、round_up=False，empty tiles retained；semantic fusion/density/null/balanced paths disabled，`test_cfg` deleted。
- Config inheritance audit 未发现 NMS/top-k/dense/RPN/RoI inference fields；剩余检查是 `OVCapFlow`/head predict/export 代码及 strict auditor。
- Code-level `OVCapFlow.pre_decoder`: training only 时复制 query groups 和 DN；eval 不复制、不含 DN，matching count=`num_queries`，encoder proposal class/coord outputs为None。
- `OVCapFlowHead.predict` 逐 query 在 class 维选一个 label，不对 query 行排序/裁剪；bbox 保持 `(Q,5)`，仅反归一化、angle scaling 和 image-bound clamp，scores/labels/boxes 同行数。
- 因此 training Q1800 不泄漏到 inference，实际 inference 是原序 Q600 all-query；仍需核对 calibration helper 与 strict auditor 的拦截/测试证据。
- Calibration helper 只在 class 维 `max(dim=-1)`，随后做 per-query score calibration；不跨 query 排序、筛选或聚合。
- Strict auditor 做项目 AST scan + runtime hooks，覆盖 torch/Tensor topk、MMCV NMS/nms_rotated、OpenCV minAreaRect，并要求每图 exactly cfg.num_queries。
- Auditor 不一般化拦截 `sort/argsort/slice`，但逐行 predict/calibration 路径未出现这些操作；还需读取 T7 实际 audit artifact/测试结果。
- Audit inventory has strict/open-vocabulary/alignment artifacts for grouped/control/repeat candidates, but no T7/E18-named strict JSON found in the current audit directory listing.
- T7 only changes data/scale/exposure/training continuation, not predict graph; nevertheless current protocol verification will use existing artifact plus focused CPU/config tests instead of assuming pass or consuming busy GPUs.
- Same-graph grouped strict artifact: `pass=true`, predictions `[600,600]`, static/runtime forbidden calls empty, uses_nms/topk/minAreaRect false.
- Same-graph open-vocabulary artifact: `pass=true`, expected_queries=600, `state_shapes_unchanged=true`; prompt variant counts still need short-tail confirmation.
- OV artifact contains canonical, reordered, synonym-expanded and a fourth variant block; exact counts are localized around lines 6167–7134 for follow-up.
- OV variants confirmed: canonical/reordered/synonym-expanded/one-new-class all return `[600,600]`; fixed classification layers, forbidden checkpoint keys and count failures are empty.
- 2026-07-21 CPU focused verification: strict head/audit, calibration, clean-start config, E24 queue/validator and GPU guard = `51 passed in 10.39s` under selected mmdet env with CUDA disabled.
- Same-graph decoder alignment: two images prediction=decoder=600, same shape/order, max abs error 0.0; checkpoint missing/unexpected keys empty.
- OV artifact tail includes a 19-class positive map with `[600,600]` result counts, consistent with new-class shape invariance; variant names/counts will be read directly before finalizing.
- Evidence label remains “same predict graph audited”, not “current E24 checkpoint audited”; E24 dump validator/strict protocol audit remains required after publication.

## E24 Autoschedule Contract

- 唯一 launch gate：`checkpoint_published && validation_finished && !training_alive && gpus_idle && !already_launched`。
- Scheduler 每 30 秒轮询，deadline 2026-07-23 23:59 +08:00；不得 kill/pause live training，必须 atomic JSONL event，防 duplicate launch。
- Evaluation 使用 non-resume base config、GPU 2/3/8/9、4 ranks、port 29791、NCCL P2P/IB disabled；成功后才运行 CPU-only validator。
- Validator production defaults：13,833 records、600 rows/image、18 labels、8,299,800 rows，并检查 unique ids、rotated `(Q,5)` boxes、shape、finite、label range。
- No E18 replay queued by default；E24 authority/dump first。只有 E24 decomposition 确需 exact trajectory comparison 时才另行调度 E18 canonical replay。
- 实时 queue state `ready=false` 与设计一致，不是调度故障。
- E24 dump 的唯一 owner 是现有 queue；只要训练仍存活或 `evaluation_launched` 已存在，禁止手工第二次 launch，也禁止复用同一 output 目录。
- E24 evaluation 刻意使用 non-resume config：它与 live resume config 的 model/test dataloader/evaluator 解析结果一致，但不会继承训练恢复语义。
- 启动前还必须同时满足 checkpoint 可 CPU load/metadata 正确、E24 validation metric 已落盘、训练退出、四卡 idle、port 29791 free、磁盘至少 5 GB。
- dump 完成判据不仅是进程退出：自动 E24 metric 与 dump evaluation metric 一致，CPU-remap 可载入，13,833 unique IDs，每图 boxes=`600x5`、scores/labels=`600`、finite、含 gt/img_id/pred，并记录 hash/size。
- D143 已用一个 433,788,897-byte、8,299,800-row CUDA-tagged dump 验证 CPU-remap validator 路径；这只证明工具链，不替代 canonical E24 产物。
- E24 后的诊断顺序固定为：容量→geometry/semantic reachability→base/novel→density→size/aspect/angle→SV density×size→score vs same-label rotated IoU→duplicate/evaluator FP→AP/recall efficiency→与 E18 比较。
- 只有 E24 仍有无法区分的 trajectory 问题时才考虑另排 E18 canonical replay；不能为了“完整”而预先占卡。

## Route-Priority Supersession

- `.lab/parking-lot.md` 的 2026-07-21 reconciliation 明确覆盖早期“D11 优先”文字。
- E24 若复现 broad high-recall/low-AP headroom 且 score–rotated-IoU 弱：quality/ranking family 获得唯一下一候选的 first claim。
- E24 若显示 exact tiny/dense geometry 与 query/refinement association 主导：才审议 fixed-Q600 coverage/initialization。
- D11 需要 AP-support query-id blind spot；D12 需要 matched shape/refinement bias。候选互斥，禁止自动 E25+、NMS/top-k/extra head。
- Q600 joint theoretical recall cap 0.934578、SV cap 0.896580，说明“仅增加查询数”不是当前已证实瓶颈；该上界不证明实际 ownership 已解决。
- OGC query 与 decoder xywh 的 omission 是真实代码事实，但目前没有 canonical raw AP 因果证据，不能据此默认启动 D11/D12。
- Quality/ranking 只有在 E24 canonical 复现 broad high-recall/low-AP 且 score–IoU 弱时才是 provisional first；coverage 必须有 exact tiny/dense 与 query/refinement linkage。
- D11 的准入证据是 AP-support query-id blind spots；D12 的准入证据是 terminal-shape/refinement bias。两者不能叠加试错。
- Prompt/text 路线只允许一个 test-only A/B，且只在 novel semantic evidence 主导时进入；不能用 prompt 改动掩盖 geometry/ranking 问题。
- 同一 D141 目录中的早期 `e18_ap70_budget_label_allocation_and_init_priority.md` 曾把 D11 放在 quality loss 前；后出的 priority reconciliation 明确撤回这一顺序，故该文只保留诊断事实，不再保留候选排序授权。

## AP70 Effect-Size and Allocation Audit

- E18 rounded class AP sum=10.947，AP70 target sum=12.600，所需 AP-sum gain=1.653（macro +0.091833）。E12→E18 的 +0.003724 只覆盖约 4% 当前 gap。
- VOC07 fixed-recall perfect-ranking oracle sum=14.272727、mAP≈0.792929；总 ranking headroom=3.325727，达到 AP70 要回收 49.703%。这是数学可达上界，不是单一 quality loss 足够的证据。
- 六个 AP≥0.70 强类只有 0.253818 headroom；即使全部回收，12 个弱类仍须贡献 1.399182，即完整 AP70 gap 的 84.645%。强类只能作为 regression guards，不能承担主要增益。
- 九个 recall≥0.70/AP<0.70 类占 fixed-recall headroom 65.98%；若强类与三个低-recall 类都不改善，这九类要实现其 perfect-order headroom 的 75.329%，实际风险很高。
- 低-recall 三类为 container-crane/helipad/SV；helipad 仅 6 GT，不能单独选路。只让九个 high-recall weak-AP 类到 0.70，再把三低-recall类的 AP 提到当前 recall，合计只比目标多 0.002 AP-sum，几乎无现实余量；长期路线最终很可能同时需要 efficiency 与 coverage，但每次仍须单变量互斥验证。
- E12→E18 所有 8,299,800 query 的最终 label allocation 明显集中：airport share 54.07%→60.97%，top-5 89.30%→91.29%，entropy 1.6225→1.4448，effective classes 5.066→4.241。airport 101 GT 却获得约 506 万 query labels。
- 这些 det counts 是所有 query 的 token argmax allocation，不是 score-ranked FP；模型无 background/no-object output，故 airport hub 可能是 unmatched token baseline，也可能是候选分配变化，必须用 E24 row-level score/same-label IoU/rank 区分，不能直接派生 loss。
- D11 的可检验事实：converter 因 Q900→Q600 丢弃 generic OGC `tgt_embed`；E18 target→OGC first600 same-index cosine≈0.0032，target stable rank 70.46 vs OGC 4.16，E12→E18 query same-row cosine 0.9980，说明 plateau 后 basin 稳定且未自动恢复 generic manifold。但 source/target 范数和低秩结构差异大，D11 不是低风险微调。
- D12 的可检验事实：generic decoder terminal xywh rows 因 4D→5D mismatch 被丢弃，但 source 权重尺度可能产生极端 zero-update boxes；D11/D12 必须互斥。当前二者排序均服从后出的 E24 canonical gate。

## Prompt and Literature Compatibility (Local Trace; External Reverification Pending)

- 18 canonical classes 中 10 个含 hyphen；当前 `clean_label_name()` 只替换 underscore，不替换 hyphen，BERT positive tokens 因而共享 `-` token。
- Hyphen group GT share 74.3149%，但 E18 all-query label share 仅 10.6083%（E12 13.3477%）；non-hyphen group 89.3917%。这是格式相关 allocation signal，不是因果：unmatched queries 也必须前景 argmax，hyphen group AP 还略升，tennis-court AP=0.894。
- 正确 natural-space A/B 不能直接改 dataset classes，否则 canonical annotation `cls_map` 可能损坏；必须在 canonical dataloader 解析后，仅替换 inference sample `text/custom_entities`，evaluator labels/order 保持 canonical。
- 预注册 prompt A/B：E24 发布且训练退出后、raw13,833、Q600、10 hyphen→space、无逐类 prompt sweep；仅当 total mAP 与 AP50 都至少 +0.005 且 novel4/base14 均不下降超过 0.001 时保留，否则关闭路线。
- 现有 evaluator 尚缺完整18类 canonical-label/alias-prompt 映射扩展；当前没有实现授权，需在 E24 前后按证据和 TDD 决定。
- 本地 literature trace 把 Stable Matching DETR、Rank-DETR、Align-DETR 作为条件 quality references；AO2/D²Q/DQ/Dome 仅作 dense/tiny/query 诊断参考，动态 query、counting head、point head、proposal top-k、minAreaRect 等均不兼容当前合同。
- 本地 trace 还列出 Grounding DINO/DetCLIP 作为 concept-text 解耦依据，以及 PET-DINO/PaQ-DETR 等 2026 工作；所有“最新/first”相关条目必须由本轮独立原始来源检索重新验证，未验证前不用于 novelty claim。
- 若 canonical E24 触发 ranking 分支，最小候选优先考虑 rotated-IoU positional positive target 或 matching-cost modulation 二选一；不重复 grouped/many-positive，不叠 decoder/query/prompt。

## Independent Literature Verification (2026-07-21)

- 已从官方 proceedings 核验 Stable Matching DETR（ICCV 2023）：positive classification 由 positional metric/IoU 监督，并可调制 matching cost；它是当前最贴近“单一训练 target/cost”的机制参考，但证据来自 COCO，不能外推 raw DOTA2 效应量。
- 已从 NeurIPS 官方页核验 Rank-DETR（NeurIPS 2023）：核心是分类分数与定位质量的排序错位；其论文语境包含排序/选择高排名预测，本项目只可借鉴 loss/cost/诊断，不能引入 inference top-k。
- 已从 BMVC 官方页核验 Align-DETR（BMVC 2024）：joint quality Align Loss + intermediate many-to-one；本项目若触发只能借 quality target/regression weighting，因 grouped training 已覆盖 many-positive 思路。
- 已从 ECCV 官方页核验 DQ-DETR（ECCV 2024）：counting/density map、feature enhancement、动态 query 数量与位置；它支持 tiny/dense diagnosis，但额外模块与动态 mouth 违反当前合同。
- 已从 NeurIPS 官方页核验 DetCLIP（NeurIPS 2022）：concept dictionary/descriptions 与 separated concept formulation 支持“canonical label 与 concept text 可解耦”的研究背景，但收益来自预训练，不保证 test-time hyphen alias A/B。
- 已从 CVF 官方页核验 PaQ-DETR 为 **CVPR 2026 正式论文**（不是仅 arXiv）：content-conditioned dynamic queries + quality-aware one-to-many assignment，报告多 backbone 1.5–4.2 mAP gains。它与“query adaptivity + quality balance”的宽泛组合高度重叠，禁止此类 first claim；其动态 query/one-to-many 也不直接兼容当前路线。
- 已从 CVF 官方页核验 PET-DINO（CVPR 2026）：visual/text prompts 与 prompt-enriched multi-route training。它提示 prompt 泛化可能依赖训练期多样性，反而支持先做一次 test-only alias A/B、失败即关闭。
- 新增正式近邻 HSA-DINO（CVPR 2026）：multi-scale prompt bank + semantic-aware inference router 面向 OVOD domain shift；会约束“domain-specific semantic augmentation”叙事，但额外 prompt bank/router 与当前最小候选不同。
- 新增正式近邻 ViTPrompt（CVPR 2026）：从首轮高置信检测提 visual tokens，第二轮同时修 box/class，training-free；它是 prompt+localization 强近邻，但 two-pass/high-confidence selection 不等价于当前 single-mouth all-Q600，不能直接采用。
- 已核验 VK-Det（arXiv:2511.18075，2025）为 OV aerial 直接近邻：使用 visual knowledge、prototype pseudo-label 和 score fusion，在 DOTA 报 novel mAP 23.3；它是 two-stage Faster R-CNN、top-500 proposals/多分数融合，合同不同但必须进入 related work 与 novelty 风险表。未发现正式 venue，暂标 preprint。
- Dome-DETR（arXiv:2505.05741）与 D2Q-DETR（arXiv:2303.00542）原始记录可追溯；前者 density mask/sparse attention/adaptive query，后者 point head/dynamic query。两者仅作 coverage 机制参考。
- 结论：当前不存在可安全书写的“首次把 query/quality/prompt 用于 DETR/OVOD/航拍”主张。未来 novelty 必须收窄到经 E24 触发、fixed-Q600、rotated-5D、open-vocabulary、strict all-query 口径下的具体新机制，并再次检索。

## Paper/Figure Asset Audit

- `projects/OVCapFlow/README.md` 描述早期 isolated extension 与 semantic/capacity/null 机制，但其“尚未 real-batch / DOTA2 protocol 未配置”状态已被当前 T7 实验事实覆盖；不能把该 README 当现状权威，后续应在安全时更新或明确 archival status。
- README 中 parent-preserving semantic residual、continuous capacity、scene count、null reservoir 均为已实现候选，不是 current best T7 active path；当前配置已确认这些 intervention disabled。
- 既有三图设计正确区分：Figure1=validated strict T7；Figure2=complete architecture且候选用amber dashed；Figure3=parent-preserving semantic-capacity equation。它是图稿规范，不是性能证据。
- Figure1 可安全表达 Q600 learned content/5D refs、x6 decoder、training-only three groups+DN、all-query class+5D box、no proposal/global top-k/NMS；不得画 semantic-capacity/null 为 T7 active。
- Figure2/3 只有 matched causal validation 后才能把候选 dashed 改为 solid；在此之前 captions 必须写“implemented candidate; excluded from current best T7”。
- Open-vocabulary 图文只能写 vocabulary-conditioned/text-prompt scoring；尚不能写 proven unseen/zero-shot、SOTA、first 或 AP70。
- 现有规范要求 draw.io源+SVG/PDF vector+300dpi PNG、178mm宽、字体≥8pt、色盲安全/线型冗余、无外部图标/栅格文本。后续三图制作应复用此规范，不重新设计视觉语法。

## OpenRSD Rotation / False-SV Historical Audit (Read-only)

- 口径锁定：OpenRSD paper-mouth 是 filtered non-empty `ss_val=6605`、scale1024、text7、`val_using_aux=False`，A10 epoch24复现0.704959/0.7050；raw all-patch `ss_val=13833` 是另一 mouth，历史约0.652 AP50，不能与当前T7 raw0.608直接当同协议优劣。
- 2026-05 的 958-job SV-shift suite 总结：rotation后 dense logits 仍向SV对齐，final SV FP含context/line activation，object flip只是子集；white-line-only under-triggers，因果判定为 `CONTEXT_PRIMARY` / general local-instance attractor，而非ball-court-only。
- P0148 是 dense-stage hub/origin signal与diagnostic stress tile，明确不在 final-test primary metrics；任何方法不能凭 P0148 改善声称泛化或AP收益。
- Cross-class历史：harbor/bridge strict GT→SV=0，baseball/soccer较高；court-conditioned 2500-tile audit只有8 tiles出现court→SV且P0148不在该cohort。whole-image SV ratio会被真实停车场混淆，必须在non-SV GT区域内统计。
- 旧 repair suite 的主链写作“SV embedding attractor→dense drift→top-k/NMS amplification→court-like context/background FP”；其中 top-k/NMS 属于旧 OpenRSD mouth，不能移植进OV-CapFlow strict all-query合同。
- 历史 orbit real inference 可降低P0148 mean final-SV ratio：naive0.7510→calibrated0.5798；但P0682 true-SV stress tile也从0.9683降至0.8606。`true_sv_recall_proxy`字段为空，因此这不是GT-matched preservation证据，反而证明必须把P0682设为负向安全门。
- `fres_071_true_sv_preservation` 与其CSV全部标 `PARTIAL_PROXY/PROXY_NOT_OFFICIAL`；不能声称历史方案已保护true SV。
- Orbit methods在heldout final-test的SV drop为负（naive -0.1926、view consensus -0.1066、calibrated -0.0718）且verdict BASELINE_ONLY/WEAK，说明hub suppression没有转化为可靠泛化。
- 旧60/40 heldout的所谓 REAL_AP表是小样本内部eval，多个类AP=0且各角结果重复；它不是DOTA2 raw13,833或paper6605全val，不能作为当前模型AP证据。
- 后续E24问题因此具体化：先确认当前all-query是否仍有P0148-like context→SV hub；任何causal intervention必须同时报告P0148 hub suppression与P0682 GT-matched true-SV recall/score/box retention，并最终以全raw evaluator AP-support FP变化裁决。
- OpenRSD epoch24 text-only 12-angle sweep（专用angle_sweep_val，不是当前raw full-val）mean mAP0.6512，angle0 0.6957，最差240° 0.6280，较同记录ss_val0.6962下降约0.045；说明rotation sensitivity是真实历史现象，但mouth/split不同，不能转成T7 AP delta。
- MESS/Fourier融合12-angle mean仅0.4586–0.4707，远低于text-only0.6512；大融合路线是明确负结果，不应在OV-CapFlow重复。
- 后续8条narrow text/head/rotation微调全部低于0.6962基线4.6–5.0 points；SV-only adapter更跌至0.4870并使SV AP=0.0222。历史表明小参数量/有界初始化也不能保证保留OV alignment，任何prompt/text训练必须有no-op/near-zero-LR control并优先test-only诊断。

## Latest Read-Only Diagnostics (E18 / Scale1280 Preflight; E24 Still Required)

- `.lab/results.tsv` 有 403 行，`.lab/log.md` 有 6,640 行；不再整段读取 TSV，使用最新尾部与 run ID 定向核验。
- E18 canonical fixed-recall perfect-ranking oracle mAP 约 0.792929；从 0.6081 到 0.70 需回收总 ranking headroom 的约 49.7034%。这使 quality/ranking 成为直接证据较强的候选族，但不证明一个 loss 足够。
- E18 scale1280 exact evaluator decomposition（非 canonical）：AP-support FP 中 localization/background 65.3577%、empty 16.5213%、duplicate 14.8881%、semantic 3.2328%；reconstructed mAP 与 official 的绝对误差 2.28e-9。
- Scale1280 matched preflight：geometry miss rate 0.337870、semantic miss 0.008927；tiny≤8 geometry miss 0.784234、density>600 geometry miss 0.846274、SV approximate geometry share 0.848341；score-IoU Spearman 0.486130，TP 内 0.277827。
- Fixed reference center 对 tiny 并不更远，但 density 越高 nearest-reference collision 越强；fixed references 仍近 41 px square，而 GT median size/aspect 14.68 px/2.32。初始 prior 不是 final box，必须用 canonical matched trajectory 区分 refinement/ownership。
- Query preflight：无 dead-query collapse（high-score effective queries 568.448，TP effective queries 590.675），无 monomorphic prior trapping final boxes；存在 top-score short-path bias 候选。
- E1/E6/E12/E18 curve：0.553674→0.604341→0.608065（E6 onward），late gain retention 0.073504；同配方 repeat-last-stage E24 projection 0.611789，optimistic late aggregate 0.658732，classwise cherrypick 0.667833，均不足以支持自然达到 0.70。
- 结论强度：当前只支持“混合 ranking + tiny/dense coverage 瓶颈”的方向性观察。必须由 canonical E24 dump 决定唯一干预，不得从 scale1280 直接启动训练。
- D143 预注册将唯一科学变量定义为 test Resize 1024→1280；batch2→1 是显存工程变更，保持 raw13,833/Q600/5D/no-NMS/no-top-k，optimizer steps=0。
- D143 exact outcome：0.6081/0.6080→0.5942/0.5940（-0.0139/-0.0140）；novel4 -0.02250、base14 -0.01150，固定1280测试尺度已关闭。SV AP +0.023 但 recall -0.007，只能说明 feature-scale/ranking tradeoff，不能称 coverage 改善。
- D143 dump integrity：433,788,897 bytes，SHA `951d06fd8f2ddb83ead62eac3f028a27ea3b76965de30e1478ed2e95375611e5`，13,833 unique records、8,299,800 rows、shape/finite/label-range bad counts=0。
- D143 暴露便携性缺陷：prediction tensors 在 CPU，但 GT rotated boxes 序列化为 cuda:0/cuda:1；官方同 run metric 不受影响，E24 validator 必须 CPU-remap 或落盘前递归 CPU 化。
- D143 是本轮 goal 启动前已完成的历史 run，曾使用物理 GPU4/5。当前用户 GPU 合同已明确禁止 4/5；该历史行为不构成今后复用授权，后续只允许 2/3/8/9 且当前先保护 live T7。
- Cross-scale class topology 稳定：1024 vs 1280 的 AP/recall/efficiency Pearson=0.9953/0.9875/0.9889；相同 12 类 AP<.70、相同 3 类 recall<.70、相同 9 类 recall≥.80/AP<.70。它降低“混合 ranking+coverage 只是 1280 artifact”的风险，但 exact FP/IoU 数值仍不能迁移。
- E18 fixed reference center 相对 25×25 grid 仅 mean 1.653px 移动；所有 size/density bin 最近中心均约16px，tiny≤8 vs >64 只差0.369px，故“tiny 中心离网格太远”被否定。
- 局部 nearest-reference collision excess 随 density 单调从5.559%升至77.408%（>600 GT/image）；它体现 allocation competition，但 decoder 可移动且 query 不硬绑定，不能当 recall ceiling。
- 600 fixed 5D refs 到 E18 仍几乎同一个约41px正方形/90° prior：size p05–p95 40.369–41.404px，aspect 1.0008–1.0215，angle 89.854–90.128°；GT median size14.680px、aspect2.323，支持差异显著。
- Tiny≤8 GT 对最近 reference scale mismatch median 6.171×，within4×=0%；这是 refinement-distance 机制证据而非 final-box ceiling。E24 应用 query id 连接 prior→final box，检验 geometry miss 是否随 log-scale refinement 与 local collision 集中。
- 输出行号可可靠作为 query id：推理无复制/筛选/排序/截断。D143 全行 prior→final center 位移 p50/p95=59.07/193.17px，scale refinement=2.651×/8.732×，说明 decoder 并未困在41px prior；但 prior center 与每-query跨图平均 final center x/y Pearson=0.99907/0.99803，属于“稳定空间身份上的图像条件化大幅 refinement”。
- 无 dead-query collapse：p99高分有效query=568.45、zero=0；evaluator TP witness有效query=590.67、zero-TP=0。增加/动态/prune query 没有此类证据。
- E18×1280 GT-matched preflight：243,632 GT 中 geometry miss 82,316 (33.787%)，semantic miss 2,175 (0.893%)，evaluator reachable 159,129 (65.315%)；仅12 GT因 best-GT competition 丢最大召回。candidate excess≈309k 主要可能伤 AP，不可再称 recall competition。
- Tiny≤8 geometry miss 78.42%/recall21.30%；>600-density geometry miss84.63%/recall14.94%；SV约69,832 geometry misses，占全部84.83%。>600桶同时含Q600不可避免容量缺口，E24需拆开容量下限与额外几何失败。
- Reachable TP witness score–IoU Spearman仅0.2778；score与center/scale refinement约-0.2365/-0.2264。quality与refinement均有关但受size/class/density混杂，不能直接选loss/initializer。
- Exact evaluator reconstruction在非canonical dump上误差2.28e-9；AP-support FP mix为 localization/background65.36%、empty16.52%、duplicate14.89%、semantic3.23%。airport数百万row多数低分无害，必须基于class-wise AP-support prefix而非all-row mass做机制判断。
- 当前分类为binary token Focal，rotated KLD已在box loss与Hungarian cost；真正代码缺口是positive classification target不编码rotated quality。Parent head不支持config-only QFL，简单换loss名是伪实验。
- temperature/power在null/capacity关闭时只是严格单调score transform，不改变排序/AP；禁止作为AP候选。

## E18 Canonical Raw Record (Direct Artifact Read)

- Exact authority: E6/E12/E18 mAP `0.553674/0.604341/0.608065`, AP50 `0.554/0.604/0.608`; E12→E18 only `+0.003724/+0.004`。
- E18 rounded mean AP/recall/efficiency `0.608167/0.816944/0.744441`; E12→E18 recall `-0.0055` while efficiency `+0.009572`，AP Shapley = recall `-0.004068` + efficiency `+0.007846`。
- Novel4 E12→E18 AP/recall `-0.01275/-0.00775`; base14 `+0.0085/-0.004857`。Late training trades group/class gains rather than universally improving coverage。
- Same 12 classes below AP0.70 at E12/E18；persistent low-recall set = container-crane, helipad, small-vehicle；same nine high-recall/low-AP classes persist。
- E18 checkpoint `2,180,744,553` bytes, SHA256 `e757915d1d9a3af89a10a2c1b79d26718c1eaeb92419a15c236eb14f3374d6de`。
- E18 record explicitly says same-route dump unavailable and ranking vs rotated localization not identifiable；continue unchanged to E24, D11/D12 unauthorized, no stacking/completion claim。

## Runtime Observations

### Snapshot 1 (2026-07-21)

- GPU 2/3/8/9 memory used: 41,351 / 40,695 / 40,743 / 40,839 MiB（总显存均 46,068 MiB）。
- 瞬时利用率分别为 3% / 47% / 52% / 35%；单点快照不能判断停滞。
- 相关 tmux：`t7_full24_rare4x_gpu2389`、`t7_full24_monitor_gpu2389`、`t7_full24_guard_gpu2389_e24`、`t7_full24_wait_e18_m5900_gpu2389`、`t7_e24_dump_queue_gpu2389`。
- 安全决定：在确认每个会话、PID、日志和 checkpoint 状态前，不启动任何新训练/推理，不结束任何会话。
- 当前沙箱中的 `pgrep` 只能看到命名空间内进程，不能用于否定宿主 PID 689012 等的存在；后续通过 tmux pane 和日志核验。
- tmux pane 显示训练 master/launcher 由 monitor/guard 以 PID 688939 跟踪；workers 与 GPU PID 映射一致。
- 训练 stdout 重定向到 `work_dirs/dotav2_cleanstart/full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2_gpu2389_e6_resume_a1_launcher.log`，capture-pane 为空是预期行为。
- E24 queue state/log：`.lab/workspace/8-t7-e24-dump-gpu2389-queue.jsonl` 与 `.lab/workspace/8-t7-e24-dump-gpu2389-queue.log`；deadline `2026-07-23T23:59:00+08:00`。
- monitor 多个采样点四卡达到 90–100% 利用率，当前没有数据加载永久阻塞的迹象；仍需用短 tail 精确确认最新 iteration/ETA。
- 16:55:59 的训练精确末行为 E21 2400/5916，ETA 9:46:49；当前审稿决定为 CONTINUE monitoring。
- 17:14:02 训练已到 E21 2820/5916，ETA 9:36:48；物理2/3/8/9均100%利用率，日志持续增长，无停滞迹象。
- E24 queue tmux pane `pane_dead=0`，仍执行唯一预注册 queue command；state JSON 只在状态变化时写入，15:00后无新行不等于队列死亡。
- Codex persistent goal 状态重新核验为 `active`。当前工具面未暴露 generic `automation_update`；72h 持续链由 active goal + live tmux monitor + guard + E24 gated queue 共同承担，不能另造重复调度器。
- 17:25:01 已推进至E21 3080/5916，ETA9:30:31；monitor至17:25:07仍`process_group_alive=true`、`fatal_pattern=null`，四卡显存稳定约40–41GiB并有持续利用率波动。决定仍为CONTINUE。
- Goal continuation重新核验：17:26:40训练到E21 3120/5916、ETA9:29:32；monitor至17:27:09仍alive/fatal-null，E24 checkpoint与dump均尚不存在，queue waiting state正确。
- D139 exact matched/evaluator分析只留下审计文档，没有可直接复用的项目脚本；当前`projects/OVCapFlow/tools`只有mouth/strict/OV/queue/validator等工具。若不固化canonical analyzer，E24到达后会重复临时分析且不利于复现。

## Git Protection Snapshot

- Branch: `research/dotav2-cleanstart-ov-e2e-ap70`。
- HEAD: `314c1311389cddd0545dbebdc87ada7fc2492f2a`。
- `git diff --stat` 为空，未发现 tracked 工作树修改。
- 本轮新增：`task_plan.md`、`findings.md`、`progress.md`。
- 其余未跟踪内容均视为用户已有资产并保护：`.superpowers/`、两周复盘目录、E24 autoschedule 计划、`queue_test_after_training.py`、`validate_dotav2_q600_dump.py`、GPU guard/queue/validator tests 等。
- 不执行 add/commit/clean/reset/checkout，不覆盖这些文件。
- 17:25复核仍无tracked diff；所有列出的`??`继续视为用户资产（仅本轮新建的三个planning files例外）。
- 磁盘决定：1.3 TB 余量允许当前 E24 dump，但因卷使用率 86%，禁止无计划复制 checkpoint 或生成重复 dump。

## 2026-07-30 — “首个 E2E 遥感开放词汇旋转框”主张核验

- 宽泛的“首个遥感开放词汇检测器”主张不成立：CastDet ECCV 2024 已正式提出 aerial OVAD；其 IJCV 2026 扩展版进一步给出 oriented OVAD。
- 宽泛的“首个遥感开放词汇旋转框检测器”也不成立：
  - Oriented CastDet 使用 Faster/Oriented R-CNN、RPN/RoI、EMA localization teacher、RemoteCLIP teacher、动态伪标签队列；
  - OpenRSD（ICCV 2025）明确支持 OBB/HBB，但基于 MMRotate RTMDet-L，训练含自标注/自训练，伪标签合并使用 class-agnostic NMS。
- “首个端到端遥感开放词汇检测器”同样不可写：
  - arXiv:2408.12246 的最新版本已改名为 RT-OVAD，明确称基于 RT-DETR 的 end-to-end framework；
  - 它设置 500 object queries，并从 encoder features 按图文相似度选择 top 500 初始化 decoder；
  - 其框回归定义与伪标签均为 4D，正文未给 oriented/rotated box 方法或实验，因此它是强 HBB 近邻，但不是当前 rotated-5D strict mouth 的直接先例。
- SOAR（AAAI 2026）也是 DETR/query 近邻，但其 unlabeled branch 从 foreground priors 生成 `K×4` pseudo boxes，并用 Refiner 伪标签训练；这与 teacher-free/pseudo-label-free rotated-5D 路线不同。
- Cross-View Open-Vocabulary Object Detection in Aerial Imagery（arXiv:2510.03858；OpenReview 仍显示 Submitted to ICLR 2026）使用 OWLv2 与 bipartite box training，但其跨视角数据构建依赖 OWLv2 伪框、NMS 和水平框；不能当作已接收论文，也不是 rotated-5D 先例。
- OpenRSD 在 DOTA-v2.0 的 text-prompt OBB 表中报告 70.1，但它使用 ORSD+ 约 47 万图像/200 类、多阶段预训练/微调/自训练、RTMDet-R 与 NMS。该数值说明 AP70 在大规模非匹配协议下可达，不能与当前 raw 13,833 all-patch、clean-start、Q600、no-NMS mouth 横比，也不能作为当前目标已达证据。
- 当前可辩护但尚不能正式宣称的差异化边界是：**teacher-free、pseudo-label-free、fixed-Q600、rotated-5D direct set prediction、all-query scoring、inference no-NMS/no-proposal-top-k 的遥感开放词汇检测器**。
- “first”仍须在成稿前做一次系统检索与逐篇排除；当前只允许将上述边界写为 working positioning，不允许写成已证实的首次性结论。
- 机制层面新增风险：RT-OVAD 已覆盖 text-guided query enhancement；SOAR 已覆盖 foreground/language-aware query selection/enhancement；PaQ-DETR 已覆盖 dynamic content query + quality-aware assignment。因此后续创新不能泛称“语义 query”“动态 query”或“quality-aware query”，必须绑定 strict rotated all-query mouth 的可复现实证缺口。
## 2026-07-21 E24 canonical analyzer design evidence

- The repository has a reusable CPU-safe dump loader/validator in `projects/OVCapFlow/tools/validate_dotav2_q600_dump.py`; it remaps CUDA-tagged pickle storage to CPU and validates the canonical `13833 x 600` record contract.
- The authoritative raw DOTA-v2 evaluator path is `DOTAMetric -> eval_rbbox_map -> tpfp_default`, with class-wise global score ordering, greedy one-GT assignment, rotated IoU, and VOC07 11-point AP. Any analyzer must reconstruct this exact path and fail closed when parity is not met.
- No reusable D139/D140 diagnostic implementation exists: the current workspace contains the runbook and evidence documents, but the prior decomposition was produced by one-off analysis. A canonical E24 analyzer is therefore the next missing reproducibility artifact.
- The analyzer should preserve prediction-row/query identity before any class filtering and should separately report GT reachability/ownership/allocation, AP-support FP causes, score-IoU calibration, fixed-recall oracle headroom, size/density/query strata, and the P0148/P0682 safety cases.
- Under the `brainstorming` skill gate, implementation is paused until the proposed design is explicitly approved.
## 2026-07-21 17:33 runtime authority correction

- Sandbox-local process listings are not authoritative for the host DDP group. The durable authority is the launcher/scalar log plus `.lab/workspace/8-t7-full24-rare4x-gpu2389-monitor.jsonl` and the queue/guard state logs.
- Latest verified live point: Epoch 21 `[3260/5916]`, monitor `process_group_alive=true`, `fatal_pattern=null`, GPUs 2/3/8/9 only. E24 remains unavailable, so E18 remains the raw metric authority.
## 2026-07-22 analyzer spec self-review findings

- P0148/P0682 cannot be assumed to exist in the canonical DOTA-v2 validation dump. Exact historical IDs are `P0148__1024__651___0` and `P0682__1024__553___0`; the canonical analyzer must report their auxiliary case gate separately from raw DOTA-v2 metrics.
- Same-dump evaluator reconstruction can use the strict `1e-7` mAP parity demonstrated by D139. Independent E24 validation-versus-replay parity is a cross-run check and uses `5e-4` plus equal rounded AP50, while exposing any delta above `1e-7`.
- The output bundle has eight files. `manifest.json` hashes the other seven; its own SHA256 belongs in the experiment ledger to avoid recursive self-hashing.

## 2026-08-03 continuation authority

- 当前用户目标取代旧资源范围：允许使用物理 GPU 0–9，要求至少 18 小时持续值守；仍禁止删除文件、抢占/终止无关进程或做破坏性 Git 操作。
- 已核验的 strict raw DOTA-v2.0 E24 权威结果仍为 `mAP=0.6064053488274416`、`AP50=0.6060`，目标是同一 `13833 × Q600` 口径达到 `mAP/AP50 >= 0.7000`。
- 已关闭路线继续视为硬负证据：D11 matched proxy candidate/control `0.4521/0.4660`；D13-N E12 candidate/control `0.838305831/0.847486973`；QAF source reachability `0.071788826` 且 placebos 不敏感。不得无新证据重复救援这些配方。
- 今日第一阶段只做运行现场、实验台账、配置与评测口径审计；在确认 GPU/进程/产物前不盲目启动训练。
- 2026-08-03 首次资源快照：GPU 0–9 均为 A40、显存各仅 `17 MiB`、利用率 `0–1%`，`nvidia-smi` 无 compute app；宿主未发现训练/评测 Python 进程。全部 GPU 当前可用，但仍需先检查历史 tmux pane 的 exit/status 与今日台账，避免重复启动已完成路线。
- Git 分支为 `research/dotav2-cleanstart-ov-e2e-ap70`；tracked tree 无 diff，列出的 `??` 均继续视为受保护资产，不做 clean/reset/删除。
- tmux 仍保留 A0、D11、D12、D13-N 等历史 session 名称；session 存在不等于进程存活，需要只读检查 pane dead/status/log 后再作队列决策。
- 今日并非“无实验”：`exp-8-omq-decoder-source-m0` 在 01:26 产出 10 个 shard 与 `source_report.json`；随后 OMQ-M1 matched pair 使用 GPU0–4 control、GPU5–9 candidate，两个 work_dir 都至少写到 `epoch_10.pth` 和逐 epoch audit（最近时间约 02:05–02:07）。当前 GPU 已空闲，但必须读取 OMQ source gate、训练末尾、逐 epoch mAP 与停止决策，才能判断是完成、失败还是意外中止。
- 历史 tmux pane 审计：A0/D11/D12 panes 为 `pane_dead=1`；D13-N panes 仍是空闲 `bash`，没有训练子进程。没有发现需要保护的 live GPU workload。
- D12 后续已有正式闭环：candidate 在 Epoch2 前遇到 Hungarian cost non-finite，bounded probe 未复现异常但落后 matched E3 control；formal control E12 `mAP=0.3943`，D12 已 discard，不能重开 rescue/stacking。
- OMQ-M0 source gate 为 `PASS`：overall primary/paired-image/spatial-shift reach=`0.501031637/0.253782669/0.257467086`；`le8px`=`0.496254457/0.160237151/0.136842527`；`small_vehicle`=`0.475290167/0.218556099/0.218556099`。这证明 frozen decoder source 具有图像特异与位置敏感的 geometry-miss reachability，但还不证明训练后 mAP 改善。
- OMQ-M1 matched proxy 曲线 E1–E10：control 恒定约 `mAP=0.8469`；candidate 为 `0.8450, 0.8470, 0.8464, 0.8475, 0.8469, 0.8470, 0.8462, 0.8461, 0.8465, 0.8466`。当前 best candidate E4 仅比 control `+0.0006`，E10 为 `-0.0003`，尚未达到 `+0.003` 机制晋升或 `+0.005` 直接晋升门槛。
- 两臂日志均在 Epoch11 未保存 checkpoint 前无报错地停止（candidate 到 step120，control 到 step140），GPU 随后空闲；当前时刻 02:17 CST，需检查启动方式/提交记录并从共同 E10 checkpoint 成对续到冻结 E12，而不是把不完整 E11 当 endpoint。
- 项目解释器已核验为 `/data/zcy/anaconda3/envs/mmdet/bin/python`：Python 3.8.19、torch 1.12.1+cu113、mmengine 0.10.4、mmrotate 1.0.0rc1。
- M1 当前并非“意外中断待续跑”：更新的冻结设计明确写明 E1–E10 是 user-directed stop 后的完整证据，M1 已关闭，禁止 resume/retune/extend/best-epoch selection。E10 checkpoint metadata 为 `epoch=10, iter=1600`；六层 gate weight L2=`1.418843,1.471636,1.446880,1.412984,1.376743,1.420758`。
- 下一正式 gate 是 held-out source actionability：rich-source vs zero MSE reduction `>=15%`，比 M1-4 高 `>=10pp`，paired-image placebo `<=2%` 且落后 rich `>=10pp`，`<=8px >=10%`（样本数至少100）。FAIL 关闭 OM-QFlow；PASS 才允许 M2。
- Task 3 数据流初审：M1 的 `source_measure_statistics()` 先对 level/point 求和、再对 head 求均值，只保留 `[dx,dy,dx²,dy²]`，因此确实丢掉 head/level topology 与 `dx·dy`；rich-source 应直接从 `RotatedMultiScaleDeformableAttention.last_source_locations/weights/references` 构造，不能再经过 M1-4 聚合。
- `audit_decoder_source_m0.py` 已有无侵入 capture hook、source 数值重建、原图 GT 归一化、pair-placebo 和十 rank shard/publish 模式；Task 3 应复用这些已验证边界，而不是新增第二套模型加载/数据遍历框架。
- `OVCapFlowHead` 已有 final matching assignment capture 路径（`_get_targets_single` 与 loss cache 邻近），下一步需逐行核实 query/group→GT index 映射与生命周期，避免用 M0 的 maximum bipartite mass matching 代替论文设计要求的现有 class-aware Hungarian assigner。
- 逐行核实后发现当前 head 的通用路径只缓存 `bbox_weights.any(-1)` 的 matched-query boolean mask，并未保存 matched GT index；D13-N 的 group mask 也只是 ownership target。因此 actionability capture 必须直接复用 `self.assigner.assign(...)` 返回的 assignment/GT indices，不能仅靠 mask 还原，也不能以 M0 的 source-mass bipartite matching 替代。
- M0 的 `capture_decoder_sources()` 是临时 forward wrapper，重算并保存 exact locations/weights，但当前没有把 layer-specific `reference_points` 写入 state；actionability target 与 M1-4/rich displacement 都必须绑定该层同一 reference，因此 Task 3 要在 capture state 中明确加入 reference tensor 并做 shape/finite/identity 校验。
- sufficient-statistic 方案可避免落盘 raw feature maps：train/val 每个 family/stratum 保存 `XTX, XTy, yTy, count`，验证 SSE 可由冻结 ridge 系数解析计算。paired-image placebo 应交换另一图的 Q600 source rows，再按当前图 Hungarian matched query IDs 取行，从而允许配对图 GT 数不同且保持 targets/references 不变。
- Parent `_get_targets_single` 的权威匹配路径已通过运行时源码确认：先把 normalized `(cx,cy,w,h,angle)` prediction 乘 `[img_w,img_h,img_w,img_h,angle_factor]`，再调用 `self.assigner.assign(pred_instances, gt_instances, img_meta)`；`assign_result.gt_inds[pos_inds]-1` 就是 matched GT index。Task 3 应抽取/复用这一调用顺序并把 GT regularization/normalization保持一致。
- Frozen split 来自 `s1_train_all18` / `s1_val_all18` 数据目录，OMQ config 只记录 manifest SHA；Task 3 在启动前必须把实际 image-id 顺序绑定到可复算 manifest/hash，而不是只信目录长度或 distributed sampler。
- OpenRSD 今日明确存在 CSER Phase-1 资产：设计/计划、10-GPU config/launcher、candidate/control 日志和 `resultmd/exp_cser_phase1_20260803/fres_cser_phase1_frozen400.md`；这是一条 DOTA1/S2 frozen400 证据线，不能与 OV-CapFlow raw DOTA-v2 13,833/Q600 mouth 横比。
- OpenRSD CSER Phase-1 权威同口径结果为 control/candidate `mAP=0.7313/0.7314`，delta `+0.0001`，远低于预注册 `+0.0030`；small-vehicle `0.6393→0.6409`（`+0.0016`），storage-tank 反而 `0.7530→0.7531` 的表面稳定之外，多个类正负混合。4099 个 adapter 参数确实更新，故是“活跃但排序收益不足”，决策 `HOLD / 不晋级`。
- CSER 使用 OpenRSD RTMDet-style head、visual 8-shot、DOTA1 15 类 S2 可恢复 `2497/2500`，训练来自 DOTA2 formatted PKL；它可作为“共享正 support 低秩反证不足”的跨模型负证据，但不能证明 OV-CapFlow 的 Q600 decoder source 机制，也不能支撑 strict no-NMS E2E claim。
- 跨项目一致教训：M1 与 CSER 都排除了“参数未更新”，但仅用强压缩的全局统计/共享低秩残差都无法产生 AP 级改善；下一步应先检验信息 actionability 与 held-out generalization，不应继续堆 adapter/router/head。
- ODQ-R1 初筛：`.lab/results.tsv/.lab/log.md/.lab/branches.md` 没有任何 ODQ 结果条目；能定位到的是隔离 `research/odq-r1` worktree 的实现、supervisor、Stage-0、配置与测试，以及一份 `full_e6_gpu0_7` config。当前未发现同名 work_dir 结果，因此必须读冻结 supervisor 计划和 worktree 状态后，才能判断它是未启动、失败未记账还是被 OMQ 主线正式取代。
- ODQ 冻结合同本身禁止在 Stage-0 五项 PASS 与 supervisor dry-run audit 前训练；首轮只允许 x/y/w/h distribution residual，angle/Q600/grouped/DN/matching/readout 保持父路径，且禁止 quality head、calibration、NMS/top-k 等叠加。该设计与当前 strict mouth 相容，但尚不能仅凭实现/配置 commit 视为实验完成。
- 更正 ODQ 初筛：实际 full-E6 产物位于 `work_dirs/odq/full_e6_gpu0_7_20260802_121440`。8-GPU 训练从 E1 `mAP=0.356974006/AP50=0.3570` 运行到 E6 `mAP=0.540336192/AP50=0.5400`，完整写出 E1/E6 checkpoint；1128 条 monitor 记录无 fatal pattern，显存/利用率证明 GPU0–7 确实参与，任务是正常完成而非运行故障。
- ODQ 的冻结 full-E6 control 是 T7 E6 `mAP=0.553674161/AP50=0.5540`；因此 ODQ total delta=`-0.013337970`，已在首个必须满足的 `mAP delta >= +0.010` 门上失败，不能晋级 E12/E24，也不能以 AP50 或选取中途点救援。计划要求的 registered Q600 diagnostics 目录当前还不存在，这构成额外 contract 缺口；但即使忽略该缺口，total gate 已足以作科学 FAIL 决策。
- Actionability 论文一致性审查发现 S1 held-out 假设失效：`build_dotav2_cleanstart_subset.py::_select_s1` 只在 density-bin 内按 tile stem 随机切 80/20，没有原始 scene group 隔离。主代理已从 runtime manifest 独立复算并精确复现：164 个原始场景跨 split，400 个 val tile 中 322 个与 train 共享场景，86 个有实际像素重叠、105 对重叠，最大 crop IoU=`0.9711260828`。runtime manifest 与历史副本 SHA256 均为 `a00b945ddd0a769008d57145feba28382fb9e56f5d6f1427413220f8008e1a2e`。这已触发 fail-closed：不得在现有 S1 上声称 held-out actionability，也不得启动 Task 3 GPU capture。
- Actionability assignment 还必须固定一次 actual assigner 产生的 query→GT id，并在 decoder layer、feature family 与 paired-placebo 之间复用；当前 head 的 boolean matched mask 不具备这个身份约束。纯 sufficient-statistics 数学核心不依赖 split，可继续完成和审查；数据捕获与 gate 必须等待 scene-disjoint、zero-pixel-overlap manifest 重新冻结。
- mAP-first 容量锚点可行性：本地官方 GroundingDINO 仓库提供 Swin-B config，README 给出的官方 B checkpoint 是 `groundingdino_swinb_cogcoor.pth`（COCO/O365/GoldG/Cap4M/OpenImage/ODinW-35/RefCOCO generic pretraining），但本地尚未下载。现有 cleanstart converter 按目标 config 做 exact name/shape 过滤并支持 Q900→Q600 contiguous query transport，可复用于 B，只需新的 target config/provenance；未触碰权重或产物。
- CPU 构建审计显示在保持 decoder/text/Q600 mouth 不变时，Swin-T parent 为 `172,844,764` 参数（backbone `27,520,506`），Swin-B override 为 `232,908,890`（backbone `86,880,120`）；B 结构可被当前 MMRotate/MMDet 环境成功实例化，neck 输入需 `[256,512,1024]`。这只证明结构兼容，不证明 checkpoint coverage、A40 训练显存或 mAP。
- 论文主线审查收敛到 incomplete-label negative-safe set prediction：严格 base14 训练时，未标 novel 实例周围的 unmatched Q600 是否被错误作为背景持续惩罚；可用两视图 inverse-warp 后稳定的 rotated query trajectory 只屏蔽可疑负梯度，不产生正伪框、teacher、新 head/router 或 inference 后处理。它与已失败的 per-query 表示/几何/readout 残差路线正交，但必须先建立 scene-disjoint strict base-only substrate 和 sealed novel final protocol。
- OMQ Task2 纯数学原语已提交 `06d070a` 并由主代理 fresh 验证：聚焦套件 `69 passed, 8 warnings in 3.77s`，`py_compile` 与 `git diff --check` 通过。实现正确区分 raw cross moment `E[dx*dy]` 与 centered covariance，并把 placebo 的 axis0=image/同 split/不同 scene 明确写成 caller contract；它不包含 manifest、scene、assignment/shard、image-macro 或最终 gate 校验，因此只保留为基础设施，不改变 OMQ 的 `STOP_CONTRACT_FAIL`。
- 新两阶段设计通过三轮论文一致性审查：B0 用官方 GroundingDINO-B 做 all18 closed-set raw mAP 强锚点；未来 N0/N1 从独立 generic clean-start 重启，绝不继承 B0 权重、缓存、epoch 或结果选择。N0 只用 base14 内预注册 meta-novel folds，official novel4 在方法/阈值/控制/seed 冻结前完全封存；N1 只能称 prediction-derived negative masking/no positive pseudo-box targets，不能再泛称 pseudo-label-free。
- B0 冻结停止门：E1 mAP `<0.4000` 停；E6 同时要求 `>=0.5800` 且比 T7 E6 精确值高 `>=0.0200`；E12 `>=0.6500`；E18 要么比 E12 `+0.0100` 要么已达 `0.6900`；E24 端点同时要求 exact mAP 与 rounded AP50 `>=0.7000`。所有 checkpoint/source/config/manifest/mouth/memory/sampler gates fail-closed，无救援与中途点选择。
- B0 实施计划经间歇论文守门复核 PASS；每份结果必须固定写 `closed_set_all18_anchor=true / strict_ov_evidence=false / innovation_claim=false`。未来 novel seal 的可审计定义是不读取 official-val novel4 annotation geometry/count/metric；公开类名字符串本身不伪装成秘密。
- B0 sampler 算术预检提前发现 world10×batch1 必然失败：47,294 对 world10 余4，现有 DDP sampler 不允许 6 个 rank 在末步为空，也不允许 pad/drop。自行审批的最小无损修订是 world10×batch2、accumulation1、update_multiple1、Swin `with_cp=True`；末个全局 batch=14，可让十个 rank 各1–2真实样本。该配方每 epoch optimizer updates 多于 T7，明确只作 non-causal anchor，必须过 PyTorch1.12 checkpoint/DDP 和 10%显存余量实测。
- 官方 GroundingDINO-B GitHub release 已完整下载：`938,057,991` bytes，checkpoint root 仅 `model`，1108 个 tensor keys，SHA256=`46270f7a822e6906b655b729c90613e48929d0f2bb8b9b76fd10a856f3ac6ab7`；先落 `.part`，hash/torch.load 通过后才原子改名为 `/data/zcy/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth`，未覆盖任何既有 checkpoint。
- batch amendment 经论文守门修订后 PASS：配置路径统一 batch2；真实 DDP 预检必须连续至少3个 optimizer updates，并覆盖 global20 与 tail-like global14（十rank各1–2样本），否则单步 smoke 不足以排除 PyTorch1.12 reentrant checkpoint ready-twice/hang。
- B0 config/cleanstart gate 通过：commit `9e6c504`，主代理 fresh 回归 `76 passed`；真实 sampler 47,294 exposures→2365 updates，global14–20/local1–2，missing/duplicate均0。官方 B 转换 coverage=`0.9992707574`，1053 matched keys/233,236,344 matched numel；24 missing、4 unexpected、14 shape mismatches 精确等于既有 allowlist。compatible SHA=`86a7d0b9…e7056`，provenance SHA=`267e085a…5777e`，payload 仅 `state_dict`。
- 更正 sampler 数值：上述 2365/global14 是无GT synthetic exposure arithmetic；真正构建 rare4x dataset 后，23 个 query-budget shrink updates 令冻结 world10 plan 为2366 updates、global17–20/local1–2，仍 missing=duplicate=0、coverage checksum 不变。max-GT=3396（index12912），max-query-area singleton对应 index17819。正式 no-clobber 路径 commit `448a153`，formal compatible SHA仍 `86a7d0b9…e7056`，新 provenance SHA=`f84c476e…64e1e`。

## 2026-08-03 — paper conclusion hierarchy before B0 execution

- **Current evidence:** E24 is an all-18 supervised closed-set substrate at exact raw mAP `0.6064053488274416` (`AP50=0.6060`) on the full `13,833 x Q600` mouth. Its defensible systems result is rotated-5D direct set prediction with all-query scoring and no proposal/top-k/NMS inference; base14/novel4 slices are not open-vocabulary evidence.
- **Conditional B0 conclusion:** only if the immutable E24 gate passes may the paper state that official generic GroundingDINO-B initialization and the frozen all18 recipe can jointly exceed exact raw mAP and rounded AP50 `0.7000` under the same mouth. Because capacity, pretraining, global batch, and optimizer-update count differ, B0 remains a non-causal closed-set engineering anchor rather than a Swin-B causal ablation, strict-OV result, or innovation.
- **Method conclusion still requiring new evidence:** the ICLR-level method hypothesis remains incomplete-label negative-safe fixed-set prediction. It needs a separately initialized, scene-disjoint, base-only/meta-novel N0/N1 evidence chain with official novel4 sealed; B0 weights, cache, metrics, and selection decisions cannot cross that firewall.
- **Forbidden claims:** no broad `first`, no zero-shot label for an all18-trained novel4 slice, no claim that B0 validates negative-safe learning, and no extrapolation from the closed D11/D12/D13-N/QAF/M1/ODQ/CSER recipes to an entire mechanism family.
- B0 single-GPU Stage-0 is real and passes: one loaded model sequentially optimized the registered typical batch2, max-query singleton, and max-GT singleton. All three produced finite loss/gradients, a nonzero clipped update, and exact Q600 rotated-5D predictions. Worst reserved memory was `37476 MiB`, leaving `8592 MiB` / `18.65%` of the frozen 46068 MiB A40 budget; report SHA256=`01d42543971594063fd4ffa410794fafeb801393efa652c1835bb9ad5854c502`.
- B0 world10 smoke r2 fails deterministically before its first update: 10/10 ranks report PyTorch 1.12 `Expected to mark a variable ready only once`, parameter index 325, for reentrant `with_cp=True` under `find_unused_parameters=True/static_graph=False`. No PASS JSON or formal training artifact exists. The only bounded runtime amendment is `static_graph=True`, explicitly suggested by the framework error and leaving all scientific ingredients fixed; it must pass the identical three-update smoke or B0 stops.
- The single-field `static_graph=True` amendment passes both final-config gates: single GPU worst reserved remains `37476 MiB`, while the identical ten-rank updates 0/1/2365 (global20/20/17) complete with zero ready-twice/fatal and at most `19700 MiB` reserved per rank. Formal r3 is consequently live on GPU0--9; its epoch0 audit exactly reproduces the frozen real plan, and Epoch1 iter40 reports finite loss `50.0203` with independent monitor fatal-null.
- B0 formal r3 运行时监控在 E1 iter320 前保持十卡进程存活、fatal-null；最新可见 loss=`30.8191`、grad_norm=`101.3346` 均有限，但这些只证明执行健康，不是 E1 mAP。`nvidia-smi memory.used` 曾在 GPU5 采样到 `42816/46068 MiB`，它不是预注册的 per-rank `torch.max_memory_reserved` preflight 指标，且当前没有 CUDA OOM/allocator/rank fatal；因此按冻结口径继续，但不得声称正式全程保有 `>=10%` 在线显存余量。
- 自动守门器 commit `eb2bde8` 的第一轮独立复审发现五类 Important 工程风险：已有终态/目标死亡时 watch 可能永久等待、已有决策前缀状态机校验不足、指标 authority 可经 symlink/路径逃逸、stop 动作先于 durable journal、最终 identity check 与 tmux send 存在 TOCTOU。阈值逻辑本身与冻结 spec 一致；在这些问题经 TDD 修复和复审前，守门器不进入 enforce，正式训练与独立只读 monitor 不受影响。
- 守门器修复 commit `6fcd0982` 由主代理 fresh 验证 `45 passed`、`py_compile`、`diff --check`，并获独立复审 PASS（0 Critical/Important）。journal 锁现在覆盖 intent fsync→identity recapture→server-side tmux condition→outcome fsync；已有终态、target-dead、非法状态前缀与 authority symlink/escape 均 fail-closed。精确绑定的 enforce watch 已在 tmux `$42/%52` 启动，绑定正式 `$40/%50`、PID391764/start_ticks71190087；receipt SHA=`6a2f3d2b…1db3a`。残余采用 at-most-once：durable intent 后崩溃或 tmux 客户端报错时不重发，信号交付状态可能 unknown；这不会造成重复停止。
- Task6 scene-only 合同 SHA=`5497bb6e…155998` 经论文审查 PASS；M0 builder 经三轮 TDD/复审最终落在 commit `6822133b`，主代理 fresh `28 passed`。真实 M0 仅用 `scandir+lstat` 在2.6秒内封印47,294 train tiles/1,664 scenes和13,833 official-val filename inventory：train/dev/test为1157/262/245 scenes与35109/5172/7013 tiles，所有声明的 scene/stem/crop/official-val overlap 为0。`COMMITTED.json`/manifest/inventory SHA分别为`2ae10b3a…8140d`、`0c2382ed…d13aa`、`bafb17b7…09ba8`；输出明确 `preparation_only=true`、`training_use_forbidden=true`、content/header seal incomplete、N0/N1未授权，不能作为 strict-OV 或方法证据。
- B0 formal r3 E1 在完整 `13,833 × Q600` raw mouth 上得到 exact mAP `0.44958066940307617`、rounded AP50 `0.4500`，以 `+0.0495806694` 通过 `0.4000` 可继续门；supervisor 记录 `stop_requested=false/ctrl_c_sent=false` 后训练无缝进入 E2。checkpoint 为2,800,650,093 bytes、SHA=`e55d1e4d…95388`；mouth audit 47,294/13,833、empty 22,575/7,228、18类、missing/unpaired=0，SHA=`286fbdba…920ba`；sampler audit SHA=`9057d97d…d263c`。按三位 console AP 重建的 novel4/base14 均值 `0.35225/0.477571` 仅是 all18-supervised diagnostic，不是 OV 证据。E1 只授权 unchanged 到 E6；下一门必须同时满足 mAP `>=0.5800` 与相对 T7 E6 `>=+0.0200`，从 E1 至少还需 `+0.1304193306`。
- E1 论文一致性复审 PASS（0 Critical/0 Important/2 Minor）：18 类 detection rows 精确合计 `8,299,800=13,833×600`，并与 config hash、692/692 progress、dataset-mouth audit 共同构成 runtime mouth 证据；单独的 mouth JSON 只审计 dataset/count/class/pairing。official novel4 的 seal 精确定义为不允许其信息参与 N0/N1 问题选择、阈值、控制或 PASS/FAIL，而不是声称 B0 all18 diagnostic 从未可见。M0 仍不污染 seal。

## 2026-08-12 — Three-month retrospective: initial evidence notes

- Audit window is fixed to `2026-05-12` through `2026-08-12` (Asia/Shanghai).
- The existing ledger already records a major methodological shift: early OpenRSD-style rotation/text/false-positive interventions were repeatedly closed by full-evaluation evidence; OV-CapFlow later moved toward fixed-query all-row evaluation, provenance sealing, parent-preserving interventions, and explicit negative-result gates.
- Existing OV-CapFlow evidence must be normalized by protocol before comparison. In particular, `raw-13833`, `filtered-6605`, proxy subsets, all18 closed-set slices, and strict base/novel evidence are not interchangeable.
- Current work is read-only. Existing dirty files, checkpoints, active experiment state, and user-owned planning records are preserved.
- OpenRSD is too large for a naive chronological file dump: `resultmd` is about 5.2 GiB with 10,178 files, and `work_dirs` is about 277 GiB. The audit therefore must route through summaries/indexes and then verify selected claims against raw tables/logs.
- The earliest in-window cluster (2026-05-12 onward) contains DOTA1 rotation baselines, prompt/feature rotation sensitivity, P0148 stage probes, the small-vehicle semantic-attractor diagnosis, causal embedding interventions, and several attempted calibration/background-prototype/anti-hub repairs. These form one coherent early phase rather than independent paper ideas.
- Broad mtime scans over `.lab` are noisy because captured per-image `.npz` artifacts dominate. OV-CapFlow reconstruction should instead use `.lab/results.tsv`, `.lab/summary.md`, `.lab/branches.md`, curated `docs/project_history`, and exact experiment reports.
- Curated OV-CapFlow branch ledger gives the first reliable coarse timeline: HRSC parent-preserving fusion improved strict all-query AP50 from C0 `0.5870` to `0.6290` and repeated at `0.6210` (two-seed mean `0.6250`), but matched DOTA2 transfer was much weaker (`mAP 0.1253`, `+0.0146` over C0). This is evidence for a transferable local mechanism, not a strong DOTA2 detector.
- The DOTA2 engineering anchor replay reached `mAP/AP50 0.6568/0.6570`; the later clean-start Q600 T7 line's canonical Epoch24 authority is exact raw `mAP 0.6064053488`, `AP50 0.6060` on all 13,833 validation tiles. These are different parents/routes and must not be presented as monotonic progress.
- Several later branches are high-value negative evidence: A1 target-only position supervision ended at matched proxy `0.4643`; pure final-center reordering was closed because only `22.256%` of geometry misses had a usable final center; OMQ M1 was near-neutral (`delta -0.000307`) and its frozen split had scene/pixel leakage; ODQ full E6 was `0.540336`, or `-0.013338` versus T7 E6.
- The Swin-B B0 route is explicitly a non-causal all18 closed-set capacity anchor. Its verified E1 is `0.449581/0.4500`; later status in the ledger was still running toward E6, so no E6/capacity conclusion may be inferred from E1 or runtime health.
- The August RISC derivative changed from generic low-rank intervention to a bounded angle-conditioned rank-8 symmetric projection (`2,089` trainable scalars) on a raw-13,833 `0.62009567` parent, then evolved into paired-orbit/POQ training. Its scientific outcome must be read from later endpoint reports, not from numerous implementation commits or early finite-loss checks.

### OpenRSD phase findings verified from curated reports

- May's strongest mechanistic finding is narrower and more useful than the original broad story: a `small_vehicle` attractor is already visible in dense classification logits, with NMS/top-k acting mainly as an amplifier. Text-encoder swaps barely changed the phenomenon, whereas visual-support swaps collapsed P0148 SV predictions from about `0.75` to `0.06–0.09`; however, those swaps also destroyed held-out AP (`0.674 → 0.092/0.096`). Thus the mechanism is diagnostic, not a deployable module.
- The best early training repair, B8k/DeHub, reported heldout-500 AP50 `0.738 → 0.805`, SV AP50 `0.741 → 0.842`, and P0148 dense SV near `0.005`, but also a documented `CLASS_DRIFT_RISK` (low-risk detections expanded by about 61%). Post-hoc calibration/ensemble repairs reduced the stress-tile SV ratio yet produced no robust held-out AP gain (`n=40` best delta `-0.009`; `n=60` methods approximately tied).
- May's engineering lesson is scientific, not merely operational: several apparent findings were invalidated or weakened by checkpoint non-injection, CSV overwrite, GPU mismatch, daemon/DataLoader failure, missing true-SV GT linkage, and incompatible evaluation sample sizes. Provenance and protocol controls therefore became part of the method-development process.
- June's text/rotation interventions repeatedly underperformed the frozen text-only anchor: candidate short runs were roughly `0.646–0.650` versus `0.6962`; an SV adapter collapsed to `0.4870`. Frozen-BN/no-op controls showed that training-mode BN drift alone could move AP materially (`lr0 no-op 0.6448`, frozen-BN `0.6597`), so short-run screening without a matched frozen-BN control was unreliable.
- The Gaussian/support/ranking family mostly found tiny or confounded gains. P12B reached `0.8745` on HRRSD but had a four-GPU/global-batch confound; SGP variants remained below the `0.8307` baseline. This phase taught that lower risk/SISE diagnostics are not equivalent to higher AP and that every auxiliary score needs a matched AP control.
- A separate structural breakthrough came from abandoning the dense-head/NMS mouth. On HRRSD, P15O-B strict set prediction reached `mAP 0.8865` versus the OpenRSD `0.8562` baseline while improving measured GPU latency (`15.42` vs `19.82 ms/img`); this was the real conceptual bridge toward OV-CapFlow.
- DOTA2 then exposed the data/optimization threshold of this route: a 500-image/80-epoch strict E2E model reached only `0.1394`, whereas the 1,000-image curriculum reached `0.6008` versus its subset OpenRSD baseline `0.5641`. Small proxy failure therefore could not be used as a route-level falsification; minimum data scale and convergence time were themselves causal variables.

### OV-CapFlow July transition findings

- The July 1–14 audit explicitly separates four mouths: `DOTA1000`, raw `13,833`, filtered `6,605`, and HRSC single-class. Its central correction is that the apparent AP70 results (`~0.7004–0.7064`) came from filtered or multi-source/router settings; they were not strict raw-full-val single-checkpoint results.
- The old strict no-head line plateaued near `0.40–0.43` even after GT pre-topk, Q changes, long training, full 47,294-image training, teacher positives, density losses, and dense-distinct constructors. This validly rejects several *specific implementations* and supports a structural bottleneck: missing query birth, local quality/objectness, instance ownership, null calibration, and continuous capacity.
- Historical structural assets were not all discarded. P149/P156/P167/P170/P175 supplied useful evidence for stable query birth, slot state, lifecycle, competitive memory, and output-budget control, while exact overwrite/suppression/post-decoder-perturbation topologies were paused with explicit revival conditions.
- OpenSetFlow's R25 query-preserving fusion improved over earlier semantic-overwrite variants, but DOTA2 remained below its frozen parent. R29 balanced reduction produced the clearest DOTA2 causal improvement (`E5 0.3698`, `+0.0205` vs matched control) and repaired empty-tile/gate drift, yet projection conditioning deteriorated. This is why OV-CapFlow inherited query identity and balanced capacity, but required null/readout/conditioning gates rather than blindly inheriting the whole stack.
- The July design's strongest methodological discipline is a claim chain, not a module list: `query identity + balanced evidence capacity + explicit null -> ownership/calibration -> strict OV oriented set prediction`. Every later module should be retained only if it moves both the mechanism mediator and the final AP under the same mouth.

### Late July–August findings

- The current ten-day authority overturns stale live-state assumptions. Verified T7 Epoch24 is raw-13,833 `0.6064053488`; Swin-B B0 later reached E6 `0.5760012269`, beating matched T7 E6 by `+0.022327` but missing its absolute `0.5800` gate by `0.003999`. It is therefore a near-miss capacity anchor, not a passed method result.
- The only clear late-window positive endpoint is the world8 rotation continuation (`0.62009567` vs matched no-rotation `0.61037731`, `+0.009718`), but the world1 E2 pair reversed direction (`-0.017873`). Rotation is a promising interaction/robustness factor whose effect is entangled with world size, effective batch, sampler, seed, or training length.
- A sequence of parent-preserving query-space interventions was legally negative or neutral: D11 `-0.0139`; D13-N `-0.009181` on its own proxy; OMQ-M1 `-0.000307`; ODQ `-0.013338`; RISC-E0 `-0.004373`; RISC-Orbit `-0.000980`; hard OBF/GC about `-0.033/-0.032`; soft projection exactly `0.000000`. This supports stopping the search for a generic post-hoc “rotation subspace” adapter.
- OMQ provides a particularly reusable lesson: its source audit passed (`0.5010` reach vs roughly `0.254–0.257` placebos), but detector action was neutral-negative and the later actionability split leaked 164 scenes / 86 pixel-overlap validation tiles. Signal existence, causal actionability, and protocol validity are three separate gates.
- The current scientific priority in the authority document is strict-OV negative-safe learning: scene-disjoint data, independent generic initialization, base-only/meta-novel development, official novel4 sealed, and a matched N0/N1 pair. M1 content/header sealing later completed over 47,294 images with zero cross-partition content-hash overlap, but it explicitly did not authorize training.
- The closest-work gate narrowed the claim to fixed-Q600 rotated OV set prediction where cross-view-stable evidence may identify harmful false-background gradients without producing positive pseudo boxes or adding top-k/NMS/proposals. N0-RI supports a rotation-interference diagnostic but used an all18-trained source, so it is not strict-OV evidence.
- RISC-ER (2026-08-11) is currently an approved design and implemented/smoked mechanism, not an AP result. The E12 filtered-6605 source is `mAP 0.6340868473`; its eight-GPU two-iteration smoke eventually passed engineering checks, then work stopped by explicit scope before 12E/raw-13,833 evaluation. Any claim that RISC-ER improves AP, P0148, or open vocabulary would be fabricated at this point.
- The immediate pre-RISC-ER parent line changed to a single official MM-Grounding-DINO-T source. It reached filtered-6605 `0.607968` at E6 and `0.634087` at E12; the same E6 checkpoint was raw-13,833 `0.579411`. No matched “same parent/schedule without POQ” endpoint exists, so the gain cannot be attributed to POQ. It is best treated as a stronger current substrate and as further proof that parent/protocol choice can dominate a small method delta.

### Initial current-literature screen for the derivative detector

- Plain HBox-to-OBB by rotation/flip consistency is already strongly covered by H2RBox and H2RBox-v2; H2RBox-v2 reports near fully supervised performance across DOTA/HRSC/FAIR1M. A proposal whose novelty is merely “use rotated views to learn angle from HBoxes” is therefore not defensible.
- A generic unified mixed-supervision framework is also covered by Wholly-WOOD (points, HBoxes, RBoxes, and mixtures). Unifying annotation types alone is not a sufficient contribution.
- Point-supervised rotation consistency is crowded: Point2RBox already combines transformed-view self-supervision with synthetic-pattern box knowledge; PointOBB-v2 and Point2RBox-v2 (2025) further strengthen point-only pseudo-RBox generation/spatial-layout modeling. “One point + rotate consistency” would be a direct prior-art collision.
- A viable differentiation must target a failure the above families do not make central: fixed-query one-to-one ownership under dense scenes, no pseudo-RBox confirmation bias, annotation-consistent latent geometry across a whole rotation orbit, or negative-safe treatment of unowned queries. This aligns much better with the project's actual evidence than adding another feature-consistency loss.

## 2026-08-12 — Final retrospective synthesis and route decision

- The three-month trajectory is best summarized as: semantic-attractor causality → deployment/mouth discipline → strict one-to-one rotated set prediction → fixed-query ownership/null/capacity → supervision-contract and negative-safety.
- Recommend HBox-supervised OBB as the derivative paper route. Full RBox supervision should be the parent/oracle; point supervision should remain a later adapter. Generic mixed supervision is too close to Wholly-WOOD/PWOOD for a primary claim.
- Provisional framework name: HALO-Set, or Horizontal Annotation Latent-Orbit Set Prediction. It treats an HBox as a feasible family of OBBs, marginalizes K orientation/shape particles, tracks annotation identity across rotation/flip views, and separates safe-background negatives from ambiguous local contenders while retaining one final owner.
- The narrow needs-search claim is the joint formulation of latent HBox geometry, orbit-consistent one-to-one ownership, and ambiguity-aware negatives in a fixed-query rotated set predictor. No `first` claim is authorized.
- First authorized future action after user approval should be an offline identifiability audit, not training: quantify HBox→OBB ambiguity versus angle/aspect/density and test whether hard pseudo-angle errors concentrate in the predicted regimes.
- Full report: `docs/project_history/exp_20260812_three_month_review/fres_20260512_20260812_openrsd_to_ovcapflow_review_and_detector_proposal_zh.md`.
- Literature package: `docs/literature-search-20260812-weakly-supervised-oriented-set-detection/` (25 screened, 15 retained).

## 2026-08-12 — CNN-HBox v2 synthesis

- The user-selected architecture boundary is now decisive: the main paper should use a CNN dense detector. Fixed-query work remains valuable as a source of failure modes and experimental discipline, not as a decoder/query design.
- The cleanest primary parent is H2RBox-v2 / Rotated FCOS R50-FPN. The local head uses center sampling, regression ranges, and minimum-area conflict resolution; unselected locations are background for focal classification. Annotation IDs aggregate angle predictions across original/rotated/flipped views, but the dense assignment field itself is not aligned across views.
- BGHR is the closest collision and makes a generic “better HBox sample assignment” claim indefensible. Its AFSM/PRA selects a best predicted RBox per GT HBox and then forms hard positives. Any new method must differ at the relation level, not merely use a different quality score.
- The revised hypothesis is that the hidden variable under HBox supervision is both orientation and location-instance ownership. A cross-view consensus soft ownership plan can preserve uncertainty before it becomes a hard predicted RBox; plan entropy can distinguish safe background from ambiguous HBox-interior negatives.
- Proposed framework: `OrbitAssign`, with only two core changes over H2RBox-v2: orbit-aligned soft instance assignment and ambiguity-safe background learning. Existing HBox projection and rotate/flip symmetry remain baseline machinery.
- No Transformer decoder, fixed queries, Hungarian matching, global Q budget, NMS-free inference, point adapter, teacher, or hard pseudo-RBox is part of v2. Standard rotated NMS remains for fair CNN evaluation.
- Rotated RTMDet is the second-stage portability parent because its native dynamic soft-label assigner would confound the first FCOS causal test.
- The primary novelty language is narrow and provisional: the date-bounded search did not find an exact method centered on transformation-aligned soft location-instance assignment distributions plus HBox-interior ambiguity-weighted background gradients. No `first` claim is authorized.
- Focused literature audit screened 26 papers and retained 18 traceable primary works. Dynamic/soft/global assignment, box equivariance, scale correction, unified supervision, and sparse-label negative safety are all occupied territories.
- Design deliverable: `docs/project_history/exp_20260812_three_month_review/fres_20260512_20260812_openrsd_to_cnn_hbox_obb_proposal_v2_zh.md`.
- Focused literature: `docs/literature-search-20260812-cnn-hbox-oriented-assignment/`.

## 2026-08-12 — FMSA fusion and OrbitAssign v3 synthesis

- The useful FMSA contribution is not its full point-supervised pipeline but the principle that weak orientation, scale, boundary, density, and instance evidence have different reliability. Under full HBox supervision, point-to-region warmup, class prototypes, hard pseudo OBBs, threshold filtering, and EMA teacher–student add confounds without addressing the clean ownership question.
- The assisted draft contains several assumptions that require correction before reuse: MessDet uses eight orientation dimensions in its reported experiments; FAA performs a 2D spatial Fourier transform rather than a 1D MessDet group-axis transform; an unoriented OBB axis requires explicit double-angle treatment; batch-max amplitude is not calibrated confidence; nearest-point distance is not object radius; and H2RBox-v2 is not a generic EMA-teacher citation.
- The optimized hypothesis factorizes three latent relations: location-instance ownership, cross-view orientation stability, and background safety. These must not be collapsed into one quality score because low directionality does not imply background and stable direction does not resolve overlap ownership.
- The recommended method remains CNN/H2RBox-v2-based and is renamed only internally as `OrbitAssign-ER`. Conflict-local soft transport operates on annotation-only HBox support-overlap components. A parameter-free second-order axial circular coefficient measures transformed-view angle concentration, but it is used only to route geometry evidence and never as a hard pseudo angle or correctness probability.
- Loss routing is the central upgrade: ownership evidence weights positive classification/centerness, ownership plus axial stability weights envelope/angle/symmetry geometry, and a separately capped background-safety weight controls HBox-interior negative gradients. Minimum coverage remains independent to prevent starvation.
- FMSA-style scalar quality is retained as a required negative/control ablation, not as the proposed method. MessDet group-harmonic evidence is deferred until the same-parent FCOS mechanism passes and the group phase law is verified by synthetic transformations.
- Incremental primary-source screening adds MessDet (ICCV 2025), Fourier Angle Alignment (CVPR 2026), PointOBB-v3, Point2RBox-v3, and the closed-access TIP 2025 `Instance-Level Orientation Enhancement` paper. The last item is a mandatory full-text novelty risk before submission.
- Deliverable: `docs/project_history/exp_20260812_three_month_review/fres_20260512_20260812_openrsd_to_cnn_hbox_obb_proposal_v3_fmsa_fusion_zh.md`.
- Literature addendum: `docs/literature-search-20260812-cnn-hbox-oriented-assignment/fmsa-fusion-addendum.md`.
