---
title: "OV-CapFlow 2026-07-29 至 2026-08-07 十日实验全面复盘、思考与后续工作记忆"
document_role: "fres_final_review_and_active_working_memory"
language: "zh-CN"
date_range_start: "2026-07-29"
date_range_end: "2026-08-07"
created_at: "2026-08-07 Asia/Shanghai"
dynamic_snapshot_at: "2026-08-07 23:58 Asia/Shanghai"
last_evidence_revision_at: "2026-08-08 Asia/Shanghai"
status: "ACTIVE_AUTHORITY"
canonical_for_window: true
ai_scan_priority: "P0"
ai_scan_when:
  - "恢复 OV-CapFlow 实验"
  - "扫描 work_dirs"
  - "提出新实验"
  - "解释过去十天结果"
  - "撰写 strict-OV 或 negative-safe 论文内容"
active_research_priority: "A_strict_open_vocabulary_negative_safe"
metric_authority: "真实日志、scalars.json、.lab/results.tsv；禁止仅凭目录名推断"
dynamic_state_warning: "POQ 为运行中快照，后续 AI 必须重新读取实时日志"
---

# OV-CapFlow 十日实验全面复盘、思考与后续工作记忆

> [AI-SCAN-P0] 本文是 2026-07-29 至 2026-08-07 时间窗的首要扫描入口。
> 任何后续 AI 在提出实验、恢复训练、解释结果或撰写论文前，都应先读本文，
> 再按第 12 节的 P0 → P1 → P2 顺序核验原始证据。

> [DECISION] 用户已选择下一阶段优先级 A：
> 严格开放词汇（strict open vocabulary）与 negative-safe 学习证据优先。
> raw mAP 工程冲高和 RISC/旋转机制深挖降为辅助路线，不能反客为主。

> [CLAIM-BOUNDARY] 当前全部 DOTA-v2 all-18 训练结果，包括 T7、B0、RISC
> 与旋转增强，只能作为闭集系统、容量或机制筛选证据。它们不能被描述为
> strict-OV、zero-shot、novel-category generalization 或论文创新已经成立。

> [REVISION-POINTER 2026-08-08] 第 7.1、7.2、9.1 和 13 节中的 M1/文献门
> 状态是 2026-08-07 快照。恢复工作时必须同时读取第 15 节：M1 content/header
> seal 与 closest-work hard gate 已完成，N0-RI 四折确认也已通过；但 N0-RI 的
> source 见过 all-18，不能替代 strict-OV seen-only N0。

## 0. 阅读规则与证据标签

本文明确区分以下标签：

- [FACT]：由真实日志、指标 JSON、checkpoint/audit 或不可变台账直接支持。
- [INFERENCE]：由多个事实共同支持，但仍需新的因果实验确认。
- [DECISION]：当前研究决策、停止门或优先级。
- [OPEN]：尚未回答，禁止用训练 loss 或直觉填补。
- [DO-NOT-REPEAT]：当前协议下已经关闭，不允许无新假设地重跑、调参救援或堆叠。
- [AI-SCAN-P0]：恢复任务时必须扫描。
- [AI-SCAN-P1]：涉及对应结论或复现实验时扫描。
- [AI-SCAN-P2]：仅在定位实现、运行时故障或复算哈希时扫描。
- [CLAIM-BOUNDARY]：论文中允许和禁止的表述边界。

### 0.1 统计口径

[FACT] 本文采用 2026-07-29 至 2026-08-07，含首尾日期，共十个自然日。

[FACT] “研究节点”按一个可以独立做出 PASS、FAIL、discard、neutral、
contract-fail 或 pending 判定的问题计数。以下项目不单独计为科学实验：

- 设计文档、计划文档和代码提交；
- 无 optimizer update 的预检；
- 同一科学问题下的兼容性重启；
- 仅验证显存、sampler、checkpoint load 或 DDP 行为的工程 smoke；
- 同一指标 JSON 在 202608*.json 与 scalars.json 中的重复记录。

[CLAIM-BOUNDARY] proxy400、proxy1600×400、raw-13,833、Swin-T、Swin-B、
不同 world size 与不同 epoch 的数字不可直接求平均。本文只在预注册 matched
control、明确父 checkpoint 或相同 raw mouth 内报告 delta。

## 1. 十日总览

### 1.1 一句话结论

[INFERENCE] 过去十天最重要的产出不是找到一个已经可发表的新模块，而是形成了
一组相当一致的排除证据：generic query 初始化、terminal-XYWH transport、
null/existence residual、query evidence flow、ODQ、一般低秩残差、硬子空间删除、
group-conditioned projection 和软投影都没有在冻结端点稳定改善最终 AP。

[FACT] 唯一清晰的正向性能端点是 world8 旋转增强：

- rotation：raw mAP 0.6200956702；
- no rotation：raw mAP 0.6103773117；
- matched delta：+0.0097183585。

[FACT] 但单卡两轮配对得到相反方向：

- rotation：0.5860218406；
- no rotation：0.6038943529；
- delta：-0.0178725123。

[INFERENCE] 因此“旋转增强有效”目前是值得追踪的信号，而不是已经稳定的因果结论。

### 1.2 节点统计

| 类别 | 数量 | 节点 |
|---|---:|---|
| 负向、失败或未过门 | 11 | D11、D12、D13-N、QAF、OMQ-M1、ODQ、B0-E6、RISC-E0、RISC-Orbit、RISC-OBF、RISC-GC |
| 严格中性 | 1 | RISC Soft Projection |
| 正向性能信号 | 1 | world8 rotation augmentation |
| 正向诊断、下游未转化 | 1 | OMQ-M0 source |
| 协议合同失败 | 1 | OMQ actionability scene/pixel leakage |
| 运行中、无最终 mAP | 1 | RISC-POQ batch8/acc2 |
| 合计 | 16 | 不是 16 次独立训练启动，而是 16 个可辨识研究节点 |

### 1.3 最重要的三条认识

1. [INFERENCE] 当前瓶颈不像是“再给 query 加一个小适配器”。
2. [INFERENCE] source existence 不等于 actionability；能测到图像相关信号，
   不代表把它注入 decoder 就会改善 mAP。
3. [DECISION] 论文级缺口已经从闭集 AP 转移到严格 OV 的合法证据链：
   独立初始化、scene-disjoint、base-only/meta-novel、official novel4 seal，
   以及 unmatched query 的 negative-safe 学习。

## 2. 权威基线与不可混淆的评价口径

### 2.1 T7 Epoch24 canonical authority

[FACT] 当前 Swin-T canonical T7 Epoch24 raw authority：

- raw full-val images：13,833；
- query rows：每图 Q600；
- rotated box dimensions：5；
- inference：all-query，no NMS，no top-k；
- exact mAP：0.6064053488；
- rounded AP50：0.6060。

[CLAIM-BOUNDARY] 这是 all-18 supervised closed-set substrate。其 novel4
分片只是闭集诊断，不是 open-vocabulary 证据。

### 2.2 不可直接比较的高值 proxy

[FACT] D13-N 和 OMQ-M1 的约 0.84 数字来自 proxy1600×400 口径，不是
raw-13,833 mAP。它们只能与各自 matched control 比较。

[DO-NOT-REPEAT] 后续 AI 不得将 0.847 proxy 写成“系统已经达到 raw 84.7 mAP”，
也不得与 raw 0.606、0.620 或 B0 0.576 排名。

## 3. 时间线

| 日期 | 研究节点 | 结果与状态 |
|---|---|---|
| 07-30 | D11 generic first-600 content query | matched endpoint -0.0139，discard |
| 07-31 | D12 terminal-XYWH transport | candidate 动态非有限，无合法 paired endpoint |
| 08-01 | D13-N null/existence | endpoint -0.009181，negative |
| 08-01 | QAF source gate | reachability 0.07179，placebo 不敏感，FAIL |
| 08-02 | ODQ full E6 | 0.540336 vs 0.553674，FAIL |
| 08-03 | OMQ-M0 source | reach 0.501032，source gate PASS |
| 08-03 | OMQ-M1 | E10 -0.000307，discard |
| 08-03 | OMQ actionability | scene/pixel leakage，STOP_CONTRACT_FAIL |
| 08-03—08-05 | Swin-B B0 | E1 0.449581；恢复后 E6 0.576001，未过 0.5800 绝对门 |
| 08-04—08-05 | rotation/no-rotation controls | world8 +0.009718；world1/E2 -0.017873，结论冲突 |
| 08-05 | RISC-E0 low-rank residual | 0.602033，低于 T7 E24 |
| 08-06 | RISC-Orbit | 0.605425，接近但仍低于 T7 E24 |
| 08-06 | RISC-OBF hard deletion | 0.591779，明显退化 |
| 08-07 | RISC-GC | 0.592630，明显退化 |
| 08-07 | RISC Soft Projection | 两轮均 0.6200956702，与父模型相同 |
| 08-07 | RISC-POQ | 运行中，尚无 mAP |

## 4. 逐实验复盘

### 4.1 D11：generic first-600 content query

[FACT] 科学问题：仅把 generic GroundingDINO OGC 的前 600 行 content query
恢复到候选，reference initializer、Q600、rotated 5D、数据、loss、optimizer
和 inference mouth 保持不变。

[FACT] 冻结 Epoch12：

| 指标 | Candidate | Control | Delta |
|---|---:|---:|---:|
| mAP | 0.4521 | 0.4660 | -0.0139 |
| AP50 | 0.4520 | 0.4660 | -0.0140 |
| novel4 | 0.26425 | 0.32475 | -0.06050 |
| base14 | 0.473571 | 0.473071 | +0.000500 |

[FACT] Candidate 在 Epoch3 曾领先 +0.0351，在自身最佳 Epoch10 仍领先
+0.0110，但 Epoch11/12 反转。

[INFERENCE] generic content initialization 会改变早期优化路径，但没有改善稳定端点；
novel4 的显著下降提示“恢复 generic query”并不自动等于保留开放语义。

[DO-NOT-REPEAT] 禁止最佳 epoch cherry-pick、额外 epoch、retune、rescue 或与其他
query adapter 堆叠。

[AI-SCAN-P1]

- docs/project_history/exp_20260715_dotav2_cleanstart_ov_e2e_ap70/fres_20260730_d11_content_query_world3_zh.md
- work_dirs/dotav2_cleanstart/d11_v2_world3_candidate_seed20260716_gpu012_batch2
- work_dirs/dotav2_cleanstart/d11_v2_world3_control_seed20260716_gpu345_batch2
- .lab/results.tsv 中 8-D149-D11-W3-V2-FINAL

### 4.2 D12：terminal-XYWH transport

[FACT] zero-step proxy400 输出有限，candidate/control mAP 分别约
0.00920/0.00750，但 zero-step 不是正式 gate。

[FACT] 正式 candidate 在 Epoch2 完成前触发 Hungarian cost invalid numeric
entries；matched control 正常完成 Epoch12，mAP 0.394293。

[FACT] bounded failure-only probe 未复现同一异常，但 Epoch3 candidate
仍低于 matched control：0.0966 vs 0.1029。

[DECISION] D12 没有合法 paired endpoint，属于无效/失败路线，不应作为普通
“候选 mAP 下降”统计。

[INFERENCE] 直接运输 terminal XYWH 会引入数值和优化不稳定；即便异常暂时不复现，
早期 AP 也没有支持继续救援。

[DO-NOT-REPEAT] 当前 terminal-XYWH 配方禁止重启、稳定器堆叠和候选单臂长训。

[AI-SCAN-P1]

- .lab/results.tsv 中 8-D150 至 8-D156
- work_dirs/dotav2_cleanstart/d12_world5_candidate_seed20260716_gpu01234_batch2
- work_dirs/dotav2_cleanstart/d12_world5_control_seed20260716_gpu56789_batch2

### 4.3 D13-N：null/existence residual

[FACT] 仅训练 257 个 scalar，zero-identity，proxy1600×400，冻结 Epoch12：

- candidate：0.838305831；
- control：0.847486973；
- delta：-0.009181142；
- Epoch2 一度仅 +0.0004875。

[INFERENCE] unmatched/null calibration 不是当前端点差距的充分解释；早期极小正偏移
不能覆盖持续训练后的负向趋势。

[DO-NOT-REPEAT] 禁止选择 Epoch2、扩大 scale、额外 epoch、叠加 existence/null
模块或改变门槛救援。

[AI-SCAN-P1]

- .lab/results.tsv 中 8-D157、8-D158
- work_dirs/dotav2_cleanstart/d13n_full_candidate_2b326af_seed20260716_gpu56789_batch2
- work_dirs/dotav2_cleanstart/d13n_full_control_2b326af_seed20260716_gpu01234_batch2

### 4.4 QAF：统一 Query Allocation Quality Flow 源门

[FACT] E24 stride-32 source preflight：

- overall reachability：0.0717888256；
- base14：0.0717736257；
- spatial shuffle：0.072623961；
- semantic shuffle：0.071837951；
- novel4 为 1/1 样本，不能作为强证据。

[FACT] primary 与 placebo 基本不可区分，低于 25% reachability 门，也未超过 prior E
+3 个百分点。

[DECISION] 当前 source family 关闭，production module 和 full training 未授权。

[INFERENCE] 低覆盖且 placebo 不敏感的 source 不值得通过更复杂 decoder adapter
强行放大。

[AI-SCAN-P1]

- .lab/workspace/exp-8-qaf-source-v1/source_report.json
- .lab/results.tsv 中 8-D159、8-D160

### 4.5 OMQ：source existence、M1 与协议失败

#### 4.5.1 M0 source

[FACT] exact rotated decoder attention source：

- primary reach：0.5010316369；
- paired-image placebo：0.2537826685；
- spatial-shift placebo：0.2574670859；
- ≤8 px primary：0.4962544566；
- small-vehicle primary：0.4752901672。

[FACT] source existence 与空间敏感性通过门。

#### 4.5.2 M1 detector action

[FACT] 十个完整 matched proxy epoch：

- candidate E10：0.8465896249；
- control：0.8468967080；
- endpoint delta：-0.0003070831；
- E1–E10 mean delta：约 -0.00038；
- 最佳短暂 delta：约 +0.0006。

[INFERENCE] 源信号真实存在，但当前注入方式没有形成 detector actionability。

#### 4.5.3 Actionability contract

[FACT] 冻结 split 仅 tile-stem-disjoint，不是 original-scene/pixel-disjoint：

- shared scenes：164；
- val tiles from shared scenes：322；
- pixel-overlap val tiles：86；
- overlap pairs：105；
- max crop IoU：0.971126。

[DECISION] Task3 capture、ridge fit、M2 与 GPU launch 均取消。

[DO-NOT-REPEAT] 不得在同一泄漏 split 上重跑；也不得把数学工具 69 tests passed
描述为科学协议通过。

[INFERENCE] “可测 source”与“可干预 source”必须分开；未来 actionability 分析要先
封住 scene/pixel leakage。

[AI-SCAN-P1]

- .lab/workspace/exp-8-omq-decoder-source-m0/source_report.json
- .lab/results.tsv 中 8-D161 至 8-D166
- work_dirs/dotav2_cleanstart/omq_m1_candidate_seed20260716_gpu56789_batch2
- work_dirs/dotav2_cleanstart/omq_m1_control_seed20260716_gpu01234_batch2

### 4.6 ODQ-R1 full E6

[FACT] 八卡正常完成，无 fatal：

- candidate E1：0.3569740057；
- candidate E6：0.5403361917；
- T7 E6 control：0.5536741614；
- delta：-0.0133379698；
- 注册要求：至少 +0.010。

[DECISION] 未进入 E12/E24。

[INFERENCE] distributional query flow 在真实 full-E6 mouth 上造成显著退化；
这比 proxy 上的局部机制改善更有裁决力。

[DO-NOT-REPEAT] 禁止延长、scale rescue 或补加未注册模块。

[AI-SCAN-P1]

- work_dirs/odq/full_e6_gpu0_7_20260802_121440
- .lab/results.tsv 中 8-D164

### 4.7 Swin-B B0：容量锚点

[FACT] B0 是 official generic GroundingDINO-B 初始化、all-18 supervised 的
闭集容量锚点，不是方法候选。

[FACT] 正式 world10：

- E1 exact mAP：0.4495806694；
- E1 continuation threshold：0.4000；
- E1 PASS；
- 用户在完整 E3 后停止；
- E3 无 checkpoint/mAP，唯一 checkpoint 仍为 E1。

[FACT] GPU8/9 从 E1 恢复：

- r1 暴露 PyTorch 1.12 reentrant checkpoint + static graph + no_sync
  accumulation 兼容性故障，无科学结果；
- r2 使用 AlwaysSyncOptimWrapper 保持每五个 microstep 一次 optimizer step；
- E6 exact mAP：0.5760012269；
- 相对 T7 E6 0.5536741614：+0.0223270655；
- 注册 E6 双门：mAP ≥0.5800 且相对 T7 E6 ≥+0.0200；
- 第二门通过，第一门差 0.0039987731，整体未通过。

[INFERENCE] 更大 backbone/pretraining capacity 在同阶段确有正收益，但当前 B0
没有达到预注册绝对门；更重要的是它仍没有回答 strict-OV 问题。

[DECISION] 在优先级 A 下，不应继续把主要 GPU 预算用于 B0 E12/E18/E24。

[CLAIM-BOUNDARY] 允许表述：B0 是工程容量近失配锚点。禁止表述：
Swin-B 已证明方法创新、strict-OV、negative-safe 或 zero-shot。

[AI-SCAN-P1]

- work_dirs/dotav2_cleanstart/swinb_b0_full24e_grouped_scale1024_rare4x_seed20260716_gpu0123456789_batch2_acc1
- work_dirs/dotav2_cleanstart/swinb_b0_gpu89_e1_to_e6_r2_sync
- .lab/results.tsv 中 8-D167 至 8-D185
- .lab/workspace/exp-8-swinb-b0-r3/RESUME_NEXT_GPU_WINDOW.md

### 4.8 Rotation augmentation controls

[FACT] world8、E24-parent continuation 到 E6：

| Arm | raw mAP | AP50 |
|---|---:|---:|
| rotation augmentation | 0.6200956702 | 0.620 |
| no rotation | 0.6103773117 | 0.610 |
| delta | +0.0097183585 | +0.010 |

[FACT] 相对 canonical T7 E24 0.6064053488：

- rotation：+0.0136903214；
- no rotation：+0.0039719629。

[FACT] world1 两轮 continuation：

| Arm | E2 mAP |
|---|---:|
| rotation | 0.5860218406 |
| no rotation | 0.6038943529 |
| delta | -0.0178725123 |

[INFERENCE] 正信号可能依赖 world size、effective batch、sampler、训练长度或 seed；
当前不能简单归因于“旋转不变性已经被模型学到”。

[DECISION] rotation 是优先级 A 的 robustness/辅助轴，而不是 strict-OV 主方法。

[OPEN] 需要在 scene-disjoint strict-OV protocol 中用 matched、至少多 seed 的
2×2 设计验证 rotation 主效应与 negative-safe interaction。

[AI-SCAN-P1]

- work_dirs/risc_controls/rotation_aug_e24_6e_world8_seed20260804
- work_dirs/risc_controls/no_rotation_e24_6e_world8_seed20260804
- work_dirs/risc_controls/gpu8_rotation_e24_1e_world1_seed20260804
- work_dirs/risc_controls/gpu9_no_rotation_e24_1e_world1_seed20260804

### 4.9 RISC-E0：一般低秩 residual

[FACT] full E6 raw mAP：0.6020325422。

[FACT] 相对 T7 E24 0.6064053488：-0.0043728066。

[INFERENCE] 一般 query-only 低秩 residual 没有建立有用方向，且会侵蚀父模型。

[DO-NOT-REPEAT] 禁止通过增加 rank、loss stack 或额外 epoch 盲目救援 E0。

[AI-SCAN-P1]

- work_dirs/risc_e0/lowrank_e24_6e_world8_seed20260805_restart1

### 4.10 RISC-Orbit Flow

[FACT] full E6 raw mAP：0.6054249406。

[FACT] 相对 T7 E24：-0.0009804082。

[INFERENCE] orbit supervision 比一般 E0 更接近父模型，但没有产生可晋级的 AP；
其价值更可能在诊断 query/orbit 可识别性，而不是当前实现本身。

[DO-NOT-REPEAT] 不得把 -0.001 的近似持平描述成提升。

[AI-SCAN-P1]

- work_dirs/risc_orbit/phase1_e24_6e_world8_seed20260806

### 4.11 RISC-OBF 与 RISC-GC

[FACT] 同一 paper-mouth parent replay：0.6248902082。

[FACT] OBF hard subspace deletion E2：0.5917792916，delta -0.0331109166。

[FACT] GC candidate E3：0.5926304460，delta -0.0322597623。

[INFERENCE] 无论硬删除还是 group-conditioned 变换，当前识别出的“旋转子空间”
都包含大量任务有效语义；将其直接消除会造成约 3.2–3.3 mAP 点损失。

[DO-NOT-REPEAT] 禁止继续调删除强度、叠加 class bias 或以训练 loss 下降为理由
长训同一 hard-removal family。

[AI-SCAN-P1]

- work_dirs/risc_obf_swinb/r1_e2
- work_dirs/risc_obf_swinb/r1_e2_papermouth6605_world10
- work_dirs/risc_gc_swinb/candidate_e3_v2

### 4.12 RISC Soft Projection

[FACT] parent raw mAP：0.6200956702。

[FACT] Epoch1 与 Epoch2 都精确得到 0.6200956702。

[FACT] Stage0 有有限但非常小的投影梯度，训练可运行，父模型加载边界正确。

[INFERENCE] 该投影在当前两轮预算和强度下对最终排序没有可见影响。它不是工程失败，
而是严格中性科学结果。

[DO-NOT-REPEAT] 同一父模型、rank、harmonic、两轮设置禁止简单续训或放大强度；
若未来重访，必须由新的可识别性诊断预先证明目标子空间与 AP error 有关。

[AI-SCAN-P1]

- work_dirs/risc_soft_projection/map62_parent_e2_gpu89_20260807

### 4.13 RISC-POQ：当前运行中的开放节点

[FACT] 当前正式运行目录：

- work_dirs/risc_poq/map62_parent_e2_gpu89_b8a2_20260807

[FACT] 2026-08-07 23:58 左右的动态快照：

- Epoch1：1420/3042；
- logger ETA：约 10:27:48；
- losses/grad finite；
- GPU8/9 进程存活；
- 尚无 raw validation mAP。

[OPEN] POQ 是否改善父模型 0.6200956702 尚未回答。

[DECISION] POQ 结束前不得根据 loss_risc_poq_*、grad_norm 或 GPU 利用率预测 mAP。

[AI-SCAN-P0] 后续 AI 必须重新读取最新日志，而不是复用上述快照：

- work_dirs/risc_poq/map62_parent_e2_gpu89_b8a2_20260807/launch.log
- work_dirs/risc_poq/map62_parent_e2_gpu89_b8a2_20260807/20260807_204637/vis_data/scalars.json
- work_dirs/risc_poq/audits/map62_e2_gpu89_b8a2_epoch.json

## 5. 数值结果矩阵

下表仅给出各自合法 comparator 内的差值，不跨协议求平均。

| 节点 | Candidate | Comparator | Delta | 状态 |
|---|---:|---:|---:|---|
| D11 E12 proxy400 | 0.4521 | 0.4660 | -0.0139 | discard |
| D12 | 无合法 endpoint | control 0.3943 | N/A | invalid/discard |
| D13-N E12 proxy | 0.838306 | 0.847487 | -0.009181 | negative |
| OMQ-M1 E10 proxy | 0.846590 | 0.846897 | -0.000307 | discard |
| ODQ E6 raw | 0.540336 | 0.553674 | -0.013338 | discard |
| B0 E6 raw | 0.576001 | T7 E6 0.553674 | +0.022327 | 仍未过绝对 0.580 |
| Rotation world8 raw | 0.620096 | no-rotation 0.610377 | +0.009718 | positive signal |
| Rotation world1 E2 | 0.586022 | no-rotation 0.603894 | -0.017873 | contradictory |
| RISC-E0 raw | 0.602033 | T7 E24 0.606405 | -0.004373 | negative |
| RISC-Orbit raw | 0.605425 | T7 E24 0.606405 | -0.000980 | near-neutral negative |
| RISC-OBF raw | 0.591779 | replay parent 0.624890 | -0.033111 | negative |
| RISC-GC raw | 0.592630 | replay parent 0.624890 | -0.032260 | negative |
| Soft Projection raw | 0.620096 | parent 0.620096 | 0.000000 | neutral |
| POQ | pending | parent 0.620096 | pending | running |

## 6. 失败类型与方法论反思

### 6.1 科学负结果不等于工程失败

[FACT] D11、D13-N、OMQ-M1、ODQ、RISC-E0、RISC-Orbit、RISC-OBF、
RISC-GC 和 Soft Projection 都有可解释的合法端点。

[DECISION] 这些应保留为科学排除证据，不能因为分数不好就从总结中删除。

### 6.2 无效结果必须与负结果分开

[FACT] D12 缺失 paired endpoint；OMQ actionability 的 split 有 scene/pixel leakage；
B0 r1 是 runtime compatibility failure。

[DECISION] 这三类不能写成“模型机制导致 mAP 下降”。

### 6.3 过去十天过度集中于 query 后处理式机制

[INFERENCE] D11、D13、QAF、OMQ、ODQ、E0、OBF、GC、Soft Projection
虽然形式不同，但大多试图在已产生的 query semantics/evidence 上做初始化、残差、
投影、删除或融合。连续负结果提示：

- 真正瓶颈可能更靠前，位于监督合同和负样本定义；
- 也可能更靠后，位于 fixed-Q all-row scoring 的可校准性；
- 但现有证据最强地指向“未标注类别被当作背景负样本”的 strict-OV 缺口，
  而不是再寻找一个低维旋转子空间。

### 6.4 source existence 不能替代干预实验

[FACT] OMQ-M0 source 明显优于 placebo，但 OMQ-M1 对 mAP 为负。

[INFERENCE] 后续任何“我们找到了某个相关 feature/source”的结果，都必须先问：

1. 它是否 scene-disjoint、无像素泄漏？
2. 它是否预测 held-out error，而非只与当前输出相关？
3. 梯度是否真的经过目标参数？
4. 它能否在 matched endpoint 改善 raw mAP？

### 6.5 父模型选择必须前置冻结

[FACT] RISC D186 最初引用了不属于当前七日 OV-CapFlow 语境的 OpenRSD C2B
0.659678，随后在零 optimizer update 前被纠正为 August-5 parent 0.62009567。

[DECISION] 未来每个候选在实现前必须记录：

- parent checkpoint path、bytes、SHA；
- parent raw mouth 与 exact metric；
- candidate 允许新增的 keys；
- zero-init 是否精确重现 parent prediction；
- comparator 是否同数据、同 seed、同 world/effective batch。

### 6.6 endpoint discipline 是过去十天做对的事

[FACT] D11 在 Epoch3 和 Epoch10 有正 delta，但冻结 E12 为负；按 endpoint
纪律正确淘汰。

[DECISION] 继续保留：不选最佳 epoch、不见结果改门、不以训练 loss 推断 AP。

## 7. 当前论文证据地图

### 7.1 已经有的证据

- [FACT] rotated 5D direct set prediction、Q600、all-query scoring、
  no proposal/top-k/NMS 的系统 substrate 可运行。
- [FACT] T7 E24 raw mAP 0.606405。
- [FACT] rotation augmentation 在一个 world8 设置中达到 0.620096。
- [FACT] 多类 query adapter/投影机制被合法排除。
- [FACT] scene-only M0 inventory 可构造 1,664 train scenes，声明的
  scene/stem/crop overlap 为零，但 content/header seal 尚未完成。

### 7.2 尚未有的证据

- [OPEN] scene-disjoint、base-only 的 strict-OV N0 baseline；
- [OPEN] negative-safe N1 matched candidate/control；
- [OPEN] official novel4 全程封存后的最终结果；
- [OPEN] meta-novel 开发集与 official novel4 测试集之间的选择防火墙；
- [OPEN] 多 seed 稳定性；
- [OPEN] closest-work novelty gate 的最新系统检索；
- [OPEN] POQ 最终 mAP。

### 7.3 当前禁止的论文主张

[CLAIM-BOUNDARY] 禁止：

- “首个遥感开放词汇检测器”；
- “首个 oriented open-vocabulary detector”；
- “all18 novel4 是 zero-shot”；
- “B0/RISC 已证明 negative-safe”；
- “rotation augmentation 已证明旋转不变机制”；
- “一个 source audit PASS 就证明 detector 可以改善”；
- “0.84 proxy 优于 0.62 raw”。

## 8. 下一阶段优先级 A：strict-OV + negative-safe

### 8.1 目标

[DECISION] 下一阶段的主问题：

> 在 teacher-free、pseudo-label-free、固定 Q600、rotated 5D、
> all-query/no-NMS 的条件下，如何避免 base-only 训练把潜在 novel object query
> 作为背景负样本压制，并获得 scene-disjoint、official-novel-sealed 的严格 OV 证据？

### 8.2 必须建立的证据防火墙

[DECISION]

1. N0/N1 使用独立 generic initialization，不继承 B0、RISC、POQ 的训练权重。
2. train/dev/test 按 original scene 划分，并完成 image content/header hash seal。
3. official novel4 不参与方法选择、loss 权重、阈值、早停、控制选择或失败救援。
4. base14 只作为训练正类；meta-novel 仅用于开发期可证伪诊断。
5. 最终 official novel4 只在冻结设计和 endpoint 后打开一次。
6. N0 与 N1 必须 matched：数据、seed、sampler、world size、effective batch、
   optimizer、epoch、Q600 mouth 全部一致。

### 8.3 候选核心假设

[INFERENCE] 当前最值得验证的不是另一种 query projection，而是
positive-unlabeled/negative-safe query classification：

- matched base positives 仍提供明确分类正监督；
- unmatched query 不应对所有潜在 novel semantics 施加同等背景负压力；
- class-agnostic objectness 与 open-vocabulary semantic score 应尽量解耦；
- negative-safe 规则必须 zero-change 或 matched-control 可审计；
- 不依赖 teacher、pseudo box、NMS 或额外 dense inference head。

[OPEN] 具体实现尚未在本文冻结。后续 brainstorming 应比较至少三种最小方案，
再由用户批准设计，不能直接编码或启动训练。

### 8.4 优先研究问题

1. [OPEN] unmatched query 的 novel-text logit 在 N0 中是否被持续压低？
2. [OPEN] false-negative pressure 是否集中于含 novel-like object 的 scene，
   而非所有背景 query？
3. [OPEN] 只改变负样本权重能否改善 meta-novel，而不伤害 base14 和 empty-image？
4. [OPEN] class-agnostic objectness 是否比 semantic projection 更能预测最终 AP？
5. [OPEN] rotation augmentation 与 negative-safe 是否存在正交增益或交互？

## 9. 建议的后续实验层次

本节是工作建议，不是已批准 spec。

### 9.1 P0：先封协议，不训练

[DECISION]

- 完成 M1 PNG header 与 streaming content hashes；
- 生成 scene-disjoint base14/meta-novel/official-novel seal；
- 冻结 N0/N1 parent、mouth、seed、sampler 与 endpoint；
- 预注册主指标、base regression、empty foreground、duplicate 与 novel seal；
- 审计没有 official novel4 信息进入任何选择。

### 9.2 P1：N0 严格基线

[INFERENCE] 在独立 generic init 上只训练 base14，先回答“当前 substrate 在合法
strict-OV protocol 下到底有多差”，而不是先假定新方法有效。

[DECISION] N0 本身必须产出完整错误分解：

- base14/meta-novel AP；
- unmatched query semantic suppression；
- empty-image foreground mass；
- duplicate/ownership；
- score-IoU/null calibration；
- 每图固定 600 rows。

### 9.3 P2：一个最小 N1

[INFERENCE] N1 只允许一个 negative-safe 变量，不同时加入 rotation、POQ、
calibration 或新 backbone。

[DECISION] 只有 N1 在 matched meta-novel gate 上通过，才允许进入 full
official-novel sealed evaluation。

### 9.4 P3：rotation 作为辅助 2×2 因果轴

[INFERENCE] 若 N1 成立，再做：

- N0 / N1；
- rotation off / on；
- 至少多 seed；
- 同 world size/effective batch。

[DECISION] 在这之前，不把 rotation augmentation 升格为主方法。

## 10. 停止清单

### 10.1 已关闭、禁止无新证据重跑

- [DO-NOT-REPEAT] D11 generic first600 content-query initialization。
- [DO-NOT-REPEAT] D12 terminal-XYWH transport。
- [DO-NOT-REPEAT] D13-N null/existence residual。
- [DO-NOT-REPEAT] 当前 QAF source family。
- [DO-NOT-REPEAT] OMQ-M1 当前注入方式。
- [DO-NOT-REPEAT] 泄漏的 OMQ actionability split。
- [DO-NOT-REPEAT] ODQ-R1。
- [DO-NOT-REPEAT] RISC-E0 一般低秩 residual。
- [DO-NOT-REPEAT] RISC-OBF hard deletion。
- [DO-NOT-REPEAT] RISC-GC 当前 group-conditioned projection。
- [DO-NOT-REPEAT] Soft Projection 同一 rank/harmonic/两轮设置。

### 10.2 暂不关闭

- [OPEN] POQ：等待 raw endpoint。
- [OPEN] rotation augmentation：有冲突结果，需要严格 matched replication。
- [OPEN] strict-OV negative-safe：主线尚未启动。

## 11. 后续 AI 的工作守则

### 11.1 恢复任务时

[AI-SCAN-P0]

1. 先读本文，不从 work_dirs 名称猜实验状态。
2. 读 docs/project_history/README.md 的当前 P0 指针。
3. 定向查 .lab/results.tsv 中 8-D149 至最新记录。
4. 读 .lab/branches.md，核对 closed/running/superseded。
5. 刷新 nvidia-smi、进程 owner 和运行日志。
6. 对 POQ 或其他运行中任务，只认最新 scalars/log 与进程，不复用本文动态快照。
7. 报告事实、推断、建议三者的边界。

### 11.2 提出新实验时

[AI-SCAN-P0]

- 主优先级固定为 strict-OV + negative-safe，除非用户明确改变。
- 先检查候选是否与第 10 节关闭路线同构。
- 先写 problem、causal hypothesis、matched control、endpoint、promotion/stop gate。
- 先封 scene/content/official-novel leakage。
- 不把 B0/RISC/POQ 权重带入 N0/N1。
- 不使用 official novel4 做调参、选择、早停或 rescue。
- 不启动多个变量同时变化的模块堆栈。

### 11.3 扫描 work_dirs 时

[AI-SCAN-P0] 优先扫描：

- work_dirs/risc_poq/map62_parent_e2_gpu89_b8a2_20260807
- work_dirs/risc_soft_projection/map62_parent_e2_gpu89_20260807
- work_dirs/risc_controls/rotation_aug_e24_6e_world8_seed20260804
- work_dirs/risc_controls/no_rotation_e24_6e_world8_seed20260804
- work_dirs/dotav2_cleanstart/swinb_b0_gpu89_e1_to_e6_r2_sync
- work_dirs/odq/full_e6_gpu0_7_20260802_121440

[AI-SCAN-P1] 涉及失败谱系时扫描：

- work_dirs/dotav2_cleanstart/d11_v2_world3_*
- work_dirs/dotav2_cleanstart/d12_world5_*
- work_dirs/dotav2_cleanstart/d13n_full_*
- work_dirs/dotav2_cleanstart/omq_m1_*
- work_dirs/risc_e0
- work_dirs/risc_orbit
- work_dirs/risc_obf_swinb
- work_dirs/risc_gc_swinb

[AI-SCAN-P2] 仅在复现/诊断时扫描：

- 大型 checkpoint 文件；
- 完整 launcher.log；
- 中间每 20 iter 的 loss rows；
- D1–D148 的历史微观诊断；
- 已被 supersede 的 config。

### 11.4 推荐的只读扫描命令

所有 shell 命令必须以 rtk 开头：

    rtk sed -n '1,260p' docs/project_history/exp_20260729_20260807_ten_day_review/fres_20260729_20260807_full_review_reflection_zh.md
    rtk rg -n '^8-D(149|15[0-9]|16[0-9]|17[0-9]|18[0-9])' .lab/results.tsv
    rtk tail -n 120 .lab/branches.md
    rtk rg -n 'dota/mAP|dota/AP50' work_dirs/risc_poq work_dirs/risc_soft_projection
    rtk rg -n 'Epoch\(train\)' work_dirs/risc_poq/map62_parent_e2_gpu89_b8a2_20260807/launch.log
    rtk nvidia-smi

[DECISION] 禁止为了“扫描方便”删除 checkpoint、清理 work_dirs、覆盖日志、
kill 进程或修改实验台账。

## 12. AI 扫描优先级地图

### P0：每次恢复必读

1. 本文。
2. docs/project_history/README.md。
3. .lab/results.tsv 的 D149 以后记录。
4. .lab/branches.md。
5. 当前 live work_dir 与 nvidia-smi。

### P1：按问题读取

1. 对应实验的 scalars.json。
2. 对应 final result/audit JSON。
3. 对应 config.py 与 sampler audit。
4. 本文第 4 节列出的证据目录。

### P2：只有发生争议或复现时

1. checkpoint SHA、state_dict keys 与 optimizer state。
2. 全量 launcher/console logs。
3. DDP、NCCL、sampler 和 runtime compatibility 记录。
4. 旧计划、已 supersede 设计与非里程碑 loss 窗口。

## 13. 当前开放问题与下次扫描触发器

| 优先级 | 开放问题 | 触发器 | 下一动作 |
|---|---|---|---|
| P0 | POQ 最终 mAP | 出现 Epoch1/Epoch2 validation 或进程结束 | 对父 0.62009567 做冻结裁决 |
| P0 | strict-OV 数据 seal | 用户批准下一阶段设计 | 完成 content/header hash，不训练 |
| P0 | N0 baseline | seal 与协议审计通过 | 独立初始化 matched baseline |
| P0 | N1 negative-safe | N0 错误分解完成 | 只引入一个负监督变量 |
| P1 | rotation 主效应 | N1 初步通过 | matched 2×2、多 seed |
| P1 | closest-work novelty | 方法设计冻结前 | 重新做真实文献检索 |
| P2 | B0 继续长训 | 仅当用户把 raw mAP 改回第一目标 | 重新设计，不自动恢复 |

## 14. 给未来 AI 的最终摘要

[AI-SCAN-P0]

如果你是后续接手本项目的 AI，请不要从“目录很多、提交很多、GPU 跑得很满”
推断研究已经取得大量正结果。过去十天的真实结论是：

1. 多数 query/semantic adapter 为负或中性；
2. rotation augmentation 有一个 +0.0097 信号，但另一个 matched 设置为负；
3. B0-E6 相对 T7-E6 有 +0.0223，但差 0.004 未过绝对门，并且不是 OV 证据；
4. OMQ 证明 source 可以存在而 actionability 仍失败；
5. 当前最重要的未解决问题是合法 strict-OV supervision；
6. 用户已明确选择 negative-safe strict-OV 为第一优先级；
7. POQ 尚未出结果，必须实时刷新；
8. 新实验应从 scene-disjoint、official-novel-sealed、独立初始化 N0/N1 开始，
   而不是继续堆叠已关闭的 query adapter。

[DECISION] 本文若与未来新证据冲突，后续 AI 应追加带日期的修订说明并保留旧结论，
不得静默改写历史；动态状态必须更新，冻结历史指标不得覆盖。

## 15. 2026-08-08 证据修订：M1、文献门与 strict-OV N0 边界

本节追加于原十日快照之后，保留第 1–14 节当时的判断，不回写历史。凡与下述
状态冲突，以本节及其列出的不可变产物为准。

### 15.1 M1 content/header seal 已完成，但仍不授权训练

[FACT] M1 完成标记为：
`.lab/workspace/exp-8-n0-scene-prep-v1/m1_n0_ri_v1/COMMITTED.json`。
该标记声明 `m1_complete=true`、`source_content_seal_complete=true`，并绑定：

- manifest SHA256：`0deb6e3ca635abc0a3491c6abfb8bf1dc00753f56f949af7f42862fcd39fdd1d`；
- content inventory SHA256：`2b1bb3b3da30688d50cef6f0fbcc0a22d4a554e6ac6c6e5e007f836ccd3caef2`；
- 47,294 个 RGB records，其中 train/dev/test 为 35,109/5,172/7,013；
- 总 content bytes 为 69,527,520,786；
- cross-partition content SHA256 overlap 为 0。

[CLAIM-BOUNDARY] 同一冻结报告仍明确记录
`preparation_only=true`、`training_use_forbidden=true`、`N0_N1_authorized=false`。
因此 M1 是已完成的数据完整性前置证据，不是启动 N0/N1 训练的授权。

### 15.2 closest-work hard gate 已完成，裁决为 NARROW

[FACT] 权威记录为
`.worktrees/n0-literature-m1/docs/literature-search-20260803-negative-safe-fixed-set/papers.md`：
筛选 21 个候选，保留并由 primary source 核验 15 篇，裁决为 `NARROW`。

[DECISION] 允许继续做可证伪机制诊断，但只允许窄化表述：研究 fixed-Q600、
rotated open-vocabulary set predictor 中，cross-view-stable evidence 能否识别并
抑制 harmful false-background classification gradients；候选干预必须是
negative-only，不产生 positive pseudo-box targets，且不引入 proposal/top-k/NMS。
文献状态本身不授权 N1；在正式冻结 N1 或论文 claim 前必须刷新检索。

### 15.3 N0-RI 四折确认通过，但不是 strict-OV 证据

[FACT] 权威裁决为
`.lab/workspace/exp-8-n0-ri-confirm-v1/confirmation_analysis/decision.json`，结果
`PASS_TO_RISC_STAGE0`。它覆盖 158 个唯一 scene、256 个四折 tile observations，
`optimizer_steps=0`，且 P0148 与 test partition 未用于决策；预注册 checks 全部通过。

[CLAIM-BOUNDARY] N0-RI 使用的 source baseline 在训练时见过 DOTA-v2 all-18
类别。因此它只支持“旋转视角下存在 semantic interference signal”的机制诊断，
不支持 strict unseen/novel generalization，也不能替代 seen-only strict-OV N0。

### 15.4 当前真正开放的主门

[OPEN] 当前需要实现并运行的是 N0 negative-gradient diagnostic：在
scene-disjoint、seen-only supervision、独立 generic initialization、official
novel4 封存的协议下，检验 withheld meta-novel 邻域 unmatched Q600 rows 是否承受
集中且方向冲突的 background focal gradients。

[DECISION] 决策顺序固定为：Fold 1 zero-step pilot → 四折 zero-step N0 → 四折
seen-only E1 N0。只有 zero-step 与 E1 的完整 N0 都独立通过预注册门，N1 才获得
单独设计与实现资格；任一阶段失败都关闭或重构该机制，不以训练救援绕过 N0。

[AI-SCAN-P0] 已批准的实施计划保存在
`docs/superpowers/plans/2026-08-08-strict-ov-n0-negative-gradient-gate.md`。
该计划只授权按门执行 N0；不预授权 N1 代码或训练。
