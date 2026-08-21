# DOTA-v2 Clean-start OV E2E AP70：过去一整天实验总结

## 1. Report metadata

| field | value |
|---|---|
| report_time | `2026-07-16 16:32 CST` |
| core_window | `2026-07-15 16:32` — `2026-07-16 16:32 CST` |
| branch | `research/dotav2-cleanstart-ov-e2e-ap70` |
| HEAD | `a3c3817d912f48d8a3489a2d47f4a7059fe791c2` |
| primary_goal | raw DOTA-v2 `mAP >= 0.7000` 且 `AP50 >= 0.7000` |
| current_state | `paused-after-epoch6-checkpoint-raw-eval-pending` |
| commits_in_window | `33` |
| authoritative_metric_mouth | raw `ss_val=13,833`, `filter_empty_gt=False` |

本报告以滚动 24 小时为核心时间窗。为了让因果链完整，报告在“前置背景”中额外
保留了 `2026-07-15 08:22` 左右完成的 P126C 强锚点复验；该结果发生在核心时间窗
之前，而且不满足 clean-start 架构约束，因此只作参照，不进入当前合规路线的成功
判定。

权威证据来自 `.lab/log.md`、`.lab/results.tsv`、冻结配置、训练日志、
`vis_data/scalars.json`、checkpoint 文件及其独立 SHA256。报告不把训练 loss 推断成
AP，不把 400 图小口径 AP 与 raw 13,833 AP 混为一谈，也不把被中断的 epoch6
validation 写成有效结果。

## 2. 一句话结论

过去一整天完成了从 generic-only 初始化到小数据筛选、工程审计、四卡全量训练和
epoch6 安全保存的完整链路：小数据主种子上，`8-S1-G` grouped O2O 以
`AP50=0.4070` 胜过 control 的 `0.3810`；随后同一合规配方在 47,294 张训练图上
完成 6 个 epoch，epoch1 raw 13,833 验证为 `mAP=0.3443`、`AP50=0.3440`，
epoch6 checkpoint 已完整保存并独立校验，但其 raw AP 按用户停卡指令尚未评测。

因此当前最准确的判断是：

- 合规训练链路已经跑通，且模型持续稳定学习；
- 小数据赢家存在明显 seed sensitivity，不能把 `+0.0260` 当成稳健增益；
- 当前唯一有效的 full-data raw AP 仍是 epoch1 的 `0.3440`；
- 距离 `0.7000` 仍有 `0.3560 AP50`，目标远未完成；
- 下一步不是盲目续训，而是先补跑 epoch6 raw 13,833 验证，再按已预注册门槛决定
  继续 scale800 还是转入 matched scale1024 小数据实验。

## 3. 目标约束与当前符合性

这里的“从头开始”采用项目已经确认的严格口径：遥感检测任务从 generic OGC
GroundingDINO compatible checkpoint 开始，可以使用通用 ImageNet/语言预训练，
但禁止加载任何 DOTA、OpenRSD、GSOVD、P126C 或其他遥感检测训练权重。这不是
所有参数完全随机初始化，而是 generic-only task clean-start。

“无头”按本项目可执行口径解释为无 dense proposal head、无 RPN/RoI 两阶段头，
只保留 DETR decoder 的 fixed-set prediction readout。完全没有任何预测映射就无法
输出类别分数和旋转框，因此不能把“无头”误写成没有 E2E set-prediction readout。

| requirement | implementation/evidence | status |
|---|---|---|
| task clean-start | generic compatible checkpoint；无 DOTA/OpenRSD/遥感 parent | pass |
| no teacher/distillation | checkpoint provenance 和 config audit 均无 teacher/distill namespace | pass |
| no pseudo labels | 训练只用真实 DOTA-v2 GT；subset manifest 明确 `uses_pseudo_labels=false` | pass |
| open vocabulary | token-similarity scoring；prompt reorder/synonym/new-class shape audits pass | pass |
| E2E fixed-set | decoder 直接输出主 Q600，保持 query 顺序 | pass |
| no dense/RoI head | 无 dense proposal、RPN、RoI 推理路径 | pass |
| no NMS/top-k | strict runtime audit 对 `topk/nms/nms_rotated/multiclass_nms` 零命中 | pass |
| rotated boxes | 每个 query 输出 5-D rotated box，decoder/export 对齐误差为 0 | pass |
| small-data first | S0 50-update smoke + S1 1,600/400 四候选筛选已完成 | pass |
| four-GPU full train | physical GPU 4–7，47,294 图，已完成 epoch1–6 | pass |
| raw DOTA Val AP >= 0.70 | 当前合规 full raw AP50 只有 epoch1 `0.3440`；epoch6 未评 | **not achieved** |

## 4. 前置背景：强锚点能做什么，不能做什么

`7-A0` 使用 P126C epoch40 已保存 checkpoint，在 raw 13,833 张 DOTA-v2 validation
上重新评估得到 `mAP=0.656813`、`AP50=0.6570`。四卡推理完整覆盖 13,833 条记录，
结果可复现，checkpoint SHA256 为
`5babf6e5a8c335299705158ec49bb8b88f70508ff0e72c75323abb4d894af979`。

这个结果证明当前机器、数据口径和 evaluator 能得到接近 0.66 的强检测性能，也说明
DOTA-v2 数据本身和评估管线不是 0.70 目标的根本障碍。但它不能作为当前目标的完成
证据，原因有三：

1. 它加载了既有遥感检测 checkpoint，不是 generic-only clean-start；
2. 历史体系包含当前目标禁止的 dense head/NMS 等路径；
3. 即便忽略架构约束，`AP50=0.6570` 仍低于 0.7000。

所以 P126C 只承担“可达上界参照”和“数据/evaluator 健康锚点”的角色，未被加载到
Experiment 8 的任何训练候选中。

## 5. 过去 24 小时实验时间线

| time | event | outcome |
|---|---|---|
| 07-15 18:19–19:22 | 设计并实现 Experiment 8 clean-start 合同 | provenance、Q600、OV、rotated、strict audits 就绪 |
| 07-15 19:22–19:37 | `8-S0` 50-update real-data smoke | 首次 dense batch OOM；按预注册规则降 batch 后完成 |
| 07-15 19:41 | 冻结 `8-S1` 2,000-tile protocol | 1,600 train / 400 val，主种子 20260715 |
| 07-15 19:41–07-16 01:17 | control/grouped/Hausdorff/Chamfer 与 repeat 并行 | grouped 主种子胜出；Hausdorff/Chamfer 淘汰；repeat 暴露 seed sensitivity |
| 07-15 21:14 | `8-M0` GPU monitor | 12 tests pass，形成 30 秒 JSONL 监控链 |
| 07-15 21:40 | `8-M1` sampler audit | 各候选 1,600/1,600，duplicate=0，missing=0 |
| 07-15 21:53 | `8-M2` sparse milestones | 只保存 epoch 1/6/12/18/24 |
| 07-15 22:13–22:24 | `8-CF0` Chamfer 实现与启动 | matcher contract pass，最终科学结果低于 control |
| 07-15 23:20 | `8-M3` full24 protocol prepared | 47,294/13,833 mouth、最密样本、world4 sampler 预检通过 |
| 07-15 23:47–23:59 | 修复 variable epoch 与 accumulation boundary | 精确总计划 71,100；所有 milestone 可恢复 |
| 07-16 00:40 | grouped repeat recovery 完成 | best AP50 0.3840；框架故障修复得到验证 |
| 07-16 01:17 | Chamfer 和 raw-mouth diagnostics 收口 | Chamfer discard；小口径到 raw 存在 -0.0980 gap |
| 07-16 02:16 | 锁定 full-data winner | 选择 grouped O2O，同时保留 seed-sensitivity warning |
| 07-16 02:29 | 四卡 full24 正式 clean restart | 修复 DDP reentrant-ready-twice 后稳定启动 |
| 07-16 04:53 | epoch1 完成并保存 | `epoch_1.pth` 写入完成 |
| 07-16 05:09 | epoch1 raw validation 完成 | `mAP=0.3443`、`AP50=0.3440` |
| 07-16 07:35 | epoch2 完成 | 无非计划 validation；进入 epoch3 |
| 07-16 10:00 | epoch3 完成 | mean loss 比 epoch2 下降 12.94% |
| 07-16 12:33 | epoch4 完成 | mean loss 比 epoch3 下降 5.48% |
| 07-16 14:30 | epoch5 完成 | mean loss 比 epoch4 下降 4.13% |
| 07-16 15:05 | 用户下达 save-then-pause | epoch6 完成后保存、校验、停卡；raw eval 延后 |
| 07-16 16:28–16:29 | epoch6 保存并暂停 | checkpoint 独立 SHA256 一致；本项目训练 PIDs 全退 |

## 6. S0：先证明合规模型真的能训练

`8-S0` 的目的不是测最终 AP，而是验证 generic-only Q600 模型能在真实 DOTA-v2
标注上完成 forward、backward、optimizer update、save/load 和 strict inference。

### 6.1 协议修正

初版继承了 `accumulative_counts=8`。MMEngine 的 train-loop iteration 与真正的
optimizer update 并不等价，如果保持这个设置，固定的 50 iterations 只会产生约
6 次参数更新。因此在运行前将 S0 改成 batch 8、accumulation 1，使 50 iterations
严格对应 50 次 optimizer update。

### 6.2 首批 OOM 与恢复

正式 seed 的第一个 batch 包含 7 张图，其中有 975-GT 和 352-GT dense tiles。
该批次到达 40.26 GiB allocated / 42.10 GiB reserved，随后申请额外 1.08 GiB 时
OOM。故障发生在 update 1 之前，没有产生 checkpoint，也没有污染科学比较。

按预注册的 staircase rule，回退到最后通过的 batch 3 / accumulation 1。恢复后：

| metric | value |
|---|---:|
| real optimizer updates | 50 |
| validation images | 96 |
| mAP | 0.0051 |
| AP50 | 0.0050 |
| final loss | 50.9032 |
| logged peak memory MiB | 19,825 |
| fixed rows/image | 600 |
| checkpoint SHA256 | `ccb0d561861c1be293fded72f5d2b1dc8b4ca34063bbf2a3695a27ed0394616d` |

该 AP 只是 50-update engineering diagnostic，不能用于判断候选好坏。真正重要的是
所有 loss 有限，checkpoint 可重新加载，canonical/reordered/synonym/new-class
prompt 均保持 600 rows，decoder/export 旋转框误差为 0，forbidden calls 为 0，
portable suite 为 `121 passed, 13 skipped`。

## 7. S1：2,000-tile 小数据筛选

### 7.1 冻结数据与训练口径

S1 使用不可变 2,000-tile manifest：1,600 train / 400 validation，保留空图，
不使用 pseudo labels。密度配额覆盖 600 张空图、900 张 1–10 GT、300 张
11–50 GT、150 张 51–200 GT，以及 50 张 >200 GT。全部候选使用相同 generic
checkpoint、seed20260715、12 epochs、effective batch 8、prompts、optimizer 和
schedule。

主验证使用 18 类；额外把 `airport`、`container-crane`、`helipad`、`helicopter`
作为 novel-4 prompt diagnostic。但这个 400 图口径只有 11 个 novel-4 GT，且
helipad 没有 GT，所以 novel-4 结果只能作为风险信号，不能当稳定泛化估计。

### 7.2 候选定义

| exp_id | candidate | only_delta |
|---|---|---|
| 8-S1-C | Q600 control | 无结构干预 |
| 8-S1-G | grouped O2O | 训练期 3 个共享 query groups；推理仍只输出主 Q600 |
| 8-S1-H | Hausdorff-DN | Hausdorff matching + periodic-angle DN + adaptive positive-Hungarian DN |
| 8-S1-CF | Chamfer | 只把 Hungarian RBoxL1Cost 换成 symmetric four-corner ChamferCost |

晋级规则在结果出现前冻结：候选必须先通过有限训练、checkpoint reload、strict
Q600、OV prompt-shape、decoder alignment 和 zero forbidden-call；再要求相对 control
满足 `delta_AP50 >= +0.020`，或 `delta_AP50 >= +0.010` 且 coverage 至少
`+0.020`，duplicate extras/GT 增幅不得超过 10%。

### 7.3 最终对照结果

| metric | 8-S1-C | 8-S1-G | 8-S1-H | 8-S1-CF |
|---|---:|---:|---:|---:|
| best epoch | 12 | 12 | 12 | 12 |
| mAP | 0.3806 | **0.4068** | 0.3409 | 0.3755 |
| AP50 | 0.3810 | **0.4070** | 0.3410 | 0.3750 |
| delta_AP50 vs control | — | **+0.0260** | -0.0400 | -0.0060 |
| base14_AP50 | 0.415 | **0.422** | 0.383 | 0.412 |
| novel4_AP50 | **0.125** | 0.030 | 0.065 | 0.077 |
| GT coverage | **0.504970** | 0.496848 | 0.484970 | 0.490667 |
| duplicate extras/GT | 0.971030 | 0.890061 | 0.952970 | **0.646667** |
| empty foreground mass | 6.384808 | 5.839650 | n/a | **4.805809** |
| rows/image | 600 | 600 | 600 | 600 |
| strict/OV/alignment | pass | pass | pass | pass |
| decision | control | promote | discard | discard |

### 7.4 科学解释

grouped O2O 是主种子唯一通过冻结 AP gate 的候选。它相对 control 提升
`+0.0260 AP50`，同时 duplicate extras/GT 从 0.971030 降到 0.890061，empty
foreground mass 也下降，没有靠增加推理 query 或后处理获得收益。

但它不是“全面更好”：coverage 轻微下降，主种子的 novel-4 AP50 从 0.125 降到
0.030。因为 novel-4 GT 极少，这不能证明真正 novel 泛化崩溃，却足以作为 full-data
阶段必须重点检查的 semantic warning。

Hausdorff-DN 的 geometry loss 更低，却最终比 control 低 0.040 AP50，说明更好的
局部几何匹配没有自动转化成分类排序 AP。Chamfer 把 duplicate extras/GT 大幅降到
0.646667，但 AP50 比 control 低 0.006，说明“去重更好”也不能替代主指标。

### 7.5 Seed20260716 matched repeat

独立 grouped repeat 在 epoch12 尾部遇到 MMEngine `loss_factor should be larger
than zero`。根因不是模型发散，而是 variable batch sampler 的逐 epoch 长度为
`402,403,403,402,402,403,402,402,402,403,403,403`，真实总数 4,830，而 stock
loop 固定估算为 `402 × 12 = 4,824`，少算 6 次。

修复后从最后一个 accumulation boundary（epoch10 iter4,024）恢复并重放完整
epochs 11–12。matched seed 结果为：

| metric | control seed20260716 | grouped seed20260716 | delta |
|---|---:|---:|---:|
| AP50 | 0.3800 | 0.3840 | +0.0040 |
| base14_AP50 | 0.434 | 0.443 | +0.009 |
| novel4_AP50 | 0.079 | 0.125 | +0.046 |
| GT coverage | 0.510667 | 0.508606 | -0.002061 |
| duplicate extras/GT | 1.119152 | 1.097939 | -0.021213 |

`+0.0040` 没有独立通过 `+0.020` promotion gate。因此 grouped 主种子仍按冻结规则
获胜，但必须明确标注 seed-sensitive，不能把它包装成稳定的两种子大幅提升。

## 8. 小数据到 raw 13,833 的迁移诊断

将主种子 S1 grouped best 不做任何更新，直接放到 raw 13,833 validation mouth：

| metric | 400-image S1 mouth | raw 13,833 mouth | delta |
|---|---:|---:|---:|
| mAP | 0.4068 | 0.3094 | -0.0974 |
| AP50 | 0.4070 | 0.3090 | -0.0980 |
| base14_AP50 | 0.422 | 0.360 | -0.062 |
| novel4_AP50 | 0.030 | 0.049 | +0.019 |
| GT coverage | 0.496848 | 0.463572 | -0.033276 |
| duplicate extras/GT | 0.890061 | 0.872656 | -0.017405 |
| empty foreground mass | 5.839650 | 8.412324 | +2.572674 |

这说明 400 图 screen 能做候选排序，但不能代表 raw DOTA Val。迁移损失不是单一
原因：coverage 下降、rare/novel 类弱、空图前景分数质量变差同时存在。因此 full
47,294-image training 是必要步骤，而不是从 0.4070 直接外推到 0.70。

## 9. Full24 四卡协议

### 9.1 数据与配置

| field | value |
|---|---|
| config | `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_full24e.py` |
| train mouth | `ss_train=47,294`, empty=22,575, `filter_empty_gt=False` |
| val mouth | raw `ss_val=13,833`, empty=7,228, `filter_empty_gt=False` |
| classes | canonical DOTA-v2 18 classes |
| physical GPUs | 4,5,6,7 |
| local batch | 4 |
| accumulation | 2 |
| effective batch | 32 |
| schedule | 24 epochs, 500-update warmup, cosine |
| validations | epochs 1,6,12,18,24 |
| checkpoints | epochs 1,6,12,18,24 |
| inference | primary Q600, 600 rows/image, no NMS/top-k |
| initialization | generic compatible checkpoint only |

账本中的 `update` 是每 rank 的 sampler/train-loop iteration 口径。由于
`accumulative_counts=2`，每两个 train-loop iterations 才完成一次参数
`optimizer.step`；报告保留日志原始 `update` 命名，避免与已有审计记录错位。

### 9.2 Sampler 与最密样本

world4 sampler 对 47,294 张训练图实现 duplicate=0、missing=0。24 个 epoch 中，
20 轮为 2,962 updates，第 6/8/12/17/20/21 轮为 2,964 updates，总计划
71,100 updates/rank。增加 `update_count_multiple=2` 后，每个 epoch 都在
accumulation boundary 结束，所有 milestone checkpoint 都能严格恢复。

最密单图包含 3,396 GT，query-area 预算要求 singleton 放行。该真实样本的 grouped
Q1800 forward/backward 在 A40 上通过，peak allocated memory 为 26,009.9 MiB，
没有丢图、拆标注或过滤 dense sample。

### 9.3 四卡启动故障

第一次 full24 formal launch 在第三次 backward 开始时失败。同步 A/B 诊断确认：
PyTorch 1.12 的 reentrant Swin checkpointing 在 DDP + accumulation=2 下，把
`backbone.stages.3.blocks.1.ffn.layers.1.bias` 标记 ready 两次；原来的
`static_graph=True` 只把它表现成较模糊的 `run_backward returned NULL`。

full24 专用修复仅关闭 `backbone.with_cp` 和 DDP `static_graph`，不改变模型数学、
loss、数据、batch、初始化或学习率。相同四卡 smoke 连续通过 8 steps 和 4 个
accumulation boundaries，peak log memory 27,904 MiB；完整回归为
`146 passed, 14 skipped`。随后在 `2026-07-16 02:29:33` 从同一个 generic
checkpoint clean restart，没有从失败进程恢复或继承任何更新。

## 10. Full-data 训练曲线

以下统计由本报告只读汇总正式 `vis_data/scalars.json`。每轮使用 148 个间隔日志点；
这些值只用于判断训练健康和优化趋势，不能替代 raw AP。

| epoch | log_rows | mean_loss | mean_cls | mean_bbox | mean_iou | mean_grad_norm | max_grad_norm | mean_time_s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 148 | 19.782379 | 0.719030 | 0.946684 | 0.633190 | 71.311346 | 94.534051 | 2.894060 |
| 2 | 148 | 9.875145 | 0.383896 | 0.344939 | 0.407664 | 69.657847 | 89.011068 | 2.945655 |
| 3 | 148 | 8.597547 | 0.335231 | 0.285501 | 0.358472 | 68.258462 | 105.759221 | 2.932295 |
| 4 | 148 | 8.126759 | 0.305674 | 0.264073 | 0.347619 | 66.753454 | 96.922107 | 3.106513 |
| 5 | 148 | 7.791503 | 0.300168 | 0.251286 | 0.336908 | 64.704638 | 89.330696 | 2.372193 |
| 6 | 148 | 7.335165 | 0.270410 | 0.235348 | 0.323358 | 62.584012 | 111.318704 | 2.395991 |

主要观察：

- mean loss 从 epoch1 的 19.7824 降到 epoch6 的 7.3352，下降约 62.9%；
- epoch2→3、3→4、4→5、5→6 的 mean loss 依次下降约
  12.94%、5.48%、4.13%、5.86%；
- mean bbox loss 从 0.946684 降到 0.235348，框回归优化持续进行；
- mean grad norm 从 71.31 降到 62.58，所有记录均为有限值；
- epoch3 和 epoch6 的单点 max grad norm 分别达到 105.76 和 111.32，但这是
  clip 前的记录，`clip_grad.max_norm=0.1` 一直启用，且没有对应 loss 发散；
- 没有 OOM、NaN/Inf、NCCL failure、dead rank 或训练数值崩溃。

这些事实证明模型仍在学习，但不能据此声称 epoch6 AP 一定超过 epoch1，更不能据此
声称可以达到 0.70。

## 11. Epoch1 唯一有效的 full raw AP

epoch1 在 `2026-07-16 04:53` 完成全部 2,962 updates，保存 checkpoint 后由同一
未重启进程完成 raw 13,833 validation：

| metric | value |
|---|---:|
| raw_val_images | 13,833 |
| filter_empty_gt | false |
| rows/image | 600 |
| mAP | 0.3443 |
| AP50 | 0.3440 |
| delta_AP50 vs pre-full 8-D1 | +0.0350 |
| estimated base14_AP50 | 0.402 |
| estimated novel4_AP50 | 0.143 |

明显弱类为 helipad 0.000、airport 0.095、container-crane 0.127、
small-vehicle 0.160、soccer-ball-field 0.172；tennis-court 0.795、plane 0.699
说明模型已经形成有效定位和排序能力。

从 pre-full raw AP50 0.3090 到 full epoch1 0.3440 的 `+0.0350` 证明全量真实数据
训练有效，但距离 0.7000 仍差 0.3560。按 optimizer-update 尺度对齐，S1 grouped
epoch7/8 线性插值预测 full epoch1 AP50 约 0.35395，与实际 0.3440 只差 0.00995；
这支持评估口径稳定，同时也暴露 S1 epoch10–12 在约 0.405–0.407 平台化，不能
仅凭延长名义 epoch 乐观外推到 0.70。

## 12. Epoch6 保存、校验与暂停

用户在 `2026-07-16 15:05` 明确要求“跑完第六轮就保存，暂停使用显卡，暂停工作”。
原先的 epoch6 raw gate controller 和 observer 被停止，替换为 save-then-pause
controller：只有 checkpoint 大于 2,000,000,000 bytes、连续 10 秒大小不变、
mtime age 至少 8 秒、完整 SHA256 计算前后大小一致，才向训练 PGID 发送 SIGINT。

epoch6 完成 2,964 updates，日志在 16:28:34 明确记录
`Saving milestone checkpoint at 6 epochs`。最终资产：

| artifact | bytes | SHA256 |
|---|---:|---|
| `epoch_1.pth` | 2,076,337,385 | `7b1ab328d85ff7f7d7f5c304822b81b8514efc95bd39d7ebaacaa317a3c14377` |
| `epoch_6.pth` | 2,087,703,721 | `3e47cf4c230313d84ec5498715d36d25ad8ba136b37c78540ff52b1a89f334c6` |

checkpoint 写完后，框架自动进入 epoch6 内置 validation。controller 在等待文件
稳定并计算完整 SHA256 的十几秒内，validation 运行到 `80/1730`，随后按用户指令
终止整个训练进程组。这个 `80/1730` 不是完整评测，不产生 mAP/AP50，也不能判断
epoch6 三重门槛通过或失败。

停机时 launcher/rank PIDs 全部退出，独立 `nvidia-smi` 显示 GPU4–7 均为
0% utilization、16 MiB memory used，证明本项目四卡已释放。报告生成期间机器上
后来出现了新的外部 GPU 负载；本项目原 PIDs 仍不存在，不能把后续其他任务的占用
误归因到本次训练。

## 13. GPU 利用率与“尽量占满显卡”的实际情况

四卡显存长期约 41–45 GiB，峰值最高约 45,217 MiB，已经接近 A40 安全上限，不能
通过继续增大 batch 来安全提高平均 utilization。训练期间多次采到四卡同时
90–100%，说明计算阶段确实能把卡吃满；但长窗口平均 utilization 在不同 epoch
约 54%–67%。

| epoch/window | GPU4 avg | GPU5 avg | GPU6 avg | GPU7 avg | all-four >=90 | fatal |
|---|---:|---:|---:|---:|---:|---:|
| epoch2 | 63.89% | 62.26% | 62.90% | 63.02% | 79/287 | 0 |
| epoch3 | 60.88% | 60.33% | 60.49% | 61.57% | 74/287 | 0 |
| epoch4 | 54.34% | 52.68% | 54.55% | 54.83% | 63/303 | 0 |
| epoch5 | 66.55% | 66.18% | 65.05% | 67.03% | 62/233 | 0 |

只读 `8-D3` sampler skew audit 显示四 rank 平均 query-area spread 只有 0.66%；
同一步 max/min query-area ratio 的 P50/P90/P95/P99/max 为
1.0809/1.7196/2.1567/2.9515/4.7555，估算 idle fraction 约 13.48%。这能解释
部分尾部 dense-update 等待，却不能解释全部利用率缺口。

源码路径显示 grouped O2O 每个 local batch 会触发大量
`cost.detach().cpu()` + SciPy `linear_sum_assignment`，与 GPU/CPU 同步脉冲相符；
但 live `perf` attach 被 `perf_event_paranoid=4` 拒绝，`profile_samples=0`，所以
“Hungarian 是主要瓶颈”仍是高概率候选，不是直接 profile 结论。为避免污染正式
训练，没有尝试更侵入的 strace/gdb attach。

## 14. 失败、纠偏与价值

| failure | root_cause | correction | scientific_impact |
|---|---|---|---|
| S0 batch8 OOM before update1 | dense first batch 超出显存 | 回退 batch3，固定 50 optimizer updates | 无科学污染；S0 仅工程 smoke |
| grouped repeat epoch12 assertion | stock loop 少算 variable epoch 总迭代 6 次 | `VariableBatchEpochBasedTrainLoop` + boundary resume | 恢复后完成；揭示 seed sensitivity |
| epoch11 direct resume unsafe | iter4,427 不在 accumulation boundary | 从 epoch10 iter4,024 重放完整 epochs11–12 | 不丢 pending half-gradient |
| full24 third backward failure | reentrant Swin checkpoint + DDP + accumulation ready-twice | full24-only `with_cp=False`, `static_graph=False` | 同一模型/数据 clean restart，科学 recipe 不变 |
| live CPU profile denied | kernel `perf_event_paranoid=4` | 停止侵入式 attach，保留静态诊断 | 没有瓶颈实测结论 |
| epoch6 validation only 80/1730 | 用户要求 checkpoint 后立即停卡 | 保存、SHA256、SIGINT、释放显卡 | epoch6 AP 明确 deferred，不算失败 |

这些失败的共同价值是把长训中的“隐性不可信”转成可复现工程边界：variable batch
计划、accumulation boundary、DDP reentrant checkpoint 和 milestone 保存现在都有
明确测试或审计，不需要靠偶然跑完来证明可靠。

## 15. 对当前方法的研究判断

### 15.1 已经成立的结论

1. generic-only、无教师、无 pseudo、OV、E2E、Q600、无 NMS/top-k、旋转框这条
   训练/推理合同可以在真实 DOTA-v2 上稳定运行。
2. grouped O2O 在主种子小数据口径上是唯一通过冻结 AP gate 的候选。
3. grouped 的收益主要是更好的总体排序和较少 duplicates，不是增加推理容量。
4. Hausdorff/Chamfer 改善局部 geometry 或 duplicates，并不自动改善 AP。
5. S1 400 图结果不能替代 raw 13,833；raw mouth 暴露 coverage、novel/rare 和
   empty-tile calibration 三重短板。
6. full-data epoch1 相对 pre-full raw mouth 有真实 `+0.0350 AP50` 增益。
7. full training 到 epoch6 数值稳定、采样完整、checkpoint 可恢复。

### 15.2 尚未成立的结论

1. 不能说当前模型已经达到或接近 AP70；唯一 full raw AP50 是 0.3440。
2. 不能说 epoch6 比 epoch1 AP 更高，因为 epoch6 尚未完成验证。
3. 不能说 grouped 是 seed-robust improvement；matched repeat 只有 +0.0040。
4. 不能说降低 training loss 必然提高 raw AP。
5. 不能说 scale1024 一定有效；目前只有原生物体尺寸的机制证据，没有 matched AP。
6. 不能说 text LR 是主要 novel-class 原因；目前只有 BERT drift 诊断，没有配对 AP。
7. 不能把 P126C 0.6570 当成合规 clean-start 结果或 0.70 完成证据。

## 16. 为什么 scale1024 成为首要备选

raw validation 共 243,632 个 GT。在当前 `1024→800` resize 后，短边 `<4 px` 和
`<8 px` 分别占 18.044% 和 51.646%；保持 1024 输入后预计降到 5.440% 和
33.337%。150,145 个 small-vehicle 在 800 下有 73.769% 短边 `<8 px`，1024 下
可降到 49.372%。bridge 的 `<8 px` 比例也从 28.990% 降到 11.528%。

这与 epoch1 的 small-vehicle AP 0.160、bridge AP 0.144 相吻合，构成独立于 AP
曲线的分辨率机制证据。但 airport/container-crane/helipad/helicopter 并非主要
小目标，因此 scale1024 只能解决部分几何可见性问题，不能代替 rare/novel semantic
策略。

直接复制 batch4 到 1024 的线性显存外推约 53.8 GiB，会超过 A40。已预注册的安全
起点是 `batch_size=2/GPU × world_size=4 × accum=1`，effective batch 仍为 8；
启动前必须对 max-query-area 与 max-GT singleton 做真实 forward/backward，要求
peak memory <=42 GiB、loss/grad finite、world4 sampler duplicate=0/missing=0。

## 17. 恢复工作后的严格顺序

当前工作按用户要求暂停。恢复后不得直接续训 epoch7，正确顺序是：

1. 使用已校验 `epoch_6.pth`，在 raw `ss_val=13,833`、
   `filter_empty_gt=False` 上完成独立 validation；
2. 解析 `mAP`、`AP50` 和 novel-4 mean AP；
3. 应用预注册 epoch6 gate：
   `mAP >= 0.5060 && AP50 >= 0.5060 && novel4 >= 0.2500`；
4. 若三项全过，恢复当前 scale800 run，下一 raw gate 为 epoch12；
5. 若任一未过，保留 epoch6 checkpoint，先运行 matched real-2000 scale1024
   screen；晋级门槛为 `AP50 >= 0.4040`、`novel4 >= 0.125`，并通过全部
   600-row/OV/E2E/no-NMS/rotated-box audits；
6. 若 scale1024 仍表现为 semantic/novel 短板，再运行单变量 text-encoder LR
   candidate：只把 `language_model` LR 从 `1e-4` 降到 `1e-5`，要求相对直接对照
   `delta_AP50 >= +0.010`、`delta_novel4 >= +0.030`、base14 回退不超过 0.003；
7. 只有小数据门槛通过的候选才能进入下一轮四卡全量训练；最终完成声明仍必须来自
   raw 13,833 的 `mAP >= 0.7000` 且 `AP50 >= 0.7000`。

## 18. 关键资产与证据路径

### Configs and protocol

- `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_full24e.py`
- `docs/project_history/exp_20260715_dotav2_cleanstart_ov_e2e_ap70/s0-result.md`
- `docs/project_history/exp_20260715_dotav2_cleanstart_ov_e2e_ap70/s1-screen.md`
- `docs/project_history/exp_20260715_dotav2_cleanstart_ov_e2e_ap70/full24e-protocol.md`
- `docs/project_history/exp_20260715_dotav2_cleanstart_ov_e2e_ap70/subset-manifest.json`

### Full run

- work dir：`work_dirs/dotav2_cleanstart/full24e_grouped`
- main log：
  `work_dirs/dotav2_cleanstart/full24e_grouped/20260716_022933/20260716_022933.log`
- scalars：
  `work_dirs/dotav2_cleanstart/full24e_grouped/20260716_022933/vis_data/scalars.json`
- monitor：
  `work_dirs/dotav2_cleanstart/audits/full24e_grouped_monitor_fc7803e.jsonl`
- epoch1 checkpoint：
  `work_dirs/dotav2_cleanstart/full24e_grouped/epoch_1.pth`
- epoch6 checkpoint：
  `work_dirs/dotav2_cleanstart/full24e_grouped/epoch_6.pth`

### Experiment ledger

- `.lab/log.md`
- `.lab/results.tsv`
- `.lab/branches.md`

### Key commits

| commit | purpose |
|---|---|
| `e02b32b` | clean-start AP70 design |
| `6e63e52` | OV prompt、strict、decoder alignment audits |
| `69be524` | S0 dense-tile OOM recovery |
| `062c263` | freeze S1 protocol |
| `b392c1e` | physical-GPU JSONL monitor |
| `475346a` | deterministic sampler audit |
| `87c1394` | milestone checkpoint hook |
| `9175961` | Chamfer candidate |
| `e148623` | full24 protocol |
| `fb85034` | exact variable-batch epoch loop |
| `ea021e7` | accumulation-boundary alignment |
| `345289a` | lock grouped full-data winner |
| `fc7803e` | stabilize four-GPU accumulation |
| `bfab5b9` | record epoch1 raw result |
| `a3c3817` | pin epoch1 checkpoint hash |

## 19. 最终状态

本轮已经完成“先小数据筛选，再四卡全量训练”的阶段性承诺，并保存了一个可严格
恢复的 epoch6 节点。科学目标本身仍未完成：当前没有任何同时满足 clean-start、
无教师蒸馏、开放词汇、E2E、无 dense/RoI head、无 NMS、旋转框，并在 raw DOTA
Val 上达到 `mAP/AP50 >= 0.70` 的实验证据。

当前最有价值的资产不是一个被夸大的 AP，而是一个口径清楚、工程边界已修复、
checkpoint 已校验、下一步门槛已预注册的可继续实验状态。epoch6 raw 13,833
validation 已按下节完成；三项门槛均未通过，因此 800 路线不直接恢复 epoch7。

## 20. Epoch6 raw-13,833 独立验证结果

按恢复顺序，使用物理 GPU2、3 对已校验 `epoch_6.pth` 完成独立双卡验证。评测
严格使用 raw `ss_val=13,833`、`filter_empty_gt=False`、scale800、Q600 和
`DOTAMetric(iou_thrs=0.5)`；这不是 filtered-6605 paper mouth，二者不得混报。

| metric | epoch1 | epoch6 | delta | gate | decision |
|---|---:|---:|---:|---:|---|
| `dota/mAP` | 0.3443 | 0.4842 | +0.1399 | 0.5060 | fail (-0.0218) |
| `dota/AP50` | 0.3440 | 0.4840 | +0.1400 | 0.5060 | fail (-0.0220) |
| `novel4_AP50` | 0.143 | 0.204 | +0.061 | 0.2500 | fail (-0.0460) |

`novel4_AP50=0.204` 由 evaluator 输出的三位小数逐类 AP 计算：airport 0.440、
container-crane 0.055、helipad 0.000、helicopter 0.321。它足以判定未过 0.2500
门槛，但不应冒充 evaluator 未输出的更高精度 novel4 数字。

完整性核验显示 prediction records=13,833、unique `img_id`=13,833、
`pred_instances` missing=0、每图 rows min/max=600/600、rows!=600=0。日志没有
Traceback、OOM、NCCL error、CUDA error 或数值异常；结束后 GPU2、3 均回到
16 MiB、0% utilization。

- checkpoint SHA256：
  `3e47cf4c230313d84ec5498715d36d25ad8ba136b37c78540ff52b1a89f334c6`
- predictions SHA256：
  `1739dfb9c2e867b934c35c00005496cf3e0e51cea5e0855ed52f496d1af446b0`
- log：
  `work_dirs/dotav2_cleanstart/eval_epoch6_raw13833_gpu23/launcher.log`
- predictions：
  `work_dirs/dotav2_cleanstart/eval_epoch6_raw13833_gpu23/predictions.pkl`

结论是 scale800 在 epoch1→6 有显著同口径提升，但三项预注册门槛全部失败，因此
不得直接恢复 epoch7。保留 `epoch_6.pth`，下一项实验应按预注册顺序执行 matched
real-2000 scale1024 screen；本次验证没有自动启动后续训练。
