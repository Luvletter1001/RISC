# DOTA-v2 A1 位置质量监督 Stage-1 最终结果

## 最终判定

**`discard A1；禁止进入 Stage 2`。** 预注册的 Epoch 12 proxy 终点为
`dota/mAP=0.4643`、`dota/AP50=0.4640`，低于匹配 control 的
`0.4691/0.4690`，也没有达到 `AP50 >= 0.4790` 推进门。固定 novel4
仅为 `0.244500`，低于 `0.3675`；base14 为 `0.493786`，通过
`0.464643` 门槛。

机制上，A1 明显提高 evaluator-TP score--assigned rotated-IoU Spearman，
并提高 micro recall，但 fixed-recall oracle headroom 从 `0.188689` 增至
`0.204120`，没有按预注册要求收缩。因此指标 gate 为 1/3 通过，机制 gate
为 2/3 通过；合取 gate 失败。

该结果关闭“只对 matched-positive token map 施加 detached rotated-IoU
质量缩放”作为第一干预。不得追加 epoch、换 seed、调 floor/exponent、改
Hungarian cost、覆盖 DN，或与其他候选机制叠加救援。

## 完整 gate 表

| Gate | Control | A1 E12 | 变化/门槛 | 判定 |
|---|---:|---:|---:|---|
| AP50 | 0.4690 | 0.4640 | `>= 0.4790` | **FAIL** |
| novel4，官方三位表含零-GT helipad | 0.367500 | 0.244500 | `>= 0.367500` | **FAIL** |
| base14，官方三位表 | 0.464643 | 0.493786 | `>= 0.464643` | PASS |
| TP score--IoU Spearman | 0.300683 | 0.485816 | `+0.185133 >= +0.03` | PASS |
| micro recall | 0.492606 | 0.512242 | `+0.019636 >= -0.005` | PASS |
| oracle headroom | 0.188689 | 0.204120 | `+0.015431`，要求严格缩小 | **FAIL** |

novel4 的正式口径为 `(airport 0.160 + container-crane 0.000 +
helicopter 0.818 + helipad 0.000) / 4 = 0.244500`。canonical analyzer 的
`groups.novel.map=0.326118` 会排除零-GT helipad，只能作为诊断值，不能替代
预注册的四类固定均值。对应地，正式 base14 使用剩余 14 类官方三位 AP，
为 `6.913 / 14 = 0.493785714`；analyzer 的未取整诊断值为 `0.493945483`。

## 完整训练轨迹

选择始终使用 Epoch 12，以下中途轨迹没有用于早停或选 best checkpoint。

| Epoch | Control mAP | A1 mAP | A1-Control |
|---:|---:|---:|---:|
| 1 | 0.0872 | 0.0664 | -0.0208 |
| 2 | 0.1598 | 0.1820 | +0.0222 |
| 3 | 0.2389 | 0.2418 | +0.0029 |
| 4 | 0.2941 | 0.3507 | +0.0566 |
| 5 | 0.3567 | 0.3897 | +0.0330 |
| 6 | 0.3915 | 0.4131 | +0.0216 |
| 7 | 0.4099 | 0.4236 | +0.0137 |
| 8 | 0.4332 | 0.4269 | -0.0063 |
| 9 | 0.4360 | 0.4422 | +0.0062 |
| 10 | 0.4583 | 0.4559 | -0.0024 |
| 11 | 0.4625 | 0.4604 | -0.0021 |
| 12 | 0.4691 | 0.4643 | -0.0048 |

A1 在 Epoch 4 出现最大中途领先，但在最终四轮中没有保持优势。这是坚持
预注册终点而不使用 best checkpoint 的直接反例证据。

## Epoch 12 类别结果

以下为同一 400 图、Q600、无 NMS/top-k 的官方 IoU=0.5 三位表。

| 类别 | GT | Recall | AP |
|---|---:|---:|---:|
| airport | 4 | 0.750 | 0.160 |
| baseball-diamond | 38 | 0.842 | 0.457 |
| basketball-court | 14 | 0.500 | 0.419 |
| bridge | 64 | 0.688 | 0.245 |
| container-crane | 4 | 0.000 | 0.000 |
| ground-track-field | 26 | 0.577 | 0.514 |
| harbor | 202 | 0.550 | 0.294 |
| helicopter | 3 | 1.000 | 0.818 |
| helipad | 0 | 0.000 | 0.000 |
| large-vehicle | 1118 | 0.554 | 0.345 |
| plane | 313 | 0.802 | 0.689 |
| roundabout | 23 | 0.696 | 0.278 |
| ship | 1050 | 0.743 | 0.656 |
| small-vehicle | 5138 | 0.415 | 0.340 |
| soccer-ball-field | 13 | 0.846 | 0.366 |
| storage-tank | 128 | 0.758 | 0.682 |
| swimming-pool | 59 | 0.932 | 0.748 |
| tennis-court | 53 | 0.981 | 0.880 |

novel4 失败主要来自 airport 大幅低于 control、container-crane 仍为零以及
零-GT helipad 的固定计入；helicopter 保持高 AP 仍不足以补偿。base14 虽然
通过安全门，但 small/large vehicle 的 AP 与整体排序 headroom 说明密集目标
排序仍未解决。

## 机制解释

| 机制量 | Control | A1 | A1-Control | 判读 |
|---|---:|---:|---:|---|
| TP score--assigned IoU Spearman | 0.300683 | 0.485816 | +0.185133 | 质量对齐显著改善 |
| evaluator-reachable TP | 4064 | 4226 | +162 | 覆盖增加 |
| total GT | 8250 | 8250 | 0 | 口径一致 |
| micro recall | 0.492606 | 0.512242 | +0.019636 | 通过安全门 |
| oracle mAP | — | 0.668449 | — | 描述项 |
| oracle headroom | 0.188689 | 0.204120 | +0.015431 | 排序余量反而增大 |

最稳妥的因果解释是：target-only quality scaling 确实把正样本分数与定位质量
对齐，并让更多 GT 达到 evaluator TP，但没有把这种局部校准转化成更好的全局
TP/FP 排序。更大的 oracle headroom 与最终 AP50 回退共同否定了“该单一机制
已解决 strict Q600 排序瓶颈”的叙事。

## 完整性与复现证据

- 固定 seed `20260716`、12 epochs、1600 train / 400 val、rare4x、Q600、
  3 groups、6 decoder layers；只修改 A1 与 bookkeeping。
- 训练全程 loss/gradient 有限；没有 fatal、NCCL 错误、guard recovery、
  restart、sampler 或资源事件。
- 物理 GPU 8/9 用于训练，GPU 2/3 用于唯一终点 dump；没有触碰 GPU
  0/1/4/5 上的既有任务。
- `epoch_12.pth` 和 `last_checkpoint` 一致，稳定大小 2,081,251,817 bytes；
  最终评估与 dump 都使用 `epoch_12.pth`，没有使用 best checkpoint。
- dump validator：400 records、400 unique image IDs、240,000 rows、每图
  Q600、18 类、全部 CPU finite。
- same-dump parity error `1.0518466708742125e-08`；training replay delta
  `0.0`。

## 运行身份、数据与预算

| 字段 | 冻结值 |
|---|---|
| Git branch / endpoint commit | `research/dotav2-cleanstart-ov-e2e-ap70` / `c0c5dbf` |
| Saved resolved-config SHA256 | `3709b9d17173839bfad605abec600fd3bdf9f1400ffa0d9413dfbcb04d77a8dd` |
| Normalized control-equivalent config SHA256 | candidate/control 均为 `4449be70d7d2872f44d64a275d8c7e7fc4979f914084f8251520063d3ad46f3b` |
| Scientific-diff SHA256 | `c541f3b7f476e1ea6da967d536489f1058cccf5b06c2e15e04dcbe778267ebcd` |
| 唯一科学变量 | `position_supervised_cfg.enabled=True`：matched-positive token map 乘 detached aligned rotated IoU |
| Source checkpoint SHA256 | `e4b612bedf7ce0d78064bfacdf5b3641e265aaf1a75c0f3c2d82a924bd337d16` |
| rare4x train-manifest SHA256 | `1457c641d91a7e0bf26a62f0cd6c70c71d9e9c6df5a2137fe4b73b8e8fc05290` |
| proxy manifest-record SHA256 | `99b46b9871370ae529f11f8322ca6fdd52113bde1ee5129fbc809b718c61eddb` |
| val manifest content SHA256 | `a00b945ddd0a769008d57145feba28382fb9e56f5d6f1427413220f8008e1a2e` |
| Sampler / audit SHA256 | `DNQueryBudgetBatchSampler` / `99d7670f26318582537dfa318bb98fd5148f31f388be5977ec150d05ce191d4e` |
| Sampler coverage | 1600/1600，duplicate=0，missing=0，800 micro-steps/epoch/rank |
| Seed / epochs | `20260716` / 12 |
| GPUs / world size | physical 8,9 / 2 |
| Per-rank batch / accumulation / effective batch | 1 / 4 / 8 |
| Micro-steps / optimizer steps | 9,600 / 2,400 |
| Optimizer | AdamW，lr `1e-4`，weight decay `1e-4`，clip max-norm `0.1` |
| Observed train+val wall time | `2026-07-23 03:16:35–07:10:17 +08:00`，3h53m42s |

scientific-diff hash 的定义是对下列确定性命令的 stdout 取 SHA256：
`git diff --no-ext-diff --no-color 7a2d5d5..c0c5dbf --
projects/OVCapFlow/ov_capflow/ov_capflow_head.py
configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch1_rare4x_position_supervised.py`。
它只覆盖 A1 生产代码和 candidate config，不把测试或文档计入科学变量。

## Score-bin calibration

以下为 candidate 全 240,000 rows；TP IoU 是 evaluator assigned rotated IoU。

| Score bin | Rows | TP | TP precision | Same-label hit rate | Mean same-label IoU | Mean TP IoU |
|---|---:|---:|---:|---:|---:|---:|
| `[0,.05)` | 216,671 | 128 | 0.000591 | 0.036719 | 0.040022 | 0.591512 |
| `[.05,.10)` | 11,757 | 189 | 0.016076 | 0.292507 | 0.240471 | 0.605163 |
| `[.10,.25)` | 6,619 | 686 | 0.103641 | 0.380873 | 0.313245 | 0.653004 |
| `[.25,.50)` | 3,339 | 1,761 | 0.527403 | 0.689128 | 0.526618 | 0.704482 |
| `[.50,.75)` | 1,488 | 1,337 | 0.898522 | 0.920027 | 0.707135 | 0.756805 |
| `[.75,1]` | 126 | 125 | 0.992063 | 0.992063 | 0.785433 | 0.791716 |

## AP-support FP 与 GT reachability

AP-support 区间包含 24,715 rows，其中 4,017 TP、20,698 FP。

| AP-support FP 类型 | Count | FP share |
|---|---:|---:|
| localization/background | 12,716 | 0.614359 |
| duplicate | 5,577 | 0.269446 |
| empty tile | 1,910 | 0.092279 |
| semantic | 495 | 0.023915 |

| GT 状态 | Count | 8250-GT rate |
|---|---:|---:|
| geometry miss | 3,732 | 0.452364 |
| semantic miss | 292 | 0.035394 |
| ownership miss | 0 | 0.000000 |
| evaluator reachable | 4,226 | 0.512242 |

另有 13,485 个 same-label IoU>=0.5 candidate excess；599/600 个 query ID
至少成为一次 reachable witness。对 reachable GT，初始 prior 到最终框的中心距离
中位数为 18.005 px，scale/aspect/angle change 中位数分别为
`1.6813/2.1234/0.7154`。这进一步说明固定 query 数不是该 proxy 的直接硬上限，
主要未覆盖量仍是几何 miss。

## Epoch 12 matching 与 DN loss

最终训练日志点为 epoch 12 micro-step 9,600；下表保留 matching 与 DN 的独立
路径，不把 A1 误写为 DN 改动。

| 路径 | Final-layer cls | Final-layer bbox | Final-layer IoU | 六层分量合计 |
|---|---:|---:|---:|---:|
| Matching | 0.218285 | 0.407996 | 0.398408 | 7.380947 |
| DN | 0.052749 | 0.075775 | 0.144417 | 2.111624 |

总 loss 为 `9.492571`，pre-clip grad norm 为 `60.882886`，全部有限。A1 只
影响 matching target；DN target builder 和 DN loss 公式未变。

## Warning、retry 与 resume 账本

- 正式 Stage-1 训练：retry=0、interruption=0、resume=0、guard recovery=0；
  所有 rank 正常退出。
- Stage-0 首次 opt-in GPU smoke 随机抽到空-GT batch，上游 DN 零查询路径在
  backward/inference 前返回 detached zero；没有 optimizer step 或 checkpoint。
  测试夹具随后只做有界“首个非空 GT batch”选择，第二次完成有效
  forward/backward/inference。因此共尝试两次，只有一次完整有效 smoke。
- 重复但非 fatal 的线程/数值 API warning：torchrun 设置
  `OMP_NUM_THREADS=1`，MMEngine 设置 `MKL_NUM_THREADS=1`；tokenizers 在
  fork 后关闭并行；`torch.meshgrid` 将要求 `indexing` 参数；PyTorch
  `__floordiv__` deprecation；MultiheadAttention 缺少 key position
  encoding；scheduler 在 accumulation/warmup 初始化时报告
  `scheduler.step()` 先于 `optimizer.step()`。
- 重复但非 fatal 的框架/数据结构 warning：`optim_wrapper` 的 mmrotate
  registry scope 查找失败后回退到 mmengine registry；SwinTransformer 构造
  时报告没有 module-local pretrained weights（随后仍按审计加载冻结的通用
  source checkpoint）；`RotatedBoxes.clip` 为 no-op；`FileClient` 将弃用；
  `HardDiskBackend` 是将弃用的 `LocalBackend` alias。
- Endpoint dump 的非 fatal warning：`DumpResults` metric prefix 未设置；
  这只影响日志命名，不影响 pickle 或 DOTAMetric 数值。
- 上述 warning 类别在 paired control/既有 substrate 中同源存在；未发现
  OOM、NCCL、NaN/Inf、sampler、缺失数据或 loader stall warning。
- Analyzer 明确输出：`case studies not_run; diagnostic bundle is incomplete`。
  本次 proxy 的 P0148/P0682 case studies 没有运行；该 warning 不影响同一
  400-row-mouth 的 metric/mechanism gate，但禁止把 bundle 描述为完整案例诊断。
- 精确随机 smoke 样本不可复放；正式 train/val 数据顺序、seed、sampler audit、
  checkpoint 和 endpoint 均可复核。

## 关键资产

- Candidate checkpoint：
  `work_dirs/dotav2_cleanstart/s1_grouped_scale1024_seed20260716_gpu89_batch1_rare4x_position_supervised/epoch_12.pth`
  - SHA256 `d9f28af5e0607a1700113c480a805d4f014cd026b45907269e265b190fab6301`
- Prediction dump：
  `work_dirs/dotav2_cleanstart/eval_a1_e12_proxy400_q600_gpu23_dump/predictions.pkl`
  - SHA256 `4d099e3c3b2fed871b2a627ab1fa6c29f75006c856c346bef41279ebdd60a850`
- Official metrics：
  `work_dirs/dotav2_cleanstart/eval_a1_e12_proxy400_q600_gpu23_dump/20260723_071154/20260723_071154.json`
  - SHA256 `51f98d6a75a2c9bc5ac225ba290482a95b02090b4f81f1e254954e06ce27c680`
- Diagnostics：
  `work_dirs/dotav2_cleanstart/eval_a1_e12_proxy400_q600_gpu23_dump/diagnostics/diagnostics.json`
  - SHA256 `16f7f9f590aa2a7b54329011a84445db1b747f27e3f6d4f4d357da3fb4fe472e`
- Per-class CSV：
  `work_dirs/dotav2_cleanstart/eval_a1_e12_proxy400_q600_gpu23_dump/diagnostics/per_class.csv`
  - SHA256 `0ce7141c19aa26c15b213b1c5336446e06567b220dad093c6c18039748ba669f`
- Control diagnostics：
  `work_dirs/dotav2_cleanstart/eval_8_t6_r_e12_proxy400_dump/diagnostics/diagnostics.json`
  - SHA256 `9fcc8b44ac1bbe702fdeb43e4b83529a7692c58162392aebbfdb5d568db3ac29`
- Approved spec / execution plan：
  `docs/superpowers/specs/2026-07-22-dotav2-rotated-iou-position-supervision-design.md` /
  `docs/superpowers/plans/2026-07-23-dotav2-rotated-iou-position-supervision.md`

## 结论边界与下一路线

本实验是 400 图 proxy 上的单 seed 因果筛选，不是论文级重复，也不是 raw
13,833 全量 AP70 证据。A1 不能被表述为已提升最终检测性能；可以保留的结论
仅是“局部 score--IoU 校准与 recall 改善，但最终 AP 和全局排序未通过门槛”。

按已批准 spec 的 fallback，下一候选应作为独立实验转向固定 Q600 的
tiny/dense geometry family，针对 small-vehicle、large-vehicle 及 >600-GT
密集分层的 evaluator coverage/geometry miss。它必须重新预注册、保持 A1
关闭，不得与 A1 叠加，也不得用 NMS/top-k 掩盖 strict set-output 问题。
