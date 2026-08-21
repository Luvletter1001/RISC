# DOTA-v2.0 AP50 强锚点最终结果

## 结论

用户批准的 A 路线已经在物理 GPU `4,5,6,7` 上通过：P126C epoch40 的已保存
checkpoint 在 DOTA-v2.0 原始完整 `ss_val=13,833` 上重新评估得到：

- `dota/mAP = 0.6568131446838379`；
- `dota/AP50 = 0.6570`。

因此 AP50 50% 目标实际超过 `15.70` 个百分点，mAP 50% 双重完整性门槛也超过
`15.6813` 个百分点。结果不是引用旧日志，而是本轮新生成的完整预测和指标。

## 口径与模型

| 字段 | 值 |
|---|---|
| dataset | DOTA-v2.0 canonical 18 classes |
| validation | raw full `ss_val`, 13,833 images |
| empty tiles | `filter_empty_gt=False`, 7,228 empty-GT tiles retained |
| geometry | resize/pad to 1024 × 1024 |
| evaluator | `DETAILDOTAMetric(metric='mAP')`, IoU 0.5 |
| model | P126C P121 semantic-state strong detector anchor |
| checkpoint | `best_dota_mAP_epoch_40.pth` |
| SHA256 | `5babf6e5a8c335299705158ec49bb8b88f70508ff0e72c75323abb4d894af979` |
| GPUs | physical 4, 5, 6, 7 |
| batch | 32/GPU, 4 ranks |
| predictions | 13,833 records |

## Overall

| model/result | raw mAP | AP50 | 相对 AP50 目标 | 判定 |
|---|---:|---:|---:|---|
| OV-CapFlow C1（既有严格结果） | 0.1253 | 0.1250 | −0.3750 | 低底座，结论保留 |
| P126C 历史 epoch40 训练内评估 | 0.660601 | 0.6610 | +0.1610 | 只作先验 |
| **P126C 本轮保存权重复验** | **0.656813** | **0.6570** | **+0.1570** | **通过** |

本轮相对 C1 提高 `+0.531513 mAP` 和 `+0.5320 AP50`。这个差值代表 A 路线
切换到强检测父模型后的项目锚点，不应写成同一个 OV-CapFlow 架构内部的单变量增益。

本轮保存权重复验比历史训练内 epoch40 指标低 `0.003788 mAP`。可能涉及训练内
EMA 状态、保存权重语义或大 batch 推理差异；本报告不猜测具体原因，也不把两者
写成精确复现。两者都明显高于当前 0.50 目标。

## 逐类 AP50

| class | AP | recall |
|---|---:|---:|
| airport | 0.7984 | 0.9010 |
| baseball-diamond | 0.5958 | 0.9234 |
| basketball-court | 0.7606 | 0.9157 |
| bridge | 0.4874 | 0.8056 |
| container-crane | 0.0179 | 0.1972 |
| ground-track-field | 0.5690 | 0.8791 |
| harbor | 0.7690 | 0.8862 |
| helicopter | 0.7634 | 0.8873 |
| helipad | 0.3441 | 0.6667 |
| large-vehicle | 0.7973 | 0.9351 |
| plane | 0.8957 | 0.9620 |
| roundabout | 0.6951 | 0.8823 |
| ship | 0.8969 | 0.9554 |
| small-vehicle | 0.5318 | 0.7360 |
| soccer-ball-field | 0.5163 | 0.6419 |
| storage-tank | 0.7766 | 0.8821 |
| swimming-pool | 0.7021 | 0.8438 |
| tennis-court | 0.9052 | 0.9647 |

当前最明显的后续弱项是 `container-crane`、`helipad` 和 `bridge`；不过本轮目标是
整体 AP50 过 0.50，弱类优化不影响本轮达标判定。

## 执行与完整性

- 四卡计算期 GPU 4–7 利用率为 98–100%；每卡约 20,293 MiB device memory。
- MMEngine 日志峰值 9,511 MiB，无 OOM、NCCL error 或 traceback。
- 完整推理 109 steps，约 460 秒内完成模型初始化、推理、18 类汇总和预测保存。
- `predictions.pkl` 顶层长度经独立 pickle 读取为 13,833。
- 展开配置明确记录 `ann_file='ss_val/annfiles'`、
  `filter_empty_gt=False`、`GPU number: 4` 和固定 checkpoint。

## 证据路径

- 配置：`configs/strong_anchors/dotav2_p126c_raw13833_ap50_eval.py`
- 指标 JSON：
  `work_dirs/dotav2_ap50_strong_anchor/p126c_epoch40_raw13833_gpu4567/20260715_082204/20260715_082204.json`
- 完整日志：
  `work_dirs/dotav2_ap50_strong_anchor/p126c_epoch40_raw13833_gpu4567/20260715_082204/20260715_082204.log`
- 展开配置：
  `work_dirs/dotav2_ap50_strong_anchor/p126c_epoch40_raw13833_gpu4567/20260715_082204/vis_data/config.py`
- 预测：
  `work_dirs/dotav2_ap50_strong_anchor/p126c_epoch40_raw13833_gpu4567/predictions.pkl`

## 边界

本结果证明当前项目已经有一个同口径 DOTA2 强检测锚点超过 50%。它不证明严格
OV-CapFlow C1 本身从 12.5% 训练到了 65.7%，也不单独证明 novel-class 开放词汇
泛化。后续若继续做方法创新，应以本锚点为 parent，并要求零更新等价后仍保持
AP50 不低于 0.50。

