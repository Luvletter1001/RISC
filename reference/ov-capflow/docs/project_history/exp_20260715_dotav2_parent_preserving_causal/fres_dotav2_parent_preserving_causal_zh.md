# DOTA-v2.0 父模型保持式融合最终结果

## 最终判定

**`promote`。** 正式 C1 的 raw-13,833 `dota/mAP=0.1253`，相对匹配 C0
`0.1107` 提升 `+0.0146`，约为预注册 `+0.003` 推进门的 4.9 倍。

该结论经过两条独立双卡全量复放和 276.66 万条预测逐值比较确认。机制证据
显示提升主要由更高目标覆盖驱动，同时 duplicate 和空图前景 score mass
变差；因此推进的是“父模型保持式融合配方”，不是“抑制/校准已经成功”的
叙事。

## 主指标与复验

| 口径 | dota/mAP | 日志 AP50 | 图像 | 结果 |
|---|---:|---:|---:|---|
| C0 训练后验证 | 0.1107 | 0.1110 | 13,833 | 控制值 |
| C0 独立双卡复放 | 0.1107 | 0.1110 | 13,833 | 精确复现 |
| 零更新 C1 双卡复放 | 0.1107 | 0.1110 | 13,833 | 精确父模型 |
| 正式 C1 训练后验证 | **0.1253** | 0.1250 | 13,833 | `+0.0146` |
| C1 独立复放 A2 | **0.1253** | 0.1250 | 13,833 | 精确复现 |
| C1 独立复放 B2 | **0.1253** | 0.1250 | 13,833 | 精确复现 |

- A2/B2 各输出 2,766,600 rows；逐图比较 boxes、scores、labels、label
  names 和 `img_id`，mismatch=0、max absolute difference=0.0。
- C1 checkpoint 完整加载 missing=0、invalid missing=0、unexpected=0；只有
  六层共 18 个 semantic-fusion 参数在训练中可更新。
- strict 输出始终为每图 200 rows，无 top-k、NMS 或 min-area-rect。

## 18 类 AP 变化

以下 AP 来自同一 raw-13,833 IoU=0.5 评测表；三位小数用于展示。

| 类别 | C0 AP | C1 AP | C1-C0 |
|---|---:|---:|---:|
| airport | 0.000 | 0.000 | +0.000 |
| baseball-diamond | 0.122 | 0.106 | -0.016 |
| basketball-court | 0.091 | 0.032 | -0.059 |
| bridge | 0.092 | 0.023 | -0.069 |
| container-crane | 0.000 | 0.000 | +0.000 |
| ground-track-field | 0.013 | 0.102 | +0.089 |
| harbor | 0.072 | 0.125 | +0.053 |
| helicopter | 0.000 | 0.000 | +0.000 |
| helipad | 0.000 | 0.000 | +0.000 |
| large-vehicle | 0.048 | 0.182 | +0.134 |
| plane | 0.384 | 0.384 | +0.000 |
| roundabout | 0.029 | 0.058 | +0.029 |
| ship | 0.145 | 0.218 | +0.073 |
| small-vehicle | 0.102 | 0.028 | -0.074 |
| soccer-ball-field | 0.004 | 0.083 | +0.079 |
| storage-tank | 0.229 | 0.227 | -0.002 |
| swimming-pool | 0.225 | 0.177 | -0.048 |
| tennis-court | 0.436 | 0.510 | +0.074 |

净增益集中在 large-vehicle、ground-track-field、soccer-ball-field、ship 和
tennis-court；small-vehicle、bridge、basketball-court 等出现明显回退。后续
工作不能只报总 mAP，必须保留该类别级取舍。

## 全量机制指标

| 指标 | C0 | C1 | 变化 | 判读 |
|---|---:|---:|---:|---|
| GT coverage | 0.221030 | 0.246450 | +0.025419 | 实质改善 |
| matched queries | 119,013 | 143,279 | +24,266 | 更多目标被命中 |
| duplicate extras/GT | 0.267465 | 0.341646 | +27.74% | 变差 |
| 空图前景 score mass | 7.339223 | 7.417990 | +1.07% | 变差 |
| matched gate mean | — | 0.460409 | — | 描述项 |
| unmatched gate mean | — | 0.468401 | — | 描述项 |
| gate gap | — | +0.007992 | — | 未达到 `<= -0.005` |

覆盖绝对提升 `+0.025419` 明显越过预注册 `+0.001` 实质改善线；但 duplicate
和空图 score mass 均向错误方向变化，且 unmatched gate 强于 matched gate。
最稳妥解释是：语义融合扩大了模型找到目标的能力，同时放大了冗余和部分
背景前景分数。

## 关键资产

- C0 `epoch_1.pth` SHA256：
  `f5ccd6e83dfcc4191046e5eee1fe3b912f9565065954c576d540dd88baab6769`
- C1 `epoch_1.pth` SHA256：
  `ff04b55271a8303cc3cfb380caeb0b7ff041fce2859919d159aea49e2bebdb67`
- C1 best checkpoint SHA256：
  `810fa8670cf636a005a4cb51386ea40b2b0885e67eba3a2b9069cae61d877356`
- C1 A2/B2 prediction SHA256：
  `ce30c23799ba8c69ca45bd11c39f54b08fead5b950aa0cb2f6f98c29fdf8f191` /
  `1fb6a79b71a006ef1325bd5344cab94c3f2b1cfb6dfb8cc6f9139abb6f12e2ed`
- 逐值比较：
  `work_dirs/ov_capflow_dotav2/audits/c1_replay_exact_comparison.json`
- C0/C1 机制指标：
  `work_dirs/ov_capflow_dotav2/audits/c0_mediators.json` /
  `work_dirs/ov_capflow_dotav2/audits/c1_mediators.json`

## 结论边界与下一步

本实验提供的是 DOTA-v2.0 全类、跨 HRSC 数据集的结构转移证据；18 类均参与
训练和验证，因此不能表述为 novel-class 或 open-vocabulary 泛化证明。

下一步应把本 C1 固定为新的候选锚点，单独预注册一个“保留 coverage、降低
duplicate/空图 score mass”的单变量实验。不能同时堆 balanced、null 和
density，也不能用后处理 NMS/top-k 掩盖 strict set-output 问题。
