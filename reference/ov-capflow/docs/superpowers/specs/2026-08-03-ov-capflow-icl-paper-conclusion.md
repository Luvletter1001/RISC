# OV-CapFlow ICLR 论文结论骨架（2026-08-03 E3 停止态）

## 核心主张结构

论文必须解耦三个问题：

1. 固定 `Q=600`、全查询计分、无 proposal/top-k/NMS 的 rotated-5D 直接集合预测是否形成可工作的严格推理口径；
2. 官方 generic GroundingDINO-B 初始化与更大闭集容量能否抬高 all-18 监督上限；
3. prediction-derived negative masking 能否缓解 base-only 不完整标注对未标 novel 实例的错误负监督，并改善真正的 strict-OV 泛化。

现有 E24 只回答第 1 项；B0 原计划回答第 2 项，但本次在 E1-only 状态停止，尚未回答容量上限问题；独立初始化且无泄漏的 N0/N1 与一次性 sealed novel4 才能回答第 3 项。三条证据链不可互相替代或继承选择信息。

## 当前已成立的结论

- E24 在完整 `13,833 × Q600` all-query raw mouth 上取得 exact mAP `0.6064053488274416`、rounded AP50 `0.6060`。
- 系统直接输出 rotated-5D boxes，不使用 proposal、top-k 或 NMS。这是 all-18 supervised closed-set substrate，不是 strict-OV 或 zero-shot 证据。
- OMQ source 显示 decoder 信息具有图像特异性与空间敏感性；其 M1 冻结端点没有改善，后续 actionability 又因 scene/pixel leakage 触发合同关闭。因此既不形成正方法结论，也不能外推为整个 geometry/query-flow 机制族无效。
- B0 已在完整 `13,833 × Q600` raw mouth 上通过 E1 可继续门：exact mAP `0.44958066940307617`、rounded AP50 `0.4500`，相对 `0.4000` 门限余量 `+0.0495806694`。随后按用户新指令完整训练至 E3 后主动停止；E3 不是注册验证/保存点，因此没有 E3 mAP 或 checkpoint，E6 与 `0.70` 目标均未观测。该结果只建立 E1 可行性和三 epoch 工程稳定性，不构成容量、创新、因果 backbone 或 OV 结论。N0/N1 未执行；official novel4 对 N0/N1 的问题选择、阈值、控制与 PASS/FAIL 决策继续封存。B0 的 all-18 supervised novel4 diagnostic 不得跨越这条信息防火墙；当前不得宣称 negative-safe 方法有效或 strict-OV 泛化已解决。

## B0 门控后的唯一允许句式

| 门控结果 | 允许写入结论的内容 |
|---|---|
| E1 mAP `<0.4000` | 当前 B0 闭集工程配方未通过初始化可行性门；不否定独立 N0/N1 假设。 |
| E1 mAP `>=0.4000` | 仅说明可继续至 E6，不构成容量、创新或 OV 结论。 |
| E1 通过后按用户指令在非评估 E3 停止 | 只报告 E1 指标、完整三 epoch 工程稳定性和主动停止；E6/0.70 均保持未解决，不从 E3 loss 推断 mAP。 |
| E6 未同时满足 `mAP>=0.5800` 且相对 T7 E6 `+0.0200` | 未建立预注册的早期闭集优势，B0 终止。 |
| E6 双门通过 | 建立早期闭集工程锚点；因容量、预训练、batch 与 update 数均改变，不是 Swin-B 因果消融。 |
| E12 mAP `<0.6500` | 未达到中期容量门，终止且不得选择更优中间 checkpoint。 |
| E12 mAP `>=0.6500` | 仅保留达到最终闭集目标的可能性。 |
| E18 未满足 `delta(E12)>=0.0100` 或 `mAP>=0.6900` | 按冻结规则判定平台期并终止。 |
| E18 满足任一条件 | 只获准继续至唯一 E24 端点，不得提前宣布成功。 |
| E24 exact mAP 与 rounded AP50 均 `>=0.7000` | “官方 generic GroundingDINO-B 初始化与冻结 all-18 配方共同在同一 raw mouth 下超过 0.70。”只能称非因果闭集容量锚点。 |
| E24 任一门失败 | 如实报告终点并判 B0 最终锚点失败；不得用 best epoch、单项指标或恢复训练改写结论。 |

## Future N0/N1 的证据责任

- N0：独立 generic clean-start；scene-disjoint base14 substrate；仅用预注册 meta-novel folds 固定训练稳定性、阈值、控制与 stopping rule；不读取 official novel4 geometry/count/metric。
- N1：与 N0 严格配对，唯一改变为 prediction-derived negative masking；不生成 positive pseudo-box targets；用于识别 incomplete-label negative supervision 的因果作用。
- sealed novel4：方法、seed、阈值和停止规则完全冻结后只解封一次；只承担最终 strict-OV 泛化判断，不参与选择、调参或救援。

## 禁止性边界

- 不使用 broad “first”；若以后需要首创性表述，必须有完整文献核验并限定任务、口径与日期。
- 不把 all18-trained novel4 slice 称为 zero-shot/open-vocabulary。
- 不把 B0 称为创新、negative-safe 证据或 Swin-B 因果消融。
- 不把 OMQ 合同失败外推为整个机制族无效。
- N0/N1 与 sealed novel4 完成前，不声称解决 incomplete-label OV detection。
- N1 只能称 “no positive pseudo-box targets”，不能泛称 pseudo-label-free。

## 中文 Conclusion Skeleton

本文将严格旋转集合预测中的推理口径、闭集容量与不完整标注负监督三者解耦。现有全18类监督模型在完整 `13,833 × Q600` 全查询口径下取得 exact raw mAP `0.6064053488` 和 AP50 `0.6060`，同时保持直接 rotated-5D 输出且不使用 proposal、top-k 或 NMS；这一结果建立的是闭集系统底座，而非开放词表证据。OMQ 审计发现 decoder source 包含图像特异且空间敏感的信息，但其匹配代理未改善冻结端点，后续 actionability 又因原场景与像素泄漏被合同关闭，因此我们不据此提出有效方法或否定整个机制族。

截至主动停止，B0 在同一完整 raw mouth 上唯一经过验证的指标是 E1 exact mAP `0.4495806694`、rounded AP50 `0.4500`，通过 `mAP>=0.4000` 的可继续门；随后完整训练至 E3，但 E3 不是注册验证/保存点。用户在 E3 post-epoch hook 后停止作业，因此 E6 与 `0.70` 目标均未观测，不能据 E3 loss 判断达到、未达到或可能达到该目标。B0 仅保留为 E1-only、非因果 all-18 闭集容量锚点尝试，并与未来 strict-OV 证据隔离。**关于 incomplete-label negative-safe learning 的结论仍待独立 base14/meta-novel 实验和 sealed novel4 最终评估。** 核心启示是：闭集工程稳定性不能替代对开放词表监督机制的独立、无泄漏验证。

## English Conclusion Skeleton

We disentangled three questions in strict rotated set prediction: the inference mouth, the closed-set capacity ceiling, and negative supervision under incomplete annotations. Our current all-18 supervised model reaches an exact raw mAP of `0.6064053488` (`0.6060` AP50) over the full `13,833 × Q600` all-query evaluation, while directly predicting rotated 5-D boxes without proposals, top-k selection, or NMS. This establishes a closed-set systems substrate, not open-vocabulary evidence. The OMQ audit further showed that decoder sources contain image-specific and spatially sensitive information, but its matched proxy did not improve the frozen endpoint, and the subsequent actionability study was closed after detecting scene and pixel leakage. We therefore draw neither a positive method claim nor a family-wide negative conclusion from OMQ.

At the user-directed stop, B0's only evaluated result is its E1 exact mAP of `0.4495806694` (`0.4500` rounded AP50) on the same full raw mouth, which passes only the `mAP>=0.4000` continuation gate. The recipe subsequently completed Epoch3, but E3 is not a registered evaluation or checkpoint milestone. The run was stopped after the Epoch3 post-epoch hook, leaving E6 and the `0.70` target unobserved; Epoch3 losses cannot establish whether that target was reached, missed, or likely. B0 therefore remains an E1-only, non-causal all-18 closed-set capacity-anchor attempt isolated from the future strict-OV evidence chain. Claims about incomplete-label negative-safe learning remain pending an independently initialized base14/meta-novel study and a final sealed-novel4 evaluation. The central takeaway is that closed-set engineering stability cannot substitute for an independent, leakage-free test of the open-vocabulary learning mechanism.

## 当前投稿成熟度判定

若 N0/N1 与 sealed novel4 未完成，当前最多形成严谨的 systems/protocol 结论；B0 的 E1 可行性与三 epoch 工程稳定性不能补足 ICLR 方法主张。只有独立、scene-disjoint、sealed 的负监督证据链成立后，论文才具备完整的方法结论。
