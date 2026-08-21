# CSER Phase-0 设计规范

## 目标

在不修改 detector、不重新推理、不启动训练的前提下，使用已有 A10 DOTA1 angle=0 原始预测与真实 GT，判断“背景 false hub”是否位于 AP 有效排序区间，并量化理想 counter-support 可恢复的 AP50 上界。

## 核心假设

CSER 将 open-vocabulary 分类改写为正支持证据与 prompt-conditioned 背景反支持证据的比值。Phase-0 不拟合反支持，只验证其必要前提：删除与任意 GT 的 rotated IoU 小于阈值的背景预测，是否能在不删除 TP 的条件下显著提高 AP50。

## 输入

- 原始预测：`resultmd/exp_rotation_semantic_attractor/evidence_closure_20260601/openvocab_dota1_ap_eval_20260603_fast_gpu45_bs64/predictions/openvocab_dota1_ap_raw_predictions.jsonl`
- 基线参考：angle=0 `mAP50=0.756870`
- GT：`/data/zcy/dataset/{trainval_ms_full,trainval_ss,trainval_1,test_ms,test_ss}/annfiles`
- 类别：标准 DOTA1 15 类。

## 单一实验

对 angle=0 的每个预测框计算其与该图像所有 GT 的最大 rotated IoU：

- `baseline`：保留全部预测；
- `bg_oracle_0p1`：删除 `max_iou_any_gt < 0.1` 的预测；
- `sv_bg_oracle_0p1`：只对 `small-vehicle` 删除上述背景预测。

三组预测使用同一个 `eval_rbbox_map(..., iou_thr=0.5, use_07_metric=True)` 评估。Phase-0 只报告 oracle 上界，不把使用验证 GT 的策略描述为可部署方法。

## 五项检查

1. `baseline_equivalence`：本地 baseline 与已有 angle=0 AP50 的绝对误差不超过 `0.002`。
2. `strict_output_scope`：输入只包含原始 `rbox/class/score`，本轮不改最终推理路径。
3. `tp_preservation`：oracle 删除的 baseline TP 数必须为 `0`。
4. `ap_active_background`：至少存在 score≥0.3 的背景 FP，且 oracle 能删除其中一部分。
5. `map_headroom`：`bg_oracle_0p1` 相对 baseline 至少提高 `0.005 mAP50`。

## 决策

- 五项全过：进入 GPU 8/9 上的 frozen-feature counter-support 拟合与零扰动检查。
- baseline 不等价：先修复 GT/metric 复现，不解释 oracle 数值。
- AP headroom 不足：CSER 降级为 reliability 诊断，不进入训练。

## 非目标

- 不实现 counter-support memory；
- 不增加 router、decoder slot 或新 loss；
- 不跑 12-angle 全量；
- 不启动 GPU 训练。
