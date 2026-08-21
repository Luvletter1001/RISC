# DOTA-v2.0 父模型保持式融合因果验证计划

## 状态

- 实验编号：Experiment 6
- 预注册时间：2026-07-15 01:10 +08:00
- 分支：`research/dotav2-parent-preserving-causal`
- 状态：已完成；C0、正式 C1、两条独立 C1 复放、逐值一致性和全量机制
  指标均通过，预注册最终判定为 `promote`
- 目标：在精确 raw-13,833 验证口径上，判断父模型保持式 C1
  语义融合是否优于同代码、同数据、同 seed 的 C0。

## 唯一科学问题

固定 OV-CapFlow Swin-T/BERT 拓扑、Q=200、seed=20260712、训练数据、
优化器和一轮预算，只把 C1 的 decoder 表达式改为：

```text
q_fused = q_parent + gate * P(q_native - q_parent)
```

gate 零初始化，因此 C1 在零更新时必须与 C0 精确相等。正式训练中只有
六层 decoder 的 `semantic_fusion` 参数可训练。balanced、null、density、
teacher、NMS/top-k 和 query-count 调参全部不进入本实验。

## 固定实验臂

| 实验臂 | 初始化 | 可训练参数 | 预算 |
|---|---|---|---|
| C0 | 本地 Swin-T 与 BERT 初始化 | 当前 OV-CapFlow 全模型 | 1 epoch |
| C1 | C0 `epoch_1.pth` | `decoder.layers.*.semantic_fusion.*` | 1 epoch |

历史 P134B checkpoint 只作为外部参考，不作为 C0/C1 parent。它是 ResNet、
两层 decoder、P15B head；当前模型是 Swin-T、六层 decoder、OVCapFlowHead，
不能做部分加载后冒充同一父模型。

## 固定数据口

- 根目录：`/data1/zcy/datasets/DOTA2_1024_500/`
- 训练：`ss_train`，47,294 张图和 47,294 份标注；保留空瓦片。
- 验证：`ss_val`，13,833 张图和 13,833 份标注；
  `filter_empty_gt=False`。
- 类别顺序：airport、baseball-diamond、basketball-court、bridge、
  container-crane、ground-track-field、harbor、helicopter、helipad、
  large-vehicle、plane、roundabout、ship、small-vehicle、
  soccer-ball-field、storage-tank、swimming-pool、tennis-court。
- 输入：800×800 等比例缩放；训练可做固定配置中的三方向随机翻转。

## 固定优化与采样

- Q=200；dynamic DN 目标 100 queries。
- 每 rank 最大 batch=6；四卡全局最大 batch=24。
- `DNQueryBudgetBatchSampler` 的每 rank query-area 上限为 50,000,000。
- 同一 seed/epoch 在每个 rank 重建同一全局顺序，再做不相交 rank 分片。
- 必须满足 47,294 全覆盖、零重复、零缺失、四 rank 更新步数相同、
  每个 local batch 非空。
- 优化器与其他 model/data/evaluator 字段在 C0/C1 展开配置中完全一致。

## GPU 4–7 调度

1. GPU4–7：四卡共同训练 C0。
2. GPU4–5：完整复放 C0；GPU6–7：并行复放零更新 C1。
3. GPU4–7：四卡共同训练 C1。
4. GPU4–5 与 GPU6–7：对同一 C1 checkpoint 做两次独立完整复验。

所有多卡命令固定带
`NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`。GPU 饱和是执行目标，但不能覆盖
数据口、等价性、strict 或数值安全门。

## 训练前工程门

以下任一不通过，禁止启动正式科学比较：

- 数据口计数、配对、类别 token 或软链接落点错误；
- sampler 覆盖不完整、重复、rank 步数不同或空 batch；
- C0/C1 真实 batch 非有限、OOM 或 C1 梯度越出融合参数；
- C0 checkpoint 不完整或加载有非法 missing/unexpected keys；
- 零更新 C1 与 C0 的七组父模型可见张量不精确相等；
- strict 推理不是每图 200 rows，或出现 top-k/NMS/min-area-rect；
- 完整验证不是 13,833 张图。

## 预注册判定

主指标是 raw-13,833 `dota/mAP`，比较 `C1 - C0`：

- `>= +0.003`：promote；
- `[0, +0.003)`：positive-under-gate，只记趋势；
- `>= -0.001` 且至少两个机制指标实质改善、coverage delta `>= -0.005`：
  park-with-mechanism-signal；
- `< -0.010` 且没有机制指标改善：stop-exact-recipe；
- 其他有限有效结果：do-not-promote；
- 任一工程门失败：invalid-engineering。

“实质改善”在结果出现前固定为：空图前景 score mass 相对下降至少 1%；
GT coverage 绝对提升至少 0.001；duplicate extras/GT 相对下降至少 1%；
或 C1 的 `unmatched_gate_mean - matched_gate_mean <= -0.005`。本轮不把
rare/dense-class recall 纳入 park 计数，因为尚未预注册独立确定性计算器。

## 结论边界

DOTA2 的 18 类都参与训练与验证。本实验能提供跨数据集结构/机制证据，
不能被表述为 novel-class 或 open-vocabulary 泛化证明。
