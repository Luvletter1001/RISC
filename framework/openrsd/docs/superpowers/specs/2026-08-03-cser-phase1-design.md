# CSER Phase-1 十卡 Frozen-Feature Gate 设计规范

## 目标

在不改变旋转框回归、assigner、NMS 和正支持路径的前提下，将 CSER 写进 OpenRSD 的分类证据方程，并在 GPU 0–9 上完成一次 400 iteration 的 frozen-feature 真实训练 gate。目标不是直接宣称最终 mAP，而是验证可学习 counter-support 能否把 Phase-0 的 oracle headroom 转化为 held-out AP50 增益。

## 已批准约束

- 用户最新授权使用 GPU 0–9，覆盖此前仅 GPU 8/9 的限制。
- 只实现一个结构性机制，不叠加 router、decoder slot、新 detection head 或额外辅助 loss。
- 复用 A10 `OpenRotatedRTMDetSepBNHead`、标准 detection loss、`MetaRemoveRunner.trainable_parameters` 和现有 DOTA 数据管线。
- 训练仍是 E2E rotated-box detector 的分类路径适配；框回归输出与 NMS 路径不变。

## 方案比较

### A. Prompt-conditioned log-evidence ratio（采用）

用共享低秩变换从当前类别 support 动态生成 counter-support，随后在 logit 空间计算正证据减反证据。它适用于运行时任意类别 support，不把 DOTA 类别固化成 15 个静态参数。

### B. Negative-text auxiliary loss（拒绝）

项目已有 `FocusNegativeTextBank`，但其契约明确是 auxiliary-only，不能改变最终推理 score；继续沿此路无法直接解释 Phase-0 的 AP-sensitive background FP。

### C. Hard-FP memory/miner（暂缓）

它需要额外 mining、memory 更新和 provenance 管线，首个真实结果前会重新进入模块堆叠。只有方案 A 的真实 gate 成功后才有讨论价值。

## 核心方程

对每个 dense query `q` 和正支持 shot `p_cj`，保留原始 OpenRSD 正支持 logit `z_pos(c)`。共享 rank-8 residual transform 生成：

```text
p_neg(c,j) = normalize(p_cj + U(V(p_cj)))
z_neg(c)   = max_j [ scale * cosine(q, p_neg(c,j)) + bias ]
z_cser(c)  = z_pos(c) - strength * softplus(z_neg(c))
```

`strength` 初始化为 0，因此关闭或初始状态严格复现原始 logit；训练时它和共享低秩变换一起由现有 detection classification loss 学习。没有额外 loss、类别静态 memory 或 GT-at-inference 路径。

## 集成边界

- 新增一个聚焦模块：`M_AD/models/utils/counter_support_evidence_ratio.py`。
- 在 `OpenRotatedRTMDetSepBNHead` 内创建一份共享 adapter，并在三个 FPN level 的原始 `cls_logit` 后调用。
- 配置 `enable=False` 时直接返回输入张量，不创建输出分支差异。
- `trainable_parameters=['bbox_head.counter_support_ratio']`，其余 checkpoint 参数全部冻结。
- 使用现有 `FreezeNormStatsHook`，避免冻结模型的 BN buffer 漂移。

## 数据与评估口径

- checkpoint：`/data1/zcy/OpenRSD_results/results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24.pth`。
- 训练：从现有 `Data1_DOTA2.zip` 只提取同时存在于 `/data/zcy/dataset/trainval_ms_full/images` 的 58,564 个 formatted PKL；它们与 S2 的 2,500 个 Phase-0 tile ID 交集为 0。
- support：`/data1/zcy/datasets/OPENRSD_DATA/DOTAV2train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl`。
- 验证：复用 Phase-0 五个 GT/image 根，构造 2,497 个可恢复 S2 tile 的只读 symlink view；3 个历史临时 GT 已丢失，保持与 Phase-0 本地 baseline `0.756489` 同口径。
- gate：400 iteration、每卡 batch size 2、global batch 20、固定 seed 2024；训练后在同一 held-out S2 view 上评估 AP50。

## 五项启动与结果检查

1. `disabled_equivalence`：关闭 adapter 时输出逐元素一致。
2. `zero_init_equivalence`：开启且 `strength=0` 时输出逐元素一致。
3. `strict_trainable_scope`：只有 `bbox_head.counter_support_ratio.*` 可训练。
4. `data_disjointness`：训练 PKL stem 与 S2 tile stem 交集为 0。
5. `real_metric_gate`：候选 held-out AP50 相比 0.756489 至少不下降 0.002；若提升至少 0.003，则进入较长训练，否则停止或重设计方程。

## 十卡启动规范

单个 DDP job 使用 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9`、`--nproc_per_node=10`，并强制 `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`。`train_dataloader.batch_size=2`、`train_dataloader.sampler.batch_size=2`、`sampler.num_gpus=10`；不添加无 `type` 的 `batch_sampler`。

## 非目标

- 不跑全量 24 epoch；
- 不引入第二个 counter-support 模块；
- 不使用 S2 验证 GT 训练 adapter；
- 不把 Phase-0 oracle 数值描述为可实现结果；
- 不在首个 AP 结果前增加审计基础设施。
