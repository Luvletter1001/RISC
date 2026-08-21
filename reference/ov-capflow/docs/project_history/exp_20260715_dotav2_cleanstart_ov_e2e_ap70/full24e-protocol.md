# DOTA-v2 Clean-start OV E2E — Full24e Protocol

## 状态

- 状态：`paused-after-epoch6-gate-fail`；正式重启时间
  `2026-07-16 02:29:33 +08:00`，launcher PID `399469`，日志时间戳
  `20260716_022933`。epoch6 checkpoint 已保存，独立 raw-13,833 验证已完成，
  未直接恢复 epoch7。
- 锁定赢家：`8-S1-G` grouped O2O，S1 best epoch 12，
  `mAP=0.4068`、`AP50=0.4070`，相对控制 AP50 `+0.0260`。
- 启动门已通过：Chamfer 最终 AP50 `0.3750`，低于控制；seed20260716
  对照/分组为 `0.3800/0.3840`，仅 `+0.0040`，因此 grouped 带有明确的
  种子敏感风险，但冻结主种子排名不变，仍是唯一可进入 full24 的候选。
- 当前执行提交：`fc7803e71ac0fc772f776214f50285e59f771472`。

## 配置与来源

- 配置：
  `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_full24e.py`
- 配置 SHA256：
  `e48eb8e5ce9cf248b73c124c78b8af34aed32d21dfeb30a4897f75d313c7190b`。
- 兼容 clean-start checkpoint SHA256：
  `e4b612bedf7ce0d78064bfacdf5b3641e265aaf1a75c0f3c2d82a924bd337d16`
- provenance JSON SHA256：
  `d73f9221b893e9248723fd7ad5b660e961b6ae63d9e41eeb7813c226c784b4ed`
- 继承 S1 grouped，不加载任何 DOTA、OpenRSD、遥感父权重；不使用 teacher、
  distillation、pseudo label、dense/RoI head、NMS 或 top-k。

## 原始数据口径

- 训练：`ss_train` 47,294 图/标注，其中空图 22,575；
  `filter_empty_gt=False`。
- 验证：raw `ss_val` 13,833 图/标注，其中空图 7,228；
  `filter_empty_gt=False`。
- 图像/标注零缺失，18 个 DOTA-v2 类齐全。
- mouth audit SHA256：
  `286fbdba409c27c235846b7609d5f00924d271adf2c4d41bc50075ea2a5920ba`。
- filtered-6605 只能作为补充结果，不能用于晋升或完成声明。

## 训练与保存

- physical GPUs：`4,5,6,7`。
- 每卡 batch 4，accumulation 2，四卡 effective batch 32。
- 24 epochs；验证在 epoch `1,6,12,18,24`。
- `MilestoneCheckpointHook` 只保存 epoch `1,6,12,18,24`。
- Q600 主查询，训练期 3 个共享 grouped O2O 组；推理只输出原顺序主 Q600。
- AdamW、500 update linear warmup、24 epoch cosine schedule；`resume=False`。
- 四卡 full24 专用执行关闭 Swin reentrant checkpointing
  (`backbone.with_cp=False`) 和 DDP static graph (`static_graph=False`)。
  这不改变模型、损失、数据或有效 batch，只保留激活以避免 PyTorch 1.12
  在 reentrant checkpoint + DDP + accumulation=2 下重复触发参数 ready hook。
- 使用 `VariableBatchEpochBasedTrainLoop` 预计算每个 epoch 的确定性采样
  长度，并以真实总和初始化梯度累积计数；这避免 query-area 动态批计划使
  原生 `首轮长度 × epoch数` 低估总迭代数。该循环不改变样本顺序、损失、
  学习率、有效 batch 或模型结构。
- 采样器设置 `update_count_multiple=2`：当原始计划为奇数 update 时，只
  把一个每卡至少含2图的大批交错等分成两个非空小批。因此每个 epoch 都在
  accumulation 边界结束，epoch 1/6/12/18/24 检查点均可严格恢复。

## 四卡采样与最坏样本预检

- world size 4 重放覆盖 47,294/47,294，duplicate=0，missing=0；每 rank
  首轮 2,962 updates，local batch 1–4，57 个批次因 query-area 预算缩小。
- 24轮逐轮 update 数为：20轮 `2,962`、第6/8/12/17/20/21轮 `2,964`；
  后6轮各有1次 accumulation 对齐拆分。总计 `71,100` updates/rank，所有
  轮次均为偶数，逐轮检查均为 duplicate=0、missing=0。
- sampler audit SHA256：
  `ab1f7ec8deaf4864684c51178a1572c0cff0221d18b91d715e26ce790f136c52`。
- 最大 GT 单图：index 13,329，3,396 GT；估算 query area 73,822,464，
  超过 50,000,000 预算，因此采样器以 singleton 放行，不能拆图或丢图。
- 该真实最坏单图的 grouped Q1800 forward/backward 已在 A40 通过，峰值
  allocated memory 26,009.9 MiB，无 OOM、NaN 或 Inf。
- 配置/里程碑/监控/严格审计回归：29 passed，9 个 opt-in GPU 测试 skipped；
  最坏样本 opt-in 测试另计 1 passed。
- 可变轮长与 accumulation 对齐加入后，完整 OV-CapFlow 回归为
  146 passed、14 个 opt-in skipped。

## 四卡启动故障与修复证据

- 首次启动在完成第2步、进入第3次反向时退出；原始
  `static_graph=True` 只暴露为 `run_backward returned NULL`。该进程没有保存
  checkpoint，正式重启仍从同一个 generic clean-start checkpoint 开始。
- 同步 CUDA A/B 复现后，关闭 static graph 得到准确错误：
  `backbone.stages.3.blocks.1.ffn.layers.1.bias` 被 reentrant checkpoint 的
  backward 标记 ready 两次。
- 同时关闭 Swin checkpointing 与 static graph 后，四卡相同数据、模型、
  accumulation=2 连续通过8步，跨过原故障点和4个累积边界；峰值日志显存
  27,904 MiB，所有 loss/grad norm 有限。
- 修复回归为 146 passed、14 skipped，`git diff --check` 通过；修复提交为
  `fc7803e71ac0fc772f776214f50285e59f771472`。
- 正式重启已通过第20/2,962步：loss `56.4908`、grad norm `14.2666`、
  日志显存 `27,913 MiB`、time `3.2825 s/update`。30秒监控无 fatal pattern，
  采样到 GPU4–7 利用率 `93/91/75/81%`，NVML 显存最高 `42,979 MiB`。
- 监控：
  `work_dirs/dotav2_cleanstart/audits/full24e_grouped_monitor_fc7803e.jsonl`。

## Epoch 1 raw-13,833 中间结果

- epoch1 于 `2026-07-16 04:53 +08:00` 完成全部 `2,962` updates，最后一个
 记录点 update2960 的 loss 为 `11.1521`、grad norm 为 `65.8228`，所有 loss
  分量均为有限值；随后保存 `epoch_1.pth`（2,076,337,385 bytes），SHA256 为
  `7b1ab328d85ff7f7d7f5c304822b81b8514efc95bd39d7ebaacaa317a3c14377`。
- 同一未重启进程完成 raw all-patch `ss_val=13,833` 验证，明确保持
  `filter_empty_gt=False`、18 类和每图 600 个原始 query rows。结果为
  `dota/mAP=0.3443`、`dota/AP50=0.3440`。
- 与 full-data 训练前同口径 8-D1 的 `AP50=0.3090` 相比提高 `+0.0350`；按
  evaluator 输出的逐类四舍五入 AP 估算，base14/novel4 AP50 约为
  `0.402/0.143`，训练前为 `0.360/0.049`。这说明 full-data 学习同时改善了
  基础类和新类，但仍未达到 `0.7000` 完成门槛。
- 当前明显弱类包括 helipad `0.000`、airport `0.095`、container-crane
  `0.127`、small-vehicle `0.160` 和 soccer-ball-field `0.172`；同时
  tennis-court `0.795`、plane `0.699` 证明模型已形成有效定位与排序能力。
- 验证于 `05:09 +08:00` 完成后自动进入 epoch2。到 update200/2962，loss
  `12.0964`、grad norm `67.4629`，四个 worker 均存活且无 fatal pattern；
  同时采样到 physical GPU4–7 利用率 `100/100/100/100%`，显存约
  `41.3–44.2 GiB`。下一次预注册的 raw 验证节点是 epoch6。

该结果只被登记为有效中间证据，不构成 AP70 完成声明，也不触发对运行中科学
配置的修改。

## Epoch 6 raw-13,833 独立验证与 gate

- checkpoint：`work_dirs/dotav2_cleanstart/full24e_grouped/epoch_6.pth`，
  SHA256 `3e47cf4c230313d84ec5498715d36d25ad8ba136b37c78540ff52b1a89f334c6`。
- mouth：raw `ss_val=13,833`，`filter_empty_gt=False`，scale800，Q600，
  `DOTAMetric(iou_thrs=0.5)`；physical GPUs `2,3`。
- result：`dota/mAP=0.4842`、`dota/AP50=0.4840`；按 evaluator 三位小数
  逐类 AP 计算的 `novel4_AP50=0.2040`。
- integrity：13,833 prediction records、13,833 unique `img_id`、每图严格
  600 rows、无缺失 `pred_instances`。
- gate：要求 `mAP>=0.5060 && AP50>=0.5060 && novel4>=0.2500`；实际三项
  分别差 `0.0218/0.0220/0.0460`，因此判定 `fail`。
- decision：不恢复 scale800 epoch7；保留 epoch6 节点，下一步执行预注册的
  matched real-2000 scale1024 screen。
- artifacts：
  `work_dirs/dotav2_cleanstart/eval_epoch6_raw13833_gpu23/launcher.log` 与
  `work_dirs/dotav2_cleanstart/eval_epoch6_raw13833_gpu23/predictions.pkl`；后者
  SHA256 为
  `1739dfb9c2e867b934c35c00005496cf3e0e51cea5e0855ed52f496d1af446b0`。

## 启动命令（赢家锁定后执行）

```bash
rtk env NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=4,5,6,7 MASTER_PORT=29670 /data/zcy/anaconda3/envs/mmdet/bin/python -m torch.distributed.run --nproc_per_node=4 tools/train.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_full24e.py --launcher pytorch --work-dir work_dirs/dotav2_cleanstart/full24e_grouped --cfg-options resume=False
```

启动后必须同时用 30 秒间隔监控 physical GPU4–7、launcher PID、训练日志、
OOM/NCCL/NaN/死 rank。epoch 1/6/12/18/24 均使用 raw 13,833 验证口径。
