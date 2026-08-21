# DOTA-v2.0 父模型保持式融合审计记录

## 环境

- Python 3.8.19
- torch 1.12.1+cu113，CUDA runtime 11.3
- mmcv 2.1.0，MMEngine 0.10.4，MMDetection 3.3.0，
  MMRotate 1.0.0rc1，transformers 4.46.3
- GPU：NVIDIA A40 46,068 MiB，driver 535.309.01
- 解释器：`/data/zcy/anaconda3/envs/mmdet/bin/python`
- BERT：`/data1/zcy/LAEDINO/weights/bert-base-uncased`
- Swin-T：`/data/zcy/swin_tiny_patch4_window7_224.pth`

## 数据口审计

机器可读文件：
`work_dirs/ov_capflow_dotav2/audits/mouth.json`

| 项目 | 训练 | 验证 |
|---|---:|---:|
| 图像 | 47,294 | 13,833 |
| 标注 | 47,294 | 13,833 |
| 空标注瓦片 | 22,575 | 7,228 |
| 缺图/缺标注 | 0/0 | 0/0 |

软链接真实落点：

- train：`ss_train_full_1024_500_20260620`
- val：`/data1/zcy/datasets/DOTA2_1024_500_valcomplete_20260618/ss_val`

审计器读取到的类别 token 与预注册 18 类集合完全一致，配置测试另行固定
airport-first 顺序。

## 外部 P134B 边界

- checkpoint：`best_dota_mAP_epoch_10.pth`
- SHA256：`f8564985ae4104961dc475300bbf4b4d83f9a6665869c7d51c39e9d5d3776f19`
- 历史拓扑：`mmdet.ResNet`、`P15BOrientedDINOSetHead`、decoder 2 层、Q=200。
- 当前拓扑：Swin-T、`OVCapFlowHead`、decoder 6 层、Q=200。
- 判定：只能作为外部性能参考，禁止作为当前 C1 parent 或部分加载控制组。

## Sampler 审计

机器可读文件：
`work_dirs/ov_capflow_dotav2/audits/sampler_preflight.json`

- seed 20260712，world size 4，dataset size 47,294；
- update count 1,973；四 rank 相同；
- duplicate 0，missing 0；
- global batch 19–24，local batch 1–6；
- shrink updates 16；
- max GT 3,396，索引 13,329，singleton query area 48,888,064；
- 最大实际批 query area 49,904,736，小于预算 50,000,000；
- coverage SHA256：
  `6f3b805c95343fe05381ef4205c18e1d7b7f397c383137eb763dedc3c1de9eea`。

## 真实 batch 与显存门

| 配置 | 普通 batch | 最大-area batch | max-GT singleton | 梯度边界 |
|---|---|---:|---:|---|
| C0 | finite forward/backward | 38,407 MiB | 18,860 MiB | 全模型 |
| C1 | finite forward/backward | 14,937 MiB | 13,353 MiB | 仅 semantic fusion |

上述峰值是 `torch.cuda.max_memory_allocated()`，不是正式 DDP 训练峰值；正式
训练仍需监视 DDP bucket 和 reserved memory。

## 正式运行证明状态

- C0/C1 checkpoint、主指标、双通道复验、逐值一致性与全量 mediator 均已
  完成，没有待补科学运行证据。

## 正式 C0 与 C1 启动门

- 科学运行提交：`cd444b0`；C0 完成全部 1,973 updates，正式 sampler
  audit 与预检一致。
- raw-13,833 训练后验证和独立双卡复放均为
  `dota/mAP=0.1107`、`dota/AP50=0.1110`。
- `epoch_1.pth`：1,978.3 MiB，SHA256
  `f5ccd6e83dfcc4191046e5eee1fe3b912f9565065954c576d540dd88baab6769`；
  C0 加载 missing=0、unexpected=0。
- C1 parent 加载只缺少预期的 18 个 semantic-fusion 参数；
  invalid missing=0、unexpected=0，冻结后 trainable set 正好为这 18 个。
- 七组 parent-visible 张量 exact=true、max abs diff=0；全量 13,833 图、
  2,766,600 prediction rows 也为 mismatch=0、max abs diff=0。
- C0 与零更新 C1 的复放预测文件大小都为 183,952,270 bytes；其 pickle
  SHA 不同是因为配置/设备元数据不同，逐值预测内容完全相等。
- C0/C1 strict：每图 200 rows，forbidden calls=0，NMS/top-k/
  min-area-rect 均为 false。
- 日志最大 memory 39,123 MiB；训练时 `nvidia-smi` 最高观察到约
  44.7 GiB reserved，未 OOM。

## C0 attempt 1 失败审计

- 失败类型：DDP 静态 unused parameters，`invalid-engineering`，不计入 C0
  科学结果。
- 数据与 sampler：有效；world size 4，seed 20260712，覆盖完整。
- 数值/资源：未见非有限、OOM 或 NCCL 错误。
- 参数索引映射：
  - 198：`bbox_head.cls_branches.6.bias`；
  - 235–240：`bbox_head.reg_branches.6.*`；
  - 691–694：`memory_trans_fc.*`、`memory_trans_norm.*`。
- 代码证据：`OVCapFlow.pre_decoder()` 使用 `FixedRotatedQueryInitializer`，
  并在训练时设置 `enc_outputs_class=None`、`enc_outputs_coord=None`，上述
  scored-encoder proposal 参数不在 loss 图中。
- 恢复边界：只允许打开 MMEngine DDP 的 unused-parameter detection；不改
  data/model/optimizer/evaluator、seed、预算或判定门。

### Recovery smoke 1

- `find_unused_parameters=True` 解决了上一轮 reduction 检查，但与 Swin
  `with_cp=True` 的 reentrant backward 在参数索引 169 上冲突。
- 参数 169：`backbone.stages.3.blocks.1.ffn.layers.1.bias`。
- PyTorch 1.12 的 DDP `static_graph` 文档明确列出支持 reentrant backward、
  多次 activation checkpointing、以及 activation checkpointing 与 unused
  parameters 的组合。
- 下一恢复只替换 DDP wrapper 配置，不关闭 Swin checkpointing，以保留
  已验证的 C0 最坏批显存边界。

### Recovery smoke 2

- 提交：`f45c7cb`；目录：
  `work_dirs/ov_capflow_dotav2/c0_ddp_static_smoke`。
- 三个训练 iteration 均完成有限 forward/backward/update；训练日志峰值显存
  33,026 MiB，保存 `iter_3.pth`，并完成 1,730 个四卡验证 update。
- 完整日志中没有 traceback、RuntimeError、DDP reduction/ready-twice、OOM、
  NaN 或 NCCL 错误，tmux 正常退出，GPU4–7 回落至各 16 MiB。
- sampler audit 与预检完全一致：world size 4、seed 20260712、1,973 updates、
  duplicate=0、missing=0、同一 coverage SHA256。
- PyTorch 提示 static graph 可自行检测 unused parameters；当前同时保留
  `find_unused_parameters=True` 作为显式边界。该提示不影响执行，且三步已
  实证通过。
- 本次只判定为 `valid-engineering`。三步后的 `dota/mAP=0.0000` 不得写入
  C0/C1 科学比较。
- attempt-1 失败目录禁止覆盖；正式恢复目录固定为
  `dotav2_c0_native_1e_recovery_static`，C1 只从该目录的 `epoch_1.pth`
  读取 parent。

## 正式 C1 审计

- 1,973/1,973 更新、raw-13,833 验证和 best-checkpoint 保存均正常完成；
  无 DDP、OOM、NaN、NCCL、RuntimeError 或 traceback。
- `dota/mAP=0.1253`、日志 `AP50=0.1250`，相对 C0 的 mAP 增量为
  `+0.0146`。该增量越过预注册 `+0.003` promote 门。
- 日志最大 memory 14,000 MiB；最后记录的 loss/grad norm 为
  16.2101/2.9479。
- `epoch_1.pth` SHA256：
  `ff04b55271a8303cc3cfb380caeb0b7ff041fce2859919d159aea49e2bebdb67`；
  best checkpoint SHA256：
  `810fa8670cf636a005a4cb51386ea40b2b0885e67eba3a2b9069cae61d877356`。
- 训练后 checkpoint：missing=0、invalid missing=0、unexpected=0；
  trainable set 仍严格为 18 个 semantic-fusion 参数，balanced/null/density
  均为 false。

## C1 双重复验审计

- A2/B2 两条独立双卡复放均完整覆盖 13,833 图，并精确复现
  `dota/mAP=0.1253`、`AP50=0.1250` 和同一 18 类 AP 表。
- 两份预测各含 2,766,600 rows；逐图/逐张量比较为 mismatch=0、
  max absolute difference=0.0。机器记录：
  `work_dirs/ov_capflow_dotav2/audits/c1_replay_exact_comparison.json`。
- 最初 A1/B1 因错误的 `PYTHONNOUSERSITE=1` 在 import 阶段退出，未加载
  模型、未产生科学结果；修正仅删除该环境隔离项，不改变配置、checkpoint、
  数据、seed 或 evaluator。
- 最终便携套件：62 passed、10 skipped。

## C0/C1 全量机制审计

| 指标 | C0 | C1 | 变化 | 预注册解释 |
|---|---:|---:|---:|---|
| GT coverage | 0.221030 | 0.246450 | +0.025419 | 实质改善 |
| matched queries | 119,013 | 143,279 | +24,266 | 覆盖增加 |
| duplicate extras/GT | 0.267465 | 0.341646 | +27.74% | 回退 |
| 空图前景 score mass | 7.339223 | 7.417990 | +1.07% | 回退 |
| matched gate mean | 不适用 | 0.460409 | — | 描述项 |
| unmatched gate mean | 不适用 | 0.468401 | — | 描述项 |
| gate gap | 不适用 | +0.007992 | — | 未达到 `<= -0.005` |

- 两边均为 image_count=13,833、prediction_count=2,766,600、
  gt_count=243,632、empty_image_count=7,228，且每图严格 200 rows。
- 机器记录：`work_dirs/ov_capflow_dotav2/audits/c0_mediators.json` 与
  `work_dirs/ov_capflow_dotav2/audits/c1_mediators.json`。
- 机制证据支持“C1 找到更多可匹配目标”，不支持“C1 更好地抑制重复或
  空图前景分数”，也不支持 gate 已学会偏向 matched query。
