# DOTA-v2.0 父模型保持式融合执行日志

## 2026-07-15 00:00–00:30：资产恢复与方案修正

- 建立分支 `research/dotav2-parent-preserving-causal`。
- 找到精确数据口 `DOTA2_1024_500/ss_train` 与 `ss_val`。
- 找到历史 P134B checkpoint，但展开配置证明其为 ResNet、两层 decoder、
  P15B head，与当前 Swin-T、六层 OV-CapFlow 拓扑不兼容。
- 因此把方案修正为：先用当前代码四卡构建一轮 C0，再训练冻结父模型的
  一轮 C1。P134B 只保留为外部参考。
- 将 GPU 调度固定为四卡训练和两条双卡复放线，物理卡只用 4、5、6、7。

## 2026-07-15 00:30–00:50：TDD 实现

- 数据口审计器先出现预期 import failure，再实现并通过 5 个测试。
- DN query-budget sampler 先出现预期 import failure，再实现并通过覆盖、
  确定性、rank 不相交、稠密缩批和 JSON 审计测试。
- DOTA2 C0/C1 配置先以 file-not-found 失败，再实现并通过完整展开差分。
- 空瓦片 score-mass 指标先因缺少 `scores` 参数失败，再做向后兼容扩展。
- 便携测试从 50 passed/5 skipped 增长到 59 passed/5 skipped；正式提交前
  仍需重新运行最新全套测试。

## 2026-07-15 00:50–01:05：真实数据与最坏批工程门

- 真实 mouth audit：47,294/13,833 图像和标注一一对应；训练空瓦片 22,575，
  验证空瓦片 7,228；18 类 token 完整。
- 四 rank sampler preflight：1,973 updates，零重复、零缺失；local batch
  范围 1–6；16 次缩批；覆盖 SHA256 为
  `6f3b805c95343fe05381ef4205c18e1d7b7f397c383137eb763dedc3c1de9eea`。
- C0/C1 普通真实 batch=6 均完成有限 loss 与 backward；C1 有效梯度只在
  semantic-fusion 参数。
- 最大 query-area 六图批次为索引
  `[9907, 27754, 28972, 36743, 42754, 43125]`，GT 数
  `[1342, 0, 0, 0, 0, 0]`。C0/C1 峰值分配显存约 38,407/14,937 MiB。
- 最大 GT 单图索引 13,329，GT=3,396。C0/C1 峰值分配显存约
  18,860/13,353 MiB。两种极端均通过有限前反向门。

## 2026-07-15 01:10：预注册锁定

- 补齐机制指标的量化阈值与未覆盖结果的 do-not-promote 归类。
- 写入 fplan/flog/faudit/fres 四份角色分离记录。
- 正式 C0 优化尚未启动；下一步是最新全套验证、提交 Experiment 6 实现，
  然后用 GPU4–7 启动 C0。

## 2026-07-15 01:15–01:17：C0 attempt 1 工程失败

- 提交 `7f642fc` 后在 GPU4–7 启动四卡 C0，端口 29661，NCCL P2P/IB
  均关闭。
- 四 rank sampler 正式审计正确写出 world_size=4、seed=20260712、
  1,973 updates、duplicate=0、missing=0，与预检覆盖哈希一致。
- 第一个 iteration 已完成，但第二次 forward 前四个 rank 同时触发 DDP
  `Expected to have finished reduction`。失败耗时约 67 秒；无 OOM、NaN 或
  NCCL 错误。
- 四 rank 未梯度参数索引完全相同：198、235–240、691–694。映射为
  `bbox_head.cls_branches.6`、`bbox_head.reg_branches.6` 和
  `memory_trans_fc/norm`。
- 根因：OVCapFlow 固定 query 路径绕过 scored encoder proposals，并把
  encoder head 输出设为 None，因此上述参数静态不参与 loss；MMEngine DDP
  默认 `find_unused_parameters=False`。
- 本次失败目录和 `launcher.log` 保留不覆盖。恢复方案先以配置测试约束
  `find_unused_parameters=True`，再提交并在独立 recovery 目录启动。

## 2026-07-15 01:22–01:23：DDP recovery smoke 1 失败

- 修复提交：`7044d4d`，共同配置加入 `find_unused_parameters=True`；配置和
  便携测试通过。
- 使用独立 `c0_ddp_smoke` 目录和端口 29662 跑四卡三步 smoke。
- sampler 再次通过，但第一个 backward 触发
  `Expected to mark a variable ready only once`，四 rank 均指向参数索引
  169，即 `backbone.stages.3.blocks.1.ffn.layers.1.bias`。
- 根因是 Swin `with_cp=True` 使用 reentrant activation checkpointing；
  PyTorch 1.12 DDP 文档明确说明普通 `find_unused_parameters=True` 不支持
  “checkpointing + unused parameters”，而 `static_graph=True` 专门支持该
  组合。
- 保留 smoke 失败日志；下一最小假设改为显式
  `MMDistributedDataParallel(static_graph=True, find_unused_parameters=True)`。

## 2026-07-15 01:27–01:40：DDP recovery smoke 2 通过

- 修复提交 `f45c7cb` 只把共同 wrapper 改为显式 static graph；模型、数据、
  sampler、优化器、seed 和判定门均未改变。
- GPU4–7 上的三步训练全部完成，loss 分别为 125.6257、429.4222、
  302.4970，均为有限值；峰值日志显存 33,026 MiB。未再出现 reduction、
  ready-twice、OOM、NaN 或 NCCL 错误。
- sampler 仍为 world size 4、seed 20260712、1,973 updates、零重复、零缺失，
  覆盖哈希与预检一致。
- 冒烟随后完整跑完 raw-13,833 验证并正常退出；三步随机初始化模型的
  `dota/mAP=0.0000` 仅是工程 smoke 观测，不计作 C0 科学结果。
- 为原样保留失败的 attempt-1 目录，正式 C0 输出固定到
  `dotav2_c0_native_1e_recovery_static`，C1 的 parent 路径同步指向其
  `epoch_1.pth`。路径约束先失败后通过，最新便携测试为
  62 passed、10 skipped。

## 2026-07-15 01:42–03:27：正式 C0 完成

- 从提交 `cd444b0` 启动 GPU4–7 正式恢复，四 rank 完成全部 1,973 updates；
  sampler 为 world size 4、seed 20260712、duplicate=0、missing=0，覆盖哈希
  与预检一致。
- 训练全程没有 DDP、OOM、NaN、NCCL 或 traceback；最后记录的 loss 为
  18.5405，日志最大 memory 为 39,123 MiB，`nvidia-smi` 观察到的最大
  reserved 水位约 44.7 GiB。
- raw-13,833 验证完整结束：`dota/mAP=0.1107`、日志
  `dota/AP50=0.1110`。独立双卡 C0 复放再次得到完全相同的数值。
- `epoch_1.pth` SHA256 为
  `f5ccd6e83dfcc4191046e5eee1fe3b912f9565065954c576d540dd88baab6769`；
  C0 完整加载 missing=0、unexpected=0。

## 2026-07-15 04:58–05:19：C1 启动硬门通过

- GPU4–5 完整复放 C0，GPU6–7 并行完整复放零更新 C1；两边均覆盖
  13,833 图并得到 `dota/mAP=0.1107`、`AP50=0.1110`，18 类表逐项一致。
- 单批七组父模型可见张量全部 `torch.equal`，max absolute difference=0；
  全量预测再比较 13,833 图、2,766,600 rows，mismatch image=0、max
  absolute difference=0。
- C1 加载 C0 时仅允许缺少六层共 18 个 semantic-fusion 参数；非法缺失=0、
  unexpected=0，可训练参数也恰好只有这 18 个。
- C0/C1 strict 审计都为每图 200 rows、forbidden calls=0，不使用 NMS、
  top-k 或 min-area-rect。正式 C1 获准启动。

## 2026-07-15 05:27–06:29：正式 C1 完成

- GPU4–7 完成全部 1,973 个优化 update；只有六层共 18 个
  `semantic_fusion` 参数可训练，未打开 balanced、null 或 density 路径。
- 训练、保存和 raw-13,833 验证均正常退出；没有 DDP、OOM、NaN、NCCL、
  RuntimeError 或 traceback。最后记录的 loss/grad norm 为 16.2101/2.9479，
  日志最大 memory 为 14,000 MiB。
- 正式验证为 `dota/mAP=0.1253`、日志 `dota/AP50=0.1250`；相对 C0
  `0.1107` 的绝对增量为 `+0.0146`，已超过预注册 promote 门 `+0.003`。
- `epoch_1.pth` SHA256 为
  `ff04b55271a8303cc3cfb380caeb0b7ff041fce2859919d159aea49e2bebdb67`；
  best checkpoint SHA256 为
  `810fa8670cf636a005a4cb51386ea40b2b0885e67eba3a2b9069cae61d877356`。
- 训练后 checkpoint 重新审计为 missing=0、invalid missing=0、
  unexpected=0；冻结后 trainable set 仍恰好是上述 18 个参数。

## 2026-07-15 06:30–07:01：双重放与全量逐值复验

- 最初 A1/B1 重放在导入阶段因命令误加 `PYTHONNOUSERSITE=1`，导致用户
  site 中的 `importlib_metadata` 不可见而立即退出；模型尚未加载，判为
  `invalid-engineering`，失败目录保留不覆盖。
- 修正后 A2 使用 GPU4–5、B2 使用 GPU6–7 并行覆盖全部 13,833 图；两边
  都得到 `dota/mAP=0.1253`、`AP50=0.1250`，18 类 AP 表逐项一致。
- A2/B2 prediction SHA256 分别为
  `ce30c23799ba8c69ca45bd11c39f54b08fead5b950aa0cb2f6f98c29fdf8f191` 和
  `1fb6a79b71a006ef1325bd5344cab94c3f2b1cfb6dfb8cc6f9139abb6f12e2ed`；
  pickle SHA 因设备/序列化元数据而不同。
- 逐图比较 `img_id`、boxes、scores、labels 和 label names，共 13,833 图、
  2,766,600 rows，`mismatch_images=0`、`max_abs_diff=0.0`。
- 最新最终便携测试再次为 62 passed、10 skipped。

## 2026-07-15 06:38–07:36：全量机制指标与最终判定

- C0/C1 分别在 GPU4/GPU7 对全部 13,833 图和 2,766,600 条预测做逐图
  rotated-IoU、coverage、duplicate、空图 score mass 和 gate 统计；两边均
  正常退出，图像、GT=243,632、空图=7,228 与 strict rows 全部对齐。
- GT coverage 从 0.221030 提升到 0.246450，绝对 `+0.025419`、相对
  `+11.50%`，matched queries 增加 24,266；满足预注册的实质改善定义。
- duplicate extras/GT 从 0.267465 增到 0.341646，相对 `+27.74%`；空图
  前景 score mass 从 7.339223 增到 7.417990，相对 `+1.07%`，两项均变差。
- C1 matched/unmatched gate mean 为 0.460409/0.468401，gate gap
  `+0.007992`，不满足期望的 `<= -0.005`。
- 主指标增量 `+0.0146 >= +0.003`，因此不依赖 mediator 数量，严格按
  预注册判为 `promote`。解释限定为“覆盖驱动的净提升，伴随冗余与空图
  score-mass 代价”，不得写成校准或未匹配抑制已成功。
