# DOTA-v2 Clean-Start 开放词汇 E2E AP70 设计

## 1. 目标与不可变约束

本阶段从一个不含 DOTA、OpenRSD 或其他遥感检测训练结果的通用视觉语言
checkpoint 开始，在 DOTA-v2.0 上重新训练一个开放词汇旋转集合检测器。最终主
目标是在 raw all-patch `ss_val=13,833` 上达到：

- `dota/AP50 >= 0.7000`；
- 同一 evaluator 输出的 `dota/mAP >= 0.7000`；
- 推理为端到端集合预测，无 dense detection head、教师网络、蒸馏、伪标签、
  NMS、rotated NMS、全局 top-k 或其他预测行裁剪。

同时单独报告 paper-mouth `filter_empty_gt=True, dataset_len=6,605`，但该口径
只作辅助结果，不能替代 raw-13,833 主目标。

这里的“无头”指没有 YOLO/RTMDet/Faster R-CNN 式独立 dense/RoI 检测头；
Transformer decoder 每个 query 必需的文本相似度投影和五维旋转框回归层属于
集合预测本体，不被伪装成额外检测头。

## 2. Clean-start 定义

允许的初始化只有与 DOTA 和遥感检测训练无关的通用预训练：

- `/data/zcy/GroundingDINO/weights/groundingdino_swint_ogc.pth`；
- SHA256：
  `3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799`；
- 本地 `bert-base-uncased` 语言模型；
- 未能从通用 checkpoint 对齐的旋转角度、query reference 和新增几何参数使用
  固定随机种子重新初始化。

禁止加载或间接复制：

- 任意 DOTA、HRSC、DIOR-R、FAIR1M 或 OpenRSD checkpoint；
- P126C、P121、A10 epoch24、既有 OV-CapFlow C0/C1 等权重；
- 教师输出、离线/在线伪标签、蒸馏 logits、proposal cache；
- 从上述模型导出的 EMA、optimizer、query embedding 或类别原型。

训练从新的 work directory、epoch 0、空 optimizer/scheduler/EMA 状态启动，
`resume=False`。启动前必须生成 checkpoint provenance JSON，枚举来源、哈希、
missing/unexpected keys，并证明没有禁止来源。

## 3. 现状与根因

当前严格 DOTA C0 使用 `Q=200`、只训练一个 epoch，raw-13,833 为
`mAP=0.1107`；parent-preserving C1 为 `mAP=0.1253/AP50=0.1250`。已有
P126C raw 结果 `mAP=0.656813/AP50=0.6570` 来自 DOTA/OpenRSD 已训练权重和
不符合本阶段约束的检测路径，只能用作性能参照。

DOTA clean train 有 47,294 张 tile 和 836,745 个旋转框，其中 22,575 张为空
图。目标数分布显示：

| 每图 GT 阈值 | 超过阈值的训练图数 |
|---:|---:|
| 200 | 875 |
| 300 | 500 |
| 400 | 281 |
| 600 | 115 |
| 900 | 47 |

`p99=307`、`p99.5≈428`、`p99.9≈896`。因此 Q200 会对一部分密集图形成硬
召回上限；Q600 是 A40 显存、绝大多数图片容量与固定行严格推理之间的首轮
折中。115 张 `GT>600` 图片必须单独报告 coverage，不得通过 NMS/top-k 掩盖。

## 4. 路线选择

### 4.1 采用：OV-RHINO-CapFlow

在当前 `RotatedGroundingDINO/OVCapFlow` 代码路径内组合三类已有证据：

1. Grounding DINO 的文本条件特征增强、跨模态 decoder 与 token-level
   分类，保留运行时可替换词表的开放词汇接口；
2. RHINO 的旋转框顶点 Hausdorff 匹配与自适应 query denoising，缓解角度边界
   和静态噪声在训练后期反而变差的问题；
3. O2-DEIM 的角度分布迭代、顶点 Chamfer 二分图代价和旋转对比去噪，作为
   RHINO 匹配仍不足时的单变量升级项。

参考证据：

- Grounding DINO：<https://arxiv.org/abs/2303.05499>；
- RHINO：<https://openaccess.thecvf.com/content/WACV2025/papers/Lee_Hausdorff_Distance_Matching_with_Adaptive_Query_Denoising_for_Rotated_Detection_WACV_2025_paper.pdf>；
- O2-DEIM：<https://arxiv.org/abs/2603.15497>。

### 4.2 未采用：完整移植 O2-DEIM 后再增加语言分支

该路线闭集旋转性能强，但会同时更换 runner、数据格式、模型和开放词汇分类，
小数据结果难以定位增益来源，且当前机器没有现成代码树。它保留为主路线连续
三次有效实验无进展后的结构性 fork。

### 4.3 未采用：只放大现有固定 query 配方

仅增加 query 数和 epoch 的实现成本最低，但不能解决旋转匹配与去噪稳定性，
相对当前 12.5% 距离过大。它只作为小数据 S0 控制组，不作为默认全量长训候选。

## 5. 模型架构

### 5.1 开放词汇主干

- Swin-T 输出四级多尺度视觉特征；
- BERT 编码以句点分隔的动态类别提示；
- visual/text feature enhancer 进行双向跨模态融合；
- decoder query 与文本 token 做相似度分类；类别数不固化进 checkpoint 的
  最后一层形状；
- 所有主干、encoder、decoder 和映射层均参与 DOTA 训练，不冻结成只训练少量
  adapter 的旧 P126C 模式。

### 5.2 严格集合预测

- 推理固定 `Q=600` 个 matching queries；
- 每个 query 输出一个类别分数和一个 `(cx, cy, w, h, angle)`；
- Hungarian 一对一匹配决定监督；
- 推理逐 query 选择词表内分数最高类别，但不对 query 行排序或裁剪；
- 输出必须恰好每图600行，strict auditor 拦截 `topk`、`nms`、
  `rotated_nms`、`multiclass_nms` 和 `minAreaRect`。

### 5.3 训练期多组一对一监督

训练可复制三组共享 decoder 参数的 matching queries，每组独立进行一对一匹配
和辅助损失；推理只实例化/保留主组。它提供更多正样本监督，但不是 dense head、
教师或推理后处理。该机制必须满足：

- 三组共享预测参数，不能新增独立 dense classifier/regressor；
- group loss 在配置中可关闭；关闭时逐值等价于 Q600 控制组；
- inference graph 不包含 group 维度；
- checkpoint audit 证明没有 teacher/dense-head 参数命名空间。

### 5.4 旋转几何

第一候选只引入一个可归因的几何包：

- 将五维框转换为四顶点；
- Hungarian cost 在原 classification/L1/rotated-IoU 基础上加入对称
  Hausdorff corner cost；
- denoising 角度噪声使用周期最短路径，尺度噪声保持正宽高；
- 噪声强度依据当前匹配质量退火，避免训练后期静态噪声过强。

Chamfer cost、angle distribution refinement 和 rotated deformable sampling
不在同一首轮实验中一起堆叠。只有小数据几何包证明正增益后，才逐项追加，保证
实验结论可解释。

## 6. 数据与实验阶段

### 6.1 S0：真实标注 smoke

- 从 DOTA clean train 按固定 seed 选择64张非空图和32张空图；
- 必须覆盖18类，并额外包含至少8张 `GT>200` 密集图；
- 单卡运行50个 optimizer updates；
- 验证 forward/backward、有限 loss、600行输出、checkpoint provenance、动态
  prompt 和 strict no-NMS auditor；
- S0 不用 AP 作科学判定，也不能晋升为性能结果。

### 6.2 S1：2,000图小数据筛选

- 使用真实 DOTA GT，不使用 P33A/online label 或任何伪标签；
- 1,600 train + 400 validation，按类别、空图和密度分层，manifest 固定并保存
  SHA256；
- 18类 closed-train AP 用于快速优化；
- 另做14 base / 4 novel 的 prompt holdout，证明 checkpoint 不依赖固定18类
  classifier 形状；novel 结果只作开放词汇能力证据，不替代主 AP；
- 依次比较 Q600 控制、训练期三组O2O、Hausdorff adaptive-DN；每次只改变一个
  变量；
- 晋升条件：相对同 seed 控制组 AP50 至少 `+0.020`，或 AP50 至少 `+0.010`
  且 GT coverage 至少 `+0.020`，同时 duplicate extras/GT 不增加超过10%；
- 连续三个有效候选都未超过控制时，触发 O2-DEIM 结构 fork，不把无效小改动
  扩展到全量。

### 6.3 S2：四卡全量训练

- train：DOTA clean `ss_train=47,294`，`filter_empty_gt=False`；
- val：raw `ss_val=13,833`，`filter_empty_gt=False`；
- physical GPUs：4、5、6、7；
- 初始建议 batch 2/GPU、梯度累积4，effective batch 32；实际 batch 只由显存
  preflight 调整；
- Q600 时同时更新 dataloader batch 与 DN budget sampler，禁止构造缺少 type 的
  `batch_sampler`；
- 首次全量训练计划总长24 epochs，验证节点为 epoch 1、6、12、18、24；若
  epoch24 仍有明确
  上升趋势且未达标，按相同 run 继续至36/48 epochs；
- cosine/step scheduler、EMA 和 optimizer 均从空状态开始；
- 四卡命令必须设置 `NCCL_P2P_DISABLE=1` 和 `NCCL_IB_DISABLE=1`。

全量训练不能因为小数据 AP 较高而提前声称完成。只有 raw-13,833 完整结果超过
0.70 才满足主目标。

## 7. GPU 使用与运行健康

启动前在 GPU4 做一次单 batch 显存阶梯：batch 1→2→3。选择不超过约42 GiB
峰值的最大安全 batch，再在4–7号卡启动相同 per-GPU batch。训练稳定阶段目标：

- 每卡 GPU utilization 的滚动中位数不低于90%；
- 每卡显存尽量达到35–42 GiB，但显存占满不是优先于稳定性的科学指标；
- 每60秒以内记录一次 GPU、进程、loss、吞吐、错误扫描；
- 第一处 OOM 只降低 per-GPU batch 并等比例增加 accumulation；模型、数据、
  seed、学习率语义和 query 数保持不变；
- NCCL、NaN 或 sampler failure 必须停止队列并诊断，不能让后续实验连续空跑。

## 8. 评测与完成证据

### 8.1 主评测

使用原始完整 DOTA-v2.0 `ss_val=13,833`：18类、1024×1024、空 tile 保留、
IoU=0.5。保存 metric JSON、完整日志、展开配置、13,833条 prediction record、
checkpoint SHA256 和18类 AP/recall。

### 8.2 辅助 paper-mouth

同一 checkpoint、同一 prompt 和模型，只切换为 `filter_empty_gt=True`，确认
`dataset_len=6,605` 后报告。结果标题必须含 `filtered-6605`，不得与 raw AP
直接做增益声明。

### 8.3 开放词汇证明

至少同时满足：

- checkpoint 可在不改变参数形状的情况下接受重排、同义词扩充和新增类别
  prompt；
- 14-base/4-novel 小数据 holdout 有非零 novel AP/recall；
- 最终配置的类别预测来自 text-token similarity，不存在固定18维可训练 linear
  classifier；
- prompt audit 保存实际文本、positive map 和 token span。

### 8.4 E2E/no-head/no-NMS 证明

- strict runtime audit 对完整构造模型执行并通过；
- 每图固定600行；
- forbidden-call hits 为0；
- state_dict 不含 teacher、distill、pseudo、dense_head、rpn、roi_head；
- 展开配置中 NMS/top-k 预测字段不存在；
- 随机抽取预测与 decoder 原始输出逐值对齐，证明没有隐藏后处理。

## 9. 测试与故障处理

代码修改采用测试先行：

1. corner/Hausdorff 周期与梯度单元测试；
2. grouped O2O 关闭等价、训练开启、推理移除测试；
3. 通用 checkpoint provenance 与禁止权重来源测试；
4. base/novel prompt 维度无关性测试；
5. Q600 DOTA config、sampler coverage 和 real-batch 测试；
6. strict 600-row forbidden-call 测试；
7. 现有 OV-CapFlow portable suite 回归测试。

工程 crash 不计入科学胜负。配置、OOM、NCCL 和数据错误各自保留独立日志；修复
后使用新 work directory 重跑。真实但不提升的结果按研究账本规则记录并回退该
实验代码，不能删除失败证据。

## 10. 交付物与终止条件

每阶段保存配置、manifest、计划、日志、指标、审计和中文结果报告，并登记到
`.lab/results.tsv`、`.lab/log.md` 与 `.lab/branches.md`。

完成必须同时有权威证据证明：

1. 从允许的通用 checkpoint 开始，DOTA optimizer/EMA/schedule 从零；
2. 没有教师、蒸馏或伪标签；
3. 开放词汇 prompt 接口和 novel holdout 通过；
4. E2E 固定集合、无额外 dense/RoI head；
5. 无 NMS/top-k；
6. 输出五维旋转框；
7. 先完成小数据筛选，再完成 GPU4–7 四卡全量训练；
8. raw-13,833 `AP50>=0.7000` 且 `mAP>=0.7000`。

未满足第8项时，项目仍处于进行中，不得用 filtered-6605、历史 checkpoint 或
局部 subset AP 宣称完成。
