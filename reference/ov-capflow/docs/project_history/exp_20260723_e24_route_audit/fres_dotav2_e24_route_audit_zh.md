# DOTA-v2 E24 严格 Q600 路线审计（工作草案）

## 阶段性判定

本档案记录 E24 raw 13,833 结果之后的只读根因审计。当前结论是：

- **A1 已淘汰**，不得进入 Stage 2，也不得与新候选叠加救援；
- **B1 当前 balanced-token loss 直接淘汰**，因为它把主分类损失缩小约
  `64.1x`，并非同尺度替换；
- **G2 统一常开 reference-size floor 暂不进入训练**：25 图观察只支持“风险
  较高、证据不足”，尚无干预性实验可证明 zero-gated 版本必然失败；
- **N1 仅在 25 图、group-0、train-mode surrogate 上未触发 TP 污染否决条件**，
  可作为冻结 E24 parent 的 surplus-query ownership probe 候选，但不是已批准
  实验，也不能单独承担 ICLR 核心新颖性；
- **G3 只支持把 fixed-Q center-location generation/transport 作为下一设计问题**：
  现成 final center 的 any-label 覆盖仅 22.256%，纯重排已不足；但目前没有 AP、
  可学习性或具体信号来源证据；
- raw 目标尚未达成：E24 `mAP=0.606405`、官方 `AP50=0.606`，距离
  `0.7000` 仍差 `0.093595`。

在用户对 N1 或 G3/A 的具体设计作出明确批准前，不创建 candidate config、
不修改生产代码、不训练，也不把本档案当作实验 spec。

## Canonical E24 身份与完整性

| 字段 | 冻结值 |
|---|---|
| Checkpoint | `work_dirs/dotav2_cleanstart/full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/epoch_24.pth` |
| Raw dump | `work_dirs/dotav2_cleanstart/eval_t7_epoch24_raw13833_gpu2389_dump/predictions.pkl` |
| Records / unique IDs | 13,833 / 13,833 |
| GT / prediction rows | 243,632 / 8,299,800 |
| Output contract | 每图恰好 Q600；无 NMS、top-k、dense extra outputs |
| Classes | 18 |
| CPU finite | 全部通过 |
| Same-dump parity error | `2.856055891786724e-08` |
| Training replay delta | `0.0` |

## Raw mAP 0.7000 目标缺口与主瓶颈

| 指标 | E24 raw |
|---|---:|
| mAP | 0.606405 |
| AP50 | 0.606 |
| Oracle mAP | 0.782828 |
| Oracle headroom | 0.176423 |
| 达到 mAP 0.7000 需兑现的 headroom 比例 | 53.05% |
| TP score--assigned-IoU Spearman | 0.365425 |
| base14 | 0.654018 |
| novel4 | 0.439762 |

243,632 个 GT 的状态为：geometry miss `76,741`（31.499%）、semantic miss
`2,440`、ownership miss `11`、evaluator reachable `164,440`（67.495%）。因此
E24 不是单一排序问题：排序存在足够 oracle 余量，但约三分之一 GT 在 Q600
原始几何中没有同类 IoU>=0.5 witness。

AP-support FP 共 161,714：localization/background 占 64.634%，empty tile 占
17.623%，duplicate 占 15.302%，semantic 占 2.441%。这说明下一机制至少要
直接影响局部/背景查询的可分性；只修 positive quality target 的 A1 已被终点
结果否定。

### 小目标与极密度

- `<=8 px` GT 38,575：geometry miss 76.366%，reachable 23.119%；
- `>600 GT/image` GT 35,739：geometry miss 82.912%，reachable 16.718%；
- small-vehicle GT 150,145：geometry miss 43.736%，reachable 55.318%，
  AP 0.319549，oracle headroom 0.225906；
- 固定 Q600 的纯容量超额仅 15,939 GT（6.542%），乐观容量 ceiling 为
  93.458%。因此 31.499% 的 geometry miss 不能被“查询数不够”单独解释。

## Dense400 确定性分层复核

Dense400 由全部 `>600` 的 33 图、全部 `401–600` 的 60 图、全部
`201–400` 的 157 图，以及 `101–200`、`51–100` 各按
`sha256("20260723:" + img_id)` 最小值选 75 图组成。已归档 400-ID manifest 的
selection-content SHA256 为
`f1d705abdc38d66c2d6a6ba14d318a850039b518e8230470fa07b2beffd8fc4d`，文件
SHA256 为 `cf187eedba4f19703a77475639887147e4e1285b28f7db4c91686c0296030594`。

以下 Dense400 mAP/reachability 是当次会话汇总，尚未落盘逐图 evaluator
diagnostics，**未独立冻结**。该口径含 123,888 GT；E24 mAP 为 0.532542，
reachable 0.501275，oracle mAP 0.696970，headroom 0.164428。

| GT/image | Reachable | Geometry miss |
|---|---:|---:|
| 51–100 | 84.996% | 13.667% |
| 101–200 | 79.300% | 19.562% |
| 201–400 | 66.169% | 33.113% |
| 401–600 | 50.012% | 49.709% |
| >600 | 16.718% | 82.912% |

| 尺寸 | Reachable | Geometry miss |
|---|---:|---:|
| <=8 px | 19.811% | 79.920% |
| 8–16 px | 42.417% | 56.873% |
| 16–32 px | 81.477% | 17.892% |

密度和尺度两个边际分桶都与 geometry miss 恶化相关，但二者可能混杂；尚未通过
联合分层或回归证明其独立性。这些结果不支持用单一全局尺度 floor 解决所有
密度层。

### G3 工作线索：final-box 几何反事实

在同一 Dense400 上，对 canonical E24 的 61,068 个 geometry-miss GT 做
GT-assisted final-box oracle：分别把预测框的 center `C`、extent `(w,h)` `E`
或 angle `A` 替换为该 GT 分量，并重算 rotated IoU。`E` 同时混合面积尺度与
长宽比，不能简称纯 scale。

结果同时报告两种候选口径：

- all-query same-label：每种干预都允许在固定预测标签正确的查询中重新选最大
  IoU，是局部存在性上界；
- fixed-original-same-query：先按原始框固定最佳同类 query，所有干预始终使用
  同一 query，更接近分量归因。

| 反事实 | All-query same-label rescue | Fixed original same-query rescue |
|---|---:|---:|
| Center `C` | 96.044% | 45.122% |
| Extent `E` | 2.658% | 2.407% |
| Angle `A` | 0.745% | 0.665% |
| Center+Extent `CE` | 99.150% | 51.937% |
| Center+Angle `CA` | 60.159% | 33.631% |
| Extent+Angle `EA` | 5.605% | 5.096% |
| Full `CEA` | 99.396% | 99.396% |

在 fixed-query 的最小修复集中，single-center 26,891，single-extent 934，
single-angle 166，CE-only 13,633，CA-only 3,930，EA-only 805，triple-only
13,208；另有 ambiguous-multiple-singles 680、ambiguous-multiple-pairs 452，
以及 369 个 GT 没有任何同类预测 query。上述类别完整覆盖 61,068 个
geometry miss。center 的线索最强，但在
`>600 GT/image` 与 `<=8 px` 分层中，fixed-query center rescue 仅 39.656% 与
34.511%，center+extent 则为 54.370% 与 50.062%，说明极密 tiny 场景更可能
需要 center 与 extent 的交互，而非统一尺寸 floor。

该结果精确复现 Dense400 state sanity：geometry miss 61,068、semantic miss
717、reachable/ownership 合计 62,103。证据文件 SHA256 为
`0d2b885e1985dd8db59c78a072b7632170eeb37b7fde1463b19f3f3e04fef776`。

它仍不能证明 center 是模型内部因果根因：per-GT 最大值会让一个 query 同时
“救回”多个 GT，尚未做候选--GT 二分匹配、角度等价表示枚举、shuffled-GT
placebo、图像级 bootstrap、长宽比分层或 IoU=0.75 复核。因此当前只把 G3
研究优先级转向 **query-center allocation/refinement + conditional extent
coupling**；不据此创建实现或声称 AP 收益。

这也不重开已经失败的 geometry-matching cost：历史同口径 proxy 中，control
AP50 为 0.381，Hausdorff-DN 为 0.341，four-corner Chamfer 为 0.375；更低的
局部 geometry cost/duplicate 没有转化为 AP。G3 若继续设计，必须作用于 center
的表示、分配或跨层更新机制，而不能只是更换 Hungarian cost 或复用
Hausdorff/Chamfer loss。

### Center query-switch 与一对一匹配控制

为排除同一 query 被多个 GT 重复使用，进一步在每图构造 GT--Q600 同类边图，
对 baseline、geometry-miss GT 的 center-oracle `C`、center+extent-oracle `CE`
分别求最大二分匹配。非 geometry-miss GT 始终保留 baseline 边。

| 口径 | 同时可匹配 GT | 123,888-GT recall | 相对 baseline |
|---|---:|---:|---:|
| Baseline | 62,102 | 50.128% | — |
| Center oracle | 99,636 | 80.424% | +37,534 GT |
| Center+extent oracle | 100,001 | 80.719% | +37,899 GT |

局部 rescue transition 为：`C=1,CE=1` 58,569，`C=1,CE=0` 83，
`C=0,CE=1` 1,980，二者均失败 436，说明 CE 并非严格单调但净增益为正。
对 geometry-miss GT，原最高分同类 query 的 C/CE rescue 为 45.279%/47.357%；
发生 winner switch 且被局部获救的 GT 占全部 geometry-miss GT 的
95.331%/98.158%；若以 C/CE 局部获救集合为分母，条件切换率分别为
99.258%/98.999%。这更支持先验证 **query--GT center allocation/association**，
再验证 fixed-query center refinement；extent 只作为 center 条件分支。

极密度层仍有明确边界：`>600 GT/image` 的 baseline/C/CE matching recall 为
16.718%/43.275%/43.001%，CE 甚至略低于 C，且均受每图 Q600 容量约束。

该一对一 matching artifact 仍是“每条边按其 GT 单独替换分量”的 oracle，
不是一组可同时部署的预测框；它约束 query 独占，却不提供 score/AP、候选数
随机子采样、placebo、
bootstrap 或 Hungarian-assigned-query 证据。机器证据 SHA256 为
`3efcc1d20e07bc7cde285629461ba915bccb525ee044a56ee4e57d07e0433435`。

### Center 候选数量与 shuffled-GT 负控

随后用 CPU 重读 canonical dump，并在 5 个运行前固定 seed 上只复算 center-oracle
`C`。脚本先精确复现 61,068 个 geometry miss、58,652 个 full local rescue、
baseline matching 62,102 和 C matching 99,636，任一锚点不一致即禁止发布。

以下 local 控制对每个 GT 最多随机保留 `m` 个同类候选；若同类候选不足则使用
全集。比例均为 5-seed rescued mean / 全部 61,068 个 geometry miss，369 个
无同类候选 GT 计作失败。

| 每 GT 最多保留同类候选数 | 5-seed mean C rescue / 61,068 |
|---:|---:|
| 1 | 37.369% |
| 5 | 72.225% |
| 10 | 81.791% |
| 25 | 89.290% |
| 50 | 92.440% |
| 100 | 94.194% |
| all | 96.044% |

matching 控制在所有图上共享同一组固定 query ID，计数分母为全体 123,888 GT；
`gain retention=(mean C-mean baseline)/(99,636-62,102)`。

| 跨图固定保留 query ID | mean baseline | mean C | C 增量 | gain retention |
|---:|---:|---:|---:|---:|
| 100 | 14,977.2 | 25,016.8 | +10,039.6 | 26.748% |
| 200 | 27,497.0 | 46,539.4 | +19,042.4 | 50.734% |
| 400 | 47,184.6 | 78,549.4 | +31,364.8 | 83.564% |
| 600 | 62,102.0 | 99,636.0 | +37,534.0 | 100.000% |

same-image/same-class shuffled-GT center placebo 覆盖 61,017/61,068
geometry miss（99.916%）。eligible 分母上的 true-C local rescue 为
58,641/61,017（96.106%）；仅对该 eligible 子集开放 true-center 边时，全图
matching count 为 99,625/123,888。5 个错配中心 seed 的 local rescue 均为
0，全图 matching 均退回 baseline 62,102。这说明 center oracle 具有强 target
specificity；同时，`m=1` 到 all 的明显增幅证明 96.044% 仍受候选数量放大，
不能把 all-query oracle 写成 fixed-query refinement 证据。

该控制仍没有 score/AP、学习过程或可部署框。零 placebo 是较容易的负控，不能
单独建立因果性或可学习性。它支持继续研究 query--GT center allocation/
association；fixed-query refinement 仍只能由原 fixed-C 45.122% 单独支持。
机器证据 `evidence/dense400_center_controls.json` 的 SHA256 为
`d2374fe60bcc330c4303017acb24113b4bc510a4d236990dd781e8ae9e3bfede`。

### 现有 final center 可用性控制

前述 `C` oracle 会把 query center 直接替换成 GT center，因此它并不回答一个
关键问题：漏检 GT 附近是否已经存在可以被重新分配的 final query center。为此，
在同一 Dense400、同一 61,068 个 geometry miss 上，将每个预测中心变换到
GT 的旋转局部坐标，并以
`max(2|dx_local|/GT_w, 2|dy_local|/GT_h) <= 1` 定义中心落入 GT。

| 口径 | 可覆盖 / 可一对一匹配 | Geometry-miss 占比 |
|---|---:|---:|
| Any-label center inside | 13,591 | 22.256% |
| Same-label center inside | 11,802 | 19.326% |
| Any-label 一对一匹配 | 13,391 | 21.928% |
| Same-label 一对一匹配 | 11,633 | 19.049% |

按 manifest 的 GT/image 分层，final-center availability 总体随密度增加而恶化，
但前两个低密度层并非严格单调：

| GT/image | Geometry miss | Any-label available | Same-label available |
|---|---:|---:|---:|
| 51–100 | 767 | 39.765% | 32.464% |
| 101–200 | 2,045 | 40.098% | 35.501% |
| 201–400 | 14,381 | 30.415% | 27.703% |
| 401–600 | 14,243 | 24.286% | 23.029% |
| >600 | 29,632 | 15.635% | 12.024% |

该表只是描述性分层：密度与尺度仍可能混杂，`>600` 层还受 Q600 固定容量约束，
不能据此把密度解释成独立因果因素。

最小 normalized Chebyshev 距离的 any-label P25/P50/P90 为
1.105/2.076/7.327，same-label 为 1.247/2.296/11.220；same-label 有限候选
覆盖 60,699 个 GT，另有 369 个 GT 没有同类 query。availability 到一对一
matching 只损失 any-label 200、same-label 169，说明在该口径下，**现有近中心
query 之间的独占冲突不是主要缺口**。更关键的是，约 77.744% 的 geometry miss
连任意标签的 final center 都未进入 GT。

这修正了方案边界：`C` oracle 的强结果不能被解释成“只需重排现成 final
centers”。若继续 fixed-Q center allocation，机制必须实际移动、产生或运输新的
center location；只在 600 个现成 final centers 间置换，最多只能解释这里的低
覆盖子集。新中心信号应来自 decoder localization state / sampling trajectory，
还是 encoder spatial features，仍是待比较假设。该控制仍不是 IoU reachability、
AP 或可学习性证据，也不证明某一 proposal/transport 机制有效。机器证据
`evidence/dense400_center_availability.json` 的 SHA256 为
`72967b785162e3d66c5eda48a89dbb9654c9e697c8e35978736dc9645c2e20b3`。

截至 2026-07-23，通用 query 路线的新颖性风险已经很高：
[DQ-DETR](https://arxiv.org/abs/2404.03507) 使用计数/密度图动态调整 query 数和
位置，[Dome-DETR](https://arxiv.org/abs/2505.05741) 使用 density-oriented
adaptive query initialization，[PaQ-DETR](https://arxiv.org/abs/2603.06917)
使用 image-specific dynamic queries；开放词汇侧的
[OV-DINO](https://arxiv.org/abs/2407.07844) 已有 language-aware query
selection，[RT-OVAD（早期版本名 OVA-DETR）](https://arxiv.org/abs/2408.12246)
已有 text-guided aerial
decoder。因此“density/text-guided query”本身不足以构成 ICLR 核心；若继续
G3，差异必须落在固定 Q600、无 hard top-k/NMS、全量 soft allocation/
ownership 以及定位训练不破坏 novel semantics 的联合约束上。

## P0148 / P0682 案例边界

- `P0148__1024__651___0` 的公开 annotation 为空。现有 OpenRSD pickle 与
  P0682 的内容逐字节相同，视觉来源只符合 P0682；因此 P0148 只能记为
  **未标注语义集中案例**，不能声称其 small-vehicle 输出是假阳性。
- P0148 的 600 行中 497 行预测 small-vehicle，均值分数 0.488795，最大值
  0.832991；这是集中现象，不是带 GT 的误检定论。
- P0682 恢复出 20 个可验证 GT，其中 10 个 small-vehicle。E24 的 600/600
  输出均为 small-vehicle，均值分数 0.885964、最大值 0.929730，但 10/10
  small-vehicle 全部为 geometry miss，600 行均为 localization/background。
- 离线压制 small-vehicle 类别会消除两图的该类输出，却不能修复 P0682 的
  10 个 geometry miss。因此语义抑制不是几何恢复机制。

## 候选 G2：pre-attention 尺度观察后暂不训练

工作诊断从五个密度层各选 5 图并读取真实六层 decoder reference。由于逐查询
明细、统计单位、条件分母和 reference-size 公式没有冻结，本草案删除原精确
百分比表，只保留治理结论：中高密度层曾出现局部尺度相关信号，但 `>600`
极密度层没有得到同方向支持。该观察不足以为统一常开 floor 提供训练依据；
zero-gated 或
条件化版本是否无害仍未知。治理结论只是“G2 暂不训练”，不是因果淘汰。

## 候选 B1：真实 batch 数值审计后淘汰

仓库中的 `balanced_cfg` helper 当前未启用。其单测通过不等于科学尺度正确。
本节精确数值来自当次会话的真实-batch 工作诊断，尚无独立日志资产，证据等级
低于 canonical E24/A1；在补齐可复放脚本和输出前不得写成冻结结论。
在一个真实 batch（6 matched / 594 unmatched queries；384 / 38,016 有效
token）上，现有 token-group mean 使主 `loss_cls` 相对 parent 只剩
`0.0155988`，约缩小 `64.1x`；bbox、IoU 与 DN loss 不变。

尝试只做内存中的 query-count normalization 也没有得到可靠同尺度替换：普通
初始化 batch 可恢复到 parent 的 99.93–99.94%，但 E12 稀疏单 GT 样本六层
比值为 `0.230/0.670/0.546/0.098/0.625/0.619`；empty 和 1,223-GT 极密度
样本还因 padded logits 非有限而产生 NaN。故当前 B1 直接 FAIL；任何修订都将
是新设计，不能借用现有 helper 直接开跑。

## 候选 N1：Hungarian false-null 预检

### 预检方法

从与 G2 相同的 25 张分层图运行 E24 的 train-mode loss surrogate：保留
3x600 grouped matching、DN 和最终 group-0 Hungarian mask，并从同一 forward
捕获最终 group-0 分类/框输出。随后按 raw mouth 的一查询一类别、无 NMS/top-k
规则，在 IoU=0.5 下将 600 个查询分为 TP、duplicate、semantic、
localization/background 或 empty。

这是只读诊断；没有 optimizer step、反向传播、checkpoint 或仓库源代码改动。
固定主 seed 为 `20260723`，每图使用 `20260723 + selection_ordinal`；模型处于
`train()`，所以 Swin `drop_path_rate=0.2` 仍激活。当前只做了一次固定-seed
运行，不能视为 canonical eval-mode raw dump 的确定性重放或稳定性估计。完整
样本 ID、状态定义和聚合量归档于 `evidence/false_null_25.json`。

### 结果

| 分层 | TP | TP 中 unmatched | TP false-null rate | 全部 unmatched |
|---|---:|---:|---:|---:|
| 1–10 | 20 | 3 | 15.000% | 2,979 |
| 201–400 | 1,084 | 28 | 2.583% | 1,544 |
| 401–600 | 1,346 | 10 | 0.743% | 637 |
| >600 | 773 | 0 | 0.000% | 0 |
| empty | 0 | 0 | — | 3,000 |
| **合计** | **3,223** | **41** | **1.272%** | **8,160** |

全部 15,000 queries 中 matched 6,840、unmatched 8,160。unmatched 组成是：
TP 41、duplicate 1,089、semantic 252、localization/background 3,778、empty
3,000。也就是说：

- 98.728% 的最终 TP 已被 Hungarian matched 覆盖；
- 误落在 unmatched 的 TP 只占全部 unmatched 的 0.502%；
- 但 unmatched 中仍有 1,130 个 same-label support（TP+duplicate）以及
  1,382 个 any-object-IoU>=0.5 查询；压低它们主要会去重和压错类，但可能减少
  某些 GT 的替代 witness；
- `>600 GT/image` 中每图 600 个 query 全部 matched，N1 在这一层没有
  matched/unmatched 对比监督，不能解决固定容量和极密度几何覆盖。

matched 也不是 detection-quality 正标签：6,840 个 matched 中只有 3,182 TP
（46.52%），另有 3,058 localization/background（44.71%）、566 duplicate
（8.27%）和 34 semantic（0.50%）。所以 N1 学到的是 Hungarian ownership，
不是 objectness、IoU quality 或几何正确性。

因此该 surrogate 的 false-null 污染没有否定 N1，但只足以让它进入**待批准的
冻结 parent surplus-query ownership probe**，不构成性能提升证据。

### 25 图 ownership-target oracle

在同一固定-seed forward 上保持 Q600、框、类别和行顺序不变，仅把 unmatched
行分数置为 `-1`，按 DOTA IoU=0.5、VOC07 11-point AP 重算：该 25 图、9 个有
GT 类别的 mean AP 从 0.591378 增至 0.619159，增加 0.027781；plane 反而下降
0.013986。完整逐类结果见 `evidence/ownership_oracle_25.json`。

这只是密度偏置样本上的 perfect-target 排序诊断：它说明 ownership target 有
非零排序价值，也说明收益远小于 raw mAP 0.7000 所需的 0.093595。由于样本、
mode 与类别支持不同，它既不是 raw 上界，也不能数学关闭或批准 N1。

## N1 待批准概念边界

N1 的最小概念是：冻结 E24 parent，用一个共享的 257 参数线性头为每个最终
matching query 输出一个零初始化标量 ownership/null residual；监督来自
detached final Hungarian mask，按每个 600-query group 独立计算。推理只把以
零点中心化的 log-non-null residual 加到已选类别的 log score；标签、框、Q600
行数和查询顺序不变。

为保护因果归因，若后续获批，必须满足：

1. parent 参数、分类与框输出完全冻结，新增参数只允许一个线性层的 257 个参数；
2. 三个 matching groups 分别监督，不能错误复用第一组 mask 或把 1,800 queries
   当作一个 reservoir；
3. 零初始化时必须 bit-exact 复现 parent 分数；
4. empty / all-matched 分组要用 finite、branchless reduction；
5. 不得启用仓库旧 null loss、capacity、semantic gate 或 B1；
6. 该候选只能主张“strict raw mouth 的 parent-preserving null calibration”，不能
   主张解决 `>600 GT/image` 的 geometry/capacity 瓶颈。

文献风险很高：[PROB](https://openaccess.thecvf.com/content/CVPR2023/html/Zohar_PROB_Probabilistic_Objectness_for_Open_World_Object_Detection_CVPR_2023_paper.html)
已从 query embedding 建模 probabilistic objectness；
[Cascade-DETR](https://openaccess.thecvf.com/content/ICCV2023/html/Ye_Cascade-DETR_Delving_into_High-Quality_Universal_Object_Detection_ICCV_2023_paper.html)
已用 expected IoU 校准 query confidence；
[Decoupled PROB](https://openaccess.thecvf.com/content/WACV2025/html/Inoue_Decoupled_PROB_Decoupled_Query_Initialization_Tasks_and_Objectness-Class_Learning_for_WACV_2025_paper.html)
专门处理 objectness/class 学习冲突；
[OWOBJ](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Open-World_Objectness_Modeling_Unifies_Novel_Object_Detection_CVPR_2025_paper.html)
进一步覆盖 open-world objectness。加上 DETR no-object 与其他 ranking 工作，N1
若有效也更适合作为 broad geometry 方法中的校准组件，而非 ICLR 核心贡献。

## 决策门与下一步

N1 当前状态为 **CONDITIONAL PASS / awaiting explicit approval**。以下只是尚未
生效的预注册提案，不能由本档案自行授予实现、训练或 raw 评估权限。建议只做
一次冻结-parent、低成本 proxy，不替代 broad geometry 主路线：

- Stage 0：零初始化 bit-exact、分组 mask、finite empty/all-matched、参数冻结和
  Q600 输出合同测试全部通过；
- Stage 1：固定 400 图、固定 E24 parent、单 seed；AP50 至少 `+0.010`，novel4
  不回退，base14 不回退超过 `0.005`；
- 机制门：AP-support localization/background 与 empty FP 同时下降，micro
  recall 不低于 parent `0.005`，oracle headroom 不扩大；
- 任一门失败即淘汰，不延长、不换 seed、不调阈值、不与 A1/B1/G2 堆叠；
- proxy 全门通过只构成“建议申请 raw 13,833 评估”的条件；是否运行仍须符合
  用户批准的完整协议或另行取得明确授权。

由于 N1 无法改善 `>600 GT/image` 的几何覆盖，即使 proxy 通过，仍需要另立且
重新批准的 broad tiny/dense geometry 机制；当前没有足够证据为其冻结具体实现。

## 关键资产

- E24 canonical diagnostics：
  `work_dirs/dotav2_cleanstart/eval_t7_epoch24_raw13833_gpu2389_dump/diagnostics/`
- A1 最终档案：
  `docs/project_history/exp_20260723_dotav2_a1_position_supervision/fres_dotav2_a1_position_supervision_zh.md`
- A1 endpoint commit：`2616ab1`
- False-null 25 图工作证据：`evidence/false_null_25.json`
  - SHA256 `8647bb2793b8f3a128151bffa8c207f310fc738f95c437343767addeab40a5ed`
- Ownership-target oracle 工作证据：`evidence/ownership_oracle_25.json`
  - SHA256 `bf080364bac4fc16d5a91b87cb8ecaec470e458cc3e7b69f5462e9a399d4a08c`
- Dense400 selection manifest：`evidence/dense400_manifest.json`
  - SHA256 `cf187eedba4f19703a77475639887147e4e1285b28f7db4c91686c0296030594`
- Dense400 final-box geometry counterfactual：
  `evidence/dense400_geometry_counterfactual.json`
  - SHA256 `0d2b885e1985dd8db59c78a072b7632170eeb37b7fde1463b19f3f3e04fef776`
- Dense400 center matching control：
  `evidence/dense400_center_matching_control.json`
  - SHA256 `3efcc1d20e07bc7cde285629461ba915bccb525ee044a56ee4e57d07e0433435`
- Dense400 center candidate-count / shuffled-GT controls：
  `evidence/dense400_center_controls.json`
  - SHA256 `d2374fe60bcc330c4303017acb24113b4bc510a4d236990dd781e8ae9e3bfede`
  - 复算脚本：`projects/OVCapFlow/tools/analyze_dense400_center_controls.py`
    - SHA256 `07ecd6cdce83c012987883f5ce6f7b9ea4c9d60de30253bc8fdd1139454866b6`
  - 专项测试：`tests/test_projects/ov_capflow/test_dense400_center_controls.py`
    - SHA256 `fd2d09f6f8fafcf4d57447bee320599f73d161eebebf2dd0154a9da501f85c6c`
- Dense400 final-center availability：
  `evidence/dense400_center_availability.json`
  - SHA256 `72967b785162e3d66c5eda48a89dbb9654c9e697c8e35978736dc9645c2e20b3`
  - 复算脚本：`projects/OVCapFlow/tools/analyze_dense400_center_availability.py`
    - SHA256 `97ee8f46bfdd1680de070c10f44ab43283fdb92df0d44a8dd685bc14ec00b123`
  - 专项测试：`tests/test_projects/ov_capflow/test_dense400_center_availability.py`
    - SHA256 `ebfbb322676d462766da7f69b01745f1378b07630f30a5fb6e00b58f6a69028a`

## 证据等级与未补档项

| 证据 | 等级 | 状态 |
|---|---|---|
| E24 raw diagnostics | frozen | 有 canonical JSON/report、完整 dump 与 parity |
| A1 Stage-1 | frozen | 有终档、checkpoint/dump hash 与提交 `2616ab1` |
| N1 false-null / ownership oracle | session evidence | 已归档样本、seed、mode、定义与聚合；仅一次 train-mode surrogate |
| Dense400 | working+artifacts | manifest、counterfactual、一对一 matching、5-seed candidate/placebo、final-center availability 已落盘；baseline 逐图 mAP 与置信区间仍缺 |
| G2 pre-attention | working | 缺逐查询明细、统计单位与 reference-size 公式冻结 |
| B1 real-batch | working | 缺可复放脚本、样本 ID 和输出日志 |
| P0148/P0682 | working | GT 来源与案例边界已说明，但聚合输出尚未落盘 |

在 working 项补齐资产和独立复核前，本文件保持“工作草案”，不作为冻结终档。

## 结论边界

本审计冻结的是 E24/A1 已有资产，并记录后续工作诊断；它不证明 G2 因果失败，
也不是 N1 有效性结果。所有数值都来自单一 E24 checkpoint；25 图机制样本是
固定 seed、train-mode 的分层 surrogate，不是完整验证集统计。不得把 N1 写成
已批准、已实现或已提升 AP，也不得把 E24 0.606 描述为 mAP 0.7000 目标已完成。
