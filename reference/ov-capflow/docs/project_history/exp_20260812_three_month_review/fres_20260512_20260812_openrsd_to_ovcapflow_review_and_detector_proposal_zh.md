# OpenRSD → OV-CapFlow 三个月实验复盘与弱监督旋转检测副产品方案

> 时间窗：2026-05-12 至 2026-08-12  
> 状态：滚动复盘快照；现有主线实验仍在推进  
> 范围：只读证据审计与研究设计，不代表已批准实现或已获得新方法结果

## 0. 一页结论

这三个月并不是“不断试模块但没有稳定涨点”，而是一条相当清晰的研究认知迁移链：

1. **OpenRSD 前期解决的是语义吸引子和后处理放大问题。** 你证明了 `small_vehicle` 偏置在 dense classification logits 中已经存在，NMS/top-k 主要负责放大，而不是凭空制造它；但凡能强力消除该现象的视觉支持干预，往往同时毁掉整体开放词汇检测能力。
2. **OpenRSD 中后期把问题从“语义修补”推进到“部署口径和端到端结构”。** 文本、旋转、支持集、后处理等大量局部修复没有稳定越过 matched baseline；相反，HRRSD 上严格无 dense head、无 NMS 的 P15O-B 说明真正可迁移的副产品是“一对一集合预测”而非某个语义去偏插件。
3. **OV-CapFlow 将主矛盾明确为 fixed-query 系统中的 query birth、ownership、null calibration、capacity 与 supervision contract。** 旧的无头路线长期停在约 0.40–0.43；T7 在完整 raw 13,833 图、Q=600、5D、无 NMS/top-k 口径到达 0.606405，但这是 all-18 closed-set 结果，不是严格开放词汇结论。
4. **7 月底至 8 月最有价值的进步是学会“主动否证”。** D11、D13-N、QAF、ODQ、RISC、POQ 等候选被 matched control、placebo、完整端点或协议审计及时关闭；OMQ 又证明“source signal 存在”不等于“可以被训练利用”，更不等于“协议无泄漏”。
5. **最值得独立成篇的副产品方向是 HBB 监督旋转框，而不是直接做全监督或点监督。** 全监督最适合作为 oracle 和工程底座，论文新意弱；点监督几何欠定且 2024–2026 已被 PointOBB、Point2RBox 系列和 PWOOD 强覆盖；HBB 保留中心、尺度与实例身份，最适合把你已经积累的 fixed-query ownership、negative-safe supervision 和 rotation-orbit 经验组合成新框架。

建议的候选方法暂名 **HALO-Set：Horizontal Annotation Latent-Orbit Set Prediction**。它不从 HBB 提前生成唯一伪旋转框，而把 HBB 对应的旋转框视为一个潜在可行集合；每个 matched query 预测多模态方向分布，在旋转/翻转轨道上保持同一实例的 ownership，并对尚未消歧的候选避免过早施加 false-background 梯度。该方案目前是**有根据、可证伪、但尚未验证的新假设**，不能宣称首创或已经有效。

## 1. 证据口径

### 1.1 本报告的证据等级

| Label | 含义 | 可用于什么结论 |
|---|---|---|
| `VERIFIED_ENDPOINT` | 有冻结配置、完整端点、明确评测口径 | 可报告绝对值和同口径差值 |
| `MECHANISM_DIAGNOSTIC` | 干预/探针揭示机制，但不一定可部署 | 可说明因果链或候选瓶颈 |
| `CONFOUNDED` | batch、BN、数据、评测 mouth、父模型或泄漏不匹配 | 只能用于方法学警示 |
| `NEGATIVE_RESULT` | 同口径、到冻结端点，候选未过门 | 可关闭该具体实现，不等于关闭整个问题 |
| `PENDING` | 仅设计、smoke、短跑或缺少 matched control | 不作 AP/创新结论 |

### 1.2 不能混排的评测 mouth

OV-CapFlow 至少存在三种长期并行口径：

- DOTA-v2 内部 `DOTA1000` 代理；
- 完整 raw 13,833 图，`filter_empty_gt=False`；
- paper 6,605 非空图过滤口径。

它们分别回答筛选速度、部署全分布、论文非空图比较问题。约 0.70 的 G17/G19/G27C 结果属于过滤或多源/router 体系，不能与严格 raw 单 checkpoint、all-query 结果直接混排。今后任何表格必须把 dataset split、empty policy、query mouth、后处理、checkpoint 和类别监督范围写进主表，而不是只写脚注。

## 2. 阶段复盘

## 2.1 2026-05-12—05-26：从 false-SV 现象到语义吸引子因果链

### 已经做实的部分

- P0148 暴露出极强的 `small_vehicle` 聚集，但它应被定位为**诊断压力样本**，不能作为最终指标。
- 偏置在 dense classification logits 已经出现，NMS/top-k 会进一步集中输出。因此“只改后处理”最多缓解可见症状，不能清除源头。
- 文本编码器替换几乎不改变 P0148 的最终 SV 比例，约为 `0.751–0.753`；说明该吸引子不是简单的文本编码器单点故障。
- 替换视觉支持能把 SV 比例压到约 `0.06–0.09`，但 held-out mAP50 同时从 `0.674` 跌到约 `0.092/0.096`。这是很强的机制证据：错误语义与有效识别共享同一视觉支持通路，不能靠粗暴删除来部署修复。
- DeHub/B8k 早期训练修复把 held-out-500 AP50 从 `0.738` 提至 `0.805`，SV AP50 从 `0.741` 提至 `0.842`，P0148 dense SV 接近 `0.005`；但低风险图检测数增加约 `61%`，触发 `CLASS_DRIFT_RISK`。它说明局部指标改善可能由全局决策边界漂移换来。
- 后处理修复在 n=40 上最好约 `-0.009`，n=60 约持平 `0.618`，没有稳健 AP 收益。

### 这一阶段真正留下的经验

1. **异常图不是主指标，而是因果显微镜。** P0148 应始终和真实 SV-rich 样本、安全集、完整 held-out 指标绑定。
2. **症状消失不等于系统变好。** 需要同时检查目标类别召回、其他类别漂移、空图 FP 和完整 AP。
3. **模块替换必须问“信号从哪里来”。** 文本不敏感、视觉支持极敏感，促使研究从 prompt engineering 转向视觉证据与决策路径。
4. **dense 输出的错误最终会被排序和后处理放大。** 这为后续从 dense/NMS 转向一对一集合预测埋下了第一条逻辑线。

## 2.2 2026-05-27—06-22：局部修复大筛选与实验控制成熟

### 主要结果

- 一批 text/rotation 短跑集中在 `0.646–0.650`，低于约 `0.6962` 的对应基线；SV-only adapter 约 `0.487`，且 SV AP 约 `0.0222`。小模块、有界初始化或“只动语义侧”并不天然保持原模型 alignment。
- BN 控制揭示了严重 no-op 混杂：eval-only C0 `0.6592`，lr=0 no-op `0.6448`，lr=0 + frozen BN `0.6597`。若没有 frozen-BN control，约 1.5 point 的变化会被错归因给候选方法。
- S3C/GS3C 对若干内部指标有微小改善，但“更低 SISE”不等价于“更高 AP”；机制代理必须经过 endpoint 校准。
- P12B 在 HRRSD 到 `0.8745`，但 4-GPU/global-batch 改变造成混杂；不能作为方法因果值。
- SGP 约 `0.8307`，低于基线。

### 方法学升级

- 形成 matched no-op control、父 checkpoint 对齐、batch/BN/worker 对齐的意识；
- 开始把 source existence、intervention sensitivity、endpoint utility 分成不同门；
- 开始将失败结果正式归档，而不是围绕 best checkpoint 继续救援；
- 工程故障被纳入科学有效性：GPU worker 全落 GPU0、pickle/daemon/DataLoader 失败、adapter checkpoint 未注入、CSV 覆盖、GPU 映射错误、缺少 true-SV GT，都会让“实验跑完”失去结论资格。

## 2.3 2026-06-23—06-26：严格端到端集合预测成为转折点

### 从失败中定位结构问题

P13/P14 的多种 fusion decoder、Gaussian prior、query transport、teacher anchor、assigned lattice 等实现出现零或近零结果。它们没有证明“DETR 路线不行”，只证明当时的 query birth、特征对齐、优化尺度或 assignment 设计不成立。

### 两条最关键证据

1. **HRRSD P15O-B**：严格无 dense head、无 NMS，mAP `0.8865`，高于 OpenRSD `0.8562`；GPU latency 约 `15.42 ms/img`，优于 `19.82 ms/img`。这是 OpenRSD 向 OV-CapFlow 转向的最强概念桥：直接集合预测不是仅为“形式纯洁”，而是可能同时带来准确率和效率。
2. **DOTA-v2 P22 数据规模对照**：500 图、80E 仅 `0.1394`；1000 图、80E 达 `0.6008`，高于同 subset OpenRSD `0.5641`。小代理或短优化失败不能轻率杀死结构路线，最小数据规模本身是一个 gate。

### 转折点的含义

到这里，研究对象已经从“如何修补 OpenRSD 的某个错误类别”变为：能否让一个固定数量 query 的旋转框系统，在不依赖 dense head/NMS 的情况下完成对象出生、归属、存活、空集拒绝与排序。

## 2.4 2026-07-01—07-14：OV-CapFlow 建立，但旧瓶颈被系统性暴露

### 结果边界

- 过滤/多源路线 G17/G19 约 `0.7004`，G27C `0.706429`；它们不是严格 raw 单模型 all-query 结论。
- 旧严格无头路线在 GT pretopk、query 数、长训、teacher、density、constructor 等多种干预后仍长期约 `0.40–0.43`。
- 因此被否定的是若干具体实现，不是五个结构问题本身：`query birth`、objectness/local quality、ownership、null calibration、capacity。

### 有保留价值的部件

- P149：稳定 query birth；
- P156：slot state；
- P167：query lifecycle；
- P170：competitive memory；
- P175：output budget；
- OpenSetFlow R25：query-preserving fusion；
- R29：balanced evidence capacity，E5 `0.3698`，比 matched control 高 `0.0205`，但 conditioning 变坏，不能直接晋升。

这些资产共同指向同一句话：**旋转集合检测的核心不是把更多特征塞进 query，而是让有限 query 在信息容量、实例所有权和空集校准之间达成稳定合同。**

## 2.5 2026-07-15—08-12：完整 raw 基线、严格否证和监督合同转向

### 权威基线与近端结果

| Experiment | Protocol | Result | Evidence type | Interpretation |
|---|---|---:|---|---|
| T7 E24 | raw 13,833, Q600, 5D, no NMS/top-k, all18 supervised | mAP `0.6064053488`, AP50 `0.6060` | `VERIFIED_ENDPOINT` | 当前 canonical closed-set substrate；不是 strict OV |
| D11 | generic first600 query, matched endpoint | `0.4521` vs `0.4660`, Δ `-0.0139` | `NEGATIVE_RESULT` | 早期正向未保留到冻结端点 |
| D12 | matched candidate/control | 数值失败，无有效 paired endpoint | `CONFOUNDED` | 丢弃，不能称科学负结果 |
| D13-N | existence residual proxy | `0.838305831` vs `0.847486973`, Δ `-0.009181` | `NEGATIVE_RESULT` | 该 proxy 候选失败 |
| QAF | source reachability + placebos | primary `0.07179`，placebo 近似相同 | `NEGATIVE_RESULT` | source 不敏感，分支关闭 |
| OMQ M0/M1 | source/action matched proxy | source `0.5010` vs placebo约 `0.254/0.257`；action Δ `-0.000307` | `CONFOUNDED` | source 存在但 action 无效，且 split 泄漏 |
| ODQ E6 | raw candidate vs T7 E6 | `0.540336` vs `0.553674`, Δ `-0.013338` | `NEGATIVE_RESULT` | 完整端点关闭 |
| Swin-B B0 E6 | closed-set capacity anchor | `0.576001` vs T7 E6 `0.553674`, Δ `+0.022327` | `MECHANISM_DIAGNOSTIC` | 容量有益，但离预注册绝对 `0.580` 差 `0.003999`，不构成方法因果 |
| Rotation world8 | matched schedule候选 | `0.620096` vs no-rot `0.610377`, Δ `+0.009718` | `CONFOUNDED` | world1 E2 反向 `-0.017873`，尚不稳定 |
| RISC | canonical-ish endpoint | E0 `0.602033`，比 T7 E24低 `0.004373` | `NEGATIVE_RESULT` | 没有越过 parent |
| RISC-Orbit | same family | `0.605425`, Δ `-0.000980` | `NEGATIVE_RESULT` | 近中性 |
| RISC OBF/GC | hard semantic removal | 相对 parent 分别约 `-0.033111/-0.032260` | `NEGATIVE_RESULT` | 删除“有害”语义也删除了有效证据 |
| soft projection | parent-preserving test | 与 parent `0.62009567` 完全相同 | `MECHANISM_DIAGNOSTIC` | 安全但中性 |
| POQ original | map62 E1/E2 | `0.6202033/0.6200977` vs `0.62009567` | `NEGATIVE_RESULT` | 实质中性 |

### OMQ 给出的额外教训

OMQ 的 source gate 很漂亮，但 action endpoint 没有收益；更严重的是 frozen split 中有 164 个共享场景、322 个来自共享场景的验证 tile、86 个存在像素重叠的验证 tile，最大 crop IoU `0.971`。因此：

> 信号存在 ≠ 梯度可利用 ≠ endpoint 提升 ≠ 协议有效。

这四层以后必须独立过门，且 scene-disjoint/content-hash seal 应在训练前完成。

### 近期 POQ/RISC-ER 的准确状态

- Hybrid RSP + POQ Stage M 从 E1 `0.5408` 到 E4 `0.5700`，但缺少同 parent、同 schedule 的 no-POQ control，不能把增长归因给 POQ。
- 单一官方 MM-Grounding-DINO-T parent：E6 filtered-6605 `0.6079678535`，同 checkpoint raw-13833 `0.5794105530`；E12 filtered-6605 `0.6340868473`。它是当前可用 substrate，不是 POQ 因果证据。
- RISC-ER 已完成设计、实现和 8-GPU 两迭代 smoke；尚未完成 12E 或 raw 评测，因此没有 AP 改善结论。
- strict-OV 的 M1 content/header seal 已完成：47,294 RGB，train/dev/test 为 `35109/5172/7013`，跨分区 content SHA 零重叠；但 N0/N1 尚未授权/完成，不能把 all18 closed-set 结果包装成开放词汇证据。

## 3. 三个月形成的研究操作系统

### 3.1 先冻结问题和口径，再允许方法存在

过去最大的无效劳动来自同一实验同时改变父模型、数据 mouth、global batch、BN 状态、评测过滤或输出规则。今后每个候选在写代码前就要写清：

- 唯一主要因果变量；
- frozen parent checkpoint 与 parent-equivalence 检查；
- dataset/split/empty policy；
- query 数、5D 定义、是否 NMS/top-k；
- source gate、action gate、endpoint gate；
- 失败后的禁止性边界。

### 3.2 代理指标只有在能预测 endpoint 时才有价值

SISE、reachability、source separation、P0148 SV ratio 都曾提供洞察，但它们不能替代完整 AP。正确流程是：先用一个历史候选校准代理是否能区分成功/失败，再把它用于低成本筛选。

### 3.3 正结果看 matched endpoint，负结果只关闭“具体配方”

- D11 的早期改善在 E12 反转，说明 early best 不可当结论；
- D12 数值失败不是科学负结果；
- P22 的 500 图失败不能否定 1000 图路线；
- rotation 在 world8 正、world1 负，说明有信号但机制与优化尚未分离。

### 3.4 不要把“去掉坏东西”当成方法

视觉支持替换和 RISC hard removal 都重复证明：被诊断为错误相关的通道往往也承载正确语义。更稳健的方向是分解、路由、置信条件化和 parent-preserving residual，而不是硬删除。

### 3.5 Query 的问题要用 ownership 语言描述

fixed-Q 系统中，一个物体是否被检测不是单纯“feature 好不好”，而是依次经过：

`可达 query → 一对一归属 → 几何收敛 → foreground/null 判定 → 质量排序`。

只优化最后一层 score，无法补救前面的 birth/ownership；只扩 query 数也无法修复错误匹配和 false-background 梯度。

### 3.6 工程 provenance 是实验变量的一部分

环境、checkpoint 注入、worker GPU、DDP sampler、resume 语义、文件覆盖、评测 tensor device、scene leakage 都必须进入实验 ledger。你的后期工作最成熟的部分，正是已经把这些问题从“工程杂务”升级为“论文证据有效性”。

### 3.7 研究节奏应从“大量候选”改为“小量强否证”

推荐以后每轮只维持：一个 canonical parent、一个候选、一个 matched control、一个机制 placebo。两个连续候选不能区分假设时，回到问题建模，而不是继续堆模块。

## 4. 当前主线还缺什么

截至本快照，OV-CapFlow 已拥有强工程与诊断资产，但论文主张仍缺三块：

1. **严格开放词汇证据**：scene-disjoint base-only/meta-novel、sealed official novel4、独立 generic init、N0/N1 matched controls；
2. **可复现的单变量方法增益**：完整 raw 13,833、相同 parent/schedule 的持续正向结果，而不是过滤 mouth 或无 control 的训练曲线；
3. **最接近工作的硬边界**：当前能守住的窄命题是 fixed-Q600 rotated set predictor 中，跨视图稳定证据如何识别有害 false-background classification gradients；不能使用宽泛的“首个开放词汇旋转检测”或“首个端到端遥感检测”措辞。

这也是为什么副产品论文应暂时与 strict-OV 主线解耦：先在 closed-set HBB→OBB 上验证 supervision/ownership 机制，避免把开放词汇、弱监督、旋转集合预测三个难题一次性叠加。

## 5. 三种副产品路线比较

| Route | Annotation cost | Geometry identifiability | 2026 novelty space | Experimental risk | Match to existing assets | Recommendation |
|---|---:|---:|---:|---:|---:|---|
| Full RBox supervision | 高 | 高 | 低；需另有强结构创新 | 低 | 高 | 做 oracle、debug 与上界，不作首选论文主线 |
| Point supervision | 低 | 很低 | 已被 PointOBB/Point2RBox 系列强占 | 高 | 中 | 作为第二阶段扩展，不在 v1 同时承担 |
| HBB supervision | 中低 | 中；中心/尺度/实例身份已知，方向欠定 | 单纯 consistency 已拥挤，但 set ownership 仍有空间 | 中 | 很高 | **首选主线** |
| Mixed RBox/HBB/point | 可调 | 中高 | Wholly-WOOD/PWOOD 已直接覆盖统一或 partial weak | 中高 | 中 | 只做标注成本曲线，不作主 claim |

选择 HBB 的核心理由不是“它最容易”，而是它恰好留下了与你的能力匹配的困难：方向与形状是潜变量，但每个实例的中心、粗尺度和类别仍然可用于一对一 matching。这样可以把 OpenRSD 的 rotation 经验和 OV-CapFlow 的 ownership/null 经验统一到一个可检验问题里。

## 6. 推荐框架：HALO-Set（暂名）

### 6.1 研究问题

给定训练图像及完整 HBB 标注

\[
H_j=(c_x,c_y,W,H,y_j),
\]

训练一个固定 Q 的端到端检测器，测试时直接输出旋转框

\[
r_i=(x,y,w,h,\theta),
\]

不使用测试时 NMS，也不把单个 hard pseudo-RBox 当作监督真值。

对于最小外接 HBB，旋转框与水平包络满足

\[
W=|w\cos\theta|+|h\sin\theta|,\qquad
H=|w\sin\theta|+|h\cos\theta|.
\]

这个约束通常不能唯一决定 \((w,h,\theta)\)，在接近 45° 时尤其病态。因此缺失的不是一个可以随便回归的角度标量，而是一个**多模态潜在几何集合**。

### 6.2 核心假设

> 若训练时保留 HBB 所允许的几何多解性，并在多视图中锁定“同一标注实例的 query ownership”，再延迟对歧义 query 的 background 判罚，那么固定查询检测器可以减少 hard pseudo-box 的确认偏差，在密集、小目标场景中学到更可靠的 OBB。

### 6.3 组件 A：Latent Feasible-Set Box Supervision

每个 matched query 不只输出单一角度，而输出 K 个低成本方向/形状粒子及其权重，角度用 doubled-angle 表示以处理 \(\theta\) 与 \(\theta+\pi\) 等价：

\[
p_i(2\theta)=\sum_{k=1}^{K}\pi_{ik}\,p_{ik}(2\theta).
\]

训练时不挑一个粒子写成伪标签，而对 HBB envelope energy 做加权边缘化：

\[
\mathcal L_{\text{feasible}}
=-\log\sum_k\pi_{ik}\exp[-d(\operatorname{AABB}(r_{ik}),H_j)/\tau].
\]

若数据中 HBB 是 coarse box 而非最小外接框，引入有界的 per-instance slack/scale nuisance；必须用 tight-HBB 与 coarse-HBB 分层实验验证，不能让 slack 无约束吞掉监督。

### 6.4 组件 B：Orbit-Consistent Ownership Tracking

对原图、旋转图和翻转图保留同一个 annotation ID。每个视图仍独立进行一对一 Hungarian matching，但匹配后通过标注 ID 形成 ownership track；query index 可以随空间变换运输，不强迫原始 slot 编号完全相同。

匹配代价只使用弱标注可观测量：

\[
C_{ij}=\lambda_{cls}C_{cls}+\lambda_c\lVert c_i-c_j\rVert_1
+\lambda_e d(\operatorname{AABB}(r_i),H_j)
+\lambda_o C_{orbit}.
\]

旋转 \(\alpha\) 后，方向后验应平移 \(2\alpha\)；翻转后按相应符号变换。比较的是整个分布/粒子集合，而不是两个可能都错的单点角度。这一点区别于“两个视图预测值直接做 L1”。

### 6.5 组件 C：Ownership-Safe Negative Supervision

标准 DETR 会把所有 unmatched query 直接作为 no-object。弱几何监督早期可能选错 owner，于是同一 HBB 内仍然合理的 query 会收到强 false-background 梯度。

建议把未匹配 query 分三类：

1. `safe background`：中心和 envelope 与所有 HBB 支持区明显分离，正常施加 no-object；
2. `local contender`：落在某个 HBB 的可行支持内，但没有成为 primary owner，早期分类 loss ignore/soften，同时通过局部 exclusivity loss 抑制重复；
3. `owned foreground`：唯一 primary query，承担类别和潜在几何监督。

随着 ownership track 稳定度与后验熵下降，逐步缩小 contender ignore 区。这样不是放任重复，而是把“背景判罚”和“同实例竞争”拆成两个不同损失。

### 6.6 组件 D：Local Exclusivity and Capacity Audit

每个 HBB 只允许一个最终 active owner；附近 contender 通过局部相斥或 overlap energy 退场。固定 Q 保持不变，但训练和评测必须报告：

- 每个标注的 candidate count；
- ownership flip rate；
- 未覆盖实例率；
- 每图 GT/Q 比与 overflow 下界；
- duplicate rate 和 empty-tile foreground；
- 小目标/密集度分桶。

当某图 GT 数超过 Q 时，容量缺口是数学下界，不能被误报成 matching 算法失败。

### 6.7 训练与推理

- 训练：原图 + 一种随机 orbit view；不需要外部 teacher，不落盘 hard pseudo-RBox。
- 推理：单视图、Q 行、每 query 选择最高权重粒子，直接输出 OBB；不增加 NMS/top-k。
- Full-RBox adapter：直接监督真值粒子，用作上界、debug 和 parent sanity check。
- Point adapter：以后可用 point + Voronoi/layout support 替换 HBB envelope，但不建议在第一版同时实现。

## 7. 与现有工作的边界

当前检索得到的结论是：

- H2RBox/H2RBox-v2 已覆盖 HBB + 旋转/翻转一致性，后者在多数据集接近 full-supervised Rotated FCOS；
- ABBSPO 已专门处理 tight/coarse HBB 的尺度偏差与对称先验；
- PointOBB、Point2RBox 及其 v2 已覆盖 point + 多视图、自监督角度、伪框生成和密集布局约束；
- Wholly-WOOD 与 PWOOD 已覆盖多种/部分弱标注统一使用；
- AO2-DETR、ARS-DETR、RHINO 等已覆盖 full-supervised rotated set prediction、旋转 matching 和 query denoising。

因此不能写的 claim 包括：

- “首次用 HBB 学旋转框”；
- “首次利用旋转/翻转一致性”；
- “首次统一 point/HBB/RBox”；
- “首次端到端旋转 DETR”。

暂时可能成立、但仍需继续 closest-work search 的窄边界是：

> 在固定查询的一对一旋转集合预测器中，将 HBB 监督表述为 latent feasible-set marginalization，并将跨轨道实例 ownership 与 ambiguity-aware negative supervision 联合建模。

截至本次检索，尚未找到完全相同的组合；这只是“未检出”，不是“首个”的证明。

## 8. 最小可信实验方案

## 8.1 先冻结协议

1. 主数据集先用 DOTA-v1.0：从 train RBox 自动生成 circumscribed HBB，训练代码只能读取 HBB；val/test RBox 密封，仅评测使用。
2. 第二数据集用 HRSC2016 验证单类长条目标和角度精度；第三数据集再选 DIOR-R 或 DOTA-v1.5/v2.0 验证类别/密度泛化。
3. patch、scale、augmentation、backbone、schedule、global batch、seed、empty policy 完全 matched。
4. 机制 sandbox 可先用现有 Q600 5D 代码；论文主结果应移植到一个已复现官方精度的强 rotated-DINO/RHINO 类 parent，避免“方法有效但底座太弱”掩盖结论。
5. 第一篇先做 closed-set HBB→OBB；不要同时引入 open vocabulary。

## 8.2 必备 baseline

- Full-RBox parent：同一 set detector 的上界；
- Native HBB set baseline：Hungarian + center/class + AABB envelope，单角度输出；
- Hard pseudo-RBox baseline：从当前最可信角度直接回归；
- H2RBox-v2、ABBSPO：同 backbone 不一定都能公平移植，因此至少同时报告官方范式复现和相同检测头近似对照；
- 强 full-supervised dense baseline：Rotated FCOS 或 Oriented R-CNN；
- 若做 point 扩展，再加入 PointOBB-v2、Point2RBox-v2，不能只与早期方法比。

## 8.3 消融矩阵

| ID | Change from native HBB set baseline | 回答的问题 |
|---|---|---|
| A1 | hard pseudo angle → latent K-particle feasible set | 是否减少伪标签确认偏差 |
| A2 | independent view matching → annotation-ID orbit track | 是否稳定 ownership |
| A3 | all unmatched hard negative → safe/contender/background split | 是否减少 false-background 梯度 |
| A4 | single-angle L1 → doubled-angle distribution | 是否处理周期/对称多模态 |
| A5 | no local exclusion → local exclusivity | 是否在 dense scene 防重复 |
| A6 | rotate only / flip only / both | 收益来自哪种群变换 |
| A7 | exact HBB / coarse HBB noise | 方法是否依赖理想包络 |
| A8 | Q300/Q600/Q900 | ownership 收益是否只是容量变化 |

每个主消融只改变一个组件；不要一次把 A1–A5 全开后再做倒序 removal，因为那无法定位优化耦合。

## 8.4 指标

### 任务指标

- DOTA official AP50；
- AP75 和可行时的 AP50:95，避免角度误差被 AP50 掩盖；
- per-class AP；
- small/medium/large、density quartile、aspect ratio、angle bin；
- HRSC angle-sensitive 指标与速度/显存/参数量。

### 机制指标

- orbit ownership retention / flip rate；
- matched query 的 posterior entropy 与校准；
- feasible-set coverage：GT RBox 是否落入训练后高概率集合；
- safe-negative precision：被判 safe background 的 query 中，是否存在高-IoU GT；
- duplicate/empty FP；
- query reachability、GT/Q overflow、每实例 contender 数；
- hard pseudo-RBox angle error 与 latent posterior oracle error。

### 标注效率

若后续引入少量 RBox，只报告 1%/5%/10% RBox + 其余 HBB 的成本曲线；它是扩展实验，不应成为 HALO-Set 的必要条件。

## 8.5 预注册晋级门

建议采用三级门，而不是一开始跑完整大实验：

1. `G0 Parent`: full-RBox parent 复现官方/历史允许误差，HBB native baseline 稳定训练，所有 annotation seal 通过；
2. `G1 Mechanism`: matched 短端点下，ownership flip、false-background 和 dense duplicate 至少两项按预期改善，且 placebo 不改善；
3. `G2 Endpoint screen`: 完整同口径 mAP 至少 `+0.005`，无关键类别超过 `-0.01` 崩塌；
4. `G3 Paper`: 三 seed 均值相对最强 reproduced HBB baseline 至少约 `+1.0 AP point`，或在相近 AP 下显著减少标注/推理成本，并在第二数据集复现；
5. `Stop`: 机制指标未动、AP 下降超过 `0.005`、收益只来自 Q/scale/batch、只在 filtered mouth 有效、或需要 hard pseudo-box/NMS 才成立，立即关闭具体配方。

`+1.0 AP point` 是研究资源门槛，不是对未来结果的承诺；若领域最强 baseline 已很高，也可用 full-supervision gap closure 和统计置信区间替代固定阈值。

## 9. 风险与反例

| Risk | Why it matters | Required countermeasure |
|---|---|---|
| Closest-work collision | HBB/point 弱监督已非常拥挤 | 继续查 transformer/set-prediction 方向，主张保持窄 |
| HBB 不唯一导致塌到错误角度 | 多视图可能一致地错 | 多模态 posterior、对称类别分层、GT 仅用于 sealed diagnostic |
| Coarse HBB 破坏精确 envelope | DIOR 类标注未必是最小外接框 | 有界 slack + tight/coarse 分层，不能无约束自适应 |
| Fixed-Q dense capacity | DOTA 局部实例可超过 Q | 显式报告 overflow，下游不把数学下界归因给方法 |
| Weak parent hides gain | 当前某些 set parent 低于 dense SOTA | sandbox 与 paper parent 两阶段 |
| Negative-safe 造成重复 | ignore contender 可能放任多框 | 独立 local exclusivity，报告 duplicate/empty FP |
| Open-vocabulary confound | 文本 alignment 会掩盖几何监督结论 | v1 closed-set；成功后再接 OV |
| 多组件故事过重 | A1–A5 可能无法归因 | 逐组件晋级，每步 matched control |

## 10. 推荐执行顺序

### Phase A：不写新大模块，先做可辨识性审计

- 用现有 full RBox checkpoint 离线计算：给定 HBB，真实 RBox 在不同角度/aspect/density 下的 feasible-set 大小和病态程度；
- 检查单点 hard pseudo-angle 的错误是否确实集中在密集、小目标、近 45° 与对称类别；
- 检索是否已有 HBB-supervised DETR 对 latent box set/ambiguous unmatched query 做同样处理。

只有这三项支持假设，才进入实现。

### Phase B：最小 HBB set baseline

- 在 closed-set Q600 5D parent 上只加入 HBB envelope matching/loss；
- 先证明无 RBox 训练输入、评测 mouth 与 full parent 对齐；
- 生成 ownership/negative-gradient 基线诊断。

### Phase C：按 A1 → A2 → A3 顺序单变量增加

- 先验证 latent feasible set；
- 再验证 orbit ownership；
- 最后验证 negative-safe + exclusivity；
- 每个组件不过机制门就不进入完整训练。

### Phase D：强 parent 与多数据集

- 把通过的最小组合移植到复现合格的 rotated-DINO/RHINO 类 parent；
- DOTA-v1.0 + HRSC2016；
- 再决定 DIOR-R、DOTA-v2.0 或 point adapter。

## 11. 最终判断

这三个月最有价值的积累不是某个现成模块，而是三条可组合的规律：

1. **旋转视图会暴露稳定与不稳定证据，但直接删特征会伤害正确语义；**
2. **固定查询系统的核心是实例 ownership 和 null/negative contract，而不是后处理；**
3. **弱监督最危险的不是监督少，而是把不确定性过早伪装成确定标签。**

HALO-Set 正好把三条规律变成一个窄而清晰的问题：在 HBB 只给出旋转框可行集合时，如何让一对一 query 在变换轨道上保持实例所有权，并只对真正安全的背景施加负梯度。

因此我的建议是：

- **主线选择 HBB-supervised OBB；**
- **full supervision 只做 oracle/parent；**
- **point supervision 延后为扩展；**
- **先做 Phase A 的离线可辨识性审计，再决定是否实现 HALO-Set。**

在获得明确批准前，本报告不授权代码实现、数据转换或训练启动。

## 12. 本地证据索引

### OpenRSD

- `resultmd/exp_resultmd_global_summaries/reports/fres_two_week_engineering_report_20260512_20260526.md`
- `resultmd/exp_two_week_experiment_summary/fres_20260624_two_week_experiment_report.md`
- `resultmd/exp_p15l_hrrsd_e2e/fres_20260624_p15po_openrsd_gpu45_big_table.md`
- `resultmd/exp_p22_dota2_strict_e2e_subset_20260626/fres_dota2_strict_e2e_train500_1000_baseline.md`

### OV-CapFlow

- `docs/project_history/exp_20260714_two_week_review/fres_20260701_20260714_full_review_zh.md`
- `docs/project_history/exp_20260729_20260807_ten_day_review/fres_20260729_20260807_full_review_reflection_zh.md`
- `docs/superpowers/specs/2026-08-11-risc-er-formal-design.md`
- `docs/project_history/exp_20260811_risc_er/flog_risc_er_zh.md`
- `.lab/workspace/exp-8-swinb-b0-r3/RESUME_NEXT_GPU_WINDOW.md`

## 13. 文献索引

系统化筛选、评分和链接见：

- `docs/literature-search-20260812-weakly-supervised-oriented-set-detection/papers.md`
- `docs/literature-search-20260812-weakly-supervised-oriented-set-detection/papers.csv`
- `docs/literature-search-20260812-weakly-supervised-oriented-set-detection/search-notes.md`
