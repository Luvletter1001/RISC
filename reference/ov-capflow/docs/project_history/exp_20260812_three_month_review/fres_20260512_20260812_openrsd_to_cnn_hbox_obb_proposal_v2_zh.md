# OpenRSD → OV-CapFlow 经验复盘与 CNN-HBox 旋转检测新方案（v2）

> 时间窗：2026-05-12 至 2026-08-12  
> 版本日期：2026-08-12  
> 状态：研究设计版；主线实验仍在推进，本文件不代表方法已经实现或取得增益  
> v2 核心修订：**以 CNN 稠密检测器为主体；fixed-query 只作为问题建模经验，不作为网络结构。**

## 0. 最终建议

新版本不再推荐固定查询的 HALO-Set 作为首篇副产品，而推荐：

> **OrbitAssign：面向水平框监督旋转检测的跨视图软实例分配。**

主模型使用 **ResNet-50 + FPN + Rotated FCOS/H2RBox-v2 head**。训练时保留原图、随机旋转图、翻转图，利用已知几何变换和 annotation ID，把每个 FPN 位置对每个标注实例的归属表示为软分配，而不是先选一个唯一伪旋转框。跨视图对齐的软分配负责正样本选择；水平框内部尚不能确认归属的位置不立即当作硬背景。测试时仍是普通 CNN 稠密输出和 rotated NMS，不引入 Transformer decoder、fixed queries、Hungarian matching 或额外推理分支。

这一选择有四个理由：

1. 它服从“CNN base”的要求，并且直接复用 OpenRSD 中已有的 Rotated FCOS、H2RBox-v2、Rotated RTMDet 资产。
2. 它没有丢掉 fixed-query 阶段真正有价值的经验，而是把 query ownership、unmatched false background、capacity/competition 转译为 CNN 的位置归属、歧义负样本和跨 FPN 竞争。
3. 它绕开了已经非常拥挤的宽命题：H2RBox/H2RBox-v2 已覆盖旋转与翻转一致性，BGHR 已覆盖 HBox 下的样本挖掘，ABBSPO 已覆盖尺度失配与对称先验，Wholly-WOOD/PWOOD/SPWOOD 已覆盖混合或部分弱监督。
4. 它把贡献压到一个可以被单独证伪的点：**一致的应该不只是预测框角度，还应包括生成监督的 location–instance assignment field。**

我对三个候选方向的排序是：

| Rank | Route | Judgment |
|---:|---|---|
| 1 | H2RBox-v2/Rotated FCOS + OrbitAssign | **主方案**；最容易与最近工作公平比较，也最能隔离样本归属因果变量 |
| 2 | Rotated RTMDet + orbit-aware dynamic assigner | 第二阶段迁移；工程强，但原生 dynamic soft assigner 会混淆首轮归因 |
| 3 | teacher/student + hard pseudo-RBox | 不作主线；确认偏差高，并与 Wholly-WOOD、PWOOD、SPWOOD 的故事靠得过近 |

## 1. v2 相对上一版改变了什么

上一版 HALO-Set 的核心是固定查询、一对一 matching、潜在旋转框集合和无 NMS 推理。v2 作以下收缩：

- 删除 Transformer decoder、fixed-Q、Hungarian、query particles 和“无 NMS”主张；
- 不再尝试同时统一 full/point/HBox 三种监督；首篇只做**完整 HBox 标注下的 HBox→RBox**；
- 保留 H2RBox-v2 的框投影与旋转/翻转对称学习，把创新变量集中到**训练样本分配**；
- fixed-query 的经验只用于解释“谁负责哪个实例、哪些负样本是安全的、竞争如何稳定”，不用于规定网络形态；
- 全监督 RBox 仅作为上界和机制 oracle，点监督只作为成功后的扩展。

这是一次主动减法。新方案只有两个核心组件：

1. **Orbit-Aligned Soft Assignment（跨视图软实例分配）**；
2. **Ambiguity-Safe Background Learning（歧义安全背景学习）**。

其余角度编码、HBox envelope loss、backbone、NMS 都使用公开基线，避免论文变成模块堆叠。

## 2. 三个月实验的重新归纳

### 2.1 认知迁移链

| Stage | 主要现象 | 经验证的结论 | 对 OrbitAssign 的含义 |
|---|---|---|---|
| OpenRSD 早期：false-SV | P0148 出现强 `small_vehicle` 语义聚集；dense logits 已有偏置，NMS/top-k 继续放大 | 错误不是后处理凭空制造；粗暴删除视觉支持会同时破坏正确检测 | 不从 NMS 端“补结果”，而从产生 logits/targets 的训练分配上游处理 |
| OpenRSD 中期：局部模块筛选 | text/rotation/adapter 多数短跑不越过 matched baseline；BN/no-op 可制造约 1.5 point 假象 | 单模块直觉不足；parent、BN、batch、mouth 必须匹配 | 新方法必须只改变 assignment/negative contract，保留相同 CNN parent |
| 严格端到端转折 | HRRSD P15O-B 严格无 dense head/NMS 达 `0.8865`；DOTA-v2 小数据规模又暴露优化门槛 | ownership 是结构问题；但端到端路线的成功依赖数据规模和正确 query birth | 吸收 ownership 语言，但不把 query 结构强行搬到 HBox 任务 |
| OV-CapFlow fixed-Q | 旧路线长期约 `0.40–0.43`；T7 raw 13,833、Q600、all18 到 `0.606405` | query birth、ownership、null、capacity 和排序构成连续因果链 | 在 CNN 中对应候选位置覆盖、实例归属、背景状态、FPN/重叠竞争和质量校准 |
| 后期强否证 | D11、D13-N、QAF、ODQ、RISC、POQ 多条路线被 matched endpoint/placebo 关闭；OMQ 还暴露 split leakage | source existence、actionability、endpoint、protocol integrity 必须分开过门 | OrbitAssign 先做 assignment 污染诊断，再做最小 action，最后才看 AP |

### 2.2 真正应保留的经验

#### 经验一：错误要在产生处修

OpenRSD 已经表明 dense logits 中的错误会被后续排序、top-k 和 NMS 放大。CNN 版本允许使用标准 rotated NMS，但不能把 NMS threshold、pre-NMS top-k 或 score calibration 当作主要方法。主因果变量必须位于训练 target 和 dense score 的生成处。

#### 经验二：ownership 比“再加一种特征”更接近根因

fixed-query 中，一个目标先要被 query 覆盖，再被唯一 query 拥有，之后才谈几何和类别。CNN 没有全局固定 Q，但仍要回答：

- 哪些 FPN locations 可以负责实例；
- 重叠 HBox 中一个 location 属于哪个实例；
- 一个实例是否至少获得足够监督；
- 未被选中的 HBox 内部位置究竟是背景，还是暂时不确定。

因此 v2 把研究对象从“预测框是否一致”改为“**监督归属关系是否一致**”。

#### 经验三：不确定性不能过早伪装成负标签

H2RBox-v2 的现有 dense target 路径使用 center sampling、regression range 和最小面积规则；没有入选的位置被设为 background。对细长、斜置、密集目标，这**可能**把 HBox 内真实物体区域或实例竞争区域变成 focal-loss 负样本；其实际比例必须由 `G1 Source` 诊断确认。BGHR 已指出 HBox 与 RBox 监督之间存在样本选择差距，但其 AFSM 先选一个最佳预测 RBox，再据此形成正样本，仍可能在早期形成单一路径确认偏差。

#### 经验四：matched control 和协议 seal 是方法的一部分

新项目必须继承后期已经成熟的实验纪律：

- 同 backbone、neck、head、schedule、batch、augmentation 和评测口径；
- 一次只替换 assignment/negative contract；
- HBox 训练 dataloader 不得接触 RBox；
- source diagnostic、action gate、endpoint gate 分开；
- 完整数据集结果优先于 proxy/best checkpoint；
- 报告负结果和关闭条件。

## 3. 为什么主干选择 Rotated FCOS，而不是先上 RTMDet

### 3.1 主基线：H2RBox-v2 / Rotated FCOS R50-FPN

H2RBox-v2 本身就是 CNN anchor-free dense detector。当前 OpenRSD 实现继承 `RotatedFCOSHead`，FPN strides 为 `8/16/32/64/128`，采用 center sampling。它非常适合作为首轮 parent：

- 与 H2RBox、H2RBox-v2、BGHR、ABBSPO 的论文口径最接近；
- target assignment 写得清楚，可以直接测量每个位置为何成为正/负；
- 不需要改 backbone 或检测头即可隔离新变量；
- 本地已有 detector、head、loss、DOTA/DIOR 配置和历史评测资产。

本地代码证据：

- `/data1/zcy/OpenRSD/mmrotate/models/dense_heads/h2rbox_v2_head.py`：center/range/min-area target assignment；未入选位置设为 background；box ID 主要用于聚合三视图的 angle prediction。
- `/data1/zcy/OpenRSD/mmrotate/models/detectors/h2rbox_v2.py`：生成 original/rotated/flipped 三视图并保留 annotation identity。
- `/data1/zcy/OpenRSD/mmrotate_configs/h2rbox/`：R50-FPN、五层 stride、中心采样的可复现基线。

### 3.2 第二基线：Rotated RTMDet

Rotated RTMDet 更适合做“可迁移性证明”，不适合做第一个因果 parent：

- 它已有 `BatchDynamicSoftLabelAssigner` 和 quality focal loss；
- 若首轮直接在 RTMDet 上改 assignment，很难区分收益来自 HBox 机制还是原生 dynamic assignment 的交互；
- OpenRSD 中 RTMDet 资产丰富，待 FCOS 上证明机制后迁移成本不高。

因此推荐顺序是：**FCOS 做论文主因果，RTMDet 做架构泛化。**

## 4. fixed-query 经验如何转译到 CNN

| fixed-query 经验 | CNN 中的对应对象 | v2 的处理 | 不照搬的部分 |
|---|---|---|---|
| query birth / reachability | 目标是否在某个 FPN level 和候选位置集合中可达 | 保留跨层候选 support，显式报告 per-level coverage | 不保留 learnable query birth |
| one-to-one ownership | location–instance assignment；一个位置最多服务一个实例 | 软分配矩阵带列容量/局部排他约束 | 不做全局 Hungarian 一对一 |
| unmatched query false background | HBox 内未选 location 被错误当背景 | 高歧义位置 ignore/soft-negative；框外才是天然 safe background | 不建立固定数量 null queries |
| Q capacity / competition | 重叠实例、FPN 层和正样本预算之间的竞争 | 每实例最低覆盖 + 每位置单实例容量；报告 overflow/conflict | 不设 Q=600 全局上限 |
| query identity across views | 同一 annotation ID 的 assignment field | 用已知变换对齐 original/rotate/flip 的软归属 | 不追踪 decoder slot identity |
| null/existence calibration | foreground/background 概率与 negative weight | 显式区分 safe background、stable local background、ambiguous band | 不沿用 DETR no-object class |
| NMS/top-k amplification | dense raw score 的源头偏差被 mouth 放大 | 同时报告 pre-NMS 和 post-NMS；方法只改训练上游 | 不把“无 NMS”当 CNN 论文要求 |

需要特别强调：**fixed-query 给 v2 的是问题语言和失败模式，不是结构答案。** OrbitAssign 没有 Transformer decoder、learnable queries、Hungarian matching、固定 Q、all-query inference 或 NMS-free claim。

## 5. 最近文献比较与可守住的边界

### 5.1 HBox 监督的直接近邻

| Work | 已占据的核心 | 对 v2 的压力 | OrbitAssign 的区别 |
|---|---|---|---|
| [H2RBox](https://arxiv.org/abs/2210.06742) | CNN 上通过旋转双视图一致性从 HBox 学角度 | “用旋转一致性学 OBB”不能再作为新意 | 不把 box consistency 当贡献；改变 location–instance supervision |
| [H2RBox-v2](https://papers.neurips.cc/paper_files/paper/2023/hash/b9603de9e49d0838e53b6c9cf9d06556-Abstract-Conference.html) | rotate + flip symmetry；报告 DOTA-v1.0 上 `72.31` 对 full Rotated FCOS `72.44` | 仅靠三视图对称损失很难再写一篇 | 从 angle/box output consistency 下沉到 assignment-field equivariance |
| [EIE-Det](https://doi.org/10.1109/TETCI.2024.3398020) | 显式/隐式 box equivariance | “box equivariance”宽命题已占 | 关注谁给谁提供监督，而不是只约束框参数 |
| [BGHR](https://ojs.aaai.org/index.php/AAAI/article/view/32310) | AFSM/PRA：为每个 GT HBox 选最佳预测 RBox，再挖正样本 | **最接近工作**；“HBox 动态样本分配”已被占据 | 不先坍缩到单个 hard RBox；保留软分配并跨视图对齐，再逐步退火 |
| [ABBSPO](https://openaccess.thecvf.com/content/CVPR2025/html/Lee_ABBSPO_Adaptive_Bounding_Box_Scaling_and_Symmetric_Prior_based_Orientation_CVPR_2025_paper.html) | coarse/tight HBox 尺度失配与 symmetric prior angle | 尺度自适应、对称先验不能作为 v2 主张 | 可作互补 baseline/plugin；v2 主变量是 assignment 与 negative safety |
| [Wholly-WOOD](https://arxiv.org/abs/2502.09471) | point/HBox/RBox/mixed 统一利用 | 阻止“统一多种监督”的宽 claim | v2 只研究完整 HBox 标注，不做统一框架 |
| [PWOOD](https://arxiv.org/abs/2507.02751) / [SPWOOD](https://openreview.net/pdf?id=PXaboOpwGv) | 部分/稀疏弱标注、未标目标和伪标签过滤 | negative safety 的宽叙事已出现 | v2 假定每个实例都有 HBox；歧义来自 HBox 内部支持，而非漏标实例 |

### 5.2 稠密分配的通用近邻

| Work | 已占据的核心 | 不能声称什么 |
|---|---|---|
| [ATSS](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Bridging_the_Gap_Between_Anchor-Based_and_Anchor-Free_Detection_via_Adaptive_CVPR_2020_paper.html) | 统计式自适应正负样本选择 | 不能声称首个 adaptive assignment |
| [OTA](https://openaccess.thecvf.com/content/CVPR2021/html/Ge_OTA_Optimal_Transport_Assignment_for_Object_Detection_CVPR_2021_paper.html) | 用最优传输全局解决 GT–anchor 分配，尤其面向拥挤场景 | 不能把 transport/全局竞争本身当创新 |
| [IQDet](https://openaccess.thecvf.com/content/CVPR2021/html/Ma_IQDet_Instance-Wise_Quality_Distribution_Sampling_for_Object_Detection_CVPR_2021_paper.html) | 实例级质量分布和概率采样，训练期辅助、推理零开销 | 不能声称首个 soft/distribution sampling |
| [DCFL](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_Dynamic_Coarse-To-Fine_Learning_for_Oriented_Tiny_Object_Detection_CVPR_2023_paper.html) | 全监督旋转微小目标中的 dynamic prior + coarse-to-fine assigner | 不能声称首个 oriented dynamic assignment |
| [RTMDet](https://arxiv.org/abs/2212.07784) | dynamic soft-label assignment 和高效 CNN detector | 不能把 soft label 或 RTMDet 移植当贡献 |
| [TS-Conv](https://arxiv.org/abs/2209.02200) | task-wise sampling、特征对齐和动态 task-aware assignment | 不能以“分类/定位采样对齐”为宽主张 |
| [FRED](https://ojs.aaai.org/index.php/AAAI/article/view/28069) | 完整 CNN 检测流程的 rotation equivariance | 不能声称首个 rotation-equivariant detector |

### 5.3 截至 2026-08-12 的暂定空隙

本次定向检索没有发现一篇工作把以下组合设为核心：

> 在**完整 HBox 标注、CNN 稠密旋转检测**中，对 original/rotated/flipped views 的 **soft location–instance assignment distribution** 进行 annotation-ID 对齐，并用分配熵控制 HBox 内部的 background gradient。

这是“当前未发现精确匹配”，不是“首个”结论。正式投稿前仍需从 H2RBox-v2、BGHR、ABBSPO、EIE-Det、Wholly-WOOD 做前向/后向引用追踪。

## 6. 研究问题的重新定义

给定训练集

\[
\mathcal D=\{(I_i,\{H_{ij},y_{ij}\}_{j=1}^{n_i})\},
\]

其中每个实例只有水平框

\[
H_j=(c_x,c_y,W,H)
\]

和类别标签。CNN 在每个 FPN location \(p\) 输出类别分数 \(s_p\)、旋转框 \(b_p=(x,y,w,h,\theta)\) 与 centerness/quality \(u_p\)。

现有问题通常被表述为“角度监督缺失”。v2 的判断是：还存在第二个隐变量：

\[
Z_{pj}\in\{0,1\},
\]

即 location \(p\) 是否应由实例 \(j\) 监督。HBox 比真实物体范围更宽，尤其在细长和斜置目标上；因此 \(Z\) 不能由“落在 HBox/中心区”完全确定。

核心假设是：

> 对同一物理图像做已知旋转或翻转后，单个 location 的离散采样会变化，但同一物理区域属于哪个 annotation instance 的关系应保持等变。若先形成跨视图共识的软归属，再决定正样本和背景权重，可以比单视图 hard RBox mining 更少地产生确认偏差与 false-background 梯度。

## 7. 三种 CNN 实现路线

### Route A：H2RBox-v2 + OrbitAssign（推荐）

- Parent：ResNet-50 + FPN + Rotated FCOS/H2RBox-v2 head；
- 保留：HBox projection regression、rotate/flip symmetry、centerness；
- 替换：静态 center/min-area target assignment 和 HBox 内硬背景；
- 优点：最近工作可比性最高，变量最干净；
- 风险：FCOS 上限可能低于更现代 detector，但这可以由 RTMDet 迁移补充。

### Route B：Rotated RTMDet + orbit-aware SimOTA

- Parent：CSPNeXt + PAFPN + RTMDet-R；
- 在 `BatchDynamicSoftLabelAssigner` 成本中加入 HBox envelope 与 cross-view plan consistency；
- 优点：性能/效率潜力强，适合后续工程落地；
- 风险：原生 dynamic assigner 已经很强，首轮消融难以讲清楚。

### Route C：EMA teacher + pseudo-RBox

- Teacher 生成 RBox，student 用 RBox assignment；
- 优点：实现直观，可能容易获得 AP；
- 风险：早期 hard pseudo-angle/shape 自举，和 BGHR、Wholly-WOOD、PWOOD/SPWOOD 距离过近，故事不干净。

结论：先做 A；A 的机制成立后，把同一训练期 assignment target 移植到 B；C 仅作为对照，不作为主方法。

## 8. OrbitAssign 设计

## 8.1 总体结构

训练视图集合为

\[
g\in\mathcal G=\{e,r_\alpha,f\},
\]

分别表示原图、随机旋转和翻转。已知图像变换为 \(T_g\)，同一标注实例在各视图中保留相同 annotation ID。

训练路径：

\[
\text{CNN/FPN predictions}
\rightarrow \text{per-view assignment cost}
\rightarrow \text{inverse warp to canonical view}
\rightarrow \text{soft ownership plan}
\rightarrow \text{soft positive/negative targets}.
\]

推理路径完全不含上述 assignment 模块：

\[
I\rightarrow \text{CNN+FPN+Rotated FCOS head}\rightarrow \text{rotated NMS}.
\]

## 8.2 候选支持域

对实例 \(j\) 和视图 \(g\)，候选位置集合 \(\mathcal C_j^g\) 满足：

- 位置位于经 \(T_g\) 精确变换后的 HBox 区域；
- 满足 FCOS 的 FPN regression range；
- 额外保留一个很小的中心 seed，确保训练初期每个实例至少可达；
- 不依赖 GT RBox、mask、SAM 或外部伪标签。

这里使用“变换后的原 HBox 区域”，不是重新从未知 RBox 构造新的 tight HBox，因此没有引入额外标注信息。

## 8.3 每视图 assignment cost

对候选 pair \((p,j)\)，定义训练期 cost：

\[
C_{jp}^{g}
=\lambda_{cls}C_{cls}
+\lambda_{env}C_{env}
+\lambda_{ctr}C_{ctr}
+\lambda_{q}C_{q}.
\]

其中：

- \(C_{cls}=-\log \sigma(s_{p,y_j})\)：类别兼容；
- \(C_{env}\)：预测 RBox 的水平包络与标注 HBox 的距离，使用 parent 的 CircumIoU/IoU/KLD 口径；
- \(C_{ctr}\)：位置到 HBox 中心的尺度归一化距离，只作弱先验；
- \(C_q=1-u_p\)：centerness/localization quality。

所有预测项在构造 assignment target 时 `stop-gradient`，避免模型通过同时改变 cost 与监督目标获得平凡解。

## 8.4 跨视图共识

把各视图的 cost 和 valid mask 用已知 \(T_g^{-1}\) 映射回 canonical image coordinates：

\[
\widetilde C^g=\mathcal W_{g\rightarrow e}(C^g),
\qquad
\bar C=\frac{\sum_g M^g\odot \widetilde C^g}{\sum_g M^g+\epsilon}.
\]

FPN location 不一定精确落在另一个视图的网格点，因此 \(\mathcal W\) 使用双线性 gather/splat，并只在共同有效区域比较。旋转不改变物理尺度，默认逐 FPN level 对齐；跨层迁移单独作为消融，避免把 level routing 混进主变量。

## 8.5 软实例分配

在 canonical cost 上求训练期的熵正则软分配：

\[
\bar P=\arg\min_{P\ge 0}
\langle P,\bar C\rangle
+\varepsilon\sum_{jp}P_{jp}(\log P_{jp}-1).
\]

使用轻量 unbalanced transport/projection 满足：

1. 每个 location 的 foreground mass 不超过 1，避免同时回归多个实例；
2. 每个实例获得有界最低 coverage，避免小目标或困难实例无监督；
3. background 吸收剩余 mass；
4. 正样本预算只作宽松上下界，不要求一个实例只有一个 location。

这不是 DETR Hungarian：它是 CNN 稠密训练中的 many-locations-to-one-instance 软计划，推理时不存在。

将共识计划推到每个视图：

\[
P^g=\mathcal W_{e\rightarrow g}(\bar P),
\]

并让该视图由自身预测得到的 assignment posterior \(Q^g\) 接近 \(P^g\)：

\[
\mathcal L_{orbit}
=\frac{1}{|\mathcal G|}\sum_g
D_{JS}(Q^g\Vert P^g).
\]

真正的新变量是 \(P\) 的跨视图等变，而不是再增加一个 box angle consistency loss。

## 8.6 歧义安全背景学习

对 location \(p\)，用软计划熵定义归属歧义：

\[
a_p=\frac{H(P_p)}{\log(|\mathcal J_p|+1)}.
\]

训练位置分为三类：

1. **Safe background**：位于所有 HBox 支持域之外，完整 HBox 标注下可正常施加 focal negative；
2. **Confident foreground**：某个 \(P_{jp}\) 高且跨视图稳定，按软权重施加分类、框回归和 centerness；
3. **Ambiguous HBox interior**：位于至少一个 HBox 内但归属熵高，初期 ignore，后期最多施加有上限的 soft-negative，绝不立即作为 full-weight background。

对 HBox 内稳定背景，可在 warm-up 后使用

\[
w_p^{bg}=\operatorname{clip}[(1-a_p)P_{0p},0,w_{max}],
\]

逐步恢复背景辨别能力。这样不会把整个 HBox 都永久 ignore，也不会在模型尚未学会方向时过早压低真实物体区域。

## 8.7 保留的 H2RBox-v2 几何监督

OrbitAssign 不重新发明角度学习。对软正样本继续使用：

- HBox envelope/projection regression；
- H2RBox-v2 rotate/flip symmetry loss；
- 现有 angle coder 和周期处理；
- centerness/quality prediction。

ABBSPO 的 adaptive scaling 可以作为独立 `+ABBS` 对照，但不默认并入主方法，否则会同时改变 scale learning 和 assignment 两个因果变量。

## 8.8 总损失

\[
\mathcal L
=\mathcal L_{cls}^{soft/safe}
+\lambda_b\mathcal L_{HBox-reg}^{soft}
+\lambda_c\mathcal L_{ctr}^{soft}
+\lambda_s\mathcal L_{sym}^{H2RBox-v2}
+\lambda_o\mathcal L_{orbit}
+\lambda_m\mathcal L_{coverage}.
\]

`coverage` 只防止实例失去全部 mass，不要求每个实例相同数量的正点。

## 8.9 训练日程

1. **Warm-up**：使用复现合格的 H2RBox-v2 static assignment，使类别和 envelope prediction 具备基本可用性；
2. **Soft transition**：\(P=(1-\rho)P_{static}+\rho P_{orbit}\)，逐步增加 \(\rho\)，保持较高温度；
3. **Orbit stage**：使用共识软计划，缓慢降低温度，但不生成离线 hard pseudo-RBox；
4. **Inference**：只保留原 CNN detector，训练期 warp/transport 全部移除。

## 9. 为什么该方案不是简单的“BGHR + consistency”

这是最需要守住的审稿边界。

BGHR 的中心动作是：从预测中为一个 GT HBox 选择最佳 RBox，再使用该 RBox 挖掘 fine-grained positives。OrbitAssign 的中心动作不同：

- 不先承诺一个唯一 predicted RBox；
- 监督对象是 location–instance relation，而不是单个 box；
- 三视图在 canonical coordinates 上共同产生 assignment target；
- assignment entropy 直接决定负样本是否安全；
- hard selection 是退火后的可能极限，不是训练起点。

必须用以下消融证明差异确实有效：

1. static H2RBox-v2 assignment；
2. BGHR-style single-best-RBox hard mining；
3. single-view soft assignment；
4. cross-view soft assignment，但所有未选位置仍为 hard background；
5. 完整 OrbitAssign。

如果 3→4 没有改善 assignment stability，或者 4→5 只增加 false positives，那么主假设失败，不能靠叠加 ABBS、teacher 或更强 backbone 救故事。

## 10. 实验设计

### 10.1 数据集顺序

| Priority | Dataset | Purpose |
|---:|---|---|
| 1 | DOTA-v1.0 | 主结果；类别、尺度、方向和密度覆盖广，最近 HBox 方法比较充分 |
| 2 | DIOR-R | 第二数据集；检验 tight/coarse HBox 与尺度失配，正面对比 BGHR/ABBSPO |
| 3 | HRSC2016 | 单类细长目标机制集；主要看 angle/aspect 和 assignment 可视化，不单独支撑通用性 |
| 4 | DOTA-v1.5/v2.0 | 通过前两数据集后再扩展；检验极密/微小实例，不作首轮算力黑洞 |

训练时只向模型暴露 HBox。原始 RBox 与训练 dataloader 分离；其用途限于标准 RBox evaluation 和冻结后的 report-only 机制分析。

### 10.2 必需基线

- Full-RBox Rotated FCOS R50-FPN：同 parent 的监督上界；
- H2RBox、H2RBox-v2：直接 parent 和 symmetry baseline；
- BGHR：最重要的 sample-mining competitor；
- ABBSPO：scale/symmetry competitor；
- EIE-Det：box-equivariance competitor；
- Wholly-WOOD HBox-only：若代码/协议可严格复现；
- Rotated RTMDet-R full/HBox native：仅在 portability 阶段报告。

reported-only 结果与 reproduced 结果必须分栏，不能混为一张排名表。

### 10.3 主消融矩阵

| ID | Assignment | Orbit align | Ambiguous BG | Purpose |
|---|---|---:|---:|---|
| A0 | FCOS static center/min-area | ✗ | hard | parent |
| A1 | BGHR-style hard predicted-RBox | ✗ | hard | closest mechanism control |
| A2 | single-view soft transport | ✗ | hard | soft assignment effect |
| A3 | cross-view soft transport | ✓ | hard | assignment equivariance effect |
| A4 | cross-view soft transport | ✓ | ignore only | negative-safety upper tendency |
| A5 | cross-view soft transport | ✓ | annealed soft-negative | full OrbitAssign |

随后才做：rotate only / flip only / both、temperature、warm-up、coverage bound、local softmax vs unbalanced transport、`+ABBS`、RTMDet transfer。

### 10.4 不能只报 AP 的机制指标

1. **assignment flip rate**：同一物理位置 warp 后，实例归属 argmax 改变的比例；
2. **soft-plan JS divergence**：各视图与 canonical plan 的距离；
3. **foreground coverage/purity**：使用 sealed RBox 仅作 report-only 分析；
4. **false-negative gradient rate**：落在真实 RBox 内却受到 background gradient 的 location 比例；
5. **overlap conflict rate**：一个位置同时被多个实例强竞争的比例；
6. **instance starvation rate**：没有获得有效正 mass 的实例比例；
7. **per-level ownership**：小/中/大目标在 P3–P7 的 mass 分布；
8. **pre-NMS duplicate、score–IoU calibration、empty-image FP**；
9. 标准 mAP/AP50/AP75、per-class AP，以及 size/aspect/density/angle 分桶。

### 10.5 预注册资源门

这些是是否继续投入算力的门，不是结果承诺：

- `G0 Reproduction`：H2RBox-v2 与 full Rotated FCOS 在允许误差内复现，HBox-only seal 通过；
- `G1 Source`：A0 确实存在可观的 HBox interior false-negative 或 cross-view assignment instability，且集中于细长/密集/小目标；若没有，停止方法；
- `G2 Action`：A3 相比 A2 明显降低 assignment flip/JS，A5 相比 A3 降低 false-negative gradient，且 instance starvation/duplicates 不恶化；
- `G3 Endpoint screen`：完整 DOTA-v1.0 同口径至少提升 `0.5 AP point`，否则不迁移 RTMDet；
- `G4 Paper`：相对最强 reproduced HBox baseline，三 seed 均值约 `+1.0 AP point`，或关闭至少 `20%` 的 full-supervision gap，并在 DIOR-R 重现；
- `Stop`：只在 post-NMS 有改善、机制指标不动、依赖 RBox 选超参、关键类退化、重复框显著增加，立即关闭当前配方。

## 11. 预期失败模式与修复边界

| Failure | Diagnosis | Allowed response | Forbidden rescue |
|---|---|---|---|
| 软计划全部塌到中心 | coverage/center cost 过强 | 降 center prior；检查 view consensus | 加新 backbone 掩盖问题 |
| 软计划全部塌到背景 | warm-up 不足或 bg supply 过强 | 延长固定 warm-up；加入最小 coverage | 用 GT RBox seed |
| HBox 内 ignore 导致重复框 | negative safety 过宽 | 启用 capped stable-background weight | 靠更激进 NMS 宣称方法有效 |
| 跨视图对齐无收益 | 静态 assignment 已近似等变，或 warp 太粗 | 关闭 OrbitAssign；保留负结果 | 换一组旋转角度追 best result |
| 只在 HRSC 有效 | 单类细长特例 | 降级为机制观察 | 宣称通用遥感框架 |
| `+ABBS` 才有收益 | 尺度而非 assignment 是主因 | 重新定义为 scale study | 把 ABBS 收益归给 OrbitAssign |
| RTMDet 无法迁移 | 方法依赖 FCOS target geometry | 如实限定 parent | 同时改 head/neck/assigner 继续堆叠 |

## 12. 论文主张应如何写

### 可以写

- “We study the equivariance of dense instance-assignment fields under HBox supervision.”
- “OrbitAssign builds a transformation-aligned soft assignment target and uses its ambiguity to regulate background gradients.”
- “The method is training-only and keeps the underlying CNN detector and inference pipeline unchanged.”
- “In our date-bounded search, we did not find an exact formulation combining these elements.”

### 不能写

- first HBox-supervised oriented detector；
- first dynamic/soft sample assignment；
- first rotation-equivariant oriented detector；
- first unified point/HBox/RBox framework；
- end-to-end NMS-free detector；
- fixed-query detector；
- 已经接近/超过全监督，除非真实实验支持。

### 推荐题目

**OrbitAssign: Equivariant Soft Instance Assignment for Horizontal-Box-Supervised Oriented Object Detection**

更保守的中文题目：

**面向水平框监督旋转目标检测的跨视图软实例分配**

## 13. 推荐执行顺序

### Phase 0：只做离线 source audit

- 在复现合格的 H2RBox-v2 checkpoint 上导出三视图 location targets/predictions；
- 测 static assignment flip、HBox interior background gradient、overlap conflict 和 instance starvation；
- 用 RBox 只做 report-only 诊断，不进入 assignment 或训练。

若问题幅度很小，直接停止，不写模块。

### Phase 1：最小 action

- 只实现 canonical warp + single-view/cross-view soft assignment；
- 保留原 background contract，先验证 A2→A3；
- 不加 ABBS、teacher、RTMDet 或新 backbone。

### Phase 2：negative-safe

- 加入三分区 background weighting；
- 重点监控 duplicate、empty FP 和 pre-NMS score；
- 完成 A3/A4/A5 因果消融。

### Phase 3：完整证据

- DOTA-v1.0 三 seed；
- DIOR-R 迁移；
- strongest reproduced baseline；
- tight/coarse HBox、density/aspect/angle 分桶；
- 训练和推理开销。

### Phase 4：架构迁移

- 只把已通过的最小组合移植到 Rotated RTMDet；
- 若 FCOS 与 RTMDet 都成立，才支持“detector-agnostic training assignment”表述；
- 点监督或 partial weak supervision另立后续项目，不能塞进首篇。

## 14. 最终判断

我的明确判断是：**CNN base 是更稳健的新版本，fixed-query 经验应保留，但必须降级为设计原则。**

这三个月真正可迁移的不是“固定查询一定优于 dense detector”，而是：

1. 检测错误常在训练监督和 raw logits 中已经形成；
2. ownership/negative contract 决定了后续几何和分类有没有机会学对；
3. 未匹配或未选中不等于真实背景；
4. 旋转视图的价值不仅是约束角度，还能检验监督归属是否稳定；
5. 方法必须经 source→action→endpoint→protocol 四层门，而不是靠最好 checkpoint 和后处理讲故事。

OrbitAssign 把这五条经验压缩成一个 CNN 问题：

> **在 HBox 不足以确定真实前景支持时，先让多视图对“哪些位置属于哪个实例”形成软共识，再决定哪些位置可以安全地成为负样本。**

它比上一版更简单、更贴近现有代码、更容易和 H2RBox-v2/BGHR/ABBSPO 公平比较，也更符合一篇独立副产品论文应该具备的可证伪性。

在用户确认该设计前，本文件不授权实现、数据转换或训练启动。

## 15. 文献与本地材料索引

### 聚焦文献包

- `docs/literature-search-20260812-cnn-hbox-oriented-assignment/papers.md`
- `docs/literature-search-20260812-cnn-hbox-oriented-assignment/papers.csv`
- `docs/literature-search-20260812-cnn-hbox-oriented-assignment/search-notes.md`

### 完整三个月旧版复盘

- `docs/project_history/exp_20260812_three_month_review/fres_20260512_20260812_openrsd_to_ovcapflow_review_and_detector_proposal_zh.md`

### 本地 CNN 实现入口

- `/data1/zcy/OpenRSD/mmrotate/models/detectors/h2rbox_v2.py`
- `/data1/zcy/OpenRSD/mmrotate/models/dense_heads/h2rbox_v2_head.py`
- `/data1/zcy/OpenRSD/mmrotate_configs/h2rbox/`
- `/data1/zcy/OpenRSD/mmrotate/models/dense_heads/rotated_rtmdet_head.py`
- `/data1/zcy/OpenRSD/mmyolo_configs/rtmdet/rotated/`
