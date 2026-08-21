# OpenRSD → OV-CapFlow 经验复盘与 CNN-HBox 旋转检测方案（v3：融合 FMSA-Point）

> 时间窗：2026-05-12 至 2026-08-12  
> 版本日期：2026-08-12  
> 状态：研究设计版；未实现、未训练、未产生方法增益  
> 版本关系：保留 v2；本文件是融合 `local:FMSA_Point_OpenDraft_Assisted_Draft_v2.md` 后的推荐 v3  
> 监督边界：首篇只研究完整 HBox 标注下的 HBox→RBox；点监督、部分弱监督与全监督统一不进入主线

## 0. 最终结论

v3 不把 FMSA-Point 整套迁移到 HBox，也不改回 point-supervised 主线。推荐方法仍以 CNN 稠密检测器和 OrbitAssign 为中心，但把 v2 的“一个 assignment entropy 处理所有不确定性”升级为：

> **OrbitAssign-ER：跨视图软实例分配 + 因子化证据路由。**

核心判断是：HBox 监督下至少存在三种不同的不确定性，不能压成一个伪框质量分数 $Q$：

1. **Ownership uncertainty**：一个 FPN location 应由哪个 annotation instance 负责；
2. **Geometry uncertainty**：该位置给出的方向和旋转几何是否在跨视图下稳定；
3. **Background uncertainty**：该位置是否有足够证据被训练为背景。

v3 对这三种证据分别建模，并只把它们送到能够被其支撑的损失：ownership 决定正样本归属，方向稳定性只调节几何监督，background safety 只调节负梯度。FMSA 草稿中最有价值的“self-awareness”因此被保留为**证据分解与路由原则**；MessDet 主干、点到区域 warmup、边界分割、类别原型、硬伪 OBB 和 teacher–student 不进入首篇核心。

推荐的一句话论文问题是：

> **When HBoxes cannot determine exact foreground support, a dense detector should not use one confidence score to decide everything; it should separately estimate who owns a location, whether its orientation evidence is stable, and whether treating it as background is safe.**

## 1. v3 相对 v2 的实质变化

| Change | v2 | v3 | Purpose |
|---|---|---|---|
| uncertainty model | 主要依赖 soft-plan entropy | ownership / geometry / background 三因子 | 避免一个不确定性替代另一个不确定性 |
| FMSA integration | 无 | 吸收方向置信、密度和多证据思想 | 保留灵感，不搬运整套 pipeline |
| Fourier role | 无 | 二阶轴向圆周矩，仅衡量跨视图方向稳定性 | 不生成硬角度伪标签，不与 FAA 主张碰撞 |
| density handling | 全局或局部 transport 概念 | annotation-only conflict graph 上的局部 transport | 把计算和竞争集中在重叠实例 |
| quality usage | assignment entropy 控制背景 | evidence-specific loss routing | 避免 FMSA 式任意线性总分 $Q$ |
| geometry in assignment | 预测 HBox envelope 直接进入 cost | envelope 延迟启用；方向稳定性默认不决定 ownership | 避免 early confirmation 与循环自证 |
| hard-instance protection | coverage lower bound | coverage lower bound + 不做阈值删除 | 吸收 fixed-query 中 starvation 经验 |
| main parent | H2RBox-v2 / Rotated FCOS | 不变 | 保持 CNN base 与因果可比性 |
| inference | standard rotated NMS | 不变 | 所有新机制仅训练期存在 |

v3 仍然只有两个论文级动作：

1. **Conflict-local Orbit Assignment**：在 HBox 冲突图内形成跨视图对齐的软 location–instance assignment；
2. **Factorized Evidence Routing**：把 ownership、orientation stability 和 background safety 分别路由到相应损失。

Fourier/circular moment 是第二个动作中的无参数估计器，不单独包装成第三个模块。

## 2. FMSA-Point 草稿的审读结论

### 2.1 草稿提供了什么真正有用的灵感

FMSA-Point 的核心直觉不是“必须使用 MessDet”或“必须做 teacher–student”，而是：弱监督生成的方向、尺度、边界和实例支持具有不同可靠性，网络应显式感知这些可靠性。这个直觉与三个月实验的以下经验一致：

- source existence 不等于 actionability；某个信号存在，不代表它能安全地产生监督；
- 一次 hard selection 会把早期偏差变成后续训练事实；
- query/location starvation 不能用简单阈值过滤解决；
- 几何、类别、存在性和背景状态不能共享一个未经校准的置信度；
- 多视图不仅能约束输出，还能测量监督关系是否稳定。

因此，FMSA 对 v3 的主要贡献是促使我们把 v2 的 assignment ambiguity 进一步拆开，而不是增加更多特征模块。

### 2.2 逐项迁移矩阵

| FMSA element | Decision | v3 translation | Reason |
|---|---|---|---|
| point-to-region warmup | Reject | 直接使用已知 HBox candidate support | HBox 已提供区域约束；额外分割 warmup 会引入新伪监督和 parent confound |
| MessDet backbone/neck | Defer | 作为 Phase-4 architecture oracle | 主线需保持 H2RBox-v2/FCOS parent；MessDet 本身改变 backbone、neck、head 和预训练 |
| group-axis Fourier angle | Adapt | 二阶轴向圆周矩估计跨视图方向集中度 | 保留频域/周期统计思想，但不把未验证相位直接当伪角度 |
| scale self-awareness | Diagnostic | assignment support covariance / per-level mass consistency | HBox 已给粗尺度；真实 OBB extent 仍未知，先测量而不新增 scale branch |
| boundary self-awareness | Defer | report-only edge/support alignment | 边缘易受纹理、阴影和相邻目标干扰，且会与已有边界/方向增强工作混淆 |
| density self-awareness | Adapt | HBox conflict graph + component-wise transport | 不假设最近点距离等于物体半径；只使用可验证的支持域重叠关系 |
| scalar uncertainty $Q$ | Reject as-is | factorized evidence routing | 异质置信度线性相加没有因果含义，也无法说明每个分数应影响哪项损失 |
| class prototype memory | Reject | 使用 annotation class | 完整 HBox 已知类别；prototype 可能放大 OpenRSD 已观察到的类别偏置 |
| hard pseudo OBB | Reject | 始终保留 soft relation target | 避免早期 angle/extent 错误自举 |
| quality threshold filtering | Reject | continuous weights + minimum coverage | `Q < threshold` 会优先删除困难、小型和密集实例，重现 query starvation |
| EMA teacher–student | Control only | 作为 hard-pseudo baseline | PWOOD/SPWOOD/Point2RBox-v3 已使该路线拥挤，且会弱化 assignment 因果故事 |
| point jitter stress test | Replace | HBox looseness、center noise、annotation perturbation | 监督协议不同，不能直接沿用 point-only 结论 |

### 2.3 必须纠正的技术点

FMSA 草稿是带占位结果的灵感稿，以下内容不能直接进入方法或论文事实：

1. **MessDet 的 orientation dimension**：MessDet 论文实验将 $N$ 设为 8，不是草稿中的“通常 $K=6$”。
2. **FAA 与草稿 Fourier 不是同一操作**：Fourier Angle Alignment 在局部二维空间特征上做 2D DFT、极坐标能量累积和主方向对齐；草稿是在 MessDet orientation-group axis 上做 1D DFT。后者需要独立证明，不能写成对 FAA 的直接采用。
3. **轴向角的谐波阶数**：OBB 方向满足 $\theta\equiv\theta+\pi$。若使用复圆周矩，应使用 $e^{i2\theta}$ 的二阶轴向表示；“对一阶谐波相位直接除以 2”只有在信号表示和群作用严格匹配时才成立，不能默认。
4. **Fourier amplitude 不是正确率**：高集中度只说明各视图方向一致，系统性一致错误仍可能获得高分。v3 将其称为 stability/reliability evidence，不称为 calibrated correctness。
5. **batch-wise max normalization 不稳定**：用批内最大振幅归一化会使同一实例的分数依赖 batch composition；v3 使用有界圆周集中度 $[0,1]$。
6. **density 等式不成立**：最近中心距离通常不等于物体长边的一半，稀疏场景尤其不成立；v3 只用 HBox 支持域重叠构图。
7. **boundary IoU 需重新定义**：区域 mask 与一像素 contour 直接求 IoU 不具有稳定几何含义；若后续启用，必须定义 tolerance band 或 distance-transform metric。
8. **H2RBox-v2 不是标准 EMA teacher–student 引用**：其核心是弱/自监督分支及 rotate/flip symmetry，不能用它为通用 teacher–student 过程背书。

这些纠正并不否定 FMSA 灵感，而是把它从“多模块概念稿”改造成可检验的 CNN-HBox 机制。

## 3. 三个月经验在 v3 中如何收敛

| Experience | Failure pattern | v3 consequence |
|---|---|---|
| OpenRSD false-SV | dense logits 已形成错误，top-k/NMS 继续放大 | 修改训练 target/gradient，不把后处理当方法 |
| matched-control lessons | BN、batch、parent 和 mouth 可制造假增益 | parent、schedule、augmentation、inference 全部冻结 |
| fixed-query ownership | birth→ownership→null→ranking 是连续链 | 在 CNN 中显式建模 reachability、ownership、background safety |
| Q/capacity lessons | 困难实例会被查询或候选预算饿死 | minimum mass coverage；禁止阈值删除实例 |
| source/action/endpoint gates | source 存在不保证 endpoint 改善 | 先证实 false-background 和 assignment instability，再实现 action |
| RISC/POQ/OMQ closures | source、endpoint、protocol leakage 必须分离 | RBox 只用于冻结后的 report-only mechanism audit |
| FMSA self-awareness | 弱监督几何证据可靠性不同 | 不再用单一 entropy 或 scalar $Q$ 决定全部监督 |

fixed-query 仍只提供问题语言：

| fixed-query concept | CNN-HBox object | v3 handling |
|---|---|---|
| query birth | FPN reachability / candidate support | per-level support + center seed + coverage audit |
| ownership | location–instance relation | conflict-local soft transport |
| unmatched false background | HBox interior unselected location | factorized background safety |
| query identity | annotation identity across views | canonical warp by annotation ID |
| null calibration | background probability | separate $q^{bg}$, not class/geometry score |
| query capacity | overlap/FPN positive budget | component-wise capacity + starvation metric |
| raw-score amplification | pre-NMS dense errors | pre/post-NMS simultaneous reporting |

## 4. 文献边界更新

### 4.1 FMSA 相关文献带来的新约束

| Work | Verified core | Claim boundary for v3 |
|---|---|---|
| [MessDet, ICCV 2025](https://arxiv.org/abs/2507.09896) | E2CNN-based rotation-equivariant CSPNeXt/PAFPN、strict/approximate downsampling、RE channel attention、orientation-group multi-branch head；实验 $N=8$ | 不能声称首个 orientation-group CNN、严格旋转等变或 multi-branch group head |
| [Fourier Angle Alignment, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Gu_Fourier_Angle_Alignment_for_Oriented_Object_Detection_in_Remote_Sensing_CVPR_2026_paper.html) | 对局部空间 feature 做 2D Fourier angle estimation，并在 FPN/head 中对齐方向 | 不能声称首个 Fourier orientation estimation/alignment；v3 只用轴向集中度做训练可靠性证据 |
| [PointOBB-v3](https://arxiv.org/abs/2501.13898) | point supervision 的 multi-view scale/angle learning、SSFF、end-to-end branch、instance-aware weighting | 不能把 multi-view scale/angle 或 instance-aware weight 作为宽创新 |
| [Point2RBox-v2, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_Point2RBox-v2_Rethinking_Point-supervised_Oriented_Object_Detection_with_Spatial_Layout_Among_CVPR_2025_paper.html) | Gaussian overlap、Voronoi watershed、transform consistency | 不能声称首个 density/layout-aware point-to-RBox；v3 的 graph 只服务 HBox ownership competition |
| [Point2RBox-v3](https://arxiv.org/abs/2509.26281) | progressive label assignment、动态 mask prior 和 pseudo-label refinement/utilization | hard/dynamic pseudo-label assignment 已非常拥挤，不应成为 v3 主线 |
| [Instance-Level Orientation Enhancement, TIP 2025](https://doi.org/10.1109/TIP.2025.3632224) | 已核验题名、期刊、DOI；属于 HBox-supervised instance-level orientation enhancement | 正式投稿前必须取得全文并做机制级碰撞审计；在此之前不声称 orientation enhancement 空白 |

### 4.2 原 v2 的直接近邻仍然成立

- H2RBox/H2RBox-v2：rotate/flip box/angle symmetry 已占据；
- BGHR：HBox 下 predicted-RBox sample mining 已占据，是最强机制对照；
- ABBSPO：HBox scale correction 与 symmetric prior 已占据；
- EIE-Det：box equivariance 已占据；
- Wholly-WOOD/PWOOD/SPWOOD：mixed、partial、sparse weak supervision 与 pseudo-label filtering 已占据；
- OTA/IQDet/ATSS/DCFL/RTMDet：soft、global、dynamic assignment 已占据。

### 4.3 v3 可守住的窄问题

截至 2026-08-12 的定向公开检索，尚未发现精确同构于以下联合命题的工作：

> 在完整 HBox 监督的 CNN 稠密旋转检测中，将 **location–instance assignment equivariance** 与 **relation-specific evidence routing** 作为共同中心：ownership evidence 决定实例归属，axial orientation stability 决定几何监督权重，background safety 决定 HBox 内负梯度。

这只是 `date-bounded search finding`，不是 `first` 结论。尤其需要继续审计 TIP 2025 的 Instance-Level Orientation Enhancement 全文及其前向/后向引用。

## 5. 问题形式化：三个潜变量，而不是一个伪框

训练集为

\[
\mathcal D=\{(I_i,\{H_{ij},y_{ij},id_{ij}\}_{j=1}^{n_i})\},
\]

其中 $H_j=(c_x,c_y,W,H)$ 是 HBox，$y_j$ 是已知类别，$id_j$ 在 original/rotate/flip views 中保持一致。CNN 在 FPN location $p$ 输出类别 $s_p$、旋转框 $b_p=(x,y,w,h,\theta)$ 和 localization quality $u_p$。

v3 显式区分三个隐变量：

\[
Z_{pj}:\text{ location }p\text{ belongs to instance }j,
\]

\[
R^{\theta}_{pj}:\text{ its orientation evidence is stable},
\]

\[
B_p:\text{ training }p\text{ as background is safe}.
\]

它们之间相关，但不等价：

- 一个位置可以明确属于某实例，但方向因物体近圆形而不可靠；
- 一个位置可以给出稳定方向，但处在两个同类 HBox 的 ownership 冲突区；
- 一个位置的方向不可靠，不代表它是真背景；
- 跨视图预测一致也不代表方向正确，只代表当前模型稳定。

因此 v3 不学习统一 $Q_{pj}$，而学习或计算 $P_{pj}$、$q^{own}_{pj}$、$q^{\theta}_{pj}$ 和 $q^{bg}_{p}$。

## 6. OrbitAssign-ER 方法设计

### 6.1 CNN parent 与训练视图

主模型保持：

- ResNet-50 + FPN；
- Rotated FCOS / H2RBox-v2 head；
- original、random rotation、flip 三视图；
- H2RBox-v2 的 HBox envelope regression、rotation/flip symmetry、angle coder 和 centerness；
- standard rotated NMS inference。

不引入 Transformer decoder、fixed queries、Hungarian matching、SAM、mask annotation、offline pseudo OBB 或新推理分支。

### 6.2 Candidate support 与 HBox conflict graph

对实例 $j$ 和视图 $g$，候选集 $\mathcal C_j^g$ 由以下条件给出：

1. location 位于已知变换 $T_g$ 后的 HBox support；
2. 满足 FCOS regression range；
3. 保留最小 center seed，防止训练早期 starvation；
4. 不使用 GT RBox、mask 或外部 proposal。

在 canonical view 上构造冲突图

\[
\mathcal G_H=(\mathcal V,\mathcal E),\qquad
(j,k)\in\mathcal E\iff \mathcal C_j\cap\mathcal C_k\neq\varnothing.
\]

只在同一 connected component 内执行跨实例竞争：

- isolated HBox：只需实例内 soft support，不制造无意义的全图竞争；
- overlapping HBoxes：使用 location capacity 和 instance coverage 解 ownership；
- component degree 直接形成 density bucket，不假设中心距离等于目标尺度。

该图完全来自 HBox annotation 与 FCOS support，训练 dataloader 不接触 RBox。

### 6.3 Per-view base cost

对候选 pair $(j,p)$，先定义

\[
C_{jp}^{g,base}
=\lambda_{cls}C_{cls}
+\lambda_{ctr}C_{ctr}
+\lambda_q C_q,
\]

其中 $C_{cls}$ 是已知 annotation class 的分类 cost，$C_{ctr}$ 是弱中心先验，$C_q=1-u_p$ 是 localization quality。所有 prediction-derived cost 在形成 target 时 `stop-gradient`。

HBox envelope 是由真实 HBox 提供的有效弱约束，但早期 predicted RBox 尚不稳定，因此只做延迟启用：

\[
C_{jp}^{g}
=C_{jp}^{g,base}
+\eta(t)\,\lambda_{env}C_{env},
\]

其中 $\eta(t)$ 从 0 平滑增加。主版本**不让** $q^{\theta}$ 反向决定 assignment cost：方向稳定性由当前预测产生，若再用它选择 ownership，容易形成“稳定预测获得更多监督、更多监督使其更稳定”的循环自证。$q^{\theta}$ 只在 assignment plan 已形成后路由方向相关损失。

将 $q^{\theta}$ 乘入 $C_{env}$ 仅作为 `theta-in-assignment` 消融，用于验证这种额外耦合是否有益；它不是推荐配置。

### 6.4 Canonical alignment 与 conflict-local soft plan

把每个视图的 cost、valid mask 和 location coordinates 经 $T_g^{-1}$ 映射回 canonical coordinates：

\[
\widetilde C^g=\mathcal W_{g\rightarrow e}(C^g),\qquad
\bar C=\frac{\sum_g M^g\odot\widetilde C^g}{\sum_g M^g+\epsilon}.
\]

逐 FPN level 使用双线性 gather/splat；跨层迁移不是默认动作。对每个 conflict component $c$ 求熵正则、轻量 unbalanced plan：

\[
P_c=\arg\min_{P\ge0}
\langle P,\bar C_c\rangle
+\varepsilon\sum_{jp}P_{jp}(\log P_{jp}-1),
\]

满足：

- 每个 location 的 foreground mass 不超过 1；
- 每个 instance 有宽松 minimum coverage；
- background 吸收剩余 mass；
- 不要求一个实例只有一个 location；
- 不删除低质量 instance。

将 canonical plan 推回各视图，约束 per-view posterior $Q^g$：

\[
\mathcal L_{orbit}
=\frac1{|\mathcal G|}\sum_gD_{JS}(Q^g\Vert\mathcal W_{e\rightarrow g}(P)).
\]

### 6.5 FMSA-inspired axial spectral reliability

对同一物理 pair $(j,p)$，先把每个视图预测方向经逆群作用映射到 canonical frame。所有方向先经 parent angle coder 解码并统一到相同 long-edge convention；近方形或可交换宽高的预测不能靠裸角数值直接比较。对旋转视图减去已知角度；对 reflection 不手写易错符号，而对解码后的主轴向量施加相应逆矩阵，再恢复轴向角。

由于 OBB 是 $\pi$-periodic axis，定义二阶圆周 Fourier coefficient：

\[
z_{jp}^{(2)}
=\frac{\sum_{g\in\mathcal V_{jp}}\omega_{jp}^{g}
\exp(i2\widetilde\theta_{jp}^{g})}
{\sum_{g\in\mathcal V_{jp}}\omega_{jp}^{g}+\epsilon},
\]

\[
q_{jp}^{\theta}=|z_{jp}^{(2)}|\in[0,1],\qquad
\bar\theta_{jp}=\tfrac12\arg z_{jp}^{(2)}.
\]

使用规则：

- $q^{\theta}$ 是跨视图轴向集中度，不是 accuracy probability；
- $\bar\theta$ 只用于 diagnostic visualization，不作为 hard pseudo angle；
- $\omega^g$ 由 valid mask 与 stop-gradient quality 给出，不做 batch-max normalization；
- $q^{\theta}$ 在参与 loss routing 时 `stop-gradient`，不能通过主动改变权重获得平凡解；
- 低 $q^{\theta}$ 只降低 angle/symmetry 等方向相关监督，不能削弱 HBox envelope 约束或把位置变成背景；
- 高 $q^{\theta}$ 但 report-only angle error 大的样本记为 `wrong-consistent`，用于校准审计。

该计算不要求 MessDet。默认直接使用 H2RBox-v2 angle head 的三视图预测，因此无额外参数。MessDet orientation-group harmonic 仅作为后续 feature-evidence 对照，并且必须先用合成旋转单元测试确定 harmonic order 与 phase law，不能沿用草稿公式。

### 6.6 Factorized evidence routing

定义 normalized ownership entropy：

\[
h_p=\frac{H(P_{\cdot p})}{\log(|\mathcal J_p|+1)},
\]

以及 per-pair view stability：

\[
v_{jp}=\exp[-D_{JS}(Q_{jp}^{views},P_{jp})/\tau_v].
\]

对 background 同样定义 $v_{0p}$。令 $\underline P_{0p}=\min_g Q_{0p}^g$，只有所有有效视图都倾向 background 时，HBox 内部才可能恢复负梯度。三类权重分别为：

\[
q_{jp}^{own}=(1-h_p)v_{jp},
\]

\[
r_{jp}^{\theta}(t)
=(1-\rho(t))+\rho(t)\left[q_{\min}+(1-q_{\min})\operatorname{sg}(q_{jp}^{\theta})\right],
\]

\[
q_{jp}^{geom}=q_{jp}^{own}r_{jp}^{\theta}(t),
\]

其中 $\rho(t)$ 在 angle warm-up 后从 0 增长，$q_{\min}>0$ 防止低方向性实例永远失去角度学习机会。令 $\mathcal S_{safe}$ 为所有有效 HBox support 之外且不属于 ignore、truncated 或 crop-uncertain 区域的位置，则

\[
q_p^{bg}=\begin{cases}
1,&p\in\mathcal S_{safe},\\
\beta(t)\operatorname{clip}[\underline P_{0p}(1-h_p)v_{0p},0,w_{max}],&\text{otherwise},
\end{cases}
\]

其中 $\beta(t)$ 从 0 缓慢增加，且 $w_{max}<1$。因此 HBox interior background 必须同时满足跨视图一致、低 ownership entropy 和保守 background lower bound，不能由单视图高背景分数直接自证。所有 routing weights 在反向传播时视为 `stop-gradient`。

损失路由遵循：

| Loss target | Weight | Evidence meaning |
|---|---|---|
| positive classification | $P_{jp}q_{jp}^{own}$ | 谁负责该位置、跨视图是否稳定 |
| HBox envelope regression | $P_{jp}q_{jp}^{own}$ after warm-up | HBox 是有效弱几何约束，不因低方向性而删除 |
| angle / rotate / flip symmetry | $P_{jp}q_{jp}^{geom}$ | 方向稳定性只监督几何 |
| centerness / localization quality | $P_{jp}q_{jp}^{own}$ | 不因近圆形目标的低方向性而完全饿死 |
| background classification | $q_p^{bg}$ | 只有背景关系足够确定时施加负梯度 |
| instance coverage | independent lower bound | 防止困难实例因低置信被删除 |

这张表是 v3 相对 FMSA scalar $Q$ 和 v2 single entropy 的关键改进。

### 6.7 Support moments：吸收 scale/boundary 灵感但先不训练

对每个实例的 soft support 计算二阶矩：

\[
\mu_j=\frac{\sum_pP_{jp}x_p}{\sum_pP_{jp}},\qquad
\Sigma_j=\frac{\sum_pP_{jp}(x_p-\mu_j)(x_p-\mu_j)^\top}{\sum_pP_{jp}}.
\]

$\Sigma_j$ 的特征值、主轴和跨视图变换一致性用于报告：

- support scale consistency；
- aspect-ratio bucket；
- dense-overlap 下 support collapse；
- assignment mass 是否只塌在中心。

第一版不把 $\Sigma_j$ 直接回归成 pseudo OBB，也不加入 Sobel/Gabor boundary loss。只有当 G1 source audit 表明 assignment 边界而非 ownership 是主要残余误差时，才允许单独开启 boundary side study。

### 6.8 总损失

\[
\mathcal L=
\mathcal L_{cls}^{routed}
+\lambda_b\mathcal L_{HBox-env}^{routed}
+\lambda_c\mathcal L_{ctr}^{routed}
+\lambda_s\mathcal L_{sym}^{routed}
+\lambda_o\mathcal L_{orbit}
+\lambda_m\mathcal L_{coverage}.
\]

没有 pseudo-box regression loss、mask loss、prototype loss 或 EMA-teacher loss。

### 6.9 训练日程

1. **Static warm-up**：复现合格的 H2RBox-v2 assignment；仅收集三视图 angle/assignment statistics；
2. **Soft ownership transition**：从 static plan 平滑过渡到 conflict-local soft plan，envelope cost ramp 仍接近 0；
3. **Evidence routing stage**：启用 $q^{own}$、$q^{geom}$、$q^{bg}$，缓慢增加 $\eta(t)$；
4. **Stable stage**：保持 soft plan，不离线导出 hard pseudo OBB；
5. **Inference**：只保留原 CNN detector 与 rotated NMS。

## 7. 三条融合路线的取舍

| Rank | Route | Definition | Judgment |
|---:|---|---|---|
| 1 | Minimal OrbitAssign-ER | FCOS parent + conflict-local assignment + factorized routing + prediction-level axial moment | **推荐主线**；变量最少、与 H2RBox-v2/BGHR 可公平比较 |
| 2 | MessDet spectral sidecar | 把 orientation-group harmonic 作为 $q^{\theta}$ 的第二估计器 | 第二阶段；可验证 feature evidence，但会更换 backbone/neck/head 和预训练 |
| 3 | FMSA-HBox teacher pipeline | warmup mask + multi-cue $Q$ + hard pseudo OBB + EMA teacher | 不推荐；模块过多、确认偏差高、与 Point/PWOOD 系列过近 |

如果 Route 1 失败，不能直接用 Route 3 堆模块“救 AP”。只能依据机制指标判断：

- assignment equivariance 无效：关闭 OrbitAssign；
- ownership 有效但 direction gate 无效：退回 v2-style assignment + factorized BG；
- only MessDet 有效：重新定义为 rotation-equivariant feature study，不能归功于 HBox assignment；
- only teacher 有效：另立 pseudo-label paper，不与本方案混写。

## 8. 实验设计

### 8.1 数据集与顺序

| Priority | Dataset | Role |
|---:|---|---|
| 1 | DOTA-v1.0 | 主结果、完整机制与三 seed |
| 2 | DIOR-R | scale/coarse-HBox 与 direct HBox baseline 迁移 |
| 3 | HRSC2016 | 单类细长目标、方向可靠性和 failure analysis |
| 4 | DOTA-v1.5/v2.0 | 通过前两数据集后再做 dense/tiny stress test |

训练仅暴露 HBox。RBox 只用于标准 evaluation 与冻结后的 report-only diagnostics；不得用于阈值、schedule、harmonic order 或 checkpoint 选择。

### 8.2 Baseline 分层

**Direct HBox baselines**：

- H2RBox、H2RBox-v2；
- BGHR；
- ABBSPO；
- EIE-Det；
- Instance-Level Orientation Enhancement（全文取得后决定 reproduced/reported-only）；
- Wholly-WOOD HBox-only（协议可匹配时）。

**Full-supervision upper bound**：

- same-parent Rotated FCOS R50-FPN；
- MessDet/FAA 只作 architecture/orientation evidence oracle，不与 HBox 方法混排监督等级。

**Adjacent controls**：

- PointOBB-v3、Point2RBox-v2/v3 只用于 related-work boundary，不作为同监督公平排名；
- PWOOD/SPWOOD 只在 partial/sparse protocol appendix 中讨论。

reported-only 与 reproduced results 必须分列。

### 8.3 主因果消融

| ID | Assignment | Evidence routing | Axial stability | Background | Purpose |
|---|---|---|---:|---|---|
| A0 | static FCOS/H2RBox-v2 | none | ✗ | hard | parent |
| A1 | BGHR-style hard predicted RBox | none | ✗ | hard | closest hard-mining control |
| A2 | single-view soft | none | ✗ | hard | soft assignment effect |
| A3 | cross-view soft | none | ✗ | hard | assignment equivariance effect |
| A4 | cross-view soft | factorized ownership/background | ✗ | routed | factorization without Fourier evidence |
| A5 | cross-view soft | scalar FMSA-style $Q$ | ✓ | scalar-gated | proves whether factorization matters |
| A6 | cross-view soft | factorized | ✓ | routed | full OrbitAssign-ER |

独立控制：

- `A6-theta-in-assignment`：额外用 $q^{\theta}$ 门控 $C_{env}$，检验是否引入循环确认偏差；
- `A6-hard-threshold`：按 scalar threshold 删除低质量实例，验证 starvation 风险；
- `A6-global-OT` vs `A6-conflict-local`：验证局部图主要降低计算而不偷换算法；
- rotate only / flip only / both；
- $e^{i\theta}$ vs $e^{i2\theta}$：前者是周期错误的 sanity control；
- prediction-level moment vs MessDet group moment：只在 Phase 4；
- teacher–student hard pseudo-RBox：作为相邻路线 control，不进入主方法。

### 8.4 机制指标

除 mAP/AP50/AP75、per-class AP、size/aspect/density/angle buckets 外，必须报告：

1. `assignment_flip_rate`；
2. `assignment_JS`；
3. `false_negative_gradient_rate`；
4. `overlap_conflict_rate`；
5. `instance_starvation_rate`；
6. `per_level_mass`；
7. `support_moment_equivariance_error`；
8. `axial_concentration` 与 angle error 的 rank correlation；
9. `selective_angle_MAE` / risk–coverage curve；
10. `wrong_consistent_rate`：高 $q^{\theta}$ 但大 angle error；
11. `background_gradient_mass` in true RBox / HBox-only band；
12. pre-NMS duplicate、score–IoU calibration、empty-image FP；
13. conflict-component size、transport time 和 peak memory。

第 8–11 项中的 GT RBox 只在冻结配方后离线计算，不反向进入训练或选择超参数。

### 8.5 FMSA 机制的必要 sanity tests

在训练前完成不需真实 RBox 的单元级验证：

- 旋转 $\phi$ 后，canonical axial moment 恢复到同一方向；
- horizontal/vertical reflection 的 inverse action 正确；
- $\theta$ 与 $\theta+\pi$ 得到相同 $e^{i2\theta}$；
- uniform/isotropic responses 给出低 concentration；
- batch composition 改变不影响单实例 concentration；
- invalid/missing view 不产生 NaN 或虚假高置信；
- low $q^{\theta}$ 不会增加 background weight；
- minimum coverage 在所有 conflict components 中成立。

## 9. 预注册 gates 与关闭条件

| Gate | Required evidence | Close condition |
|---|---|---|
| G0 Reproduction | H2RBox-v2、same-parent full FCOS 与 protocol seal 通过 | parent 或评测未复现，不实现新方法 |
| G1 Source | static assignment 在细长/重叠/小目标上存在 instability 或 false-negative gradient | 问题幅度很小，关闭 OrbitAssign |
| G1.5 Spectral sanity | axial moment 变换律通过；与 report-only angle error 有基本 risk ranking | concentration 与误差无关或 wrong-consistent 过高，删除 Fourier gate |
| G2 Action | A3 相对 A2 降低 flip/JS；A4 降低 false-negative gradient 且不显著增 FP | 机制指标不动，停止 endpoint 扩展 |
| G2.5 Fusion | A6 相对 A4 改善 geometry diagnostics；A6 优于 scalar A5 | Fourier evidence 无独立价值，保留 A4 作为简化版 |
| G3 Endpoint | full DOTA-v1.0 至少有可复现的约 `+0.5 AP point` screen gain | 不迁移 MessDet/RTMDet |
| G4 Paper | 相对最强 reproduced HBox baseline 三 seed 约 `+1.0 AP point`，或有清晰 gap closure，并在 DIOR-R 重现 | 降级为机制/负结果报告 |

所有数值是资源决策门，不是结果预测或承诺。

立即关闭当前配方的条件：

- 只有 post-NMS 改善，raw assignment/gradient 指标不动；
- 依赖 GT RBox 选 threshold、harmonic order、epoch 或 checkpoint；
- 低方向性类别/近圆形目标被系统性饿死；
- HBox interior negative 减弱导致 duplicate 或 empty-image FP 显著增加；
- scalar $Q$ 和 factorized routing 没有可区分行为；
- 更换 MessDet 后才有收益，same-parent FCOS 无收益；
- 关键收益来自 boundary/scale/plugin，而非 assignment/evidence routing。

## 10. 预期审稿问题与回答边界

### Q1：这是不是 BGHR + uncertainty？

不是。BGHR 先从预测中选一个 best RBox 再挖 positives；v3 不先坍缩为单个 RBox，监督对象是跨视图对齐的 location–instance distribution，而且三类证据分别路由。必须由 A1/A3/A6 证明差异。

### Q2：这是不是 FAA 用在弱监督？

不是。FAA 在二维空间特征上估计并对齐方向；v3 的二阶圆周系数只汇总已知变换下的 axial prediction stability，不旋转 feature、不生成角度伪标签，也不改变 inference head。

### Q3：为什么不用 MessDet group feature？

为了 first causal study 保持 parent 不变。MessDet 同时改变 backbone、neck、head、orientation dimension 和预训练成本。只有 FCOS 上机制成立后才作为 portability/evidence-source study。

### Q4：为什么不合成一个质量分数？

因为低方向稳定性不等于背景，ownership ambiguity 也不等于类别错误。统一 $Q$ 会让一个证据越权决定无关损失；A5 是专门的 scalar-Q control。

### Q5：高 Fourier concentration 能证明方向正确吗？

不能。它只证明跨视图稳定。论文必须报告 wrong-consistent rate、risk–coverage 和校准失败案例，不能称其为 accuracy probability。

### Q6：density graph 是否就是 Point2RBox-v2？

不是。Point2RBox-v2 在 point supervision 下用 Gaussian overlap 和 Voronoi/watershed 推断尺度边界；v3 已知 HBox，只用 candidate-overlap graph 限定 ownership competition 和 transport computation，不生成 mask/size pseudo label。

## 11. 论文主张与题目

### 11.1 推荐主张

可以写：

- “We factorize uncertainty in HBox-supervised dense detection into ownership, orientation stability, and background safety.”
- “OrbitAssign-ER aligns soft location–instance assignments across known transformations and routes relation-specific evidence to compatible supervision terms.”
- “An axial second-order circular coefficient measures cross-view orientation stability without producing hard pseudo boxes.”
- “The method is training-only and leaves the CNN inference pipeline unchanged.”

不能写：

- first Fourier weakly-supervised oriented detector；
- first uncertainty-aware / self-aware detector；
- first dynamic or soft assignment；
- first rotation-equivariant CNN；
- Fourier confidence equals correctness probability；
- point/HBox/RBox unified framework；
- 接近或超过全监督，除非真实 matched results 支持。

### 11.2 推荐题目

首选英文：

**OrbitAssign: Factorized Evidence Routing for Horizontal-Box-Supervised Oriented Object Detection**

备选英文：

**Equivariant Soft Assignment with Relation-Specific Reliability for HBox-Supervised Oriented Detection**

中文：

**面向水平框监督旋转检测的跨视图软分配与因子化证据路由**

题目不放 Fourier、MessDet 或 self-aware，避免把辅助证据估计器写成整篇主张。

## 12. 推荐执行顺序

### Phase 0：只做 source/sanity audit

- 导出 H2RBox-v2 三视图 targets、angle predictions、annotation IDs；
- 完成 axial transform 单元测试；
- 测 static flip/JS、false-background、starvation、conflict graph degree；
- 用 RBox 只做 sealed report-only reliability audit。

### Phase 1：最小 ownership action

- 实现 canonical warp、single-view soft 与 cross-view soft；
- 完成 A2/A3；
- 不加 Fourier gate、background routing、MessDet、teacher 或 boundary cue。

### Phase 2：factorized routing

- 加入 $q^{own}$ 和 $q^{bg}$，完成 A4；
- 同时监控 duplicate、empty FP 和 pre-NMS score；
- 若 negative safety 失败，先关闭而非靠 NMS rescue。

### Phase 3：FMSA-inspired orientation evidence

- 加入 axial second-order concentration、方向损失 floor/ramp 与 factorized routing；
- 完成 scalar A5 与 factorized A6；
- 只有 G1.5/G2.5 通过才保留该组件。

### Phase 4：完整证据与迁移

- DOTA-v1.0 三 seed、DIOR-R 迁移、HRSC mechanism study；
- 获得并精读 TIP 2025 ILOE 全文；
- 再决定 MessDet group-harmonic 或 RTMDet portability；
- 点监督 FMSA-Point 作为独立后续项目，不塞入首篇。

## 13. 最终判断

融合 FMSA 后，v2 不应该变成更大的 pipeline，而应该变得更精确。

v2 已经抓住了“assignment field 是 HBox 监督下的隐藏变量”；FMSA 最有价值的补充是提醒我们：**assignment 不确定、方向不稳定和背景不安全并非同一件事。** v3 因此把“self-awareness”从六个并列模块收缩成一条原则：

> **证据只能监督它有资格证明的关系。**

最终推荐仍是 CNN/H2RBox-v2 parent，仍吸收 fixed-query 的 ownership、null、capacity 和 starvation 经验；新增的 Fourier 统计只做方向稳定性测量，不生成硬伪框。这样既融合了 FMSA 的有效灵感，又避免了 point warmup、MessDet、mask、prototype、scalar-Q 和 teacher–student 同时进入所造成的不可归因性。

在用户确认本 v3 设计前，本文件不授权代码实现、数据转换或训练启动。

## 14. 材料索引

### Version artifacts

- v2：`docs/project_history/exp_20260812_three_month_review/fres_20260512_20260812_openrsd_to_cnn_hbox_obb_proposal_v2_zh.md`
- v3：本文件
- 三个月完整复盘：`docs/project_history/exp_20260812_three_month_review/fres_20260512_20260812_openrsd_to_ovcapflow_review_and_detector_proposal_zh.md`
- FMSA reference：`local:FMSA_Point_OpenDraft_Assisted_Draft_v2.md`

### Literature artifacts

- `docs/literature-search-20260812-cnn-hbox-oriented-assignment/papers.md`
- `docs/literature-search-20260812-cnn-hbox-oriented-assignment/papers.csv`
- `docs/literature-search-20260812-cnn-hbox-oriented-assignment/search-notes.md`
- `docs/literature-search-20260812-cnn-hbox-oriented-assignment/fmsa-fusion-addendum.md`

### Local implementation evidence

- `local:OpenRSD/mmrotate/models/detectors/h2rbox_v2.py`
- `local:OpenRSD/mmrotate/models/dense_heads/h2rbox_v2_head.py`
- `local:OpenRSD/mmrotate_configs/h2rbox/`
- `local:OpenRSD/mmrotate/models/dense_heads/rotated_rtmdet_head.py`
