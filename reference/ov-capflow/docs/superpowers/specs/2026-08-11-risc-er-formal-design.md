# RISC-ER 正式方法规格：旋转超额语义风险与强检测底座

**状态：** 2026-08-11 用户书面确认，作为后续设计、实现和实验的最高方法边界。

**目标会议：** ICLR 2027。

**核心原则：** 强检测底座负责通用检测能力；RISC-ER 只解决旋转引起、且超过普通重复噪声的语义风险。两者不得混写成一个方法贡献。

---

## 1. 一句话问题与一句话方法

**问题：** 在遥感检测中，图像旋转可能保持目标几何可达性，却改变查询特征与文本原型的关系，使真实类别 margin 下降、错误类别形成 false hub，并增加围绕同一目标的错误竞争查询。

**方法：** RISC-ER 使用同像素 identity-repeat 作为噪声对照，只惩罚旋转视图相对该对照新增的对象级语义风险；训练时在查询排列商空间中按 GT object identity 对齐，推理时只对单视图查询语义施加有界、轨道零均值的低秩残差。

## 2. 已核实事实、推断和未决信息

### 2.1 已核实事实

- `[FACT]` 当前 E12 在 filtered-6605、scale-1024 口径上的 `dota/mAP=0.6340868473`、`AP50=0.6340`。
- `[FACT]` 本地复现 OpenRSD 在相同 paper mouth 上约为 `0.7049593925/0.7050`，差距约 `7.10` 点。
- `[FACT]` small vehicle、storage tank、roundabout、baseball diamond 四类的差距约解释其中 `5.0` 个 mAP 点。
- `[FACT]` DOTA2 验证集有 13,833 张图、243,632 个 GT；33 张图超过 600 个 GT。Q600 的最小容量缺口为 15,939 个 GT，即微平均 `6.5422%`。
- `[FACT]` 当前 E12 没有同 checkpoint 的全查询预测 dump，因此当前 geometry/semantic/duplicate/ranking 错误比例未知；旧 E24 分解不得直接迁移。
- `[FACT]` scene-disjoint N0-RI 已在 158 个 scene、4,147 个几何稳定身份上确认 rotation-induced class-flip excess `0.157421`，10,000 次 bootstrap 区间 `[0.130640, 0.184247]`，四折方向一致。
- `[FACT]` P0148 的公开标注文件为空，并且它不属于当前 DOTA2 验证集；支持向量干预证明预测敏感性，但不证明预测正误。
- `[FACT]` 现有 POQ 使用一个 identity 加两个非零旋转视图，并优化 feature、margin、background 三个独立损失；它不是本规格定义的 identity-controlled excess risk。
- `[FACT]` 现有低秩 gate 包含 query term 和 rank bias，未强制在旋转轨道上零均值，因此理论上可以删除与旋转无关的稳定语义。
- `[FACT]` 当前 E12 没有同父权重、同训练 schedule、仅关闭 POQ 的 matched control，现阶段不能归因 POQ 的 AP 效果。

### 2.2 有依据但仍需验证的推断

- `[INFERENCE]` 当前与 OpenRSD 的差距更可能是多因素叠加：遥感域父模型、tiny feature resolution、训练数据/时长、提示集、排序与固定 Q600 容量，而不是单一 RISC 缺陷。
- `[INFERENCE]` small-vehicle false hub 是值得优先研究的机制，但当前 E12 上它占多少 AP 缺口尚未测量。
- `[INFERENCE]` 对象级 owner/contender 风险比全局 unmatched background 分布更贴近 P0148 与 N0 的因果链，但其增益必须由 matched control 验证。
- `[INFERENCE]` 低秩、轨道零均值和范数上限有望避免语义塌缩；这不是已证明结论。

### 2.3 未决信息

- `[OPEN]` 当前 E12 的 perfect-ranking、duplicate-removal、perfect-class、geometry 和 Q-capacity oracle headroom。
- `[OPEN]` 当前 E12 上 POQ 相对 matched no-POQ control 的真实增量。
- `[OPEN]` 强遥感父权重的训练数据是否包含 DOTA val/test 或等价裁片。
- `[OPEN]` all-18 结果能否转化为严格 held-out-class open-vocabulary 改善。
- `[OPEN]` arbitrary-angle 改善是否部分来自学习插值伪影，而非旋转语义干扰。

## 3. 明确否定的前提和非目标

- `[NON-CLAIM]` 不声称 RISC 解决通用密集目标覆盖。
- `[NON-CLAIM]` 不声称 Q600 是当前 7.10 AP 差距的唯一或主要原因。
- `[NON-CLAIM]` 不用检测行数差异证明重复框是主要损失。
- `[NON-CLAIM]` 不用 P0148 作为定量 correctness 证据；未补权威标注前只作定性发现图。
- `[NON-CLAIM]` 不把遥感 backbone 初始化称为完整遥感父模型。
- `[NON-CLAIM]` 不把 all-18 prompt 训练称为严格 open-vocabulary generalization。
- `[NON-CLAIM]` 不声称完全没有 top-k。当前 two-stage GroundingDINO 的 encoder proposal selection 保留；禁止的是 decoder 输出后的全局 top-k/NMS 口径漂移。
- `[NON-GOAL]` RISC-ER 不移动 reference point，不修改 bbox regression，不增加 objectness/quality head，不做 `class_score × quality_score` 的后验乘法。
- `[NON-GOAL]` 不重新包装 OTA、DDQ、RQFormer、Rank-DETR 或跨视图 query matching 为本文创新。

## 4. 双层系统边界

### 4.1 B*：强检测底座，非论文贡献

B* 继续使用 OV-CapFlow/GroundingDINO 主体。它可以包含经过当前 E12 oracle 诊断证明必要的标准改进，例如：

- 兼容的遥感域预训练与完整检测器适配；
- tiny 目标确有 geometry headroom 时的高分辨率特征层；
- 固定框 perfect-ranking headroom 足够大时的既有排序方法；
- Q600 容量反事实确有收益时的查询数消融。

所有 B* 组件必须同时出现在 `B0-strong` 与 `B1-strong`，不得只给 RISC 组使用。B* 的改进写作定位是“更强且公平的 substrate”，不是 RISC 的模块。

RSP-Swin-T-E300 只能作为 backbone 初始化资产。若目标是追近 OpenRSD，还必须补齐完整模型的遥感域训练、数据规模、提示协议和 provenance 审计。OpenRSD 本身是多数据、多阶段、多提示系统，而不是一个可无损塞入 GroundingDINO 的单一父权重。

### 4.2 RISC-ER：唯一论文方法

RISC-ER 的贡献边界只有三点：

1. 定义并测量 **rotation excess semantic risk**，显式扣除 identity-repeat 噪声；
2. 在 `rotation × query permutation` 的商空间中按对象身份对齐，而不假设固定 query index；
3. 用单视图可执行、轨道零均值的低秩语义残差降低该风险，同时保持几何路径不被 RISC 辅助损失更新。

## 5. 训练视图与对象对应

### 5.1 三视图角色固定

每个 orbit update 构造：

\[
x^{0a},\quad x^{0b},\quad x^{\theta}.
\]

- `0a`：原输入的深拷贝，像素和标注不变；
- `0b`：同一原输入的第二次深拷贝，像素和标注不变；
- `theta`：同一输入的一个非零固定画布旋转，使用既有合法角度集合和可见区域过滤。

`0a` 与 `0b` 的差异只来自模型训练态随机性及数值噪声，不引入新的图像增强。旋转视图是唯一保留 RISC-ER 辅助梯度的视图；两个 identity 视图均 `no_grad`。

为隔离旋转增强本身，matched B0 必须使用完全相同的 orbit 采样、旋转检测损失、前向次数和 RNG 消耗，只关闭 quotient 与 excess-risk 梯度。这不是数据增强 sweep，而是不可省略的因果对照。

### 5.2 查询排列商空间

每个视图独立运行正常 Hungarian assignment。通过变换前写入的 `orbit_instance_ids` 连接同一 GT：

\[
q^{0a}_j \sim q^{0b}_j \sim q^\theta_j,
\]

其中三个 query index 可以不同。只有在三个视图中均可见且各自恰好有一个 owner 的对象进入 RISC-ER 分母。

不建立 Q600 到约 21k encoder proposals 的全局 OT；也不把 query-query Sinkhorn 作为方法贡献。

## 6. 轨道零均值低秩残差

对 decoder query `z` 和该 query 的预测旋转角 `phi`，沿用只作用于 classification readout 的低秩子空间：

\[
U=\operatorname{qr}(U_{raw}),\qquad
z_{stable}=z-\Delta(z,\phi).
\]

设角度特征为与 `le90` 的 \(\pi\)-周期一致的二阶、四阶谐波。对固定的 12 点半圆群积分网格 \(\Phi\)，定义：

\[
\tilde g(z,\phi)=\tanh(W_q\hat z+W_\phi h(\phi)+b),
\]

\[
g(z,\phi)=\frac{g_{max}}{2}
\left(
\tilde g(z,\phi)-
\frac{1}{|\Phi|}\sum_{\psi\in\Phi}\tilde g(z,\psi)
\right).
\]

因此在离散群积分上严格满足：

\[
\frac{1}{|\Phi|}\sum_{\phi\in\Phi}g(z,\phi)=0,
\qquad |g|\le g_{max}.
\]

残差仍受 `max_delta_norm_ratio` 限制。该中心化消除了“任意删除与角度无关语义”的自由度，是从旧 POQ 到 RISC-ER 的必要结构修订。

零初始化时 `z_stable == z` 必须逐元素完全成立，从而 B1 在 step 0 与父模型预测等价。

## 7. 对象级 owner/contender 风险

### 7.1 owner 与 contender

对视图 `v` 中对象 `j`：

- owner `q_j^v`：正常 Hungarian assignment 分配给该 GT 的 primary Q600 query；
- contender：其余 primary query 中，与某个 GT 的最大 rotated IoU 至少为 `0.5` 的 query；每个 query 只归属最大 IoU 的那个 GT，owner 排除在外。

`0.5` 与主 AP50 口径一致，不新增可调 top-k。IoU 选择和 bbox 坐标全部 stop-gradient。

### 7.2 单一对象风险

从 stable query 与冻结文本 readout 得到 class log score `s`。真实类别为 `y_j`。对象风险由同一个 log-sum-exp 聚合：

\[
\mathcal E_j^v=
\left\{
s^v_{q_j,c}-s^v_{q_j,y_j}:c\ne y_j
\right\}
\cup
\left\{
\max_c s^v_{q,c}-s^v_{q_j,y_j}:q\in C_j^v
\right\},
\]

\[
R_j^v=\tau\log\sum_{e\in\mathcal E_j^v}\exp(e/\tau).
\]

实现中固定 `tau=1`，不把它作为调参自由度。第一部分度量 owner 的错误类 hub；第二部分度量几何上围绕同一对象的竞争查询。它不惩罚没有几何联系的所有背景 query，也不声称消除普通场景的全部重复候选。

### 7.3 唯一 RISC-ER 目标

定义风险增量：

\[
d_j(a\rightarrow b)=[R_j^b-R_j^a]_+.
\]

唯一辅助目标为：

\[
\mathcal L_{ER}=\frac{1}{|\mathcal J|}
\sum_{j\in\mathcal J}
\left[
d_j(0a\rightarrow\theta)
-\operatorname{sg}\big(d_j(0a\rightarrow0b)\big)
\right]_+.
\]

不再并列 feature、margin、background 三个论文损失，也不引入额外 margin 超参数。空共享对象集合返回连接到 quotient 参数的有限零值。

为防止辅助目标通过共享视觉表示牺牲定位，计算 `L_ER` 时使用 detach 后的 raw query、预测角、文本和 bbox，重新经过 quotient 得到 active stable query；因此 `L_ER` 只更新 quotient 参数。正常检测损失仍按原模型训练视觉主干、decoder、分类和回归。

## 8. 训练和推理数据流

### 8.1 普通 update

```text
single input
  -> unchanged OV-CapFlow forward
  -> quotient-stabilized classification / unchanged bbox regression
  -> normal detection losses
```

### 8.2 orbit update

```text
same sample
  -> identity A (no_grad) -> assignment/state/risk baseline
  -> identity B (no_grad) -> identity-repeat noise
  -> rotated view (grad)  -> normal detection loss
                           -> quotient-only RISC-ER loss
```

使用同一 optimizer 和同一基础学习率。`orbit_interval=4` 是三视图目标的固定稀疏估计，并按 4 倍保持其期望权重；只有在 update schedule 与样本内容独立时才能称为无偏，因此必须审计各类/密度在 orbit update 上的覆盖。它不是两套 optimizer 或两时间尺度学习率。

### 8.3 单视图推理

```text
one image -> Q600 decoder queries
          -> predicted query angle
          -> bounded orbit-centered quotient
          -> text similarity scores + original boxes
          -> all Q600 rows
```

推理不需要 paired views、GT、Hungarian matching、contender 构造或 RISC loss。

## 9. 因果实验矩阵

### 9.1 主 2×2 矩阵

| substrate | matched control | RISC-ER |
|---|---|---|
| 当前底座 | `B0-weak` | `B1-weak` |
| 强遥感底座 B* | `B0-strong` | `B1-strong` |

四组保持：数据、seed、batch、累计步数、LR、prompt、orbit 采样、检测 loss、验证 mouth 和 checkpoint 选择规则一致。

### 9.2 必要消融

- `matched orbit control`：相同旋转训练但无 quotient/ER；
- `quotient without ER`：检验结构本身是否只是额外参数；
- `raw consistency`：不扣 identity-repeat 的普通一致性基线；
- `without contender energy`：只保留 owner wrong-class risk；
- `without orbit centering`：验证零均值可识别性；
- `lossless 90°` 与 arbitrary angle 分开报告；
- label-permuted orbit placebo；
- strict held-out-class folds，仅在结果成立后提出 OV claim。

### 9.3 指标

**主任务指标：** exact filtered-6605 AP50/mAP、raw-13833 AP50/mAP、per-class AP、small-vehicle AP。

**主机制指标：** scene-macro class-flip excess、true-vs-best-wrong margin drift excess、JS excess、false-hub rate、contender excess、geometry retention。

**安全指标：** true small-vehicle recall、tiny `<=8 px`、density `>600`、empty-tile foreground、novel/held-out AP、训练时间、峰值显存、单视图推理延迟。

## 10. 晋级和停止规则

### 10.1 实现前硬门

在正式 B0/B1 训练前必须完成当前 E12 的同模型全查询 dump 和 oracle 分解。旧 E24 结果只能作为历史先验。

若 current-E12 诊断显示 small-vehicle 等主差距主要是 geometry-unreachable，优先建设 B*，不得把该缺口归因给 RISC-ER。

### 10.2 Stage-0 机制门

RISC-ER 只有同时满足以下条件才进入完整训练：

- matched proxy 上 rotation flip/margin/hub excess 方向一致改善；
- geometry retention 不下降；
- true small-vehicle recall 不出现实质性下降；
- identity-repeat risk 不被人为放大来逃避 excess hinge；
- quotient gate、norm、有限值和 owner coverage 审计通过。

### 10.3 AP 晋级门

- 单 seed 完整同口径结果必须优于 matched B0，不能只优于旧 checkpoint；
- 若提升不足 `+0.3` AP 且机制指标不显著，停止该配方；
- 若提升达到 `+0.3` AP 且机制成立，进入至少 3 seeds；
- `AP50>=0.70` 是系统工程目标，不是 RISC-ER 的预注册承诺。

## 11. 被忽略变量、成本和偏差

- **父权重泄漏：** 遥感预训练可能包含 DOTA 或等价裁片；必须发布数据 provenance。
- **评测口径偏差：** filtered-6605 与 raw-13833 永久分表；OpenRSD 的 threshold/NMS 输出不能与 all-Q600 行数作因果比较。
- **prompt 偏差：** canonical 与 text7 分报，不能只选最优 prompt。
- **模型选择偏差：** 预先固定 endpoint 和 early-stop gate，不能从 12 个 epoch 中事后挑最好值。
- **多重尝试偏差：** 历史负实验必须进入附录/工作账本，不能只报告最终成功路线。
- **尺度与密度混杂：** density 高的图通常 tiny 目标更多；需要二维 strata，不做单变量相关即因果的表述。
- **插值伪影：** arbitrary-angle 旋转同时改变插值和有效画布；必须有 lossless 90° 与 identity control。
- **开放词汇泄漏：** all-18 GT loss 只能证明已知类机制；OV claim 依赖严格 held-out fold 或跨数据集结果。
- **算力成本：** interval-4 三视图的平均前向量约为普通训练的 1.5 倍；实测时间和显存必须报告。
- **容量上限：** Q600 对 `>600 GT` 图存在硬缺口，RISC-ER 不可能恢复不存在的查询容量。

## 12. 最近邻工作与 novelty 边界

- OTA：全局最优传输分配，CVPR 2021，https://openaccess.thecvf.com/content/CVPR2021/papers/Ge_OTA_Optimal_Transport_Assignment_for_Object_Detection_CVPR_2021_paper.pdf
- DDQ：dense distinct queries，CVPR 2023，https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Dense_Distinct_Query_for_End-to-End_Object_Detection_CVPR_2023_paper.pdf
- Rank-DETR：排序/定位对齐，NeurIPS 2023，https://proceedings.neurips.cc/paper_files/paper/2023/hash/34074479ee2186a9f236b8fd03635372-Abstract-Conference.html
- Semi-DETR：跨视图 query consistency，CVPR 2023，https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Semi-DETR_Semi-Supervised_Object_Detection_With_Detection_Transformers_CVPR_2023_paper.pdf
- RQFormer：遥感密集旋转目标的 selective distinct queries，https://arxiv.org/abs/2311.17629
- ReDet/FRED：遥感旋转等变检测，https://openaccess.thecvf.com/content/CVPR2021/papers/Han_ReDet_A_Rotation-Equivariant_Detector_for_Aerial_Object_Detection_CVPR_2021_paper.pdf 与 https://ojs.aaai.org/index.php/AAAI/article/view/28069
- OpenRSD：多提示、多任务、多阶段遥感开放检测，ICCV 2025，https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_OpenRSD_Towards_Open-prompts_for_Object_Detection_in_Remote_Sensing_Images_ICCV_2025_paper.pdf

允许的 novelty 表述是：

> We formulate rotation-induced semantic interference as excess object-level risk over an identity-repeat control in the quotient of image rotation and query permutation, and learn a bounded orbit-centered semantic residual that is used with a single view at inference.

在完成更新至投稿截止日前的系统检索前，不使用“first”或“首次”。

## 13. 论文主张—证据矩阵

| 论文主张 | 最低证据 | 当前状态 |
|---|---|---|
| 旋转会在几何稳定时造成额外语义漂移 | scene-disjoint N0、identity control、bootstrap | 已有 N0 支持 |
| RISC-ER 降低 rotation excess risk | matched B0/B1、机制指标、多 seed | 未执行 |
| 改善不是定位退化造成 | geometry retention、bbox stop-gradient、oracle | 未执行 |
| single-view inference 有效 | 正常推理配置与延迟审计 | 结构上可行，未测 |
| 对 OV 类别可迁移 | strict held-out fold/跨数据集 | 无证据，禁止当前声称 |
| 系统追近 OpenRSD | 同 mouth B*-control/RISC 与 provenance | 未执行 |

## 14. 正式决策

- `[DECISION]` 停止把 global Query Ownership Flow 作为主贡献。
- `[DECISION]` B* 与 RISC-ER 分层，所有强底座改进必须在 control/candidate 中对称。
- `[DECISION]` RISC-ER 使用 identity A、identity B、一个 rotated view。
- `[DECISION]` 只保留一个 rotation-excess object-risk 目标。
- `[DECISION]` quotient 在离散旋转群积分上强制零均值。
- `[DECISION]` RISC 辅助损失只更新 quotient，不直接更新 geometry/shared visual path。
- `[DECISION]` P0148 只作定性发现，除非补权威标注。
- `[DECISION]` 当前 E12 诊断先于正式新训练；不再用旧 E24 分解代替。
- `[DECISION]` 不承诺 RISC 单独达到 AP50 70；系统目标与方法因果增量分别报告。
