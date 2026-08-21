# FMSA Fusion Literature Addendum

> Snapshot: 2026-08-12  
> Scope: 对 OrbitAssign v2 融合 FMSA-Point 灵感所需的增量文献核验  
> Privacy: 检索仅使用公开题名、任务和方法关键词，未提交私有草稿原文

## Incremental papers

| Rank | Paper | Year/Venue | Verified mechanism | Relevance | Grade | Canonical URL |
|---:|---|---|---|---|:---:|---|
| 1 | Measuring the Impact of Rotation Equivariance on Aerial Object Detection (MessDet) | ICCV 2025 | E2CNN-based strict/approximate rotation-equivariant CSPNeXt/PAFPN、RE channel attention、orientation-group multi-branch head；$N=8$ | FMSA 的 group-feature 来源；不能直接当弱监督方向置信证据 | S | https://arxiv.org/abs/2507.09896 |
| 2 | Fourier Angle Alignment for Oriented Object Detection in Remote Sensing | CVPR 2026 | local 2D FFT、polar angular energy、feature orientation alignment in FPN/head | 阻止“首个 Fourier angle”主张；与 group-axis/circular moment 严格区分 | S | https://openaccess.thecvf.com/content/CVPR2026/html/Gu_Fourier_Angle_Alignment_for_Oriented_Object_Detection_in_Remote_Sensing_CVPR_2026_paper.html |
| 3 | PointOBB-v3: Expanding Performance Boundaries of Single Point-Supervised Oriented Object Detection | 2025 preprint | multi-view scale/angle learning、SSFF、end-to-end branch、instance-aware weighting | multi-view scale/angle 与 quality weighting 的相邻边界 | A | https://arxiv.org/abs/2501.13898 |
| 4 | Point2RBox-v2: Rethinking Point-supervised Oriented Object Detection with Spatial Layout Among Instances | CVPR 2025 | Gaussian overlap、Voronoi watershed、consistency | density/layout-aware point supervision 的直接边界 | S | https://openaccess.thecvf.com/content/CVPR2025/html/Yu_Point2RBox-v2_Rethinking_Point-supervised_Oriented_Object_Detection_with_Spatial_Layout_Among_CVPR_2025_paper.html |
| 5 | Point2RBox-v3: Self-Bootstrapping from Point Annotations via Integrated Pseudo-Label Refinement and Utilization | 2025 preprint | progressive label assignment、prior-guided dynamic mask、pseudo-label refinement/utilization | 说明 teacher/pseudo-label/dynamic assignment 路线已拥挤 | A | https://arxiv.org/abs/2509.26281 |
| 6 | Instance-Level Orientation Enhancement for Horizontal Box Supervised Oriented Object Detection in Remote Sensing Images | IEEE TIP 2025 | 题名、作者、期刊卷页与 DOI 已核验；全文机制尚待取得 | HBox orientation enhancement 的高风险近邻，投稿前必须全文审计 | S-risk | https://doi.org/10.1109/TIP.2025.3632224 |

## Verified corrections to the assisted draft

1. MessDet 主实验的 orientation dimension 是 $N=8$，不是 $K=6$。
2. MessDet 使用 orientation-group features 构造 multi-branch head，但其论文没有把 1D group-axis Fourier amplitude 定义为弱监督伪框置信度。
3. FAA 对局部二维空间 feature 做 2D DFT，并通过极坐标角能量的 argmax 估计主方向；它不是对 MessDet group axis 做一阶 DFT。
4. 对 $\pi$-periodic OBB axis，稳定的无向轴统计应显式处理二倍角 $e^{i2\theta}$。具体 group harmonic 与 phase law 必须由表示和群作用推导/单元测试，不能默认。
5. PointOBB-v3 已覆盖 multi-view scale/angle、instance-aware weighting；Point2RBox-v2/v3 已覆盖 layout、progressive assignment 与 pseudo-label refinement。因此 FMSA 的模块列表不能作为组合式新颖性来源。

## Design implication

推荐保留的不是 `Fourier + MessDet + six awareness modules`，而是：

- 用 prediction-level axial circular concentration 衡量跨视图方向稳定性；
- 不把 concentration 当 correctness probability 或 hard pseudo angle；
- 将 ownership、geometry 和 background evidence 分别路由到对应 loss；
- 用 HBox conflict graph 取代最近点距离—尺度等式；
- MessDet/group harmonic 只作 second-stage evidence-source study。

## Remaining literature risk

- 获取并精读 IEEE TIP 2025 Instance-Level Orientation Enhancement 全文；
- 从该文、BGHR、ABBSPO、H2RBox-v2、FAA、MessDet 做 backward/forward citation trace；
- 正式投稿前重复搜索 `assignment equivariance`、`relation-specific reliability`、`orientation confidence HBox supervision`；
- 继续避免 `first`、`unprecedented` 和未校准的 probability 表述。
