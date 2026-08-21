# CNN HBox-Supervised Oriented Detection and Dense Assignment — Traceable Literature

> 检索快照：2026-08-12（Asia/Shanghai）  
> 工作范围：完整 HBox 标注下的 CNN 旋转检测、跨视图等变性、dense label assignment、soft sampling 与 negative safety  
> 时间范围：重点 2021–2026；FCOS/ATSS 作为必要基础工作向前追溯  
> 筛选结果：26 篇具体候选中保留 18 篇；优先官方 proceedings、出版社/DOI、OpenReview 和作者 arXiv 页面

## 质量等级

- `S`：与拟议方法高度接近、顶会/顶刊且链接可核验，必须精读和正面对比；
- `A`：直接支撑方法设计或限定主张边界；
- `B`：基础设施、扩展路线或相邻问题的重要参考；
- 等级表示“对当前课题的价值”，不是对论文本身作绝对质量排名。

## 最终文献集

| Rank | 文献 | 作者 | 年份/会议期刊 | 监督/架构 | 对当前课题的作用 | 等级 | DOI/真实地址 |
|---:|---|---|---|---|---|:---:|---|
| 1 | [BGHR: Bridging the Gap Between HBox-Supervised and RBox-Supervised Oriented Object Detection via Adaptive Fine-Grained Sample Mining](https://ojs.aaai.org/index.php/AAAI/article/view/32310) | Chenlin Fu; Yingying Zhu | AAAI 2025 | HBox; CNN/FCOS | 最接近工作：AFSM/PRA 选择最佳预测 RBox 并挖掘正样本；直接阻止“首个 HBox sample assignment”主张 | S | [10.1609/aaai.v39i3.32310](https://doi.org/10.1609/aaai.v39i3.32310) |
| 2 | [H2RBox-v2: Incorporating Symmetry for Boosting Horizontal Box Supervised Oriented Object Detection](https://papers.neurips.cc/paper_files/paper/2023/hash/b9603de9e49d0838e53b6c9cf9d06556-Abstract-Conference.html) | Yi Yu; Xue Yang; Qingyun Li; Yue Zhou; Gefan Zhang; Feipeng Da; Junchi Yan | NeurIPS 2023 | HBox; Rotated FCOS | 原图/旋转/翻转对称学习和主要 parent；必须证明 assignment consistency 不等于已有 box/angle consistency | S | [10.52202/075280-2581](https://doi.org/10.52202/075280-2581) |
| 3 | [ABBSPO: Adaptive Bounding Box Scaling and Symmetric Prior based Orientation Prediction for Detecting Aerial Image Objects](https://openaccess.thecvf.com/content/CVPR2025/html/Lee_ABBSPO_Adaptive_Bounding_Box_Scaling_and_Symmetric_Prior_based_Orientation_CVPR_2025_paper.html) | Woojin Lee; Hyugjae Chang; Jaeho Moon; Jaehyup Lee; Munchurl Kim | CVPR 2025 | HBox; CNN | 处理 tight/coarse HBox 尺度失配与三视图一致错误；限定 scale/symmetry 主张 | S | [Official paper PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_ABBSPO_Adaptive_Bounding_Box_Scaling_and_Symmetric_Prior_based_Orientation_CVPR_2025_paper.pdf) |
| 4 | [Explicit and Implicit Box Equivariance Learning for Weakly-Supervised Rotated Object Detection](https://doi.org/10.1109/TETCI.2024.3398020) | Linfei Wang; Yibing Zhan; Xu Lin; Baosheng Yu; Liang Ding; Jianqing Zhu; Dapeng Tao | IEEE TETCI 2025 | HBox; CNN | EIE-Det 已覆盖显式/隐式 box equivariance；新方案需把等变对象限定为 assignment field | S | [10.1109/TETCI.2024.3398020](https://doi.org/10.1109/TETCI.2024.3398020) |
| 5 | [H2RBox: Horizontal Box Annotation is All You Need for Oriented Object Detection](https://arxiv.org/abs/2210.06742) | Xue Yang; Gefan Zhang; Wentong Li; Xuehui Wang; Yue Zhou; Junchi Yan | ICLR 2023 | HBox; Rotated FCOS | HBox→RBox 双视图一致性范式和直接 baseline | S | [arXiv:2210.06742](https://arxiv.org/abs/2210.06742) |
| 6 | [Wholly-WOOD: Wholly Leveraging Diversified-quality Labels for Weakly-supervised Oriented Object Detection](https://arxiv.org/abs/2502.09471) | Yi Yu; Xue Yang; Yansheng Li; Zhenjun Han; Feipeng Da; Junchi Yan | IEEE TPAMI 2025 | point/HBox/RBox/mixed; CNN | 阻止“统一多质量标注框架”宽主张；提供 HBox-only 强对照 | S | [10.1109/TPAMI.2025.3542542](https://doi.org/10.1109/TPAMI.2025.3542542) |
| 7 | [OTA: Optimal Transport Assignment for Object Detection](https://openaccess.thecvf.com/content/CVPR2021/html/Ge_OTA_Optimal_Transport_Assignment_for_Object_Detection_CVPR_2021_paper.html) | Zheng Ge; Songtao Liu; Zeming Li; Osamu Yoshie; Jian Sun | CVPR 2021 | full HBox; CNN | 全局 GT–anchor transport、拥挤场景和列竞争基础；阻止把 transport 本身写成创新 | A | [Official paper PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Ge_OTA_Optimal_Transport_Assignment_for_Object_Detection_CVPR_2021_paper.pdf) |
| 8 | [IQDet: Instance-wise Quality Distribution Sampling for Object Detection](https://openaccess.thecvf.com/content/CVPR2021/html/Ma_IQDet_Instance-Wise_Quality_Distribution_Sampling_for_Object_Detection_CVPR_2021_paper.html) | Yuchen Ma; Songtao Liu; Zeming Li; Jian Sun | CVPR 2021 | full HBox; CNN | 实例级概率分布采样与训练期零推理开销先例；阻止“首个 soft distribution sampling”主张 | A | [arXiv:2104.06936](https://arxiv.org/abs/2104.06936) |
| 9 | [Dynamic Coarse-To-Fine Learning for Oriented Tiny Object Detection](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_Dynamic_Coarse-To-Fine_Learning_for_Oriented_Tiny_Object_Detection_CVPR_2023_paper.html) | Chang Xu; Jian Ding; Jinwang Wang; Wen Yang; Huai Yu; Lei Yu; Gui-Song Xia | CVPR 2023 | full RBox; CNN | rotated tiny objects 的 dynamic prior/assignment；用于小目标和 per-level 对照 | A | [10.1109/CVPR52729.2023.00707](https://doi.org/10.1109/CVPR52729.2023.00707) |
| 10 | [Task-wise Sampling Convolutions for Arbitrary-Oriented Object Detection in Aerial Images](https://arxiv.org/abs/2209.02200) | Zhanchao Huang; Wei Li; Xiang-Gen Xia; Hao Wang; Ran Tao | IEEE TNNLS 2025 | full RBox; CNN | task-wise feature sampling + dynamic task-aware assignment；限定分类/定位对齐主张 | A | [10.1109/TNNLS.2024.3367331](https://doi.org/10.1109/TNNLS.2024.3367331) |
| 11 | [FRED: Towards a Full Rotation-Equivariance in Aerial Image Object Detection](https://ojs.aaai.org/index.php/AAAI/article/view/28069) | Chanho Lee; Jinsu Son; Hyounguk Shon; Yunho Jeon; Junmo Kim | AAAI 2024 | full RBox; CNN | 完整检测流程 rotation equivariance；阻止宽泛“首个旋转等变”主张 | A | [10.1609/aaai.v38i4.28069](https://doi.org/10.1609/aaai.v38i4.28069) |
| 12 | [Point2RBox-v2: Rethinking Point-supervised Oriented Object Detection with Spatial Layout Among Instances](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_Point2RBox-v2_Rethinking_Point-supervised_Oriented_Object_Detection_with_Spatial_Layout_Among_CVPR_2025_paper.html) | Yi Yu; Botao Ren; Peiyuan Zhang; Mingxin Liu; Junwei Luo; Shaofeng Zhang; Feipeng Da; Junchi Yan; Xue Yang | CVPR 2025 | point; CNN | 密集实例的 Gaussian overlap/Voronoi/consistency；说明 point/layout 路线拥挤，作为局部排他边界 | A | [arXiv:2502.04268](https://arxiv.org/abs/2502.04268) |
| 13 | [Partial Weakly-Supervised Oriented Object Detection](https://arxiv.org/abs/2507.02751) | Mingxin Liu; Peiyuan Zhang; Yuan Liu; Wei Zhang; Yue Zhou; Ning Liao; Ziyang Gong; Junwei Luo; Zhirui Wang; Yi Yu; Xue Yang | CVPR 2026 | partial HBox/point + unlabeled; teacher-student | PWOOD 覆盖部分弱标注、OS-Student 与 class-agnostic pseudo-label filtering | A | [arXiv:2507.02751](https://arxiv.org/abs/2507.02751) |
| 14 | [SPWOOD: Sparse Partial Weakly-Supervised Oriented Object Detection](https://openreview.net/pdf?id=PXaboOpwGv) | Wei Zhang; Xiang Liu; Ningjing Liu; Mingxin Liu; Wei Liao; Chunyan Xu; Xue Yang | ICLR 2026 | sparse partial weak + unlabeled; teacher-student | 处理漏标目标的 hard-negative 和多级伪标签过滤；与完整 HBox 内部歧义作边界区分 | A | [OpenReview](https://openreview.net/forum?id=PXaboOpwGv) |
| 15 | [Oriented RepPoints for Aerial Object Detection](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Oriented_RepPoints_for_Aerial_Object_Detection_CVPR_2022_paper.html) | Wentong Li; Yijie Chen; Kaixuan Hu; Jianke Zhu | CVPR 2022 | full RBox; CNN | adaptive points、质量评估与 sample assignment 的 oriented 先例 | B | [Official paper PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Oriented_RepPoints_for_Aerial_Object_Detection_CVPR_2022_paper.pdf) |
| 16 | [RTMDet: An Empirical Study of Designing Real-Time Object Detectors](https://arxiv.org/abs/2212.07784) | Chengqi Lyu; Wenwei Zhang; Haian Huang; Yue Zhou; Yudong Wang; Yanyi Liu; Shilong Zhang; Kai Chen | arXiv 2022 | full; CNN | dynamic soft-label assigner 与第二阶段迁移 parent；作为工程基础而非 novelty 证据 | B | [arXiv:2212.07784](https://arxiv.org/abs/2212.07784) |
| 17 | [Bridging the Gap Between Anchor-Based and Anchor-Free Detection via Adaptive Training Sample Selection](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Bridging_the_Gap_Between_Anchor-Based_and_Anchor-Free_Detection_via_Adaptive_CVPR_2020_paper.html) | Shifeng Zhang; Cheng Chi; Yongqiang Yao; Zhen Lei; Stan Z. Li | CVPR 2020 | full HBox; CNN | ATSS 奠定“正负样本定义决定 detector 差异”的基础 | B | [Official paper PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Bridging_the_Gap_Between_Anchor-Based_and_Anchor-Free_Detection_via_Adaptive_CVPR_2020_paper.pdf) |
| 18 | [FCOS: Fully Convolutional One-Stage Object Detection](https://openaccess.thecvf.com/content_ICCV_2019/html/Tian_FCOS_Fully_Convolutional_One-Stage_Object_Detection_ICCV_2019_paper.html) | Zhi Tian; Chunhua Shen; Hao Chen; Tong He | ICCV 2019 | full HBox; CNN | H2RBox-v2 的 anchor-free dense parent；解释 center sampling、centerness 和 location targets | B | [Official paper PDF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Tian_FCOS_Fully_Convolutional_One-Stage_Object_Detection_ICCV_2019_paper.pdf) |

## 文献综合结论

### 已被占据的命题

1. HBox→RBox 的 rotate/flip consistency：H2RBox、H2RBox-v2。
2. 显式/隐式 box equivariance：EIE-Det。
3. HBox 下的 fine-grained sample mining / predicted-RBox assignment：BGHR。
4. HBox scale correction 与 symmetric prior：ABBSPO。
5. point/HBox/RBox/mixed 或 partial/sparse weak supervision：Wholly-WOOD、PWOOD、SPWOOD。
6. soft/probabilistic/global/dynamic assignment：IQDet、OTA、ATSS、DCFL、RTMDet、TS-Conv。
7. full rotation equivariance：FRED。

### 当前可继续检验的窄命题

> 将跨视图等变约束施加到 **CNN dense location–instance soft assignment distribution**，并用其不确定性调节完整 HBox 内部的 background gradients。

该命题需要与 BGHR 作最强消融，因为 BGHR 已经明确指出 HBox 与 RBox 监督的 sample-selection gap。安全表述是“本次定向检索未发现精确同构方法”，不能使用 `first`。

## 推荐精读顺序

1. H2RBox-v2 → 本地 `h2rbox_v2_head.py`：核对 box consistency 与 static target assignment 的实际分界。
2. BGHR：逐式复现 PRA/AFSM，设计 single-best-RBox hard control。
3. ABBSPO + EIE-Det：排除 scale/symmetry/box-equivariance 的贡献混淆。
4. OTA + IQDet + DCFL：确定 soft plan、coverage 和拥挤竞争的最简实现。
5. Wholly-WOOD/PWOOD/SPWOOD：收紧 mixed/partial/negative-safe 相关措辞。
6. RTMDet：只用于第二阶段 portability，不用于首轮 novelty。

## 可追溯性说明

- 所有最终条目均有官方 proceedings、出版社 DOI、OpenReview 或作者 arXiv 地址。
- 未使用博客、聚合站或代码 README 作为方法事实的最终依据。
- MDPI 检索结果未进入最终集。
- ABBSPO 的官方 CVPR 页面与 PDF 已核验；其 arXiv 版本较晚，不作为年份判断依据。
- RTMDet 在本集合中标为基础设施预印本，不与正式顶会论文等同计权。

