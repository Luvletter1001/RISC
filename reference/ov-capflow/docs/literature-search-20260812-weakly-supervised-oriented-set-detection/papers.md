# Weakly Supervised Oriented Set Detection — Curated Literature

> Search snapshot: 2026-08-12  
> Scope: HBox/point/partial supervision for oriented detection; rotation-equivariance; rotated DETR/set prediction; ownership and negative supervision  
> Decision use: define the novelty boundary for a fixed-query HBox→OBB framework

## Ranking rubric

Each paper is scored from 1–5 on: topical fit (`Fit`), method proximity (`Close`), source/evidence reliability (`Evidence`), recency (`Recent`), and design usefulness (`Action`). `Total` is the unweighted sum out of 25. Scores are routing aids, not paper-quality judgments.

## Final set

| Rank | Paper | Venue/year | Supervision | Fit | Close | Evidence | Recent | Action | Total | Decision |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | [H2RBox-v2: Incorporating Symmetry for Boosting Horizontal Box Supervised Oriented Object Detection](https://papers.neurips.cc/paper_files/paper/2023/hash/b9603de9e49d0838e53b6c9cf9d06556-Abstract-Conference.html) | NeurIPS 2023 | HBox | 5 | 5 | 5 | 3 | 5 | 23 | `CORE`: rotate/flip consistency and full-supervision-gap reference |
| 2 | [ABBSPO: Adaptive Bounding Box Scaling and Symmetric Prior based Orientation Prediction](https://openaccess.thecvf.com/content/CVPR2025/html/Lee_ABBSPO_Adaptive_Bounding_Box_Scaling_and_Symmetric_Prior_based_Orientation_CVPR_2025_paper.html) | CVPR 2025 | HBox | 5 | 5 | 5 | 5 | 5 | 25 | `CORE`: tight/coarse HBox mismatch and symmetry collision |
| 3 | [BGHR: Bridging the Gap Between HBox-Supervised and RBox-Supervised Oriented Object Detection](https://ojs.aaai.org/index.php/AAAI/article/view/32310) | AAAI 2025 | HBox | 5 | 5 | 5 | 5 | 5 | 25 | `CORE`: sample mining, RBox assignment, symmetric self-supervision |
| 4 | [Wholly-WOOD: Wholly Leveraging Diversified-Quality Labels for Weakly-Supervised Oriented Object Detection](https://doi.org/10.1109/TPAMI.2025.3542542) | TPAMI 2025 | point/HBox/RBox/mixed | 5 | 4 | 5 | 5 | 5 | 24 | `CORE`: blocks broad “unified supervision” claim |
| 5 | [Partial Weakly-Supervised Oriented Object Detection](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_Partial_Weakly-Supervised_Oriented_Object_Detection_CVPR_2026_paper.html) | CVPR 2026 | partial weak HBox/point | 5 | 4 | 5 | 5 | 5 | 24 | `CORE`: blocks broad partial/mixed-weak claim; strongest current boundary |
| 6 | [H2RBox: Horizontal Box Annotation is All You Need for Oriented Object Detection](https://arxiv.org/abs/2210.06742) | ICLR 2023 | HBox | 5 | 5 | 5 | 3 | 5 | 23 | `CORE`: original HBox self-supervised paradigm |
| 7 | [Point2RBox-v2: Rethinking Point-supervised Oriented Object Detection with Spatial Layout Among Instances](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_Point2RBox-v2_Rethinking_Point-supervised_Oriented_Object_Detection_with_Spatial_Layout_Among_CVPR_2025_paper.html) | CVPR 2025 | point | 4 | 4 | 5 | 5 | 5 | 23 | `CORE`: dense-scene layout upper/lower bounds and consistency |
| 8 | [PointOBB-v2: Towards Simpler, Faster, and Stronger Single Point Supervised Oriented Object Detection](https://proceedings.iclr.cc/paper_files/paper/2025/hash/636d57c09a5baacd83722639265802f6-Abstract-Conference.html) | ICLR 2025 | point | 4 | 3 | 5 | 5 | 4 | 21 | `CORE`: class map/PCA pseudo-RBox and strong point-only baseline |
| 9 | [PointOBB: Learning Oriented Object Detection via Single Point Supervision](https://openaccess.thecvf.com/content/CVPR2024/html/Luo_PointOBB_Learning_Oriented_Object_Detection_via_Single_Point_Supervision_CVPR_2024_paper.html) | CVPR 2024 | point | 4 | 4 | 5 | 4 | 4 | 21 | `CORE`: multi-view scale/angle consistency prior art |
| 10 | [Point2RBox: Combine Knowledge from Synthetic Visual Patterns for End-to-end Oriented Object Detection](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_Point2RBox_Combine_Knowledge_from_Synthetic_Visual_Patterns_for_End-to-end_Oriented_CVPR_2024_paper.html) | CVPR 2024 | point | 4 | 3 | 5 | 4 | 4 | 20 | `CORE`: synthetic-pattern knowledge plus transform self-supervision |
| 11 | [Relational Matching for Weakly Semi-Supervised Oriented Object Detection](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Relational_Matching_for_Weakly_Semi-Supervised_Oriented_Object_Detection_CVPR_2024_paper.html) | CVPR 2024 | partial RBox + point | 4 | 4 | 5 | 4 | 5 | 22 | `CORE`: rotation-modulated relational matching and ambiguous points |
| 12 | [Hausdorff Distance Matching with Adaptive Query Denoising for Rotated Detection Transformer](https://openaccess.thecvf.com/content/WACV2025/html/Lee_Hausdorff_Distance_Matching_with_Adaptive_Query_Denoising_for_Rotated_Detection_WACV_2025_paper.html) | WACV 2025 | full RBox | 4 | 5 | 5 | 5 | 5 | 24 | `CORE`: strongest direct neighbor on rotated query matching/denoising |
| 13 | [AO2-DETR: Arbitrary-Oriented Object Detection Transformer](https://arxiv.org/abs/2205.12785) | preprint 2022 | full RBox | 4 | 5 | 4 | 2 | 4 | 19 | `CORE`: oriented proposals, refinement, rotation-aware set matching |
| 14 | [FRED: Towards a Full Rotation-Equivariance in Aerial Image Object Detection](https://ojs.aaai.org/index.php/AAAI/article/view/28069) | AAAI 2024 | full RBox | 4 | 3 | 5 | 4 | 4 | 20 | `ADJACENT`: blocks broad rotation-equivariance claim |
| 15 | [Open-World Objectness Modeling Unifies Novel Object Detection](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Open-World_Objectness_Modeling_Unifies_Novel_Object_Detection_CVPR_2025_paper.html) | CVPR 2025 | open-world, incomplete labels | 3 | 4 | 5 | 5 | 5 | 22 | `ADJACENT`: unmatched-query/objectness negative-safety reference |

## What the set says

### Crowded claims

- HBox→RBox with rotate/flip consistency is already established by H2RBox/H2RBox-v2.
- HBox scale mismatch and symmetry priors are explicitly addressed by ABBSPO.
- HBox sample assignment/mining is central in BGHR.
- Point-only angle/scale recovery, pseudo-RBox generation, and dense layout have strong 2024–2025 solutions.
- Unified or partially weak use of point/HBox/RBox is covered by Wholly-WOOD and PWOOD.
- Full-supervised rotated DETR/set matching is covered by AO2-DETR, ARS-DETR, RHINO and related work.

### Surviving provisional gap

The current defensible search target is narrower:

> HBox supervision as a latent feasible set inside a fixed-query one-to-one rotated detector, with annotation-identity orbit ownership and ambiguity-aware negative supervision, without hard pseudo-RBoxes or inference NMS.

No final-set paper was found to make exactly this joint formulation central. This is a `needs-search` gap, not a novelty verdict.

## Design implications

1. Do not use “rotation consistency” as the method name or sole contribution.
2. Do not make generic mixed-supervision unification the paper claim.
3. Compare directly with H2RBox-v2, BGHR and ABBSPO on HBox supervision.
4. Include RHINO/AO2-DETR-style full-supervised set predictors as parent/upper-bound context.
5. Report dense-scene ownership and negative-gradient diagnostics, not only AP.
6. Keep point supervision as a later adapter unless it creates a distinct experiment rather than a second paper inside the first.

