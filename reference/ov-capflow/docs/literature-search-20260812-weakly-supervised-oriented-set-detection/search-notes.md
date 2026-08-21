# Search Notes

## Research question

Can the OpenRSD→OV-CapFlow lessons support a novel, credible remote-sensing framework that learns OBBs from full RBox, single points, or HBoxes, especially through fixed-query set prediction, cross-view ownership, and safe negative supervision?

## Search date and cutoff

- Executed: 2026-08-12 (Asia/Shanghai)
- Coverage emphasis: 2022–2026, with older foundational set-prediction work considered only when necessary
- Sources: official conference proceedings, OpenReview/arXiv author manuscripts, AAAI proceedings, IEEE DOI metadata
- Exclusion: MDPI results were not used; secondary aggregators were not used as evidence when a primary source was available

## Query families

- `horizontal box supervised oriented object detection`
- `H2RBox H2RBox-v2 symmetry rotate flip`
- `point supervised oriented object detection dense layout`
- `PointOBB Point2RBox v2`
- `partial weakly supervised oriented object detection`
- `weakly supervised oriented DETR set prediction`
- `horizontal box supervised transformer oriented detection`
- `rotated detection transformer query matching denoising`
- `ambiguity unmatched query negative supervision objectness`
- `rotation equivariant oriented object detector`

## Screening ledger

| Candidate family | Screen decision | Reason |
|---|---|---|
| H2RBox / H2RBox-v2 | Include | Direct HBox→RBox and symmetry baselines |
| BGHR / ABBSPO | Include | Current HBox sample mining and box-quality/symmetry boundary |
| PointOBB / Point2RBox families | Include | Direct collision for point + multi-view/layout ideas |
| Wholly-WOOD / PWOOD | Include | Direct collision for unified or partial weak annotations |
| Relational Matching | Include | Closest point-ambiguity + cross-model relational matching work |
| AO2-DETR / RHINO | Include | Direct full-supervised set matching/query denoising parent context |
| FRED | Include as adjacent | Rotation equivariance claim boundary |
| Open-World Objectness Modeling | Include as adjacent | Negative/background treatment under incomplete labels |
| Generic horizontal detectors | Exclude | Do not address OBB weak supervision |
| Generic semi-supervised DETR | Exclude from final set | Useful background but insufficiently specific after stronger matches |
| Non-primary blogs/repositories | Exclude as evidence | Retained only for navigation when official paper was available |
| MDPI articles | Exclude | User/skill evidence policy |

### Item-level screening count

Twenty-five concrete papers were screened; 15 were retained in the final set. The ten additional items below were useful for boundary checking but were not promoted because a more direct paper already occupied their role.

| ID | Paper | Decision | Reason |
|---:|---|---|---|
| 16 | [ARS-DETR](https://arxiv.org/abs/2303.04989) | Exclude from final | Important full-RBox transformer, but RHINO is a closer matching/denoising neighbor for the proposed method |
| 17 | [O2DETR](https://arxiv.org/abs/2106.03146) | Exclude from final | Foundational rotated transformer; AO2-DETR gives a closer one-to-one matching formulation |
| 18 | [RotaTR](https://arxiv.org/abs/2312.02821) | Exclude from final | Dense/rotated attention context, but does not address weak HBox supervision |
| 19 | [EIE-Det](https://doi.org/10.1109/TETCI.2024.3398020) | Exclude from final | Box equivariance is relevant, but H2RBox-v2/FRED more directly define the symmetry boundary |
| 20 | [AFWS](https://doi.org/10.1109/TGRS.2024.3485590) | Exclude from final | Angle-free HBox supervision is relevant; displaced by stronger direct HBox baselines in the 15-paper cap |
| 21 | [WSODet](https://doi.org/10.1109/TGRS.2023.3247578) | Exclude from final | Image-level/proposal-style weak supervision is less aligned with complete HBox instance ownership |
| 22 | [Knowledge Combination To Learn Rotated Detection Without Rotated Annotation](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_Knowledge_Combination_To_Learn_Rotated_Detection_Without_Rotated_Annotation_CVPR_2023_paper.html) | Exclude from final | Synthetic knowledge route is represented more directly by Point2RBox for the planned comparison |
| 23 | [Omni-DETR](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Omni-DETR_Omni-Supervised_Object_Detection_With_Transformers_CVPR_2022_paper.html) | Exclude from final | Generic omni-supervised horizontal detection; useful background but not oriented-HBox specific |
| 24 | [Semi-DETR](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Semi-DETR_Semi-Supervised_Object_Detection_With_Detection_Transformers_CVPR_2023_paper.html) | Exclude from final | Generic semi-supervised DETR; weaker fit than oriented relational matching |
| 25 | [Scaling Novel Object Detection With Weakly Supervised Detection Transformers](https://openaccess.thecvf.com/content/WACV2023/html/LaBonte_Scaling_Novel_Object_Detection_With_Weakly_Supervised_Detection_Transformers_WACV_2023_paper.html) | Exclude from final | Relevant to OV negatives, but Open-World Objectness is newer and more actionable for unmatched queries |

## Search conclusions

### Confirmed occupied territory

1. HBox-supervised orientation through augmentation symmetry.
2. Point-supervised orientation/scale through multi-view consistency or pseudo boxes.
3. Dense point-supervised instance layout through Gaussian/Voronoi constraints.
4. Unified use of point/HBox/RBox and partial weak annotation mixtures.
5. Full-supervised rotated DETR matching and query denoising.

### Needs continued search

1. Whether any HBox-supervised rotated DETR explicitly marginalizes a distribution over the HBox-consistent OBB feasible set.
2. Whether any set detector carries annotation identity across group-transformed views as an ownership track rather than only prediction consistency.
3. Whether weakly supervised OBB work separates safe-background negatives from ambiguous local contenders in unmatched queries.
4. Whether a 2025–2026 journal/preprint combines all three above even if it uses different terminology.

## Claim policy

- Allowed now: “we did not find an exact match in this date-bounded search.”
- Not allowed now: “first,” “the first HBox-supervised DETR,” or “no prior work.”
- A first-claim would require backward/forward citation chasing from H2RBox-v2, ABBSPO, BGHR, Wholly-WOOD, PWOOD and RHINO; author/code repository inspection; and a second independent search pass.
