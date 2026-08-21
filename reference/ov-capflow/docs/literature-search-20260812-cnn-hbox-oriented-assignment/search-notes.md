# Search Notes — CNN HBox-Supervised Oriented Assignment

## Working scope

- 主题边界：完整 HBox instance annotations 训练 CNN oriented detector；重点是 dense sample/label assignment、rotation/flip equivariance 和 ambiguous background gradients。
- 时间窗：2021–2026 为主；FCOS 2019、ATSS 2020 作为必要基础例外。
- 来源优先级：官方会议 proceedings / 出版社或 DOI / OpenReview / 作者 arXiv。
- 目标数量：12–18 篇可追溯核心文献；最终保留 18 篇。
- 输出：核心表、CSV 元数据、检索/筛选账本和 novelty boundary。

## Keyword plan

### 中文关键词组

- `水平框监督 旋转目标检测 样本分配`
- `遥感 旋转检测 动态标签分配`
- `水平框 弱监督 旋转等变 一致性`
- `密集小目标 正负样本 归属`
- `旋转目标检测 软分配 最优传输`

### English keyword groups

- `horizontal box supervised oriented object detection sample assignment`
- `HBox supervised rotated detection adaptive fine-grained sample mining`
- `rotation flip equivariance weakly supervised oriented detection`
- `dense object detection soft instance-wise distribution sampling`
- `oriented tiny object detection dynamic label assignment`
- `partial sparse weakly supervised oriented detection hard negatives`

### Venue-aware combinations

- `site:ojs.aaai.org HBox supervised oriented object detection`
- `site:openaccess.thecvf.com oriented object detection label assignment`
- `site:papers.neurips.cc horizontal box supervised oriented detection`
- `site:openreview.net sparse partial weakly supervised oriented detection`
- `site:arxiv.org RTMDet dynamic soft label assignment`

## Retrieval passes

### Pass 1: direct HBox supervision

Seeds: H2RBox, H2RBox-v2, BGHR, ABBSPO, EIE-Det, Wholly-WOOD.

Purpose: identify whether orientation consistency, symmetry, scale correction, box equivariance, sample mining, and mixed-quality labels are already occupied.

### Pass 2: dense assignment foundations

Seeds: FCOS, ATSS, OTA, IQDet, DCFL, RTMDet, TS-Conv, Oriented RepPoints.

Purpose: prevent generic claims around adaptive, dynamic, probabilistic, distributional, global, or task-aligned assignment.

### Pass 3: adjacent supervision and negative safety

Seeds: Point2RBox-v2, PWOOD, SPWOOD, PointOBB/PointOBB-v2, Open-World Objectness Modeling.

Purpose: distinguish complete-HBox interior ambiguity from point-layout uncertainty and missing-instance annotations.

### Pass 4: claim collision search

Queries targeted phrases such as `cross-view label assignment`, `assignment equivariance`, `consistent sample assignment`, and `soft instance ownership` without including private project names or full proposed formulations.

Result: no exact primary-source match was found for transformation-aligned soft location–instance assignment plus HBox-interior ambiguity-safe negatives. This remains a search result, not a novelty verdict.

## Screening ledger

Twenty-six concrete papers were screened; 18 were retained.

| ID | Candidate | Decision | Reason |
|---:|---|---|---|
| 1 | H2RBox | Include `S` | Foundational HBox→RBox CNN consistency |
| 2 | H2RBox-v2 | Include `S` | Direct parent; rotate/flip symmetry |
| 3 | BGHR | Include `S` | Closest HBox sample-assignment competitor |
| 4 | ABBSPO | Include `S` | Scale mismatch and symmetric-prior boundary |
| 5 | EIE-Det | Include `S` | Explicit/implicit box equivariance boundary |
| 6 | Wholly-WOOD | Include `S` | Unified diversified-label boundary |
| 7 | OTA | Include `A` | Global transport and crowded assignment |
| 8 | IQDet | Include `A` | Instance-wise probabilistic quality distribution |
| 9 | DCFL | Include `A` | Oriented tiny-object dynamic assignment |
| 10 | TS-Conv | Include `A` | Task-wise sampling and dynamic assignment |
| 11 | FRED | Include `A` | Full rotation-equivariance claim boundary |
| 12 | Point2RBox-v2 | Include `A` | Dense-layout and local-exclusivity boundary |
| 13 | PWOOD | Include `A` | Partial weak annotation + unlabeled data boundary |
| 14 | SPWOOD | Include `A` | Sparse weak annotation and false-background boundary |
| 15 | Oriented RepPoints | Include `B` | Oriented quality/sample-assignment precedent |
| 16 | RTMDet | Include `B` | Secondary CNN parent and dynamic soft labels |
| 17 | ATSS | Include `B` | Foundational adaptive sample selection |
| 18 | FCOS | Include `B` | Foundational dense location target framework |
| 19 | PointOBB | Exclude from final | Point route represented more recently by Point2RBox-v2; retained in previous broad package |
| 20 | PointOBB-v2 | Exclude from final | Strong point baseline but less direct to complete-HBox assignment |
| 21 | Point2RBox | Exclude from final | Synthetic-pattern point route; v2 is closer to dense-layout concern |
| 22 | MCL | Exclude from final | Semi-supervised RBox setting; useful but weaker fit than SPWOOD/PWOOD |
| 23 | Open-World Objectness Modeling | Exclude from final | Incomplete/open-world negatives differ from fully annotated HBox setting |
| 24 | OASL | Exclude from final | Full-supervised orientation-aware sampling; TS-Conv/DCFL give stronger boundary coverage |
| 25 | Metric-aligned Sample Selection | Exclude from final | Full-RBox sampling; displaced by DCFL/TS-Conv in the 18-paper cap |
| 26 | TOOD | Exclude from final | Generic task alignment; OTA/IQDet/RTMDet cover the required dense-assignment foundations more directly |

## Verification notes

- BGHR facts were checked on the AAAI page and official PDF: AFSM, PRA/KLD, single best predicted RBox per GT HBox, and no added inference overhead.
- H2RBox-v2 facts and DOTA/HRSC/FAIR1M numbers were checked on the NeurIPS proceedings page.
- ABBSPO was verified on the CVPR proceedings PDF/supplement; the official page is preferred over its later arXiv upload.
- EIE-Det title, authors, journal volume/pages, and DOI were cross-checked through DOI/DBLP metadata.
- PWOOD has an official CVPR 2026 proceedings page and arXiv record; SPWOOD has an ICLR 2026 OpenReview paper.
- RTMDet is retained as an arXiv/infrastructure reference and is not graded as a peer-reviewed core paper.
- No unverified DOI was guessed; blank DOI fields in CSV remain blank.

## Main synthesis

The literature does not support a broad claim around `dynamic assignment`, `soft sampling`, `rotation consistency`, `box equivariance`, or `negative-safe weak supervision`. The surviving question is relation-specific:

> Is the location–instance assignment distribution itself equivariant across known image transformations, and can its uncertainty identify when a location inside a complete HBox should not be trained as hard background?

## Claim policy

- Allowed: `In a date-bounded primary-source search, we did not find an exact formulation that makes transformation-aligned dense assignment distributions and HBox-interior negative weighting the joint center of the method.`
- Not allowed: `first`, `no prior work`, `first HBox label assignment`, `first soft assignment`, or `first rotation-equivariant detector`.
- Before submission: perform backward/forward citation chasing from H2RBox-v2, BGHR, ABBSPO, EIE-Det, Wholly-WOOD, OTA and IQDet; inspect author repositories; repeat search independently.

