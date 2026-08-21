# Stable Strict-E2E Route After P15-SCOD Smoke

Date: 2026-06-23

## Scope

This memo studies how to make the strict end-to-end oriented detector more stable after the first P15-SCOD HRRSD overfit run.

Constraints remain strict:

- final inference must directly output a fixed set of `rbox + cls posterior`;
- no traditional dense detection head as final output;
- no NMS;
- no score-threshold or per-class quota fallback;
- support fusion must enter query, memory, attention, or posterior formation, not post-processing.

## Current P15-SCOD Observation

The first HRRSD train20 overfit run finished successfully but did not pass the overfit gate.

- Best epoch: epoch 19, `mAP=0.0086`, `AP50=0.0090`.
- Final epoch: epoch 20, `mAP=0.0013`, `AP50=0.0010`.
- Regression loss decreased strongly, so the geometry path is learning.
- AP is unstable and very low, so the main failure is posterior/query-label stability, not only box geometry.

Code-level risk in the current head:

- `P15OrientedDINOSetHead` uses per-class sigmoid focal targets with no explicit `background/no-object` class.
- Inference uses `sigmoid().max()` as the final score and class.
- There is no separate objectness posterior.
- Matching cost uses sigmoid class probability, so early matching can be unstable when all queries are poorly calibrated.

## Literature Signals

### DETR

DETR frames detection as direct set prediction with bipartite matching and removes NMS and anchor generation. This is the correct conceptual family for our strict E2E requirement.

Source: https://arxiv.org/abs/2005.12872

Design lesson:

- Keep one-to-one set prediction as the final inference contract.
- Do not rescue with NMS or threshold fallback.
- Add an explicit no-object/background modeling path; DETR-style set prediction expects unmatched queries to learn "not an object" rather than being silent all-zero multi-label vectors.

### Deformable DETR

Deformable DETR addresses slow convergence and small-object difficulty by attending to sparse sampling points around reference points and using multi-scale features.

Source: https://arxiv.org/abs/2010.04159

Design lesson:

- The rotated deformable decoder is the right backbone for aerial/HRRSD objects.
- Query references must be high quality; bad references waste decoder capacity.

### DN-DETR

DN-DETR identifies early unstable bipartite matching as a major cause of slow convergence and adds denoising reconstruction from noisy ground-truth boxes.

Source: https://arxiv.org/abs/2203.01305

Design lesson:

- Rotated DN queries should not be ignored in P15 loss. They need explicit DN classification, box, angle, and geometry losses.
- DN positives are training-only and do not violate strict E2E inference.

### DAB-DETR

DAB-DETR treats box coordinates as dynamic queries and updates them layer by layer, turning DETR queries into an explicit positional prior.

Source: https://arxiv.org/abs/2201.12329

Design lesson:

- P15 support fusion should condition both query content and query geometry/reference.
- Support-only additive content conditioning is too weak if geometry references remain poorly calibrated.

### DINO

DINO combines improved denoising, mixed query selection for anchor initialization, and look-forward-twice box prediction.

Source: https://arxiv.org/abs/2203.03605

Design lesson:

- The next P15 implementation should be closer to DINO's full training recipe, especially DN loss accounting and mixed query selection.
- Current P15 only uses part of this recipe, which explains the weak posterior.

### RT-DETR

RT-DETR keeps the end-to-end no-NMS benefit while improving practicality through efficient hybrid encoding and uncertainty-minimal query selection.

Source: https://arxiv.org/abs/2304.08069

Design lesson:

- For HRRSD, use uncertainty-minimal/top-quality query selection as initializer, but keep it as initializer only.
- Do not convert it into dense final detections.

### H-DETR / Group DETR / Co-DETR

These works show the same bottleneck from different angles: pure one-to-one matching supplies too few positives, so training is sparse and unstable. They add train-time one-to-many or auxiliary supervision while preserving one-to-one inference.

Sources:

- H-DETR: https://arxiv.org/abs/2207.13080
- Group DETR: https://arxiv.org/abs/2207.13085
- Co-DETR: https://arxiv.org/abs/2211.12860

Design lesson:

- Training-only extra positives are a proven way to stabilize E2E DETR.
- Under the user's strict requirement, avoid traditional dense auxiliary heads for now.
- A safer variant is Group-DETR-style query groups: multiple one-to-one groups during training, one group at inference. This adds supervision without a traditional dense head or NMS.

### D2Q-DETR

D2Q-DETR is directly about oriented object detection with transformers. It removes rotated NMS/RRoI heuristics, decouples classification and regression features, uses dynamic queries, and revisits label assignment.

Source: https://arxiv.org/abs/2303.00542

Design lesson:

- Decouple classification query features and regression query features in P15.
- Dynamic query pruning/re-weighting is useful for dense aerial scenes, but the final output must remain a fixed query set for now.

### OrientedFormer

OrientedFormer targets end-to-end oriented detection in remote sensing and highlights three oriented-specific issues: angle encoding, missing geometric relations in self-attention, and cross-attention misalignment. It uses Gaussian positional encoding, Wasserstein self-attention, and oriented cross-attention.

Source: https://arxiv.org/abs/2409.19648

Design lesson:

- The Gaussian idea should move from "standalone Gaussian posterior head" to "Gaussian query geometry encoding and relation bias".
- P15 v2 should not only apply GWD as a loss; it should let Gaussian geometry affect query attention or class/regression posterior formation.

### GWD / KFIoU

GWD maps rotated boxes to 2D Gaussian distributions to reduce boundary discontinuity and keep useful gradients when boxes do not overlap. KFIoU uses Gaussian product to approximate SkewIoU behavior in a differentiable way.

Sources:

- GWD: https://arxiv.org/abs/2101.11952
- KFIoU: https://arxiv.org/abs/2201.12558

Design lesson:

- Keep Gaussian geometry loss, but do not rely on it to solve query-class posterior collapse.
- For posterior stability, Gaussian product should contribute to quality/objectness, not replace set prediction.

## Diagnosis

The current P15 implementation is strict E2E, but it is not yet a stable DETR-style detector.

Root causes likely ranked by importance:

1. No explicit `background/no-object` class. Unmatched queries are trained as all-zero sigmoid vectors, but inference still forces every query to pick a foreground class by max sigmoid.
2. No objectness branch. Classification and object existence are entangled, so high false positives and low ranking stability are expected.
3. DN losses are not separated and may not supervise denoising queries correctly.
4. Support conditioning is only additive query content modulation, not geometry/reference modulation.
5. Classification and regression share the same hidden feature path; D2Q-DETR suggests decoupling helps oriented precision.
6. One-to-one positives are too sparse for a 20-image smoke run; query groups or DN positives are needed before scaling.

## Recommended Next Route: P15B-SCOD

P15B should be a stability patch, not a new model family.

### Change 1: Explicit Background Set Classification

Replace 13-way sigmoid focal posterior with a DETR-style `num_classes + 1` softmax CE path.

Training:

- matched queries target real class `0..12`;
- unmatched queries target background `13`;
- class loss normalized by number of positives plus controlled background weight.

Inference:

- foreground posterior is `softmax(logits)[..., :num_classes]`;
- background is ignored for final labels;
- still output the fixed query set;
- no score threshold.

Expected effect:

- reduces forced foreground collapse;
- gives unmatched queries a stable target;
- makes Hungarian class cost cleaner.

### Change 2: Query Objectness Posterior

Add a one-logit objectness branch per query.

Training:

- matched queries target objectness `1`;
- unmatched queries target objectness `0`;
- DN positives target `1`;
- use BCE/focal objectness with mild weight.

Inference:

- final score = `objectness.sigmoid() * foreground_softmax.max(-1)`;
- label = foreground argmax;
- still no threshold and no NMS.

Expected effect:

- separates "is there an object" from "which class";
- improves AP ranking without post-processing.

### Change 3: Proper Rotated DN Loss Split

Do not `del dn_meta`. Split decoder outputs into DN and matching segments, then compute:

- `loss_dn_cls`;
- `loss_dn_obj`;
- `loss_dn_bbox`;
- `loss_dn_angle`;
- `loss_dn_gwd` or `loss_dn_kfiou`.

Expected effect:

- stabilizes early training by giving known positive query-box pairs;
- directly addresses DN-DETR's matching instability diagnosis.

### Change 4: Group-DETR-Style Training-Only Query Groups

Use 2-4 query groups during training, one group during inference.

Constraints:

- each group uses one-to-one matching;
- groups share weights;
- self-attention mask prevents cross-group leakage if needed;
- inference keeps only the primary group.

Expected effect:

- more positive supervision without dense heads;
- stricter than Co-DETR because it avoids ATSS/Faster-RCNN auxiliary heads.

### Change 5: D2Q-Style Feature Decoupling

Split decoder hidden state into classification and regression streams:

- classification stream: support-conditioned class/objectness posterior;
- regression stream: box/angle/Gaussian geometry.

Expected effect:

- reduces class-regression gradient conflict;
- directly targets current observation: geometry improves but class posterior remains weak.

### Change 6: Gaussian Support Product As Quality Bias

Keep Gaussian modeling, but use it as a differentiable quality bias:

- convert predicted rbox to Gaussian;
- derive class-conditioned support Gaussian prototypes or learned geometry priors;
- combine class logit, objectness logit, and Gaussian quality logit before posterior.

Strict rule:

- this must be inside the forward posterior computation;
- no post-hoc reranking, NMS, or threshold fallback.

## Minimal Smoke Plan

Do not start full HRRSD training until these gates pass.

### Gate A: Contract Unit Tests

Required tests:

- `uses_nms=False`;
- `uses_dense_detection_head=False`;
- output count equals `num_queries`;
- classification logits shape is `num_classes + 1`;
- objectness shape is `[batch, queries, 1]`;
- final score equals objectness times foreground posterior;
- no threshold mask in predict;
- DN loss keys exist when `dn_meta` is present.

### Gate B: One-Image CUDA Smoke

Expected:

- no NaN;
- `loss_cls`, `loss_obj`, `loss_bbox`, `loss_angle`, `loss_gwd`, and DN losses are finite;
- background CE does not dominate by more than 5x bbox+geometry loss after the first few iterations.

### Gate C: HRRSD Train20 Overfit

Expected before mini:

- AP50 should exceed `0.05` within 20 epochs as a weak viability signal;
- target remains `AP50 >= 0.5` before any serious full experiment;
- if best AP50 stays below `0.02`, stop and inspect query assignment/score histograms.

### Gate D: Query Diagnostics

Log per epoch:

- number of matched positives;
- top-300 score histogram;
- foreground/background posterior mean;
- objectness positive/negative mean;
- class distribution of predictions;
- matched IoU/GWD histogram.

## Decision

Continue P15, but rename the next implementation as `P15B-SCOD`.

Do not return to P13C/P14 ranking-style fixes. They can produce valid plumbing but do not solve strict E2E candidate formation.

Do not use Co-DETR-style dense auxiliary heads yet. They are effective in the literature but too close to the user's forbidden "traditional dense head" boundary. If all strict query-only stabilization fails, revisit it explicitly as a training-only exception and document the tradeoff.

The immediate next coding task should be:

1. add background softmax classification;
2. add objectness posterior;
3. implement DN loss split instead of ignoring `dn_meta`;
4. add contract tests;
5. run one-image CUDA smoke;
6. rerun HRRSD train20 overfit only.

