# P15-SCOD Design: Support-Conditioned Oriented DINO

Date: 2026-06-23

## Status

Approved direction: mature DETR-style set prediction first, then inject support fusion into query initialization, encoder memory modulation, or decoder attention.

This spec is design-only. It does not authorize a training launch until the gates below pass.

## Problem

P13/P14 strict E2E experiments proved that the inference contract is implementable, but the current fixed-query posterior family is not a viable detector. The best reliable P14 mini result is `mAP=0.0048`, P14-N full is `mAP=0.0009`, and the P14-N full GT-oracle one-to-one rerank upper bound is only `mAP=0.1818`. That upper bound is far below the HRRSD 3-epoch dense baseline `mAP=0.8307`.

The failure mode is candidate generation and query-object binding, not merely score calibration. Therefore P15 must stop modifying the P14 fixed-query head family and instead reuse a mature query-set detector recipe.

## Goal

Build a strict E2E oriented set detector named `P15-SCOD`:

- final inference output is a fixed set of `rbox(cx, cy, w, h, angle) + cls posterior`;
- no traditional dense detection head as final output;
- no NMS;
- no score-threshold fallback;
- no teacher model in inference;
- support information conditions the set detector through query/memory/attention, not through post-processing.

## Non-Goals

- Do not continue P14 rank/quality/score-floor variants.
- Do not use P13C/P13C-obj as the final detector.
- Do not add NMS, per-class quota fallback, or empty-result fallback to rescue AP.
- Do not treat Gaussian product posterior as a standalone detector. Gaussian components are allowed only as geometry encoding, loss, or attention relation.
- Do not start 4GPU 3-epoch training until overfit and mini gates pass.

## Local Components

Reusable:

- `mmdet/models/detectors/dino.py`: complete DINO detector flow, including two-stage top-k query selection and denoising query integration.
- `mmdet/models/dense_heads/dino_head.py`: DINO loss split for matching/denoising and encoder proposal loss.
- `M_AD/models/layers/transformer/dinor_layersv2.py`: `DinoRTransformerDecoder` with rotated deformable decoder layers, angle refinement, and `CdnRQueryGenerator`.
- `M_AD/models/layers/transformer/deformable_detr_layers.py`: `RotatedMultiScaleDeformableAttention`.
- `M_AD/models/task_modules/assigners/hungarian_assigner.py`: `RHungarianAssigner`.
- `M_AD/models/task_modules/assigners/match_cost.py`: rotated focal, L1, IoU, and angle costs.
- HRRSD data root: `/data1/zcy/datasets/HRRSD_800_0/internal_split_20260619/`.

Not suitable as the main starting point:

- `M_AD/models/detectors/E_Rtmdet_v2_Any_to_Rotate.py`: useful as historical reference, but it mixes RTMDet, ROI prediction, and noisy GT proposals. Its inference shape is not clean enough for the strict P15 contract.

## Architecture Options

### Option A: DINO Skeleton + Rotated Decoder/Head

Start from `mmdet.models.detectors.DINO`, replace the HBB decoder/head path with rotated references and rotated losses:

```text
image/support inputs
  -> backbone + neck
  -> optional support-conditioned memory modulation
  -> DINO two-stage query initializer
  -> train-only denoising GT rbox queries
  -> rotated deformable decoder
  -> oriented set posterior head
  -> fixed rbox + cls output
```

Pros:

- Closest to DINO/Deformable DETR literature.
- Uses existing complete detector lifecycle.
- Best chance to stop the `mAP=0` loop.

Cons:

- Requires careful adaptation of box representation, assigner, losses, and result conversion.
- The DINO encoder top-k proposal path must be audited so it remains a query initializer, not a final dense head.

Recommendation: use this path.

### Option B: OrientedFormer-Lite Gaussian Query Decoder

Implement Gaussian positional encoding and Wasserstein-style query relation in a lighter rotated decoder.

Pros:

- Closest to the Gaussian-oriented idea.
- Potentially stronger novelty if it works.

Cons:

- Larger implementation risk.
- No local complete OrientedFormer detector skeleton exists.
- Bad fit for immediate smoke recovery.

Recommendation: keep as P16 only after P15 proves a working E2E oriented set baseline.

### Option C: DDQ/RT-DETR Query Selection + Strict Output

Use dense feature priors to initialize distinct queries, but final output remains a fixed set.

Pros:

- Literature suggests better recall than sparse queries.
- Good for dense aerial scenes.

Cons:

- Highest risk of crossing into dense-head semantics.
- Requires strict audit that dense priors never become final detections or NMS candidates.

Recommendation: backup route only.

## Selected Design

P15-SCOD uses Option A.

The first milestone is not support fusion. The first milestone is a clean no-support oriented DINO-style set detector that can overfit HRRSD. Support conditioning is added only after the detector proves it can generate useful oriented candidates.

### Phase 0: Clean Oriented DINO Baseline

Create a minimal `P15OrientedDINO` detector path:

- inherit the DINO detector lifecycle where practical;
- use rotated references `cx, cy, w, h, angle`;
- use train-only rotated denoising queries;
- use rotated Hungarian assignment;
- output `RotatedBoxes` predictions directly;
- expose `e2e_debug` flags proving no NMS/dense final head/threshold fallback.

### Phase 1: Support-Conditioned Query Initializer

Add a support adapter that conditions only the query/memory path:

- support descriptor from existing support/prototype branch;
- query content modulation through FiLM or additive adapter;
- optional class-conditioned query embedding bias;
- no post-hoc score multiplication by support quality.

### Phase 2: Support-Conditioned Memory or Attention

Only after Phase 1 passes gates:

- modulate encoder memory tokens with support descriptors; or
- add support-conditioned bias to decoder cross-attention; or
- add Gaussian/Wasserstein relation between rotated query references and support-conditioned class geometry priors.

## Inference Contract

The final prediction path must be:

```text
features/support -> query-set detector -> fixed set rbox + cls posterior
```

Required flags:

- `strict_e2e=True`
- `uses_nms=False`
- `uses_dense_detection_head=False`
- `uses_score_threshold_postprocess=False`
- `outputs_fixed_query_set=True`
- `teacher_training_only=True` if teacher targets are used

Forbidden calls in P15 inference:

- `batched_nms`
- rotated NMS variants
- score threshold filtering as fallback
- P13C teacher inference
- RTMDet/Retina/S2A dense final head prediction

## Losses

Core losses:

- `loss_cls`: focal or quality focal loss over matched queries.
- `loss_bbox`: L1 over normalized `cx, cy, w, h`.
- `loss_riou`: rotated IoU or KFIoU/GWD-style geometry loss.
- `loss_angle`: periodic angle loss.
- `loss_dn_cls`, `loss_dn_bbox`, `loss_dn_angle`: denoising reconstruction losses.

Optional after baseline:

- `loss_rank`: Rank-DETR-style localization-aware ranking, only if candidate oracle is strong but score ranking is weak.
- `loss_ms_o2m`: MS-DETR-style mixed supervision on primary queries, train-only.
- `loss_support_align`: support descriptor alignment, only after no-support detector passes gates.

## Gates

### Gate 1: Build and Contract

Required:

- config builds;
- forward loss on one synthetic or one real HRRSD batch;
- predict returns fixed query set;
- no NMS, no threshold fallback, no dense final head;
- teacher branch absent from inference.

No GPU training before this gate passes.

### Gate 2: 20-Image Overfit

Run on 20 HRRSD training images.

Required:

- train loss decreases;
- train AP50 becomes clearly non-zero;
- target threshold: train AP50 >= `0.5` before any full-val inference;
- predicted box centers and sizes are not template-collapsed.

If this fails, fix the detector adaptation. Do not add support fusion.

### Gate 3: Mini Candidate Oracle

Run mini validation with fixed set outputs and GT-oracle one-to-one rerank.

Required:

- oracle mAP target: at least `0.5`;
- if oracle mAP remains below `0.3`, the candidate generator is not viable.

This gate prevents another P14-style ranking-only loop.

### Gate 4: Real Mini AP

Run 300-500 iter mini training/eval.

Required:

- real AP50 target: at least `0.05`;
- if below target, stop before 4GPU.

### Gate 5: Support Safety

After adding support conditioning:

- support version must not reduce candidate oracle mAP versus no-support P15-0;
- support version must improve either real mini AP50, ranking quality, or class confusion without harming candidate recall.

## Experiment Sequence

| exp_id | purpose | action | promotion rule |
|---|---|---|---|
| P15-0 | no-support oriented DINO adaptation | implement detector and contract tests | passes Gate 1 |
| P15-0-overfit | check detector validity | 20-image HRRSD overfit | train AP50 >= 0.5 |
| P15-0-mini | check candidate generator | mini val + oracle rerank | oracle mAP >= 0.5 and real AP50 >= 0.05 |
| P15-A | support-conditioned query initializer | support FiLM/additive query adapter | no oracle degradation, real AP improves |
| P15-B | support-conditioned memory modulation | encoder memory adapter | only if P15-A weak but safe |
| P15-C | Gaussian relation attention | query/support geometry relation | only after P15-0 candidate generator is strong |

## Decision Rules

- If P15-0 cannot overfit, the implementation is wrong or the local DINO adaptation is unsuitable.
- If P15-0 overfits but oracle mAP is low, candidate generation is still weak; do not tune scores.
- If oracle mAP is high but real AP is low, then Rank-DETR/MS-DETR-style ranking supervision is justified.
- If support conditioning lowers oracle mAP, remove it or move it to a weaker adapter point.
- If real mini AP50 reaches the gate, then and only then schedule 4GPU 3e.

## Expected Implementation Touch Points

Likely new files:

- `M_AD/models/detectors/p15_oriented_dino.py`
- `M_AD/models/dense_heads/p15_oriented_dino_head.py`
- `M_configs/Diagnostics/hrrsd_p15_oriented_dino_overfit.py`
- `M_configs/Diagnostics/hrrsd_p15_oriented_dino_mini.py`
- `M_Tools/experiments/run_hrrsd_p15_overfit_20260623.sh`
- `M_Tools/analysis/oracle_rank_fixed_query_predictions.py` reuse or extension
- `tests/test_p15_oriented_dino_contract.py`

Likely reused files:

- `mmdet/models/detectors/dino.py`
- `mmdet/models/dense_heads/dino_head.py`
- `M_AD/models/layers/transformer/dinor_layersv2.py`
- `M_AD/models/task_modules/assigners/hungarian_assigner.py`
- `M_AD/models/task_modules/assigners/match_cost.py`

## Review Checklist

- No final dense detection head.
- No NMS.
- No score-threshold fallback.
- No teacher inference.
- No P14 fixed-query ranking-only continuation.
- No support fusion before no-support detector validity.
- Promotion gates are numeric and enforce stop conditions.

## Open Risk

The largest risk is adapting HBB DINO code to rotated boxes cleanly. That risk is acceptable only if the first implementation target is a no-support overfit gate. Adding support before the detector can overfit would repeat the P13/P14 failure pattern.
