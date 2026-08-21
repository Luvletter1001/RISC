# DOTA-v2 Rotated-IoU Position Supervision Design

## 1. Status and purpose

This design was approved on 2026-07-22 as **A1: target-only rotated-IoU
position supervision**. It defines the smallest causal quality/ranking
intervention justified by the canonical T7 Epoch-24 Q600 evidence.

The intervention changes only the classification targets of Hungarian-matched
positive queries. It does not change matching, regression, denoising,
inference, the fixed query mouth, or the data recipe.

The purpose is to test one falsifiable hypothesis:

> In the current fixed-Q600 open-vocabulary rotated detector, part of the raw
> AP deficit is caused by classification scores that are insufficiently
> aligned with rotated localization quality. Supervising each matched
> positive token target with detached aligned rotated IoU should improve
> score ordering without reducing geometric reachability.

This is an experiment design, not a claim that a single loss change can close
the full AP70 gap. Implementation and training remain separately gated.

## 2. Evidence that triggered the design

The authoritative E24 dump is the raw DOTA-v2.0 validation mouth:

- 13,833 unique images, including empty-GT tiles;
- 600 rows per image and 8,299,800 finite rows;
- exact reconstructed mAP `0.6064053488274416`;
- official `dota/mAP=0.6064` and `dota/AP50=0.6060`;
- evaluator-TP score--rotated-IoU Spearman `0.36542497075651204`;
- fixed-recall perfect-ranking mAP `0.7828282929129071`;
- AP70 consumes `53.0513%` of perfect-ranking headroom;
- AP-support FPs are `64.6339%` localization/background, `17.6231%`
  empty-tile, `15.3023%` duplicate, and `2.4407%` semantic;
- GT states are `31.4987%` geometry miss, `1.0015%` semantic miss,
  `0.0045%` ownership miss, and `67.4952%` evaluator reachable.

The final analyzer manifest is
`bd1ce0a7b98a9e03f4b647fe194290bab4c229225e53304c270b2fc3d4166dfe`.
The canonical report is
`work_dirs/dotav2_cleanstart/eval_t7_epoch24_raw13833_gpu2389_dump/diagnostics/report.md`.

Five of six independent routing perspectives ranked quality/ranking first.
The dissenting tiny/dense perspective remains the explicit fallback because
geometry misses are still large, especially for <=8-pixel and >600-GT
strata. The evidence selects the first family to test; it does not establish
practical sufficiency.

The design is mechanism-compatible with:

- [Detection Transformer with Stable Matching, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Detection_Transformer_with_Stable_Matching_ICCV_2023_paper.html),
  which uses positional quality to supervise positive classification scores;
- [Rank-DETR, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/34074479ee2186a9f236b8fd03635372-Abstract-Conference.html),
  which studies localization-aware classification and ranking;
- [Align-DETR, BMVC 2024](https://bmvc2024.org/proceedings/211/),
  which confirms the classification--regression alignment problem but adds
  joint confidence, ranking, and many-to-one mechanisms that are excluded
  from this first intervention.

The cited work supplies a prior, not an expected DOTA-v2 effect size or a
novelty claim.

## 3. Frozen scientific contract

The A1 candidate preserves all of the following:

- DOTA-v2.0 canonical 18-class order and base14/novel4 split;
- raw 13,833-image final validation, including empty-GT tiles;
- fixed Q600 train and inference mouth;
- exactly one selected class and one score per query row;
- no NMS, rotated NMS, score threshold, top-k, or row deletion;
- 5-D `(cx, cy, w, h, angle)` rotated boxes;
- the current three independently matched training query groups;
- the current six decoder layers and their auxiliary losses;
- the current 100-query DN branch with its original hard/fractional targets;
- the current `mmdet.FocalLoss`, regression L1 loss, and KLD/GD loss;
- the current BinaryFocal + rotated L1 + KLD Hungarian costs;
- the scale-1024, rare4x, clean-start, optimizer, scheduler, batch, sampler,
  prompt, seed, and evaluator recipes;
- the current inference token aggregation and score readout;
- physical GPUs restricted to `2,3,8,9` for any later authorized run.

The sole scientific delta is the positive classification target after the
existing Hungarian assignment.

## 4. Position-supervised target

For matched prediction `i`, let:

- `b_i` be its predicted rotated box at the current decoder layer;
- `g_i` be the normalized target returned by the existing assignment;
- `Y_i` be the existing Grounding DINO token positive map;
- `q_i` be their aligned rotated IoU in pixel/radian coordinates.

The target is:

```text
q_i = stop_gradient(clamp(rotated_iou(b_i, g_i), 0, 1))
Y_i_quality = q_i * Y_i
```

Unmatched targets remain exactly zero. `label_weights`, `bbox_targets`,
`bbox_weights`, `pos_inds`, `neg_inds`, normalization factors, and the
assignment itself remain unchanged.

The original token positive map is normalized across the tokens belonging to
one entity. Multiplying the full map by `q_i` preserves its token proportions
and scales its original normalized positive mass by `q_i`. Setting every
positive token directly to `q_i` is forbidden because it would make the
intervention depend on the number of tokens in a class name.

The first causal experiment uses identity rotated IoU. It has no quality
floor, exponent, epsilon rescaling, confidence term, curriculum, learnable
coefficient, or class-dependent weight. Those additions would introduce
unidentified degrees of freedom.

## 5. Code boundary and data flow

The implementation is local to
`projects/OVCapFlow/ov_capflow/ov_capflow_head.py`.

`OVCapFlowHead` receives one opt-in constructor field:

```text
position_supervised_cfg = {enabled: bool}
```

The default is disabled. Unknown keys are rejected. The enabled configuration
is incompatible with `balanced_cfg.enabled=True`; construction must fail
early rather than silently combine two classification reweighting mechanisms.

The minimal integration point is an `OVCapFlowHead._get_targets_single()`
override:

1. Call the parent method once to perform the unchanged Hungarian assignment
   and construct all original targets.
2. Return immediately when the feature is disabled or no positive exists.
3. Convert only matched predicted and target boxes to pixel/radian coordinates
   using `[image_width, image_height, image_width, image_height,
   angle_factor]`.
4. Compute aligned rotated IoU in float32 with
   `rbbox_overlaps(..., is_aligned=True)`.
5. Explicitly detach and clamp quality to `[0,1]`.
6. Fail on non-finite quality; do not repair it with `nan_to_num`.
7. Multiply the existing positive-token rows by their aligned quality and
   return every other target unchanged.

This location is preferred over rewriting `loss_by_feat_single()` because it
does not repeat Hungarian assignment, fork loss normalization, or duplicate
the parent regression path. It is preferred over a config-only
`QualityFocalLoss` swap because both Grounding DINO matching and DN paths
explicitly reject that loss and its negative/reduction semantics would be a
second change.

The runtime data flow is:

```text
decoder layer outputs
  -> existing per-group Hungarian assignment
  -> original token/bbox targets
  -> A1 scales matched token targets by detached aligned rotated IoU
  -> unchanged FocalLoss + unchanged L1/KLD losses
```

## 6. Included and excluded supervision paths

### Included

- **All six main decoder layers.** Each layer uses its own prediction,
  assignment, and aligned quality. Restricting A1 to only the final layer
  would mix hard and quality-aware classification objectives in shared
  decoder parameters.
- **All three matching groups.** Each group remains independently matched and
  the three loss dictionaries remain arithmetically averaged. Applying A1 to
  only one group would introduce asymmetric supervision.

### Excluded

- **DN and adaptive DN.** DN uses separate target builders. Its label-noise
  reconstruction curriculum remains unchanged; adding predicted quality to
  DN would be a second mechanism.
- **Encoder outputs.** OVCapFlow's fixed-query `pre_decoder()` sets encoder
  class and coordinate outputs to `None`. A contract test must prevent future
  configs from silently expanding the A1 scope.
- **Hungarian costs.** Binary focal, rotated L1, and KLD costs are unchanged.
- **Regression.** L1 and KLD/GD targets, weights, and losses are unchanged.
- **Inference.** Q600 output shape, token-to-class aggregation, calibration,
  labels, scores, and boxes are unchanged.

The final matching-mask cache may call the same target builder. A1 may change
the returned labels in that read-only call, but it must not change the cached
assignment mask or any indices/weights.

## 7. Error handling and invariants

The implementation must fail fast on:

- invalid or unknown `position_supervised_cfg` values;
- simultaneous position supervision and balanced classification;
- non-finite aligned IoU;
- rotated target shapes other than five coordinates;
- mismatch between positive indices and aligned quality count;
- an enabled run that unexpectedly supplies encoder output supervision.

Near-degenerate positive widths/heights use the existing `rbbox_overlaps`
minimum-size handling. Quality is computed in float32 even under mixed
precision, then cast to the label dtype. No exception may be converted into a
zero-quality target silently.

When disabled, the method must return the parent targets tensor-for-tensor
without running the rotated-IoU operation. This is the parent-equivalence
contract and rollback path.

## 8. Test-driven verification

Implementation begins with failing tests. The minimum suite covers:

1. Disabled mode returns every target exactly equal to the parent behavior.
2. A perfectly overlapping positive retains its original token positive map.
3. A known partial overlap multiplies every original positive-map value by
   the exact aligned rotated IoU while preserving token ratios.
4. Unmatched token targets stay zero and pos/neg indices stay unchanged.
5. Empty-GT and zero-positive inputs preserve shapes and finite outputs.
6. Quality targets are detached; classification backward cannot send a
   gradient through the IoU target into box predictions.
7. All three grouped branches and all six layers receive the same target rule,
   while group loss aggregation remains an arithmetic mean.
8. DN targets and fixed-input DN classification loss are identical with A1
   disabled and enabled.
9. The live config preserves encoder outputs `None`, groups=3, Q600,
   `balanced_cfg.enabled=False`, FocalLoss, and all matching costs.
10. Inference output retains exactly 600 finite scores, labels, and 5-D boxes
    per image with no row sorting or deletion inside the head.
11. Near-degenerate rotated boxes produce finite clamped quality; deliberately
    non-finite input fails instead of being repaired.
12. A real-batch forward/backward smoke has finite total/matching/DN losses,
    finite gradients, and no unexpected state-dict key changes.

The focused suite must pass before the broader OVCapFlow project tests. A
config build and zero-update checkpoint audit must pass before any optimizer
step is authorized.

## 9. Experiment stages and gates

### Stage 0: implementation and zero-update audit

Stage 0 may build the model and run tests but may not start formal training.
It must establish:

- default-disabled parent equivalence;
- exact candidate config diff containing only the A1 flag and bookkeeping;
- unchanged checkpoint load coverage and state-dict shapes;
- unchanged inference mouth and official evaluator configuration;
- finite real-batch forward/backward behavior;
- no live training or evaluation process and allowed GPUs only.

Failure closes the candidate until the implementation is corrected and
reviewed. It never triggers a reduced or silently modified recipe.

### Stage 1: matched 1,600/400 proxy

Only after explicit run authorization, train A1 from the same generic Q600
clean-start checkpoint on the immutable scale-1024 rare4x proxy used by
`8-T6-R`. Preserve its train/validation manifests, seed `20260716`, 12-epoch
endpoint, effective batch, sampler, optimizer, and evaluator.

`8-T6-R-E12` may be reused as the control only after a hash audit proves that
the candidate differs solely by A1 and bookkeeping. Otherwise a paired
control must be run; a near-match is not sufficient.

The proxy has three decision layers:

1. **Integrity gate:** strict OV/rotated/E2E/Q600/no-NMS/no-top-k audit passes;
   no invalid loss, gradient, sampler, or resource event occurs.
2. **Metric gate:** endpoint AP50 is at least `0.4790`, a `+0.0100` margin over
   the immutable `0.4690` control; novel4 is at least `0.3675`; base14 is at
   least `0.464643`. Selection uses Epoch 12 only, never the best checkpoint.
3. **Mechanism gate:** on identical proxy dumps, evaluator-TP score--rotated-
   IoU Spearman improves by at least `+0.03`, total recall does not fall by
   more than `0.005`, and the fixed-recall ranking headroom shrinks rather
   than grows.

All three gates are conjunctive. A proxy miss discards A1 as the first
intervention. It does not authorize a floor, exponent, cost modulation, DN
quality target, extra epoch, alternate seed, or candidate stack.

### Stage 2: matched full-data raw run

Only a Stage-1 pass may request a full-data run. Start from the same generic
Q600 checkpoint as T7 and reproduce the scale-1024 rare4x 24-epoch recipe,
changing only A1 and bookkeeping. Do not fine-tune E24 as the primary causal
test: position supervision is a training objective and an E24 continuation
would not test its full optimization path.

Evaluate scheduled immutable checkpoints using the raw 13,833-image mouth.
Epoch 24 is the selection authority; intermediate checkpoints are trajectory
evidence only. Save the final all-Q600 dump and analyze it with the canonical
analyzer.

Outcomes are classified as:

- **AP70 complete:** both raw official `dota/mAP` and `dota/AP50` are at least
  `0.7000`, with all integrity gates passing.
- **Mechanism-positive but incomplete:** both official metrics improve by at
  least `+0.0100` over E24, TP score--IoU Spearman improves by at least
  `+0.03`, total recall does not fall by more than `0.005`, and neither
  base14 nor novel4 falls by more than `0.005`; AP70 is still not claimed.
- **Discard:** integrity fails, either official metric regresses, the ranking
  mechanism gate fails, or group/recall safety limits fail.

A mechanism-positive result may justify a separately reviewed tiny/dense
geometry intervention. It does not authorize stacking in the same run. A
paper-level effect claim requires an independently preregistered repeat after
the first full-data result; the repeat is not automatic.

## 10. Reporting requirements

Every proxy and raw report records:

- resolved config and scientific-diff hashes;
- source checkpoint, dataset manifests, seed, sampler, GPU IDs, world size,
  effective batch, optimizer steps, and wall time;
- official total mAP/AP50 and all 18 class AP/recall values;
- base14 and novel4 summaries;
- TP score--rotated-IoU Spearman and score-bin calibration;
- exact AP-support FP taxonomy;
- geometry/semantic/ownership/reachable GT decomposition;
- fixed-recall oracle and consumed/remaining headroom;
- DN losses separately from matching losses;
- every warning, retry, interruption, and resume boundary.

## 11. Explicit non-goals

A1 does not include:

- position-modulated or high-order matching cost;
- QualityFocalLoss, VarifocalLoss, or a new classification head;
- confidence--IoU joint targets;
- an IoU floor, square, square root, rescaling, or learned transform;
- DN quality targets or adaptive DN changes;
- additional matching groups or many-to-one supervision;
- query rank layers, content-query transport, or decoder-head transport;
- dynamic queries, density/counting heads, point heads, or proposal top-k;
- NMS, score filtering, top-k, per-class row limits, or second-stage heads;
- prompt aliases, prompt sweeps, or text-encoder retraining;
- E25+ continuation of the baseline;
- combining A1 with a geometry, semantic, initialization, or prompt change;
- claiming the published Stable/Rank/Align mechanisms as original work;
- claiming AP70 before both raw official metrics reach `0.7000`.

## 12. Rollback and fallback

Rollback is configuration-only: disable `position_supervised_cfg`. The default
path must remain parent-equivalent and checkpoints must retain the same state
dict because A1 introduces no learned parameter.

If A1 fails its proxy or raw mechanism gate, close target-only quality
supervision as the first intervention. Preserve the result as negative causal
evidence and route next to the separately designed fixed-Q600 tiny/dense
geometry family. Do not rescue A1 by adding cost modulation or hyperparameter
sweeps inside this experiment.

## 13. Definition of done

The design is implemented only when:

- all TDD and project regression tests pass;
- an independent code review finds no critical or important issue;
- default-disabled parent equivalence is demonstrated;
- the candidate config has exactly one scientific delta;
- Stage-0 zero-update and real-batch audits pass;
- the written run command and promotion gates are reviewed before launch.

The research goal is complete only when a canonical raw run records both
`dota/mAP>=0.7000` and `dota/AP50>=0.7000` under the unchanged Q600 rotated
end-to-end mouth. Passing this implementation specification alone is not
completion and does not authorize training.
