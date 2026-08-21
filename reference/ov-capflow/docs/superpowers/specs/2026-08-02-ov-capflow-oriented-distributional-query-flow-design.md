# OV-CapFlow Oriented Distributional Query Flow Design

**Date:** 2026-08-02  
**Status:** approved direction; written-spec review gate  
**Target:** ICLR paper mainline with raw DOTA-v2 mAP recovery  
**Authoritative parent:** T7 Epoch 24  
**Training hardware after implementation approval:** physical GPUs 8 and 9 only

## 1. Decision

The next OV-CapFlow mainline is **Oriented Distributional Query Flow
(ODQ-Flow)**. A query carries one predictive geometric distribution through
decoder refinement, matching, and final localization-aware scoring. The
distribution is the method's shared state; matching and scoring must not grow
independent quality, objectness, proposal, or calibration branches.

This document freezes the full architectural direction and the first
implementation cycle:

1. `ODQ-S0`: a parent-equivalent distribution scaffold and five core checks;
2. `ODQ-R1`: geometry-only distributional `xywh` refinement on the existing
   T7 training and inference paths;
3. a scientific gate deciding whether the same distribution may later enter
   matching and all-row scoring.

Distribution-aware matching, localization-aware readout, circular angle
refinement, and distribution-conditioned cross-attention are deliberately not
implemented in the first cycle. They remain conditional consumers of the
same state and require a new written-spec approval after `ODQ-R1` passes.

## 2. Evidence That Determines the Design

### 2.1 Authoritative local evidence

The canonical parent is:

- config:
  `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_full24e_scale1024_rare4x_gpu89_batch2.py`;
- checkpoint:
  `work_dirs/dotav2_cleanstart/full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/epoch_24.pth`;
- raw mAP/AP50: `0.606405/0.6060`;
- best observed T7 raw mAP/AP50 at Epoch 18: `0.6081/0.608`;
- base14/novel4 at Epoch 24: `0.654018/0.439762`;
- fixed-recall oracle mAP: `0.782828`, leaving `+0.176423` ranking headroom;
- TP score--rotated-IoU Spearman: `0.365425`;
- geometry misses: `31.499%` of GT, versus approximately `1%` semantic
  misses;
- `77.744%` of geometry misses have no nearby final query center;
- objects no larger than eight pixels dominate geometry misses;
- AP-support false positives are dominated by localization/background
  (`64.634%`), not semantic confusion (`2.441%`).

The frozen encoder-source audit is a scientific failure, not an infrastructure
failure:

- report:
  `.lab/workspace/exp-8-qaf-source-v1/source_report.json`;
- source reach: `0.07178882557149407`;
- base14 reach: `0.0717736256898161`;
- spatial placebo: `0.07262396017554201`;
- semantic placebo: `0.07183795113643807`;
- report SHA-256:
  `d51f88a677af48ddbdc0455f4300452bcaf51e44bbddec7c22176555b034ca33`.

Therefore the method must change decoder-native geometry. It cannot be a
score-only rescue or another stride-32 encoder evidence adapter.

### 2.2 Closed local branches

The following branches cannot be renamed or reintroduced as ODQ-Flow:

- A1 detached rotated-IoU target scaling;
- D11 and D12 transport families;
- D13-N detached existence residual;
- balanced matched/unmatched classification;
- Hausdorff or Chamfer matching and denoising;
- fixed-1280 scaling;
- blind continuation beyond T7 Epoch 24;
- semantic-FP-first optimization;
- any alternate encoder source, feature level, threshold, seed, or subset
  rescue of the closed QAF source family.

## 3. Literature Boundary and Claim Discipline

The public literature already covers each isolated ingredient:

- D-FINE uses fine-grained distribution refinement for horizontal DETR box
  localization;
- Gaussian/KLD rotated detection treats an oriented box as a Gaussian object
  for regression and assignment;
- D2Q-DETR uses point distributions and dynamic queries for oriented
  detection;
- Group DETR, MS-DETR, Stable Matching DETR, and DEIM improve one-to-many or
  matchability-aware DETR supervision;
- Rank-DETR, Align-DETR, Cascade-DETR, UN-DETR, PROB, and OWOBJ address
  localization-aware ranking, presence, or objectness;
- DQ-DETR, DDQ, Dome-DETR, and PaQ-DETR change query population, selection,
  density allocation, or query specialization;
- RHINO and Fourier Angle Alignment address rotated matching, denoising, or
  angle-aligned features;
- OpenRSD and CastDet already establish open-prompt/open-vocabulary remote
  sensing and oriented variants.

Accordingly, the paper must not claim any of the following:

- first distributional box detector;
- first probabilistic or Gaussian rotated detector;
- first quality-aware DETR;
- first objectness-aware open-vocabulary detector;
- first dynamic or density-aware query detector;
- first open-vocabulary oriented detector.

The current candidate gap is narrower:

> A fixed-population, open-vocabulary, oriented set predictor in which one
> decoder-native predictive geometry distribution is reused across iterative
> refinement, matching, and unfiltered all-row scoring.

This is a **candidate gap**, not a first claim. A broader citation-chain and
concurrent-work audit is required before manuscript claim freezing.

## 4. Frozen Scientific Contract

ODQ-Flow must preserve all of the following:

1. exactly 600 matching queries at inference;
2. exactly three independently matched groups of 600 queries during training;
3. the existing denoising prefix and attention-mask isolation;
4. open-vocabulary token similarity rather than a fixed 18-way classifier;
5. one class selected for every query without ranking or removing query rows;
6. all 600 rows returned to the raw evaluator;
7. no NMS, top-k, proposal preselection, duplicate suppression, or test-time
   filtering;
8. no scored encoder proposal source and no dynamic query count;
9. the DOTA-v2 1024 input, rare-positive exposure 4x, full empty-image
   coverage, prompt protocol, and canonical validation mouth;
10. physical GPUs 8 and 9 as the only devices for later training;
11. matched control and candidate runs must share data order, seed, effective
    batch, schedule, evaluation mouth, and checkpoint provenance;
12. raw mAP is authoritative; oracle and diagnostic metrics may explain a
    result but cannot promote one.

### 4.1 Rotated E2E invariant

“Rotated E2E” is an indivisible method contract:

- every prediction consumed by assignment, loss, validation, dump analysis,
  and reporting is a five-dimensional `(cx, cy, w, h, theta)` rotated box;
- the first-cycle ODQ residual changes only `xywh`, but the fifth angle value
  is inherited row-for-row from the parent rotated regression path;
- the existing rotated Hungarian geometry costs and rotated KLD/GDLoss remain
  active; they may not be replaced by an HBB surrogate gate;
- input quadrilateral annotations may undergo the existing deterministic
  `qbox -> rbox` conversion, but predictions may not be converted to HBB for a
  promotion metric;
- E2E means that the fixed 600-query set directly produces the 600 evaluated
  rotated boxes: no RPN, proposal selector, NMS/rotated-NMS, top-k, box voting,
  post-hoc refiner, or external detector is allowed;
- all reported mAP/AP50 promotion values use the canonical raw DOTA rotated-IoU
  evaluator at IoU `0.5`.

## 5. Architecture

### 5.1 One state, not multiple modules

For decoder layer `l`, let `h_l` be the query state, `r_l` the incoming 5-D
reference, and `d_l^parent` the existing point-regression delta. ODQ-Flow adds
a factorized predictive residual distribution in the parent's inverse-sigmoid
refinement coordinates:

```text
h_l -> distribution logits z_l [..., 4, K]
                 |
                 +-> centered expectation e_l [..., 4]
                 +-> target log probability / entropy telemetry

d_l^parent + alpha * e_l -> existing iterative reference update
```

The first cycle distributes only `x`, `y`, `w`, and `h`. The fifth
angle coordinate is copied exactly from the parent regression branch. This is
intentional: the local failure is primarily center/extent reach for tiny
objects, while circular angle decoding would add an independent risk before
the core geometry source is validated.

The distribution is nevertheless **oriented** because it refines a 5-D
oriented query state and all supervision, assignment, evaluation, and box
quality remain rotated. A later angle extension must use a circular
distribution derived from the same state; it may not be an auxiliary angle
head added to a failed `ODQ-R1`.

### 5.2 Distribution parameterization

The scaffold uses these frozen choices:

- four factorized categorical distributions, one each for `x`, `y`, `w`, and
  `h` in the parent's inverse-sigmoid coordinate system;
- `K = 17` symmetric support bins, including an exact zero bin;
- normalized symmetric support in `[-1, 1]`;
- fixed inverse-sigmoid coordinate radii `(2.0, 2.0, 2.0, 2.0)`;
- a two-layer prediction branch matching the existing regression branch's
  hidden width and activation pattern;
- the final distribution-logit layer is initialized to zero;
- no learned temperature, learned support, mixture count, covariance head, or
  separate confidence scalar in the first cycle.

The centered expectation is computed relative to the zero-logit distribution:

```text
e(z) = sum softmax(z)_k * support_k
       - sum softmax(zeros_like(z))_k * support_k
```

Therefore a zero-initialized distribution branch emits an exact zero residual
through the same numerical operations. The fixed `alpha` is a configuration
value, not a learned gate:

- `ODQ-S0`: `alpha = 0.0`, distribution loss weight `0.0`;
- `ODQ-R1`: `alpha = 1.0`, distribution loss weight `1.0`.

No trainable gate is allowed because a collapsing gate would make a negative
scientific result ambiguous.

### 5.3 Coordinate semantics

The distributional expectation is an additive correction to the existing
regression delta in the parent's inverse-sigmoid refinement space. For the
first four normalized coordinates, the refined value is

```text
b_refined = sigmoid(inverse_sigmoid(r_l) + d_l^parent + alpha * e_l)
```

The angle coordinate is exactly
`sigmoid(inverse_sigmoid(r_l_angle) + d_l_angle^parent)`. Coordinate radii
limit `e_l` before the normal reference update. Values outside the configured
support are clipped only for distribution-target construction; the existing
point regression and GDLoss remain responsible for the full target range.

The parent regression branch remains trainable and unchanged. ODQ-Flow is a
residual refinement, not a replacement checkpoint or a second detector.

### 5.4 Training target

For each decoder layer and each positive Hungarian assignment:

1. compute the parent-only point prediction from `d_l^parent` and `r_l`;
2. detach that parent-only prediction for distribution-target construction;
3. clamp normalized parent and GT `xywh` to `[1e-4, 1 - 1e-4]` and calculate
   `inverse_sigmoid(GT) - inverse_sigmoid(parent-only prediction)`;
4. normalize that four-coordinate residual by the frozen radii;
5. project the normalized residual onto its two adjacent support bins;
6. optimize cross-entropy to the resulting two-bin soft target.

Only actual Hungarian positives receive the distribution loss. Unmatched
queries receive no fabricated zero-residual target. The distribution loss is
averaged over positive coordinates and then across the existing three query
groups. Empty-GT batches return a graph-connected exact zero.

The existing classification, L1, rotated KLD/GDLoss, grouped assignment, and
denoising losses remain in place. The first cycle adds no distribution term to
the Hungarian cost and adds no independent matching pass.

### 5.5 Inference

`ODQ-R1` changes box geometry only. It does not alter the semantic score:

```text
score_q = existing open-vocabulary token score_q
```

Distribution entropy and support-boundary hit rates may be dumped as detached
telemetry, but they cannot affect scores or remove rows. This preserves a clean
answer to the first question: does decoder-native distributional refinement
recover geometry and raw mAP?

## 6. Code Boundaries

The first implementation cycle is intentionally concentrated:

### 6.1 New focused file

`projects/OVCapFlow/ov_capflow/oriented_distribution.py`

Responsibilities:

- validate `K`, radii, tensor shapes, floating types, and finiteness;
- build symmetric support;
- compute centered expectations;
- build two-bin soft targets;
- compute positive-only distribution loss;
- provide the distributional residual regression branch;
- expose detached entropy and support-boundary telemetry.

It must not import datasets, evaluators, prompt logic, query grouping, or
experiment configuration.

### 6.2 Minimal existing-file changes

`projects/OVCapFlow/ov_capflow/ov_capflow_head.py`

- construct distributional regression branches only when enabled;
- preserve the parent branch and all existing disabled-path behavior;
- collect per-layer distribution losses using existing assignments;
- expose detached distribution telemetry;
- leave `_predict_by_feat_single` score and row selection untouched.

`projects/OVCapFlow/ov_capflow/__init__.py`

- export only the new public branch/helper required by config construction and
  tests.

No first-cycle changes are allowed in:

- `ov_capflow.py` query initialization or pre-decoder;
- `ov_capflow_layers.py` cross-attention;
- `grouped_queries.py` matching-group semantics;
- `adaptive_dn.py` denoising policy;
- `calibration.py`, `null_reservoir.py`, `semantic_capacity.py`, or
  `existence_residual.py`;
- raw prediction row selection or evaluator code.

### 6.3 Configs

Two configs derive from the authoritative T7 recipe:

- `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_full24e_scale1024_rare4x_gpu89_batch2_odq_s0.py`:
  disabled/equality scaffold;
- `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_full24e_scale1024_rare4x_gpu89_batch2_odq_r1.py`:
  geometry-only candidate.

They may change only the ODQ distribution configuration, work directory, and
explicit experiment identity. Data, optimizer, schedule, query geometry, DN,
prompts, evaluator, and physical GPU declaration remain inherited.

## 7. Error Handling and Telemetry

The implementation must fail closed on:

- even `K`, `K < 3`, or support without an exact zero bin;
- radii with wrong length, non-positive values, or non-finite values;
- hidden states whose last dimension differs from the branch input width;
- logits not shaped `[..., 4, K]`;
- non-finite logits, expectations, targets, loss, or refined deltas;
- assignment masks that do not match the batch/query axes;
- inference query count other than 600;
- training metadata other than `3 x 600` for the promoted recipe;
- any attempt to enable distribution-aware scoring or matching in the first
  cycle.

Required detached telemetry per evaluated layer:

- mean entropy for each of the four coordinates;
- 5th/50th/95th expectation quantiles;
- lower- and upper-support target hit counts;
- positive target count;
- correction-to-parent magnitude quantiles;
- angle exact-equality count versus the parent output.

Telemetry is diagnostic only and must never enter gradients, evaluation
scores, promotion gates, or control/candidate data paths.

## 8. Five Core Checks

### Check 1: shape and mouth

- inference emits exactly `[B, 600, 5]` boxes and 600 scores/labels;
- training matching suffix is exactly `[B, 3, 600, ...]`;
- DN remains a prefix and cannot enter the matching suffix;
- all 600 inference rows reach the raw evaluator.

### Check 2: zero-step parent equivalence

With `ODQ-S0` and the authoritative T7 checkpoint:

- every parent-visible tensor group loads from the same source;
- classification logits, boxes, scores, labels, and parent losses are exactly
  equal on the frozen real batch;
- the ODQ loss is a graph-connected positive zero;
- angle output is exactly equal, not merely close;
- no parent parameter changes value before an optimizer step.

### Check 3: finite distribution behavior

- centered expectation is zero for zero logits;
- support extremes remain finite;
- empty-positive and empty-GT cases return finite graph-connected zero;
- one real forward/backward step has finite loss and gradients;
- target clipping and boundary-hit telemetry agree exactly.

### Check 4: optimizer and provenance

- new parameters appear exactly once in optimizer groups;
- parent parameters retain the T7 learning-rate multipliers and weight decay;
- checkpoint missing keys are restricted to the registered ODQ branch;
- no unexpected keys are accepted;
- config, parent checkpoint, dump, and audit SHA-256 values are recorded.

### Check 5: contract and anti-shortcut audit

- no NMS, top-k, proposal selection, dynamic query count, or score threshold;
- no encoder source or alternative feature-level source;
- no change to prompt order, class mapping, validation tiles, or raw metric;
- no use of distribution entropy in scoring;
- launch command exposes only physical GPUs 8 and 9.

Failure of any core check blocks training.

## 9. Experiment Ladder and Gates

### 9.1 `ODQ-S0`: no-training equality audit

Run config construction, synthetic unit tests, checkpoint audit, frozen real
batch forward/loss/predict equality, and one finite backward check. This stage
uses no scheduled training.

Promotion requires all five core checks. Tolerance may be used only for the
separate finite backward check; parent-equivalence tensors themselves must be
bitwise equal where the existing framework is deterministic.

### 9.2 `ODQ-R1-P`: matched proxy

Run candidate and control with the same proxy dataset, seed, data order,
effective batch, schedule, evaluation mouth, and checkpoint source. Training
may use only physical GPUs 8 and 9.

Pre-registered promotion gates:

- raw proxy mAP improvement at least `+0.010` absolute;
- raw AP50 improvement at least `+0.010` absolute;
- novel4 change no worse than `-0.005` absolute;
- either small-vehicle AP improves by at least `+0.015` or frozen geometry
  reach improves by at least `+0.020` absolute;
- all rows, losses, gradients, boxes, and telemetry remain finite;
- no support boundary has more than `25%` of positive targets; otherwise the
  radii are invalid and the run is diagnostic, not promotable.

The thresholds are continue/stop gates, not paper claims.

### 9.3 `ODQ-R1-F`: full-data promotion

Only a passing proxy may enter full-data training. Evaluate at matched Epoch
6 and Epoch 12 before authorizing Epoch 24.

- Epoch 6 continue gate: raw mAP delta at least `+0.010`;
- Epoch 12 continue gate: raw mAP delta at least `+0.015`, novel4 no worse
  than `-0.005`, and geometry-reach improvement remains positive;
- Epoch 24 paper-parent gate: raw mAP delta at least `+0.020`, novel4 no worse
  than `-0.005`, and small-object/localization diagnostics support the same
  causal account.

The project stretch target remains raw mAP `0.70`. A smaller gate pass justifies
continuing the structural research; it does not establish the final target.

### 9.4 Stop and branch rules

- If raw mAP fails but geometry reach rises, inspect score--IoU coupling and
  consider the separately specified distribution-aware readout stage.
- If localization among reached GT improves but center reach does not,
  consider the separately specified distribution-conditioned cross-attention
  stage.
- If geometry reach and raw mAP both fail, close ODQ-Flow; do not add matching,
  readout, angle, attention, or hyperparameter modules to rescue it.
- A radius or loss-weight change is allowed only after a registered boundary
  saturation or loss-scale failure. It creates a new experiment identity and
  never rewrites a failed result.

## 10. Required Ablations After a Positive Main Result

These are evidence requirements, not first-cycle implementation work:

1. point parent versus distributional refinement;
2. expectation-only versus expectation plus proper distribution loss;
3. center-only versus `xywh` distribution;
4. final-only versus all-layer distributional refinement;
5. factorized distribution versus equal-parameter scalar residual control;
6. entropy/magnitude stratification by object size and base/novel split;
7. fixed-Q all-row output audit for every reported model;
8. at least three seeds for the final promoted recipe and matched control.

## 11. Paper-Shaped Hypotheses

The first implementation cycle tests only the first two hypotheses:

- **H1:** point estimates provide weak gradients for tiny-object query
  refinement; a decoder-native residual distribution improves center/extent
  reach under the same fixed query population.
- **H2:** the improvement appears in raw rotated mAP and localization error,
  not only in oracle metrics or filtered predictions.
- **H3, conditional:** the same predictive distribution can align assignment
  and final localization confidence without an independent quality head.
- **H4, conditional:** distribution scale can control attention search range
  when geometry reach, rather than regression precision, remains limiting.

A defensible future paper contribution would be the verified coupling of H1,
H3, and possibly H4. `ODQ-R1` alone is a recovery substrate until those claims
receive direct experimental support.

## 12. Integrity Rules

- Every result is recorded, including failed zero-step checks, proxy runs, and
  non-promoted endpoints.
- Diagnostic subsets cannot replace the canonical raw validation result.
- Base14 and novel4 are always reported together with aggregate mAP.
- No checkpoint averaging, filtered mouth, alternate prompt order, or hidden
  post-processing may be used to cross a gate.
- Literature wording distinguishes verified public facts, local empirical
  facts, and unverified research hypotheses.
- A failed stage is closed under its frozen configuration and retained in
  project history.

## 13. Written-Spec Acceptance Criteria

This specification is ready for implementation planning only if the reviewer
accepts all of the following:

1. the first cycle is limited to `ODQ-S0` and geometry-only `ODQ-R1`;
2. `xywh` is distributional while angle remains exactly parent-controlled;
3. the distribution is a residual on the existing point branch, not a new
   detector;
4. matching, scoring, angle, and attention consumers are future gated work;
5. five core checks block all GPU training;
6. later training uses only physical GPUs 8 and 9;
7. promotion is based on raw mAP plus base/novel and geometry evidence.
