# Map-First Swin-B Anchor and Negative-Safe Strict-OV Design

**Date:** 2026-08-03  
**Status:** proposed; no download, implementation, dataset mutation, or GPU
launch before explicit approval

## 1. Evidence boundary

The current authority is an all-18-supervised strict-mouth substrate, not a
strict open-vocabulary detector:

- E24 raw DOTA-v2.0 mAP/AP50: `0.606405349 / 0.6060` on `13,833 x Q600`;
- no proposal/top-k/NMS, six decoder layers, direct rotated 5-D boxes;
- base14/novel4 slicing of this checkpoint is descriptive only because all 18
  categories were present during training;
- the tested D11, D12, D13-N, QAF, M1, ODQ, and OpenRSD-CSER instantiations
  were negative; they are evidence against repeating those exact recipes, not
  proof that every member of their broader mechanism families must fail;
- OMQ actionability stopped before capture because its S1 train/validation
  split leaks 164 original scenes and 105 positive pixel-overlap tile pairs.

No broad `first` claim is allowed. The working system description is
teacher-free, pseudo-label-free, fixed-Q600 rotated-5D direct set prediction
with all-query scoring and no inference NMS/proposal-top-k.

## 2. Options considered

### Option A — recommended two-stage program

First establish a stronger generic-pretrained Swin-B raw-mAP anchor without
calling it an innovation. Then study negative-safe fixed-set learning under
strict base-only incomplete annotations. This follows the user's ordering:
reach the raw mAP target first, then build the paper method on a valid strict-OV
protocol.

### Option B — strict negative-safe method immediately

This has the strongest ICLR research question but leaves the raw `0.7000`
target tied to the weaker Swin-T substrate. It is scientifically clean but
does not follow the requested mAP-first priority.

### Option C — set-level ownership transport

Mass-conserving soft ownership across the complete Q600 set could address
dense/tiny misses, but its readiness is lower and novelty collision with
Group/H-DETR, Align-DETR, PaQ-DETR, and quality-aware assignment is high. It is
kept as a fallback only after a trajectory oracle, not the next experiment.

## 3. Stage B0: raw-mAP capacity anchor

### Question

Can a stronger generic GroundingDINO-B initialization cross the raw DOTA-v2
`mAP/AP50 >= 0.7000` target while preserving the exact strict rotated mouth?

This is a strong anchor, not a causal backbone ablation: the official B model
also has a larger generic pretraining corpus than the existing T source.

### Frozen model and data contract

- Source: official `groundingdino_swinb_cogcoor.pth` only; compute and freeze
  its SHA256 after download. No added DOTA/OpenRSD/HRSC/project checkpoint;
  record the official pretraining-data statement as provenance rather than
  inferring absence of every target-like image from a filename.
- Target structure: Swin-B `embed_dims=128`, depths `[2,2,18,2]`, heads
  `[4,8,16,32]`, window `12`, neck inputs `[256,512,1024]`; decoder, language
  path, loss, and rotated head remain the existing OV-CapFlow recipe.
- Current CPU construction check: 232,908,890 total parameters, including
  86,880,120 in the backbone.
- Initialization: reuse the audited clean-start name/shape converter. Require
  all generic backbone/encoder/decoder/language prefixes; additional missing
  tensors beyond the known Q600 initializer, classification bias, and 4-D to
  rotated-5-D regression mismatch are a contract stop.
- Training data: the existing full rare4x all-18 recipe, scale 1024, seed
  20260716. This stage is supervised raw-mAP engineering and is not used as
  strict-OV evidence. Sampler audit checks the pre-registered per-original-stem
  exposure multiplicity: every designated rare stem exactly four times and
  every other stem exactly once; only missing or non-pre-registered repetition
  is a failure.
- Inference: exactly 600 rows/image, six rotated decoder layers, empty
  `test_cfg`, no proposal selection, top-k, NMS, class filtering, or box
  duplication.
- Resources: physical GPUs 0--9; every DDP launch sets
  `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1`. Use batch 2/GPU,
  accumulation 1, and Swin backbone activation checkpointing. This amendment
  is frozen before implementation because 47,294 exposures cannot give every
  world10 rank a non-empty singleton final update. Before GT-aware splitting,
  world10 x batch2 has a nominal final 14-sample remainder; the actual frozen
  sampler plan makes 23 query-budget shrink updates and redistributes to 2,366
  updates with global batch 17--20 and local batch 1--2. It uses every real
  exposure once without padding, dropping, or duplication. Never guess memory
  capacity.

### Preflight gates

1. source/config/output/provenance hashes and forbidden-source scan pass;
2. converted checkpoint coverage and missing/mismatch allowlist pass;
3. CPU model build and load have no unexpected model namespace;
4. one real sparse, one dense, and one `GT>600` train batch are finite;
5. one forward/backward/step is finite with exact Q600 inference replay;
6. measured peak memory leaves at least 10% A40 headroom on every rank;
7. sampler covers the frozen exposure manifest exactly, including intentional
   rare4x aliases, with no missing or unintended duplicate exposure, and
   records variable-batch shrink events.

The batch amendment means B0 is not an optimizer-matched Swin-T comparison:
it uses 17--20 real samples per optimizer update, the inherited learning
rate/schedule, and more optimizer updates per epoch than T7. This is disclosed
capacity/pretraining engineering, never a causal backbone claim.

Any deterministic, provenance, mouth, numerical, or memory failure stops B0.
There is no silent batch/scale/query/loss change.

### Frozen training gates

- E1: report raw mAP/AP50, per-class AP, runtime, sampler, and mouth audit. If
  mAP is below `0.4000`, stop as a failed initialization.
- E6: require raw mAP at least `0.5800` and at least `+0.0200` over the T7 E6
  authority `0.553674161`; otherwise stop without E12.
- E12: require raw mAP at least `0.6500`; otherwise stop without E18/E24.
- E18: continue only if mAP improves E12 by at least `+0.0100` or already
  reaches `0.6900`.
- E24: final target is both exact mAP and rounded AP50 at least `0.7000` on
  all 13,833 validation tiles. Never select an intermediate checkpoint as the
  endpoint.

Only milestone checkpoints E1/E6/E12/E18/E24 are retained. A gate failure
preserves all evidence and releases GPUs; it does not authorize rescue.

## 4. Firewall between B0 and strict-OV development

B0 is only a closed-set engineering anchor. N0/N1 always initialize directly
from the official generic Swin-B checkpoint through a separately hashed
clean-start converter output. They never load a B0 checkpoint, optimizer,
feature cache, class head, or selected epoch. B0 pass/fail, its novel4
per-class values, and its training trajectory cannot choose the N0/N1
backbone, augmentation, threshold, mask budget, class fold, or checkpoint.

Therefore the two stages are chronological but not a scientific ancestry:
B0 may establish a stronger raw-mAP substrate design, while strict-OV evidence
starts again from generic initialization. Any N0/N1 implementation requires a
separate approved design; approval of this document does not yet authorize it.

## 5. Stage N0: strict base-only substrate and diagnostic

N0 starts only after B0 reaches its terminal gate or is formally closed.

### Split and sealing

- Parse every tiled training stem as `(original_scene, crop_size, x, y)`.
- Split original scenes before assigning tiles. Train/dev/test scene
  intersections and all cross-split crop pixel intersections must be zero.
- Training annotations and prompts contain the active fold's seen classes
  only. Official novel4 boxes, names, rare-positive exposure, metrics, and
  gradients are absent from all development training and model selection.
- N0 development uses only pre-registered meta-novel folds drawn from the 14
  base categories. For each fold, selected base categories are temporarily
  hidden and treated as pseudo-novel solely to test the learning problem and
  choose the frozen mechanism contract.
- The official DOTA-v2 novel4 labels and GT remain sealed from problem
  selection, threshold/mask-budget choice, architecture, checkpointing, and
  every PASS/FAIL decision until one final strict all18-prompt evaluation.

### Diagnostic question

Under a meta-fold's seen-only training, are Hungarian-unmatched Q600 rows near
withheld meta-novel objects assigned unusually large negative classification
gradients compared with matched-background, spatial, scene, and random-row
placebos? If not across the pre-registered base-category folds,
incomplete-label negative suppression is closed before a new model is written.

The diagnostic must report concentration, support, image-macro and row-micro
statistics, tiny/dense strata, effect-size confidence intervals, and a
meta-novel oracle-negative-ignore upper bound. Oracle masking is strictly a
no-optimizer, no-checkpoint offline gradient/loss counterfactual; it never
trains an arm, selects a threshold, becomes a positive target, or enters
inference.

## 6. Stage N1: negative-safe fixed-set prediction

Only a passing N0 permits N1.

### Mechanism boundary

For two geometric views of the same training image, map rotated query boxes
back to the common image frame. Among rows that are unmatched under the
seen-class-only Hungarian assignment, identify pre-registered cross-view-stable
query trajectories and mask only their suspected-invalid background
classification gradient. The rows receive no positive class/box target. This
is prediction-derived negative masking; it must not be described broadly as
`pseudo-label-free`.

Forbidden additions:

- teacher/EMA, pseudo boxes, self-training, external region proposals;
- new quality/objectness head, router, source-stat adapter, or score residual;
- query count/selection, top-k, NMS, duplicate suppression, or inference-time
  auxiliary path;
- novel names, boxes, metrics, or class frequency in training/model selection.

Inference remains the unchanged single-view Q600 rotated all-query model.

### Controls and promotion

Use matched generic checkpoint, data exposure, seed, Q600, optimizer updates,
and evaluator for:

1. base-only control;
2. random unmatched-negative masking with the same mask budget;
3. confidence-only masking;
4. cross-view-stability masking;
5. offline meta-novel oracle masking as a no-training upper bound.

Arms 1--4 all process the same two geometric views and use the same number of
forwards, optimizer updates, and augmentation exposures. Arm 1 is the true
unmodified control and always masks zero rows; any compute sham must leave its
loss exactly unchanged. Only Arms 2--4 receive the same pre-registered nonzero
per-image unmatched-row mask budget. The random and confidence controls differ
from stability only in how they select that fixed budget. The offline oracle
is not a matched training arm.

Before official novel4 is unsealed, all method code, controls, thresholds,
mask budgets, augmentations, base-category meta-novel folds, seeds, checkpoint
rules, and independent-dataset protocol must be frozen from seen/base evidence
only. Final evidence reports base14 mAP, novel4 mAP, harmonic mean, raw all18
mAP/AP50, per-class AP, tiny/dense strata, and AP-support FP decomposition.
The primary method must improve novel mAP by at least `+0.0300` over the
matched base-only control, keep base mAP within `-0.0050`, beat both
non-oracle masking controls, and preserve every strict mouth audit.

All gates are conjunctive: raw all18 mAP/AP50 may not regress by more than
`0.0050`; empty-tile foreground FP may not increase; AP-support
background/localization and duplicate FP may not increase; and Q600 row count,
no-NMS/no-top-k, base retention, and finite-runtime gates must all pass. A
failed one-time official novel4 evaluation closes the mechanism; it cannot be
used to revise the method and rerun the same final split.

## 7. Pre-N1 closest-work hard gate

Before an N1 specification or implementation, freeze a primary-source
closest-work table covering positive-unlabeled detection, incomplete-label
detection, open-world unknown mining, cross-view DETR consistency, negative
query filtering, and open-vocabulary oriented detection. The proposed novelty
must survive as all of: negative-only masking, no positive pseudo-box target,
no teacher, fixed-set Q600 rotated all-query inference, and no NMS/top-k. If a
prior method already instantiates the same mechanism, stop or change the
research question before any N1 code.

## 8. Paper claim boundary

The candidate contribution is:

> negative-safe fixed-set prediction under incomplete category annotations,
> using equivariant rotated-query stability as prediction-derived negative
> masking while retaining a proposal-free Q600 set and using no positive
> pseudo-box targets or teacher.

This wording is provisional until the pre-N1 literature gate passes. It must
not be shortened to `pseudo-label-free`. D11/D12/D13-N/QAF/M1/ODQ/CSER are
internal design evidence, not seven paper contributions.

## 9. Approval boundary

Approval of this design authorizes only:

1. B0 implementation by TDD, official B weight download with recorded hash,
   preflight, then gate-controlled GPU0--9 training;
2. concurrent CPU-only construction and audit of future scene-disjoint
   meta-novel/base-only manifests, without accessing official novel4 labels;
3. no N0/N1 model implementation until a separate design, closest-work gate,
   folds/seeds/controls contract, and sealed-novel protocol are approved.
