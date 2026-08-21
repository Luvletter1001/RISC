# OV-CapFlow HRSC Fast-Iteration Design

## 1. Purpose

Use HRSC2016 ship1 as a fast structural gate for OV-CapFlow before returning to
DOTA2. The HRSC stage must answer whether fixed native queries, query-preserving
evidence fusion, balanced supervision, and an explicit null reservoir can be
trained reproducibly while preserving strict all-query oriented inference.

HRSC is a single-class closed-set benchmark in this stage. Its results may
support claims about rotated geometry, query ownership, optimization stability,
and null calibration. They must not be used as evidence for arbitrary-vocabulary
recognition, novel-class transfer, or absent-prompt generalization.

## 2. Fixed experimental contract

- Dataset: HRSC2016 ship1.
- Local data source: `/data/zcy/dataset/HRSC_unzip`.
- Training split: `ImageSets/trainval.txt`.
- Validation split: `ImageSets/test.txt`.
- Input scale: `(800, 512)` with aspect ratio preserved.
- Angle convention: `le90` direct rotated boxes.
- Primary metric: `dota/mAP` from `DOTAMetric(iou_thrs=0.5)`, interpreted as
  HRSC AP50; higher is better.
- Secondary metrics: exact mAP, recall, predictions per image, GT coverage,
  duplicate extras per GT, matched and unmatched gate means, gate gap, null
  Brier score, capacity mass, gradient norms, peak memory, and iteration time.
- Inference contract: fixed matching-query count, one prediction per matching
  query, no encoder-proposal top-k in the strict arms, no global query-class
  top-k, no survivor pruning, no dense inference head, no NMS, and no
  `minAreaRect` conversion.
- Devices: physical GPUs 4, 5, 6, and 7.
- Per-run wall-clock limit: two hours. A run that exceeds this limit is stopped
  and recorded as a timeout rather than silently receiving more budget.
- Common randomness: seed `20260712` for the first matrix; a promoted winner is
  repeated with seed `20260713`.

All compared arms use the same data ordering, augmentation, optimizer, learning
rate, matching-query count, batch budget, initialization source, epoch count,
evaluator, and checkpoint-selection rule. A setting changed for engineering
reasons must be applied to every arm and recorded before the formal matrix.

## 3. Why a staged parent is required

The historical P149/R25 checkpoints belong to the older GSOVD/OpenSetFlow
architecture. They are retained as external evidence but are not treated as
state-compatible OV-CapFlow parents. Directly importing their adapter results
would confound codebase, parameterization, and evaluator differences.

The current repository has a usable Swin-T initialization and a standard BERT
text backbone, but no trained CastDet/OV-CapFlow HRSC checkpoint. The first
stage therefore constructs two matched local anchors from the same public
initialization:

1. upstream Oriented GroundingDINO, which retains its proposal selection and
   serves as an engineering/performance reference;
2. OV-CapFlow strict native-only, which becomes the parent for all subsequent
   strict structural arms.

Both anchors train for 10 epochs. They run in parallel when resources allow,
but their results remain separate because only the strict native-only anchor
satisfies the OV-CapFlow inference contract.

## 4. Components and configuration boundaries

### 4.1 HRSC dataset configuration

Create an OV-CapFlow HRSC configuration that reuses the repository's
`HRSCDataset` pipeline while setting the absolute local data root through a
single config field. The config must expose the split names, input scale,
class text (`ship`), batch size, query count, seed, training length, validation
interval, and work directory without editing shared upstream dataset files.

The dataset build is verified before GPU training by checking dataset lengths,
the first packed sample, its text metadata, and the converted five-dimensional
rotated boxes.

### 4.2 Explicit structural switches

The OV-CapFlow decoder and detector need independent config switches for:

- semantic evidence fusion;
- continuous density capacity and its losses;
- balanced matched/unmatched reduction;
- explicit null-reservoir calibration.

Disabling all four switches must produce the strict native-only arm. A disabled
module must not contribute a loss, gradient, score transformation, or hidden
state update. Density remains disabled in the first HRSC causal matrix because
the historical HRSC evidence already showed that scanning density strength
without better null calibration was not informative.

### 4.3 Balanced supervision

Balanced reduction applies only to matching-query classification/calibration
supervision. Denoising losses retain the upstream behavior. Matched and
unmatched matching queries are reduced as separate means and combined with
weights fixed in the config before training. Images with no members in a group
produce a finite zero contribution for that group rather than a NaN or a
division by zero.

The implementation records effective matched and unmatched counts and their
loss contributions. Adding extra unmatched queries with identical predictions
must not linearly increase the matched contribution.

### 4.4 Explicit null reservoir

The null reservoir is a continuous capacity outlet, not a hard background mask.
It supplies:

- a per-query null probability;
- matched versus unmatched null calibration;
- a foreground-mass penalty for background-heavy images;
- a scene-level residual between predicted non-null capacity and GT count;
- validation diagnostics for null Brier score and gate ordering.

All matching queries remain in the decoder and raw prediction set. No threshold,
sorting operation, or top-k may turn null probability into survivor selection.

### 4.5 Stable per-query readout

Class selection and score calibration are separate operations. Label selection
uses stable log-domain values and remains invariant under a class-common
positive temperature or power transformation. Score calibration may use null
and capacity, but it must remain finite for extreme fp16 logits and must not
change the number of prediction rows.

Although HRSC has one foreground class and cannot validate multi-class label
competition empirically, unit tests enforce this contract before training.

## 5. Execution stages

### Stage H0: engineering gate

Before formal training:

1. build the HRSC dataset and both model configs;
2. load one real batch;
3. run complete loss forward and backward;
4. verify every active loss is finite and non-constant;
5. verify the expected query, reference, box, gate, null, capacity, and count
   shapes;
6. verify gradients reach every enabled intervention;
7. run a prediction forward and confirm the strict arms emit exactly the fixed
   matching-query count;
8. scan executable OV-CapFlow paths for forbidden selection and post-processing.

An H0 failure is `invalid-engineering`. It is fixed and rerun with the same
design; it is not counted as evidence against a structure.

### Stage H1: local anchors

Train for 10 epochs from the same Swin-T/BERT initialization:

- `HRSC-P0`: upstream Oriented GroundingDINO reference;
- `HRSC-C0`: OV-CapFlow strict native-only parent.

Report AP50 at epochs 5 and 10, plus best and final checkpoints. `HRSC-C0` is
eligible as the structural parent when its run completes without non-finite
values, passes the strict audit, reaches AP50 of at least `0.50`, and repeats
evaluation of the same checkpoint within `0.001` AP50. A lower score stops H2
and sends the work back to initialization, loss-routing, and optimization
diagnosis instead of testing adapters on a weak parent.

The upstream and strict scores are not expected to be equal. Their gap is
reported as the cost of the strict substrate, not attributed to later adapters.

### Stage H2: four-arm causal matrix

Initialize every arm from the same `HRSC-C0` epoch-10 checkpoint. Freeze the
detector and established box trajectory; intervention arms train only the
explicitly listed fusion/balanced/null parameters for five epochs.

| GPU | Experiment | Active intervention |
|---:|---|---|
| 4 | `HRSC-C0R` | zero-update native-only checkpoint replay/control |
| 5 | `HRSC-C1` | identity query-preserving evidence fusion |
| 6 | `HRSC-C2` | C1 plus balanced matched/unmatched reduction |
| 7 | `HRSC-C3` | C2 plus explicit null reservoir |

The control replay loads and evaluates the frozen checkpoint at the start and
again after the intervention runs. It has no optimizer step because there is no
adapter in the native-only arm. It uses the same validation loader, evaluator,
and effective prediction settings as the intervention arms, detecting drift
caused by checkpoint loading, data order, or evaluator changes without changing
the parent trajectory.

### Stage H3: reproducibility and short continuation

Promote the best valid H2 arm only when:

- AP50 improves by at least `0.002` over `HRSC-C0R`; or
- AP50 is within `0.001` of the control while at least two ownership/null
  diagnostics improve materially and recall does not decrease by more than
  `0.005`.

For the second rule, a material mediator improvement means one of: null Brier
decreases by at least `10%`; positive `unmatched_gate - matched_gate` decreases
by at least `20%` or becomes non-positive; duplicate extras per GT decreases by
at least `10%`; or GT coverage increases by at least `0.005`.

Repeat a promoted arm with seed `20260713`. If both runs retain the gate, allow
a short continuation to epoch 10 with calibration early stopping. If the repeat
fails, classify the result as single-seed/parked rather than promoted.

## 6. Measurement and experiment ledger

Research iterations use an untracked `.lab/` ledger. It records the objective,
commands, commit SHA, parent experiment, primary and secondary metrics, runtime,
keep/discard decision, crashes, and causal interpretation. `.lab/` is excluded
from git but remains available across experiment branches and resets.

Every repository-changing experiment is committed before it runs. Every result
is logged before a discarded experiment is reset. Formal work directories use
the experiment IDs above and never overwrite another arm.

Primary keep/discard decisions use AP50 with a `0.001` noise band. Secondary
metrics break ties and can mark a result `keep*` or `interesting`, but they do
not convert a clear AP regression into a promotion.

## 7. Failure handling and stopping rules

- Missing data, missing pretrained weights, config-build failure, loss-routing
  failure, NaN/Inf, OOM before useful optimization, and incomplete evaluation
  are engineering failures and are rerun after repair.
- A checkpoint whose load operation overwrites the requested intervention is
  invalid. Effective switches and trainable parameter names are serialized
  after loading.
- Three consecutive valid discards trigger an assumption review before another
  experiment.
- Five consecutive valid discards trigger a strategy fork rather than a finer
  sweep of the same mechanism.
- A two-hour timeout is recorded as a timeout. Two consecutive timeouts require
  reducing the run unit or revisiting the approach.
- Density strength, query count, geometry loss, and teacher/pseudo-label routes
  are outside the initial HRSC matrix and are not used to rescue a failed arm.
- The stage stops and returns to DOTA2 when a reproducible H3 winner exists, or
  when the balanced/null axis reaches the discard guardrail without mediator
  improvement.

## 8. Test and audit requirements

The change set is acceptable only when all of the following pass:

- existing 17 OV-CapFlow focused tests;
- config-build tests for upstream HRSC and each strict switch combination;
- a real HRSC batch forward/backward integration test;
- balanced reduction tests for group-size invariance, empty groups, and
  positive/negative gradients;
- null-reservoir tests for bounded probabilities, finite empty/non-empty losses,
  count residual, and absence of hard query deletion;
- log-space readout tests for extreme fp16 logits, temperature/power label
  invariance, and exactly one label per query;
- checkpoint-load tests that report missing, unexpected, and intentionally new
  parameters;
- a strict inference scan showing no top-k, NMS, dense inference head, survivor
  mask, or minimum-area rectangle operation in executable OV-CapFlow paths.

## 9. Deliverables

The HRSC stage produces:

1. committed HRSC configs and implementation;
2. real-batch H0 audit;
3. upstream and strict H1 parent reports;
4. four-arm H2 AP50 and mediator table;
5. repeated H3 winner or a documented stop/park decision;
6. checkpoint hashes, effective configs, strict scans, and `.lab` history;
7. a concise transfer recommendation identifying exactly which mechanism is
   allowed to enter the later DOTA2 matrix.
