# DOTA-v2 D13-N Parent-Preserving Existence Residual Design

## 1. Status and purpose

The user selected approach A on 2026-07-31. This document freezes the written
design for review before implementation.

D13-N tests one causal hypothesis:

> A frozen strict-Q600 parent already contains useful boxes and semantic
> labels, but forces every surplus query to expose a foreground score. A
> learned, query-local existence residual can improve all-query score ordering
> without changing query identity, class selection, rotated boxes, or the
> no-NMS/no-top-k inference contract.

D13-N is a calibration probe, not a claim that calibration alone solves the
full geometry deficit or reaches raw mAP 0.7000. It must not be described as
the paper's final innovation unless later evidence and literature review
support a broader method.

## 2. Evidence and limits

The authoritative parent is T7 Epoch 24 on all 13,833 raw DOTA-v2 validation
tiles:

- exact mAP `0.6064053488274416` and official AP50 `0.6060`;
- 600 predictions per image and 8,299,800 total rows;
- base14 mAP `0.6540175995656422` and novel4 mAP
  `0.43976247124373913`;
- AP-support false-positive shares: localization/background `0.646339`,
  empty `0.176231`, duplicate `0.153023`, semantic `0.024407`;
- TP score--rotated-IoU Spearman `0.365425`;
- fixed-recall oracle mAP `0.782828`.

The 25-image false-null audit found that 98.728% of final true positives were
Hungarian matched. Only 0.502% of all unmatched queries were true positives,
while a perfect unmatched-suppression oracle raised the supported-class mean
AP by `0.027781`. This supports a score-calibration test but is neither a raw
upper bound nor evidence of a learned gain.

Known limits are part of the contract:

- matched status is an ownership surrogate, not objectness or IoU quality;
- 44.71% of matched queries in the 25-image audit were still
  localization/background, so the target is noisy;
- in `>600 GT/image` scenes every query can be matched, so this mechanism
  cannot repair the fixed-capacity geometry ceiling;
- D13-N cannot be stacked with A1, balanced classification, semantic fusion,
  density capacity, D11, or D12 in this experiment;
- existing objectness/null/ranking literature makes a standalone novelty
  claim high risk.

## 3. Frozen scientific contract

Candidate and control both preserve:

- parent checkpoint
  `work_dirs/dotav2_cleanstart/full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/epoch_24.pth`;
- parent SHA256
  `a4f2661e6c1645b08f296dfb2bbebfd76afe8c366d6152dbbc333bbf840af6b8`;
- the canonical T7 model, prompt, 18-class order, base14/novel4 split, fixed
  Q600 initializer, six-layer direct rotated 5-D decoder, DN100 path, token
  aggregation, Hungarian costs, and all parent losses;
- the immutable 1,600-train/400-validation rare4x proxy manifests and scale
  `1024 x 1024`;
- three independently matched 600-query training groups;
- one label, score, and rotated box for every inference query in original row
  order;
- no query deletion, sorting, thresholding, global top-k, NMS, rotated NMS,
  proposal selection, dense/RoI/RPN head, teacher, pseudo-label, or remote-
  sensing parent;
- seed `20260716`, Epoch-12 endpoint, optimizer family, scheduler, data order,
  batch size, and distributed topology.

Both arms instantiate the same head-owned module:

```text
existence_logit = Linear(stop_gradient(final_query), 1)
weight = 0
bias = 0
trainable scalars = 256 + 1 = 257
```

Construction of `existence_residual` itself must consume no random numbers:
with the PyTorch RNG state snapshotted immediately before and after only that
module's constructor, the states must be bitwise equal. Its parameters are
allocated as explicit zeros or `reset_parameters()` is overridden with
deterministic zero fills. This requirement does not claim that construction of
the full parent model is RNG-free.

Every parent parameter is frozen. The only scientific difference is the
existence-loss coefficient:

```text
candidate: existence_loss_weight = 1.0
control:   existence_loss_weight = 0.0
```

The zero-weight control retains a differentiable zero loss so the identical
optimizer path can execute, but its head must remain bitwise zero after every
step. Candidate and control may differ only in the 257 head scalars and their
optimizer state. No gradient from the head may update parent features or any
parent parameter.

The proxy is a deliberately narrow head-screening mouth, not independent
generalization evidence. Its 400 images come from the DOTA training pool seen
by the E24 parent; the raw 13,833-image validation mouth is therefore the only
promotion evidence. The frozen manifests are:

```text
rare4x train1600 SHA256 1457c641d91a7e0bf26a62f0cd6c70c71d9e9c6df5a2137fe4b73b8e8fc05290
proxy400 parent SHA256 a00b945ddd0a769008d57145feba28382fb9e56f5d6f1427413220f8008e1a2e
```

The train manifest contains 1,600 rows, including 372 empty tiles and 36 rare
images repeated four times for 144 rows. Proxy400 contains 120 empty tiles.
These counts and hashes are Stage-0 provenance gates, not mutable recipe
choices.

## 4. Architecture and data flow

### 4.1 Independent head-owned path

The new module is owned by `OVCapFlowHead`, not the decoder. The old
`ExplicitNullReservoir`, decoder null plumbing, C3 mass loss, gate-order loss,
and capacity path remain disabled and untouched.

During training, the head takes the explicit DN-excluding slice
`hidden_states[-1][:, -1800:, :]`, with shape `[B,1800,256]`. It removes no
matching row, reshapes the slice to `[B,3,600,256]`, detaches it, and produces
existence logits `[B,3,600]`. No DN-prefix row enters this path. During
inference there is one fixed matching group, so the head maps the complete
`hidden_states[-1][:, -600:, :]` slice directly to `[B,600]`.

The E24 checkpoint must load with exactly two expected missing tensors:

```text
bbox_head.existence_residual.weight  # [1,256]
bbox_head.existence_residual.bias    # [1]
```

All parent tensors must load without another missing, unexpected, or
shape-mismatched key.

### 4.2 Per-group Hungarian targets

Final predictions from each training group are independently passed through
the existing assignment path. The head constructs a detached boolean target
with shape `[B,3,600]`, where `True` means Hungarian matched.

It is forbidden to:

- reuse group 0's mask for groups 1 and 2;
- flatten 1,800 queries before assignment;
- include DN queries;
- add the residual to matcher costs;
- use an IoU oracle, evaluator label, threshold, or validation result as a
  training target.

Matched counts are recorded per image and group and must equal
`min(num_gt,600)`. If deterministic re-assignment can differ from the actual
parent loss assignment under a cost tie, implementation must capture the
actual final-layer assignments instead of accepting an unverifiable replay.

### 4.3 Branchless grouped loss

The target is `1` for matched and `0` for unmatched. In each image and group,
compute float32 positive and negative BCE means independently, activate only
the nonempty strata, and average the active stratum means:

```text
Lm = sum(m * softplus(-e)) / clamp(sum(m), min=1)
Lu = sum((1-m) * softplus(e)) / clamp(sum(1-m), min=1)
Lg = (1[sum(m)>0] * Lm + 1[sum(1-m)>0] * Lu)
     / (1[sum(m)>0] + 1[sum(1-m)>0])
loss_exist = mean_B(mean_G(Lg))
```

All indicators, clamps, and reductions are tensor operations; there is no
data-dependent Python branch. Empty images use only `Lu`, all-matched groups
use only `Lm`, and mixed groups weight matched and unmatched strata equally.
This is head-local target balance that prevents unmatched-query prevalence
from collapsing the adapter to uniform suppression. It does not reuse the
failed balanced token-classification objective. No count/mass loss, semantic
gate loss, capacity loss, legacy C3 loss, or parent classification
reweighting is active.

### 4.4 Centered log-space readout

Class labels are selected from unchanged class log scores before existence
calibration. For selected parent log score `s` and existence logit `e`, use:

```text
r(e) = logsigmoid(e.float()) - logsigmoid(zeros_like(e).float())
s_calibrated = s + r(e)
score = exp(clamp(s_calibrated, min=-80, max=0))
```

The zero baseline uses the same `logsigmoid` kernel, not a separately rounded
constant. At `e=0`, the residual is exact positive zero in float32. The
implementation must prove `torch.equal` with the no-residual parent for
scores, labels, and boxes.

The residual is passed to `calibrate_selected_log_scores` as an optional
already-centered scalar log residual. It is added after the unchanged parent
temperature/power computation and before the existing clamp/exp. It never
participates in class argmax or the box clone/rescale path.

### 4.5 Frozen analyzer and metric definitions

Proxy and raw candidate/control diagnostics use exactly:

```text
projects/OVCapFlow/tools/analyze_dotav2_q600_dump.py
SHA256 f4f10734948b714dcbae726142e859b3ddac793ee5ab5c78d9a3f3436d99fa23
projects/OVCapFlow/tools/dotav2_q600_diagnostics.py
SHA256 bd29dc88650e0db545479c60adaa6a9ee64da1734d3a1756c7cd787c7269c92d
source commit abd860d157d562aa9d30422c05e41e246effc8a7
```

Changing either analyzer hash after Stage 0 invalidates the formal pair. The
gate freezes these definitions:

- `mAP` is the analyzer's full-precision macro VOC07 AP at rotated IoU 0.5
  over supported classes; `AP50` is its official three-decimal mouth.
- all-query micro recall is `sum_c(tp_count) / sum_c(num_gts)` over supported
  classes after the analyzer's complete 600-row-per-image evaluation.
- fixed-recall oracle mAP is the macro mean of `perfect_ranking_ap` applied to
  each class's analyzer TP/FP outcomes; oracle headroom is
  `oracle_map - map`.
- an AP-support prefix ends, per class, one rank after the maximum rank used by
  any point of the VOC07 11-point precision envelope. FP categories and counts
  are exactly `fp_regions.ap_support` from the frozen analyzer.
- endpoint matched/unmatched separation uses a diagnostic-only Hungarian
  assignment on the single 600-query validation group with unchanged parent
  costs and no existence residual in the matcher. Global means aggregate all
  rows in each nonempty stratum; empty and all-matched images are also reported
  separately but do not invent a missing stratum.

Because score ordering participates in the evaluator's greedy assignment,
unchanged labels and boxes do not imply bitwise-equal recall or oracle mAP.
Every non-strict floating gate comparison uses an absolute arithmetic
tolerance of `1e-12`: lower bounds use
`observed + 1e-12 >= lower_bound`, while upper bounds use
`observed <= upper_bound + 1e-12`. An explicitly strict inequality uses the
margin stated at that gate. Integer gates use exact integer arithmetic;
identity gates use `torch.equal` or byte hashes and have no tolerance.

## 5. Paired proxy configuration and five-GPU sequential layout

Two thin config wrappers inherit the scale-1024 rare4x proxy recipe but load
the frozen E24 parent rather than generic initialization:

```text
..._world5_d13n_control.py
..._world5_d13n_candidate.py
```

Both explicitly disable:

```text
decoder null reservoir
semantic fusion
density capacity
balanced classification
position-supervised classification
adaptive DN
capacity readout
legacy null/density losses
```

The freeze allowlist is exactly:

```text
^bbox_head\.existence_residual\.(weight|bias)$
```

Parent parameters are set to `requires_grad=False` during model construction,
before the optimizer is built. The resolved optimizer parameter groups must
contain exactly the two allowlisted head tensors and no parent tensor. During
head training, the detector remains in training mode so DN/group construction
is unchanged, while stochastic parent feature modules (backbone, language
model, encoder, decoder, and the parent portion of the box head) remain in
evaluation mode; only `existence_residual` is in training mode.

A D13-N-only mode-enforcement hook re-applies this policy in
`before_train_iter`, after MMEngine's per-epoch `runner.model.train()` call and
before every forward. A `before_train` or `before_train_epoch`-only hook is
forbidden because MMEngine would overwrite it. Candidate and control use the
identical hook. Stage 0 calls `model.train()` before enforcement and then
proves the exact module modes and bitwise-deterministic repeated parent
features.

Only physical GPUs `5,6,7,8,9` may be used. The zero-residual control runs to
the frozen endpoint first; only after it passes its integrity gate may the
candidate run on the same five GPUs. Candidate and control are never run
concurrently. Each arm has:

- world size `5`, per-rank batch `2`, accumulation `1`, nominal global batch
  `10`, and exactly `160` synchronized updates per epoch;
- exactly 1,600 manifest rows once per epoch, comprising 1,492 unique image
  IDs plus only the 108 preregistered rare4x repeat exposures, with zero
  missing row indices, sampler padding, or unregistered duplicate row indices;
  every manifest row index appears on exactly one rank, while duplicate image
  IDs are allowed only through the 108 registered repeat rows;
- seed `20260716`, `diff_rank_seed=False`, and `resume=False`;
- `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1`;
- candidate port `29841` and control port `29842`, subject to a final
  fail-closed listener check;
- unique work directory, sampler audit, console, tmux session, output dump,
  checkpoint, and postrun-gate paths;
- all epoch checkpoints retained without deletion or best-checkpoint
  replacement.

Apart from sequential role/port/path bookkeeping and existence-loss
coefficient `1.0`
versus `0.0`, resolved configs must be identical. Both arms use the inherited
AdamW `1e-4` schedule and train through all 12 epochs: 160 updates per epoch
and exactly 1,920 updates per arm. Epoch 12 is the only selection endpoint.

## 6. Stage-0 gates

Stage 0 publishes no formal checkpoint and performs no formal optimizer step.
All gates are conjunctive.

### 6.1 Static model and freeze audit

- Verify parent/config/data SHA256 values.
- Freeze the Stage-0 environment fingerprint: conda interpreter
  `/data/zcy/anaconda3/envs/mmdet/bin/python`, Python `3.8.19`, Torch
  `1.12.1+cu113` with Torch CUDA runtime `11.3`, MMEngine `0.10.4`, MMDetection
  `3.3.0`, MMRotate `1.0.0rc1`, NVIDIA driver `580.173.02`, reported driver
  CUDA `13.0`, and physical GPU5--9 model/UUID mapping. All five GPUs must be
  `NVIDIA A40`; their exact UUIDs are captured in the hashed Stage-0 report.
  The expected mapping is GPU5
  `GPU-6c98da7c-69bd-4c20-5094-ff3b60141515`, GPU6
  `GPU-b30e9f80-7925-6816-9014-d228784c5a12`, GPU7
  `GPU-d614452d-c145-3751-ce23-f9e21dd0b535`, GPU8
  `GPU-f8f2c2c6-4c20-237b-f6df-ba5b5408573e`, and GPU9
  `GPU-ff760a2d-9a1b-38ee-06a0-8c284e002545`.
- Require exactly the two expected missing head keys and zero unexpected keys.
- Require exactly 257 trainable scalars and only the two allowlisted names.
- Require all parent parameters `requires_grad=False`.
- Require the built optimizer to contain exactly the two allowlisted tensors.
- Require the detector/group logic to remain in training mode, stochastic
  parent feature modules in evaluation mode, and only the residual head in
  training mode; identical repeated inputs must yield bitwise-equal parent
  features.
- Prove every parent tensor is bitwise equal across candidate, control, and
  E24 after initialization and load.
- Prove both heads are bitwise zero and construction of each residual module,
  scoped by an immediate before/after snapshot, does not alter the PyTorch RNG
  state.
- Prove all excluded mechanisms in Section 5 are disabled.

### 6.2 Complete zero-update equivalence

Candidate, control, and E24 parent are evaluated on the complete 400-image
proxy before training. Require exactly 240,000 finite rows per arm and bitwise
equal scores, labels, rotated boxes, and row order. A single-batch equality
test cannot replace this gate.

All three also pass strict-Q600, open-vocabulary, rotated-mouth, checkpoint-
load, and forbidden-call audits.

### 6.3 Group-target and numerical stress audit

Use fixed empty, ordinary, and 1,223-GT examples. Require:

- target/logit shapes `[B,3,600]` and independent group assignments;
- finite loss and gradients for mixed, empty, and all-matched targets;
- gradients only on the two head tensors;
- no parameter change before an optimizer step;
- one disposable candidate step changes at least one head scalar and no
  parent scalar;
- one disposable control step changes no scalar;
- finite logit/residual quantiles and clamp-hit statistics;
- no Hungarian invalid matrix, NaN/Inf, OOM, NCCL error, or query-budget
  violation.

Disposable smoke paths are unique and cannot be reused by the formal run.

## 7. Formal paired training

After Stage 0 and a pre-run commit, a fail-closed owner waits for two
consecutive idle polls of physical GPUs5--9 and performs a final GPU, port,
artifact, commit, and config recheck. It launches the world5 zero-residual
control first. Only a complete, valid control Epoch 12 that is bitwise equal
to its zero-step proxy replay can reach the candidate release gate. Before
releasing a no-replace barrier, the owner again requires two idle polls of
GPUs5--9 and repeats the port, artifact-collision, commit, resolved-config,
parent/data/analyzer hash, and complete Stage-0 software/GPU fingerprint
checks. The candidate then receives the same GPUs5--9. Any control failure or
any intervening drift closes the pair before candidate launch.

The owner distinguishes the persistent `rtk env` wrapper from the actual
Python torchrun launcher. It validates exactly one real launcher and local
ranks `0..4` from worker environments, treating the wrapper only as an
ancestor. Missing or unreadable rank environments fail closed. This prevents
the D12 double-launcher false positive.

The monitor records every 60 seconds:

- GPU memory/utilization and compute PIDs on physical GPUs5--9;
- wrapper, launcher, worker topology and master ports;
- epoch/iteration, LR, parent losses, existence loss, gradient norm, data
  time, and iteration time;
- sampler coverage/checksum, checkpoints, console growth, and fatal patterns;
- per-group matched counts and existence-logit/residual/clamp quantiles.

At every checkpoint, the monitor also hashes every parent parameter and
persistent buffer against initialization, audits the two optimizer parameter
names, and verifies the control head is still bitwise zero.

No automatic retry, resume, batch change, epoch extension, coefficient
change, or single-arm continuation is allowed after a scientific or numerical
failure. All artifacts are preserved; no reset, clean, deletion, or overwrite
occurs.

Sequential execution does not weaken the matched baseline: the control's
entire trainable state is the zero head and must remain bitwise equal to E0,
while every parent parameter and buffer is frozen. Nevertheless, results are
adjudicated only after both formal arms complete; a candidate without its
preceding valid control is non-adjudicative.

## 8. Frozen proxy endpoint gate

After both arms complete Epoch 12, the postrun gate creates full 400-image
prediction dumps with exactly 240,000 rows per arm.

The pair is scientifically valid only if:

1. Control Epoch-12 predictions are bitwise equal to its complete zero-step
   proxy replay.
2. Candidate/control labels, rotated boxes, row counts, and row order are
   bitwise equal for every row.
3. Every parent tensor is bitwise equal; control head remains all-zero; only
   candidate head tensors may differ.
4. Both arms are finite, exact-cover, strict Q600, open-vocabulary, and free of
   forbidden inference calls.

D13-N advances to raw evaluation only if all conditions hold at the fixed
Epoch-12 endpoint:

- candidate minus control mAP `>= +0.0100`;
- candidate minus control AP50 `>= +0.0100`;
- candidate novel4 is not below control;
- candidate base14 is no more than `0.0050` below control;
- all-query micro-recall delta and fixed-recall oracle-mAP delta are each
  `>= -0.0050`;
- oracle headroom is no greater than control plus `1e-12`;
- the integer sum of AP-support localization/background and empty false
  positives falls by at least 10% (`10 * candidate <= 9 * control`), with
  neither category's integer count increasing;
- duplicate AP-support false positives increase by no more than 5%
  (`100 * candidate <= 105 * control`);
- mean existence probability is higher for matched than unmatched queries,
  with finite values on ordinary, empty, and all-matched strata.

Earlier epochs are trajectory evidence only. Best-epoch selection is
forbidden. Any failure means discard with no LR/coefficient sweep, alternate
seed, extra epoch, threshold tuning, rescue, or stacking.

## 9. Conditional raw-13,833 promotion

Passing the proxy gate automatically authorizes one paired test-only raw
evaluation using the exact Epoch-12 candidate and control checkpoints. It
does not authorize more training.

Control and candidate raw replays run sequentially on GPUs `5..9`, in that
order, with new ports and output roots. Before each replay, including the
candidate after the control delay, the owner repeats the idle, port,
artifact-collision, commit, checkpoint, resolved-config, data/analyzer hash,
and complete Stage-0 software/GPU fingerprint checks. Each dump must contain
exactly 13,833 unique image IDs, 600 rows per image, 8,299,800 finite rows, all
18 classes, and `filter_empty_gt=False`.

The raw control's decoded image IDs, row order, scores, labels, and rotated-box
tensors must be bitwise equal to the canonical E24 dump; its reconstructed
mAP must be within `1e-7` of `0.6064053488274416`, and parsed official AP50
must equal `0.6060` exactly. Candidate labels and boxes must remain bitwise
equal to control. D13-N is kept only if:

- raw candidate minus control mAP `>= +0.0050`;
- raw candidate minus control AP50 `>= +0.0050`;
- base14 and novel4 each regress by no more than `0.0010`;
- all strict/E2E/open-vocabulary/no-forbidden-call gates pass;
- all-query micro-recall delta and fixed-recall oracle-mAP delta are each
  `>= -0.0050`, while oracle headroom is strictly below control by more than
  `1e-12`.

A raw pass is a calibration keep and evidence for the larger paper program;
it is not permission to call D13-N the final ICLR contribution. A raw failure
closes this null-only family without retuning.

## 10. Failure semantics

- Any NaN/Inf, invalid Hungarian matrix, OOM, NCCL failure, query-row change,
  forbidden call, parent-tensor change, nonzero control head, output collision,
  provenance mismatch, or incomplete rank topology stops the pair.
- Infrastructure failure before a valid optimizer step is not a scientific
  metric result; diagnose once and require a fresh no-overwrite launch.
- A failure after formal optimizer steps is preserved and recorded; it is not
  silently rescued by resume.
- Candidate-only results never adjudicate the hypothesis.
- The active read-only D12 monitor remains independent and is never stopped or
  repurposed by D13-N.

## 11. Test-driven implementation boundary

Implementation follows RED -> GREEN -> REFACTOR and covers at least:

1. Zero initialization, RNG-state preservation, and exact 257-scalar count.
2. Centered log-residual identity at zero for extreme fp16/fp32 logits.
3. Nonzero residual changes only scores, never labels/boxes/rows/order.
4. DN slicing and independent `[B,3,600]` assignment targets.
5. Branchless, stratum-balanced, finite mixed, empty, and all-matched BCE
   reductions.
6. Legacy HRSC null behavior remains unchanged and inactive in D13-N.
7. Pre-optimizer exact freeze allowlist, head-only optimizer groups,
   deterministic parent feature mode, and parent-gradient exclusion.
8. Candidate one-step/head-only change and control one-step/no change.
9. E24 checkpoint-load allowlist and complete proxy zero-step equality.
10. World5 config parity, exact 1,600-manifest-row coverage, ports, paths, and
    checkpoint retention.
11. Strict Q600, open-vocabulary, evaluator-mouth, no-NMS/no-top-k, and
    forbidden-source audits.
12. Real empty/ordinary/1,223-GT forward/backward numerical stress.
13. Correct wrapper/launcher/worker recognition under `rtk`.
14. Endpoint-only parsing, tensor-delta audit, proxy thresholds, mechanism
    gates, and conditional raw authorization.
15. Atomic no-replace publication for reports, dumps, and gates.

## 12. Claim boundary

Until the paired raw gate passes, the only permitted statement is that D13-N
is an approved experiment. If it passes, the permitted statement is that a
parent-preserving existence residual improves strict all-query score
calibration on this parent. It cannot by itself support claims of solving
tiny/dense geometry, universal semantic capacity flow, open-vocabulary
generalization, or being the first method of its kind.
