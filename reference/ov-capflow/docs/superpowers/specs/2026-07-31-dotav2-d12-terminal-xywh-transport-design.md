# DOTA-v2 D12 Decoder-Terminal XYWH Transport Design

## 1. Status and purpose

The user approved the D12 mechanism on 2026-07-31. This document freezes the
written design for review before implementation.

Approved initialization erratum (2026-07-31): the first pair-builder attempt
failed closed before publishing any output because the written branch-0 bias
expectation did not match the resolved runtime model. Direct inspection after
the normal initialization path confirmed that inherited `as_two_stage=True`
resets every one of the seven terminal weights and biases to all-zero. The
user approved correcting the written contract to this observed runtime state;
the candidate's transported rows and every other scientific condition remain
unchanged.

D12 tests one causal hypothesis:

> The generic GroundingDINO checkpoint contains a useful decoder regression
> prior whose first four output rows already mean `(cx, cy, w, h)`. The
> conservative `4-D -> 5-D` shape filter discards those rows together with the
> incompatible angle row. Restoring only the six decoder branches' generic
> XYWH rows may improve optimization of the clean-start rotated Q600 detector.

D12 is an initialization experiment. It does not add a module, loss, teacher,
pseudo-label, proposal stage, post-processing operation, or inference-time
parameter. Passing this specification is not evidence that D12 improves mAP.

## 2. Evidence and risk

The read-only experiment `8-D12` established:

- six decoder regression heads have `24/24` exactly mapped hidden-layer
  weight/bias tensors;
- each generic terminal is `4 x 256`, whereas each rotated terminal is
  `5 x 256`;
- all six generic decoder terminal weights are identical and have
  L2 norm `19.562620`;
- at T7 Epoch 1, the learned target XYWH terminals have only `1.1328%` of the
  source norm, source-row mean cosine `0.109304`, and relative L2 `0.999416`;
- at Epoch 6, those values are `2.0954%`, `0.165996`, and `0.998155`;
- the seventh encoder-output regression branch stays all-zero and is not used
  to initialize fixed queries.

The source-to-target norm gap is the main safety risk. A generic terminal that
is semantically aligned can still produce saturated or numerically unstable
rotated boxes after transport. Therefore D12 cannot take an optimizer step
until checkpoint, zero-update prediction, real-GT loss, backward, sampler,
strict-inference, and open-vocabulary gates all pass.

The authoritative baseline remains T7 Epoch 24 on raw DOTA-v2.0:

- exact mAP `0.6064053488274416`;
- official `dota/mAP=0.6064`, `dota/AP50=0.6060`;
- target `mAP>=0.7000` and `AP50>=0.7000`.

D12 first uses the immutable 1,600-train/400-validation rare4x proxy. A proxy
result cannot complete the raw-data goal.

## 3. Frozen scientific contract

Candidate and control preserve:

- the same raw generic source checkpoint
  `/data/zcy/GroundingDINO/weights/groundingdino_swint_ogc.pth`;
- source SHA256
  `3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799`;
- DOTA-v2.0 canonical 18-class order and base14/novel4 split;
- the same immutable rare4x train and proxy validation manifests;
- scale `1024 x 1024`;
- fixed Q600 matching queries at train and inference;
- three independently matched training query groups;
- six decoder layers and the existing 100-query DN path;
- direct rotated 5-D `(cx, cy, w, h, angle)` predictions;
- exactly one class, one score, and one box per inference query;
- all-query scoring with no query-row deletion or sorting;
- no encoder proposal selection, proposal top-k, NMS, rotated NMS, score
  threshold, min-area conversion, dense head, RPN, or RoI head;
- the current prompt, token aggregation, losses, Hungarian costs, optimizer,
  scheduler, seed `20260716`, and Epoch-12 endpoint;
- no D11 content-query transport;
- no remote-sensing checkpoint, teacher, distillation state, pseudo-label, or
  optimizer/scheduler/EMA state in the initialization checkpoint.

The only scientific difference between the paired models is:

```text
for decoder branches b = 0..5:
    candidate.reg_terminal[b].weight[0:4]
        = generic.reg_terminal[b].weight[0:4]
    candidate.reg_terminal[b].bias[0:4]
        = generic.reg_terminal[b].bias[0:4]
```

The control keeps the target model's original initialized rows. Candidate and
control angle row 4 are bitwise identical. Branch 6 is bitwise identical and
all-zero in both. No other tensor may differ after model initialization and
checkpoint load.

## 4. Paired checkpoint construction

A dedicated pair builder will import the existing generic conversion and
source-safety helpers instead of duplicating them.

It performs one deterministic target-model construction:

1. Bind the exact source path and expected source SHA before loading.
2. Reject forbidden source names, training state, and forbidden namespaces.
3. Convert the generic state with the existing official mapping rules.
4. Build the target model from the frozen proxy control config.
5. Call the normal model initialization path so terminal values match the
   runtime model contract:
   - inherited `as_two_stage=True` invokes the head post-initialization reset;
   - every one of the seven terminal weights starts at zero;
   - every one of the seven five-element terminal biases starts at zero.
6. Create one compatible generic state shared by both outputs.
7. Materialize all seven complete `5 x 256` terminal weights and five-element
   biases from that one target state into the control checkpoint.
8. Clone the control state for the candidate and replace only rows `0:4` of
   the six decoder terminal weights and biases with their exact generic rows.
9. Leave branch 6 and all six angle rows untouched.
10. Publish two model-only checkpoints and one manifest through atomic
    no-replace writes. Existing destinations cause a fail-closed exit.

The checkpoint payload contains only `state_dict`. It must not contain
optimizer, scheduler, scaler, EMA, teacher, or runtime state.

The paired manifest records:

- schema version, git commit, source/config paths and SHA256 values;
- control and candidate output paths, sizes, and SHA256 values;
- all included keys and shapes;
- exactly twelve changed keys: six terminal weights and six terminal biases;
- exactly `6,168` authorized transported scalar positions:
  `6 * (4 * 256 + 4)`;
- the actual unequal scalar count, reported separately rather than assumed
  equal to the authorized count;
- an explicit `query_transport.enabled=false`;
- bitwise-equality results outside the twelve allowed keys;
- exact source-row equality for every candidate XYWH row;
- exact control/candidate angle-row equality;
- exact branch-6 equality and all-zero status;
- absence of all forbidden namespaces and training state.

Output names are new and immutable:

```text
work_dirs/dotav2_cleanstart/checkpoints/
  groundingdino_swint_ogc_q600_d12_w5_control_seed20260716.pth
  groundingdino_swint_ogc_q600_d12_w5_xywh_seed20260716.pth
  groundingdino_swint_ogc_q600_d12_w5_pair_manifest.json
```

## 5. Runtime configuration and ten-GPU pairing

Two new config wrappers inherit the same rare4x scale-1024 proxy parent:

```text
..._batch2_rare4x_world5_d12_control.py
..._batch2_rare4x_world5_d12_xywh.py
```

The control uses physical GPUs `5,6,7,8,9`. The candidate uses
`0,1,2,3,4`. Each has:

- `selected_world_size=5`;
- per-rank batch size `2`;
- `accumulative_counts=1`;
- nominal global batch `10`;
- `update_count_multiple=1`;
- seed `20260716`, `diff_rank_seed=False`, `resume=False`;
- a unique work directory, sampler audit, console log, tmux session, and
  distributed master port.

The 1,600-image train set divides exactly into `160` synchronized updates:

```text
1600 images / (5 ranks * 2 images) = 160 updates
```

Every rank must receive exactly 320 unique sample indices per epoch. The
sampler audit must prove zero missing, zero duplicate, zero padded, and zero
cross-rank repeated indices, with identical global ordering/checksum for
candidate and control.

The base config already defines
`train_dataloader.batch_sampler.type=DNQueryBudgetBatchSampler`. The wrappers
modify its existing fields; they may not construct an untyped batch sampler.

Both jobs launch with:

```text
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
```

under the repository-required `rtk env ...` command shape.

Checkpoint retention is overridden for both jobs with no best-checkpoint
replacement and no retention limit, so every epoch checkpoint is kept. No run
hook may remove older checkpoints, and no existing work directory, log,
audit, or checkpoint may be reused or overwritten.

## 6. Stage-0 preflight gates

Stage 0 performs no optimizer step. All gates are conjunctive.

### 6.1 Static checkpoint and config integrity

- Pair manifest and both checkpoint SHA256 values verify.
- Fully initialized and loaded candidate/control model states are bitwise
  identical outside the twelve permitted terminal keys.
- Candidate XYWH rows equal the raw generic source exactly.
- Angle rows, branch 6, Q600 content queries, fixed 5-D references, DN label
  table, classification biases, and every other tensor are bitwise identical.
- Branch 6 remains all-zero.
- Config parity differs only in GPU IDs, role/bookkeeping, load path, work
  path, sampler audit path, log/session/port metadata, and the D12 scientific
  role string.
- Resolved configs retain Q600, three train/matching groups, DN100,
  scale1024, Epoch12, official rotated evaluator, and no forbidden component.

### 6.2 Zero-update prediction distribution

With an identical fixed real proxy batch, seed, eval mode, and no optimizer:

- candidate and control output exactly 600 finite scores, labels, and 5-D
  boxes per image;
- every normalized candidate `(cx, cy, w, h)` value is finite and lies in
  `[0,1]`; every width and height is strictly positive;
- the fraction of candidate coordinates within `1e-6` of 0 or 1 is at most
  `0.05` and no more than `0.05` above control;
- candidate/control median predicted box-area ratio is within `[1/16, 16]`;
- candidate p99 predicted normalized area is at most `1.0`;
- no forbidden inference function, NMS, top-k, threshold, or row-reduction
  call is observed;
- repeated candidate inference on the same input is deterministic within the
  existing floating-point tolerance.

These bounds reject catastrophic saturation without requiring D12 to mimic
the control distribution.

### 6.3 Real-GT forward/backward

On the same real batch for both sides:

- every individual loss and total loss is finite;
- candidate total loss is at most `20x` the positive control total loss;
- all candidate trainable gradients are finite;
- candidate global pre-clip gradient norm is at most `100x` the positive
  control norm;
- no parameter changes because the optimizer is not stepped;
- Q600 prediction still succeeds after backward;
- peak estimated query area is within the frozen `50,000,000` budget.

Any threshold miss closes the launch. It does not authorize source scaling,
row normalization, interpolation, clipping, partial-layer transport, or a
different bias rule.

### 6.4 Strict and open-vocabulary audits

Both sides independently pass:

- the strict Q600 inference audit;
- the open-vocabulary audit;
- forbidden source/namespace checks;
- official config build and evaluator-mouth checks.

Failure on either side prevents both jobs from launching.

## 7. Stage-1 paired training

After Stage 0 and a pre-run commit, launch both five-GPU jobs concurrently in
new tmux sessions. Candidate and control must start within 60 seconds of one
another. A post-run owner waits fail-closed for:

- both exact process groups to exit;
- finite Epoch-12 metrics;
- atomically published Epoch-12 checkpoints;
- both work roots and sampler audits;
- physical GPUs 0-9 to become idle;
- no fatal OOM, NCCL, NaN, traceback, or sampler signature.

The post-run owner then runs the frozen paired metric gate and repeats strict
Q600/open-vocabulary audits on the exact Epoch-12 checkpoints.

Epoch 12 is the only selection checkpoint. Earlier epochs are trajectory
evidence only; best-checkpoint selection is forbidden.

## 8. Frozen decision gate

D12 is promoted from the proxy only if all conditions hold:

1. Candidate `AP50 >= 0.4890`.
2. Candidate minus matched-control `AP50 >= +0.0200`.
3. Candidate minus matched-control mAP `>= +0.0200`.
4. Candidate novel4 `>= 0.3675`.
5. Candidate novel4 is not below matched control.
6. Candidate base14 `>= 0.464643`.
7. Candidate base14 is no more than `0.005` below matched control.
8. Both sampler, numerical, strict-Q600, open-vocabulary, rotated-box, and
   no-forbidden-call gates pass.

The gate is evaluated on the frozen endpoint even if an earlier candidate
checkpoint is better. Failure means `discard` with no retuning, rescue,
source scaling, extra epoch, alternate seed, best-epoch cherry-pick, D11
stack, loss change, or prompt change.

A pass establishes only that D12 deserves a separately reviewed full-data
experiment. It does not automatically authorize a full 24-epoch run and does
not establish AP70.

## 9. Monitoring and failure handling

Monitoring records:

- exact process trees, master ports, PIDs, GPU IDs, utilization and memory;
- epoch/iteration, learning rate, all matching and DN losses, gradient norm,
  iteration time, and data time;
- sampler checksum and exact coverage;
- checkpoint/log/work-root existence and hashes;
- fatal-pattern scans and every restart or interruption boundary.

Rules:

- no automatic fallback to fewer GPUs, smaller batch, or a different
  scientific recipe;
- no automatic resume into a partial or ambiguous checkpoint;
- no timeout extension, extra epoch, or hidden relaunch;
- a fast failure is diagnosed from the first traceback before any retry;
- output collision, missing provenance, mixed process ownership, or audit
  failure is an infrastructure failure, not a D12 metric result;
- a scientific discard is still a valid completed experiment;
- all checkpoints, logs, dead tmux panes, manifests, audits, and failed
  artifacts remain preserved;
- no `git reset`, `git clean`, file deletion, or overwrite is performed.

## 10. Test-driven implementation

Implementation begins with failing tests and follows RED -> GREEN -> REFACTOR.
The minimum suite covers:

1. Exact decoder source-to-target key mapping for branches 0-5.
2. Exact XYWH weight and bias transport for all six decoder branches.
3. Angle-row preservation for every decoder branch.
4. Branch-6 preservation and all-zero enforcement.
5. Rejection of missing keys, wrong source/target shapes, dtype mismatch,
   non-finite source values, and already-populated forbidden outputs.
6. Rejection of query transport and any changed key outside the twelve-key
   allowlist.
7. Exact authorized transported-element count `6,168`, plus the independently
   computed actual unequal count.
8. Deterministic, atomic, no-replace checkpoint/manifest publication.
9. Config resolution and candidate/control parity.
10. World5 batch2 exact 1,600-image coverage on all five ranks.
11. Static full-model bitwise-delta audit after checkpoint load.
12. Zero-update box-distribution gates.
13. Real-GT finite forward/backward with no optimizer step.
14. Strict Q600, open-vocabulary, no-NMS/no-top-k, and evaluator-mouth gates.
15. Post-run endpoint parsing, class aggregation, threshold boundaries, and
    scientific-discard return semantics.

Focused tests pass before broader OVCapFlow regressions. Syntax-only checks
are insufficient; both MMEngine resolved-config parsing and real model
construction are required.

## 11. Artifacts and reporting

The experiment receives a new execution identifier under the existing
Experiment 8 ledger; the historic read-only `8-D12` audit remains unchanged.

Before any real run, `.lab/log.md` records a THINK entry covering convergence,
untested assumptions, invalidation risk, and the next hypothesis. The pre-run
commit and all hashes are appended to `.lab/results.tsv`.

The final Chinese result record includes:

- source, config, code, pair-manifest, and checkpoint hashes;
- exact scientific delta and forbidden-component audit;
- GPUs, world size, global batch, optimizer steps, sampler coverage, seed,
  duration, and terminal process status;
- per-epoch candidate/control mAP and AP50;
- Epoch-12 total, all 18 classes, novel4, and base14;
- all frozen gate calculations;
- warnings, failures, retries, and audit results;
- explicit promote/discard status and the next allowed action.

## 12. Explicit non-goals

D12 does not include:

- D11 content-query transport;
- encoder-output terminal transport;
- a new reference-point initializer;
- terminal rescaling, normalization, clipping, interpolation, or row
  selection;
- a learned angle initializer or generic horizontal-to-angle mapping;
- query selection, dynamic queries, encoder proposal ranking, or top-k;
- a quality/ranking loss, assignment change, DN change, or extra head;
- teacher, EMA teacher, pseudo-label, unlabeled-data, or external aerial
  checkpoint use;
- prompt aliases, prompt sweeps, or text-encoder changes;
- NMS, score filtering, row pruning, ensemble, TTA, or second-stage inference;
- a second seed, extra epoch, or hyperparameter sweep;
- automatic stacking or automatic full-data promotion;
- an AP70, SOTA, ICLR-readiness, or first-method claim.

## 13. Definition of done

The written design is ready for implementation only after user review.

The Stage-0 implementation is ready to launch only when:

- every TDD and project regression test passes;
- both resolved configs and the pair manifest pass independent audits;
- full loaded-model delta is restricted to the twelve allowed keys and the
  `6,168` authorized transported positions;
- all zero-update, real-GT, sampler, strict-Q600, and open-vocabulary gates
  pass;
- a pre-run commit and replayable launch commands exist;
- physical GPUs 0-9 are idle and exclusively available.

D12 is scientifically complete only when both five-GPU jobs reach the frozen
Epoch-12 endpoint and all post-run audits publish. The broader research goal
remains incomplete until a canonical raw run reaches both
`dota/mAP>=0.7000` and `dota/AP50>=0.7000`, followed by separately validated
paper-level novelty evidence.
