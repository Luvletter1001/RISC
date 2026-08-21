# OV-CapFlow ODQ Overnight Supervisor Design

**Date:** 2026-08-02  
**Status:** approved direction; written-spec review gate  
**Scope:** bounded unattended audit and execution through the full-data Epoch 6 gate  
**Devices:** physical GPUs 8 and 9 only  
**Scientific contract:** open-vocabulary, rotated-box, end-to-end, fixed Q600

## 1. Decision

Build one fail-closed overnight supervisor for the first ODQ-Flow cycle. The
supervisor may execute and audit this bounded state sequence without human
intervention:

```text
PREFLIGHT
  -> ODQ-S0
  -> PROXY_CONTROL + PROXY_CANDIDATE
  -> PROXY_EVALUATION
  -> PROXY_GATE
  -> FULL_E6_CANDIDATE
  -> FULL_E6_EVALUATION
  -> FINAL_REPORT
```

Every arrow is conditional. A stage advances only after its required artifacts
are atomically published and its frozen gates pass. Any unclassified state,
contract violation, deterministic failure, exhausted retry budget, or missing
evidence moves to `STOPPED`, never to the next experiment.

The supervisor does not invent scientific decisions. It executes a finite,
pre-registered decision table and preserves the evidence needed for later
human review.

## 2. Scope and Non-Goals

### 2.1 Included

- wait for physical GPUs 8 and 9 to become exclusively idle;
- run the five ODQ-S0 core checks;
- run matched proxy control and candidate jobs;
- monitor processes, GPUs, logs, checkpoints, and progress heartbeats;
- classify known infrastructure, numerical, configuration, and scientific
  outcomes;
- resume an unchanged job after narrowly defined transient failures;
- run canonical raw rotated evaluation and Q600 dump validation;
- calculate the frozen proxy and Epoch 6 promotion gates;
- start one two-GPU full-data candidate only after a passing proxy;
- stop after the full-data Epoch 6 decision;
- write append-only machine evidence and a final Chinese audit report.

### 2.2 Excluded

- changing model code after preflight starts;
- changing batch size, optimizer, learning rate, loss weights, support radii,
  seed, data order, prompt order, evaluator, or checkpoint source;
- using any GPU except physical 8 and 9;
- automatically entering Epoch 12, Epoch 24, matching-aware, readout-aware,
  circular-angle, or cross-attention experiments;
- killing, pausing, or signaling an unrelated process;
- deleting, overwriting, or reusing an older experiment directory;
- selecting a filtered or post-processed result over the raw rotated result;
- interpreting a near-threshold result as a pass.

## 3. Rotated-Box E2E Contract

The supervisor must audit the following before every launch and evaluation:

1. model output shape is five-dimensional `(cx, cy, w, h, theta)`;
2. the candidate's ODQ branch changes only `xywh`; angle remains the parent
   rotated-regression value in the first cycle;
3. training targets use the existing `qbox -> rbox` conversion;
4. Hungarian matching retains `RBoxL1Cost` and rotated Gaussian/KLD geometry
   cost;
5. training retains rotated KLD/GDLoss;
6. inference returns exactly 600 rotated boxes, scores, and labels per image;
7. validation uses the canonical raw DOTA rotated-IoU metric at IoU `0.5`;
8. no RPN, proposal selection, HBB surrogate evaluation, NMS, rotated-NMS,
   top-k, score threshold, duplicate suppression, box voting, or post-hoc box
   refiner appears in the effective config or command;
9. open-vocabulary token similarity and canonical prompt order remain intact;
10. dump validation proves that all 600 rows reach the evaluator.

Any violation is a contract failure. The supervisor records it and stops before
scientific training or promotion.

## 4. Autonomy Boundary

### 4.1 Scientific boundary

The maximum automatic scientific promotion is the full-data Epoch 6 result.
Even a passing Epoch 6 result cannot start Epoch 12 or Epoch 24 without a new
human approval and written execution plan.

### 4.2 Time boundary

The unattended period has a coverage target and a hard ceiling:

- `overnight_coverage_target = launch_time + 20 hours`;
- `hard_deadline = launch_time + 48 hours`.

The supervisor may finish earlier after an authoritative Epoch 6 result or a
terminal failure. It may continue beyond 20 hours only to reach the already
authorized Epoch 6 boundary. At 48 hours it cannot launch or resume work. If a
job is still healthy at the hard deadline, the supervisor records
`hard_deadline_healthy` and leaves the live job unchanged; it does not kill a
healthy job merely because the audit deadline elapsed. Automatic recovery is
disabled after that event.

### 4.3 Process authority

The supervisor may signal only a process group that it launched and whose
session ID is stored in its manifest. For a confirmed orphaned owned group it
sends `SIGTERM`, waits up to 60 seconds, then may send `SIGKILL` to the same
recorded group. It never acts on an unrelated PID merely because that PID uses
GPU 8 or 9.

## 5. Runtime Architecture

### 5.1 One supervisor process

One Python process owns the state machine. It launches stage processes in new
sessions, samples them directly, and writes state before and after every
external action. A detached `tmux` session keeps the supervisor alive when the
interactive shell disconnects.

The implementation reuses existing pure helpers where their contracts are
sufficient:

- `projects/OVCapFlow/tools/monitor_gpu_run.py` for GPU parsing, incremental
  log reading, fatal-pattern labels, and process-tree observation;
- `projects/OVCapFlow/tools/queue_test_after_training.py` for physical-GPU
  occupancy, complete-checkpoint, validation-complete, port, test command, and
  dump-validator helpers;
- `.lab/workspace/check_complete_checkpoint.py` behavior for the historical
  checkpoint publication contract;
- `projects/OVCapFlow/tools/validate_dotav2_q600_dump.py` for raw dump shape and
  row-contract validation;
- existing raw DOTA evaluator and frozen diagnostic analyzers.

The old `.lab/workspace/guard_gpu89_resume.py` is not used as the top-level
controller. It restarts every launcher exit without a failure fingerprint or
retry class, does not require owned GPUs to be idle before resume, and returns
success at its deadline even when its target is missing. Relevant pure command
builders may be migrated into the new tested supervisor, but blind restart
behavior is forbidden.

### 5.2 State persistence

Each run gets a new immutable run ID and directory:

```text
.lab/workspace/exp-odq-overnight-<run_id>/
├── manifest.json
├── state.json
├── events.jsonl
├── supervisor.lock
├── gpu_monitor.jsonl
├── preflight/
├── odq_s0/
├── proxy_control/
├── proxy_candidate/
├── proxy_evaluation/
├── proxy_gate/
├── full_e6/
└── final/
```

- `manifest.json` is written once before the first launch and is immutable;
- `events.jsonl` is append-only and flushed after each record;
- `state.json` is written to a sibling temporary file, fsynced, and atomically
  replaced;
- `supervisor.lock` contains PID, process start time, run ID, and repository
  root; a live matching lock blocks duplicates;
- stage outputs use exclusive creation and never overwrite an existing path.

### 5.3 Manifest

The manifest records:

- repository root, git commit, and exact allowed dirty/untracked paths;
- supervisor/config/checkpoint/dataset-manifest SHA-256 values;
- parent T7 checkpoint and provenance paths;
- physical indices and UUIDs for GPUs 8 and 9;
- Python executable, Python/PyTorch/CUDA/MMCV/MMEngine/MMRotate versions;
- complete commands and environment variables for every registered stage;
- `CUDA_VISIBLE_DEVICES`, `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`,
  `PYTHONNOUSERSITE=1`, and `OMP_NUM_THREADS=1`;
- query counts, rotated-E2E invariants, seeds, data order, effective batch,
  prompt protocol, and raw evaluator identity;
- all scientific thresholds, retry budgets, heartbeat limits, and deadlines;
- output collision checks and available disk at admission time.

If any launch-time value differs from the immutable manifest, the launch is
rejected.

## 6. State Machine

### 6.1 `PREFLIGHT`

Required checks:

- repository HEAD and allowed dirty set match the manifest candidate;
- correct conda environment and dependency imports are usable;
- configs compile and parse through MMEngine;
- model construction succeeds for ODQ-S0, proxy control, proxy candidate, and
  full-E6 candidate configs;
- parent checkpoint exists, has the registered size/SHA, and loads with only
  registered ODQ missing keys;
- dataset roots, annotation counts, prompt order, base14/novel4 split, and
  canonical validation mouth match the frozen protocol;
- at least 100 GiB is free on the work filesystem;
- GPU indices 8 and 9 map to the UUIDs captured at admission;
- no compute process occupies GPU 8 or 9;
- no matching train/eval/supervisor process or output path already exists;
- rendezvous ports are available;
- rotated-E2E anti-shortcut audit passes.

Failure has no retry. The run moves to `STOPPED_PREFLIGHT`.

### 6.2 `ODQ_S0`

Run the five core checks from the ODQ design:

1. shape and Q600 mouth;
2. zero-step parent equivalence;
3. finite distribution behavior and one real backward;
4. optimizer grouping and checkpoint provenance;
5. rotated-E2E anti-shortcut contract.

All five reports must be present, finite, and internally consistent. Failure
moves to `STOPPED_S0`; proxy jobs are never launched.

### 6.3 `PROXY_PAIR`

After both GPUs are idle:

- physical GPU 8 runs the frozen control;
- physical GPU 9 runs the ODQ-R1 candidate;
- both receive the same seed, sample order, effective batch, prompt protocol,
  parent source, epoch count, and rotated evaluator;
- each job has an independent log, monitor stream, work directory, and process
  group;
- a failure in one job does not silently relabel the other as valid evidence.

The supervisor waits for both terminal states. A valid pair requires both
complete checkpoints and both canonical validation/dump artifacts.

### 6.4 `PROXY_EVALUATION`

For each member of the pair:

- require the training process group to have exited;
- require its requested GPU to be idle;
- require an atomically published checkpoint;
- run the canonical raw rotated test command;
- validate the Q600 dump on CPU with GPUs hidden;
- extract raw mAP/AP50, base14, novel4, small-vehicle AP, geometry reach,
  finiteness, entropy, expectation magnitude, and support-boundary hit rate;
- bind every metric to checkpoint, config, dump, and analyzer hashes.

An evaluation-only transient may be retried once. Training is never repeated
because evaluation failed.

### 6.5 `PROXY_GATE`

Promotion requires all of:

- candidate raw rotated mAP minus control at least `+0.010`;
- candidate raw rotated AP50 minus control at least `+0.010`;
- candidate novel4 minus control no worse than `-0.005`;
- either small-vehicle AP improves by at least `+0.015` or frozen geometry
  reach improves by at least `+0.020`;
- every box, score, loss, gradient, and telemetry value is finite;
- no coordinate has more than `25%` of positive targets at either support
  boundary;
- angle output satisfies the registered parent-preservation audit;
- raw dump remains open-vocabulary, rotated, E2E, and exactly Q600.

Threshold equality passes; values below the threshold by any amount fail.
Missing metrics fail closed. No confidence interval, rounding, or oracle result
can replace a missing raw gate.

### 6.6 `FULL_E6_CANDIDATE`

Only a passing proxy can launch this stage. The candidate uses both physical
GPUs 8 and 9 under two-rank DDP with P2P and InfiniBand disabled. It inherits
the authoritative T7 full-data recipe and stops at the Epoch 6 boundary while
retaining the original 24-epoch scheduler shape.

The stage must not start a second full control run automatically. Its matched
control is the registered T7 Epoch 6 checkpoint/result after the preflight
proves config, seed, data, schedule, and disabled-path equality. If that audit
cannot establish equivalence, full promotion is blocked rather than launching
an unregistered substitute.

Completion requires:

- complete Epoch 6 checkpoint publication;
- completed raw rotated Epoch 6 validation;
- Q600 dump validation;
- candidate-control raw mAP delta at least `+0.010`;
- novel4 no worse than `-0.005`;
- positive geometry-reach change;
- all runtime, box, score, and telemetry values finite;
- no rotated-E2E contract violation.

The supervisor stops after recording pass or fail. It cannot start Epoch 12.

## 7. Monitoring

### 7.1 Sampling cadence

Every 30 seconds record:

- timestamp and state;
- launcher PID, session ID, known descendant PIDs, and liveness;
- physical GPU index and UUID;
- utilization, used/total memory, temperature, and power draw;
- train-log byte offset, last scalar timestamp, epoch, microstep, and global
  iteration when available;
- fatal pattern and failure fingerprint;
- checkpoint pointer, size, and modification time;
- free filesystem bytes.

GPU-query failure is recorded as unknown/busy and fails closed for launch or
resume decisions.

### 7.2 Heartbeat

A running training stage must advance either its parsed scalar step or its log
offset within 15 minutes. If neither advances:

1. sample the owned process tree and both GPUs again;
2. wait one additional 120-second confirmation window;
3. classify a hang only if progress is still absent;
4. terminate only the owned process group;
5. apply the transient-recovery budget.

Checkpoint writing and validation are separate phases with their own expected
log markers and do not use the training heartbeat rule while active.

## 8. Failure Classification and Recovery

### 8.1 No-retry failures

The following stop immediately:

- syntax, import, registry, config-merge, missing-key allowlist, or dataset
  error;
- repeated traceback with the same normalized fingerprint;
- NaN/Inf in loss, gradient, box, score, or ODQ telemetry;
- deterministic first-step OOM while both owned GPUs were otherwise empty;
- rotated-E2E or Q600 contract violation;
- missing or malformed required evidence;
- support-boundary saturation above the frozen limit;
- scientific gate failure.

### 8.2 Retryable transient failures

The following may consume a retry:

- NCCL timeout or communicator abort after prior healthy progress;
- rank death without a preceding deterministic Python traceback;
- external termination or host-side launcher loss;
- confirmed log/process hang;
- OOM only when the evidence proves a newly arrived unrelated GPU process
  reduced available memory after prior healthy progress;
- evaluation process failure without a deterministic traceback.

Training recovery requirements:

- all owned ranks are dead or have been terminated as one owned group;
- no unrelated process on the requested GPU is killed;
- requested GPUs are idle and UUID-stable;
- the resume checkpoint is fully published and newer than the prior resume
  source;
- command, environment, config hash, and scientific manifest are unchanged;
- 120 seconds have elapsed since teardown;
- no identical failure fingerprint has already recurred.

Retry budgets:

- maximum two transient training recoveries per stage;
- maximum one evaluation retry per checkpoint;
- zero retries for configuration, numerical, contract, or scientific failure.

Exhaustion moves to `STOPPED_RETRY_EXHAUSTED`.

## 9. Result and Response Matrix

| Observed outcome | Classification | Automatic response |
|---|---|---|
| Any S0 core check fails | implementation/contract failure | archive and stop before training |
| Control proxy fails deterministically | invalid comparison | stop; candidate alone cannot promote |
| Candidate proxy fails deterministically | candidate failure | retain control evidence and stop |
| Both proxies complete but dump is not rotated Q600 | contract failure | stop |
| mAP/AP50 gate fails | scientific negative | close ODQ-R1 first cycle and stop |
| aggregate passes but novel4 regresses too much | open-vocabulary negative | stop |
| geometry improves but raw mAP does not | diagnostic positive, scientific negative | record; do not add readout automatically |
| raw mAP improves but geometry reach does not | causally ambiguous | stop for human review |
| support boundary exceeds 25% | invalid radius coverage | stop; do not tune radius automatically |
| all proxy gates pass | promotable | launch full-E6 candidate on GPUs 8 and 9 |
| full E6 delta fails | full-data negative | stop and report |
| full E6 passes | positive recovery evidence | stop at E6 and report; no E12 launch |
| unrelated process occupies a requested GPU | external resource wait | wait; never kill it |
| healthy job reaches hard deadline | incomplete but healthy | record, disable recovery, leave job unchanged |

## 10. Outputs

Runtime artifacts remain under the run-specific `.lab/workspace` directory.
After a terminal state, write:

```text
docs/project_history/exp_20260802_odq_overnight/
├── fplan_odq_overnight_zh.md
├── faudit_odq_overnight_zh.md
├── fres_odq_overnight_zh.md
└── evidence/
```

The final report contains:

- terminal state and reason;
- exact completed and skipped stages;
- control/candidate raw rotated metrics and deltas;
- base14, novel4, small-vehicle, and geometry evidence;
- Q600/rotated-E2E audit status;
- restart history and failure fingerprints;
- checkpoint, config, dump, analyzer, and report hashes;
- GPU/resource summary;
- explicit statement that no unregistered scientific variable changed;
- next action as `stop`, `human_review`, or `eligible_for_new_E12_plan`.

The report may never say that a hypothesis passed when only an infrastructure
stage passed.

## 11. Implementation Boundary

The supervisor implementation is operational infrastructure, separate from
ODQ model code. The intended files are:

- create `projects/OVCapFlow/tools/run_odq_overnight_supervisor.py`;
- create `tests/test_projects/ov_capflow/test_odq_overnight_supervisor.py`;
- reuse and, only where tests require, minimally extend
  `monitor_gpu_run.py` and `queue_test_after_training.py`;
- create run-specific config wrappers and audit commands in the separate ODQ
  implementation plan;
- create runtime records only after every preflight test passes.

No training launch is part of implementing the supervisor. A later launch
requires fresh verification of the implemented ODQ code, supervisor tests,
five core checks, GPU UUID/occupancy, disk, configs, and manifests.

## 12. Written-Spec Acceptance Criteria

The written supervisor specification is accepted only if the reviewer agrees
that:

1. rotated-box E2E and Q600 are hard admission and promotion contracts;
2. autonomy ends at the full-data Epoch 6 decision;
3. the supervisor never tunes a scientific variable;
4. deterministic, numerical, contract, and scientific failures never retry;
5. transient recovery is exact-recipe, checkpoint-complete, and budgeted;
6. only owned process groups may be terminated;
7. only physical GPUs 8 and 9 may be used;
8. every decision is append-only, hash-bound, and fail-closed;
9. a passing infrastructure stage is never reported as a passing hypothesis.
