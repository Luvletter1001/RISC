# OV-CapFlow ODQ Overnight Supervisor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, auditable, no-human-intervention supervisor that uses only physical GPUs 8 and 9, launches the frozen rotated-box E2E ODQ sequence, classifies failures, retries only transient faults within strict budgets, applies numerical scientific gates, and stops safely at the full E6 decision boundary.

**Architecture:** A single Python supervisor owns an immutable run manifest, an atomic state snapshot, an append-and-fsync event ledger, and only the process groups it launches. Pure decision functions are separated from subprocess control so every transition and failure policy can be unit-tested. The supervisor runs ODQ-S0 and ODQ-R1 proxy jobs concurrently on one GPU each, evaluates them, applies the frozen recovery gate, and launches a two-GPU R1 E6 confirmation only on a proxy pass; it never proceeds automatically to E12 or E24.

**Tech Stack:** Python 3.8 standard library, PyTorch/MMEngine configuration parsing, MMDetection distributed launcher, NVIDIA `nvidia-smi`, pytest, existing OV-CapFlow monitoring/audit utilities, physical GPUs 8 and 9 with NCCL P2P and IB disabled.

---

## Frozen operating contract

- Coverage target: remain capable of supervising for at least 20 hours; hard deadline 48 hours after manifest creation.
- Devices: physical GPU IDs exactly `(8, 9)`. Proxy S0 owns GPU 8; proxy R1 owns GPU 9; full R1 E6 owns both 8 and 9.
- E2E geometry: rotated boxes with the fixed Q600 mouth. The supervisor must reject NMS, top-k, query truncation, encoder proposal selection, and horizontal-box substitution.
- Required predecessor: an immutable ODQ stage-zero report with `verdict == 'PASS'` whose hashes match the configs, code commit, and canonical parent checkpoint in the manifest.
- Proxy decision uses unrounded values:
  - raw mAP delta at least `+0.010`;
  - raw AP50 delta at least `+0.010`;
  - novel4 delta at least `-0.005`;
  - either small-vehicle AP delta at least `+0.015` or geometry-reach delta at least `+0.020`;
  - all metrics finite;
  - ODQ boundary-hit rate at most `0.25`.
- Full E6 uses the registered T7 Epoch 6 checkpoint/result as its control and
  must not train a second full control. It requires raw mAP delta at least
  `+0.010`, novel4 delta at least `-0.005`, strictly positive geometry-reach
  change, finite runtime/box/score/telemetry values, and an intact rotated-E2E
  Q600 dump. AP50 and small-vehicle AP remain reported evidence but are not
  additional full-E6 promotion thresholds.
- Retry policy: at most two transient training recoveries for each stage, one transient evaluation retry, and zero retries for deterministic code/config, numerical, contract, or scientific-gate failures.
- Safety: the supervisor may signal only process groups whose PID, start time, process-group ID, command fingerprint, and ownership nonce are recorded by that manifest. It may never kill an external process merely because it uses GPU 8 or 9.
- Stop boundary: `COMPLETE_E6_PASS`, `STOP_SCIENTIFIC_FAIL`, `STOP_CONTRACT_FAIL`, `STOP_DETERMINISTIC_FAIL`, `STOP_NUMERICAL_FAIL`, `STOP_RETRY_EXHAUSTED`, or `STOP_HARD_DEADLINE`. There is no automatic E12/E24 branch.

## Environment and launch command contract

All commands run from `/data1/zcy/OV-CapFlow` and begin with `rtk`. The supervisor must construct subprocess environments equivalent to:

```bash
rtk env CUDA_VISIBLE_DEVICES=8,9 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow OMP_NUM_THREADS=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_odq_overnight_supervisor.py -q
```

For one-GPU proxies it changes `CUDA_VISIBLE_DEVICES` to exactly `8` or `9`. It records physical IDs and GPU UUIDs before each launch and verifies local rank mappings without assuming that local CUDA index equals the physical index.

## Task 1: Implement pure stage, failure, retry, and scientific-gate decisions

**Files:**

- Create: `projects/OVCapFlow/tools/run_odq_overnight_supervisor.py`
- Create: `tests/test_projects/ov_capflow/test_odq_overnight_supervisor.py`

- [ ] **Step 1: Write failing tests for enum coverage and terminal outcomes**

Define expected stages and failure classes in tests:

```python
EXPECTED_STAGES = {
    'PREFLIGHT', 'S0_PROXY_TRAIN', 'R1_PROXY_TRAIN',
    'S0_PROXY_EVAL', 'R1_PROXY_EVAL', 'PROXY_GATE',
    'R1_FULL_E6_TRAIN', 'R1_FULL_E6_EVAL', 'FULL_E6_GATE',
    'COMPLETE_E6_PASS', 'STOP_SCIENTIFIC_FAIL',
    'STOP_CONTRACT_FAIL', 'STOP_DETERMINISTIC_FAIL',
    'STOP_NUMERICAL_FAIL', 'STOP_RETRY_EXHAUSTED',
    'STOP_HARD_DEADLINE',
}

EXPECTED_FAILURES = {
    'TRANSIENT_RESOURCE', 'TRANSIENT_LAUNCH', 'TRANSIENT_EVAL',
    'DETERMINISTIC_CODE_CONFIG', 'NUMERICAL', 'CONTRACT',
    'SCIENTIFIC_GATE',
}
```

Assert every failure maps to exactly one retry rule and every terminal stage is absorbing.

- [ ] **Step 2: Run the tests and confirm the module is absent**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_odq_overnight_supervisor.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement immutable enums and retry policy**

Use string enums and a frozen policy:

```python
class FailureClass(str, Enum):
    TRANSIENT_RESOURCE = 'TRANSIENT_RESOURCE'
    TRANSIENT_LAUNCH = 'TRANSIENT_LAUNCH'
    TRANSIENT_EVAL = 'TRANSIENT_EVAL'
    DETERMINISTIC_CODE_CONFIG = 'DETERMINISTIC_CODE_CONFIG'
    NUMERICAL = 'NUMERICAL'
    CONTRACT = 'CONTRACT'
    SCIENTIFIC_GATE = 'SCIENTIFIC_GATE'


@dataclass(frozen=True)
class RetryPolicy:
    train_recoveries_per_stage: int = 2
    eval_retries_per_stage: int = 1

    def budget(self, failure: FailureClass, *, evaluation: bool) -> int:
        if failure == FailureClass.TRANSIENT_EVAL and evaluation:
            return self.eval_retries_per_stage
        if failure in {FailureClass.TRANSIENT_RESOURCE,
                       FailureClass.TRANSIENT_LAUNCH} and not evaluation:
            return self.train_recoveries_per_stage
        return 0
```

- [ ] **Step 4: Write exhaustive failing gate tests at every boundary**

Use `math.nextafter` so pass/fail behavior is tested immediately below and at each threshold. Cover NaN, positive/negative infinity, absent metrics, and values whose rounded display would cross a gate.

```python
def passing_proxy_metrics():
    return {
        'control_map': 0.500,
        'candidate_map': 0.510,
        'control_ap50': 0.700,
        'candidate_ap50': 0.710,
        'control_novel4': 0.400,
        'candidate_novel4': 0.395,
        'control_small_vehicle_ap': 0.300,
        'candidate_small_vehicle_ap': 0.315,
        'control_geometry_reach': 0.200,
        'candidate_geometry_reach': 0.200,
        'boundary_hit_rate': 0.25,
    }
```

Assert the exact threshold passes, one ULP below fails, and the secondary evidence uses logical OR.

- [ ] **Step 5: Implement pure gate functions**

Return a structured decision, not a bare boolean:

```python
@dataclass(frozen=True)
class GateThresholds:
    map_delta: float = 0.010
    ap50_delta: float = 0.010
    novel4_floor: float = -0.005
    small_vehicle_delta: float = 0.015
    geometry_reach_delta: float = 0.020
    boundary_hit_max: float = 0.25


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    deltas: Mapping[str, float]
    checks: Mapping[str, bool]
    failures: Tuple[str, ...]


def evaluate_proxy_gate(metrics: Mapping[str, float],
                        thresholds: GateThresholds) -> GateDecision:
    required = (
        'control_map', 'candidate_map',
        'control_ap50', 'candidate_ap50',
        'control_novel4', 'candidate_novel4',
        'control_small_vehicle_ap', 'candidate_small_vehicle_ap',
        'control_geometry_reach', 'candidate_geometry_reach',
        'boundary_hit_rate',
    )
    missing = tuple(name for name in required if name not in metrics)
    if missing:
        return GateDecision(False, {}, {'complete': False},
                            tuple(f'missing:{name}' for name in missing))
    values = {name: float(metrics[name]) for name in required}
    non_finite = tuple(name for name, value in values.items()
                       if not math.isfinite(value))
    if non_finite:
        return GateDecision(False, {}, {'finite': False},
                            tuple(f'non_finite:{name}' for name in non_finite))
    deltas = {
        'map': values['candidate_map'] - values['control_map'],
        'ap50': values['candidate_ap50'] - values['control_ap50'],
        'novel4': values['candidate_novel4'] - values['control_novel4'],
        'small_vehicle_ap': (values['candidate_small_vehicle_ap'] -
                             values['control_small_vehicle_ap']),
        'geometry_reach': (values['candidate_geometry_reach'] -
                           values['control_geometry_reach']),
    }
    checks = {
        'finite': True,
        'map': deltas['map'] >= thresholds.map_delta,
        'ap50': deltas['ap50'] >= thresholds.ap50_delta,
        'novel4': deltas['novel4'] >= thresholds.novel4_floor,
        'secondary': (deltas['small_vehicle_ap'] >= thresholds.small_vehicle_delta or
                      deltas['geometry_reach'] >= thresholds.geometry_reach_delta),
        'boundary_hit': values['boundary_hit_rate'] <= thresholds.boundary_hit_max,
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    return GateDecision(not failures, deltas, checks, failures)
```

- [ ] **Step 6: Run the pure decision tests**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_odq_overnight_supervisor.py -q -k "stage or failure or retry or gate"
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit pure decisions**

```bash
rtk git add projects/OVCapFlow/tools/run_odq_overnight_supervisor.py tests/test_projects/ov_capflow/test_odq_overnight_supervisor.py
rtk git diff --cached --check
rtk git commit -m "feat: add ODQ supervisor decision model"
```

## Task 2: Add immutable manifest, atomic state, event ledger, and singleton lock

**Files:**

- Modify: `projects/OVCapFlow/tools/run_odq_overnight_supervisor.py`
- Modify: `tests/test_projects/ov_capflow/test_odq_overnight_supervisor.py`
- Reuse: `projects/OVCapFlow/ov_capflow/no_replace.py`

- [ ] **Step 1: Write failing filesystem tests in temporary directories**

Test these invariants:

- manifest is created exclusively and never overwritten;
- manifest SHA256 is stored in every state/event record;
- state write uses a same-directory temporary file, flush, `os.fsync`, `os.replace`, and parent-directory fsync;
- events are one JSON object per line, flushed and fsynced before returning;
- a live lock holder prevents a second supervisor;
- a stale lock is not silently stolen; `--recover-stale-lock` requires matching manifest hash and a proven-dead owner;
- malformed or truncated state fails closed while the append-only ledger remains readable;
- resuming with a different config, checkpoint, Git SHA, GPU UUID, or stage-zero report hash is rejected.

- [ ] **Step 2: Run the tests and observe missing storage classes**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_odq_overnight_supervisor.py -q -k "manifest or state or ledger or lock"
```

- [ ] **Step 3: Implement canonical JSON and fingerprints**

Use UTF-8 JSON with sorted keys and compact separators for hashing:

```python
def canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
```

The manifest records absolute repo/run paths, config hashes, checkpoint hash, stage-zero report hash, Git SHA, dirty diff hash, interpreter and package versions, command templates, GPU physical IDs/UUIDs, thresholds, retry budgets, seed, start/deadline timestamps, and an ownership nonce generated with `secrets.token_hex(32)`.

- [ ] **Step 4: Implement durable stores and lock lifecycle**

Create `ManifestStore`, `AtomicStateStore`, `EventLedger`, and `SupervisorLock`. Use `fcntl.flock(LOCK_EX | LOCK_NB)` for the live lock and retain the open file descriptor for the supervisor lifetime. Lock content includes PID, `/proc/<pid>/stat` start time, hostname, manifest hash, and nonce.

- [ ] **Step 5: Run storage tests**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_odq_overnight_supervisor.py -q -k "manifest or state or ledger or lock"
```

- [ ] **Step 6: Commit durable state support**

```bash
rtk git add projects/OVCapFlow/tools/run_odq_overnight_supervisor.py tests/test_projects/ov_capflow/test_odq_overnight_supervisor.py
rtk git diff --cached --check
rtk git commit -m "feat: persist ODQ supervisor state safely"
```

## Task 3: Implement preflight, GPU identity, and rotated E2E contract audit

**Files:**

- Modify: `projects/OVCapFlow/tools/run_odq_overnight_supervisor.py`
- Modify: `tests/test_projects/ov_capflow/test_odq_overnight_supervisor.py`
- Reuse: `projects/OVCapFlow/tools/monitor_gpu_run.py`
- Reuse: `projects/OVCapFlow/tools/queue_test_after_training.py`
- Reuse: `projects/OVCapFlow/tools/audit_odq_stage0.py`

- [ ] **Step 1: Write failing tests for GPU and preflight evidence**

Feed recorded `nvidia-smi` CSV into a pure parser. Require exactly these fields per physical GPU: index, UUID, utilization, used memory, total memory, temperature, and power draw. Test rejection of missing IDs, duplicate UUIDs, MIG remapping, non-idle GPUs, and identity changes between manifest creation and launch.

Define idle as both utilization at most 5 percent and used memory at most 512 MiB for three samples at least 10 seconds apart. A busy GPU waits without consuming a retry; it does not preempt or kill any process.

- [ ] **Step 2: Write failing effective-config contract tests**

Load the real config objects and assert preflight rejects:

- any physical GPU set other than exactly the required role set;
- missing/failed/stale stage-zero report or a hash mismatch;
- free disk below 100 GiB in the run filesystem;
- Q other than 600;
- non-five-dimensional final boxes;
- non-rotated assigner/evaluator/data conversion;
- NMS, top-k, max-per-image truncation, query selection, or encoder proposal selection;
- S0/R1 proxy differences beyond ODQ switches, work directory, and GPU role;
- a full E6 recipe that alters the architecture, optimizer, data order, seed, or scheduler instead of only shortening the stop epoch.
- a registered T7 Epoch 6 control bundle whose checkpoint, effective config,
  training audit, raw metric record, or Q600 diagnostic hashes are absent or do
  not prove equality with the candidate's parent recipe.

- [ ] **Step 3: Run the preflight tests and observe failures**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_odq_overnight_supervisor.py -q -k "gpu or idle or preflight or contract"
```

- [ ] **Step 4: Implement GPU query and stable-idle sampling**

Invoke only this fixed argv through `subprocess.run` without a shell:

```python
(
    'nvidia-smi',
    '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
    '--format=csv,noheader,nounits',
)
```

Parse strictly, store raw samples in the ledger, and compare UUIDs with the manifest before every launch and resume.

- [ ] **Step 5: Implement preflight as explicit named checks**

Return a list of `{name, passed, evidence, failure_class}` records. Reuse pure helpers from existing tools where their behavior is already tested, but keep a single top-level supervisor verdict. Include:

- environment identity and package versions;
- repository SHA and dirty diff hash;
- stage-zero report verification;
- config/checkpoint hashes;
- checkpoint completeness;
- disk capacity;
- port availability;
- stable GPU identity/idle state;
- exact rotated E2E effective-config contract;
- dry-run argv/environment for every stage;
- uniqueness of work/log/report paths.

- [ ] **Step 6: Run preflight tests**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_odq_overnight_supervisor.py -q -k "gpu or idle or preflight or contract"
```

- [ ] **Step 7: Commit preflight**

```bash
rtk git add projects/OVCapFlow/tools/run_odq_overnight_supervisor.py tests/test_projects/ov_capflow/test_odq_overnight_supervisor.py
rtk git diff --cached --check
rtk git commit -m "audit: enforce ODQ overnight preflight"
```

## Task 4: Implement owned-process monitoring and bounded recovery

**Files:**

- Modify: `projects/OVCapFlow/tools/run_odq_overnight_supervisor.py`
- Modify: `tests/test_projects/ov_capflow/test_odq_overnight_supervisor.py`
- Reuse: `projects/OVCapFlow/tools/monitor_gpu_run.py`

- [ ] **Step 1: Write failing process-ownership tests with injected fakes**

Use fake clocks, fake process readers, and temporary scripts. Do not spawn GPU jobs in unit tests. Assert:

- a launched child uses a new process group and is recorded before monitoring begins;
- ownership requires exact PID, `/proc` start time, process-group ID, nonce, and normalized argv fingerprint;
- a PID reused with a different start time is external and never signaled;
- an external process on GPU 8/9 is logged but never killed;
- graceful stop sends SIGTERM to only the owned group, waits a bounded interval, then SIGKILLs only that same verified group;
- a healthy process crossing the 20-hour coverage target continues until stage completion;
- the hard 48-hour deadline stops an owned process safely and writes `STOP_HARD_DEADLINE`;
- retry counters persist across supervisor restarts and cannot reset by relaunching the supervisor.

- [ ] **Step 2: Write failure-classification tests**

Classify from exit code, log tail, checkpoint evidence, and metrics:

- CUDA OOM, NCCL transport/bootstrap timeout, transient dataloader I/O, and unavailable launch port are transient only when their explicit signatures match;
- syntax/import/config/type/shape/assertion errors are deterministic;
- NaN, Inf, non-finite loss/gradient, and anomaly-detection failures are numerical;
- GPU identity/config/hash/Q600/rotated-E2E violations are contract failures;
- a completed run below the metric gate is scientific failure;
- unknown failures are deterministic and receive no retry.

- [ ] **Step 3: Run selected tests and observe missing monitor logic**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_odq_overnight_supervisor.py -q -k "owned or process or signal or classify or deadline or recovery"
```

- [ ] **Step 4: Implement `ProcessSpec` and `OwnedProcess`**

Use frozen specs with tuples for argv and an allowlisted environment:

```python
@dataclass(frozen=True)
class ProcessSpec:
    stage: Stage
    argv: Tuple[str, ...]
    cwd: str
    environment: Mapping[str, str]
    log_path: str
    checkpoint_dir: str
    expected_gpu_ids: Tuple[int, ...]


@dataclass(frozen=True)
class OwnedProcess:
    pid: int
    proc_start_ticks: int
    pgid: int
    ownership_nonce: str
    command_fingerprint: str
    launched_at: str
```

Launch with `shell=False`, `start_new_session=True`, explicit cwd, explicit log file, and an allowlisted environment formed from required runtime variables. Never interpolate user content into a shell string.

- [ ] **Step 5: Implement heartbeat and progress health**

Sample every 120 seconds and emit a durable heartbeat at least every 15 minutes. Each heartbeat records stage, elapsed/deadline time, owned PID identity, GPU samples, latest epoch/iteration, latest finite losses, checkpoint modification time, log offset, retry counts, and decision. Treat a log that advances but has not yet emitted an iteration as startup, bounded by an explicit 20-minute startup timeout.

- [ ] **Step 6: Implement bounded recovery**

For a retryable training failure:

1. Re-verify ownership and terminate only the failed owned group.
2. Re-verify config/code/checkpoint hashes and GPU UUIDs.
3. Require stable idle samples.
4. Resume only from a checkpoint that passes completeness and provenance checks.
5. Use the identical normalized recipe and increment the persisted stage retry count.
6. Stop when the budget is exhausted.

Evaluation retry repeats the exact evaluator command once and never retrains. Numerical, deterministic, contract, and scientific failures stop immediately.

- [ ] **Step 7: Run process and recovery tests**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_odq_overnight_supervisor.py -q -k "owned or process or signal or classify or deadline or recovery or heartbeat"
```

- [ ] **Step 8: Commit monitoring and recovery**

```bash
rtk git add projects/OVCapFlow/tools/run_odq_overnight_supervisor.py tests/test_projects/ov_capflow/test_odq_overnight_supervisor.py
rtk git diff --cached --check
rtk git commit -m "feat: supervise owned ODQ processes with bounded recovery"
```

## Task 5: Freeze proxy-control, proxy-candidate, and full-E6 recipes

**Files:**

- Create: `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_odq_proxy_control_gpu8.py`
- Create: `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_odq_proxy_candidate_gpu9.py`
- Create: `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_odq_full_e6_gpu89.py`
- Create: `tests/test_projects/ov_capflow/test_odq_supervisor_configs.py`

- [ ] **Step 1: Write failing effective-config equivalence tests**

The proxy pair must have identical:

- canonical T7 checkpoint and initialization policy;
- seed and deterministic data ordering;
- 1,600-image training subset;
- 160 optimizer updates per epoch for 12 proxy epochs;
- proxy-400 validation subset and complete 18-class rotated DOTA evaluator;
- one image per GPU, optimizer, scheduler, augmentations, query count, matching groups, DN, costs, GDLoss, text prompts, and all non-ODQ model settings.

The only scientific difference is S0 versus R1 `alpha/loss_weight`; GPU role and output path are operational differences. The full E6 recipe must inherit the R1 full recipe, set `max_epochs=6`, retain scheduler horizons from the 24-epoch recipe, use two GPUs with the inherited per-GPU batch, and evaluate on the complete agreed validation mouth.

The registered full-control evidence is anchored to these existing artifacts:

```text
work_dirs/dotav2_cleanstart/full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/epoch_6.pth
work_dirs/dotav2_cleanstart/full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/20260718_113515/vis_data/20260718_113515.json
work_dirs/dotav2_cleanstart/audits/full24_scale1024_rare4x_seed20260716_gpu89_b2_epochs/epoch_6.json
```

Before any automatic launch, preflight also requires a registered Q600 E6
diagnostic bundle at
`work_dirs/odq/registered_controls/t7_e6_q600/diagnostics/diagnostics.json`.
If that evidence is absent, it is a contract stop; the supervisor does not
generate it, launch a substitute control, or weaken the full-E6 reach gate.

- [ ] **Step 2: Run tests and confirm the recipe files are absent**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_odq_supervisor_configs.py -q
```

- [ ] **Step 3: Create thin, auditable config wrappers**

Reuse the already validated subset definitions from the D13 proxy family, but
inherit the new ODQ S0/R1 configs. The supervisor sets stage work directories as
children of its explicit run root, for example
`work_dirs/odq/overnight/20260802_odq_first/proxy_s0`, rather than editing files
after manifest creation. Record an `experiment_contract` dictionary with stage,
expected physical GPU role, dataset fingerprints, parent config, checkpoint
hash, and rotated E2E mouth.

- [ ] **Step 4: Parse configs and run equivalence tests**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_odq_configs.py tests/test_projects/ov_capflow/test_odq_supervisor_configs.py -q
```

- [ ] **Step 5: Commit recipes**

```bash
rtk git add configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_odq_proxy_control_gpu8.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_odq_proxy_candidate_gpu9.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_odq_full_e6_gpu89.py tests/test_projects/ov_capflow/test_odq_supervisor_configs.py
rtk git diff --cached --check
rtk git commit -m "config: freeze ODQ overnight stage recipes"
```

## Task 6: Implement deterministic stage orchestration and metrics extraction

**Files:**

- Modify: `projects/OVCapFlow/tools/run_odq_overnight_supervisor.py`
- Modify: `tests/test_projects/ov_capflow/test_odq_overnight_supervisor.py`
- Reuse: `projects/OVCapFlow/tools/d12_postrun_gate.py`
- Reuse: `projects/OVCapFlow/tools/analyze_dotav2_q600_dump.py`
- Reuse: `projects/OVCapFlow/tools/dotav2_q600_diagnostics.py`

- [ ] **Step 1: Write an end-to-end fake-backend state-machine test**

Inject a backend whose train/eval calls return recorded artifacts. Test the successful sequence exactly:

```text
PREFLIGHT
  -> S0_PROXY_TRAIN and R1_PROXY_TRAIN
  -> S0_PROXY_EVAL and R1_PROXY_EVAL
  -> PROXY_GATE
  -> R1_FULL_E6_TRAIN
  -> R1_FULL_E6_EVAL
  -> FULL_E6_GATE
  -> COMPLETE_E6_PASS
```

Although proxy processes run concurrently, transitions are serialized in the ledger. Test every stop outcome, supervisor restart at every non-terminal state, and the invariant that no full job is launched before a durable proxy-gate pass record.

- [ ] **Step 2: Write strict metric-parser tests**

Use complete recorded 18-class DOTA tables. Extract raw mAP, AP50, base14, novel4, the frozen small-vehicle class aggregate, geometry reach, boundary-hit rate, dataset/evaluator fingerprints, and prediction count. Reject truncated tables, duplicate/missing classes, rounded-only summaries, non-finite values, mismatched sample counts, wrong evaluator, and any mouth other than Q600 all rows.

- [ ] **Step 3: Run selected tests and observe missing orchestration**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_odq_overnight_supervisor.py -q -k "orchestr or transition or metric or resume or successful"
```

- [ ] **Step 4: Implement strict metrics extraction**

Reuse tested DOTA table parsing from `d12_postrun_gate.py` and diagnostics from the Q600 tools. Extract from machine-readable artifacts whenever available, use log parsing only as a cross-check, and record SHA256 for every source artifact. Define the small-vehicle class list once in the manifest and refuse runtime changes.

- [ ] **Step 5: Implement proxy concurrency and join semantics**

After one shared preflight and stable-idle check:

- launch S0 with `CUDA_VISIBLE_DEVICES=8` and R1 with `CUDA_VISIBLE_DEVICES=9`;
- use distinct ports, work directories, logs, and process groups;
- monitor both without treating the first completion as permission to reuse its GPU;
- do not evaluate or gate until both training outcomes are terminal-success;
- if either has a non-retryable failure, terminate the other only if it is owned, record the stop reason, and do not launch full E6.

- [ ] **Step 6: Implement evaluation, proxy gate, and full E6 gate**

Evaluation uses the checkpoint selected by the frozen policy, validates completeness, executes once with one allowed transient retry, and writes immutable metric artifacts. A proxy pass enables the exact two-GPU E6 R1 job with:

```text
CUDA_VISIBLE_DEVICES=8,9
NCCL_P2P_DISABLE=1
NCCL_IB_DISABLE=1
OMP_NUM_THREADS=1
```

Use `torch.distributed.run --nproc_per_node=2` or the repository's tested launcher with a verified free port. Never use more than these two physical GPUs. After E6 evaluation and gate, write the terminal report and stop.

The full gate reads only the preflight-registered T7 E6 control bundle. Require
candidate-control raw mAP delta at least `+0.010`, novel4 delta at least
`-0.005`, geometry-reach delta strictly greater than zero, finite values, and a
valid Q600 rotated dump. Record AP50 and small-vehicle deltas without using them
to change this decision. If the registered control bundle is missing or fails
provenance equivalence, stop with `STOP_CONTRACT_FAIL`; never launch a second
full control automatically.

- [ ] **Step 7: Implement the terminal report**

The report includes manifest hash, code/config/checkpoint/report hashes, full event-ledger hash, all stage timings, commands and allowlisted environments, GPU identities, retry history, artifact paths/hashes, raw metrics/deltas/checks, failure classifications, final outcome, and an explicit `automatic_continuation_allowed: false`.

- [ ] **Step 8: Run all supervisor tests**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_odq_overnight_supervisor.py tests/test_projects/ov_capflow/test_odq_supervisor_configs.py -q
```

- [ ] **Step 9: Commit orchestration**

```bash
rtk git add projects/OVCapFlow/tools/run_odq_overnight_supervisor.py tests/test_projects/ov_capflow/test_odq_overnight_supervisor.py
rtk git diff --cached --check
rtk git commit -m "feat: orchestrate gated ODQ overnight experiments"
```

## Task 7: Add dry-run, check-once, and safe resume CLI modes

**Files:**

- Modify: `projects/OVCapFlow/tools/run_odq_overnight_supervisor.py`
- Modify: `tests/test_projects/ov_capflow/test_odq_overnight_supervisor.py`

- [ ] **Step 1: Write failing CLI tests**

Require these arguments:

```text
--run-root
--stage0-report
--parent-checkpoint
--full-e6-control-checkpoint
--full-e6-control-training-metrics
--full-e6-control-training-audit
--full-e6-control-diagnostics
--proxy-control-config
--proxy-candidate-config
--full-e6-config
--seed
--minimum-coverage-hours
--hard-deadline-hours
--dry-run
--check-once
--resume
--recover-stale-lock
```

Test that `--dry-run` performs config/environment/report/hash/GPU/disk/port checks and prints normalized commands without creating a manifest or launching a process. `--check-once` may create or resume state, samples health once, writes one durable event, and exits without changing stage. `--resume` requires an existing exact manifest. Conflicting modes fail before mutation.

- [ ] **Step 2: Run CLI tests and observe failure**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_odq_overnight_supervisor.py -q -k "cli or dry_run or check_once or resume"
```

- [ ] **Step 3: Implement argument validation and command preview**

Defaults are `seed=20260802`, `minimum_coverage_hours=20.0`, and `hard_deadline_hours=48.0`. Require absolute `run-root` after resolution and require it to be a new child beneath `work_dirs/odq/overnight`; refuse repository root, home, or filesystem-root targets. Print and record normalized argv arrays, not shell strings.

- [ ] **Step 4: Implement dry-run side-effect boundary**

Dry-run may read files, query GPUs, check disk/ports, import configs, validate the stage-zero report, and print the manifest preview. It must not create a run directory, acquire a persistent lock, start a process, signal a PID, write state, or change configs.

- [ ] **Step 5: Run CLI tests and syntax check**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_odq_overnight_supervisor.py -q
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m py_compile projects/OVCapFlow/tools/run_odq_overnight_supervisor.py
```

- [ ] **Step 6: Commit the CLI**

```bash
rtk git add projects/OVCapFlow/tools/run_odq_overnight_supervisor.py tests/test_projects/ov_capflow/test_odq_overnight_supervisor.py
rtk git diff --cached --check
rtk git commit -m "feat: add safe ODQ supervisor CLI modes"
```

## Task 8: Verify the supervisor and prepare, but do not launch, the overnight run

**Files:**

- Verify all files changed by Tasks 1–7

- [ ] **Step 1: Run the full focused suite**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_oriented_distribution.py tests/test_projects/ov_capflow/test_odq_head.py tests/test_projects/ov_capflow/test_odq_configs.py tests/test_projects/ov_capflow/test_odq_stage0.py tests/test_projects/ov_capflow/test_odq_supervisor_configs.py tests/test_projects/ov_capflow/test_odq_overnight_supervisor.py -q
```

- [ ] **Step 2: Run the existing OV-CapFlow regression suite**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow -q
```

- [ ] **Step 3: Run a no-write dry-run using the immutable stage-zero PASS report**

Use the explicit first-run path below if it does not exist. If it already exists,
choose a new explicit suffixed sibling before invoking the dry-run; dry-run must
not create either path.

```bash
rtk env CUDA_VISIBLE_DEVICES=8,9 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow OMP_NUM_THREADS=1 /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/run_odq_overnight_supervisor.py --run-root /data1/zcy/OV-CapFlow/work_dirs/odq/overnight/20260802_odq_first --stage0-report /data1/zcy/OV-CapFlow/work_dirs/odq/stage0/20260802_odq_r1_first/report.json --parent-checkpoint /data1/zcy/OV-CapFlow/work_dirs/dotav2_cleanstart/full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/epoch_24.pth --full-e6-control-checkpoint /data1/zcy/OV-CapFlow/work_dirs/dotav2_cleanstart/full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/epoch_6.pth --full-e6-control-training-metrics /data1/zcy/OV-CapFlow/work_dirs/dotav2_cleanstart/full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/20260718_113515/vis_data/20260718_113515.json --full-e6-control-training-audit /data1/zcy/OV-CapFlow/work_dirs/dotav2_cleanstart/audits/full24_scale1024_rare4x_seed20260716_gpu89_b2_epochs/epoch_6.json --full-e6-control-diagnostics /data1/zcy/OV-CapFlow/work_dirs/odq/registered_controls/t7_e6_q600/diagnostics/diagnostics.json --proxy-control-config configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_odq_proxy_control_gpu8.py --proxy-candidate-config configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_odq_proxy_candidate_gpu9.py --full-e6-config configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_odq_full_e6_gpu89.py --seed 20260802 --minimum-coverage-hours 20 --hard-deadline-hours 48 --dry-run
```

Expected: preflight `PASS`, physical GPU IDs/UUIDs 8 and 9, exact command
previews, no created run directory, and no process launch.

- [ ] **Step 4: Verify there are no training processes created by the implementation**

```bash
rtk pgrep -af "tools/train.py|torch.distributed.run|run_odq_overnight_supervisor.py"
rtk nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
```

Existing unrelated processes may be present; record them and do not signal them. The implementation itself must have launched none.

- [ ] **Step 5: Scan for unsafe or incomplete implementation**

```bash
rtk rg -n "TODO|TBD|FIXME|NotImplementedError|shell=True|os\.system|killall|pkill|CUDA_VISIBLE_DEVICES=0|nproc_per_node=10|topk|nms|max_per_img" projects/OVCapFlow/tools/run_odq_overnight_supervisor.py configs/ov_capflow/dotav2/*odq*.py tests/test_projects/ov_capflow/test_odq_overnight_supervisor.py tests/test_projects/ov_capflow/test_odq_supervisor_configs.py
```

Every match must be an intentional rejection test or removed before completion.

- [ ] **Step 6: Review repository state and evidence**

```bash
rtk git status --short
rtk git diff --check
rtk git log --oneline -10
```

Do not commit work-directory artifacts, audit reports, generated metrics, checkpoints, logs, locks, or state files.

## Launch handoff boundary

Implementation is complete only when all tests pass, the five-check ODQ stage-zero report is immutable and `PASS`, and the supervisor dry-run passes with physical GPU UUIDs for exactly GPUs 8 and 9. The actual non-dry launch is a separate execution action. Once explicitly entered, the supervisor owns the approved S0/R1 proxy-to-E6 sequence for 20-hour minimum coverage within a 48-hour hard deadline, applies the frozen retry/gate policies without human approval, and stops after E6 with no automatic E12/E24 continuation.
