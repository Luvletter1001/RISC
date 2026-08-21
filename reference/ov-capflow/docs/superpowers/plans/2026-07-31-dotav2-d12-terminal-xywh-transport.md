# D12 Terminal XYWH Transport Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Use `test-driven-development` for every production-code change, `project-conda-env-selection` before Python execution, `distributed-training-nccl-a40` for the dual five-GPU launch, `openrsd-batch-sampler-safety` for sampler/config changes, and `verification-before-completion` before any completion claim.

**Goal:** Run one frozen, paired DOTA-v2 proxy experiment that tests only whether transporting the generic GroundingDINO decoder terminal XYWH rows into OV-CapFlow improves the existing strict Q600 end-to-end rotated detector.

**Architecture:** Build a no-clobber pair of complete target-state checkpoints from one initialized target model, then mutate exactly the first four terminal-regression output rows of decoder branches 0–5 in the candidate clone. Drive candidate and control through matched world-size-5 configs, a fail-closed Stage 0 audit, concurrent 5+5 GPU training, and a postrun gate that evaluates only epoch 12 against the frozen thresholds.

**Tech Stack:** Python 3.8, PyTorch 1.12.1, MMEngine 0.10.4, MMDetection 3.3.0, MMRotate 1.0.0rc1, pytest, torchrun/MMEngine distributed launcher, tmux, NVIDIA A40.

**Approved design:** `docs/superpowers/specs/2026-07-31-dotav2-d12-terminal-xywh-transport-design.md`

**Selected interpreter:** `/data/zcy/anaconda3/envs/mmdet/bin/python`

---

## File map and responsibilities

| File | Responsibility |
|---|---|
| `projects/OVCapFlow/tools/prepare_d12_checkpoint_pair.py` | Validate the raw generic source, initialize one full target model, build complete paired checkpoints, prove the 12-key/6,168-position boundary, and publish three artifacts atomically without replacement. |
| `tests/test_projects/ov_capflow/test_d12_checkpoint_pair.py` | Unit-test allowed keys, exact row transport, preserved angle/branch 6, fail-closed validation, deterministic manifest accounting, and no-clobber publication. |
| `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_d12_control.py` | Frozen control protocol on physical GPUs 5–9, complete control checkpoint, exact-cover batch settings, and retain-all epoch checkpoints. |
| `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_d12_xywh.py` | Candidate wrapper on physical GPUs 0–4 whose only scientific delta is the D12 checkpoint path/role. |
| `tests/test_projects/ov_capflow/test_d12_world5_configs.py` | Prove pair parity, world-size/global-batch arithmetic, typed sampler preservation, exact 1,600-example cover, seed/resume invariants, and checkpoint retention. |
| `projects/OVCapFlow/tools/audit_d12_preflight.py` | Fail-closed checkpoint/model delta audit plus fixed-real-batch no-update prediction/loss/gradient checks; emit one immutable Stage 0 JSON report. |
| `tests/test_projects/ov_capflow/test_d12_preflight.py` | Unit-test numerical gates, delta-boundary rejection, saturation/area summaries, loss/gradient ratio checks, determinism checks, and no-clobber report publication. |
| `projects/OVCapFlow/tools/d12_postrun_gate.py` | Wait for exact paired processes to exit, require epoch-12 metrics/checkpoints/sampler audits/GPU idleness, run strict/open-vocabulary audits, compute frozen deltas, and emit an immutable promote/discard report. |
| `tests/test_projects/ov_capflow/test_d12_postrun_gate.py` | Unit-test process matching, completion evidence, epoch-12-only parsing, metric thresholds, integrity conjunction, and no-clobber output behavior. |
| `.lab/log.md` | Append-only experiment decision/work log; record spec confirmation, preflight evidence, launch identity, monitoring evidence, and final result. |
| `docs/project_history/exp_20260731_d12_terminal_xywh_transport.md` | Chinese result record containing provenance, exact commands, hashes, Stage 0 evidence, epoch-12 metrics, integrity audits, and the frozen decision. |

Existing D11 artifacts and untracked user files are read-only references. Do not overwrite, delete, move, reset, clean, or reuse their output paths.

---

### Task 1: Freeze the implementation plan

**Files:**

- Create: `docs/superpowers/plans/2026-07-31-dotav2-d12-terminal-xywh-transport.md`
- Modify: `.lab/log.md`

- [ ] **Step 1: Verify the approved spec and plan exist**

Run:

```bash
rtk test -s docs/superpowers/specs/2026-07-31-dotav2-d12-terminal-xywh-transport-design.md
rtk test -s docs/superpowers/plans/2026-07-31-dotav2-d12-terminal-xywh-transport.md
```

Expected: both commands exit 0.

- [ ] **Step 2: Self-review the plan**

Run:

```bash
rtk rg -n '[T]ODO|[T]BD|place[h]older|fill in lat[e]r' docs/superpowers/plans/2026-07-31-dotav2-d12-terminal-xywh-transport.md
rtk rg -n 'prepare_d12_checkpoint_pair|world5_d12|audit_d12_preflight|d12_postrun_gate|6168|0\\.489|0\\.020|0\\.3675|0\\.464643' docs/superpowers/plans/2026-07-31-dotav2-d12-terminal-xywh-transport.md
```

Expected: the first command has no matches; the second covers every implementation surface and frozen threshold.

- [ ] **Step 3: Record approval and commit only the plan**

Append an ISO-8601 entry to `.lab/log.md` stating that the user confirmed the written spec and implementation planning began. Then run:

```bash
rtk git status --short
rtk git add docs/superpowers/plans/2026-07-31-dotav2-d12-terminal-xywh-transport.md
rtk git diff --cached --check
rtk git commit -m "docs: plan D12 terminal XYWH transport"
```

Expected: the commit contains only the plan; `.lab/log.md` remains append-only and may be ignored by Git.

---

### Task 2: Build paired complete checkpoints with TDD

**Files:**

- Create: `tests/test_projects/ov_capflow/test_d12_checkpoint_pair.py`
- Create: `projects/OVCapFlow/tools/prepare_d12_checkpoint_pair.py`
- Read/import: `projects/OVCapFlow/tools/prepare_cleanstart_checkpoint.py`

- [ ] **Step 1: Confirm the selected environment before Python work**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -c "import sys, torch, mmengine, mmdet, mmrotate; print(sys.executable); print(torch.__version__, mmengine.__version__, mmdet.__version__, mmrotate.__version__)"
```

Expected: `/data/zcy/anaconda3/envs/mmdet/bin/python`, PyTorch `1.12.1+cu113`, MMEngine `0.10.4`, MMDetection `3.3.0`, MMRotate `1.0.0rc1`.

- [ ] **Step 2: Write the failing pair-construction tests**

The tests use tiny ordered tensor mappings and temporary paths; they must not load the production model. Cover:

```python
def test_allowed_terminal_keys_are_exactly_decoder_branches_zero_to_five():
    assert allowed_transport_keys() == tuple(
        f"bbox_head.reg_branches.{branch}.4.{suffix}"
        for branch in range(6)
        for suffix in ("weight", "bias")
    )

def test_transport_replaces_only_xywh_and_preserves_angle_and_branch_six():
    control, candidate, summary = build_d12_pair_states(target, converted)
    assert summary["authorized_position_count"] == 6168
    for branch in range(6):
        assert torch.equal(candidate[f"...{branch}.4.weight"][:4],
                           converted[f"...{branch}.4.weight"])
        assert torch.equal(candidate[f"...{branch}.4.bias"][:4],
                           converted[f"...{branch}.4.bias"])
        assert torch.equal(candidate[f"...{branch}.4.weight"][4:],
                           control[f"...{branch}.4.weight"][4:])
        assert torch.equal(candidate[f"...{branch}.4.bias"][4:],
                           control[f"...{branch}.4.bias"][4:])
    assert torch.equal(candidate["bbox_head.reg_branches.6.4.weight"],
                       control["bbox_head.reg_branches.6.4.weight"])

def test_transport_fails_closed_on_missing_key_wrong_shape_or_nonfinite_source():
    ...

def test_publish_artifacts_refuses_existing_final_or_pending_paths():
    ...
```

Also assert control/candidate are complete clones, source tensors are never mutated, `actual_unequal_position_count` is counted independently, the manifest has the exact schema, `query_transport` is false, and repeated pure construction is deterministic.

- [ ] **Step 3: Run RED**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d12_checkpoint_pair.py -q
```

Expected: failure because `prepare_d12_checkpoint_pair.py` does not exist.

- [ ] **Step 4: Implement the minimal pure core**

Implement these public seams:

```python
TERMINAL_LAYER_INDEX = 4
DECODER_BRANCHES = tuple(range(6))
XYWH_ROWS = slice(0, 4)
AUTHORIZED_POSITION_COUNT = 6 * (4 * 256 + 4)

def allowed_transport_keys() -> tuple[str, ...]: ...

def build_d12_pair_states(
    target_state: Mapping[str, Tensor],
    converted_source: Mapping[str, Tensor],
) -> tuple[OrderedDict, OrderedDict, dict]: ...

def validate_pair_delta(
    control: Mapping[str, Tensor],
    candidate: Mapping[str, Tensor],
    converted_source: Mapping[str, Tensor],
) -> dict: ...

def publish_artifacts_no_clobber(
    payloads: Sequence[tuple[Path, object]],
) -> None: ...
```

`build_d12_pair_states` must clone every target tensor into two complete state dicts; validate 7 branches shaped `[5, 256]`/`[5]`; copy only source rows `0:4` for branches `0:6`; and prove all other values bitwise equal. Publication writes same-directory `.pending.<pid>` files, fsyncs, hard-links each final name only if all final and pending paths are absent, fsyncs directories, and preserves evidence on any failure.

- [ ] **Step 5: Run GREEN and focused regression**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d12_checkpoint_pair.py tests/test_projects/ov_capflow/test_cleanstart_checkpoint.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Add the production CLI**

The CLI must:

1. Require the exact raw source SHA-256 `3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799`.
2. Reuse `reject_forbidden_source`, `unwrap_model_state`, `reject_forbidden_namespaces`, `convert_groundingdino_state_dict`, and `sha256_file`.
3. Build the approved D12 control config model, call `model.init_weights()` exactly once, and capture the full CPU target state.
4. Assert that, under the resolved inherited `as_two_stage=True` model, all seven initialized terminal weights and all seven terminal biases are zero.
5. Produce model-only `{"state_dict": ...}` payloads at the approved control/candidate paths.
6. Produce a manifest with source/config/output hashes, exact 12 allowed keys, 6,168 authorized positions, independently counted unequal positions, and explicit false values for query/reference/DN/classification transport.
7. Refuse all collisions and never overwrite.

- [ ] **Step 7: Verify and commit the checkpoint builder**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d12_checkpoint_pair.py tests/test_projects/ov_capflow/test_cleanstart_checkpoint.py -q
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m py_compile projects/OVCapFlow/tools/prepare_d12_checkpoint_pair.py
rtk git diff --check
rtk git add projects/OVCapFlow/tools/prepare_d12_checkpoint_pair.py tests/test_projects/ov_capflow/test_d12_checkpoint_pair.py
rtk git commit -m "feat: build auditable D12 checkpoint pair"
```

Expected: tests and compilation pass; commit contains only builder and tests.

---

### Task 3: Add matched world-size-5 configs with TDD

**Files:**

- Create: `tests/test_projects/ov_capflow/test_d12_world5_configs.py`
- Create: `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_d12_control.py`
- Create: `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_d12_xywh.py`

- [ ] **Step 1: Write the failing config contract**

Assert both resolved configs have:

```python
assert cfg["selected_world_size"] == 5
assert cfg["train_dataloader"]["batch_size"] == 2
assert cfg["train_dataloader"]["batch_sampler"]["type"] == "DNQueryBudgetBatchSampler"
assert cfg["train_dataloader"]["batch_sampler"]["num_matching_queries"] == 1800
assert cfg["train_dataloader"]["batch_sampler"]["update_count_multiple"] == 1
assert cfg["optim_wrapper"]["accumulative_counts"] == 1
assert 2 * 5 * 1 == 10
assert divmod(1600, 10) == (160, 0)
assert cfg["randomness"]["seed"] == 20260716
assert cfg["resume"] is False
assert cfg["model"]["num_queries"] == 600
assert cfg["default_hooks"]["checkpoint"]["interval"] == 1
assert cfg["default_hooks"]["checkpoint"]["max_keep_ckpts"] == -1
assert cfg["default_hooks"]["checkpoint"].get("save_best") is None
```

Candidate GPUs must be `(0, 1, 2, 3, 4)`, control GPUs `(5, 6, 7, 8, 9)`. After removing physical GPU role, role metadata, `load_from`, `work_dir`, and sampler `audit_path`, resolved configs must be identical.

- [ ] **Step 2: Run RED**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d12_world5_configs.py -q
```

Expected: failure because the D12 configs do not exist.

- [ ] **Step 3: Implement the control config and thin candidate wrapper**

The control config inherits the existing rare4x protocol and overrides only:

```python
physical_gpus = (5, 6, 7, 8, 9)
selected_world_size = 5
d12_pair_id = "D12-Q600-terminal-xywh-world5-seed20260716"
d12_role = "control"
d12_only_scientific_delta = "none"
train_dataloader = dict(
    batch_size=2,
    batch_sampler=dict(
        update_count_multiple=1,
        audit_path="work_dirs/dotav2_cleanstart/audits/d12_world5_control_seed20260716_gpu56789_sampler.json"))
optim_wrapper = dict(accumulative_counts=1)
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True, interval=1, max_keep_ckpts=-1,
        save_best=None, save_last=True))
```

The candidate inherits control and changes only GPU tuple, role/scientific-delta metadata, audit path, `load_from`, and `work_dir`.

- [ ] **Step 4: Run GREEN plus sampler contract**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d12_world5_configs.py tests/test_projects/ov_capflow/test_d11_world3_configs.py -q
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_real_batch.py -q
```

Expected: config tests pass; opt-in real-data tests are skipped, not failed.

- [ ] **Step 5: Commit configs**

Run:

```bash
rtk git diff --check
rtk git add configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_d12_control.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_d12_xywh.py tests/test_projects/ov_capflow/test_d12_world5_configs.py
rtk git commit -m "config: freeze D12 world5 paired proxy"
```

Expected: commit contains only the two configs and their contract test.

---

### Task 4: Implement fail-closed Stage 0 preflight with TDD

**Files:**

- Create: `tests/test_projects/ov_capflow/test_d12_preflight.py`
- Create: `projects/OVCapFlow/tools/audit_d12_preflight.py`

- [ ] **Step 1: Write failing pure-gate tests**

Test these pure functions with small tensors/dicts:

```python
def compare_complete_states(control, candidate, converted_source) -> dict: ...
def summarize_normalized_cxywh(boxes: Tensor, epsilon=1e-6) -> dict: ...
def enforce_prediction_pair(candidate_summary, control_summary) -> None: ...
def enforce_loss_gradient_pair(candidate_stats, control_stats) -> None: ...
def publish_json_no_clobber(report: Mapping, output: Path) -> None: ...
```

Required rejection cases: an unequal non-allowed parameter; a mismatched source XYWH row; changed angle or branch 6; nonfinite/out-of-range boxes; saturation fraction above `0.05`; candidate saturation increase above `0.05`; median area ratio outside `[1/16, 16]`; p99 area above `1`; total-loss ratio above `20`; gradient-norm ratio above `100`; changed parameter after no-update checks; nondeterministic repeated output; existing final/pending output.

- [ ] **Step 2: Run RED**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d12_preflight.py -q
```

Expected: failure because the preflight module does not exist.

- [ ] **Step 3: Implement the pure gate and CLI orchestration**

The CLI takes candidate/control config, candidate/control checkpoint, raw source, fixed batch seed, device, strict-audit script, open-vocabulary-audit script, and output. It must:

1. Load two independently built model instances and compare their complete loaded states.
2. Re-run source-row provenance checks against the converted raw source.
3. Require identical query/reference/DN/classification tensors; require branch 6 all-zero and unchanged.
4. Build one deterministic real DOTA-v2 batch with nonempty GT and feed the same preprocessed tensors to both models.
5. Run prediction twice per role and require bitwise-identical raw normalized box tensors and exactly 600 finite predictions.
6. Compute saturation, median area, and p99 area before rescaling.
7. Run loss/backward without constructing or stepping an optimizer; require finite losses/gradients, total-loss and gradient-norm ratios within bounds, and model states bitwise unchanged.
8. Run `audit_strict_inference.py` and `audit_open_vocabulary.py` for both configs/checkpoints and incorporate their pass/fail JSON.
9. Publish a single immutable report only when every conjunct passes.

Do not call NMS, proposal top-k, dense heads, RoI heads, teachers, pseudo labels, or optimizer steps.

- [ ] **Step 4: Run GREEN and integration-safe regressions**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d12_preflight.py tests/test_projects/ov_capflow/test_strict_audit.py tests/test_projects/ov_capflow/test_open_vocabulary_audit.py tests/test_projects/ov_capflow/test_cleanstart_checkpoint.py -q
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m py_compile projects/OVCapFlow/tools/audit_d12_preflight.py
```

Expected: all focused tests and compilation pass.

- [ ] **Step 5: Commit Stage 0 tooling**

Run:

```bash
rtk git diff --check
rtk git add projects/OVCapFlow/tools/audit_d12_preflight.py tests/test_projects/ov_capflow/test_d12_preflight.py
rtk git commit -m "test: add fail-closed D12 preflight"
```

Expected: commit contains only Stage 0 audit code and tests.

---

### Task 5: Implement the epoch-12 postrun gate with TDD

**Files:**

- Create: `tests/test_projects/ov_capflow/test_d12_postrun_gate.py`
- Create: `projects/OVCapFlow/tools/d12_postrun_gate.py`
- Read/reuse behavior from: `.lab/workspace/exp-8-d149/d11_pair_postrun_queue.py`

- [ ] **Step 1: Write failing scheduler/gate tests**

Cover:

```python
def exact_training_process_exists(config: Path, proc_root=Path("/proc")) -> bool: ...
def extract_complete_epoch_metric(text: str, target_epoch=12) -> dict | None: ...
def pair_completion_status(...) -> dict: ...
def evaluate_frozen_gate(candidate: Mapping, control: Mapping,
                         integrity: Mapping) -> dict: ...
def publish_json_no_clobber(...) -> None: ...
```

`evaluate_frozen_gate` must be a conjunction:

```python
candidate["AP50"] >= 0.489
candidate["AP50"] - control["AP50"] >= 0.020
candidate["mAP"] - control["mAP"] >= 0.020
candidate["novel4"] >= 0.3675
candidate["novel4"] >= control["novel4"]
candidate["base14"] >= 0.464643
candidate["base14"] >= control["base14"] - 0.005
all(integrity.values())
```

Test equality at every threshold, one failure below each threshold, NaN/Inf rejection, epoch 11/best-checkpoint rejection, live exact-process waiting, missing/truncated checkpoint waiting, incomplete sampler audit waiting, non-idle GPU waiting, and output collision rejection.

- [ ] **Step 2: Run RED**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d12_postrun_gate.py -q
```

Expected: failure because the postrun module does not exist.

- [ ] **Step 3: Implement the fail-closed postrun CLI**

Adapt the proven D11 scheduler without importing lab-only modules. Require:

- exact candidate/control config argv matching in `/proc`;
- completed epoch-12 validation rows, epoch-12 checkpoint files, and `last_checkpoint` pointers;
- sampler audit evidence of 1,600 unique examples, no duplicates/omissions, 160 updates, five ranks;
- no fatal traceback/OOM/NCCL failure in either console;
- physical GPUs 0–9 compute-idle before audits;
- frozen epoch-12 candidate/control classwise metrics;
- strict-inference and open-vocabulary audit success for both roles.

The output contains all raw evidence, each Boolean conjunct, `decision` equal to `promote` or `discard`, and the explicit prohibition on automatic full24, rescue, retuning, extra epochs/seeds, checkpoint cherry-picking, and D11 stacking.

- [ ] **Step 4: Run GREEN and compile**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d12_postrun_gate.py -q
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m py_compile projects/OVCapFlow/tools/d12_postrun_gate.py
```

Expected: all tests and compilation pass.

- [ ] **Step 5: Commit postrun gate**

Run:

```bash
rtk git diff --check
rtk git add projects/OVCapFlow/tools/d12_postrun_gate.py tests/test_projects/ov_capflow/test_d12_postrun_gate.py
rtk git commit -m "feat: add frozen D12 postrun gate"
```

Expected: commit contains only postrun tooling and tests.

---

### Task 6: Verify the implementation and freeze the experiment commit

**Files:**

- Modify: `.lab/log.md`

- [ ] **Step 1: Run the complete focused test set**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d12_checkpoint_pair.py tests/test_projects/ov_capflow/test_d12_world5_configs.py tests/test_projects/ov_capflow/test_d12_preflight.py tests/test_projects/ov_capflow/test_d12_postrun_gate.py tests/test_projects/ov_capflow/test_cleanstart_checkpoint.py tests/test_projects/ov_capflow/test_d11_world3_configs.py tests/test_projects/ov_capflow/test_dotav2_real_batch.py tests/test_projects/ov_capflow/test_strict_audit.py tests/test_projects/ov_capflow/test_open_vocabulary_audit.py -q
```

Expected: all non-opt-in tests pass; real DOTA-v2 integration tests are skipped.

- [ ] **Step 2: Audit the diff and repository state**

Run:

```bash
rtk git diff --check
rtk git status --short
rtk git log --oneline -6
```

Expected: no unintended tracked edits; pre-existing untracked user assets remain untouched.

- [ ] **Step 3: Record the frozen code identity**

Append a `.lab/log.md` THINK entry with branch, HEAD, approved scientific delta, exact configs, exact checkpoint paths, frozen thresholds, and the statement that there will be no code/config edits after Stage 0 begins. If that entry is tracked in this repository, commit only that entry:

```bash
rtk git status --short .lab/log.md
```

Expected: the experiment code identity is a clean committed SHA.

---

### Task 7: Materialize the immutable checkpoint pair

**Files produced without overwrite:**

- `work_dirs/dotav2_cleanstart/checkpoints/groundingdino_swint_ogc_q600_d12_w5_control_seed20260716.pth`
- `work_dirs/dotav2_cleanstart/checkpoints/groundingdino_swint_ogc_q600_d12_w5_xywh_seed20260716.pth`
- `work_dirs/dotav2_cleanstart/checkpoints/groundingdino_swint_ogc_q600_d12_w5_pair_manifest.json`

- [ ] **Step 1: Prove output absence and source identity**

Run:

```bash
rtk test ! -e work_dirs/dotav2_cleanstart/checkpoints/groundingdino_swint_ogc_q600_d12_w5_control_seed20260716.pth
rtk test ! -e work_dirs/dotav2_cleanstart/checkpoints/groundingdino_swint_ogc_q600_d12_w5_xywh_seed20260716.pth
rtk test ! -e work_dirs/dotav2_cleanstart/checkpoints/groundingdino_swint_ogc_q600_d12_w5_pair_manifest.json
rtk sha256sum /data/zcy/GroundingDINO/weights/groundingdino_swint_ogc.pth
```

Expected: outputs absent; source digest is `3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799`.

- [ ] **Step 2: Build once**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/prepare_d12_checkpoint_pair.py --source /data/zcy/GroundingDINO/weights/groundingdino_swint_ogc.pth --config configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_d12_control.py --control-output work_dirs/dotav2_cleanstart/checkpoints/groundingdino_swint_ogc_q600_d12_w5_control_seed20260716.pth --candidate-output work_dirs/dotav2_cleanstart/checkpoints/groundingdino_swint_ogc_q600_d12_w5_xywh_seed20260716.pth --manifest-output work_dirs/dotav2_cleanstart/checkpoints/groundingdino_swint_ogc_q600_d12_w5_pair_manifest.json
```

Expected: exactly three new artifacts, no pending file, 12 allowed keys, 6,168 authorized positions, query transport false, and printed SHA-256 values.

- [ ] **Step 3: Independently inspect artifacts**

Run:

```bash
rtk sha256sum work_dirs/dotav2_cleanstart/checkpoints/groundingdino_swint_ogc_q600_d12_w5_control_seed20260716.pth work_dirs/dotav2_cleanstart/checkpoints/groundingdino_swint_ogc_q600_d12_w5_xywh_seed20260716.pth work_dirs/dotav2_cleanstart/checkpoints/groundingdino_swint_ogc_q600_d12_w5_pair_manifest.json
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m json.tool work_dirs/dotav2_cleanstart/checkpoints/groundingdino_swint_ogc_q600_d12_w5_pair_manifest.json
```

Expected: all manifest hashes match the files and every boundary assertion passes.

---

### Task 8: Run Stage 0 strict preflight

**Files produced without overwrite:**

- `work_dirs/dotav2_cleanstart/audits/d12_stage0_preflight_seed20260716.json`
- Per-role strict/open-vocabulary audit JSON referenced by the Stage 0 report.

- [ ] **Step 1: Verify GPU idleness and no conflicting training**

Run:

```bash
rtk nvidia-smi --query-gpu=index,pci.bus_id,memory.used,utilization.gpu,pstate --format=csv,noheader
rtk pgrep -af 'tools/train.py|torchrun|torch.distributed.run'
rtk tmux list-sessions
```

Expected: GPUs 0–9 have no compute processes; no live training argv uses either D12 config. Dead historical panes are evidence, not deletion targets.

- [ ] **Step 2: Run exact-cover sampler integration**

Run:

```bash
rtk env RUN_OVCAPFLOW_DOTAV2_PREFLIGHT=1 OVCAPFLOW_DOTA2_CONFIG=configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_d12_control.py PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_real_batch.py -k sampler -q
```

Expected: five rank views cover all 1,600 items exactly once in 160 ten-sample optimizer updates, with no duplicate/omitted sample and query area at most 50,000,000.

- [ ] **Step 3: Run paired real-batch preflight**

Run:

```bash
rtk env CUDA_VISIBLE_DEVICES=0 RUN_OVCAPFLOW_DOTAV2_INTEGRATION=1 OVCAPFLOW_PREFLIGHT_BATCH=2 PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/audit_d12_preflight.py --candidate-config configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_d12_xywh.py --control-config configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_d12_control.py --candidate-checkpoint work_dirs/dotav2_cleanstart/checkpoints/groundingdino_swint_ogc_q600_d12_w5_xywh_seed20260716.pth --control-checkpoint work_dirs/dotav2_cleanstart/checkpoints/groundingdino_swint_ogc_q600_d12_w5_control_seed20260716.pth --source /data/zcy/GroundingDINO/weights/groundingdino_swint_ogc.pth --device cuda:0 --seed 20260716 --output work_dirs/dotav2_cleanstart/audits/d12_stage0_preflight_seed20260716.json
```

Expected: report `pass=true`; Q600 exact; all finite; saturation/area, loss ratio, and gradient ratio within approved limits; no parameter changed; strict and open-vocabulary audits pass for both roles.

- [ ] **Step 4: Freeze Stage 0 evidence**

Append artifact hashes and all scalar gate values to `.lab/log.md`. Do not modify code/config/checkpoints after this point. Any failure is a D12 discard unless it is an execution-only issue proven unrelated to the scientific delta; repair of scientific behavior requires a new reviewed design.

---

### Task 9: Launch the matched 5+5 GPU proxy

**Immutable runtime assignments:**

- Candidate: physical GPUs `0,1,2,3,4`
- Control: physical GPUs `5,6,7,8,9`
- Both: world size 5, batch size 2/rank, accumulation 1, global batch 10, seed 20260716, 12 epochs, epoch-12-only selection.

- [ ] **Step 1: Recheck all launch conditions immediately before mutation**

Run:

```bash
rtk nvidia-smi --query-gpu=index,memory.used,utilization.gpu,pstate --format=csv,noheader
rtk pgrep -af 'tools/train.py|torchrun|torch.distributed.run'
rtk test -s work_dirs/dotav2_cleanstart/audits/d12_stage0_preflight_seed20260716.json
rtk git status --short
rtk git rev-parse HEAD
```

Expected: GPUs idle, no conflicting live processes, Stage 0 pass exists, committed code identity unchanged, user assets untouched.

- [ ] **Step 2: Start candidate and control within 60 seconds**

Run:

```bash
rtk tmux new-session -d -s d12_xywh_w5_20260731 "cd /data1/zcy/OV-CapFlow && env CUDA_VISIBLE_DEVICES=0,1,2,3,4 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m torch.distributed.run --nproc_per_node=5 --master_port=29612 tools/train.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_d12_xywh.py --launcher pytorch 2>&1 | tee work_dirs/dotav2_cleanstart/d12_world5_candidate_seed20260716_gpu01234_batch2/console.log"
rtk tmux new-session -d -s d12_control_w5_20260731 "cd /data1/zcy/OV-CapFlow && env CUDA_VISIBLE_DEVICES=5,6,7,8,9 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m torch.distributed.run --nproc_per_node=5 --master_port=29613 tools/train.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_d12_control.py --launcher pytorch 2>&1 | tee work_dirs/dotav2_cleanstart/d12_world5_control_seed20260716_gpu56789_batch2/console.log"
```

Expected: both tmux sessions exist and their recorded start timestamps differ by at most 60 seconds.

- [ ] **Step 3: Verify ten-rank health**

Run:

```bash
rtk tmux list-sessions
rtk pgrep -af 'world5_d12_(xywh|control).*--launcher pytorch'
rtk nvidia-smi --query-gpu=index,memory.used,utilization.gpu,pstate --format=csv,noheader
rtk rg -n 'Traceback|CUDA out of memory|NCCL|nan|inf' work_dirs/dotav2_cleanstart/d12_world5_candidate_seed20260716_gpu01234_batch2/console.log work_dirs/dotav2_cleanstart/d12_world5_control_seed20260716_gpu56789_batch2/console.log
```

Expected: ten worker ranks; GPUs 0–9 active; no fatal, OOM, NCCL error, NaN, or Inf.

- [ ] **Step 4: Start the fail-closed postrun owner**

Run:

```bash
rtk tmux new-session -d -s d12_postrun_20260731 "cd /data1/zcy/OV-CapFlow && env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/d12_postrun_gate.py --candidate-config configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_d12_xywh.py --control-config configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_d12_control.py --candidate-work-dir work_dirs/dotav2_cleanstart/d12_world5_candidate_seed20260716_gpu01234_batch2 --control-work-dir work_dirs/dotav2_cleanstart/d12_world5_control_seed20260716_gpu56789_batch2 --candidate-sampler-audit work_dirs/dotav2_cleanstart/audits/d12_world5_candidate_seed20260716_gpu01234_sampler.json --control-sampler-audit work_dirs/dotav2_cleanstart/audits/d12_world5_control_seed20260716_gpu56789_sampler.json --target-epoch 12 --poll-seconds 30 --output work_dirs/dotav2_cleanstart/audits/d12_epoch12_frozen_gate_seed20260716.json 2>&1 | tee work_dirs/dotav2_cleanstart/audits/d12_postrun_owner_seed20260716.log"
```

Expected: owner waits without modifying training; it cannot publish a decision before both exact jobs exit and every completion/integrity conjunct exists.

---

### Task 10: Monitor for at least 18 hours and evaluate only epoch 12

**Files:**

- Modify append-only: `.lab/log.md`
- Produce: `work_dirs/dotav2_cleanstart/audits/d12_epoch12_frozen_gate_seed20260716.json`

- [ ] **Step 1: Monitor without changing scientific state**

At each check, record timestamp, live exact processes, both latest epoch/iteration/loss lines, GPU 0–9 memory/utilization, sampler audit state, checkpoint list, and any fatal signature. Use:

```bash
rtk date --iso-8601=seconds
rtk pgrep -af 'world5_d12_(xywh|control).*--launcher pytorch'
rtk tail -n 80 work_dirs/dotav2_cleanstart/d12_world5_candidate_seed20260716_gpu01234_batch2/console.log
rtk tail -n 80 work_dirs/dotav2_cleanstart/d12_world5_control_seed20260716_gpu56789_batch2/console.log
rtk nvidia-smi --query-gpu=index,memory.used,utilization.gpu,pstate --format=csv,noheader
rtk rg -n 'Traceback|CUDA out of memory|NCCL|nan|inf' work_dirs/dotav2_cleanstart/d12_world5_candidate_seed20260716_gpu01234_batch2/console.log work_dirs/dotav2_cleanstart/d12_world5_control_seed20260716_gpu56789_batch2/console.log
```

Expected: paired progress remains comparable; no intervention, resume, retune, extra seed, or checkpoint selection. Continue monitoring for at least 18 hours or until the complete paired postrun result is published, whichever is later.

- [ ] **Step 2: Verify the immutable postrun result**

Run:

```bash
rtk test -s work_dirs/dotav2_cleanstart/audits/d12_epoch12_frozen_gate_seed20260716.json
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m json.tool work_dirs/dotav2_cleanstart/audits/d12_epoch12_frozen_gate_seed20260716.json
rtk sha256sum work_dirs/dotav2_cleanstart/audits/d12_epoch12_frozen_gate_seed20260716.json
```

Expected: report cites only epoch 12, includes both classwise metrics and all integrity evidence, and returns exactly one frozen decision.

- [ ] **Step 3: Apply the approved terminal decision**

If any conjunct fails: label D12 `discard`; preserve every artifact; do not scale, retune, rescue, add epochs/seeds, cherry-pick, stack D11, or change prompts/losses.

If all conjuncts pass: label D12 `promote-to-separate-full-run-review`; do not launch full24 automatically. Prepare a new written design and obtain separate approval.

---

### Task 11: Write and verify the Chinese experiment record

**Files:**

- Create: `docs/project_history/exp_20260731_d12_terminal_xywh_transport.md`
- Modify append-only: `.lab/log.md`

- [ ] **Step 1: Write the evidence-backed result record**

Include:

- problem/hypothesis and narrow novelty boundary;
- branch/commit/config/checkpoint/source hashes;
- exact 12 transported keys, 6,168 authorized positions, and actual unequal count;
- Stage 0 box/loss/gradient/determinism/integrity evidence;
- launch timestamps, physical GPU mapping, world size, global batch, seed;
- candidate/control epoch-12 `mAP`, `AP50`, `novel4`, and `base14`;
- every frozen Boolean conjunct and final decision;
- explicit statement that no forbidden inference/training mechanism or rescue was used;
- artifact paths and SHA-256 values.

- [ ] **Step 2: Verify claims against source artifacts**

Run:

```bash
rtk rg -n 'mAP|AP50|novel4|base14|decision|sha256|6168|epoch 12|Q600|NMS|top-k|teacher|pseudo' docs/project_history/exp_20260731_d12_terminal_xywh_transport.md
rtk git diff --check
rtk git status --short
```

Expected: every reported number/path/hash is traceable to an immutable artifact; no unsupported “first” claim; no unrelated file change.

- [ ] **Step 3: Commit only the final record**

Run:

```bash
rtk git add docs/project_history/exp_20260731_d12_terminal_xywh_transport.md
rtk git diff --cached --check
rtk git commit -m "docs: record D12 terminal XYWH proxy"
```

Expected: final documentation commit contains only the D12 record.

---

## Final verification checklist

- [ ] Approved written spec and implementation plan are committed.
- [ ] Pair builder tests prove the exact 12-key/6,168-position boundary.
- [ ] Candidate/control are complete target-state checkpoints from one initialized model.
- [ ] Angle row, branch 6, queries, references, DN state, and classification biases are identical.
- [ ] World-size-5 configs are identical outside approved operational/role/checkpoint fields.
- [ ] Typed exact-cover sampler yields 1,600 unique samples in 160 updates.
- [ ] Stage 0 passes Q600, finite-box, saturation/area, real-GT loss/gradient, determinism, strict-inference, and open-vocabulary gates.
- [ ] Candidate/control launch within 60 seconds with `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1`.
- [ ] All twelve epoch checkpoints are preserved for both roles.
- [ ] Postrun decision uses epoch 12 only and all frozen thresholds conjunctively.
- [ ] At least 18 hours of monitoring evidence is recorded.
- [ ] No delete/reset/clean/overwrite, rescue, retune, extra seed/epoch, cherry-pick, D11 stack, or automatic full24 launch occurred.
- [ ] Chinese result record is committed with traceable hashes and no invalid broad novelty claim.
