# D12 Five-GPU Sequential Pair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Preserve a matched D12 control/candidate comparison while occupying
only physical GPUs0-4 after all ten GPUs become idle.

**Architecture:** Add two thin world5 configs whose scientific checkpoint
delta is unchanged but whose runtime assignment and output roots are both
GPU0-4. A fail-closed sequential owner waits for global idleness, runs the
control through Epoch 12, then runs the candidate through Epoch 12 on the same
five GPUs, and finally invokes the existing frozen gate with all four strict
audits serialized on GPU0. The result is explicitly labelled sequential and
temporally confounded; it cannot claim the original concurrent-start control.

**Tech Stack:** MMEngine configs, PyTorch distributed world5, tmux, Python
JSONL owners, pytest, existing D12 checkpoint/preflight/gate utilities.

---

### Task 1: Parameterize postrun audit GPU assignment

**Files:**
- Modify: `projects/OVCapFlow/tools/d12_postrun_gate.py`
- Modify: `tests/test_projects/ov_capflow/test_d12_postrun_gate.py`

- [ ] Add failing tests proving `build_audit_specs` accepts explicit candidate
  and control physical GPUs, defaults remain `0/5`, and `0/0` places all four
  sequential audits on physical GPU0.
- [ ] Add parser options `--candidate-audit-gpu` and `--control-audit-gpu`,
  each restricted to integers `0..9`, and thread them into
  `build_audit_specs`.
- [ ] Run:
  `rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d12_postrun_gate.py -q`
  and require zero failures.
- [ ] Commit only these two files with message
  `feat: allow D12 postrun audits on reserved GPUs`.

### Task 2: Add immutable sequential world5 config pair

**Files:**
- Create:
  `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_d12_seq5_control.py`
- Create:
  `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_d12_seq5_xywh.py`
- Modify: `tests/test_projects/ov_capflow/test_d12_world5_configs.py`

- [ ] Add failing config tests requiring both roles to use
  `physical_gpus=(0,1,2,3,4)`, world size 5, batch2, accumulation1, seed
  20260716, Epoch12, distinct append-only work/sampler paths, identical
  normalized configs outside role/checkpoint/output bookkeeping, and the same
  twelve-key D12 checkpoint delta.
- [ ] Implement thin wrappers over the approved D12 configs. Use work roots
  `d12_seq5_control_seed20260716_gpu01234_batch2` and
  `d12_seq5_candidate_seed20260716_gpu01234_batch2`, and corresponding new
  sampler-audit paths.
- [ ] Run the focused config tests and resolved-config construction.
- [ ] Commit the two configs and tests with message
  `config: add D12 sequential five-GPU pair`.

### Task 3: Build and test the fail-closed sequential owner

**Files:**
- Create: `.lab/workspace/exp-8-d153/d12_seq5_owner.py`
- Create: `.lab/workspace/exp-8-d153/test_d12_seq5_owner.py`

- [ ] Test first: exact HEAD/Stage-0 hashes, ten-GPU consecutive-idle gate,
  final PID recheck, no-clobber paths, control-first state machine, exact
  torchrun parent plus environment ranks `0..4`, Epoch12 metric/checkpoint and
  sampler completion before candidate release, GPU0-4 idleness before each
  launch, and postrun arguments `--candidate-audit-gpu 0
  --control-audit-gpu 0`.
- [ ] Implement one-shot append-only JSONL state transitions:
  `waiting_all_idle -> control_running -> control_complete ->
  candidate_running -> candidate_complete -> postrun_running`.
- [ ] Never signal external processes, delete artifacts, overwrite paths,
  resume, retry, change batch/world size, add epochs, or launch GPU5-9.
- [ ] Run the owner tests and `py_compile`; do not launch the owner in this
  task.

### Task 4: Re-run sequential Stage 0 and launch the owner

**Files produced without overwrite:**
- `work_dirs/dotav2_cleanstart/audits/d12_seq5_stage0_preflight_seed20260716.json`
- `work_dirs/dotav2_cleanstart/audits/d12_seq5_owner_20260731.jsonl`

- [ ] Run the complete focused D12 regression suite.
- [ ] Run `audit_d12_preflight.py` on GPU0 with the sequential configs and
  original immutable checkpoint pair; require all checkpoint, numerical,
  strict-Q600, open-vocabulary and sampler gates to pass.
- [ ] Record the new code/config/Stage-0 hashes in `.lab/log.md` and
  `.lab/results.tsv`.
- [ ] Launch `d12_seq5_owner_20260731` in tmux. Verify it observes all ten
  GPUs but never allocates GPU5-9.

### Task 5: Review and monitoring

- [ ] Obtain spec-compliance review, then code-quality review, fixing and
  re-reviewing every Important/Critical issue before launch.
- [ ] Monitor owner JSONL, exact processes, GPU allocation, console growth,
  fatal signatures, Epoch12 checkpoints and final frozen gate.
- [ ] Report the result as a sequential matched pair with temporal-order
  confounding; do not present it as the original concurrent D12 protocol.
