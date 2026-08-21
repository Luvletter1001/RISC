# E24 GPU2389 Dump Autoschedule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Queue the canonical raw-13,833 Epoch 24 evaluation and all-Q600 prediction dump on physical GPUs 2,3,8,9 without interrupting or contending with the live T7 training run.

**Architecture:** A small testable scheduler polls four independent readiness gates: the Epoch 24 checkpoint is fully published, the scheduled Epoch 24 validation metric is present, no matching training process remains, and GPUs 2,3,8,9 have no compute processes. It then launches the existing non-resume test config with four ranks and NCCL P2P/IB disabled, waits for completion, and invokes a separate CPU-remap validator for the dump.

**Tech Stack:** Python 3.8, PyTorch 1.12, MMEngine 0.10, pytest, `nvidia-smi`, PyTorch distributed launcher, tmux.

---

### Task 1: Readiness and launch specification

**Files:**
- Create: `projects/OVCapFlow/tools/queue_test_after_training.py`
- Test: `tests/test_projects/ov_capflow/test_queue_test_after_training.py`

- [x] **Step 1: Write failing tests for the validation marker and launch specification**

```python
def test_validation_complete_requires_target_epoch_metric():
    text = 'Epoch(val) [24][1730/1730] dota/mAP: 0.6100 dota/AP50: 0.6100'
    assert validation_complete(text, target_epoch=24)
    assert not validation_complete(text, target_epoch=18)


def test_build_test_spec_keeps_fixed_gpu_and_mouth_contract():
    command, environment = build_test_spec(
        python=Path('/env/bin/python'),
        config=Path('/repo/base.py'),
        checkpoint=Path('/repo/epoch_24.pth'),
        work_dir=Path('/repo/eval'),
        prediction_out=Path('/repo/eval/predictions.pkl'),
        gpu_indices=(2, 3, 8, 9),
        port=29791,
    )
    assert '--nproc_per_node=4' in command
    assert 'tools/test.py' in command
    assert '--out' in command
    assert environment['CUDA_VISIBLE_DEVICES'] == '2,3,8,9'
    assert environment['NCCL_P2P_DISABLE'] == '1'
    assert environment['NCCL_IB_DISABLE'] == '1'
```

- [x] **Step 2: Run the tests and verify RED**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_queue_test_after_training.py -q
```

Expected: FAIL because `queue_test_after_training.py` does not exist.

- [x] **Step 3: Implement the pure readiness and command helpers**

Implement:

```python
def validation_complete(text: str, target_epoch: int) -> bool:
    pattern = rf'Epoch\(val\) \[{target_epoch}\]\[\d+/\d+\].*dota/mAP:.*dota/AP50:'
    return re.search(pattern, text) is not None


def build_test_spec(python, config, checkpoint, work_dir, prediction_out,
                    gpu_indices, port):
    command = [
        str(python), '-m', 'torch.distributed.launch',
        f'--nproc_per_node={len(gpu_indices)}', f'--master_port={port}',
        'tools/test.py', str(config), str(checkpoint),
        '--launcher', 'pytorch', '--work-dir', str(work_dir),
        '--out', str(prediction_out),
    ]
    environment = {
        'CUDA_VISIBLE_DEVICES': ','.join(map(str, gpu_indices)),
        'NCCL_P2P_DISABLE': '1', 'NCCL_IB_DISABLE': '1',
        'OMP_NUM_THREADS': '1', 'PYTHONNOUSERSITE': '1',
    }
    return command, environment
```

- [x] **Step 4: Run the tests and verify GREEN**

Run the command from Step 2. Expected: PASS.

### Task 2: Full scheduler gate

**Files:**
- Modify: `projects/OVCapFlow/tools/queue_test_after_training.py`
- Modify: `tests/test_projects/ov_capflow/test_queue_test_after_training.py`

- [x] **Step 1: Add failing tests for all readiness inputs and duplicate-launch protection**

The tests must prove that launch is denied when any of checkpoint publication, validation completion, training exit, or GPU idleness is false, and that a prior `evaluation_launched` state prevents a second launch.

- [x] **Step 2: Verify RED with the Task 1 pytest command**

Expected: FAIL because the combined gate is absent.

- [x] **Step 3: Implement `ready_to_launch`, state-log parsing, `/proc` training matching, per-GPU compute-process checks, and atomic JSONL event writes**

The loop must only return ready for:

```python
checkpoint_published and validation_finished and not training_alive and gpus_idle and not already_launched
```

It must poll every 30 seconds until `2026-07-23T23:59:00+08:00`, write a stable reason on every state transition, and never kill or pause the training process.

- [x] **Step 4: Verify GREEN**

Run the Task 1 pytest command. Expected: PASS.

### Task 3: Portable dump validation

**Files:**
- Create: `projects/OVCapFlow/tools/validate_dotav2_q600_dump.py`
- Test: `tests/test_projects/ov_capflow/test_validate_dotav2_q600_dump.py`

- [x] **Step 1: Write failing tests using a small pickle with CUDA-tag-independent CPU tensors**

Cover record count, unique image ids, `(Q,5)` rotated boxes, score/label shapes, label range, finite values, and total row count. The public helper must accept configurable expected counts so tests do not allocate 8.3 million rows.

- [x] **Step 2: Verify RED**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_validate_dotav2_q600_dump.py -q
```

Expected: FAIL because the validator does not exist.

- [x] **Step 3: Implement `CpuUnpickler` and structural validation**

`CpuUnpickler.find_class` must only intercept `torch.storage._load_from_bytes` and call `torch.load(io.BytesIO(payload), map_location='cpu')`. Production defaults must be 13,833 records, 600 rows/image, 18 labels, and 8,299,800 total rows.

- [x] **Step 4: Verify GREEN**

Run the Task 3 pytest command. Expected: PASS.

### Task 4: Integrate and dry-run

**Files:**
- Modify: `projects/OVCapFlow/tools/queue_test_after_training.py`
- Modify: `tests/test_projects/ov_capflow/test_queue_test_after_training.py`

- [x] **Step 1: Add a failing test proving the validator command has `CUDA_VISIBLE_DEVICES` empty and runs only after evaluation exit code 0**
- [x] **Step 2: Verify RED**
- [x] **Step 3: Implement post-evaluation validation and terminal state events**
- [x] **Step 4: Run both focused test files and the existing GPU guard tests**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_queue_test_after_training.py tests/test_projects/ov_capflow/test_validate_dotav2_q600_dump.py tests/test_projects/ov_capflow/test_gpu_resume_guard.py -q
```

Expected: all PASS.

- [x] **Step 5: Run scheduler `--check-once` against the live job**

Expected state: not ready because T7 is alive and Epoch 24 validation is absent. No subprocess may launch.

### Task 5: Start the detached GPU2389 queue

**Files:**
- Runtime state: `.lab/workspace/8-t7-e24-dump-gpu2389-queue.jsonl`
- Runtime log: `.lab/workspace/8-t7-e24-dump-gpu2389-queue.log`
- Evaluation output: `work_dirs/dotav2_cleanstart/eval_t7_epoch24_raw13833_gpu2389_dump/`

- [x] **Step 1: Confirm the output directory and tmux session do not already exist**
- [x] **Step 2: Start `t7_e24_dump_queue_gpu2389` in tmux with the mmdet interpreter**

The scheduler arguments must name physical GPUs `2 3 8 9`, target Epoch 24, port 29791, the non-resume base config, `epoch_24.pth`, the live training log/config, and the frozen deadline.

- [x] **Step 3: Verify the queue state is waiting and T7 remains alive**

Expected: state log says checkpoint/validation/training/GPU gates are not all ready; live T7 PID 688939 and monitor remain healthy; no evaluation rank exists.

- [x] **Step 4: Record the queue session, state log, command, and expected post-E24 behavior in `.lab/log.md`**

No E18 replay is queued by default. The E24 authority and dump are first; E18 canonical replay is only scheduled afterward if the E24 decomposition needs an exact trajectory comparison.
