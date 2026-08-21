# DOTA-v2 T4 Text-LR Preservation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch the pre-registered matched scale-1024 T4 experiment on physical GPUs 8 and 9 with `language_model` peak LR reduced from `1e-4` to `1e-5` as the only scientific delta.

**Architecture:** Inherit the completed batch-1 scale-1024 control config, deep-merge one optimizer `paramwise_cfg.custom_keys.language_model.lr_mult=0.1` entry, and isolate operational outputs in a T4 work directory and sampler-audit file. A structural config test removes only operational metadata and the one intended optimizer key, then proves the resolved candidate equals the control. Training uses the unchanged world-2, batch-1/GPU, accumulation-4, effective-batch-8, seed-20260716, 12-epoch protocol.

**Tech Stack:** MMEngine configs, PyTorch/MMRotate, pytest, existing `DNQueryBudgetBatchSampler`, existing GPU monitor and auto-resume guard.

---

### Task 1: Freeze the T4 single-variable contract

**Files:**
- Modify: `tests/test_projects/ov_capflow/test_dotav2_cleanstart_config.py`

- [ ] **Step 1: Add the candidate constant and a failing structural-diff test**

Add:

```python
S1_GROUPED_SCALE1024_BATCH1_TEXTLR1E5 = (
    'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch1_textlr1e5.py')


def test_t4_textlr1e5_changes_only_language_model_lr_and_paths():
    control = _load(S1_GROUPED_SCALE1024_BATCH1).to_dict()
    candidate = _load(S1_GROUPED_SCALE1024_BATCH1_TEXTLR1E5).to_dict()

    assert candidate['t4_parent'] == '8-T2-R-B1-E12'
    assert candidate['optim_wrapper']['optimizer']['lr'] == 1e-4
    assert candidate['optim_wrapper']['paramwise_cfg']['custom_keys'][
        'language_model'] == {'lr_mult': 0.1}
    assert candidate['work_dir'].endswith(
        's1_grouped_scale1024_seed20260716_gpu89_batch1_textlr1e5')

    candidate.pop('t4_parent')
    candidate.pop('t4_only_scientific_delta')
    candidate['optim_wrapper']['paramwise_cfg']['custom_keys'].pop(
        'language_model')
    candidate['train_dataloader']['batch_sampler']['audit_path'] = control[
        'train_dataloader']['batch_sampler']['audit_path']
    candidate['work_dir'] = control['work_dir']
    assert candidate == control
```

- [ ] **Step 2: Run the test and confirm it fails because the config does not exist**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_cleanstart_config.py::test_t4_textlr1e5_changes_only_language_model_lr_and_paths -q
```

Expected: FAIL while loading the missing T4 config.

### Task 2: Add the minimal T4 config

**Files:**
- Create: `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch1_textlr1e5.py`

- [ ] **Step 1: Create the inherited one-variable candidate**

```python
_base_ = [
    './ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch1.py'
]

t4_parent = '8-T2-R-B1-E12'
t4_only_scientific_delta = 'language_model_lr_mult=0.1'

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys=dict(language_model=dict(lr_mult=0.1))))
train_dataloader = dict(
    batch_sampler=dict(
        audit_path=(
            'work_dirs/dotav2_cleanstart/audits/'
            's1_grouped_scale1024_seed20260716_gpu89_batch1_'
            'textlr1e5_sampler.json')))
work_dir = (
    'work_dirs/dotav2_cleanstart/'
    's1_grouped_scale1024_seed20260716_gpu89_batch1_textlr1e5')
resume = False
```

- [ ] **Step 2: Run the focused structural test**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_cleanstart_config.py::test_t4_textlr1e5_changes_only_language_model_lr_and_paths -q
```

Expected: `1 passed`.

- [ ] **Step 3: Run the complete clean-start config test file**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_cleanstart_config.py -q
```

Expected: all tests pass.

### Task 3: Audit sampler coverage and resolved optimizer policy

**Files:**
- Create at runtime: `work_dirs/dotav2_cleanstart/audits/t4_textlr1e5_epoch0_world2.json`

- [ ] **Step 1: Replay the world-2 sampler**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/audit_sampler_coverage.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch1_textlr1e5.py --world-size 2 --epoch 0 --output work_dirs/dotav2_cleanstart/audits/t4_textlr1e5_epoch0_world2.json
```

Expected: rank update counts `[800, 800]`, `duplicate_count=0`, and `missing_count=0`.

- [ ] **Step 2: Inspect the resolved config**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python -c "from mmengine import Config; p='configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch1_textlr1e5.py'; c=Config.fromfile(p); print(c.optim_wrapper.paramwise_cfg.custom_keys); print(c.train_dataloader.batch_size, c.optim_wrapper.accumulative_counts, c.randomness.seed, c.train_cfg.max_epochs)"
```

Expected: existing `absolute_pos_embed` and `backbone` entries remain, `language_model.lr_mult=0.1` is present, and protocol values are `1 4 20260716 12`.

### Task 4: Launch and guard T4 on GPUs 8 and 9

**Files:**
- Create at runtime: `work_dirs/dotav2_cleanstart/s1_grouped_scale1024_seed20260716_gpu89_batch1_textlr1e5_launcher.log`
- Create at runtime: `.lab/workspace/8-t4-textlr1e5-gpu89-monitor.jsonl`

- [ ] **Step 1: Verify GPUs 8 and 9 are free and no matching launcher exists**

```bash
rtk nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader -i 8,9
rtk pgrep -af 'textlr1e5.py|s1_grouped_scale1024_seed20260716_gpu89_batch1_textlr1e5'
```

Expected: 16 MiB per GPU and no matching training process.

- [ ] **Step 2: Launch the unchanged two-rank recipe with the T4 config**

```bash
rtk env CUDA_VISIBLE_DEVICES=8,9 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTHONNOUSERSITE=1 rtk setsid -f /data/zcy/anaconda3/envs/mmdet/bin/python -m torch.distributed.launch --nproc_per_node=2 --master_port=41977 tools/train.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch1_textlr1e5.py --launcher pytorch
```

Expected: one launcher, two ranks, generic compatible checkpoint load, epoch 1 progress, and no resume from the control checkpoint.

- [ ] **Step 3: Resolve the launcher PID and bind GPU/fatal monitoring**

```bash
rtk pgrep -af '^/data/zcy/anaconda3/envs/mmdet/bin/python -m torch.distributed.launch.*textlr1e5.py'
rtk setsid -f /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/monitor_gpu_run.py --pid "$(rtk pgrep -n -f '^/data/zcy/anaconda3/envs/mmdet/bin/python -m torch.distributed.launch.*textlr1e5.py')" --gpu-indices 8 9 --interval 30 --train-log work_dirs/dotav2_cleanstart/s1_grouped_scale1024_seed20260716_gpu89_batch1_textlr1e5_launcher.log --output .lab/workspace/8-t4-textlr1e5-gpu89-monitor.jsonl
```

Expected: one launcher PID and monitor samples for physical GPUs 8/9 with `fatal_pattern=null`.

- [ ] **Step 4: Bind same-config auto-resume through the 10:00 deadline**

```bash
rtk setsid -f /data/zcy/anaconda3/envs/mmdet/bin/python .lab/workspace/guard_gpu89_resume.py --pid "$(rtk pgrep -n -f '^/data/zcy/anaconda3/envs/mmdet/bin/python -m torch.distributed.launch.*textlr1e5.py')" --deadline 2026-07-18T10:00:00+08:00 --config configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch1_textlr1e5.py --work-dir work_dirs/dotav2_cleanstart/s1_grouped_scale1024_seed20260716_gpu89_batch1_textlr1e5 --state-log .lab/workspace/8-t4-textlr1e5-auto-resume-guard.jsonl --interval 30
```

Expected: guard state identifies the T4 config/work directory and target `epoch_12.pth`.

- [ ] **Step 5: Record the launch and protocol hash in `.lab/results.tsv`, `.lab/log.md`, and `.lab/branches.md`**

Expected: T4 status changes from `thought` to `running`; the control endpoint remains `AP50=0.4030`, `novel4=0.11225`, and no raw-13,833 claim is made.

### Task 5: Verify and commit the isolated implementation

**Files:**
- Modify: `tests/test_projects/ov_capflow/test_dotav2_cleanstart_config.py`
- Create: `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch1_textlr1e5.py`
- Create: `docs/superpowers/plans/2026-07-18-dotav2-t4-text-lr.md`

- [ ] **Step 1: Check formatting and the focused diff**

```bash
rtk git diff --check -- tests/test_projects/ov_capflow/test_dotav2_cleanstart_config.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch1_textlr1e5.py docs/superpowers/plans/2026-07-18-dotav2-t4-text-lr.md
rtk git diff -- tests/test_projects/ov_capflow/test_dotav2_cleanstart_config.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch1_textlr1e5.py docs/superpowers/plans/2026-07-18-dotav2-t4-text-lr.md
```

Expected: no whitespace errors and no scientific delta beyond `language_model.lr_mult=0.1`.

- [ ] **Step 2: Commit only the plan, test, and config**

```bash
rtk git add docs/superpowers/plans/2026-07-18-dotav2-t4-text-lr.md tests/test_projects/ov_capflow/test_dotav2_cleanstart_config.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch1_textlr1e5.py
rtk git commit -m 'experiment: add DOTA-v2 T4 text LR screen'
```

Expected: one focused commit without staging unrelated worktree changes.
