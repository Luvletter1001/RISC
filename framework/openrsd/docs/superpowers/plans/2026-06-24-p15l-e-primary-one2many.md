# P15L-E Primary One-to-Many Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a training-only MS-DETR-style one-to-many auxiliary supervision path to the stable P15L-B strict-E2E head and launch HRRSD overfit100 80epoch batch8*4 validation.

**Architecture:** Keep P15L-B query selection and inference unchanged. Add optional P15L-E parameters to `P15BOrientedDINOSetHead`; when enabled, `_loss_single()` computes standard Hungarian one-to-one losses, then adds a low-weight auxiliary loss for unmatched low-cost primary queries.

**Tech Stack:** PyTorch, MMEngine/MMRotate, HRRSD overfit100 configs, 4GPU DDP runner with NCCL P2P/IB disabled.

---

### Task 1: Contract Tests

**Files:**
- Modify: `tests/test_p15_oriented_dino_contract.py`

- [ ] **Step 1: Add tests before production code**

Add tests that assert:
- defaults keep `uses_aux_one2many_primary_supervision=False`
- enabling `aux_one2many_topk=1` selects only unmatched queries
- warmup scale is `0.0` at step 0 and `1.0` after warmup
- `predict()` still returns all fixed queries with `objectness_sigmoid_x_foreground_softmax`

- [ ] **Step 2: Verify RED**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mplconfig /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_p15_oriented_dino_contract.py -q
```

Expected: fail because P15L-E parameters/helpers are missing.

### Task 2: Head Implementation

**Files:**
- Modify: `M_AD/models/dense_heads/p15_oriented_dino_head.py`

- [ ] **Step 1: Add optional parameters**

Add `aux_one2many_topk`, `aux_one2many_loss_weight`, and `aux_one2many_warmup_iters` to `P15BOrientedDINOSetHead.__init__`, defaulting to disabled.

- [ ] **Step 2: Reuse matching cost**

Refactor the existing Hungarian cost calculation into a helper that can return the cost matrix and assignment without changing default behavior.

- [ ] **Step 3: Add auxiliary loss**

In final-layer `_loss_single(prefix='')`, exclude Hungarian positives, select the lowest-cost unmatched query per GT, and add low-weight cls/obj/bbox/angle/GWD losses. Do not add auxiliary losses to encoder, intermediate decoder layers, or DN outputs in v1.

- [ ] **Step 4: Verify GREEN**

Run focused pytest and py_compile. Expected: all P15 contract tests pass.

### Task 3: Config and Runner

**Files:**
- Create: `M_configs/Diagnostics/hrrsd_p15l_e_one2many_primary_overfit100_4gpu_b8_v5.py`
- Create: `M_Tools/experiments/run_hrrsd_p15l_e_one2many_primary_overfit100_4gpu_b8_v5_20260624.sh`

- [ ] **Step 1: Config**

Inherit from P15L-B repeat config, set `aux_one2many_topk=1`, `aux_one2many_loss_weight=0.10`, `aux_one2many_warmup_iters=80`, keep HRRSD overfit100, `batch_size=8`, `val_interval=5`.

- [ ] **Step 2: Runner**

Use GPUs `0,1,6,7`, `NPROC=4`, per-GPU batch 8, `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`, and write logs under `work_dirs/p15l_e_one2many_primary_hrrsd_20260624`.

### Task 4: Preflight and Launch

- [ ] **Step 1: Preflight**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mplconfig /data/zcy/anaconda3/envs/openrsd/bin/python -m py_compile M_AD/models/dense_heads/p15_oriented_dino_head.py M_configs/Diagnostics/hrrsd_p15l_e_one2many_primary_overfit100_4gpu_b8_v5.py
rtk bash -n M_Tools/experiments/run_hrrsd_p15l_e_one2many_primary_overfit100_4gpu_b8_v5_20260624.sh
rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mplconfig /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_p15_oriented_dino_contract.py -q
```

- [ ] **Step 2: Config parse**

Confirm batch and sampler:

```bash
rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mplconfig /data/zcy/anaconda3/envs/openrsd/bin/python -c "from mmengine.config import Config; p='M_configs/Diagnostics/hrrsd_p15l_e_one2many_primary_overfit100_4gpu_b8_v5.py'; cfg=Config.fromfile(p); td=cfg.train_dataloader; print(td.get('batch_size'), td.get('sampler',{}).get('batch_size'), 'batch_sampler' in td)"
```

- [ ] **Step 3: Launch**

Start tmux `p15l_e_one2many_20260624` and monitor epoch 1/eval logs plus GPU 0/1/6/7 memory.
