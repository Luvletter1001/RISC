# P5C AP-Rank-Protected Geometry Gaussian Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and run a P5C detector-head experiment where class-conditioned geometry Gaussian support changes train-time ranking through an AP/rank-protected pair-margin loss.

**Architecture:** Reuse the existing `GaussianSemanticScaleDensityHead` and geometry support priors. Add one focused pair-margin loss that selects geometry-low-support hard negatives but optimizes the relative GT-vs-hard-negative delta margin, then expose it through `GSRRotatedRTMDetSepBNHead` config/debug fields. Launch two HRRSD variants on GPU6/GPU7 only after unit tests, py_compile, and MMEngine config parsing pass.

**Tech Stack:** PyTorch, MMRotate/MMEngine configs, OpenRSD `GSRRotatedRTMDetSepBNHead`, pytest, tmux, single-GPU A40 runs.

---

### Task 1: Pair-Margin Utility Loss

**Files:**
- Modify: `tests/test_gaussian_semantic_scale_density_head.py`
- Modify: `M_AD/models/utils/gaussian_semantic_scale.py`

- [ ] **Step 1: Write the failing test**

Add a test that imports `semantic_scale_density_pair_margin_loss`, builds one GT-positive row and one high-score low-support hard negative, and expects a positive loss plus debug counts. The first run should fail because the function does not exist.

- [ ] **Step 2: Run the failing test**

Run: `rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/openrsd_mplconfig /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_gaussian_semantic_scale_density_head.py::test_density_pair_margin_loss_protects_gt_against_geometry_hardneg -q`

Expected: import failure for `semantic_scale_density_pair_margin_loss`.

- [ ] **Step 3: Implement the utility loss**

Implement `semantic_scale_density_pair_margin_loss(logit_delta, log_prob, labels, assign_metrics, valid_mask, cls_logits, min_hardneg_score, min_logprob_gap, margin, max_hardneg)` so selected pairs optimize `relu(margin + hardneg_logit + hardneg_delta - gt_logit - gt_delta)` with assign-metric weights.

- [ ] **Step 4: Run the utility test**

Run the same pytest command and expect one passing test.

### Task 2: Dense Head Integration

**Files:**
- Modify: `tests/test_gs3c_closed_set_rtmdet_head.py`
- Modify: `M_AD/models/dense_heads/gs3c_rtmdet_head.py`

- [ ] **Step 1: Write the failing head test**

Add a test enabling `gaussian_semantic_density_pair_margin_loss_enable=True`, `use_geometry_logprob=True`, and geometry priors where class 1 is low support for a square GT box. Expect positive loss and debug keys `pair_margin_num_hardneg=1`, `pair_margin_active_violation_count=1`, `geometry_logprob_used=True`.

- [ ] **Step 2: Run the failing head test**

Run: `rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/openrsd_mplconfig /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_gs3c_closed_set_rtmdet_head.py::test_closed_set_density_pair_margin_loss_uses_geometry_support -q`

Expected: missing config attribute or debug key failure.

- [ ] **Step 3: Wire config and loss**

Add density-head config fields for `pair_margin_loss_enable`, `pair_margin_loss_weight`, `pair_margin_loss_margin`, `pair_margin_loss_min_logprob_gap`, `pair_margin_loss_min_hardneg_score`, and `pair_margin_loss_max_hardneg`. Call the new utility inside `_loss_gaussian_scale_density` using `support_log_prob`, so geometry mode reuses the same path.

- [ ] **Step 4: Run focused head tests**

Run: `rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/openrsd_mplconfig /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_gs3c_closed_set_rtmdet_head.py tests/test_gaussian_semantic_scale_density_head.py -q`

Expected: all focused tests pass.

### Task 3: P5C Configs and Launcher

**Files:**
- Create: `M_configs/Diagnostics/hrrsd_rtmdet_l_dota_init_internal_bass_gsf_p5c_pairmargin_train_gpu67.py`
- Create: `M_configs/Diagnostics/hrrsd_rtmdet_l_dota_init_internal_bass_gsf_p5c_pairmargin_eval_gpu67.py`
- Create: `M_Tools/experiments/run_hrrsd_bass_gsf_p5c_pairmargin_train_20260621.sh`

- [ ] **Step 1: Add train/eval configs**

Inherit from P5B geometry configs. Enable pair-margin loss, keep geometry support enabled, set P5C-specific work dirs, and keep `batch_sampler` absent.

- [ ] **Step 2: Add launcher**

Create a single-GPU launcher accepting GPU id, pair-margin weight, margin, hardneg score threshold, and max epochs. It should train, eval, and run the existing deployment-risk audit against no-G3 and density controls.

- [ ] **Step 3: Verify configs**

Run py_compile on changed Python files/configs and parse configs with MMEngine. Confirm `train_dataloader` has no `batch_sampler`.

### Task 4: Launch GPU6/GPU7 Runs

**Files:**
- Logs under `work_dirs/train_queue_logs/`
- Result checkpoint/eval dirs under `work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/`

- [ ] **Step 1: Start two tmux sessions**

GPU6 mild: small pair-margin weight and margin. GPU7 stronger: larger weight/margin but still score-gated.

- [ ] **Step 2: Health check**

Within one minute, inspect tmux sessions, GPU memory, and the first training log. Stop only if there is an immediate config crash.

- [ ] **Step 3: Record status**

Write a `resultmd/exp_p4_scale_semantic_validation/fstatus_20260621_p5c_pairmargin_gpu67_live.md` status file with command lines, success gates, and current health.
