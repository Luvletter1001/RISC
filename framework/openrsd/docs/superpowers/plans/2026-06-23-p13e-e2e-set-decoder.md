# P13E End-to-End Set Decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch four HRRSD 3epoch P13E smoke experiments where support fusion directly emits a fixed set of rotated detections with one-to-one matching.

**Architecture:** Add a new `P13EDenseSeedGaussianSetHead` that receives FPN features, selects internal Top-K seed queries, runs a lightweight decoder, and directly outputs class logits, quality, and rotated boxes. Training uses per-image Hungarian one-to-one matching; prediction emits one label per query with no dense anchor head and no NMS.

**Tech Stack:** PyTorch, MMEngine/MMDetection single-stage detector interface, MMRotate DOTA metric, HRRSD internal split.

---

## File Structure

- `M_AD/models/dense_heads/p13e_set_decoder_head.py`: strict end-to-end set-prediction head.
- `tests/test_p13e_set_decoder_head.py`: forward/loss/predict tests for fixed-query one-to-one behavior.
- `M_configs/Diagnostics/hrrsd_rtmdet_l_dota_init_internal_p13e_*_train_gpu67.py`: four HRRSD configs.
- `M_Tools/experiments/run_hrrsd_p13e_single_20260623.sh`: one experiment train/eval/risk wrapper.
- `M_Tools/experiments/run_hrrsd_p13e_4card_20260623.sh`: GPU0/1/6/7 parallel launcher.
- `resultmd/exp_p13_fusion_decoder_hrrsd/`: result and failure/lesson records.

## Tasks

### Task 1: P13E Head

**Files:**
- Create: `M_AD/models/dense_heads/p13e_set_decoder_head.py`

- [ ] Implement `P13EDenseSeedGaussianSetHead` with:
  - `forward(feats) -> (cls_logits, box_preds, quality_logits, seed_logits)`.
  - fixed query count `num_queries`.
  - internal dense seed selection only for query initialization.
  - support Gaussian/product class logits through learned class support tokens.
  - one box and one class score per query at prediction time.

### Task 2: One-to-One Loss

**Files:**
- Modify: `M_AD/models/dense_heads/p13e_set_decoder_head.py`

- [ ] Add per-image Hungarian matching using class focal cost, normalized `cxcywh` L1 cost, and angle L1 cost.
- [ ] Add losses: `loss_cls`, `loss_bbox`, `loss_angle`, `loss_quality`, and internal `loss_seed`.
- [ ] Keep all detection outputs end-to-end; `loss_seed` only trains query seed selection and is not a detection output.

### Task 3: Unit Tests

**Files:**
- Create: `tests/test_p13e_set_decoder_head.py`

- [ ] Test forward tensor shapes.
- [ ] Test loss returns finite values on synthetic rotated boxes.
- [ ] Test prediction emits at most `num_queries` boxes and sets `e2e_debug['uses_nms'] == False`.

### Task 4: Four Configs

**Files:**
- Create four `M_configs/Diagnostics/hrrsd_rtmdet_l_dota_init_internal_p13e_*_train_gpu67.py` configs.

- [ ] `p13e_k100_seed`: conservative query count.
- [ ] `p13e_k200_seed`: balanced query count.
- [ ] `p13e_k300_seed`: recall-heavy query count.
- [ ] `p13e_k200_seed_strong`: stronger decoder/objectness loss.

### Task 5: Launch Scripts

**Files:**
- Create: `M_Tools/experiments/run_hrrsd_p13e_single_20260623.sh`
- Create: `M_Tools/experiments/run_hrrsd_p13e_4card_20260623.sh`

- [ ] Train each config for 3 epochs.
- [ ] Evaluate mAP/AP50 after checkpoint.
- [ ] Run baseline and P11 risk comparisons.
- [ ] Launch on GPUs 0/1/6/7 as four independent single-GPU jobs.

### Task 6: Verification Before Launch

- [ ] Run `py_compile` for the new head, test, configs, and scripts.
- [ ] Run `pytest tests/test_p13e_set_decoder_head.py -q`.
- [ ] Parse configs with `Config.fromfile`.
- [ ] Build one full model from config.
- [ ] Check GPUs and launch tmux only after the above checks pass.
