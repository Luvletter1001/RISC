# P13N Teacher-Anchored Gaussian Set Posterior Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and smoke-test P13N-A, a strict E2E/no-NMS set-posterior detector trained with P13C teacher supervision but not using the P13C dense head at inference.

**Architecture:** P13N inherits the strict fixed set posterior family but does not inherit P13L's inference quota/threshold fallback. During training only, GT boxes and optional P13C teacher boxes are converted into a class-wise Gaussian teacher field. Query logits, quality, boxes, and rank are pulled toward teacher-supported targets while GT boxes keep priority over teacher false positives. Inference is exactly `fixed queries -> posterior scores -> class-locked box/cls` with no dense detection head, no NMS, no score-threshold filtering, and no empty-result fallback.

**Tech Stack:** PyTorch, MMRotate/MMEngine, HRRSD configs, existing P13 set decoder utilities.

---

### Task 1: TDD Teacher Field API

**Files:**
- Modify: `tests/test_p13e_set_decoder_head.py`
- Modify: `M_AD/models/dense_heads/p13e_set_decoder_head.py`

- [x] Add a failing test importing `P13NTeacherAnchoredPosteriorHead`.
- [x] Test that GT boxes override lower-score teacher boxes near the same class and center.
- [x] Test that a teacher-only high score creates soft objectness and finite teacher losses.
- [x] Test that prediction returns the fixed query set directly without threshold fallback, NMS, or per-class quota selection.
- [x] Implement `_teacher_augmented_targets` and `_teacher_anchored_loss_single`.
- [x] Run `pytest tests/test_p13e_set_decoder_head.py -q`.

### Task 2: Config And Runner

**Files:**
- Create: `M_configs/Diagnostics/hrrsd_rtmdet_l_dota_init_internal_p13n_teacher_anchored_posterior_e2e_4gpu.py`
- Create: `M_Tools/experiments/run_hrrsd_p13n_4gpu_20260623.sh`

- [x] Create a P13N config inheriting P13K training settings but overriding inference to strict all-query posterior output.
- [x] Keep `train_dataloader=dict(batch_size=2)` only; do not add `batch_sampler`.
- [x] Set `score_thr=0.0`, `max_per_img=num_queries`; do not use `per_class_max_per_img` for P13N inference.
- [x] Set teacher weights conservatively so GT remains dominant.
- [x] Create a 4GPU train+4GPU eval runner with `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`.

### Task 3: Verification And Launch

**Files:**
- Modify: `CODEX_WORKLOG.md`
- Create: `resultmd/exp_p13_fusion_decoder_hrrsd/fres_20260623_p13n_teacher_anchored_result.md` after results.

- [x] Run unit tests.
- [x] Run py_compile for head/config/runner syntax.
- [x] Parse config with MMEngine and build model.
- [x] Launch tmux 4GPU train+eval.
- [x] Monitor train/eval logs.
- [x] Record mAP/AP50, prediction statistics, and decision.
