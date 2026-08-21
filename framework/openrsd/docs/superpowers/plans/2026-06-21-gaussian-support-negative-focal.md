# Gaussian Support Negative Focal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a train-only Gaussian support-aware negative focal loss that targets low-support, high-score semantic false positives without changing inference-time logits or positive-sample ranking.

**Architecture:** The new loss lives in `M_AD/models/utils/gaussian_semantic_scale.py` as a pure tensor utility, then is called from `GSRRotatedRTMDetSepBNHead._loss_gaussian_scale_density`. It uses decoded predicted boxes for negative locations, computes class support from the existing area/geometry Gaussian priors, and adds a small auxiliary BCE penalty only for negative class-location pairs whose score is high and whose class support is much worse than the best supported class at that location.

**Tech Stack:** PyTorch, MMRotate RTMDet head, existing G-S3C Gaussian prior utilities, pytest.

---

## Failure Lessons From P5B/P5C

- P5B directly changed inference logits and reduced mAP, so the next method must not introduce inference-time suppression.
- P5C tried to protect AP through pair-margin on positives, but still spent optimization budget around positive ranking and did not beat no-G3.
- The observed useful signal is not “always lower low-support classes”; it is “when a background/negative location assigns a high score to a semantically unsupported class, train that class score down harder.”
- Therefore the new method should be train-only, negative-only, and support-gated.

## Task 1: Utility Loss

**Files:**
- Modify: `tests/test_gaussian_semantic_scale_density_head.py`
- Modify: `M_AD/models/utils/gaussian_semantic_scale.py`

- [ ] **Step 1: Write the failing test**

Add a test named `test_support_negative_focal_loss_penalizes_low_support_high_score_negatives` that constructs three locations and two classes:

```python
cls_logits = torch.tensor([
    [torch.logit(torch.tensor(0.90)), torch.logit(torch.tensor(0.10))],
    [torch.logit(torch.tensor(0.20)), torch.logit(torch.tensor(0.80))],
    [torch.logit(torch.tensor(0.95)), torch.logit(torch.tensor(0.20))],
], requires_grad=True)
log_prob = torch.tensor([
    [-0.2, -4.5],
    [-0.4, -0.3],
    [-0.2, -5.0],
])
labels = torch.tensor([2, 2, 0])
valid = torch.tensor([True, True])
```

Expected behavior:

- location 0 class 1 is low support but low score, so score-gated out;
- location 1 has high score but no support gap, so not selected;
- location 2 is a positive label and must not be selected;
- with score threshold 0.05, location 0 class 1 becomes selected and produces positive BCE gradient.

- [ ] **Step 2: Verify RED**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/openrsd_mplconfig /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_gaussian_semantic_scale_density_head.py::test_support_negative_focal_loss_penalizes_low_support_high_score_negatives -q
```

Expected: import failure for `gaussian_support_negative_focal_loss`.

- [ ] **Step 3: Implement the minimal utility**

Add `gaussian_support_negative_focal_loss` with this behavior:

- negative locations are `labels >= num_classes`;
- candidates satisfy `best_valid_log_prob - class_log_prob >= min_logprob_gap`;
- optional score gate uses `sigmoid(cls_logits) >= min_score`;
- select up to `max_hardneg` per negative location by score;
- loss is `softplus(selected_logit) * score.detach() ** gamma * gap_weight`, normalized by selected count.

- [ ] **Step 4: Verify GREEN**

Run the same focused utility test, then the full density utility test file.

## Task 2: Head Integration

**Files:**
- Modify: `tests/test_gs3c_closed_set_rtmdet_head.py`
- Modify: `M_AD/models/dense_heads/gs3c_rtmdet_head.py`

- [ ] **Step 1: Write the failing head test**

Add `test_closed_set_density_support_negative_loss_uses_predicted_geometry_only_for_negatives`:

- enable density head and support-negative loss;
- set `density_loss_weight=0.0`;
- pass one negative label and one positive label;
- pass decoded predicted boxes where the negative location has geometry matching class 0 but high logit for class 1;
- assert the loss is positive, `support_negative_selected_pairs == 1`, and positive labels are not selected.

- [ ] **Step 2: Verify RED**

Run the single head test and confirm the missing config/debug path fails.

- [ ] **Step 3: Implement head wiring**

Add density config keys:

- `support_negative_loss_enable`
- `support_negative_loss_weight`
- `support_negative_loss_min_score`
- `support_negative_loss_min_logprob_gap`
- `support_negative_loss_gamma`
- `support_negative_loss_gap_scale`
- `support_negative_loss_max_extra_weight`
- `support_negative_loss_max_hardneg`

Use predicted boxes for this auxiliary loss, not bbox targets. Reuse geometry support when `density_head.use_geometry_logprob=True`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/openrsd_mplconfig /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_gaussian_semantic_scale_density_head.py tests/test_gs3c_closed_set_rtmdet_head.py -q
```

Expected: all focused tests pass.

## Task 3: Diagnostic Config

**Files:**
- Add: `M_configs/Diagnostics/hrrsd_rtmdet_l_dota_init_internal_bass_gsf_p6_supportneg_train_gpu67.py`
- Add: `M_configs/Diagnostics/hrrsd_rtmdet_l_dota_init_internal_bass_gsf_p6_supportneg_eval_gpu67.py`
- Add: `resultmd/exp_p4_scale_semantic_validation/fplan_20260621_p6_support_negative_focal.md`

- [ ] **Step 1: Add train config**

Base it on the P5C train config, but disable inference-time density deltas and all P5B/P5C positive/pair/delta losses:

```python
model = dict(
    bbox_head=dict(
        gaussian_semantic_scale=dict(
            density_head=dict(
                apply_logit_delta=False,
                delta_loss_enable=False,
                positive_delta_loss_enable=False,
                pair_margin_loss_enable=False,
                support_negative_loss_enable=True,
                support_negative_loss_weight=0.01,
                support_negative_loss_min_score=0.05,
                support_negative_loss_min_logprob_gap=1.5,
                support_negative_loss_gamma=2.0,
                support_negative_loss_gap_scale=8.0,
                support_negative_loss_max_extra_weight=2.0,
                support_negative_loss_max_hardneg=1,
            )
        )
    )
)
```

- [ ] **Step 2: Add eval config**

Keep `apply_logit_delta=False`; this method is train-only and evaluation must use normal logits.

- [ ] **Step 3: Verify syntax and config parse**

Run `py_compile` and a small `Config.fromfile` parse for both config files.

## Task 4: Evidence Record

**Files:**
- Add: `resultmd/exp_p4_scale_semantic_validation/fplan_20260621_p6_support_negative_focal.md`

- [ ] **Step 1: Record the hypothesis**

Summarize why P5B/P5C failed and why P6 is a better test:

- train-only;
- negative-only;
- predicted-geometry support;
- no inference-time calibration.

- [ ] **Step 2: Record launch criteria**

Only launch if focused tests and config parse pass.
