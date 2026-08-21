# P4 GSRL Non-Oracle Ranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a non-oracle Gaussian semantic-support ranker that can be compared against the current P3 AP-constrained oracle projection baseline.

**Architecture:** Add one focused offline prediction transformer that reads existing `predictions.pkl`, class-conditioned log-area Gaussian priors, and detector class order. It updates detection scores in logit space using positive support reward and low-support penalty, without reading validation annotations, changing boxes, or using bbox Gaussian distances.

**Tech Stack:** Python, PyTorch tensor-safe pickle transforms, existing OpenRSD `evaluate_sise_score_calibration.py` helpers, `pytest`, existing DOTA annfiles-only AP evaluator, existing SISE deployment risk evaluator.

---

### Task 1: Add P4/GSRL Transformer

**Files:**
- Create: `M_Tools/analysis/apply_p4_gsrl_support_ranker.py`
- Test: `tests/test_p4_gsrl_support_ranker.py`

- [ ] **Step 1: Write failing tests**

Create tests that build a tiny `predictions.pkl` and priors CSV. Assert that an in-support detection is boosted, an out-of-support detection is penalized, scores remain in `[0, 1]`, and the summary records `uses_gt_at_inference=False`.

- [ ] **Step 2: Implement the transformer**

Implement `apply_p4_gsrl_ranker(...)` with:

```text
z = abs((log(area) - mu_class) / sigma_class)
reward = max(0, reward_z - z)
penalty = max(0, z - penalty_z)
score' = sigmoid(logit(score) + clamp(alpha * reward - beta * penalty))
```

Only use predicted class, predicted box area, score, and train-set prior.

- [ ] **Step 3: Add CLI**

Expose `--input-pkl`, `--output-pkl`, `--class-area-priors-csv`, `--config`, `--class-names`, `--alpha`, `--beta`, `--reward-z`, `--penalty-z`, `--min-score`, `--max-up-delta`, `--max-down-delta`, and `--summary-json`.

### Task 2: Run Candidate Sweeps

**Files:**
- Output: `work_dirs/semantic_scale_p4_gsrl_20260621/**`

- [ ] **Step 1: Run P4 on DIOR-R baseline predictions**

Use:

```bash
rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/openrsd_mplconfig /data/zcy/anaconda3/envs/openrsd/bin/python M_Tools/analysis/apply_p4_gsrl_support_ranker.py ...
```

- [ ] **Step 2: Evaluate AP and deployment risk**

Use existing annfiles-only metric eval and SISE deployment risk scripts. Compare against DIOR-R baseline and P3D.

- [ ] **Step 3: Optionally repeat best setting on HRRSD**

Only repeat if DIOR-R shows a useful AP/risk frontier, because the user asked for paper-impactful evidence rather than occupying GPUs.

### Task 3: Write Paper Evidence

**Files:**
- Create: `resultmd/exp_p4_scale_semantic_validation/fres_20260621_p4_gsrl_non_oracle_ranker.md`
- Modify: `paper/drafts/semantic_scale_support_mismatch_iclr_draft_v0_4_20260620.md`
- Modify: `paper/semantic_scale_support_iclr/main.tex`

- [ ] **Step 1: Record real results**

Write whether P4 exceeds the current P3 partial-oracle projection baseline. Do not claim it exceeds a true global oracle.

- [ ] **Step 2: Update manuscript**

Add a concise P4 paragraph/table row with the non-oracle boundary, Gaussian support formula, and result.

### Self-Review

- Spec coverage: covers implementation, offline evaluation, risk audit, and manuscript update.
- Placeholder scan: no metric placeholder is used as a result; actual result values must come from generated JSON.
- Scope check: single offline experiment package; no new training job or GPU dependency unless later promoted.
