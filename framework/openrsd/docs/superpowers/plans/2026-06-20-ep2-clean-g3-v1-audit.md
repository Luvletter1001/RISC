# E-P2 Clean and G3-v1 Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a gate-first causal/path intervention auditor for E-P2-clean and summarize existing G3-v1 dense-head results as negative/partial before any new GPU experiment.

**Architecture:** Add one standard-library E-P2-clean analyzer that consumes path-probe CSV rows and reports whether scale-support advantage causally tracks hard-negative logit margin under fixed context/object constraints. Add one G3-v1 result auditor that reads existing HRRSD eval JSON and deployment-risk CSV files and writes a Chinese resultmd summary comparing G3/density variants against matched no-G3 controls.

**Tech Stack:** Python standard library, pytest, existing OpenRSD resultmd/work_dirs layout.

---

### Task 1: E-P2-clean Path Auditor

**Files:**
- Create: `tests/test_ep2_clean_path_intervention.py`
- Create: `M_Tools/analysis/evaluate_ep2_clean_path_intervention.py`

- [ ] **Step 1: Write failing tests**

Create tests for:
- clean path passes when hard-negative support advantage and hard-negative margin rise together under fixed context/object.
- shuffled-prior control is rejected even if present in the same CSV.
- varying `context_id` invalidates a case as confounded.

- [ ] **Step 2: Run the test and verify it fails**

Run: `rtk /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_ep2_clean_path_intervention.py -q`

Expected: import failure because `M_Tools.analysis.evaluate_ep2_clean_path_intervention` does not exist.

- [ ] **Step 3: Implement the analyzer**

Implement:
- CSV loader/writer.
- per-row `support_advantage = abs_z_gt - abs_z_hardneg`.
- per-row `hardneg_margin = logit_hardneg - logit_gt`.
- per-case Spearman correlation and endpoint margin/support deltas.
- clean gate requiring fixed context/object, enough points, positive rho, positive margin delta, positive support delta.
- control specificity requiring clean pass rate to exceed control pass rate.
- CLI outputs: `ep2_clean_case_summary.csv`, `ep2_clean_control_summary.csv`, `ep2_clean_review.json`, `ep2_clean_report.md`.

- [ ] **Step 4: Run tests and py_compile**

Run:
`rtk /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_ep2_clean_path_intervention.py -q`
`rtk /data/zcy/anaconda3/envs/openrsd/bin/python -m py_compile M_Tools/analysis/evaluate_ep2_clean_path_intervention.py`

Expected: pass.

### Task 2: G3-v1 Existing Result Audit

**Files:**
- Create: `M_Tools/analysis/audit_g3_v1_dense_head_results.py`
- Create via script: `resultmd/exp_p4_scale_semantic_validation/fres_20260620_g3_v1_negative_partial_audit.md`

- [ ] **Step 1: Implement standard-library result reader**

Read existing HRRSD G3/density/no-G3 deployment-risk CSVs and eval JSONs from:
- `work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619`
- `work_dirs/gs3c_sise_problem_reframing_20260619`

- [ ] **Step 2: Compute matched-control interpretation**

Report:
- baseline mAP.
- best no-G3 same-epoch control.
- best G3 by mAP.
- best G3 by total `sise_logz_wrong_excl_sibling` reduction.
- whether any G3 beats best no-G3 control in mAP.

- [ ] **Step 3: Write resultmd with strict reviewer conclusion**

Conclusion must explicitly say G3-v1 is `negative/partial`: network path works, risk can move, but G3-v1 has not beaten matched no-G3 control.

### Task 3: Gate Before New GPU Runs

**Files:**
- Update: `resultmd/exp_p4_scale_semantic_validation/fres_20260620_g3_v1_negative_partial_audit.md`

- [ ] **Step 1: Add G3-v2 gate**

State no new GPU run should start until:
- E-P2-clean script has a real path-probe CSV.
- G3-v2 design has a matched no-G3 same-seed control.
- success means beating matched no-G3 in mAP or AP-sensitive utility, not merely beating epoch3 baseline.
