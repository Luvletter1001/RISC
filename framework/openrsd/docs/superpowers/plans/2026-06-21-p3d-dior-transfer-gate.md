# P3D DIOR-R Transfer Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second-dataset P3D transfer evidence gate using DIOR-R so the stricter 10-gate audit can reach 8/10 only when cross-dataset Gaussian semantic-support transfer passes.

**Architecture:** Reuse the existing P3A AP-constrained projection algorithm without changing its semantics. P3D adds dataset-specific artifact paths, a result summarizer, and a gate reader that treats DIOR-R transfer as applicability breadth evidence only when mAP/topK precision are preserved and semantic false-alarm risk decreases without harming correct positives.

**Tech Stack:** Python, pytest, existing OpenRSD analysis scripts, `predictions.pkl`, DOTA-format annfiles, JSON/Markdown result artifacts.

---

### Task 1: Run DIOR-R P3D Projection

**Files:**
- Read: `M_Tools/analysis/apply_p3a_ap_constrained_support_projection.py`
- Read: `work_dirs/gs3c_dior_r_rtmdetl_dota_init_20260619/eval_epoch3_plus3_baseline_full/predictions.pkl`
- Read: `/data1/zcy/datasets/DIOR_R_dota/test/labelTxt`
- Read: `work_dirs/gs3c_dior_scope_20260619/dior_r_trainval_area_priors.csv`
- Create: `work_dirs/gs3c_dior_r_rtmdetl_dota_init_20260619/eval_epoch3_plus3_baseline_p3d_z2_s01/predictions.pkl`
- Create: `work_dirs/gs3c_dior_r_rtmdetl_dota_init_20260619/eval_epoch3_plus3_baseline_p3d_z2_s01/p3d_projection_summary.json`

- [ ] **Step 1: Run AP-constrained Gaussian semantic-support projection**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/openrsd_mplconfig /data/zcy/anaconda3/envs/openrsd/bin/python M_Tools/analysis/apply_p3a_ap_constrained_support_projection.py --input-pkl work_dirs/gs3c_dior_r_rtmdetl_dota_init_20260619/eval_epoch3_plus3_baseline_full/predictions.pkl --output-pkl work_dirs/gs3c_dior_r_rtmdetl_dota_init_20260619/eval_epoch3_plus3_baseline_p3d_z2_s01/predictions.pkl --ann-dir /data1/zcy/datasets/DIOR_R_dota/test/labelTxt --class-area-priors-csv work_dirs/gs3c_dior_scope_20260619/dior_r_trainval_area_priors.csv --config M_configs/Diagnostics/dior_r_rtmdet_l_dota_init_epoch3_plus3_eval_gpu0189.py --iou-thr 0.5 --z-thr 2.0 --lambda 0.5 --min-score 0.1 --summary-json work_dirs/gs3c_dior_r_rtmdetl_dota_init_20260619/eval_epoch3_plus3_baseline_p3d_z2_s01/p3d_projection_summary.json
```

Expected: command exits 0 and the summary reports `changed_correct=0`.

### Task 2: Evaluate DIOR-R AP And Deployment Risk

**Files:**
- Read: `M_Tools/analysis/eval_predictions_annfiles_metric_json.py`
- Read: `M_Tools/analysis/evaluate_sise_deployment_risk.py`
- Create: `work_dirs/gs3c_dior_r_rtmdetl_dota_init_20260619/eval_epoch3_plus3_baseline_p3d_z2_s01/offline_eval.json`
- Create: `work_dirs/gs3c_dior_r_rtmdetl_dota_init_20260619/eval_epoch3_plus3_baseline_p3d_z2_s01/offline_eval.md`
- Create: `work_dirs/gs3c_dior_r_rtmdetl_dota_init_20260619/sise_eval_p3d_transfer_z2_s01/deployment_risk_summary.json`

- [ ] **Step 1: Run annfiles-only AP evaluation**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/openrsd_mplconfig /data/zcy/anaconda3/envs/openrsd/bin/python M_Tools/analysis/eval_predictions_annfiles_metric_json.py --config M_configs/Diagnostics/dior_r_rtmdet_l_dota_init_epoch3_plus3_eval_gpu0189.py --predictions work_dirs/gs3c_dior_r_rtmdetl_dota_init_20260619/eval_epoch3_plus3_baseline_p3d_z2_s01/predictions.pkl --ann-dir /data1/zcy/datasets/DIOR_R_dota/test/labelTxt --out-json work_dirs/gs3c_dior_r_rtmdetl_dota_init_20260619/eval_epoch3_plus3_baseline_p3d_z2_s01/offline_eval.json --out-md work_dirs/gs3c_dior_r_rtmdetl_dota_init_20260619/eval_epoch3_plus3_baseline_p3d_z2_s01/offline_eval.md
```

Expected: `missing_ann=0` and mAP is not lower than `0.6446871757507324`.

- [ ] **Step 2: Run deployment risk and rank-tail audit**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/openrsd_mplconfig /data/zcy/anaconda3/envs/openrsd/bin/python M_Tools/analysis/evaluate_sise_deployment_risk.py --variant dior_baseline:work_dirs/gs3c_dior_r_rtmdetl_dota_init_20260619/eval_epoch3_plus3_baseline_full/predictions.pkl --variant p3d_dior_z2_s01:work_dirs/gs3c_dior_r_rtmdetl_dota_init_20260619/eval_epoch3_plus3_baseline_p3d_z2_s01/predictions.pkl --ann-dir /data1/zcy/datasets/DIOR_R_dota/test/labelTxt --class-area-priors-csv work_dirs/gs3c_dior_scope_20260619/dior_r_trainval_area_priors.csv --config M_configs/Diagnostics/dior_r_rtmdet_l_dota_init_epoch3_plus3_eval_gpu0189.py --out-dir work_dirs/gs3c_dior_r_rtmdetl_dota_init_20260619/sise_eval_p3d_transfer_z2_s01 --iou-thr 0.5 --z-thr 2.0 --topk 100,500,1000,5000 --rank-tail-audit
```

Expected: risk summary exits 0 and includes a `rank_tail_summary` row for `p3d_dior_z2_s01`.

### Task 3: Add P3D Summary Reader And Gate Integration

**Files:**
- Create: `M_Tools/analysis/summarize_p3d_transfer_results.py`
- Create: `tests/test_p3d_transfer_summary.py`
- Modify: `M_Tools/analysis/build_iclr95_evidence_gate.py`
- Modify: `M_Tools/analysis/refresh_bass_rankdelta_paper_package.py`
- Modify: `tests/test_iclr95_evidence_gate_closure_plan.py`
- Modify: `tests/test_bass_rankdelta_refresh_package.py`
- Create: `work_dirs/semantic_scale_six_experiments_20260620/p3d_transfer_summary.json`
- Create: `resultmd/exp_p4_scale_semantic_validation/fres_20260621_p3d_dior_transfer_gate.md`

- [ ] **Step 1: Write summary-reader tests**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_p3d_transfer_summary.py -q
```

Expected before implementation: import or assertion failure.

- [ ] **Step 2: Implement the P3D summary reader**

The reader consumes baseline eval JSON, P3D eval JSON, projection summary JSON, and risk JSON. It writes a JSON/Markdown summary and returns `strict_transfer_pass=True` only when mAP and top5000 precision are preserved, at least one top5000 semantic-risk measure improves, `changed_correct=0`, and `tp_rank_harm=0`.

- [ ] **Step 3: Wire P3D into the gate**

`build_iclr95_evidence_gate.py` reads `p3d_transfer_summary.json`. The `Applicability breadth` gate passes when P3D strict transfer passes and P3A strict pass exists.

- [ ] **Step 4: Wire P3D into refresh**

`refresh_bass_rankdelta_paper_package.py` runs the P3D summarizer before `build_iclr95_gate` and records `p3d_status`, `p3d_main_gate`, and the updated gate count in the manifest.

### Task 4: Refresh Paper Package And Verify

**Files:**
- Modify: `paper/drafts/semantic_scale_support_mismatch_iclr_draft_v0_4_20260620.md`
- Modify: `paper/semantic_scale_support_iclr/main.tex`
- Modify: `resultmd/exp_p4_scale_semantic_validation/freview_20260620_iclr95_evidence_gate_machine_audit.md`
- Modify: `resultmd/exp_p4_scale_semantic_validation/fstatus_20260620_bass_rankdelta_paper_package_refresh.md`

- [ ] **Step 1: Run focused tests**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_p3d_transfer_summary.py tests/test_iclr95_evidence_gate_closure_plan.py tests/test_bass_rankdelta_refresh_package.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Refresh package**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/openrsd_mplconfig /data/zcy/anaconda3/envs/openrsd/bin/python M_Tools/analysis/refresh_bass_rankdelta_paper_package.py
```

Expected: `iclr95_8of10_current=8/10` if P3D strict transfer passes. If it does not pass, the result must stay below 8/10 and the blocker must name the failed DIOR-R condition.
