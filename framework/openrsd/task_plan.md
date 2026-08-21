# Rotation Semantic Attractor Experiment Deployment Plan

Goal: deploy `experiments/rotation_semantic_attractor/` and run the smoke test for the full experiment pipeline using GPUs 6,7,8,9.

## Phases

| Phase | Status | Notes |
|---|---|---|
| Read workflow rules and current repo state | complete | Skills loaded; old planning files were from a previous visualization task. |
| Inventory existing assets and reusable entry points | complete | DOTA angle-sweep data, weights, configs, and cached predictions exist; initial sandbox CUDA issue was resolved by GPU-visible smoke execution. |
| Implement experiment module skeleton | complete | `experiments/rotation_semantic_attractor/` package, configs, scripts, adapters, metrics, tests, and report builder added. |
| Implement geometry, adapters, metrics, report, and scripts | complete | Rotation/GT helpers, mmrotate adapter, closed-set registries, false-hub metrics, smoke placeholders, and report generation implemented. |
| Run unit tests | complete | `rtk python3 -m pytest -o addopts='' experiments/rotation_semantic_attractor/tests/test_rotation_geometry.py -q` passed with 3 tests. |
| Run smoke test on GPUs 6,7,8,9 | complete | GPU-visible smoke ran with `CUDA_VISIBLE_DEVICES=6,7,8,9`, PyTorch saw 4 CUDA devices, and rotated RetinaNet loaded on `cuda:0`. |
| Audit outputs and summarize issues | complete | Initial smoke limitations were recorded before the status-semantics repair; scientific gap audit is now in `experiments/rotation_semantic_attractor/reports/scientific_gap_audit.md`. |
| Repair status semantics and capability-driven reporting | complete | Added unified statuses, capability profiles, matrix smoke split, and report filtering so architecture N/A, unavailable assets, proxy/schema rows, and real measurements are not conflated. |
| Add DONE_FULL gates and tests | complete | DONE_FULL now requires real full-split/full-angle evidence and rejects smoke/proxy/schema-only rows per experiment family. |
| Inventory OpenRSD/open-vocabulary assets | complete | Local OpenRSD config/checkpoint/support candidates were found; final inventory reports 5 runnable OpenRSD pairs. |
| Implement real OpenRSD adapter/hook smoke | complete | OpenRSD hook smoke runs real forward inference on GPU and captures dense logits, pre-NMS head outputs, visual-support embeddings, and projection/features. |
| Implement real closed-set intervention smoke | complete | Closed-set classifier weight intervention modifies the small-vehicle channel and reruns inference; rows are `DONE_SMOKE`, not proxy. |
| Implement real image-level context counterfactual smoke | complete | Object-only/context-only edited images are generated and rerun through inference; rows are `DONE_SMOKE`, not proxy. |
| Implement real DeHub safety smoke | complete | Baseline and DeHub repair checkpoints are run on the same smoke images/angles with class drift and low-risk inflation metrics. |
| Prepare full benchmark runner | complete | `16_run_full_closedset_benchmark.py` creates a 10+ model, 12-angle, S2 final-test command plan without executing the heavy benchmark by default. |
| Verify repaired smoke matrix | complete | Closed-set smoke: `PASS_WITH_NOT_APPLICABLE`; open-vocab OpenRSD smoke: `DONE_SMOKE`; matrix smoke: `PASS_WITH_NOT_APPLICABLE`. No `DONE_FULL` scientific rows are emitted. |

## Constraints

- Prefix shell commands with `rtk`.
- Use `CUDA_VISIBLE_DEVICES=6,7,8,9` for smoke commands that may touch GPU.
- Do not run the full 2500-tile benchmark by default.
- Do not invent results; missing assets produce `NOT_RUN` or `UNSUPPORTED` with reasons.
- Do not treat architecture-level non-applicability as unsupported failure. Closed-set query logits must be reported as `NOT_APPLICABLE`.
- Do not include smoke proxy/schema rows in scientific-result tables.
- Do not modify core training or inference scripts unless an adapter absolutely requires it.
- Do not upgrade tiny smoke runs, adapter probes, or schema/proxy rows to `DONE_FULL`.
- Do not claim a full causal or repair result until the S2 full split, 12-angle, multi-model benchmark and the required paired causal/safety checks have actually run.

## Errors Encountered

| Error | Attempt | Resolution |
|---|---|---|
| OpenRSD conda lacks pytest | Geometry test red attempt | Used system `python3 -m pytest -o addopts=''` because project `pytest.ini` requires missing xdoctest plugin. |
| User site scipy breaks mmrotate imports | Initial mmdet API probe | Use `PYTHONNOUSERSITE=1` for OpenRSD Python commands. |
| Sandbox-limited PyTorch could not see CUDA | Initial smoke/diagnostic attempt | Reran the decisive CUDA diagnostic and smoke outside the sandbox with approved escalation; main evidence is the GPU run. |
| Stage decomposition hooks unavailable | Smoke audit | Marked pre-NMS, dense logits, and query logits as `UNSUPPORTED` instead of inventing causal-stage results. |
| Intervention/counterfactual experiments not executed in smoke | Smoke audit | Wrote explicit `NOT_RUN` outputs for open-vocab intervention, closed-set intervention, context counterfactual, and DeHub safety scripts. |
| Model registry count command assumed list-shaped data | Final audit | Re-read registry and corrected the parser for the actual `models` dictionary shape; registry has 15 methods, 11 with checkpoints. |
| Four registry methods lacked checkpoints | Repair pass | Added ai4rs/OpenMMLab checkpoint URLs and downloaded RoI Transformer, S2ANet, Rotated FCOS, and Rotated ATSS weights into `weights/`; inventory now reports 15/15 models available. |
| Stage decomposition was fully `UNSUPPORTED` | Repair pass | Added adapter-side single-stage pre-NMS/dense probe using `extract_feat` and `bbox_head.predict_by_feat(..., with_nms=False)`; repaired smoke reports `PARTIAL` because closed-set models still do not expose query logits. |
| Intervention/counterfactual/safety rows were empty `NOT_RUN` | Repair pass | Replaced smoke placeholders with posthoc/proxy rows: closed-set channel proxy, context metric proxy, and DeHub safety smoke proxy. |
| Closed-set query logits were classified like unsupported hooks | Status semantics repair | Added capability profiles and report rendering; closed-set dense/head detectors now show `query logits: N/A by architecture` with status `NOT_APPLICABLE`. |
| Proxy/schema rows could be read as scientific results | Status semantics repair | Added explicit `SMOKE_PROXY` and `SCHEMA_ONLY` metadata with `is_scientific_result=false`, `include_in_main_table=false`, and `claim_level=none`. |
| Open-vocabulary intervention was shown as ordinary `NOT_RUN` in closed-set smoke | Status semantics repair | Split smoke scripts; closed-set-only smoke now records open-vocabulary intervention as `NOT_SELECTED_IN_THIS_SMOKE`, while open-vocab smoke records `NOT_AVAILABLE_ASSET` until a real asset exists. |

## Repair Verification

- `rtk python3 -m pytest -o addopts='' experiments/rotation_semantic_attractor/tests -q`: 7 passed.
- `rtk python3 -m compileall -q experiments/rotation_semantic_attractor`: exit 0.
- GPU smoke on `CUDA_VISIBLE_DEVICES=6,7,8,9`: `PASS_WITH_PARTIAL_HOOKS`.
- Smoke run dir: `experiments/rotation_semantic_attractor/outputs/runs/smoke_2tiles_2angles_gpu_hooks`.
- Repaired status test suite: `rtk python3 -m pytest -o addopts='' experiments/rotation_semantic_attractor/tests -q` -> 13 passed.
- Repaired closed-set smoke on `CUDA_VISIBLE_DEVICES=6,7,8,9`: `PASS_WITH_NOT_APPLICABLE`.
- Repaired open-vocabulary smoke: `PASS_SCHEMA_ONLY` with `openvocab_status=NOT_AVAILABLE_ASSET`.
- Repaired matrix smoke on `CUDA_VISIBLE_DEVICES=6,7,8,9`: `PASS_WITH_NOT_APPLICABLE`.
- Scientific buildout verification: `rtk python3 -m pytest -o addopts='' experiments/rotation_semantic_attractor/tests -q` -> 35 passed before the final open-vocab smoke integration patch.
- OpenRSD open-vocab smoke integration: `run_smoke_openvocab.sh` now runs asset inventory plus `15_openrsd_hook_smoke.py`; latest `outputs/smoke/openvocab/smoke_verdict.txt` reports `verdict=PASS` and `openvocab_status=DONE_SMOKE`.
- Latest matrix verdict: `experiments/rotation_semantic_attractor/outputs/smoke/matrix/smoke_verdict.txt` reports `PASS_WITH_NOT_APPLICABLE` with closed-set query logits as architecture-level N/A and open-vocab OpenRSD smoke as `DONE_SMOKE`.

## DONE_FULL Execution Addendum

| Phase | Status | Notes |
|---|---|---|
| Full readiness audit | complete | Wrote `experiments/rotation_semantic_attractor/reports/full_execution_readiness_audit.md` and `.json`; DONE_FULL count remains 0 until the heavy run completes. |
| Freeze S2/S3 full splits | complete | Reused existing S2 final-test split and rebuilt/froze S3 only because the old S3 lacked the required low-risk normal stratum. |
| Full closed-set execution path | complete | Added resumable per-model manifests, per-model rotated asset namespace, central run-status rebuild, and model-parallel full runner. |
| Full closed-set S2 12-angle benchmark | complete | 10/10 selected closed-set models reached DONE_FULL on S2 final-test, 12 angles, with 300000 raw and 300000 canonical prediction JSONs. |
| Postprocess full closed-set metrics/report | complete | False-hub taxonomy, stage decomposition, full summary, and scientific audit artifacts were generated; validated audit artifacts live under `/data/zcy/OpenRSD_artifacts/closedset_scientific_audit_20260531`. |

## Full Open-Vocab/Causal/Context/DeHub Continuation (2026-05-31)

| Phase | Status | Notes |
|---|---|---|
| Reconcile existing state and storage | in_progress | `/data1` has about 286G free and `/data` has about 272G free. The previous full closed-set run still occupies about 204G after rotated images were removed/indexed. |
| Audit current 07-10 script readiness | complete | `07_intervention_open_vocab.py` is capability/status-only; `08/09/10` can run real smoke-style paths but still mark results as smoke and depend on persisted rotated image paths. |
| Define full output roots on `/data` | complete | Full outputs now use `/data/zcy/OpenRSD_runs/rotation_semantic_attractor/...`; project files on `/data1` hold only scripts/planning records. |
| Implement full open-vocab S2 12-angle runner | complete | Added `21_run_full_openrsd_streaming.py`; it streams rotated assets through scratch chunks and retains CSV/JSON metrics plus transient asset indexes only. |
| Implement full causal intervention runner | complete | The same streaming runner executes OpenRSD visual-support interventions with paired baseline/intervention rows and full-vs-limit status gating. |
| Implement full context counterfactual runner | complete | The runner regenerates rotated assets, creates `object_only`/`context_only` images per chunk, runs inference, then deletes scratch images. |
| Implement full DeHub safety runner | complete | The runner evaluates baseline and repair checkpoints on the frozen S3 split and writes DeHub safety/class drift/low-risk/true-SV summary files. |
| Launch resumable background tasks | in_progress | Four tmux sessions are running: `rsa_full_openvocab_s2` on GPU5, `rsa_full_causal_s3` on GPU6, `rsa_full_context_s3` on GPU7, and `rsa_full_dehub_s3` on GPU8. Status is summarized by `22_check_full_openrsd_status.py`. |

## FOCUS-OVD Prototype (2026-06-08)

Goal: implement a default-off Fourier Orientation-Conditioned Residual Support Adapter for OpenRSD, preserving native DINOv2/A10 support geometry and providing smoke/report scaffolding without running full benchmarks.

| Phase | Status | Notes |
|---|---|---|
| Preflight inventory | complete | Wrote `resultmd/exp_focus_ovd_20260608/preflight_focus_inventory.md` and `.json`. |
| Fourier orientation learner tests | complete | Added synthetic direction, low-texture confidence, finite-output, and Fourier-code tests. |
| Residual support adapter tests | complete | Added zero-init equivalence, class gating, runtime class-name override, confidence gating, delta clipping, and geometry diagnostics tests. |
| FOCUS modules | complete | Added Fourier orientation learner, support residual adapter, conditioned contrastive embed, and attractor loss helpers. |
| OpenRSD integration | complete | Added config-gated dense-head path with default `use_focus_ovd=False`; zero residual falls back to native contrastive matmul. |
| Scripts and reports | complete | Added 29-35 FOCUS scripts and generated audit, equivalence, orientation, train/eval, ablation, and method-report artifacts. |
| Verification | complete | `py_compile` passed for modified modules/scripts; focused pytest reports 14 passed. No full benchmark/training was run. |

## FOCUS-OVD Scheduled Training (2026-06-08)

Goal: schedule a safe GPU6/9 FOCUS adapter-only full train after `2026-06-09 03:00:00 Asia/Shanghai`, with smoke-first execution and fatal checkpoint/config preflight.

| Phase | Status | Notes |
|---|---|---|
| Add adapter-only enforcement | complete | `MetaRemoveRunner` now supports pre-optimizer `trainable_parameters`; this keeps the optimizer scoped to `bbox_head.focus_support_adapter.*`. |
| Add audit hook | complete | `FocusOVDTrainableAuditHook` fails fast if unexpected parameters are trainable or no adapter parameter is trainable. |
| Add full/smoke configs | complete | Full config inherits A10 formal and enables FOCUS for `small-vehicle`; smoke uses the same path with `max_iter_per_epoch=20`. |
| Add 03:00 watcher | complete | `focus_ovd_wait_and_launch_gpu69.py` waits, polls GPU 6/9, runs preflight, smoke, then full tmux launch. |
| Immediate verification | complete | `py_compile` passed; config dry-run passed; focused pytest reports 16 passed; checkpoint preflight is OK. |
| Launch watcher | complete | tmux session `focus_ovd_wait_gpu69_20260609` is sleeping until 03:00 and will then wait for both GPUs to become idle. |

## FOCUS-OVD Module Attribution (2026-06-09)

Goal: implement the paper-grade module attribution scaffold for the FOCUS-OVD method line, with strict default-off module switches, no DeCLIP support replacement as a main path, and no full benchmark or full training.

| Phase | Status | Notes |
|---|---|---|
| Read request, skills, and current repo state | complete | Attached brief is treated as the approved experiment spec; worklog start entry is recorded. |
| Create dated experiment result tree and manifest | complete | Created `resultmd/exp_focus_ovd_module_attribution_20260609/` with required subdirectories and `manifest.json`. |
| Inventory current FOCUS implementation | complete | Added and ran `36_focus_module_inventory.py`; all required guardrail checks now report true. |
| Generate attribution matrix | complete | Added and ran `37_focus_design_module_attribution_matrix.py`; generated 36 variants V00-V63. |
| Define effectiveness metrics | complete | Added and ran `38_focus_define_effectiveness_metrics.py`; generated 34 metric definitions and decision rules. |
| Implement controlled diagnostic modules | complete | Added default-off utilities and audit scripts for cluster mining, attribute gate, channel mask, safety gate, orbit teacher, head consensus, negative prompt, and DINO CCL control. |
| Verify new files | complete | `py_compile` passed for touched files; focused attribution utility pytest reports 7 passed. |
| Run P0 attribution planner | complete | P0 runner wrote 9 rows; V01 exact baseline-equivalence passed and V10 orientation probe was written; train/eval variants were not run. |
| Build interim attribution report | complete | Generated report, tables, and figures without AP/safety improvement claims. |
| Ingest full effect judgment | complete | Parsed the completed DOTA2-only recovery full run and baseline CSVs; V11 is AP-positive but fails the true-SV recall safety gate. |

## FOCUS-OVD P0 Small Train/Eval (2026-06-09)

Goal: move P0 core FOCUS variants from planned/not-evaluated into a bounded verified-crop-label proxy small-train/eval pass, without full benchmark, full detector retraining, DeCLIP support replacement, text-prompt main path, annotation-missing true-vehicle negatives, or degenerate/padding corrected-FSV loss.

| Phase | Status | Notes |
|---|---|---|
| P0 preflight | complete | `49_focus_p0_preflight.py` passed after using the OpenRSD Python environment and recursive base-config freeze checks. |
| Orientation confidence calibration | complete | `50_focus_orientation_confidence_calibration.py` found high confidence on padding/degenerate rows, so orientation confidence is not a sufficient safety gate. |
| P0 split build | complete | `51_focus_build_p0_splits.py` produced train/eval/safety splits with no leakage: train 316, eval 329, safety 329. |
| Variant config generation | complete | `52_focus_generate_p0_configs.py` generated V00/V01/V02/V03/V10/V11/V20/V21/V22/V23/V24 configs. |
| Control evaluation | complete | V00/V01/V02/V03/V10 were evaluated before training variants; V02/V03 are negative controls. |
| Proxy small-train | complete | `53_focus_run_p0_training.py` wrote proxy adapter-train records for V11/V20/V21/V22/V23/V24 with base model frozen. |
| Full P0 proxy eval and verdict | complete | `54`/`55`/`56` produced eval, safety, and report artifacts under `resultmd/exp_focus_ovd_p0_train_eval_20260609/`. |
| Verification | complete | `py_compile` passed for scripts 49-56 plus common helper; focused pytest reports 10 passed. |

## FOCUS-OVD P1 Detector-Level Validation (2026-06-09)

Goal: convert P0 proxy evidence into detector-level evidence only if real detector loss wiring and corrected-FSV label-to-loss mapping can be proven. Stop before training if either cannot be proven.

| Phase | Status | Notes |
|---|---|---|
| Read P1 request and start worklog | complete | User explicitly required P1 preflight first and stop if loss wiring cannot be proven. |
| Audit existing detector loss path | complete | Dense head exposes FOCUS inference/support adapter path and training logits/support inputs, but no `bbox_head.focus_losses` detector loss branch or `loss_focus_*` logging exists. |
| Audit corrected-FSV label mapping | complete | P0 splits contain crop/tile/angle metadata only; no `prediction_id`/`train_sample_id`/feature-level/grid fields or explicit mapper into detector training loss masks exists. |
| Add P1 preflight script | complete | Added `57_focus_p1_preflight.py`, creating the P1 experiment tree, manifest, and preflight JSON/Markdown. |
| Run P1 preflight | blocked | Status is `BLOCKED_LABEL_TO_LOSS_MAPPING`; blocking checks are `detector_focus_loss_path_wired` and `label_to_loss_mapping`. |
| Stop detector training/eval | complete | Scripts 58-63 and P1 detector training/eval were intentionally not run because preflight failed the stop rule. |
| Verification | complete | `py_compile` passed for script 57; focused pytest reports 12 passed across P1/P0/attribution tests. |

## FOCUS-OVD P1 Unblock Loss Mapping (2026-06-09)

Goal: unblock the P1 detector-loss gate by wiring real focus loss keys, converting verified crop labels into spatial detector-region targets, and explicitly separating P1A spatial-region loss evidence from P1B exact pre-NMS provenance.

| Phase | Status | Notes |
|---|---|---|
| Add TDD coverage | complete | Added tests for focus loss config validation, spatial target assignment, and exact provenance matching; RED run failed before helper/script implementation. |
| Implement focus loss target helpers | complete | Added `focus_loss_targets.py` and `focus_spatial_target_assigner.py`; tests now cover mapping fields, polygon-to-feature assignment, and preserve-over-anti conflict resolution. |
| Wire detector focus loss branch | complete | `Flex_Rrtmdet_head_v3_1.py` now accepts `focus_losses`, records per-level focus debug, and emits `loss_focus_support_distill`, `loss_focus_anti`, `loss_focus_preserve`, `loss_focus_migration`, and `loss_focus_total` when enabled. |
| Build P1A spatial-region targets | complete | Script 65 produced 316 train-split rows: 108 valid anti negatives and 207 valid preserve positives under `resultmd/exp_focus_ovd_p1_unblock_loss_mapping_20260609/targets/`. |
| Diagnose P1B exact provenance | blocked | Scripts 66/67 show existing metadata is final-stage only; exact pre-NMS coverage is 0.0, so P1B remains blocked. |
| One-batch loss smoke | complete | Script 68 passed synthetic detector-head tensor backward: finite losses, adapter grad flow true, frozen grad false, assigned points 128. |
| Unblock report and P1 preflight refresh | complete | Script 69 reports `PASS_P1A_SPATIAL_REGION_LOSS_PARTIAL_UNBLOCK`; updated script 57 now returns `PASS_P1A_SPATIAL_REGION_LOSS` when reading `--unblock-exp`. |
| Detector training continuation | blocked | `can_continue_to_detector_training=false` because real batch target-mask injection is not proven; no full detector train/AP/safety/DeCLIP result is claimed. |
| Verification | complete | Focused pytest reports 7 passed; `py_compile` passed for dense head, helpers, losses, scripts 57 and 64-69. |

## FOCUS-EQText Detector-Level Short Run (2026-06-10)

Goal: implement and run the minimal FOCUS-EQText DOTA detector-level experiment on physical GPUs 6 and 9, using the passed P1A spatial pseudo-region realbatch loss path and exactly the requested five core variants.

| Phase | Status | Notes |
|---|---|---|
| Read current FOCUS/P1A state | complete | P1A realbatch path exists in scripts 70-74; P1B exact provenance remains blocked and is not claimed. |
| Add module tests | complete | Covered EQText zero-init, alpha-zero gradient reachability, dual zero baseline equivalence, broadcast fusion, loss behavior, and negative bank guardrails. |
| Implement EQText modules | complete | Added text residual adapter, dual fusion, losses, auxiliary negative text bank, and detector/dense-head support routing without DeCLIP or support-bank replacement. |
| Add scripts 93-97 | complete | Added preflight, config generation, short train, short eval, safety/report builder, and generated a P2 dual-GPU launcher. |
| Verify and run | complete | py_compile passed; focused pytest reports 10 passed; preflight/configs/train/eval/report all passed. V10/V20/V30 each ran 1000 P1A realbatch steps. |

## FOCUS-EQText Constraints

- Use only physical GPUs 6 and 9 for detector-level train/eval commands.
- Keep native OpenRSD DINO/A10 support; do not enable DeCLIP support or replace support bank.
- Freeze text encoder and base model; train only focus head residual adapter, focus text residual adapter, dual support fusion gate, and alpha parameters.
- Use `target_mapping_mode=spatial_region` and the passed P1A realbatch loss path.
- Never use annotation-missing true vehicles as negatives.
- Exclude `degenerate_large_sv_box` and `padding_artifact` from corrected-FSV loss.
- Run only EQ_V00, EQ_V01, EQ_V10, EQ_V20, and EQ_V30 unless explicitly extended.

## EQText Strict Epoch2 Audit (2026-06-10)

Goal: stop long EQ_V33 continuation and build a strict, layered, epoch2-direct-val validation scaffold for TEXT-side changes only.

| Phase | Status | Notes |
|---|---|---|
| Stop EQ_V33 long training | complete | User requested stop; GPU 6/9 V33 processes were terminated after epoch2 val and epoch3 partial progress. |
| Create strict audit result tree | complete | Root is `resultmd/exp_eqtext_strict_epoch2_audit_20260610/` with preflight/configs/train/eval/tables/figures/reports/logs. |
| Add V33 epoch2 failure audit | complete | Script 104 parsed V33 epoch2 and wrote `FAILED_TEXT_CALIBRATION_EPOCH2` with stop recommendation. |
| Generate strict TEXT-side configs | complete | Script 105 generated V00/V01/V10/V20/V21/V30/V31/V32/V40 from the FOCUS-OVD positive config path, not EQ_V30/V33. |
| Add epoch2 direct-val runner | complete | Script 106 defaults to dry-run, enforces max_epochs=2/val_interval=2 safety scanning, and only runs training with `--execute`. |
| Verification | complete | `py_compile` passed; V33 audit, config generation, and runner dry-run completed without launching training. |

Constraints:

- Do not resume EQ_V33 long training.
- Do not run long training.
- Any TEXT-side training command must be two epochs with epoch2 validation and kill-rule parsing.

## FOCUS-OVD Dual Text GPU67 Best-Ckpt Run (2026-06-11)

Goal: run the text-side dual-branch FOCUS/Mess-Fourier experiment on physical GPUs 6 and 7, keep evaluation fixed to the best checkpoint, and use the run to debug the text-fusion route.

| Phase | Status | Notes |
|---|---|---|
| Recover current dual-text context | complete | Existing 6GPU/autotune runs and watchdog artifacts were found; latest r2 six-card run had stopped after SIGHUP around epoch 15. |
| Build GPU67/best-checkpoint launch path | complete | Added a two-card config plus train-then-best-eval launcher using `save_best='dota/mAP'` and physical GPU 6,7 defaults. |
| Smoke-check config and training path | complete | Focused dual-text pytest passed; shell launcher syntax passed; GPU 6/7 were available before launch. |
| Launch and monitor | in_progress | tmux session `focus_mess_gpu67_best_20260611` is running. Log: `work_dirs/train_queue_logs/focus_mess_fourier_gpu67_best_20260611.log`; work_dir: `work_dirs/focus_ovd_a10_mess_fourier_dual_text_gpu67_best_20260611`. |
| Record outcome | complete | First train log reached epoch 1 iter 50/400 with no immediate OOM/import failure; best-checkpoint eval is wired to `best_*.pth` after training completes. |

## Past Two Weeks Experiment Summary (2026-06-10)

Goal: summarize all traceable experiment work from `2026-05-27` through `2026-06-10` into one Markdown report, with a numeric table for each experiment.

| Phase | Status | Notes |
|---|---|---|
| Define date window and output path | complete | Date window is inclusive: `2026-05-27` to `2026-06-10`; target report is under `resultmd/exp_past_two_weeks_experiments_20260610/`. |
| Inventory recent experiment directories | complete | Scanned `resultmd/exp_*`, `experiments/rotation_semantic_attractor`, and relevant `work_dirs`/logs for dated artifacts. |
| Extract numeric evidence | complete | Used existing CSV/JSON/Markdown tables; blocked/proxy/dry-run evidence is labeled explicitly. |
| Write consolidated report | complete | Report contains 27 sections and 33 Markdown tables, including a numeric table in every section. |
| Verify and finish worklog | complete | Confirmed the report exists and references source artifacts; finish worklog entry recorded. |

## Experiment ABCD Audit (2026-06-10)

Goal: scan current OpenRSD experiment records and outputs, then classify experiments into A/B/C/D:
A = recent and very useful; B = recent and generally useful; C = at least last-month/older or low-significance; D = failed dirty files or meaningless experiments that can be proposed for deletion.

| Phase | Status | Notes |
|---|---|---|
| Load workflow rules and existing records | complete | Skills loaded; current root planning files already summarize major experiments from 2026-05-30 through 2026-06-10. |
| Inventory experiment roots | complete | Lightweight inventory wrote 1141 roots to `resultmd/exp_experiment_abcd_audit_20260610/tables/experiment_inventory_lite.csv`; detailed JSON saved beside it. |
| Extract evidence | complete | Evidence came from existing planning files plus key reports for FOCUS-EQText, FOCUS-OVD P0/P1/P1A, module attribution, DeCLIP+CCL, EQText strict audit, FOCUS-T-Safe, FOCUS-TAC, and rotation semantic full closed-set. |
| Assign ABCD grades | complete | Manual classification table written to `resultmd/exp_experiment_abcd_audit_20260610/tables/manual_abcd_classification.csv`. |
| Write final audit report | complete | Final report written to `resultmd/exp_experiment_abcd_audit_20260610/reports/fres_abcd_experiment_audit_20260610.md`; no deletion was performed. |
- Keep `support_type='text'`, `with_aux_bbox_head=True`, `use_declip_support=False`.
- Do not replace the native OpenRSD support bank.
- Do not enable anti-attractor loss without verified DOTA2 hard-negative targets.
- Do not use DOTA1 target CSVs.
- Treat direction hard prompt as a negative control only.

## FOCUS-T-Safe Conservative Text Shadow (2026-06-10)

Goal: implement the most conservative text-side Fourier path on top of the positive FOCUS-OVD line: shadow text diagnostics first, optional loss-only smoke, and offline one-way calibration only if shadow signal passes.

| Phase | Status | Notes |
|---|---|---|
| Read brief and reconcile running state | complete | Attachment requires no direct text/logit fusion, no V33 continuation, no DeCLIP support, and no anti loss without verified DOTA2 hard negatives. A prior FOCUS-TAC eval is still running; T-Safe did not start GPU training. |
| Add T-Safe module tests | complete | Added `tests/test_focus_tsafe_modules.py` and verified RED failures before implementation. |
| Implement T-Safe modules | complete | Added `focus_tsafe_text_adapter.py`, `focus_tsafe_negative_bank.py`, and `focus_tsafe_text_losses.py`. |
| Add scripts 113-118 | complete | Implemented zero-disturbance report, shadow diagnostic, shadow analysis, loss-only smoke, offline tiny calibration, and final report builder. |
| Run non-training verification | complete | `py_compile` passed; focused pytest reports 6 passed; CPU/offline scripts generated reports. No long training or final-logit text fusion was launched. |

Constraints:

- FOCUS-OVD remains the main method.
- Keep `support_type='text'`, `with_aux_bbox_head=True`, and DeCLIP support disabled.
- Do not replace the native support bank.
- Do not directly fuse text into final logits or NMS.
- Do not train the full model.
- Do not use DOTA1 hard negatives for DOTA2.
- Do not enable anti-attractor without verified DOTA2 hard negatives.
- Direction hard prompt only remains a negative control.
