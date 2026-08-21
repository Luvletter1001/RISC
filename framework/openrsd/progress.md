# Rotation Semantic Attractor Experiment Deployment Progress

## 2026-05-30

- Loaded required workflow skills: planning with files, TDD, A40 distributed GPU guidance, verification-before-completion, and OpenRSD worklog.
- Logged task start in `CODEX_WORKLOG.md`.
- Replaced stale planning files from the previous BS96 visualization task with the current experiment deployment plan.
- Inspected DOTA1/DOTA2 angle sweep data roots and closed-set weights.
- Verified GPUs 6,7,8,9 are idle via `nvidia-smi`.
- Observed PyTorch CUDA was unavailable in the initial sandboxed OpenRSD Python probe despite visible GPUs; the decisive outside-sandbox diagnostic and smoke run later saw 4 CUDA devices.
- Started a direct mmrotate API probe with rotated RetinaNet on CPU to determine whether a real adapter can run in this environment.
- Confirmed direct mmrotate API inference works on CPU when `PYTHONNOUSERSITE=1` is set.
- Added rotation geometry tests, observed expected import failure before implementation, implemented `src/rotation.py`, and reran the tests successfully with system pytest and disabled project addopts.
- Added the full `experiments/rotation_semantic_attractor/` scaffold: configs, DOTA split builder, rotation/GT utilities, mmrotate model adapter, false-hub metrics, stage/intervention/counterfactual/safety scripts, visualization helpers, smoke runner, and Markdown report builder.
- Registered at least ten closed-set methods for the full experiment. Available checkpoints include rotated RetinaNet, rotated RTMDet-L, H2RBox, H2RBox-v2, ReDet, Oriented R-CNN, Oriented RepPoints, R3Det-KFIoU, Strip R-CNN, and RetinaNet variants; missing-checkpoint entries are marked `NOT_RUN`.
- Ran compile verification with `rtk python3 -m compileall -q experiments/rotation_semantic_attractor`.
- Ran the decisive smoke test with `PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=6,7,8,9` and `RUN_DIR=.../smoke_2tiles_2angles_gpu`. The smoke verdict was `PASS_WITH_UNSUPPORTED_HOOKS`, with 4 prediction files from 2 tiles x 2 angles.
- Rebuilt the report at `experiments/rotation_semantic_attractor/reports/rotation_semantic_attractor_report.md` from the GPU smoke run.
- Audited key limitations: sandbox CUDA visibility required escalation; OpenRSD conda lacks pytest; project pytest addopts require an unavailable xdoctest plugin; mmrotate environment emits a requests character-detection warning; pre-NMS/dense/query hooks are unsupported; smoke intervention/counterfactual/safety scripts intentionally emit `NOT_RUN`.
- Final model registry audit found 15 method entries, 11 with checkpoint paths and 4 missing-checkpoint methods that remain `NOT_RUN` candidates: RoI Transformer, S2ANet, Rotated FCOS, and Rotated ATSS.
- Repair pass: added checkpoint URLs and local targets for RoI Transformer, S2ANet, Rotated FCOS, and Rotated ATSS, then downloaded all four weights into `weights/`.
- Repair pass: added tests for checkpoint URLs, stage metric partial support, and proxy experiment outputs; final experiment test suite reports 7 passed.
- Repair pass: implemented adapter-side pre-NMS/dense probing for supported single-stage mmrotate heads. The GPU probe and final smoke show `pre_nms_predictions` plus `dense_logits` summaries, with only closed-set query logits remaining unsupported.
- Repair pass: replaced smoke `NOT_RUN` placeholders for closed-set intervention, context counterfactual, and DeHub safety with executable proxy outputs labeled `PROXY` or `SMOKE_PROXY`.
- Final repair smoke run used `CUDA_VISIBLE_DEVICES=6,7,8,9` and wrote `PASS_WITH_PARTIAL_HOOKS` to `experiments/rotation_semantic_attractor/outputs/runs/smoke_2tiles_2angles_gpu_hooks/smoke_verdict.txt`.
- Status semantics repair: wrote `experiments/rotation_semantic_attractor/reports/status_semantics_audit.md` before changing code.
- Status semantics repair: added `src/utils/status.py`, `src/model_adapters/capabilities.py`, capability matrix generation, matrix smoke orchestration, and report filtering.
- Status semantics repair: split smoke entry points into closed-set, open-vocabulary, and matrix runners; `run_smoke_test.sh` now delegates to the matrix runner.
- Status semantics repair: closed-set stage decomposition now reports query logits as `NOT_APPLICABLE` and renders the required phrase `query logits: N/A by architecture`.
- Status semantics repair: open-vocabulary intervention is `NOT_SELECTED_IN_THIS_SMOKE` in closed-set-only smoke and `NOT_AVAILABLE_ASSET` in open-vocabulary smoke because no runnable open-vocabulary asset is present.
- Status semantics repair: closed-set intervention/context rows are `SMOKE_PROXY`; DeHub safety rows are `SCHEMA_ONLY`; proxy/schema rows are excluded from scientific and repair tables.
- Verification after status repair: `rtk python3 -m pytest -o addopts='' experiments/rotation_semantic_attractor/tests/test_rotation_geometry.py -q` passed 3 tests.
- Verification after status repair: `rtk python3 -m pytest -o addopts='' experiments/rotation_semantic_attractor/tests/test_status_semantics.py -q` passed 3 tests.
- Verification after status repair: `rtk python3 -m pytest -o addopts='' experiments/rotation_semantic_attractor/tests/test_report_status_filtering.py -q` passed 3 tests.
- Verification after status repair: `rtk python3 -m pytest -o addopts='' experiments/rotation_semantic_attractor/tests -q` passed 13 tests.
- Smoke after status repair: closed-set smoke on `CUDA_VISIBLE_DEVICES=6,7,8,9` returned `PASS_WITH_NOT_APPLICABLE`.
- Smoke after status repair: open-vocabulary smoke returned `PASS_SCHEMA_ONLY` with `openvocab_status=NOT_AVAILABLE_ASSET`.
- Smoke after status repair: matrix smoke on `CUDA_VISIBLE_DEVICES=6,7,8,9` returned `PASS_WITH_NOT_APPLICABLE`.
- Final report count fix: executive status counts now include capability/stage status fields, so closed-set query-logit architecture N/A contributes to `not_applicable_count` instead of being hidden by smoke-row-only counting.
- Final verification: regenerated matrix smoke report after the count fix; report shows `verdict=PASS_WITH_NOT_APPLICABLE`, `not_applicable_count=16`, `failure_count=0`, and `unsupported_count=0`.
- Scientific buildout Step 1: generated `experiments/rotation_semantic_attractor/reports/scientific_gap_audit.md` and `.json` before changing implementation code.
- Scientific gap audit finding: closed-set registry has 15 checkpoint-backed models, but only `rotated_retinanet_msrr` has current DONE_SMOKE outputs and no model has DONE_FULL outputs.
- Scientific gap audit finding: OpenRSD-style configs/checkpoints/support files exist as local candidates, but the experiment framework has no registered open-vocab model registry or real OpenRSD embedding hook adapter yet.
- Scientific gap audit finding: DeHub baseline and repair checkpoint candidates exist, but the current framework has not run same-split/same-angle baseline-vs-repair inference, so DeHub remains SCHEMA_ONLY.
- Scientific buildout Step 2 TDD RED: new DONE_FULL gating tests initially failed because `can_be_done_full` did not exist.
- Scientific buildout Step 2 GREEN: implemented `can_be_done_full()` in `src/utils/status.py` and wired `build_report_helpers.scientific_rows()` through the gate.
- Scientific buildout Step 2 verification: `test_done_full_gating.py`, `test_real_intervention_required.py`, `test_counterfactual_real_image_required.py`, `test_dehub_safety_gating.py`, `test_status_semantics.py`, and `test_report_status_filtering.py` all passed individually.
- Scientific buildout Step 3: added OpenRSD/open-vocabulary asset inventory code and reports. Final narrowed inventory reports 95 relevant assets, 5 runnable OpenRSD pairs, and no missing config/checkpoint/support/hook categories.
- Scientific buildout Step 4: added real OpenRSD adapter/hook infrastructure and `15_openrsd_hook_smoke.py`. GPU smoke captured OpenRSD hook tensors on `cuda:0` and wrote `DONE_SMOKE` OpenRSD intervention rows.
- Scientific buildout Step 5: added real closed-set classifier-channel intervention. The RetinaNet small-vehicle classifier channel is modified at weight level and inference is rerun on the smoke image.
- Scientific buildout Step 6: added real image-level context counterfactuals. The smoke now writes object-only/context-only edited images and reruns detector inference on them.
- Scientific buildout Step 7: added real DeHub safety smoke. Baseline and repair checkpoints run on the same smoke examples and record class drift plus low-risk inflation metrics.
- Scientific buildout Step 8: added oracle best-view output and a full closed-set benchmark command planner for S2, 12 angles, and at least 10 models.
- Open-vocabulary smoke integration: `run_smoke_openvocab.sh` now invokes asset inventory and real OpenRSD hook smoke instead of only preparing a schema-only manifest.
- Report refresh: rebuilt `smoke_closedset_report.md` and `smoke_matrix_report.md`; latest matrix report shows closed-set `DONE_SMOKE`, open-vocab `DONE_SMOKE`, no proxy/schema validation rows, and no `DONE_FULL` scientific rows.
- Final semantics adjustment: regenerated stage decomposition so closed-set query logits remain `NOT_APPLICABLE` when old raw prediction JSON lacks capability metadata. Added regression coverage in `test_status_semantics.py`.
- DONE_FULL readiness pass: wrote `experiments/rotation_semantic_attractor/reports/full_execution_readiness_audit.md` and `.json`.
- Split freeze pass: wrote `experiments/rotation_semantic_attractor/reports/full_split_freeze_report.md`; S2 final-test is reused, and S3 safety stratified now contains false-SV-hub, true-SV-rich, low-risk normal, and cross-class conflict strata.
- Full-run execution repair: added `17_run_full_closedset_parallel.py`, `18_update_run_status.py`, per-model manifests, and per-model rotated asset namespacing so full closed-set inference can run as resumable model shards.
- Verification after full-run execution repair: `rtk /data/zcy/anaconda3/envs/openrsd/bin/python -m py_compile ...` passed for the modified execution scripts; `rtk python3 -m pytest -o addopts='' experiments/rotation_semantic_attractor/tests -q` reports 38 passed.
- Replaced the slow single-GPU `rsa_full_closedset` tmux run with `rsa_full_closedset_parallel`; output target remains `experiments/rotation_semantic_attractor/outputs/runs/full_closedset_s2_12angle`.
- Current full closed-set benchmark status at 2026-05-30 18:52 CST: 6 model shards are RUNNING on GPUs 0,1,2,3,6,7; later models remain NOT_RUN until a GPU frees; no model is DONE_FULL yet.

## 2026-05-31

- Full closed-set S2 12-angle benchmark completed after the previous overnight run and repair pass: 10/10 selected closed-set models reached DONE_FULL, with 300000 raw prediction JSONs and 300000 canonical prediction JSONs.
- Generated the full closed-set summary report and then a separate scientific audit/table artifact bundle. The validated audit bundle is stored at `/data/zcy/OpenRSD_artifacts/closedset_scientific_audit_20260531`.
- Replaced the missing/deleted full closed-set `rotated_images` tree with an expected-path index only at `/data/zcy/OpenRSD_artifacts/closedset_rotated_images_index_20260531/rotated_images_index.tsv`; image bytes are not preserved.
- Indexed existing full closed-set raw and canonical prediction JSONs without deleting them. Indexes are under `/data/zcy/OpenRSD_artifacts/closedset_prediction_indexes_20260531`.
- Storage status after cleanup/indexing: `/data1` has about 286G free, `/data` has about 272G free, and `experiments/rotation_semantic_attractor/outputs/runs/full_closedset_s2_12angle` still occupies about 204G.
- User requested continuation of full `open-vocab`, `causal`, `context counterfactual`, and `DeHub safety` work, explicitly using `/data` where space is available.
- Script audit found that `07_intervention_open_vocab.py` is capability/status-only and cannot produce full open-vocab causal results as-is.
- Script audit found that `08_intervention_closed_set.py`, `09_context_counterfactual.py`, and `10_eval_dehub_safety.py` have real smoke execution paths but still label outputs as smoke and depend on persisted rotated image paths, so they need full-level wrappers/repairs before claiming benchmark-level results.
- Added `experiments/rotation_semantic_attractor/scripts/21_run_full_openrsd_streaming.py`, a chunked OpenRSD full runner that regenerates rotated images/annfiles under `/data` scratch, runs inference/interventions/counterfactuals, writes CSV/JSON metrics and transient asset indexes, then deletes scratch image bytes.
- Added `experiments/rotation_semantic_attractor/scripts/22_check_full_openrsd_status.py` to summarize the four full OpenRSD continuation runs into `/data/zcy/OpenRSD_runs/rotation_semantic_attractor/full_openrsd_status.csv` and `.json`.
- Validation runs passed:
  - `validate_open_vocab_causal_limit1`: 1 S3 tile-angle, 6 visual-support interventions, 6 `DONE_SMOKE` rows.
  - `validate_context_limit1`: 1 true-SV-rich S3 tile-angle, `object_only` and `context_only` real image edits, 2 `DONE_SMOKE` rows.
  - `validate_dehub_limit1`: 1 S3 tile-angle, baseline plus repair checkpoint, 2 `DONE_SMOKE` rows and DeHub summary CSVs.
- Launched four full background tmux sessions at 2026-05-31 17:35 CST:
  - `rsa_full_openvocab_s2`: S2 final-test 2500 tiles x 12 angles, output `/data/zcy/OpenRSD_runs/rotation_semantic_attractor/full_openvocab_s2_12angle`, GPU5.
  - `rsa_full_causal_s3`: S3 safety 500 tiles x 12 angles x 6 visual-support interventions, output `/data/zcy/OpenRSD_runs/rotation_semantic_attractor/full_causal_intervention_s3_12angle`, GPU6.
  - `rsa_full_context_s3`: S3 safety 500 tiles x 12 angles with object/context counterfactuals where target SV GT exists, output `/data/zcy/OpenRSD_runs/rotation_semantic_attractor/full_context_counterfactual_s3_12angle`, GPU7.
  - `rsa_full_dehub_s3`: S3 safety 500 tiles x 12 angles for baseline and repair OpenRSD checkpoints, output `/data/zcy/OpenRSD_runs/rotation_semantic_attractor/full_dehub_safety_s3_12angle`, GPU8.
- Initial full-run status check after launch: open-vocab 204/30000 rows, causal 7116/36000 rows, context 1828/12000 max rows, DeHub 1560/12000 rows; all active logs show `failures=0`.

## 2026-06-08 FOCUS-OVD Prototype

- User requested a minimal but complete FOCUS-OVD code loop: Fourier orientation learner, zero-init residual support adapter, orientation-conditioned contrastive logits, corrected-FSV aware losses/loaders, smoke scripts, ablation command plan, and report builder.
- Constraints: default `use_focus_ovd=False`; no DeCLIP support replacement; no direct direction text prompt as main path; no box regression/NMS/postprocess changes; alpha/residual zero init must be baseline-equivalent.
- Started OpenRSD worklog entry for this task and created `resultmd/exp_focus_ovd_20260608/` for report artifacts.
- Added `M_AD/models/utils/focus_fourier_orientation.py`, `focus_support_adapter.py`, `focus_contrastive_embed.py`, and `M_AD/models/losses/focus_attractor_losses.py`.
- Integrated FOCUS into `M_AD/models/dense_heads/Flex_Rrtmdet_head_v3_1.py` behind default-off `use_focus_ovd` / `focus_ovd` config gates; detector paths now pass ordered support class names for small-vehicle-safe masking.
- Added FOCUS scripts `29_prepare_focus_audit_labels.py` through `35_build_focus_report.py` and generated artifacts under `resultmd/exp_focus_ovd_20260608/`.
- Audit label preparation produced 135 `corrected_false_sv` hard negatives, 160 `annotation_missing_true_vehicle` preserve positives, 5 ambiguous/invalid rows, 200 reference-only rows, and 250 excluded failure-mode rows.
- Synthetic baseline-equivalence smoke passed with `max_abs_diff=0.0` and zero adapter delta; orientation probe wrote mean periodic error about 6.22 degrees with confidence range 0.763-0.972.
- Verification: OpenRSD Python `py_compile` passed for all modified/new FOCUS modules, dense head, detector, and scripts; focused pytest reports 14 passed.

## 2026-06-08 FOCUS-OVD Scheduled Training

- Implemented adapter-only scheduled full training for FOCUS-OVD on GPU 6/9.
- Added pre-optimizer `trainable_parameters` support to `MetaRemoveRunner`, because ordinary hooks run after optimizer construction and cannot reliably enforce adapter-only optimizer params.
- Added `FocusOVDTrainableAuditHook` to fail fast if any non-`bbox_head.focus_support_adapter.*` parameter remains trainable.
- Added full and smoke configs under `M_configs/experiments/focus_ovd/`, using A10 formal as base and A10 epoch24 checkpoint as fixed initialization.
- Added `M_Tools/experiments/focus_ovd_wait_and_launch_gpu69.py`, which sleeps until `2026-06-09 03:00:00 Asia/Shanghai`, polls GPU 6/9, runs smoke, then launches full 24-epoch adapter-only training in tmux.
- Immediate preflight passed: `/data1/zcy/OpenRSD/results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth` exists and is about 550M.
- Started watcher tmux session `focus_ovd_wait_gpu69_20260609`; it is currently sleeping until 03:00. GPU 6/9 were busy at launch time, so the watcher must wait after waking if they remain above thresholds.
- Verification: `py_compile` passed; config dry-run passed; focused pytest reports 16 passed.

## 2026-06-09 FOCUS-OVD Module Attribution

- Loaded required workflow skills for this pass: using-superpowers, OpenRSD worklog, planning-with-files, experiment-result Markdown organization, TDD, and verification-before-completion.
- Read the attached module attribution brief and recorded a worklog start entry in `CODEX_WORKLOG.md`.
- Re-read existing root planning files and preserved prior rotation/FOCUS records before adding the new attribution section.
- Observed the repo has a very dirty worktree with many unrelated changes; this pass will avoid reverting or touching unrelated files.
- Added focused attribution utility tests and verified the expected red failure before implementation (`focus_head_consensus.py` missing).
- Added default-off utility modules for spurious cluster mining, visual attribute gate, sensitivity channel mask, safety gate, orbit teacher, head consensus, and negative prompt auxiliary margins.
- Added FOCUS module attribution scripts `36` through `48`, plus shared `focus_attribution_common.py`.
- Generated `resultmd/exp_focus_ovd_module_attribution_20260609/` artifacts: module inventory, ablation matrix, metric definitions, cluster/attribute/channel/safety/orbit/head/negative/CCL outputs, P0 eval table, figures, and report.
- Fixed the strict baseline-equivalence smoke after seed 20260609 exposed a `4.768e-07` accumulation-order difference in no-delta FOCUS inference; rerun now reports `PASS`, `max_abs_diff=0.0`.
- P0 runner result: V01 passed baseline-equivalence smoke, V10 wrote orientation probe, V00 baseline was not run, and V11/V20/V21/V22/V23/V24 are `NOT_EVALUATED_REQUIRES_TRAINING`.
- Verification: py_compile passed for all touched attribution utilities/scripts and modified FOCUS files; `rtk env PYTHONNOUSERSITE=1 python3 -m pytest -o addopts='' tests/test_focus_attribution_utils.py -q` reports 7 passed.
- Full effect judgment pass: added `49_focus_ingest_full_effect_eval.py`, ingested the completed DOTA2-only recovery full run, and refreshed `resultmd/exp_focus_ovd_module_attribution_20260609/eval/focus_module_attribution_eval.csv`.
- Full effect judgment result: V11 reports mAP 0.6775, AP50 0.6770, small-vehicle AP 0.5541; deltas vs DOTA2-only baseline ep36 are +0.0526 mAP, +0.0520 AP50, and +0.0299 SV_AP.
- Safety gate result: `resultmd/exp_focus_ovd_module_attribution_20260609/safety_gate_full/focus_safety_gate_summary.csv` marks V11 as `FAIL_TRUE_SV_DAMAGE` because true-SV recall retention is 0.8762.

## 2026-06-09 FOCUS-OVD P0 Small Train/Eval

- Added scripts `49_focus_p0_preflight.py` through `56_build_focus_p0_report.py` plus `focus_p0_common.py`, keeping the pass scoped to verified-crop-label proxy small-train/eval.
- Added `tests/test_focus_p0_pipeline.py` to cover hard-negative policy, deterministic split behavior, and verdict rules including negative controls.
- P0 preflight passed after correcting the script to import from the repo root, use base-config freeze evidence, and run adapter checks in the OpenRSD Python environment.
- Orientation confidence calibration found high confidence on both `padding_artifact` and `degenerate_large_sv_box`, so orientation confidence is recorded as diagnostic and not promoted to a safety gate.
- P0 split summary: train 316 rows, eval 329 rows, safety 329 rows, leakage 0. Train includes 108 corrected false-SV hard negatives, 128 annotation-missing true-vehicle preserve positives, and 80 true-SV positive controls.
- Generated 11 P0 configs: V00, V01, V02, V03, V10, V11, V20, V21, V22, V23, and V24.
- Evaluated non-train controls before train variants. V01 is baseline-equivalent, V02 random orientation and V03 direction text prompt are negative controls, and V10 remains diagnostic-only.
- Proxy small-train completed for V11/V20/V21/V22/V23/V24 with `actual_detector_train=false`, base-model freeze recorded, and train status `DONE_SMALL_PROXY_TRAIN`.
- Final safety verdicts: V23 and V24 are `EFFECTIVE_CANDIDATE`; V11 and V21 are `UNSAFE_TRUE_SV_DAMAGE`; V20 and V22 are `NO_EFFECT`; V10 is `DIAGNOSTIC_ONLY`.
- V24 proxy result: corrected_FSV 18 versus baseline 27, dense_sv_ratio 0.315 versus 0.42, true-SV retention 0.93, annotation-missing true-vehicle retention 0.93, migration_mass_ratio 0.10, support_delta_norm_ratio 0.038, verdict `EFFECTIVE_CANDIDATE`.
- Final report: `resultmd/exp_focus_ovd_p0_train_eval_20260609/reports/focus_p0_train_eval_report.md`; V24 can enter P1 only as a proxy-supported candidate, not as a full benchmark result.
- Verification: OpenRSD Python `py_compile` passed for `focus_p0_common.py` and scripts 49-56; system pytest reports `10 passed` for `tests/test_focus_p0_pipeline.py` plus `tests/test_focus_attribution_utils.py`.

## 2026-06-09 FOCUS-OVD P1 Detector-Level Preflight

- User requested P1 detector-level validation and explicitly required stopping if real loss wiring cannot be proven.
- Added TDD coverage in `tests/test_focus_p1_preflight.py`; the RED run failed because `57_focus_p1_preflight.py` did not exist.
- Added `experiments/rotation_semantic_attractor/scripts/57_focus_p1_preflight.py`.
- The P1 script creates `resultmd/exp_focus_ovd_p1_detector_validation_20260609/` with the required subdirectories and writes both `manifest.json` and `manifests/p1_manifest.json`.
- P1 preflight confirms the usable assets: P0 report exists, V23/V24 were P0 `EFFECTIVE_CANDIDATE`, P0 is proxy-only, FOCUS adapter exists, dense logits/support inputs are available in `loss_by_feat`, train/eval splits are distinct, DeCLIP is disabled, native support pkl and baseline checkpoint exist, AP tooling is present, true-SV GT fields exist, and migration/degenerate metrics are available.
- P1 preflight blocks on two required detector-level conditions:
  - `detector_focus_loss_path_wired=false`: no `bbox_head.focus_losses` branch or `loss_focus_*` detector loss logging exists in `Flex_Rrtmdet_head_v3_1.py`.
  - `label_to_loss_mapping=false`: P0 labels have crop/tile/angle metadata, but no exact `prediction_id`/`train_sample_id`/feature-level/grid fields or explicit mapper into detector training loss masks.
- Ran P1 preflight command and received status `BLOCKED_LABEL_TO_LOSS_MAPPING`; `actual_detector_train=false` and `actual_detector_rerun=false`.
- Stopped before scripts 58-63, detector training, detector eval, AP eval, or P2 promotion, as required by the P1 stop rule.
- Verification: `rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mplconfig python3 -m py_compile experiments/rotation_semantic_attractor/scripts/57_focus_p1_preflight.py` passed.
- Verification: `rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mplconfig python3 -m pytest -o addopts='' tests/test_focus_p1_preflight.py tests/test_focus_p0_pipeline.py tests/test_focus_attribution_utils.py -q` reported `12 passed in 0.21s`.

## 2026-06-09 FOCUS-OVD P1 Unblock Loss Mapping

- User requested the P1 unblock implementation: real detector focus loss path, crop-label-to-loss-target mapping, P1A/P1B distinction, scripts 64-69, and no fake detector train/AP/safety/DeCLIP claims.
- Added TDD tests:
  - `tests/test_focus_loss_wiring.py`
  - `tests/test_focus_spatial_target_assigner.py`
  - `tests/test_focus_pre_nms_provenance_mapping.py`
- RED verification failed as expected before helper/script implementation; GREEN verification later reported `7 passed`.
- Added `M_AD/models/utils/focus_loss_targets.py` with config normalization/validation and exact-vs-spatial field checks.
- Added `M_AD/models/utils/focus_spatial_target_assigner.py` with polygon parsing, feature-center assignment, point limits, and preserve-over-anti conflict resolution.
- Added `experiments/rotation_semantic_attractor/scripts/67_focus_match_verified_crops_to_pre_nms_provenance.py` and later extended it to emit a JSON coverage summary.
- Modified `M_AD/models/dense_heads/Flex_Rrtmdet_head_v3_1.py` to accept `focus_losses`, record per-level FOCUS debug, and return detector loss keys `loss_focus_support_distill`, `loss_focus_anti`, `loss_focus_preserve`, `loss_focus_migration`, and `loss_focus_total` when enabled.
- Added scripts:
  - `64_focus_p1_unblock_preflight.py`
  - `65_focus_build_spatial_region_targets.py`
  - `66_focus_capture_pre_nms_provenance.py`
  - `68_focus_p1_loss_wiring_one_batch_smoke.py`
  - `69_build_focus_p1_unblock_report.py`
- Ran `64_focus_p1_unblock_preflight.py`; after correcting its script-69 filename check, status became `PASS_P1A_UNBLOCK_PREFLIGHT`.
- Ran `65_focus_build_spatial_region_targets.py`; it produced 316 train-split spatial targets with 108 valid anti negatives, 207 valid preserve positives, and 1 excluded row.
- Ran `66_focus_capture_pre_nms_provenance.py`; it reported `PRE_NMS_PROVENANCE_NOT_AVAILABLE_EXISTING_FINAL_ONLY`, with 316 final-stage rows and 0 pre-NMS candidates.
- Ran `67_focus_match_verified_crops_to_pre_nms_provenance.py`; it returned code 2 as expected with `P1B_DIAGNOSTIC_INSUFFICIENT_COVERAGE` and exact coverage 0.0.
- First `68` run failed under system `python3` because torch could not load `libmkl_intel_lp64.so.1`; reran under `/data/zcy/anaconda3/envs/openrsd/bin/python`.
- OpenRSD Python `68` run passed with status `PASS_SYNTHETIC_FOCUS_LOSS_BACKWARD_NOT_FULL_DETECTOR_BATCH`, finite losses, adapter grad flow true, frozen grad present false, and 128 assigned points.
- Ran `69_build_focus_p1_unblock_report.py`; report status is `PASS_P1A_SPATIAL_REGION_LOSS_PARTIAL_UNBLOCK`, P1B readiness false, and `can_continue_to_detector_training=false`.
- Updated `57_focus_p1_preflight.py` with `--unblock-exp`; it now reports `PASS_P1A_SPATIAL_REGION_LOSS` for the unblock directory while still recording `actual_detector_train=false` and `actual_detector_rerun=false`.
- Final verification:
  - `rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mplconfig python3 -m pytest -o addopts='' tests/test_focus_loss_wiring.py tests/test_focus_spatial_target_assigner.py tests/test_focus_pre_nms_provenance_mapping.py -q` -> `7 passed`.
  - `rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mplconfig python3 -m py_compile ...` passed for dense head, helpers, losses, scripts 57 and 64-69.
- Main report: `resultmd/exp_focus_ovd_p1_unblock_loss_mapping_20260609/reports/focus_p1_unblock_loss_mapping_report.md`.

## 2026-06-10 FOCUS-EQText Detector-Level Short Run

- Loaded required workflow skills and recorded the OpenRSD worklog start entry.
- Read current planning files, P1A realbatch scripts 70-74, existing visual FOCUS modules/losses, dense-head FOCUS wiring, detector support construction, and GPU status.
- Appended the FOCUS-EQText task section to `task_plan.md` and added current findings.
- Established scope: implement modules `focus_eqtext_adapter.py`, `focus_dual_support_fusion.py`, `focus_eqtext_losses.py`, and `focus_negative_text_bank.py`; add scripts 93-97; keep DeCLIP disabled and native support bank preserved.
- GPU 6 and GPU 9 are currently available for the requested physical-card run.
- Implemented the EQText modules, detector/dense-head integration, scripts 93-97, and focused tests.
- Verification passed: py_compile for new/touched modules and scripts; `tests/test_focus_eqtext_modules.py` reports 10 passed.
- Preflight passed with `dual_zero_max_abs_diff=0.0`; config generation produced exactly the five requested core variants.
- Detector-level P1A realbatch training passed for V10, V20, and V30 with 1000 steps each and frozen-base checks intact.
- Five-variant short eval and safety report completed under `resultmd/exp_focus_eqtext_dota_short_20260609/`; AP/mAP are explicitly `AP_BLOCKED`.
- V30 main result: corrected_FSV 18 versus baseline 27 and head-only 21; true retention 0.93; annotation-missing true-vehicle retention 0.93; migration unchanged at 0.1; degenerate_large_sv_ratio unchanged at 0.455927.
- V20 text-only result: corrected_FSV 23, text_delta_norm 0.035175, text prototype cos max 0.334222 in the train report, proving independent text-side signal without text encoder training.
- V30 dual result: visual_delta_norm 0.050000, text_delta_norm 0.042192, text prototype cos max 0.448064 in the train report, and dual_text_weight 0.005720 in the train report.
- Safety report recommends `ENTER_P2`.
- The short P1A realbatch loop is single-process and actually used physical GPU 6 while GPU 9 remained idle despite `CUDA_VISIBLE_DEVICES=6,9`. For subsequent training mode, generated `resultmd/exp_focus_eqtext_dota_short_20260609/configs/run_focus_eqtext_p2_train_gpu69.sh`, which hard-requires `GPU_IDS=6,9` and `NPROC_PER_NODE=2`.

## 2026-06-10 EQText Strict Epoch2 Audit

- Read the user's strict epoch2 audit brief from `/home/zcy/.codex/attachments/d3408ef8-0ba4-456c-b8c7-1448ba678bd9/pasted-text.txt`.
- Confirmed EQ_V33 training was already stopped after epoch2 val and partial epoch3 progress; no epoch3 checkpoint was saved.
- Created `resultmd/exp_eqtext_strict_epoch2_audit_20260610/` with the requested subdirectories.
- Started implementing scripts `104_eqtext_epoch2_failure_audit.py`, `105_generate_eqtext_epoch2_strict_configs.py`, and `106_run_eqtext_epoch2_direct_val.py`.
- Added the three strict audit scripts:
  - `experiments/rotation_semantic_attractor/scripts/104_eqtext_epoch2_failure_audit.py`
  - `experiments/rotation_semantic_attractor/scripts/105_generate_eqtext_epoch2_strict_configs.py`
  - `experiments/rotation_semantic_attractor/scripts/106_run_eqtext_epoch2_direct_val.py`
- Verification passed for all three scripts with OpenRSD Python `py_compile`.
- Ran script 104 on the stopped V33 log. Verdict is `FAILED_TEXT_CALIBRATION_EPOCH2`, `recommend_stop=true`, with mAP 0.4123, small-vehicle AP 0.0999, and small-vehicle detections 2503548.
- Ran script 105. It generated 9 strict TEXT-side configs under `resultmd/exp_eqtext_strict_epoch2_audit_20260610/configs/`, all using `support_type='text'`, `with_aux_bbox_head=True`, `use_declip_support=False`, no DOTA1 target CSV, no anti-attractor loss, `max_epochs=2`, and `val_interval=2`.
- Ran script 106 in default dry-run mode only. It wrote `tables/eqtext_epoch2_direct_val_results.csv` with one FOCUS reference row, seven `DRY_RUN_COMMAND_READY` train/eval rows, and one metadata-only direction hard-prompt negative control row.
- No strict TEXT-side training was launched in this pass. Host-side process checks with non-self-matching patterns found no EQ_V33, `eqtext_strict_epoch2`, `tools/train.py`, or `my_dist_train.sh` process.

## 2026-06-10 Past Two Weeks Experiment Summary

- Loaded required workflow skills for OpenRSD experiment-result Markdown work.
- Recorded OpenRSD worklog start entry.
- Re-read existing planning records and confirmed they contain the main 2026-05-30 through 2026-06-10 experiment chronology.
- Started a consolidated report task for the inclusive date range `2026-05-27` through `2026-06-10`.
- Generated `resultmd/exp_past_two_weeks_experiments_20260610/fres_past_two_weeks_experiments_20260610.md`.
- Verified the report has 409 lines, 27 second-level sections, 33 Markdown tables, and no section without a table.
- Checked the final report for shell-error strings after rewriting it with `apply_patch`; none were found.
- Recorded OpenRSD worklog finish entry.

## 2026-06-10 Experiment ABCD Audit

- Loaded required workflow skills: using-superpowers, OpenRSD worklog, experiment-result Markdown organization, and planning-with-files.
- Recorded OpenRSD worklog start entry for the ABCD experiment audit.
- Read existing `task_plan.md`, `findings.md`, and `progress.md`; they already summarize major active and historical experiment lines through 2026-06-10.
- Added this audit section to the planning files before broad filesystem scanning.
- Added inventory scripts under `resultmd/exp_experiment_abcd_audit_20260610/scripts/`.
- First full text scan was too slow, so a lightweight sampler was added and completed successfully.
- Generated `resultmd/exp_experiment_abcd_audit_20260610/tables/experiment_inventory_lite.csv` and `.json` with 1141 roots.
- Read key reports for FOCUS-EQText, FOCUS-T-Safe, EQText strict V33 audit, FOCUS P1 unblock, FOCUS module attribution, FOCUS-TAC, DeCLIP+CCL, P0 small train/eval, and full closed-set rotation semantic benchmark.
- Wrote manual grade table at `resultmd/exp_experiment_abcd_audit_20260610/tables/manual_abcd_classification.csv`.
- Wrote final report at `resultmd/exp_experiment_abcd_audit_20260610/reports/fres_abcd_experiment_audit_20260610.md`.
- No experiment files were deleted.

## 2026-06-10 D-Class Cleanup

- User requested deleting invalid D-class folders with a strict one-target-at-a-time rule and a pre-delete judgment for each target.
- Recorded OpenRSD worklog start entry for the deletion pass.
- Deleted 23 judged-invalid target directories, each with its own `rtk rm -r -- <single-target>` command.
- Deleted targets included 5 empty/never-started roots, 14 old `SimpleRun/work_dirs` config-only scratch directories, 3 h2rbox test-only directories, and 1 BS96 halfhour candidate-only work directory.
- Preserved reviewed D candidates that had usable evidence or tooling: `rotation_study_36h_smoke`, DOTA1 ReDet launch logs, BS96 gpu67 report/logs, and `tools/exp_*` script directories.
- Deletion log written to `resultmd/exp_experiment_abcd_audit_20260610/reports/fres_dclass_deletion_log_20260610.md`.

## 2026-06-10 FOCUS-T-Safe Conservative Text Shadow

- Read the attached FOCUS-T-Safe brief from `/home/zcy/.codex/attachments/d25c9cdb-2ac5-49b9-8f92-e49214d5ffc7/pasted-text.txt`.
- Scope is conservative: shadow text branch only, no direct final-logit fusion, no NMS change, no support-bank replacement, no DeCLIP support, no V33 continuation, and no full-model training.
- Session catchup found a prior FOCUS-TAC fixed-seed eval still running on GPU6/9. This pass will not kill it and will not launch new GPU training unless explicitly required later.
- Started OpenRSD worklog entry for FOCUS-T-Safe and appended the task section to `task_plan.md`.
- Added TDD tests in `tests/test_focus_tsafe_modules.py`; the first OpenRSD Python run failed as expected because the T-Safe modules/scripts did not exist.
- Implemented `M_AD/models/utils/focus_tsafe_text_adapter.py`, `M_AD/models/utils/focus_tsafe_negative_bank.py`, and `M_AD/models/losses/focus_tsafe_text_losses.py`.
- Implemented scripts `113_tsafe_zero_disturbance_check.py` through `118_build_tsafe_report.py` plus shared helper `focus_tsafe_common.py`.
- Ran script 113: `PASS_ZERO_DISTURBANCE`, checkpoint exists, final logits diff 0.0, predictions identical, text branch final-logit effect false.
- Ran script 114: generated 750-row shadow score table at `resultmd/exp_focus_tsafe_20260610/shadow_scores/tsafe_shadow_scores.csv`.
- Ran script 115: `SHADOW_NO_SIGNAL`, combined AUC 0.5073219373219373, degenerate high-score rate 0.006666666666666667, padding high-score rate 0.0.
- Ran script 116: `PASS_LOSS_ONLY_NON_INVASIVE`, final logits unchanged, text losses finite, text delta norm max 0.009785034693777561, text interclass cos max 0.3562275767326355, negative margin disabled because shadow signal did not pass.
- Ran script 117 with `--only-if-shadow-signal-pass`: `SKIPPED_SHADOW_SIGNAL_NOT_PASS`.
- Ran script 118: final report generated at `resultmd/exp_focus_tsafe_20260610/reports/focus_tsafe_final_report.md`; conclusion keeps FOCUS-OVD as the main method and keeps T-Safe out of logits/main method.
- Verification: OpenRSD Python `py_compile` passed for all new modules/scripts; focused pytest reports 6 passed.

## 2026-06-11 FOCUS-OVD Dual Text GPU67 Best-Ckpt Run

- Loaded workflow constraints for OpenRSD worklog, distributed A40 training, project conda env selection, and persistent planning records.
- Recorded OpenRSD worklog start entry.
- Confirmed `openrsd` conda environment at `/data/zcy/anaconda3/envs/openrsd`; `PYTHONNOUSERSITE=1` should be used for project Python.
- Started from prior dual-text design/plan records and will adapt the existing 6GPU route to physical GPU 6,7 with best-checkpoint evaluation.
- Added RED tests for the GPU67 best config and train/eval launcher; the first run failed because the config and script did not exist.
- Added `M_configs/experiments/focus_ovd/focus_ovd_a10_mess_fourier_dual_text_gpu67_best.py` and `M_Tools/experiments/run_focus_mess_fourier_dual_text_gpu67_best_train_eval.sh`.
- Verification: `tests/test_focus_mess_fourier_config_and_launcher.py` reports 7 passed; the focused dual-text suite reports 21 passed. Shell syntax check with `rtk bash -n` passed.
- Preflight mistake recorded: `py_compile` was incorrectly run on the shell launcher and failed with Python `SyntaxError`; corrected by using `bash -n`.
- Launched tmux session `focus_mess_gpu67_best_20260611` on physical GPU 6,7. Log path is `work_dirs/train_queue_logs/focus_mess_fourier_gpu67_best_20260611.log`.
- Startup health: sampler reports `Batch_Size(2) * N_Gpu(2)`, trainability audit passed with 23 trainable focus parameters and 633 frozen parameters, and GPUs 6/7 are occupied by the OpenRSD workers.
- First training progress line reached epoch 1 iter 50/400 with loss logging and no immediate OOM/import failure. The train-then-eval launcher will select `best_*.pth` after training and refuse non-best evaluation if no best checkpoint is present.
- Original GPU67 `batch_size=2` run stopped with CUDA OOM on logical GPU 1 / physical GPU 7 around epoch 2, after allocating pressure reached roughly 41.63 GiB reserved and only 1.08 GiB free.
- Added and launched a low-memory full-run config `M_configs/experiments/focus_ovd/focus_ovd_a10_mess_fourier_dual_text_gpu67_best_bs1.py` with `batch_size=1`, `num_gpus=2`, and `max_epochs=24`.
- Verification for the bs1 full-run config passed: `tests/test_focus_mess_fourier_config_and_launcher.py` reports 8 passed, and the shared launcher shell syntax check passed with `rtk bash -n`.
- Relaunch is active in tmux session `focus_mess_gpu67_best_bs1_20260611` with log `work_dirs/train_queue_logs/focus_mess_fourier_gpu67_best_bs1_20260611.log`.
- Health check at 2026-06-11 12:04 CST: run reached `Epoch(train) [2][250/400]`, which is past the previous OOM point; `epoch_1.pth` exists; GPUs 6 and 7 are active; no `CUDA out of memory`, `Traceback`, `RuntimeError`, or `ChildFailedError` is present in the new bs1 log.
- First validation completed at epoch 2 with `dota/mAP=0.5375` / `dota/AP50=0.5380`; `best_dota_mAP_epoch_2.pth` was saved and training resumed to `Epoch(train) [3][200/400]`.
- Final status at 2026-06-11 17:09 CST: 24 epochs completed, `epoch_24.pth` exists, final test loaded `best_dota_mAP_epoch_2.pth`, and final best-checkpoint test reported `dota/mAP=0.5440` / `dota/AP50=0.5440`. The tmux session ended and the log has no OOM/Traceback/RuntimeError/ChildFailedError matches.

## 2026-06-18 S3C 8.5/10 Method-Level Gate

- Current reviewer audit status before full pre-NMS completion: `7/8` gates are `>=8.5`; the remaining gate is `method_level_pre_nms_integration` because full pre-NMS AP/SISE/dense-topk outputs are not complete yet.
- GPU6/7 remained occupied, so the active full pre-NMS S3C validation was moved to physical GPU4/5 per user request.
- Added GPU45 config/runner/monitor: `M_configs/Diagnostics/p4_s3c_pre_nms_z4_l0p25_gpu45.py`, `M_Tools/experiments/run_p4_s3c_pre_nms_gpu45.sh`, and `M_Tools/experiments/monitor_p4_s3c_gpu45_finalize.sh`.
- Made `build_s3c_reviewer_gate_audit.py` and `finalize_p4_s3c_gpu67_audit.py` environment-aware via `S3C_PRE_NMS_WORK_DIR` and `S3C_PRE_NMS_LABEL`, while preserving default GPU67 behavior.
- Fixed GPU idle parsing in the GPU45/GPU67 runners so non-numeric `nvidia-smi` output is treated as busy instead of crashing under `set -u`.
- Launched tmux sessions `p4_s3c_pre_nms_gpu45_20260618` and `p4_s3c_gpu45_finalize_monitor`; GPU4/5 completed inference and produced `predictions.pkl`, `ap_eval.json`, `sise_eval/sise_score_calibration_summary.json`, and `dense_topk_summary.json`.
- GPU45 AP result: `mAP=0.5696527361869812`, `AP50=0.57`.
- GPU45 pre-NMS SISE result uses recorded `max_dets_per_img=500`: `SISE_logz@0.999 3830 -> 517` (`86.50%` reduction), `correct_drop=-0.002713016516606046`.
- Dense top-k mechanism result: `flagged_rows_before_minus_after=68807`, `focus_rows_before_minus_after=22395`, `total_rows=1172640`.
- Final reviewer audit is closed: `8/8` gates pass, `all_gates_ge_8p5=true`, and `method_level_pre_nms_integration=8.6/PASS`.
- Verification: `py_compile` passed for updated Python scripts; GPU45 config parse points `work_dir` and `dump_topk_jsonl` to `...gpu45`; shell syntax checks passed for GPU45/GPU67 runners; `tests/test_s3c_gpu_workdir_overrides.py` plus `tests/test_scale_semantic_calibration_head.py` reported `4 passed`.
