# Rotation Semantic Attractor Experiment Deployment Findings

- User wants the experiment plan deployed through smoke testing, using GPUs 6,7,8,9.
- The pasted requirements demand a self-contained module under `experiments/rotation_semantic_attractor/`, with inventory, registries, split builder, rotation/GT tools, model adapters, false-hub metrics, stage decomposition, intervention stubs, report builder, and smoke script.
- Root `task_plan.md/findings.md/progress.md` have been replaced with the current rotation semantic attractor deployment records.
- The repository contains many mmrotate configs and DOTA baseline configs, including Rotated RetinaNet, Rotated Faster R-CNN, RoI Transformer, Oriented R-CNN, ReDet, R3Det/KFIoU, S2ANet, Oriented RepPoints, Rotated FCOS, Rotated RTMDet, H2RBox, and Rotated ATSS.
- Closed-set detectors can only test rotation-conditioned class-channel/logit hubs; they cannot validate an open-vocabulary embedding attractor.
- `data/DOTA1_1024_500/angle_sweep_val/realistic/angle_000/` exists with `images/` and DOTA text `annfiles/`.
- `weights/` contains real closed-set weights including rotated RetinaNet, rotated RTMDet-L, H2RBox, H2RBox-v2, Oriented R-CNN, Oriented RepPoints, R3Det-KFIoU, ReDet, and StripNet.
- `M_configs/RotationStudy/` has DOTA1 eval configs for rotated RTMDet-L, H2RBox, H2RBox-v2, ReDet, and rotated RetinaNet.
- `nvidia-smi` showed GPUs 6,7,8,9 idle. The sandboxed Python process initially could not see CUDA, but the approved outside-sandbox diagnostic and smoke run saw 4 CUDA devices with `CUDA_VISIBLE_DEVICES=6,7,8,9`.
- The decisive smoke evidence is a new GPU-backed mmrotate inference run, not a cached-prediction fallback: `outputs/runs/smoke_2tiles_2angles_gpu/manifest.json` records `device: cuda:0`, `torch_cuda_available: true`, and 4 visible CUDA devices.
- The model registry now contains 15 closed-set methods; 11 have checkpoint paths and 4 are kept as missing-checkpoint `NOT_RUN` candidates.
- After the repair pass, all 15 registry methods have local checkpoint paths. The four added weights are RoI Transformer, S2ANet, Rotated FCOS, and Rotated ATSS from the ai4rs/OpenMMLab mmrotate v0.1.0 metadata.
- Stage decomposition now has measured pre-NMS and dense small-vehicle score summaries for supported single-stage models. The repaired smoke status is `PARTIAL`, not `UNSUPPORTED`, because query logits remain unavailable for closed-set detectors.
- Smoke intervention/counterfactual/DeHub safety scripts now emit executable proxy rows (`PROXY` or `SMOKE_PROXY`) instead of empty `NOT_RUN` rows. These are smoke proxies, not full causal image-editing or weight-level interventions.
- Status semantics have been repaired with a unified status vocabulary. Closed-set query logits are now `NOT_APPLICABLE` because dense/head closed-set detectors do not use query logits.
- Capability-driven reporting now separates capability matrix, smoke execution matrix, actual scientific tables, proxy/schema validation, stage decomposition, intervention, DeHub safety, and appendices.
- Open-vocabulary smoke is schema-only until a runnable open-vocabulary/OpenRSD model asset exists. Closed-set-only smoke records open-vocabulary intervention as `NOT_SELECTED_IN_THIS_SMOKE`; open-vocabulary smoke records the missing asset as `NOT_AVAILABLE_ASSET`.
- Closed-set intervention and context counterfactual rows are `SMOKE_PROXY` with `is_scientific_result=false` and `include_in_main_table=false`. DeHub safety rows are `SCHEMA_ONLY` with `include_in_repair_table=false`.

## Resolved Inventory Questions

- Dataset paths and DOTA text annotations are present for the DOTA1 angle-sweep validation tiles used by smoke.
- At least one actual mmrotate model loads and runs in smoke; rotated RetinaNet MS RR loaded on logical `cuda:0`.
- Full-run registry has 15 methods available by checkpoint, satisfying the closed-set expansion requirement.
- Pre-NMS/dense hooks are implemented for supported single-stage heads. Query logits remain a follow-up for open-vocabulary/query-based detectors, not for ordinary closed-set mmrotate heads.
- The current repaired smoke verdict is `PASS_WITH_NOT_APPLICABLE`, not a full scientific completion signal. The experiment still needs a full run and real intervention assets before making causal or repair claims.

## Scientific Buildout Findings

- DONE_FULL gates are now explicit and conservative. Smoke/proxy/schema rows cannot enter scientific tables even if they have numeric metrics.
- OpenRSD/open-vocabulary assets are available locally. The asset inventory found runnable OpenRSD config+checkpoint+visual-support triples, including the A10 formal checkpoint and DINOv2 visual support pkl.
- The OpenRSD hook smoke now runs real forward inference on GPU. It captures dense logits, pre-NMS head tensors, visual-support embeddings, text-support mapping non-execution, class-embedding projection, dense features, and alignment candidates.
- OpenRSD `zero_sv` is an embedding-level visual-support intervention, not prompt-only text editing. In the latest smoke it changes the visual-support embedding checksum and drops small-vehicle predictions on the smoke image from 935/1348 to 0/480 detections.
- Closed-set intervention is no longer a postprocess proxy in the current smoke artifact. It modifies the RetinaNet small-vehicle classifier weights, reruns inference, and records paired baseline/intervention predictions.
- Context counterfactual is no longer just a ledger/proxy in the current smoke artifact. It writes real `object_only.png`, `context_only.png`, and `target_mask.png` images, then reruns inference on the modified images.
- DeHub safety is no longer schema-only in the current smoke artifact. Baseline and DeHub repair checkpoints are both run on the same smoke split/angles; the smoke records class-distribution JS/KL and low-risk inflation.
- Full scientific conclusions are still intentionally blocked because the full S2 final-test, 12-angle, 10+ closed-set model benchmark and the full open-vocabulary/causal/safety benchmark have not been executed.
- Full execution readiness audit currently reports no DONE_FULL scientific rows before heavy execution completes; this is expected and is not a bug.
- S2 final-test is frozen at 2500 tiles with seed 20260530. S3 safety stratified has been rebuilt/frozen to include false-SV-hub negatives, true-SV-rich positives, low-risk normal tiles, and cross-class conflict tiles.
- The original single-GPU full closed-set command was functional but impractical: about 973 tile-angle predictions after roughly 12 minutes, implying a very long wall-clock time for 300000 required predictions.
- The active full closed-set benchmark now uses independent model shards in one output directory. Per-model rotated assets prevent file races, and `18_update_run_status.py` reconstructs the central manifest and `metrics/run_status_by_model.csv` from file counts.
- DONE_FULL promotion for false-hub rows now checks the corresponding model's manifest status, so a partial or failed model cannot be promoted just because the run uses S2 and 12 angles.

## 2026-05-31 Continuation Findings

- Full closed-set S2 12-angle inference is complete and audited. The validated closed-set scientific audit bundle is under `/data/zcy/OpenRSD_artifacts/closedset_scientific_audit_20260531`.
- The full closed-set `rotated_images` directory is absent; only a reconstructed path index exists under `/data/zcy/OpenRSD_artifacts/closedset_rotated_images_index_20260531`. Any later task that needs pixels must regenerate rotated images from the original frozen split, not rely on old closed-set rotated-image paths.
- Full closed-set raw/canonical JSON outputs remain useful and have path-size-mtime indexes under `/data/zcy/OpenRSD_artifacts/closedset_prediction_indexes_20260531`. The indexes are catalogs only and cannot reconstruct deleted prediction contents if the JSONs are later removed.
- After cleanup, `/data1` has about 286G free and `/data` has about 272G free. Future full open-vocab/causal/context/DeHub outputs should write to `/data` and aggressively avoid storing large intermediate rotated/counterfactual images.
- GPU status at continuation start: no active `03_run_angle_sweep`, `07_intervention_open_vocab`, `08_intervention_closed_set`, `09_context_counterfactual`, `10_eval_dehub_safety`, `tools/train.py`, `torchrun`, or queue runner process was found; GPUs 5-9 were mostly free.
- `03_run_angle_sweep.py` only supports `mmdet` adapters. OpenRSD/Open-vocabulary full inference needs a dataset-loader based runner similar to `15_openrsd_hook_smoke.py` or `10_eval_dehub_safety.py`.
- `07_intervention_open_vocab.py` currently writes only capability/status rows. It does not run full OpenRSD inference or full visual-support intervention.
- `08_intervention_closed_set.py` can rerun real classifier-channel interventions, but it reads `metadata.rotated_image_path` from raw prediction JSONs. Because full closed-set rotated images were removed, a full runner must regenerate or stream rotated images before inference.
- `09_context_counterfactual.py` can generate real object-only/context-only edited images and rerun inference, but the current path is smoke-oriented and depends on persisted rotated image/GT paths. Full context evaluation needs regenerated image/GT assets or a streaming image-edit path.
- `10_eval_dehub_safety.py` can run baseline and repair OpenRSD checkpoints, but the current selector defaults to smoke-sized rows and also depends on rotated assets from a source run. Full safety evaluation should use frozen S3 directly.
- `21_run_full_openrsd_streaming.py` now provides the missing streaming path for OpenRSD full continuation. It avoids long-lived `rotated_images` by regenerating per-chunk image/ann assets under `/data/zcy/OpenRSD_runs/rotation_semantic_attractor/scratch_openrsd_streaming`, recording a transient asset index, and deleting chunk files after inference.
- The OpenRSD full continuation uses four separate run directories:
  - `/data/zcy/OpenRSD_runs/rotation_semantic_attractor/full_openvocab_s2_12angle`
  - `/data/zcy/OpenRSD_runs/rotation_semantic_attractor/full_causal_intervention_s3_12angle`
  - `/data/zcy/OpenRSD_runs/rotation_semantic_attractor/full_context_counterfactual_s3_12angle`
  - `/data/zcy/OpenRSD_runs/rotation_semantic_attractor/full_dehub_safety_s3_12angle`
- `22_check_full_openrsd_status.py` is the monitor entry point for these four runs and writes `/data/zcy/OpenRSD_runs/rotation_semantic_attractor/full_openrsd_status.csv` plus `.json`.

## 2026-06-08 FOCUS-OVD Findings

- FOCUS-OVD should operate in the native OpenRSD DINOv2/A10 support space. The key implementation invariant is additive zero-initialized residual modulation after the existing support mapping, not replacing support banks.
- The first supported class target is `small-vehicle`; all-class modulation is an ablation only.
- Corrected-FSV labels must split annotation-missing true vehicles from confirmed false-SV, because prior audit shows over half of decidable valid unmatched SV crops are true vehicles or missed matches.
- The implementation now uses support labels plus ordered class names for class gating, so support-shot rows for `small-vehicle` can be selected without assuming one support row per class.
- Zero residual/alpha is an exact classifier fallback, not just close numerically: `30_focus_baseline_equivalence_smoke.py` reported `max_abs_diff=0.0`.
- The corrected-FSV index excludes degenerate large-SV and padding categories from hard negatives, keeping them available as failure-mode references rather than negative training evidence.
- Full AAAI/ICCV-level claims still require real adapter training and paired evaluation. Current artifacts are a code-complete prototype plus smoke/report scaffold, not a completed benchmark result.
- `MetaRemoveRunner.train()` builds the optimizer after runner construction but before hook `before_run`, so adapter-only training cannot rely on a freeze hook alone. The reliable mechanism is a pre-optimizer trainable allowlist in the runner plus a hook audit.
- The A10 epoch24 checkpoint now exists at `/data1/zcy/OpenRSD/results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth`; immediate preflight passed after earlier planning had observed an empty directory.
- At watcher launch time, GPU 6 and GPU 9 were still busy at about 34GB each, so the scheduled launcher is correctly parked until both cards satisfy the memory/util thresholds after 03:00.

## 2026-06-09 FOCUS-OVD Module Attribution Findings

- The new task is not a full benchmark or blind module expansion. It requires module attribution first, then controlled default-off integration of diagnostic/safety modules, then a paper-grade evidence report.
- Hard constraints from the brief: keep native OpenRSD support bank as the main path; do not make DeCLIP support, direction text prompts, all-class residuals, or unmatched-SV hard negatives into default/main-method behavior.
- Current repo already has FOCUS prototype files and tests from 2026-06-08: Fourier orientation, residual support adapter, orientation-conditioned contrastive embed, attractor losses, baseline-equivalence smoke, orientation probe, train/eval smoke scaffolds, corrected-FSV audit labels, and support geometry-related adapter diagnostics.
- The dated output root for this pass is `resultmd/exp_focus_ovd_module_attribution_20260609/`; experiment-specific Markdown will stay inside that directory per result organization rules.
- The new module inventory reports 20 rows and confirms the required guardrails: `use_focus_ovd=False` baseline path, alpha-zero smoke availability, SV-only adapter default, zero-init residual, delta norm cap, support cosine debug, detach-able orientation, corrected-FSV label separation, DeCLIP disabled by default, and no default direction prompt injection.
- The new attribution matrix contains 36 variants covering V00-V63. P0 rows are V00, V01, V10, V11, and V20-V24.
- Cluster mining and attribute-gate audits consumed 750 human-audit rows. Channel mask and orbit teacher wrote `NO_INPUT` artifacts because no channel sensitivity/orbit consistency CSV was supplied. Head consensus wrote `UNSUPPORTED_BY_CURRENT_CODE` because no multi-head scores/hooks were supplied.
- The first run of the new baseline-equivalence smoke with seed 20260609 exposed a strict equivalence bug (`max_abs_diff=4.768e-07`) caused by accumulation-order drift in the zero-residual FOCUS path. The fix uses the exact baseline matmul in no-grad zero-delta inference while leaving the training path differentiable.
- After the fix, `V01_focus_zero` baseline-equivalence smoke passed with `max_abs_diff=0.0` and `max_delta_norm_ratio=0.0`.
- `V10_orientation_probe_only` wrote a diagnostic synthetic probe with mean periodic error about 6.2167 degrees and confidence range 0.7628-0.9722.
- No AP, corrected-FSV reduction, true-SV retention, migration, support-collapse, or det/img improvement was demonstrated in this pass; train/eval-dependent P0 variants are explicitly `NOT_EVALUATED_REQUIRES_TRAINING`.
- The previously completed FOCUS DOTA2-only recovery full run is a V11-style result: SV-only orientation residual adapter, losses off, adapter-only trainable audit passed, checkpoint at `work_dirs/focus_ovd_a10_sv_only_dota2_recovery_full_gpu69_20260609/epoch_24.pth`.
- Full effect judgment for V11 is mixed: AP metrics are positive versus DOTA2-only baseline ep36 (mAP +0.0526, AP50 +0.0520, small-vehicle AP +0.0299), but small-vehicle recall drops from 0.7317 to 0.6411 and small-vehicle detections drop from 523202 to 353603.
- The rule-based safety gate marks V11 as `FAIL_TRUE_SV_DAMAGE` because true-SV recall retention is 0.8762, below the 0.90 minimum. This blocks main-method promotion until corrected-FSV, dense-SV, migration, and false-positive/false-negative audits are completed.
- V20-V24 remain not evaluated as full loss-attribution variants; the current full run does not exercise support-distill, anti-attractor, preserve, or their combined core loss path.

## 2026-06-09 FOCUS-OVD P0 Small Train/Eval Findings

- The P0 pass is a verified-crop-label proxy experiment. It does not run the full detector benchmark, does not retrain the full model, and does not prove final AP-level gains.
- Existing detector integration exposes the zero-init support residual adapter path, but the support-distill, anti-attractor, preserve, and migration losses are still not wired into a real detector training loss path. The new P0 train script therefore records bounded proxy small-train outcomes rather than claiming real loss-backprop detector training.
- Preflight assets are available: zero-equivalence smoke passed, orientation probe exists, A10 baseline checkpoint exists, native DINOv2 support pkl exists, DeCLIP support is not enabled, and adapter-only freeze support exists through base config plus runner/hook code.
- Label policy is preserved: only 135 `corrected_false_sv` rows are hard negatives; 160 `annotation_missing_true_vehicle` rows are preserve positives; degenerate large-SV boxes and padding artifacts are excluded failure modes, not corrected-FSV loss.
- Orientation confidence is unsafe as a standalone filter because degenerate rows have mean confidence 0.913 and high-confidence rate 0.70, while padding rows have mean confidence 0.862 and high-confidence rate 0.43.
- V02 random orientation and V03 direction text prompt did not create meaningful corrected-FSV improvement, so they behave as negative controls. Their migration/degenerate side metrics remain reported but do not turn them into effect candidates.
- V11 orientation-adapter-only remains unsafe in P0 because true-SV and annotation-missing true-vehicle retention are 0.86, below the 0.90 threshold.
- V21 anti-attractor-only strongly reduces corrected_FSV from 27 to 18 but damages true-SV retention to 0.78 and annotation-missing retention to 0.76, confirming that anti-attractor alone is too blunt.
- V20 support-distill-only and V22 preserve-only do not produce enough corrected-FSV reduction, so they are stabilizers rather than independent effect modules in this proxy.
- V23 anti-plus-preserve is the simpler effective ablation: corrected_FSV 21, dense_sv_ratio 0.3486, true-SV retention 0.92, migration_mass_ratio 0.10, support_delta_norm_ratio 0.035.
- V24 full core is the strongest proxy candidate: corrected_FSV 18, dense_sv_ratio 0.315, true-SV retention 0.93, annotation-missing retention 0.93, det/img 4.20 within the 20% safety bound, and support geometry below collapse thresholds.
- V24 can enter P1 only as a proxy-supported candidate. P1 must wire the real loss path or otherwise run a detector-level adapter training/eval loop before any paper-grade main-method claim.

## 2026-06-09 FOCUS-OVD P1 Detector-Level Findings

- P1 cannot start detector training from the current code without additional implementation. The preflight status is `BLOCKED_LABEL_TO_LOSS_MAPPING`.
- FOCUS adapter inference exists in `M_AD/models/dense_heads/Flex_Rrtmdet_head_v3_1.py`, and `loss_by_feat` receives dense classification scores plus support features/labels, so the necessary tensors are nearby.
- The real detector loss path does not yet include a `bbox_head.focus_losses` config branch, `loss_focus_support_distill`, `loss_focus_anti`, `loss_focus_preserve`, `loss_focus_total`, or calls to `anti_attractor_loss`/`preserve_loss` from detector training.
- `M_AD/models/losses/focus_attractor_losses.py` contains helper functions, but they are not wired into detector loss computation.
- P0 corrected-FSV rows have `crop_id`, `tile_id`, `angle`, `metadata_json`, class/score/box audit fields, and focus labels, but they do not have exact detector training mask fields such as `prediction_id`, `train_sample_id`, `feature_level`, `grid_x`, or `grid_y`.
- Because the P0 label split is tile/angle proxy metadata rather than detector training anchors/grid cells, it cannot safely drive `anti_attractor_loss` or `preserve_loss` during detector training yet.
- The P1 preflight correctly preserves the hard constraints: DeCLIP is disabled, native DINOv2/A10 support pkl exists, baseline checkpoint exists, annotation-missing true vehicles are not negatives, and degenerate/padding rows are not corrected-FSV negatives.
- No P1 detector-level effectiveness conclusion exists. V24 remains a P0 proxy candidate only, not P1 evidence.
- Required next engineering step before P1 can proceed: implement a real label-to-loss mapper that converts audited corrected-FSV/preserve rows into detector-level training masks or sample weights with explicit provenance, then wire `focus_losses` into `Flex_Rrtmdet_head_v3_1.loss_by_feat` and prove gradient flow with the planned one-batch smoke.

## 2026-06-09 FOCUS-OVD P1 Unblock Findings

- The previous `detector_focus_loss_path_wired=false` blocker is resolved for the dense head: `Flex_Rrtmdet_head_v3_1.py` now has a `focus_losses` config branch and returns `loss_focus_support_distill`, `loss_focus_anti`, `loss_focus_preserve`, `loss_focus_migration`, and `loss_focus_total` when enabled.
- The target mapping blocker is only partially resolved. P1A now has detector-level spatial pseudo-region targets, but P1B exact pre-NMS provenance remains unavailable.
- P1A spatial target counts from the P0 train split are 316 total rows, 108 valid `anti_negative` rows, 207 valid `preserve_positive` rows, and 1 excluded row.
- Existing verified crop metadata is final-stage only. `raw_prediction_record.stage` is `final` for all 316 spatial target rows, and there are no `feature_level`, `grid_x`, `grid_y`, or `anchor_id_or_point_id` fields.
- Exact pre-NMS matching coverage is 0.0, so `P1B_EXACT_PROVENANCE_LOSS_READY=false` and no exact pre-NMS detector loss claim is valid.
- The one-batch smoke proves the loss tensors and adapter gradient path on synthetic detector-head tensors: all losses are finite, adapter grad flow is true, frozen grad present is false, and 128 target points are assigned.
- The one-batch smoke is not actual detector training. It records `actual_detector_train=false` and `detector_batch_source=synthetic_detector_head_tensors`.
- The updated P1 preflight can now return `PASS_P1A_SPATIAL_REGION_LOSS` when pointed at the unblock experiment, but this is not an AP/safety/full-training result.
- Detector training still cannot be launched as a proven next step because real batch target-mask injection is not proven. The consolidated report therefore records `can_continue_to_detector_training=false` and `next_training_command_if_ready=NOT_READY`.
- Main evidence lives under `resultmd/exp_focus_ovd_p1_unblock_loss_mapping_20260609/`; the key report is `reports/focus_p1_unblock_loss_mapping_report.md`.

## 2026-06-10 FOCUS-EQText Findings

- User requested a key TEXT-vs-HEAD detector-level test: Fourier image phase conditions a frozen text-support residual while the existing head-side FOCUS path is kept as the visual/head branch.
- Required core variants are exactly five: `EQ_V00_baseline`, `EQ_V01_focus_zero_dual`, `EQ_V10_head_only`, `EQ_V20_text_eq_only`, and `EQ_V30_dual_eqtext`.
- The passed P1A realbatch path is available through scripts 70-74. It uses a real dataloader detector batch, spatial pseudo-region focus target masks, finite focus losses, adapter gradient evidence, and frozen-base checks.
- P1B exact pre-NMS provenance remains unavailable and must not be implied by the EQText experiment.
- Detector support data already contains both `visual_embeds` and `text_embeds` in the native DINOv2/A10 support pkl. DeCLIP is controlled by a separate switch and remains disabled.
- Current detector code can build visual and text support mappings, but the dense head needs explicit text-support and visual-support tensors to support dual fusion without replacing the native support bank.
- GPU status at start: physical GPUs 6 and 9 are available; multi-GPU commands should expose only `CUDA_VISIBLE_DEVICES=6,9` and set `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`.
- Implemented `FourierEquivariantTextAdapter` with bounded signed `alpha_t` in `[-0.05, 0.05]`, zero residual at init, prototype-level text geometry monitoring, and no text encoder training.
- Implemented `FocusDualSupportFusion` with exact disabled/zero-alpha fallback for equivalence checks and a training-time straight-through path so alpha-zero text residuals can receive gradients.
- Implemented EQText auxiliary losses and negative text bank. The negative bank is auxiliary-only and does not replace class support.
- Detector support routing now keeps native chosen support while separately passing mapped native visual and mapped native text supports into the FOCUS classification head.
- The P1A spatial target policy is preserved: annotation-missing true vehicles are preserve positives, not negatives; degenerate large-SV and padding artifacts are excluded from corrected-FSV loss.
- Preflight passed and proved focus-zero-dual baseline equivalence with `max_abs_diff=0.0`.
- All three train variants passed actual P1A realbatch detector training evidence for 1000 steps: V10 head-only, V20 text-eq-only, and V30 dual-eqtext.
- V20 text-only has independent signal in the short proxy eval: corrected_FSV 23 versus baseline 27, text_delta_norm 0.035175, text prototype cos max 0.334222 in the train report, and true/annotation retention 0.94.
- V30 dual-eqtext exceeds head-only in the short proxy eval: corrected_FSV 18 versus V10 head-only 21 and baseline 27, with true retention 0.93 and annotation-missing retention 0.93.
- V30 train report shows bounded geometry with anchor enabled: visual_delta_norm 0.050000, text_delta_norm 0.042192, text prototype cos max 0.448064, dual_text_weight 0.005720, and `loss_focus_text_anchor` present in `loss_focus_total`.
- Migration and degenerate ratios do not worsen in the short safety table; AP/mAP remain blocked because no full DOTA AP evaluation was run.
- The final safety report recommends `ENTER_P2`, but the current short realbatch loop is single-process and actually used physical GPU 6 only. A generated future training launcher hard-requires `GPU_IDS=6,9` and `NPROC_PER_NODE=2` for P2/training-mode runs.

## 2026-06-10 EQText Strict Epoch2 Audit Findings

- EQ_V33 has been stopped and should not be continued as a long run.
- The strict audit target is not to prove TEXT-side equivariance by long training; it is to reject unsafe TEXT changes by epoch2 AP/detection calibration.
- Known FOCUS epoch2 reference from `resultmd/exp_focus_ovd_eqv30_val14_compare/tables/val14_map_key_ap.csv`: mAP 0.6755, small-vehicle AP 0.5512, small-vehicle detections 347552.
- Known EQ_V33 epoch2 result from `work_dirs/focus_eqtext_dota_short_20260609/EQ_V33_text_eq_clean_dota2_targets/20260610_110750/20260610_110750.log`: mAP 0.4123, small-vehicle AP 0.0999, small-vehicle detections 2503548, bridge detections 955260, helipad detections 82104.
- Current dense-head training loss wiring logs `loss_focus_text_anchor`, but `loss_focus_eqtext_consistency` and `loss_focus_text_negative_margin` helper functions are not wired into `loss_focus_by_feat` logs. Strict runner output must mark them as `NOT_LOGGED` unless future wiring is added.
- The current `focus_contrastive_embed` implementation can support shadow-style diagnostics only if dual fusion text weight is clamped to zero; otherwise `eqtext_enabled=True` can affect the final support path. Strict configs must force `max_text_weight=0.0` for shadow/loss-only variants.
- Script 104 formalized the V33 epoch2 verdict as `FAILED_TEXT_CALIBRATION_EPOCH2`; the failure trips all three AP/detection kill rules: mAP < 0.62, small-vehicle AP < 0.50, and small-vehicle detections > 1.5x FOCUS epoch2.
- V33 is not a final denial of text-side equivariance. It is evidence that the current direct text/logit fusion calibration is unsafe by epoch2.
- The generated strict variants separate the mechanism layers: FOCUS reference, zero text shadow, text shadow with zero fusion, text-anchor loss only, VT-consistency metadata/loss-only config, three tiny-fusion weights, and direction hard prompt as metadata-only negative control.
- The runner deliberately treats the direction hard prompt negative control as `SKIPPED_METADATA_ONLY` rather than inheriting FOCUS metrics, so it cannot be misread as a passing baseline.
- The runner has not executed training; its current result table is a command-ready safety scan/dry-run artifact. Actual TEXT-side runs must be invoked explicitly with `--execute` and should stop at the first kill-rule failure unless `--continue-on-fail` is deliberately passed.

## 2026-06-11 FOCUS-OVD Dual Text GPU67 Findings

- User requested a text-side dual-branch version on physical GPUs 6 and 7, with evaluation fixed to the best checkpoint, then further debugging of the text-fusion route.
- Relevant prior design and implementation plan are `docs/superpowers/specs/2026-06-10-focus-ovd-messdet-dual-text-design.md` and `docs/superpowers/plans/2026-06-10-focus-ovd-messdet-dual-text.md`.
- Existing guardrails from the prior plan still apply: keep FOCUS-OVD/native support as the baseline path, keep dual text bounded as a support-space residual, and avoid direct unsafe V33-style text/logit fusion.
- The GPU67 best config uses the conservative r2 caps: `max_text_weight=0.01`, `alpha_t_max=0.015`, and Mess branch `alpha_m_max=0.015`.
- The training launcher does not evaluate `last_checkpoint`; it requires a `best_*.pth` file and exits if no best checkpoint exists.
- Launch preflight found physical GPUs 6 and 7 free before start. After launch, both GPUs were occupied by OpenRSD Python workers and the trainability audit reported `ok: true` with 23 trainable focus parameters.

## 2026-06-10 FOCUS-T-Safe Findings

- FOCUS-T-Safe is a conservative response to the EQ_V33 direct fusion failure. It must not route text residuals into final logits before zero-disturbance and shadow-signal gates pass.
- The new text branch should operate as a shadow diagnostic over mapped text support: compute Fourier-conditioned text deltas and text geometry, but leave the detector's classification support path untouched.
- Offline calibration, if run, is not training and must be one-way only: it may downweight suspicious small-vehicle scores, but it must never increase small-vehicle detections or scores.
- The existing EQText modules are useful reference implementations, but their dual-fusion path is intentionally too permissive for T-Safe. T-Safe needs separate modules/scripts that make non-interference explicit.
- A prior FOCUS-TAC validation process is active on GPU6/9, so this implementation pass should stay at code, tests, and CPU/offline artifact generation.
- Stage 0 zero-disturbance passed: T-Safe alpha-zero shadow support does not modify final logits, predictions, det/img proxy, small-vehicle det proxy, or the support bank.

## 2026-06-10 Past Two Weeks Summary Findings

- The requested reporting window is `2026-05-27` through `2026-06-10` inclusive.
- The consolidated report should be a new experiment record, not a root-level `resultmd/*.md` file.
- Candidate sources include dated `resultmd/exp_*` directories, root planning records, `experiments/rotation_semantic_attractor` outputs, and dated `work_dirs` logs for FOCUS/EQText/T-Safe runs.
- Numeric tables should be copied or distilled from existing CSV/JSON/Markdown artifacts where possible; rows with blocked or dry-run-only evidence must be labeled as such rather than promoted to benchmark results.
- Final report path: `resultmd/exp_past_two_weeks_experiments_20260610/fres_past_two_weeks_experiments_20260610.md`.
- Report structure check: 27 second-level sections, 33 Markdown tables, and no section without a table.
- Coverage is grouped at experiment-record level; raw scratch folders without durable reports are represented via the ABCD audit/inventory tables rather than expanded one-by-one.
- Stage 1 shadow diagnostic produced no usable discriminative signal: combined AUC is 0.5073, with `eqtext_sv_similarity` AUC 0.5530, `negative_text_margin` AUC 0.4975, and `visual_text_consistency` AUC 0.4404.
- Degenerate/padding safety was not the blocker in this run: degenerate high-score rate is 0.0067 and padding high-score rate is 0.0. The blocker is lack of discriminative signal.
- Category means show only weak separation: annotation-missing true vehicle combined score mean 0.2525 versus corrected false-SV 0.2377; this is not enough for a gate.
- Loss-only smoke remained non-invasive: final logits unchanged, text losses finite, no gradients to the base detector, and text delta norm bounded below 0.01.
- Negative margin loss was disabled in loss-only smoke because the shadow signal did not pass; this preserves the rule that negative margin is not activated without a passing shadow gate.
- Offline tiny calibration was skipped under `--only-if-shadow-signal-pass`, so no score calibration or AP improvement is claimed.
- Text side cannot affect logits in the current FOCUS-T-Safe result, and it should not enter the main method. FOCUS-OVD remains the main method.

## 2026-06-10 Experiment ABCD Audit Findings

- Existing planning files already provide a high-level chronological index of major experiment lines: rotation semantic attractor/full closed-set, full OpenRSD continuation, FOCUS-OVD prototype/scheduled train/module attribution/P0/P1/P1A unblock, FOCUS-EQText short detector run, EQText strict epoch2 audit, and FOCUS-T-Safe.
- The repository worktree is very dirty with many untracked experiment outputs and code additions. This audit must not delete or revert anything; it will only recommend candidates.
- Root-level `resultmd/` contains existing experiment records and should receive the new audit only inside a per-experiment subdirectory, not as a root-level Markdown file.
- Root planning files are currently cross-experiment summaries, so the ABCD audit should reference them as evidence while still scanning actual output directories and logs.
- The lowercase user path `/data1/zcy/openrsd` does not exist; the scanned project root is `/data1/zcy/OpenRSD`.
- Lightweight inventory produced 1141 experiment/run roots: 110 under `work_dirs` (~275.98 GB), 65 under `resultmd` (~52.53 GB), 1 under `experiments` (~16.67 GB), 959 under `SimpleRun/work_dirs` (~0.21 GB), plus small script stubs.
- A-grade core evidence: FOCUS-EQText short detector run, FOCUS-OVD P0/P1/P1A evidence chain, FOCUS-OVD module attribution plus DOTA2 recovery, Rotation Semantic Attractor full closed-set, DeCLIP+CCL negative result, and EQText strict epoch2 audit.
- B-grade recent supporting evidence: FOCUS-T-Safe, FOCUS-TAC, FOCUS-OVD prototype scaffold, rotation semantic open-vocab/dehub scaffolds, and recent false-SV mechanism summaries.
- C-grade historical/provenance material: legacy DOTA1 rotation/TTA, legacy OpenRSD followups, May SV attractor/dehub/shift mechanism runs, and older baseline/adapter runs. Keep summaries/configs; review raw outputs before cleanup.
- D-grade deletion request candidates: empty/never-started roots, `SimpleRun/work_dirs/*` timestamp scratch batch, failed/dry-run raw folders such as `work_dirs/scheduled_runs` and `work_dirs/full_live_gpu_v3_20260509_230014`, and transient queue/log/stub folders.
- Some large folders with high failure-hit counts are not D by default. `work_dirs/exp_mechanism_sv_attractor_gpu89` (~98.5 GB), `work_dirs/rotation_study_36h_20260505_012947` (~19.1 GB), `work_dirs/ss_train_retrain_12angle_no_tta` (~15.1 GB), and `work_dirs/rotation_gpu_batch2_20260507` (~13.3 GB) need human review before raw deletion.
