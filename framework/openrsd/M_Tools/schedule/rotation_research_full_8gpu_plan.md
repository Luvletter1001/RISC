# Full Rotation Robustness Experiment Plan

This run targets the research question:

> Do oriented remote-sensing detectors preserve object identity under whole-image rotation, or do they rely on common geographic/object pose shortcuts?

## Scope

- GPUs: `0,1,4,5,6,7,8,9` by default.
- Distributed eval: `NPROC_PER_NODE=8`.
- Per-GPU batch fallback: `16 -> 8 -> 4 -> 2 -> 1`.
- NCCL guards: `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`.
- The runner waits for the selected GPUs to have no compute processes before each GPU task.
- All results are continuously written to one new Markdown file:
  `work_dirs/scheduled_runs/rotation_research_full_8gpu_<timestamp>/rotation_research_full_results.md`.

## Models

| model | config | checkpoint |
|---|---|---|
| ReDet_Re50_epoch4 | `M_configs/DOTA2OfficialAdapters/redet_re50_refpn_dotav2_fullinit.py` | `work_dirs/dotav2_official_adapters/redet_re50_refpn_fullinit/epoch_4.pth` |
| ORCNN_R50_epoch8 | `M_configs/DOTA2OfficialAdapters/oriented_rcnn_r50_fpn_dotav2_fullinit.py` | `work_dirs/dotav2_official_adapters/oriented_rcnn_r50_fpn_fullinit/epoch_8.pth` |
| R3Det_KFIoU_R50_epoch12 | `M_configs/DOTA2OfficialAdapters/r3det_kfiou_r50_fpn_dotav2_fullinit.py` | `work_dirs/dotav2_official_adapters/r3det_kfiou_r50_fpn_fullinit/epoch_12.pth` |

## Experiments

1. Prepare angle-sweep DOTA2 validation splits.
   - Angles: `0,15,30,...,345`.
   - Existing 30-degree data is reused with hardlinks by default.
   - Missing 15-degree-offset splits are generated from `ss_val`.

2. Angle response curve.
   - Runs all three models on all 24 angles.
   - Records `mAP`, `AP50`, checkpoint, predictions, logs.
   - The Markdown report computes:
     - `mAP@0`
     - mean rotated mAP
     - rotation gap
     - worst-angle mAP
     - best-angle mAP
     - RSI
     - right-angle mean
     - non-right-angle mean

3. Classification-vs-localization collapse diagnostics.
   - Uses saved prediction pkls.
   - Writes `diagnostics/rotation_detection_diagnostics.tsv`.
   - Reports recall at IoU 0.5, class-wrong rate, localization-fail rate, missed rate, and false positives per image.

4. Rotation TTA upper bound.
   - `tta_4angle`: `0,90,180,270`.
   - `tta_8angle`: `0,45,90,135,180,225,270,315`.
   - Predictions are inverse-rotated back to `ss_val`, merged by rotated NMS, then evaluated offline.

## Failure Handling

- Every task is resumable through `results.tsv`.
- Completed `OK` rows are skipped on restart.
- Failed GPU evals retry with smaller batch sizes.
- CUDA illegal-memory failures retry with `CUDA_LAUNCH_BLOCKING=1` on the next attempt.
- Missing angle-data failures trigger angle-split preparation again before retry.
- A watchdog tmux wrapper restarts the runner with the same `OUT_ROOT`, so it continues from the last completed task.
- Failures and skips are preserved in the same Markdown report with log paths and diagnosis tags.

## Entry Points

Dry run:

```bash
rtk env DRY_RUN=1 bash M_Tools/schedule/start_rotation_research_full_8gpu_tmux.sh
```

Start real run:

```bash
rtk bash M_Tools/schedule/start_rotation_research_full_8gpu_tmux.sh
```

Attach:

```bash
rtk tmux attach -t rotation_research_full_8gpu
```
