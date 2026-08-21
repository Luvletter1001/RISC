# DOTA-v2 Clean-start OV E2E AP70 — Locked S1 Screen

## Immutable evidence

- Protocol commit before freeze: `39bde54d0bfc9f79ceb490bebf3cb19af5b9384b`
- Random seed: `20260715`
- Generic compatible checkpoint SHA256:
  `e4b612bedf7ce0d78064bfacdf5b3641e265aaf1a75c0f3c2d82a924bd337d16`
- Checkpoint provenance JSON SHA256:
  `d73f9221b893e9248723fd7ad5b660e961b6ae63d9e41eeb7813c226c784b4ed`
- Canonical subset manifest SHA256:
  `a00b945ddd0a769008d57145feba28382fb9e56f5d6f1427413220f8008e1a2e`
- Frozen manifest copy:
  `docs/project_history/exp_20260715_dotav2_cleanstart_ov_e2e_ap70/subset-manifest.json`
- Train/validation: 1,600/400 immutable real DOTA-v2 tiles, with empty tiles
  retained and no pseudo labels.
- Density quotas over all 2,000 tiles: 600 empty, 900 with 1–10 GT, 300
  with 11–50 GT, 150 with 51–200 GT, and 50 with more than 200 GT.
- Classes: all 18 for training and primary validation. Novel prompt holdout is
  `airport`, `container-crane`, `helipad`, `helicopter`; the other 14 classes
  form the base prompt set.
- Schedule: 12 epochs, validation every epoch, linear warmup plus cosine,
  batch 4 / accumulation 2 / effective batch 8 on one A40 per candidate.
- Inference contract: primary Q600 only, token-similarity classification,
  5-D rotated boxes, same query order, no top-k/NMS/crop postprocessing.

## Locked candidates

| ID | Candidate | Config SHA256 | Only intervention | GPU |
|---|---|---|---|---:|
| 8-S1-C | Q600 control | `d5fea58428fe50a018e6190463194501493f44710a5ade24b51eee9381b41f67` | none | 4 |
| 8-S1-G | grouped O2O | `bf30033d9078f522821b5bdf454f65deb7c9c0dc576303c59934b92220635c1a` | three training-only shared query groups | 5 |
| 8-S1-H | Hausdorff-DN | `b05fbb91f19d440cf008579f202ad641839db4408660e63a6157de4f545f8939` | Hausdorff matching, periodic angle noise, adaptive positive-Hungarian DN | 6 |

Batch-4 real forward/backward preflights passed with peak allocated memory
16,169.7 MiB (C), 20,396.2 MiB (G), and 15,804.8 MiB (H). The three runs are
independent and execute concurrently; no candidate consumes another
candidate's checkpoint.

## Promotion rule

Every candidate must first pass finite training, checkpoint reload, strict
Q600 row count, open-vocabulary prompt-shape, decoder alignment, and zero
forbidden-call audits. Relative to 8-S1-C, an intervention is promotable only
if either:

1. AP50 improves by at least `+0.020`; or
2. AP50 improves by at least `+0.010` and GT coverage improves by at least
   `+0.020`.

In both cases duplicate extras per GT may rise by no more than 10%. Among valid
promotable candidates, select the highest AP50. If neither intervention passes,
the control may enter only a short full-data diagnostic, not a 24-epoch claim.

## Commands

```bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=4 /data/zcy/anaconda3/envs/mmdet/bin/python tools/train.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_control.py --work-dir work_dirs/dotav2_cleanstart/s1_control --cfg-options resume=False
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=5 /data/zcy/anaconda3/envs/mmdet/bin/python tools/train.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped.py --work-dir work_dirs/dotav2_cleanstart/s1_grouped --cfg-options resume=False
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=6 /data/zcy/anaconda3/envs/mmdet/bin/python tools/train.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_hausdorff_dn.py --work-dir work_dirs/dotav2_cleanstart/s1_hausdorff_dn --cfg-options resume=False
```

## Pre-registered results

| Metric | 8-S1-C | 8-S1-G | 8-S1-H |
|---|---:|---:|---:|
| Completion / best epoch | complete / 12 | complete / 12 | complete / 12 |
| all-18 mAP | 0.3806 | 0.4068 | 0.3409 |
| all-18 AP50 | 0.3810 | 0.4070 | 0.3410 |
| base-14 AP50 | 0.415 | 0.422 | 0.383 |
| novel-4 AP50 | 0.125 | 0.030 | 0.065 |
| GT coverage | 0.504970 | 0.496848 | 0.484970 |
| Duplicate extras / GT | 0.971030 | 0.890061 | 0.952970 |
| Fixed rows / image | 600 | 600 | 600 |
| Maximum GPU memory MiB | 28,981 | 32,831 | 28,980 |
| Strict / OV / alignment audits | pass | pass | pass |
| Promotion rule | control | pass (+0.0260) | fail (-0.0400) |

## Decision

Winner: `8-S1-G` grouped O2O under the frozen seed-20260715 primary rule.
The follow-up Chamfer candidate is lower than the control. The matched
seed-20260716 repeat shows only a `+0.0040` grouped AP50 gain, so the selected
intervention is explicitly seed-sensitive even though it remains the sole
primary candidate that clears the pre-registered promotion gate.

## Seed20260716 repeat recovery note

The grouped repeat stopped in epoch 12 with MMEngine's
`loss_factor should be larger than zero` assertion. This was an engineering
failure rather than a model or numerical failure: the query-budget batch
sampler produced deterministic epoch lengths
`402,403,403,402,402,403,402,402,402,403,403,403` (4,830 total), while the
stock epoch loop fixed its optimizer budget to `402 * 12 = 4,824`. The first
11 epochs contain exactly 4,427 completed iterations in `epoch_11.pth`; the
six-iteration underestimate caused the assertion near the end of epoch 12.

`VariableBatchEpochBasedTrainLoop` now enumerates every deterministic epoch
plan before training and sets the exact optimizer iteration budget without
changing data order, losses, learning rates, effective batch size, or model
code. Because epoch 11 ended at odd global iteration 4,427, its checkpoint
does not contain the pending half accumulation. Recovery therefore uses the
last exact accumulation boundary, epoch 10 at iteration 4,024, and replays
complete epochs 11–12. Its result remains pending until validation and audits
pass.

The clean recovery completed all 4,830 planned iterations with exit code 0.
Epoch 11 remained the best at `dota/mAP=0.3837`, `dota/AP50=0.3840`; epoch 12
finished at `0.3833/0.3830`. The recovered best checkpoint SHA256 is
`65312a3a0d3144ffd6490658d80547542ec1dd074ba7f029a286eb892c560716`.
Fresh audits against that exact file pass checkpoint reload, strict Q600,
open-vocabulary prompt-shape, and decoder-alignment checks. Across 400 images,
GT coverage is `0.508606`, duplicate extras/GT is `1.097939`, and empty-image
foreground score mass is `6.803284`; base-14/novel-4 AP50 is `0.443/0.125`.
Artifacts are isolated under
`work_dirs/dotav2_cleanstart/audits/s1_grouped_seed20260716_recovered_final_e11/`.
This repeat is robustness evidence only and does not alter the frozen
seed-20260715 primary ranking.

The matched seed-20260716 control completed at `dota/mAP=0.3802`,
`dota/AP50=0.3800`; its exact checkpoint SHA256 is
`76a0b7e3e0c8ecb4b6a4b01f65629d1c05b17ec0c936d2eb9d3cff2f8d808ece`.
Fresh audits pass reload, strict Q600, open-vocabulary prompt-shape, and
decoder alignment. Coverage is `0.510667`, duplicate extras/GT is `1.119152`,
empty foreground mass is `6.303545`, and base-14/novel-4 AP50 is
`0.434/0.079`. Against this matched control, grouped improves all-18 AP50 by
only `+0.0040`, base-14 by `+0.009`, and novel-4 by `+0.046`, while coverage
changes by `-0.002061` and duplicate extras/GT by `-0.021212`. This repeat
does not independently pass the promotion threshold and is recorded as a
seed-sensitivity warning rather than used to rewrite the frozen primary rank.

## Chamfer follow-up endpoint

The isolated four-corner Chamfer candidate completed all 12 epochs. Its final
and best checkpoint is epoch 12 at `dota/mAP=0.3755`, `dota/AP50=0.3750`,
which is `-0.0060` below the frozen control and `-0.0320` below grouped O2O.
The exact checkpoint SHA256 is
`5d2176a8c4494de3557774ee73942dcaa4d8f779b8dfd0e5255188fddff4369e`.
Fresh final-file audits pass reload, strict Q600, open-vocabulary prompt-shape,
and decoder alignment with zero forbidden calls and zero box error. Coverage
is `0.490667`, duplicate extras/GT is `0.646667`, empty-image foreground mass
is `4.805809`, and base-14/novel-4 AP50 is `0.412/0.077`. Chamfer substantially
reduces duplicate predictions but fails the frozen AP promotion gate and is
discarded.

## Raw-13,833 diagnostic

The frozen seed-20260715 grouped S1 best was evaluated without updates on the
full raw `ss_val` mouth (`13,833` images, empty tiles retained) using two GPUs.
The run completed with exit code 0 at `dota/mAP=0.3094`, `dota/AP50=0.3090`,
versus `0.4068/0.4070` on the 400-image S1 screen. This `-0.0980` AP50 transfer
gap is diagnostic evidence only: it does not alter S1 ranking, but confirms
that the AP60 path must rely on full 47,294-image training rather than the
small-screen validation mouth. Monitoring collected 53 samples with no fatal
pattern. Additional single-GPU raw diagnostics find base-14/novel-4 AP50 of
`0.360/0.049`, GT coverage `0.463572`, duplicate extras/GT `0.872656`, and
empty-image foreground score mass `8.412324`, always with exactly 600 rows per
image. The transfer deficit therefore combines reduced coverage, weak novel
classes, and worse empty-tile calibration rather than a metric-mouth artifact.
