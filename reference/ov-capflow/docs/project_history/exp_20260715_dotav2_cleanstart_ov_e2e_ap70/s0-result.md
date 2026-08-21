# DOTA-v2 Clean-start OV E2E AP70 — S0 Engineering Smoke

## Frozen protocol

- Branch: `research/dotav2-cleanstart-ov-e2e-ap70`
- Final launch code commit: `69be524da55bf7d8143db98ccc3d9e5a9b869b05`
- Config: `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s0.py`
- Config SHA256: `4471f4bf6c5e817cab8080da9884f692a2bc0aa1086a45f2912faefbab539949`
- Compatible generic checkpoint SHA256: `e4b612bedf7ce0d78064bfacdf5b3641e265aaf1a75c0f3c2d82a924bd337d16`
- Immutable S0/S1 manifest SHA256: `a00b945ddd0a769008d57145feba28382fb9e56f5d6f1427413220f8008e1a2e`
- Data: 96 real DOTA-v2 tiles (64 nonempty, 32 empty), all 18 classes,
  including 8 tiles with more than 200 GT instances.
- Budget: exactly 50 optimizer updates; this is an engineering gate, not a
  scientific AP result.
- Batch/accumulation: 3/1 after the seeded dense first batch made batch 8 OOM;
  each train-loop iteration still performs one optimizer step.
- Inference contract: open-vocabulary token similarity, fixed Q600, 5-D
  rotated boxes, no top-k, no NMS, no crop postprocessing.

## Command

```bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=4 /data/zcy/anaconda3/envs/mmdet/bin/python tools/train.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s0.py --work-dir work_dirs/dotav2_cleanstart/s0_control --cfg-options resume=False
```

## Pre-registered results

| Check | Result |
|---|---:|
| Batch 1 peak allocated MiB | 4483.3 |
| Batch 2 peak allocated MiB | 8272.3 |
| Batch 3 peak allocated MiB | 12033.8 |
| Batch 8 ordinary preflight peak allocated MiB | 30998.1 |
| Batch 8 seeded dense launch | OOM before update 1; 40.26 GiB allocated / 42.10 GiB reserved |
| Selected batch / accumulation | 3 / 1 |
| Completed updates | 50 / 50 |
| Finite loss range | 50.4284–56.5499 across 10 logged windows; final 50.9032 |
| Maximum logged training memory MiB | 19825 |
| S0 diagnostic mAP / AP50 | 0.0051 / 0.0050 (not a scientific candidate result) |
| Saved checkpoint SHA256 | `ccb0d561861c1be293fded72f5d2b1dc8b4ca34063bbf2a3695a27ed0394616d` |
| Strict 600-row audit | pass; static/runtime forbidden calls 0 |
| Open-vocabulary prompt audit | pass; all four variants 600 rows/image; shape invariant |
| Decoder alignment audit | pass; same order, maximum absolute error 0.0 |
| Portable tests | 121 passed, 13 opt-in skipped |
| S0 engineering gate | **PASS** |

## Evidence paths

- Training log: `work_dirs/dotav2_cleanstart/s0_control/20260715_193106/vis_data/20260715_193106.json`
- Checkpoint: `work_dirs/dotav2_cleanstart/s0_control/iter_50.pth`
- Strict audit: `work_dirs/dotav2_cleanstart/audits/s0_strict_inference.json`
- Open-vocabulary audit: `work_dirs/dotav2_cleanstart/audits/s0_open_vocabulary.json`
- Decoder alignment audit: `work_dirs/dotav2_cleanstart/audits/s0_decoder_alignment.json`
