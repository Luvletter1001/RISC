# Rotation Stage Diagnostics

Diagnose where rotated OpenRSD inputs start collapsing toward
`small-vehicle`: backbone, neck, head feature branch, cls logits, or final
detection/NMS.

The probe script reuses the same prompt-style inference path as
`SimpleRun/step1_inference.py`: it builds the OpenRSD runner from the requested
config/checkpoint, calls `model.prompt_extract_feats()`, then calls
`model.prompt_predict()`. It registers hooks from actual `model.named_modules()`
and saves module inventory to:

```text
resultmd/exp_rotation_stage_probe_P0148/fres_module_inventory.md
```

Intermediate tensors are not saved in full. The probe saves scalar statistics,
global pooled vectors, channel-mean heatmaps, and downsampled feature maps for
rotation-aligned comparison.

Use the OpenRSD conda environment interpreter on this server:

```bash
PY=/data/zcy/anaconda3/envs/openrsd/bin/python
```

## Smoke Test

```bash
cd /data1/zcy/OpenRSD
PY=/data/zcy/anaconda3/envs/openrsd/bin/python

CUDA_VISIBLE_DEVICES=8 $PY tools/rotation_diagnostics/probe_rotated_stage_outputs.py \
  --config M_configs/Step2_A10_Large_Pretrain_Stage3/A10_flex_rtm_v3_1_formal.py \
  --checkpoint results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth \
  --image-dir vis/P0148__1024__651___0/dataset/images \
  --angles 0 5 \
  --out-dir work_dirs/rotation_stage_probe_P0148_smoke

$PY tools/rotation_diagnostics/analyze_rotated_stage_outputs.py \
  --probe-dir work_dirs/rotation_stage_probe_P0148_smoke \
  --out-md resultmd/exp_rotation_stage_probe_P0148/fres_stage_probe_smoke_test.md
```

## Key Angles

```bash
cd /data1/zcy/OpenRSD
PY=/data/zcy/anaconda3/envs/openrsd/bin/python

CUDA_VISIBLE_DEVICES=8 $PY tools/rotation_diagnostics/probe_rotated_stage_outputs.py \
  --config M_configs/Step2_A10_Large_Pretrain_Stage3/A10_flex_rtm_v3_1_formal.py \
  --checkpoint results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth \
  --image-dir vis/P0148__1024__651___0/dataset/images \
  --angles 0 5 10 45 90 180 270 \
  --out-dir work_dirs/rotation_stage_probe_P0148_key_angles

$PY tools/rotation_diagnostics/analyze_rotated_stage_outputs.py \
  --probe-dir work_dirs/rotation_stage_probe_P0148_key_angles \
  --out-md resultmd/exp_rotation_stage_probe_P0148/fres_stage_probe_key_angles.md
```

## Full 72 Angles

Single process:

```bash
cd /data1/zcy/OpenRSD
PY=/data/zcy/anaconda3/envs/openrsd/bin/python

CUDA_VISIBLE_DEVICES=8,9 $PY tools/rotation_diagnostics/probe_rotated_stage_outputs.py \
  --config M_configs/Step2_A10_Large_Pretrain_Stage3/A10_flex_rtm_v3_1_formal.py \
  --checkpoint results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth \
  --image-dir vis/P0148__1024__651___0/dataset/images \
  --angle-step 5 \
  --out-dir work_dirs/rotation_stage_probe_P0148_full

$PY tools/rotation_diagnostics/analyze_rotated_stage_outputs.py \
  --probe-dir work_dirs/rotation_stage_probe_P0148_full \
  --out-md resultmd/exp_rotation_stage_probe_P0148/fres_rotation_stage_probe_summary.md
```

Two-process split:

```bash
cd /data1/zcy/OpenRSD
PY=/data/zcy/anaconda3/envs/openrsd/bin/python

CUDA_VISIBLE_DEVICES=8 $PY tools/rotation_diagnostics/probe_rotated_stage_outputs.py \
  --config M_configs/Step2_A10_Large_Pretrain_Stage3/A10_flex_rtm_v3_1_formal.py \
  --checkpoint results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth \
  --image-dir vis/P0148__1024__651___0/dataset/images \
  --angles 0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150 155 160 165 170 175 \
  --out-dir work_dirs/rotation_stage_probe_P0148_part0 &

CUDA_VISIBLE_DEVICES=9 $PY tools/rotation_diagnostics/probe_rotated_stage_outputs.py \
  --config M_configs/Step2_A10_Large_Pretrain_Stage3/A10_flex_rtm_v3_1_formal.py \
  --checkpoint results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth \
  --image-dir vis/P0148__1024__651___0/dataset/images \
  --angles 180 185 190 195 200 205 210 215 220 225 230 235 240 245 250 255 260 265 270 275 280 285 290 295 300 305 310 315 320 325 330 335 340 345 350 355 \
  --out-dir work_dirs/rotation_stage_probe_P0148_part1 &

wait
```

The split outputs can be analyzed separately, or merged by copying the per-angle
files into one probe directory before running the analysis script. The analysis
script can do that merge directly:

```bash
$PY tools/rotation_diagnostics/analyze_rotated_stage_outputs.py \
  --merge-probe-dirs \
    work_dirs/rotation_stage_probe_P0148_part0 \
    work_dirs/rotation_stage_probe_P0148_part1 \
  --probe-dir work_dirs/rotation_stage_probe_P0148_full_gpu8_9 \
  --out-md resultmd/exp_rotation_stage_probe_P0148/fres_rotation_stage_probe_summary.md
```
