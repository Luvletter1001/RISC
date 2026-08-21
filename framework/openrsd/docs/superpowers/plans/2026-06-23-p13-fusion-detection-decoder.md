# P13 Fusion Detection Decoder Smoke

## Goal

Rename the planned B-route experiment line to `P13` and start an HRRSD 3-epoch smoke sequence. P13 tests whether support fusion can directly produce classification logits and box/angle outputs, leaving the existing MMRotate head path only as a thin loss/decode wrapper.

## Design

Implement `P13FusionDecoderRotatedRTMDetSepBNHead` as a new head file, separate from SGP. It will not use Gaussian product posterior logits. Instead, it builds a dense support-conditioned decoder:

- project dense detection features into query embeddings;
- compare query embeddings with learnable class support tokens for classification logits;
- add optional objectness bias to logits;
- predict bounded support-conditioned residuals for box distances and angle;
- reuse RTMDet training assignment, loss, and prediction wrappers for the smoke only.

## Smoke Matrix

All runs use HRRSD internal split under `/data1/zcy/datasets/HRRSD_800_0/internal_split_20260619/` and 3 epochs.

- `p13c_dense_fdd_safe`: conservative objectness and small box/angle residuals.
- `p13c_dense_fdd_box`: stronger box/angle residuals.
- `p13c_dense_fdd_obj`: stronger objectness branch.
- `p13c_dense_fdd_nobox`: classification-only decoder control, no box/angle residuals.

## Implementation Steps

1. Add the P13 head implementation under `M_AD/models/dense_heads/`.
2. Add unit tests for support-token initialization and forward tensor shapes/debug flags.
3. Add four HRRSD P13 config files under `M_configs/Diagnostics/`.
4. Add a P13 single-GPU train/eval/risk launcher and use four GPUs in parallel.
5. Run syntax/config/test checks before launching the smoke sequence.
6. Record launch state and later results under `resultmd/exp_p13_fusion_decoder_hrrsd/`.

## Acceptance Gate

P13 is considered worth extending only if at least one variant reaches near-baseline mAP directionally in 3 epochs, or shows a clear mAP/risk improvement over SGP-v1/v2 while keeping train/eval stable.
