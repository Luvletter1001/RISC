#!/usr/bin/env bash
# Smoke preflight on physical GPU 9 (openrsd env).
set -euo pipefail
REPO=/data1/zcy/OpenRSD
PYTHON="${PYTHON_BIN:-/data/zcy/anaconda3/envs/openrsd/bin/python}"
TS=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${WORK_DIR:-$REPO/work_dirs/verify_sv_attractor_gpu89_${TS}}"
export CUDA_VISIBLE_DEVICES=9
exec "$PYTHON" "$REPO/tools/verify_sv_attractor_gpu89/run_verify_sv_attractor_gpu89.py" \
  --repo-root "$REPO" \
  --config "$REPO/M_configs/Step2_A10_Large_Pretrain_Stage3/A10_flex_rtm_v3_1_formal.py" \
  --checkpoint "$REPO/results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth" \
  --result-md-dir "$REPO/resultmd/exp_verify_sv_attractor_gpu89" \
  --work-dir "$WORK_DIR" \
  --gpu 9 \
  --exp preflight \
  --mode smoke \
  "$@"
