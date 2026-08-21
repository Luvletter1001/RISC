#!/usr/bin/env bash
# Smoke repair: GPU8 R1+R2, GPU9 R3+eval (same WORK_DIR).
set -euo pipefail
REPO=/data1/zcy/OpenRSD
PY="${PYTHON_BIN:-/data/zcy/anaconda3/envs/openrsd/bin/python}"
TS=$(date +%Y%m%d_%H%M%S)
WORK="${WORK_DIR:-$REPO/work_dirs/sv_attractor_repair_gpu89_${TS}}"
VERIFY="$REPO/work_dirs/verify_sv_attractor_gpu89_20260519_113039"
COMMON=(
  --repo-root "$REPO"
  --config "$REPO/M_configs/Step2_A10_Large_Pretrain_Stage3/A10_flex_rtm_v3_1_formal.py"
  --checkpoint "$REPO/results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth"
  --support-pkl "$REPO/data/DOTA1_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl"
  --verify-work-dir "$VERIFY"
  --work-dir "$WORK"
  --result-md-dir "$REPO/resultmd/exp_sv_attractor_repair_gpu89"
  --mode smoke
  --train-max-images 128
  --heldout-max-images 64
  --eval-max-images 64
  --angles 0,90
  --iters 300
  --batch-size 1
)

export CUDA_VISIBLE_DEVICES=8
"$PY" "$REPO/tools/sv_attractor_repair_gpu89/run_sv_attractor_repair_gpu89.py" \
  "${COMMON[@]}" --gpu 8 --exp preflight,baseline,r1_calib,r2_negative &
PID8=$!

export CUDA_VISIBLE_DEVICES=9
"$PY" "$REPO/tools/sv_attractor_repair_gpu89/run_sv_attractor_repair_gpu89.py" \
  "${COMMON[@]}" --gpu 9 --exp preflight,r3_antihub,combined,eval,visual,summary --resume &
PID9=$!

wait "$PID8" "$PID9"
echo "WORK_DIR=$WORK"
