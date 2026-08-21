#!/usr/bin/env bash
# After preflight OK: run mapping+intervention+bg_gt on GPU9 and dense+postprocess on GPU8 (smoke).
set -euo pipefail
REPO=/data1/zcy/OpenRSD
PYTHON="${PYTHON_BIN:-/data/zcy/anaconda3/envs/openrsd/bin/python}"
WORK_DIR="${1:?Usage: $0 WORK_DIR}"
ANGLES="${ANGLES:-0,45,90}"

export CUDA_VISIBLE_DEVICES=9
"$PYTHON" "$REPO/tools/verify_sv_attractor_gpu89/run_verify_sv_attractor_gpu89.py" \
  --repo-root "$REPO" \
  --work-dir "$WORK_DIR" \
  --result-md-dir "$REPO/resultmd/exp_verify_sv_attractor_gpu89" \
  --gpu 9 \
  --exp mapping,intervention,bg_gt \
  --mode smoke \
  --angles "$ANGLES" \
  --tiles P0148__1024__651___0 &

PID9=$!
export CUDA_VISIBLE_DEVICES=8
"$PYTHON" "$REPO/tools/verify_sv_attractor_gpu89/run_verify_sv_attractor_gpu89.py" \
  --repo-root "$REPO" \
  --work-dir "$WORK_DIR" \
  --result-md-dir "$REPO/resultmd/exp_verify_sv_attractor_gpu89" \
  --gpu 8 \
  --exp dense,postprocess \
  --mode smoke \
  --angles "$ANGLES" \
  --tiles P0148__1024__651___0 &

PID8=$!
wait "$PID9" "$PID8"
echo "Smoke GPU8+GPU9 done. work_dir=$WORK_DIR"
