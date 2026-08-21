#!/usr/bin/env bash
# Hourly patrol loop for SV-DeHub-Lite v1 (run in nohup).
set -euo pipefail
REPO="/data1/zcy/OpenRSD"
PY="/data/zcy/anaconda3/envs/openrsd/bin/python"
TOOL="$REPO/tools/exp_sv_dehub_lite_train_gpu89"
RESULT="$REPO/resultmd/exp_sv_dehub_lite_train_gpu89"
WORK="$REPO/work_dirs/exp_sv_dehub_lite_train_gpu89/train_v1"
LOG="$RESULT/log_hourly_monitor_loop.txt"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RESULT"

exec >>"$LOG" 2>&1
echo "hourly loop start $(date -Iseconds)"
while true; do
  "$PY" "$TOOL/hourly_monitor_sv_dehub.py" \
    --project-dir "$REPO" \
    --result-dir "$RESULT" \
    --work-dir "$WORK" \
    --debug-if-crashed --resume-if-fixed
  sleep 3600
done
