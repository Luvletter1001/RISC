#!/usr/bin/env bash
set -euo pipefail
REPO="/data1/zcy/OpenRSD"
PY="/data/zcy/anaconda3/envs/openrsd/bin/python"
TOOL="$REPO/tools/exp_formal_success_check_gpu89"
RES="$REPO/resultmd/exp_formal_success_check_gpu89"
mkdir -p "$RES" "$REPO/work_dirs/exp_formal_success_check_gpu89"

exec > >(tee -a "$RES/log_pipeline_formal_check.txt") 2>&1
echo "[$(date -Iseconds)] pipeline start"

"$PY" "$TOOL/run_preflight.py"
"$PY" "$TOOL/import_dehub_heldout.py"

CUDA_VISIBLE_DEVICES=8 nohup "$PY" "$TOOL/run_class_drift_gpu8.py" --gpu 8 \
  >> "$RES/log_gpu8_class_drift.txt" 2>&1 &
PID8=$!
echo "GPU8 drift pid=$PID8"

CUDA_VISIBLE_DEVICES=9 nohup "$PY" "$TOOL/run_step23_gpu9.py" --task all --gpu 9 --heldout-n 500 \
  >> "$RES/log_gpu9_step23_ap.txt" 2>&1 &
PID9=$!
echo "GPU9 step23 pid=$PID9"

wait "$PID8" || echo "GPU8 exit $?"
wait "$PID9" || echo "GPU9 exit $?"

"$PY" "$TOOL/build_final_report.py"
echo "[$(date -Iseconds)] pipeline done"
