#!/usr/bin/env bash
set -euo pipefail
REPO="/data1/zcy/OpenRSD"
PY="/data/zcy/anaconda3/envs/openrsd/bin/python"
TOOL="$REPO/tools/exp_formal_success_check_gpu89"
RES="$REPO/resultmd/exp_formal_success_check_gpu89"
mkdir -p "$RES" "$REPO/work_dirs/exp_formal_success_check_gpu89"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$RES/log_pipeline_gpu89.txt"; }

log "preflight"
"$PY" "$TOOL/run_preflight.py"
"$PY" "$TOOL/import_dehub_heldout.py"

log "GPU8 class drift (background)"
CUDA_VISIBLE_DEVICES=8 nohup "$PY" "$TOOL/run_class_drift_gpu8.py" --gpu 8 \
  >> "$RES/log_gpu8_class_drift.txt" 2>&1 &
PID8=$!

log "GPU9 step23 AP (background)"
CUDA_VISIBLE_DEVICES=9 nohup "$PY" "$TOOL/run_step23_gpu9.py" --task all --gpu 9 --heldout-n 500 \
  >> "$RES/log_gpu9_step23_ap.txt" 2>&1 &
PID9=$!

log "wait GPU8=$PID8 GPU9=$PID9"
wait "$PID8" || log "GPU8 drift exit $?"
wait "$PID9" || log "GPU9 step23 exit $?"

log "build final report"
"$PY" "$TOOL/build_final_report.py" >> "$RES/log_build_report.txt" 2>&1

log "pipeline done"
