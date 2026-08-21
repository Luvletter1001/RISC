#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/data1/zcy/OpenRSD}"
RUNNER="${RUNNER:-$ROOT_DIR/M_Tools/schedule/run_rotation_research_full_8gpu.sh}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/work_dirs/scheduled_runs/rotation_research_full_8gpu_manual}"
MAX_WATCHDOG_RESTARTS="${MAX_WATCHDOG_RESTARTS:-20}"
WATCHDOG_SLEEP_SECONDS="${WATCHDOG_SLEEP_SECONDS:-120}"
WATCHDOG_LOG="$OUT_ROOT/watchdog.log"

main() {
  cd "$ROOT_DIR" || exit 1
  rtk /usr/bin/mkdir -p "$OUT_ROOT" || exit 1
  local attempt=0
  local code=0
  while [[ "$attempt" -lt "$MAX_WATCHDOG_RESTARTS" ]]; do
    attempt=$((attempt + 1))
    printf '[%s] starting runner attempt=%s\n' "$(rtk /usr/bin/date '+%F %T')" "$attempt" >> "$WATCHDOG_LOG"
    rtk env \
      ROOT_DIR="$ROOT_DIR" \
      OUT_ROOT="$OUT_ROOT" \
      GPUS="${GPUS:-0,1,4,5,6,7,8,9}" \
      NPROC_PER_NODE="${NPROC_PER_NODE:-8}" \
      BATCH_CANDIDATES="${BATCH_CANDIDATES:-16 8 4 2 1}" \
      NUM_WORKERS="${NUM_WORKERS:-8}" \
      PYTHON_BIN="${PYTHON_BIN:-/data/zcy/anaconda3/envs/openrsd/bin/python}" \
      RESUME=1 \
      DRY_RUN="${DRY_RUN:-0}" \
      PREPARE_ANGLES="${PREPARE_ANGLES:-1}" \
      RUN_ANGLE_SWEEP="${RUN_ANGLE_SWEEP:-1}" \
      RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}" \
      RUN_TTA="${RUN_TTA:-1}" \
      DIAGNOSTIC_MAX_DETS_PER_IMG="${DIAGNOSTIC_MAX_DETS_PER_IMG:-300}" \
      TTA_PRE_NMS_TOPK="${TTA_PRE_NMS_TOPK:-4000}" \
      TTA_SCORE_THR="${TTA_SCORE_THR:-0.05}" \
      TTA_NMS_IOU="${TTA_NMS_IOU:-0.1}" \
      bash "$RUNNER"
    code=$?
    printf '[%s] runner_exit=%s attempt=%s\n' "$(rtk /usr/bin/date '+%F %T')" "$code" "$attempt" >> "$WATCHDOG_LOG"
    if [[ "$code" -eq 0 ]]; then
      exit 0
    fi
    rtk /usr/bin/sleep "$WATCHDOG_SLEEP_SECONDS"
  done
  exit "$code"
}

main "$@"
