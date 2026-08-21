#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/data1/zcy/OpenRSD}"
RUNNER="${RUNNER:-$ROOT_DIR/M_Tools/schedule/run_rotation_research_full_8gpu.sh}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/work_dirs/scheduled_runs/rotation_research_full_8gpu_20260504_014158}"
GPUS="${GPUS:-0,1,4,5,6,7,8,9}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PYTHON_BIN="${PYTHON_BIN:-/data/zcy/anaconda3/envs/openrsd/bin/python}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
SLEEP_SECONDS="${SLEEP_SECONDS:-120}"
TTA_PRE_NMS_TOPK="${TTA_PRE_NMS_TOPK:-1000}"
DIAGNOSTIC_MAX_DETS_PER_IMG="${DIAGNOSTIC_MAX_DETS_PER_IMG:-50}"
LOG="$OUT_ROOT/gpu_rerun_then_fill_watchdog.log"

run_phase() {
  local phase="$1"
  shift
  printf '[%s] phase_start=%s\n' "$(rtk /usr/bin/date '+%F %T')" "$phase" >> "$LOG"
  rtk env \
    ROOT_DIR="$ROOT_DIR" \
    OUT_ROOT="$OUT_ROOT" \
    GPUS="$GPUS" \
    NPROC_PER_NODE="$NPROC_PER_NODE" \
    NUM_WORKERS="$NUM_WORKERS" \
    PYTHON_BIN="$PYTHON_BIN" \
    "$@" \
    bash "$RUNNER"
  local code=$?
  printf '[%s] phase_exit=%s code=%s\n' "$(rtk /usr/bin/date '+%F %T')" "$phase" "$code" >> "$LOG"
  return "$code"
}

main() {
  cd "$ROOT_DIR" || exit 1
  rtk /usr/bin/mkdir -p "$OUT_ROOT" || exit 1
  local attempt=0
  local code=0
  while [[ "$attempt" -lt "$MAX_RESTARTS" ]]; do
    attempt=$((attempt + 1))
    printf '[%s] attempt_start=%s\n' "$(rtk /usr/bin/date '+%F %T')" "$attempt" >> "$LOG"

    run_phase gpu_angle_sweep_rerun \
      RESUME=1 \
      PREPARE_ANGLES=0 \
      RUN_ANGLE_SWEEP=1 \
      RUN_DIAGNOSTICS=0 \
      RUN_TTA=0 \
      WAIT_FOR_FREE_GPUS=1 \
      CHECK_GPUS_EACH_TASK=1 \
      FORCE_RERUN_STAGES=angle_sweep
    code=$?
    if [[ "$code" -ne 0 ]]; then
      printf '[%s] retry_after_gpu_phase code=%s\n' "$(rtk /usr/bin/date '+%F %T')" "$code" >> "$LOG"
      rtk /usr/bin/sleep "$SLEEP_SECONDS"
      continue
    fi

    run_phase fill_na_postprocess \
      RESUME=1 \
      PREPARE_ANGLES=0 \
      RUN_ANGLE_SWEEP=0 \
      RUN_DIAGNOSTICS=1 \
      RUN_TTA=1 \
      WAIT_FOR_FREE_GPUS=0 \
      CHECK_GPUS_EACH_TASK=0 \
      DIAGNOSTIC_MAX_DETS_PER_IMG="$DIAGNOSTIC_MAX_DETS_PER_IMG" \
      TTA_PRE_NMS_TOPK="$TTA_PRE_NMS_TOPK"
    code=$?
    if [[ "$code" -eq 0 ]]; then
      printf '[%s] all_done attempt=%s\n' "$(rtk /usr/bin/date '+%F %T')" "$attempt" >> "$LOG"
      exit 0
    fi
    printf '[%s] retry_after_postprocess code=%s\n' "$(rtk /usr/bin/date '+%F %T')" "$code" >> "$LOG"
    rtk /usr/bin/sleep "$SLEEP_SECONDS"
  done
  exit "$code"
}

main "$@"
