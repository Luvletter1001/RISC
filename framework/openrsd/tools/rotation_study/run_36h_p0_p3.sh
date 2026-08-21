#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/data1/zcy/OpenRSD}"
PYTHON_BIN="${PYTHON_BIN:-/data/zcy/anaconda3/envs/openrsd/bin/python}"
RUN_TS="${RUN_TS:-$(rtk /usr/bin/date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/work_dirs/rotation_study_36h_${RUN_TS}}"
GPUS="${GPUS:-0,1,4,5,6,7,8,9}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TOTAL_TIMEOUT="${TOTAL_TIMEOUT:-36h}"
RESULT_MD="${RESULT_MD:-$OUT_ROOT/rotation_study_36h_results.md}"
RUN_LOG="$OUT_ROOT/run_36h.log"

log() {
  local line
  line="[$(rtk /usr/bin/date '+%F %T')] $*"
  printf '%s\n' "$line"
  printf '%s\n' "$line" >> "$RUN_LOG"
}

refresh_md() {
  rtk env PYTHONPATH="$ROOT_DIR:$ROOT_DIR/tools" "$PYTHON_BIN" \
    tools/rotation_study/summarize_rotation_study_md.py \
    --out-root "$OUT_ROOT" \
    --out-md "$RESULT_MD" \
    --title "36h Rotation Study P0-P3 Results" || true
}

run_phase() {
  local phase="$1"
  local script="$2"
  log "START $phase: $script"
  (
    cd "$ROOT_DIR" || exit 1
    rtk env \
      OUT_ROOT="$OUT_ROOT" \
      GPUS="$GPUS" \
      NPROC_PER_NODE="$NPROC_PER_NODE" \
      BATCH_SIZE="$BATCH_SIZE" \
      NUM_WORKERS="$NUM_WORKERS" \
      PYTHON_BIN="$PYTHON_BIN" \
      "$script"
  ) >> "$RUN_LOG" 2>&1
  local exit_code=$?
  log "END $phase exit_code=$exit_code"
  refresh_md
  return "$exit_code"
}

main_impl() {
  cd "$ROOT_DIR" || exit 1
  rtk /usr/bin/mkdir -p "$OUT_ROOT"
  : > "$RUN_LOG"
  log "36h rotation study started"
  log "OUT_ROOT=$OUT_ROOT"
  log "RESULT_MD=$RESULT_MD"
  log "GPUS=$GPUS NPROC_PER_NODE=$NPROC_PER_NODE BATCH_SIZE=$BATCH_SIZE NUM_WORKERS=$NUM_WORKERS"
  refresh_md

  run_phase P0 tools/rotation_study/run_p0_matrix.sh || true
  run_phase P1 tools/rotation_study/run_p1_fair1m_granularity.sh || true
  run_phase P2 tools/rotation_study/run_p2_angle_sweep.sh || true
  run_phase P3 tools/rotation_study/run_p3_error_decomp.sh || true

  log "36h rotation study finished all queued P0-P3 phases"
  refresh_md
}

if [[ "${ROTATION_STUDY_INNER:-0}" == "1" ]]; then
  main_impl "$@"
else
  rtk /usr/bin/mkdir -p "$OUT_ROOT"
  rtk /usr/bin/timeout "$TOTAL_TIMEOUT" rtk env ROTATION_STUDY_INNER=1 "$0" "$@"
  exit_code=$?
  if [[ "$exit_code" -eq 124 || "$exit_code" -eq 137 ]]; then
    {
      printf '[%s] TOTAL_TIMEOUT reached: %s\n' "$(rtk /usr/bin/date '+%F %T')" "$TOTAL_TIMEOUT"
    } >> "$RUN_LOG"
    refresh_md
    exit 0
  fi
  exit "$exit_code"
fi
