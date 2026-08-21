#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/data1/zcy/OpenRSD}"
RUNNER="${RUNNER:-$ROOT_DIR/M_Tools/schedule/run_rotation_research_full_8gpu.sh}"
WATCHER="${WATCHER:-$ROOT_DIR/M_Tools/schedule/watch_rotation_research_full_8gpu.sh}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/work_dirs/scheduled_runs/rotation_research_full_8gpu_20260504_014158}"
SESSION="${SESSION:-rotation_research_postprocess_resume}"
GPUS="${GPUS:-0,1,4,5,6,7,8,9}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PYTHON_BIN="${PYTHON_BIN:-/data/zcy/anaconda3/envs/openrsd/bin/python}"
MAX_WATCHDOG_RESTARTS="${MAX_WATCHDOG_RESTARTS:-20}"
WATCHDOG_SLEEP_SECONDS="${WATCHDOG_SLEEP_SECONDS:-120}"

# Angle-sweep mAP is already complete. This wrapper resumes failed postprocess.
PREPARE_ANGLES="${PREPARE_ANGLES:-0}"
RUN_ANGLE_SWEEP="${RUN_ANGLE_SWEEP:-0}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-0}"
RUN_TTA="${RUN_TTA:-1}"
WAIT_FOR_FREE_GPUS="${WAIT_FOR_FREE_GPUS:-0}"
TTA_PRE_NMS_TOPK="${TTA_PRE_NMS_TOPK:-1000}"
DIAGNOSTIC_MAX_DETS_PER_IMG="${DIAGNOSTIC_MAX_DETS_PER_IMG:-50}"

main() {
  cd "$ROOT_DIR" || exit 1
  rtk /usr/bin/mkdir -p "$OUT_ROOT" || exit 1

  if rtk tmux has-session -t "$SESSION" >/dev/null 2>&1; then
    printf 'tmux session already exists: %s\n' "$SESSION"
    printf 'Attach with: rtk tmux attach -t %s\n' "$SESSION"
    exit 1
  fi

  rtk tmux new-session -d -s "$SESSION" -c "$ROOT_DIR" \
    "rtk env ROOT_DIR='$ROOT_DIR' RUNNER='$RUNNER' OUT_ROOT='$OUT_ROOT' GPUS='$GPUS' NPROC_PER_NODE='$NPROC_PER_NODE' NUM_WORKERS='$NUM_WORKERS' PYTHON_BIN='$PYTHON_BIN' MAX_WATCHDOG_RESTARTS='$MAX_WATCHDOG_RESTARTS' WATCHDOG_SLEEP_SECONDS='$WATCHDOG_SLEEP_SECONDS' PREPARE_ANGLES='$PREPARE_ANGLES' RUN_ANGLE_SWEEP='$RUN_ANGLE_SWEEP' RUN_DIAGNOSTICS='$RUN_DIAGNOSTICS' RUN_TTA='$RUN_TTA' WAIT_FOR_FREE_GPUS='$WAIT_FOR_FREE_GPUS' TTA_PRE_NMS_TOPK='$TTA_PRE_NMS_TOPK' DIAGNOSTIC_MAX_DETS_PER_IMG='$DIAGNOSTIC_MAX_DETS_PER_IMG' bash '$WATCHER'" || exit 1

  printf 'Started tmux session: %s\n' "$SESSION"
  printf 'Output root: %s\n' "$OUT_ROOT"
  printf 'Markdown summary: %s/rotation_research_full_results.md\n' "$OUT_ROOT"
  printf 'Run log: %s/run.log\n' "$OUT_ROOT"
  printf 'Watchdog log: %s/watchdog.log\n' "$OUT_ROOT"
}

main "$@"
