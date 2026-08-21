#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/data1/zcy/OpenRSD}"
RUNNER="$ROOT_DIR/M_Tools/schedule/run_rotation_research_full_8gpu.sh"
WATCHER="$ROOT_DIR/M_Tools/schedule/watch_rotation_research_full_8gpu.sh"
SESSION="${SESSION:-rotation_research_full_8gpu}"
RUN_TS="${RUN_TS:-$(rtk /usr/bin/date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/work_dirs/scheduled_runs/rotation_research_full_8gpu_$RUN_TS}"
GPUS="${GPUS:-0,1,4,5,6,7,8,9}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_CANDIDATES="${BATCH_CANDIDATES:-16 8 4 2 1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PYTHON_BIN="${PYTHON_BIN:-/data/zcy/anaconda3/envs/openrsd/bin/python}"
MAX_WATCHDOG_RESTARTS="${MAX_WATCHDOG_RESTARTS:-20}"
WATCHDOG_SLEEP_SECONDS="${WATCHDOG_SLEEP_SECONDS:-120}"
DRY_RUN="${DRY_RUN:-0}"

main() {
  cd "$ROOT_DIR" || exit 1
  rtk /usr/bin/mkdir -p "$OUT_ROOT" || exit 1

  if [[ ! -f "$RUNNER" ]]; then
    printf 'Missing runner: %s\n' "$RUNNER"
    exit 2
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    rtk env \
      ROOT_DIR="$ROOT_DIR" \
      OUT_ROOT="$OUT_ROOT" \
      GPUS="$GPUS" \
      NPROC_PER_NODE="$NPROC_PER_NODE" \
      BATCH_CANDIDATES="$BATCH_CANDIDATES" \
      NUM_WORKERS="$NUM_WORKERS" \
      PYTHON_BIN="$PYTHON_BIN" \
      DRY_RUN=1 \
      RESUME=1 \
      bash "$RUNNER"
    exit $?
  fi

  if rtk tmux has-session -t "$SESSION" >/dev/null 2>&1; then
    printf 'tmux session already exists: %s\n' "$SESSION"
    printf 'Attach with: rtk tmux attach -t %s\n' "$SESSION"
    printf 'Output root: %s\n' "$OUT_ROOT"
    exit 1
  fi

  rtk tmux new-session -d -s "$SESSION" -c "$ROOT_DIR" \
    "rtk env ROOT_DIR='$ROOT_DIR' RUNNER='$RUNNER' OUT_ROOT='$OUT_ROOT' GPUS='$GPUS' NPROC_PER_NODE='$NPROC_PER_NODE' BATCH_CANDIDATES='$BATCH_CANDIDATES' NUM_WORKERS='$NUM_WORKERS' PYTHON_BIN='$PYTHON_BIN' MAX_WATCHDOG_RESTARTS='$MAX_WATCHDOG_RESTARTS' WATCHDOG_SLEEP_SECONDS='$WATCHDOG_SLEEP_SECONDS' bash '$WATCHER'" || exit 1

  printf 'Started tmux session: %s\n' "$SESSION"
  printf 'Attach with: rtk tmux attach -t %s\n' "$SESSION"
  printf 'Output root: %s\n' "$OUT_ROOT"
  printf 'Markdown summary: %s/rotation_research_full_results.md\n' "$OUT_ROOT"
  printf 'Run log: %s/run.log\n' "$OUT_ROOT"
  printf 'Watchdog log: %s/watchdog.log\n' "$OUT_ROOT"
}

main "$@"
