#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/data1/zcy/OpenRSD}"
SCHEDULE_ROOT="${SCHEDULE_ROOT:-$ROOT_DIR/work_dirs/scheduled_runs}"
WAITER="$ROOT_DIR/M_Tools/schedule/wait_and_run_may02_0167_full_series.sh"
SESSION="${SESSION:-may02_0167_full_series}"
GPUS="${GPUS:-0,1,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PYTHON_BIN="${PYTHON_BIN:-/data/zcy/anaconda3/envs/openrsd/bin/python}"
DRY_RUN="${DRY_RUN:-0}"
SCHEDULER_TS="${SCHEDULER_TS:-$(rtk /usr/bin/date +%Y%m%d_%H%M%S)}"
SCHEDULER_LOG="$SCHEDULE_ROOT/may02_0167_scheduler_$SCHEDULER_TS.log"

main() {
  cd "$ROOT_DIR" || exit 1
  rtk /usr/bin/mkdir -p "$SCHEDULE_ROOT" || exit 1

  if [[ "$DRY_RUN" == "1" ]]; then
    rtk env \
      ROOT_DIR="$ROOT_DIR" \
      SCHEDULE_ROOT="$SCHEDULE_ROOT" \
      SCHEDULER_LOG="$SCHEDULER_LOG" \
      GPUS="$GPUS" \
      NPROC_PER_NODE="$NPROC_PER_NODE" \
      BATCH_SIZE="$BATCH_SIZE" \
      NUM_WORKERS="$NUM_WORKERS" \
      PYTHON_BIN="$PYTHON_BIN" \
      DRY_RUN=1 \
      bash "$WAITER"
    exit $?
  fi

  if rtk tmux has-session -t "$SESSION" >/dev/null 2>&1; then
    printf 'tmux session already exists: %s\n' "$SESSION"
    printf 'Attach with: rtk tmux attach -t %s\n' "$SESSION"
    printf 'Scheduler log: %s\n' "$SCHEDULER_LOG"
    exit 1
  fi

  local cmd
  cmd="rtk env ROOT_DIR=$ROOT_DIR SCHEDULE_ROOT=$SCHEDULE_ROOT SCHEDULER_LOG=$SCHEDULER_LOG GPUS=$GPUS NPROC_PER_NODE=$NPROC_PER_NODE BATCH_SIZE=$BATCH_SIZE NUM_WORKERS=$NUM_WORKERS PYTHON_BIN=$PYTHON_BIN bash $WAITER"
  rtk tmux new-session -d -s "$SESSION" -c "$ROOT_DIR" "$cmd" || exit 1

  printf 'Started tmux session: %s\n' "$SESSION"
  printf 'Attach with: rtk tmux attach -t %s\n' "$SESSION"
  printf 'Scheduler log: %s\n' "$SCHEDULER_LOG"
}

main "$@"
