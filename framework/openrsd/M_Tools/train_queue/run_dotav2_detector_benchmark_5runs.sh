#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data1/zcy/OpenRSD"
QUEUE_FILE="$ROOT_DIR/M_Tools/train_queue/dotav2_detector_benchmark_5runs.queue"
SESSION_NAME="${SESSION_NAME:-dotav2_detector_benchmark_5runs}"
LOG_DIR="$ROOT_DIR/work_dirs/train_queue_logs"
TMUX_LOG="$LOG_DIR/${SESSION_NAME}_tmux.log"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  echo "Attach with: rtk tmux attach -t $SESSION_NAME"
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" \
  "cd '$ROOT_DIR' && GPUS='${GPUS:-0,1,2,3}' NPROC_PER_NODE='${NPROC_PER_NODE:-4}' MASTER_PORT_START='${MASTER_PORT_START:-29901}' CONTINUE_ON_FAILURE='${CONTINUE_ON_FAILURE:-0}' rtk bash M_Tools/train_queue/run_train_queue.sh '$QUEUE_FILE' > '$TMUX_LOG' 2>&1"

echo "Started tmux session: $SESSION_NAME"
echo "Queue file: $QUEUE_FILE"
echo "tmux log: $TMUX_LOG"
echo "Status/log files will be written under: $LOG_DIR"
