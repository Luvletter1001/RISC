#!/usr/bin/env bash
set -u

ROOT_DIR="/data1/zcy/OpenRSD"
TRAIN_SESSION="baseline6_dotav2_ss_orcnn_r50_20260610"
RUN_LOG="$ROOT_DIR/work_dirs/train_queue_logs/orcnn_r50_dotav2_ss_baseline_6gpu_20260610_run1_20260610_205807.log"
WORK_DIR="$ROOT_DIR/work_dirs/baseline6_dotav2_ss_orcnn_r50_bs2_20260610"
MONITOR_LOG="$ROOT_DIR/work_dirs/train_queue_logs/orcnn_r50_dotav2_ss_baseline_6gpu_20260610_stop_after_ep4.log"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$MONITOR_LOG"
}

stop_train_session() {
  log "Stopping train session: $TRAIN_SESSION"
  rtk tmux send-keys -t "$TRAIN_SESSION" C-c >>"$MONITOR_LOG" 2>&1 || true
  sleep 20
  if rtk tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
    log "Session still alive after first interrupt; sending C-c again"
    rtk tmux send-keys -t "$TRAIN_SESSION" C-c >>"$MONITOR_LOG" 2>&1 || true
    sleep 10
  fi
  if rtk tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
    log "Session still alive after interrupts; killing target session only"
    rtk tmux kill-session -t "$TRAIN_SESSION" >>"$MONITOR_LOG" 2>&1 || true
  fi
}

log "Monitor started; will stop after epoch4 validation metric is logged"

while true; do
  if [[ ! -f "$RUN_LOG" ]]; then
    log "Waiting for run log: $RUN_LOG"
    sleep 10
    continue
  fi

  if grep -Eq 'Epoch\(val\) *\[4\]\[[0-9]+/[0-9]+\].*dota/mAP' "$RUN_LOG"; then
    log "Detected epoch4 validation complete"
    if [[ -f "$WORK_DIR/epoch_4.pth" ]]; then
      log "Found checkpoint: $WORK_DIR/epoch_4.pth"
    else
      log "Warning: epoch4 validation complete but epoch_4.pth not found yet"
    fi
    stop_train_session
    log "Monitor finished after epoch4 val"
    exit 0
  fi

  if grep -Eq 'Epoch\(train\) +\[5\]' "$RUN_LOG"; then
    log "Detected epoch5 training started; stopping immediately as safety fallback"
    if [[ -f "$WORK_DIR/epoch_4.pth" ]]; then
      log "Found checkpoint: $WORK_DIR/epoch_4.pth"
    fi
    stop_train_session
    log "Monitor finished after epoch5 fallback"
    exit 0
  fi

  if ! rtk tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
    log "Train session no longer exists before epoch4 val trigger"
    exit 1
  fi

  sleep 10
done
