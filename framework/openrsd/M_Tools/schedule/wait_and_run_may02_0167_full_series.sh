#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/data1/zcy/OpenRSD}"
SCHEDULE_ROOT="${SCHEDULE_ROOT:-$ROOT_DIR/work_dirs/scheduled_runs}"
RUNNER="$ROOT_DIR/M_Tools/schedule/run_may02_0167_full_series.sh"
GPUS="${GPUS:-0,1,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PYTHON_BIN="${PYTHON_BIN:-/data/zcy/anaconda3/envs/openrsd/bin/python}"
DRY_RUN="${DRY_RUN:-0}"
SCHEDULER_TS="${SCHEDULER_TS:-$(rtk /usr/bin/date +%Y%m%d_%H%M%S)}"
SCHEDULER_LOG="${SCHEDULER_LOG:-$SCHEDULE_ROOT/may02_0167_scheduler_$SCHEDULER_TS.log}"

SLOTS=(
  "2026-05-02 12:00:00"
  "2026-05-02 14:00:00"
  "2026-05-02 16:00:00"
  "2026-05-02 20:00:00"
)

GPU_BUSY_REASON=""

log() {
  local ts line
  ts="$(rtk /usr/bin/date '+%F %T')"
  line="[$ts] $*"
  printf '%s\n' "$line"
  printf '%s\n' "$line" >> "$SCHEDULER_LOG"
}

is_selected_gpu() {
  local idx="$1"
  local old_ifs="$IFS"
  local selected
  IFS=','
  for selected in $GPUS; do
    if [[ "$selected" == "$idx" ]]; then
      IFS="$old_ifs"
      return 0
    fi
  done
  IFS="$old_ifs"
  return 1
}

check_selected_gpus_free() {
  declare -A bus_to_index=()
  local idx bus pid proc mem
  GPU_BUSY_REASON=""

  while IFS=',' read -r idx bus; do
    idx="${idx//[[:space:]]/}"
    bus="${bus//[[:space:]]/}"
    [[ -n "$idx" && -n "$bus" ]] || continue
    bus_to_index["$bus"]="$idx"
  done < <(rtk nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader)

  while IFS=',' read -r bus pid proc mem; do
    bus="${bus//[[:space:]]/}"
    pid="${pid//[[:space:]]/}"
    proc="${proc#"${proc%%[![:space:]]*}"}"
    mem="${mem//[[:space:]]/}"
    idx="${bus_to_index[$bus]:-}"
    [[ -n "$idx" ]] || continue
    if is_selected_gpu "$idx"; then
      GPU_BUSY_REASON+="GPU $idx pid=$pid mem=${mem}MiB proc=$proc; "
    fi
  done < <(rtk nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader,nounits)

  [[ -z "$GPU_BUSY_REASON" ]]
}

wait_until_slot() {
  local slot="$1"
  local target_epoch now_epoch sleep_seconds
  target_epoch="$(rtk /usr/bin/date -d "$slot" +%s)" || return 1
  now_epoch="$(rtk /usr/bin/date +%s)" || return 1
  if (( now_epoch < target_epoch )); then
    sleep_seconds=$((target_epoch - now_epoch))
    log "Waiting ${sleep_seconds}s until $slot Asia/Shanghai"
    rtk /usr/bin/sleep "$sleep_seconds"
  else
    log "Slot $slot is already due; checking immediately"
  fi
}

write_failure_summary() {
  local out_root="$1"
  local reason="$2"
  rtk /usr/bin/mkdir -p "$out_root" || return 1
  printf 'stage|model|dataset|epoch|train_map|status|map|ap50|checkpoint|log|predictions\n' > "$out_root/results.tsv"
  printf 'scheduler|may02_0167_full_series|ALL|NA|NA|FAILED_TO_START|NA|NA||%s|\n' "$out_root/scheduler.log" >> "$out_root/results.tsv"
  {
    printf '# May 02 0167 Full Series Summary\n\n'
    printf -- '- generated_at: `%s`\n' "$(rtk /usr/bin/date '+%F %T')"
    printf -- '- status: `FAILED_TO_START`\n'
    printf -- '- GPUs: `%s`\n' "$GPUS"
    printf -- '- retry_slots: `12:00 -> 14:00 -> 16:00 -> 20:00`\n'
    printf -- '- reason: `%s`\n' "$reason"
    printf -- '- scheduler_log: `%s`\n' "$SCHEDULER_LOG"
    printf -- '- output_root: `%s`\n' "$out_root"
  } > "$out_root/summary.md"
  {
    printf '[%s] FAILED_TO_START: %s\n' "$(rtk /usr/bin/date '+%F %T')" "$reason"
    printf '[%s] Scheduler log: %s\n' "$(rtk /usr/bin/date '+%F %T')" "$SCHEDULER_LOG"
  } > "$out_root/scheduler.log"
}

dry_run_report() {
  local slot
  log "DRY_RUN=1: no waiting and no tmux/train/eval launch"
  log "DRY_RUN=1: GPU compute-process check target is exactly GPUS=$GPUS"
  for slot in "${SLOTS[@]}"; do
    log "DRY_RUN=1: would wait until $slot Asia/Shanghai, then check GPUs $GPUS"
  done
  log "DRY_RUN=1: first free slot would run OUT_ROOT=$SCHEDULE_ROOT/may02_0167_full_series_<timestamp> rtk bash $RUNNER"
}

main() {
  cd "$ROOT_DIR" || exit 1
  rtk /usr/bin/mkdir -p "$SCHEDULE_ROOT" || exit 1
  : > "$SCHEDULER_LOG"

  log "May 02 scheduler started"
  log "Retry slots: ${SLOTS[*]}"
  log "Target GPUs: $GPUS"

  if [[ "$DRY_RUN" == "1" ]]; then
    dry_run_report
    exit 0
  fi

  local slot run_ts out_root status
  for slot in "${SLOTS[@]}"; do
    wait_until_slot "$slot" || exit 1
    log "Checking selected GPU compute processes for $GPUS at slot $slot"
    if check_selected_gpus_free; then
      run_ts="$(rtk /usr/bin/date +%Y%m%d_%H%M%S)"
      out_root="$SCHEDULE_ROOT/may02_0167_full_series_$run_ts"
      rtk /usr/bin/mkdir -p "$out_root" || exit 1
      log "GPUs are free. Starting full series with OUT_ROOT=$out_root"
      rtk env \
        ROOT_DIR="$ROOT_DIR" \
        OUT_ROOT="$out_root" \
        GPUS="$GPUS" \
        NPROC_PER_NODE="$NPROC_PER_NODE" \
        BATCH_SIZE="$BATCH_SIZE" \
        NUM_WORKERS="$NUM_WORKERS" \
        PYTHON_BIN="$PYTHON_BIN" \
        NCCL_P2P_DISABLE=1 \
        NCCL_IB_DISABLE=1 \
        bash "$RUNNER"
      status=$?
      log "Full series runner exited with status=$status"
      exit "$status"
    fi

    log "GPUs are busy at $slot: $GPU_BUSY_REASON"
    if [[ "$slot" == "2026-05-02 20:00:00" ]]; then
      run_ts="$(rtk /usr/bin/date +%Y%m%d_%H%M%S)"
      out_root="$SCHEDULE_ROOT/may02_0167_full_series_${run_ts}_failed_to_start"
      write_failure_summary "$out_root" "20:00 retry still busy: $GPU_BUSY_REASON"
      log "20:00 retry still busy. Wrote failure summary to $out_root/summary.md"
      exit 1
    fi
  done
}

main "$@"
