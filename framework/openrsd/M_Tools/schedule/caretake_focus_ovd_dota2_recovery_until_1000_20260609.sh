#!/usr/bin/env bash
set -u

ROOT="/data1/zcy/OpenRSD"
SESSION="focus_ovd_dota2_recovery_full_gpu69_20260609"
LOG_DIR="${ROOT}/resultmd/exp_focus_ovd_20260608/train_full_gpu69_20260609"
FULL_LOG="${LOG_DIR}/full_train_tmux.log"
CARE_LOG="${LOG_DIR}/caretaker_dota2_recovery_until_1000.log"
END_AT="2026-06-09 10:00:00"
POLL_SECONDS=120

mkdir -p "${LOG_DIR}"

log() {
  printf '%s %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$*" | tee -a "${CARE_LOG}"
}

session_alive() {
  tmux has-session -t "${SESSION}" >/dev/null 2>&1
}

last_train_line() {
  if [[ -f "${FULL_LOG}" ]]; then
    grep -E 'Epoch\(train\).*loss:' "${FULL_LOG}" | tail -n 1 || true
  fi
}

fatal_tail() {
  if [[ -f "${FULL_LOG}" ]]; then
    grep -E 'Traceback|RuntimeError|AssertionError|FileNotFoundError|CUDA out of memory|NCCL.*(error|fail)' "${FULL_LOG}" | tail -n 8 || true
  fi
}

gpu_snapshot() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | awk -F, '$1 ~ /^[[:space:]]*(6|9)[[:space:]]*$/ {gsub(/^ +| +$/, "", $1); gsub(/^ +| +$/, "", $2); gsub(/^ +| +$/, "", $3); printf("gpu%s_mem=%sMiB_util=%s%% ", $1, $2, $3)}'
}

end_ts="$(date -d "${END_AT}" +%s)"
log "caretaker started session=${SESSION} end_at=${END_AT}"

while (( "$(date +%s)" < end_ts )); do
  if session_alive; then
    line="$(last_train_line)"
    gpu="$(gpu_snapshot)"
    if [[ -n "${line}" ]]; then
      log "alive ${gpu}last='${line}'"
    else
      log "alive ${gpu}last='<no train line yet>'"
    fi
  else
    fatal="$(fatal_tail)"
    if [[ -n "${fatal}" ]]; then
      log "blocked session_missing fatal_tail_begin"
      printf '%s\n' "${fatal}" | tee -a "${CARE_LOG}"
      log "blocked session_missing fatal_tail_end"
    else
      log "session_missing no_fatal_detected; likely completed or exited cleanly"
    fi
    break
  fi
  sleep "${POLL_SECONDS}"
done

log "caretaker stopped"
