#!/usr/bin/env bash
# Strict queue: r0->r1->r2->r3->r4 with auto-restart on crash.
set -euo pipefail

REPO="/data1/zcy/OpenRSD"
WORK="${REPO}/work_dirs/exp_rotation_gt_shift_taxonomy_20260527"
LOG="${WORK}/logs/watchdog_outer.log"
PYTHON="/data/zcy/anaconda3/envs/openrsd/bin/python"
COHORT_SIZE="${COHORT_SIZE:-2500}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-48}"
R2_WORKERS="${R2_WORKERS:-512}"
SLEEP_SEC="${SLEEP_SEC:-120}"

mkdir -p "${WORK}/logs"

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "${LOG}"
}

log "watchdog outer loop start cohort=${COHORT_SIZE} batch=${BATCH_SIZE} gpus=${GPU_IDS}"

while [[ ! -f "${WORK}/.done_pipeline" ]]; do
  if ! "${PYTHON}" "${REPO}/M_Tools/analysis/run_rsv_taxonomy_watchdog.py" \
      --cohort-size "${COHORT_SIZE}" \
      --infer-gpu-ids "${GPU_IDS}" \
      --infer-batch-size "${BATCH_SIZE}" \
      --r2-workers "${R2_WORKERS}" \
      --once; then
    log "watchdog pass failed; sleep ${SLEEP_SEC}s and retry"
    sleep "${SLEEP_SEC}"
  else
    if [[ -f "${WORK}/.done_pipeline" ]]; then
      log "pipeline finished"
      break
    fi
    log "watchdog pass ok but pipeline not done; sleep ${SLEEP_SEC}s"
    sleep "${SLEEP_SEC}"
  fi
done

log "ALL DONE: ${WORK}/.done_pipeline"
