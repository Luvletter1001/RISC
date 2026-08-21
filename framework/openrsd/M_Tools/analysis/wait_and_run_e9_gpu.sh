#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  cat <<'USAGE'
Usage:
  wait_and_run_e9_gpu.sh GPU_ID MODEL_NAMES MAX_CASES OUT_DIR [CHECK_INTERVAL_SEC]

Example:
  wait_and_run_e9_gpu.sh 6 faahead_lsknet,lsknet_orcnn 8 work_dirs/semantic_ambiguity_study_20260617/e9_gpu6
USAGE
  exit 2
fi

GPU_ID="$1"
MODEL_NAMES="$2"
MAX_CASES="$3"
OUT_DIR="$4"
CHECK_INTERVAL_SEC="${5:-120}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="/data/zcy/anaconda3/envs/openrsd/bin/python"
SCRIPT_PATH="$REPO_ROOT/M_Tools/analysis/run_e9_scale_counterfactual_probe.py"
LOG_DIR="$REPO_ROOT/work_dirs/semantic_ambiguity_study_20260617/logs"
LOG_PATH="$LOG_DIR/e9_wait_gpu${GPU_ID}_$(date '+%Y%m%d_%H%M%S').log"
MAX_MEMORY_USED_MB="${MAX_MEMORY_USED_MB:-10000}"
MAX_UTIL_PCT="${MAX_UTIL_PCT:-10}"
REQUIRED_IDLE_CHECKS="${REQUIRED_IDLE_CHECKS:-3}"

mkdir -p "$LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "$LOG_PATH"
}

query_gpu() {
  rtk nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits |
    awk -F', *' -v id="$GPU_ID" '$1 == id {print $2" "$3}'
}

idle_checks=0
log "wait_start gpu=$GPU_ID model_names=$MODEL_NAMES max_cases=$MAX_CASES out_dir=$OUT_DIR max_mem_mb=$MAX_MEMORY_USED_MB max_util_pct=$MAX_UTIL_PCT required_idle_checks=$REQUIRED_IDLE_CHECKS"

while true; do
  read -r memory_used util_pct < <(query_gpu)
  if [ -z "${memory_used:-}" ] || [ -z "${util_pct:-}" ]; then
    log "gpu_query_failed gpu=$GPU_ID"
    sleep "$CHECK_INTERVAL_SEC"
    continue
  fi

  log "gpu_status gpu=$GPU_ID memory_used_mb=$memory_used util_pct=$util_pct idle_checks=$idle_checks"
  if [ "$memory_used" -le "$MAX_MEMORY_USED_MB" ] && [ "$util_pct" -le "$MAX_UTIL_PCT" ]; then
    idle_checks=$((idle_checks + 1))
  else
    idle_checks=0
  fi

  if [ "$idle_checks" -ge "$REQUIRED_IDLE_CHECKS" ]; then
    break
  fi
  sleep "$CHECK_INTERVAL_SEC"
done

log "launch_probe gpu=$GPU_ID"
cd "$REPO_ROOT"
set +e
CUDA_VISIBLE_DEVICES="$GPU_ID" \
PYTHONNOUSERSITE=1 \
PYTHONPATH="$REPO_ROOT/tools:$REPO_ROOT" \
"$PYTHON_BIN" "$SCRIPT_PATH" \
  --model-names "$MODEL_NAMES" \
  --max-cases "$MAX_CASES" \
  --device cuda:0 \
  --out-dir "$OUT_DIR" 2>&1 | tee -a "$LOG_PATH"
probe_exit="${PIPESTATUS[0]}"
set -e

log "probe_finished gpu=$GPU_ID exit_code=$probe_exit out_dir=$OUT_DIR"
exit "$probe_exit"
