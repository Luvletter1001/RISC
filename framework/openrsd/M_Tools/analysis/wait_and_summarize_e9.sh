#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="/data/zcy/anaconda3/envs/openrsd/bin/python"
SUMMARY_SCRIPT="$REPO_ROOT/M_Tools/analysis/summarize_e9_counterfactual_results.py"
GPU6_DIR="${GPU6_DIR:-work_dirs/semantic_ambiguity_study_20260617/e9_scale_counterfactual_gpu6}"
GPU7_DIR="${GPU7_DIR:-work_dirs/semantic_ambiguity_study_20260617/e9_scale_counterfactual_gpu7}"
OUT_DIR="${OUT_DIR:-work_dirs/semantic_ambiguity_study_20260617/e9_scale_counterfactual_combined}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-120}"
LOG_DIR="$REPO_ROOT/work_dirs/semantic_ambiguity_study_20260617/logs"
LOG_PATH="$LOG_DIR/e9_wait_summary_$(date '+%Y%m%d_%H%M%S').log"

mkdir -p "$LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "$LOG_PATH"
}

gpu6_csv="$REPO_ROOT/$GPU6_DIR/e9_counterfactual_probe.csv"
gpu7_csv="$REPO_ROOT/$GPU7_DIR/e9_counterfactual_probe.csv"

log "summary_wait_start gpu6_csv=$gpu6_csv gpu7_csv=$gpu7_csv out_dir=$OUT_DIR"

while true; do
  gpu6_ready=0
  gpu7_ready=0
  [ -s "$gpu6_csv" ] && gpu6_ready=1
  [ -s "$gpu7_csv" ] && gpu7_ready=1
  log "summary_wait_status gpu6_ready=$gpu6_ready gpu7_ready=$gpu7_ready"
  if [ "$gpu6_ready" -eq 1 ] && [ "$gpu7_ready" -eq 1 ]; then
    break
  fi
  sleep "$CHECK_INTERVAL_SEC"
done

log "summary_launch"
cd "$REPO_ROOT"
rtk "$PYTHON_BIN" "$SUMMARY_SCRIPT" \
  --input-dirs "$GPU6_DIR,$GPU7_DIR" \
  --out-dir "$OUT_DIR" 2>&1 | tee -a "$LOG_PATH"
log "summary_finished out_dir=$OUT_DIR"
