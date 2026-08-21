#!/usr/bin/env bash
set -euo pipefail
REPO=/data1/zcy/OpenRSD
PY=/data/zcy/anaconda3/envs/openrsd/bin/python
TOOL=$REPO/tools/exp_sv_dehub_lite_train_gpu89/plan3h
VD=$REPO/resultmd/exp_sv_dehub_lite_train_gpu89/verify_ap_ablation_plan3h_20260521
LOG=$VD/log_plan3h_master.txt
export PYTHONPATH=$REPO
T0=$(date +%s)
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }
monitor() {
  local em=$1 t8=$2 t9=$3 err=${4:-} act=${5:-record} nxt=${6:-}
  $PY $TOOL/plan3h_monitor_sv_dehub_verify.py --elapsed-min "$em" --task-gpu8 "$t8" --task-gpu9 "$t9" --error "$err" --action "$act" --next "$nxt" || true
}
elapsed() { echo "scale=1; ($(date +%s)-$T0)/60" | bc; }

log "=== Plan3H pipeline start ==="
nvidia-smi >> "$LOG" 2>&1
monitor 0 "idle" "preflight"

log "Exp0 patched baseline (GPU9)"
CUDA_VISIBLE_DEVICES=9 $PY $TOOL/run_patched_baseline_audit_gpu9.py 2>&1 | tee -a "$VD/log_gpu9_patched_baseline_consistency.txt"
RC0=${PIPESTATUS[0]}
VERDICT=$(grep -oP 'verdict: \*\*\K[^*]+' "$VD/fres_00_patched_baseline_consistency.md" 2>/dev/null || echo UNKNOWN)
log "Exp0 rc=$RC0 verdict=$VERDICT"
monitor "$(elapsed)" "idle" "exp0_done" "" "gate" "$VERDICT"

if [[ "$RC0" -eq 2 ]] || [[ "$VERDICT" == "BROKEN" ]]; then
  log "STOP: patched baseline BROKEN"
  $PY $TOOL/build_plan3h_summary.py --verdict SUPPORT_FIX_CONFOUNDED 2>&1 | tee -a "$LOG"
  exit 2
fi

# GPU8 ablation in background
log "Start GPU8 ablation train (B1-B3)"
(
  export CUDA_VISIBLE_DEVICES=8
  $PY $TOOL/run_ablation_train_gpu8.py --gpu 8 2>&1 | tee -a "$VD/log_gpu8_ablation_train.txt"
) &
PID8=$!

log "ExpA full AP heldout 200 (GPU9)"
CUDA_VISIBLE_DEVICES=9 $PY $TOOL/run_full_ap_eval_gpu9.py --heldout-n 200 2>&1 | tee -a "$VD/log_gpu9_full_ap_eval.txt" || {
  log "AP200 failed, try heldout 100"
  CUDA_VISIBLE_DEVICES=9 $PY $TOOL/run_full_ap_eval_gpu9.py --heldout-n 100 2>&1 | tee -a "$VD/log_gpu9_full_ap_eval.txt"
}

wait $PID8 || log "ablation train had errors"
monitor "$(elapsed)" "ablation_done" "ap_done" "" "continue" "ablation_eval"

log "Ablation eval GPU9"
CUDA_VISIBLE_DEVICES=9 $PY $TOOL/run_ablation_eval_gpu9.py --heldout-n 100 2>&1 | tee -a "$VD/log_gpu9_ablation_eval.txt" || true

log "ExpC low-risk audit"
$PY $TOOL/audit_lowrisk_enhanced.py --eval-csv "$REPO/resultmd/exp_sv_dehub_lite_train_gpu89/ftable_eval_sv_dehub_lite_raw.csv" 2>&1 | tee -a "$VD/log_lowrisk_audit.txt" || true

log "Final summary"
$PY $TOOL/build_plan3h_summary.py 2>&1 | tee -a "$LOG"
monitor "$(elapsed)" "done" "done" "" "final" "report_written"
log "=== Plan3H pipeline end ==="
