#!/usr/bin/env bash
# Overnight pipeline: train A/B on GPU8/6, eval GPU9, official GPU7, monitor loop.
set -euo pipefail
REPO="/data1/zcy/OpenRSD"
PY="/data/zcy/anaconda3/envs/openrsd/bin/python"
TOOL="$REPO/tools/exp_next_plan_sv_dehub_step23_overnight"
RES="$REPO/resultmd/exp_next_plan_sv_dehub_step23_overnight"
mkdir -p "$RES"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$RES/log_pipeline_overnight.txt"; }

log "T0 start overnight pipeline"

# GPU9: baseline heldout-500 first (background)
CUDA_VISIBLE_DEVICES=9 nohup "$PY" "$TOOL/run_heldout_eval_gpu9.py" \
  --heldout-n 500 --baseline-only --gpu 9 \
  >> "$RES/log_eval_gpu9_baseline500.txt" 2>&1 &
EVAL9_PID=$!
log "GPU9 baseline heldout-500 pid=$EVAL9_PID"

# GPU7: fusion audit + official highlow (background)
CUDA_VISIBLE_DEVICES=7 nohup "$PY" "$TOOL/run_official_step23.py" --task all --gpu 7 \
  >> "$RES/log_official_gpu7.txt" 2>&1 &
OFF7_PID=$!
log "GPU7 official pid=$OFF7_PID"

# GPU8: A1 resume iter1000 -> 8k
CUDA_VISIBLE_DEVICES=8 nohup "$PY" "$TOOL/run_train_branch.py" --branch A --gpu 8 \
  >> "$RES/log_train_A_iter1000_to8k.txt" 2>&1 &
TRA8_PID=$!
log "GPU8 train A pid=$TRA8_PID"

# GPU6: B2 fresh epoch24 -> 8k (after brief wait if needed)
CUDA_VISIBLE_DEVICES=6 nohup "$PY" "$TOOL/run_train_branch.py" --branch B --gpu 6 \
  >> "$RES/log_train_B_fresh_epoch24_to8k.txt" 2>&1 &
TRA6_PID=$!
log "GPU6 train B pid=$TRA6_PID"

# Monitor loop (background)
nohup "$PY" "$TOOL/monitor_plan_overnight.py" --interval-min 50 \
  >> "$RES/log_monitor.txt" 2>&1 &
MON_PID=$!
log "monitor pid=$MON_PID"

# Wait for training; poll checkpoints and enqueue eval
wait_train_and_eval() {
  local branch=$1
  local wd="$REPO/work_dirs/exp_next_plan_sv_dehub_step23_overnight/branch_${branch}_*"
  while true; do
    if ! kill -0 "$TRA8_PID" 2>/dev/null && [ "$branch" = "A" ]; then break; fi
    if ! kill -0 "$TRA6_PID" 2>/dev/null && [ "$branch" = "B" ]; then break; fi
    sleep 300
    for ck in $(find "$REPO/work_dirs/exp_next_plan_sv_dehub_step23_overnight" -name 'iter_*.pth' 2>/dev/null | sort); do
      mark="$RES/.eval_done_$(basename "$ck")"
      [ -f "$mark" ] && continue
      if [[ "$ck" == *iter_4000* ]] || [[ "$ck" == *iter_8000* ]] || [[ "$ck" == *iter_2000* ]]; then
        log "enqueue partial eval $ck"
        CUDA_VISIBLE_DEVICES=9 "$PY" "$TOOL/run_heldout_eval_gpu9.py" --heldout-n 200 --gpu 9 \
          >> "$RES/log_eval_gpu9_partial.txt" 2>&1 || true
        touch "$mark"
      fi
    done
    sleep 600
  done
}

# Foreground wait for primary trains then full eval
wait "$TRA8_PID" || log "train A exited non-zero"
wait "$TRA6_PID" || log "train B exited non-zero"

log "training done; full heldout-500 + mechanism eval"
CUDA_VISIBLE_DEVICES=9 "$PY" "$TOOL/run_heldout_eval_gpu9.py" --heldout-n 500 --gpu 9 \
  >> "$RES/log_eval_gpu9_full.txt" 2>&1 || true

wait "$OFF7_PID" 2>/dev/null || true
wait "$EVAL9_PID" 2>/dev/null || true

"$PY" "$TOOL/build_overnight_summary.py" || log "summary build skipped"

log "pipeline finished"
