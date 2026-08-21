#!/usr/bin/env bash
# Poll for new checkpoints; run heldout-200/500 + mechanism on GPU9.
set -euo pipefail
REPO="/data1/zcy/OpenRSD"
PY="/data/zcy/anaconda3/envs/openrsd/bin/python"
TOOL="$REPO/tools/exp_next_plan_sv_dehub_step23_overnight"
RES="$REPO/resultmd/exp_next_plan_sv_dehub_step23_overnight"
WD="$REPO/work_dirs/exp_next_plan_sv_dehub_step23_overnight"
MARK_DIR="$RES/.eval_marks"
mkdir -p "$MARK_DIR"

while true; do
  for ck in $(find "$WD" -name 'iter_*.pth' 2>/dev/null | sort); do
    base=$(basename "$ck")
    mark="$MARK_DIR/${base}.done"
    [[ -f "$mark" ]] && continue
    case "$base" in
      iter_2000.pth|iter_4000.pth|iter_6000.pth|iter_8000.pth)
        echo "[$(date -Iseconds)] eval $ck" >> "$RES/log_eval_gpu9_watch.txt"
        CUDA_VISIBLE_DEVICES=9 "$PY" "$TOOL/run_heldout_eval_gpu9.py" --heldout-n 200 --gpu 9 \
          >> "$RES/log_eval_gpu9_partial.txt" 2>&1 || true
        if [[ "$base" == "iter_8000.pth" ]]; then
          CUDA_VISIBLE_DEVICES=9 "$PY" "$TOOL/run_heldout_eval_gpu9.py" --heldout-n 500 --gpu 9 \
            >> "$RES/log_eval_gpu9_full.txt" 2>&1 || true
          CUDA_VISIBLE_DEVICES=9 "$PY" "$TOOL/build_overnight_summary.py" \
            >> "$RES/log_summary_build.txt" 2>&1 || true
        fi
        touch "$mark"
        ;;
    esac
  done
  # Exit when both trains done and iter_8000 marks exist for A and B
  if ! pgrep -f 'run_train_branch.py --branch A' >/dev/null 2>&1 \
     && ! pgrep -f 'run_train_branch.py --branch B' >/dev/null 2>&1; then
    if [[ -f "$MARK_DIR/iter_8000.pth.done" ]] || [[ -f "$WD/branch_A_iter1000_to8k/iter_8000.pth" ]]; then
      echo "[$(date -Iseconds)] trains finished; final eval" >> "$RES/log_eval_gpu9_watch.txt"
      CUDA_VISIBLE_DEVICES=9 "$PY" "$TOOL/run_heldout_eval_gpu9.py" --heldout-n 500 --gpu 9 \
        >> "$RES/log_eval_gpu9_full.txt" 2>&1 || true
      CUDA_VISIBLE_DEVICES=9 "$PY" "$TOOL/build_overnight_summary.py" \
        >> "$RES/log_summary_build.txt" 2>&1 || true
      break
    fi
  fi
  sleep 600
done
