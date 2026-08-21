#!/usr/bin/env bash
# Full SV-DeHub-Lite v1 pipeline: train (GPU8) -> eval (GPU9) -> report
set -euo pipefail
REPO="/data1/zcy/OpenRSD"
PY="/data/zcy/anaconda3/envs/openrsd/bin/python"
TOOL="$REPO/tools/exp_sv_dehub_lite_train_gpu89"
RESULT="$REPO/resultmd/exp_sv_dehub_lite_train_gpu89"
WORK="$REPO/work_dirs/exp_sv_dehub_lite_train_gpu89/train_v1"
LOG="$RESULT/log_pipeline_full.txt"
MAX_ITERS="${MAX_ITERS:-3000}"
SAVE_EVERY="${SAVE_EVERY:-1000}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RESULT" "$WORK"

exec > >(tee -a "$LOG") 2>&1
echo "========== pipeline start $(date -Iseconds) max_iters=$MAX_ITERS =========="

cd "$REPO"
"$PY" "$TOOL/discover_sv_dehub_context.py" --result-dir "$RESULT"
"$PY" "$TOOL/build_hard_negative_patch_bank.py" --result-dir "$RESULT"

# fresh loss curve for this run
: > "$RESULT/ftable_train_loss_curve.csv"
echo 'iter,total_loss,loss_cls,loss_bbox,loss_dehub,dehub_ratio,lr,status' > "$RESULT/ftable_train_loss_curve.csv"

echo "========== GPU8 train =========="
CUDA_VISIBLE_DEVICES=8 "$PY" "$TOOL/run_train_sv_dehub_lite_gpu8.py" \
  --max-iters "$MAX_ITERS" --save-every "$SAVE_EVERY" --gpu 8 \
  --work-dir "$WORK" --result-dir "$RESULT"
TRAIN_RC=$?
echo "train exit_code=$TRAIN_RC"

if ! grep -q loss_dehub "$RESULT/log_gpu8_train_sv_dehub_lite_v1.txt" 2>/dev/null; then
  echo "WARN: loss_dehub not in mmengine log (check ftable_train_loss_curve.csv)"
fi
DEHUB_SUM=$(awk -F, 'NR>1 {s+=$5} END {print s+0}' "$RESULT/ftable_train_loss_curve.csv")
echo "loss_dehub sum in curve csv: $DEHUB_SUM"

echo "========== GPU9 eval =========="
CUDA_VISIBLE_DEVICES=9 "$PY" "$TOOL/run_eval_sv_dehub_lite_gpu9.py" \
  --work-dir "$WORK" --result-dir "$RESULT"
EVAL_RC=$?
echo "eval exit_code=$EVAL_RC"

echo "========== final report =========="
"$PY" "$TOOL/build_sv_dehub_final_report.py" --result-dir "$RESULT"

echo "========== pipeline done $(date -Iseconds) =========="
echo "RESULT_DIR=$RESULT"
echo "WORK_DIR=$WORK"
echo "REPORT=$RESULT/fres_sv_dehub_lite_train_summary.md"
ls -la "$WORK"/*.pth 2>/dev/null || true
