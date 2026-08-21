#!/usr/bin/env bash
# Verification pipeline: AP (GPU9) || ablation train (GPU8) -> eval -> audit -> summary
set -euo pipefail
REPO="/data1/zcy/OpenRSD"
PY="/data/zcy/anaconda3/envs/openrsd/bin/python"
TOOL="$REPO/tools/exp_sv_dehub_lite_train_gpu89/verify"
VERIFY="$REPO/resultmd/exp_sv_dehub_lite_train_gpu89/verify_ap_ablation_20260521"
HELDOUT_N="${HELDOUT_N:-100}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$VERIFY"

echo "=== C: low-risk audit ==="
"$PY" "$TOOL/audit_lowrisk_side_effect.py"

echo "=== B: ablation train GPU8 ==="
CUDA_VISIBLE_DEVICES=8 "$PY" "$TOOL/run_ablation_train_gpu8.py" --max-iters 1000

echo "=== A: full AP GPU9 ==="
CUDA_VISIBLE_DEVICES=9 "$PY" "$TOOL/run_full_ap_eval_gpu9.py" --heldout-n "$HELDOUT_N"

echo "=== B eval GPU9 ==="
CUDA_VISIBLE_DEVICES=9 "$PY" "$TOOL/run_ablation_eval_gpu9.py" --heldout-n "$HELDOUT_N"

echo "=== final summary ==="
"$PY" "$TOOL/build_verify_summary.py"

echo "DONE $VERIFY"
