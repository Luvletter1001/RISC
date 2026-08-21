#!/usr/bin/env bash
# Mechanism study orchestrator — GPU8 + GPU9 in parallel (no DDP).
set -euo pipefail
REPO=/data1/zcy/OpenRSD
PY="${PYTHON_BIN:-/data/zcy/anaconda3/envs/openrsd/bin/python}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

mkdir -p "$REPO/resultmd/exp_mechanism_sv_attractor_gpu89"
mkdir -p "$REPO/work_dirs/exp_mechanism_sv_attractor_gpu89/logs"

echo "[preflight]"
"$PY" "$REPO/tools/exp_mechanism_sv_attractor_gpu89/discover_existing_artifacts.py"

echo "[GPU9 atlas] background"
CUDA_VISIBLE_DEVICES=9 "$PY" "$REPO/tools/exp_mechanism_sv_attractor_gpu89/run_gpu9_attractor_atlas.py" --gpu 9 \
  > "$REPO/work_dirs/exp_mechanism_sv_attractor_gpu89/logs/gpu9_atlas.log" 2>&1 &
PID9=$!

echo "[GPU8 decomp + bg] background"
CUDA_VISIBLE_DEVICES=8 "$PY" "$REPO/tools/exp_mechanism_sv_attractor_gpu89/run_gpu8_logit_embedding_decomposition.py" --gpu 8 \
  > "$REPO/work_dirs/exp_mechanism_sv_attractor_gpu89/logs/gpu8_decomp.log" 2>&1 &
PID8A=$!

wait "$PID9" || echo "WARN: atlas exit $?"
wait "$PID8A" || echo "WARN: decomp exit $?"

CUDA_VISIBLE_DEVICES=8 "$PY" "$REPO/tools/exp_mechanism_sv_attractor_gpu89/run_gpu8_background_counterfactual.py" --gpu 8 \
  || echo "WARN: bg cf failed"

"$PY" "$REPO/tools/exp_mechanism_sv_attractor_gpu89/analyze_variance_decomposition.py"
CUDA_VISIBLE_DEVICES=9 "$PY" "$REPO/tools/exp_mechanism_sv_attractor_gpu89/run_minimal_repair_mechanism.py" --gpu 9 \
  || echo "WARN: repair mech failed"
"$PY" "$REPO/tools/exp_mechanism_sv_attractor_gpu89/analyze_risk_predictor.py" \
  || echo "WARN: risk predictor failed"

"$PY" "$REPO/tools/exp_mechanism_sv_attractor_gpu89/build_final_mechanism_report.py"

"$PY" "$REPO/tools/exp_mechanism_sv_attractor_gpu89/run_audit_and_resume.py"
"$PY" "$REPO/tools/exp_mechanism_sv_attractor_gpu89/build_final_mechanism_report.py"
