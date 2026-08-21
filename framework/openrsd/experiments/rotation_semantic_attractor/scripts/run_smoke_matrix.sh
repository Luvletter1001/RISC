#!/usr/bin/env bash
set -u

ROOT="/data1/zcy/OpenRSD"
EXP="$ROOT/experiments/rotation_semantic_attractor"
PY="${PY:-/data/zcy/anaconda3/envs/openrsd/bin/python}"
CLOSEDSET_RUN_DIR="${CLOSEDSET_RUN_DIR:-$EXP/outputs/smoke/closedset}"
OPENVOCAB_RUN_DIR="${OPENVOCAB_RUN_DIR:-$EXP/outputs/smoke/openvocab}"
MATRIX_RUN_DIR="${MATRIX_RUN_DIR:-$EXP/outputs/smoke/matrix}"
LOG_DIR="$MATRIX_RUN_DIR/logs"
mkdir -p "$LOG_DIR"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7,8,9}"

cd "$ROOT" || exit 1

echo "[matrix-smoke] closedset"
RUN_DIR="$CLOSEDSET_RUN_DIR" "$EXP/scripts/run_smoke_closedset.sh" > "$LOG_DIR/closedset.log" 2>&1
closed_code=$?

echo "[matrix-smoke] openvocab"
RUN_DIR="$OPENVOCAB_RUN_DIR" OPENVOCAB_SOURCE_RUN_DIR="$CLOSEDSET_RUN_DIR" "$EXP/scripts/run_smoke_openvocab.sh" > "$LOG_DIR/openvocab.log" 2>&1
open_code=$?

"$PY" "$EXP/scripts/14_build_smoke_matrix.py" --closedset-run-dir "$CLOSEDSET_RUN_DIR" --openvocab-run-dir "$OPENVOCAB_RUN_DIR" --output-dir "$MATRIX_RUN_DIR" > "$LOG_DIR/matrix.log" 2>&1
matrix_code=$?

"$PY" "$EXP/scripts/11_build_report.py" --run-dir "$CLOSEDSET_RUN_DIR" --openvocab-run-dir "$OPENVOCAB_RUN_DIR" --matrix-run-dir "$MATRIX_RUN_DIR" --output "$EXP/reports/smoke_matrix_report.md" --seed 20260530 > "$LOG_DIR/report.log" 2>&1
report_code=$?

cat "$MATRIX_RUN_DIR/smoke_verdict.txt"

if [ "$closed_code" -ne 0 ] || [ "$matrix_code" -ne 0 ] || [ "$report_code" -ne 0 ]; then
  exit 1
fi

if [ "$open_code" -ne 0 ]; then
  exit 1
fi
