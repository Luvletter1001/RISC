#!/usr/bin/env bash
set -u

ROOT="/data1/zcy/OpenRSD"
EXP="$ROOT/experiments/rotation_semantic_attractor"

echo "[deprecated] run_smoke_test.sh now delegates to run_smoke_matrix.sh"
"$EXP/scripts/run_smoke_matrix.sh"
