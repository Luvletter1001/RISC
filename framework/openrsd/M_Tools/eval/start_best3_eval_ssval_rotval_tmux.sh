#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/data1/zcy/OpenRSD}"
SESSION="${SESSION:-best3_eval_0167}"
GPUS="${GPUS:-0,1,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MASTER_PORT_START="${MASTER_PORT_START:-30501}"
REQUIRE_FREE_GPUS="${REQUIRE_FREE_GPUS:-1}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/work_dirs/dotav2_official_adapters/best3_eval_ssval_rotval_$(date +%Y%m%d_%H%M%S)}"

rtk /usr/bin/mkdir -p "$OUT_ROOT"

if rtk tmux has-session -t "$SESSION" >/dev/null 2>&1; then
  echo "tmux session already exists: $SESSION" >&2
  echo "Attach with: rtk tmux attach -t $SESSION" >&2
  exit 1
fi

rtk tmux new-session -d -s "$SESSION" -c "$ROOT_DIR" \
  "OUT_ROOT='$OUT_ROOT' GPUS='$GPUS' NPROC_PER_NODE='$NPROC_PER_NODE' BATCH_SIZE='$BATCH_SIZE' NUM_WORKERS='$NUM_WORKERS' MASTER_PORT_START='$MASTER_PORT_START' REQUIRE_FREE_GPUS='$REQUIRE_FREE_GPUS' rtk bash M_Tools/eval/run_best3_eval_ssval_rotval.sh > '$OUT_ROOT/launcher.log' 2>&1"

echo "Started tmux session: $SESSION"
echo "Output root: $OUT_ROOT"
echo "Live log: $OUT_ROOT/launcher.log"
echo "Final summary: $OUT_ROOT/summary.md"
echo "Attach: rtk tmux attach -t $SESSION"
