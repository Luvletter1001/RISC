#!/usr/bin/env bash
set -u

ROOT="/data1/zcy/OpenRSD"
EXP="$ROOT/experiments/rotation_semantic_attractor"
PY="${PY:-/data/zcy/anaconda3/envs/openrsd/bin/python}"
RUN_DIR="${RUN_DIR:-$EXP/outputs/smoke/openvocab}"
SOURCE_RUN_DIR="${OPENVOCAB_SOURCE_RUN_DIR:-$EXP/outputs/smoke/closedset}"
INVENTORY_DIR="${OPENVOCAB_INVENTORY_DIR:-$EXP/outputs/open_vocab_assets}"
INVENTORY_JSON="$INVENTORY_DIR/open_vocab_asset_inventory.json"
DEVICE="${DEVICE:-cuda:0}"
ANGLES="${OPENVOCAB_ANGLES:-0}"
LIMIT="${OPENVOCAB_LIMIT:-1}"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7,8,9}"
cd "$ROOT" || exit 1

echo "[openvocab-smoke] inventory"
"$PY" "$EXP/scripts/12_open_vocab_asset_inventory.py" --output-dir "$INVENTORY_DIR" > "$LOG_DIR/inventory.log" 2>&1
inventory_code=$?

echo "[openvocab-smoke] openrsd_hook_smoke"
"$PY" "$EXP/scripts/15_openrsd_hook_smoke.py" \
  --inventory "$INVENTORY_JSON" \
  --source-run-dir "$SOURCE_RUN_DIR" \
  --output-dir "$RUN_DIR" \
  --report-dir "$EXP/reports" \
  --angles "$ANGLES" \
  --limit "$LIMIT" \
  --device "$DEVICE" \
  --seed 20260530 > "$LOG_DIR/openrsd_hook_smoke.log" 2>&1
hook_code=$?

status="FAILED"
if [ "$inventory_code" -eq 0 ] && [ "$hook_code" -eq 0 ]; then
  status=$(grep -o 'status=.*' "$LOG_DIR/openrsd_hook_smoke.log" | tail -n 1 | cut -d= -f2)
fi

if [ "$status" = "NOT_AVAILABLE_ASSET" ]; then
  verdict="PASS_SCHEMA_ONLY"
elif [ "$status" = "UNSUPPORTED_BY_CURRENT_CODE" ]; then
  verdict="PASS_WITH_UNSUPPORTED_HOOKS"
elif [ "$status" = "DONE_SMOKE" ]; then
  verdict="PASS"
else
  verdict="FAIL"
fi

{
  echo "verdict=$verdict"
  echo "run_dir=$RUN_DIR"
  echo "openvocab_status=$status"
  echo "source_run_dir=$SOURCE_RUN_DIR"
  echo "inventory_json=$INVENTORY_JSON"
  echo "device=$DEVICE"
  echo "angles=$ANGLES"
  echo "limit=$LIMIT"
} | tee "$RUN_DIR/smoke_verdict.txt"

if [ "$verdict" = "FAIL" ]; then
  exit 1
fi
