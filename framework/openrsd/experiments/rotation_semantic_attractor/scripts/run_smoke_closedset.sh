#!/usr/bin/env bash
set -u

ROOT="/data1/zcy/OpenRSD"
EXP="$ROOT/experiments/rotation_semantic_attractor"
PY="${PY:-/data/zcy/anaconda3/envs/openrsd/bin/python}"
RUN_DIR="${RUN_DIR:-$EXP/outputs/smoke/closedset}"
MODEL="${MODEL:-rotated_retinanet_msrr}"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7,8,9}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/openrsd_mpl_rotation_semantic_attractor}"
mkdir -p "$MPLCONFIGDIR"

cd "$ROOT" || exit 1

verdict="FAIL"
notes=()

run_step() {
  local name="$1"
  shift
  echo "[closedset-smoke] $name"
  "$@" > "$LOG_DIR/${name}.log" 2>&1
  local code=$?
  if [ "$code" -ne 0 ]; then
    notes+=("$name failed with exit code $code; see $LOG_DIR/${name}.log")
    return "$code"
  fi
  return 0
}

run_step inventory "$PY" "$EXP/scripts/00_inventory.py" --output-dir "$EXP/outputs/inventory" --seed 20260530
run_step capability_matrix "$PY" "$EXP/scripts/04_build_capability_matrix.py" --output-dir "$EXP/outputs" --seed 20260530
run_step build_split "$PY" "$EXP/scripts/01_build_tile_split.py" --dataset DOTA --limit 2 --output-dir "$EXP/outputs/splits" --seed 20260530
run_step angle_sweep "$PY" "$EXP/scripts/03_run_angle_sweep.py" --split "$EXP/outputs/splits/S0_discovery_debug.json" --models "$MODEL" --angles 0,90 --limit 2 --output-dir "$RUN_DIR" --seed 20260530
run_step false_hub "$PY" "$EXP/scripts/05_eval_false_hub_taxonomy.py" --run-dir "$RUN_DIR" --iou-thr 0.30 --seed 20260530
run_step stage_decomposition "$PY" "$EXP/scripts/06_stage_decomposition.py" --run-dir "$RUN_DIR" --seed 20260530
run_step open_vocab_intervention "$PY" "$EXP/scripts/07_intervention_open_vocab.py" --run-dir "$RUN_DIR" --seed 20260530
run_step closedset_intervention "$PY" "$EXP/scripts/08_intervention_closed_set.py" --run-dir "$RUN_DIR" --seed 20260530
run_step context_counterfactual "$PY" "$EXP/scripts/09_context_counterfactual.py" --run-dir "$RUN_DIR" --seed 20260530
run_step dehub_safety "$PY" "$EXP/scripts/10_eval_dehub_safety.py" --run-dir "$RUN_DIR" --seed 20260530
run_step build_report "$PY" "$EXP/scripts/11_build_report.py" --run-dir "$RUN_DIR" --output "$EXP/reports/smoke_closedset_report.md" --seed 20260530

prediction_count=$(find "$RUN_DIR/canonical_predictions" -name 'angle_*.json' 2>/dev/null | wc -l | tr -d ' ')
false_hub_csv="$RUN_DIR/metrics/false_hub_tile_angle.csv"
stage_csv="$RUN_DIR/metrics/stage_decomposition_summary.csv"
report_md="$EXP/reports/smoke_closedset_report.md"

if [ "$prediction_count" -ge 4 ] && [ -s "$false_hub_csv" ] && [ -s "$stage_csv" ] && [ -s "$report_md" ]; then
  if grep -q "UNSUPPORTED_BY_CURRENT_CODE" "$stage_csv"; then
    verdict="PASS_WITH_UNSUPPORTED_HOOKS"
  elif grep -q "NOT_APPLICABLE" "$stage_csv" || grep -q "NOT_SELECTED_IN_THIS_SMOKE" "$RUN_DIR/metrics/open_vocab_intervention.csv"; then
    verdict="PASS_WITH_NOT_APPLICABLE"
  else
    verdict="PASS"
  fi
else
  notes+=("missing required closed-set smoke outputs: predictions=$prediction_count false_hub=$false_hub_csv stage=$stage_csv report=$report_md")
fi

{
  echo "verdict=$verdict"
  echo "run_dir=$RUN_DIR"
  echo "prediction_count=$prediction_count"
  echo "false_hub_csv=$false_hub_csv"
  echo "stage_csv=$stage_csv"
  echo "report_md=$report_md"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  if [ "${#notes[@]}" -gt 0 ]; then
    echo "notes:"
    printf -- '- %s\n' "${notes[@]}"
  fi
} | tee "$RUN_DIR/smoke_verdict.txt"

if [ "$verdict" = "FAIL" ]; then
  exit 1
fi
