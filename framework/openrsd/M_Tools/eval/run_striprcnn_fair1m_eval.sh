#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/data1/zcy/OpenRSD}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data/fair1m/dair1m_1024}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/work_dirs/striprcnn_fair1m_eval_4ds_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0,1,4,5,6,7,8,9}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MASTER_PORT_START="${MASTER_PORT_START:-30601}"
PYTHON_BIN="${PYTHON_BIN:-/data/zcy/anaconda3/envs/openrsd/bin/python}"

CONFIG="/data1/zcy/OpenRSD/work_dirs/striprcnn_fair1m_eval.py"
CHECKPOINT="/data1/zcy/OpenRSD/weights/stripnet_s_fair1m.pth"

RESULTS_MD="$OUT_ROOT/results.md"
DETAILS_MD="$OUT_ROOT/details.md"

DATASETS=(
  "ss_val|val/annfiles/|val/images/"
  "rot_val_standard|rot_val_standard/annfiles/|rot_val_standard/images/"
  "rot_val_standard_0_90_180_270|rot_val_standard_0_90_180_270/annfiles/|rot_val_standard_0_90_180_270/images/"
  "rot_val_standard_30_60_120_150_210_240_300_330|rot_val_standard_30_60_120_150_210_240_300_330/annfiles/|rot_val_standard_30_60_120_150_210_240_300_330/images/"
)

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

parse_metric_line() {
  local log_file="$1"
  local line metric_line="" map="NA" ap50="NA"
  while IFS= read -r line; do
    if [[ "$line" == *"dota/mAP:"* ]]; then
      metric_line="$line"
      if [[ "$line" =~ dota/mAP:[[:space:]]*([0-9.]+) ]]; then
        map="${BASH_REMATCH[1]}"
      fi
      if [[ "$line" =~ dota/AP50:[[:space:]]*([0-9.]+) ]]; then
        ap50="${BASH_REMATCH[1]}"
      fi
    fi
  done < "$log_file"
  printf '%s|%s|%s\n' "$map" "$ap50" "$metric_line"
}

run_eval() {
  local dataset="$1"
  local ann_file="$2"
  local img_path="$3"
  local run_index="$4"

  local safe_dataset run_dir log_file pred_file status exit_code
  safe_dataset="${dataset//[^A-Za-z0-9_]/_}"
  run_dir="$OUT_ROOT/${run_index}_${safe_dataset}"
  log_file="$run_dir/test.log"
  pred_file="$run_dir/predictions.pkl"

  mkdir -p "$run_dir"
  log "Evaluating on $dataset -> $run_dir"

  local master_port=$((MASTER_PORT_START + run_index))

  (
    cd "$ROOT_DIR" || exit 1
    env \
      CUDA_VISIBLE_DEVICES="$GPUS" \
      NCCL_P2P_DISABLE=1 \
      NCCL_IB_DISABLE=1 \
      PYTHONNOUSERSITE=1 \
      MPLCONFIGDIR=/tmp/mplconfig \
      PYTHONPATH="$ROOT_DIR:$ROOT_DIR/tools" \
      "$PYTHON_BIN" -m torch.distributed.launch \
      "--nproc_per_node=$NPROC_PER_NODE" \
      "--master_port=$master_port" \
      tools/openrsd_test.py \
      "$CONFIG" \
      "$CHECKPOINT" \
      --launcher pytorch \
      --work-dir "$run_dir" \
      --out "$pred_file" \
      --cfg-options \
      "test_dataloader.batch_size=$BATCH_SIZE" \
      "test_dataloader.num_workers=$NUM_WORKERS" \
      "test_dataloader.dataset.data_root=$DATA_ROOT" \
      "test_dataloader.dataset.ann_file=$ann_file" \
      "test_dataloader.dataset.data_prefix.img_path=$img_path"
  ) > "$log_file" 2>&1
  exit_code=$?

  if [[ "$exit_code" -eq 0 ]]; then
    status="OK"
  else
    status="FAIL($exit_code)"
  fi

  local parsed map ap50 metric_line
  parsed="$(parse_metric_line "$log_file")"
  IFS='|' read -r map ap50 metric_line <<< "$parsed"

  printf '| %s | %s | %s | %s | %s |\n' \
    "$dataset" "$status" "$map" "$ap50" "$log_file" >> "$RESULTS_MD"
  log "Finished $dataset: status=$status mAP=$map AP50=$ap50"
}

main() {
  cd "$ROOT_DIR" || die "Cannot cd to $ROOT_DIR"
  mkdir -p "$OUT_ROOT"

  # Write results header
  {
    printf '# StripRCNN FAIR1M Evaluation Results\n\n'
    printf -- '- generated_at: `%s`\n' "$(date '+%F %T')"
    printf -- '- config: `%s`\n' "$CONFIG"
    printf -- '- checkpoint: `%s`\n' "$CHECKPOINT"
    printf -- '- GPUs: `%s`\n' "$GPUS"
    printf -- '- data_root: `%s`\n' "$DATA_ROOT"
    printf -- '- out_root: `%s`\n\n' "$OUT_ROOT"

    printf '## Results\n\n'
    printf '| Dataset | Status | mAP | AP50 | Log |\n'
    printf '|---|---|---|---|---|\n'
  } > "$RESULTS_MD"

  local dataset_entry dataset ann_file img_path
  local idx=0
  for dataset_entry in "${DATASETS[@]}"; do
    IFS='|' read -r dataset ann_file img_path <<< "$dataset_entry"
    idx=$((idx + 1))
    run_eval "$dataset" "$ann_file" "$img_path" "$idx"
  done

  log "All evaluations complete. Results: $RESULTS_MD"
}

main "$@"
