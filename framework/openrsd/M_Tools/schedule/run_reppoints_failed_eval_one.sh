#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/data1/zcy/OpenRSD}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data/DOTA2_1024_500}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
DATASET_NAME="${DATASET_NAME:?DATASET_NAME is required}"
ANN_FILE="${ANN_FILE:?ANN_FILE is required}"
IMG_PATH="${IMG_PATH:?IMG_PATH is required}"
GPUS="${GPUS:?GPUS is required}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MASTER_PORT="${MASTER_PORT:-31101}"
PYTHON_BIN="${PYTHON_BIN:-/data/zcy/anaconda3/envs/openrsd/bin/python}"

MODEL_NAME="RepPoints_R50_best"
CONFIG="M_configs/DOTA2OfficialAdapters/oriented_reppoints_r50_fpn_dotav2_fullinit.py"
CHECKPOINT="$ROOT_DIR/work_dirs/scheduled_runs/may02_0167_full_series_20260502_120001/train_reppoints_fullinit/epoch_4.pth"
RESULTS_TSV="$OUT_ROOT/results.tsv"
SUMMARY_MD="$OUT_ROOT/summary.md"
RUN_LOG="$OUT_ROOT/run.log"
RUN_DIR="$OUT_ROOT/${MODEL_NAME}_${DATASET_NAME}_b${BATCH_SIZE}"
TEST_LOG="$RUN_DIR/test.log"
PRED_FILE="$RUN_DIR/predictions.pkl"

log() {
  local line
  line="[$(rtk /usr/bin/date '+%F %T')] $*"
  printf '%s\n' "$line"
  printf '%s\n' "$line" >> "$RUN_LOG"
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

build_summary() {
  {
    printf '# RepPoints Failed Dataset Retry\n\n'
    printf -- '- generated_at: `%s`\n' "$(rtk /usr/bin/date '+%F %T')"
    printf -- '- output_root: `%s`\n' "$OUT_ROOT"
    printf -- '- dataset: `%s`\n' "$DATASET_NAME"
    printf -- '- GPUs: `%s`\n' "$GPUS"
    printf -- '- batch_size_per_gpu: `%s`\n' "$BATCH_SIZE"
    printf -- '- num_workers_per_rank: `%s`\n\n' "$NUM_WORKERS"
    printf '| model | dataset | status | eval mAP | eval AP50 | log | predictions |\n'
    printf '|---|---|---|---:|---:|---|---|\n'
    local model dataset status map ap50 log_file pred_file
    while IFS='|' read -r model dataset status map ap50 log_file pred_file; do
      [[ "$model" != "model" ]] || continue
      printf '| %s | %s | %s | %s | %s | `%s` | `%s` |\n' \
        "$model" "$dataset" "$status" "$map" "$ap50" "$log_file" "$pred_file"
    done < "$RESULTS_TSV"
  } > "$SUMMARY_MD"
}

main() {
  cd "$ROOT_DIR" || exit 1
  rtk /usr/bin/mkdir -p "$RUN_DIR"
  : > "$RUN_LOG"
  printf 'model|dataset|status|map|ap50|log|predictions\n' > "$RESULTS_TSV"
  build_summary

  log "Retrying $MODEL_NAME on $DATASET_NAME with GPUs=$GPUS batch_size=$BATCH_SIZE"
  (
    cd "$ROOT_DIR" || exit 1
    rtk env \
      CUDA_VISIBLE_DEVICES="$GPUS" \
      NCCL_P2P_DISABLE=1 \
      NCCL_IB_DISABLE=1 \
      PYTHONNOUSERSITE=1 \
      MPLCONFIGDIR=/tmp/mplconfig \
      PYTHONPATH="$ROOT_DIR:$ROOT_DIR/tools" \
      "$PYTHON_BIN" -m torch.distributed.launch \
      "--nproc_per_node=$NPROC_PER_NODE" \
      "--master_port=$MASTER_PORT" \
      tools/openrsd_test.py \
      "$CONFIG" \
      "$CHECKPOINT" \
      --launcher pytorch \
      --work-dir "$RUN_DIR" \
      --out "$PRED_FILE" \
      --cfg-options \
      "test_dataloader.batch_size=$BATCH_SIZE" \
      "test_dataloader.num_workers=$NUM_WORKERS" \
      "test_dataloader.dataset.data_root=$DATA_ROOT" \
      "test_dataloader.dataset.ann_file=$ANN_FILE" \
      "test_dataloader.dataset.data_prefix.img_path=$IMG_PATH"
  ) > "$TEST_LOG" 2>&1
  local exit_code=$?

  local status="OK"
  if [[ "$exit_code" -ne 0 ]]; then
    status="FAIL($exit_code)"
  fi
  local parsed map ap50 metric_line
  parsed="$(parse_metric_line "$TEST_LOG")"
  IFS='|' read -r map ap50 metric_line <<< "$parsed"
  printf '%s|%s|%s|%s|%s|%s|%s\n' \
    "$MODEL_NAME" "$DATASET_NAME" "$status" "$map" "$ap50" "$TEST_LOG" "$PRED_FILE" >> "$RESULTS_TSV"
  build_summary
  log "Finished $MODEL_NAME on $DATASET_NAME: status=$status mAP=$map AP50=$ap50"
  exit "$exit_code"
}

main "$@"
