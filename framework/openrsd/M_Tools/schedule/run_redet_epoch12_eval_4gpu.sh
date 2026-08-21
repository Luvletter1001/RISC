#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/data1/zcy/OpenRSD}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data/DOTA2_1024_500}"
RUN_TS="${RUN_TS:-$(rtk /usr/bin/date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/work_dirs/scheduled_runs/may03_redet_epoch12_eval_${RUN_TS}}"
GPUS="${GPUS:-0,1,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MASTER_PORT_START="${MASTER_PORT_START:-31101}"
PYTHON_BIN="${PYTHON_BIN:-/data/zcy/anaconda3/envs/openrsd/bin/python}"

MODEL_NAME="ReDet_Re50_epoch12"
CONFIG="M_configs/DOTA2OfficialAdapters/redet_re50_refpn_dotav2_fullinit.py"
CHECKPOINT="$ROOT_DIR/work_dirs/dotav2_official_adapters/redet_re50_refpn_fullinit/epoch_12.pth"
EPOCH="12"
TRAIN_MAP="NA"

RESULTS_TSV="$OUT_ROOT/results.tsv"
SUMMARY_MD="$OUT_ROOT/summary.md"
RUN_LOG="$OUT_ROOT/run.log"

DATASETS=(
  "ss_val|ss_val/annfiles/|ss_val/images/"
  "rot_val_standard|rot_val_standard/annfiles/|rot_val_standard/images/"
  "rot_val_standard_0_90_180_270|rot_val_standard_0_90_180_270/annfiles/|rot_val_standard_0_90_180_270/images/"
  "rot_val_standard_30_60_120_150_210_240_300_330|rot_val_standard_30_60_120_150_210_240_300_330/annfiles/|rot_val_standard_30_60_120_150_210_240_300_330/images/"
)

log() {
  local line
  line="[$(rtk /usr/bin/date '+%F %T')] $*"
  printf '%s\n' "$line"
  printf '%s\n' "$line" >> "$RUN_LOG"
}

parse_metric_line() {
  local log_file="$1"
  local line map="NA" ap50="NA"
  while IFS= read -r line; do
    if [[ "$line" == *"dota/mAP:"* ]]; then
      if [[ "$line" =~ dota/mAP:[[:space:]]*([0-9.]+) ]]; then
        map="${BASH_REMATCH[1]}"
      fi
      if [[ "$line" =~ dota/AP50:[[:space:]]*([0-9.]+) ]]; then
        ap50="${BASH_REMATCH[1]}"
      fi
    fi
  done < "$log_file"
  printf '%s|%s\n' "$map" "$ap50"
}

build_summary() {
  {
    printf '# ReDet Epoch 12 Evaluation\n\n'
    printf -- '- generated_at: `%s`\n' "$(rtk /usr/bin/date '+%F %T')"
    printf -- '- output_root: `%s`\n' "$OUT_ROOT"
    printf -- '- checkpoint: `%s`\n' "$CHECKPOINT"
    printf -- '- GPUs: `%s`\n' "$GPUS"
    printf -- '- batch_size_per_gpu: `%s`\n\n' "$BATCH_SIZE"
    printf '| model | dataset | epoch | status | eval mAP | eval AP50 | log | predictions |\n'
    printf '|---|---|---:|---|---:|---:|---|---|\n'
    local model dataset epoch status map ap50 log_file pred_file
    while IFS='|' read -r model dataset epoch status map ap50 log_file pred_file; do
      [[ "$model" != "model" ]] || continue
      printf '| %s | %s | %s | %s | %s | %s | `%s` | `%s` |\n' \
        "$model" "$dataset" "$epoch" "$status" "$map" "$ap50" "$log_file" "$pred_file"
    done < "$RESULTS_TSV"
  } > "$SUMMARY_MD"
}

run_eval() {
  local run_index="$1"
  local dataset="$2"
  local ann_file="$3"
  local img_path="$4"
  local safe_dataset="${dataset//[^A-Za-z0-9_]/_}"
  local run_dir="$OUT_ROOT/${run_index}_${MODEL_NAME}_${safe_dataset}"
  local log_file="$run_dir/test.log"
  local pred_file="$run_dir/predictions.pkl"
  local master_port=$((MASTER_PORT_START + run_index))

  rtk /usr/bin/mkdir -p "$run_dir"
  log "Evaluating $MODEL_NAME on $dataset -> $run_dir"

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
  local exit_code=$?

  local status="OK"
  if [[ "$exit_code" -ne 0 ]]; then
    status="FAIL($exit_code)"
  fi

  local parsed map ap50
  parsed="$(parse_metric_line "$log_file")"
  IFS='|' read -r map ap50 <<< "$parsed"
  printf '%s|%s|%s|%s|%s|%s|%s|%s\n' \
    "$MODEL_NAME" "$dataset" "$EPOCH" "$status" "$map" "$ap50" "$log_file" "$pred_file" >> "$RESULTS_TSV"
  build_summary
  log "Finished $MODEL_NAME on $dataset: status=$status mAP=$map AP50=$ap50"
}

main() {
  cd "$ROOT_DIR" || exit 1
  rtk /usr/bin/mkdir -p "$OUT_ROOT"
  : > "$RUN_LOG"
  printf 'model|dataset|epoch|status|map|ap50|log|predictions\n' > "$RESULTS_TSV"
  build_summary

  if [[ ! -f "$CHECKPOINT" ]]; then
    log "Missing checkpoint: $CHECKPOINT"
    exit 2
  fi

  log "Starting $MODEL_NAME evaluation on GPUs=$GPUS"
  local run_index=0 dataset_entry dataset ann_file img_path
  for dataset_entry in "${DATASETS[@]}"; do
    IFS='|' read -r dataset ann_file img_path <<< "$dataset_entry"
    run_index=$((run_index + 1))
    run_eval "$run_index" "$dataset" "$ann_file" "$img_path"
  done
  build_summary
  log "Finished $MODEL_NAME queue. Summary: $SUMMARY_MD"
}

main "$@"
