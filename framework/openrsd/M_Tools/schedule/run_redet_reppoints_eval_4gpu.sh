#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/data1/zcy/OpenRSD}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data/DOTA2_1024_500}"
RUN_TS="${RUN_TS:-$(rtk /usr/bin/date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/work_dirs/scheduled_runs/may03_redet_reppoints_eval_${RUN_TS}}"
GPUS="${GPUS:-2,3,8,9}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MASTER_PORT_START="${MASTER_PORT_START:-30901}"
PYTHON_BIN="${PYTHON_BIN:-/data/zcy/anaconda3/envs/openrsd/bin/python}"

RESULTS_TSV="$OUT_ROOT/results.tsv"
SUMMARY_MD="$OUT_ROOT/summary.md"
RUN_LOG="$OUT_ROOT/run.log"

MODELS=(
  "ReDet_Re50_epoch4|M_configs/DOTA2OfficialAdapters/redet_re50_refpn_dotav2_fullinit.py|work_dirs/dotav2_official_adapters/redet_re50_refpn_fullinit/epoch_4.pth|4|0.4534"
  "RepPoints_R50_best|M_configs/DOTA2OfficialAdapters/oriented_reppoints_r50_fpn_dotav2_fullinit.py|work_dirs/scheduled_runs/may02_0167_full_series_20260502_120001/train_reppoints_fullinit/epoch_4.pth|4|0.4268"
)

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

path_abs() {
  local p="$1"
  if [[ "$p" == /* ]]; then
    printf '%s\n' "$p"
  else
    printf '%s/%s\n' "$ROOT_DIR" "$p"
  fi
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
    printf '# ReDet + RepPoints Parallel Evaluation\n\n'
    printf -- '- generated_at: `%s`\n' "$(rtk /usr/bin/date '+%F %T')"
    printf -- '- output_root: `%s`\n' "$OUT_ROOT"
    printf -- '- GPUs: `%s`\n' "$GPUS"
    printf -- '- nproc_per_node: `%s`\n' "$NPROC_PER_NODE"
    printf -- '- batch_size_per_gpu: `%s`\n' "$BATCH_SIZE"
    printf -- '- num_workers_per_rank: `%s`\n\n' "$NUM_WORKERS"
    printf '| model | dataset | epoch | train val mAP | status | eval mAP | eval AP50 | log | predictions |\n'
    printf '|---|---|---:|---:|---|---:|---:|---|---|\n'
    local model dataset epoch train_map status map ap50 log_file pred_file
    while IFS='|' read -r model dataset epoch train_map status map ap50 log_file pred_file; do
      [[ "$model" != "model" ]] || continue
      printf '| %s | %s | %s | %s | %s | %s | %s | `%s` | `%s` |\n' \
        "$model" "$dataset" "$epoch" "$train_map" "$status" "$map" "$ap50" "$log_file" "$pred_file"
    done < "$RESULTS_TSV"
  } > "$SUMMARY_MD"
}

run_eval() {
  local run_index="$1"
  local model="$2"
  local config="$3"
  local ckpt="$4"
  local epoch="$5"
  local train_map="$6"
  local dataset="$7"
  local ann_file="$8"
  local img_path="$9"

  local safe_model="${model//[^A-Za-z0-9_]/_}"
  local safe_dataset="${dataset//[^A-Za-z0-9_]/_}"
  local run_dir="$OUT_ROOT/${run_index}_${safe_model}_${safe_dataset}"
  local log_file="$run_dir/test.log"
  local pred_file="$run_dir/predictions.pkl"
  local master_port=$((MASTER_PORT_START + run_index))
  local ckpt_abs
  ckpt_abs="$(path_abs "$ckpt")"

  rtk /usr/bin/mkdir -p "$run_dir"
  log "Evaluating $model on $dataset -> $run_dir"

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
      "$config" \
      "$ckpt_abs" \
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

  local parsed map ap50 metric_line
  parsed="$(parse_metric_line "$log_file")"
  IFS='|' read -r map ap50 metric_line <<< "$parsed"
  printf '%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
    "$model" "$dataset" "$epoch" "$train_map" "$status" "$map" "$ap50" "$log_file" "$pred_file" >> "$RESULTS_TSV"
  build_summary
  log "Finished $model on $dataset: status=$status mAP=$map AP50=$ap50"
}

main() {
  cd "$ROOT_DIR" || exit 1
  rtk /usr/bin/mkdir -p "$OUT_ROOT"
  : > "$RUN_LOG"
  printf 'model|dataset|epoch|train_map|status|map|ap50|log|predictions\n' > "$RESULTS_TSV"
  build_summary

  log "Starting ReDet + RepPoints parallel queue on GPUs=$GPUS"
  local run_index=0 entry model config ckpt epoch train_map dataset_entry dataset ann_file img_path
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r model config ckpt epoch train_map <<< "$entry"
    for dataset_entry in "${DATASETS[@]}"; do
      IFS='|' read -r dataset ann_file img_path <<< "$dataset_entry"
      run_index=$((run_index + 1))
      run_eval "$run_index" "$model" "$config" "$ckpt" "$epoch" "$train_map" "$dataset" "$ann_file" "$img_path"
    done
  done
  build_summary
  log "Finished ReDet + RepPoints queue. Summary: $SUMMARY_MD"
}

main "$@"
