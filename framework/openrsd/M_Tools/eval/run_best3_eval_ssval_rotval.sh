#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/data1/zcy/OpenRSD}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data/DOTA2_1024_500}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/work_dirs/dotav2_official_adapters/best3_eval_ssval_rotval_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0,1,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MASTER_PORT_START="${MASTER_PORT_START:-30501}"
PYTHON_BIN="${PYTHON_BIN:-/data/zcy/anaconda3/envs/openrsd/bin/python}"
REQUIRE_FREE_GPUS="${REQUIRE_FREE_GPUS:-1}"

RESULTS_TSV="$OUT_ROOT/results.tsv"
DETAILS_MD="$OUT_ROOT/details.md"
SUMMARY_MD="$OUT_ROOT/summary.md"

MODELS=(
  "ORCNN_R50|M_configs/DOTA2OfficialAdapters/oriented_rcnn_r50_fpn_dotav2_fullinit.py|work_dirs/dotav2_official_adapters/oriented_rcnn_r50_fpn_fullinit/epoch_8.pth|8|0.4503"
  "R3Det_KFIoU_R50|M_configs/DOTA2OfficialAdapters/r3det_kfiou_r50_fpn_dotav2_fullinit.py|work_dirs/dotav2_official_adapters/r3det_kfiou_r50_fpn_fullinit/epoch_12.pth|12|0.3813"
  "ReDet_Re50|M_configs/DOTA2OfficialAdapters/redet_re50_refpn_dotav2_fullinit.py|work_dirs/dotav2_official_adapters/redet_re50_refpn_fullinit/epoch_4.pth|4|0.4534"
)

DATASETS=(
  "ss_val|ss_val/annfiles/|ss_val/images/"
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

require_file() {
  [[ -f "$1" ]] || die "Missing file: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "Missing directory: $1"
}

is_selected_gpu() {
  local idx="$1"
  local old_ifs="$IFS"
  IFS=','
  for selected in $GPUS; do
    if [[ "$selected" == "$idx" ]]; then
      IFS="$old_ifs"
      return 0
    fi
  done
  IFS="$old_ifs"
  return 1
}

check_selected_gpus_free() {
  if [[ "$REQUIRE_FREE_GPUS" != "1" ]]; then
    log "Skipping GPU compute-process check because REQUIRE_FREE_GPUS=$REQUIRE_FREE_GPUS"
    return 0
  fi

  declare -A bus_to_index=()
  local line idx bus busy=0
  while IFS=',' read -r idx bus; do
    idx="${idx//[[:space:]]/}"
    bus="${bus//[[:space:]]/}"
    [[ -n "$idx" && -n "$bus" ]] || continue
    bus_to_index["$bus"]="$idx"
  done < <(rtk nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader)

  while IFS=',' read -r bus pid proc mem; do
    bus="${bus//[[:space:]]/}"
    pid="${pid//[[:space:]]/}"
    proc="${proc#"${proc%%[![:space:]]*}"}"
    mem="${mem//[[:space:]]/}"
    idx="${bus_to_index[$bus]:-}"
    [[ -n "$idx" ]] || continue
    if is_selected_gpu "$idx"; then
      log "GPU $idx already has compute process pid=$pid mem=${mem}MiB proc=$proc"
      busy=1
    fi
  done < <(rtk nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader,nounits)

  if [[ "$busy" -ne 0 ]]; then
    die "Selected GPUs ($GPUS) are not free. Set REQUIRE_FREE_GPUS=0 only if you intentionally want to share them."
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

append_class_table() {
  local log_file="$1"
  local in_table=0
  local line
  while IFS= read -r line; do
    if [[ "$line" == *"| class"* && "$line" == *"| ap"* ]]; then
      in_table=1
    fi
    if [[ "$in_table" -eq 1 ]]; then
      printf '%s\n' "$line" >> "$DETAILS_MD"
      if [[ "$line" == *"| mAP"* ]]; then
        break
      fi
    fi
  done < "$log_file"
}

append_detail() {
  local model="$1"
  local dataset="$2"
  local status="$3"
  local map="$4"
  local ap50="$5"
  local metric_line="$6"
  local run_dir="$7"
  local log_file="$8"
  local pred_file="$9"

  {
    printf '\n## %s / %s\n\n' "$model" "$dataset"
    printf -- '- status: `%s`\n' "$status"
    printf -- '- mAP: `%s`\n' "$map"
    printf -- '- AP50: `%s`\n' "$ap50"
    printf -- '- work_dir: `%s`\n' "$run_dir"
    printf -- '- predictions: `%s`\n' "$pred_file"
    printf -- '- log: `%s`\n' "$log_file"
    if [[ -n "$metric_line" ]]; then
      printf -- '- metric line: `%s`\n' "$metric_line"
    fi
    printf '\n'
  } >> "$DETAILS_MD"

  if [[ "$status" == "OK" ]]; then
    append_class_table "$log_file"
    printf '\n' >> "$DETAILS_MD"
  else
    {
      printf 'Last 80 log lines:\n\n'
      printf '```text\n'
    } >> "$DETAILS_MD"
    rtk /usr/bin/tail -n 80 "$log_file" >> "$DETAILS_MD" 2>/dev/null || true
    printf '```\n' >> "$DETAILS_MD"
  fi
}

build_summary_md() {
  {
    printf '# DOTAv2 Best3 Evaluation Summary\n\n'
    printf -- '- generated_at: `%s`\n' "$(date '+%F %T')"
    printf -- '- GPUs: `%s`\n' "$GPUS"
    printf -- '- nproc_per_node: `%s`\n' "$NPROC_PER_NODE"
    printf -- '- batch_size_per_gpu: `%s`\n' "$BATCH_SIZE"
    printf -- '- num_workers_per_rank: `%s`\n' "$NUM_WORKERS"
    printf -- '- data_root: `%s`\n' "$DATA_ROOT"
    printf -- '- output_root: `%s`\n\n' "$OUT_ROOT"

    printf '## Selected Checkpoints\n\n'
    printf '| model | best epoch | training val mAP | checkpoint |\n'
    printf '|---|---:|---:|---|\n'
    local entry model config ckpt epoch train_map
    for entry in "${MODELS[@]}"; do
      IFS='|' read -r model config ckpt epoch train_map <<< "$entry"
      printf '| %s | %s | %s | `%s` |\n' "$model" "$epoch" "$train_map" "$ROOT_DIR/$ckpt"
    done

    printf '\n## Evaluation Results\n\n'
    printf '| model | dataset | best epoch | train val mAP | status | eval mAP | eval AP50 | log | predictions |\n'
    printf '|---|---|---:|---:|---|---:|---:|---|---|\n'
    local line dataset status map ap50 log_file pred_file
    while IFS='|' read -r model dataset epoch train_map status map ap50 log_file pred_file; do
      [[ "$model" != "model" ]] || continue
      printf '| %s | %s | %s | %s | %s | %s | %s | `%s` | `%s` |\n' \
        "$model" "$dataset" "$epoch" "$train_map" "$status" "$map" "$ap50" "$log_file" "$pred_file"
    done < "$RESULTS_TSV"

    printf '\n## Details\n'
    if [[ -f "$DETAILS_MD" ]]; then
      while IFS= read -r line; do
        printf '%s\n' "$line"
      done < "$DETAILS_MD"
    fi
  } > "$SUMMARY_MD"
}

run_eval() {
  local model="$1"
  local config="$2"
  local ckpt="$3"
  local epoch="$4"
  local train_map="$5"
  local dataset="$6"
  local ann_file="$7"
  local img_path="$8"
  local run_index="$9"

  local safe_model safe_dataset run_dir log_file pred_file status exit_code
  safe_model="${model//[^A-Za-z0-9_]/_}"
  safe_dataset="${dataset//[^A-Za-z0-9_]/_}"
  run_dir="$OUT_ROOT/${run_index}_${safe_model}_${safe_dataset}"
  log_file="$run_dir/test.log"
  pred_file="$run_dir/predictions.pkl"

  rtk /usr/bin/mkdir -p "$run_dir"
  log "Running $model on $dataset -> $run_dir"

  local master_port=$((MASTER_PORT_START + run_index - 1))
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
      "$ckpt" \
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

  printf '%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
    "$model" "$dataset" "$epoch" "$train_map" "$status" "$map" "$ap50" "$log_file" "$pred_file" >> "$RESULTS_TSV"
  append_detail "$model" "$dataset" "$status" "$map" "$ap50" "$metric_line" "$run_dir" "$log_file" "$pred_file"
  build_summary_md

  log "Finished $model on $dataset: status=$status mAP=$map AP50=$ap50"
  [[ "$exit_code" -eq 0 ]]
}

main() {
  cd "$ROOT_DIR" || die "Cannot cd to $ROOT_DIR"
  rtk /usr/bin/mkdir -p "$OUT_ROOT"

  local dataset_entry dataset ann_file img_path
  for dataset_entry in "${DATASETS[@]}"; do
    IFS='|' read -r dataset ann_file img_path <<< "$dataset_entry"
    require_dir "$DATA_ROOT/${ann_file%/}"
    require_dir "$DATA_ROOT/${img_path%/}"
  done

  local entry model config ckpt epoch train_map
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r model config ckpt epoch train_map <<< "$entry"
    require_file "$ROOT_DIR/$config"
    require_file "$ROOT_DIR/$ckpt"
  done

  check_selected_gpus_free

  printf 'model|dataset|epoch|train_map|status|map|ap50|log|predictions\n' > "$RESULTS_TSV"
  : > "$DETAILS_MD"
  build_summary_md

  log "Output root: $OUT_ROOT"
  log "Summary md: $SUMMARY_MD"

  local failed=0 run_index=0
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r model config ckpt epoch train_map <<< "$entry"
    for dataset_entry in "${DATASETS[@]}"; do
      IFS='|' read -r dataset ann_file img_path <<< "$dataset_entry"
      run_index=$((run_index + 1))
      if ! run_eval "$model" "$config" "$ckpt" "$epoch" "$train_map" "$dataset" "$ann_file" "$img_path" "$run_index"; then
        failed=1
      fi
    done
  done

  build_summary_md
  log "All runs finished. Summary: $SUMMARY_MD"
  exit "$failed"
}

main "$@"
