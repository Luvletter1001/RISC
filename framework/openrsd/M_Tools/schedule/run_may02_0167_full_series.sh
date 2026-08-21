#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/data1/zcy/OpenRSD}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data/DOTA2_1024_500}"
RUN_TS="${RUN_TS:-$(rtk /usr/bin/date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/work_dirs/scheduled_runs/may02_0167_full_series_$RUN_TS}"
GPUS="${GPUS:-0,1,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MASTER_PORT_START="${MASTER_PORT_START:-30701}"
PYTHON_BIN="${PYTHON_BIN:-/data/zcy/anaconda3/envs/openrsd/bin/python}"
REQUIRE_FREE_GPUS="${REQUIRE_FREE_GPUS:-1}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_REPPOINTS_TRAIN="${SKIP_REPPOINTS_TRAIN:-0}"
REPPOINTS_EXISTING_CKPT="${REPPOINTS_EXISTING_CKPT:-}"
REPPOINTS_EXISTING_EPOCH="${REPPOINTS_EXISTING_EPOCH:-NA}"
REPPOINTS_EXISTING_MAP="${REPPOINTS_EXISTING_MAP:-NA}"

SCHEDULER_LOG="$OUT_ROOT/scheduler.log"
RESULTS_TSV="$OUT_ROOT/results.tsv"
DETAILS_MD="$OUT_ROOT/details.md"
SUMMARY_MD="$OUT_ROOT/summary.md"

ORCNN_CONFIG="M_configs/DOTA2OfficialAdapters/oriented_rcnn_r50_fpn_dotav2_fullinit.py"
R3DET_CONFIG="M_configs/DOTA2OfficialAdapters/r3det_kfiou_r50_fpn_dotav2_fullinit.py"
REDET_CONFIG="M_configs/DOTA2OfficialAdapters/redet_re50_refpn_dotav2_fullinit.py"
REPPOINTS_CONFIG="M_configs/DOTA2OfficialAdapters/oriented_reppoints_r50_fpn_dotav2_fullinit.py"
ORCNN_IMAGENET_CONFIG="M_configs/DOTA2OfficialAdapters/oriented_rcnn_r50_fpn_dotav2_imagenetonly.py"
ORCNN_24E_CONFIG="M_configs/DOTA2OfficialAdapters/oriented_rcnn_r50_fpn_dotav2_fullinit_24e.py"
ORCNN_ROTATE_CONFIG="M_configs/DOTA2OfficialAdapters/oriented_rcnn_r50_fpn_dotav2_fullinit_randomrotate.py"
LSKNET_CONFIG="work_dirs/lsknet_dotav2_ss_orcnn_bs2_fullinit/G02_Baselines_Data1_DOTA2_M5_ORCNN_LSKNet.py"

ORCNN_EPOCH8_CKPT="work_dirs/dotav2_official_adapters/oriented_rcnn_r50_fpn_fullinit/epoch_8.pth"
ORCNN_EPOCH12_CKPT="work_dirs/dotav2_official_adapters/oriented_rcnn_r50_fpn_fullinit/epoch_12.pth"
R3DET_CKPT="work_dirs/dotav2_official_adapters/r3det_kfiou_r50_fpn_fullinit/epoch_12.pth"
REDET_CKPT="work_dirs/dotav2_official_adapters/redet_re50_refpn_fullinit/epoch_4.pth"
LSKNET_CKPT="work_dirs/lsknet_dotav2_ss_orcnn_bs2_fullinit/epoch_4.pth"

DATASETS=(
  "ss_val|ss_val/annfiles/|ss_val/images/"
  "rot_val_standard|rot_val_standard/annfiles/|rot_val_standard/images/"
  "rot_val_standard_0_90_180_270|rot_val_standard_0_90_180_270/annfiles/|rot_val_standard_0_90_180_270/images/"
  "rot_val_standard_30_60_120_150_210_240_300_330|rot_val_standard_30_60_120_150_210_240_300_330/annfiles/|rot_val_standard_30_60_120_150_210_240_300_330/images/"
)

BASELINE_MODELS=(
  "ORCNN_R50_epoch8|$ORCNN_CONFIG|$ORCNN_EPOCH8_CKPT|8|0.4503"
  "R3Det_KFIoU_R50_epoch12|$R3DET_CONFIG|$R3DET_CKPT|12|0.3813"
  "ReDet_Re50_epoch4|$REDET_CONFIG|$REDET_CKPT|4|0.4534"
)

SELECTED_MODELS=()
RUN_INDEX=0
TRAIN_INDEX=0
FAILED=0

log() {
  local ts line
  ts="$(rtk /usr/bin/date '+%F %T')"
  line="[$ts] $*"
  printf '%s\n' "$line"
  printf '%s\n' "$line" >> "$SCHEDULER_LOG"
}

die() {
  log "ERROR: $*"
  exit 1
}

path_abs() {
  local p="$1"
  if [[ "$p" == /* ]]; then
    printf '%s\n' "$p"
  else
    printf '%s/%s\n' "$ROOT_DIR" "$p"
  fi
}

require_file() {
  local abs
  abs="$(path_abs "$1")"
  [[ -f "$abs" ]] || die "Missing file: $abs"
}

require_dir() {
  local abs
  abs="$(path_abs "$1")"
  [[ -d "$abs" ]] || die "Missing directory: $abs"
}

format_command() {
  local arg q out=""
  for arg in "$@"; do
    printf -v q '%q' "$arg"
    out+="$q "
  done
  printf '%s' "${out% }"
}

is_selected_gpu() {
  local idx="$1"
  local old_ifs="$IFS"
  local selected
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
  local line idx bus pid proc mem busy=0
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
    return 1
  fi
  return 0
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

append_result() {
  local stage="$1"
  local model="$2"
  local dataset="$3"
  local epoch="$4"
  local train_map="$5"
  local status="$6"
  local eval_map="$7"
  local eval_ap50="$8"
  local checkpoint="$9"
  local log_file="${10}"
  local pred_file="${11}"

  printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
    "$stage" "$model" "$dataset" "$epoch" "$train_map" "$status" \
    "$eval_map" "$eval_ap50" "$checkpoint" "$log_file" "$pred_file" >> "$RESULTS_TSV"
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
  local stage="$1"
  local model="$2"
  local dataset="$3"
  local status="$4"
  local map="$5"
  local ap50="$6"
  local metric_line="$7"
  local run_dir="$8"
  local log_file="$9"
  local pred_file="${10}"

  {
    printf '\n## %s / %s / %s\n\n' "$stage" "$model" "$dataset"
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
  elif [[ -f "$log_file" ]]; then
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
    printf '# May 02 0167 Full Series Summary\n\n'
    printf -- '- generated_at: `%s`\n' "$(rtk /usr/bin/date '+%F %T')"
    printf -- '- output_root: `%s`\n' "$OUT_ROOT"
    printf -- '- GPUs: `%s`\n' "$GPUS"
    printf -- '- nproc_per_node: `%s`\n' "$NPROC_PER_NODE"
    printf -- '- eval_batch_size_per_gpu: `%s`\n' "$BATCH_SIZE"
    printf -- '- eval_num_workers_per_rank: `%s`\n' "$NUM_WORKERS"
    printf -- '- data_root: `%s`\n' "$DATA_ROOT"
    printf -- '- NCCL_P2P_DISABLE: `1`\n'
    printf -- '- NCCL_IB_DISABLE: `1`\n'
    printf -- '- dry_run: `%s`\n\n' "$DRY_RUN"

    printf '## Datasets\n\n'
    printf '| dataset | ann_file | image_path |\n'
    printf '|---|---|---|\n'
    local dataset_entry dataset ann_file img_path
    for dataset_entry in "${DATASETS[@]}"; do
      IFS='|' read -r dataset ann_file img_path <<< "$dataset_entry"
      printf '| %s | `%s` | `%s` |\n' "$dataset" "$DATA_ROOT/$ann_file" "$DATA_ROOT/$img_path"
    done

    printf '\n## Selected Evaluation Checkpoints\n\n'
    printf '| model | epoch | training val mAP | checkpoint |\n'
    printf '|---|---:|---:|---|\n'
    local entry model config ckpt epoch train_map
    for entry in "${SELECTED_MODELS[@]}"; do
      IFS='|' read -r model config ckpt epoch train_map <<< "$entry"
      printf '| %s | %s | %s | `%s` |\n' "$model" "$epoch" "$train_map" "$(path_abs "$ckpt")"
    done

    printf '\n## Results\n\n'
    printf '| stage | model | dataset | epoch | train val mAP | status | eval mAP | eval AP50 | checkpoint | log | predictions |\n'
    printf '|---|---|---|---:|---:|---|---:|---:|---|---|---|\n'
    local stage status map ap50 log_file pred_file
    while IFS='|' read -r stage model dataset epoch train_map status map ap50 ckpt log_file pred_file; do
      [[ "$stage" != "stage" ]] || continue
      printf '| %s | %s | %s | %s | %s | %s | %s | %s | `%s` | `%s` | `%s` |\n' \
        "$stage" "$model" "$dataset" "$epoch" "$train_map" "$status" \
        "$map" "$ap50" "$ckpt" "$log_file" "$pred_file"
    done < "$RESULTS_TSV"

    printf '\n## Notes\n\n'
    printf -- '- `four_model_four_dataset` evaluates ORCNN epoch 8, R3Det epoch 12, ReDet epoch 4, and RepPoints best if RepPoints training succeeds.\n'
    printf -- '- `orcnn_epoch12_comparison` adds ORCNN epoch 12; compare it with the ORCNN epoch 8 rows above.\n'
    printf -- '- If a train/eval task fails, the failure is recorded here and the next task continues.\n\n'

    printf '## Details\n'
    if [[ -f "$DETAILS_MD" ]]; then
      while IFS= read -r line; do
        printf '%s\n' "$line"
      done < "$DETAILS_MD"
    fi
  } > "$SUMMARY_MD"
}

select_best_checkpoint() {
  local log_file="$1"
  local work_dir="$2"
  rtk "$PYTHON_BIN" - "$log_file" "$work_dir" <<'PY'
import os
import re
import sys

log_path, work_dir = sys.argv[1], sys.argv[2]
epoch_re = re.compile(r'Epoch\(val\)\s*\[([0-9]+)\]')
map_re = re.compile(r'dota/mAP:\s*([0-9]*\.?[0-9]+)')
records = []
last_epoch = None

try:
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            epoch_match = epoch_re.search(line)
            if epoch_match:
                last_epoch = int(epoch_match.group(1))
            map_match = map_re.search(line)
            if map_match and last_epoch is not None:
                records.append((float(map_match.group(1)), last_epoch, map_match.group(1)))
except FileNotFoundError:
    pass

if records:
    best_map, best_epoch, best_map_text = max(records, key=lambda x: (x[0], x[1]))
    print(f"{best_epoch}|{best_map_text}|{os.path.join(work_dir, f'epoch_{best_epoch}.pth')}")
    raise SystemExit(0)

last_checkpoint = os.path.join(work_dir, 'last_checkpoint')
if os.path.exists(last_checkpoint):
    with open(last_checkpoint, 'r', encoding='utf-8', errors='replace') as f:
        checkpoint = f.read().strip()
    if checkpoint and not os.path.isabs(checkpoint):
        checkpoint = os.path.join(work_dir, checkpoint)
    print(f"NA|NA|{checkpoint}")
else:
    print("NA|NA|")
PY
}

run_train() {
  local model="$1"
  local config="$2"
  local work_dir="$3"
  local stage="$4"
  local train_log="$work_dir/train.log"
  local master_port exit_code status

  TRAIN_INDEX=$((TRAIN_INDEX + 1))
  master_port=$((MASTER_PORT_START + 500 + TRAIN_INDEX))
  rtk /usr/bin/mkdir -p "$work_dir" || return 1

  local cmd=(
    rtk env
    CUDA_VISIBLE_DEVICES="$GPUS"
    NCCL_P2P_DISABLE=1
    NCCL_IB_DISABLE=1
    PYTHONNOUSERSITE=1
    MPLCONFIGDIR=/tmp/mplconfig
    PYTHONPATH="$ROOT_DIR:$ROOT_DIR/tools"
    "$PYTHON_BIN" -m torch.distributed.launch
    "--nproc_per_node=$NPROC_PER_NODE"
    "--master_port=$master_port"
    tools/train.py
    "$config"
    --launcher pytorch
    --work-dir "$work_dir"
  )

  log "Training $model -> $work_dir"
  log "Command: $(format_command "${cmd[@]}")"

  if [[ "$DRY_RUN" == "1" ]]; then
    {
      printf 'DRY_RUN training command:\n'
      format_command "${cmd[@]}"
      printf '\n'
    } > "$train_log"
    append_result "$stage" "$model" "TRAIN" "NA" "NA" "DRY_RUN" "NA" "NA" "" "$train_log" ""
    build_summary_md
    return 0
  fi

  (
    cd "$ROOT_DIR" || exit 1
    "${cmd[@]}"
  ) > "$train_log" 2>&1
  exit_code=$?

  if [[ "$exit_code" -eq 0 ]]; then
    status="OK"
  else
    status="FAIL($exit_code)"
    FAILED=1
  fi

  append_result "$stage" "$model" "TRAIN" "NA" "NA" "$status" "NA" "NA" "" "$train_log" ""
  append_detail "$stage" "$model" "TRAIN" "$status" "NA" "NA" "" "$work_dir" "$train_log" ""
  build_summary_md
  log "Finished training $model: status=$status"
  [[ "$exit_code" -eq 0 ]]
}

run_eval() {
  local stage="$1"
  local model="$2"
  local config="$3"
  local ckpt="$4"
  local epoch="$5"
  local train_map="$6"
  local dataset="$7"
  local ann_file="$8"
  local img_path="$9"
  local run_index="${10}"

  local safe_model safe_dataset run_dir log_file pred_file master_port
  local status exit_code parsed map ap50 metric_line ckpt_abs
  safe_model="${model//[^A-Za-z0-9_]/_}"
  safe_dataset="${dataset//[^A-Za-z0-9_]/_}"
  run_dir="$OUT_ROOT/${run_index}_${stage}_${safe_model}_${safe_dataset}"
  log_file="$run_dir/test.log"
  pred_file="$run_dir/predictions.pkl"
  ckpt_abs="$(path_abs "$ckpt")"

  rtk /usr/bin/mkdir -p "$run_dir" || return 1

  master_port=$((MASTER_PORT_START + run_index))
  local cmd=(
    rtk env
    CUDA_VISIBLE_DEVICES="$GPUS"
    NCCL_P2P_DISABLE=1
    NCCL_IB_DISABLE=1
    PYTHONNOUSERSITE=1
    MPLCONFIGDIR=/tmp/mplconfig
    PYTHONPATH="$ROOT_DIR:$ROOT_DIR/tools"
    "$PYTHON_BIN" -m torch.distributed.launch
    "--nproc_per_node=$NPROC_PER_NODE"
    "--master_port=$master_port"
    tools/openrsd_test.py
    "$config"
    "$ckpt_abs"
    --launcher pytorch
    --work-dir "$run_dir"
    --out "$pred_file"
    --cfg-options
    "test_dataloader.batch_size=$BATCH_SIZE"
    "test_dataloader.num_workers=$NUM_WORKERS"
    "test_dataloader.dataset.data_root=$DATA_ROOT"
    "test_dataloader.dataset.ann_file=$ann_file"
    "test_dataloader.dataset.data_prefix.img_path=$img_path"
  )

  log "Evaluating $stage / $model on $dataset -> $run_dir"
  log "Command: $(format_command "${cmd[@]}")"

  if [[ "$DRY_RUN" == "1" ]]; then
    {
      printf 'DRY_RUN eval command:\n'
      format_command "${cmd[@]}"
      printf '\n'
    } > "$log_file"
    append_result "$stage" "$model" "$dataset" "$epoch" "$train_map" "DRY_RUN" "NA" "NA" "$ckpt_abs" "$log_file" "$pred_file"
    append_detail "$stage" "$model" "$dataset" "DRY_RUN" "NA" "NA" "" "$run_dir" "$log_file" "$pred_file"
    build_summary_md
    return 0
  fi

  if [[ ! -f "$ckpt_abs" ]]; then
    status="SKIP(checkpoint_missing)"
    append_result "$stage" "$model" "$dataset" "$epoch" "$train_map" "$status" "NA" "NA" "$ckpt_abs" "$log_file" "$pred_file"
    append_detail "$stage" "$model" "$dataset" "$status" "NA" "NA" "" "$run_dir" "$log_file" "$pred_file"
    build_summary_md
    log "Skipping $model on $dataset: missing checkpoint $ckpt_abs"
    FAILED=1
    return 1
  fi

  (
    cd "$ROOT_DIR" || exit 1
    "${cmd[@]}"
  ) > "$log_file" 2>&1
  exit_code=$?

  if [[ "$exit_code" -eq 0 ]]; then
    status="OK"
  else
    status="FAIL($exit_code)"
    FAILED=1
  fi

  parsed="$(parse_metric_line "$log_file")"
  IFS='|' read -r map ap50 metric_line <<< "$parsed"

  append_result "$stage" "$model" "$dataset" "$epoch" "$train_map" "$status" "$map" "$ap50" "$ckpt_abs" "$log_file" "$pred_file"
  append_detail "$stage" "$model" "$dataset" "$status" "$map" "$ap50" "$metric_line" "$run_dir" "$log_file" "$pred_file"
  build_summary_md
  log "Finished $stage / $model on $dataset: status=$status mAP=$map AP50=$ap50"
  [[ "$exit_code" -eq 0 ]]
}

eval_model_entries() {
  local stage="$1"
  shift
  local entry model config ckpt epoch train_map dataset_entry dataset ann_file img_path
  for entry in "$@"; do
    IFS='|' read -r model config ckpt epoch train_map <<< "$entry"
    for dataset_entry in "${DATASETS[@]}"; do
      IFS='|' read -r dataset ann_file img_path <<< "$dataset_entry"
      RUN_INDEX=$((RUN_INDEX + 1))
      run_eval "$stage" "$model" "$config" "$ckpt" "$epoch" "$train_map" \
        "$dataset" "$ann_file" "$img_path" "$RUN_INDEX" || true
    done
  done
}

skip_model_on_all_datasets() {
  local stage="$1"
  local model="$2"
  local config="$3"
  local ckpt="$4"
  local epoch="$5"
  local train_map="$6"
  local reason="$7"
  local dataset_entry dataset ann_file img_path
  for dataset_entry in "${DATASETS[@]}"; do
    IFS='|' read -r dataset ann_file img_path <<< "$dataset_entry"
    append_result "$stage" "$model" "$dataset" "$epoch" "$train_map" "$reason" "NA" "NA" "$ckpt" "" ""
  done
  build_summary_md
}

train_and_eval_best() {
  local model="$1"
  local config="$2"
  local work_dir="$3"
  local train_stage="$4"
  local eval_stage="$5"
  local best_info best_epoch best_map best_ckpt entry

  if run_train "$model" "$config" "$work_dir" "$train_stage"; then
    if [[ "$DRY_RUN" == "1" ]]; then
      best_epoch="DRY_RUN"
      best_map="NA"
      best_ckpt="$work_dir/epoch_DRYRUN.pth"
    else
      best_info="$(select_best_checkpoint "$work_dir/train.log" "$work_dir")"
      IFS='|' read -r best_epoch best_map best_ckpt <<< "$best_info"
    fi
    if [[ -n "$best_ckpt" && -f "$best_ckpt" ]]; then
      entry="$model|$config|$best_ckpt|$best_epoch|$best_map"
      SELECTED_MODELS+=("$entry")
      build_summary_md
      log "Selected best checkpoint for $model: epoch=$best_epoch mAP=$best_map checkpoint=$best_ckpt"
      eval_model_entries "$eval_stage" "$entry"
    elif [[ "$DRY_RUN" == "1" ]]; then
      entry="$model|$config|$best_ckpt|$best_epoch|$best_map"
      SELECTED_MODELS+=("$entry")
      build_summary_md
      log "DRY_RUN selected placeholder checkpoint for $model: $best_ckpt"
      eval_model_entries "$eval_stage" "$entry"
    else
      log "No usable best checkpoint for $model after training. Parsed: $best_info"
      skip_model_on_all_datasets "$eval_stage" "$model" "$config" "$best_ckpt" "$best_epoch" "$best_map" "SKIP(best_checkpoint_missing)"
      FAILED=1
    fi
  else
    skip_model_on_all_datasets "$eval_stage" "$model" "$config" "" "NA" "NA" "SKIP(train_failed)"
  fi
}

validate_static_inputs() {
  local dataset_entry dataset ann_file img_path
  for dataset_entry in "${DATASETS[@]}"; do
    IFS='|' read -r dataset ann_file img_path <<< "$dataset_entry"
    require_dir "$DATA_ROOT/${ann_file%/}"
    require_dir "$DATA_ROOT/${img_path%/}"
  done

  require_file "$ORCNN_CONFIG"
  require_file "$R3DET_CONFIG"
  require_file "$REDET_CONFIG"
  require_file "$REPPOINTS_CONFIG"
  require_file "$ORCNN_IMAGENET_CONFIG"
  require_file "$ORCNN_24E_CONFIG"
  require_file "$ORCNN_ROTATE_CONFIG"

  require_file "$ORCNN_EPOCH8_CKPT"
  require_file "$ORCNN_EPOCH12_CKPT"
  require_file "$R3DET_CKPT"
  require_file "$REDET_CKPT"
}

main() {
  cd "$ROOT_DIR" || exit 1
  rtk /usr/bin/mkdir -p "$OUT_ROOT" || exit 1
  printf 'stage|model|dataset|epoch|train_map|status|map|ap50|checkpoint|log|predictions\n' > "$RESULTS_TSV"
  : > "$DETAILS_MD"
  : > "$SCHEDULER_LOG"

  SELECTED_MODELS=("${BASELINE_MODELS[@]}")
  build_summary_md

  log "Starting May 02 0167 full series"
  log "Output root: $OUT_ROOT"
  log "GPUs: $GPUS"

  validate_static_inputs

  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN=1: would check only selected GPUs $GPUS for compute processes"
  elif ! check_selected_gpus_free; then
    die "Selected GPUs ($GPUS) are not free at run start"
  fi

  local reppoints_work_dir="$OUT_ROOT/train_reppoints_fullinit"
  local reppoints_best reppoints_epoch reppoints_map reppoints_ckpt reppoints_entry
  if [[ "$SKIP_REPPOINTS_TRAIN" == "1" ]]; then
    reppoints_ckpt="$REPPOINTS_EXISTING_CKPT"
    reppoints_epoch="$REPPOINTS_EXISTING_EPOCH"
    reppoints_map="$REPPOINTS_EXISTING_MAP"
    if [[ -n "$reppoints_ckpt" && -f "$(path_abs "$reppoints_ckpt")" ]]; then
      reppoints_ckpt="$(path_abs "$reppoints_ckpt")"
      reppoints_entry="RepPoints_R50_best|$REPPOINTS_CONFIG|$reppoints_ckpt|$reppoints_epoch|$reppoints_map"
      SELECTED_MODELS+=("$reppoints_entry")
      append_result "train_reppoints" "RepPoints_R50" "TRAIN" "$reppoints_epoch" "$reppoints_map" "SKIP(using_existing_checkpoint)" "NA" "NA" "$reppoints_ckpt" "" ""
      log "Skipping RepPoints training; using existing checkpoint epoch=$reppoints_epoch mAP=$reppoints_map checkpoint=$reppoints_ckpt"
    else
      reppoints_entry=""
      append_result "train_reppoints" "RepPoints_R50" "TRAIN" "$reppoints_epoch" "$reppoints_map" "SKIP(existing_checkpoint_missing)" "NA" "NA" "$reppoints_ckpt" "" ""
      log "Requested SKIP_REPPOINTS_TRAIN=1 but checkpoint is missing: $reppoints_ckpt"
      FAILED=1
    fi
  elif run_train "RepPoints_R50" "$REPPOINTS_CONFIG" "$reppoints_work_dir" "train_reppoints"; then
    if [[ "$DRY_RUN" == "1" ]]; then
      reppoints_epoch="DRY_RUN"
      reppoints_map="NA"
      reppoints_ckpt="$reppoints_work_dir/epoch_DRYRUN.pth"
      reppoints_best="$reppoints_epoch|$reppoints_map|$reppoints_ckpt"
    else
      reppoints_best="$(select_best_checkpoint "$reppoints_work_dir/train.log" "$reppoints_work_dir")"
      IFS='|' read -r reppoints_epoch reppoints_map reppoints_ckpt <<< "$reppoints_best"
    fi
    if [[ -n "$reppoints_ckpt" && -f "$reppoints_ckpt" ]]; then
      reppoints_entry="RepPoints_R50_best|$REPPOINTS_CONFIG|$reppoints_ckpt|$reppoints_epoch|$reppoints_map"
      SELECTED_MODELS+=("$reppoints_entry")
      log "Selected RepPoints best checkpoint: epoch=$reppoints_epoch mAP=$reppoints_map checkpoint=$reppoints_ckpt"
    elif [[ "$DRY_RUN" == "1" ]]; then
      reppoints_entry="RepPoints_R50_best|$REPPOINTS_CONFIG|$reppoints_ckpt|$reppoints_epoch|$reppoints_map"
      SELECTED_MODELS+=("$reppoints_entry")
      log "DRY_RUN selected placeholder RepPoints checkpoint: $reppoints_ckpt"
    else
      reppoints_entry=""
      log "RepPoints training finished but no usable best checkpoint was found. Parsed: $reppoints_best"
      FAILED=1
    fi
  else
    reppoints_entry=""
    log "RepPoints training failed; continuing with existing ORCNN/R3Det/ReDet evaluations"
  fi
  build_summary_md

  local four_model_entries=("${BASELINE_MODELS[@]}")
  if [[ -n "$reppoints_entry" ]]; then
    four_model_entries+=("$reppoints_entry")
  else
    skip_model_on_all_datasets "four_model_four_dataset" "RepPoints_R50_best" "$REPPOINTS_CONFIG" "${reppoints_ckpt:-}" "${reppoints_epoch:-NA}" "${reppoints_map:-NA}" "SKIP(reppoints_missing)"
  fi
  eval_model_entries "four_model_four_dataset" "${four_model_entries[@]}"

  local orcnn_epoch12_entry="ORCNN_R50_epoch12|$ORCNN_CONFIG|$ORCNN_EPOCH12_CKPT|12|0.4499"
  SELECTED_MODELS+=("$orcnn_epoch12_entry")
  build_summary_md
  eval_model_entries "orcnn_epoch12_comparison" "$orcnn_epoch12_entry"

  train_and_eval_best \
    "ORCNN_R50_ImageNetOnly_12e" \
    "$ORCNN_IMAGENET_CONFIG" \
    "$OUT_ROOT/train_orcnn_imagenetonly_12e" \
    "train_orcnn_imagenetonly_12e" \
    "eval_orcnn_imagenetonly_12e"

  train_and_eval_best \
    "ORCNN_R50_DetectorPretrain_24e" \
    "$ORCNN_24E_CONFIG" \
    "$OUT_ROOT/train_orcnn_detectorpretrain_24e" \
    "train_orcnn_detectorpretrain_24e" \
    "eval_orcnn_detectorpretrain_24e"

  train_and_eval_best \
    "ORCNN_R50_RandomRotate_rectfix" \
    "$ORCNN_ROTATE_CONFIG" \
    "$OUT_ROOT/train_orcnn_randomrotate_rectfix" \
    "train_orcnn_randomrotate_rectfix" \
    "eval_orcnn_randomrotate_rectfix"

  if [[ -f "$(path_abs "$LSKNET_CONFIG")" && -f "$(path_abs "$LSKNET_CKPT")" ]]; then
    local lsknet_entry="LSKNet_S_epoch4|$LSKNET_CONFIG|$LSKNET_CKPT|4|0.6011"
    SELECTED_MODELS+=("$lsknet_entry")
    build_summary_md
    eval_model_entries "lsknet_upper_bound" "$lsknet_entry"
  else
    log "Skipping LSKNet upper-bound evaluation because config or checkpoint is missing"
    skip_model_on_all_datasets "lsknet_upper_bound" "LSKNet_S_epoch4" "$LSKNET_CONFIG" "$LSKNET_CKPT" "4" "0.6011" "SKIP(checkpoint_or_config_missing)"
  fi

  build_summary_md
  log "Full series finished. Summary: $SUMMARY_MD"
  exit "$FAILED"
}

main "$@"
