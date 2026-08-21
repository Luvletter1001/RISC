#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/data1/zcy/OpenRSD}"
PYTHON_BIN="${PYTHON_BIN:-/data/zcy/anaconda3/envs/openrsd/bin/python}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/work_dirs/rotation_study_36h}"
GPUS="${GPUS:-0,1,4,5,6,7,8,9}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-35100}"
SMOKE_TEST_SECONDS="${SMOKE_TEST_SECONDS:-0}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"

log() {
  local msg
  msg="[$(rtk /usr/bin/date '+%F %T')] $*"
  printf '%s\n' "$msg"
}

safe_name() {
  local value="$1"
  value="${value//[^A-Za-z0-9_]/_}"
  printf '%s\n' "$value"
}

ensure_root() {
  cd "$ROOT_DIR" || exit 1
  rtk /usr/bin/mkdir -p "$OUT_ROOT"/{logs,predictions,metrics,angle_response,error_decomp,p1_granularity}
}

assert_file() {
  if [[ ! -f "$1" ]]; then
    log "ERROR missing file: $1"
    exit 2
  fi
}

assert_dir() {
  if [[ ! -d "$1" ]]; then
    log "ERROR missing directory: $1"
    exit 2
  fi
}

first_gpu() {
  local old_ifs="$IFS"
  local gpu
  IFS=','
  for gpu in $GPUS; do
    printf '%s\n' "$gpu"
    IFS="$old_ifs"
    return 0
  done
  IFS="$old_ifs"
}

gpu_for_task() {
  local task_idx="$1"
  local -a gpus=()
  local old_ifs="$IFS"
  local gpu idx
  IFS=','
  for gpu in $GPUS; do
    gpus+=("$gpu")
  done
  IFS="$old_ifs"
  if [[ "${#gpus[@]}" -eq 0 ]]; then
    printf '0\n'
    return 0
  fi
  idx=$((task_idx % ${#gpus[@]}))
  printf '%s\n' "${gpus[$idx]}"
}

parse_metrics_from_log() {
  local log_file="$1"
  rtk "$PYTHON_BIN" - "$log_file" <<'PY'
import re
import sys
path = sys.argv[1]
try:
    text = open(path, 'r', encoding='utf-8', errors='replace').read()
except FileNotFoundError:
    text = ''
def last(pattern):
    m = re.findall(pattern, text)
    return m[-1] if m else 'NA'
keys = [
    ('mAP', r'["\']?dota/mAP["\']?\s*[:=]\s*([0-9]*\.?[0-9]+)'),
    ('AP50', r'["\']?dota/AP50["\']?\s*[:=]\s*([0-9]*\.?[0-9]+)'),
    ('ap07_mAP', r'["\']?dota_ap07/mAP["\']?\s*[:=]\s*([0-9]*\.?[0-9]+)'),
    ('ap12_mAP', r'["\']?dota_ap12/mAP["\']?\s*[:=]\s*([0-9]*\.?[0-9]+)'),
]
print('|'.join(last(p) for _, p in keys))
PY
}

append_csv_header() {
  local csv="$1"
  if [[ ! -f "$csv" ]]; then
    printf 'stage,dataset,task_granularity,model,checkpoint,eval_split,rotation_type,angle,status,mAP,AP50,ap07_mAP,ap12_mAP,prediction_path,log_path\n' > "$csv"
  fi
}

run_eval_task() {
  local stage="$1"
  local dataset="$2"
  local granularity="$3"
  local model="$4"
  local config="$5"
  local checkpoint="$6"
  local data_root="$7"
  local ann_file="$8"
  local img_path="$9"
  local split="${10}"
  local rotation_type="${11}"
  local angle="${12}"
  local task_idx="${13:-0}"

  assert_file "$config"
  assert_file "$checkpoint"
  assert_dir "$data_root/$ann_file"
  assert_dir "$data_root/$img_path"

  local safe_stage safe_dataset safe_model safe_split run_dir log_file pred_file csv port cuda_devices nproc
  safe_stage="$(safe_name "$stage")"
  safe_dataset="$(safe_name "$dataset")"
  safe_model="$(safe_name "$model")"
  safe_split="$(safe_name "$split")"
  run_dir="$OUT_ROOT/${safe_stage}/${safe_dataset}/${safe_model}/${safe_split}"
  log_file="$run_dir/test.log"
  pred_file="$run_dir/predictions.pkl"
  csv="$OUT_ROOT/results_rotation_study.csv"
  port=$((MASTER_PORT_BASE + task_idx))
  nproc="$NPROC_PER_NODE"
  cuda_devices="$GPUS"
  if [[ "$nproc" == "1" ]]; then
    cuda_devices="$(gpu_for_task "$task_idx")"
  fi

  rtk /usr/bin/mkdir -p "$run_dir"
  append_csv_header "$csv"
  log "RUN $stage $dataset $model $split on CUDA_VISIBLE_DEVICES=$cuda_devices"

  local cmd=(
    rtk env
    CUDA_VISIBLE_DEVICES="$cuda_devices"
    NCCL_P2P_DISABLE=1
    NCCL_IB_DISABLE=1
    PYTHONNOUSERSITE=1
    MPLCONFIGDIR="$MPLCONFIGDIR"
    PYTHONPATH="$ROOT_DIR:$ROOT_DIR/tools"
    "$PYTHON_BIN"
  )
  if [[ "$nproc" == "1" ]]; then
    cmd+=(
      tools/openrsd_test.py
      "$config"
      "$checkpoint"
      --work-dir "$run_dir"
      --out "$pred_file"
      --cfg-options
      "test_dataloader.batch_size=$BATCH_SIZE"
      "test_dataloader.num_workers=$NUM_WORKERS"
      "test_dataloader.dataset.data_root=$data_root"
      "test_dataloader.dataset.ann_file=$ann_file"
      "test_dataloader.dataset.data_prefix.img_path=$img_path"
    )
  else
    cmd+=(
      -m torch.distributed.launch
      "--nproc_per_node=$nproc"
      "--master_port=$port"
      tools/openrsd_test.py
      "$config"
      "$checkpoint"
      --launcher pytorch
      --work-dir "$run_dir"
      --out "$pred_file"
      --cfg-options
      "test_dataloader.batch_size=$BATCH_SIZE"
      "test_dataloader.num_workers=$NUM_WORKERS"
      "test_dataloader.dataset.data_root=$data_root"
      "test_dataloader.dataset.ann_file=$ann_file"
      "test_dataloader.dataset.data_prefix.img_path=$img_path"
    )
  fi

  local status exit_code metrics map ap50 ap07 ap12
  if [[ "$SMOKE_TEST_SECONDS" -gt 0 ]]; then
    (
      cd "$ROOT_DIR" || exit 1
      rtk /usr/bin/timeout --preserve-status "$SMOKE_TEST_SECONDS" "${cmd[@]}"
    ) > "$log_file" 2>&1
    exit_code=$?
    if [[ "$exit_code" -eq 143 || "$exit_code" -eq 124 ]]; then
      status="SMOKE_OK"
    elif [[ "$exit_code" -eq 0 ]]; then
      status="OK"
    else
      status="FAIL($exit_code)"
    fi
  else
    (
      cd "$ROOT_DIR" || exit 1
      "${cmd[@]}"
    ) > "$log_file" 2>&1
    exit_code=$?
    if [[ "$exit_code" -eq 0 ]]; then
      status="OK"
    else
      status="FAIL($exit_code)"
    fi
  fi

  metrics="$(parse_metrics_from_log "$log_file")"
  IFS='|' read -r map ap50 ap07 ap12 <<< "$metrics"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$stage" "$dataset" "$granularity" "$model" "$checkpoint" "$split" \
    "$rotation_type" "$angle" "$status" "$map" "$ap50" "$ap07" "$ap12" \
    "$pred_file" "$log_file" >> "$csv"

  log "DONE $stage $dataset $model $split status=$status mAP=$map AP50=$ap50"
  if [[ "$status" == FAIL* ]]; then
    return 1
  fi
}
