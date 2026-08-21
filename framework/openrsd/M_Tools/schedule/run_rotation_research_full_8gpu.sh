#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/data1/zcy/OpenRSD}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data/DOTA2_1024_500}"
RUN_TS="${RUN_TS:-$(rtk /usr/bin/date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/work_dirs/scheduled_runs/rotation_research_full_8gpu_$RUN_TS}"
GPUS="${GPUS:-0,1,4,5,6,7,8,9}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_CANDIDATES="${BATCH_CANDIDATES:-16 8 4 2 1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MASTER_PORT_START="${MASTER_PORT_START:-32101}"
PYTHON_BIN="${PYTHON_BIN:-/data/zcy/anaconda3/envs/openrsd/bin/python}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-1}"
PREPARE_ANGLES="${PREPARE_ANGLES:-1}"
RUN_ANGLE_SWEEP="${RUN_ANGLE_SWEEP:-1}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"
RUN_TTA="${RUN_TTA:-1}"
WAIT_FOR_FREE_GPUS="${WAIT_FOR_FREE_GPUS:-1}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-300}"
GPU_MAX_WAIT_SECONDS="${GPU_MAX_WAIT_SECONDS:-0}"
CHECK_GPUS_EACH_TASK="${CHECK_GPUS_EACH_TASK:-1}"
FORCE_RERUN_STAGES="${FORCE_RERUN_STAGES:-}"
ANGLE_PREP_NPROC="${ANGLE_PREP_NPROC:-16}"
ANGLE_LINK_MODE="${ANGLE_LINK_MODE:-hardlink}"
DIAGNOSTIC_SCORE_THR="${DIAGNOSTIC_SCORE_THR:-0.05}"
DIAGNOSTIC_MAX_DETS_PER_IMG="${DIAGNOSTIC_MAX_DETS_PER_IMG:-300}"
TTA_SCORE_THR="${TTA_SCORE_THR:-0.05}"
TTA_PRE_NMS_TOPK="${TTA_PRE_NMS_TOPK:-4000}"
TTA_NMS_IOU="${TTA_NMS_IOU:-0.1}"

ANGLE_SWEEP_ROOT="$DATA_ROOT/angle_sweep_val"
RESULTS_TSV="$OUT_ROOT/results.tsv"
SUMMARY_MD="$OUT_ROOT/rotation_research_full_results.md"
RUN_LOG="$OUT_ROOT/run.log"
DIAGNOSTICS_TSV="$OUT_ROOT/diagnostics/rotation_detection_diagnostics.tsv"

ORCNN_CONFIG="M_configs/DOTA2OfficialAdapters/oriented_rcnn_r50_fpn_dotav2_fullinit.py"
R3DET_CONFIG="M_configs/DOTA2OfficialAdapters/r3det_kfiou_r50_fpn_dotav2_fullinit.py"
REDET_CONFIG="M_configs/DOTA2OfficialAdapters/redet_re50_refpn_dotav2_fullinit.py"

ORCNN_CKPT="work_dirs/dotav2_official_adapters/oriented_rcnn_r50_fpn_fullinit/epoch_8.pth"
R3DET_CKPT="work_dirs/dotav2_official_adapters/r3det_kfiou_r50_fpn_fullinit/epoch_12.pth"
REDET_CKPT="work_dirs/dotav2_official_adapters/redet_re50_refpn_fullinit/epoch_4.pth"

ANGLES=(0 15 30 45 60 75 90 105 120 135 150 165 180 195 210 225 240 255 270 285 300 315 330 345)
MODELS=(
  "ReDet_Re50_epoch4|$REDET_CONFIG|$REDET_CKPT|4"
  "ORCNN_R50_epoch8|$ORCNN_CONFIG|$ORCNN_CKPT|8"
  "R3Det_KFIoU_R50_epoch12|$R3DET_CONFIG|$R3DET_CKPT|12"
)
DIAGNOSTIC_MODELS="${DIAGNOSTIC_MODELS:-ReDet_Re50_epoch4 ORCNN_R50_epoch8 R3Det_KFIoU_R50_epoch12}"
TTA_SETS=(
  "tta_4angle|0,90,180,270"
  "tta_8angle|0,45,90,135,180,225,270,315"
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

safe_name() {
  local value="$1"
  value="${value//[^A-Za-z0-9_]/_}"
  printf '%s\n' "$value"
}

angle_name() {
  printf '%03d\n' "$1"
}

sanitize_cell() {
  local value="$1"
  value="${value//$'\n'/ }"
  value="${value//|/;}"
  printf '%s\n' "$value"
}

format_command() {
  local arg q out=""
  for arg in "$@"; do
    printf -v q '%q' "$arg"
    out+="$q "
  done
  printf '%s' "${out% }"
}

require_file() {
  local abs
  abs="$(path_abs "$1")"
  if [[ ! -f "$abs" ]]; then
    log "ERROR missing file: $abs"
    return 1
  fi
}

require_dir() {
  local abs
  abs="$(path_abs "$1")"
  if [[ ! -d "$abs" ]]; then
    log "ERROR missing directory: $abs"
    return 1
  fi
}

ensure_results_header() {
  if [[ "$RESUME" == "1" && -f "$RESULTS_TSV" ]]; then
    return 0
  fi
  printf 'stage|model|dataset|angle|epoch|batch_size|status|map|ap50|checkpoint|log|predictions|diagnosis\n' > "$RESULTS_TSV"
}

append_result() {
  local stage="$1"
  local model="$2"
  local dataset="$3"
  local angle="$4"
  local epoch="$5"
  local batch="$6"
  local status="$7"
  local map="$8"
  local ap50="$9"
  local checkpoint="${10}"
  local log_file="${11}"
  local predictions="${12}"
  local diagnosis="${13}"

  diagnosis="$(sanitize_cell "$diagnosis")"
  status="$(sanitize_cell "$status")"
  printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
    "$stage" "$model" "$dataset" "$angle" "$epoch" "$batch" "$status" \
    "$map" "$ap50" "$checkpoint" "$log_file" "$predictions" "$diagnosis" >> "$RESULTS_TSV"
}

refresh_summary() {
  rtk env \
    PYTHONPATH="$ROOT_DIR:$ROOT_DIR/tools" \
    "$PYTHON_BIN" \
    M_Tools/analysis/summarize_rotation_research_results.py \
    --results-tsv "$RESULTS_TSV" \
    --out-md "$SUMMARY_MD" \
    --out-root "$OUT_ROOT" \
    --title "Full Rotation Robustness Results" \
    --gpus "$GPUS" \
    --data-root "$DATA_ROOT" \
    --diagnostics-tsv "$DIAGNOSTICS_TSV" \
    --run-log "$RUN_LOG" >/dev/null 2>&1 || true
}

task_has_ok() {
  local stage="$1"
  local model="$2"
  local dataset="$3"
  local angle="$4"
  local row_stage row_model row_dataset row_angle row_epoch row_batch row_status rest
  local old_ifs token
  old_ifs="$IFS"
  IFS=', '
  for token in $FORCE_RERUN_STAGES; do
    if [[ "$token" == "$stage" || "$token" == "all" ]]; then
      IFS="$old_ifs"
      return 1
    fi
  done
  IFS="$old_ifs"
  [[ -f "$RESULTS_TSV" ]] || return 1
  while IFS='|' read -r row_stage row_model row_dataset row_angle row_epoch row_batch row_status rest; do
    [[ "$row_stage" != "stage" ]] || continue
    if [[ "$row_stage" == "$stage" && "$row_model" == "$model" && "$row_dataset" == "$dataset" && "$row_angle" == "$angle" && "$row_status" == "OK" ]]; then
      return 0
    fi
  done < "$RESULTS_TSV"
  return 1
}

latest_prediction_for() {
  local stage="$1"
  local model="$2"
  local dataset="$3"
  local angle="$4"
  local row_stage row_model row_dataset row_angle row_epoch row_batch row_status row_map row_ap50 row_ckpt row_log row_pred row_diag
  local found=""
  [[ -f "$RESULTS_TSV" ]] || return 1
  while IFS='|' read -r row_stage row_model row_dataset row_angle row_epoch row_batch row_status row_map row_ap50 row_ckpt row_log row_pred row_diag; do
    [[ "$row_stage" != "stage" ]] || continue
    if [[ "$row_stage" == "$stage" && "$row_model" == "$model" && "$row_dataset" == "$dataset" && "$row_angle" == "$angle" && "$row_status" == "OK" ]]; then
      found="$row_pred"
    fi
  done < "$RESULTS_TSV"
  [[ -n "$found" ]] || return 1
  printf '%s\n' "$found"
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

check_selected_gpus_free_once() {
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
      log "GPU $idx busy: pid=$pid mem=${mem}MiB proc=$proc"
      busy=1
    fi
  done < <(rtk nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader,nounits)

  [[ "$busy" -eq 0 ]]
}

wait_for_selected_gpus() {
  if [[ "$WAIT_FOR_FREE_GPUS" != "1" || "$DRY_RUN" == "1" ]]; then
    return 0
  fi

  local waited=0
  while ! check_selected_gpus_free_once; do
    if [[ "$GPU_MAX_WAIT_SECONDS" -gt 0 && "$waited" -ge "$GPU_MAX_WAIT_SECONDS" ]]; then
      log "ERROR GPUs did not become free within ${GPU_MAX_WAIT_SECONDS}s"
      return 1
    fi
    log "Waiting ${GPU_WAIT_SECONDS}s for selected GPUs to become free: $GPUS"
    rtk /usr/bin/sleep "$GPU_WAIT_SECONDS"
    waited=$((waited + GPU_WAIT_SECONDS))
  done
  return 0
}

parse_metrics() {
  local log_file="$1"
  rtk "$PYTHON_BIN" - "$log_file" <<'PY'
import re
import sys

path = sys.argv[1]
text = ''
try:
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
except FileNotFoundError:
    pass

def last(pattern):
    matches = re.findall(pattern, text)
    return matches[-1] if matches else 'NA'

map_value = last(r'["\']?dota/mAP["\']?\s*[:=]\s*([0-9]*\.?[0-9]+)')
ap50_value = last(r'["\']?dota/AP50["\']?\s*[:=]\s*([0-9]*\.?[0-9]+)')
print(f'{map_value}|{ap50_value}')
PY
}

diagnose_log() {
  local log_file="$1"
  rtk "$PYTHON_BIN" - "$log_file" <<'PY'
import re
import sys

path = sys.argv[1]
try:
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
except FileNotFoundError:
    print('log_missing')
    raise SystemExit

patterns = [
    ('cuda_oom', r'CUDA out of memory|OutOfMemoryError'),
    ('cuda_illegal_memory', r'illegal memory access'),
    ('nccl', r'\bNCCL\b|ProcessGroupNCCL|nccl'),
    ('missing_file', r'No such file or directory|FileNotFoundError|cannot find'),
    ('dataset_empty', r'No samples|empty dataset|len\(dataset\).*0'),
    ('killed', r'\bKilled\b|exit code 137|SIGKILL'),
    ('traceback', r'Traceback \(most recent call last\)'),
    ('assertion', r'AssertionError'),
]
hits = [name for name, pattern in patterns if re.search(pattern, text, re.I)]
print(','.join(hits) if hits else 'unknown_failure')
PY
}

validate_static_inputs() {
  require_dir "$DATA_ROOT/ss_val/images" || return 1
  require_dir "$DATA_ROOT/ss_val/annfiles" || return 1
  require_file "$ORCNN_CONFIG" || return 1
  require_file "$R3DET_CONFIG" || return 1
  require_file "$REDET_CONFIG" || return 1
  require_file "$ORCNN_CKPT" || return 1
  require_file "$R3DET_CKPT" || return 1
  require_file "$REDET_CKPT" || return 1
  require_file "tools/openrsd_test.py" || return 1
  require_file "tools/openrsd_eval_metric.py" || return 1
  require_file "M_Tools/Data1_DOTA2/prepare_dotav2_angle_sweep.py" || return 1
  require_file "M_Tools/analysis/rotation_detection_diagnostics.py" || return 1
  require_file "M_Tools/analysis/rotation_tta_merge.py" || return 1
  require_file "M_Tools/analysis/summarize_rotation_research_results.py" || return 1
}

prepare_angle_splits() {
  if [[ "$PREPARE_ANGLES" != "1" ]]; then
    log "Skipping angle split preparation because PREPARE_ANGLES=$PREPARE_ANGLES"
    return 0
  fi
  if task_has_ok "prepare" "angle_sweep_val" "all" "all"; then
    log "Angle split preparation already completed in $RESULTS_TSV"
    return 0
  fi

  local log_file="$OUT_ROOT/prepare_angle_sweep.log"
  rtk /usr/bin/mkdir -p "$OUT_ROOT" || return 1
  log "Preparing angle-sweep validation data under $ANGLE_SWEEP_ROOT"

  local cmd=(
    rtk env
    PYTHONPATH="$ROOT_DIR:$ROOT_DIR/M_Tools/Data1_DOTA2:$ROOT_DIR/tools"
    "$PYTHON_BIN"
    M_Tools/Data1_DOTA2/prepare_dotav2_angle_sweep.py
    --data-root "$DATA_ROOT"
    --out-root "$ANGLE_SWEEP_ROOT"
    --nproc "$ANGLE_PREP_NPROC"
    --link-mode "$ANGLE_LINK_MODE"
  )
  log "Command: $(format_command "${cmd[@]}")"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN prepare command: %s\n' "$(format_command "${cmd[@]}")" > "$log_file"
    append_result "prepare" "angle_sweep_val" "all" "all" "NA" "NA" "DRY_RUN" "NA" "NA" "" "$log_file" "" "dry_run"
    refresh_summary
    return 0
  fi

  (
    cd "$ROOT_DIR" || exit 1
    "${cmd[@]}"
  ) > "$log_file" 2>&1
  local exit_code=$?
  local diagnosis
  diagnosis="$(diagnose_log "$log_file")"
  if [[ "$exit_code" -eq 0 ]]; then
    append_result "prepare" "angle_sweep_val" "all" "all" "NA" "NA" "OK" "NA" "NA" "" "$log_file" "$ANGLE_SWEEP_ROOT/realistic/manifest_angle_sweep.json" "$diagnosis"
    refresh_summary
    return 0
  fi
  append_result "prepare" "angle_sweep_val" "all" "all" "NA" "NA" "FAIL($exit_code)" "NA" "NA" "" "$log_file" "" "$diagnosis"
  refresh_summary
  return "$exit_code"
}

run_eval_task() {
  local stage="$1"
  local model="$2"
  local config="$3"
  local ckpt="$4"
  local epoch="$5"
  local dataset="$6"
  local angle="$7"
  local ann_file="$8"
  local img_path="$9"

  if task_has_ok "$stage" "$model" "$dataset" "$angle"; then
    log "Skip completed task: $stage $model $dataset angle=$angle"
    return 0
  fi
  if [[ "$CHECK_GPUS_EACH_TASK" == "1" ]]; then
    wait_for_selected_gpus || return 1
  fi

  local ckpt_abs safe_model safe_dataset batch attempt run_dir log_file pred_file master_port
  local exit_code status parsed map ap50 diagnosis extra_cuda_launch_blocking
  ckpt_abs="$(path_abs "$ckpt")"
  safe_model="$(safe_name "$model")"
  safe_dataset="$(safe_name "$dataset")"
  attempt=0
  extra_cuda_launch_blocking=0

  if [[ ! -f "$ckpt_abs" ]]; then
    append_result "$stage" "$model" "$dataset" "$angle" "$epoch" "NA" "SKIP(checkpoint_missing)" "NA" "NA" "$ckpt_abs" "" "" "checkpoint_missing"
    refresh_summary
    return 1
  fi

  for batch in $BATCH_CANDIDATES; do
    attempt=$((attempt + 1))
    run_dir="$OUT_ROOT/tasks/${stage}/${safe_model}/${safe_dataset}/angle_${angle}/b${batch}_try${attempt}"
    log_file="$run_dir/test.log"
    pred_file="$run_dir/predictions.pkl"
    master_port=$((MASTER_PORT_START + attempt + angle + ${#model} + ${#stage}))
    rtk /usr/bin/mkdir -p "$run_dir" || return 1

    local cmd=(
      rtk env
      CUDA_VISIBLE_DEVICES="$GPUS"
      NCCL_P2P_DISABLE=1
      NCCL_IB_DISABLE=1
      NCCL_ASYNC_ERROR_HANDLING=1
      TORCH_NCCL_ASYNC_ERROR_HANDLING=1
      PYTHONNOUSERSITE=1
      MPLCONFIGDIR=/tmp/mplconfig
      PYTHONPATH="$ROOT_DIR:$ROOT_DIR/tools"
    )
    if [[ "$extra_cuda_launch_blocking" == "1" ]]; then
      cmd+=(CUDA_LAUNCH_BLOCKING=1)
    fi
    cmd+=(
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
      "test_dataloader.batch_size=$batch"
      "test_dataloader.num_workers=$NUM_WORKERS"
      "test_dataloader.dataset.data_root=$DATA_ROOT"
      "test_dataloader.dataset.ann_file=$ann_file"
      "test_dataloader.dataset.data_prefix.img_path=$img_path"
    )

    log "Running $stage / $model / $dataset / angle=$angle / batch=$batch"
    log "Command: $(format_command "${cmd[@]}")"

    if [[ "$DRY_RUN" == "1" ]]; then
      printf 'DRY_RUN eval command: %s\n' "$(format_command "${cmd[@]}")" > "$log_file"
      append_result "$stage" "$model" "$dataset" "$angle" "$epoch" "$batch" "DRY_RUN" "NA" "NA" "$ckpt_abs" "$log_file" "$pred_file" "dry_run"
      refresh_summary
      return 0
    fi

    (
      cd "$ROOT_DIR" || exit 1
      "${cmd[@]}"
    ) > "$log_file" 2>&1
    exit_code=$?
    parsed="$(parse_metrics "$log_file")"
    IFS='|' read -r map ap50 <<< "$parsed"
    diagnosis="$(diagnose_log "$log_file")"

    if [[ "$exit_code" -eq 0 ]]; then
      status="OK"
      append_result "$stage" "$model" "$dataset" "$angle" "$epoch" "$batch" "$status" "$map" "$ap50" "$ckpt_abs" "$log_file" "$pred_file" "$diagnosis"
      refresh_summary
      log "Completed $stage / $model / $dataset / angle=$angle mAP=$map AP50=$ap50"
      return 0
    fi

    status="FAIL($exit_code)"
    append_result "$stage" "$model" "$dataset" "$angle" "$epoch" "$batch" "$status" "$map" "$ap50" "$ckpt_abs" "$log_file" "$pred_file" "$diagnosis"
    refresh_summary
    log "Failed $stage / $model / $dataset / angle=$angle batch=$batch diagnosis=$diagnosis"

    if [[ "$diagnosis" == *"cuda_illegal_memory"* ]]; then
      extra_cuda_launch_blocking=1
      log "Next retry will use CUDA_LAUNCH_BLOCKING=1 for clearer diagnostics."
    fi
    if [[ "$diagnosis" == *"missing_file"* && "$stage" == "angle_sweep" ]]; then
      log "Missing file detected; rerunning angle split preparation before retry."
      prepare_angle_splits || true
    fi
  done
  log "All batch fallbacks failed for $stage / $model / $dataset / angle=$angle"
  return 1
}

run_angle_sweep() {
  if [[ "$RUN_ANGLE_SWEEP" != "1" ]]; then
    log "Skipping angle sweep because RUN_ANGLE_SWEEP=$RUN_ANGLE_SWEEP"
    return 0
  fi

  local entry model config ckpt epoch angle aname dataset ann_file img_path
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r model config ckpt epoch <<< "$entry"
    for angle in "${ANGLES[@]}"; do
      aname="$(angle_name "$angle")"
      dataset="angle_${aname}"
      ann_file="angle_sweep_val/realistic/angle_${aname}/annfiles/"
      img_path="angle_sweep_val/realistic/angle_${aname}/images/"
      run_eval_task "angle_sweep" "$model" "$config" "$ckpt" "$epoch" "$dataset" "$angle" "$ann_file" "$img_path" || true
    done
  done
}

model_entry_by_name() {
  local needle="$1"
  local entry model config ckpt epoch
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r model config ckpt epoch <<< "$entry"
    if [[ "$model" == "$needle" ]]; then
      printf '%s\n' "$entry"
      return 0
    fi
  done
  return 1
}

run_diagnostics() {
  if [[ "$RUN_DIAGNOSTICS" != "1" ]]; then
    log "Skipping diagnostics because RUN_DIAGNOSTICS=$RUN_DIAGNOSTICS"
    return 0
  fi

  local model entry config ckpt epoch angle aname dataset pred ann_dir safe_model log_file status diagnosis append_flag
  rtk /usr/bin/mkdir -p "$OUT_ROOT/diagnostics" || return 1
  for model in $DIAGNOSTIC_MODELS; do
    entry="$(model_entry_by_name "$model" || true)"
    [[ -n "$entry" ]] || continue
    IFS='|' read -r model config ckpt epoch <<< "$entry"
    safe_model="$(safe_name "$model")"
    for angle in "${ANGLES[@]}"; do
      if task_has_ok "diagnostic" "$model" "angle_sweep" "$angle"; then
        log "Skip completed diagnostic: $model angle=$angle"
        continue
      fi
      aname="$(angle_name "$angle")"
      dataset="angle_${aname}"
      pred="$(latest_prediction_for "angle_sweep" "$model" "$dataset" "$angle" || true)"
      log_file="$OUT_ROOT/diagnostics/${safe_model}_angle_${aname}.log"
      ann_dir="$DATA_ROOT/angle_sweep_val/realistic/angle_${aname}/annfiles"
      if [[ -z "$pred" || ! -f "$pred" ]]; then
        append_result "diagnostic" "$model" "angle_sweep" "$angle" "$epoch" "NA" "SKIP(prediction_missing)" "NA" "NA" "$(path_abs "$ckpt")" "$log_file" "" "prediction_missing"
        refresh_summary
        continue
      fi
      append_flag=()
      if [[ -f "$DIAGNOSTICS_TSV" && -s "$DIAGNOSTICS_TSV" ]]; then
        append_flag=(--append)
      fi
      local cmd=(
        rtk env
        PYTHONPATH="$ROOT_DIR:$ROOT_DIR/tools"
        "$PYTHON_BIN"
        M_Tools/analysis/rotation_detection_diagnostics.py
        --predictions "$pred"
        --ann-dir "$ann_dir"
        --angle "$angle"
        --model "$model"
        --out-tsv "$DIAGNOSTICS_TSV"
        --score-thr "$DIAGNOSTIC_SCORE_THR"
        --max-dets-per-img "$DIAGNOSTIC_MAX_DETS_PER_IMG"
        "${append_flag[@]}"
      )
      log "Running diagnostic $model angle=$angle"
      if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY_RUN diagnostic command: %s\n' "$(format_command "${cmd[@]}")" > "$log_file"
        append_result "diagnostic" "$model" "angle_sweep" "$angle" "$epoch" "NA" "DRY_RUN" "NA" "NA" "$(path_abs "$ckpt")" "$log_file" "$DIAGNOSTICS_TSV" "dry_run"
        refresh_summary
        continue
      fi
      (
        cd "$ROOT_DIR" || exit 1
        "${cmd[@]}"
      ) > "$log_file" 2>&1
      local exit_code=$?
      diagnosis="$(diagnose_log "$log_file")"
      if [[ "$exit_code" -eq 0 ]]; then
        status="OK"
      else
        status="FAIL($exit_code)"
      fi
      append_result "diagnostic" "$model" "angle_sweep" "$angle" "$epoch" "NA" "$status" "NA" "NA" "$(path_abs "$ckpt")" "$log_file" "$DIAGNOSTICS_TSV" "$diagnosis"
      refresh_summary
    done
  done
}

run_tta_one() {
  local model="$1"
  local config="$2"
  local ckpt="$3"
  local epoch="$4"
  local tta_name="$5"
  local angles_csv="$6"
  local dataset="ss_val_${tta_name}"
  local safe_model safe_tta merge_dir merge_log merged_pkl eval_log status diagnosis
  local pred_args=()
  local angle_args=()
  local old_ifs angle aname pred parsed map ap50 exit_code

  if task_has_ok "tta_eval" "$model" "$dataset" "merged"; then
    log "Skip completed TTA eval: $model $tta_name"
    return 0
  fi

  old_ifs="$IFS"
  IFS=','
  for angle in $angles_csv; do
    aname="$(angle_name "$angle")"
    pred="$(latest_prediction_for "angle_sweep" "$model" "angle_${aname}" "$angle" || true)"
    if [[ -z "$pred" || ! -f "$pred" ]]; then
      IFS="$old_ifs"
      append_result "tta_eval" "$model" "$dataset" "merged" "$epoch" "NA" "SKIP(prediction_missing_angle_${aname})" "NA" "NA" "$(path_abs "$ckpt")" "" "" "prediction_missing"
      refresh_summary
      return 1
    fi
    pred_args+=(--prediction "$pred")
    angle_args+=(--angle "$angle")
  done
  IFS="$old_ifs"

  safe_model="$(safe_name "$model")"
  safe_tta="$(safe_name "$tta_name")"
  merge_dir="$OUT_ROOT/tasks/tta/${safe_model}/${safe_tta}"
  merge_log="$merge_dir/merge.log"
  eval_log="$merge_dir/eval_metric.log"
  merged_pkl="$merge_dir/merged_predictions.pkl"
  rtk /usr/bin/mkdir -p "$merge_dir" || return 1

  local merge_cmd=(
    rtk env
    PYTHONPATH="$ROOT_DIR:$ROOT_DIR/tools"
    "$PYTHON_BIN"
    M_Tools/analysis/rotation_tta_merge.py
    "${pred_args[@]}"
    "${angle_args[@]}"
    --out "$merged_pkl"
    --score-thr "$TTA_SCORE_THR"
    --pre-nms-topk "$TTA_PRE_NMS_TOPK"
    --nms-iou "$TTA_NMS_IOU"
  )
  log "Merging TTA predictions for $model $tta_name"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN TTA merge command: %s\n' "$(format_command "${merge_cmd[@]}")" > "$merge_log"
    append_result "tta_eval" "$model" "$dataset" "merged" "$epoch" "NA" "DRY_RUN" "NA" "NA" "$(path_abs "$ckpt")" "$merge_log" "$merged_pkl" "dry_run"
    refresh_summary
    return 0
  fi

  (
    cd "$ROOT_DIR" || exit 1
    "${merge_cmd[@]}"
  ) > "$merge_log" 2>&1
  exit_code=$?
  if [[ "$exit_code" -ne 0 ]]; then
    diagnosis="$(diagnose_log "$merge_log")"
    append_result "tta_eval" "$model" "$dataset" "merged" "$epoch" "NA" "FAIL_MERGE($exit_code)" "NA" "NA" "$(path_abs "$ckpt")" "$merge_log" "$merged_pkl" "$diagnosis"
    refresh_summary
    return 1
  fi

  local eval_cmd=(
    rtk env
    PYTHONNOUSERSITE=1
    MPLCONFIGDIR=/tmp/mplconfig
    PYTHONPATH="$ROOT_DIR:$ROOT_DIR/tools"
    "$PYTHON_BIN"
    tools/openrsd_eval_metric.py
    "$config"
    "$merged_pkl"
    --cfg-options
    "test_dataloader.dataset.data_root=$DATA_ROOT"
    "test_dataloader.dataset.ann_file=ss_val/annfiles/"
    "test_dataloader.dataset.data_prefix.img_path=ss_val/images/"
  )
  log "Evaluating merged TTA predictions for $model $tta_name"
  (
    cd "$ROOT_DIR" || exit 1
    "${eval_cmd[@]}"
  ) > "$eval_log" 2>&1
  exit_code=$?
  parsed="$(parse_metrics "$eval_log")"
  IFS='|' read -r map ap50 <<< "$parsed"
  diagnosis="$(diagnose_log "$eval_log")"
  if [[ "$exit_code" -eq 0 ]]; then
    status="OK"
  else
    status="FAIL_EVAL($exit_code)"
  fi
  append_result "tta_eval" "$model" "$dataset" "merged" "$epoch" "NA" "$status" "$map" "$ap50" "$(path_abs "$ckpt")" "$eval_log" "$merged_pkl" "$diagnosis"
  refresh_summary
  [[ "$exit_code" -eq 0 ]]
}

run_tta() {
  if [[ "$RUN_TTA" != "1" ]]; then
    log "Skipping TTA because RUN_TTA=$RUN_TTA"
    return 0
  fi
  local entry model config ckpt epoch tta_entry tta_name angles_csv
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r model config ckpt epoch <<< "$entry"
    for tta_entry in "${TTA_SETS[@]}"; do
      IFS='|' read -r tta_name angles_csv <<< "$tta_entry"
      run_tta_one "$model" "$config" "$ckpt" "$epoch" "$tta_name" "$angles_csv" || true
    done
  done
}

main() {
  cd "$ROOT_DIR" || exit 1
  rtk /usr/bin/mkdir -p "$OUT_ROOT" || exit 1
  if [[ "$RESUME" != "1" || ! -f "$RUN_LOG" ]]; then
    : > "$RUN_LOG"
  fi
  ensure_results_header
  refresh_summary

  log "Starting full rotation robustness experiment"
  log "OUT_ROOT=$OUT_ROOT"
  log "SUMMARY_MD=$SUMMARY_MD"
  log "GPUS=$GPUS NPROC_PER_NODE=$NPROC_PER_NODE BATCH_CANDIDATES=$BATCH_CANDIDATES"
  log "RUN_ANGLE_SWEEP=$RUN_ANGLE_SWEEP RUN_DIAGNOSTICS=$RUN_DIAGNOSTICS RUN_TTA=$RUN_TTA DRY_RUN=$DRY_RUN RESUME=$RESUME"

  validate_static_inputs || {
    refresh_summary
    exit 2
  }

  wait_for_selected_gpus || {
    refresh_summary
    exit 3
  }

  prepare_angle_splits || {
    log "Angle split preparation failed. Existing completed tasks remain resumable in $RESULTS_TSV"
    refresh_summary
    exit 4
  }
  run_angle_sweep
  run_diagnostics
  run_tta
  refresh_summary
  log "Finished full rotation robustness experiment. Summary: $SUMMARY_MD"
}

main "$@"
