#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'USAGE'
Usage:
  rtk bash M_Tools/train_queue/run_train_queue.sh [QUEUE_FILE]

QUEUE_FILE format:
  CONFIG|WORK_DIR|EXTRA_CFG_OPTIONS

Example:
  M_configs/example.py|work_dirs/example_run|train_dataloader.batch_size=1
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

QUEUE_FILE="${1:-M_Tools/train_queue/faa_dotav2_3runs.queue}"
ROOT_DIR="/data1/zcy/OpenRSD"
LOG_DIR="$ROOT_DIR/work_dirs/train_queue_logs"
GPUS="${GPUS:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT_START="${MASTER_PORT_START:-29801}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-1}"
PYTHON_BIN="${PYTHON_BIN:-/data/zcy/anaconda3/envs/openrsd/bin/python}"

if [[ ! -f "$QUEUE_FILE" ]]; then
  echo "Missing queue file: $QUEUE_FILE" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
timestamp="$(date +%Y%m%d_%H%M%S)"
queue_name="$(basename "$QUEUE_FILE" .queue)"
summary_log="$LOG_DIR/${queue_name}_${timestamp}.log"
status_file="$LOG_DIR/${queue_name}_${timestamp}.status"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$summary_log"
}

log "Queue started"
log "Queue file: $QUEUE_FILE"
log "GPUs: $GPUS"
log "Status file: $status_file"

run_index=0
failed=0

while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  line="${raw_line#"${raw_line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"
  if [[ -z "$line" || "${line:0:1}" == "#" ]]; then
    continue
  fi

  IFS='|' read -r config work_dir extra_cfg_options <<< "$line"
  if [[ -z "${config:-}" || -z "${work_dir:-}" ]]; then
    log "Malformed queue line: $raw_line"
    failed=1
    if [[ "$CONTINUE_ON_FAILURE" != "1" ]]; then
      break
    fi
    continue
  fi

  if [[ ! -f "$ROOT_DIR/$config" ]]; then
    log "Missing config: $config"
    failed=1
    if [[ "$CONTINUE_ON_FAILURE" != "1" ]]; then
      break
    fi
    continue
  fi

  run_index=$((run_index + 1))
  master_port=$((MASTER_PORT_START + run_index - 1))
  run_log="$LOG_DIR/${queue_name}_run${run_index}_${timestamp}.log"

  log "Run $run_index starting"
  log "  config: $config"
  log "  work_dir: $work_dir"
  log "  master_port: $master_port"
  log "  run_log: $run_log"
  if [[ -n "${extra_cfg_options:-}" ]]; then
    log "  cfg_options: $extra_cfg_options"
  fi

  cmd=(
    rtk env
    "CUDA_VISIBLE_DEVICES=$GPUS"
    NCCL_P2P_DISABLE=1
    NCCL_IB_DISABLE=1
    PYTHONNOUSERSITE=1
    MPLCONFIGDIR=/tmp/mplconfig
    PYTHONPATH="$ROOT_DIR:$ROOT_DIR/tools"
    "$PYTHON_BIN"
    -m torch.distributed.launch
    "--nproc_per_node=$NPROC_PER_NODE"
    "--master_port=$master_port"
    tools/train.py
    "$config"
    --launcher pytorch
    --work-dir "$work_dir"
  )

  if [[ -n "${extra_cfg_options:-}" ]]; then
    read -r -a extra_opts <<< "$extra_cfg_options"
    cmd+=(--cfg-options "${extra_opts[@]}")
  fi

  echo "RUN $run_index START $(date '+%F %T') $config $work_dir" >> "$status_file"
  (
    cd "$ROOT_DIR" || exit 1
    "${cmd[@]}"
  ) > "$run_log" 2>&1
  exit_code=$?

  if [[ "$exit_code" -eq 0 ]]; then
    log "Run $run_index finished OK"
    echo "RUN $run_index OK $(date '+%F %T') $config $work_dir" >> "$status_file"
  else
    log "Run $run_index failed with exit code $exit_code"
    echo "RUN $run_index FAIL exit=$exit_code $(date '+%F %T') $config $work_dir" >> "$status_file"
    failed=1
    if [[ "$CONTINUE_ON_FAILURE" != "1" ]]; then
      break
    fi
  fi
done < "$QUEUE_FILE"

if [[ "$failed" -eq 0 ]]; then
  log "Queue finished OK"
else
  log "Queue finished with failures"
fi

exit "$failed"
