#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  rtk bash M_Tools/train_queue/run_next_after.sh [WAIT_PATTERN] [ENV_FILE]

WAIT_PATTERN:
  A string matched against process command lines. The script waits while any
  matching process exists. If omitted, it starts the next job immediately.

ENV_FILE:
  Queue config file. Default:
  M_Tools/train_queue/next_train.env
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

WAIT_PATTERN="${1:-}"
ENV_FILE="${2:-M_Tools/train_queue/next_train.env}"

if [[ -n "$WAIT_PATTERN" ]]; then
  echo "Waiting for current job matching: $WAIT_PATTERN"
  while true; do
    matches="$(pgrep -af "$WAIT_PATTERN" || true)"
    matches="$(printf '%s\n' "$matches" | grep -v "run_next_after.sh" || true)"
    if [[ -z "$matches" ]]; then
      break
    fi
    sleep 60
  done
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE" >&2
  exit 1
fi

while true; do
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a

  if [[ -n "${NEXT_CONFIG:-}" ]]; then
    break
  fi
  echo "NEXT_CONFIG is empty in $ENV_FILE; waiting for you to set it..."
  sleep 60
done

if [[ ! -f "$NEXT_CONFIG" ]]; then
  echo "NEXT_CONFIG does not exist: $NEXT_CONFIG" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
config_name="$(basename "$NEXT_CONFIG" .py)"
work_dir="${NEXT_WORK_DIR:-work_dirs/${config_name}_${timestamp}}"
log_dir="work_dirs/train_queue_logs"
mkdir -p "$log_dir"
queue_log="$log_dir/${config_name}_${timestamp}.log"

echo "Starting next training:"
echo "  config:   $NEXT_CONFIG"
echo "  work_dir: $work_dir"
echo "  log:      $queue_log"

cmd=(
  rtk env
  "CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3}"
  "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}"
  "NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}"
  PYTHONNOUSERSITE=1
  MPLCONFIGDIR=/tmp/mplconfig
  PYTHONPATH=/data1/zcy/OpenRSD:/data1/zcy/OpenRSD/tools
  /data/zcy/anaconda3/envs/openrsd/bin/python
  -m torch.distributed.launch
  "--nproc_per_node=${NPROC_PER_NODE:-4}"
  "--master_port=${MASTER_PORT:-29731}"
  tools/train.py
  "$NEXT_CONFIG"
  --launcher pytorch
  --work-dir "$work_dir"
)

if [[ -n "${EXTRA_CFG_OPTIONS:-}" ]]; then
  # Intentionally split EXTRA_CFG_OPTIONS like a shell command line.
  # Use simple key=value tokens here.
  read -r -a extra_opts <<< "$EXTRA_CFG_OPTIONS"
  cmd+=(--cfg-options "${extra_opts[@]}")
fi

nohup "${cmd[@]}" > "$queue_log" 2>&1 &
echo "Started PID: $!"
