#!/usr/bin/env bash
set -euo pipefail

cd /data1/zcy/OpenRSD

MONITOR_DIR=/data1/zcy/OpenRSD/resultmd/exp_rotation_sv_repair_20260524/overnight_monitor
LOG_DIR=/data1/zcy/OpenRSD/work_dirs/exp_rotation_sv_repair_20260524/logs
mkdir -p "${MONITOR_DIR}" "${LOG_DIR}"

export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=/tmp/mplconfig
export PYTHONPATH=/data1/zcy/OpenRSD:/data1/zcy/OpenRSD/tools
export CUDA_VISIBLE_DEVICES=8,9
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

/data/zcy/anaconda3/envs/openrsd/bin/python M_Tools/rotation_sv_repair/overnight_watchdog.py \
  --repo-root /data1/zcy/OpenRSD \
  --work-dir /data1/zcy/OpenRSD/work_dirs/exp_rotation_sv_repair_20260524 \
  --result-dir /data1/zcy/OpenRSD/resultmd/exp_rotation_sv_repair_20260524 \
  --monitor-dir "${MONITOR_DIR}" \
  --gpu-ids 8,9 \
  --interval-minutes 30 \
  --max-hours 12 \
  --mode aggressive \
  --resume \
  --auto-fix \
  --auto-rerun \
  --no-delete \
  --respect-baseline-lock
