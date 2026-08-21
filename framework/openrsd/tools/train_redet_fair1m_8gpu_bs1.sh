#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data1/zcy/OpenRSD"
CONFIG="${ROOT_DIR}/M_configs/RedetFair1M/redet_re50_refpn_1x_fair1m_bs1_8gpu.py"
WORK_DIR="${ROOT_DIR}/work_dirs/redet_fair1m_bs2_8gpu"
LOG_DIR="${WORK_DIR}/logs"
GPUS="${GPUS:-8}"
PORT="${PORT:-29611}"
CONDA_ENV="${CONDA_ENV:-openrsd}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,4,5,6,7,8,9}"
CONDA_SH="${CONDA_SH:-/data/zcy/anaconda3/etc/profile.d/conda.sh}"

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

setsid bash -lc "
    source '${CONDA_SH}' &&
    conda activate '${CONDA_ENV}' &&
    cd '${ROOT_DIR}' &&
    PORT='${PORT}' bash tools/my_dist_train.sh \
        '${CONFIG}' \
        '${GPUS}' \
        --work-dir '${WORK_DIR}'
" > "${LOG_FILE}" 2>&1 < /dev/null &

PID="$!"
echo "${PID}" > "${WORK_DIR}/train.pid"
echo "Started ReDet FAIR1M training"
echo "PID: ${PID}"
echo "Config: ${CONFIG}"
echo "Work dir: ${WORK_DIR}"
echo "Log: ${LOG_FILE}"
