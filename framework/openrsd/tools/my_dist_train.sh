#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PYTHON=${PYTHON:-/data/zcy/anaconda3/envs/openrsd/bin/python}

export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

PYTHONPATH="$(dirname "$0")/..":${PYTHONPATH:-} \
"$PYTHON" -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train_rotate.py \
    $CONFIG \
    --launcher pytorch ${@:3}
