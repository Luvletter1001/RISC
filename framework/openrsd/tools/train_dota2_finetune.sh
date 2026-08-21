#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

GPUS="${GPUS:-1}"
if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then
    GPUS="$1"
    shift
fi

PYTHON="${PYTHON:-/data/zcy/anaconda3/envs/openrsd/bin/python}"
CONFIG="${CONFIG:-M_configs/Step3_A12_SelfTrain/A12_flex_rtm_v3_1_DOTA2only_ss_train.py}"
WORK_DIR="${WORK_DIR:-./results/MMR_AD_A12_flex_rtm_v3_1_DOTA2only_ss_train_prob10_48e}"
PORT="${PORT:-29500}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

TRAIN_ANN_DIR="${TRAIN_ANN_DIR:-./data/DOTA2_1024_500/ss_train/annfiles}"
TRAIN_LABEL_DIR="${TRAIN_LABEL_DIR:-./data/DOTA2_1024_500/ss_train/Step6_Format_labels}"
SUPPORT_PKL="${SUPPORT_PKL:-./data/DOTA2_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl}"
LOAD_FROM="${LOAD_FROM:-./results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24.pth}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

if [ ! -f "$SUPPORT_PKL" ]; then
    echo "Missing support pkl: $SUPPORT_PKL" >&2
    exit 1
fi

if [ ! -f "$LOAD_FROM" ]; then
    echo "Missing fine-tuning checkpoint: $LOAD_FROM" >&2
    exit 1
fi

"$PYTHON" M_Tools/Data1_DOTA2/convert_dota2_txt_to_openrsd_pkl.py \
    --ann-dir "$TRAIN_ANN_DIR" \
    --out-dir "$TRAIN_LABEL_DIR"

CFG_OPTIONS=(
    "num_gpus=$GPUS"
    "train_dataloader.sampler.num_gpus=$GPUS"
    "load_from=$LOAD_FROM"
    "model.support_feat_dict.Data1_DOTA2=$SUPPORT_PKL"
)

if [ "$GPUS" -eq 1 ]; then
    PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" "$PYTHON" tools/train_rotate.py \
        "$CONFIG" \
        --work-dir "$WORK_DIR" \
        --cfg-options "${CFG_OPTIONS[@]}" \
        "$@"
else
    PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" "$PYTHON" -m torch.distributed.launch \
        --nproc_per_node="$GPUS" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$PORT" \
        tools/train_rotate.py \
        "$CONFIG" \
        --launcher pytorch \
        --work-dir "$WORK_DIR" \
        --cfg-options "${CFG_OPTIONS[@]}" \
        "$@"
fi
