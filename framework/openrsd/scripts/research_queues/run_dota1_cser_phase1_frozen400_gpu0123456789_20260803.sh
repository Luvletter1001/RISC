#!/usr/bin/env bash
set -euo pipefail

exec rtk env \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 \
  NCCL_P2P_DISABLE=1 \
  NCCL_IB_DISABLE=1 \
  PYTHONNOUSERSITE=1 \
  MPLCONFIGDIR=/tmp/zcy-cser-mpl \
  TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=4 \
  PYTHONPATH=/data1/zcy/OpenRSD \
  /data/zcy/anaconda3/envs/openrsd/bin/python -m torch.distributed.run \
  --nproc_per_node=10 \
  --master_port=29683 \
  /data1/zcy/OpenRSD/tools/train.py \
  /data1/zcy/OpenRSD/M_configs/Diagnostics/dota1_cser_phase1_frozen400_gpu0123456789_20260803.py \
  --launcher pytorch
