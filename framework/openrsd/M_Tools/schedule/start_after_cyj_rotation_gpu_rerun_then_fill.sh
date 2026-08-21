#!/usr/bin/env bash
set -uo pipefail

# Saved command for resuming after cyj's experiment finishes.
# This starts a tmux job that:
# 1) re-runs the GPU angle-sweep stage on GPUs 0,1,4,5,6,7,8,9,
# 2) then fills remaining postprocess/NA rows into the same Markdown report.

cd /data1/zcy/OpenRSD || exit 1

rtk env \
  OUT_ROOT=/data1/zcy/OpenRSD/work_dirs/scheduled_runs/rotation_research_full_8gpu_20260504_014158 \
  SESSION=rotation_research_gpu_rerun_then_fill \
  GPUS=0,1,4,5,6,7,8,9 \
  NPROC_PER_NODE=8 \
  NUM_WORKERS=8 \
  PYTHON_BIN=/data/zcy/anaconda3/envs/openrsd/bin/python \
  TTA_PRE_NMS_TOPK=1000 \
  DIAGNOSTIC_MAX_DETS_PER_IMG=50 \
  bash M_Tools/schedule/start_rotation_research_gpu_rerun_then_fill_tmux.sh
