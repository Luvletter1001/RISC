#!/usr/bin/env bash
# Wait for RSV taxonomy pipeline, then render Top-100 LR vis with high parallelism.
set -euo pipefail

REPO="/data1/zcy/OpenRSD"
WORK="${REPO}/work_dirs/exp_rotation_gt_shift_taxonomy_20260527"
PYTHON="/data/zcy/anaconda3/envs/openrsd/bin/python"
VIS_WORKERS="${VIS_WORKERS:-512}"
TOP_N="${TOP_N:-100}"
LOG="${WORK}/logs/vis_top100_case1.log"

cd "${REPO}"
echo "[$(date -Iseconds)] wait for .done_pipeline ..."
while [[ ! -f "${WORK}/.done_pipeline" ]]; do
  sleep 20
done
echo "[$(date -Iseconds)] pipeline done; vis workers=${VIS_WORKERS} top=${TOP_N}"

exec "${PYTHON}" M_Tools/analysis/draw_top_case1_rsv_vis.py \
  --top-n "${TOP_N}" \
  --workers "${VIS_WORKERS}"
