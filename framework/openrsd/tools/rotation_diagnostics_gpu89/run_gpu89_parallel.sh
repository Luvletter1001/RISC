#!/usr/bin/env bash
set -u

cd /data1/zcy/OpenRSD

PY=/data/zcy/anaconda3/envs/openrsd/bin/python
TS=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="work_dirs/rotation_causal_probe_P0148_gpu89_${TS}"
REPORT_DIR="resultmd/exp_rotation_causal_probe_P0148_gpu89"
mkdir -p "${OUT_ROOT}/logs" "${REPORT_DIR}"

CUDA_VISIBLE_DEVICES=8 "${PY}" tools/rotation_diagnostics_gpu89/run_p0148_causal_probe_gpu8.py \
  --project-root /data1/zcy/OpenRSD \
  --image-dir vis/P0148__1024__651___0/dataset/images \
  --visual-probe-dir work_dirs/rotation_stage_probe_P0148_full_physgpu8_9 \
  --text-probe-dir work_dirs/rotation_stage_probe_P0148_text_full_physgpu8_9 \
  --out-dir "${OUT_ROOT}/p0148_gpu8" \
  --report-dir "${REPORT_DIR}" \
  --angles 0 5 45 50 55 60 75 90 180 225 230 240 250 260 270 \
  > "${OUT_ROOT}/logs/gpu8_p0148.log" 2>&1 &
PID8=$!

CUDA_VISIBLE_DEVICES=9 "${PY}" tools/rotation_diagnostics_gpu89/run_cross_tile_probe_gpu9.py \
  --project-root /data1/zcy/OpenRSD \
  --out-dir "${OUT_ROOT}/cross_tile_gpu9" \
  --report-dir "${REPORT_DIR}" \
  --angles 0 45 90 180 270 \
  --max-tiles 12 \
  > "${OUT_ROOT}/logs/gpu9_cross_tile.log" 2>&1 &
PID9=$!

wait "${PID8}"
EC8=$?
wait "${PID9}"
EC9=$?

"${PY}" tools/rotation_diagnostics_gpu89/make_gpu89_final_summary.py \
  --out-root "${OUT_ROOT}" \
  --report-dir "${REPORT_DIR}" \
  --gpu8-exit-code "${EC8}" \
  --gpu9-exit-code "${EC9}" \
  > "${OUT_ROOT}/logs/final_summary.log" 2>&1

echo "GPU8 exit code: ${EC8}"
echo "GPU9 exit code: ${EC9}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "REPORT_DIR=${REPORT_DIR}"

exit $(( EC8 != 0 || EC9 != 0 ))
