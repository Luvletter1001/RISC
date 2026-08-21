#!/usr/bin/env bash
set -u

cd /data1/zcy/OpenRSD

TS=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="work_dirs/rotation_overnight_gpu89_${TS}"
REPORT_DIR="resultmd/exp_rotation_overnight_gpu89"
mkdir -p "${OUT_ROOT}/logs" "${REPORT_DIR}"

ANGLES_FULL=(0 45 50 55 60 75 90 180 225 230 240 250 260 270)
ANGLES_CROSS=(0 45 90 180 270)

if [[ "${1:-}" == "--smoke" ]]; then
  ANGLES_DENSE=(0 50)
  ANGLES_EMB=(0 50)
  MAX_CROPS=20
  MAX_TILES=3
  EMB_INTS=(original zero_sv swap_sv_lv)
else
  ANGLES_DENSE=("${ANGLES_FULL[@]}")
  ANGLES_EMB=("${ANGLES_FULL[@]}")
  MAX_CROPS=300
  MAX_TILES=30
  EMB_INTS=(original zero_sv swap_sv_lv norm_sv_mean normalize_all random_sv)
fi

env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OpenRSD CUDA_VISIBLE_DEVICES=8 \
  /data/zcy/anaconda3/envs/openrsd/bin/python tools/rotation_overnight_gpu89/run_dense_prenms_gpu8.py \
  --project-root /data1/zcy/OpenRSD \
  --image-dir vis/P0148__1024__651___0/dataset/images \
  --out-dir "${OUT_ROOT}/gpu8_dense" \
  --report-dir "${REPORT_DIR}" \
  --angles "${ANGLES_DENSE[@]}" \
  > "${OUT_ROOT}/logs/gpu8_dense.log" 2>&1 &
PID8=$!

env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OpenRSD CUDA_VISIBLE_DEVICES=9 \
  /data/zcy/anaconda3/envs/openrsd/bin/python tools/rotation_overnight_gpu89/run_embedding_intervention_gpu9.py \
  --project-root /data1/zcy/OpenRSD \
  --image-dir vis/P0148__1024__651___0/dataset/images \
  --out-dir "${OUT_ROOT}/gpu9_embedding" \
  --report-dir "${REPORT_DIR}" \
  --angles "${ANGLES_EMB[@]}" \
  --interventions "${EMB_INTS[@]}" \
  > "${OUT_ROOT}/logs/gpu9_embedding.log" 2>&1
EC_EMB=$?

env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OpenRSD CUDA_VISIBLE_DEVICES=9 \
  /data/zcy/anaconda3/envs/openrsd/bin/python tools/rotation_overnight_gpu89/export_p0148_detection_audit_gpu9.py \
  --out-dir "${OUT_ROOT}/gpu9_audit" \
  --report-dir "${REPORT_DIR}" \
  --max-crops "${MAX_CROPS}" \
  > "${OUT_ROOT}/logs/gpu9_audit.log" 2>&1
EC_AUDIT=$?

env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OpenRSD CUDA_VISIBLE_DEVICES=9 \
  /data/zcy/anaconda3/envs/openrsd/bin/python tools/rotation_overnight_gpu89/run_cross_tile_extended_gpu9.py \
  --project-root /data1/zcy/OpenRSD \
  --out-dir "${OUT_ROOT}/gpu9_cross_tile_ext" \
  --report-dir "${REPORT_DIR}" \
  --angles "${ANGLES_CROSS[@]}" \
  --max-tiles "${MAX_TILES}" \
  > "${OUT_ROOT}/logs/gpu9_cross_tile_ext.log" 2>&1
EC_CROSS=$?

wait ${PID8}
EC_DENSE=$?

env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OpenRSD \
  /data/zcy/anaconda3/envs/openrsd/bin/python tools/rotation_overnight_gpu89/make_overnight_summary.py \
  --out-root "${OUT_ROOT}" \
  --report-dir "${REPORT_DIR}" \
  --gpu8-dense-exit-code "${EC_DENSE}" \
  --gpu9-embedding-exit-code "${EC_EMB}" \
  --gpu9-audit-exit-code "${EC_AUDIT}" \
  --gpu9-cross-tile-exit-code "${EC_CROSS}" \
  > "${OUT_ROOT}/logs/overnight_summary.log" 2>&1 || true

echo "OUT_ROOT=${OUT_ROOT}"
echo "REPORT_DIR=${REPORT_DIR}"
echo "GPU8_DENSE_EXIT=${EC_DENSE}"
echo "GPU9_EMBEDDING_EXIT=${EC_EMB}"
echo "GPU9_AUDIT_EXIT=${EC_AUDIT}"
echo "GPU9_CROSS_TILE_EXIT=${EC_CROSS}"
cat "${OUT_ROOT}/logs/overnight_summary.log" || true

exit $(( EC_DENSE != 0 || EC_EMB != 0 || EC_AUDIT != 0 || EC_CROSS != 0 ))
