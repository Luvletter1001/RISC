#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/data1/zcy/OpenRSD}"
source "$ROOT_DIR/tools/rotation_study/lib_rotation_study.sh"

ANGLES="${ANGLES:-0 30 60 90 120 150 180 210 240 270 300 330}"
SMOKE_MAX_TASKS="${SMOKE_MAX_TASKS:-1}"

angle_name() {
  printf '%03d\n' "$1"
}

maybe_stop_smoke() {
  local done_count="$1"
  if [[ "$SMOKE_TEST_SECONDS" -gt 0 && "$done_count" -ge "$SMOKE_MAX_TASKS" ]]; then
    log "SMOKE test reached $done_count task(s); stopping P2 early."
    exit 0
  fi
}

main() {
  ensure_root
  local idx=2000
  local aname angle

  local D2_ROOT="$ROOT_DIR/data/DOTA2_1024_500"
  local D1_ROOT="$ROOT_DIR/data/DOTA1_1024_500"
  local F_ROOT="$ROOT_DIR/data/fair1m/dair1m_1024"

  local D2_CFG="$ROOT_DIR/M_configs/DOTA2OfficialAdapters/redet_re50_refpn_dotav2_fullinit.py"
  local D1_CFG="$ROOT_DIR/M_configs/RotationStudy/redet_re50_refpn_dota1_eval.py"
  local F_CFG="$ROOT_DIR/M_configs/RedetFair1M/redet_re50_refpn_1x_fair1m_bs1_8gpu.py"

  local D2_CKPT="$ROOT_DIR/work_dirs/dotav2_official_adapters/redet_re50_refpn_fullinit/epoch_4.pth"
  local D1_CKPT="$ROOT_DIR/weights/ReDet_re50_refpn_1x_dota1-a025e6b1.pth"
  local F_CKPT="$ROOT_DIR/work_dirs/redet_fair1m_bs2_8gpu/epoch_12.pth"

  local need_prepare=0
  for angle in $ANGLES; do
    aname="$(angle_name "$angle")"
    if [[ ! -d "$D2_ROOT/angle_sweep_val/realistic/angle_${aname}/annfiles" ]]; then
      need_prepare=1
      break
    fi
  done
  if [[ "$need_prepare" == "1" ]]; then
    log "Preparing DOTA2 angle_sweep_val for angles: $ANGLES"
    rtk env PYTHONPATH="$ROOT_DIR:$ROOT_DIR/M_Tools/Data1_DOTA2:$ROOT_DIR/tools" "$PYTHON_BIN" \
      M_Tools/Data1_DOTA2/prepare_dotav2_angle_sweep.py \
      --data-root "$D2_ROOT" \
      --out-root "$D2_ROOT/angle_sweep_val" \
      --angles $ANGLES \
      --nproc "${ANGLE_PREP_NPROC:-16}" \
      --link-mode "${ANGLE_LINK_MODE:-hardlink}"
  fi

  for angle in $ANGLES; do
    aname="$(angle_name "$angle")"
    idx=$((idx + 1))
    run_eval_task P2 DOTA2 multi_class "ReDet_Re50_DOTA2_e4" "$D2_CFG" "$D2_CKPT" "$D2_ROOT" "angle_sweep_val/realistic/angle_${aname}/annfiles/" "angle_sweep_val/realistic/angle_${aname}/images/" "angle_${aname}" "angle_sweep" "$angle" "$idx" || true
    maybe_stop_smoke "$((idx - 2000))"
  done

  for angle in $ANGLES; do
    aname="$(angle_name "$angle")"
    idx=$((idx + 1))
    run_eval_task P2 DOTA_v1 multi_class "ReDet_Re50_DOTA1_official" "$D1_CFG" "$D1_CKPT" "$D1_ROOT" "angle_sweep_val/realistic/angle_${aname}/annfiles/" "angle_sweep_val/realistic/angle_${aname}/images/" "angle_${aname}" "angle_sweep" "$angle" "$idx" || true
    maybe_stop_smoke "$((idx - 2000))"
  done

  for angle in $ANGLES; do
    aname="$(angle_name "$angle")"
    idx=$((idx + 1))
    run_eval_task P2 FAIR1M fine_37 "ReDet_Re50_FAIR1M_e12" "$F_CFG" "$F_CKPT" "$F_ROOT" "rot_val_standard/realistic/angle_${aname}/annfiles/" "rot_val_standard/realistic/angle_${aname}/images/" "angle_${aname}" "angle_sweep" "$angle" "$idx" || true
    maybe_stop_smoke "$((idx - 2000))"
  done
}

main "$@"
