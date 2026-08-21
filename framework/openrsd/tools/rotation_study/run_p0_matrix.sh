#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/data1/zcy/OpenRSD}"
source "$ROOT_DIR/tools/rotation_study/lib_rotation_study.sh"

SMOKE_MAX_TASKS="${SMOKE_MAX_TASKS:-1}"

DOTA1_CLASSES="plane,baseball-diamond,bridge,ground-track-field,small-vehicle,large-vehicle,ship,tennis-court,basketball-court,storage-tank,soccer-ball-field,roundabout,harbor,swimming-pool,helicopter"
DOTA2_CLASSES="$DOTA1_CLASSES,container-crane,airport,helipad"

maybe_stop_smoke() {
  local done_count="$1"
  if [[ "$SMOKE_TEST_SECONDS" -gt 0 && "$done_count" -ge "$SMOKE_MAX_TASKS" ]]; then
    log "SMOKE test reached $done_count task(s); stopping P0 early."
    exit 0
  fi
}

main() {
  ensure_root
  local idx=0

  local D2_ROOT="$ROOT_DIR/data/DOTA2_1024_500"
  local D1_ROOT="$ROOT_DIR/data/DOTA1_1024_500"
  local F_ROOT="$ROOT_DIR/data/fair1m/dair1m_1024"
  local H_ROOT="$ROOT_DIR/data/HRSC_unzip/dota"

  local RED_D2_CFG="$ROOT_DIR/M_configs/DOTA2OfficialAdapters/redet_re50_refpn_dotav2_fullinit.py"
  local ORC_D2_CFG="$ROOT_DIR/M_configs/DOTA2OfficialAdapters/oriented_rcnn_r50_fpn_dotav2_fullinit.py"
  local R3D_D2_CFG="$ROOT_DIR/M_configs/DOTA2OfficialAdapters/r3det_kfiou_r50_fpn_dotav2_fullinit.py"
  local RED_D1_CFG="$ROOT_DIR/M_configs/RotationStudy/redet_re50_refpn_dota1_eval.py"
  local ORC_D1_CFG="$ROOT_DIR/M_configs/OfficialMMRotateWeightRepro/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py"
  local FAIR_CFG="$ROOT_DIR/M_configs/RedetFair1M/redet_re50_refpn_1x_fair1m_bs1_8gpu.py"
  local HRSC_CFG="$ROOT_DIR/M_configs/OfficialMMRotateWeightRepro/rtmdet/rotated_rtmdet_tiny_hrsc_dota_eval.py"

  local RED_D2_CKPT="$ROOT_DIR/work_dirs/dotav2_official_adapters/redet_re50_refpn_fullinit/epoch_4.pth"
  local ORC_D2_CKPT="$ROOT_DIR/work_dirs/dotav2_official_adapters/oriented_rcnn_r50_fpn_fullinit/epoch_8.pth"
  local R3D_D2_CKPT="$ROOT_DIR/work_dirs/dotav2_official_adapters/r3det_kfiou_r50_fpn_fullinit/epoch_12.pth"
  local RED_D1_CKPT="$ROOT_DIR/weights/ReDet_re50_refpn_1x_dota1-a025e6b1.pth"
  local ORC_D1_CKPT="$ROOT_DIR/weights/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth"
  local FAIR_CKPT="$ROOT_DIR/work_dirs/redet_fair1m_bs2_8gpu/epoch_12.pth"
  local HRSC_CKPT="$ROOT_DIR/weights/rotated_rtmdet_tiny-9x-hrsc-9f2e3ca6.pth"

  local split_entry split ann img rotation angle model_entry model cfg ckpt gran

  local D2_SPLITS=(
    "ss_val|ss_val/annfiles/|ss_val/images/|clean|0"
    "rot_val_standard|rot_val_standard/annfiles/|rot_val_standard/images/|rotated|all"
    "rot_val_right|rot_val_standard_0_90_180_270/annfiles/|rot_val_standard_0_90_180_270/images/|right_angle|0_90_180_270"
    "rot_val_nonright|rot_val_standard_30_60_120_150_210_240_300_330/annfiles/|rot_val_standard_30_60_120_150_210_240_300_330/images/|non_right|30_60_120_150_210_240_300_330"
  )
  local D2_MODELS=(
    "ReDet_Re50_DOTA2_e4|$RED_D2_CFG|$RED_D2_CKPT"
    "ORCNN_R50_DOTA2_e8|$ORC_D2_CFG|$ORC_D2_CKPT"
    "R3Det_KFIoU_R50_DOTA2_e12|$R3D_D2_CFG|$R3D_D2_CKPT"
  )
  for model_entry in "${D2_MODELS[@]}"; do
    IFS='|' read -r model cfg ckpt <<< "$model_entry"
    for split_entry in "${D2_SPLITS[@]}"; do
      IFS='|' read -r split ann img rotation angle <<< "$split_entry"
      idx=$((idx + 1))
      run_eval_task P0 DOTA2 multi_class "$model" "$cfg" "$ckpt" "$D2_ROOT" "$ann" "$img" "$split" "$rotation" "$angle" "$idx" || true
      maybe_stop_smoke "$idx"
    done
  done

  local D1_SPLITS=(
    "ss_val|ss_val/annfiles/|ss_val/images/|clean|0"
    "angle_030|angle_sweep_val/realistic/angle_030/annfiles/|angle_sweep_val/realistic/angle_030/images/|rotated|30"
    "angle_060|angle_sweep_val/realistic/angle_060/annfiles/|angle_sweep_val/realistic/angle_060/images/|rotated|60"
    "angle_090|angle_sweep_val/realistic/angle_090/annfiles/|angle_sweep_val/realistic/angle_090/images/|right_angle|90"
  )
  local D1_MODELS=(
    "ReDet_Re50_DOTA1_official|$RED_D1_CFG|$RED_D1_CKPT"
    "ORCNN_R50_DOTA1_official|$ORC_D1_CFG|$ORC_D1_CKPT"
  )
  for model_entry in "${D1_MODELS[@]}"; do
    IFS='|' read -r model cfg ckpt <<< "$model_entry"
    for split_entry in "${D1_SPLITS[@]}"; do
      IFS='|' read -r split ann img rotation angle <<< "$split_entry"
      idx=$((idx + 1))
      run_eval_task P0 DOTA_v1 multi_class "$model" "$cfg" "$ckpt" "$D1_ROOT" "$ann" "$img" "$split" "$rotation" "$angle" "$idx" || true
      maybe_stop_smoke "$idx"
    done
  done

  local F_SPLITS=(
    "val|val/annfiles/|val/images/|clean|0"
    "rot_val_standard|rot_val_standard/annfiles/|rot_val_standard/images/|rotated|all"
    "rot_val_right|rot_val_standard_0_90_180_270/annfiles/|rot_val_standard_0_90_180_270/images/|right_angle|0_90_180_270"
    "rot_val_nonright|rot_val_standard_30_60_120_150_210_240_300_330/annfiles/|rot_val_standard_30_60_120_150_210_240_300_330/images/|non_right|30_60_120_150_210_240_300_330"
  )
  for split_entry in "${F_SPLITS[@]}"; do
    IFS='|' read -r split ann img rotation angle <<< "$split_entry"
    idx=$((idx + 1))
    run_eval_task P0 FAIR1M fine_37 "ReDet_Re50_FAIR1M_e12" "$FAIR_CFG" "$FAIR_CKPT" "$F_ROOT" "$ann" "$img" "$split" "$rotation" "$angle" "$idx" || true
    maybe_stop_smoke "$idx"
  done

  local H_SPLITS=(
    "ss_val|ss_val/annfiles/|ss_val/images/|clean|0"
    "rot_val_standard|rot_val_standard/annfiles/|rot_val_standard/images/|rotated|all"
    "rot_val_right|rot_val_standard_0_90_180_270/annfiles/|rot_val_standard_0_90_180_270/images/|right_angle|0_90_180_270"
    "rot_val_nonright|rot_val_standard_30_60_120_150_210_240_300_330/annfiles/|rot_val_standard_30_60_120_150_210_240_300_330/images/|non_right|30_60_120_150_210_240_300_330"
  )
  for split_entry in "${H_SPLITS[@]}"; do
    IFS='|' read -r split ann img rotation angle <<< "$split_entry"
    idx=$((idx + 1))
    run_eval_task P0 HRSC single_class_ship "RTMDet_tiny_HRSC_official" "$HRSC_CFG" "$HRSC_CKPT" "$H_ROOT" "$ann" "$img" "$split" "$rotation" "$angle" "$idx" || true
    maybe_stop_smoke "$idx"
  done
}

main "$@"
