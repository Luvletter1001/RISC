#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/data1/zcy/OpenRSD}"
source "$ROOT_DIR/tools/rotation_study/lib_rotation_study.sh"

DOTA1_CLASSES="plane,baseball-diamond,bridge,ground-track-field,small-vehicle,large-vehicle,ship,tennis-court,basketball-court,storage-tank,soccer-ball-field,roundabout,harbor,swimming-pool,helicopter"
DOTA2_CLASSES="plane,baseball-diamond,bridge,ground-track-field,small-vehicle,large-vehicle,ship,tennis-court,basketball-court,storage-tank,soccer-ball-field,roundabout,harbor,swimming-pool,helicopter,container-crane,airport,helipad"
FAIR_CLASSES="a220,a321,a330,a350,arj21,baseball_field,basketball_court,boeing737,boeing747,boeing777,boeing787,bridge,bus,c919,cargo_truck,dry_cargo_ship,dump_truck,engineering_ship,excavator,fishing_boat,football_field,intersection,liquid_cargo_ship,motorboat,other-airplane,other-ship,other-vehicle,passenger_ship,roundabout,small_car,tennis_court,tractor,trailer,truck_tractor,tugboat,van,warship"
HRSC_CLASSES="ship"

run_decomp() {
  local dataset="$1"
  local model="$2"
  local split="$3"
  local pred="$4"
  local ann_dir="$5"
  local classes="$6"
  local out_csv="$OUT_ROOT/error_decomp/p3_error_decomp.csv"
  if [[ ! -f "$pred" ]]; then
    log "SKIP P3 $dataset $split: prediction missing: $pred"
    return 0
  fi
  log "Running P3 error decomposition for $dataset $model $split"
  local append_flag=()
  if [[ -f "$out_csv" ]]; then
    append_flag=(--append)
  fi
  rtk env PYTHONPATH="$ROOT_DIR:$ROOT_DIR/tools" "$PYTHON_BIN" \
    tools/rotation_study/error_decomp_obb.py \
    --predictions "$pred" \
    --ann-dir "$ann_dir" \
    --classes "$classes" \
    --dataset "$dataset" \
    --model "$model" \
    --split "$split" \
    --out-csv "$out_csv" \
    --max-images "${MAX_IMAGES:-500}" \
    "${append_flag[@]}"
}

main() {
  ensure_root
  local F_ROOT="$ROOT_DIR/data/fair1m/dair1m_1024"
  local D2_ROOT="$ROOT_DIR/data/DOTA2_1024_500"
  local D1_ROOT="$ROOT_DIR/data/DOTA1_1024_500"
  local H_ROOT="$ROOT_DIR/data/HRSC_unzip/dota"

  local D2_CFG="$ROOT_DIR/M_configs/DOTA2OfficialAdapters/redet_re50_refpn_dotav2_fullinit.py"
  local D1_CFG="$ROOT_DIR/M_configs/RotationStudy/redet_re50_refpn_dota1_eval.py"
  local F_CFG="$ROOT_DIR/M_configs/RedetFair1M/redet_re50_refpn_1x_fair1m_bs1_8gpu.py"
  local H_CFG="$ROOT_DIR/M_configs/OfficialMMRotateWeightRepro/rtmdet/rotated_rtmdet_tiny_hrsc_dota_eval.py"

  local D2_CKPT="$ROOT_DIR/work_dirs/dotav2_official_adapters/redet_re50_refpn_fullinit/epoch_4.pth"
  local D1_CKPT="$ROOT_DIR/weights/ReDet_re50_refpn_1x_dota1-a025e6b1.pth"
  local F_CKPT="$ROOT_DIR/work_dirs/redet_fair1m_bs2_8gpu/epoch_12.pth"
  local H_CKPT="$ROOT_DIR/weights/rotated_rtmdet_tiny-9x-hrsc-9f2e3ca6.pth"

  local idx=3001
  run_eval_task P3 FAIR1M fine_37 "ReDet_Re50_FAIR1M_e12" "$F_CFG" "$F_CKPT" "$F_ROOT" "val/annfiles/" "val/images/" "val" "clean" "0" "$idx" || true
  if [[ "$SMOKE_TEST_SECONDS" -gt 0 ]]; then
    log "SMOKE test finished P3 launch check before CPU decomposition."
    exit 0
  fi
  idx=$((idx + 1))
  run_eval_task P3 FAIR1M fine_37 "ReDet_Re50_FAIR1M_e12" "$F_CFG" "$F_CKPT" "$F_ROOT" "rot_val_standard/annfiles/" "rot_val_standard/images/" "rot_val_standard" "rotated" "all" "$idx" || true
  idx=$((idx + 1))
  run_eval_task P3 DOTA2 multi_class "ReDet_Re50_DOTA2_e4" "$D2_CFG" "$D2_CKPT" "$D2_ROOT" "ss_val/annfiles/" "ss_val/images/" "ss_val" "clean" "0" "$idx" || true
  idx=$((idx + 1))
  run_eval_task P3 DOTA2 multi_class "ReDet_Re50_DOTA2_e4" "$D2_CFG" "$D2_CKPT" "$D2_ROOT" "rot_val_standard/annfiles/" "rot_val_standard/images/" "rot_val_standard" "rotated" "all" "$idx" || true
  idx=$((idx + 1))
  run_eval_task P3 DOTA_v1 multi_class "ReDet_Re50_DOTA1_official" "$D1_CFG" "$D1_CKPT" "$D1_ROOT" "ss_val/annfiles/" "ss_val/images/" "ss_val" "clean" "0" "$idx" || true
  idx=$((idx + 1))
  run_eval_task P3 HRSC single_class_ship "RTMDet_tiny_HRSC_official" "$H_CFG" "$H_CKPT" "$H_ROOT" "ss_val/annfiles/" "ss_val/images/" "ss_val" "clean" "0" "$idx" || true

  run_decomp FAIR1M "ReDet_Re50_FAIR1M_e12" clean "$OUT_ROOT/P3/FAIR1M/ReDet_Re50_FAIR1M_e12/val/predictions.pkl" "$F_ROOT/val/annfiles" "$FAIR_CLASSES"
  run_decomp FAIR1M "ReDet_Re50_FAIR1M_e12" rotated "$OUT_ROOT/P3/FAIR1M/ReDet_Re50_FAIR1M_e12/rot_val_standard/predictions.pkl" "$F_ROOT/rot_val_standard/annfiles" "$FAIR_CLASSES"
  run_decomp DOTA2 "ReDet_Re50_DOTA2_e4" clean "$OUT_ROOT/P3/DOTA2/ReDet_Re50_DOTA2_e4/ss_val/predictions.pkl" "$D2_ROOT/ss_val/annfiles" "$DOTA2_CLASSES"
  run_decomp DOTA2 "ReDet_Re50_DOTA2_e4" rotated "$OUT_ROOT/P3/DOTA2/ReDet_Re50_DOTA2_e4/rot_val_standard/predictions.pkl" "$D2_ROOT/rot_val_standard/annfiles" "$DOTA2_CLASSES"
  run_decomp DOTA_v1 "ReDet_Re50_DOTA1_official" clean "$OUT_ROOT/P3/DOTA_v1/ReDet_Re50_DOTA1_official/ss_val/predictions.pkl" "$D1_ROOT/ss_val/annfiles" "$DOTA1_CLASSES"
  run_decomp HRSC "RTMDet_tiny_HRSC_official" clean "$OUT_ROOT/P3/HRSC/RTMDet_tiny_HRSC_official/ss_val/predictions.pkl" "$H_ROOT/ss_val/annfiles" "$HRSC_CLASSES"
}

main "$@"
