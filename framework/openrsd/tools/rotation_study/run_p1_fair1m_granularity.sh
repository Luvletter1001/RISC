#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/data1/zcy/OpenRSD}"
source "$ROOT_DIR/tools/rotation_study/lib_rotation_study.sh"

main() {
  ensure_root
  local F_ROOT="$ROOT_DIR/data/fair1m/dair1m_1024"
  local CFG="$ROOT_DIR/M_configs/RedetFair1M/redet_re50_refpn_1x_fair1m_bs1_8gpu.py"
  local CKPT="$ROOT_DIR/work_dirs/redet_fair1m_bs2_8gpu/epoch_12.pth"
  local model="ReDet_Re50_FAIR1M_e12"
  local idx=1001

  run_eval_task P1 FAIR1M fine_37 "$model" "$CFG" "$CKPT" "$F_ROOT" "val/annfiles/" "val/images/" "val" "clean" "0" "$idx" || true
  if [[ "$SMOKE_TEST_SECONDS" -gt 0 ]]; then
    log "SMOKE test finished P1 launch check before FAIR1M-5 remap."
    exit 0
  fi
  idx=$((idx + 1))
  run_eval_task P1 FAIR1M fine_37 "$model" "$CFG" "$CKPT" "$F_ROOT" "rot_val_standard/annfiles/" "rot_val_standard/images/" "rot_val_standard" "rotated" "all" "$idx" || true

  local clean_pred="$OUT_ROOT/P1/FAIR1M/ReDet_Re50_FAIR1M_e12/val/predictions.pkl"
  local rot_pred="$OUT_ROOT/P1/FAIR1M/ReDet_Re50_FAIR1M_e12/rot_val_standard/predictions.pkl"
  local clean_out="$OUT_ROOT/p1_granularity/fair1m5_clean_ap50.csv"
  local rot_out="$OUT_ROOT/p1_granularity/fair1m5_rotated_ap50.csv"

  if [[ -f "$clean_pred" ]]; then
    log "Running FAIR1M-5 remap eval for clean val."
    rtk env PYTHONPATH="$ROOT_DIR:$ROOT_DIR/tools" "$PYTHON_BIN" \
      tools/rotation_study/remap_fair1m_eval.py \
      --predictions "$clean_pred" \
      --ann-dir "$F_ROOT/val/annfiles" \
      --split clean \
      --out-csv "$clean_out"
  fi
  if [[ -f "$rot_pred" ]]; then
    log "Running FAIR1M-5 remap eval for rotated val."
    rtk env PYTHONPATH="$ROOT_DIR:$ROOT_DIR/tools" "$PYTHON_BIN" \
      tools/rotation_study/remap_fair1m_eval.py \
      --predictions "$rot_pred" \
      --ann-dir "$F_ROOT/rot_val_standard/annfiles" \
      --split rotated \
      --out-csv "$rot_out"
  fi
}

main "$@"
