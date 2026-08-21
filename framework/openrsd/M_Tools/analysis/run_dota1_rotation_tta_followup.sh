#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/data1/zcy/OpenRSD}"
PYTHON_BIN="${PYTHON_BIN:-/data/zcy/anaconda3/envs/openrsd/bin/python}"
GPUS="${GPUS:-4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-39200}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/work_dirs/dota1_exp_ab_20260507_003353}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$OUT_ROOT/exp_a_tta_followup_analysis}"
RESULTMD_DIR="${RESULTMD_DIR:-$ROOT_DIR/resultmd}"
RUN_GPU_SMOKE="${RUN_GPU_SMOKE:-1}"
FORCE_SMOKE="${FORCE_SMOKE:-0}"
MODELS="${MODELS:-rtmdet_l h2rbox_v2 retinanet_msrr}"

declare -A CFG
declare -A CKPT
declare -A BATCH
declare -A PORT_OFFSET

CFG[rtmdet_l]="$ROOT_DIR/M_configs/RotationStudy/rotated_rtmdet_l_dota1_ms_eval.py"
CKPT[rtmdet_l]="$ROOT_DIR/weights/rotated_rtmdet_l-3x-dota_ms-2738da34.pth"
BATCH[rtmdet_l]=24
PORT_OFFSET[rtmdet_l]=0

CFG[h2rbox_v2]="$ROOT_DIR/M_configs/RotationStudy/h2rbox_v2_r50_fpn_dota1_ms_rr_eval.py"
CKPT[h2rbox_v2]="$ROOT_DIR/weights/h2rbox_v2-le90_r50_fpn_ms_rr-1x_dota-5e0e53e1.pth"
BATCH[h2rbox_v2]=32
PORT_OFFSET[h2rbox_v2]=100

CFG[retinanet_msrr]="$ROOT_DIR/M_configs/RotationStudy/rotated_retinanet_r50_msrr_dota1_eval.py"
CKPT[retinanet_msrr]="$ROOT_DIR/weights/rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90-1da1ec9c.pth"
BATCH[retinanet_msrr]=48
PORT_OFFSET[retinanet_msrr]=200

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    log "ERROR missing file: $path"
    exit 2
  fi
}

run_gpu_smoke_one() {
  local model="$1"
  local cfg="${CFG[$model]}"
  local ckpt="${CKPT[$model]}"
  local batch="${BATCH[$model]}"
  local port=$((MASTER_PORT_BASE + PORT_OFFSET[$model]))
  local out_dir="$ANALYSIS_ROOT/gpu_smoke/$model/angle_000"
  local pred="$out_dir/predictions.pkl"
  local log_file="$out_dir/test.log"

  if (( batch < 16 )); then
    log "ERROR batch size for $model is below 16: $batch"
    exit 3
  fi
  require_file "$cfg"
  require_file "$ckpt"

  if [[ "$FORCE_SMOKE" != "1" && -f "$pred" ]]; then
    log "Skip existing GPU smoke: $model -> $pred"
    return 0
  fi

  rtk /usr/bin/mkdir -p "$out_dir"
  log "GPU smoke: model=$model gpus=$GPUS per_gpu_batch=$batch out=$out_dir"
  (
    cd "$ROOT_DIR"
    rtk env \
      CUDA_VISIBLE_DEVICES="$GPUS" \
      NCCL_P2P_DISABLE=1 \
      NCCL_IB_DISABLE=1 \
      PYTHONNOUSERSITE=1 \
      MPLCONFIGDIR=/tmp/mplconfig \
      PYTHONPATH="$ROOT_DIR:$ROOT_DIR/tools" \
      "$PYTHON_BIN" -m torch.distributed.launch \
      "--nproc_per_node=$NPROC_PER_NODE" \
      "--master_port=$port" \
      tools/openrsd_test.py \
      "$cfg" \
      "$ckpt" \
      --launcher pytorch \
      --work-dir "$out_dir" \
      --out "$pred" \
      --cfg-options \
      "test_dataloader.batch_size=$batch" \
      "test_dataloader.num_workers=$NUM_WORKERS" \
      "test_dataloader.persistent_workers=False" \
      "test_dataloader.dataset.data_root=$ROOT_DIR/data/DOTA1_1024_500" \
      "test_dataloader.dataset.ann_file=angle_sweep_val/realistic/angle_000/annfiles/" \
      "test_dataloader.dataset.data_prefix.img_path=angle_sweep_val/realistic/angle_000/images/"
  ) > "$log_file" 2>&1
}

main() {
  rtk /usr/bin/mkdir -p "$ANALYSIS_ROOT" "$RESULTMD_DIR"

  if [[ "$RUN_GPU_SMOKE" == "1" ]]; then
    for model in $MODELS; do
      run_gpu_smoke_one "$model"
    done
  else
    log "Skipping GPU smoke because RUN_GPU_SMOKE=$RUN_GPU_SMOKE"
  fi

  log "Running follow-up analysis and md generation"
  (
    cd "$ROOT_DIR"
    rtk env \
      PYTHONNOUSERSITE=1 \
      MPLCONFIGDIR=/tmp/mplconfig \
      PYTHONPATH="$ROOT_DIR:$ROOT_DIR/tools" \
      "$PYTHON_BIN" \
      M_Tools/analysis/dota1_rotation_tta_followup.py \
      --task all \
      --models $MODELS \
      --out-root "$OUT_ROOT" \
      --analysis-root "$ANALYSIS_ROOT" \
      --resultmd-dir "$RESULTMD_DIR" \
      --python-bin "$PYTHON_BIN" \
      --gpus "$GPUS"
  )
}

main "$@"
