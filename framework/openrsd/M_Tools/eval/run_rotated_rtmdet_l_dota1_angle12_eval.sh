#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/data1/zcy/OpenRSD}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data/DOTA1_1024_500}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/work_dirs/rotated_rtmdet_l_dota1_angle12_eval_4gpu_$(date +%Y%m%d_%H%M%S)}"
RESULT_MD="${RESULT_MD:-$ROOT_DIR/resultmd/rotated_rtmdet_l_dota1_angle12_eval_4gpu_$(date +%Y%m%d_%H%M%S).md}"
GPUS="${GPUS:-4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MASTER_PORT_START="${MASTER_PORT_START:-30901}"
PYTHON_BIN="${PYTHON_BIN:-/data/zcy/anaconda3/envs/openrsd/bin/python}"

CONFIG="/data1/zcy/OpenRSD/M_configs/RotationStudy/rotated_rtmdet_l_dota1_ms_eval.py"
CHECKPOINT="/data1/zcy/OpenRSD/weights/rotated_rtmdet_l-3x-dota_ms-2738da34.pth"

ANGLES=(000 030 060 090 120 150 180 210 240 270 300 330)

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

parse_metrics() {
  local log_file="$1"
  local line map="NA" ap50="NA" raw_line=""
  while IFS= read -r line; do
    if [[ "$line" == *"dota/mAP:"* ]]; then
      raw_line="$line"
      if [[ "$line" =~ dota/mAP:[[:space:]]*([0-9.]+) ]]; then
        map="${BASH_REMATCH[1]}"
      fi
      if [[ "$line" =~ dota/AP50:[[:space:]]*([0-9.]+) ]]; then
        ap50="${BASH_REMATCH[1]}"
      fi
    fi
  done < "$log_file"
  printf '%s|%s|%s\n' "$map" "$ap50" "$raw_line"
}

main() {
  cd "$ROOT_DIR" || die "Cannot cd to $ROOT_DIR"
  mkdir -p "$OUT_ROOT"
  mkdir -p "$(dirname "$RESULT_MD")"

  local total=${#ANGLES[@]}
  local ok=0 fail=0

  # Results arrays
  declare -a angle_list status_list map_list ap50_list raw_list log_list pred_list

  for i in "${!ANGLES[@]}"; do
    angle="${ANGLES[$i]}"
    safe_angle="angle_${angle}"
    run_dir="$OUT_ROOT/$safe_angle"
    log_file="$run_dir/test.log"
    pred_file="$run_dir/predictions.pkl"
    mkdir -p "$run_dir"

    local master_port=$((MASTER_PORT_START + i))

    log "[$((i+1))/${total}] Evaluating angle ${angle}°..."

    (
      env \
        CUDA_VISIBLE_DEVICES="$GPUS" \
        NCCL_P2P_DISABLE=1 \
        NCCL_IB_DISABLE=1 \
        PYTHONNOUSERSITE=1 \
        MPLCONFIGDIR=/tmp/mplconfig \
        PYTHONPATH="$ROOT_DIR:$ROOT_DIR/tools" \
        "$PYTHON_BIN" -m torch.distributed.launch \
        "--nproc_per_node=$NPROC_PER_NODE" \
        "--master_port=$master_port" \
        tools/openrsd_test.py \
        "$CONFIG" \
        "$CHECKPOINT" \
        --launcher pytorch \
        --work-dir "$run_dir" \
        --out "$pred_file" \
        --cfg-options \
        "test_dataloader.batch_size=$BATCH_SIZE" \
        "test_dataloader.num_workers=$NUM_WORKERS" \
        "test_dataloader.dataset.data_root=$DATA_ROOT" \
        "test_dataloader.dataset.ann_file=angle_sweep_val/realistic/${safe_angle}/annfiles/" \
        "test_dataloader.dataset.data_prefix.img_path=angle_sweep_val/realistic/${safe_angle}/images/"
    ) > "$log_file" 2>&1
    exit_code=$?

    if [[ "$exit_code" -eq 0 ]]; then
      status="OK"
      ok=$((ok+1))
    else
      status="FAIL($exit_code)"
      fail=$((fail+1))
    fi

    parsed=$(parse_metrics "$log_file")
    IFS='|' read -r map_val ap50_val raw_metrics <<< "$parsed"

    angle_list+=("$angle")
    status_list+=("$status")
    map_list+=("$map_val")
    ap50_list+=("$ap50_val")
    raw_list+=("$raw_metrics")
    log_list+=("$log_file")
    pred_list+=("$pred_file")

    log "  -> angle ${angle}°: status=$status mAP=$map_val AP50=$ap50_val"
  done

  # ── Write result MD ──
  {
    printf '# Rotated RTMDet-L 3x MS DOTA1 Angle Sweep Evaluation\n\n'
    printf -- '- generated_at: \`%s\`\n' "$(date '+%F %T')"
    printf -- '- out_root: \`%s\`\n' "$OUT_ROOT"
    printf -- '- config: \`%s\`\n' "$CONFIG"
    printf -- '- checkpoint: \`%s\`\n' "$CHECKPOINT"
    printf -- '- GPUs: \`%s\`\n' "$GPUS"
    printf '\n## Summary\n\n'
    printf '| total | OK | non-OK |\n'
    printf '|---:|---:|---:|\n'
    printf '| %d | %d | %d |\n\n' "$total" "$ok" "$fail"
    printf '## Metrics\n\n'
    printf '| angle | status | mAP | AP50 | predictions | log |\n'
    printf '|---:|---|---:|---:|---|---|\n'

    for j in "${!angle_list[@]}"; do
      printf '| %s | %s | %s | %s | \`%s\` | \`%s\` |\n' \
        "${angle_list[$j]}" "${status_list[$j]}" "${map_list[$j]}" "${ap50_list[$j]}" \
        "${pred_list[$j]}" "${log_list[$j]}"
    done

    printf '\n## Raw Metric Lines\n\n'
    printf '| angle | raw metrics from log |\n'
    printf '|---:|---|\n'
    for j in "${!angle_list[@]}"; do
      local safe_raw="${raw_list[$j]}"
      printf '| %s | %s |\n' "${angle_list[$j]}" "$safe_raw"
    done

  } > "$RESULT_MD"

  log "Done! Results saved to $RESULT_MD"
  log "Detailed logs in $OUT_ROOT"
}

main "$@"
