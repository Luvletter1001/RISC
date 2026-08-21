#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/zcy/OpenRSD/work_dirs/scheduled_runs/retry_reppoints_rot_val_standard_b4_8gpu_$(date +%Y%m%d_%H%M%S)"
TASK="$ROOT/RepPoints_R50_best_rot_val_standard_b4"
mkdir -p "$TASK"
echo "$ROOT" > /data1/zcy/OpenRSD/work_dirs/scheduled_runs/latest_retry_reppoints_rot_val_standard_b4_8gpu.txt

LOG="$TASK/test.log"
RUNLOG="$ROOT/run.log"
RESULTS="$ROOT/results.tsv"
SUMMARY="$ROOT/summary.md"
CONFIG="/data1/zcy/OpenRSD/M_configs/DOTA2OfficialAdapters/oriented_reppoints_r50_fpn_dotav2_fullinit.py"
CKPT="/data1/zcy/OpenRSD/work_dirs/scheduled_runs/may02_0167_full_series_20260502_120001/train_reppoints_fullinit/epoch_4.pth"

echo "[$(date '+%F %T')] Retrying RepPoints_R50_best on rot_val_standard with GPUs=0,1,4,5,6,7,8,9 batch_size=4" | tee "$RUNLOG"

set +e
env \
  CUDA_VISIBLE_DEVICES=0,1,4,5,6,7,8,9 \
  NCCL_P2P_DISABLE=1 \
  NCCL_IB_DISABLE=1 \
  PYTHONNOUSERSITE=1 \
  MPLCONFIGDIR=/tmp/mplconfig \
  PYTHONPATH=/data1/zcy/OpenRSD:/data1/zcy/OpenRSD/tools \
  /data/zcy/anaconda3/envs/openrsd/bin/python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=31383 \
    tools/openrsd_test.py \
    "$CONFIG" \
    "$CKPT" \
    --launcher pytorch \
    --work-dir "$TASK" \
    --out "$TASK/predictions.pkl" \
    --cfg-options \
      test_dataloader.batch_size=4 \
      test_dataloader.num_workers=8 \
      test_dataloader.dataset.data_root=/data1/zcy/OpenRSD/data/DOTA2_1024_500 \
      test_dataloader.dataset.ann_file=rot_val_standard/annfiles/ \
      test_dataloader.dataset.data_prefix.img_path=rot_val_standard/images/ \
  2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e

map="NA"
ap50="NA"
if [ "$status" -eq 0 ]; then
  line="$(grep -E 'Epoch\(test\).*dota/mAP' "$LOG" | tail -1 || true)"
  if [ -n "$line" ]; then
    map="$(printf "%s\n" "$line" | sed -n 's/.*dota\/mAP: \([0-9.]*\).*/\1/p')"
    ap50="$(printf "%s\n" "$line" | sed -n 's/.*dota\/AP50: \([0-9.]*\).*/\1/p')"
  fi
  label="OK"
else
  label="FAIL($status)"
fi

printf "model|dataset|status|map|ap50|log|predictions\n" > "$RESULTS"
printf "RepPoints_R50_best|rot_val_standard|%s|%s|%s|%s|%s\n" \
  "$label" "$map" "$ap50" "$LOG" "$TASK/predictions.pkl" >> "$RESULTS"

{
  echo "# RepPoints rot_val_standard 8GPU Retry"
  echo
  echo "- generated_at: \`$(date '+%F %T')\`"
  echo "- output_root: \`$ROOT\`"
  echo "- dataset: \`rot_val_standard\`"
  echo "- checkpoint: \`$CKPT\`"
  echo "- GPUs: \`0,1,4,5,6,7,8,9\`"
  echo "- batch_size_per_gpu: \`4\`"
  echo
  echo "| model | dataset | status | eval mAP | eval AP50 | log | predictions |"
  echo "|---|---|---|---:|---:|---|---|"
  echo "| RepPoints_R50_best | rot_val_standard | $label | $map | $ap50 | \`$LOG\` | \`$TASK/predictions.pkl\` |"
} > "$SUMMARY"

echo "[$(date '+%F %T')] Finished RepPoints_R50_best on rot_val_standard: status=$label mAP=$map AP50=$ap50" | tee -a "$RUNLOG"
exit "$status"
