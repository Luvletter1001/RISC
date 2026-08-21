#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/data1/zcy/OpenRSD}"
PYTHON_BIN="${PYTHON_BIN:-/data/zcy/anaconda3/envs/openrsd/bin/python}"
GPUS="${GPUS:-4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-24}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MASTER_PORT_START="${MASTER_PORT_START:-37100}"

CONFIG="${CONFIG:-$ROOT_DIR/M_configs/RotationStudy/redet_re50_refpn_dota1_eval.py}"
CHECKPOINT="${CHECKPOINT:-$ROOT_DIR/weights/redet_re50_fpn_1x_dota_ms_rr_le90-fc9217b5.pth}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data/DOTA1_1024_500}"
ANGLE_ROOT="${ANGLE_ROOT:-$DATA_ROOT/angle_sweep_val/realistic}"

RUN_TS="${RUN_TS:-$(rtk /usr/bin/date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/work_dirs/dota1_angle12_redet_eval_4gpu_$RUN_TS}"
RESULT_MD="${RESULT_MD:-$ROOT_DIR/resultmd/dota1_angle12_redet_eval_4gpu_$RUN_TS.md}"
RESULT_CSV="$OUT_ROOT/results.csv"
RUN_LOG="$OUT_ROOT/run.log"

ANGLES="${ANGLES:-000 030 060 090 120 150 180 210 240 270 300 330}"

log() {
  local line
  line="[$(rtk /usr/bin/date '+%F %T')] $*"
  printf '%s\n' "$line"
  printf '%s\n' "$line" >> "$RUN_LOG"
}

assert_file() {
  if [[ ! -f "$1" ]]; then
    log "ERROR missing file: $1"
    exit 2
  fi
}

assert_dir() {
  if [[ ! -d "$1" ]]; then
    log "ERROR missing directory: $1"
    exit 2
  fi
}

parse_metrics() {
  local log_file="$1"
  rtk "$PYTHON_BIN" - "$log_file" <<'PY'
import json
import re
import sys

path = sys.argv[1]
try:
    text = open(path, 'r', encoding='utf-8', errors='replace').read()
except FileNotFoundError:
    text = ''

def last(pattern):
    matches = re.findall(pattern, text)
    return matches[-1] if matches else 'NA'

map_value = last(r'["\']?dota/mAP["\']?\s*[:=]\s*([0-9]*\.?[0-9]+)')
ap50_value = last(r'["\']?dota/AP50["\']?\s*[:=]\s*([0-9]*\.?[0-9]+)')
ap07_value = last(r'["\']?dota_ap07/mAP["\']?\s*[:=]\s*([0-9]*\.?[0-9]+)')
ap12_value = last(r'["\']?dota_ap12/mAP["\']?\s*[:=]\s*([0-9]*\.?[0-9]+)')

metric_lines = []
for line in text.splitlines():
    if re.search(r'(dota[/_][A-Za-z0-9_./-]+|mAP|AP50)', line):
        if any(token in line for token in ('dota/', 'dota_', 'mAP', 'AP50')):
            metric_lines.append(line.strip())
metric_raw = ' || '.join(metric_lines[-12:])
metric_raw = metric_raw.replace('|', '/')
print('|'.join([map_value, ap50_value, ap07_value, ap12_value, metric_raw]))
PY
}

write_md() {
  rtk "$PYTHON_BIN" - "$RESULT_CSV" "$RESULT_MD" "$OUT_ROOT" "$CONFIG" "$CHECKPOINT" "$GPUS" <<'PY'
import csv
import sys
from datetime import datetime
from pathlib import Path

csv_path, md_path, out_root, config, checkpoint, gpus = sys.argv[1:]
rows = []
path = Path(csv_path)
if path.exists():
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))

def esc(v):
    return str(v or '').replace('|', '\\|')

def code(v):
    return f'`{esc(v)}`' if v else ''

def fmt(v):
    if v in ('', 'NA', None):
        return 'NA'
    try:
        return f'{float(v):.4f}'
    except ValueError:
        return esc(v)

lines = [
    '# DOTA1 Angle Sweep ReDet Validation Evaluation',
    '',
    f'- generated_at: `{datetime.now().strftime("%F %T")}`',
    f'- out_root: `{out_root}`',
    f'- config: `{config}`',
    f'- checkpoint: `{checkpoint}`',
    f'- GPUs: `{gpus}`',
    f'- result_csv: `{csv_path}`',
    '',
]
if rows:
    ok = sum(1 for r in rows if r.get('status') == 'OK')
    fail = sum(1 for r in rows if r.get('status') != 'OK')
    lines += [
        '## Summary',
        '',
        '| total | OK | non-OK |',
        '|---:|---:|---:|',
        f'| {len(rows)} | {ok} | {fail} |',
        '',
        '## Metrics',
        '',
        '| angle | status | mAP | AP50 | AP07 mAP | AP12 mAP | predictions | log |',
        '|---:|---|---:|---:|---:|---:|---|---|',
    ]
    for r in rows:
        lines.append(
            f'| {esc(r.get("angle"))} | {esc(r.get("status"))} | '
            f'{fmt(r.get("mAP"))} | {fmt(r.get("AP50"))} | '
            f'{fmt(r.get("ap07_mAP"))} | {fmt(r.get("ap12_mAP"))} | '
            f'{code(r.get("prediction_path"))} | {code(r.get("log_path"))} |')
    lines.append('')
    lines.append('## Raw Metric Lines')
    lines.append('')
    lines.append('| angle | raw metrics from log |')
    lines.append('|---:|---|')
    for r in rows:
        lines.append(f'| {esc(r.get("angle"))} | {esc(r.get("metric_raw"))} |')
    lines.append('')
else:
    lines += ['No finished evaluations yet.', '']

Path(md_path).parent.mkdir(parents=True, exist_ok=True)
Path(md_path).write_text('\n'.join(lines) + '\n')
PY
}

main() {
  cd "$ROOT_DIR" || exit 1
  rtk /usr/bin/mkdir -p "$OUT_ROOT" "$(rtk /usr/bin/dirname "$RESULT_MD")"
  : > "$RUN_LOG"
  printf 'angle,status,mAP,AP50,ap07_mAP,ap12_mAP,prediction_path,log_path,metric_raw\n' > "$RESULT_CSV"

  assert_file "$CONFIG"
  assert_file "$CHECKPOINT"
  assert_dir "$ANGLE_ROOT"

  log "Starting DOTA1 12-angle ReDet eval"
  log "GPUS=$GPUS NPROC_PER_NODE=$NPROC_PER_NODE BATCH_SIZE=$BATCH_SIZE NUM_WORKERS=$NUM_WORKERS"
  log "CONFIG=$CONFIG"
  log "CHECKPOINT=$CHECKPOINT"
  log "ANGLE_ROOT=$ANGLE_ROOT"
  log "RESULT_MD=$RESULT_MD"
  write_md

  local idx=0
  local angle split_dir ann_dir img_dir run_dir log_file pred_file port
  local exit_code status parsed map ap50 ap07 ap12 metric_raw

  for angle in $ANGLES; do
    idx=$((idx + 1))
    split_dir="$ANGLE_ROOT/angle_$angle"
    ann_dir="$split_dir/annfiles"
    img_dir="$split_dir/images"
    assert_dir "$ann_dir"
    assert_dir "$img_dir"

    run_dir="$OUT_ROOT/angle_$angle"
    log_file="$run_dir/test.log"
    pred_file="$run_dir/predictions.pkl"
    port=$((MASTER_PORT_START + idx))
    rtk /usr/bin/mkdir -p "$run_dir"

    log "Evaluating angle_$angle -> $run_dir"
    (
      cd "$ROOT_DIR" || exit 1
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
        "$CONFIG" \
        "$CHECKPOINT" \
        --launcher pytorch \
        --work-dir "$run_dir" \
        --out "$pred_file" \
        --cfg-options \
        "test_dataloader.batch_size=$BATCH_SIZE" \
        "test_dataloader.num_workers=$NUM_WORKERS" \
        "test_dataloader.dataset.data_root=$DATA_ROOT" \
        "test_dataloader.dataset.ann_file=angle_sweep_val/realistic/angle_$angle/annfiles/" \
        "test_dataloader.dataset.data_prefix.img_path=angle_sweep_val/realistic/angle_$angle/images/"
    ) > "$log_file" 2>&1
    exit_code=$?

    if [[ "$exit_code" -eq 0 ]]; then
      status="OK"
    else
      status="FAIL($exit_code)"
    fi

    parsed="$(parse_metrics "$log_file")"
    IFS='|' read -r map ap50 ap07 ap12 metric_raw <<< "$parsed"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,"%s"\n' \
      "$angle" "$status" "$map" "$ap50" "$ap07" "$ap12" \
      "$pred_file" "$log_file" "$metric_raw" >> "$RESULT_CSV"
    write_md
    log "Finished angle_$angle status=$status mAP=$map AP50=$ap50"
  done

  write_md
  log "Finished all angles. Markdown: $RESULT_MD"
}

main "$@"
