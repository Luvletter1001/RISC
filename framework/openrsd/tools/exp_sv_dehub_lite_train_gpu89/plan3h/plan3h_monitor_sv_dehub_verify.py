#!/usr/bin/env python
"""Append plan3h patrol row to monitor CSV."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import sys
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tools.exp_sv_dehub_lite_train_gpu89.plan3h import common_plan3h_utils as P

FIELDS = [
    'check_time', 'elapsed_minutes', 'gpu8_process_alive', 'gpu9_process_alive',
    'gpu8_memory_used', 'gpu9_memory_used', 'current_task_gpu8', 'current_task_gpu9',
    'latest_ablation_variant', 'latest_train_iter', 'latest_checkpoint',
    'latest_ap_eval_checkpoint', 'ap_eval_rows', 'ablation_eval_rows',
    'patched_baseline_status', 'latest_high_final_sv', 'latest_low_final_sv',
    'latest_AP50', 'latest_sv_AP50', 'detected_error', 'action_taken', 'next_action',
]


def gpu_mem(gpu_id: int) -> str:
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', f'--id={gpu_id}'],
            text=True, timeout=10)
        return out.strip().split('\n')[0]
    except Exception:
        return 'NA'


def proc_alive(gpu: int) -> str:
    try:
        out = subprocess.check_output(['pgrep', '-af', f'CUDA_VISIBLE_DEVICES={gpu}'], text=True)
        return 'yes' if out.strip() else 'no'
    except subprocess.CalledProcessError:
        return 'no'


def read_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding='utf-8') as f:
        return max(0, sum(1 for _ in f) - 1)


def latest_metric(path: Path, col: str, ckpt: str = '') -> str:
    if not path.exists():
        return 'NA'
    with open(path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if ckpt:
        rows = [r for r in rows if r.get('checkpoint') == ckpt]
    if not rows:
        return 'NA'
    return rows[-1].get(col, 'NA')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--elapsed-min', type=float, default=0)
    ap.add_argument('--task-gpu8', default='')
    ap.add_argument('--task-gpu9', default='')
    ap.add_argument('--error', default='')
    ap.add_argument('--action', default='record')
    ap.add_argument('--next', default='')
    args = ap.parse_args()
    P.ensure_dirs()
    t0_path = P.VERIFY_DIR / 'fmeta_plan3h_T0.txt'
    baseline_md = P.VERIFY_DIR / 'fres_00_patched_baseline_consistency.md'
    pb_status = 'NA'
    if baseline_md.exists():
        for line in baseline_md.read_text().splitlines():
            if 'verdict' in line and '**' in line:
                pb_status = line.split('**')[1] if '**' in line else 'NA'
                break
    row = dict(
        check_time=datetime.now().isoformat(timespec='seconds'),
        elapsed_minutes=args.elapsed_min,
        gpu8_process_alive=proc_alive(8),
        gpu9_process_alive=proc_alive(9),
        gpu8_memory_used=gpu_mem(8),
        gpu9_memory_used=gpu_mem(9),
        current_task_gpu8=args.task_gpu8,
        current_task_gpu9=args.task_gpu9,
        latest_ablation_variant='',
        latest_train_iter='',
        latest_checkpoint='',
        latest_ap_eval_checkpoint='',
        ap_eval_rows=read_csv_rows(P.VERIFY_DIR / 'ftable_01_full_ap_eval_summary.csv'),
        ablation_eval_rows=read_csv_rows(P.VERIFY_DIR / 'ftable_02_ablation_eval_summary.csv'),
        patched_baseline_status=pb_status,
        latest_high_final_sv=latest_metric(P.VERIFY_DIR / 'ftable_01_full_ap_eval_summary.csv', 'high_final_sv'),
        latest_low_final_sv=latest_metric(P.VERIFY_DIR / 'ftable_01_full_ap_eval_summary.csv', 'low_final_sv'),
        latest_AP50=latest_metric(P.VERIFY_DIR / 'ftable_01_full_ap_eval_summary.csv', 'ap50'),
        latest_sv_AP50=latest_metric(P.VERIFY_DIR / 'ftable_01_full_ap_eval_summary.csv', 'sv_ap50'),
        detected_error=args.error,
        action_taken=args.action,
        next_action=args.next,
    )
    mon = P.VERIFY_DIR / 'ftable_plan3h_monitor.csv'
    new = not mon.exists() or mon.stat().st_size == 0
    with open(mon, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow(row)
    (P.VERIFY_DIR / 'fres_plan3h_monitor.md').write_text(
        f'# Monitor @ {row["check_time"]}\n\n- elapsed: {args.elapsed_min} min\n- patched_baseline: {pb_status}\n- AP rows: {row["ap_eval_rows"]}\n',
        encoding='utf-8')
    print(json.dumps(row, indent=2))


if __name__ == '__main__':
    main()
