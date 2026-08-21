#!/usr/bin/env python
"""45–60 min overnight monitor; append ftable_00_monitor.csv."""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.exp_next_plan_sv_dehub_step23_overnight import common_overnight_utils as U


def gpu_status() -> str:
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
             '--format=csv,noheader'], text=True, timeout=30)
        return '; '.join(out.strip().splitlines())
    except Exception as exc:
        return str(exc)


def latest_iter(work: Path) -> str:
    best = 0
    for p in work.glob('iter_*.pth'):
        try:
            best = max(best, int(p.stem.split('_')[-1]))
        except ValueError:
            pass
    return str(best)


def running_pids() -> str:
    try:
        out = subprocess.check_output(
            ['pgrep', '-af', 'exp_next_plan_sv_dehub_step23_overnight'], text=True, timeout=10)
        return out.strip()[:500]
    except subprocess.CalledProcessError:
        return ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--interval-min', type=int, default=50)
    args = ap.parse_args()
    U.ensure_dirs()
    mon_csv = U.RESULT_DIR / 'ftable_00_monitor.csv'
    mon_md = U.RESULT_DIR / 'fres_00_monitor.md'

    def tick():
        row = dict(
            check_time=datetime.now().isoformat(timespec='seconds'),
            gpu_status=gpu_status(),
            running_tasks=running_pids(),
            latest_train_iter_A=latest_iter(U.BRANCHES['A']['work']),
            latest_train_iter_B=latest_iter(U.BRANCHES['B']['work']),
            latest_train_iter_C=latest_iter(U.BRANCHES['C']['work']),
            latest_ckpt_A=str(list(U.BRANCHES['A']['work'].glob('iter_*.pth'))[-1:])
            if list(U.BRANCHES['A']['work'].glob('iter_*.pth')) else 'NA',
            latest_ckpt_B=str(list(U.BRANCHES['B']['work'].glob('iter_*.pth'))[-1:])
            if list(U.BRANCHES['B']['work'].glob('iter_*.pth')) else 'NA',
            latest_ckpt_C=str(list(U.BRANCHES['C']['work'].glob('iter_*.pth'))[-1:])
            if list(U.BRANCHES['C']['work'].glob('iter_*.pth')) else 'NA',
            action_taken='monitor_tick',
            next_action='continue',
        )
        U.append_csv(mon_csv, row, U.SCHEMA_MONITOR)
        mon_md.write_text(f'# Monitor\n\n- last: {row["check_time"]}\n\n```\n{row["gpu_status"]}\n```\n',
                          encoding='utf-8')
        print('monitor', row['check_time'], 'A', row['latest_train_iter_A'],
              'B', row['latest_train_iter_B'])

    tick()
    if args.once:
        return 0
    import time
    while True:
        time.sleep(max(60, args.interval_min * 60))
        tick()


if __name__ == '__main__':
    sys.exit(main() or 0)
