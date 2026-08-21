#!/usr/bin/env python
"""Experiment B: minimal ablation trains (dehub-only, paste-only)."""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.exp_sv_dehub_lite_train_gpu89.verify import common_verify_utils as V

VARIANTS = [
    ('dehub_only', REPO / 'M_configs/experiments/sv_dehub_ablation_dehub_only_v1.py',
     REPO / 'work_dirs/exp_sv_dehub_lite_train_gpu89/verify_ablation/dehub_only'),
    ('paste_only', REPO / 'M_configs/experiments/sv_dehub_ablation_paste_only_v1.py',
     REPO / 'work_dirs/exp_sv_dehub_lite_train_gpu89/verify_ablation/paste_only'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-iters', type=int, default=1000)
    ap.add_argument('--gpu', type=int, default=8)
    args = ap.parse_args()
    V.ensure_dirs()
    log = V.VERIFY_DIR / 'log_gpu8_ablation_train.txt'
    loss_csv = V.VERIFY_DIR / 'ftable_ablation_train_loss.csv'
    loss_csv.write_text(
        'variant,iter,total_loss,loss_cls,loss_bbox,loss_dehub,dehub_ratio,lr,status\n',
        encoding='utf-8',
    )

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    env['PYTHONPATH'] = str(REPO) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    py = str(V.PYTHON)

    with open(log, 'a', encoding='utf-8') as logf:
        for variant, cfg, work_dir in VARIANTS:
            work_dir.mkdir(parents=True, exist_ok=True)
            hook_csv = V.VERIFY_DIR / f'ftable_ablation_{variant}_loss.csv'
            hook_csv.write_text(
                'iter,total_loss,loss_cls,loss_bbox,loss_dehub,dehub_ratio,lr,status\n',
                encoding='utf-8',
            )
            cmd = [
                py, str(REPO / 'tools/train.py'), str(cfg),
                '--work-dir', str(work_dir),
                '--cfg-options',
                f'train_cfg.max_iters={args.max_iters}',
                'train_cfg.type=IterBasedTrainLoop',
                'train_cfg.val_interval=1001',
                f'default_hooks.checkpoint.interval={args.max_iters}',
                'default_hooks.checkpoint.by_epoch=False',
                f'custom_hooks.0.log_csv={hook_csv}',
                f'load_from={V.BASE_CKPT}',
            ]
            logf.write(f'\n=== {variant} {datetime.now().isoformat()} ===\n')
            logf.write(' '.join(cmd) + '\n')
            logf.flush()
            proc = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=logf, stderr=subprocess.STDOUT)
            logf.write(f'exit_code={proc.returncode}\n')
            if proc.returncode != 0:
                print(f'FAIL {variant} rc={proc.returncode}')
                return proc.returncode
            with open(loss_csv, 'a', encoding='utf-8') as out, open(hook_csv, encoding='utf-8') as inp:
                r = csv.DictReader(inp)
                for row in r:
                    out.write(f"{variant},{row.get('iter','')},{row.get('total_loss','')},"
                              f"{row.get('loss_cls','')},{row.get('loss_bbox','')},"
                              f"{row.get('loss_dehub','')},{row.get('dehub_ratio','')},"
                              f"{row.get('lr','')},{row.get('status','')}\n")
            print(f'OK {variant}')
    print('ablation train done')
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
