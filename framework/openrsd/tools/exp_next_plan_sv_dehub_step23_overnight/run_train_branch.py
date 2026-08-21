#!/usr/bin/env python
"""Launch one SV-DeHub overnight training branch on a single GPU."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.exp_next_plan_sv_dehub_step23_overnight import common_overnight_utils as U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--branch', choices=['A', 'B', 'C'], required=True)
    ap.add_argument('--max-iters', type=int, default=None)
    ap.add_argument('--gpu', type=int, default=None)
    args = ap.parse_args()
    U.ensure_dirs()
    spec = U.BRANCHES[args.branch]
    work = spec['work']
    work.mkdir(parents=True, exist_ok=True)
    max_iters = args.max_iters or spec['max_iters']
    gpu = args.gpu if args.gpu is not None else spec['gpu']
    cfg = spec.get('config', U.DEBUB_CONFIG)
    log = U.RESULT_DIR / f"log_train_{args.branch}_{'iter1000_to8k' if args.branch == 'A' else 'fresh'}.txt"

    cfg_opts = [
        f'train_cfg.max_iters={max_iters}',
        'default_hooks.checkpoint.interval=1000',
        f'work_dir={work}',
    ]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env['PYTHONPATH'] = str(REPO) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    cmd = [
        str(U.PYTHON), str(REPO / 'tools/train.py'), str(cfg),
        '--work-dir', str(work), '--cfg-options', *cfg_opts,
    ]
    resume = spec.get('resume')
    if resume is not None:
        if not Path(resume).exists():
            raise FileNotFoundError(resume)
        cmd.extend(['--resume', str(resume)])
    log.write_text(f'# {datetime.now().isoformat()}\nCMD: {" ".join(cmd)}\n\n', encoding='utf-8')
    with open(log, 'a', encoding='utf-8') as lf:
        proc = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=lf, stderr=subprocess.STDOUT)
    print('branch', args.branch, 'exit', proc.returncode, 'log', log)
    return proc.returncode


if __name__ == '__main__':
    sys.exit(main() or 0)
