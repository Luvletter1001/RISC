#!/usr/bin/env python
"""Train SV-DeHub-Lite v1 on GPU8."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.exp_sv_dehub_lite_train_gpu89 import common_dehub_utils as U


def write_train_report(result_dir: Path, work_dir: Path, max_iters: int, cmd: list):
    lines = [
        '# Train SV-DeHub-Lite v1',
        '',
        f'- started: {datetime.now().isoformat()}',
        f'- config: `{U.DEBUB_CONFIG}`',
        f'- work_dir: `{work_dir}`',
        f'- max_iters: {max_iters}',
        f'- load_from: `{U.BASE_CKPT}`',
        '',
        '## Code changes',
        '- `M_AD/models/losses/sv_dehub_loss.py` — BackgroundSVDeHubLoss',
        '- `Flex_Rrtmdet_head_v3_1.py` — optional loss_dehub in loss_by_feat',
        '- `hard_negative_paste.py` — HardNegativeBackgroundPaste transform',
        '- `sv_dehub_train_hook.py` — loss logging / weight cap',
        '',
        '## Frozen',
        '- backbone, neck (config frozen_parameters)',
        '',
        f'## Command\n```bash\n{" ".join(cmd)}\n```',
    ]
    (result_dir / 'fres_train_sv_dehub_lite_v1.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-dir', type=Path, default=REPO)
    ap.add_argument('--result-dir', type=Path, default=U.RESULT_MD)
    ap.add_argument('--work-dir', type=Path, default=U.WORK_ROOT / 'train_v1')
    ap.add_argument('--max-iters', type=int, default=3000)
    ap.add_argument('--save-every', type=int, default=1000)
    ap.add_argument('--max-images', type=int, default=400)
    ap.add_argument('--gpu', type=int, default=8)
    ap.add_argument('--resume-from', type=Path, default=None,
                    help='Resume from checkpoint (.pth); passes --resume to train.py')
    args = ap.parse_args()
    U.ensure_dirs()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.result_dir / 'log_gpu8_train_sv_dehub_lite_v1.txt'

    cfg_opts = [
        f'train_cfg.max_iters={args.max_iters}',
        f'default_hooks.checkpoint.interval={args.save_every}',
        f'work_dir={args.work_dir}',
    ]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    env['PYTHONPATH'] = str(args.project_dir) + (
        os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    cmd = [
        str(U.PYTHON), str(REPO / 'tools/train.py'),
        str(U.DEBUB_CONFIG),
        '--work-dir', str(args.work_dir),
        '--cfg-options', *cfg_opts,
    ]
    if args.resume_from is not None:
        if not args.resume_from.exists():
            raise FileNotFoundError(f'resume checkpoint missing: {args.resume_from}')
        cmd.extend(['--resume', str(args.resume_from)])
    write_train_report(args.result_dir, args.work_dir, args.max_iters, cmd)
    log_path.write_text('CMD: ' + ' '.join(cmd) + '\n\n', encoding='utf-8')
    with open(log_path, 'a', encoding='utf-8') as logf:
        proc = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=logf, stderr=subprocess.STDOUT)
    try:
        from tools.exp_sv_dehub_lite_train_gpu89.debug_log import dbg
        dbg('A', 'run_train_sv_dehub_lite_gpu8.py:main', 'train subprocess finished',
            {'exit_code': proc.returncode, 'max_iters': args.max_iters}, run_id='post-fix')
    except Exception:
        pass
    print('train exit', proc.returncode, 'log', log_path)
    return proc.returncode


if __name__ == '__main__':
    sys.exit(main() or 0)
