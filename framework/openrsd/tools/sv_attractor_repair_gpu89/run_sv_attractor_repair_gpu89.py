#!/usr/bin/env python
"""
Master runner: small-vehicle attractor repair (GPU 8/9).

Use openrsd python:
  /data/zcy/anaconda3/envs/openrsd/bin/python tools/sv_attractor_repair_gpu89/run_sv_attractor_repair_gpu89.py ...
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.sv_attractor_repair_gpu89 import common as C
from tools.sv_attractor_repair_gpu89 import (
    exp_baseline,
    exp_combined,
    exp_eval,
    exp_full_ready,
    exp_heldout_eval,
    exp_preflight,
    exp_r1,
    exp_r2,
    exp_r2b_eval,
    exp_r3,
    exp_summary,
    exp_visual,
)

EXP_MAP = {
    'preflight': exp_preflight,
    'baseline': exp_baseline,
    'r1_calib': exp_r1,
    'r2_negative': exp_r2,
    'r3_antihub': exp_r3,
    'combined': exp_combined,
    'eval': exp_eval,
    'heldout_eval': exp_heldout_eval,
    'r2b_eval': exp_r2b_eval,
    'full_ready': exp_full_ready,
    'visual': exp_visual,
    'summary': exp_summary,
}

GPU8 = {'preflight', 'baseline', 'r1_calib', 'r2_negative', 'heldout_eval'}
GPU9 = {'preflight', 'r3_antihub', 'r2b_eval', 'combined', 'eval', 'heldout_eval',
        'full_ready', 'visual', 'summary'}


def parse_exps(s: str) -> list:
    alias = {
        'all': list(EXP_MAP.keys()),
        'r1': ['r1_calib'],
        'r2': ['r2_negative'],
        'r3': ['r3_antihub'],
    }
    out = []
    for t in s.split(','):
        t = t.strip()
        if not t:
            continue
        out.extend(alias.get(t, [t]))
    seen = []
    for e in out:
        if e not in seen:
            seen.append(e)
    return seen


def nvidia_smi() -> str:
    try:
        return subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader'], text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        return str(exc)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--repo-root', default=str(REPO_ROOT))
    p.add_argument('--config', default=str(REPO_ROOT / C.verify.DEFAULT_CONFIG))
    p.add_argument('--checkpoint', default=str(REPO_ROOT / C.verify.DEFAULT_CHECKPOINT))
    p.add_argument('--support-pkl',
                   default=str(REPO_ROOT / 'data/DOTA1_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl'))
    p.add_argument('--verify-work-dir',
                   default=str(REPO_ROOT / 'work_dirs/verify_sv_attractor_gpu89_20260519_113039'))
    p.add_argument('--work-dir', default='')
    p.add_argument('--result-md-dir', default=str(REPO_ROOT / 'resultmd/exp_sv_attractor_repair_gpu89'))
    p.add_argument('--gpu', type=int, choices=[8, 9], required=True)
    p.add_argument('--exp', default='preflight')
    p.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    p.add_argument('--train-max-images', type=int, default=128)
    p.add_argument('--heldout-max-images', type=int, default=64)
    p.add_argument('--eval-max-images', type=int, default=64)
    p.add_argument('--angles', default='')
    p.add_argument('--tiles', default=C.DEFAULT_TILE)
    p.add_argument('--iters', type=int, default=300)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--force', action='store_true')
    p.add_argument('--resume', action='store_true')
    return p.parse_args()


def build_ctx(args) -> C.RepairContext:
    repo = Path(args.repo_root).resolve()
    work = Path(args.work_dir) if args.work_dir else (
        repo / 'work_dirs' / f'sv_attractor_repair_gpu89_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    work.mkdir(parents=True, exist_ok=True)
    angles = ([int(a) for a in args.angles.split(',') if a.strip()]
              if args.angles.strip() else (C.FULL_ANGLES if args.mode == 'full' else C.SMOKE_ANGLES))
    ctx = C.RepairContext(
        repo_root=repo,
        config=Path(args.config),
        checkpoint=Path(args.checkpoint),
        support_pkl=Path(args.support_pkl),
        verify_work_dir=Path(args.verify_work_dir),
        work_dir=work,
        result_md_dir=Path(args.result_md_dir).resolve(),
        gpu=args.gpu,
        mode=args.mode,
        angles=angles,
        tiles=C.verify.parse_tiles(args.tiles, repo),
        train_max_images=args.train_max_images,
        heldout_max_images=args.heldout_max_images,
        eval_max_images=args.eval_max_images,
        iters=args.iters if args.mode == 'full' else min(args.iters, 300),
        batch_size=args.batch_size,
        lr=args.lr,
        force=args.force,
        resume=args.resume,
    )
    ctx.load_progress()
    if args.force:
        ctx.progress.pop('blocked', None)
        ctx.progress.pop('blocked_reason', None)
    ctx.progress['gpu'] = args.gpu
    ctx.progress['mode'] = args.mode
    ctx.save_progress()
    return ctx


def log_run(ctx: C.RepairContext, name: str, rc: int, t0: float, t1: float):
    entry = dict(
        experiment=name,
        command=f'run_sv_attractor_repair_gpu89.py --exp {name}',
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        start_time=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0)),
        end_time=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t1)),
        return_code=rc,
        nvidia_smi_before=ctx.progress.get('_smi_before', ''),
        nvidia_smi_after=nvidia_smi(),
    )
    p = ctx.log_dir() / f'{name}.json'
    p.write_text(json.dumps(entry, indent=2))
    ctx.progress.setdefault('command_log', []).append(str(p))
    ctx.save_progress()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    exps = parse_exps(args.exp)
    ctx = build_ctx(args)
    allowed = GPU8 if ctx.gpu == 8 else GPU9

    print(f'work_dir={ctx.work_dir}')
    print(f'gpu={ctx.gpu} exps={exps} mode={ctx.mode}')

    order = ['preflight', 'baseline', 'r1_calib', 'r2_negative', 'r3_antihub', 'r2b_eval',
             'combined', 'heldout_eval', 'eval', 'full_ready', 'visual', 'summary']
    for name in order:
        if name not in exps:
            continue
        if name not in allowed and name != 'preflight':
            print(f'SKIP {name} (GPU{ctx.gpu})')
            continue
        if ctx.blocked() and name not in ('preflight', 'summary'):
            print(f'SKIP {name} blocked')
            continue
        if ctx.resume and ctx.step_ok(name) and not ctx.force:
            print(f'SKIP {name} done')
            continue
        ctx.progress['_smi_before'] = nvidia_smi()
        t0 = time.time()
        try:
            mod = EXP_MAP[name]
            result = mod.run(ctx)
            rc = 0 if result.get('status') != 'FAILED' else 1
        except Exception as exc:
            result = dict(status='FAILED', error=str(exc))
            rc = 1
            fres = {
                'preflight': 'fres_00_repair_preflight.md',
                'baseline': 'fres_01_baseline_reconfirm.md',
                'r1_calib': 'fres_02_classwise_calibration.md',
                'r2_negative': 'fres_03_negative_background_prototypes.md',
                'r3_antihub': 'fres_04_antihub_projection_adapter.md',
                'combined': 'fres_05_combined_repair_ablation.md',
                'eval': 'fres_06_eval_angle_cross_tile_ap.md',
                'visual': 'fres_07_visual_failure_success_pack.md',
                'summary': 'fres_summary_repair_verdict.md',
            }.get(name, f'fres_{name}_FAILED.md')
            with open(ctx.fres(fres), 'w') as f:
                f.write(f'# {name} FAILED\n\n```\n{exc}\n```\n')
            ctx.mark(name, 'FAILED', error=str(exc))
        log_run(ctx, name, rc, t0, time.time())
        print(f'EXP {name} -> {result}')

    print(f'DONE {ctx.work_dir}')


if __name__ == '__main__':
    main()
