#!/usr/bin/env python
"""
Master runner: OpenRSD small-vehicle attractor causal verification (GPU 8/9).

Usage (preflight on GPU9):
  CUDA_VISIBLE_DEVICES=9 python tools/verify_sv_attractor_gpu89/run_verify_sv_attractor_gpu89.py \\
    --repo-root /data1/zcy/OpenRSD \\
    --work-dir /data1/zcy/OpenRSD/work_dirs/verify_sv_attractor_gpu89_YYYYMMDD_HHMMSS \\
    --gpu 9 --exp preflight --mode smoke
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

from tools.verify_sv_attractor_gpu89 import common as C
from tools.verify_sv_attractor_gpu89 import (
    exp_bg_gt,
    exp_dense,
    exp_intervention,
    exp_mapping,
    exp_postprocess,
    exp_preflight,
    exp_summary,
    exp_tile_angle,
    exp_visual_pack,
)


EXP_ALIASES = {
    'all': ['preflight', 'mapping', 'dense', 'intervention', 'bg_gt',
            'postprocess', 'tile_angle', 'visual_pack', 'summary'],
    'preflight': ['preflight'],
    'mapping': ['mapping'],
    'dense': ['dense'],
    'intervention': ['intervention'],
    'bg_gt': ['bg_gt'],
    'postprocess': ['postprocess'],
    'tile_angle': ['tile_angle'],
    'visual_pack': ['visual_pack'],
    'summary': ['summary'],
}

GPU8_EXPS = {'dense', 'postprocess', 'tile_angle'}
GPU9_EXPS = {'preflight', 'mapping', 'intervention', 'bg_gt',
             'tile_angle', 'visual_pack', 'summary'}


def parse_args():
    p = argparse.ArgumentParser(description='Verify SV attractor (GPU 8/9)')
    p.add_argument('--repo-root', default=str(REPO_ROOT))
    p.add_argument('--config', default=str(REPO_ROOT / C.DEFAULT_CONFIG))
    p.add_argument('--checkpoint', default=str(REPO_ROOT / C.DEFAULT_CHECKPOINT))
    p.add_argument('--support-pkl', default='')
    p.add_argument('--work-dir', default='')
    p.add_argument('--result-md-dir',
                   default=str(REPO_ROOT / 'resultmd/exp_verify_sv_attractor_gpu89'))
    p.add_argument('--gpu', type=int, choices=[8, 9], required=True)
    p.add_argument('--exp', default='preflight',
                   help='comma-separated or all')
    p.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    p.add_argument('--angles', default='')
    p.add_argument('--tiles', default=C.DEFAULT_TILE)
    p.add_argument('--force', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max-images', type=int, default=0)
    p.add_argument('--score-thr', type=float, default=0.01)
    p.add_argument('--nms-iou', type=float, default=0.5)
    p.add_argument('--topk', type=int, default=1000)
    return p.parse_args()


def resolve_experiments(exp_str: str) -> list:
    parts = []
    for token in exp_str.split(','):
        token = token.strip()
        if not token:
            continue
        parts.extend(EXP_ALIASES.get(token, [token]))
    seen = []
    for e in parts:
        if e not in seen:
            seen.append(e)
    return seen


def nvidia_smi() -> str:
    try:
        return subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader'],
            text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        return f'nvidia-smi failed: {exc}'


def log_command(ctx: C.RunContext, exp_name: str, cmd_desc: str,
                return_code: int, t0: float, t1: float,
                stdout_path: str = '', stderr_path: str = ''):
    entry = dict(
        experiment=exp_name,
        command=cmd_desc,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        start_time=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0)),
        end_time=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t1)),
        return_code=return_code,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        nvidia_smi_before=ctx.progress.get('_last_smi_before', ''),
        nvidia_smi_after=nvidia_smi(),
    )
    log_path = ctx.log_dir() / f'{exp_name}.json'
    log_path.write_text(json.dumps(entry, indent=2))
    ctx.progress.setdefault('command_log', []).append(str(log_path))
    ctx.save_progress()
    return entry


def build_context(args) -> C.RunContext:
    repo = Path(args.repo_root).resolve()
    if not args.work_dir:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        work_dir = repo / 'work_dirs' / f'verify_sv_attractor_gpu89_{ts}'
    else:
        work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    config = Path(args.config)
    if not config.is_absolute():
        config = repo / config
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = repo / checkpoint

    support = None
    support_source = ''
    if args.support_pkl:
        support = Path(args.support_pkl)
        if not support.is_absolute():
            support = repo / support
        support_source = 'cli'
    else:
        support, support_source = C.resolve_support_from_config(config)

    ctx = C.RunContext(
        repo_root=repo,
        config=config,
        checkpoint=checkpoint,
        support_pkl=support,
        work_dir=work_dir,
        result_md_dir=Path(args.result_md_dir).resolve(),
        gpu=args.gpu,
        mode=args.mode,
        angles=C.parse_angles(args.angles, args.mode),
        tiles=C.parse_tiles(args.tiles, repo),
        score_thr=args.score_thr,
        nms_iou=args.nms_iou,
        topk=args.topk,
        max_images=args.max_images,
        force=args.force,
        resume=args.resume,
    )
    ctx.load_progress()
    if args.force:
        ctx.progress.pop('blocked', None)
        ctx.progress.pop('blocked_reason', None)
    ctx.progress['support_source'] = support_source
    ctx.progress['gpu'] = args.gpu
    ctx.progress['mode'] = args.mode
    ctx.save_progress()
    return ctx


def run_experiment(ctx: C.RunContext, name: str, args) -> dict:
    allowed = GPU8_EXPS if ctx.gpu == 8 else GPU9_EXPS
    if name not in allowed and name != 'preflight':
        # preflight allowed on both; summary on 9
        if name == 'summary' and ctx.gpu != 9:
            return dict(status='SKIPPED', reason=f'summary should run on GPU9')
        if name not in allowed:
            return dict(status='SKIPPED',
                        reason=f'{name} not assigned to GPU{ctx.gpu}')

    if ctx.resume and ctx.step_done(name) and not ctx.force:
        print(f'SKIP {name} (already OK, use --force to rerun)')
        return dict(status='SKIPPED', reason='resume')

    ctx.progress['_last_smi_before'] = nvidia_smi()
    t0 = time.time()
    result = {}
    try:
        if name == 'preflight':
            result = exp_preflight.run(ctx)
        elif name == 'mapping':
            result = exp_mapping.run(ctx)
        elif name == 'dense':
            result = exp_dense.run(ctx)
        elif name == 'intervention':
            result = exp_intervention.run(ctx)
        elif name == 'bg_gt':
            result = exp_bg_gt.run(ctx)
        elif name == 'postprocess':
            result = exp_postprocess.run(ctx)
        elif name == 'tile_angle':
            result = exp_tile_angle.run_and_write_md(ctx, ctx.gpu)
        elif name == 'visual_pack':
            result = exp_visual_pack.run(ctx)
        elif name == 'summary':
            result = exp_summary.run(ctx)
        else:
            result = dict(status='FAILED', error=f'unknown exp {name}')
    except Exception as exc:
        result = dict(status='FAILED', error=str(exc))
        ctx.mark_step(name, 'FAILED', error=str(exc))
        fres = ctx.fres_path({
            'preflight': 'fres_00_preflight_inventory.md',
            'mapping': 'fres_01_mapping_sanity.md',
            'dense': 'fres_02_dense_logit_attractor.md',
            'intervention': 'fres_03_embedding_causal_intervention.md',
            'bg_gt': 'fres_04_background_gt_stratification.md',
            'postprocess': 'fres_05_postprocess_amplifier.md',
            'tile_angle': 'fres_06_tile_angle_generalization.md',
            'visual_pack': 'fres_07_visual_evidence_pack.md',
            'summary': 'fres_summary_sv_attractor_verdict.md',
        }.get(name, f'fres_{name}_FAILED.md'))
        with open(fres, 'w') as f:
            f.write(f'# {name} FAILED\n\n```\n{exc}\n```\n')

    t1 = time.time()
    log_command(ctx, name, f'python run_verify_sv_attractor_gpu89.py --exp {name}',
                0 if result.get('status') != 'FAILED' else 1, t0, t1)
    print(f'EXP {name} -> {result}')
    if ctx.blocked() and name not in ('preflight', 'mapping', 'summary'):
        print(f'BLOCKED after {name}: {ctx.progress.get("blocked_reason")}')
    return result


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    exps = resolve_experiments(args.exp)
    ctx = build_context(args)

    print(f'work_dir={ctx.work_dir}')
    print(f'result_md_dir={ctx.result_md_dir}')
    print(f'CUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]}')
    print(f'experiments={exps} mode={ctx.mode} angles={ctx.angles}')

    # Ordered pipeline
    order = ['preflight', 'mapping', 'dense', 'intervention', 'bg_gt',
             'postprocess', 'tile_angle', 'visual_pack', 'summary']
    for name in order:
        if name not in exps:
            continue
        if ctx.blocked() and name not in ('preflight', 'summary'):
            print(f'SKIP {name} — blocked: {ctx.progress.get("blocked_reason")}')
            continue
        run_experiment(ctx, name, args)

    print(f'DONE work_dir={ctx.work_dir}')


if __name__ == '__main__':
    main()
