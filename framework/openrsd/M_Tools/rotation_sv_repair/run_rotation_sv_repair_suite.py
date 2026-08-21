#!/usr/bin/env python3
"""
Rotation-aware Open-Vocabulary Teacher-Student Repair Suite (OC-OVD / ROVA).

Unified entry for dryrun / smoke / full repair experiments on rotation-induced SV drift.

Recommended Python: /data/zcy/anaconda3/envs/openrsd/bin/python
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'tools'))

from M_Tools.rotation_sv_repair.common import SuiteContext, setup_env, write_failed_fres
from M_Tools.rotation_sv_repair.discovery import run_dryrun, run_hooks, run_inventory
from M_Tools.rotation_sv_repair.experiments import exp_runner as ER
from M_Tools.rotation_sv_repair.tile_splits import run_splits

DEFAULT_WORK = REPO_ROOT / 'work_dirs/exp_rotation_sv_repair_20260524'
DEFAULT_RESULT = REPO_ROOT / 'resultmd/exp_rotation_sv_repair_20260524'

EXP_ORDER = [
    ('dryrun', lambda ctx: run_dryrun(ctx)),
    ('inventory', lambda ctx: run_inventory(ctx)),
    ('hooks', lambda ctx: run_hooks(ctx)),
    ('tests', lambda ctx: ER.run_unit_tests(ctx)),
    ('splits', lambda ctx: run_splits(ctx)),
    ('baseline', lambda ctx: ER.run_baseline(ctx)),
    ('postprocess', lambda ctx: ER.run_postprocess(ctx)),
    ('embedding', lambda ctx: ER.run_embedding(ctx)),
    ('prompt', lambda ctx: ER.run_prompt(ctx)),
    ('vlm_encoder', lambda ctx: ER.run_vlm_encoder_compare(ctx)),
    # OpenRSD head teacher and CastDet-style queue are submodes inside this step.
    ('ovd_teacher_student', lambda ctx: ER.run_ovd(ctx)),
    ('orbit', lambda ctx: ER.run_orbit(ctx)),
    ('adapter', lambda ctx: ER.run_adapter(ctx)),
    ('official_ap', lambda ctx: ER.run_official_ap(ctx)),
    ('true_sv', lambda ctx: ER.run_true_sv(ctx)),
    ('oracle', lambda ctx: ER.run_oracle(ctx)),
    ('ablation', lambda ctx: ER.run_ablation(ctx)),
    ('stats', lambda ctx: ER.run_stats(ctx)),
    ('audit', lambda ctx: ER.run_audit(ctx)),
    ('verdict', lambda ctx: ER.run_verdict(ctx)),
]

EXP_ALIASES = {
    'all': [e[0] for e in EXP_ORDER],
    'head_teacher': ['hooks', 'ovd_teacher_student'],
    'castdet': ['ovd_teacher_student'],
    'distill': ['orbit', 'adapter', 'ovd_teacher_student'],
    'vlm': ['inventory', 'prompt', 'vlm_encoder'],
    'makeup': ['inventory', 'splits', 'baseline', 'prompt', 'vlm_encoder', 'ovd_teacher_student', 'adapter', 'verdict'],
    'real_repair': ['inventory', 'splits', 'baseline', 'postprocess', 'embedding', 'official_ap', 'verdict'],
    'paper_method': [
        'splits', 'baseline', 'adapter', 'orbit', 'ovd_teacher_student', 'official_ap', 'verdict',
    ],
    'paper_method_retry': ['adapter', 'orbit', 'ovd_teacher_student', 'verdict'],
    'official_ap_repair': ['official_ap'],
    'official_ap_paper': ['official_ap'],
}


def parse_args():
    p = argparse.ArgumentParser(description='Rotation SV Repair Suite')
    p.add_argument('--repo-root', default=str(REPO_ROOT))
    p.add_argument('--work-dir', default=str(DEFAULT_WORK))
    p.add_argument('--result-dir', default=str(DEFAULT_RESULT))
    p.add_argument('--gpu-ids', default='8,9')
    p.add_argument('--mode', choices=['dryrun', 'smoke', 'full', 'debug'], default='dryrun')
    p.add_argument('--exp', default='all')
    p.add_argument('--teacher', default='ensemble')
    p.add_argument('--student', default='adapter')
    p.add_argument('--pseudo-label-mode', default='dynamic_queue')
    p.add_argument('--vocab-mode', default='dynamic_vocabulary')
    p.add_argument('--teacher-vlm', default='available_auto')
    p.add_argument('--only-model', default='')
    p.add_argument('--only-checkpoint', default='')
    p.add_argument('--only-config', default='')
    p.add_argument('--only-angle', default='')
    p.add_argument('--only-tile', default='')
    p.add_argument('--max-images', type=int, default=0)
    p.add_argument('--force', action='store_true')
    p.add_argument('--resume', action='store_true', default=True)
    p.add_argument('--no-resume', dest='resume', action='store_false')
    p.add_argument('--no-train', action='store_true')
    p.add_argument('--train-adapter', action='store_true')
    p.add_argument('--distill', action='store_true')
    p.add_argument('--debug-small', action='store_true')
    p.add_argument('--batch-size', type=int, default=0,
                   help='Inference batch size (0=auto: 8 full, 4 smoke, 2 debug)')
    p.add_argument('--num-workers', type=int, default=0,
                   help='Dataloader workers (0=auto: 4 full, 2 smoke)')
    return p.parse_args()


def resolve_exps(exp_str: str) -> list:
    parts = [x.strip() for x in exp_str.split(',') if x.strip()]
    out = []
    for p in parts:
        if p in EXP_ALIASES:
            out.extend(EXP_ALIASES[p])
        else:
            out.append(p)
    if 'all' in out:
        return [e[0] for e in EXP_ORDER]
    seen = []
    for e in out:
        if e not in seen:
            seen.append(e)
    return seen


def ensure_dirs(ctx: SuiteContext):
    for d in (ctx.work_dir, ctx.result_dir, ctx.tables_dir, ctx.figures_dir,
              ctx.human_audit_dir, ctx.logs_dir, ctx.adapter_ckpt_dir, ctx.pseudo_label_dir):
        d.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    # Prefer openrsd env when launched with system python lacking MKL/torch deps.
    if 'openrsd' not in sys.executable:
        openrsd_py = Path('/data/zcy/anaconda3/envs/openrsd/bin/python')
        if openrsd_py.exists() and os.environ.get('ROTATION_SV_REPAIR_NO_REEXEC') != '1':
            os.environ['ROTATION_SV_REPAIR_NO_REEXEC'] = '1'
            os.execv(str(openrsd_py), [str(openrsd_py)] + sys.argv)
    setup_env(args.gpu_ids)
    if args.batch_size <= 0:
        args.batch_size = 2 if args.mode == 'debug' else (4 if args.mode == 'smoke' else 8)
    if args.num_workers <= 0:
        args.num_workers = 2 if args.mode in ('debug', 'smoke') else 4
    ctx = SuiteContext(
        repo_root=Path(args.repo_root).resolve(),
        work_dir=Path(args.work_dir).resolve(),
        result_dir=Path(args.result_dir).resolve(),
        gpu_ids=args.gpu_ids,
        mode=args.mode,
        exp=args.exp,
        teacher=args.teacher,
        student=args.student,
        pseudo_label_mode=args.pseudo_label_mode,
        vocab_mode=args.vocab_mode,
        teacher_vlm=args.teacher_vlm,
        only_model=args.only_model,
        only_checkpoint=args.only_checkpoint,
        only_config=args.only_config,
        only_angle=args.only_angle,
        only_tile=args.only_tile,
        max_images=args.max_images,
        force=args.force,
        resume=args.resume,
        no_train=args.no_train,
        train_adapter=args.train_adapter,
        distill=args.distill,
        debug_small=args.debug_small,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    ensure_dirs(ctx)
    ctx.load_progress()
    ctx.progress['started_at'] = datetime.now().isoformat(timespec='seconds')
    ctx.progress['mode'] = ctx.mode
    ctx.save_progress()

    log_main = ctx.logs_dir / f'suite_{ctx.mode}_{datetime.now().strftime("%H%M%S")}.log'
    exps = resolve_exps(args.exp)
    if ctx.mode == 'dryrun':
        exps = ['dryrun', 'inventory', 'hooks', 'tests', 'splits'] if 'all' in args.exp else exps

    _exp_tokens = [x.strip() for x in args.exp.split(',')]
    if any(x in _exp_tokens for x in ('paper_method', 'paper_method_retry', 'official_ap_paper')):
        ctx.progress['paper_method_real'] = True
        ctx.save_progress()
    elif 'official_ap_repair' in _exp_tokens:
        ctx.progress.pop('paper_method_real', None)
        ctx.save_progress()

    results = {}
    blocked = False
    for name, fn in EXP_ORDER:
        if name not in exps:
            continue
        print(f'[{datetime.now().strftime("%H:%M:%S")}] >>> exp={name} mode={ctx.mode}')
        try:
            if blocked and name not in ('verdict', 'dryrun'):
                results[name] = dict(status='BLOCKED', reason='prior_critical_failure')
                continue
            res = fn(ctx)
            results[name] = res
            if res.get('status') == 'FAILED' and name == 'tests' and ctx.mode == 'full':
                blocked = True
                ctx.progress['blocked'] = True
                ctx.save_progress()
        except Exception as exc:
            results[name] = dict(status='FAILED', error=str(exc))
            fres_map = {
                'baseline': 'fres_010_baseline_reconfirm.md',
                'postprocess': 'fres_020_postprocess_repair.md',
            }
            if name in fres_map:
                write_failed_fres(ctx, fres_map[name], name, exc)
            traceback.print_exc()
            if name in ('tests', 'baseline') and ctx.mode == 'full':
                blocked = True

    summary = '\n'.join(f'{k}: {v.get("status", v)}' for k, v in results.items())
    log_main.write_text(summary, encoding='utf-8')
    print(summary)
    return 0 if all(v.get('status') != 'FAILED' for v in results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
