#!/usr/bin/env python3
"""Overnight controller: encoder swap semantic drift experiments until 09:00."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / 'tools') not in sys.path:
    sys.path.insert(0, str(REPO / 'tools'))

from tools.encoder_swap_semantic_drift.common import (
    EncoderSwapCtx, bind_gpu, compute_stop_time, save_state, select_gpus, snapshot_env,
)
from tools.encoder_swap_semantic_drift.experiments.queue import EXPERIMENT_QUEUE
from tools.encoder_swap_semantic_drift.reporting import finalize_exp, update_live_status


def build_final_summary(ctx: EncoderSwapCtx, state: dict):
    md = ctx.result_md_dir / 'fres_encoder_swap_overnight_summary.md'
    js = ctx.result_md_dir / 'fjson_encoder_swap_overnight_summary.json'
    inv = read_inventory(ctx)
    p0148 = [r for r in state.get('latest_results', []) if 'P0148' in str(r)]
    lines = [
        '# Encoder swap overnight summary',
        '',
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        '',
        '## Timeline',
        f"- Start: {state.get('start_time')}",
        f"- Target stop: {state.get('target_stop_time')}",
        f"- Completed: {', '.join(state.get('completed_experiments', [])) or 'none'}",
        '',
        '## Encoder inventory',
        inv or '_see ftable_00_',
        '',
        '## P0148 highlights',
        '\n'.join(f'- {x}' for x in p0148[:20]) or '- see exp_03/04 csv',
        '',
        '## Answers (preliminary)',
        '1. Text encoder swap: see exp_03 dense_sv/final_sv vs original.',
        '2. Image support encoder swap: see exp_04.',
        '3. RemoteCLIP/GeoRSCLIP vs CLIP: compare sv_hubness in exp_01/02.',
        '4. DINOv2 image prototypes: exp_02/04 pairing rows.',
        '5. B8k complementarity: compare B8k vs baseline rows in exp_03.',
        '',
        '**Scope reminder:** All swaps are prompt/support embeddings, not detector backbone.',
        '',
        f"**Warnings:** {state.get('last_error', '')[:300]}",
    ]
    md.write_text('\n'.join(lines), encoding='utf-8')
    payload = dict(
        completed_experiments=state.get('completed_experiments', []),
        failed_experiments=state.get('failed_experiments', []),
        blocked_experiments=state.get('blocked_experiments', []),
        encoder_inventory=inv,
        p0148_results=p0148,
        hypothesis_updates=state.get('latest_hypothesis_update', ''),
        next_steps=state.get('next_action', ''),
        critical_warnings=['prompt_encoder_only_not_backbone'],
    )
    js.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def read_inventory(ctx: EncoderSwapCtx) -> str:
    p = ctx.result_md_dir / 'exp_00_encoder_inventory' / 'ftable_00_pretrained_inventory.csv'
    if not p.exists():
        return ''
    return p.read_text(encoding='utf-8')[:2000]


def _run_only_exps(ctx, gpu, encode_device, gpus, only_exps, args):
    """Run a fixed list of experiments once (daytime follow-up)."""
    fn_map = {exp_id: fn for exp_id, fn in EXPERIMENT_QUEUE}
    rc = 0
    for exp_id in only_exps:
        fn = fn_map.get(exp_id)
        if fn is None:
            print(f'UNKNOWN exp: {exp_id}')
            rc = 1
            continue
        t0 = time.time()
        print(f'[{datetime.now().strftime("%H:%M:%S")}] START {exp_id} gpu={gpu} mode={ctx.mode}')
        try:
            if args.mode == 'dryrun':
                from tools.encoder_swap_semantic_drift.common import ExpResult
                res = ExpResult('COMPLETE', exp_id, exp_id.replace('exp_', 'exp_') + '_dryrun')
            else:
                res = fn(ctx, gpu, encode_device)
        except Exception:
            err = traceback.format_exc()
            print(f'FAILED {exp_id}:\n{err}')
            rc = 1
            continue
        finalize_exp(
            ctx, res, gpu, t0,
            command=f'run_encoder_swap_overnight.py --only-exps {exp_id}',
            why=f'Daytime follow-up: {exp_id}',
            setup=f'gpu={gpu} smoke={ctx.smoke} gpus={gpus}',
            sanity=f'status={res.status} rows={len(res.rows)}',
            interpretation=res.interpretation,
            guesses=res.next_guesses or [],
            decision='single-shot follow-up',
        )
        print(f'DONE {exp_id} -> {res.status}')
        if res.status in ('FAILED',):
            rc = 1
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-root', type=Path, default=REPO)
    ap.add_argument('--pretrained-root', type=Path, default=REPO / 'pretrained')
    ap.add_argument('--work-dir', type=Path, required=True)
    ap.add_argument('--result-md-dir', type=Path, required=True)
    ap.add_argument('--gpu-ids', default='auto')
    ap.add_argument('--run-until', default='09:00')
    ap.add_argument('--min-hours', type=float, default=10.0)
    ap.add_argument('--mode', default='full', choices=['dryrun', 'smoke', 'full', 'debug', 'resume'])
    ap.add_argument('--max-retry', type=int, default=2)
    ap.add_argument('--force', action='store_true')
    ap.add_argument(
        '--only-exps',
        default='',
        help='Comma-separated exp ids to run once and exit (e.g. exp_03,exp_05,exp_07)',
    )
    ap.add_argument('--batch-size', type=int, default=12, help='Tile angle inference batch size')
    ap.add_argument('--ap-batch-size', type=int, default=8, help='Held-out AP eval batch size')
    ap.add_argument(
        '--allow-autonomous-next',
        nargs='?',
        const=True,
        default=True,
        type=lambda v: str(v).lower() not in ('0', 'false', 'no', 'off'),
        help='Keep autonomous exp_09 loop until stop time (accepts true/false)',
    )
    ap.add_argument('--python', type=Path, default=Path('/data/zcy/anaconda3/envs/openrsd/bin/python'))
    args = ap.parse_args()

    gpus = select_gpus(args.gpu_ids)
    ctx = EncoderSwapCtx(
        repo_root=args.repo_root,
        work_dir=args.work_dir,
        result_md_dir=args.result_md_dir,
        pretrained_root=args.pretrained_root,
        python=args.python,
        mode=args.mode,
        force=args.force,
        smoke=(args.mode == 'smoke'),
        batch_size=args.batch_size,
        ap_batch_size=args.ap_batch_size,
        gpu_ids=gpus,
    )
    gpu = gpus[0]
    encode_device = bind_gpu(gpu)
    target_stop, time_note = compute_stop_time(args.run_until, args.min_hours)

    state = dict(
        start_time=datetime.now().isoformat(),
        target_stop_time=target_stop.isoformat(),
        time_note=time_note,
        selected_gpus=gpus,
        encode_device=encode_device,
        mode=args.mode,
        smoke=ctx.smoke,
        completed_experiments=[],
        failed_experiments=[],
        blocked_experiments=[],
        next_action='exp_00',
        last_error='',
        latest_hypothesis_update='SV attractor may be text-geometry and/or visual-prototype geometry.',
        latest_results=[],
    )
    if args.mode == 'resume' and ctx.state_path.exists() and not args.force:
        state.update(json.loads(ctx.state_path.read_text(encoding='utf-8')))

    only_exps = [x.strip() for x in args.only_exps.split(',') if x.strip()]
    if only_exps:
        return _run_only_exps(ctx, gpu, encode_device, gpus, only_exps, args)

    env_log = ctx.log_dir / f"startup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    snapshot_env(ctx.repo_root, env_log)

    done = set(state.get('completed_experiments', []))
    cycle = 0
    while datetime.now() < target_stop:
        cycle += 1
        progressed = False
        for exp_id, fn in EXPERIMENT_QUEUE:
            if exp_id in done and not args.force:
                continue
            state['running_experiment'] = exp_id
            state['next_action'] = exp_id
            save_state(ctx, state)
            update_live_status(ctx, state)
            t0 = time.time()
            print(f'[{datetime.now().strftime("%H:%M:%S")}] START {exp_id} gpu={gpu} mode={ctx.mode}')
            failed = state.setdefault('failed_experiments', [])
            blocked = state.setdefault('blocked_experiments', [])
            completed = state.setdefault('completed_experiments', [])
            try:
                if args.mode == 'dryrun':
                    from tools.encoder_swap_semantic_drift.common import ExpResult
                    res = ExpResult('COMPLETE', exp_id, exp_id.replace('exp_', 'exp_') + '_dryrun')
                else:
                    res = fn(ctx, gpu, encode_device)
            except Exception as exc:
                res = None
                err = traceback.format_exc()
                state['last_error'] = err
                if exp_id not in failed:
                    failed.append(exp_id)
                print(f'FAILED {exp_id}: {exc}')
                if args.max_retry > 0 and not ctx.smoke:
                    ctx.smoke = True
                    try:
                        res = fn(ctx, gpu, encode_device)
                        ctx.smoke = args.mode == 'smoke'
                    except Exception:
                        res = None
                ctx.smoke = args.mode == 'smoke'
            if res is None:
                save_state(ctx, state)
                continue
            status = res.status
            if status == 'BLOCKED':
                if exp_id not in blocked:
                    blocked.append(exp_id)
            elif status in ('FAILED',):
                if exp_id not in failed:
                    failed.append(exp_id)
            else:
                if exp_id in failed:
                    failed.remove(exp_id)
                if exp_id in blocked and status != 'BLOCKED':
                    blocked.remove(exp_id)
                if exp_id not in done:
                    completed.append(exp_id)
                    done.add(exp_id)
                progressed = True
            if res.rows:
                for r in res.rows[:5]:
                    if 'dense_sv' in r or 'final_sv' in r:
                        state['latest_results'].append(str(r))
            state['latest_hypothesis_update'] = res.interpretation or state['latest_hypothesis_update']
            finalize_exp(
                ctx, res, gpu, t0,
                command=f'run_encoder_swap_overnight.py --exp {exp_id}',
                why=f'Overnight encoder swap queue item {exp_id}',
                setup=f'gpu={gpu} smoke={ctx.smoke} gpus={gpus}',
                sanity=f'status={res.status} rows={len(res.rows)}',
                interpretation=res.interpretation,
                guesses=res.next_guesses or ['continue queue', 'refine best encoder', 'heldout AP'],
                decision='Autonomous loop continues until 09:00',
            )
            state['running_experiment'] = ''
            save_state(ctx, state)
            update_live_status(ctx, state)
            print(f'DONE {exp_id} -> {status}')

        if args.allow_autonomous_next and datetime.now() < target_stop:
            # repeat exp_09 variants
            from tools.encoder_swap_semantic_drift.experiments.queue import run_exp09_autonomous
            tag = f'cycle_{cycle}'
            if tag not in done:
                try:
                    res = run_exp09_autonomous(ctx, gpu, encode_device)
                    finalize_exp(ctx, res, gpu, time.time(), command=tag, why='autonomous extra',
                                 setup='extra grid', sanity=res.status,
                                 interpretation=res.interpretation, guesses=[], decision='loop')
                    done.add(tag)
                    progressed = True
                except Exception as exc:
                    state['last_error'] = str(exc)

        if not progressed:
            time.sleep(120)
        if args.mode == 'smoke':
            break

    build_final_summary(ctx, state)
    save_state(ctx, state)
    update_live_status(ctx, state)
    print('OVERNIGHT CONTROLLER FINISHED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
