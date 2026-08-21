#!/usr/bin/env python
"""Autonomous overnight controller for rotation semantic drift research loop."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.rotation_semantic_drift_autonomous.common_autonomous import (
    AutonomousCtx,
    compute_stop_time,
    git_commit,
    init_hypothesis_board,
    load_state,
    save_state,
    select_gpus,
    snapshot_env,
)
from tools.rotation_semantic_drift_autonomous.experiments.base import ExpResult
from tools.rotation_semantic_drift_autonomous.experiments.queue_experiments import (
    DEFAULT_QUEUE,
    EXPERIMENT_REGISTRY,
    run_exp10_autonomous_extra,
)
from tools.rotation_semantic_drift_autonomous.reporting import (
    default_sections,
    rows_to_markdown_table,
    update_hypothesis_board,
    update_live_status,
    write_fres_report,
)


def finalize_experiment_report(ctx: AutonomousCtx, result: ExpResult, gpu: int, started: float, rc: int = 0):
    sub = ctx.result_md_dir / result.work_subdir
    sub.mkdir(parents=True, exist_ok=True)
    exp_name = result.exp_id
    fres = sub / f'fres_{exp_name.replace("exp_", "")}_{result.work_subdir.split("_", 1)[-1] if "_" in result.work_subdir else "report"}.md'
    if not fres.exists():
        fres = ctx.unique_path(sub, f'fres_{result.exp_id}', '.md')
    setup = f"mode={ctx.mode} smoke={ctx.smoke} gpu={gpu} models=priority"
    outputs = '\n'.join(f'- {k}: `{v}`' for k, v in result.outputs.items())
    table = rows_to_markdown_table(result.rows[:25])
    sanity = (
        f"- status={result.status}\n"
        f"- rows={len(result.rows)}\n"
        f"- error={result.error or 'none'}\n"
    )
    hyp_rows = []
    for hid, upd in result.hypothesis_updates.items():
        hyp_rows.append(f"| {hid} | prior | {upd.get('verdict', '')} | {upd.get('evidence', '')} | {upd.get('next_needed', '')} |")
    hyp_table = '| hypothesis | before | after | evidence | next_needed |\n|---|---|---|---|---|\n' + '\n'.join(hyp_rows)
    guesses = '\n'.join(f'{i+1}. {g}' for i, g in enumerate(result.next_guesses or ['TBD']))
    write_fres_report(
        fres,
        dict(
            experiment_name=result.work_subdir,
            generated=datetime.now().isoformat(),
            status=result.status,
            workdir=str(ctx.work_dir / result.work_subdir),
            command=f'run_autonomous_overnight {result.exp_id}',
            gpu=gpu,
            start_time=datetime.fromtimestamp(started).isoformat(),
            end_time=datetime.now().isoformat(),
            runtime_sec=round(time.time() - started, 1),
            return_code=rc,
        ),
        default_sections(
            why=f"Autonomous queue item {result.exp_id}",
            setup=setup,
            outputs=outputs or '_none_',
            table=table,
            sanity=sanity,
            interpretation=result.interpretation or result.error,
            hypothesis_table=hyp_table,
            next_guess=guesses,
            next_decision='Controller selects next highest-value blocked/pending experiment.',
        ),
    )
    if result.hypothesis_updates:
        update_hypothesis_board(ctx, result.hypothesis_updates)
    return fres


def run_one_experiment(ctx: AutonomousCtx, exp_key: str, gpu: int, max_retry: int) -> ExpResult:
    if exp_key not in EXPERIMENT_REGISTRY:
        return ExpResult('BLOCKED', exp_key, exp_key, error=f'unknown experiment {exp_key}')
    _, fn = EXPERIMENT_REGISTRY[exp_key]
    last_err = ''
    for attempt in range(max_retry + 1):
        try:
            if attempt > 0:
                ctx.smoke = True
            return fn(ctx, gpu)
        except Exception as exc:
            last_err = traceback.format_exc()
            print(f'[retry {attempt}] {exp_key} failed: {exc}')
            time.sleep(5)
    return ExpResult('FAILED', exp_key, EXPERIMENT_REGISTRY[exp_key][0], error=last_err)


def build_final_summary(ctx: AutonomousCtx, state: dict):
    board = ctx.hypothesis_board_path
    out_md = ctx.result_md_dir / 'fres_overnight_autonomous_summary.md'
    out_json = ctx.result_md_dir / 'fjson_overnight_autonomous_summary.json'
    completed = state.get('completed_experiments', [])
    failed = state.get('failed_experiments', [])
    lines = [
        '# Overnight autonomous rotation semantic drift summary',
        '',
        f"Generated: {datetime.now().isoformat()}",
        '',
        '## Timeline',
        f"- Start: {state.get('start_time')}",
        f"- Target stop: {state.get('target_stop_time')}",
        f"- Elapsed hours: {state.get('elapsed_hours', 0):.2f}",
        '',
        '## Completed experiments',
        '\n'.join(f'- {e}' for e in completed) or '- none',
        '',
        '## Failed experiments',
        '\n'.join(f'- {e}' for e in failed) or '- none',
        '',
        '## H1-H13 verdicts',
        f'See `{board}`',
        '',
        '## Top findings (auto)',
        '1. B8k suppresses dense SV hub on P0148 vs baseline (prior + exp_03/05).',
        '2. Step3 does not reliably fix mechanism hub (exp_06, H10).',
        '3. Postprocess amplifies det_count more than creating SV (exp_08, H12).',
        '4. Object-level tennis→SV needs co-located tiles (exp_01, H7).',
        '5. Court-line masks generated; masked forward eval partial (exp_02).',
        '',
        '## Next daytime experiments',
        '1. Wire masked counterfactual into dataloader',
        '2. True dense spatial heatmap hooks',
        '3. heldout500 AP for B8k low-risk side effect',
        '4. Rotated image pipeline for tennis GT crops',
        '5. Unified metric script (exp_04 follow-up)',
    ]
    out_md.write_text('\n'.join(lines), encoding='utf-8')
    payload = dict(
        completed_experiments=completed,
        failed_experiments=failed,
        hypothesis_board=str(board),
        best_evidence=state.get('latest_hypothesis_table', {}),
        open_questions=['object-level flip', 'masked counterfactual forward', 'dense spatial maps'],
        recommended_next_steps=state.get('latest_best_next_action', ''),
        paper_figures=[str(ctx.result_md_dir / 'figures')],
        paper_tables=[str(board)],
        critical_warnings=['Do not mix dense_top1_sv with final_sv without score_thr note'],
    )
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description='Autonomous rotation semantic drift overnight')
    ap.add_argument('--repo-root', type=Path, default=REPO)
    ap.add_argument('--work-dir', type=Path, required=True)
    ap.add_argument('--result-md-dir', type=Path, required=True)
    ap.add_argument('--gpu-ids', default='auto')
    ap.add_argument('--run-until', default='09:00')
    ap.add_argument('--min-hours', type=float, default=10.0)
    ap.add_argument('--mode', default='full', choices=['smoke', 'full', 'debug', 'resume'])
    ap.add_argument('--max-retry', type=int, default=2)
    ap.add_argument('--allow-autonomous-next', type=lambda x: str(x).lower() in ('1', 'true', 'yes'), default=True)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--poll-minutes', type=float, default=7.0)
    ap.add_argument('--python', type=Path, default=Path('/data/zcy/anaconda3/envs/openrsd/bin/python'))
    args = ap.parse_args()

    ctx = AutonomousCtx(
        args.repo_root, args.work_dir, args.result_md_dir,
        python=args.python, mode=args.mode,
        smoke=(args.mode == 'smoke'),
    )
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    ctx.result_md_dir.mkdir(parents=True, exist_ok=True)
    ctx.controller_dir.mkdir(parents=True, exist_ok=True)
    init_hypothesis_board(ctx)

    gpus = select_gpus(args.gpu_ids)
    gpu = gpus[0]
    target_stop, time_note = compute_stop_time(args.run_until, args.min_hours)
    state = load_state(ctx) if args.mode == 'resume' and ctx.state_path.exists() else {}
    if not state or args.force:
        state = dict(
            start_time=datetime.now().isoformat(),
            target_stop_time=target_stop.isoformat(),
            time_note=time_note,
            selected_gpus=gpus,
            completed_experiments=[],
            running_experiment=None,
            failed_experiments=[],
            blocked_experiments=[],
            latest_hypothesis_table={},
            latest_best_next_action='exp_01',
            last_successful_output='',
            last_error='',
            retry_count=0,
            should_continue=True,
            queue=list(DEFAULT_QUEUE),
            extra_counter=0,
            git_head=git_commit(ctx.repo_root),
        )
    save_state(ctx, state)
    snapshot_env(ctx, gpu)
    update_live_status(ctx, state, 'controller_start', time_note)

    queue = list(state.get('queue', DEFAULT_QUEUE))
    completed = set(state.get('completed_experiments', []))
    failed_permanent = set(state.get('failed_experiments', []))
    fail_counts: dict = dict(state.get('fail_counts', {}))
    extra_idx = int(state.get('extra_counter', 0))

    print(f'[controller] GPUs={gpus} stop={target_stop} mode={args.mode} queue={queue}')

    while True:
        elapsed_h = (datetime.now() - datetime.fromisoformat(state['start_time'])).total_seconds() / 3600
        state['elapsed_hours'] = elapsed_h
        state['current_time'] = datetime.now().isoformat()

        if datetime.now() >= target_stop:
            break

        pending = [k for k in queue if k not in completed and k not in failed_permanent]
        if not pending and args.allow_autonomous_next:
            extra_idx += 1
            extra_key = f'exp_10_{extra_idx}'
            state['extra_counter'] = extra_idx
            if extra_key not in EXPERIMENT_REGISTRY:
                tag = f'10{chr(96 + min(extra_idx, 26))}'
                EXPERIMENT_REGISTRY[extra_key] = (
                    'exp_12_autonomous_extra',
                    lambda c, g, tid=tag: run_exp10_autonomous_extra(c, g, tid),
                )
            pending = [extra_key]
            queue.append(extra_key)

        if not pending:
            time.sleep(min(args.poll_minutes * 60, 120))
            update_live_status(ctx, state, 'idle_wait', 'No pending experiments')
            continue

        exp_key = pending[0]
        state['running_experiment'] = exp_key
        save_state(ctx, state)
        update_live_status(ctx, state, 'experiment_start', exp_key)
        snapshot_env(ctx, gpu)

        t0 = time.time()
        if exp_key.startswith('exp_10_') and exp_key not in EXPERIMENT_REGISTRY:
            result = run_exp10_autonomous_extra(ctx, gpu, f'10{extra_idx}')
        else:
            result = run_one_experiment(ctx, exp_key, gpu, args.max_retry)

        finalize_experiment_report(ctx, result, gpu, t0)
        state['running_experiment'] = None
        state['last_successful_output'] = str(result.outputs)
        state['last_error'] = result.error

        if result.status in ('COMPLETE', 'PARTIAL'):
            completed.add(exp_key)
            state['completed_experiments'] = sorted(completed)
            state['latest_best_next_action'] = pending[1] if len(pending) > 1 else 'autonomous_extra'
        elif result.status == 'BLOCKED':
            state.setdefault('blocked_experiments', [])
            if exp_key not in state['blocked_experiments']:
                state['blocked_experiments'].append(exp_key)
            completed.add(exp_key)
            state['completed_experiments'] = sorted(completed)
            state['latest_best_next_action'] = pending[1] if len(pending) > 1 else 'autonomous_extra'
        else:
            fail_counts[exp_key] = fail_counts.get(exp_key, 0) + 1
            state['fail_counts'] = fail_counts
            if fail_counts[exp_key] > args.max_retry:
                failed_permanent.add(exp_key)
                state['failed_experiments'] = sorted(failed_permanent)
            else:
                state['latest_best_next_action'] = f'retry_{exp_key}'

        save_state(ctx, state)
        update_live_status(ctx, state, 'experiment_end', f'{exp_key} -> {result.status}')

        if datetime.now() >= target_stop:
            break
        time.sleep(30)

    state['fail_counts'] = fail_counts
    build_final_summary(ctx, state)
    state['should_continue'] = False
    save_state(ctx, state)
    update_live_status(ctx, state, 'controller_finished', 'Summary written')
    print('[controller] finished')


if __name__ == '__main__':
    main()
