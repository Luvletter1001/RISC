#!/usr/bin/env python
"""Hourly patrol for SV-DeHub-Lite v1 train/eval on GPU8/GPU9."""
from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.exp_sv_dehub_lite_train_gpu89 import common_dehub_utils as U
from tools.exp_sv_dehub_lite_train_gpu89 import patch_config_sv_dehub_lite as C


def _run(cmd: list[str], timeout: int = 30) -> str:
    try:
        return subprocess.check_output(cmd, text=True, timeout=timeout, stderr=subprocess.STDOUT)
    except Exception:
        return ''


def pgrep_pattern(pat: str) -> bool:
    out = _run(['pgrep', '-af', pat])
    for line in out.splitlines():
        if 'pgrep' in line or 'hourly_monitor' in line:
            continue
        if re.search(pat, line):
            return True
    return False


def gpu_mem(gpu_id: int) -> str:
    out = _run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits',
                f'--id={gpu_id}'])
    lines = [x.strip() for x in out.strip().splitlines() if x.strip()]
    return lines[0] if lines else 'NA'


def tail_log(path: Path, n: int = 80) -> str:
    if not path.exists():
        return ''
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    return '\n'.join(lines[-n:])


def classify_error(log_tail: str, curve_rows: list) -> str:
    t = log_tail.lower()
    if 'out of memory' in t or 'cuda out of memory' in t:
        return 'CUDA_OOM'
    if 'illegal memory access' in t:
        return 'CUDA_ILLEGAL_MEMORY'
    if 'filenotfound' in t or 'no such file' in t:
        return 'PATH_MISSING'
    if 'modulenotfound' in t or 'importerror' in t:
        return 'IMPORT_ERROR'
    if 'keyerror' in t:
        return 'CONFIG_ERROR'
    if 'nan' in t and 'loss' in t:
        return 'NAN_LOSS'
    if 'traceback' in t:
        return 'UNKNOWN'
    for row in curve_rows[-5:]:
        for k in ('total_loss', 'loss_dehub'):
            v = row.get(k, '')
            try:
                fv = float(v)
                if math.isnan(fv) or math.isinf(fv):
                    return 'NAN_LOSS'
            except (TypeError, ValueError):
                pass
    return ''


def latest_curve_row(curve_path: Path) -> dict:
    rows = U.read_csv(curve_path)
    return rows[-1] if rows else {}


def eval_means(eval_path: Path, ckpt: str) -> tuple[float, float]:
    rows = U.read_csv(eval_path)
    h = [float(r['final_sv_ratio']) for r in rows
         if r.get('checkpoint') == ckpt and r.get('group') == 'high']
    l = [float(r['final_sv_ratio']) for r in rows
         if r.get('checkpoint') == ckpt and r.get('group') == 'low']
    return (
        statistics.mean(h) if h else float('nan'),
        statistics.mean(l) if l else float('nan'),
    )


def save_debug_failure(result_dir: Path, err: str, log_tail: str, state: dict):
    ckpts = U.list_checkpoints(U.WORK_ROOT / 'train_v1')
    body = [
        '# Debug Latest Failure',
        '',
        f'- time: {datetime.now().isoformat()}',
        f'- error_tag: {err}',
        f'- train_alive: {state.get("train_alive")}',
        f'- eval_alive: {state.get("eval_alive")}',
        '',
        '## Checkpoints',
        *[f'- {p.name}' for p in ckpts],
        '',
        '## CSV rows',
        f'- train_curve: {state.get("train_csv_rows")}',
        f'- eval_raw: {state.get("eval_csv_rows")}',
        '',
        '## Log tail',
        '```',
        log_tail[-12000:],
        '```',
    ]
    (result_dir / 'fres_debug_latest_failure.md').write_text('\n'.join(body), encoding='utf-8')


def append_hourly(row: dict, result_dir: Path):
    csv_path = result_dir / 'ftable_hourly_monitor.csv'
    new = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=C.HOURLY_FIELDS, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, 'NA') for k in C.HOURLY_FIELDS})

    md_path = result_dir / 'fres_hourly_monitor.md'
    line = (
        f"\n### {row['check_time']}\n"
        f"- train: {row['gpu8_train_process_alive']} | eval: {row['gpu9_eval_process_alive']}\n"
        f"- iter: {row['latest_train_iter']} | ckpt: {row['latest_checkpoint']}\n"
        f"- dehub: {row['latest_dehub_loss']} (ratio {row['latest_dehub_ratio']})\n"
        f"- highrisk final_sv: {row['latest_highrisk_final_sv']} | low: {row['latest_lowrisk_final_sv']}\n"
        f"- error: {row['detected_error']} | action: {row['action_taken']}\n"
        f"- next: {row['next_action']}\n"
    )
    if not md_path.exists():
        md_path.write_text('# Hourly Monitor SV-DeHub-Lite v1\n', encoding='utf-8')
    with open(md_path, 'a', encoding='utf-8') as f:
        f.write(line)


def maybe_start_eval_watcher(result_dir: Path, work_dir: Path, train_alive: bool):
    """Start GPU9 eval watcher if checkpoints exist and eval not running."""
    if pgrep_pattern('run_eval_sv_dehub'):
        return 'eval_already_running'
    ckpts = U.list_checkpoints(work_dir)
    if not ckpts and train_alive:
        return 'wait_for_ckpt'
    if not ckpts:
        return 'no_ckpt'
    py = str(C.PYTHON)
    script = str(C.TOOL_DIR / 'run_eval_sv_dehub_lite_gpu9.py')
    log = result_dir / 'log_gpu9_eval_sv_dehub_lite_v1.txt'
    cmd = (
        f'cd {REPO} && CUDA_VISIBLE_DEVICES={C.EVAL_GPU} PYTHONPATH={REPO} '
        f'nohup {py} {script} --result-dir {result_dir} --work-dir {work_dir} '
        f'--watch-checkpoints --poll-sec 300 --eval-highrisk --eval-lowrisk --eval-ap-smoke '
        f'>> {log} 2>&1 &'
    )
    subprocess.Popen(['bash', '-c', cmd], cwd=str(REPO))
    return 'started_eval_watcher'


def maybe_resume_train(result_dir: Path, work_dir: Path, err: str):
    if err not in ('CUDA_OOM', 'NAN_LOSS', 'UNKNOWN', 'CONFIG_ERROR', 'IMPORT_ERROR', ''):
        return 'no_resume'
    ckpts = sorted(work_dir.glob('iter_*.pth'), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        return 'no_ckpt_for_resume'
    latest = ckpts[-1]
    py = str(C.PYTHON)
    script = str(C.TOOL_DIR / 'run_train_sv_dehub_lite_gpu8.py')
    log = result_dir / 'log_gpu8_train_sv_dehub_lite_v1.txt'
    cmd = (
        f'cd {REPO} && CUDA_VISIBLE_DEVICES={C.TRAIN_GPU} PYTHONPATH={REPO} '
        f'nohup {py} {script} --result-dir {result_dir} --work-dir {work_dir} '
        f'--resume-from {latest} --max-iters {C.MAX_ITERS_DEFAULT} '
        f'>> {log} 2>&1 &'
    )
    subprocess.Popen(['bash', '-c', cmd], cwd=str(REPO))
    notes = result_dir / 'fres_debug_patch_notes.md'
    with open(notes, 'a', encoding='utf-8') as f:
        f.write(f'\n## Resume {datetime.now().isoformat()}\n- from {latest}\n- err {err}\n')
    return f'resumed_from_{latest.name}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-dir', type=Path, default=REPO)
    ap.add_argument('--result-dir', type=Path, default=C.RESULT_DIR)
    ap.add_argument('--work-dir', type=Path, default=C.TRAIN_WORK_DIR)
    ap.add_argument('--debug-if-crashed', action='store_true')
    ap.add_argument('--resume-if-fixed', action='store_true')
    args = ap.parse_args()
    U.ensure_dirs()
    now = datetime.now().isoformat(timespec='seconds')

    train_alive = pgrep_pattern(r'train\.py.*sv_dehub|train\.py.*sv_dehub_lite')
    eval_alive = pgrep_pattern('run_eval_sv_dehub')

    curve_path = args.result_dir / 'ftable_train_loss_curve.csv'
    eval_path = args.result_dir / 'ftable_eval_sv_dehub_lite_raw.csv'
    train_log = args.result_dir / 'log_gpu8_train_sv_dehub_lite_v1.txt'
    eval_log = args.result_dir / 'log_gpu9_eval_sv_dehub_lite_v1.txt'

    curve_rows = U.read_csv(curve_path)
    eval_rows = U.read_csv(eval_path)
    last = latest_curve_row(curve_path)

    ckpts = U.list_checkpoints(args.work_dir)
    latest_ckpt = ckpts[-1].name if ckpts else 'none'
    latest_iter = last.get('iter', 'NA')

    log_tail = tail_log(train_log, 200)
    err = '' if train_alive else classify_error(log_tail, curve_rows)
    if not train_alive and not ckpts and 'traceback' in log_tail.lower():
        err = err or 'UNKNOWN'

    dehub = float(last.get('loss_dehub', 0) or 0)
    dehub_ratio = float(last.get('dehub_ratio', 0) or 0)
    if dehub <= 0 and curve_rows and train_alive:
        err = err or 'LOSS_NOT_BACKPROP'

    best_dehub_ck = 'iter_3000'
    for name in ('iter_3000', 'iter_2000', 'iter_1000', 'baseline_epoch24'):
        if any(r.get('checkpoint') == name for r in eval_rows):
            best_dehub_ck = name
            break
    hr_sv, lr_sv = eval_means(eval_path, best_dehub_ck)

    action = 'record_only'
    next_action = 'sleep_3600_and_recheck'

    if not train_alive and ckpts and int(str(latest_iter).replace('NA', '0') or 0) >= C.MAX_ITERS_DEFAULT:
        train_status = 'completed'
        if not eval_rows or len({r['checkpoint'] for r in eval_rows}) < 2:
            action = maybe_start_eval_watcher(args.result_dir, args.work_dir, False)
            next_action = 'wait_eval_complete'
        else:
            action = 'train_done_eval_ok'
            next_action = 'run_analyze_and_final_report'
    elif train_alive:
        train_status = 'running'
        if any(p.name == 'iter_1000.pth' for p in ckpts) and not eval_alive:
            action = maybe_start_eval_watcher(args.result_dir, args.work_dir, True)
    elif err and args.debug_if_crashed:
        train_status = 'crashed'
        save_debug_failure(args.result_dir, err, log_tail, {
            'train_alive': train_alive, 'eval_alive': eval_alive,
            'train_csv_rows': len(curve_rows), 'eval_csv_rows': len(eval_rows),
        })
        action = f'debug_saved:{err}'
        if args.resume_if_fixed:
            action = maybe_resume_train(args.result_dir, args.work_dir, err)
            next_action = 'verify_resume'
    else:
        train_status = 'idle'

    if dehub_ratio > 0.30:
        err = err or 'NAN_LOSS'
        action = 'warn_dehub_ratio_high'

    row = dict(
        check_time=now,
        gpu8_train_process_alive=train_alive,
        gpu9_eval_process_alive=eval_alive,
        gpu8_memory_used=gpu_mem(C.TRAIN_GPU),
        gpu9_memory_used=gpu_mem(C.EVAL_GPU),
        latest_train_iter=latest_iter,
        latest_checkpoint=latest_ckpt,
        train_log_updated=train_log.exists(),
        eval_log_updated=eval_log.exists(),
        train_csv_rows=len(curve_rows),
        eval_csv_rows=len(eval_rows),
        latest_total_loss=last.get('total_loss', 'NA'),
        latest_cls_loss=last.get('loss_cls', 'NA'),
        latest_dehub_loss=last.get('loss_dehub', 'NA'),
        latest_dehub_ratio=last.get('dehub_ratio', 'NA'),
        latest_eval_checkpoint=best_dehub_ck,
        latest_highrisk_final_sv=f'{hr_sv:.4f}' if hr_sv == hr_sv else 'NA',
        latest_lowrisk_final_sv=f'{lr_sv:.4f}' if lr_sv == lr_sv else 'NA',
        detected_error=err or 'none',
        action_taken=action,
        next_action=next_action,
    )
    append_hourly(row, args.result_dir)

    print('[SV-DeHub Monitor]')
    print('time:', now)
    print('train_status:', train_status)
    print('eval_status:', 'running' if eval_alive else ('done' if eval_rows else 'idle'))
    print('latest_iter:', latest_iter)
    print('latest_checkpoint:', latest_ckpt)
    print('latest_dehub_loss:', row['latest_dehub_loss'])
    print('latest_highrisk_final_sv:', row['latest_highrisk_final_sv'])
    print('detected_error:', row['detected_error'])
    print('action_taken:', action)
    print('next_check:', next_action)


if __name__ == '__main__':
    main()
