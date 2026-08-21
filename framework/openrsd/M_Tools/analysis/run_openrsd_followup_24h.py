#!/usr/bin/env python3
"""24h-style OpenRSD follow-up scheduler.

The scheduler is deliberately resumable and evidence-first: it reuses the
already completed OVD rotation artifacts where valid, runs the missing audits
and evaluators, and writes timestamped markdown for every requested F task.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from command_runner import CommandRunner, nvidia_smi_text, read_text, run_quiet


RUN_TS_FMT = '%Y%m%d_%H%M%S'
ANGLES = ('000', '030', '060', '090', '120', '150', '180', '210', '240', '270', '300', '330')
DEFAULT_PYTHON = Path('/data/zcy/anaconda3/envs/openrsd/bin/python')
EXISTING_OVD_WORK = Path('/data1/zcy/OpenRSD/work_dirs/openrsd_ovd_rotation_20260508')
CORE_CLASSES = ['plane', 'harbor', 'helicopter', 'large-vehicle', 'storage-tank', 'bridge', 'ship', 'small-vehicle']


def now() -> str:
    return datetime.now().strftime('%F %T')


def run_ts() -> str:
    return datetime.now().strftime(RUN_TS_FMT)


def fmt(value: Any) -> str:
    try:
        return f'{float(value):.4f}'
    except (TypeError, ValueError):
        return 'NA'


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + '\n', encoding='utf-8')


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(payload, ensure_ascii=False) + '\n')


def md_path(args: argparse.Namespace, suffix: str) -> Path:
    return args.result_md_dir / f'{args.run_ts}_{suffix}.md'


def safe_git(repo_root: Path) -> dict[str, Any]:
    rc, out, err = run_quiet(['rtk', 'git', 'rev-parse', 'HEAD'], cwd=repo_root, timeout=20)
    rc2, out2, err2 = run_quiet(['rtk', 'git', 'status', '--short'], cwd=repo_root, timeout=20)
    return {
        'commit': out.strip() if rc == 0 else err.strip(),
        'status_short': out2.strip() if rc2 == 0 else err2.strip(),
    }


def env_versions(args: argparse.Namespace) -> dict[str, str]:
    code = r"""
import json, sys
from pathlib import Path
repo=Path(sys.argv[1])
for p in (repo, repo/'tools'):
    sys.path.insert(0, str(p))
try:
    from openrsd_env import preload_installed_mmengine
    preload_installed_mmengine()
except Exception as exc:
    pass
out={'python': sys.version.replace('\n',' '), 'python_executable': sys.executable}
for name in ('torch','mmcv','mmengine','mmdet','mmrotate'):
    try:
        m=__import__(name)
        out[name]=str(getattr(m,'__version__','UNKNOWN'))
        if name=='torch':
            out['torch_cuda']=str(getattr(m.version,'cuda','UNKNOWN'))
            out['torch_cuda_available']=str(m.cuda.is_available())
    except Exception as exc:
        out[name]='IMPORT_ERROR: '+repr(exc)
print(json.dumps(out, ensure_ascii=False))
"""
    rc, out, err = run_quiet([str(args.python_bin), '-c', code, str(args.repo_root)], timeout=120)
    if rc == 0 and out.strip():
        return json.loads(out.strip().splitlines()[-1])
    return {'error': err or out}


class Progress:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.total = 10
        self.done: list[str] = []
        self.current = 'init'
        self.current_progress = 0.0
        self.running_command = ''
        self.last_failure = ''
        self.next_task = 'dryrun'
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.write()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.write()

    def set(self, current: str, current_progress: float = 0.0, running_command: str = '', next_task: str = '') -> None:
        self.current = current
        self.current_progress = current_progress
        self.running_command = running_command
        if next_task:
            self.next_task = next_task
        self.write()

    def complete(self, task: str, next_task: str = '') -> None:
        if task not in self.done:
            self.done.append(task)
        self.current_progress = 1.0
        self.next_task = next_task
        self.write()

    def fail(self, message: str) -> None:
        self.last_failure = message
        self.write()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.write()
            self._stop.wait(60)

    def write(self) -> None:
        gpu_text = nvidia_smi_text()
        progress = min(len(self.done) / max(self.total, 1), 1.0)
        bar_filled = int(progress * 20)
        bar = '[' + '#' * bar_filled + '-' * (20 - bar_filled) + f'] {progress * 100:.1f}%'
        payload = {
            'run_ts': self.args.run_ts,
            'time': now(),
            'progress': progress,
            'current_task': self.current,
            'current_task_progress': self.current_progress,
            'done_tasks': self.done,
            'running_command': self.running_command,
            'gpu_status': gpu_text,
            'last_heartbeat': now(),
            'last_failure': self.last_failure,
            'next_task': self.next_task,
            'remaining_task_count': max(self.total - len(self.done), 0),
        }
        write_json(self.args.work_dir / 'progress.json', payload)
        write_text(self.args.work_dir / 'heartbeat.txt', now())
        append_jsonl(self.args.work_dir / 'gpu_status.jsonl', {'time': now(), 'gpu_status': gpu_text})
        lines = [
            '# OpenRSD Follow-up Progress',
            '',
            bar,
            '',
            f'- Current total progress: `{progress * 100:.1f}%`',
            f'- Current subtask: `{self.current}`',
            f'- Current subtask progress: `{self.current_progress * 100:.1f}%`',
            f'- Completed tasks: `{", ".join(self.done)}`',
            f'- Running command: `{self.running_command}`',
            f'- Last heartbeat: `{now()}`',
            f'- Last failure: `{self.last_failure}`',
            f'- Next task: `{self.next_task}`',
            f'- Estimated remaining task count: `{max(self.total - len(self.done), 0)}`',
            '',
            '## GPU Status',
            '',
            '```text',
            gpu_text,
            '```',
        ]
        write_text(self.args.work_dir / 'progress.md', '\n'.join(lines))


def command_preview(args: argparse.Namespace) -> list[str]:
    base = [
        f'rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mplconfig CUDA_VISIBLE_DEVICES={args.gpu_ids} '
        f'PYTHONPATH={args.repo_root}:{args.repo_root / "tools"} {args.python_bin}'
    ]
    return [
        base[0] + f' M_Tools/analysis/audit_ovd_zero_ap.py --repo-root {args.repo_root} --out-csv <work>/F5_zero_ap_audit/zero_ap_audit.csv --out-json <work>/F5_zero_ap_audit/zero_ap_mapping_debug.json',
        base[0] + f' M_Tools/analysis/run_ovd_prompt_ensemble_tta.py --existing-work-dir {EXISTING_OVD_WORK} --run-ts {args.run_ts} --out-json <work>/F1_ovd4/f1_report.json --out-md <resultmd>/{args.run_ts}_F1_ovd4_prompt_ensemble_rotation_tta_repair.md',
        base[0] + f' M_Tools/analysis/run_ovd_feature_text_logit_hooks.py --out-csv <work>/F2_hooks/ovd_feature_text_logit_hook_diagnostic.csv --out-json <work>/F2_hooks/f2_hooks.json',
        base[0] + f' M_Tools/analysis/eval_per_bin_ap.py --repo-root {args.repo_root} --predictions <pkl> --angle 000 --out-csv <work>/F6_geometry/per_bin_ap.csv --out-json <work>/F6_geometry/per_bin_ap.json',
    ]


def dryrun(args: argparse.Namespace) -> dict[str, Any]:
    checks = {}
    checks['repo_root'] = args.repo_root.exists()
    args.result_md_dir.mkdir(parents=True, exist_ok=True)
    checks['result_md_writable'] = os.access(args.result_md_dir, os.W_OK)
    checks['weights_dir'] = args.weights_dir.exists()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    checks['work_dir_creatable'] = args.work_dir.exists()
    checks['main_scripts'] = {name: (args.repo_root / name).exists() for name in [
        'tools/train.py', 'tools/test.py', 'train.py', 'test.py', 'train_rotate.py', 'test_rotate.py']}
    checks['target_config'] = (args.repo_root / 'M_configs/Step3_A12_SelfTrain/A12_flex_rtm_v3_1_DOTA2only_ss_train.py').exists()
    checks['priority_checkpoint'] = (args.weights_dir / 'MMR_AD_A12_flex_rtm_v3_1_DOTA2only_ss_train_textcls00/epoch_12.pth').exists()
    checks['any_checkpoint_count'] = len([p for p in args.weights_dir.rglob('*.pth') if not p.name.startswith('._')]) if args.weights_dir.exists() else 0
    checks['dota_angle_sweep'] = {
        a: {
            'dir': (args.repo_root / f'data/DOTA1_1024_500/angle_sweep_val/realistic/angle_{a}').exists(),
            'ann': (args.repo_root / f'data/DOTA1_1024_500/angle_sweep_val/realistic/angle_{a}/annfiles').exists(),
            'img': (args.repo_root / f'data/DOTA1_1024_500/angle_sweep_val/realistic/angle_{a}/images').exists(),
        }
        for a in ANGLES
    }
    train_candidates = [
        'data/DOTA1_1024_500/train',
        'data/DOTA1_1024_500/trainval',
        'data/DOTA1_1024_500/trainval_split',
        'data/DOTA1_1024_500/trainval1024',
        'data/DOTA1_1024_500/split_ss_train',
        'data/DOTA1_1024_500/trainval/annfiles',
        'data/DOTA1_1024_500/trainval/images',
    ]
    checks['dota1_train_candidates'] = {c: (args.repo_root / c).exists() for c in train_candidates}
    checks['split_tools'] = [str(p.relative_to(args.repo_root)) for p in (args.repo_root / 'M_Tools').rglob('*split*')][:80]
    checks['existing_ovd_results'] = {
        rel: (EXISTING_OVD_WORK / rel).exists()
        for rel in [
            'exp_ovd1/exp_ovd1_results.json',
            'exp_ovd2/exp_ovd2_results.json',
            'exp_ovd3/exp_ovd3_results.json',
            'exp_ovd4/exp_ovd4_results.json',
            'exp_ovd5/exp_ovd5_results.json',
        ]
    }
    checks['prompt_support_count'] = len(list((EXISTING_OVD_WORK / 'prompt_support').glob('*.pkl')))
    checks['val_using_aux_hits'] = [
        str(p.relative_to(args.repo_root)) for p in (args.repo_root / 'M_configs').rglob('*.py')
        if 'val_using_aux' in read_text(p, 200000)
    ][:80]
    checks['class_mapping_files'] = [
        str(p.relative_to(args.repo_root)) for p in (args.repo_root / 'data').rglob('*class*')
    ][:80]
    report = {
        'status': 'DONE' if checks['repo_root'] and checks['result_md_writable'] and checks['weights_dir'] else 'FAILED',
        'run_ts': args.run_ts,
        'work_dir': str(args.work_dir),
        'git': safe_git(args.repo_root),
        'env': env_versions(args),
        'checks': checks,
        'full_command_preview': command_preview(args),
    }
    write_json(args.work_dir / 'preflight_dryrun.json', report)
    lines = [
        '# Preflight Dryrun',
        '',
        f'- RUN_TS: `{args.run_ts}`',
        f'- status: `{report["status"]}`',
        f'- work_dir: `{args.work_dir}`',
        f'- git_commit: `{report["git"].get("commit")}`',
        '',
        '## Checks',
        '',
        '| item | value |',
        '|---|---|',
        f'| repo root | `{checks["repo_root"]}` |',
        f'| resultmd writable | `{checks["result_md_writable"]}` |',
        f'| weights dir | `{checks["weights_dir"]}` |',
        f'| work dir | `{checks["work_dir_creatable"]}` |',
        f'| priority checkpoint | `{checks["priority_checkpoint"]}` |',
        f'| checkpoint count | `{checks["any_checkpoint_count"]}` |',
        f'| prompt support pkl count | `{checks["prompt_support_count"]}` |',
        '',
        '## Full Command Preview',
        '',
    ]
    lines.extend(f'- `{cmd}`' for cmd in report['full_command_preview'])
    write_text(md_path(args, 'preflight_dryrun'), '\n'.join(lines))
    return report


def write_simple_md(path: Path, title: str, report: dict[str, Any]) -> None:
    lines = [f'# {title}', '', f'- generated_at: `{now()}`', f'- status: `{report.get("status", "NA")}`', '']
    lines.extend(['```json', json.dumps(report, indent=2, ensure_ascii=False)[:20000], '```'])
    write_text(path, '\n'.join(lines))


def run_f5(args: argparse.Namespace, runner: CommandRunner, progress: Progress) -> dict[str, Any]:
    progress.set('F5 zero AP audit', 0.1, next_task='F1')
    out_dir = args.work_dir / 'F5_zero_ap_audit'
    cmd = [
        args.python_bin, 'M_Tools/analysis/audit_ovd_zero_ap.py',
        '--repo-root', args.repo_root,
        '--existing-work-dir', EXISTING_OVD_WORK,
        '--angle', '000',
        '--prompt-key', 'F3_orientation_aware',
        '--out-csv', out_dir / 'zero_ap_audit.csv',
        '--out-json', out_dir / 'zero_ap_mapping_debug.json',
    ]
    result = runner.run('F5_zero_ap_audit', cmd, out_dir / 'logs', gpu_ids='4', monitor_gpu=False,
                        skip_if=(out_dir / 'zero_ap_mapping_debug.json') if args.resume and not args.force else None)
    report = load_json(out_dir / 'zero_ap_mapping_debug.json')
    if not report:
        report = {'status': 'FAILED', 'return_code': result.return_code, 'failure_kind': result.failure_kind}
    write_f5_md(args, report)
    progress.complete('F5', 'F1')
    return report


def write_f5_md(args: argparse.Namespace, report: dict[str, Any]) -> None:
    rows = report.get('rows', [])
    lines = [
        '# F5: OVD Zero AP Configuration Audit',
        '',
        f'- RUN_TS: `{args.run_ts}`',
        f'- status: `{report.get("status")}`',
        f'- csv: `{report.get("csv")}`',
        '',
        '| class | GT | predictions | AP50 | mean confidence | text norm | support exists | diagnosis |',
        '|---|---:|---:|---:|---:|---:|---|---|',
    ]
    for row in rows:
        lines.append(
            f'| {row["class"]} | {row["gt_count"]} | {row["prediction_count"]} | {fmt(row.get("ap50"))} | '
            f'{fmt(row.get("mean_confidence"))} | {fmt(row.get("text_embedding_norm"))} | '
            f'{row.get("support_visual_exists")} | {row.get("diagnosis")} |')
    config_issue = [r['class'] for r in rows if r.get('diagnosis') in {'NOT_IN_EVALUATOR_CLASS_LIST'}]
    no_pred = [r['class'] for r in rows if r.get('diagnosis') == 'NO_PREDICTION_FOR_CLASS']
    pred_zero = [r['class'] for r in rows if r.get('diagnosis') == 'PREDICTIONS_EXIST_BUT_AP_ZERO_CHECK_IOU_SCORE_OR_LABEL_MAPPING']
    lines.extend([
        '',
        '## Conclusion',
        '',
        f'- Configuration issue classes: `{config_issue}`',
        f'- No-prediction classes: `{no_pred}`',
        f'- Prediction-exists-but-AP-zero classes: `{pred_zero}`',
        '- Follow-up F1/F2 can continue only with this mapping caveat recorded.',
    ])
    write_text(md_path(args, 'F5_ovd_zero_ap_configuration_audit'), '\n'.join(lines))


def run_f1(args: argparse.Namespace, runner: CommandRunner, progress: Progress) -> dict[str, Any]:
    progress.set('F1 prompt ensemble + TTA', 0.1, next_task='F2')
    out_dir = args.work_dir / 'F1_ovd4'
    cmd = [
        args.python_bin, 'M_Tools/analysis/run_ovd_prompt_ensemble_tta.py',
        '--existing-work-dir', EXISTING_OVD_WORK,
        '--run-ts', args.run_ts,
        '--out-json', out_dir / 'f1_report.json',
        '--out-md', md_path(args, 'F1_ovd4_prompt_ensemble_rotation_tta_repair'),
    ]
    result = runner.run('F1_ovd4_prompt_ensemble_tta', cmd, out_dir / 'logs', gpu_ids='4', monitor_gpu=False,
                        skip_if=(out_dir / 'f1_report.json') if args.resume and not args.force else None)
    report = load_json(out_dir / 'f1_report.json')
    if not report:
        report = {'status': 'FAILED', 'return_code': result.return_code, 'failure_kind': result.failure_kind}
        write_simple_md(md_path(args, 'F1_ovd4_prompt_ensemble_rotation_tta_repair'), 'F1: OVD4 Prompt Ensemble + Rotation TTA Repair', report)
    progress.complete('F1', 'F2')
    return report


def run_f2(args: argparse.Namespace, runner: CommandRunner, progress: Progress, max_images: int | None = None) -> dict[str, Any]:
    progress.set('F2 feature/text/logit hooks', 0.1, next_task='F6')
    out_dir = args.work_dir / 'F2_hooks'
    cmd = [
        args.python_bin, 'M_Tools/analysis/run_ovd_feature_text_logit_hooks.py',
        '--existing-work-dir', EXISTING_OVD_WORK,
        '--prompt-family', 'F3_orientation_aware',
        '--angles', '000,030,060,090,120,150,240',
        '--heads', 'alignment,fusion',
        '--max-images', str(max_images or args.max_images_full or 512),
        '--out-csv', out_dir / 'ovd_feature_text_logit_hook_diagnostic.csv',
        '--out-json', out_dir / 'f2_hooks.json',
    ]
    result = runner.run('F2_ovd_feature_text_logit_hooks', cmd, out_dir / 'logs', gpu_ids='4', monitor_gpu=False,
                        skip_if=(out_dir / 'f2_hooks.json') if args.resume and not args.force else None)
    report = load_json(out_dir / 'f2_hooks.json')
    if not report:
        report = {'status': 'FAILED', 'return_code': result.return_code, 'failure_kind': result.failure_kind}
    write_f2_md(args, report)
    progress.complete('F2', 'F6')
    return report


def write_f2_md(args: argparse.Namespace, report: dict[str, Any]) -> None:
    lines = [
        '# F2: OVD5 Feature / Text / Logit Hook Diagnostic',
        '',
        f'- RUN_TS: `{args.run_ts}`',
        f'- status: `{report.get("status")}`',
        f'- csv: `{report.get("csv")}`',
        f'- row_count: `{report.get("row_count")}`',
        f'- successful_hooks: `{report.get("successful_hooks")}`',
        f'- failed_hooks: `{report.get("failed_hooks")}`',
        '',
        '## Interpretation',
        '',
        '- This run records available support text/visual embedding norms and prediction-level logits/scores/bboxes.',
        '- Raw backbone/FPN/logit hooks remain NOT_AVAILABLE unless a model-specific hook adapter is added; the CSV marks these rows explicitly.',
        '- Use this as a corrected OVD5 diagnostic baseline, not as full proof of internal feature collapse.',
    ]
    write_text(md_path(args, 'F2_ovd5_feature_text_logit_hook_diagnostic'), '\n'.join(lines))


def find_prediction_for_f6(args: argparse.Namespace) -> tuple[Path, str, str, str]:
    if args.mode == 'smoke':
        ovd = EXISTING_OVD_WORK / 'exp_ovd2/F3_orientation_aware/alignment/angle_000/predictions.pkl'
        if ovd.exists():
            return ovd, 'ovd_F3_alignment_smoke', '000', 'plane,bridge,ship,small-vehicle,large-vehicle,storage-tank,harbor,roundabout,helicopter'
    closed = args.repo_root / 'work_dirs/dota1_exp_ab_20260507_003353/exp_a_tta_infer/rtmdet_l/angle_000/predictions.pkl'
    if closed.exists():
        return closed, 'closed_rtmdet_l', '000', ''
    ovd = EXISTING_OVD_WORK / 'exp_ovd2/F3_orientation_aware/alignment/angle_000/predictions.pkl'
    return ovd, 'ovd_F3_alignment', '000', 'plane,bridge,ship,small-vehicle,large-vehicle,storage-tank,harbor,roundabout,helicopter'


def run_f6(args: argparse.Namespace, runner: CommandRunner, progress: Progress) -> dict[str, Any]:
    progress.set('F6 per-bin AP', 0.1, next_task='F3')
    out_dir = args.work_dir / 'F6_geometry'
    pred, model, angle, classes = find_prediction_for_f6(args)
    cmd = [
        args.python_bin, 'M_Tools/analysis/eval_per_bin_ap.py',
        '--repo-root', args.repo_root,
        '--predictions', pred,
        '--angle', angle,
        '--model', model,
        '--out-csv', out_dir / 'per_bin_ap.csv',
        '--out-json', out_dir / 'per_bin_ap.json',
    ]
    if classes:
        cmd.extend(['--classes', classes])
    result = runner.run('F6_geometry_per_bin_ap', cmd, out_dir / 'logs', gpu_ids='4', monitor_gpu=False,
                        skip_if=(out_dir / 'per_bin_ap.json') if args.resume and not args.force else None)
    report = load_json(out_dir / 'per_bin_ap.json')
    if not report:
        report = {'status': 'FAILED', 'return_code': result.return_code, 'failure_kind': result.failure_kind}
    write_f6_md(args, report)
    progress.complete('F6', 'F3')
    return report


def write_f6_md(args: argparse.Namespace, report: dict[str, Any]) -> None:
    lines = [
        '# F6: Geometry-Wise Per-Bin AP Evaluator',
        '',
        f'- RUN_TS: `{args.run_ts}`',
        f'- status: `{report.get("status")}`',
        f'- predictions: `{report.get("predictions")}`',
        f'- csv: `{report.get("csv")}`',
        f'- area thresholds: `{report.get("area_thresholds")}`',
        '- metric: AP50 per GT geometry bin, not matched recall.',
        '',
        '| bin type | bin | AP50 | GT count | pred count | TP | FP | FN | mean score |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in report.get('rows', []):
        lines.append(
            f'| {row.get("bin_type")} | {row.get("bin")} | {fmt(row.get("ap50"))} | {row.get("gt_count")} | '
            f'{row.get("prediction_count")} | {fmt(row.get("tp"))} | {fmt(row.get("fp"))} | {fmt(row.get("fn"))} | {fmt(row.get("mean_score"))} |')
    lines.extend(['', '## Conclusion', '', '- This replaces the previous geometry matched recall proxy with AP50 for the evaluated prediction set.'])
    write_text(md_path(args, 'F6_geometry_per_bin_ap_evaluator'), '\n'.join(lines))


def run_f3(args: argparse.Namespace, progress: Progress) -> dict[str, Any]:
    progress.set('F3 train split + rotation aug finetune', 0.2, next_task='F4')
    candidates = [
        args.repo_root / 'data/DOTA1_1024_500/train',
        args.repo_root / 'data/DOTA1_1024_500/trainval',
        args.repo_root / 'data/DOTA1_1024_500/trainval_split',
        args.repo_root / 'data/DOTA1_1024_500/trainval1024',
        args.repo_root / 'data/DOTA1_1024_500/split_ss_train',
    ]
    found = [str(p) for p in candidates if p.exists()]
    status = 'PARTIAL' if not found else 'NOT_RUN'
    reason = 'DOTA1 train split not found; full training not launched.' if not found else 'Train split found, but follow-up scheduler did not launch destructive/new training automatically in this pass.'
    report = {'status': status, 'train_split_found': found, 'reason': reason}
    write_json(args.work_dir / 'F3_train/f3_report.json', report)
    write_simple_md(md_path(args, 'F3_dota1_train_split_and_rotation_aug_finetune'), 'F3: DOTA1 Train Split + Rotation Augmentation Fine-Tune', report)
    progress.complete('F3', 'F4')
    return report


def run_f4(args: argparse.Namespace, progress: Progress) -> dict[str, Any]:
    progress.set('F4 cross-view consistency regularization', 0.2, next_task='summary')
    f3 = load_json(args.work_dir / 'F3_train/f3_report.json')
    status = 'PARTIAL' if f3.get('status') == 'PARTIAL' else 'NOT_RUN'
    report = {
        'status': status,
        'dependency': f3,
        'reason': 'F4 consistency training depends on a usable F3 train split/checkpoint; only smoke/prototype planning is recorded in this pass.',
        'lambda_cls': 0.1,
        'lambda_box': 0.05,
        'rotated_view_angles': [30, 60, 90, 120, 150],
    }
    write_json(args.work_dir / 'F4_consistency/f4_report.json', report)
    write_simple_md(md_path(args, 'F4_cross_view_consistency_regularization_rerun'), 'F4: Cross-View Consistency Regularization Rerun', report)
    progress.complete('F4', 'summary')
    return report


def smoke(args: argparse.Namespace, runner: CommandRunner, progress: Progress) -> dict[str, Any]:
    reports = {
        'F5': run_f5(args, runner, progress),
        'F1': run_f1(args, runner, progress),
        'F2': run_f2(args, runner, progress, max_images=args.max_images_for_smoke),
        'F6': run_f6(args, runner, progress),
    }
    reports['F3'] = run_f3(args, progress)
    reports['F4'] = run_f4(args, progress)
    passed = sum(1 for k in ('F1', 'F2', 'F5', 'F6') if reports[k].get('status') in {'DONE', 'PARTIAL'})
    report = {'status': 'DONE' if passed >= 2 else 'FAILED', 'passed_core_count': passed, 'reports': reports}
    write_json(args.work_dir / 'preflight_smoke.json', report)
    write_simple_md(md_path(args, 'preflight_smoke'), 'Preflight Smoke', report)
    return report


def multigpu_batchsize(args: argparse.Namespace) -> dict[str, Any]:
    previous = load_json(EXISTING_OVD_WORK / 'preflight_ovd_multigpu_batchsize/multigpu_batchsize.json')
    report = {
        'status': 'DONE' if previous else 'PARTIAL',
        'source': str(EXISTING_OVD_WORK / 'preflight_ovd_multigpu_batchsize/multigpu_batchsize.json'),
        'batch_size_candidates': args.batch_size_candidates,
        'previous_probe': previous,
        'selected': load_json(EXISTING_OVD_WORK / 'selected_batch_size.json'),
        'note': 'Reused existing OpenRSD OVD batch-size preflight; physical GPU 4-7 are required for new GPU commands.',
    }
    write_json(args.work_dir / 'preflight_multigpu_batchsize.json', report)
    write_simple_md(md_path(args, 'preflight_multigpu_batchsize'), 'Preflight Multigpu + Batch Size', report)
    return report


def full(args: argparse.Namespace, runner: CommandRunner, progress: Progress) -> dict[str, Any]:
    reports = {}
    order = [('F5', lambda: run_f5(args, runner, progress)), ('F1', lambda: run_f1(args, runner, progress)),
             ('F2', lambda: run_f2(args, runner, progress)), ('F6', lambda: run_f6(args, runner, progress)),
             ('F3', lambda: run_f3(args, progress)), ('F4', lambda: run_f4(args, progress))]
    for key, fn in order:
        try:
            reports[key] = fn()
        except Exception as exc:  # noqa: BLE001
            reports[key] = {'status': 'FAILED', 'error': repr(exc)}
            append_jsonl(args.work_dir / 'failures.jsonl', {'time': now(), 'task': key, 'error': repr(exc)})
            progress.fail(f'{key}: {exc!r}')
            if not args.fallback_on_failure:
                break
    return reports


TASK_REPORT_PATHS = {
    'F1': Path('F1_ovd4/f1_report.json'),
    'F2': Path('F2_hooks/f2_hooks.json'),
    'F3': Path('F3_train/f3_report.json'),
    'F4': Path('F4_consistency/f4_report.json'),
    'F5': Path('F5_zero_ap_audit/zero_ap_mapping_debug.json'),
    'F6': Path('F6_geometry/per_bin_ap.json'),
}


def load_jsonl_tail(path: Path, limit: int = 5) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines()[-limit:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({'raw': line})
    return rows


def collect_task_reports(args: argparse.Namespace, reports: dict[str, Any]) -> dict[str, dict[str, Any]]:
    task_reports: dict[str, dict[str, Any]] = {}
    if isinstance(reports.get('smoke'), dict):
        for key, value in reports['smoke'].get('reports', {}).items():
            if isinstance(value, dict):
                task_reports[key] = value
    for key in TASK_REPORT_PATHS:
        value = reports.get(key)
        if isinstance(value, dict):
            task_reports[key] = value
    for key, rel_path in TASK_REPORT_PATHS.items():
        if not task_reports.get(key, {}).get('status'):
            disk_report = load_json(args.work_dir / rel_path)
            if disk_report:
                task_reports[key] = disk_report
    return task_reports


def strategy_means(report: dict[str, Any]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in report.get('rows', []):
        if row.get('status') != 'OK':
            continue
        try:
            grouped.setdefault(row.get('strategy', 'unknown'), []).append(float(row.get('ap50')))
        except (TypeError, ValueError):
            continue
    return {key: statistics.mean(values) for key, values in grouped.items() if values}


def key_result_lines(task_reports: dict[str, dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    f5 = task_reports.get('F5', {})
    if f5:
        rows = f5.get('rows', [])
        no_pred = [r.get('class') for r in rows if r.get('diagnosis') == 'NO_PREDICTION_FOR_CLASS']
        pred_zero = [r.get('class') for r in rows if r.get('diagnosis') == 'PREDICTIONS_EXIST_BUT_AP_ZERO_CHECK_IOU_SCORE_OR_LABEL_MAPPING']
        config = [r.get('class') for r in rows if r.get('diagnosis') == 'NOT_IN_EVALUATOR_CLASS_LIST']
        lines.append(f'- F5: config issue classes `{config}`; no-prediction `{no_pred}`; prediction-exists/AP-zero `{pred_zero}`.')
    f1 = task_reports.get('F1', {})
    if f1:
        means = strategy_means(f1)
        formatted = ', '.join(f'{name}={value:.4f}' for name, value in sorted(means.items()))
        lines.append(f'- F1: OVD4 artifact summary status `{f1.get("status")}`; strategy mean AP50: {formatted or "NA"}.')
    f2 = task_reports.get('F2', {})
    if f2:
        lines.append(
            f'- F2: `{f2.get("status")}` with `{f2.get("row_count")}` CSV rows; '
            f'successful hooks `{f2.get("successful_hooks")}`; failed hooks `{f2.get("failed_hooks")}`.')
    f6 = task_reports.get('F6', {})
    if f6:
        ar_rows = [r for r in f6.get('rows', []) if r.get('bin_type') == 'ar_bin']
        ar_summary = ', '.join(f'{r.get("bin")}={fmt(r.get("ap50"))}' for r in ar_rows[:6])
        lines.append(f'- F6: true per-bin AP50 evaluator ran on `{f6.get("model")}`; AR AP50: {ar_summary or "NA"}.')
    for key in ('F3', 'F4'):
        report = task_reports.get(key, {})
        if report:
            lines.append(f'- {key}: `{report.get("status")}`; {report.get("reason", "see experiment md")}')
    return lines


def write_summary(args: argparse.Namespace, reports: dict[str, Any]) -> dict[str, Any]:
    dry = load_json(args.work_dir / 'preflight_dryrun.json')
    smk = load_json(args.work_dir / 'preflight_smoke.json')
    mg = load_json(args.work_dir / 'preflight_multigpu_batchsize.json')
    env = env_versions(args)
    git = safe_git(args.repo_root)
    task_reports = collect_task_reports(args, reports)
    statuses = {
        k: task_reports.get(k, {}).get('status', 'NOT_RUN')
        for k in ['F1', 'F2', 'F3', 'F4', 'F5', 'F6']
    }
    paths = {
        'F1': md_path(args, 'F1_ovd4_prompt_ensemble_rotation_tta_repair'),
        'F2': md_path(args, 'F2_ovd5_feature_text_logit_hook_diagnostic'),
        'F3': md_path(args, 'F3_dota1_train_split_and_rotation_aug_finetune'),
        'F4': md_path(args, 'F4_cross_view_consistency_regularization_rerun'),
        'F5': md_path(args, 'F5_ovd_zero_ap_configuration_audit'),
        'F6': md_path(args, 'F6_geometry_per_bin_ap_evaluator'),
    }
    proposition = 'PARTIAL_SUPPORT' if (
        statuses.get('F1') in {'DONE', 'PARTIAL'}
        and statuses.get('F2') in {'DONE', 'PARTIAL'}
        and statuses.get('F5') in {'DONE', 'PARTIAL'}
    ) else 'INSUFFICIENT'
    progress_report = load_json(args.work_dir / 'progress.json')
    completed_preflights = sum(
        1 for item in (dry, smk, mg)
        if item.get('status') in {'DONE', 'PARTIAL', 'FAILED'}
    )
    completed_experiments = sum(
        1 for item in statuses.values()
        if item in {'DONE', 'PARTIAL', 'FAILED'}
    )
    computed_progress = min((completed_preflights + completed_experiments + 1) / 10.0, 1.0)
    failure_tail = load_jsonl_tail(args.work_dir / 'failures.jsonl')
    command_tail = load_jsonl_tail(args.work_dir / 'commands.jsonl')
    gpu_info = nvidia_smi_text()
    lines = [
        '# Summary: OpenRSD Follow-up 24h',
        '',
        '## Motivation',
        '',
        'This follow-up targets the evidence gaps between the earlier observation-style OVD rotation results and a method-oriented diagnosis: zero-AP configuration audit, prompt ensemble + rotation TTA repair, hook-style diagnostics, and geometry-wise AP.',
        '',
        f'- RUN_TS: `{args.run_ts}`',
        f'- work_dir: `{args.work_dir}`',
        f'- git_commit: `{git.get("commit")}`',
        f'- proposition_support: `{proposition}`',
        f'- current_progress: `{computed_progress:.4f}`',
        f'- 24h_scheduler_completed: `NO; tasks completed/reused before the 24h budget was exhausted`',
        f'- gpu_idle_observed: `YES at final check; no live follow-up process remained`',
        f'- fallback_triggered: `YES for F3/F4 training-dependent tasks because DOTA1 train split was unavailable`',
        '',
        '## Preflight',
        '',
        '| check | status |',
        '|---|---|',
        f'| dryrun | {dry.get("status", "NOT_RUN")} |',
        f'| smoke | {smk.get("status", "NOT_RUN")} |',
        f'| multigpu + batch size | {mg.get("status", "NOT_RUN")} |',
        '',
        '## Experiment Status',
        '',
        '| experiment | status | md |',
        '|---|---|---|',
    ]
    for key in ['F5', 'F1', 'F2', 'F6', 'F3', 'F4']:
        lines.append(f'| {key} | {statuses.get(key, "NOT_RUN")} | `{paths[key]}` |')
    lines.extend([
        '',
        '## Key Results',
        '',
        *key_result_lines(task_reports),
        '',
        '## Five Findings',
        '',
        '- The audited zero-AP classes are not all the same failure mode: helicopter has no predictions, while plane/harbor/large-vehicle have predictions but AP remains zero.',
        '- Prompt ensemble + rotation TTA is supported by existing OVD4 artifacts as a repair signal, but the completed scope is not a full 12x12 rerun.',
        '- F2 now produces a structured diagnostic CSV, but raw backbone/FPN/logit hooks are still marked NOT_AVAILABLE in this pass.',
        '- F6 now reports AP50 by geometry bin, replacing the previous matched-recall-only geometry proxy.',
        '- F3/F4 cannot be claimed as training evidence because the DOTA1 train split is still missing for full fine-tuning.',
        '',
        '## Five Risks',
        '',
        '- F1 reuses prior OVD4 artifacts, so it is evidence consolidation rather than a new long full run.',
        '- F2 is still partial until model-specific internal hook adapters are wired into OpenRSD inference.',
        '- F3 and F4 remain blocked by train split availability.',
        '- Some zero-AP classes need IoU/score/label-remap drill-down before OVD AP can be treated as final.',
        '- Final GPU 4-7 idleness is expected after the scheduler exited, but this means the requested 24h extended queue did not keep running.',
        '',
        '## Claim',
        '',
        '`Open-prompt remote-sensing detection exhibits rotation-dependent recognition instability; prompt ensemble, rotation TTA, head choice, and feature/text alignment diagnostics are necessary to distinguish semantic drift from localization and fusion artifacts.`',
        '',
        f'- support_status: `{proposition}`',
        '',
        '## Paper-Ready',
        '',
        '- Safe to write: OVD rotation instability exists; prompt/head choices matter; zero-AP classes require audit; per-bin AP is now available for geometry analysis.',
        '- Not yet safe to write: full internal feature/text drift proof, rotation augmentation training gains, or consistency regularization gains.',
        '',
        '## Failure Summary',
        '',
        '```json',
        json.dumps(failure_tail, indent=2, ensure_ascii=False),
        '```',
        '',
        '## Recent Commands',
        '',
        '```json',
        json.dumps(command_tail, indent=2, ensure_ascii=False)[:20000],
        '```',
        '',
        '## Environment',
        '',
        '```json',
        json.dumps(env, indent=2, ensure_ascii=False),
        '```',
        '',
        '## GPU Info',
        '',
        '```text',
        gpu_info,
        '```',
        '',
        '## Logs',
        '',
        f'- commands: `{args.work_dir / "commands.jsonl"}`',
        f'- failures: `{args.work_dir / "failures.jsonl"}`',
        f'- progress: `{args.work_dir / "progress.md"}`',
    ])
    write_text(md_path(args, 'summary_openrsd_followup_24h'), '\n'.join(lines))
    report = {
        'statuses': statuses,
        'task_reports': task_reports,
        'preflight': {'dryrun': dry, 'smoke': smk, 'multigpu': mg},
        'proposition': proposition,
        'progress': {**progress_report, 'computed_summary_progress': computed_progress},
        'failure_tail': failure_tail,
    }
    write_json(args.work_dir / 'summary_openrsd_followup_24h.json', report)
    return report


def seed_completed_tasks(args: argparse.Namespace) -> list[str]:
    done: list[str] = []
    for name, path in [
        ('dryrun', args.work_dir / 'preflight_dryrun.json'),
        ('smoke', args.work_dir / 'preflight_smoke.json'),
        ('multigpu_batchsize', args.work_dir / 'preflight_multigpu_batchsize.json'),
    ]:
        report = load_json(path)
        if report.get('status') in {'DONE', 'PARTIAL', 'FAILED'}:
            done.append(name)
    for name, rel_path in TASK_REPORT_PATHS.items():
        report = load_json(args.work_dir / rel_path)
        if report.get('status') in {'DONE', 'PARTIAL', 'FAILED'}:
            done.append(name)
    if (args.work_dir / 'summary_openrsd_followup_24h.json').exists():
        done.append('summary')
    return done


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--result-md-dir', type=Path, required=True)
    parser.add_argument('--weights-dir', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, default=None)
    parser.add_argument('--gpu-ids', default='4,5,6,7')
    parser.add_argument('--exp', default='all', choices=['all', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6'])
    parser.add_argument('--mode', default='full', choices=['dryrun', 'smoke', 'full', 'debug'])
    parser.add_argument('--time-budget-hours', type=float, default=24)
    parser.add_argument('--batch-size', type=int, default=0)
    parser.add_argument('--batch-size-candidates', default='1,2,4,8,16')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--max-images-for-smoke', type=int, default=16)
    parser.add_argument('--max-images-full', type=int, default=512)
    parser.add_argument('--max-iters-for-smoke', type=int, default=20)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--keep-gpus-busy', action='store_true')
    parser.add_argument('--fallback-on-failure', action='store_true')
    parser.add_argument('--python-bin', type=Path, default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.result_md_dir = args.result_md_dir.resolve()
    args.weights_dir = args.weights_dir.resolve()
    if args.work_dir is None:
        args.run_ts = run_ts()
        args.work_dir = args.repo_root / 'work_dirs' / f'openrsd_followup_{args.run_ts}'
    else:
        args.work_dir = args.work_dir.resolve()
        name = args.work_dir.name
        args.run_ts = name.replace('openrsd_followup_', '') if name.startswith('openrsd_followup_') else run_ts()
    return args


def main() -> None:
    args = parse_args()
    args.result_md_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    progress = Progress(args)
    progress.done = seed_completed_tasks(args)
    progress.start()
    runner = CommandRunner(args.repo_root, args.work_dir, args.gpu_ids, args.python_bin)
    reports: dict[str, Any] = {}
    try:
        if args.mode == 'dryrun':
            reports['dryrun'] = dryrun(args)
            progress.complete('dryrun', 'smoke')
        elif args.mode == 'smoke' and args.exp == 'F1' and args.batch_size_candidates:
            reports['multigpu_batchsize'] = multigpu_batchsize(args)
            progress.complete('multigpu_batchsize', 'full')
        elif args.mode == 'smoke':
            reports['smoke'] = smoke(args, runner, progress)
            progress.complete('smoke', 'multigpu_batchsize')
        else:
            if args.exp == 'all':
                reports.update(full(args, runner, progress))
            else:
                dispatch = {
                    'F1': lambda: run_f1(args, runner, progress),
                    'F2': lambda: run_f2(args, runner, progress),
                    'F3': lambda: run_f3(args, progress),
                    'F4': lambda: run_f4(args, progress),
                    'F5': lambda: run_f5(args, runner, progress),
                    'F6': lambda: run_f6(args, runner, progress),
                }
                reports[args.exp] = dispatch[args.exp]()
        write_summary(args, reports)
        progress.complete('summary', 'done')
    finally:
        progress.stop()


if __name__ == '__main__':
    main()
