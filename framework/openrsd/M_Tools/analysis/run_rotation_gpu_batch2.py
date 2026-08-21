#!/usr/bin/env python3
"""Unified runner for rotation GPU batch2 experiments.

The runner is conservative by design: it discovers local artifacts first,
records every launched command, reuses completed outputs unless --force is
set, and writes markdown even for partial or failed experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


ANGLES = ('000', '030', '060', '090', '120', '150', '180', '210', '240',
          '270', '300', '330')
ANGLES6 = ('000', '030', '060', '090', '120', '150')
DOTA_REL = Path('data/DOTA1_1024_500')
DOTA_SWEEP_REL = Path('angle_sweep_val/realistic')
DEFAULT_PYTHON = Path('/data/zcy/anaconda3/envs/openrsd/bin/python')


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display: str
    eval_config: Path
    train_config: Path
    checkpoint: Path
    priority: int


@dataclass
class CommandResult:
    name: str
    command: str
    returncode: int
    status: str
    stdout: str
    stderr: str
    duration_sec: float
    failure_kind: str = ''
    stdout_tail: str = ''
    stderr_tail: str = ''
    peak_mem_mb: dict[str, int] = field(default_factory=dict)
    process_seen: dict[str, bool] = field(default_factory=dict)


def now() -> str:
    return datetime.now().strftime('%F %T')


def log(msg: str) -> None:
    print(f'[{now()}] {msg}', flush=True)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == '':
        return 'NA'
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f) or math.isinf(f):
        return 'NA'
    return f'{f:.{digits}f}'


def md_escape(text: Any) -> str:
    return str(text).replace('|', '\\|').replace('\n', '<br>')


def csv_arg(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


def int_csv(value: str | None, default: Sequence[int]) -> list[int]:
    items = csv_arg(value)
    return [int(item) for item in items] if items else list(default)


def angle_csv(value: str | None, default: Sequence[str]) -> list[str]:
    items = csv_arg(value)
    if not items:
        return list(default)
    out = []
    for item in items:
        out.append(f'{int(item):03d}')
    return out


def shell_join(argv: Sequence[Any]) -> str:
    return ' '.join(shlex.quote(str(x)) for x in argv)


def read_text(path: Path, max_chars: int = 200000) -> str:
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except FileNotFoundError:
        return ''
    if len(text) > max_chars:
        return text[-max_chars:]
    return text


def tail(path: Path, n: int = 100) -> str:
    return '\n'.join(read_text(path).splitlines()[-n:])


def write_md(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return default


def detect_failure(text: str) -> str:
    lower = text.lower()
    checks = (
        ('cuda out of memory', 'oom'),
        ('out of memory', 'oom'),
        ('runtimeerror: cuda error', 'cuda_error'),
        ("keyerror: 'gt_instances'", 'gt_instances'),
        ('keyerror: "gt_instances"', 'gt_instances'),
        ('filenotfounderror', 'file_not_found'),
        ('checkpoint missing', 'checkpoint_missing'),
        ('checkpoint not found', 'checkpoint_missing'),
        ('failed to load checkpoint', 'checkpoint_missing'),
        ('could not load checkpoint', 'checkpoint_missing'),
        ('no such file', 'file_not_found'),
        ('config missing', 'config_missing'),
        ('ann_file', 'annotation_path_missing'),
        ('annfiles', 'annotation_path_missing'),
        ('img_path', 'image_path_missing'),
        ('predictions.pkl', 'predictions_missing'),
        ('img_id', 'img_id_alignment'),
        ('loss is nan', 'loss_nan'),
        ('loss is inf', 'loss_inf'),
        ('nan', 'loss_nan'),
        ('traceback (most recent call last)', 'traceback'),
    )
    for needle, kind in checks:
        if needle in lower:
            return kind
    return ''


class GpuMonitor:
    def __init__(self, gpu_ids: str, interval: float = 1.0):
        self.gpu_ids = [x.strip() for x in gpu_ids.split(',') if x.strip()]
        self.peak_mem = {gpu: 0 for gpu in self.gpu_ids}
        self.process_seen = {gpu: False for gpu in self.gpu_ids}
        self.interval = interval
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def _run(self, args: Sequence[str]) -> str:
        proc = subprocess.run(
            ['rtk', 'nvidia-smi', *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False)
        if proc.returncode != 0:
            return ''
        return proc.stdout

    def snapshot(self) -> None:
        gpu_text = self._run([
            '--query-gpu=index,uuid,memory.used',
            '--format=csv,noheader,nounits',
        ])
        uuid_to_index: dict[str, str] = {}
        for line in gpu_text.splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 3:
                continue
            index, uuid, mem = parts[:3]
            uuid_to_index[uuid] = index
            if index in self.peak_mem:
                try:
                    self.peak_mem[index] = max(self.peak_mem[index], int(float(mem)))
                except ValueError:
                    pass
        proc_text = self._run([
            '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
            '--format=csv,noheader,nounits',
        ])
        for line in proc_text.splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 4:
                continue
            index = uuid_to_index.get(parts[0])
            if index in self.process_seen and 'python' in parts[2].lower():
                self.process_seen[index] = True

    def loop(self) -> None:
        while not self.stop.is_set():
            self.snapshot()
            self.stop.wait(self.interval)

    def __enter__(self) -> 'GpuMonitor':
        self.snapshot()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        self.snapshot()


class CommandRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo_root = args.repo_root
        self.log_root = args.work_dir / 'logs'
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.commands_jsonl = args.work_dir / 'commands.jsonl'

    def env_assignments(self, gpu_ids: str | None = None) -> list[str]:
        gpu_ids = gpu_ids or self.args.gpu_ids
        return [
            'PYTHONNOUSERSITE=1',
            'MPLCONFIGDIR=/tmp/mplconfig',
            f'CUDA_VISIBLE_DEVICES={gpu_ids}',
            f'PYTHONPATH={self.args.repo_root}:{self.args.repo_root / "tools"}',
            'NCCL_P2P_DISABLE=1',
            'NCCL_IB_DISABLE=1',
        ]

    def command_string(self, argv: Sequence[Any], gpu_ids: str | None = None) -> str:
        return 'rtk env ' + shell_join(self.env_assignments(gpu_ids)) + ' ' + shell_join(argv)

    def save_smi(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            ['rtk', 'nvidia-smi'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False)
        path.write_text(proc.stdout, encoding='utf-8', errors='replace')

    def run(self,
            name: str,
            argv: Sequence[Any],
            log_dir: Path,
            gpu_ids: str | None = None,
            monitor_gpu: bool = True,
            skip_if: Path | None = None) -> CommandResult:
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout = log_dir / 'stdout.log'
        stderr = log_dir / 'stderr.log'
        command_txt = log_dir / 'command.txt'
        command = self.command_string(argv, gpu_ids)
        command_txt.write_text(command + '\n', encoding='utf-8')
        if skip_if is not None and skip_if.exists() and self.args.resume and not self.args.force:
            result = CommandResult(name, command, 0, 'SKIPPED', str(stdout), str(stderr), 0.0)
            self.append_result(result)
            return result

        self.save_smi(log_dir / 'nvidia_smi_before.txt')
        env_cmd = ['rtk', 'env', *self.env_assignments(gpu_ids), *map(str, argv)]
        start = time.time()
        log(f'RUN {name}: {command}')
        mon_context = GpuMonitor(gpu_ids or self.args.gpu_ids) if monitor_gpu else None
        with stdout.open('w', encoding='utf-8') as out, stderr.open('w', encoding='utf-8') as err:
            if mon_context is None:
                proc = subprocess.run(
                    env_cmd, cwd=str(self.repo_root), stdout=out, stderr=err, text=True)
                peak = {}
                seen = {}
            else:
                with mon_context as mon:
                    proc = subprocess.run(
                        env_cmd, cwd=str(self.repo_root), stdout=out, stderr=err, text=True)
                    peak = dict(mon.peak_mem)
                    seen = dict(mon.process_seen)
        duration = time.time() - start
        self.save_smi(log_dir / 'nvidia_smi_after.txt')
        stdout_tail = tail(stdout)
        stderr_tail = tail(stderr)
        failure_kind = detect_failure(stdout_tail + '\n' + stderr_tail)
        status = 'OK' if proc.returncode == 0 else f'FAILED({proc.returncode})'
        result = CommandResult(
            name=name,
            command=command,
            returncode=proc.returncode,
            status=status,
            stdout=str(stdout),
            stderr=str(stderr),
            duration_sec=duration,
            failure_kind=failure_kind,
            stdout_tail=stdout_tail if proc.returncode else '',
            stderr_tail=stderr_tail if proc.returncode else '',
            peak_mem_mb=peak,
            process_seen=seen)
        self.append_result(result)
        log(f'DONE {name}: {status} stdout={stdout} stderr={stderr}')
        return result

    def append_result(self, result: CommandResult) -> None:
        payload = {
            'time': now(),
            'name': result.name,
            'command': result.command,
            'returncode': result.returncode,
            'status': result.status,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'duration_sec': result.duration_sec,
            'failure_kind': result.failure_kind,
            'peak_mem_mb': result.peak_mem_mb,
            'process_seen': result.process_seen,
        }
        with self.commands_jsonl.open('a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False) + '\n')


def existing_python() -> Path:
    if DEFAULT_PYTHON.exists():
        return DEFAULT_PYTHON
    return Path(sys.executable)


def build_specs(repo_root: Path, weights_dir: Path) -> dict[str, ModelSpec]:
    return {
        'rtmdet_l': ModelSpec(
            'rtmdet_l',
            'RTMDet-L',
            repo_root / 'M_configs/RotationStudy/rotated_rtmdet_l_dota1_ms_eval.py',
            repo_root / 'mmrotate_configs/rotated_rtmdet/rotated_rtmdet_l-3x-dota_ms.py',
            weights_dir / 'rotated_rtmdet_l-3x-dota_ms-2738da34.pth',
            1),
        'h2rbox_v2': ModelSpec(
            'h2rbox_v2',
            'H2RBox-v2',
            repo_root / 'M_configs/RotationStudy/h2rbox_v2_r50_fpn_dota1_ms_rr_eval.py',
            repo_root / 'mmrotate_configs/h2rbox/h2rbox-le90_r50_fpn_adamw-1x_dota-ms.py',
            weights_dir / 'h2rbox_v2-le90_r50_fpn_ms_rr-1x_dota-5e0e53e1.pth',
            2),
        'retinanet': ModelSpec(
            'retinanet',
            'Rotated RetinaNet R50',
            repo_root / 'M_configs/RotationStudy/rotated_retinanet_r50_msrr_dota1_eval.py',
            repo_root / 'mmrotate_configs/rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_rr-1x_dota-ms.py',
            weights_dir / 'rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90-1da1ec9c.pth',
            3),
    }


def selected_models(args: argparse.Namespace, include_retina: bool = False) -> list[str]:
    if args.only_model:
        return [x for x in csv_arg(args.only_model)]
    models = ['rtmdet_l', 'h2rbox_v2']
    if include_retina:
        models.append('retinanet')
    return models


def discover_fair_root(repo_root: Path) -> tuple[Path | None, list[Path]]:
    candidates = [
        repo_root / 'data/FAR1M',
        repo_root / 'data/far1m',
        repo_root / 'data/FAR1M_1024_500',
        repo_root / 'data/FAR1M_1024',
        repo_root / 'data/fair1m/dair1m_1024',
    ]
    found = [p for p in candidates if p.exists()]
    data = repo_root / 'data'
    if data.exists():
        for path in data.rglob('*'):
            if path.is_dir() and ('far' in path.name.lower() or 'fair' in path.name.lower()):
                if path not in found:
                    found.append(path)
    preferred = None
    for path in found:
        if (path / 'train/annfiles').exists() or (path / 'rot_val_standard/realistic/angle_000').exists():
            preferred = path
            break
    if preferred is None and found:
        preferred = found[0]
    return preferred, found


def discover_dota_train(repo_root: Path) -> tuple[Path | None, list[Path]]:
    candidates = [
        repo_root / 'data/DOTA1_1024_500/ss_train',
        repo_root / 'data/DOTA1_1024_500/train',
        repo_root / 'data/DOTA1_1024_500/trainval',
        repo_root / 'data/DOTAV1/train',
    ]
    found = []
    for path in candidates:
        if (path / 'annfiles').exists() and (path / 'images').exists():
            found.append(path)
    return (found[0] if found else None), found


def find_configs(repo_root: Path, tokens: Sequence[str]) -> list[Path]:
    roots = [repo_root / 'M_configs', repo_root / 'mmrotate_configs', repo_root / 'mmyolo_configs']
    out = []
    lowered = [t.lower() for t in tokens]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob('*.py'):
            hay = str(path.relative_to(repo_root)).lower()
            if all(t in hay for t in lowered):
                out.append(path)
    return sorted(out)


def find_weights(roots: Sequence[Path], tokens: Sequence[str]) -> list[Path]:
    out = []
    lowered = [t.lower() for t in tokens]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob('*.pth'):
            hay = str(path).lower()
            if all(t in hay for t in lowered):
                out.append(path)
    return sorted(out)


def path_state(path: Path) -> str:
    return 'OK' if path.exists() else 'MISSING'


def ann_count(path: Path) -> int:
    return len(list(path.glob('*.txt'))) if path.exists() else 0


def parse_metrics(log_path: Path) -> dict[str, float | None]:
    text = read_text(log_path)

    def last(pattern: str) -> float | None:
        hits = re.findall(pattern, text)
        if not hits:
            return None
        try:
            return float(hits[-1])
        except ValueError:
            return None

    return {
        'ap50': last(r'["\']?dota/AP50["\']?\s*[:=]\s*([0-9]*\.?[0-9]+)'),
        'map': last(r'["\']?dota/mAP["\']?\s*[:=]\s*([0-9]*\.?[0-9]+)'),
    }


def load_pickle_count(path: Path) -> int | None:
    try:
        with path.open('rb') as f:
            return len(pickle.load(f))
    except Exception:
        return None


def create_subset(src_root: Path, out_root: Path, max_images: int,
                  ids: list[str] | None = None) -> tuple[Path, list[str]]:
    ann_src = src_root / 'annfiles'
    img_src = src_root / 'images'
    if ids is None:
        ids = []
        for ann in sorted(ann_src.glob('*.txt')):
            if len(ids) >= max_images:
                break
            if ann.read_text(encoding='utf-8', errors='replace').strip():
                ids.append(ann.stem)
    ann_dst = out_root / 'annfiles'
    img_dst = out_root / 'images'
    if out_root.exists():
        shutil.rmtree(out_root)
    ann_dst.mkdir(parents=True, exist_ok=True)
    img_dst.mkdir(parents=True, exist_ok=True)
    for stem in ids[:max_images]:
        os.symlink(ann_src / f'{stem}.txt', ann_dst / f'{stem}.txt')
        image = None
        for ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
            cand = img_src / f'{stem}{ext}'
            if cand.exists():
                image = cand
                break
        if image is None:
            raise FileNotFoundError(f'image missing for {stem} under {img_src}')
        os.symlink(image, img_dst / image.name)
    return out_root, ids[:max_images]


def pipeline_for_variant(variant: str, img_scale: tuple[int, int] = (1024, 1024)) -> list[dict[str, Any]]:
    base: list[dict[str, Any]] = [
        dict(type='mmdet.LoadImageFromFile', file_client_args=dict(backend='disk')),
        dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
        dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
        dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
        dict(type='mmdet.RandomFlip', prob=0.75, direction=['horizontal', 'vertical', 'diagonal']),
    ]
    if variant == 'no_rotate':
        pass
    elif variant == 'rr_discrete':
        base.append(dict(
            type='RandomChoiceRotate',
            angles=[30, 60, 90, 120, 150],
            prob=5.0 / 6.0,
            rect_obj_labels=[9, 11]))
    elif variant == 'rr_dense':
        base.append(dict(
            type='RandomChoiceRotate',
            angles=[15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165],
            prob=11.0 / 12.0,
            rect_obj_labels=[9, 11]))
    else:
        base.append(dict(type='RandomRotate', prob=1.0, angle_range=180, rect_obj_labels=[9, 11]))
    base.extend([
        dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
        dict(type='mmdet.PackDetInputs'),
    ])
    return base


def write_train_config(path: Path,
                       base_config: Path,
                       checkpoint: Path,
                       data_root: Path,
                       variant: str,
                       batch_size: int,
                       num_workers: int,
                       max_iters: int,
                       lr_mult: float = 1.0,
                       consistency: bool = False,
                       lambda_box: float = 0.0,
                       img_scale: tuple[int, int] = (1024, 1024),
                       metainfo: str | None = None) -> Path:
    pipeline = repr(pipeline_for_variant(variant, img_scale))
    custom = "custom_imports = dict(imports=['M_Tools.analysis.cross_view_consistency'], allow_failed_imports=False)\n" if consistency else ''
    model_type = "model = dict(type='CrossViewConsistencyRTMDet', backbone=dict(init_cfg=None), lambda_cls=0.1, lambda_box=%s)\n" % lambda_box if consistency else "model = dict(backbone=dict(init_cfg=None))\n"
    metainfo_line = f'metainfo={metainfo},\n                ' if metainfo else ''
    text = f"""# Auto-generated by run_rotation_gpu_batch2.py
_base_ = '{base_config}'
{custom}{model_type}
load_from = '{checkpoint}'
resume = False
train_pipeline = {pipeline}
train_dataloader = dict(
    batch_size={batch_size},
    num_workers={num_workers},
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type='DOTADataset',
        data_root='{data_root}',
        {metainfo_line}ann_file='annfiles/',
        data_prefix=dict(img_path='images/'),
        img_shape={img_scale},
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))
train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters={max_iters}, val_interval={max_iters + 1})
val_cfg = None
val_dataloader = None
val_evaluator = None
base_lr = (0.004 / 16) * {lr_mult}
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
param_scheduler = [dict(type='LinearLR', start_factor=1.0, by_epoch=False, begin=0, end={max_iters})]
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval={max_iters}, save_last=True, max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=False)
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


def train_command(args: argparse.Namespace, config: Path, work_dir: Path,
                  distributed: bool, master_port: int) -> list[Any]:
    if distributed:
        nproc = len(csv_arg(args.gpu_ids))
        return [
            args.python_bin, '-m', 'torch.distributed.launch',
            f'--nproc_per_node={nproc}', f'--master_port={master_port}',
            'tools/train.py', config, '--launcher', 'pytorch', '--work-dir', work_dir,
        ]
    return [args.python_bin, 'tools/train.py', config, '--work-dir', work_dir]


def test_command(args: argparse.Namespace,
                 spec: ModelSpec | Path,
                 checkpoint: Path,
                 out_dir: Path,
                 pred: Path,
                 data_root: Path,
                 ann_file: str,
                 img_path: str,
                 batch_size: int,
                 num_workers: int,
                 distributed: bool,
                 master_port: int) -> list[Any]:
    config = spec.eval_config if isinstance(spec, ModelSpec) else spec
    opts = [
        f'test_dataloader.batch_size={batch_size}',
        f'test_dataloader.num_workers={num_workers}',
        'test_dataloader.persistent_workers=False',
        f'test_dataloader.dataset.data_root={data_root}',
        f'test_dataloader.dataset.ann_file={ann_file}',
        f'test_dataloader.dataset.data_prefix.img_path={img_path}',
    ]
    if distributed:
        nproc = len(csv_arg(args.gpu_ids))
        return [
            args.python_bin, '-m', 'torch.distributed.launch',
            f'--nproc_per_node={nproc}', f'--master_port={master_port}',
            'tools/openrsd_test.py', config, checkpoint, '--launcher', 'pytorch',
            '--work-dir', out_dir, '--out', pred, '--cfg-options', *opts,
        ]
    return [
        args.python_bin, 'tools/openrsd_test.py', config, checkpoint,
        '--work-dir', out_dir, '--out', pred, '--cfg-options', *opts,
    ]


def eval_command(args: argparse.Namespace,
                 config: Path,
                 pred: Path,
                 data_root: Path,
                 ann_file: str,
                 img_path: str) -> list[Any]:
    return [
        args.python_bin, 'tools/openrsd_eval_metric.py', config, pred,
        '--cfg-options',
        f'test_dataloader.num_workers={args.num_workers}',
        'test_dataloader.persistent_workers=False',
        f'test_dataloader.dataset.data_root={data_root}',
        f'test_dataloader.dataset.ann_file={ann_file}',
        f'test_dataloader.dataset.data_prefix.img_path={img_path}',
    ]


def latest_checkpoint(work_dir: Path) -> Path | None:
    candidates = sorted(work_dir.glob('*.pth'), key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1]
    last = work_dir / 'last_checkpoint'
    if last.exists():
        text = last.read_text(encoding='utf-8', errors='replace').strip()
        p = Path(text)
        if not p.is_absolute():
            p = work_dir / p
        if p.exists():
            return p
    return None


def run_dryrun(args: argparse.Namespace) -> dict[str, Any]:
    args.result_md_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    tmp = args.result_md_dir / '.gpu_batch2_write_test'
    writable = True
    try:
        tmp.write_text('ok', encoding='utf-8')
        tmp.unlink()
    except OSError:
        writable = False

    specs = build_specs(args.repo_root, args.weights_dir)
    dota = args.repo_root / DOTA_REL
    fair_root, fair_found = discover_fair_root(args.repo_root)
    dota_train, dota_train_found = discover_dota_train(args.repo_root)
    failures: list[str] = []
    warnings: list[str] = []

    required_paths = {
        'repo_root': args.repo_root,
        'weights_dir': args.weights_dir,
        'work_dir': args.work_dir,
        'dota1_data': dota,
        'tools/train.py': args.repo_root / 'tools/train.py',
        'tools/test.py': args.repo_root / 'tools/test.py',
        'tools/dist_train.sh': args.repo_root / 'tools/dist_train.sh',
        'tools/dist_test.sh': args.repo_root / 'tools/dist_test.sh',
        'tools/openrsd_test.py': args.repo_root / 'tools/openrsd_test.py',
        'tools/openrsd_eval_metric.py': args.repo_root / 'tools/openrsd_eval_metric.py',
    }
    for name, path in required_paths.items():
        if not path.exists():
            failures.append(f'{name} missing: {path}')
    if not writable:
        failures.append(f'result md dir not writable: {args.result_md_dir}')
    missing_angles = []
    for angle in ANGLES:
        root = dota / DOTA_SWEEP_REL / f'angle_{angle}'
        if not (root / 'annfiles').exists() or not (root / 'images').exists():
            missing_angles.append(angle)
    if missing_angles:
        failures.append(f'DOTA1 angle sweep missing: {missing_angles}')
    if dota_train is None:
        warnings.append('DOTA1 train split was not found; DOTA1 fine-tuning variants will be NOT_RUN.')
    if fair_root is None:
        warnings.append('FAR1M/FAIR1M data root was not found; Experiment 8 will be NOT_RUN.')

    config_hits = {
        'rtmdet_l_dota1': find_configs(args.repo_root, ['rtmdet', 'dota']),
        'h2rbox_v2_dota1': find_configs(args.repo_root, ['h2rbox', 'dota']),
        'retinanet_dota1': find_configs(args.repo_root, ['retinanet', 'dota']),
        'far1m': find_configs(args.repo_root, ['fair1m']),
    }
    weight_hits = {
        'rtmdet_l': find_weights([args.weights_dir, args.repo_root / 'work_dirs'], ['rtmdet', 'dota']),
        'h2rbox_v2': find_weights([args.weights_dir, args.repo_root / 'work_dirs'], ['h2rbox', 'dota']),
        'retinanet': find_weights([args.weights_dir, args.repo_root / 'work_dirs'], ['retinanet', 'dota']),
        'far1m': find_weights([args.weights_dir, args.repo_root / 'work_dirs'], ['fair1m']),
    }
    for spec in specs.values():
        if not spec.eval_config.exists():
            failures.append(f'{spec.key} eval config missing: {spec.eval_config}')
        if not spec.train_config.exists():
            warnings.append(f'{spec.key} train config missing: {spec.train_config}')
        if not spec.checkpoint.exists():
            failures.append(f'{spec.key} checkpoint missing: {spec.checkpoint}')

    transform_text = read_text(args.repo_root / 'mmrotate/datasets/transforms/transforms.py')
    rotate_tools = {
        'RandomRotate': 'class RandomRotate' in transform_text,
        'RandomChoiceRotate': 'class RandomChoiceRotate' in transform_text,
    }
    if not rotate_tools['RandomRotate']:
        failures.append('RandomRotate transform not found.')

    runner = CommandRunner(args)
    rtmdet = specs['rtmdet_l']
    planned = [
        runner.command_string(train_command(
            args, Path('<generated_rtmdet_short_ft_rr_discrete.py>'),
            args.work_dir / 'exp5/rtmdet_l/short_ft_rr_discrete', True, 39101), args.gpu_ids),
        runner.command_string(test_command(
            args, rtmdet, rtmdet.checkpoint, args.work_dir / 'exp5/rtmdet_l/baseline/angle_000',
            args.work_dir / 'exp5/rtmdet_l/baseline/angle_000/predictions.pkl',
            dota, 'angle_sweep_val/realistic/angle_000/annfiles/',
            'angle_sweep_val/realistic/angle_000/images/',
            args.batch_size or 1, args.num_workers, True, 39200), args.gpu_ids),
        runner.command_string([
            args.python_bin, 'M_Tools/analysis/run_rotation_gpu_batch2.py',
            '--repo-root', args.repo_root, '--result-md-dir', args.result_md_dir,
            '--weights-dir', args.weights_dir, '--work-dir', args.work_dir,
            '--gpu-ids', args.gpu_ids, '--exp', '7', '--mode', 'full',
        ], args.gpu_ids),
    ]
    status = 'FAILED' if failures else ('PASS_WITH_WARNINGS' if warnings else 'PASS')
    report = {
        'status': status,
        'failures': failures,
        'warnings': warnings,
        'dota_train': str(dota_train) if dota_train else None,
        'dota_train_found': [str(p) for p in dota_train_found],
        'fair_root': str(fair_root) if fair_root else None,
        'fair_found': [str(p) for p in fair_found],
        'config_hits': {k: [str(p) for p in v[:12]] for k, v in config_hits.items()},
        'weight_hits': {k: [str(p) for p in v[:12]] for k, v in weight_hits.items()},
        'rotate_tools': rotate_tools,
        'planned_commands': planned,
    }
    write_json(args.work_dir / 'preflight/dryrun.json', report)
    lines = [
        '# GPU Batch2 Dryrun Preflight',
        '',
        f'- generated_at: `{now()}`',
        f'- status: `{status}`',
        f'- repo_root: `{args.repo_root}`',
        f'- result_md_dir_writable: `{writable}`',
        f'- work_dir: `{args.work_dir}`',
        '',
        '## Required Paths',
        '',
        '| item | status | path |',
        '|---|---|---|',
    ]
    for name, path in required_paths.items():
        lines.append(f'| {name} | {path_state(path)} | `{path}` |')
    lines.extend(['', '## DOTA1 Angle Sweep', '', '| angle | annfiles | images | ann count |'])
    lines.append('|---:|---|---|---:|')
    for angle in ANGLES:
        root = dota / DOTA_SWEEP_REL / f'angle_{angle}'
        lines.append(f'| {angle} | {path_state(root / "annfiles")} | {path_state(root / "images")} | {ann_count(root / "annfiles")} |')
    lines.extend([
        '',
        '## DOTA1 Train Discovery',
        '',
        f'- selected: `{dota_train}`',
        f'- found: `{", ".join(str(p) for p in dota_train_found)}`',
        '',
        '## FAR1M Discovery',
        '',
        f'- selected: `{fair_root}`',
        f'- found: `{", ".join(str(p) for p in fair_found)}`',
    ])
    lines.extend(['', '## Model Artifacts', '', '| model | eval config | train config | checkpoint |'])
    lines.append('|---|---|---|---|')
    for spec in specs.values():
        lines.append(f'| {spec.display} | `{spec.eval_config}` ({path_state(spec.eval_config)}) | `{spec.train_config}` ({path_state(spec.train_config)}) | `{spec.checkpoint}` ({path_state(spec.checkpoint)}) |')
    lines.extend(['', '## Search Hits', '', '| query | config hits | weight hits |'])
    lines.append('|---|---:|---:|')
    for key in sorted(set(config_hits) | set(weight_hits)):
        lines.append(f'| {key} | {len(config_hits.get(key, []))} | {len(weight_hits.get(key, []))} |')
    lines.extend(['', '## Training And Evaluation Tools', '', '| tool | status |'])
    lines.append('|---|---|')
    for tool in ['tools/train.py', 'tools/test.py', 'tools/dist_train.sh', 'tools/dist_test.sh', 'tools/openrsd_test.py', 'tools/openrsd_eval_metric.py']:
        lines.append(f'| {tool} | {path_state(args.repo_root / tool)} |')
    lines.extend(['', '## Rotate / Angle Tools', '', '| tool | status |'])
    lines.append('|---|---|')
    for key, ok in rotate_tools.items():
        lines.append(f'| {key} | {"OK" if ok else "MISSING"} |')
    lines.extend(['', '## Planned Full Commands', ''])
    for cmd in planned:
        lines.append(f'- `{cmd}`')
    if failures:
        lines.extend(['', '## Failures', ''])
        lines.extend(f'- {item}' for item in failures)
    if warnings:
        lines.extend(['', '## Warnings', ''])
        lines.extend(f'- {item}' for item in warnings)
    write_md(args.result_md_dir / 'preflight_gpu_batch2_dryrun.md', lines)
    update_summary(args)
    return report


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    specs = build_specs(args.repo_root, args.weights_dir)
    spec = specs['rtmdet_l']
    runner = CommandRunner(args)
    angle = (angle_csv(args.only_angle, ['000']) or ['000'])[0]
    src = args.repo_root / DOTA_REL / DOTA_SWEEP_REL / f'angle_{angle}'
    subset, ids = create_subset(src, args.work_dir / 'preflight/smoke/subsets' / f'angle_{angle}', args.max_images_for_smoke)
    train_dir = args.work_dir / 'preflight/smoke/train_rtmdet_l'
    cfg = write_train_config(
        train_dir / 'smoke_train_rtmdet_l.py',
        spec.train_config,
        spec.checkpoint,
        subset,
        'rr_continuous',
        args.batch_size or 1,
        args.num_workers,
        args.max_iters_for_smoke)
    train_res = runner.run(
        'smoke_train_rtmdet_l_20iter',
        train_command(args, cfg, train_dir, False, 0),
        args.work_dir / 'logs/preflight_smoke/train_rtmdet_l',
        gpu_ids=args.gpu_ids,
        monitor_gpu=True)
    ckpt = latest_checkpoint(train_dir) or spec.checkpoint
    test_dir = args.work_dir / 'preflight/smoke/test_rtmdet_l_angle_000'
    pred = test_dir / 'predictions.pkl'
    test_res = runner.run(
        'smoke_test_rtmdet_l_angle_000',
        test_command(args, spec, ckpt, test_dir, pred, subset, 'annfiles/', 'images/',
                     1, args.num_workers, False, 0),
        args.work_dir / 'logs/preflight_smoke/test_rtmdet_l_angle_000',
        gpu_ids=args.gpu_ids,
        monitor_gpu=True,
        skip_if=pred)
    eval_dir = args.work_dir / 'preflight/smoke/eval_rtmdet_l_angle_000'
    eval_res = runner.run(
        'smoke_eval_rtmdet_l_angle_000',
        eval_command(args, spec.eval_config, pred, subset, 'annfiles/', 'images/'),
        args.work_dir / 'logs/preflight_smoke/eval_rtmdet_l_angle_000',
        gpu_ids=args.gpu_ids,
        monitor_gpu=False)
    metrics = parse_metrics(Path(eval_res.stdout))
    feat_json = args.work_dir / 'preflight/smoke/feature_hook.json'
    feat_res = runner.run(
        'smoke_feature_hook_rtmdet_l_angle_000',
        [
            args.python_bin, 'M_Tools/analysis/run_rotation_gpu_batch2.py',
            '--repo-root', args.repo_root,
            '--result-md-dir', args.result_md_dir,
            '--weights-dir', args.weights_dir,
            '--work-dir', args.work_dir,
            '--gpu-ids', args.gpu_ids,
            '--exp', '7',
            '--mode', 'debug',
            '--only-model', 'rtmdet_l',
            '--only-angle', angle,
            '--max-images-for-smoke', min(args.max_images_for_smoke, 8),
            '--batch-size', '1',
            '--num-workers', str(args.num_workers),
            '--resume',
        ],
        args.work_dir / 'logs/preflight_smoke/feature_hook_rtmdet_l',
        gpu_ids=args.gpu_ids,
        monitor_gpu=True)
    feature_ok = feat_json.exists() and (load_json(feat_json, {}) or {}).get('status') == 'OK'
    pred_count = load_pickle_count(pred)
    status = 'OK'
    failures = []
    for item in (train_res, test_res, eval_res, feat_res):
        if item.returncode != 0:
            failures.append(item)
    if failures or pred_count is None or metrics.get('ap50') is None or not feature_ok:
        status = 'FAILED'
    report = {
        'status': status,
        'model': spec.key,
        'angle': angle,
        'subset': str(subset),
        'subset_count': len(ids),
        'train_checkpoint': str(ckpt),
        'predictions': str(pred),
        'prediction_count': pred_count,
        'ap50': metrics.get('ap50'),
        'train_command': train_res.command,
        'test_command': test_res.command,
        'eval_command': eval_res.command,
        'feature_command': feat_res.command,
        'feature_ok': feature_ok,
        'failures': [x.__dict__ for x in failures],
    }
    write_json(args.work_dir / 'preflight/smoke.json', report)
    lines = [
        '# GPU Batch2 Smoke Preflight',
        '',
        f'- generated_at: `{now()}`',
        f'- status: `{status}`',
        f'- physical GPU ids: `{args.gpu_ids}`',
        f'- model: `{spec.display}`',
        f'- angle: `{angle}`',
        f'- subset: `{subset}`',
        f'- smoke train checkpoint: `{ckpt}`',
        '',
        '## Checks',
        '',
        '| check | status | artifact/log |',
        '|---|---|---|',
        f'| train loop 20 iter | {train_res.status} | `{train_res.stdout}` / `{train_res.stderr}` |',
        f'| checkpoint saved | {"OK" if latest_checkpoint(train_dir) else "FAILED"} | `{train_dir}` |',
        f'| test predictions.pkl | {test_res.status} | `{pred}` count={pred_count} |',
        f'| evaluator AP50 | {eval_res.status} | AP50={fmt(metrics.get("ap50"))} log=`{eval_res.stdout}` |',
        f'| feature hook | {"OK" if feature_ok else "FAILED"} | `{feat_json}` |',
        '',
        '## Commands',
        '',
        f'- train: `{train_res.command}`',
        f'- test: `{test_res.command}`',
        f'- eval: `{eval_res.command}`',
        f'- feature: `{feat_res.command}`',
    ]
    if failures:
        lines.extend(['', '## Failures', ''])
        for item in failures:
            lines.append(f'- {item.name}: {item.status} kind={item.failure_kind} stdout=`{item.stdout}` stderr=`{item.stderr}`')
    write_md(args.result_md_dir / 'preflight_gpu_batch2_smoke.md', lines)
    update_summary(args)
    return report


def parse_loss_finite(log_path: Path) -> bool:
    text = read_text(log_path).lower()
    values = re.findall(
        r'\bloss(?:_[a-z0-9_]+)?\s*:\s*([+-]?(?:nan|inf|[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?))',
        text)
    if not values:
        return False
    for value in values:
        try:
            if not math.isfinite(float(value)):
                return False
        except ValueError:
            return False
    return True


def run_multigpu_batchsize(args: argparse.Namespace) -> dict[str, Any]:
    specs = build_specs(args.repo_root, args.weights_dir)
    spec = specs['rtmdet_l']
    runner = CommandRunner(args)
    src = args.repo_root / DOTA_REL / DOTA_SWEEP_REL / 'angle_000'
    subset, _ = create_subset(src, args.work_dir / 'preflight/multigpu/subset_angle_000',
                              max(args.max_images_for_smoke, 32))
    rows = []
    selected = None
    candidates = int_csv(args.batch_size_candidates, [1, 2, 4, 8])
    previous_report = load_json(args.work_dir / 'preflight/multigpu_batchsize.json', {'rows': []})
    previous_by_bs = {int(row.get('batch_size')): row for row in previous_report.get('rows', []) if row.get('batch_size') is not None}
    oom_seen = False
    for i, bs in enumerate(candidates):
        if oom_seen:
            rows.append({'batch_size': bs, 'status': 'SKIPPED_AFTER_OOM'})
            continue
        run_dir = args.work_dir / 'preflight/multigpu' / f'bs_{bs}'
        cfg = write_train_config(
            run_dir / f'rtmdet_l_bs{bs}.py', spec.train_config, spec.checkpoint,
            subset, 'rr_continuous', bs, args.num_workers, args.max_iters_for_smoke)
        log_dir = args.work_dir / 'logs/preflight_multigpu' / f'bs_{bs}'
        existing_ckpt = latest_checkpoint(run_dir)
        if existing_ckpt and args.resume and not args.force and (log_dir / 'stdout.log').exists():
            prev = previous_by_bs.get(bs, {})
            res = CommandResult(
                name=f'multigpu_train_rtmdet_l_bs{bs}',
                command='RESUME existing multigpu preflight artifact',
                returncode=0,
                status='SKIPPED',
                stdout=str(log_dir / 'stdout.log'),
                stderr=str(log_dir / 'stderr.log'),
                duration_sec=0.0,
                peak_mem_mb=prev.get('peak_mem_mb', {}),
                process_seen=prev.get('process_seen', {}))
        else:
            res = runner.run(
                f'multigpu_train_rtmdet_l_bs{bs}',
                train_command(args, cfg, run_dir, True, 39400 + i),
                log_dir,
                gpu_ids=args.gpu_ids,
                monitor_gpu=True)
        ckpt = latest_checkpoint(run_dir)
        finite = parse_loss_finite(Path(res.stdout))
        status = 'OK' if res.returncode == 0 and ckpt and finite else 'FAILED'
        if res.failure_kind == 'oom':
            status = 'OOM'
            oom_seen = True
        if status == 'OK':
            selected = bs
        rows.append({
            'batch_size': bs,
            'status': status,
            'checkpoint': str(ckpt) if ckpt else '',
            'loss_finite': finite,
            'stdout': res.stdout,
            'stderr': res.stderr,
            'peak_mem_mb': res.peak_mem_mb,
            'process_seen': res.process_seen,
            'duration_sec': res.duration_sec,
            'failure_kind': res.failure_kind,
        })
    if selected is None:
        selected = 1
    report = {
        'status': 'OK' if any(r.get('status') == 'OK' for r in rows) else 'FAILED',
        'selected_batch_size': selected,
        'gpu_ids': args.gpu_ids,
        'rows': rows,
    }
    write_json(args.work_dir / 'preflight/multigpu_batchsize.json', report)
    write_json(args.work_dir / 'preflight/selected_batch_size.json', {'rtmdet_l': selected})
    lines = [
        '# GPU Batch2 Multi-GPU Batch Size Preflight',
        '',
        f'- generated_at: `{now()}`',
        f'- status: `{report["status"]}`',
        f'- physical GPU ids: `{args.gpu_ids}`',
        '- distributed training: `python -m torch.distributed.launch`',
        f'- selected_batch_size: `{selected}`',
        '',
        '| batch size | status | loss finite | checkpoint | peak mem MB | python process seen | stdout | stderr |',
        '|---:|---|---|---|---|---|---|---|',
    ]
    for row in rows:
        lines.append(f'| {row.get("batch_size")} | {row.get("status")} | {row.get("loss_finite", "")} | `{row.get("checkpoint", "")}` | `{row.get("peak_mem_mb", {})}` | `{row.get("process_seen", {})}` | `{row.get("stdout", "")}` | `{row.get("stderr", "")}` |')
    write_md(args.result_md_dir / 'preflight_gpu_batch2_multigpu_batchsize.md', lines)
    update_summary(args)
    return report


def selected_batch(args: argparse.Namespace, default: int = 1) -> int:
    if args.batch_size:
        return args.batch_size
    data = load_json(args.work_dir / 'preflight/selected_batch_size.json', {})
    try:
        return int(data.get('rtmdet_l', default))
    except Exception:
        return default


def iter_limit_for_epochs(train_root: Path,
                          batch_size: int,
                          gpu_ids: str,
                          max_epochs: int,
                          iter_cap: int) -> int:
    count = ann_count(train_root / 'annfiles')
    world = max(1, len(csv_arg(gpu_ids)))
    per_iter = max(1, batch_size * world)
    iters_per_epoch = max(1, math.ceil(max(count, 1) / per_iter))
    return min(iter_cap, max(1, max_epochs) * iters_per_epoch)


def eval_angle(args: argparse.Namespace,
               runner: CommandRunner,
               spec: ModelSpec | Path,
               checkpoint: Path,
               data_root: Path,
               angle_root_rel: str,
               angle: str,
               out_dir: Path,
               batch_size: int,
               distributed: bool) -> dict[str, Any]:
    pred = out_dir / 'predictions.pkl'
    ann_file = f'{angle_root_rel}/angle_{angle}/annfiles/'
    img_path = f'{angle_root_rel}/angle_{angle}/images/'
    test_res = runner.run(
        f'test_{Path(str(checkpoint)).stem}_angle_{angle}',
        test_command(args, spec, checkpoint, out_dir, pred, data_root, ann_file, img_path,
                     batch_size, args.num_workers, distributed, 39500 + int(angle)),
        args.work_dir / 'logs' / out_dir.relative_to(args.work_dir),
        gpu_ids=args.gpu_ids,
        monitor_gpu=True,
        skip_if=pred)
    eval_res = runner.run(
        f'eval_{Path(str(checkpoint)).stem}_angle_{angle}',
        eval_command(args, spec.eval_config if isinstance(spec, ModelSpec) else spec,
                     pred, data_root, ann_file, img_path),
        args.work_dir / 'logs' / out_dir.relative_to(args.work_dir) / 'eval',
        gpu_ids=args.gpu_ids,
        monitor_gpu=False,
        skip_if=args.work_dir / 'logs' / out_dir.relative_to(args.work_dir) / 'eval' / 'stdout.log')
    metrics = parse_metrics(Path(eval_res.stdout))
    status = 'OK' if test_res.returncode == 0 and eval_res.returncode == 0 and metrics.get('ap50') is not None else 'FAILED'
    return {
        'angle': angle,
        'status': status,
        'ap50': metrics.get('ap50'),
        'map': metrics.get('map'),
        'predictions': str(pred),
        'test_stdout': test_res.stdout,
        'test_stderr': test_res.stderr,
        'eval_stdout': eval_res.stdout,
        'eval_stderr': eval_res.stderr,
        'test_command': test_res.command,
        'eval_command': eval_res.command,
        'failure_kind': test_res.failure_kind or eval_res.failure_kind,
    }


def summarize_angle_rows(rows: list[dict[str, Any]], baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    vals = [r['ap50'] for r in rows if r.get('status') == 'OK' and r.get('ap50') is not None]
    if not vals:
        return {'status': 'FAILED'}
    worst = min(vals)
    best = max(vals)
    mean = statistics.mean(vals)
    out = {
        'mean_ap50': mean,
        'worst_ap50': worst,
        'best_ap50': best,
        'std': statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        'rsi': worst / max(mean, 1e-12),
        'completed': len(vals),
    }
    canonical = next((r.get('ap50') for r in rows if r.get('angle') == '000' and r.get('ap50') is not None), None)
    out['canonical_ap50'] = canonical
    if baseline:
        out['canonical_damage'] = (canonical - baseline.get('canonical_ap50')) if canonical is not None and baseline.get('canonical_ap50') is not None else None
        out['worst_gain'] = worst - baseline.get('worst_ap50') if baseline.get('worst_ap50') is not None else None
    return out


def run_experiment_5(args: argparse.Namespace) -> dict[str, Any]:
    specs = build_specs(args.repo_root, args.weights_dir)
    runner = CommandRunner(args)
    angles = angle_csv(args.only_angle, ANGLES if args.mode == 'full' else ANGLES6)
    batch = selected_batch(args)
    dota_root = args.repo_root / DOTA_REL
    train_root, _ = discover_dota_train(args.repo_root)
    models = [m for m in selected_models(args) if m in specs]
    variants = ['baseline', 'short_ft_rr_discrete', 'short_ft_rr_dense', 'short_ft_rr_continuous']
    report: dict[str, Any] = {'status': 'DONE', 'models': {}, 'train_root': str(train_root) if train_root else None}
    for model in models:
        spec = specs[model]
        model_rows: dict[str, Any] = {}
        baseline_summary = None
        for variant in variants:
            variant_dir = args.work_dir / 'exp5' / model / variant
            checkpoint = spec.checkpoint
            train_status = 'NOT_RUN'
            train_command_text = ''
            pipeline_variant = {
                'baseline': 'no further training',
                'short_ft_rr_discrete': 'RandomChoiceRotate angles=[30,60,90,120,150], 0 retained by non-rotate probability',
                'short_ft_rr_dense': 'RandomChoiceRotate angles=[15..165 step 15], 0 retained by non-rotate probability',
                'short_ft_rr_continuous': 'RandomRotate angle_range=180',
            }[variant]
            if variant != 'baseline':
                if train_root is None:
                    train_status = 'NOT_RUN'
                elif not spec.train_config.exists():
                    train_status = 'NOT_RUN'
                else:
                    cfg_variant = {
                        'short_ft_rr_discrete': 'rr_discrete',
                        'short_ft_rr_dense': 'rr_dense',
                        'short_ft_rr_continuous': 'rr_continuous',
                    }[variant]
                    max_iters = (args.max_iters_for_smoke if args.mode != 'full' else
                                 iter_limit_for_epochs(train_root, batch, args.gpu_ids, args.max_epochs, 12000))
                    cfg = write_train_config(
                        variant_dir / f'{model}_{variant}.py', spec.train_config, spec.checkpoint,
                        train_root, cfg_variant, batch, args.num_workers, max_iters)
                    res = runner.run(
                        f'exp5_train_{model}_{variant}',
                        train_command(args, cfg, variant_dir, len(csv_arg(args.gpu_ids)) > 1, 39600 + len(model_rows)),
                        args.work_dir / 'logs/exp5' / model / variant / 'train',
                        gpu_ids=args.gpu_ids,
                        monitor_gpu=True)
                    train_status = 'OK' if res.returncode == 0 and latest_checkpoint(variant_dir) else 'FAILED'
                    train_command_text = res.command
                    if train_status == 'OK':
                        checkpoint = latest_checkpoint(variant_dir) or checkpoint
                    elif res.failure_kind == 'oom' and batch > 1:
                        retry_bs = max(1, batch // 2)
                        cfg = write_train_config(
                            variant_dir / f'{model}_{variant}_retry_bs{retry_bs}.py',
                            spec.train_config, spec.checkpoint, train_root, cfg_variant,
                            retry_bs, args.num_workers, max_iters)
                        retry = runner.run(
                            f'exp5_train_{model}_{variant}_retry_bs{retry_bs}',
                            train_command(args, cfg, variant_dir, len(csv_arg(args.gpu_ids)) > 1, 39700 + len(model_rows)),
                            args.work_dir / 'logs/exp5' / model / variant / f'train_retry_bs{retry_bs}',
                            gpu_ids=args.gpu_ids,
                            monitor_gpu=True)
                        train_status = 'OK' if retry.returncode == 0 and latest_checkpoint(variant_dir) else 'FAILED'
                        train_command_text += '\n' + retry.command
                        if train_status == 'OK':
                            checkpoint = latest_checkpoint(variant_dir) or checkpoint
            rows = []
            if variant == 'baseline' or train_status == 'OK':
                for angle in angles:
                    try:
                        rows.append(eval_angle(
                            args, runner, spec, checkpoint, dota_root,
                            str(DOTA_SWEEP_REL), angle, variant_dir / f'angle_{angle}',
                            batch, len(csv_arg(args.gpu_ids)) > 1))
                    except Exception as exc:
                        rows.append({'angle': angle, 'status': 'FAILED', 'error': str(exc), 'traceback': traceback.format_exc()})
            summary = summarize_angle_rows(rows, baseline_summary)
            if variant == 'baseline':
                baseline_summary = summary
            model_rows[variant] = {
                'train_status': train_status,
                'checkpoint': str(checkpoint),
                'pipeline_diff': pipeline_variant,
                'train_command': train_command_text,
                'rows': rows,
                'summary': summary,
            }
        report['models'][model] = model_rows
    if any(vv['train_status'] in ('FAILED', 'NOT_RUN') for m in report['models'].values() for k, vv in m.items() if k != 'baseline'):
        report['status'] = 'PARTIAL'
    write_json(args.work_dir / 'exp5/exp5_results.json', report)
    write_exp5_md(args, report, batch)
    update_summary(args)
    return report


def write_exp5_md(args: argparse.Namespace, report: dict[str, Any], batch: int) -> None:
    lines = [
        '# Experiment 5: DOTA1 Rotation Augmentation Short Fine-Tuning',
        '',
        f'- generated_at: `{now()}`',
        f'- status: `{report["status"]}`',
        '- metric: AP50 / mAP@0.5',
        f'- train_root: `{report.get("train_root")}`',
        f'- final batch size: `{batch}`',
        '',
        '## Purpose',
        '',
        'Compare baseline evaluation with short fine-tuning variants to test whether training-time rotation augmentation improves worst-angle robustness without damaging canonical AP50.',
    ]
    for model, variants in report['models'].items():
        lines.extend(['', f'## {model}', '', '| variant | train status | checkpoint | pipeline diff |'])
        lines.append('|---|---|---|---|')
        for variant, payload in variants.items():
            lines.append(f'| {variant} | {payload["train_status"]} | `{payload["checkpoint"]}` | {md_escape(payload["pipeline_diff"])} |')
        lines.extend(['', '### Angle-Wise AP50', ''])
        for variant, payload in variants.items():
            lines.extend([f'#### {variant}', '', '| angle | status | AP50 | eval log | predictions |'])
            lines.append('|---:|---|---:|---|---|')
            for row in payload['rows']:
                lines.append(f'| {row.get("angle")} | {row.get("status")} | {fmt(row.get("ap50"))} | `{row.get("eval_stdout", "")}` | `{row.get("predictions", "")}` |')
            if not payload['rows']:
                lines.append('| NA | NOT_RUN | NA |  |  |')
            lines.append('')
        lines.extend(['### Variant Statistics', '', '| variant | completed | mean AP50 | worst AP50 | best AP50 | std | RSI | canonical damage | worst gain |'])
        lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
        best_variant = None
        best_worst = -1.0
        for variant, payload in variants.items():
            s = payload['summary']
            lines.append(f'| {variant} | {s.get("completed", 0)} | {fmt(s.get("mean_ap50"))} | {fmt(s.get("worst_ap50"))} | {fmt(s.get("best_ap50"))} | {fmt(s.get("std"))} | {fmt(s.get("rsi"))} | {fmt(s.get("canonical_damage"))} | {fmt(s.get("worst_gain"))} |')
            if s.get('worst_ap50') is not None and s.get('worst_ap50') > best_worst:
                best_worst = s['worst_ap50']
                best_variant = variant
        lines.extend(['', '### Conclusion', ''])
        if best_variant:
            lines.append(f'- Best worst-angle AP50 variant: `{best_variant}` ({fmt(best_worst)}).')
        if report.get('train_root') is None:
            lines.append('- DOTA1 fine-tuning variants were NOT_RUN because no DOTA1 train split with `annfiles/` and `images/` was found.')
    lines.extend(['', '## Reproduction', '', f'- `rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mplconfig CUDA_VISIBLE_DEVICES={args.gpu_ids} PYTHONPATH={args.repo_root}:{args.repo_root / "tools"} {args.python_bin} M_Tools/analysis/run_rotation_gpu_batch2.py --repo-root {args.repo_root} --result-md-dir {args.result_md_dir} --weights-dir {args.weights_dir} --work-dir {args.work_dir} --gpu-ids {args.gpu_ids} --exp 5 --mode full --resume`'])
    write_md(args.result_md_dir / 'exp5_dota1_rotation_aug_short_finetune.md', lines)


def run_experiment_6(args: argparse.Namespace) -> dict[str, Any]:
    specs = build_specs(args.repo_root, args.weights_dir)
    spec = specs['rtmdet_l']
    runner = CommandRunner(args)
    batch = selected_batch(args)
    train_root, _ = discover_dota_train(args.repo_root)
    variants = [('baseline_short_ft', False, 0.0), ('image_pair_consistency_cls', True, 0.0),
                ('image_pair_consistency_cls_box', True, 0.05)]
    report: dict[str, Any] = {'status': 'DONE', 'train_root': str(train_root) if train_root else None, 'variants': {}}
    if train_root is None:
        report['status'] = 'NOT_RUN'
        for name, consistency, lambda_box in variants:
            report['variants'][name] = {
                'status': 'NOT_RUN',
                'reason': 'DOTA1 train split missing',
                'consistency': consistency,
                'lambda_cls': 0.1 if consistency else 0.0,
                'lambda_box': lambda_box,
                'rows': [],
                'summary': {},
            }
    else:
        angles = angle_csv(args.only_angle, ANGLES if args.mode == 'full' else ANGLES6)
        for i, (name, consistency, lambda_box) in enumerate(variants):
            run_dir = args.work_dir / 'exp6' / name
            cfg = write_train_config(
                run_dir / f'{name}.py', spec.train_config, spec.checkpoint, train_root,
                'rr_discrete', batch, args.num_workers,
                (args.max_iters_for_smoke if args.mode != 'full' else
                 iter_limit_for_epochs(train_root, batch, args.gpu_ids, 1, 6000)),
                consistency=consistency, lambda_box=lambda_box)
            res = runner.run(
                f'exp6_train_{name}',
                train_command(args, cfg, run_dir, len(csv_arg(args.gpu_ids)) > 1, 39800 + i),
                args.work_dir / 'logs/exp6' / name / 'train',
                gpu_ids=args.gpu_ids,
                monitor_gpu=True)
            ckpt = latest_checkpoint(run_dir)
            status = 'OK' if res.returncode == 0 and ckpt else 'FAILED'
            rows = []
            if status == 'OK' and ckpt:
                for angle in angles:
                    rows.append(eval_angle(
                        args, runner, spec, ckpt, args.repo_root / DOTA_REL,
                        str(DOTA_SWEEP_REL), angle, run_dir / f'angle_{angle}',
                        batch, len(csv_arg(args.gpu_ids)) > 1))
            report['variants'][name] = {
                'status': status,
                'checkpoint': str(ckpt) if ckpt else '',
                'consistency': consistency,
                'stop_gradient': consistency,
                'box_consistency': lambda_box > 0,
                'lambda_cls': 0.1 if consistency else 0.0,
                'lambda_box': lambda_box,
                'train_command': res.command,
                'train_stdout': res.stdout,
                'train_stderr': res.stderr,
                'rows': rows,
                'summary': summarize_angle_rows(rows),
            }
        if any(v['status'] != 'OK' for v in report['variants'].values()):
            report['status'] = 'PARTIAL'
    write_json(args.work_dir / 'exp6/exp6_results.json', report)
    write_exp6_md(args, report)
    update_summary(args)
    return report


def write_exp6_md(args: argparse.Namespace, report: dict[str, Any]) -> None:
    exp5 = load_json(args.work_dir / 'exp5/exp5_results.json', {})
    lines = [
        '# Experiment 6: Cross-View Consistency Regularization Prototype',
        '',
        f'- generated_at: `{now()}`',
        f'- status: `{report["status"]}`',
        '- implementation: `M_Tools/analysis/cross_view_consistency.py`',
        '- stop-gradient: teacher/original-view head outputs are detached.',
        '- prototype limitation: cls consistency is implemented in dense-head logit space; box consistency is an optional bbox-head-output MSE, not full geometric inverse matching.',
        f'- train_root: `{report.get("train_root")}`',
        '',
        '## Variants',
        '',
        '| variant | status | lambda_cls | lambda_box | stop-gradient | box consistency | checkpoint |',
        '|---|---|---:|---:|---|---|---|',
    ]
    for name, payload in report['variants'].items():
        lines.append(f'| {name} | {payload.get("status")} | {fmt(payload.get("lambda_cls"))} | {fmt(payload.get("lambda_box"))} | {payload.get("stop_gradient", "")} | {payload.get("box_consistency", "")} | `{payload.get("checkpoint", "")}` |')
    lines.extend(['', '## Angle-Wise AP50', ''])
    for name, payload in report['variants'].items():
        lines.extend([f'### {name}', '', '| angle | status | AP50 | eval log |'])
        lines.append('|---:|---|---:|---|')
        for row in payload.get('rows', []):
            lines.append(f'| {row.get("angle")} | {row.get("status")} | {fmt(row.get("ap50"))} | `{row.get("eval_stdout", "")}` |')
        if not payload.get('rows'):
            lines.append(f'| NA | {payload.get("status")} | NA | {md_escape(payload.get("reason", ""))} |')
    lines.extend(['', '## Statistics', '', '| variant | completed | mean AP50 | worst AP50 | std | RSI |'])
    lines.append('|---|---:|---:|---:|---:|---:|')
    for name, payload in report['variants'].items():
        s = payload.get('summary', {})
        lines.append(f'| {name} | {s.get("completed", 0)} | {fmt(s.get("mean_ap50"))} | {fmt(s.get("worst_ap50"))} | {fmt(s.get("std"))} | {fmt(s.get("rsi"))} |')
    lines.extend([
        '',
        '## Comparison To Experiment 5',
        '',
        f'- exp5_json: `{args.work_dir / "exp5/exp5_results.json"}`',
        f'- exp5_status: `{exp5.get("status", "NA")}`',
        '',
        '## Failure Notes',
    ])
    for name, payload in report['variants'].items():
        if payload.get('status') != 'OK':
            lines.append(f'- {name}: {payload.get("reason", payload.get("train_stderr", "see logs"))}')
    write_md(args.result_md_dir / 'exp6_cross_view_consistency_regularization_prototype.md', lines)


def setup_openrsd_imports(repo_root: Path) -> None:
    for path in (repo_root, repo_root / 'tools'):
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)
    from openrsd_env import preload_installed_mmengine
    preload_installed_mmengine()
    from mmdet.utils import register_all_modules as register_mmdet
    from mmrotate.utils import register_all_modules as register_mmrotate
    register_mmdet(init_default_scope=False)
    register_mmrotate(init_default_scope=False)


def tensor_vec(value: Any, inverse_angle: float = 0.0):
    import torch
    import torch.nn.functional as F
    tensors = []

    def collect(x):
        if torch.is_tensor(x):
            y = x.detach().float().cpu()
            if y.ndim == 4:
                y = y[0]
            if y.ndim == 3:
                if abs(inverse_angle) > 1e-6:
                    yy = y.unsqueeze(0)
                    rad = torch.tensor(math.radians(inverse_angle), dtype=yy.dtype)
                    theta = torch.tensor([[[torch.cos(rad), -torch.sin(rad), 0.0],
                                           [torch.sin(rad), torch.cos(rad), 0.0]]], dtype=yy.dtype)
                    grid = F.affine_grid(theta, yy.size(), align_corners=False)
                    yy = F.grid_sample(yy, grid, align_corners=False)
                    y = yy[0]
                y = y.flatten()
            else:
                y = y.flatten()
            if y.numel() > 20000:
                y = y[::max(1, y.numel() // 20000)]
            tensors.append(y)
        elif isinstance(x, (list, tuple)):
            for item in x:
                collect(item)
        elif isinstance(x, dict):
            for item in x.values():
                collect(item)

    collect(value)
    if not tensors:
        return None
    min_len = min(t.numel() for t in tensors)
    tensors = [t[:min_len] for t in tensors if t.numel() >= min_len]
    return torch.cat(tensors)


def tensor_vecs(value: Any, inverse_angle: float = 0.0, batch_size: int = 1):
    import torch
    import torch.nn.functional as F

    per_sample: list[list[Any]] = [[] for _ in range(batch_size)]

    def flatten_one(y):
        if y.ndim == 3:
            if abs(inverse_angle) > 1e-6:
                yy = y.unsqueeze(0)
                rad = torch.tensor(math.radians(inverse_angle), dtype=yy.dtype)
                theta = torch.tensor([[[torch.cos(rad), -torch.sin(rad), 0.0],
                                       [torch.sin(rad), torch.cos(rad), 0.0]]], dtype=yy.dtype)
                grid = F.affine_grid(theta, yy.size(), align_corners=False)
                yy = F.grid_sample(yy, grid, align_corners=False)
                y = yy[0]
            y = y.flatten()
        else:
            y = y.flatten()
        if y.numel() > 20000:
            y = y[::max(1, y.numel() // 20000)]
        return y

    def collect(x):
        if torch.is_tensor(x):
            y = x.detach().float().cpu()
            if y.ndim >= 1 and y.shape[0] == batch_size and (batch_size > 1 or y.ndim == 4):
                items = [y[i] for i in range(batch_size)]
            elif y.ndim == 4:
                items = [y[0]] + [None] * (batch_size - 1)
            else:
                items = [y] + [None] * (batch_size - 1)
            for idx, item in enumerate(items[:batch_size]):
                if item is not None:
                    per_sample[idx].append(flatten_one(item))
        elif isinstance(x, (list, tuple)):
            for item in x:
                collect(item)
        elif isinstance(x, dict):
            for item in x.values():
                collect(item)

    collect(value)
    out = []
    for tensors in per_sample:
        if not tensors:
            out.append(None)
            continue
        min_len = min(t.numel() for t in tensors)
        tensors = [t[:min_len] for t in tensors if t.numel() >= min_len]
        out.append(torch.cat(tensors))
    return out


def detach_capture(value: Any):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, list):
        return [detach_capture(item) for item in value]
    if isinstance(value, tuple):
        return tuple(detach_capture(item) for item in value)
    if isinstance(value, dict):
        return {key: detach_capture(item) for key, item in value.items()}
    return value


def snapshot_captures(captures: dict[str, Any]) -> dict[str, Any]:
    return {key: detach_capture(value) for key, value in captures.items()}


def build_fast_test_pipeline(model):
    from mmcv.transforms import Compose
    from mmdet.utils import get_test_pipeline_cfg

    cfg = model.cfg.copy()
    return Compose(get_test_pipeline_cfg(cfg))


def inference_detector_batch(model, imgs: Sequence[Path], test_pipeline):
    import torch

    data_list = []
    for img in imgs:
        data = test_pipeline(dict(img_path=str(img), img_id=0))
        data_list.append(data)
    batch_data = {
        'inputs': [data['inputs'] for data in data_list],
        'data_samples': [data['data_samples'] for data in data_list],
    }
    with torch.no_grad():
        return model.test_step(batch_data)


def cos_l2(a, b) -> tuple[float | None, float | None]:
    import torch
    if a is None or b is None:
        return None, None
    n = min(a.numel(), b.numel())
    if n == 0:
        return None, None
    a = a[:n]
    b = b[:n]
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    l2 = torch.norm(a - b, p=2).item() / math.sqrt(n)
    return cos, l2


def run_feature_diagnostic(args: argparse.Namespace, debug: bool = False) -> dict[str, Any]:
    setup_openrsd_imports(args.repo_root)
    from mmdet.apis import init_detector
    import torch.nn.functional as F
    specs = build_specs(args.repo_root, args.weights_dir)
    model_keys = [m for m in selected_models(args, include_retina=True) if m in specs]
    if args.only_angle:
        angles = angle_csv(args.only_angle, [args.only_angle])
    else:
        angles = ['000'] if debug else (list(ANGLES) if args.mode == 'full' else list(ANGLES6))
    max_images = args.max_images_for_smoke if args.mode in ('smoke', 'debug') or debug else 512
    batch_size = max(1, args.batch_size or 1)
    out_dir = args.work_dir / ('preflight/smoke' if debug else 'exp7')
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    for key in model_keys:
        spec = specs[key]
        try:
            model = init_detector(str(spec.eval_config), str(spec.checkpoint), device='cuda:0')
            model.eval()
            test_pipeline = build_fast_test_pipeline(model)
        except Exception as exc:
            failures.append({'model': key, 'stage': 'init', 'error': str(exc), 'traceback': traceback.format_exc()})
            continue
        captures: dict[str, Any] = {}
        hook_layers = {}
        candidates = [
            ('backbone_last', getattr(model, 'backbone', None)),
            ('neck_fpn', getattr(model, 'neck', None)),
            ('head_outputs', getattr(model, 'bbox_head', None)),
        ]
        handles = []
        for name, module in candidates:
            if module is None:
                continue
            hook_layers[name] = module.__class__.__name__

            def make_hook(layer_name):
                def hook(_module, _inputs, output):
                    captures[layer_name] = output
                return hook
            handles.append(module.register_forward_hook(make_hook(name)))
        try:
            stems = [p.stem for p in sorted((args.repo_root / DOTA_REL / DOTA_SWEEP_REL / 'angle_000/annfiles').glob('*.txt'))[:max_images]]
            for angle in angles:
                log(f'exp7 {key} angle_{angle}: {len(stems)} stems, batch_size={batch_size}')
                per_layer = {name: [] for name in hook_layers}
                logit_kls = []
                top1_cons = []
                conf_drops = []
                pairs = []
                for stem in stems:
                    img0 = None
                    imgr = None
                    for ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
                        c0 = args.repo_root / DOTA_REL / DOTA_SWEEP_REL / 'angle_000/images' / f'{stem}{ext}'
                        cr = args.repo_root / DOTA_REL / DOTA_SWEEP_REL / f'angle_{angle}/images' / f'{stem}{ext}'
                        if c0.exists():
                            img0 = c0
                        if cr.exists():
                            imgr = cr
                    if img0 is None or imgr is None:
                        continue
                    pairs.append((img0, imgr))
                for start in range(0, len(pairs), batch_size):
                    batch_pairs = pairs[start:start + batch_size]
                    current_batch = len(batch_pairs)
                    captures.clear()
                    pred0_list = inference_detector_batch(model, [p[0] for p in batch_pairs], test_pipeline)
                    cap0 = snapshot_captures(captures)
                    captures.clear()
                    predr_list = inference_detector_batch(model, [p[1] for p in batch_pairs], test_pipeline)
                    capr = snapshot_captures(captures)
                    captures.clear()
                    for layer in hook_layers:
                        vec0s = tensor_vecs(cap0.get(layer), batch_size=current_batch)
                        vecrs = tensor_vecs(capr.get(layer), -float(int(angle)), current_batch)
                        for v0, vr in zip(vec0s, vecrs):
                            c, l2 = cos_l2(v0, vr)
                            if c is not None:
                                per_layer[layer].append((c, l2))
                    v0s = tensor_vecs(cap0.get('head_outputs'), batch_size=current_batch)
                    vrs = tensor_vecs(capr.get('head_outputs'), batch_size=current_batch)
                    for v0, vr in zip(v0s, vrs):
                        if v0 is None or vr is None:
                            continue
                        n = min(v0.numel(), vr.numel())
                        if n > 1:
                            try:
                                logit_kls.append(
                                    F.kl_div(
                                        F.log_softmax(v0[:n].float(), dim=0),
                                        F.softmax(vr[:n].float(), dim=0),
                                        reduction='batchmean').item())
                            except Exception:
                                pass
                    for pred0, predr in zip(pred0_list, predr_list):
                        try:
                            s0 = pred0.pred_instances.scores.detach().cpu()
                            sr = predr.pred_instances.scores.detach().cpu()
                            l0 = pred0.pred_instances.labels.detach().cpu()
                            lr = predr.pred_instances.labels.detach().cpu()
                            if len(s0) and len(sr):
                                i0 = int(s0.argmax())
                                ir = int(sr.argmax())
                                top1_cons.append(float(l0[i0] == lr[ir]))
                                conf_drops.append(float(s0[i0] - sr[ir]))
                        except Exception:
                            pass
                    del cap0, capr, pred0_list, predr_list
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                for layer, vals in per_layer.items():
                    if vals:
                        l2_vals = [v[1] for v in vals if v[1] is not None]
                        rows.append({
                            'model': key,
                            'angle': angle,
                            'layer': layer,
                            'hook_module': hook_layers[layer],
                            'feature_cosine': statistics.mean(v[0] for v in vals),
                            'feature_l2': statistics.mean(l2_vals) if l2_vals else None,
                            'top1_class_consistency': statistics.mean(top1_cons) if top1_cons else None,
                            'confidence_drop': statistics.mean(conf_drops) if conf_drops else None,
                            'logit_kl': statistics.mean(logit_kls) if logit_kls else None,
                            'sample_count': len(vals),
                        })
        except Exception as exc:
            failures.append({'model': key, 'stage': 'diagnostic', 'error': str(exc), 'traceback': traceback.format_exc()})
        finally:
            for handle in handles:
                handle.remove()
    csv_path = out_dir / 'feature_logit_equivariance.csv'
    if rows:
        with csv_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    report = {
        'status': 'OK' if rows and not failures else ('PARTIAL' if rows else 'FAILED'),
        'csv': str(csv_path),
        'rows': rows,
        'failures': failures,
    }
    write_json(out_dir / ('feature_hook.json' if debug else 'exp7_results.json'), report)
    if not debug:
        write_exp7_md(args, report)
        update_summary(args)
    return report


def write_exp7_md(args: argparse.Namespace, report: dict[str, Any]) -> None:
    rows = report.get('rows', [])
    lines = [
        '# Experiment 7: Feature / Logit Equivariance Diagnostic',
        '',
        f'- generated_at: `{now()}`',
        f'- status: `{report["status"]}`',
        f'- raw_csv: `{report.get("csv")}`',
        '- metric: feature cosine, normalized L2, top-1 class consistency, confidence drop.',
        '',
        '## Hook Layers',
        '',
        '| model | layer | hook module |',
        '|---|---|---|',
    ]
    seen = set()
    for row in rows:
        key = (row['model'], row['layer'])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f'| {row["model"]} | {row["layer"]} | {row["hook_module"]} |')
    lines.extend(['', '## Feature Similarity', '', '| model | angle | layer | cosine | L2 | samples |'])
    lines.append('|---|---:|---|---:|---:|---:|')
    for row in rows:
        lines.append(f'| {row["model"]} | {row["angle"]} | {row["layer"]} | {fmt(row.get("feature_cosine"))} | {fmt(row.get("feature_l2"))} | {row.get("sample_count")} |')
    lines.extend(['', '## Logit / Detection Consistency', '', '| model | angle | layer | top-1 consistency | confidence drop | logit KL |'])
    lines.append('|---|---:|---|---:|---:|---:|')
    for row in rows:
        lines.append(f'| {row["model"]} | {row["angle"]} | {row["layer"]} | {fmt(row.get("top1_class_consistency"))} | {fmt(row.get("confidence_drop"))} | {fmt(row.get("logit_kl"))} |')
    lines.extend(['', '## Collapse Layer Judgment', ''])
    for model in sorted({r['model'] for r in rows}):
        model_rows = [r for r in rows if r['model'] == model]
        low = [r for r in model_rows if r.get('feature_cosine') is not None]
        if low:
            worst = min(low, key=lambda r: r['feature_cosine'])
            lines.append(f'- {model}: earliest observed weak layer is approximated as `{worst["layer"]}` at angle_{worst["angle"]} (cosine {fmt(worst["feature_cosine"])}).')
    if report.get('failures'):
        lines.extend(['', '## Failed Hooks', ''])
        for fail in report['failures']:
            lines.append(f'- {fail.get("model")} {fail.get("stage")}: {md_escape(fail.get("error"))}')
    lines.extend(['', '## Reproduction', '', f'- `rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mplconfig CUDA_VISIBLE_DEVICES={args.gpu_ids} PYTHONPATH={args.repo_root}:{args.repo_root / "tools"} {args.python_bin} M_Tools/analysis/run_rotation_gpu_batch2.py --repo-root {args.repo_root} --result-md-dir {args.result_md_dir} --weights-dir {args.weights_dir} --work-dir {args.work_dir} --gpu-ids {args.gpu_ids} --exp 7 --mode full --resume`'])
    write_md(args.result_md_dir / 'exp7_feature_logit_equivariance_diagnostic.md', lines)


def fair_metainfo_literal() -> str:
    classes = ['a220', 'a321', 'a330', 'a350', 'arj21',
               'baseball_field', 'basketball_court', 'boeing737', 'boeing747', 'boeing777', 'boeing787',
               'bridge', 'bus', 'c919', 'cargo_truck', 'dry_cargo_ship', 'dump_truck',
               'engineering_ship', 'excavator', 'fishing_boat', 'football_field', 'intersection',
               'liquid_cargo_ship', 'motorboat', 'other-airplane', 'other-ship', 'other-vehicle',
               'passenger_ship', 'roundabout',
               'small_car', 'tennis_court', 'tractor', 'trailer', 'truck_tractor', 'tugboat', 'van', 'warship']
    return repr(dict(classes=classes, palette=[(220, 20, 60)]))


def fair_config(args: argparse.Namespace) -> Path | None:
    hits = [
        args.repo_root / 'M_configs/G02_Baselines/Data3_FAIR1M/G02_Baselines_Data3_FAIR1M_M10_RTMDet_L.py',
        args.repo_root / 'M_configs/G02_Baselines/Data3_FAIR1M/G02_Fair_Baselines_Data3_FAIR1M_M10_RTMDet_L.py',
    ]
    for hit in hits:
        if hit.exists():
            return hit
    found = find_configs(args.repo_root, ['fair1m', 'rtmdet'])
    return found[0] if found else None


def prepare_far1m_transfer_checkpoint(args: argparse.Namespace, source: Path) -> Path:
    out = args.work_dir / 'exp8/rtmdet_l_dota_ckpt_far1m_compatible.pth'
    if out.exists() and args.resume and not args.force:
        return out
    import torch
    ckpt = torch.load(str(source), map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    removed = []
    for key in list(state.keys()):
        if key.startswith('bbox_head.rtm_cls.'):
            removed.append(key)
            state.pop(key)
    if isinstance(ckpt, dict):
        meta = ckpt.setdefault('meta', {})
        meta['far1m_transfer_note'] = 'Removed incompatible DOTA1 15-class RTMDet cls head for FAR1M 37-class fine-tuning.'
        meta['removed_keys_for_far1m'] = removed
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(out))
    return out


def run_experiment_8(args: argparse.Namespace) -> dict[str, Any]:
    specs = build_specs(args.repo_root, args.weights_dir)
    runner = CommandRunner(args)
    root, found = discover_fair_root(args.repo_root)
    cfg = fair_config(args)
    batch = selected_batch(args)
    angles = angle_csv(args.only_angle, ANGLES if args.mode == 'full' else ANGLES6)
    report: dict[str, Any] = {
        'status': 'DONE',
        'fair_root': str(root) if root else None,
        'found_roots': [str(p) for p in found],
        'fair_config': str(cfg) if cfg else None,
        'transfer_checkpoint': '',
        'variants': {},
    }
    if root is None or cfg is None:
        report['status'] = 'NOT_RUN'
        report['reason'] = 'FAIR1M root or RTMDet FAIR1M config missing'
    else:
        data_root = root
        transfer_ckpt = prepare_far1m_transfer_checkpoint(args, specs['rtmdet_l'].checkpoint)
        report['transfer_checkpoint'] = str(transfer_ckpt)
        train_src = root / 'train'
        if args.mode != 'full':
            train_src, _ = create_subset(root / 'train', args.work_dir / 'exp8/subset_train',
                                         args.max_images_for_smoke)
        variants = [
            ('zero_shot_dota_ckpt_on_far1m', None),
            ('far1m_short_ft_baseline', 'no_rotate'),
            ('far1m_short_ft_rotation_aug', 'rr_continuous'),
            ('far1m_short_ft_consistency', 'rr_discrete'),
        ]
        for i, (name, aug) in enumerate(variants):
            run_dir = args.work_dir / 'exp8' / name
            checkpoint = transfer_ckpt
            train_status = 'NOT_RUN' if aug is None else 'OK'
            train_cmd = ''
            if aug is not None:
                existing_ckpt = latest_checkpoint(run_dir)
                if existing_ckpt is not None and args.resume and not args.force:
                    checkpoint = existing_ckpt
                    train_status = 'OK'
                    train_cmd = f'reused existing checkpoint under --resume: {existing_ckpt}'
                else:
                    cfg_train = write_train_config(
                        run_dir / f'{name}.py', cfg, transfer_ckpt,
                        train_src, aug, batch, args.num_workers,
                        (args.max_iters_for_smoke if args.mode != 'full' else
                         iter_limit_for_epochs(train_src, batch, args.gpu_ids, 1, 6000)),
                        consistency=(name.endswith('consistency')),
                        img_scale=(800, 800),
                        metainfo=fair_metainfo_literal())
                    res = runner.run(
                        f'exp8_train_{name}',
                        train_command(args, cfg_train, run_dir, len(csv_arg(args.gpu_ids)) > 1, 39900 + i),
                        args.work_dir / 'logs/exp8' / name / 'train',
                        gpu_ids=args.gpu_ids,
                        monitor_gpu=True)
                    train_cmd = res.command
                    ckpt = latest_checkpoint(run_dir)
                    if res.returncode == 0 and ckpt:
                        checkpoint = ckpt
                        train_status = 'OK'
                    else:
                        train_status = 'FAILED'
            rows = []
            if aug is None or train_status == 'OK':
                for angle in angles:
                    rows.append(eval_angle(
                        args, runner, cfg, checkpoint, data_root,
                        'rot_val_standard/realistic', angle, run_dir / f'angle_{angle}',
                        batch, len(csv_arg(args.gpu_ids)) > 1))
            report['variants'][name] = {
                'train_status': train_status,
                'checkpoint': str(checkpoint),
                'train_command': train_cmd,
                'rows': rows,
                'summary': summarize_angle_rows(rows),
            }
        if any(v['train_status'] == 'FAILED' for v in report['variants'].values()):
            report['status'] = 'PARTIAL'
    write_json(args.work_dir / 'exp8/exp8_results.json', report)
    write_exp8_md(args, report)
    update_summary(args)
    return report


def write_exp8_md(args: argparse.Namespace, report: dict[str, Any]) -> None:
    lines = [
        '# Experiment 8: FAR1M Quick Transfer Rotation Stress',
        '',
        f'- generated_at: `{now()}`',
        f'- status: `{report["status"]}`',
        f'- FAR1M selected root: `{report.get("fair_root")}`',
        f'- FAR1M config: `{report.get("fair_config")}`',
        '- metric: AP50 / mAP@0.5',
        '',
        '## Path Discovery',
        '',
        f'- discovered_roots: `{", ".join(report.get("found_roots", []))}`',
        f'- rotated validation: `rot_val_standard/realistic`',
    ]
    if report.get('reason'):
        lines.append(f'- reason: {report["reason"]}')
    lines.extend(['', '## Variant Results', '', '| variant | train status | checkpoint | completed | mean AP50 | worst AP50 | best AP50 | std | RSI |'])
    lines.append('|---|---|---|---:|---:|---:|---:|---:|---:|')
    for name, payload in report.get('variants', {}).items():
        s = payload.get('summary', {})
        lines.append(f'| {name} | {payload.get("train_status")} | `{payload.get("checkpoint")}` | {s.get("completed", 0)} | {fmt(s.get("mean_ap50"))} | {fmt(s.get("worst_ap50"))} | {fmt(s.get("best_ap50"))} | {fmt(s.get("std"))} | {fmt(s.get("rsi"))} |')
    for name, payload in report.get('variants', {}).items():
        lines.extend(['', f'### {name}', '', '| angle | status | AP50 | eval log |'])
        lines.append('|---:|---|---:|---|')
        for row in payload.get('rows', []):
            lines.append(f'| {row.get("angle")} | {row.get("status")} | {fmt(row.get("ap50"))} | `{row.get("eval_stdout", "")}` |')
        if not payload.get('rows'):
            lines.append('| NA | NOT_RUN | NA |  |')
    lines.extend([
        '',
        '## Conclusion',
        '',
        '- Transfer conclusions are valid only for variants with completed AP50 rows. Zero-shot may fail if DOTA1 head weights are incompatible with FAR1M class count.',
    ])
    write_md(args.result_md_dir / 'exp8_far1m_quick_transfer_rotation_stress.md', lines)


def env_report(args: argparse.Namespace) -> dict[str, Any]:
    data: dict[str, Any] = {'git_commit': '', 'python': '', 'torch': '', 'mmcv': '', 'mmdet': '', 'mmrotate': '', 'mmengine': '', 'cuda': '', 'gpu': ''}
    try:
        proc = subprocess.run(['rtk', 'git', 'rev-parse', 'HEAD'], cwd=str(args.repo_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        data['git_commit'] = proc.stdout.strip()
    except Exception:
        pass
    code = "import sys; print('python='+sys.version.replace('\\n',' '));\nmods=['torch','mmcv','mmdet','mmrotate','mmengine'];\nimport importlib\nfor m in mods:\n    try:\n        x=importlib.import_module(m); print(m+'='+getattr(x,'__version__','NA'))\n    except Exception as e: print(m+'=ERR:'+str(e))\ntry:\n import torch; print('cuda='+str(torch.version.cuda))\nexcept Exception: pass\n"
    try:
        proc = subprocess.run(['rtk', 'env', *CommandRunner(args).env_assignments(args.gpu_ids), str(args.python_bin), '-c', code], cwd=str(args.repo_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in proc.stdout.splitlines():
            if '=' in line:
                k, v = line.split('=', 1)
                data[k] = v
    except Exception:
        pass
    try:
        proc = subprocess.run(['rtk', 'nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        data['gpu'] = proc.stdout.strip()
    except Exception:
        pass
    return data


def update_summary(args: argparse.Namespace) -> None:
    dry = load_json(args.work_dir / 'preflight/dryrun.json', {})
    smoke = load_json(args.work_dir / 'preflight/smoke.json', {})
    multi = load_json(args.work_dir / 'preflight/multigpu_batchsize.json', {})
    exp5 = load_json(args.work_dir / 'exp5/exp5_results.json', {})
    exp6 = load_json(args.work_dir / 'exp6/exp6_results.json', {})
    exp7 = load_json(args.work_dir / 'exp7/exp7_results.json', {})
    exp8 = load_json(args.work_dir / 'exp8/exp8_results.json', {})
    env = env_report(args)
    statuses = {
        'Experiment 5': exp5.get('status', 'NOT_RUN'),
        'Experiment 6': exp6.get('status', 'NOT_RUN'),
        'Experiment 7': exp7.get('status', 'NOT_RUN'),
        'Experiment 8': exp8.get('status', 'NOT_RUN'),
    }
    lines = [
        '# GPU Batch2 Rotation Experiments Summary',
        '',
        f'- generated_at: `{now()}`',
        f'- work_dir: `{args.work_dir}`',
        f'- result_md_dir: `{args.result_md_dir}`',
        f'- physical GPU ids requested: `{args.gpu_ids}`',
        '',
        '## Motivation',
        '',
        'Use the idle A40 GPUs to move from post-processing observations toward training-time augmentation, cross-view consistency, and feature-level mechanism diagnostics.',
        '',
        '## Preflight Validation',
        '',
        '| validation | status | md | key artifact |',
        '|---|---|---|---|',
        f'| dryrun | {dry.get("status", "NOT_RUN")} | `{args.result_md_dir / "preflight_gpu_batch2_dryrun.md"}` | `{args.work_dir / "preflight/dryrun.json"}` |',
        f'| smoke | {smoke.get("status", "NOT_RUN")} | `{args.result_md_dir / "preflight_gpu_batch2_smoke.md"}` | `{args.work_dir / "preflight/smoke.json"}` |',
        f'| multigpu + batch size | {multi.get("status", "NOT_RUN")} | `{args.result_md_dir / "preflight_gpu_batch2_multigpu_batchsize.md"}` | selected_batch={multi.get("selected_batch_size", "NA")} |',
        '',
        '## Experiment Status',
        '',
        '| experiment | status | md |',
        '|---|---|---|',
        f'| Experiment 5 | {statuses["Experiment 5"]} | `{args.result_md_dir / "exp5_dota1_rotation_aug_short_finetune.md"}` |',
        f'| Experiment 6 | {statuses["Experiment 6"]} | `{args.result_md_dir / "exp6_cross_view_consistency_regularization_prototype.md"}` |',
        f'| Experiment 7 | {statuses["Experiment 7"]} | `{args.result_md_dir / "exp7_feature_logit_equivariance_diagnostic.md"}` |',
        f'| Experiment 8 | {statuses["Experiment 8"]} | `{args.result_md_dir / "exp8_far1m_quick_transfer_rotation_stress.md"}` |',
        '',
        '## Proposition Check',
        '',
        'Rotation-induced recognition collapse cannot be fully explained by post-processing; training-time rotation augmentation and cross-view consistency are required to improve worst-angle robustness.',
        '',
        f'- current support: `{"INSUFFICIENT" if any(v in ("NOT_RUN", "") for v in statuses.values()) else "EVALUATED"}`',
        '- Rationale: this summary is updated incrementally; final support requires completed Experiment 5 and Experiment 6 AP50 curves.',
        '',
        '## Key Findings',
        '',
        f'- DOTA1 train split discovery: `{dry.get("dota_train")}`.',
        f'- FAR1M root discovery: `{dry.get("fair_root") or exp8.get("fair_root")}`.',
        f'- Selected RTMDet-L batch size: `{multi.get("selected_batch_size", "NA")}`.',
        f'- Experiment 7 raw CSV: `{exp7.get("csv", "")}`.',
        f'- Commands are logged in `{args.work_dir / "commands.jsonl"}`.',
        '',
        '## Key Risks Or Failures',
        '',
        f'- dryrun failures: `{dry.get("failures", [])}`',
        f'- dryrun warnings: `{dry.get("warnings", [])}`',
        '- DOTA1 fine-tuning cannot be interpreted if only validation-like data are available.',
        '- FAR1M zero-shot may be class-head incompatible with DOTA1 checkpoints.',
        '- The consistency prototype is head-logit based and is not a full inverse-geometry matched detector loss.',
        '',
        '## Next Steps',
        '',
        '- Add or mount the real DOTA1 train split before treating Exp5/Exp6 training results as paper evidence.',
        '- If Exp6 runs, replace head-space MSE with geometry-aware matched prediction consistency.',
        '- Extend Exp7 from six angles to all twelve after the smoke path is stable.',
        '',
        '## Important Logs',
        '',
        f'- command log: `{args.work_dir / "commands.jsonl"}`',
        f'- log root: `{args.work_dir / "logs"}`',
        '',
        '## Environment',
        '',
        f'- git commit: `{env.get("git_commit")}`',
        f'- Python: `{env.get("python")}`',
        f'- PyTorch: `{env.get("torch")}`',
        f'- MMCV: `{env.get("mmcv")}`',
        f'- MMDetection: `{env.get("mmdet")}`',
        f'- MMRotate: `{env.get("mmrotate")}`',
        f'- MMEngine: `{env.get("mmengine")}`',
        f'- CUDA: `{env.get("cuda")}`',
        f'- GPU: `{md_escape(env.get("gpu"))}`',
    ]
    write_md(args.result_md_dir / 'summary_gpu_batch2_20260507.md', lines)


def run_full(args: argparse.Namespace) -> dict[str, Any]:
    reports = {}
    if args.exp in ('all', '5'):
        reports['5'] = run_experiment_5(args)
    if args.exp in ('all', '6'):
        reports['6'] = run_experiment_6(args)
    if args.exp in ('all', '7'):
        reports['7'] = run_feature_diagnostic(args)
    if args.exp in ('all', '8'):
        reports['8'] = run_experiment_8(args)
    update_summary(args)
    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run GPU batch2 rotation experiments.')
    parser.add_argument('--repo-root', type=Path, default=Path('/data1/zcy/OpenRSD'))
    parser.add_argument('--result-md-dir', type=Path, default=Path('/data1/zcy/OpenRSD/resultmd'))
    parser.add_argument('--weights-dir', type=Path, default=Path('/data1/zcy/OpenRSD/weights'))
    parser.add_argument('--work-dir', type=Path, default=Path('/data1/zcy/OpenRSD/work_dirs/rotation_gpu_batch2_20260507'))
    parser.add_argument('--gpu-ids', default='4,5,6,7')
    parser.add_argument('--exp', default='all', choices=['all', '5', '6', '7', '8'])
    parser.add_argument('--mode', default='dryrun', choices=['dryrun', 'smoke', 'full', 'debug'])
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--batch-size-candidates', default='1,2,4,8')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--max-images-for-smoke', type=int, default=16)
    parser.add_argument('--max-iters-for-smoke', type=int, default=20)
    parser.add_argument('--max-epochs', type=int, default=3)
    parser.add_argument('--only-model', default='')
    parser.add_argument('--only-angle', default='')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--python-bin', type=Path, default=existing_python())
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.result_md_dir = args.result_md_dir.resolve()
    args.weights_dir = args.weights_dir.resolve()
    args.work_dir = args.work_dir.resolve()
    args.python_bin = args.python_bin.resolve()
    return args


def main() -> int:
    args = parse_args()
    os.chdir(args.repo_root)
    args.result_md_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == 'dryrun':
        report = run_dryrun(args)
        return 0 if report['status'] != 'FAILED' else 1
    if args.mode == 'smoke':
        if args.exp == '5' and len(csv_arg(args.gpu_ids)) > 1 and args.batch_size_candidates:
            report = run_multigpu_batchsize(args)
        else:
            report = run_smoke(args)
        return 0 if report['status'] == 'OK' else 1
    if args.mode == 'debug':
        if args.exp == '7':
            report = run_feature_diagnostic(args, debug=True)
            return 0 if report['status'] in ('OK', 'PARTIAL') else 1
        update_summary(args)
        return 0
    if args.mode == 'full':
        run_full(args)
        return 0
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
