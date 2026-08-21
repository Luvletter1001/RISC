#!/usr/bin/env python3
"""Unified runner for the 2026-05-07 rotation repair experiment suite."""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import math
import os
import pickle
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


ANGLES = ('000', '030', '060', '090', '120', '150', '180', '210', '240',
          '270', '300', '330')
DOTA_DATA_REL = Path('data/DOTA1_1024_500')
DOTA_SWEEP_REL = Path('angle_sweep_val/realistic')
DEFAULT_OLD_DOTA_OUT = Path('work_dirs/dota1_exp_ab_20260507_003353')
DEFAULT_PYTHON = Path('/data/zcy/anaconda3/envs/openrsd/bin/python')
DOTA_CLASSES = (
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter')
FOCUS_CLASSES = {
    'plane', 'bridge', 'small-vehicle', 'large-vehicle', 'ship', 'harbor',
    'roundabout', 'storage-tank', 'helicopter'
}
AR_BINS = (
    ('AR 1-3', 1.0, 3.0),
    ('AR 3-6', 3.0, 6.0),
    ('AR 6-9', 6.0, 9.0),
    ('AR 9-12', 9.0, 12.0),
    ('AR 12-30', 12.0, 30.0),
    ('AR > 30', 30.0, float('inf')),
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display: str
    config: Path
    checkpoint: Path
    infer_root: Path
    smoke_batch: int


@dataclass
class CommandResult:
    name: str
    command: str
    returncode: int
    log_path: str
    duration_sec: float
    status: str
    failure_kind: str = ''
    tail: str = ''
    peak_mem_mb: dict[str, int] | None = None
    process_seen: dict[str, bool] | None = None


_MM_READY = False


def now() -> str:
    return datetime.now().strftime('%F %T')


def log(message: str) -> None:
    print(f'[{now()}] {message}', flush=True)


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return 'NA'
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 'NA'
    if math.isnan(value):
        return 'NA'
    return f'{value:.{digits}f}'


def parse_csv_arg(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


def parse_angle_list(value: str | None, default: Sequence[str]) -> list[str]:
    raw = parse_csv_arg(value)
    if not raw:
        return list(default)
    return [f'{int(item):03d}' for item in raw]


def parse_int_csv(value: str | None, default: Sequence[int]) -> list[int]:
    raw = parse_csv_arg(value)
    if not raw:
        return list(default)
    return [int(item) for item in raw]


def shell_join(argv: Sequence[str]) -> str:
    import shlex

    return ' '.join(shlex.quote(str(item)) for item in argv)


def read_text(path: Path, max_chars: int = 20000) -> str:
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except FileNotFoundError:
        return ''
    if len(text) > max_chars:
        return text[-max_chars:]
    return text


def last_lines(path: Path, count: int = 100) -> str:
    text = read_text(path, 200000)
    return '\n'.join(text.splitlines()[-count:])


def detect_failure(text: str) -> str:
    lowered = text.lower()
    checks = (
        ('CUDA out of memory', 'oom'),
        ('out of memory', 'oom'),
        ('KeyError: \'gt_instances\'', 'gt_instances'),
        ('keyerror: "gt_instances"', 'gt_instances'),
        ('FileNotFoundError', 'file_not_found'),
        ('checkpoint', 'checkpoint_missing'),
        ('config', 'config_missing'),
        ('ann_file', 'annotation_path_missing'),
        ('img_path', 'image_path_missing'),
        ('predictions.pkl', 'predictions_missing'),
        ('img_id', 'img_id_alignment'),
    )
    for needle, kind in checks:
        if needle.lower() in lowered:
            return kind
    if 'traceback (most recent call last)' in lowered:
        return 'traceback'
    return ''


def setup_repo_imports(repo_root: Path) -> None:
    for path in (repo_root, repo_root / 'tools'):
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)


def ensure_mm(repo_root: Path) -> None:
    global _MM_READY
    if _MM_READY:
        return
    setup_repo_imports(repo_root)
    from openrsd_env import preload_installed_mmengine

    preload_installed_mmengine()
    from mmdet.utils import register_all_modules as register_mmdet
    from mmrotate.utils import register_all_modules as register_mmrotate

    register_mmdet(init_default_scope=False)
    register_mmrotate(init_default_scope=False)
    _MM_READY = True


class GpuMonitor:
    def __init__(self, gpu_ids: str, interval: float = 1.0):
        self.gpu_ids = [item.strip() for item in gpu_ids.split(',') if item.strip()]
        self.interval = interval
        self.peak_mem = {gpu: 0 for gpu in self.gpu_ids}
        self.process_seen = {gpu: False for gpu in self.gpu_ids}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run_nvidia_smi(self, args: list[str]) -> str:
        cmd = ['rtk', 'nvidia-smi', *args]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False)
        if result.returncode != 0:
            return ''
        return result.stdout

    def _snapshot(self) -> None:
        gpu_text = self._run_nvidia_smi([
            '--query-gpu=index,uuid,memory.used',
            '--format=csv,noheader,nounits'
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

        proc_text = self._run_nvidia_smi([
            '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
            '--format=csv,noheader,nounits'
        ])
        for line in proc_text.splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 4:
                continue
            index = uuid_to_index.get(parts[0])
            if index in self.process_seen:
                self.process_seen[index] = True

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._snapshot()
            self._stop.wait(self.interval)

    def __enter__(self) -> 'GpuMonitor':
        self._snapshot()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._snapshot()


class CommandRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo_root = args.repo_root
        self.work_dir = args.work_dir
        self.command_log = self.work_dir / 'commands.jsonl'
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.command_log.parent.mkdir(parents=True, exist_ok=True)

    def env_assignments(self, gpu_ids: str | None = None) -> list[str]:
        gpu_ids = gpu_ids or self.args.gpu_ids
        return [
            'PYTHONNOUSERSITE=1',
            'MPLCONFIGDIR=/tmp/mplconfig',
            f'CUDA_VISIBLE_DEVICES={gpu_ids}',
            f'PYTHONPATH={self.repo_root}:{self.repo_root / "tools"}',
            'NCCL_P2P_DISABLE=1',
            'NCCL_IB_DISABLE=1',
        ]

    def command_string(self, argv: Sequence[str], gpu_ids: str | None = None) -> str:
        return 'rtk env ' + shell_join(self.env_assignments(gpu_ids)) + ' ' + shell_join(argv)

    def run(self,
            name: str,
            argv: Sequence[str],
            log_path: Path,
            gpu_ids: str | None = None,
            monitor_gpu: bool = False,
            allow_skip: bool = False) -> CommandResult:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = self.command_string(argv, gpu_ids)
        if allow_skip:
            result = CommandResult(name, command, 0, str(log_path), 0.0, 'SKIPPED')
            self._append_command(result)
            with log_path.open('a', encoding='utf-8') as f:
                f.write(f'[{now()}] SKIPPED existing artifact\n')
                f.write(command + '\n')
            return result

        env_cmd = ['rtk', 'env', *self.env_assignments(gpu_ids), *map(str, argv)]
        start = time.time()
        log(f'RUN {name}: {command}')
        monitor = GpuMonitor(gpu_ids or self.args.gpu_ids) if monitor_gpu else contextlib.nullcontext()
        with log_path.open('w', encoding='utf-8') as f:
            f.write(f'[{now()}] command={command}\n')
            f.write(f'cwd={self.repo_root}\n\n')
            with monitor as mon:
                proc = subprocess.run(
                    env_cmd,
                    cwd=str(self.repo_root),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False)
        duration = time.time() - start
        text_tail = last_lines(log_path, 100)
        failure_kind = detect_failure(text_tail)
        status = 'OK' if proc.returncode == 0 else f'FAILED({proc.returncode})'
        peak = mon.peak_mem if monitor_gpu else None  # type: ignore[name-defined]
        seen = mon.process_seen if monitor_gpu else None  # type: ignore[name-defined]
        result = CommandResult(
            name=name,
            command=command,
            returncode=proc.returncode,
            log_path=str(log_path),
            duration_sec=duration,
            status=status,
            failure_kind=failure_kind,
            tail=text_tail if proc.returncode else '',
            peak_mem_mb=peak,
            process_seen=seen)
        self._append_command(result)
        log(f'DONE {name}: {status} log={log_path}')
        return result

    def _append_command(self, result: CommandResult) -> None:
        payload = {
            'time': now(),
            'name': result.name,
            'command': result.command,
            'returncode': result.returncode,
            'status': result.status,
            'failure_kind': result.failure_kind,
            'log_path': result.log_path,
            'duration_sec': result.duration_sec,
        }
        with self.command_log.open('a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False) + '\n')


def parse_metrics_from_log(log_path: Path) -> dict:
    text = read_text(log_path, 200000)

    def last(pattern: str):
        matches = re.findall(pattern, text)
        return float(matches[-1]) if matches else None

    return {
        'map': last(r'["\']?dota/mAP["\']?\s*[:=]\s*([0-9]*\.?[0-9]+)'),
        'ap50': last(r'["\']?dota/AP50["\']?\s*[:=]\s*([0-9]*\.?[0-9]+)'),
    }


def parse_classwise_from_eval_log(log_path: Path,
                                  classes: Sequence[str]) -> list[dict]:
    text = read_text(log_path, 500000)
    class_set = set(classes)
    parsed: list[dict] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith('|'):
            continue
        cells = [cell.strip() for cell in stripped.strip('|').split('|')]
        if len(cells) < 5 or cells[0] not in class_set:
            continue
        try:
            parsed.append({
                'class': cells[0],
                'ap50': float(cells[4]),
                'num_gts': int(float(cells[1])),
                'num_dets': int(float(cells[2])),
            })
        except ValueError:
            continue
    if not parsed:
        return []
    last_by_class = {row['class']: row for row in parsed}
    if not all(class_name in last_by_class for class_name in classes):
        return []
    return [last_by_class[class_name] for class_name in classes]


def load_pickle(path: Path):
    with path.open('rb') as f:
        return pickle.load(f)


def normalize_img_id(img_id) -> str:
    if isinstance(img_id, int):
        return str(img_id)
    text = str(img_id)
    if text.startswith('angle_') and '__' in text:
        return text.split('__', 1)[1]
    return text


def to_numpy(value):
    import numpy as np

    if hasattr(value, 'tensor'):
        value = value.tensor
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    if hasattr(value, 'cpu'):
        return value.cpu().numpy()
    return np.asarray(value)


def classwise_payload_from_eval_log(log_path: Path, pkl_path: Path,
                                    out_json: Path,
                                    classes: Sequence[str],
                                    resume: bool,
                                    force: bool) -> dict | None:
    if out_json.exists() and resume and not force:
        return json.loads(out_json.read_text(encoding='utf-8'))
    rows = parse_classwise_from_eval_log(log_path, classes)
    if not rows:
        return None
    metrics = parse_metrics_from_log(log_path)
    payload = {
        'map': metrics.get('map'),
        'ap50': metrics.get('ap50'),
        'classes': rows,
        'pkl': str(pkl_path),
        'source_log': str(log_path),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding='utf-8')
    return payload


def polygon_to_rbox(coords: Sequence[float]) -> tuple[float, float, float, float, float]:
    xs = [float(coords[i]) for i in range(0, 8, 2)]
    ys = [float(coords[i]) for i in range(1, 8, 2)]
    cx = sum(xs) / 4.0
    cy = sum(ys) / 4.0
    edge0 = (xs[1] - xs[0], ys[1] - ys[0])
    edge1 = (xs[2] - xs[1], ys[2] - ys[1])
    w = math.hypot(*edge0)
    h = math.hypot(*edge1)
    angle = math.atan2(edge0[1], edge0[0])
    if h > w:
        w, h = h, w
        angle += math.pi / 2.0
    while angle < -math.pi / 2.0:
        angle += math.pi
    while angle >= math.pi / 2.0:
        angle -= math.pi
    return cx, cy, max(w, 1e-6), max(h, 1e-6), angle


def load_dota_ann_gt(data_root: Path, angle: str,
                     classes: Sequence[str]) -> dict[str, dict]:
    import numpy as np

    class_to_id = {name: idx for idx, name in enumerate(classes)}
    ann_dir = data_root / DOTA_SWEEP_REL / f'angle_{angle}' / 'annfiles'
    gt_by_id: dict[str, dict] = {}
    for ann_file in sorted(ann_dir.glob('*.txt')):
        boxes = []
        labels = []
        for line in ann_file.read_text(encoding='utf-8', errors='replace').splitlines():
            parts = line.strip().split()
            if len(parts) < 9 or parts[8] not in class_to_id:
                continue
            try:
                coords = [float(item) for item in parts[:8]]
            except ValueError:
                continue
            boxes.append(polygon_to_rbox(coords))
            labels.append(class_to_id[parts[8]])
        gt_by_id[normalize_img_id(ann_file.stem)] = {
            'bboxes': np.asarray(boxes, dtype=np.float32).reshape(-1, 5),
            'labels': np.asarray(labels, dtype=np.int64),
        }
    return gt_by_id


def geometry_recall_from_annfiles(args: argparse.Namespace, pkl_path: Path,
                                  angle: str, out_json: Path,
                                  score_thr: float = 0.05) -> dict:
    if out_json.exists() and args.resume and not args.force:
        return json.loads(out_json.read_text(encoding='utf-8'))
    ensure_mm(args.repo_root)
    import numpy as np
    import torch
    from mmcv.ops import box_iou_rotated

    gt_by_id = load_dota_ann_gt(args.repo_root / DOTA_DATA_REL, angle,
                                DOTA_CLASSES)
    predictions = load_pickle(pkl_path)
    stats = {name: {'gt': 0, 'matched': 0} for name, _, _ in AR_BINS}
    for pred in predictions:
        pid = normalize_img_id(pred.get('img_id'))
        gt = gt_by_id.get(pid)
        if not gt or len(gt['labels']) == 0:
            continue
        gt_boxes = gt['bboxes']
        gt_labels = gt['labels']
        pred_inst = pred['pred_instances']
        pred_boxes = to_numpy(pred_inst['bboxes']).astype(np.float32)
        pred_labels = to_numpy(pred_inst['labels']).astype(np.int64)
        pred_scores = to_numpy(pred_inst['scores']).astype(np.float32)
        keep = pred_scores >= score_thr
        pred_boxes, pred_labels = pred_boxes[keep], pred_labels[keep]
        gt_bins = []
        for gt_box in gt_boxes:
            ar = max(float(gt_box[2]), float(gt_box[3])) / max(
                min(float(gt_box[2]), float(gt_box[3])), 1e-6)
            bin_name = next(name for name, lo, hi in AR_BINS if ar >= lo and ar < hi)
            stats[bin_name]['gt'] += 1
            gt_bins.append(bin_name)
        if len(pred_boxes) == 0:
            continue
        matched = np.zeros(len(gt_boxes), dtype=bool)
        for label in np.unique(gt_labels):
            gt_idx = np.where(gt_labels == label)[0]
            pred_idx = np.where(pred_labels == label)[0]
            if len(gt_idx) == 0 or len(pred_idx) == 0:
                continue
            ious = box_iou_rotated(
                torch.from_numpy(pred_boxes[pred_idx].astype(np.float32)),
                torch.from_numpy(gt_boxes[gt_idx].astype(np.float32))).cpu().numpy()
            if ious.size:
                matched[gt_idx] = ious.max(axis=0) >= 0.5
        for idx, is_matched in enumerate(matched):
            if is_matched:
                stats[gt_bins[idx]]['matched'] += 1
    rows = []
    for name in stats:
        gt_count = stats[name]['gt']
        matched_count = stats[name]['matched']
        rows.append({
            'bin': name,
            'gt': gt_count,
            'matched': matched_count,
            'recall50': matched_count / gt_count if gt_count else 0.0,
        })
    payload = {'rows': rows, 'pkl': str(pkl_path), 'score_thr': score_thr}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding='utf-8')
    return payload


def select_model_keys(args: argparse.Namespace) -> list[str]:
    if args.only_model:
        return parse_csv_arg(args.only_model)
    return ['rtmdet_l', 'h2rbox_v2', 'retinanet_msrr']


def build_dota_specs(repo_root: Path, weights_dir: Path) -> dict[str, ModelSpec]:
    infer_root = repo_root / DEFAULT_OLD_DOTA_OUT / 'exp_a_tta_infer'
    return {
        'rtmdet_l': ModelSpec(
            key='rtmdet_l',
            display='RTMDet-L 3xMS',
            config=repo_root / 'M_configs/RotationStudy/rotated_rtmdet_l_dota1_ms_eval.py',
            checkpoint=weights_dir / 'rotated_rtmdet_l-3x-dota_ms-2738da34.pth',
            infer_root=infer_root / 'rtmdet_l',
            smoke_batch=4),
        'h2rbox_v2': ModelSpec(
            key='h2rbox_v2',
            display='H2RBox-v2 1xMS+RR',
            config=repo_root / 'M_configs/RotationStudy/h2rbox_v2_r50_fpn_dota1_ms_rr_eval.py',
            checkpoint=weights_dir / 'h2rbox_v2-le90_r50_fpn_ms_rr-1x_dota-5e0e53e1.pth',
            infer_root=infer_root / 'h2rbox_v2',
            smoke_batch=4),
        'retinanet_msrr': ModelSpec(
            key='retinanet_msrr',
            display='Rotated RetinaNet R50 MS+RR',
            config=repo_root / 'M_configs/RotationStudy/rotated_retinanet_r50_msrr_dota1_eval.py',
            checkpoint=weights_dir / 'rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90-1da1ec9c.pth',
            infer_root=infer_root / 'retinanet_msrr',
            smoke_batch=4),
    }


def path_ok(path: Path) -> str:
    return 'OK' if path.exists() else 'MISSING'


def discover_paths(repo_root: Path, names: Sequence[str], suffix: str = '.py') -> list[Path]:
    hits = []
    lowered = [name.lower() for name in names]
    for root in (repo_root / 'M_configs', repo_root / 'mmrotate_configs', repo_root / 'work_dirs'):
        if not root.exists():
            continue
        for path in root.rglob(f'*{suffix}'):
            hay = str(path.relative_to(repo_root)).lower()
            if all(token in hay for token in lowered):
                hits.append(path)
    return sorted(hits)


def discover_weights(search_roots: Sequence[Path], tokens: Sequence[str]) -> list[Path]:
    hits = []
    lowered = [token.lower() for token in tokens]
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob('*.pth'):
            hay = str(path).lower()
            if all(token in hay for token in lowered):
                hits.append(path)
    return sorted(hits)


def discover_fair1m_root(repo_root: Path) -> tuple[Path | None, list[Path]]:
    candidates = [
        repo_root / 'data/FAR1M',
        repo_root / 'data/far1m',
        repo_root / 'data/FAR1M_1024_500',
        repo_root / 'data/FAR1M_1024',
        repo_root / 'data/fair1m/dair1m_1024',
        repo_root / 'data/FAIR1M_2_800_400',
    ]
    found = [p for p in candidates if p.exists()]
    data_dir = repo_root / 'data'
    if data_dir.exists():
        for path in data_dir.rglob('*'):
            if path.is_dir() and ('far' in path.name.lower() or 'fair' in path.name.lower()):
                if path not in found:
                    found.append(path)
    preferred = None
    for path in found:
        if (path / 'rot_val_standard/realistic/angle_000').exists():
            preferred = path
            break
    if preferred is None and found:
        preferred = found[0]
    return preferred, found


def write_md(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')


def markdown_link(path: Path | str) -> str:
    return f'`{path}`'


def count_ann_files(ann_dir: Path) -> int:
    return len(list(ann_dir.glob('*.txt'))) if ann_dir.exists() else 0


def run_dryrun(args: argparse.Namespace) -> dict:
    args.result_md_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    tmp = args.result_md_dir / '.write_test_rotation_suite.tmp'
    writable = True
    try:
        tmp.write_text('ok', encoding='utf-8')
        tmp.unlink()
    except OSError:
        writable = False

    dota_root = args.repo_root / DOTA_DATA_REL
    fair_root, fair_found = discover_fair1m_root(args.repo_root)
    specs = build_dota_specs(args.repo_root, args.weights_dir)
    hard_failures: list[str] = []
    warnings_list: list[str] = []

    required = {
        'repo_root': args.repo_root,
        'result_md_dir_writable': args.result_md_dir,
        'weights_dir': args.weights_dir,
        'dota1_root': dota_root,
        'eval_metric': args.repo_root / 'tools/openrsd_eval_metric.py',
        'merge_script': args.repo_root / 'M_Tools/analysis/rotation_tta_merge.py',
    }
    for name, path in required.items():
        if name == 'result_md_dir_writable':
            if not writable:
                hard_failures.append(f'{name}: not writable')
        elif not path.exists():
            hard_failures.append(f'{name}: {path} missing')

    missing_angles = []
    for angle in ANGLES:
        ann = dota_root / DOTA_SWEEP_REL / f'angle_{angle}/annfiles'
        img = dota_root / DOTA_SWEEP_REL / f'angle_{angle}/images'
        if not ann.exists() or not img.exists():
            missing_angles.append(angle)
    if missing_angles:
        hard_failures.append(f'DOTA1 angle sweep missing angles: {missing_angles}')

    for spec in specs.values():
        if not spec.config.exists():
            hard_failures.append(f'{spec.key} config missing: {spec.config}')
        if not spec.checkpoint.exists():
            hard_failures.append(f'{spec.key} checkpoint missing: {spec.checkpoint}')
        missing_pred = [
            angle for angle in ANGLES
            if not (spec.infer_root / f'angle_{angle}/predictions.pkl').exists()
        ]
        if missing_pred:
            warnings_list.append(
                f'{spec.key} existing DOTA1 predictions missing for {missing_pred}; full runner will infer if needed.')

    eval_script = args.repo_root / 'tools/analysis_tools/eval_metric.py'
    eval_text = read_text(eval_script, 80000)
    wrapper_text = read_text(args.repo_root / 'tools/openrsd_eval_metric.py', 20000)
    merge_text = read_text(args.repo_root / 'M_Tools/analysis/rotation_tta_merge.py', 40000)
    eval_support = {
        'wrapper_exists': (args.repo_root / 'tools/openrsd_eval_metric.py').exists(),
        'implementation_exists': eval_script.exists(),
        'predictions_only': '--predictions-only' in eval_text,
        'gt_merge': 'merge_predictions_with_ground_truth' in eval_text,
        'empty_gt_fix': '_empty_gt_instances' in eval_text and 'gt_instances' in eval_text,
        'wrapper_imports_main': 'analysis_tools.eval_metric' in wrapper_text,
    }
    if not all(eval_support.values()):
        hard_failures.append(f'openrsd_eval_metric.py missing required GT merge/predictions-only support: {eval_support}')
    if '--target-angle' not in merge_text:
        hard_failures.append('rotation_tta_merge.py lacks --target-angle support')

    fair_angle_dirs = []
    if fair_root:
        for angle in ANGLES:
            p = fair_root / f'rot_val_standard/realistic/angle_{angle}'
            if p.exists():
                fair_angle_dirs.append(p)
    else:
        warnings_list.append('FAR1M/FAIR1M root not found under data/.')

    fair_config_hits = {
        'rtmdet_l': discover_paths(args.repo_root, ['fair1m', 'rtmdet']),
        'h2rbox_v2': discover_paths(args.repo_root, ['fair1m', 'h2rbox']),
        'retinanet_msrr': discover_paths(args.repo_root, ['fair1m', 'retinanet']),
    }
    fair_weight_hits = {
        'rtmdet_l': discover_weights([args.weights_dir, args.repo_root / 'work_dirs'], ['fair1m', 'rtmdet']),
        'h2rbox_v2': discover_weights([args.weights_dir, args.repo_root / 'work_dirs'], ['fair1m', 'h2rbox']),
        'retinanet_msrr': discover_weights([args.weights_dir, args.repo_root / 'work_dirs'], ['fair1m', 'retinanet']),
    }
    for key in ('rtmdet_l', 'h2rbox_v2', 'retinanet_msrr'):
        if not fair_config_hits[key]:
            warnings_list.append(f'FAIR1M requested model config not found for {key}.')
        if not fair_weight_hits[key]:
            warnings_list.append(f'FAIR1M requested model checkpoint not found for {key}.')

    runner = CommandRunner(args)
    planned_commands = []
    rtmdet = specs['rtmdet_l']
    smoke_out = args.work_dir / 'preflight/smoke/rtmdet_l/angle_000/predictions.pkl'
    planned_commands.append(
        runner.command_string([
            str(args.python_bin), 'tools/openrsd_test.py', str(rtmdet.config),
            str(rtmdet.checkpoint), '--work-dir',
            str(smoke_out.parent), '--out',
            str(smoke_out), '--cfg-options',
            'test_dataloader.batch_size=1',
            f'test_dataloader.num_workers={args.num_workers}',
            'test_dataloader.dataset.data_root=<smoke_subset>',
            'test_dataloader.dataset.ann_file=annfiles/',
            'test_dataloader.dataset.data_prefix.img_path=images/',
        ], args.gpu_ids))
    planned_commands.append(
        runner.command_string([
            str(args.python_bin), 'M_Tools/analysis/rotation_tta_merge.py',
            '--out', '<merged_predictions.pkl>', '--target-angle', '0',
            '--img-shape', '1024', '1024', '--prediction',
            '<angle_000/predictions.pkl>', '--angle', '0', '--prediction',
            '<angle_030/predictions.pkl>', '--angle', '30'
        ], args.gpu_ids))
    planned_commands.append(
        runner.command_string([
            str(args.python_bin), 'tools/openrsd_eval_metric.py',
            str(rtmdet.config), '<merged_predictions.pkl>', '--cfg-options',
            'test_dataloader.dataset.data_root=<target_subset>',
            'test_dataloader.dataset.ann_file=annfiles/',
            'test_dataloader.dataset.data_prefix.img_path=images/',
        ], args.gpu_ids))

    status = 'FAILED' if hard_failures else ('PASS_WITH_WARNINGS' if warnings_list else 'PASS')
    report = {
        'status': status,
        'hard_failures': hard_failures,
        'warnings': warnings_list,
        'dota_root': str(dota_root),
        'fair_root': str(fair_root) if fair_root else None,
        'fair_found': [str(p) for p in fair_found],
        'fair_angle_count': len(fair_angle_dirs),
        'dota_specs': {
            key: {
                'config': str(spec.config),
                'checkpoint': str(spec.checkpoint),
                'infer_root': str(spec.infer_root),
            }
            for key, spec in specs.items()
        },
        'eval_support': eval_support,
        'planned_commands': planned_commands,
        'fair_config_hits': {k: [str(p) for p in v[:8]] for k, v in fair_config_hits.items()},
        'fair_weight_hits': {k: [str(p) for p in v[:8]] for k, v in fair_weight_hits.items()},
    }
    out_json = args.work_dir / 'preflight/dryrun.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    lines = [
        '# Preflight Dryrun',
        '',
        f'- generated_at: `{now()}`',
        f'- status: `{status}`',
        f'- repo_root: `{args.repo_root}`',
        f'- result_md_dir_writable: `{writable}`',
        f'- dryrun_json: `{out_json}`',
        '',
        '## Required Paths',
        '',
        '| item | status | path |',
        '|---|---|---|',
    ]
    for name, path in required.items():
        state = 'OK' if (name == 'result_md_dir_writable' and writable) or (name != 'result_md_dir_writable' and path.exists()) else 'MISSING'
        lines.append(f'| {name} | {state} | `{path}` |')
    lines.extend(['', '## DOTA1 Angle Sweep', '', '| angle | annfiles | images | ann count |'])
    lines.append('|---:|---|---|---:|')
    for angle in ANGLES:
        ann = dota_root / DOTA_SWEEP_REL / f'angle_{angle}/annfiles'
        img = dota_root / DOTA_SWEEP_REL / f'angle_{angle}/images'
        lines.append(
            f'| {angle} | {path_ok(ann)} | {path_ok(img)} | {count_ann_files(ann)} |')
    lines.extend(['', '## DOTA1 Models', '', '| model | config | checkpoint | existing 12-view predictions |'])
    lines.append('|---|---|---|---:|')
    for spec in specs.values():
        count = sum(1 for angle in ANGLES if (spec.infer_root / f'angle_{angle}/predictions.pkl').exists())
        lines.append(
            f'| {spec.display} | `{spec.config}` ({path_ok(spec.config)}) | '
            f'`{spec.checkpoint}` ({path_ok(spec.checkpoint)}) | {count}/12 |')
    lines.extend(['', '## FAR1M Discovery', '', f'- selected_root: `{fair_root}`'])
    lines.append(f'- discovered_roots: `{", ".join(str(p) for p in fair_found)}`')
    lines.append(f'- angle_sweep_dirs_found: `{len(fair_angle_dirs)}/12`')
    lines.extend(['', '| requested model | config hits | checkpoint hits |'])
    lines.append('|---|---:|---:|')
    for key in ('rtmdet_l', 'h2rbox_v2', 'retinanet_msrr'):
        lines.append(
            f'| {key} | {len(fair_config_hits[key])} | {len(fair_weight_hits[key])} |')
    lines.extend(['', '## Evaluator And Merge Checks', '', '| check | status |'])
    lines.append('|---|---|')
    for key, value in eval_support.items():
        lines.append(f'| {key} | {"OK" if value else "MISSING"} |')
    lines.append(f'| rotation_tta_merge_target_angle | {"OK" if "--target-angle" in merge_text else "MISSING"} |')
    lines.extend(['', '## Planned Commands', ''])
    for command in planned_commands:
        lines.append(f'- `{command}`')
    if hard_failures:
        lines.extend(['', '## Hard Failures', ''])
        lines.extend([f'- {item}' for item in hard_failures])
    if warnings_list:
        lines.extend(['', '## Warnings', ''])
        lines.extend([f'- {item}' for item in warnings_list])
    write_md(args.result_md_dir / 'preflight_dryrun.md', lines)
    return report


def create_dota_subset(args: argparse.Namespace,
                       angle: str,
                       max_images: int,
                       ids: list[str] | None = None) -> tuple[Path, list[str]]:
    src = args.repo_root / DOTA_DATA_REL / DOTA_SWEEP_REL / f'angle_{angle}'
    ann_src = src / 'annfiles'
    img_src = src / 'images'
    if ids is None:
        ids = []
        for ann_file in sorted(ann_src.glob('*.txt')):
            if len(ids) >= max_images:
                break
            text = ann_file.read_text(encoding='utf-8', errors='replace')
            if text.strip():
                ids.append(ann_file.stem)
    subset = args.work_dir / 'preflight/subsets/dota1' / f'angle_{angle}_n{max_images}'
    ann_dst = subset / 'annfiles'
    img_dst = subset / 'images'
    if subset.exists() and args.force:
        shutil.rmtree(subset)
    ann_dst.mkdir(parents=True, exist_ok=True)
    img_dst.mkdir(parents=True, exist_ok=True)
    for stem in ids[:max_images]:
        ann_link = ann_dst / f'{stem}.txt'
        if not ann_link.exists():
            os.symlink(ann_src / f'{stem}.txt', ann_link)
        img_match = None
        for ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
            candidate = img_src / f'{stem}{ext}'
            if candidate.exists():
                img_match = candidate
                break
        if img_match is None:
            raise FileNotFoundError(f'image missing for subset id={stem} angle={angle}')
        img_link = img_dst / img_match.name
        if not img_link.exists():
            os.symlink(img_match, img_link)
    return subset, ids[:max_images]


def test_cfg_options(data_root: Path, ann_file: str, img_path: str,
                     batch_size: int, num_workers: int) -> list[str]:
    return [
        f'test_dataloader.batch_size={batch_size}',
        f'test_dataloader.num_workers={num_workers}',
        'test_dataloader.persistent_workers=False',
        f'test_dataloader.dataset.data_root={data_root}',
        f'test_dataloader.dataset.ann_file={ann_file}',
        f'test_dataloader.dataset.data_prefix.img_path={img_path}',
    ]


def inference_command(args: argparse.Namespace,
                      spec: ModelSpec,
                      out_dir: Path,
                      pred_path: Path,
                      data_root: Path,
                      ann_file: str,
                      img_path: str,
                      batch_size: int,
                      num_workers: int,
                      distributed: bool,
                      master_port: int) -> list[str]:
    cfg_options = test_cfg_options(data_root, ann_file, img_path, batch_size, num_workers)
    if distributed:
        nproc = len(parse_csv_arg(args.gpu_ids))
        return [
            str(args.python_bin), '-m', 'torch.distributed.launch',
            f'--nproc_per_node={nproc}', f'--master_port={master_port}',
            'tools/openrsd_test.py', str(spec.config), str(spec.checkpoint),
            '--launcher', 'pytorch', '--work-dir', str(out_dir), '--out',
            str(pred_path), '--cfg-options', *cfg_options
        ]
    return [
        str(args.python_bin), 'tools/openrsd_test.py', str(spec.config),
        str(spec.checkpoint), '--work-dir', str(out_dir), '--out',
        str(pred_path), '--cfg-options', *cfg_options
    ]


def eval_command(args: argparse.Namespace,
                 spec: ModelSpec,
                 pkl_path: Path,
                 data_root: Path,
                 ann_file: str,
                 img_path: str) -> list[str]:
    cfg_options = [
        f'test_dataloader.num_workers={args.num_workers}',
        'test_dataloader.persistent_workers=False',
        f'test_dataloader.dataset.data_root={data_root}',
        f'test_dataloader.dataset.ann_file={ann_file}',
        f'test_dataloader.dataset.data_prefix.img_path={img_path}',
    ]
    return [
        str(args.python_bin), 'tools/openrsd_eval_metric.py', str(spec.config),
        str(pkl_path), '--cfg-options', *cfg_options
    ]


def ensure_prediction(args: argparse.Namespace,
                      runner: CommandRunner,
                      spec: ModelSpec,
                      angle: str,
                      out_root: Path,
                      batch_size: int,
                      distributed: bool,
                      master_port: int) -> tuple[Path, CommandResult | None]:
    existing = spec.infer_root / f'angle_{angle}/predictions.pkl'
    if existing.exists() and not args.force:
        return existing, None
    out_dir = out_root / spec.key / f'angle_{angle}'
    pred_path = out_dir / 'predictions.pkl'
    if pred_path.exists() and args.resume and not args.force:
        return pred_path, None
    argv = inference_command(
        args=args,
        spec=spec,
        out_dir=out_dir,
        pred_path=pred_path,
        data_root=args.repo_root / DOTA_DATA_REL,
        ann_file=f'{DOTA_SWEEP_REL}/angle_{angle}/annfiles/',
        img_path=f'{DOTA_SWEEP_REL}/angle_{angle}/images/',
        batch_size=batch_size,
        num_workers=args.num_workers,
        distributed=distributed,
        master_port=master_port)
    result = runner.run(
        f'infer_{spec.key}_angle_{angle}',
        argv,
        out_dir / 'test.log',
        gpu_ids=args.gpu_ids,
        monitor_gpu=True)
    return pred_path, result


def merge_command(args: argparse.Namespace,
                  source_pkls: Sequence[Path],
                  source_angles: Sequence[str],
                  out_path: Path,
                  target_angle: str,
                  score_thr: float = 0.05,
                  nms_iou: float = 0.1,
                  max_per_img: int = 2000) -> list[str]:
    argv = [
        str(args.python_bin), 'M_Tools/analysis/rotation_tta_merge.py',
        '--out', str(out_path), '--target-angle', str(int(target_angle)),
        '--img-shape', '1024', '1024', '--score-thr', str(score_thr),
        '--pre-nms-topk', '4000', '--nms-iou', str(nms_iou), '--max-per-img',
        str(max_per_img)
    ]
    for pkl_path, angle in zip(source_pkls, source_angles):
        argv.extend(['--prediction', str(pkl_path), '--angle', str(int(angle))])
    return argv


def ensure_merge(args: argparse.Namespace,
                 runner: CommandRunner,
                 spec: ModelSpec,
                 source_pkls: Sequence[Path],
                 source_angles: Sequence[str] | None,
                 target_angle: str,
                 out_dir: Path,
                 score_thr: float = 0.05,
                 nms_iou: float = 0.1,
                 max_per_img: int = 2000,
                 name: str | None = None) -> tuple[Path, CommandResult | None]:
    out_path = out_dir / 'merged_predictions.pkl'
    log_path = out_dir / 'merge.log'
    if out_path.exists() and args.resume and not args.force:
        return out_path, None
    if source_angles is None:
        source_angles = ANGLES[:len(source_pkls)]
    argv = merge_command(args, source_pkls, source_angles, out_path,
                         target_angle, score_thr, nms_iou, max_per_img)
    result = runner.run(
        name or f'merge_{spec.key}_target_{target_angle}',
        argv,
        log_path,
        gpu_ids=args.gpu_ids)
    return out_path, result


def evaluate_pkl(args: argparse.Namespace,
                 runner: CommandRunner,
                 spec: ModelSpec,
                 pkl_path: Path,
                 angle: str,
                 out_dir: Path,
                 subset_root: Path | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'eval.log'
    cache_path = out_dir / 'eval.json'
    if cache_path.exists() and log_path.exists() and args.resume and not args.force:
        return json.loads(cache_path.read_text(encoding='utf-8'))
    if subset_root is None:
        data_root = args.repo_root / DOTA_DATA_REL
        ann_file = f'{DOTA_SWEEP_REL}/angle_{angle}/annfiles/'
        img_path = f'{DOTA_SWEEP_REL}/angle_{angle}/images/'
    else:
        data_root = subset_root
        ann_file = 'annfiles/'
        img_path = 'images/'
    result = runner.run(
        f'eval_{spec.key}_angle_{angle}_{out_dir.name}',
        eval_command(args, spec, pkl_path, data_root, ann_file, img_path),
        log_path,
        gpu_ids=args.gpu_ids)
    metrics = parse_metrics_from_log(log_path)
    payload = {
        'status': 'OK' if result.returncode == 0 and metrics.get('ap50') is not None else 'FAILED',
        'metrics': metrics,
        'command': result.command,
        'log': str(log_path),
        'pkl': str(pkl_path),
        'returncode': result.returncode,
        'failure_kind': result.failure_kind,
        'tail': result.tail,
    }
    cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    return payload


def pkl_ids(path: Path) -> set[str]:
    rows = load_pickle(path)
    return {normalize_img_id(row.get('img_id', '')) for row in rows}


def run_smoke(args: argparse.Namespace) -> dict:
    specs = build_dota_specs(args.repo_root, args.weights_dir)
    model = parse_csv_arg(args.only_model)[0] if args.only_model else 'rtmdet_l'
    angles = parse_angle_list(args.only_angle, ['000', '030'])
    if model not in specs:
        raise KeyError(f'unknown smoke model: {model}')
    if len(angles) < 2:
        angles = angles + ['030']
    spec = specs[model]
    runner = CommandRunner(args)
    base_subset, ids = create_dota_subset(args, angles[0], args.max_images_for_smoke)
    subsets = {angles[0]: base_subset}
    for angle in angles[1:]:
        subset, _ = create_dota_subset(args, angle, args.max_images_for_smoke, ids)
        subsets[angle] = subset

    rows = []
    pred_paths: dict[str, Path] = {}
    failures = []
    for angle in angles[:2]:
        out_dir = args.work_dir / 'preflight/smoke' / spec.key / f'angle_{angle}'
        pred_path = out_dir / 'predictions.pkl'
        if pred_path.exists() and args.resume and not args.force:
            command_result = None
        else:
            argv = inference_command(
                args, spec, out_dir, pred_path, subsets[angle], 'annfiles/',
                'images/', args.batch_size or 1, args.num_workers, False, 0)
            command_result = runner.run(
                f'smoke_infer_{spec.key}_angle_{angle}',
                argv,
                out_dir / 'test.log',
                gpu_ids=args.gpu_ids,
                monitor_gpu=True)
            if command_result.returncode != 0:
                failures.append(command_result)
        pred_paths[angle] = pred_path
        readable = False
        count = 0
        ids_ok = False
        if pred_path.exists():
            try:
                pred_rows = load_pickle(pred_path)
                readable = True
                count = len(pred_rows)
                pred_ids = pkl_ids(pred_path)
                ids_ok = pred_ids.issubset(set(ids)) and bool(pred_ids)
            except Exception:  # noqa: BLE001
                readable = False
        rows.append({
            'angle': angle,
            'predictions': str(pred_path),
            'readable': readable,
            'count': count,
            'ids_ok': ids_ok,
            'log': str(out_dir / 'test.log'),
        })

    merge_dir = args.work_dir / 'preflight/smoke' / spec.key / 'merge_target_000'
    merge_pkl = merge_dir / 'merged_predictions.pkl'
    merge_eval = {
        'status': 'FAILED',
        'metrics': {'ap50': None, 'map': None},
        'log': str(merge_dir / 'eval' / 'eval.log'),
    }
    single_eval = {
        'status': 'FAILED',
        'metrics': {'ap50': None, 'map': None},
        'log': str(args.work_dir / 'preflight/smoke' / spec.key / 'single_eval' / 'eval.log'),
    }
    merged_readable = False
    merged_count = 0
    merged_ids_ok = False
    try:
        merge_pkl, merge_result = ensure_merge(
            args,
            runner,
            spec,
            [pred_paths[angles[0]], pred_paths[angles[1]]],
            angles[:2],
            angles[0],
            merge_dir,
            name='smoke_merge_rtmdet_l_000_030')
        if merge_result and merge_result.returncode != 0:
            failures.append(merge_result)
        single_eval = evaluate_pkl(args, runner, spec, pred_paths[angles[0]], angles[0],
                                   args.work_dir / 'preflight/smoke' / spec.key / 'single_eval',
                                   subset_root=subsets[angles[0]])
        merge_eval = evaluate_pkl(args, runner, spec, merge_pkl, angles[0],
                                  merge_dir / 'eval', subset_root=subsets[angles[0]])
        if merge_pkl.exists():
            try:
                merged_rows = load_pickle(merge_pkl)
                merged_readable = True
                merged_count = len(merged_rows)
                merged_ids_ok = pkl_ids(merge_pkl).issubset(set(ids)) and bool(pkl_ids(merge_pkl))
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001
        failures.append(CommandResult(
            name='smoke_merge_or_eval',
            command='merge/eval',
            returncode=1,
            log_path=str(merge_dir / 'merge.log'),
            duration_sec=0.0,
            status='FAILED',
            failure_kind='traceback',
            tail=traceback.format_exc(),
        ))

    status = 'OK'
    if failures or not merged_readable or not merged_ids_ok or merge_eval['status'] != 'OK':
        status = 'FAILED'
    report = {
        'status': status,
        'model': model,
        'angles': angles[:2],
        'subset_ids': ids,
        'rows': rows,
        'merged_predictions': str(merge_pkl),
        'merged_readable': merged_readable,
        'merged_count': merged_count,
        'merged_ids_ok': merged_ids_ok,
        'single_eval': single_eval,
        'merge_eval': merge_eval,
        'failures': [r.__dict__ for r in failures],
    }
    out_json = args.work_dir / 'preflight/smoke/smoke.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = [
        '# Preflight Smoke',
        '',
        f'- generated_at: `{now()}`',
        f'- status: `{status}`',
        f'- model: `{spec.display}`',
        f'- physical GPU ids: `{args.gpu_ids}`',
        f'- max_images_for_smoke: `{args.max_images_for_smoke}`',
        f'- smoke_json: `{out_json}`',
        '',
        '## Prediction Read Checks',
        '',
        '| angle | readable | sample count | img_id aligns GT subset | predictions | log |',
        '|---:|---|---:|---|---|---|',
    ]
    for row in rows:
        lines.append(
            f'| {row["angle"]} | {row["readable"]} | {row["count"]} | '
            f'{row["ids_ok"]} | `{row["predictions"]}` | `{row["log"]}` |')
    lines.extend([
        '',
        '## Merge And Eval',
        '',
        f'- merged_predictions: `{merge_pkl}`',
        f'- merged_readable: `{merged_readable}`',
        f'- merged_count: `{merged_count}`',
        f'- merged_img_id_aligns_gt: `{merged_ids_ok}`',
        f'- single AP50: `{fmt(single_eval["metrics"].get("ap50"))}`',
        f'- merged AP50: `{fmt(merge_eval["metrics"].get("ap50"))}`',
        f'- single eval log: `{single_eval["log"]}`',
        f'- merged eval log: `{merge_eval["log"]}`',
    ])
    if failures:
        lines.extend(['', '## Failures', ''])
        for failure in failures:
            lines.append(f'- {failure.name}: `{failure.status}` log=`{failure.log_path}` kind=`{failure.failure_kind}`')
    write_md(args.result_md_dir / 'preflight_smoke.md', lines)
    return report


def run_multigpu_batchsize(args: argparse.Namespace) -> dict:
    specs = build_dota_specs(args.repo_root, args.weights_dir)
    spec = specs['rtmdet_l']
    runner = CommandRunner(args)
    candidates = parse_int_csv(args.batch_size_candidates, [1, 2, 4, 8])
    subset, ids = create_dota_subset(args, '000', max(args.max_images_for_smoke, 32))
    rows = []
    stable = None
    previous_oom = False
    for index, batch_size in enumerate(candidates):
        if previous_oom and batch_size > (stable or 1):
            rows.append({
                'batch_size': batch_size,
                'status': 'SKIPPED_AFTER_OOM',
                'log': '',
                'predictions': '',
                'duration_sec': 0,
                'avg_time_per_image': None,
                'peak_mem_mb': {},
                'process_seen': {},
            })
            continue
        out_dir = args.work_dir / 'preflight/multigpu_batchsize' / spec.key / f'bs_{batch_size}'
        pred_path = out_dir / 'predictions.pkl'
        if pred_path.exists() and args.resume and not args.force:
            result = None
        else:
            argv = inference_command(
                args, spec, out_dir, pred_path, subset, 'annfiles/',
                'images/', batch_size, args.num_workers, True,
                38200 + index)
            result = runner.run(
                f'multigpu_bs_{batch_size}',
                argv,
                out_dir / 'test.log',
                gpu_ids=args.gpu_ids,
                monitor_gpu=True)
        complete = False
        count = 0
        if pred_path.exists():
            try:
                count = len(load_pickle(pred_path))
                complete = count == len(ids)
            except Exception:  # noqa: BLE001
                complete = False
        status = 'OK' if (result is None or result.returncode == 0) and complete else 'FAILED'
        failure_kind = '' if result is None else result.failure_kind
        if failure_kind == 'oom' or (result and 'out of memory' in read_text(out_dir / 'test.log').lower()):
            status = 'OOM'
            previous_oom = True
        if status == 'OK':
            stable = batch_size
        duration = 0.0 if result is None else result.duration_sec
        rows.append({
            'batch_size': batch_size,
            'status': status,
            'log': str(out_dir / 'test.log'),
            'predictions': str(pred_path),
            'sample_count': count,
            'expected_count': len(ids),
            'duration_sec': duration,
            'avg_time_per_image': duration / max(count, 1) if duration else None,
            'peak_mem_mb': {} if result is None else (result.peak_mem_mb or {}),
            'process_seen': {} if result is None else (result.process_seen or {}),
            'failure_kind': failure_kind,
        })
    if stable is None:
        stable = 1
    report = {
        'status': 'OK' if any(row['status'] == 'OK' for row in rows) else 'FAILED',
        'selected_batch_size': stable,
        'rows': rows,
        'subset': str(subset),
        'gpu_ids': args.gpu_ids,
    }
    out_json = args.work_dir / 'preflight/multigpu_batchsize/multigpu_batchsize.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = [
        '# Preflight Multi-GPU Batch Size',
        '',
        f'- generated_at: `{now()}`',
        f'- status: `{report["status"]}`',
        f'- physical GPU ids: `{args.gpu_ids}`',
        f'- distributed inference: `python -m torch.distributed.launch`',
        f'- selected_batch_size: `{stable}`',
        f'- report_json: `{out_json}`',
        '',
        '| batch size | status | pkl count | avg sec/img | peak mem MB by GPU | process seen by GPU | log |',
        '|---:|---|---:|---:|---|---|---|',
    ]
    for row in rows:
        lines.append(
            f'| {row["batch_size"]} | {row["status"]} | '
            f'{row.get("sample_count", 0)}/{row.get("expected_count", len(ids))} | '
            f'{fmt(row.get("avg_time_per_image"))} | '
            f'`{row.get("peak_mem_mb", {})}` | `{row.get("process_seen", {})}` | '
            f'`{row.get("log", "")}` |')
    write_md(args.result_md_dir / 'preflight_multigpu_batchsize.md', lines)
    (args.work_dir / 'preflight/selected_batch_size.json').write_text(
        json.dumps({'rtmdet_l': stable}, indent=2), encoding='utf-8')
    return report


def source_pkls_for_model(args: argparse.Namespace,
                          runner: CommandRunner,
                          spec: ModelSpec,
                          batch_size: int) -> tuple[dict[str, Path], list[dict]]:
    pkls = {}
    failures = []
    for idx, angle in enumerate(ANGLES):
        pred, result = ensure_prediction(
            args, runner, spec, angle, args.work_dir / 'exp1/infer',
            batch_size, distributed=len(parse_csv_arg(args.gpu_ids)) > 1,
            master_port=38300 + idx)
        pkls[angle] = pred
        if result and result.returncode != 0:
            failures.append(result.__dict__)
    return pkls, failures


def summarize_model_curve(rows: list[dict]) -> dict:
    ok_rows = [row for row in rows if row.get('status') == 'OK']
    if not ok_rows:
        return {}
    single_vals = [row['single_ap50'] for row in ok_rows]
    tta_vals = [row['tta_ap50'] for row in ok_rows]
    deltas = [row['delta_ap50'] for row in ok_rows]
    single_worst = min(ok_rows, key=lambda row: row['single_ap50'])
    single_best = max(ok_rows, key=lambda row: row['single_ap50'])
    tta_worst = min(ok_rows, key=lambda row: row['tta_ap50'])
    tta_best = max(ok_rows, key=lambda row: row['tta_ap50'])
    return {
        'model': ok_rows[0]['model'],
        'display': ok_rows[0]['display'],
        'single_mean': statistics.mean(single_vals),
        'tta_mean': statistics.mean(tta_vals),
        'single_std': statistics.pstdev(single_vals),
        'tta_std': statistics.pstdev(tta_vals),
        'single_worst_angle': single_worst['angle'],
        'single_worst_ap50': single_worst['single_ap50'],
        'single_best_angle': single_best['angle'],
        'single_best_ap50': single_best['single_ap50'],
        'tta_worst_angle': tta_worst['angle'],
        'tta_worst_ap50': tta_worst['tta_ap50'],
        'tta_best_angle': tta_best['angle'],
        'tta_best_ap50': tta_best['tta_ap50'],
        'worst_angle_gain': single_worst['delta_ap50'],
        'best_angle_damage': single_best['delta_ap50'],
        'single_rsi': single_worst['single_ap50'] / max(statistics.mean(single_vals), 1e-12),
        'tta_rsi': tta_worst['tta_ap50'] / max(statistics.mean(tta_vals), 1e-12),
        'mean_delta': statistics.mean(deltas),
    }


def run_experiment_1(args: argparse.Namespace) -> dict:
    specs_all = build_dota_specs(args.repo_root, args.weights_dir)
    model_keys = [key for key in select_model_keys(args) if key in specs_all]
    angles = parse_angle_list(args.only_angle, ANGLES)
    runner = CommandRunner(args)
    selected_batch = args.batch_size or read_selected_batch_size(args, default=1)
    rows = []
    failures = []
    commands = []
    for key in model_keys:
        spec = specs_all[key]
        source_pkls, pred_failures = source_pkls_for_model(args, runner, spec, selected_batch)
        failures.extend(pred_failures)
        for target in angles:
            row = {
                'model': spec.key,
                'display': spec.display,
                'angle': target,
                'status': 'OK',
                'config': str(spec.config),
                'checkpoint': str(spec.checkpoint),
            }
            try:
                source_list = [source_pkls[angle] for angle in ANGLES]
                merge_dir = args.work_dir / 'exp1/merge' / spec.key / f'target_{target}'
                merged_pkl, merge_result = ensure_merge(args, runner, spec, source_list,
                                                        ANGLES, target, merge_dir)
                if merge_result:
                    commands.append(merge_result.command)
                    if merge_result.returncode != 0:
                        raise RuntimeError(f'merge failed: {merge_result.log_path}')
                single_eval = evaluate_pkl(
                    args, runner, spec, source_pkls[target], target,
                    args.work_dir / 'exp1/eval' / spec.key / f'target_{target}/single')
                tta_eval = evaluate_pkl(
                    args, runner, spec, merged_pkl, target,
                    args.work_dir / 'exp1/eval' / spec.key / f'target_{target}/tta')
                row.update({
                    'single_ap50': single_eval['metrics']['ap50'],
                    'tta_ap50': tta_eval['metrics']['ap50'],
                    'delta_ap50': tta_eval['metrics']['ap50'] - single_eval['metrics']['ap50'],
                    'single_map': single_eval['metrics']['map'],
                    'tta_map': tta_eval['metrics']['map'],
                    'single_pkl': str(source_pkls[target]),
                    'tta_pkl': str(merged_pkl),
                    'single_log': single_eval['log'],
                    'tta_log': tta_eval['log'],
                })
                if single_eval['status'] != 'OK' or tta_eval['status'] != 'OK':
                    row['status'] = 'FAILED'
                    failures.append(row)
            except Exception as exc:  # noqa: BLE001
                row['status'] = 'FAILED'
                row['error'] = str(exc)
                failures.append(row)
            rows.append(row)

    report = {'status': 'DONE' if not failures else 'PARTIAL', 'rows': rows, 'failures': failures}
    out_json = args.work_dir / 'exp1/exp1_results.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    write_exp1_md(args, report, selected_batch)
    return report


def write_exp1_md(args: argparse.Namespace, report: dict, batch_size: int) -> None:
    rows = report['rows']
    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(row['model'], []).append(row)
    lines = [
        '# Experiment 1: DOTA1 Full-Angle TTA Repair Curve',
        '',
        f'- generated_at: `{now()}`',
        f'- status: `{report["status"]}`',
        '- metric: DOTA AP50 / mAP@0.5',
        f'- data_path: `{args.repo_root / DOTA_DATA_REL / DOTA_SWEEP_REL}`',
        f'- work_dir: `{args.work_dir / "exp1"}`',
        f'- final batch size for missing inference: `{batch_size}`',
        '- Rotation Stability Index (RSI): `worst-angle AP50 / mean AP50`.',
        '',
        '## Model Inputs',
        '',
        '| model | config | checkpoint |',
        '|---|---|---|',
    ]
    for model_rows in by_model.values():
        first = model_rows[0]
        lines.append(f'| {first["display"]} | `{first["config"]}` | `{first["checkpoint"]}` |')
    lines.extend(['', '## Per-Angle Results', ''])
    for model, model_rows in by_model.items():
        model_rows = sorted(model_rows, key=lambda row: int(row['angle']))
        lines.append(f'### {model_rows[0]["display"]}')
        lines.extend(['', '| target angle | status | single AP50 | TTA AP50 | delta | single log | TTA log |'])
        lines.append('|---:|---|---:|---:|---:|---|---|')
        for row in model_rows:
            lines.append(
                f'| {row["angle"]} | {row.get("status", "NA")} | '
                f'{fmt(row.get("single_ap50"))} | {fmt(row.get("tta_ap50"))} | '
                f'{fmt(row.get("delta_ap50"))} | `{row.get("single_log", "")}` | '
                f'`{row.get("tta_log", "")}` |')
        lines.append('')
    lines.extend(['## Model Statistics', '', '| model | single mean | TTA mean | single worst | TTA worst | single best | TTA best | single std | TTA std | worst-angle gain | best-angle damage | single RSI | TTA RSI |'])
    lines.append('|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|')
    summaries = []
    for model_rows in by_model.values():
        summary = summarize_model_curve(model_rows)
        if not summary:
            continue
        summaries.append(summary)
        lines.append(
            f'| {summary["display"]} | {fmt(summary["single_mean"])} | {fmt(summary["tta_mean"])} | '
            f'angle_{summary["single_worst_angle"]} ({fmt(summary["single_worst_ap50"])}) | '
            f'angle_{summary["tta_worst_angle"]} ({fmt(summary["tta_worst_ap50"])}) | '
            f'angle_{summary["single_best_angle"]} ({fmt(summary["single_best_ap50"])}) | '
            f'angle_{summary["tta_best_angle"]} ({fmt(summary["tta_best_ap50"])}) | '
            f'{fmt(summary["single_std"])} | {fmt(summary["tta_std"])} | '
            f'{fmt(summary["worst_angle_gain"])} | {fmt(summary["best_angle_damage"])} | '
            f'{fmt(summary["single_rsi"])} | {fmt(summary["tta_rsi"])} |')
    failed = [row for row in rows if row.get('status') != 'OK']
    lines.extend(['', '## Failed Angles', ''])
    if failed:
        for row in failed:
            lines.append(f'- {row.get("display", row.get("model"))} angle_{row.get("angle")}: {row.get("error", "see logs")}')
    else:
        lines.append('- None.')
    lines.extend(['', '## Reproduction Commands', ''])
    lines.append(
        f'- `rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mplconfig CUDA_VISIBLE_DEVICES={args.gpu_ids} '
        f'PYTHONPATH={args.repo_root}:{args.repo_root / "tools"} {args.python_bin} '
        f'M_Tools/analysis/run_rotation_repair_suite.py --repo-root {args.repo_root} '
        f'--result-md-dir {args.result_md_dir} --weights-dir {args.weights_dir} '
        f'--work-dir {args.work_dir} --gpu-ids {args.gpu_ids} --exp 1 --mode full --resume`')
    lines.extend(['', '## Conclusions', ''])
    for summary in summaries:
        raises_worst = summary['worst_angle_gain'] > 0
        lowers_var = summary['tta_std'] < summary['single_std']
        canonical = next((row for row in by_model[summary['model']] if row['angle'] == '000' and row.get('status') == 'OK'), None)
        canonical_delta = canonical.get('delta_ap50') if canonical else None
        lines.append(
            f'- {summary["display"]}: worst-angle AP50 is {"raised" if raises_worst else "not raised"} '
            f'({fmt(summary["worst_angle_gain"])} at single worst angle), angle variance is '
            f'{"reduced" if lowers_var else "not reduced"} ({fmt(summary["single_std"])} -> {fmt(summary["tta_std"])}), '
            f'canonical delta is {fmt(canonical_delta)}.')
    if summaries:
        best_gain = max(summaries, key=lambda row: row['mean_delta'])
        worst_gain = min(summaries, key=lambda row: row['mean_delta'])
        lines.append(
            f'- Across the three models, mean TTA effect is strongest for {best_gain["display"]} '
            f'({fmt(best_gain["mean_delta"])}) and weakest for {worst_gain["display"]} ({fmt(worst_gain["mean_delta"])}).')
    write_md(args.result_md_dir / 'exp1_dota1_full_angle_tta_repair_curve.md', lines)


def read_selected_batch_size(args: argparse.Namespace, default: int = 1) -> int:
    path = args.work_dir / 'preflight/selected_batch_size.json'
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return int(data.get('rtmdet_l', default))
    except Exception:  # noqa: BLE001
        return default


def run_experiment_2(args: argparse.Namespace) -> dict:
    fair_root, fair_found = discover_fair1m_root(args.repo_root)
    angle_dirs = []
    if fair_root:
        angle_dirs = [
            fair_root / f'rot_val_standard/realistic/angle_{angle}'
            for angle in ANGLES
            if (fair_root / f'rot_val_standard/realistic/angle_{angle}').exists()
        ]
    fair_config_hits = {
        'rtmdet_l': discover_paths(args.repo_root, ['fair1m', 'rtmdet']),
        'h2rbox_v2': discover_paths(args.repo_root, ['fair1m', 'h2rbox']),
        'retinanet_msrr': discover_paths(args.repo_root, ['fair1m', 'retinanet']),
    }
    fair_weight_hits = {
        'rtmdet_l': discover_weights([args.weights_dir, args.repo_root / 'work_dirs'], ['fair1m', 'rtmdet']),
        'h2rbox_v2': discover_weights([args.weights_dir, args.repo_root / 'work_dirs'], ['fair1m', 'h2rbox']),
        'retinanet_msrr': discover_weights([args.weights_dir, args.repo_root / 'work_dirs'], ['fair1m', 'retinanet']),
    }
    redet_existing = sorted((args.repo_root / 'work_dirs').glob('rotation_study_36h_*/P2/FAIR1M/ReDet_Re50_FAIR1M_e12/angle_*/predictions.pkl'))
    runnable = [
        key for key in ('rtmdet_l', 'h2rbox_v2', 'retinanet_msrr')
        if fair_config_hits[key] and fair_weight_hits[key]
    ]
    status = 'NOT_RUN' if not runnable else 'PARTIAL'
    reasons = []
    if not fair_root:
        reasons.append('FAR1M/FAIR1M data root not found.')
    if len(angle_dirs) == 0:
        reasons.append('FAIR1M angle sweep split not found.')
    for key in ('rtmdet_l', 'h2rbox_v2', 'retinanet_msrr'):
        if not fair_config_hits[key]:
            reasons.append(f'{key}: FAIR1M config missing.')
        if not fair_weight_hits[key]:
            reasons.append(f'{key}: FAIR1M checkpoint missing.')
    report = {
        'status': status,
        'fair_root': str(fair_root) if fair_root else None,
        'fair_found': [str(p) for p in fair_found],
        'angle_dirs_found': [str(p) for p in angle_dirs],
        'fair_config_hits': {k: [str(p) for p in v[:10]] for k, v in fair_config_hits.items()},
        'fair_weight_hits': {k: [str(p) for p in v[:10]] for k, v in fair_weight_hits.items()},
        'runnable_models': runnable,
        'reasons': reasons,
        'auxiliary_redet_existing_predictions': [str(p) for p in redet_existing[:12]],
    }
    out_json = args.work_dir / 'exp2/exp2_discovery.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = [
        '# Experiment 2: FAR1M Angle Sweep Cross-Dataset Validation',
        '',
        f'- generated_at: `{now()}`',
        f'- status: `{status}`',
        '- metric target: DOTA-style AP50 / mAP@0.5 for oriented boxes.',
        f'- discovery_json: `{out_json}`',
        '',
        '## FAR1M Data Discovery',
        '',
        f'- selected_root: `{fair_root}`',
        f'- discovered_roots: `{", ".join(str(p) for p in fair_found)}`',
        f'- angle_sweep_dirs_found: `{len(angle_dirs)}/12`',
        '',
        '| angle | split path | annfiles | images |',
        '|---:|---|---|---|',
    ]
    for angle in ANGLES:
        p = fair_root / f'rot_val_standard/realistic/angle_{angle}' if fair_root else Path('')
        lines.append(f'| {angle} | `{p}` | {path_ok(p / "annfiles")} | {path_ok(p / "images")} |')
    lines.extend(['', '## Requested Model Artifacts', '', '| model | config hits | checkpoint hits | first config | first checkpoint |'])
    lines.append('|---|---:|---:|---|---|')
    for key in ('rtmdet_l', 'h2rbox_v2', 'retinanet_msrr'):
        lines.append(
            f'| {key} | {len(fair_config_hits[key])} | {len(fair_weight_hits[key])} | '
            f'`{fair_config_hits[key][0] if fair_config_hits[key] else ""}` | '
            f'`{fair_weight_hits[key][0] if fair_weight_hits[key] else ""}` |')
    lines.extend(['', '## Result', ''])
    if reasons:
        lines.append('Experiment 2 was not run for the requested model set because:')
        for reason in reasons:
            lines.append(f'- {reason}')
    if redet_existing:
        lines.extend([
            '',
            '## Existing Auxiliary FAIR1M Artifacts',
            '',
            '- Existing ReDet FAIR1M angle-sweep predictions were found, but ReDet is outside the requested RTMDet-L / H2RBox-v2 / RetinaNet model priority and was not counted as this experiment result.',
        ])
        for path in redet_existing[:12]:
            lines.append(f'- `{path}`')
    lines.extend([
        '',
        '## DOTA1 Comparison Summary',
        '',
        '- Requested-model FAR1M AP50 curves are unavailable, so no valid cross-dataset comparison is made here.',
        '- The missing artifact pattern is a reproducibility blocker: FAIR1M data are prepared, but matching requested-model checkpoints are not present.',
        '',
        '## Reproduction Command',
        '',
        f'- `rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mplconfig CUDA_VISIBLE_DEVICES={args.gpu_ids} '
        f'PYTHONPATH={args.repo_root}:{args.repo_root / "tools"} {args.python_bin} '
        f'M_Tools/analysis/run_rotation_repair_suite.py --repo-root {args.repo_root} '
        f'--result-md-dir {args.result_md_dir} --weights-dir {args.weights_dir} '
        f'--work-dir {args.work_dir} --gpu-ids {args.gpu_ids} --exp 2 --mode full --resume`',
    ])
    write_md(args.result_md_dir / 'exp2_far1m_angle_sweep_cross_dataset.md', lines)
    return report


def run_experiment_3(args: argparse.Namespace) -> dict:
    specs = build_dota_specs(args.repo_root, args.weights_dir)
    spec = specs['rtmdet_l']
    runner = CommandRunner(args)
    selected_batch = args.batch_size or read_selected_batch_size(args, default=1)
    source_pkls, failures = source_pkls_for_model(args, runner, spec, selected_batch)
    single_eval = evaluate_pkl(
        args, runner, spec, source_pkls['000'], '000',
        args.work_dir / 'exp3/baseline/single_angle000')
    default_merge_dir = args.work_dir / 'exp3/baseline/default_tta'
    default_pkl, default_merge_result = ensure_merge(
        args, runner, spec, [source_pkls[a] for a in ANGLES], ANGLES,
        '000', default_merge_dir)
    if default_merge_result and default_merge_result.returncode != 0:
        failures.append(default_merge_result.__dict__)
    default_eval = evaluate_pkl(args, runner, spec, default_pkl, '000',
                                default_merge_dir / 'eval')
    combos = list(itertools.product([0.1, 0.3, 0.5, 0.7], [0.001, 0.01, 0.05],
                                    [1000, 2000]))
    rows = []
    for nms_iou, score_thr, max_per_img in combos:
        combo_name = f'nms{nms_iou}_score{score_thr}_max{max_per_img}'.replace('.', 'p')
        merge_dir = args.work_dir / 'exp3/ablations' / combo_name
        row = {
            'nms_iou': nms_iou,
            'score_thr': score_thr,
            'max_per_img': max_per_img,
            'status': 'OK',
            'merge_dir': str(merge_dir),
        }
        try:
            start = time.time()
            merged_pkl, merge_result = ensure_merge(
                args,
                runner,
                spec,
                [source_pkls[a] for a in ANGLES],
                ANGLES,
                '000',
                merge_dir,
                score_thr=score_thr,
                nms_iou=nms_iou,
                max_per_img=max_per_img,
                name=f'exp3_merge_{combo_name}')
            if merge_result and merge_result.returncode != 0:
                raise RuntimeError(f'merge failed: {merge_result.log_path}')
            eval_payload = evaluate_pkl(args, runner, spec, merged_pkl, '000',
                                        merge_dir / 'eval')
            stats = compute_merge_stats_for_pkl(source_pkls, merged_pkl, '000',
                                                score_thr, nms_iou, max_per_img)
            row.update({
                'ap50': eval_payload['metrics']['ap50'],
                'delta_vs_single': eval_payload['metrics']['ap50'] - single_eval['metrics']['ap50'],
                'delta_vs_default_tta': eval_payload['metrics']['ap50'] - default_eval['metrics']['ap50'],
                'avg_boxes_before_nms': stats['avg_before'],
                'avg_boxes_after_nms': stats['avg_after'],
                'abnormal_images': stats['abnormal_images'],
                'duration_sec': time.time() - start,
                'merged_pkl': str(merged_pkl),
                'eval_log': eval_payload['log'],
            })
        except Exception as exc:  # noqa: BLE001
            row['status'] = 'FAILED'
            row['error'] = str(exc)
            failures.append(row)
        rows.append(row)
    status = 'DONE' if not [r for r in rows if r['status'] != 'OK'] else 'PARTIAL'
    report = {
        'status': status,
        'baseline_single': single_eval,
        'baseline_default_tta': default_eval,
        'rows': rows,
        'failures': failures,
    }
    out_json = args.work_dir / 'exp3/exp3_ablation_results.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    write_exp3_md(args, report)
    return report


def compute_merge_stats_for_pkl(source_pkls: dict[str, Path], merged_pkl: Path,
                                target_angle: str, score_thr: float,
                                nms_iou: float, max_per_img: int) -> dict:
    ensure_mm(Path(__file__).resolve().parents[2])
    import cv2
    import numpy as np
    import torch
    from mmcv.ops import nms_rotated
    from mmrotate.structures.bbox import qbox2rbox, rbox2qbox

    def rotation_matrix(width: int, height: int, angle: float):
        center = ((width - 1) / 2.0, (height - 1) / 2.0)
        return cv2.getRotationMatrix2D(center, float(angle), 1.0)

    def transform_qboxes(qboxes, matrix):
        if qboxes.size == 0:
            return qboxes
        points = qboxes.reshape(-1, 2).astype(np.float32)
        hom = np.concatenate(
            [points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
        transformed = hom @ matrix.T
        return transformed.reshape(-1, 8).astype(np.float32)

    def transform_boxes_to_target(rboxes, source_angle, target_angle, width,
                                  height):
        if len(rboxes) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        qboxes = rbox2qbox(torch.from_numpy(rboxes.astype(np.float32))).numpy()
        matrix = rotation_matrix(width, height, target_angle - source_angle)
        target_qboxes = transform_qboxes(qboxes, matrix)
        return qbox2rbox(torch.from_numpy(target_qboxes)).numpy().astype(np.float32)

    def merge_one_image(items):
        if not items:
            return 0, 0
        boxes = np.concatenate([item['bboxes'] for item in items], axis=0)
        labels = np.concatenate([item['labels'] for item in items], axis=0)
        scores = np.concatenate([item['scores'] for item in items], axis=0)
        keep = scores >= score_thr
        boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
        if len(scores) > max_per_img:
            order = np.argsort(-scores)[:max_per_img]
            boxes, labels, scores = boxes[order], labels[order], scores[order]
        before = int(len(scores))
        out_boxes = []
        for label in sorted(set(labels.tolist())):
            inds = np.where(labels == label)[0]
            if len(inds) == 0:
                continue
            label_boxes = torch.from_numpy(boxes[inds].astype(np.float32))
            label_scores = torch.from_numpy(scores[inds].astype(np.float32))
            nms_dets, _ = nms_rotated(label_boxes, label_scores, nms_iou)
            out_boxes.append(nms_dets[:, :5])
        after = int(sum(len(b) for b in out_boxes))
        return before, after

    grouped: dict[str, list[dict]] = {}
    for source_angle, pkl_path in source_pkls.items():
        angle_int = int(source_angle)
        for sample in load_pickle(pkl_path):
            img_id = normalize_img_id(sample['img_id'])
            pred = sample['pred_instances']
            boxes = to_numpy(pred['bboxes']).astype(np.float32)
            labels = to_numpy(pred['labels']).astype(np.int64)
            scores = to_numpy(pred['scores']).astype(np.float32)
            inv_boxes = transform_boxes_to_target(boxes, angle_int,
                                                  int(target_angle), 1024,
                                                  1024)
            grouped.setdefault(img_id, []).append(
                {'bboxes': inv_boxes, 'labels': labels, 'scores': scores})

    merged_rows = load_pickle(merged_pkl)
    merged_map = {normalize_img_id(sample['img_id']): sample for sample in merged_rows}
    before_counts = []
    after_counts = []
    abnormal = []
    for img_id, items in grouped.items():
        before, after = merge_one_image(items)
        before_counts.append(before)
        after_counts.append(after)
        merged_sample = merged_map.get(img_id)
        if merged_sample is None:
            abnormal.append(img_id)
        else:
            merged_scores = to_numpy(merged_sample['pred_instances']['scores'])
            if len(merged_scores) == 0:
                abnormal.append(img_id)
    return {
        'avg_before': statistics.mean(before_counts) if before_counts else 0.0,
        'avg_after': statistics.mean(after_counts) if after_counts else 0.0,
        'abnormal_images': abnormal[:20],
        'merged_count': len(merged_map),
    }


def write_exp3_md(args: argparse.Namespace, report: dict) -> None:
    rows = report['rows']
    single = report['baseline_single']['metrics']['ap50']
    default = report['baseline_default_tta']['metrics']['ap50']
    ok_rows = [row for row in rows if row.get('status') == 'OK']
    best = max(ok_rows, key=lambda row: row['ap50']) if ok_rows else None
    lines = [
        '# Experiment 3: RTMDet-L TTA Merge Ablation',
        '',
        f'- generated_at: `{now()}`',
        f'- status: `{report["status"]}`',
        '- target: `RTMDet-L 3xMS on DOTA1 angle_000`',
        '- metric: DOTA AP50 / mAP@0.5',
        '',
        '## Baseline',
        '',
        f'- single angle_000 AP50: `{fmt(single)}`',
        f'- default TTA merge AP50: `{fmt(default)}`',
        f'- single eval log: `{report["baseline_single"]["log"]}`',
        f'- default TTA eval log: `{report["baseline_default_tta"]["log"]}`',
        '',
        '## Ablation Grid',
        '',
        '| nms_iou | score_thr | max_per_img | status | AP50 | delta vs single | delta vs default TTA | avg boxes before NMS | avg boxes after NMS | abnormal images | sec | eval log |',
        '|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        lines.append(
            f'| {fmt(row["nms_iou"], 1)} | {fmt(row["score_thr"], 3)} | {row["max_per_img"]} | '
            f'{row["status"]} | {fmt(row.get("ap50"))} | {fmt(row.get("delta_vs_single"))} | '
            f'{fmt(row.get("delta_vs_default_tta"))} | {fmt(row.get("avg_boxes_before_nms"))} | '
            f'{fmt(row.get("avg_boxes_after_nms"))} | {len(row.get("abnormal_images", []))} | '
            f'{fmt(row.get("duration_sec"))} | `{row.get("eval_log", "")}` |')
    lines.extend(['', '## Best Setting', ''])
    if best:
        lines.append(
            f'- best: `nms_iou={best["nms_iou"]}, score_thr={best["score_thr"]}, max_per_img={best["max_per_img"]}` AP50=`{fmt(best["ap50"])}`')
        if best['ap50'] >= single:
            lines.append('- Interpretation: best TTA reaches or exceeds single angle_000, so the default canonical drop is mainly consistent with a merge-artifact explanation.')
        else:
            lines.append('- Interpretation: best TTA still remains below single angle_000, so cross-view prediction inconsistency remains after merge tuning.')
    failed = [row for row in rows if row.get('status') != 'OK']
    lines.extend(['', '## Failed Combinations', ''])
    if failed:
        for row in failed:
            lines.append(f'- nms={row["nms_iou"]} score={row["score_thr"]} max={row["max_per_img"]}: {row.get("error", "see logs")}')
    else:
        lines.append('- None.')
    lines.extend([
        '',
        '## Reproduction Command',
        '',
        f'- `rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mplconfig CUDA_VISIBLE_DEVICES={args.gpu_ids} '
        f'PYTHONPATH={args.repo_root}:{args.repo_root / "tools"} {args.python_bin} '
        f'M_Tools/analysis/run_rotation_repair_suite.py --repo-root {args.repo_root} '
        f'--result-md-dir {args.result_md_dir} --weights-dir {args.weights_dir} '
        f'--work-dir {args.work_dir} --gpu-ids {args.gpu_ids} --exp 3 --mode full --resume`',
    ])
    write_md(args.result_md_dir / 'exp3_rtmdet_l_tta_merge_ablation.md', lines)


class DotaEvalContext:
    def __init__(self, args: argparse.Namespace, spec: ModelSpec, angle: str):
        ensure_mm(args.repo_root)
        import torch
        from mmengine import Config
        from mmengine.registry import init_default_scope
        from mmengine.runner import Runner
        from mmdet.registry import DATASETS

        self.args = args
        self.spec = spec
        self.angle = angle
        self.torch = torch
        self.cfg = Config.fromfile(str(spec.config))
        init_default_scope(self.cfg.get('default_scope', 'mmdet'))
        self.cfg.merge_from_dict({
            'test_dataloader.dataset.data_root':
            str(args.repo_root / DOTA_DATA_REL),
            'test_dataloader.dataset.ann_file':
            f'{DOTA_SWEEP_REL}/angle_{angle}/annfiles/',
            'test_dataloader.dataset.data_prefix.img_path':
            f'{DOTA_SWEEP_REL}/angle_{angle}/images/',
            'test_dataloader.num_workers':
            args.num_workers,
            'test_dataloader.persistent_workers':
            False,
        })
        self.dataset = DATASETS.build(self.cfg.test_dataloader.dataset)
        self.dataset_meta = self.dataset.metainfo
        self.classes = tuple(self.dataset_meta['classes'])
        evaluator_cfg = self.cfg.val_evaluator
        if isinstance(evaluator_cfg, (list, tuple)):
            evaluator_cfg = evaluator_cfg[0]
        self.predict_box_type = evaluator_cfg.get('predict_box_type', 'rbox')
        self.use_07_metric = evaluator_cfg.get('eval_mode', '11points') == '11points'
        self.gt_by_id = {}
        dataloader = Runner.build_dataloader(self.cfg.test_dataloader)
        for batch in dataloader:
            for data_sample in batch['data_samples']:
                data = data_sample.to_dict()
                self.gt_by_id[normalize_img_id(data['img_id'])] = data

    def empty_gt(self):
        return {
            'labels': self.torch.zeros((0, ), dtype=self.torch.long),
            'bboxes': self.torch.zeros((0, 5), dtype=self.torch.float32),
        }

    def merge_with_gt(self, predictions: list[dict]) -> list[dict]:
        pred_map = {normalize_img_id(pred['img_id']): pred for pred in predictions}
        samples = []
        for pid in sorted(pred_map):
            pred = pred_map[pid]
            if pid in self.gt_by_id:
                sample = dict(self.gt_by_id[pid])
                if not sample.get('gt_instances'):
                    sample['gt_instances'] = self.empty_gt()
                if not sample.get('ignored_instances'):
                    sample['ignored_instances'] = self.empty_gt()
            else:
                sample = {
                    'img_id': pred['img_id'],
                    'gt_instances': self.empty_gt(),
                    'ignored_instances': self.empty_gt(),
                }
            sample['pred_instances'] = pred['pred_instances']
            samples.append(sample)
        return samples

    def classwise_ap(self, pkl_path: Path, out_json: Path, nproc: int = 4) -> dict:
        if out_json.exists() and self.args.resume and not self.args.force:
            return json.loads(out_json.read_text(encoding='utf-8'))
        ensure_mm(self.args.repo_root)
        import numpy as np
        from mmrotate.evaluation import eval_rbbox_map

        samples = self.merge_with_gt(load_pickle(pkl_path))
        annotations = []
        det_results = []
        for sample in samples:
            gt = sample['gt_instances']
            ignored = sample['ignored_instances']
            if not gt:
                ann = {
                    'labels': np.zeros((0, ), dtype=np.int64),
                    'bboxes': np.zeros((0, 5), dtype=np.float32),
                    'labels_ignore': np.zeros((0, ), dtype=np.int64),
                    'bboxes_ignore': np.zeros((0, 5), dtype=np.float32),
                }
            else:
                ann = {
                    'labels': to_numpy(gt['labels']).astype(np.int64),
                    'bboxes': to_numpy(gt['bboxes']).astype(np.float32),
                    'labels_ignore': to_numpy(ignored['labels']).astype(np.int64) if ignored else np.zeros((0, ), dtype=np.int64),
                    'bboxes_ignore': to_numpy(ignored['bboxes']).astype(np.float32) if ignored else np.zeros((0, 5), dtype=np.float32),
                }
            annotations.append(ann)
            pred = sample['pred_instances']
            boxes = to_numpy(pred['bboxes']).astype(np.float32)
            labels = to_numpy(pred['labels']).astype(np.int64)
            scores = to_numpy(pred['scores']).astype(np.float32)
            per_class = []
            for label in range(len(self.classes)):
                keep = labels == label
                per_class.append(np.hstack([boxes[keep], scores[keep, None]]))
            det_results.append(per_class)
        mean_ap, cls_results = eval_rbbox_map(
            det_results,
            annotations,
            iou_thr=0.5,
            use_07_metric=self.use_07_metric,
            box_type=self.predict_box_type,
            dataset=self.classes,
            logger='silent',
            nproc=nproc)
        rows = []
        for class_name, cls_result in zip(self.classes, cls_results):
            ap = cls_result['ap']
            if isinstance(ap, np.ndarray):
                ap = float(np.asarray(ap).reshape(-1)[0])
            rows.append({
                'class': class_name,
                'ap50': float(ap),
                'num_gts': int(np.asarray(cls_result['num_gts']).reshape(-1)[0]),
                'num_dets': int(cls_result['num_dets']),
            })
        payload = {'map': float(mean_ap), 'classes': rows, 'pkl': str(pkl_path)}
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
        return payload

    def classwise_ap_from_log(self, log_path: Path, pkl_path: Path,
                              out_json: Path) -> dict | None:
        if out_json.exists() and self.args.resume and not self.args.force:
            return json.loads(out_json.read_text(encoding='utf-8'))
        rows = parse_classwise_from_eval_log(log_path, self.classes)
        if not rows:
            return None
        metrics = parse_metrics_from_log(log_path)
        payload = {
            'map': metrics.get('map'),
            'ap50': metrics.get('ap50'),
            'classes': rows,
            'pkl': str(pkl_path),
            'source_log': str(log_path),
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
        return payload

    def geometry_recall(self, pkl_path: Path, out_json: Path, score_thr: float = 0.05) -> dict:
        if out_json.exists() and self.args.resume and not self.args.force:
            return json.loads(out_json.read_text(encoding='utf-8'))
        ensure_mm(self.args.repo_root)
        import numpy as np
        import torch
        from mmcv.ops import box_iou_rotated

        samples = self.merge_with_gt(load_pickle(pkl_path))
        stats = {
            name: {
                'gt': 0,
                'matched': 0,
            }
            for name, _, _ in AR_BINS
        }
        for sample in samples:
            gt = sample['gt_instances']
            if not gt:
                continue
            gt_boxes = to_numpy(gt['bboxes']).astype(np.float32)
            gt_labels = to_numpy(gt['labels']).astype(np.int64)
            pred = sample['pred_instances']
            pred_boxes = to_numpy(pred['bboxes']).astype(np.float32)
            pred_labels = to_numpy(pred['labels']).astype(np.int64)
            pred_scores = to_numpy(pred['scores']).astype(np.float32)
            keep = pred_scores >= score_thr
            pred_boxes, pred_labels = pred_boxes[keep], pred_labels[keep]
            for idx, gt_box in enumerate(gt_boxes):
                ar = max(float(gt_box[2]), float(gt_box[3])) / max(min(float(gt_box[2]), float(gt_box[3])), 1e-6)
                bin_name = next(name for name, lo, hi in AR_BINS if ar >= lo and ar < hi)
                stats[bin_name]['gt'] += 1
                same = pred_labels == gt_labels[idx]
                if not np.any(same):
                    continue
                ious = box_iou_rotated(
                    torch.from_numpy(pred_boxes[same].astype(np.float32)),
                    torch.from_numpy(gt_box[None].astype(np.float32))).cpu().numpy()
                if ious.size and float(ious.max()) >= 0.5:
                    stats[bin_name]['matched'] += 1
        rows = []
        for name in stats:
            gt = stats[name]['gt']
            matched = stats[name]['matched']
            rows.append({
                'bin': name,
                'gt': gt,
                'matched': matched,
                'recall50': matched / gt if gt else 0.0,
            })
        payload = {'rows': rows, 'pkl': str(pkl_path), 'score_thr': score_thr}
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
        return payload


def run_experiment_4(args: argparse.Namespace) -> dict:
    specs_all = build_dota_specs(args.repo_root, args.weights_dir)
    model_keys = [key for key in select_model_keys(args) if key in specs_all]
    exp1_path = args.work_dir / 'exp1/exp1_results.json'
    if not exp1_path.exists():
        exp1_report = run_experiment_1(args)
    else:
        exp1_report = json.loads(exp1_path.read_text(encoding='utf-8'))
    rows_by_key = {
        (row['model'], row['angle']): row
        for row in exp1_report.get('rows', [])
        if row.get('status') == 'OK'
    }
    context_cache: dict[tuple[str, str], DotaEvalContext] = {}
    class_rows = []
    geom_rows = []
    failures = []
    for key in model_keys:
        spec = specs_all[key]
        for angle in ANGLES:
            row = rows_by_key.get((key, angle))
            if not row:
                failures.append({'model': key, 'angle': angle, 'error': 'missing exp1 row'})
                continue
            cache_key = (key, angle)
            base_dir = args.work_dir / 'exp4' / key / f'angle_{angle}'
            try:
                log(f'EXP4 {key} angle_{angle}: classwise')
                single = classwise_payload_from_eval_log(
                    Path(row.get('single_log', '')),
                    Path(row['single_pkl']),
                    base_dir / 'single_classwise.json',
                    DOTA_CLASSES,
                    args.resume,
                    args.force)
                if single is None:
                    if cache_key not in context_cache:
                        context_cache[cache_key] = DotaEvalContext(args, spec, angle)
                    ctx = context_cache[cache_key]
                    single = ctx.classwise_ap(
                        Path(row['single_pkl']), base_dir / 'single_classwise.json')
                tta = classwise_payload_from_eval_log(
                    Path(row.get('tta_log', '')),
                    Path(row['tta_pkl']),
                    base_dir / 'tta_classwise.json',
                    DOTA_CLASSES,
                    args.resume,
                    args.force)
                if tta is None:
                    if cache_key not in context_cache:
                        context_cache[cache_key] = DotaEvalContext(args, spec, angle)
                    ctx = context_cache[cache_key]
                    tta = ctx.classwise_ap(
                        Path(row['tta_pkl']), base_dir / 'tta_classwise.json')
                single_by_class = {item['class']: item for item in single['classes']}
                tta_by_class = {item['class']: item for item in tta['classes']}
                for class_name in DOTA_CLASSES:
                    s = single_by_class[class_name]
                    t = tta_by_class[class_name]
                    class_rows.append({
                        'model': key,
                        'display': spec.display,
                        'angle': angle,
                        'class': class_name,
                        'single_ap50': s['ap50'],
                        'tta_ap50': t['ap50'],
                        'delta_ap50': t['ap50'] - s['ap50'],
                        'num_gts': s['num_gts'],
                    })
                log(f'EXP4 {key} angle_{angle}: geometry')
                single_geom = geometry_recall_from_annfiles(
                    args, Path(row['single_pkl']), angle,
                    base_dir / 'single_geometry.json')
                tta_geom = geometry_recall_from_annfiles(
                    args, Path(row['tta_pkl']), angle,
                    base_dir / 'tta_geometry.json')
                single_geom_by_bin = {item['bin']: item for item in single_geom['rows']}
                tta_geom_by_bin = {item['bin']: item for item in tta_geom['rows']}
                for bin_name, _, _ in AR_BINS:
                    s = single_geom_by_bin[bin_name]
                    t = tta_geom_by_bin[bin_name]
                    geom_rows.append({
                        'model': key,
                        'display': spec.display,
                        'angle': angle,
                        'bin': bin_name,
                        'gt': s['gt'],
                        'single_recall50': s['recall50'],
                        'tta_recall50': t['recall50'],
                        'delta_recall50': t['recall50'] - s['recall50'],
                    })
            except Exception as exc:  # noqa: BLE001
                failures.append({'model': key, 'angle': angle, 'error': str(exc), 'traceback': traceback.format_exc()})
    report = {
        'status': 'DONE' if not failures else 'PARTIAL',
        'class_rows': class_rows,
        'geometry_rows': geom_rows,
        'failures': failures,
    }
    out_json = args.work_dir / 'exp4/exp4_breakdown_results.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    write_exp4_md(args, report)
    return report


def group_by(rows: Iterable[dict], key: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return grouped


def aggregate_class_rows(rows: list[dict]) -> list[dict]:
    out = []
    for model, model_rows in group_by(rows, 'model').items():
        for class_name, class_rows in group_by(model_rows, 'class').items():
            single_vals = [r['single_ap50'] for r in class_rows]
            tta_vals = [r['tta_ap50'] for r in class_rows]
            delta_vals = [r['delta_ap50'] for r in class_rows]
            out.append({
                'model': model,
                'display': class_rows[0]['display'],
                'class': class_name,
                'single_mean_ap50': statistics.mean(single_vals),
                'tta_mean_ap50': statistics.mean(tta_vals),
                'delta_mean_ap50': statistics.mean(delta_vals),
                'single_std': statistics.pstdev(single_vals),
                'tta_std': statistics.pstdev(tta_vals),
                'single_worst': min(single_vals),
                'single_best': max(single_vals),
                'tta_worst': min(tta_vals),
                'tta_best': max(tta_vals),
                'num_gts': max(r['num_gts'] for r in class_rows),
            })
    for model, model_rows in group_by(out, 'model').items():
        sorted_rows = sorted(model_rows, key=lambda r: r['delta_mean_ap50'], reverse=True)
        for rank, row in enumerate(sorted_rows, 1):
            row['delta_rank'] = rank
    return out


def aggregate_geom_rows(rows: list[dict]) -> list[dict]:
    out = []
    for model, model_rows in group_by(rows, 'model').items():
        for bin_name, bin_rows in group_by(model_rows, 'bin').items():
            single_vals = [r['single_recall50'] for r in bin_rows]
            tta_vals = [r['tta_recall50'] for r in bin_rows]
            out.append({
                'model': model,
                'display': bin_rows[0]['display'],
                'bin': bin_name,
                'gt_count': int(statistics.mean([r['gt'] for r in bin_rows])),
                'single_recall50': statistics.mean(single_vals),
                'tta_recall50': statistics.mean(tta_vals),
                'delta_recall50': statistics.mean([r['delta_recall50'] for r in bin_rows]),
                'single_std': statistics.pstdev(single_vals),
                'tta_std': statistics.pstdev(tta_vals),
            })
    return out


def write_exp4_md(args: argparse.Namespace, report: dict) -> None:
    class_agg = aggregate_class_rows(report['class_rows'])
    geom_agg = aggregate_geom_rows(report['geometry_rows'])
    lines = [
        '# Experiment 4: Class-Wise / Geometry-Wise Breakdown',
        '',
        f'- generated_at: `{now()}`',
        f'- status: `{report["status"]}`',
        '- class-wise metric: DOTA AP50 / mAP@0.5.',
        '- geometry-wise uses matched recall@0.5 instead of AP50.',
        '- geometry score threshold: `0.05`.',
        '',
    ]
    for model, model_rows in group_by(class_agg, 'model').items():
        model_rows = sorted(model_rows, key=lambda r: r['delta_mean_ap50'], reverse=True)
        lines.append(f'## {model_rows[0]["display"]} Class-Wise')
        lines.extend(['', '| rank | class | single AP50 | TTA AP50 | delta | single std | TTA std | single worst | single best | TTA worst | TTA best | GT count |'])
        lines.append('|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for row in model_rows:
            lines.append(
                f'| {row["delta_rank"]} | {row["class"]} | {fmt(row["single_mean_ap50"])} | '
                f'{fmt(row["tta_mean_ap50"])} | {fmt(row["delta_mean_ap50"])} | '
                f'{fmt(row["single_std"])} | {fmt(row["tta_std"])} | '
                f'{fmt(row["single_worst"])} | {fmt(row["single_best"])} | '
                f'{fmt(row["tta_worst"])} | {fmt(row["tta_best"])} | {row["num_gts"]} |')
        lines.append('')
        gains = model_rows[:5]
        damages = sorted(model_rows, key=lambda r: r['delta_mean_ap50'])[:5]
        lines.append('Top TTA gains: ' + ', '.join(f'{r["class"]} ({fmt(r["delta_mean_ap50"])})' for r in gains))
        lines.append('')
        lines.append('Top TTA damages: ' + ', '.join(f'{r["class"]} ({fmt(r["delta_mean_ap50"])})' for r in damages))
        lines.append('')
    lines.append('## Geometry-Wise Matched Recall@0.5')
    lines.append('')
    for model, model_rows in group_by(geom_agg, 'model').items():
        lines.append(f'### {model_rows[0]["display"]}')
        lines.extend(['', '| aspect ratio bin | GT count | single recall@0.5 | TTA recall@0.5 | delta | single std | TTA std |'])
        lines.append('|---|---:|---:|---:|---:|---:|---:|')
        order = {name: idx for idx, (name, _, _) in enumerate(AR_BINS)}
        for row in sorted(model_rows, key=lambda r: order[r['bin']]):
            lines.append(
                f'| {row["bin"]} | {row["gt_count"]} | {fmt(row["single_recall50"])} | '
                f'{fmt(row["tta_recall50"])} | {fmt(row["delta_recall50"])} | '
                f'{fmt(row["single_std"])} | {fmt(row["tta_std"])} |')
        lines.append('')
    lines.extend(['## Interpretation', ''])
    for model, model_rows in group_by(class_agg, 'model').items():
        unstable = sorted(model_rows, key=lambda r: r['single_std'], reverse=True)[:5]
        repaired = sorted(model_rows, key=lambda r: r['delta_mean_ap50'], reverse=True)[:5]
        damaged = sorted(model_rows, key=lambda r: r['delta_mean_ap50'])[:5]
        focus = [r for r in model_rows if r['class'] in FOCUS_CLASSES]
        focus_note = ', '.join(f'{r["class"]}:{fmt(r["delta_mean_ap50"])}' for r in sorted(focus, key=lambda r: r['class']))
        lines.append(f'- {model_rows[0]["display"]}: most unstable classes by single-view std are ' +
                     ', '.join(f'{r["class"]} ({fmt(r["single_std"])})' for r in unstable) + '.')
        lines.append(f'- {model_rows[0]["display"]}: strongest repairs are ' +
                     ', '.join(f'{r["class"]} ({fmt(r["delta_mean_ap50"])})' for r in repaired) +
                     '; strongest damages are ' +
                     ', '.join(f'{r["class"]} ({fmt(r["delta_mean_ap50"])})' for r in damaged) + '.')
        lines.append(f'- Focus directional classes: {focus_note}.')
    lines.extend([
        '',
        '## Failures',
        '',
    ])
    if report['failures']:
        for item in report['failures']:
            lines.append(f'- {item.get("model")} angle_{item.get("angle")}: {item.get("error")}')
    else:
        lines.append('- None.')
    lines.extend([
        '',
        '## Reproduction Command',
        '',
        f'- `rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mplconfig CUDA_VISIBLE_DEVICES={args.gpu_ids} '
        f'PYTHONPATH={args.repo_root}:{args.repo_root / "tools"} {args.python_bin} '
        f'M_Tools/analysis/run_rotation_repair_suite.py --repo-root {args.repo_root} '
        f'--result-md-dir {args.result_md_dir} --weights-dir {args.weights_dir} '
        f'--work-dir {args.work_dir} --gpu-ids {args.gpu_ids} --exp 4 --mode full --resume`',
    ])
    write_md(args.result_md_dir / 'exp4_classwise_geometrywise_breakdown.md', lines)


def git_hash(repo_root: Path) -> str:
    head = repo_root / '.git/HEAD'
    if not head.exists():
        return 'not a git repo'
    text = head.read_text(encoding='utf-8', errors='replace').strip()
    if text.startswith('ref:'):
        ref = repo_root / '.git' / text.split(':', 1)[1].strip()
        if ref.exists():
            return ref.read_text(encoding='utf-8', errors='replace').strip()[:12]
    return text[:12]


def environment_info(args: argparse.Namespace) -> dict:
    ensure_mm(args.repo_root)
    import torch
    import mmcv
    import mmdet
    import mmengine
    import mmrotate

    gpu_info = ''
    try:
        out = subprocess.run(
            ['rtk', 'nvidia-smi', '--query-gpu=index,name,memory.total,driver_version',
             '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False)
        gpu_info = out.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        gpu_info = str(exc)
    return {
        'python': sys.version.replace('\n', ' '),
        'python_executable': str(args.python_bin),
        'torch': torch.__version__,
        'torch_cuda': torch.version.cuda,
        'mmcv': mmcv.__version__,
        'mmdet': mmdet.__version__,
        'mmengine': mmengine.__version__,
        'mmrotate': getattr(mmrotate, '__version__', 'NA'),
        'git_hash': git_hash(args.repo_root),
        'gpu_info': gpu_info,
    }


def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def write_summary(args: argparse.Namespace) -> dict:
    exp1 = read_json_if_exists(args.work_dir / 'exp1/exp1_results.json')
    exp2 = read_json_if_exists(args.work_dir / 'exp2/exp2_discovery.json')
    exp3 = read_json_if_exists(args.work_dir / 'exp3/exp3_ablation_results.json')
    exp4 = read_json_if_exists(args.work_dir / 'exp4/exp4_breakdown_results.json')
    preflight = {
        'dryrun': read_json_if_exists(args.work_dir / 'preflight/dryrun.json'),
        'smoke': read_json_if_exists(args.work_dir / 'preflight/smoke/smoke.json'),
        'multigpu_batchsize': read_json_if_exists(args.work_dir / 'preflight/multigpu_batchsize/multigpu_batchsize.json'),
    }
    env = environment_info(args)
    paths = {
        'exp1': args.result_md_dir / 'exp1_dota1_full_angle_tta_repair_curve.md',
        'exp2': args.result_md_dir / 'exp2_far1m_angle_sweep_cross_dataset.md',
        'exp3': args.result_md_dir / 'exp3_rtmdet_l_tta_merge_ablation.md',
        'exp4': args.result_md_dir / 'exp4_classwise_geometrywise_breakdown.md',
    }
    statuses = {
        'Experiment 1': exp1.get('status', 'NOT_RUN'),
        'Experiment 2': exp2.get('status', 'NOT_RUN'),
        'Experiment 3': exp3.get('status', 'NOT_RUN'),
        'Experiment 4': exp4.get('status', 'NOT_RUN'),
    }
    exp1_summaries = []
    if exp1.get('rows'):
        for model_rows in group_by(exp1['rows'], 'model').values():
            summary = summarize_model_curve(model_rows)
            if summary:
                exp1_summaries.append(summary)
    proposition = 'PARTIAL'
    if exp1_summaries and exp3.get('rows'):
        canonical_damage = any(
            row.get('angle') == '000' and row.get('delta_ap50', 0) < 0
            for row in exp1.get('rows', [])
            if row.get('status') == 'OK')
        best_exp3 = max((row for row in exp3['rows'] if row.get('status') == 'OK'),
                        key=lambda row: row.get('ap50', -1),
                        default=None)
        single = exp3.get('baseline_single', {}).get('metrics', {}).get('ap50')
        if canonical_damage and best_exp3 and single is not None and best_exp3['ap50'] < single:
            proposition = 'SUPPORTED_BY_DOTA1'
        elif canonical_damage:
            proposition = 'MIXED'
    lines = [
        '# Summary: Four Rotation Repair Experiments 2026-05-07',
        '',
        f'- generated_at: `{now()}`',
        f'- repo_root: `{args.repo_root}`',
        f'- work_dir: `{args.work_dir}`',
        f'- git_commit: `{env["git_hash"]}`',
        '',
        '## Experiment Overview',
        '',
        '| experiment | status | md path |',
        '|---|---|---|',
        f'| Experiment 1 | {statuses["Experiment 1"]} | `{paths["exp1"]}` |',
        f'| Experiment 2 | {statuses["Experiment 2"]} | `{paths["exp2"]}` |',
        f'| Experiment 3 | {statuses["Experiment 3"]} | `{paths["exp3"]}` |',
        f'| Experiment 4 | {statuses["Experiment 4"]} | `{paths["exp4"]}` |',
        '',
        '## Preflight Validation',
        '',
        '| check | status | md/log |',
        '|---|---|---|',
        f'| dryrun | {preflight["dryrun"].get("status", "NOT_RUN")} | `{args.result_md_dir / "preflight_dryrun.md"}` |',
        f'| smoke | {preflight["smoke"].get("status", "NOT_RUN")} | `{args.result_md_dir / "preflight_smoke.md"}` |',
        f'| multigpu + batch size | {preflight["multigpu_batchsize"].get("status", "NOT_RUN")} | `{args.result_md_dir / "preflight_multigpu_batchsize.md"}` |',
        '',
        '## Model-Level DOTA1 Findings',
        '',
        '| model | single mean AP50 | TTA mean AP50 | mean delta | single std | TTA std | single RSI | TTA RSI |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for summary in exp1_summaries:
        lines.append(
            f'| {summary["display"]} | {fmt(summary["single_mean"])} | {fmt(summary["tta_mean"])} | '
            f'{fmt(summary["mean_delta"])} | {fmt(summary["single_std"])} | {fmt(summary["tta_std"])} | '
            f'{fmt(summary["single_rsi"])} | {fmt(summary["tta_rsi"])} |')
    lines.extend([
        '',
        '## Proposition',
        '',
        f'- `Rotation TTA cannot universally solve rotation-induced recognition collapse; the deeper issue is cross-view recognition inconsistency.`',
        f'- support_status: `{proposition}`',
        '',
        '## Five Key Findings',
        '',
    ])
    findings = []
    for summary in exp1_summaries:
        findings.append(
            f'{summary["display"]}: mean TTA delta {fmt(summary["mean_delta"])}, single std {fmt(summary["single_std"])} -> TTA std {fmt(summary["tta_std"])}.')
    if exp3.get('rows'):
        best = max((row for row in exp3['rows'] if row.get('status') == 'OK'),
                   key=lambda row: row.get('ap50', -1),
                   default=None)
        if best:
            findings.append(
                f'RTMDet-L best merge ablation AP50 {fmt(best["ap50"])} at nms={best["nms_iou"]}, score={best["score_thr"]}, max={best["max_per_img"]}.')
    findings.append('FAIR1M angle-sweep data exist, but requested-model FAIR1M checkpoints are missing.')
    findings.append('Geometry-wise analysis is reported as matched recall@0.5 because AP per aspect-ratio bin would require a separate stratified evaluator.')
    for item in findings[:5]:
        lines.append(f'- {item}')
    lines.extend(['', '## Five Key Failures Or Risks', ''])
    risks = [
        'Experiment 2 is blocked for the requested model set by missing FAIR1M checkpoints/configs.',
        'Existing DOTA1 prediction pkls are reused by default; use `--force` to regenerate inference.',
        'Merged TTA AP depends on rotated NMS and score thresholds.',
        'Geometry-wise recall is not AP50 and should not be compared numerically to classwise AP50.',
        'All commands assume physical GPU ids are exposed as CUDA_VISIBLE_DEVICES and map internally to 0..N-1.',
    ]
    for item in risks:
        lines.append(f'- {item}')
    lines.extend([
        '',
        '## Next Experiments',
        '',
        '- Train or add FAIR1M RTMDet-L / H2RBox-v2 / RetinaNet checkpoints and rerun Experiment 2.',
        '- Add per-bin AP evaluator for geometry-wise AP50, replacing recall@0.5.',
        '- Run Exp3 ablations on RTMDet-L worst and best DOTA1 angles if canonical tuning is inconclusive.',
        '- Inspect cross-view classification logits for classes with large TTA damage.',
        '',
        '## Important Logs',
        '',
        f'- command log: `{args.work_dir / "commands.jsonl"}`',
        f'- work_dir: `{args.work_dir}`',
        '',
        '## Environment',
        '',
        f'- Python: `{env["python"]}`',
        f'- Python executable: `{env["python_executable"]}`',
        f'- PyTorch: `{env["torch"]}` CUDA `{env["torch_cuda"]}`',
        f'- MMEngine: `{env["mmengine"]}`',
        f'- MMCV: `{env["mmcv"]}`',
        f'- MMDetection: `{env["mmdet"]}`',
        f'- MMRotate: `{env["mmrotate"]}`',
        '',
        '## GPU Info',
        '',
        '```text',
        env['gpu_info'],
        '```',
    ])
    out_path = args.result_md_dir / 'summary_four_experiments_20260507.md'
    write_md(out_path, lines)
    report = {
        'statuses': statuses,
        'preflight': preflight,
        'proposition': proposition,
        'env': env,
        'summary_md': str(out_path),
    }
    (args.work_dir / 'summary_report.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--result-md-dir', type=Path, required=True)
    parser.add_argument('--weights-dir', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--gpu-ids', default='4,5,6,7')
    parser.add_argument('--exp', choices=['all', '1', '2', '3', '4'], default='all')
    parser.add_argument('--mode', choices=['dryrun', 'smoke', 'full', 'debug'], default='full')
    parser.add_argument('--batch-size', type=int, default=0)
    parser.add_argument('--batch-size-candidates', default='1,2,4,8')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--max-images-for-smoke', type=int, default=16)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--only-model', default='')
    parser.add_argument('--only-angle', default='')
    parser.add_argument('--python-bin', type=Path, default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.result_md_dir = args.result_md_dir.resolve()
    args.weights_dir = args.weights_dir.resolve()
    args.work_dir = args.work_dir.resolve()
    setup_repo_imports(args.repo_root)
    args.result_md_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == 'dryrun':
        report = run_dryrun(args)
        write_summary(args)
        if report['status'] == 'FAILED':
            raise SystemExit(2)
        return

    if args.mode in ('smoke', 'debug'):
        if args.batch_size_candidates:
            # Multi-GPU batch-size preflight is selected when more than one
            # GPU is requested and angle_000 is the only requested angle.
            only_angles = parse_angle_list(args.only_angle, [])
            if len(parse_csv_arg(args.gpu_ids)) > 1 and (not only_angles or only_angles == ['000']):
                run_multigpu_batchsize(args)
                write_summary(args)
                return
        run_smoke(args)
        write_summary(args)
        return

    if args.exp in ('all', '1'):
        run_experiment_1(args)
    if args.exp in ('all', '2'):
        run_experiment_2(args)
    if args.exp in ('all', '3'):
        run_experiment_3(args)
    if args.exp in ('all', '4'):
        run_experiment_4(args)
    write_summary(args)


if __name__ == '__main__':
    main()
