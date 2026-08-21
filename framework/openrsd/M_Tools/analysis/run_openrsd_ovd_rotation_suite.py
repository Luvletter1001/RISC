#!/usr/bin/env python3
"""Unified runner for OpenRSD open-vocabulary rotation experiments.

The runner is intentionally conservative: it discovers local OpenRSD assets,
generates reproducible configs under the requested work directory, logs every
child command with separate stdout/stderr files, and writes markdown even when
an experiment is blocked.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
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
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import open_clip
import numpy as np
import torch


ANGLES = (
    '000', '030', '060', '090', '120', '150', '180', '210', '240', '270',
    '300', '330'
)
DOTA_SWEEP_REL = Path('data/DOTA1_1024_500/angle_sweep_val/realistic')
DOTA1_CLASSES = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter'
]
DOTA1_RAW_PROMPTS = [
    'plane', 'baseball diamond', 'bridge', 'ground track field',
    'small vehicle', 'large vehicle', 'ship', 'tennis court',
    'basketball court', 'storage tank', 'soccer ball field', 'roundabout',
    'harbor', 'swimming pool', 'helicopter'
]
DOTA1_AERIAL_PROMPTS = [f'aerial image of a {name}' for name in DOTA1_RAW_PROMPTS]
CORE9 = [
    'plane', 'bridge', 'ship', 'small-vehicle', 'large-vehicle',
    'storage-tank', 'harbor', 'roundabout', 'helicopter'
]
PROMPT_FAMILIES_OVD2 = {
    'F0_raw_class': [
        'plane', 'bridge', 'ship', 'small vehicle', 'large vehicle',
        'storage tank', 'harbor', 'roundabout', 'helicopter'
    ],
    'F1_aerial_context': [
        'aerial photo of a plane', 'aerial photo of a bridge',
        'aerial photo of a ship', 'aerial photo of a small vehicle',
        'aerial photo of a large vehicle', 'aerial photo of a storage tank',
        'aerial photo of a harbor', 'aerial photo of a roundabout',
        'aerial photo of a helicopter'
    ],
    'F2_remote_sensing_context': [
        'remote sensing object plane', 'remote sensing object bridge',
        'remote sensing object ship', 'remote sensing object small vehicle',
        'remote sensing object large vehicle',
        'remote sensing object storage tank', 'remote sensing object harbor',
        'remote sensing object roundabout',
        'remote sensing object helicopter'
    ],
    'F3_orientation_aware': [
        'rotated plane in an aerial image',
        'rotated bridge in an aerial image',
        'rotated ship in an aerial image',
        'rotated small vehicle in an aerial image',
        'rotated large vehicle in an aerial image',
        'rotated storage tank in an aerial image',
        'rotated harbor in an aerial image',
        'rotated roundabout in an aerial image',
        'rotated helicopter in an aerial image'
    ],
    'F4_shape_aware': [
        'elongated plane seen from above',
        'long narrow bridge seen from above',
        'elongated ship seen from above',
        'small vehicle seen from above',
        'large vehicle seen from above',
        'circular storage tank seen from above',
        'harbor area seen from above',
        'circular roundabout seen from above',
        'helicopter seen from above'
    ],
}
SCRIPT_NAMES = [
    'tools/train.py',
    'tools/test.py',
    'train.py',
    'test.py',
    'train_rotate.py',
    'test_rotate.py',
    'M_Tools/Eval_Tools/eval_diff_epochs.py',
]
DEFAULT_ENV_KEYS = [
    'PYTHONNOUSERSITE=1',
    'MPLCONFIGDIR=/tmp/mplconfig',
]
PROMPT_ENCODER_MODEL = 'ViT-L-14'
PROMPT_ENCODER_PRETRAINED = 'openai'
_PROMPT_ENCODER_CACHE: dict[str, Any] = {}
ERROR_PATTERNS = (
    ('CUDA out of memory', 'CUDA_OOM'),
    ('out of memory', 'OOM'),
    ('RuntimeError: CUDA error', 'CUDA_RUNTIME'),
    ('FileNotFoundError', 'FILE_MISSING'),
    ('No such file or directory', 'FILE_MISSING'),
    ('checkpoint', 'CHECKPOINT'),
    ('Config File', 'CONFIG'),
    ('prompt', 'PROMPT'),
    ('category', 'CATEGORY_MAPPING'),
    ('class number', 'CLASS_NUMBER_MISMATCH'),
    ('shape mismatch', 'EMBEDDING_SHAPE_MISMATCH'),
    ('val_using_aux', 'VAL_USING_AUX'),
    ('gt_instances', 'GT_INSTANCES'),
    ('loss is nan', 'LOSS_NAN'),
    ('evaluation metric', 'EVAL_METRIC'),
    ('Traceback (most recent call last)', 'TRACEBACK'),
)


@dataclass
class CommandResult:
    name: str
    command: str
    start_time: str
    end_time: str
    returncode: int
    stdout_path: str
    stderr_path: str
    nvidia_smi_before: str
    nvidia_smi_after: str
    duration_sec: float
    failure_kind: str = ''
    stdout_tail: str = ''
    stderr_tail: str = ''
    peak_mem_mb: dict[str, int] = field(default_factory=dict)
    process_seen: dict[str, bool] = field(default_factory=dict)


@dataclass
class ModelChoice:
    key: str
    config: Path
    checkpoint: Path
    score: int
    reason: str


def now() -> str:
    return datetime.now().strftime('%F %T')


def log(message: str) -> None:
    print(f'[{now()}] {message}', flush=True)


def fmt(value: Any, digits: int = 4) -> str:
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


def parse_angles(value: str | None, default: Sequence[str] = ANGLES) -> list[str]:
    raw = parse_csv_arg(value)
    if not raw:
        return list(default)
    return [f'{int(item):03d}' for item in raw]


def parse_ints(value: str | None, default: Sequence[int]) -> list[int]:
    raw = parse_csv_arg(value)
    if not raw:
        return list(default)
    return [int(item) for item in raw]


def shell_join(argv: Sequence[Any]) -> str:
    import shlex

    return ' '.join(shlex.quote(str(item)) for item in argv)


def read_text(path: Path, max_chars: int = 200000) -> str:
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except FileNotFoundError:
        return ''
    if len(text) > max_chars:
        return text[-max_chars:]
    return text


def tail(path: Path, count: int = 100) -> str:
    return '\n'.join(read_text(path).splitlines()[-count:])


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + '\n', encoding='utf-8')


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def write_md(path: Path, lines: Sequence[str]) -> None:
    write_text(path, '\n'.join(lines))


def path_state(path: Path) -> str:
    return 'OK' if path.exists() else 'MISSING'


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def detect_failure(stdout: str, stderr: str) -> str:
    text = f'{stdout}\n{stderr}'
    lowered = text.lower()
    for needle, kind in ERROR_PATTERNS:
        if needle.lower() in lowered:
            return kind
    return ''


def run_quiet(argv: Sequence[str], cwd: Path | None = None, timeout: int = 30) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            list(argv),
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False)
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as exc:  # noqa: BLE001
        return 1, '', str(exc)


def nvidia_smi_text() -> str:
    rc, out, err = run_quiet(['rtk', 'nvidia-smi'], timeout=20)
    return out if rc == 0 else err


class GpuMonitor:
    def __init__(self, gpu_ids: str, interval: float = 1.0):
        self.gpu_ids = parse_csv_arg(gpu_ids)
        self.interval = interval
        self.peak_mem = {gpu: 0 for gpu in self.gpu_ids}
        self.process_seen = {gpu: False for gpu in self.gpu_ids}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _snapshot(self) -> None:
        rc, out, _ = run_quiet([
            'rtk', 'nvidia-smi', '--query-gpu=index,memory.used',
            '--format=csv,noheader,nounits'
        ], timeout=10)
        if rc == 0:
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 2:
                    continue
                if parts[0] in self.peak_mem:
                    try:
                        self.peak_mem[parts[0]] = max(self.peak_mem[parts[0]], int(float(parts[1])))
                    except ValueError:
                        pass
        rc, out, _ = run_quiet([
            'rtk', 'nvidia-smi',
            '--query-compute-apps=gpu_bus_id,pid,process_name,used_memory',
            '--format=csv,noheader,nounits'
        ], timeout=10)
        if rc == 0 and out.strip():
            # nvidia-smi does not expose physical index in this query on all
            # drivers. If any Python process exists during the monitored
            # interval, mark all requested GPUs as having process activity.
            for gpu in self.gpu_ids:
                self.process_seen[gpu] = True

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
        self.command_log = args.work_dir / 'commands.jsonl'
        self.command_log.parent.mkdir(parents=True, exist_ok=True)

    def env_assignments(self, gpu_ids: str | None = None) -> list[str]:
        gpu_ids = gpu_ids or self.args.gpu_ids
        return [
            *DEFAULT_ENV_KEYS,
            f'CUDA_VISIBLE_DEVICES={gpu_ids}',
            f'PYTHONPATH={self.args.repo_root}:{self.args.repo_root / "tools"}',
        ]

    def command_string(self, argv: Sequence[Any], gpu_ids: str | None = None) -> str:
        return 'rtk env ' + shell_join(self.env_assignments(gpu_ids)) + ' ' + shell_join(argv)

    def run(self,
            name: str,
            argv: Sequence[Any],
            log_dir: Path,
            gpu_ids: str | None = None,
            monitor_gpu: bool = False,
            skip_if: Path | None = None) -> CommandResult:
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / 'stdout.log'
        stderr_path = log_dir / 'stderr.log'
        before_path = log_dir / 'nvidia_smi_before.txt'
        after_path = log_dir / 'nvidia_smi_after.txt'
        command = self.command_string(argv, gpu_ids)
        start_time = now()
        start = time.time()
        before = nvidia_smi_text()
        write_text(before_path, before or 'nvidia-smi unavailable')

        if skip_if is not None and skip_if.exists() and not self.args.force:
            write_text(stdout_path, f'[{now()}] SKIPPED existing artifact: {skip_if}\n{command}')
            write_text(stderr_path, '')
            after = nvidia_smi_text()
            write_text(after_path, after or 'nvidia-smi unavailable')
            result = CommandResult(
                name=name,
                command=command,
                start_time=start_time,
                end_time=now(),
                returncode=0,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                nvidia_smi_before=str(before_path),
                nvidia_smi_after=str(after_path),
                duration_sec=0.0)
            self._append(result)
            return result

        log(f'RUN {name}: {command}')
        env_cmd = ['rtk', 'env', *self.env_assignments(gpu_ids), *map(str, argv)]
        monitor = GpuMonitor(gpu_ids or self.args.gpu_ids) if monitor_gpu else contextlib.nullcontext()
        with stdout_path.open('w', encoding='utf-8') as stdout_f, stderr_path.open('w', encoding='utf-8') as stderr_f:
            stdout_f.write(f'[{start_time}] command={command}\n')
            stdout_f.write(f'cwd={self.repo_root}\n\n')
            stdout_f.flush()
            with monitor as mon:
                proc = subprocess.run(
                    env_cmd,
                    cwd=str(self.repo_root),
                    stdout=stdout_f,
                    stderr=stderr_f,
                    text=True,
                    check=False)
        duration = time.time() - start
        after = nvidia_smi_text()
        write_text(after_path, after or 'nvidia-smi unavailable')
        stdout_tail = tail(stdout_path, 100)
        stderr_tail = tail(stderr_path, 100)
        result = CommandResult(
            name=name,
            command=command,
            start_time=start_time,
            end_time=now(),
            returncode=proc.returncode,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            nvidia_smi_before=str(before_path),
            nvidia_smi_after=str(after_path),
            duration_sec=duration,
            failure_kind=detect_failure(stdout_tail, stderr_tail),
            stdout_tail=stdout_tail if proc.returncode else '',
            stderr_tail=stderr_tail if proc.returncode else '',
            peak_mem_mb={} if not monitor_gpu else (mon.peak_mem or {}),  # type: ignore[name-defined]
            process_seen={} if not monitor_gpu else (mon.process_seen or {}),  # type: ignore[name-defined]
        )
        self._append(result)
        log(f'DONE {name}: rc={proc.returncode} stdout={stdout_path} stderr={stderr_path}')
        return result

    def _append(self, result: CommandResult) -> None:
        payload = result.__dict__.copy()
        with self.command_log.open('a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False) + '\n')


def safe_import_versions(repo_root: Path, python_bin: Path) -> dict[str, str]:
    versions: dict[str, str] = {
        'python': sys.version.replace('\n', ' '),
        'python_executable': sys.executable,
        'runner_python_bin': str(python_bin),
    }
    code = r"""
import json
import sys
from pathlib import Path
repo_root = Path(sys.argv[1])
for path in (repo_root, repo_root / 'tools'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
try:
    from openrsd_env import preload_installed_mmengine
    preload_installed_mmengine()
except Exception as exc:
    print(json.dumps({'openrsd_env_preload_error': repr(exc)}))
    raise SystemExit(0)
out = {'python': sys.version.replace('\n', ' '), 'python_executable': sys.executable}
for module_name in ('torch', 'mmcv', 'mmengine', 'mmdet', 'mmrotate'):
    try:
        module = __import__(module_name)
        out[module_name] = str(getattr(module, '__version__', 'UNKNOWN'))
        if module_name == 'torch':
            out['torch_cuda'] = str(getattr(module.version, 'cuda', 'UNKNOWN'))
            out['torch_cuda_available'] = str(module.cuda.is_available())
    except Exception as exc:
        out[module_name] = f'IMPORT_ERROR: {exc!r}'
print(json.dumps(out, ensure_ascii=False))
"""
    rc, out, err = run_quiet([str(python_bin), '-c', code, str(repo_root)], timeout=120)
    if rc == 0 and out.strip():
        try:
            versions.update(json.loads(out.strip().splitlines()[-1]))
        except Exception:  # noqa: BLE001
            versions['version_probe_output'] = out.strip()
            if err.strip():
                versions['version_probe_error'] = err.strip()
    else:
        versions['version_probe_error'] = err.strip() or out.strip()
    return versions


def iter_files(roots: Sequence[Path], suffixes: tuple[str, ...]) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob('*'):
            if path in seen or not path.is_file():
                continue
            if path.suffix.lower() in suffixes:
                seen.add(path)
                yield path


def list_limited(paths: Sequence[Path], root: Path, limit: int = 80) -> list[str]:
    return [safe_rel(p, root) for p in sorted(paths)[:limit]]


def discover_configs(repo_root: Path) -> dict[str, Any]:
    roots = [
        repo_root / 'M_configs',
        repo_root / 'mmdet_configs',
        repo_root / 'mmrotate_configs',
        repo_root / 'EXP_CONFIG',
        repo_root / 'SimpleRun',
        repo_root / 'work_dirs',
        repo_root / 'workdir_vis',
    ]
    filename_tokens = [
        'openrsd', 'prompt', 'text', 'image', 'alignment', 'align', 'fusion',
        'dota', 'fair', 'fair1m', 'dior', 'dior_r'
    ]
    token_hits: dict[str, list[str]] = {token: [] for token in filename_tokens}
    openrsd_configs: list[Path] = []
    content_hits: dict[str, list[str]] = {
        'OpenRTMDet': [],
        'support_feat_dict': [],
        'val_support_classes': [],
        'val_using_aux': [],
        'prompt_predict': [],
    }
    for path in iter_files(roots, ('.py',)):
        rel_lower = safe_rel(path, repo_root).lower()
        for token in filename_tokens:
            if token in rel_lower:
                token_hits[token].append(safe_rel(path, repo_root))
        text = ''
        if path.stat().st_size < 2_000_000:
            text = read_text(path, 2_000_000)
        is_openrsd = False
        for needle in content_hits:
            if needle in text:
                content_hits[needle].append(safe_rel(path, repo_root))
                is_openrsd = True
        if is_openrsd and ('M_configs' in path.parts or path.name.endswith('.py')):
            openrsd_configs.append(path)
    return {
        'filename_token_hits': {k: v[:120] for k, v in token_hits.items()},
        'content_hits': {k: v[:120] for k, v in content_hits.items()},
        'openrsd_configs': list_limited(openrsd_configs, repo_root, 120),
    }


def discover_checkpoints(weights_dir: Path, repo_root: Path) -> dict[str, Any]:
    roots = [weights_dir]
    # Keep this secondary discovery read-only. User supplied results/ as the
    # target, but local runs sometimes keep old baseline weights in weights/.
    if (repo_root / 'weights').exists() and (repo_root / 'weights') not in roots:
        roots.append(repo_root / 'weights')
    all_ckpts: list[Path] = []
    primary_ckpts: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob('*.pth'):
            if path.name.startswith('._'):
                continue
            all_ckpts.append(path)
            if root == weights_dir:
                primary_ckpts.append(path)
    priority_tokens = [
        'openrsd', 'prompt', 'text', 'image', 'align', 'fusion', 'dota',
        'fair', 'a13', 'a12', 'a10', 'hin', 'flex', 'best', 'epoch',
        'latest'
    ]
    priority = [
        p for p in primary_ckpts
        if any(token in str(p).lower() for token in priority_tokens)
    ]
    return {
        'primary_root': str(weights_dir),
        'primary_all': list_limited(primary_ckpts, repo_root, 200),
        'primary_priority': list_limited(priority, repo_root, 200),
        'secondary_all': list_limited([p for p in all_ckpts if p not in primary_ckpts], repo_root, 80),
    }


def discover_prompt_files(repo_root: Path) -> list[str]:
    suffixes = ('.txt', '.json', '.yaml', '.yml', '.pkl')
    tokens = ('class', 'category', 'prompt', 'text', 'label', 'vocab', 'support', 'normalized')
    roots = [repo_root / 'data', repo_root / 'M_Tools', repo_root / 'SimpleRun', repo_root / 'tools']
    hits = []
    for path in iter_files(roots, suffixes):
        rel = safe_rel(path, repo_root)
        if any(token in rel.lower() for token in tokens):
            hits.append(path)
    return list_limited(hits, repo_root, 220)


def discover_data_paths(repo_root: Path) -> dict[str, Any]:
    dota_angles = []
    for angle in ANGLES:
        p = repo_root / DOTA_SWEEP_REL / f'angle_{angle}'
        dota_angles.append({
            'angle': angle,
            'path': str(p),
            'annfiles': (p / 'annfiles').exists(),
            'images': (p / 'images').exists(),
            'ann_count': len(list((p / 'annfiles').glob('*.txt'))) if (p / 'annfiles').exists() else 0,
            'image_count': len(list((p / 'images').glob('*'))) if (p / 'images').exists() else 0,
        })
    fair_candidates = [
        repo_root / 'data/FAR1M',
        repo_root / 'data/FAIR1M',
        repo_root / 'data/far1m',
        repo_root / 'data/fair1m',
        repo_root / 'data/FAR1M_1024_500',
        repo_root / 'data/FAIR1M_1024_500',
        repo_root / 'data/FAIR1M_2_800_400',
    ]
    dior_candidates = [
        repo_root / 'data/DIOR',
        repo_root / 'data/DIOR_R',
        repo_root / 'data/DIOR_R_dota',
        repo_root / 'data/dior',
        repo_root / 'data/dior_r',
    ]
    return {
        'dota1_sweep_root': str(repo_root / DOTA_SWEEP_REL),
        'dota1_angles': dota_angles,
        'fair_candidates': [{'path': str(p), 'exists': p.exists()} for p in fair_candidates],
        'dior_candidates': [{'path': str(p), 'exists': p.exists()} for p in dior_candidates],
    }


def infer_capabilities(repo_root: Path, config_info: dict[str, Any]) -> dict[str, bool]:
    hay_paths = []
    for hits in config_info.get('content_hits', {}).values():
        hay_paths.extend(hits[:50])
    hay_paths.extend([
        'README.md',
        'README_en.md',
        'M_AD/models/detectors/Flex_Rtmdet_v3_1_formal.py',
        'M_AD/models/detectors/Hindsight_Rtmdet_v2_NearestHead.py',
    ])
    text_chunks = []
    for rel in sorted(set(hay_paths)):
        p = repo_root / rel
        if p.exists() and p.stat().st_size < 2_000_000:
            text_chunks.append(read_text(p, 2_000_000))
    text = '\n'.join(text_chunks).lower()
    return {
        'text_prompt': 'text_embeds' in text or 'text prompt' in text or 'support_type' in text,
        'image_prompt': 'visual_embeds' in text or 'image prompt' in text or 'prompt_extract_feats' in text,
        'alignment_head': 'val_using_aux=false' in text or 'align head' in text or 'with_obj_align' in text,
        'fusion_head': 'val_using_aux=true' in text or 'fusion head' in text or 'aux_bbox_head' in text,
        'oriented_bbox': 'rbox' in text or 'nms_rotated' in text or 'rotated' in text,
        'horizontal_bbox': 'hbb' in text or 'bbox' in text,
    }


def config_stem_key(path: Path) -> str:
    return path.stem.replace('.py', '')


def select_model_choice(args: argparse.Namespace, config_info: dict[str, Any], ckpt_info: dict[str, Any]) -> ModelChoice | None:
    config_paths = [args.repo_root / rel for rel in config_info.get('openrsd_configs', [])]
    config_paths = [p for p in config_paths if p.exists()]
    ckpt_paths = [args.repo_root / rel for rel in ckpt_info.get('primary_all', [])]
    ckpt_paths = [p for p in ckpt_paths if p.exists() and not p.name.startswith('._')]
    if args.only_model:
        only = args.only_model.lower()
        config_paths = [p for p in config_paths if only in str(p).lower()] or config_paths
        ckpt_paths = [p for p in ckpt_paths if only in str(p).lower()] or ckpt_paths
    if not config_paths or not ckpt_paths:
        return None

    best: ModelChoice | None = None
    for cfg in config_paths:
        stem = config_stem_key(cfg)
        for ckpt in ckpt_paths:
            hay = f'{ckpt.parent.name}/{ckpt.name}'.lower()
            score = 0
            reason = []
            if stem.lower() in hay:
                score += 100
                reason.append('exact config stem in checkpoint path')
            stripped = stem.replace('_vis', '')
            if stripped.lower() in hay:
                score += 80
                reason.append('stripped config stem in checkpoint path')
            for token, value in [
                ('a13', 28), ('hin', 25), ('nearestmem', 25),
                ('a12', 22), ('dota2only', 20), ('dotaonly', 18),
                ('a10', 15), ('formal', 14), ('flex', 12), ('rtm', 10),
                ('textcls', 5), ('epoch_24', 3), ('epoch_12', 2),
            ]:
                if token in hay or token in str(cfg).lower():
                    score += value
            if 'hbb' in str(cfg).lower():
                score -= 15
            if 'vis' in str(cfg).lower():
                score -= 20
            if best is None or score > best.score:
                best = ModelChoice(
                    key=ckpt.parent.name.replace('MMR_AD_', ''),
                    config=cfg,
                    checkpoint=ckpt,
                    score=score,
                    reason=', '.join(reason) or f'score={score}')
    return best


def inventory(args: argparse.Namespace) -> dict[str, Any]:
    args.result_md_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    write_test = args.result_md_dir / '.openrsd_ovd_write_test.tmp'
    writable = False
    try:
        write_test.write_text('ok', encoding='utf-8')
        write_test.unlink()
        writable = True
    except OSError:
        writable = False

    git_rc, git_out, git_err = run_quiet(['rtk', 'git', 'rev-parse', 'HEAD'], cwd=args.repo_root)
    git_status_rc, git_status_out, _ = run_quiet(['rtk', 'git', 'status', '--short'], cwd=args.repo_root)
    config_info = discover_configs(args.repo_root)
    ckpt_info = discover_checkpoints(args.weights_dir, args.repo_root)
    prompt_files = discover_prompt_files(args.repo_root)
    data_info = discover_data_paths(args.repo_root)
    capabilities = infer_capabilities(args.repo_root, config_info)
    version_info = safe_import_versions(args.repo_root, args.python_bin)
    gpu_rc, gpu_query, gpu_err = run_quiet([
        'rtk', 'nvidia-smi', '--query-gpu=index,name,memory.total,driver_version',
        '--format=csv,noheader'
    ])
    model_choice = select_model_choice(args, config_info, ckpt_info)
    support_assets = {
        'pca_meta': str(args.repo_root / 'data/7_25_pca_meta_DINOv2_256.pkl'),
        'pca_meta_exists': (args.repo_root / 'data/7_25_pca_meta_DINOv2_256.pkl').exists(),
        'normalized_class_dict': str(args.repo_root / 'data/normalized_class_dict.pkl'),
        'normalized_class_dict_exists': (args.repo_root / 'data/normalized_class_dict.pkl').exists(),
        'neg_support_data': str(args.repo_root / 'data/Neg_supports_v2.pkl'),
        'neg_support_data_exists': (args.repo_root / 'data/Neg_supports_v2.pkl').exists(),
        'dota2_support': str(args.repo_root / 'data/DOTA2_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl'),
        'dota2_support_exists': (args.repo_root / 'data/DOTA2_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl').exists(),
        'dotav2_support': str(args.repo_root / 'data/DOTAV2/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl'),
        'dotav2_support_exists': (args.repo_root / 'data/DOTAV2/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl').exists(),
        'dota1_support': str(args.repo_root / 'data/DOTA_800_600/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl'),
        'dota1_support_exists': (args.repo_root / 'data/DOTA_800_600/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl').exists(),
    }
    all_angles_ok = all(item['annfiles'] and item['images'] for item in data_info['dota1_angles'])
    runnable_base = bool(model_choice and all_angles_ok and support_assets['dota2_support_exists'])
    prompt_reencode_available, reencode_reason = probe_prompt_reencode_available()
    exp_matrix = {
        'ovd0': {'status': 'RUNNABLE', 'reason': 'Static inventory only.'},
        'ovd1': {
            'status': 'RUNNABLE' if runnable_base and prompt_reencode_available else 'BLOCKED',
            'reason': 'Needs OpenRSD config/checkpoint, DOTA1 angle sweep, support embeddings, and prompt re-encoding.'
        },
        'ovd2': {
            'status': 'RUNNABLE' if runnable_base and prompt_reencode_available else 'BLOCKED',
            'reason': reencode_reason if runnable_base else 'Base OpenRSD inference is blocked.'
        },
        'ovd3': {
            'status': 'RUNNABLE' if runnable_base and capabilities.get('alignment_head') and capabilities.get('fusion_head') else 'PARTIAL',
            'reason': 'val_using_aux controls alignment/fusion head when supported.'
        },
        'ovd4': {
            'status': 'RUNNABLE' if runnable_base and prompt_reencode_available else 'BLOCKED',
            'reason': 'Prompt ensemble needs prompt re-encoding for a faithful prompt-family ensemble.'
        },
        'ovd5': {
            'status': 'PARTIAL' if runnable_base else 'BLOCKED',
            'reason': 'Diagnostic hooks require a successful smoke model load first.'
        },
    }
    runner = CommandRunner(args)
    planned_commands = planned_suite_commands(args, runner, model_choice)
    report = {
        'generated_at': now(),
        'repo_root_exists': args.repo_root.exists(),
        'result_md_dir_writable': writable,
        'weights_dir_exists': args.weights_dir.exists(),
        'work_dir_created': args.work_dir.exists(),
        'git_commit': git_out.strip() if git_rc == 0 else f'ERROR: {git_err.strip()}',
        'git_status_short': git_status_out.strip() if git_status_rc == 0 else '',
        'versions': version_info,
        'launcher_python_executable': sys.executable,
        'gpu_info': gpu_query.strip() if gpu_rc == 0 else gpu_err.strip(),
        'scripts': {name: (args.repo_root / name).exists() for name in SCRIPT_NAMES},
        'configs': config_info,
        'checkpoints': ckpt_info,
        'prompt_files': prompt_files,
        'data': data_info,
        'capabilities': capabilities,
        'support_assets': support_assets,
        'selected_model': None if model_choice is None else {
            'key': model_choice.key,
            'config': str(model_choice.config),
            'checkpoint': str(model_choice.checkpoint),
            'score': model_choice.score,
            'reason': model_choice.reason,
        },
        'prompt_reencode_available': prompt_reencode_available,
        'prompt_reencode_reason': reencode_reason,
        'experiment_matrix': exp_matrix,
        'planned_commands': planned_commands,
    }
    hard_blockers = []
    if not args.repo_root.exists():
        hard_blockers.append('repo_root missing')
    if not writable:
        hard_blockers.append('result-md-dir not writable')
    if not args.weights_dir.exists():
        hard_blockers.append('weights-dir missing')
    if not all_angles_ok:
        hard_blockers.append('DOTA1 12-angle sweep incomplete')
    if model_choice is None:
        hard_blockers.append('No usable OpenRSD config/checkpoint pair found under weights-dir.')
    if not support_assets['dota2_support_exists'] and not support_assets['dota1_support_exists']:
        hard_blockers.append('No local DOTA support embedding pkl found.')
    report['status'] = 'BLOCKED' if hard_blockers else 'DONE'
    report['hard_blockers'] = hard_blockers
    write_json(args.work_dir / 'exp_ovd0' / 'openrsd_inventory.json', report)
    write_ovd0_md(args, report)
    write_summary(args, {'ovd0': report})
    ensure_placeholder_mds(args, report)
    return report


def planned_suite_commands(args: argparse.Namespace, runner: CommandRunner, model_choice: ModelChoice | None) -> list[str]:
    base = [
        str(args.python_bin), 'M_Tools/analysis/run_openrsd_ovd_rotation_suite.py',
        '--repo-root', str(args.repo_root),
        '--result-md-dir', str(args.result_md_dir),
        '--weights-dir', str(args.weights_dir),
        '--work-dir', str(args.work_dir),
    ]
    commands = [
        runner.command_string([*base, '--gpu-ids', '4,5,6,7', '--exp', 'all', '--mode', 'dryrun'], '4,5,6,7'),
        runner.command_string([*base, '--gpu-ids', '4', '--exp', 'ovd1', '--mode', 'smoke', '--only-dataset', 'dota1', '--only-angle', '000', '--batch-size', '1', '--max-images-for-smoke', str(args.max_images_for_smoke)], '4'),
        runner.command_string([*base, '--gpu-ids', '4,5,6,7', '--exp', 'ovd1', '--mode', 'smoke', '--only-dataset', 'dota1', '--only-angle', '000', '--batch-size-candidates', args.batch_size_candidates, '--max-images-for-smoke', '32'], '4,5,6,7'),
    ]
    for exp in ('ovd1', 'ovd2', 'ovd3', 'ovd4', 'ovd5'):
        commands.append(runner.command_string([*base, '--gpu-ids', '4,5,6,7', '--exp', exp, '--mode', 'full', '--resume'], '4,5,6,7'))
    if model_choice is not None:
        commands.append(runner.command_string([
            str(args.python_bin), 'tools/openrsd_test.py',
            '<generated_eval_config.py>', str(model_choice.checkpoint),
            '--work-dir', '<angle_work_dir>', '--out', '<predictions.pkl>'
        ], args.gpu_ids))
    return commands


def write_ovd0_md(args: argparse.Namespace, report: dict[str, Any]) -> None:
    lines = [
        '# Experiment OVD0: OpenRSD Inventory / Environment / Capability Check',
        '',
        f'- generated_at: `{report["generated_at"]}`',
        f'- status: `{report["status"]}`',
        f'- repo_root: `{args.repo_root}`',
        f'- work_dir: `{args.work_dir}`',
        f'- inventory_json: `{args.work_dir / "exp_ovd0/openrsd_inventory.json"}`',
        '',
        '## Repository',
        '',
        f'- git_commit: `{report["git_commit"]}`',
        f'- git_status_short: `{report.get("git_status_short", "") or "clean-or-not-recorded"}`',
        '',
        '## Versions',
        '',
        '| component | version |',
        '|---|---|',
    ]
    for key, value in report['versions'].items():
        lines.append(f'| {key} | `{value}` |')
    lines.extend(['', '## GPU Info', '', '```text', report.get('gpu_info') or 'nvidia-smi unavailable', '```'])
    lines.extend(['', '## Required Scripts', '', '| script | status |', '|---|---|'])
    for script, exists in report['scripts'].items():
        lines.append(f'| `{script}` | {"OK" if exists else "MISSING"} |')
    lines.extend(['', '## OpenRSD Config Discovery', ''])
    lines.append(f'- content OpenRTMDet config hits: `{len(report["configs"]["content_hits"].get("OpenRTMDet", []))}`')
    lines.append(f'- selected config: `{report.get("selected_model", {}).get("config") if report.get("selected_model") else "NONE"}`')
    lines.extend(['', '| token | filename hits | first hits |', '|---|---:|---|'])
    for token, hits in report['configs']['filename_token_hits'].items():
        lines.append(f'| {token} | {len(hits)} | `{", ".join(hits[:5])}` |')
    lines.extend(['', '## Checkpoint Discovery', ''])
    lines.append(f'- weights_dir: `{report["checkpoints"]["primary_root"]}`')
    lines.append(f'- primary checkpoints: `{len(report["checkpoints"]["primary_all"])}`')
    lines.append(f'- selected checkpoint: `{report.get("selected_model", {}).get("checkpoint") if report.get("selected_model") else "NONE"}`')
    lines.extend(['', '| checkpoint |', '|---|'])
    for ckpt in report['checkpoints']['primary_priority'][:40]:
        lines.append(f'| `{ckpt}` |')
    lines.extend(['', '## Prompt / Class / Category / Vocab Files', '', '| path |', '|---|'])
    for path in report['prompt_files'][:80]:
        lines.append(f'| `{path}` |')
    lines.extend(['', '## Dataset Paths', '', '### DOTA1 Angle Sweep', '', '| angle | annfiles | images | ann count | image count |', '|---:|---|---|---:|---:|'])
    for row in report['data']['dota1_angles']:
        lines.append(f'| {row["angle"]} | {row["annfiles"]} | {row["images"]} | {row["ann_count"]} | {row["image_count"]} |')
    lines.extend(['', '### FAIR1M / FAR1M', '', '| path | exists |', '|---|---|'])
    for row in report['data']['fair_candidates']:
        lines.append(f'| `{row["path"]}` | {row["exists"]} |')
    lines.extend(['', '### DIOR / DIOR-R', '', '| path | exists |', '|---|---|'])
    for row in report['data']['dior_candidates']:
        lines.append(f'| `{row["path"]}` | {row["exists"]} |')
    lines.extend(['', '## OpenRSD Capability Judgment', '', '| capability | available |', '|---|---|'])
    for key, value in report['capabilities'].items():
        lines.append(f'| {key} | {value} |')
    lines.extend(['', '## Support Embedding Assets', '', '| asset | exists | path |', '|---|---|---|'])
    for key, value in report['support_assets'].items():
        if key.endswith('_exists'):
            continue
        lines.append(f'| {key} | {report["support_assets"].get(key + "_exists", "NA")} | `{value}` |')
    lines.extend(['', '## Runnable Experiment Matrix', '', '| exp | status | reason |', '|---|---|---|'])
    for exp, info in report['experiment_matrix'].items():
        lines.append(f'| {exp} | {info["status"]} | {info["reason"]} |')
    if report.get('hard_blockers'):
        lines.extend(['', '## Blockers', ''])
        for blocker in report['hard_blockers']:
            lines.append(f'- {blocker}')
    lines.extend(['', '## Dryrun / Planned Commands', ''])
    for command in report['planned_commands']:
        lines.append(f'- `{command}`')
    write_md(args.result_md_dir / 'exp_ovd0_preflight_openrsd_inventory.md', lines)


def ensure_placeholder_mds(args: argparse.Namespace, inv: dict[str, Any]) -> None:
    specs = {
        'exp_ovd1_zero_shot_prompt_rotation_curve.md': ('OVD1', 'zero-shot prompt rotation curve'),
        'exp_ovd2_prompt_engineering_rotation_sensitivity.md': ('OVD2', 'prompt engineering rotation sensitivity'),
        'exp_ovd3_alignment_vs_fusion_head_rotation.md': ('OVD3', 'alignment vs fusion head rotation'),
        'exp_ovd4_prompt_ensemble_and_rotation_tta_repair.md': ('OVD4', 'prompt ensemble + rotation TTA repair'),
        'exp_ovd5_open_vocab_cross_view_diagnostic.md': ('OVD5', 'open-vocabulary cross-view diagnostic'),
    }
    for filename, (exp, title) in specs.items():
        path = args.result_md_dir / filename
        if path.exists():
            continue
        key = exp.lower()
        matrix = inv.get('experiment_matrix', {}).get(key, {})
        status = 'NOT_RUN'
        reason = matrix.get('reason', 'Not run yet.')
        lines = [
            f'# Experiment {exp}: {title}',
            '',
            f'- generated_at: `{now()}`',
            f'- status: `{status}`',
            f'- reason: `{reason}`',
            f'- work_dir: `{args.work_dir / key}`',
            '',
            'This file is a placeholder from preflight. It will be overwritten when the experiment is run.',
        ]
        write_md(path, lines)


def load_pickle(path: Path) -> Any:
    with path.open('rb') as f:
        return pickle.load(f)


def dump_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        pickle.dump(obj, f)


def canonicalize_class(name: str) -> str:
    return name.strip().lower().replace('_', '-').replace(' ', '-')


def get_prompt_encoder() -> tuple[Any, Any]:
    cache_key = f'{PROMPT_ENCODER_MODEL}:{PROMPT_ENCODER_PRETRAINED}'
    if cache_key not in _PROMPT_ENCODER_CACHE:
        model, _, _ = open_clip.create_model_and_transforms(
            PROMPT_ENCODER_MODEL, pretrained=PROMPT_ENCODER_PRETRAINED)
        tokenizer = open_clip.get_tokenizer(PROMPT_ENCODER_MODEL)
        model = model.eval().cpu()
        _PROMPT_ENCODER_CACHE[cache_key] = (model, tokenizer)
    return _PROMPT_ENCODER_CACHE[cache_key]


def encode_prompt_texts(prompt_texts: Sequence[str]) -> list[list[float]]:
    model, tokenizer = get_prompt_encoder()
    tokens = tokenizer(list(prompt_texts))
    with torch.no_grad():
        feats = model.encode_text(tokens)
    feats = feats.detach().cpu().float()
    feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return feats.numpy().tolist()


def probe_prompt_reencode_available() -> tuple[bool, str]:
    try:
        embeds = encode_prompt_texts(['aerial image of a plane'])
        if len(embeds) != 1 or len(embeds[0]) != 768:
            return False, f'Unexpected embedding shape: {len(embeds)} x {len(embeds[0]) if embeds else 0}'
        return True, f'{PROMPT_ENCODER_MODEL}/{PROMPT_ENCODER_PRETRAINED} text encoder available.'
    except Exception as exc:  # noqa: BLE001
        return False, f'Prompt re-encode probe failed: {exc!r}'


def prompt_support_sources(args: argparse.Namespace) -> list[Path]:
    return [
        args.repo_root / 'data/DOTAV2/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
        args.repo_root / 'data/DOTA2_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
        args.repo_root / 'data/DOTA_800_600/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
    ]


def select_prompt_support_source(args: argparse.Namespace,
                                 class_names: Sequence[str]) -> Path | None:
    requested = [canonicalize_class(name) for name in class_names]
    best: tuple[int, int, Path] | None = None
    for path in prompt_support_sources(args):
        if not path.exists():
            continue
        try:
            support = load_pickle(path)
        except Exception:  # noqa: BLE001
            continue
        available = {canonicalize_class(name) for name in support.keys()}
        coverage = sum(1 for name in requested if name in available)
        score = coverage * 100 + len(available)
        candidate = (score, coverage, path)
        if best is None or candidate > best:
            best = candidate
    return best[2] if best else None


def match_prompt_to_class(prompt_text: str, class_names: Sequence[str]) -> str | None:
    prompt_norm = canonicalize_class(prompt_text)
    lookup = {canonicalize_class(name): name for name in class_names}
    if prompt_norm in lookup:
        return lookup[prompt_norm]
    matches = [norm for norm in lookup if norm in prompt_norm]
    if matches:
        best_norm = max(matches, key=len)
        return lookup[best_norm]
    prompt_tokens = [token for token in prompt_norm.split('-') if token]
    if not prompt_tokens:
        return None
    best_name = None
    best_score = -1
    for norm, name in lookup.items():
        class_tokens = [token for token in norm.split('-') if token]
        overlap = len(set(prompt_tokens) & set(class_tokens))
        if overlap > best_score:
            best_score = overlap
            best_name = name
    return best_name if best_score > 0 else None


def build_prompt_support(args: argparse.Namespace,
                         prompt_key: str,
                         prompt_texts: Sequence[str],
                         class_names: Sequence[str]) -> tuple[Path | None, dict[str, Any]]:
    """Create a support pkl selecting local OpenRSD support embeddings.

    Visual embeddings are reused from the source support pkl and prompt texts
    are re-encoded locally with OpenCLIP into the support text embeddings.
    """
    src = select_prompt_support_source(args, class_names)
    out_dir = args.work_dir / 'prompt_support'
    out_pkl = out_dir / f'{prompt_key}.pkl'
    meta_path = out_dir / f'{prompt_key}.json'
    requested_classes = [canonicalize_class(name) for name in class_names]
    existing_meta: dict[str, Any] | None = None
    if meta_path.exists():
        try:
            existing_meta = json.loads(meta_path.read_text(encoding='utf-8'))
        except Exception:  # noqa: BLE001
            existing_meta = None
    if (
        out_pkl.exists() and
        existing_meta and
        existing_meta.get('selected_classes') == requested_classes and
        existing_meta.get('source_support') == (str(src) if src else None) and
        existing_meta.get('prompt_reencoded') is True and
        existing_meta.get('text_embedding_source') == f'open_clip::{PROMPT_ENCODER_MODEL}::{PROMPT_ENCODER_PRETRAINED}' and
        not args.force
    ):
        try:
            existing_support = load_pickle(out_pkl)
            existing_classes = [canonicalize_class(name) for name in existing_support.keys()]
            if (
                sorted(existing_classes) == sorted(requested_classes)
                and existing_meta.get('prompt_texts') == list(prompt_texts)
            ):
                return out_pkl, existing_meta
        except Exception:  # noqa: BLE001
            pass
    if src is None:
        meta = {
            'prompt_key': prompt_key,
            'prompt_texts': list(prompt_texts),
            'class_names': list(class_names),
            'selected_classes': requested_classes,
            'text_embedding_source': 'none',
            'image_embedding_source': 'none',
            'class_mapping': {text: match_prompt_to_class(text, class_names) for text in prompt_texts},
            'prompt_reencoded': False,
            'note': 'No source support pkl found.',
            'source_support': None,
            'error': 'No source support pkl found.',
        }
        write_json(meta_path, meta)
        return None, meta
    meta = {
        'prompt_key': prompt_key,
        'prompt_texts': list(prompt_texts),
        'class_names': list(class_names),
        'selected_classes': requested_classes,
        'text_embedding_source': f'open_clip::{PROMPT_ENCODER_MODEL}::{PROMPT_ENCODER_PRETRAINED}',
        'image_embedding_source': 'existing_openrsd_support_visual_embeds',
        'class_mapping': {text: match_prompt_to_class(text, class_names) for text in prompt_texts},
        'prompt_reencoded': True,
        'note': 'Prompt text embeddings were re-encoded locally with OpenCLIP; visual support embeddings were reused from the source support pkl.',
        'source_support': str(src) if src else None,
    }
    try:
        support = load_pickle(src)
        norm_support = {canonicalize_class(k): v for k, v in support.items()}
        selected = {}
        missing = []
        prompt_embeds = encode_prompt_texts(prompt_texts)
        if len(prompt_embeds) != len(prompt_texts):
            raise RuntimeError('Prompt embedding count mismatch.')
        if len(prompt_texts) != len(class_names):
            raise RuntimeError(
                f'Prompt text count {len(prompt_texts)} must match class name count {len(class_names)} for prompt re-encoding.')
        prompt_map = {
            canonicalize_class(class_name): {
                'prompt_text': prompt_text,
                'text_embed': prompt_embed,
            }
            for class_name, prompt_text, prompt_embed in zip(class_names, prompt_texts, prompt_embeds)
        }
        for class_name in requested_classes:
            key = canonicalize_class(class_name)
            if key not in norm_support:
                missing.append(key)
                continue
            selected[key] = deepcopy(norm_support[key])
            prompt_info = prompt_map.get(key)
            if prompt_info is not None:
                selected[key]['texts'] = [prompt_info['prompt_text']]
                selected[key]['text_embeds'] = np.asarray([prompt_info['text_embed']], dtype=np.float32)
        meta['missing_classes'] = missing
        meta['selected_classes'] = list(selected.keys())
        if not selected:
            meta['error'] = 'No requested classes found in source support.'
            write_json(meta_path, meta)
            return None, meta
        tmp_pkl = out_pkl.with_suffix(out_pkl.suffix + '.tmp')
        if tmp_pkl.exists():
            tmp_pkl.unlink()
        dump_pickle(tmp_pkl, selected)
        tmp_pkl.replace(out_pkl)
        write_json(meta_path, meta)
        return out_pkl, meta
    except Exception as exc:  # noqa: BLE001
        meta['error'] = repr(exc)
        meta['traceback'] = traceback.format_exc()
        write_json(meta_path, meta)
        return None, meta


def create_subset(args: argparse.Namespace, angle: str, max_images: int, ids: list[str] | None = None) -> tuple[Path, list[str]]:
    src = args.repo_root / DOTA_SWEEP_REL / f'angle_{angle}'
    ann_src = src / 'annfiles'
    img_src = src / 'images'
    if ids is None:
        ids = []
        for ann_file in sorted(ann_src.glob('*.txt')):
            if len(ids) >= max_images:
                break
            if ann_file.read_text(encoding='utf-8', errors='replace').strip():
                ids.append(ann_file.stem)
    subset = args.work_dir / 'subsets' / 'dota1' / f'angle_{angle}_n{max_images}'
    if subset.exists() and args.force:
        shutil.rmtree(subset)
    ann_dst = subset / 'annfiles'
    img_dst = subset / 'images'
    ann_dst.mkdir(parents=True, exist_ok=True)
    img_dst.mkdir(parents=True, exist_ok=True)
    for stem in ids[:max_images]:
        ann_link = ann_dst / f'{stem}.txt'
        if not ann_link.exists():
            os.symlink(ann_src / f'{stem}.txt', ann_link)
        img_match = None
        for ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
            p = img_src / f'{stem}{ext}'
            if p.exists():
                img_match = p
                break
        if img_match is None:
            raise FileNotFoundError(f'image missing for {stem} angle {angle}')
        img_link = img_dst / img_match.name
        if not img_link.exists():
            os.symlink(img_match, img_link)
    return subset, ids[:max_images]


def py_literal(value: Any) -> str:
    return repr(value)


def write_eval_config(args: argparse.Namespace,
                      base_config: Path,
                      out_config: Path,
                      data_root: Path,
                      ann_file: str,
                      img_path: str,
                      class_names: Sequence[str],
                      support_pkl: Path,
                      prompt_key: str,
                      head: str,
                      batch_size: int,
                      num_workers: int) -> None:
    val_using_aux = head == 'fusion'
    support_dict = {'Data1_DOTA2': str(support_pkl)}
    lines = [
        f"_base_ = {py_literal(str(base_config))}",
        '',
        f'ovd_prompt_key = {prompt_key!r}',
        f'ovd_head = {head!r}',
        f'val_support_classes = {py_literal(list(class_names))}',
        f'metainfo = dict(classes={py_literal(list(class_names))}, palette=[(220, 20, 60)])',
        '',
        'file_client_args = dict(backend="disk")',
        'val_pipeline = [',
        '    dict(type="mmdet.LoadImageFromFile", file_client_args=file_client_args),',
        '    dict(type="mmdet.Resize", scale=(1024, 1024), keep_ratio=True),',
        '    dict(type="mmdet.LoadAnnotations", with_bbox=True, box_type="qbox"),',
        '    dict(type="ConvertBoxType", box_type_mapping=dict(gt_bboxes="rbox")),',
        '    dict(type="mmdet.Pad", size=(1024, 1024), pad_val=dict(img=(114, 114, 114))),',
        '    dict(type="mmdet.PackDetInputs", meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor")),',
        ']',
        '',
        'model = dict(',
        f'    support_feat_dict=dict(_delete_=True, **{py_literal(support_dict)}),',
        '    val_support_classes=val_support_classes,',
        '    val_dataset_flag="Data1_DOTA2",',
        f'    val_using_aux={val_using_aux},',
        '    support_type="text",',
        f'    pca_meta_pth={py_literal(str(args.repo_root / "data/7_25_pca_meta_DINOv2_256.pkl"))},',
        f'    neg_support_data={py_literal(str(args.repo_root / "data/Neg_supports_v2.pkl"))},',
        f'    normalized_class_dict={py_literal(str(args.repo_root / "data/normalized_class_dict.pkl"))},',
        ')',
        '',
        'test_dataloader = dict(',
        '    _delete_=True,',
        f'    batch_size={batch_size},',
        f'    num_workers={num_workers},',
        '    persistent_workers=False,',
        '    drop_last=False,',
        '    sampler=dict(type="DefaultSampler", shuffle=False),',
        '    dataset=dict(',
        '        type="DOTADataset",',
        f'        data_root={py_literal(str(data_root))},',
        '        metainfo=metainfo,',
        f'        ann_file={ann_file!r},',
        f'        data_prefix=dict(img_path={img_path!r}),',
        '        img_shape=(1024, 1024),',
        '        filter_cfg=dict(filter_empty_gt=False),',
        '        pipeline=val_pipeline,',
        '    ),',
        ')',
        'val_dataloader = test_dataloader',
        'val_evaluator = dict(type="DETAILDOTAMetric", metric="mAP", iou_thrs=[0.5])',
        'test_evaluator = val_evaluator',
    ]
    write_text(out_config, '\n'.join(lines))


def parse_metrics_from_logs(stdout_path: Path, stderr_path: Path | None = None) -> dict[str, Any]:
    text = read_text(stdout_path)
    if stderr_path is not None:
        text += '\n' + read_text(stderr_path)
    out: dict[str, Any] = {'ap50': None, 'map': None, 'per_class_ap50': {}}
    patterns = [
        ('ap50', r"['\"]?AP50['\"]?\s*[:,=]\s*([0-9]*\.?[0-9]+)"),
        ('map', r"['\"]?mAP['\"]?\s*[:,=]\s*([0-9]*\.?[0-9]+)"),
    ]
    for key, pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            try:
                out[key] = float(matches[-1])
            except ValueError:
                pass
    for cls in DOTA1_CLASSES:
        pattern = rf"['\"]{re.escape(cls)}['\"].{{0,120}}['\"]ap['\"]\s*:\s*([0-9]*\.?[0-9]+)"
        matches = re.findall(pattern, text)
        if matches:
            out['per_class_ap50'][cls] = float(matches[-1])
    return out


def prediction_stats(path: Path) -> dict[str, Any]:
    stats = {
        'exists': path.exists(),
        'sample_count': 0,
        'mean_confidence': None,
        'avg_detections_per_image': None,
        'detections': 0,
    }
    if not path.exists():
        return stats
    try:
        rows = load_pickle(path)
        stats['sample_count'] = len(rows)
        scores = []
        det_count = 0
        for row in rows:
            pred = None
            if isinstance(row, dict):
                pred = row.get('pred_instances') or row.get('pred_instance')
            else:
                pred = getattr(row, 'pred_instances', None)
            if pred is None:
                continue
            row_scores = getattr(pred, 'scores', None)
            if row_scores is None and isinstance(pred, dict):
                row_scores = pred.get('scores')
            if row_scores is None:
                continue
            if hasattr(row_scores, 'detach'):
                arr = row_scores.detach().cpu().numpy().tolist()
            elif hasattr(row_scores, 'cpu'):
                arr = row_scores.cpu().numpy().tolist()
            else:
                arr = list(row_scores)
            scores.extend(float(x) for x in arr)
            det_count += len(arr)
        stats['detections'] = det_count
        stats['mean_confidence'] = statistics.mean(scores) if scores else None
        stats['avg_detections_per_image'] = det_count / max(len(rows), 1)
    except Exception as exc:  # noqa: BLE001
        stats['error'] = repr(exc)
    return stats


def run_single_inference(args: argparse.Namespace,
                         runner: CommandRunner,
                         model: ModelChoice,
                         angle: str,
                         prompt_key: str,
                         prompt_texts: Sequence[str],
                         class_names: Sequence[str],
                         head: str,
                         exp_dir: Path,
                         batch_size: int,
                         num_workers: int,
                         max_images: int | None,
                         gpu_ids: str,
                         distributed: bool = False) -> dict[str, Any]:
    support_pkl, support_meta = build_prompt_support(args, prompt_key, prompt_texts, class_names)
    out_dir = exp_dir / prompt_key / head / f'angle_{angle}'
    pred_path = out_dir / 'predictions.pkl'
    eval_config = out_dir / 'eval_config.py'
    if support_pkl is None:
        payload = {
            'status': 'FAILED',
            'angle': angle,
            'prompt_key': prompt_key,
            'head': head,
            'error': support_meta.get('error', 'support pkl unavailable'),
            'support_meta': support_meta,
        }
        write_json(out_dir / 'result.json', payload)
        return payload
    if max_images is not None:
        data_root, _ = create_subset(args, angle, max_images)
        ann_file = 'annfiles/'
        img_path = 'images/'
    else:
        data_root = args.repo_root / DOTA_SWEEP_REL / f'angle_{angle}'
        ann_file = 'annfiles/'
        img_path = 'images/'
    write_eval_config(
        args=args,
        base_config=model.config,
        out_config=eval_config,
        data_root=data_root,
        ann_file=ann_file,
        img_path=img_path,
        class_names=class_names,
        support_pkl=support_pkl,
        prompt_key=prompt_key,
        head=head,
        batch_size=batch_size,
        num_workers=num_workers)
    if distributed:
        nproc = len(parse_csv_arg(gpu_ids))
        argv: list[Any] = [
            str(args.python_bin), '-m', 'torch.distributed.launch',
            f'--nproc_per_node={nproc}', '--master_port', str(39400 + int(angle)),
            'tools/openrsd_test.py', str(eval_config), str(model.checkpoint),
            '--launcher', 'pytorch', '--work-dir', str(out_dir), '--out',
            str(pred_path)
        ]
    else:
        argv = [
            str(args.python_bin), 'tools/openrsd_test.py', str(eval_config),
            str(model.checkpoint), '--work-dir', str(out_dir), '--out',
            str(pred_path)
        ]
    result = runner.run(
        f'{prompt_key}_{head}_angle_{angle}',
        argv,
        out_dir / 'logs',
        gpu_ids=gpu_ids,
        monitor_gpu=True,
        skip_if=pred_path if args.resume else None)
    metrics = parse_metrics_from_logs(Path(result.stdout_path), Path(result.stderr_path))
    pred_stats = prediction_stats(pred_path)
    status = 'OK' if result.returncode == 0 and pred_path.exists() else 'FAILED'
    payload = {
        'status': status,
        'angle': angle,
        'prompt_key': prompt_key,
        'prompt_type': 'text',
        'head': head,
        'ap50': metrics.get('ap50'),
        'map': metrics.get('map'),
        'per_class_ap50': metrics.get('per_class_ap50', {}),
        'mean_confidence': pred_stats.get('mean_confidence'),
        'avg_detections_per_image': pred_stats.get('avg_detections_per_image'),
        'detections': pred_stats.get('detections'),
        'sample_count': pred_stats.get('sample_count'),
        'unmatched_prompt_classes': len(support_meta.get('missing_classes', [])),
        'fps_or_images_per_sec': (pred_stats.get('sample_count') or 0) / result.duration_sec if result.duration_sec > 0 else None,
        'config': str(eval_config),
        'checkpoint': str(model.checkpoint),
        'predictions': str(pred_path),
        'stdout': result.stdout_path,
        'stderr': result.stderr_path,
        'command': result.command,
        'returncode': result.returncode,
        'failure_kind': result.failure_kind,
        'text_embedding_source': support_meta.get('text_embedding_source'),
        'image_embedding_source': support_meta.get('image_embedding_source'),
        'class_mapping': support_meta.get('class_mapping'),
        'support_meta': support_meta,
        'peak_mem_mb': result.peak_mem_mb,
        'process_seen': result.process_seen,
        'duration_sec': result.duration_sec,
    }
    write_json(out_dir / 'result.json', payload)
    return payload


def latest_inventory(args: argparse.Namespace) -> dict[str, Any]:
    path = args.work_dir / 'exp_ovd0/openrsd_inventory.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return inventory(args)


def model_from_inventory(args: argparse.Namespace, inv: dict[str, Any]) -> ModelChoice | None:
    item = inv.get('selected_model')
    if not item:
        return None
    cfg = Path(item['config'])
    ckpt = Path(item['checkpoint'])
    if cfg.exists() and ckpt.exists():
        return ModelChoice(item['key'], cfg, ckpt, int(item.get('score', 0)), item.get('reason', 'inventory'))
    return None


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    inv = latest_inventory(args)
    model = model_from_inventory(args, inv)
    out_dir = args.work_dir / 'preflight_ovd_smoke'
    gpu_ids = '4'
    if parse_csv_arg(args.gpu_ids) == ['4']:
        gpu_ids = args.gpu_ids
    status = 'BLOCKED'
    failures = []
    rows = []
    if model is None:
        failures.append('No selected OpenRSD config/checkpoint pair.')
    if inv.get('experiment_matrix', {}).get('ovd1', {}).get('status') == 'BLOCKED':
        failures.extend(inv.get('hard_blockers', []))
    if failures:
        report = {
            'status': status,
            'failures': failures,
            'gpu_ids': gpu_ids,
            'checkpoint': None,
            'config': None,
        }
        write_json(out_dir / 'smoke.json', report)
        write_smoke_md(args, report)
        write_summary(args, {'ovd0': inv, 'smoke': report})
        return report
    runner = CommandRunner(args)
    try:
        requested_prompt_subset = ['plane', 'ship', 'bridge', 'small vehicle', 'large vehicle', 'storage tank']
        result = run_single_inference(
            args=args,
            runner=runner,
            model=model,
            angle='000',
            prompt_key='smoke_minimal_dota',
            prompt_texts=requested_prompt_subset,
            class_names=DOTA1_CLASSES,
            head='alignment',
            exp_dir=out_dir,
            batch_size=args.batch_size or 1,
            num_workers=args.num_workers,
            max_images=args.max_images_for_smoke,
            gpu_ids=gpu_ids,
            distributed=False)
        rows.append(result)
        status = 'DONE' if result.get('status') == 'OK' else 'FAILED'
        if status == 'FAILED':
            stdout_path = Path(result.get('stdout', ''))
            stderr_path = Path(result.get('stderr', ''))
            failures.append({
                'last_stdout_100': tail(stdout_path, 100) if stdout_path.exists() else '',
                'last_stderr_100': tail(stderr_path, 100) if stderr_path.exists() else '',
                'config_dump': result.get('config'),
                'checkpoint': result.get('checkpoint'),
                'prompt_file': result.get('support_meta', {}).get('source_support'),
                'dataset_path': str(args.repo_root / DOTA_SWEEP_REL / 'angle_000'),
                'traceback_summary': result.get('failure_kind'),
            })
    except Exception as exc:  # noqa: BLE001
        status = 'FAILED'
        failures.append({'error': repr(exc), 'traceback': traceback.format_exc()})
    report = {
        'status': status,
        'gpu_ids': gpu_ids,
        'config': str(model.config),
        'checkpoint': str(model.checkpoint),
        'requested_prompt_subset': requested_prompt_subset,
        'actual_class_list': DOTA1_CLASSES,
        'rows': rows,
        'failures': failures,
    }
    write_json(out_dir / 'smoke.json', report)
    write_smoke_md(args, report)
    write_summary(args, {'ovd0': inv, 'smoke': report})
    return report


def write_smoke_md(args: argparse.Namespace, report: dict[str, Any]) -> None:
    lines = [
        '# Preflight OVD Smoke',
        '',
        f'- generated_at: `{now()}`',
        f'- status: `{report["status"]}`',
        f'- physical_gpu_ids: `{report.get("gpu_ids", "4")}`',
        f'- config: `{report.get("config")}`',
        f'- checkpoint: `{report.get("checkpoint")}`',
        f'- requested_prompt_subset: `{report.get("requested_prompt_subset", [])}`',
        f'- actual_class_list: `{report.get("actual_class_list", [])}`',
        f'- report_json: `{args.work_dir / "preflight_ovd_smoke/smoke.json"}`',
        '',
        '## Smoke Requirements',
        '',
        '| item | result |',
        '|---|---|',
    ]
    row = (report.get('rows') or [{}])[0]
    lines.extend([
        f'| config load | `{row.get("status") == "OK"}` |',
        f'| checkpoint load | `{row.get("status") == "OK"}` |',
        f'| prompt embedding/support available | `{bool(row.get("support_meta"))}` |',
        f'| test loop | `{row.get("status") == "OK"}` |',
        f'| predictions saved | `{Path(row.get("predictions", "")).exists() if row.get("predictions") else False}` |',
        f'| evaluator/log metrics | `{fmt(row.get("ap50"))}` AP50 |',
    ])
    lines.extend(['', '## Result Row', '', '| angle | status | AP50 | mean conf | det/img | predictions | stdout | stderr |', '|---:|---|---:|---:|---:|---|---|---|'])
    for item in report.get('rows', []):
        lines.append(
            f'| {item.get("angle")} | {item.get("status")} | {fmt(item.get("ap50"))} | '
            f'{fmt(item.get("mean_confidence"))} | {fmt(item.get("avg_detections_per_image"))} | '
            f'`{item.get("predictions", "")}` | `{item.get("stdout", "")}` | `{item.get("stderr", "")}` |')
    if report.get('failures'):
        lines.extend(['', '## Failures / Debug', ''])
        for failure in report['failures']:
            if isinstance(failure, str):
                lines.append(f'- {failure}')
            else:
                lines.append(f'- `{json.dumps(failure, ensure_ascii=False)[:2000]}`')
    write_md(args.result_md_dir / 'preflight_ovd_smoke.md', lines)


def run_multigpu_batchsize(args: argparse.Namespace) -> dict[str, Any]:
    inv = latest_inventory(args)
    smoke_path = args.work_dir / 'preflight_ovd_smoke/smoke.json'
    smoke_ok = smoke_path.exists() and json.loads(smoke_path.read_text(encoding='utf-8')).get('status') == 'DONE'
    model = model_from_inventory(args, inv)
    out_dir = args.work_dir / 'preflight_ovd_multigpu_batchsize'
    if model is None or not smoke_ok:
        report = {
            'status': 'NOT_RUN',
            'reason': 'Smoke did not pass; multi-GPU batch-size test is intentionally skipped.',
            'rows': [],
            'selected_batch_size': 1,
        }
        write_json(out_dir / 'multigpu_batchsize.json', report)
        write_multigpu_md(args, report)
        write_summary(args, {'ovd0': inv, 'multigpu': report})
        return report
    runner = CommandRunner(args)
    rows = []
    stable = None
    for batch_size in parse_ints(args.batch_size_candidates, [1, 2, 4, 8, 16]):
        if stable is not None and rows and rows[-1].get('status') == 'OOM':
            rows.append({'batch_size': batch_size, 'status': 'SKIPPED_AFTER_OOM'})
            continue
        result = run_single_inference(
            args=args,
            runner=runner,
            model=model,
            angle='000',
            prompt_key=f'batchsize_text_bs{batch_size}',
            prompt_texts=DOTA1_RAW_PROMPTS,
            class_names=DOTA1_CLASSES,
            head='alignment',
            exp_dir=out_dir,
            batch_size=batch_size,
            num_workers=args.num_workers,
            max_images=max(args.max_images_for_smoke, 32),
            gpu_ids=args.gpu_ids,
            distributed=len(parse_csv_arg(args.gpu_ids)) > 1)
        status = result.get('status', 'FAILED')
        text = tail(Path(result.get('stdout', '')), 100) + tail(Path(result.get('stderr', '')), 100) if result.get('stdout') else ''
        if 'out of memory' in text.lower():
            status = 'OOM'
        if status == 'OK':
            stable = batch_size
        rows.append({
            'batch_size': batch_size,
            'status': status,
            'peak_mem_mb': result.get('peak_mem_mb', {}),
            'process_seen': result.get('process_seen', {}),
            'duration_sec': result.get('duration_sec'),
            'avg_time_per_image': result.get('duration_sec', 0) / max(result.get('sample_count') or 1, 1),
            'predictions': result.get('predictions'),
            'stdout': result.get('stdout'),
            'stderr': result.get('stderr'),
            'failure_kind': result.get('failure_kind'),
        })
        if status == 'OOM':
            break
    report = {
        'status': 'DONE' if any(row.get('status') == 'OK' for row in rows) else 'FAILED',
        'selected_batch_size': stable or 1,
        'rows': rows,
        'inference_mode': 'distributed inference' if len(parse_csv_arg(args.gpu_ids)) > 1 else 'single-gpu fallback',
    }
    write_json(out_dir / 'multigpu_batchsize.json', report)
    write_multigpu_md(args, report)
    write_json(args.work_dir / 'selected_batch_size.json', {'alignment': report['selected_batch_size']})
    write_summary(args, {'ovd0': inv, 'multigpu': report})
    return report


def write_multigpu_md(args: argparse.Namespace, report: dict[str, Any]) -> None:
    lines = [
        '# Preflight OVD Multi-GPU + Batch Size',
        '',
        f'- generated_at: `{now()}`',
        f'- status: `{report["status"]}`',
        f'- physical_gpu_ids: `{args.gpu_ids}`',
        f'- inference_mode: `{report.get("inference_mode", "not_run")}`',
        f'- selected_batch_size: `{report.get("selected_batch_size")}`',
        f'- report_json: `{args.work_dir / "preflight_ovd_multigpu_batchsize/multigpu_batchsize.json"}`',
    ]
    if report.get('reason'):
        lines.append(f'- reason: `{report["reason"]}`')
    lines.extend(['', '| batch size | status | avg sec/img | peak mem MB | process seen | predictions | stdout | stderr |', '|---:|---|---:|---|---|---|---|---|'])
    for row in report.get('rows', []):
        lines.append(
            f'| {row.get("batch_size")} | {row.get("status")} | {fmt(row.get("avg_time_per_image"))} | '
            f'`{row.get("peak_mem_mb", {})}` | `{row.get("process_seen", {})}` | '
            f'`{row.get("predictions", "")}` | `{row.get("stdout", "")}` | `{row.get("stderr", "")}` |')
    write_md(args.result_md_dir / 'preflight_ovd_multigpu_batchsize.md', lines)


def selected_batch(args: argparse.Namespace) -> int:
    if args.batch_size:
        return args.batch_size
    p = args.work_dir / 'selected_batch_size.json'
    if p.exists():
        try:
            return int(json.loads(p.read_text(encoding='utf-8')).get('alignment', 1))
        except Exception:  # noqa: BLE001
            return 1
    return 1


def angle_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row.get('status') == 'OK' and row.get('ap50') is not None]
    vals = [float(row['ap50']) for row in ok]
    if not vals:
        return {}
    worst = min(ok, key=lambda r: float(r['ap50']))
    best = max(ok, key=lambda r: float(r['ap50']))
    worst_angle = worst.get('angle', worst.get('target_angle', 'NA'))
    best_angle = best.get('angle', best.get('target_angle', 'NA'))
    mean = statistics.mean(vals)
    return {
        'mean': mean,
        'worst': float(worst['ap50']),
        'worst_angle': worst_angle,
        'best': float(best['ap50']),
        'best_angle': best_angle,
        'std': statistics.pstdev(vals),
        'range': max(vals) - min(vals),
        'rsi': min(vals) / max(mean, 1e-12),
    }


def classwise_mean_ap(rows: list[dict[str, Any]]) -> dict[str, float]:
    bucket: dict[str, list[float]] = {}
    for row in rows:
        if row.get('status') != 'OK':
            continue
        for cls_name, ap in (row.get('per_class_ap50') or {}).items():
            try:
                bucket.setdefault(cls_name, []).append(float(ap))
            except (TypeError, ValueError):
                continue
    return {
        cls_name: statistics.mean(values)
        for cls_name, values in bucket.items()
        if values
    }


def classwise_best_worst(grouped_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    per_class: dict[str, dict[str, float]] = {}
    for group_name, rows in grouped_rows.items():
        means = classwise_mean_ap(rows)
        for cls_name, ap in means.items():
            per_class.setdefault(cls_name, {})[group_name] = ap
    out = []
    for cls_name, by_group in sorted(per_class.items()):
        if not by_group:
            continue
        best_group, best_ap = max(by_group.items(), key=lambda item: item[1])
        worst_group, worst_ap = min(by_group.items(), key=lambda item: item[1])
        out.append({
            'class_name': cls_name,
            'best_group': best_group,
            'best_ap50': best_ap,
            'worst_group': worst_group,
            'worst_ap50': worst_ap,
            'sensitivity': best_ap - worst_ap,
        })
    return out


def prompt_rank_std(by_group_rows: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    angles = sorted({
        row.get('angle')
        for rows in by_group_rows.values()
        for row in rows
        if row.get('status') == 'OK'
    })
    if not angles:
        return {}
    rank_series: dict[str, list[int]] = {group: [] for group in by_group_rows}
    for angle in angles:
        angle_scores = []
        for group, rows in by_group_rows.items():
            match = next((row for row in rows if row.get('angle') == angle and row.get('status') == 'OK'), None)
            if match is None or match.get('ap50') is None:
                continue
            angle_scores.append((group, float(match['ap50'])))
        angle_scores.sort(key=lambda item: item[1], reverse=True)
        ranks = {group: rank + 1 for rank, (group, _) in enumerate(angle_scores)}
        for group in rank_series:
            if group in ranks:
                rank_series[group].append(ranks[group])
    return {
        group: statistics.pstdev(values) if len(values) > 1 else 0.0
        for group, values in rank_series.items()
        if values
    }


def load_json_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:  # noqa: BLE001
        return None


def group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row.get(key, 'unknown'), []).append(row)
    return grouped


def best_group_by_mean_ap(grouped_rows: dict[str, list[dict[str, Any]]]) -> tuple[str | None, dict[str, float]]:
    means: dict[str, float] = {}
    for group, rows in grouped_rows.items():
        stats = angle_stats(rows)
        if stats:
            means[group] = float(stats['mean'])
    if not means:
        return None, {}
    best_group = max(means.items(), key=lambda item: item[1])[0]
    return best_group, means


def find_row_by_group_angle(rows: list[dict[str, Any]],
                            group_key: str,
                            group_value: str,
                            angle: str,
                            head: str | None = None) -> dict[str, Any] | None:
    for row in rows:
        if row.get(group_key) != group_value:
            continue
        if row.get('angle') != angle:
            continue
        if head is not None and row.get('head') != head:
            continue
        return row
    return None


def top1_labels_from_prediction_pkl(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    if not path.exists():
        return labels
    try:
        samples = load_pickle(path)
    except Exception:  # noqa: BLE001
        return labels
    for sample in samples:
        img_id = str(sample.get('img_id'))
        pred = sample.get('pred_instances') or {}
        scores = pred.get('scores')
        pred_labels = pred.get('labels')
        if scores is None or pred_labels is None:
            continue
        if len(scores) == 0:
            labels[img_id] = -1
            continue
        if hasattr(scores, 'detach'):
            score_vals = scores.detach().cpu()
        else:
            score_vals = torch.as_tensor(scores)
        if hasattr(pred_labels, 'detach'):
            label_vals = pred_labels.detach().cpu()
        else:
            label_vals = torch.as_tensor(pred_labels)
        top_idx = int(torch.argmax(score_vals))
        labels[img_id] = int(label_vals[top_idx])
    return labels


def run_ovd1(args: argparse.Namespace) -> dict[str, Any]:
    inv = latest_inventory(args)
    model = model_from_inventory(args, inv)
    smoke_ok = (args.work_dir / 'preflight_ovd_smoke/smoke.json').exists() and json.loads((args.work_dir / 'preflight_ovd_smoke/smoke.json').read_text(encoding='utf-8')).get('status') == 'DONE'
    exp_dir = args.work_dir / 'exp_ovd1'
    if model is None or (args.mode == 'full' and not smoke_ok):
        report = {
            'status': 'NOT_RUN',
            'reason': 'OpenRSD model unavailable or smoke did not pass.',
            'rows': [],
            'checkpoint': None if model is None else str(model.checkpoint),
            'config': None if model is None else str(model.config),
        }
        write_json(exp_dir / 'exp_ovd1_results.json', report)
        write_ovd1_md(args, report)
        return report
    if args.mode == 'smoke':
        angles = parse_angles(args.only_angle, ['000'])
        max_images = args.max_images_for_smoke
        gpu_ids = args.gpu_ids if parse_csv_arg(args.gpu_ids) == ['4'] else '4'
    else:
        angles = parse_angles(args.only_angle, ANGLES)
        max_images = args.max_images_full
        gpu_ids = args.gpu_ids
    prompt_sets = {
        'dota_class_names_raw': DOTA1_RAW_PROMPTS,
        'remote_sensing_noun_phrase': DOTA1_AERIAL_PROMPTS,
    }
    if args.only_prompt_set:
        prompt_sets = {k: v for k, v in prompt_sets.items() if k == args.only_prompt_set}
    runner = CommandRunner(args)
    rows = []
    for prompt_key, prompt_texts in prompt_sets.items():
        for angle in angles:
            try:
                row = run_single_inference(
                    args=args,
                    runner=runner,
                    model=model,
                    angle=angle,
                    prompt_key=prompt_key,
                    prompt_texts=prompt_texts,
                    class_names=DOTA1_CLASSES,
                    head=args.only_head or 'alignment',
                    exp_dir=exp_dir,
                    batch_size=selected_batch(args),
                    num_workers=args.num_workers,
                    max_images=max_images,
                    gpu_ids=gpu_ids,
                    distributed=args.mode == 'full' and len(parse_csv_arg(gpu_ids)) > 1)
                rows.append(row)
            except Exception as exc:  # noqa: BLE001
                rows.append({
                    'status': 'FAILED',
                    'prompt_key': prompt_key,
                    'angle': angle,
                    'error': repr(exc),
                    'traceback': traceback.format_exc(),
                })
    status = 'DONE' if rows and all(r.get('status') == 'OK' for r in rows) else ('PARTIAL' if any(r.get('status') == 'OK' for r in rows) else 'FAILED')
    report = {
        'status': status,
        'rows': rows,
        'checkpoint': str(model.checkpoint),
        'config': str(model.config),
        'prompt_sets': prompt_sets,
        'batch_size': selected_batch(args),
        'note': 'Prompt texts are recorded; local support text embeddings are reused unless prompt_reencoded=true in support metadata.',
    }
    write_json(exp_dir / 'exp_ovd1_results.json', report)
    write_ovd1_md(args, report)
    return report


def write_ovd1_md(args: argparse.Namespace, report: dict[str, Any]) -> None:
    rows = report.get('rows', [])
    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_prompt.setdefault(row.get('prompt_key', 'unknown'), []).append(row)
    lines = [
        '# Experiment OVD1: Zero-Shot Prompt Rotation Curve',
        '',
        f'- generated_at: `{now()}`',
        f'- status: `{report["status"]}`',
        '- metric: `AP50 / mAP@0.5`',
        f'- checkpoint: `{report.get("checkpoint")}`',
        f'- config: `{report.get("config")}`',
        f'- batch_size: `{report.get("batch_size", "NA")}`',
        f'- work_dir: `{args.work_dir / "exp_ovd1"}`',
        f'- note: `{report.get("note", "")}`',
        '',
        '## Prompt Sets',
        '',
    ]
    for key, prompts in report.get('prompt_sets', {}).items():
        lines.append(f'### {key}')
        for prompt in prompts:
            lines.append(f'- {prompt}')
        family_rows = by_prompt.get(key, [])
        support_meta = next((row.get('support_meta', {}) for row in family_rows if row.get('support_meta')), {})
        lines.extend([
            '',
            f'- prompt type: `text`',
            f'- head type: `{family_rows[0].get("head", "alignment") if family_rows else "alignment"}`',
            f'- text embedding source: `{support_meta.get("text_embedding_source", "NA")}`',
            f'- image embedding source: `{support_meta.get("image_embedding_source", "NA")}`',
            f'- prompt reencoded: `{support_meta.get("prompt_reencoded", False)}`',
            f'- seen/unseen split: `seen-only` for the selected DOTA2-trained checkpoint; no unseen class split on this DOTA1 sweep.',
            f'- class mapping: `prompt order is aligned 1:1 with DOTA1 class order`',
        ])
        lines.append('')
    lines.extend(['## Angle-Wise AP50', ''])
    for prompt_key, prompt_rows in by_prompt.items():
        lines.append(f'### {prompt_key}')
        lines.extend(['', '| angle | status | AP50 | mean conf | det/img | unmatched prompts | images/sec | stdout | stderr |', '|---:|---|---:|---:|---:|---:|---:|---|---|'])
        for row in sorted(prompt_rows, key=lambda r: r.get('angle', '999')):
            lines.append(
                f'| {row.get("angle")} | {row.get("status")} | {fmt(row.get("ap50"))} | '
                f'{fmt(row.get("mean_confidence"))} | {fmt(row.get("avg_detections_per_image"))} | '
                f'{row.get("unmatched_prompt_classes", "NA")} | {fmt(row.get("fps_or_images_per_sec"))} | '
                f'`{row.get("stdout", "")}` | `{row.get("stderr", "")}` |')
        stats = angle_stats(prompt_rows)
        if stats:
            lines.extend([
                '',
                f'- mean: `{fmt(stats["mean"])}`',
                f'- worst: `angle_{stats["worst_angle"]}` `{fmt(stats["worst"])}`',
                f'- best: `angle_{stats["best_angle"]}` `{fmt(stats["best"])}`',
                f'- std: `{fmt(stats["std"])}`',
                f'- range: `{fmt(stats["range"])}`',
                f'- RSI: `{fmt(stats["rsi"])}`',
                '',
            ])
        class_mean = classwise_mean_ap(prompt_rows)
        if class_mean:
            lines.extend(['| class | mean AP50 |', '|---|---:|'])
            for cls_name, ap in sorted(class_mean.items()):
                lines.append(f'| {cls_name} | {fmt(ap)} |')
            lines.append('')
    failed = [r for r in rows if r.get('status') != 'OK']
    lines.extend(['## Failed Angles', ''])
    if failed:
        for row in failed:
            lines.append(f'- {row.get("prompt_key")} angle_{row.get("angle")}: `{row.get("error", row.get("failure_kind", "see logs"))}`')
    else:
        lines.append('- None.')
    lines.extend([
        '',
        '## Qualitative Closed-Set Contrast',
        '',
        '- Open-vocabulary instability is visible as rotation-dependent AP50 variation on the DOTA1 sweep.',
        '- Prompt expression effects are observable here because prompt text embeddings were re-encoded locally with OpenCLIP.',
        '',
        '## Reproduction',
        '',
        f'- `rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mplconfig CUDA_VISIBLE_DEVICES={args.gpu_ids} PYTHONPATH={args.repo_root}:{args.repo_root / "tools"} {args.python_bin} M_Tools/analysis/run_openrsd_ovd_rotation_suite.py --repo-root {args.repo_root} --result-md-dir {args.result_md_dir} --weights-dir {args.weights_dir} --work-dir {args.work_dir} --gpu-ids {args.gpu_ids} --exp ovd1 --mode {args.mode} --resume`',
    ])
    write_md(args.result_md_dir / 'exp_ovd1_zero_shot_prompt_rotation_curve.md', lines)


def write_blocked_exp_md(args: argparse.Namespace,
                         filename: str,
                         exp_title: str,
                         status: str,
                         reason: str,
                         extra_lines: Sequence[str] = ()) -> dict[str, Any]:
    lines = [
        f'# {exp_title}',
        '',
        f'- generated_at: `{now()}`',
        f'- status: `{status}`',
        f'- reason: `{reason}`',
        f'- work_dir: `{args.work_dir}`',
        '',
        *extra_lines,
        '',
        '## Reproduction',
        '',
        f'- `rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mplconfig CUDA_VISIBLE_DEVICES={args.gpu_ids} PYTHONPATH={args.repo_root}:{args.repo_root / "tools"} {args.python_bin} M_Tools/analysis/run_openrsd_ovd_rotation_suite.py --repo-root {args.repo_root} --result-md-dir {args.result_md_dir} --weights-dir {args.weights_dir} --work-dir {args.work_dir} --gpu-ids {args.gpu_ids} --exp {filename.split("_")[1]} --mode {args.mode} --resume`',
    ]
    write_md(args.result_md_dir / filename, lines)
    return {'status': status, 'reason': reason}


def run_ovd2(args: argparse.Namespace) -> dict[str, Any]:
    inv = latest_inventory(args)
    model = model_from_inventory(args, inv)
    smoke_ok = (args.work_dir / 'preflight_ovd_smoke/smoke.json').exists() and json.loads((args.work_dir / 'preflight_ovd_smoke/smoke.json').read_text(encoding='utf-8')).get('status') == 'DONE'
    exp_dir = args.work_dir / 'exp_ovd2'
    if model is None or not smoke_ok:
        report = write_blocked_exp_md(
            args,
            'exp_ovd2_prompt_engineering_rotation_sensitivity.md',
            'Experiment OVD2: Prompt Engineering Rotation Sensitivity',
            'NOT_RUN',
            'Smoke did not pass; prompt-family curves require a loaded OpenRSD model.',
            ['- prompt re-encoding is now available, but the model baseline is not yet validated.'])
        write_json(args.work_dir / 'exp_ovd2/exp_ovd2_results.json', report)
        return report
    if args.mode == 'smoke':
        angles = parse_angles(args.only_angle, ['000'])
        max_images = args.max_images_for_smoke
        gpu_ids = args.gpu_ids if parse_csv_arg(args.gpu_ids) == ['4'] else '4'
    else:
        angles = parse_angles(args.only_angle, ['000', '030', '060', '090', '120', '150']) if not args.only_angle else parse_angles(args.only_angle, ANGLES)
        max_images = args.max_images_full
        gpu_ids = args.gpu_ids if len(parse_csv_arg(args.gpu_ids)) == 1 else '4'
    prompt_families = PROMPT_FAMILIES_OVD2
    if args.only_prompt_set:
        prompt_families = {k: v for k, v in prompt_families.items() if k == args.only_prompt_set}
    runner = CommandRunner(args)
    rows = []
    for family_key, prompts in prompt_families.items():
        for angle in angles:
            try:
                rows.append(run_single_inference(
                    args=args,
                    runner=runner,
                    model=model,
                    angle=angle,
                    prompt_key=family_key,
                    prompt_texts=prompts,
                    class_names=CORE9,
                    head=args.only_head or 'alignment',
                    exp_dir=exp_dir,
                    batch_size=selected_batch(args),
                    num_workers=args.num_workers,
                    max_images=max_images,
                    gpu_ids=gpu_ids,
                    distributed=args.mode == 'full' and len(parse_csv_arg(gpu_ids)) > 1))
            except Exception as exc:  # noqa: BLE001
                rows.append({'status': 'FAILED', 'prompt_key': family_key, 'angle': angle, 'error': repr(exc), 'traceback': traceback.format_exc()})
    status = 'DONE' if rows and all(r.get('status') == 'OK' for r in rows) else ('PARTIAL' if any(r.get('status') == 'OK' for r in rows) else 'FAILED')
    report = {'status': status, 'rows': rows, 'config': str(model.config), 'checkpoint': str(model.checkpoint), 'prompt_families': prompt_families, 'batch_size': selected_batch(args)}
    write_json(exp_dir / 'exp_ovd2_results.json', report)
    write_ovd2_md(args, report)
    return report


def write_ovd2_md(args: argparse.Namespace, report: dict[str, Any]) -> None:
    rows = report.get('rows', [])
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(row.get('prompt_key', 'unknown'), []).append(row)
    lines = [
        '# Experiment OVD2: Prompt Engineering Rotation Sensitivity',
        '',
        f'- generated_at: `{now()}`',
        f'- status: `{report["status"]}`',
        f'- config: `{report.get("config")}`',
        f'- checkpoint: `{report.get("checkpoint")}`',
        f'- batch_size: `{report.get("batch_size", "NA")}`',
        f'- work_dir: `{args.work_dir / "exp_ovd2"}`',
        '',
        '## Prompt Families',
        '',
    ]
    for key, prompts in report.get('prompt_families', {}).items():
        lines.append(f'### {key}')
        for prompt in prompts:
            lines.append(f'- {prompt}')
        family_rows = by_family.get(key, [])
        support_meta = next((row.get('support_meta', {}) for row in family_rows if row.get('support_meta')), {})
        lines.extend([
            '',
            f'- prompt type: `text`',
            f'- head type: `{family_rows[0].get("head", "alignment") if family_rows else "alignment"}`',
            f'- text embedding source: `{support_meta.get("text_embedding_source", "NA")}`',
            f'- image embedding source: `{support_meta.get("image_embedding_source", "NA")}`',
            f'- prompt reencoded: `{support_meta.get("prompt_reencoded", False)}`',
            f'- seen/unseen split: `seen-only` for the selected DOTA2-trained checkpoint; no unseen class split on this core-9 sweep.',
            f'- class mapping: `prompt order is aligned 1:1 with CORE9 class order`',
        ])
        lines.append('')
    lines.extend(['## Angle-Wise AP50', ''])
    for family_key, family_rows in by_family.items():
        lines.append(f'### {family_key}')
        lines.extend(['', '| angle | status | AP50 | mean conf | det/img | unmatched prompts | images/sec | stdout | stderr |', '|---:|---|---:|---:|---:|---:|---:|---|---|'])
        for row in sorted(family_rows, key=lambda r: r.get('angle', '999')):
            lines.append(
                f'| {row.get("angle")} | {row.get("status")} | {fmt(row.get("ap50"))} | '
                f'{fmt(row.get("mean_confidence"))} | {fmt(row.get("avg_detections_per_image"))} | '
                f'{row.get("unmatched_prompt_classes", "NA")} | {fmt(row.get("fps_or_images_per_sec"))} | '
                f'`{row.get("stdout", "")}` | `{row.get("stderr", "")}` |')
        stats = angle_stats(family_rows)
        if stats:
            lines.extend([
                '',
                f'- mean: `{fmt(stats["mean"])}`',
                f'- worst: `angle_{stats["worst_angle"]}` `{fmt(stats["worst"])}`',
                f'- best: `angle_{stats["best_angle"]}` `{fmt(stats["best"])}`',
                f'- std: `{fmt(stats["std"])}`',
                f'- range: `{fmt(stats["range"])}`',
                f'- RSI: `{fmt(stats["rsi"])}`',
                '',
            ])
        class_mean = classwise_mean_ap(family_rows)
        if class_mean:
            lines.extend(['| class | mean AP50 |', '|---|---:|'])
            for cls_name, ap in sorted(class_mean.items()):
                lines.append(f'| {cls_name} | {fmt(ap)} |')
            lines.append('')
    best_worst = classwise_best_worst(by_family)
    if best_worst:
        lines.extend(['## Best / Worst Prompt Family by Class', '', '| class | best family | best AP50 | worst family | worst AP50 | sensitivity |', '|---|---|---:|---|---:|---:|'])
        for row in best_worst:
            lines.append(
                f'| {row["class_name"]} | {row["best_group"]} | {fmt(row["best_ap50"])} | '
                f'{row["worst_group"]} | {fmt(row["worst_ap50"])} | {fmt(row["sensitivity"])} |')
        lines.append('')
    rank_stds = prompt_rank_std(by_family)
    if rank_stds:
        lines.extend(['## Angle-Prompt Rank Interaction', '', '| family | rank std |', '|---|---:|'])
        for family, value in sorted(rank_stds.items()):
            lines.append(f'| {family} | {fmt(value)} |')
        lines.append(f'- mean rank std: `{fmt(statistics.mean(rank_stds.values()))}`')
        lines.append('')
    lines.extend(['## Failed Angles', ''])
    failed = [r for r in rows if r.get('status') != 'OK']
    if failed:
        for row in failed:
            lines.append(f'- {row.get("prompt_key")} angle_{row.get("angle")}: `{row.get("error", row.get("failure_kind", "see logs"))}`')
    else:
        lines.append('- None.')
    lines.extend([
        '',
        '## Prompt Sensitivity',
        '',
        '- Prompt effect is measured with locally re-encoded OpenCLIP text embeddings.',
        '',
        '## Reproduction',
        '',
        f'- `rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mplconfig CUDA_VISIBLE_DEVICES={args.gpu_ids} PYTHONPATH={args.repo_root}:{args.repo_root / "tools"} {args.python_bin} M_Tools/analysis/run_openrsd_ovd_rotation_suite.py --repo-root {args.repo_root} --result-md-dir {args.result_md_dir} --weights-dir {args.weights_dir} --work-dir {args.work_dir} --gpu-ids {args.gpu_ids} --exp ovd2 --mode {args.mode} --resume`',
    ])
    write_md(args.result_md_dir / 'exp_ovd2_prompt_engineering_rotation_sensitivity.md', lines)


def run_ovd3(args: argparse.Namespace) -> dict[str, Any]:
    inv = latest_inventory(args)
    model = model_from_inventory(args, inv)
    smoke_ok = (args.work_dir / 'preflight_ovd_smoke/smoke.json').exists() and json.loads((args.work_dir / 'preflight_ovd_smoke/smoke.json').read_text(encoding='utf-8')).get('status') == 'DONE'
    if model is None or not smoke_ok:
        report = write_blocked_exp_md(
            args,
            'exp_ovd3_alignment_vs_fusion_head_rotation.md',
            'Experiment OVD3: Alignment vs Fusion Head Rotation Robustness',
            'NOT_RUN',
            'Smoke did not pass; head comparison requires a successful model load.',
            [
                '## Head Switch',
                '',
                '- `val_using_aux=False`: alignment head according to README.',
                '- `val_using_aux=True`: fusion head according to README.',
            ])
        write_json(args.work_dir / 'exp_ovd3/exp_ovd3_results.json', report)
        return report
    angles = parse_angles(args.only_angle, ANGLES if args.mode == 'full' else ['000'])
    if args.only_head:
        heads = [args.only_head]
    else:
        heads = ['alignment', 'fusion']
    prompt_key = 'ovd3_F1_aerial_context'
    prompt_texts = [f'aerial photo of a {p.replace("-", " ")}' for p in DOTA1_CLASSES]
    class_names = DOTA1_CLASSES
    ovd2_report = load_json_report(args.work_dir / 'exp_ovd2/exp_ovd2_results.json')
    if ovd2_report and ovd2_report.get('rows'):
        best_family, _ = best_group_by_mean_ap(group_rows(
            [row for row in ovd2_report['rows'] if row.get('status') == 'OK'],
            'prompt_key'))
        if best_family in PROMPT_FAMILIES_OVD2:
            prompt_key = f'ovd3_{best_family}'
            prompt_texts = PROMPT_FAMILIES_OVD2[best_family]
            class_names = CORE9
    runner = CommandRunner(args)
    rows = []
    for head in heads:
        for angle in angles:
            try:
                rows.append(run_single_inference(
                    args=args,
                    runner=runner,
                    model=model,
                    angle=angle,
                    prompt_key=prompt_key,
                    prompt_texts=prompt_texts,
                    class_names=class_names,
                    head=head,
                    exp_dir=args.work_dir / 'exp_ovd3',
                    batch_size=selected_batch(args),
                    num_workers=args.num_workers,
                    max_images=args.max_images_full if args.mode == 'full' else args.max_images_for_smoke,
                    gpu_ids=args.gpu_ids,
                    distributed=args.mode == 'full' and len(parse_csv_arg(args.gpu_ids)) > 1))
            except Exception as exc:  # noqa: BLE001
                rows.append({'status': 'FAILED', 'head': head, 'angle': angle, 'error': repr(exc), 'traceback': traceback.format_exc()})
    status = 'DONE' if rows and all(r.get('status') == 'OK' for r in rows) else ('PARTIAL' if any(r.get('status') == 'OK' for r in rows) else 'FAILED')
    report = {
        'status': status,
        'rows': rows,
        'config': str(model.config),
        'checkpoint': str(model.checkpoint),
        'prompt_key': prompt_key,
        'prompt_texts': prompt_texts,
        'class_names': class_names,
    }
    write_json(args.work_dir / 'exp_ovd3/exp_ovd3_results.json', report)
    write_ovd3_md(args, report)
    return report


def write_ovd3_md(args: argparse.Namespace, report: dict[str, Any]) -> None:
    rows = report.get('rows', [])
    by_head: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_head.setdefault(row.get('head', 'unknown'), []).append(row)
    def collect_head_disagreement(head_a: str = 'alignment', head_b: str = 'fusion') -> dict[str, float]:
        angle_rates = []
        score_gaps = []
        for angle in sorted({
            row.get('angle') for row in rows if row.get('status') == 'OK'
        }):
            row_a = next((r for r in by_head.get(head_a, []) if r.get('angle') == angle and r.get('predictions') and Path(r['predictions']).exists()), None)
            row_b = next((r for r in by_head.get(head_b, []) if r.get('angle') == angle and r.get('predictions') and Path(r['predictions']).exists()), None)
            if row_a is None or row_b is None:
                continue
            preds_a = load_pickle(Path(row_a['predictions']))
            preds_b = load_pickle(Path(row_b['predictions']))
            if len(preds_a) != len(preds_b):
                continue
            disagree = 0
            total = 0
            for sample_a, sample_b in zip(preds_a, preds_b):
                pa = sample_a.get('pred_instances', {})
                pb = sample_b.get('pred_instances', {})
                if not pa or not pb:
                    continue
                score_a = pa['scores']
                score_b = pb['scores']
                label_a = pa['labels']
                label_b = pb['labels']
                if len(score_a) == 0 or len(score_b) == 0:
                    continue
                idx_a = int(torch.argmax(score_a))
                idx_b = int(torch.argmax(score_b))
                if int(label_a[idx_a]) != int(label_b[idx_b]):
                    disagree += 1
                score_gaps.append(float(score_b[idx_b]) - float(score_a[idx_a]))
                total += 1
            if total:
                angle_rates.append(disagree / total)
        return {
            'mean_disagreement': statistics.mean(angle_rates) if angle_rates else float('nan'),
            'mean_score_gap': statistics.mean(score_gaps) if score_gaps else float('nan'),
        }
    disagreement = collect_head_disagreement()
    lines = [
        '# Experiment OVD3: Alignment vs Fusion Head Rotation Robustness',
        '',
        f'- generated_at: `{now()}`',
        f'- status: `{report["status"]}`',
        f'- config: `{report.get("config")}`',
        f'- checkpoint: `{report.get("checkpoint")}`',
        f'- prompt_key: `{report.get("prompt_key", "NA")}`',
        f'- class_names: `{report.get("class_names", [])}`',
        '- head switch: `val_using_aux=False` alignment, `val_using_aux=True` fusion',
        f'- head disagreement rate: `{fmt(disagreement.get("mean_disagreement"))}`',
        f'- score gap (fusion - alignment): `{fmt(disagreement.get("mean_score_gap"))}`',
        '',
        '## Prompt Set',
        '',
    ]
    for prompt in report.get('prompt_texts', []):
        lines.append(f'- {prompt}')
    lines.append('')
    for head in ['alignment', 'fusion']:
        rows = [r for r in report.get('rows', []) if r.get('head') == head]
        if not rows:
            continue
        lines.extend([f'## {head.title()} Head', '', '| angle | status | AP50 | mean conf | det/img | stdout | stderr |', '|---:|---|---:|---:|---:|---|---|'])
        for row in rows:
            lines.append(f'| {row.get("angle")} | {row.get("status")} | {fmt(row.get("ap50"))} | {fmt(row.get("mean_confidence"))} | {fmt(row.get("avg_detections_per_image"))} | `{row.get("stdout", "")}` | `{row.get("stderr", "")}` |')
        stats = angle_stats(rows)
        if stats:
            lines.append(f'- stability: mean `{fmt(stats["mean"])}`, std `{fmt(stats["std"])}`, RSI `{fmt(stats["rsi"])}`')
            lines.append('')
        class_mean = classwise_mean_ap(rows)
        if class_mean:
            lines.extend(['| class | mean AP50 |', '|---|---:|'])
            for cls_name, ap in sorted(class_mean.items()):
                lines.append(f'| {cls_name} | {fmt(ap)} |')
            lines.append('')
    if 'alignment' in by_head and 'fusion' in by_head:
        stats_a = angle_stats(by_head['alignment'])
        stats_b = angle_stats(by_head['fusion'])
        if stats_a and stats_b:
            lines.extend([
                '## Head Comparison',
                '',
                '| metric | alignment | fusion | delta (fusion - alignment) |',
                '|---|---:|---:|---:|',
                f'| mean AP50 | {fmt(stats_a["mean"])} | {fmt(stats_b["mean"])} | {fmt(stats_b["mean"] - stats_a["mean"])} |',
                f'| worst AP50 | {fmt(stats_a["worst"])} | {fmt(stats_b["worst"])} | {fmt(stats_b["worst"] - stats_a["worst"])} |',
                f'| std | {fmt(stats_a["std"])} | {fmt(stats_b["std"])} | {fmt(stats_b["std"] - stats_a["std"])} |',
                f'| RSI | {fmt(stats_a["rsi"])} | {fmt(stats_b["rsi"])} | {fmt(stats_b["rsi"] - stats_a["rsi"])} |',
                '',
            ])
    write_md(args.result_md_dir / 'exp_ovd3_alignment_vs_fusion_head_rotation.md', lines)


def run_ovd4(args: argparse.Namespace) -> dict[str, Any]:
    inv = latest_inventory(args)
    smoke_ok = (args.work_dir / 'preflight_ovd_smoke/smoke.json').exists() and json.loads((args.work_dir / 'preflight_ovd_smoke/smoke.json').read_text(encoding='utf-8')).get('status') == 'DONE'
    exp_dir = args.work_dir / 'exp_ovd4'
    ovd2_path = args.work_dir / 'exp_ovd2/exp_ovd2_results.json'
    ovd2 = load_json_report(ovd2_path)
    model = model_from_inventory(args, inv)
    if model is None or not smoke_ok or not ovd2 or not ovd2.get('rows'):
        report = write_blocked_exp_md(
            args,
            'exp_ovd4_prompt_ensemble_and_rotation_tta_repair.md',
            'Experiment OVD4: Prompt Ensemble + Rotation TTA Repair',
            'NOT_RUN',
            'OVD2 results are required before prompt ensemble / TTA repair can be merged and evaluated.',
            [
                '## Strategies',
                '',
                '- A. single_prompt_single_view',
                '- B. prompt_ensemble_single_view',
                '- C. single_prompt_rotation_tta',
                '- D. prompt_ensemble_rotation_tta',
            ])
        write_json(args.work_dir / 'exp_ovd4/exp_ovd4_results.json', report)
        return report

    rows_ovd2 = [row for row in ovd2['rows'] if row.get('status') == 'OK']
    by_family = group_rows(rows_ovd2, 'prompt_key')
    best_family, family_means = best_group_by_mean_ap(by_family)
    if best_family is None:
        report = write_blocked_exp_md(
            args,
            'exp_ovd4_prompt_ensemble_and_rotation_tta_repair.md',
            'Experiment OVD4: Prompt Ensemble + Rotation TTA Repair',
            'NOT_RUN',
            'No successful OVD2 prompt family rows were available.',
        )
        write_json(args.work_dir / 'exp_ovd4/exp_ovd4_results.json', report)
        return report

    source_angles = sorted({row['angle'] for row in rows_ovd2})
    if args.only_angle:
        target_angles = parse_angles(args.only_angle, source_angles)
    else:
        target_angles = source_angles
    target_angle_set = set(target_angles)
    worst_angle = angle_stats(by_family[best_family]).get('worst_angle', target_angles[0] if target_angles else '000')
    default_score_thr = 0.01
    default_nms_iou = 0.1
    default_score_rule = 'max'
    runner = CommandRunner(args)
    rows: list[dict[str, Any]] = []

    def eval_pkl(config_path: str, pkl_path: Path, tag: str, gpu_ids: str = '4') -> tuple[dict[str, Any], CommandResult]:
        eval_dir = exp_dir / 'eval' / tag
        cmd = [
            str(args.python_bin), 'tools/openrsd_eval_metric.py',
            config_path, str(pkl_path),
        ]
        result = runner.run(
            f'ovd4_eval_{tag}',
            cmd,
            eval_dir / 'logs',
            gpu_ids=gpu_ids,
            monitor_gpu=False)
        metrics = parse_metrics_from_logs(Path(result.stdout_path), Path(result.stderr_path))
        payload = {
            'status': 'OK' if result.returncode == 0 else 'FAILED',
            'ap50': metrics.get('ap50'),
            'map': metrics.get('map'),
            'per_class_ap50': metrics.get('per_class_ap50', {}),
            'stdout': result.stdout_path,
            'stderr': result.stderr_path,
            'command': result.command,
            'returncode': result.returncode,
            'failure_kind': result.failure_kind,
            'duration_sec': result.duration_sec,
        }
        return payload, result

    def merge_pkl(prediction_pairs: list[tuple[str, str]], target_angle: str, tag: str,
                  score_thr: float, nms_iou: float, score_rule: str) -> tuple[Path, CommandResult]:
        merge_dir = exp_dir / 'merge' / tag
        merged_pkl = merge_dir / 'merged_predictions.pkl'
        cmd = [
            str(args.python_bin), 'M_Tools/analysis/rotation_tta_merge.py',
            '--out', str(merged_pkl),
            '--score-thr', f'{score_thr}',
            '--pre-nms-topk', '4000',
            '--nms-iou', f'{nms_iou}',
            '--max-per-img', '2000',
            '--score-rule', score_rule,
            '--img-shape', '1024', '1024',
            '--target-angle', str(int(target_angle)),
        ]
        for pred_path, angle in prediction_pairs:
            cmd.extend(['--prediction', pred_path, '--angle', str(int(angle))])
        result = runner.run(
            f'ovd4_merge_{tag}',
            cmd,
            merge_dir / 'logs',
            gpu_ids='4',
            monitor_gpu=False)
        return merged_pkl, result

    def strategy_row(strategy: str, target_angle: str, base_row: dict[str, Any],
                     merged_pkl: Path | None, eval_payload: dict[str, Any] | None,
                     merge_result: CommandResult | None,
                     source_angles_used: list[str]) -> dict[str, Any]:
        pred_stats = prediction_stats(merged_pkl) if merged_pkl else {}
        row = {
            'strategy': strategy,
            'target_angle': target_angle,
            'status': 'OK' if (eval_payload is None or eval_payload.get('status') == 'OK') else 'FAILED',
            'ap50': base_row.get('ap50') if eval_payload is None else eval_payload.get('ap50'),
            'mean_confidence': base_row.get('mean_confidence') if merged_pkl is None else pred_stats.get('mean_confidence'),
            'avg_detections_per_image': base_row.get('avg_detections_per_image') if merged_pkl is None else pred_stats.get('avg_detections_per_image'),
            'per_class_ap50': base_row.get('per_class_ap50') if eval_payload is None else eval_payload.get('per_class_ap50', {}),
            'predictions': str(merged_pkl) if merged_pkl else base_row.get('predictions'),
            'stdout': base_row.get('stdout') if eval_payload is None else eval_payload.get('stdout', ''),
            'stderr': base_row.get('stderr') if eval_payload is None else eval_payload.get('stderr', ''),
            'merge_stdout': merge_result.stdout_path if merge_result else '',
            'merge_stderr': merge_result.stderr_path if merge_result else '',
            'merge_command': merge_result.command if merge_result else '',
            'eval_command': base_row.get('command') if eval_payload is None else eval_payload.get('command', ''),
            'source_angles': source_angles_used,
            'prompt_family': base_row.get('prompt_key'),
            'head': base_row.get('head', 'alignment'),
            'config': base_row.get('config'),
            'checkpoint': base_row.get('checkpoint'),
        }
        return row

    # Strategy A and B share target-angle rows from OVD2; C/D need merged pkl + eval.
    for target_angle in target_angles:
        base_row = find_row_by_group_angle(rows_ovd2, 'prompt_key', best_family, target_angle)
        if base_row is None:
            continue
        # A. single_prompt_single_view
        rows.append(strategy_row(
            'single_prompt_single_view',
            target_angle,
            base_row,
            None,
            None,
            None,
            [target_angle],
        ))

        # B. prompt_ensemble_single_view
        ensemble_pairs = []
        for family in by_family:
            family_row = find_row_by_group_angle(rows_ovd2, 'prompt_key', family, target_angle)
            if family_row is None:
                continue
            ensemble_pairs.append((family_row['predictions'], target_angle))
        if ensemble_pairs:
            merged_pkl, merge_result = merge_pkl(
                ensemble_pairs, target_angle,
                f'prompt_ensemble_single_view_angle_{target_angle}',
                default_score_thr, default_nms_iou, default_score_rule)
            eval_payload, _ = eval_pkl(base_row['config'], merged_pkl,
                                       f'prompt_ensemble_single_view_angle_{target_angle}')
            rows.append(strategy_row(
                'prompt_ensemble_single_view',
                target_angle,
                base_row,
                merged_pkl,
                eval_payload,
                merge_result,
                [target_angle],
            ))

        # C. single_prompt_rotation_tta
        single_pairs = []
        for source_angle in source_angles:
            family_row = find_row_by_group_angle(rows_ovd2, 'prompt_key', best_family, source_angle)
            if family_row is None:
                continue
            single_pairs.append((family_row['predictions'], source_angle))
        if single_pairs:
            merged_pkl, merge_result = merge_pkl(
                single_pairs, target_angle,
                f'single_prompt_rotation_tta_angle_{target_angle}',
                default_score_thr, default_nms_iou, default_score_rule)
            eval_payload, _ = eval_pkl(base_row['config'], merged_pkl,
                                       f'single_prompt_rotation_tta_angle_{target_angle}')
            rows.append(strategy_row(
                'single_prompt_rotation_tta',
                target_angle,
                base_row,
                merged_pkl,
                eval_payload,
                merge_result,
                source_angles,
            ))

        # D. prompt_ensemble_rotation_tta
        ensemble_tta_pairs = []
        for family in by_family:
            for source_angle in source_angles:
                family_row = find_row_by_group_angle(rows_ovd2, 'prompt_key', family, source_angle)
                if family_row is None:
                    continue
                ensemble_tta_pairs.append((family_row['predictions'], source_angle))
        if ensemble_tta_pairs:
            merged_pkl, merge_result = merge_pkl(
                ensemble_tta_pairs, target_angle,
                f'prompt_ensemble_rotation_tta_angle_{target_angle}',
                default_score_thr, default_nms_iou, default_score_rule)
            eval_payload, _ = eval_pkl(base_row['config'], merged_pkl,
                                       f'prompt_ensemble_rotation_tta_angle_{target_angle}')
            rows.append(strategy_row(
                'prompt_ensemble_rotation_tta',
                target_angle,
                base_row,
                merged_pkl,
                eval_payload,
                merge_result,
                source_angles,
            ))

    # Limited ablation grid on canonical and worst angle for the strongest strategy.
    ablation_rows: list[dict[str, Any]] = []
    target_angle_grid = sorted(set(['000', str(worst_angle)]))
    for target_angle in target_angle_grid:
        base_row = find_row_by_group_angle(rows_ovd2, 'prompt_key', best_family, target_angle)
        if base_row is None:
            continue
        ensemble_tta_pairs = []
        for family in by_family:
            for source_angle in source_angles:
                family_row = find_row_by_group_angle(rows_ovd2, 'prompt_key', family, source_angle)
                if family_row is None:
                    continue
                ensemble_tta_pairs.append((family_row['predictions'], source_angle))
        for score_rule in ('max', 'mean'):
            for score_thr in (0.001, 0.01, 0.05):
                for nms_iou in (0.1, 0.3, 0.5):
                    tag = f'ablation_{target_angle}_thr{score_thr}_iou{nms_iou}_{score_rule}'
                    merged_pkl, merge_result = merge_pkl(
                        ensemble_tta_pairs,
                        target_angle,
                        tag,
                        score_thr,
                        nms_iou,
                        score_rule)
                    eval_payload, _ = eval_pkl(base_row['config'], merged_pkl, tag)
                    ablation_rows.append({
                        'target_angle': target_angle,
                        'score_thr': score_thr,
                        'nms_iou': nms_iou,
                        'score_rule': score_rule,
                        'ap50': eval_payload.get('ap50'),
                        'mean_confidence': prediction_stats(merged_pkl).get('mean_confidence'),
                        'avg_detections_per_image': prediction_stats(merged_pkl).get('avg_detections_per_image'),
                        'merged_pkl': str(merged_pkl),
                        'merge_log': merge_result.stdout_path,
                        'eval_log': eval_payload.get('stdout', ''),
                        'status': eval_payload.get('status', 'FAILED'),
                    })

    # Semantic drift: compare top-1 labels between single view and repaired D strategy.
    drift_rows = []
    by_strategy = group_rows(rows, 'strategy')
    if 'single_prompt_single_view' in by_strategy and 'prompt_ensemble_rotation_tta' in by_strategy:
        for target_angle in target_angles:
            row_a = find_row_by_group_angle(by_strategy['single_prompt_single_view'], 'strategy', 'single_prompt_single_view', target_angle, None)
            row_d = find_row_by_group_angle(by_strategy['prompt_ensemble_rotation_tta'], 'strategy', 'prompt_ensemble_rotation_tta', target_angle, None)
            if row_a is None or row_d is None:
                continue
            top_a = top1_labels_from_prediction_pkl(Path(row_a['predictions']))
            top_d = top1_labels_from_prediction_pkl(Path(row_d['predictions']))
            common_ids = set(top_a) & set(top_d)
            if not common_ids:
                continue
            drift = sum(int(top_a[img_id] != top_d[img_id]) for img_id in common_ids) / len(common_ids)
            drift_rows.append({'target_angle': target_angle, 'semantic_drift_rate': drift})

    expected_strategies = {
        'single_prompt_single_view',
        'prompt_ensemble_single_view',
        'single_prompt_rotation_tta',
        'prompt_ensemble_rotation_tta',
    }
    observed_strategies = {row.get('strategy') for row in rows}
    status = 'DONE' if (
        rows
        and expected_strategies.issubset(observed_strategies)
        and all(r.get('status') == 'OK' for r in rows if r.get('strategy'))
    ) else ('PARTIAL' if rows else 'FAILED')
    report = {
        'status': status,
        'config': str(model.config),
        'checkpoint': str(model.checkpoint),
        'best_prompt_family': best_family,
        'family_mean_ap50': family_means,
        'source_angles': source_angles,
        'target_angles': target_angles,
        'default_score_thr': default_score_thr,
        'default_nms_iou': default_nms_iou,
        'default_score_rule': default_score_rule,
        'worst_angle': str(worst_angle),
        'rows': rows,
        'ablation_rows': ablation_rows,
        'semantic_drift_rows': drift_rows,
    }
    write_json(exp_dir / 'exp_ovd4_results.json', report)
    write_ovd4_md(args, report)
    return report


def write_ovd4_md(args: argparse.Namespace, report: dict[str, Any]) -> None:
    rows = report.get('rows', [])
    by_strategy = group_rows(rows, 'strategy')
    summary_rows = []
    for strategy, strategy_rows in by_strategy.items():
        stats = angle_stats(strategy_rows)
        if not stats:
            continue
        mean_dets = statistics.mean(
            [float(r['avg_detections_per_image']) for r in strategy_rows if r.get('avg_detections_per_image') is not None]
        ) if any(r.get('avg_detections_per_image') is not None for r in strategy_rows) else None
        summary_rows.append({
            'strategy': strategy,
            'mean': stats['mean'],
            'worst': stats['worst'],
            'worst_angle': stats['worst_angle'],
            'best': stats['best'],
            'best_angle': stats['best_angle'],
            'std': stats['std'],
            'range': stats['range'],
            'rsi': stats['rsi'],
            'mean_dets': mean_dets,
        })
    summary_rows.sort(key=lambda r: r['mean'], reverse=True)
    best_strategy = summary_rows[0]['strategy'] if summary_rows else 'NA'
    semantic_drift = report.get('semantic_drift_rows', [])
    drift_mean = statistics.mean([float(r['semantic_drift_rate']) for r in semantic_drift]) if semantic_drift else None
    a_rows = by_strategy.get('single_prompt_single_view', [])
    d_rows = by_strategy.get('prompt_ensemble_rotation_tta', [])
    a_by_angle = {r['target_angle']: r for r in a_rows}
    d_by_angle = {r['target_angle']: r for r in d_rows}
    canonical_damage = None
    worst_gain = None
    worst_angle = report.get('worst_angle')
    if '000' in a_by_angle and '000' in d_by_angle:
        try:
            canonical_damage = float(d_by_angle['000']['ap50']) - float(a_by_angle['000']['ap50'])
        except (TypeError, ValueError):
            canonical_damage = None
    if worst_angle in a_by_angle and worst_angle in d_by_angle:
        try:
            worst_gain = float(d_by_angle[worst_angle]['ap50']) - float(a_by_angle[worst_angle]['ap50'])
        except (TypeError, ValueError):
            worst_gain = None
    lines = [
        '# Experiment OVD4: Prompt Ensemble + Rotation TTA Repair',
        '',
        f'- generated_at: `{now()}`',
        f'- status: `{report["status"]}`',
        f'- config: `{report.get("config")}`',
        f'- checkpoint: `{report.get("checkpoint")}`',
        f'- best_prompt_family: `{report.get("best_prompt_family")}`',
        f'- source_angles: `{report.get("source_angles", [])}`',
        f'- target_angles: `{report.get("target_angles", [])}`',
        f'- default_score_thr: `{report.get("default_score_thr")}`',
        f'- default_nms_iou: `{report.get("default_nms_iou")}`',
        f'- default_score_rule: `{report.get("default_score_rule")}`',
        f'- worst_angle: `{report.get("worst_angle")}`',
        '',
        '## Strategies',
        '',
        '- A. single_prompt_single_view',
        '- B. prompt_ensemble_single_view',
        '- C. single_prompt_rotation_tta',
        '- D. prompt_ensemble_rotation_tta',
        '',
        '## Strategy Summary',
        '',
        '| strategy | mean AP50 | worst AP50 | worst angle | best AP50 | best angle | std | range | RSI | mean det/img |',
        '|---|---:|---:|---|---:|---|---:|---:|---:|---:|',
    ]
    for row in summary_rows:
        lines.append(
            f'| {row["strategy"]} | {fmt(row["mean"])} | {fmt(row["worst"])} | angle_{row["worst_angle"]} | '
            f'{fmt(row["best"])} | angle_{row["best_angle"]} | {fmt(row["std"])} | {fmt(row["range"])} | '
            f'{fmt(row["rsi"])} | {fmt(row["mean_dets"])} |')
    lines.extend([
        '',
        '## Key Deltas',
        '',
        f'- canonical damage (D - A @ angle_000): `{fmt(canonical_damage)}`',
        f'- worst-angle gain (D - A @ angle_{report.get("worst_angle")}): `{fmt(worst_gain)}`',
        f'- mean semantic drift rate (A -> D): `{fmt(drift_mean)}`',
        '',
        '## Angle-Wise AP50',
        '',
    ])
    for strategy, strategy_rows in by_strategy.items():
        lines.extend([f'### {strategy}', '', '| angle | AP50 | mean conf | det/img | status | predictions |', '|---:|---:|---:|---:|---|---|'])
        for row in sorted(strategy_rows, key=lambda r: r.get('target_angle', '999')):
            lines.append(
                f'| {row.get("target_angle")} | {fmt(row.get("ap50"))} | {fmt(row.get("mean_confidence"))} | '
                f'{fmt(row.get("avg_detections_per_image"))} | {row.get("status")} | `{row.get("predictions", "")}` |')
        class_mean = classwise_mean_ap(strategy_rows)
        if class_mean:
            lines.extend(['', '| class | mean AP50 |', '|---|---:|'])
            for cls_name, ap in sorted(class_mean.items()):
                lines.append(f'| {cls_name} | {fmt(ap)} |')
        lines.append('')
    if semantic_drift:
        lines.extend(['## Semantic Drift', '', '| target angle | drift rate |', '|---:|---:|'])
        for row in semantic_drift:
            lines.append(f'| {row["target_angle"]} | {fmt(row["semantic_drift_rate"])} |')
        lines.append('')
    if report.get('ablation_rows'):
        lines.extend(['## Ablation Table', '', '| target angle | score thr | NMS IoU | score rule | AP50 | mean conf | det/img | status |', '|---:|---:|---:|---|---:|---:|---:|---|'])
        for row in sorted(report['ablation_rows'], key=lambda r: (r['target_angle'], r['score_rule'], r['score_thr'], r['nms_iou'])):
            lines.append(
                f'| {row["target_angle"]} | {row["score_thr"]} | {row["nms_iou"]} | {row["score_rule"]} | '
                f'{fmt(row.get("ap50"))} | {fmt(row.get("mean_confidence"))} | {fmt(row.get("avg_detections_per_image"))} | {row.get("status", "NA")} |')
        lines.append('')
    lines.extend([
        '## Conclusion',
        '',
        f'- best strategy by mean AP50: `{best_strategy}`',
        f'- prompt ensemble / TTA repair on the worst angle: `{fmt(worst_gain)}`',
        f'- canonical damage on angle_000: `{fmt(canonical_damage)}`',
        '',
        '## Reproduction',
        '',
        f'- `rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mplconfig CUDA_VISIBLE_DEVICES={args.gpu_ids} PYTHONPATH={args.repo_root}:{args.repo_root / "tools"} {args.python_bin} M_Tools/analysis/run_openrsd_ovd_rotation_suite.py --repo-root {args.repo_root} --result-md-dir {args.result_md_dir} --weights-dir {args.weights_dir} --work-dir {args.work_dir} --gpu-ids {args.gpu_ids} --exp ovd4 --mode {args.mode} --resume`',
    ])
    write_md(args.result_md_dir / 'exp_ovd4_prompt_ensemble_and_rotation_tta_repair.md', lines)


def run_ovd5(args: argparse.Namespace) -> dict[str, Any]:
    csv_path = args.work_dir / 'exp_ovd5/open_vocab_cross_view_diagnostic.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        'model', 'head', 'prompt_family', 'image_id', 'angle',
        'class_name', 'feature_layer', 'feature_cosine', 'feature_l2',
        'text_image_similarity', 'top1_prompt', 'top1_consistent',
        'top5_consistent', 'confidence', 'confidence_drop',
        'matched_iou', 'box_angle_diff', 'semantic_consistent'
    ]
    inv = latest_inventory(args)
    model = model_from_inventory(args, inv)
    ovd3 = load_json_report(args.work_dir / 'exp_ovd3/exp_ovd3_results.json')
    ovd2 = load_json_report(args.work_dir / 'exp_ovd2/exp_ovd2_results.json')

    def canon_img_id(img_id: Any) -> str:
        text = str(img_id)
        if text.startswith('angle_') and '__' in text:
            return text.split('__', 1)[1]
        return text

    def top_info(sample: dict[str, Any], class_names: list[str]) -> dict[str, Any]:
        pred = sample.get('pred_instances') or {}
        scores = pred.get('scores')
        labels = pred.get('labels')
        boxes = pred.get('bboxes')
        if scores is None or labels is None or boxes is None or len(scores) == 0:
            return {
                'label': -1,
                'class_name': '',
                'score': float('nan'),
                'box': None,
                'top5': [],
            }
        if hasattr(scores, 'detach'):
            scores_t = scores.detach().cpu()
        else:
            scores_t = torch.as_tensor(scores)
        if hasattr(labels, 'detach'):
            labels_t = labels.detach().cpu()
        else:
            labels_t = torch.as_tensor(labels)
        if hasattr(boxes, 'detach'):
            boxes_t = boxes.detach().cpu()
        else:
            boxes_t = torch.as_tensor(boxes)
        order = torch.argsort(scores_t, descending=True)
        top_idx = int(order[0])
        label = int(labels_t[top_idx])
        class_name = class_names[label] if 0 <= label < len(class_names) else str(label)
        top5 = []
        for idx in order[:5]:
            label_i = int(labels_t[int(idx)])
            top5.append(class_names[label_i] if 0 <= label_i < len(class_names) else str(label_i))
        return {
            'label': label,
            'class_name': class_name,
            'score': float(scores_t[top_idx]),
            'box': boxes_t[top_idx].numpy().astype(np.float32),
            'top5': top5,
        }

    def compare_prediction_pkls(source_row: dict[str, Any],
                                canon_row: dict[str, Any],
                                class_names: list[str],
                                head: str,
                                prompt_family: str) -> list[dict[str, Any]]:
        from mmcv.ops import box_iou_rotated
        from M_Tools.analysis.rotation_tta_merge import transform_boxes_to_target

        angle = source_row.get('angle')
        if not angle or not source_row.get('predictions') or not canon_row.get('predictions'):
            return []
        source_path = Path(source_row['predictions'])
        canon_path = Path(canon_row['predictions'])
        if not source_path.exists() or not canon_path.exists():
            return []
        source_samples = load_pickle(source_path)
        canon_samples = load_pickle(canon_path)
        source_map = {canon_img_id(sample.get('img_id')): top_info(sample, class_names) for sample in source_samples}
        canon_map = {canon_img_id(sample.get('img_id')): top_info(sample, class_names) for sample in canon_samples}
        out = []
        for img_id in sorted(set(source_map) & set(canon_map)):
            source = source_map[img_id]
            canon = canon_map[img_id]
            matched_iou = float('nan')
            box_angle_diff = float('nan')
            if source['box'] is not None and canon['box'] is not None:
                source_box = transform_boxes_to_target(
                    source['box'][None, :].astype(np.float32),
                    int(angle),
                    0,
                    1024,
                    1024)
                iou = box_iou_rotated(
                    torch.from_numpy(source_box.astype(np.float32)),
                    torch.from_numpy(canon['box'][None, :].astype(np.float32)))
                matched_iou = float(iou.item())
                diff = (float(source_box[0, 4]) - float(canon['box'][4]) + math.pi / 2) % math.pi - math.pi / 2
                box_angle_diff = abs(diff)
            top1_consistent = source['label'] == canon['label']
            top5_consistent = canon['class_name'] in source['top5']
            confidence_drop = float(canon['score'] - source['score']) if not math.isnan(canon['score']) and not math.isnan(source['score']) else float('nan')
            out.append({
                'model': model.key if model else 'NA',
                'head': head,
                'prompt_family': prompt_family,
                'image_id': img_id,
                'angle': angle,
                'class_name': source['class_name'],
                'feature_layer': 'prediction_top1',
                'feature_cosine': 'NA',
                'feature_l2': 'NA',
                'text_image_similarity': 'NA',
                'top1_prompt': source['class_name'],
                'top1_consistent': top1_consistent,
                'top5_consistent': top5_consistent,
                'confidence': source['score'],
                'confidence_drop': confidence_drop,
                'matched_iou': matched_iou,
                'box_angle_diff': box_angle_diff,
                'semantic_consistent': top1_consistent,
            })
        return out

    diagnostic_rows: list[dict[str, Any]] = []
    source_name = 'none'
    prompt_family = 'NA'
    class_names = DOTA1_CLASSES
    if ovd3 and ovd3.get('rows'):
        source_name = 'ovd3'
        prompt_family = ovd3.get('prompt_key', 'ovd3')
        class_names = ovd3.get('class_names') or DOTA1_CLASSES
        by_head = group_rows([row for row in ovd3['rows'] if row.get('status') == 'OK'], 'head')
        for head, head_rows in by_head.items():
            canon_row = next((row for row in head_rows if row.get('angle') == '000'), None)
            if canon_row is None:
                continue
            for row in sorted(head_rows, key=lambda item: item.get('angle', '999')):
                if row.get('angle') == '000':
                    continue
                diagnostic_rows.extend(compare_prediction_pkls(row, canon_row, class_names, head, prompt_family))
    if not diagnostic_rows and ovd2 and ovd2.get('rows'):
        source_name = 'ovd2'
        ok_rows = [row for row in ovd2['rows'] if row.get('status') == 'OK']
        by_family = group_rows(ok_rows, 'prompt_key')
        best_family, _ = best_group_by_mean_ap(by_family)
        prompt_family = best_family or 'unknown'
        class_names = CORE9
        if best_family:
            family_rows = by_family[best_family]
            canon_row = next((row for row in family_rows if row.get('angle') == '000'), None)
            if canon_row is not None:
                for row in sorted(family_rows, key=lambda item: item.get('angle', '999')):
                    if row.get('angle') == '000':
                        continue
                    diagnostic_rows.extend(compare_prediction_pkls(row, canon_row, class_names, row.get('head', 'alignment'), prompt_family))

    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in diagnostic_rows:
            writer.writerow(row)

    summary: dict[str, dict[str, Any]] = {}
    for row in diagnostic_rows:
        key = f'{row["head"]}|{row["angle"]}'
        item = summary.setdefault(key, {
            'head': row['head'],
            'angle': row['angle'],
            'top1': [],
            'top5': [],
            'conf_drop': [],
            'iou': [],
            'angle_diff': [],
        })
        item['top1'].append(bool(row['top1_consistent']))
        item['top5'].append(bool(row['top5_consistent']))
        if isinstance(row['confidence_drop'], float) and not math.isnan(row['confidence_drop']):
            item['conf_drop'].append(row['confidence_drop'])
        if isinstance(row['matched_iou'], float) and not math.isnan(row['matched_iou']):
            item['iou'].append(row['matched_iou'])
        if isinstance(row['box_angle_diff'], float) and not math.isnan(row['box_angle_diff']):
            item['angle_diff'].append(row['box_angle_diff'])
    summary_rows = []
    for item in summary.values():
        summary_rows.append({
            'head': item['head'],
            'angle': item['angle'],
            'top1_consistency': statistics.mean(item['top1']) if item['top1'] else None,
            'top5_consistency': statistics.mean(item['top5']) if item['top5'] else None,
            'mean_confidence_drop': statistics.mean(item['conf_drop']) if item['conf_drop'] else None,
            'mean_matched_iou': statistics.mean(item['iou']) if item['iou'] else None,
            'mean_box_angle_diff': statistics.mean(item['angle_diff']) if item['angle_diff'] else None,
            'samples': len(item['top1']),
        })
    status = 'PARTIAL' if diagnostic_rows else 'NOT_RUN'
    report = {
        'status': status,
        'source': source_name,
        'csv_path': str(csv_path),
        'prompt_family': prompt_family,
        'class_names': class_names,
        'successful_hooks': ['prediction_top1'],
        'failed_hooks': [
            'backbone last layer',
            'neck/FPN P3/P4/P5',
            'text embedding',
            'image prompt embedding',
            'alignment head logits',
            'fusion head logits',
            'objectness score',
            'class score',
            'bbox prediction hook',
        ],
        'summary_rows': sorted(summary_rows, key=lambda r: (r['head'], r['angle'])),
        'row_count': len(diagnostic_rows),
    }
    write_json(args.work_dir / 'exp_ovd5/exp_ovd5_results.json', report)
    write_ovd5_md(args, report)
    return report


def write_ovd5_md(args: argparse.Namespace, report: dict[str, Any]) -> None:
    lines = [
        '# Experiment OVD5: Open-Vocabulary Cross-View Diagnostic',
        '',
        f'- generated_at: `{now()}`',
        f'- status: `{report["status"]}`',
        f'- source: `{report.get("source")}`',
        f'- csv_path: `{report.get("csv_path")}`',
        f'- prompt_family: `{report.get("prompt_family")}`',
        f'- row_count: `{report.get("row_count")}`',
        '',
        '## Hook Layers',
        '',
        '| module | status |',
        '|---|---|',
    ]
    for hook in report.get('successful_hooks', []):
        lines.append(f'| {hook} | SUCCESS |')
    for hook in report.get('failed_hooks', []):
        lines.append(f'| {hook} | NOT_RUN |')
    lines.extend(['', '## Prediction-Level Diagnostic Summary', '', '| head | angle | top1 consistency | top5 consistency | conf drop | matched IoU | box angle diff | samples |', '|---|---:|---:|---:|---:|---:|---:|---:|'])
    for row in report.get('summary_rows', []):
        lines.append(
            f'| {row["head"]} | {row["angle"]} | {fmt(row.get("top1_consistency"))} | '
            f'{fmt(row.get("top5_consistency"))} | {fmt(row.get("mean_confidence_drop"))} | '
            f'{fmt(row.get("mean_matched_iou"))} | {fmt(row.get("mean_box_angle_diff"))} | {row.get("samples")} |')
    lines.extend([
        '',
        '## Interpretation',
        '',
        '- This run is a prediction-level cross-view diagnostic. Feature/logit hooks were not executed, so visual-feature and text-image embedding collapse are not directly attributed here.',
        '- `prediction_top1` rows still diagnose cross-view semantic consistency, confidence drop, and box consistency for the completed OpenRSD predictions.',
        '',
        '## Reproduction',
        '',
        f'- `rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mplconfig CUDA_VISIBLE_DEVICES={args.gpu_ids} PYTHONPATH={args.repo_root}:{args.repo_root / "tools"} {args.python_bin} M_Tools/analysis/run_openrsd_ovd_rotation_suite.py --repo-root {args.repo_root} --result-md-dir {args.result_md_dir} --weights-dir {args.weights_dir} --work-dir {args.work_dir} --gpu-ids {args.gpu_ids} --exp ovd5 --mode {args.mode} --resume`',
    ])
    write_md(args.result_md_dir / 'exp_ovd5_open_vocab_cross_view_diagnostic.md', lines)


def collect_existing_reports(args: argparse.Namespace) -> dict[str, Any]:
    mapping = {
        'ovd0': args.work_dir / 'exp_ovd0/openrsd_inventory.json',
        'smoke': args.work_dir / 'preflight_ovd_smoke/smoke.json',
        'multigpu': args.work_dir / 'preflight_ovd_multigpu_batchsize/multigpu_batchsize.json',
        'ovd1': args.work_dir / 'exp_ovd1/exp_ovd1_results.json',
        'ovd2': args.work_dir / 'exp_ovd2/exp_ovd2_results.json',
        'ovd3': args.work_dir / 'exp_ovd3/exp_ovd3_results.json',
        'ovd4': args.work_dir / 'exp_ovd4/exp_ovd4_results.json',
        'ovd5': args.work_dir / 'exp_ovd5/exp_ovd5_results.json',
    }
    reports = {}
    for key, path in mapping.items():
        if path.exists():
            try:
                reports[key] = json.loads(path.read_text(encoding='utf-8'))
            except Exception:  # noqa: BLE001
                reports[key] = {'status': 'FAILED_TO_READ', 'path': str(path)}
    return reports


def write_summary(args: argparse.Namespace, updates: dict[str, Any] | None = None) -> None:
    reports = collect_existing_reports(args)
    if updates:
        reports.update(updates)
    inv = reports.get('ovd0', {})
    md_paths = {
        'OVD0': args.result_md_dir / 'exp_ovd0_preflight_openrsd_inventory.md',
        'OVD1': args.result_md_dir / 'exp_ovd1_zero_shot_prompt_rotation_curve.md',
        'OVD2': args.result_md_dir / 'exp_ovd2_prompt_engineering_rotation_sensitivity.md',
        'OVD3': args.result_md_dir / 'exp_ovd3_alignment_vs_fusion_head_rotation.md',
        'OVD4': args.result_md_dir / 'exp_ovd4_prompt_ensemble_and_rotation_tta_repair.md',
        'OVD5': args.result_md_dir / 'exp_ovd5_open_vocab_cross_view_diagnostic.md',
    }
    lines = [
        '# Summary: OpenRSD OVD Rotation Experiments 20260508',
        '',
        f'- generated_at: `{now()}`',
        '- motivation: test whether open-vocabulary / open-prompt remote-sensing detectors suffer rotation-induced recognition inconsistency and whether failures arise from visual-text alignment drift rather than only box localization.',
        f'- repo_root: `{args.repo_root}`',
        f'- work_dir: `{args.work_dir}`',
        f'- result_md_dir: `{args.result_md_dir}`',
        '',
        '## Repository And Environment',
        '',
        f'- git_commit: `{inv.get("git_commit", "NA")}`',
        f'- selected_model: `{inv.get("selected_model", {})}`',
        '',
        '| component | version |',
        '|---|---|',
    ]
    for key, value in inv.get('versions', {}).items():
        lines.append(f'| {key} | `{value}` |')
    lines.extend(['', '## CUDA / GPU', '', '```text', inv.get('gpu_info', 'NA'), '```'])
    lines.extend(['', '## Three-Round Preflight', '', '| stage | status | key note |', '|---|---|---|'])
    lines.append(f'| dryrun | {reports.get("ovd0", {}).get("status", "NOT_RUN")} | inventory and planned commands |')
    lines.append(f'| smoke | {reports.get("smoke", {}).get("status", "NOT_RUN")} | `{args.result_md_dir / "preflight_ovd_smoke.md"}` |')
    lines.append(f'| multigpu + batch size | {reports.get("multigpu", {}).get("status", "NOT_RUN")} | selected batch `{reports.get("multigpu", {}).get("selected_batch_size", "NA")}` |')
    lines.extend(['', '## Experiment Status', '', '| experiment | status | md path | key result |', '|---|---|---|---|'])
    for exp in ('OVD0', 'OVD1', 'OVD2', 'OVD3', 'OVD4', 'OVD5'):
        key = exp.lower()
        status = reports.get(key, {}).get('status', 'NOT_RUN')
        if exp == 'OVD0':
            status = reports.get('ovd0', {}).get('status', 'NOT_RUN')
        note = reports.get(key, {}).get('reason', '')
        if key == 'ovd1' and reports.get(key, {}).get('rows'):
            stats = angle_stats(reports[key]['rows'])
            note = f'mean AP50 {fmt(stats.get("mean"))}, worst {stats.get("worst_angle", "NA")}' if stats else 'rows exist, metrics unavailable'
        elif key == 'ovd2' and reports.get(key, {}).get('rows'):
            best_family, means = best_group_by_mean_ap(group_rows(reports[key]['rows'], 'prompt_key'))
            note = f'best prompt {best_family}, mean AP50 {fmt(means.get(best_family) if best_family else None)}'
        elif key == 'ovd3' and reports.get(key, {}).get('rows'):
            best_head, means = best_group_by_mean_ap(group_rows(reports[key]['rows'], 'head'))
            note = f'best head {best_head}, mean AP50 {fmt(means.get(best_head) if best_head else None)}'
        elif key == 'ovd4' and reports.get(key, {}).get('rows'):
            best_strategy, means = best_group_by_mean_ap(group_rows(reports[key]['rows'], 'strategy'))
            note = f'best strategy {best_strategy}, mean AP50 {fmt(means.get(best_strategy) if best_strategy else None)}'
        elif key == 'ovd5' and reports.get(key, {}).get('csv_path'):
            note = f'prediction diagnostic rows {reports[key].get("row_count", "NA")}, source {reports[key].get("source", "NA")}'
        lines.append(f'| {exp} | {status} | `{md_paths[exp]}` | {note} |')
    proposition = 'NOT_EVALUABLE'
    if reports.get('ovd1', {}).get('status') in ('DONE', 'PARTIAL') and angle_stats(reports.get('ovd1', {}).get('rows', [])):
        proposition = 'PARTIALLY_SUPPORTED_BY_COMPLETED_ROWS'
    lines.extend([
        '',
        '## Proposition',
        '',
        f'- Open-vocabulary remote-sensing detectors also suffer from rotation-induced recognition inconsistency, and the failure may arise from visual-text alignment drift rather than only box localization: `{proposition}`',
        '',
        '## Key Findings',
        '',
        f'- OpenRSD config/checkpoint pair selected: `{inv.get("selected_model", "NA")}`.',
        f'- DOTA1 12-angle sweep complete: `{all(r.get("annfiles") and r.get("images") for r in inv.get("data", {}).get("dota1_angles", [])) if inv else "NA"}`.',
        f'- Text prompt support exists in code: `{inv.get("capabilities", {}).get("text_prompt", "NA")}`.',
        f'- Image prompt support exists in code: `{inv.get("capabilities", {}).get("image_prompt", "NA")}`.',
        f'- Prompt text re-encoding was verified: `{inv.get("prompt_reencode_available", False)}`.',
        '',
        '## Risks / Failures',
        '',
    ])
    blockers = inv.get('hard_blockers', []) or []
    if blockers:
        for blocker in blockers[:5]:
            lines.append(f'- {blocker}')
    else:
        lines.extend([
            '- OVD2/OVD4 are 32-image subset experiments in this run unless rerun without `--max-images-full`.',
            '- OVD5 is prediction-level unless the requested feature/logit hooks are enabled in a follow-up run.',
            '- Fusion head may fail independently if `val_using_aux=True` is incompatible with the selected checkpoint.',
            '- Distributed inference may fall back to task-level parallelism if OpenRSD runner launch fails.',
            '- Per-class AP depends on DETAILDOTAMetric log serialization.',
        ])
    lines.extend([
        '',
        '## Closed-Set Relation',
        '',
        '- Open-vocabulary instability vs closed-set instability: OVD1 supplies the direct angle curve for comparison.',
        '- Prompt effect on rotation robustness: OVD2 uses locally re-encoded OpenCLIP prompt embeddings.',
        '- Head choice effect: OVD3 records `val_using_aux=False` alignment and `val_using_aux=True` fusion behavior.',
        '- Prompt ensemble / TTA repair: OVD4 reuses validated OVD2 predictions and `rotation_tta_merge.py` for prompt/rotation merging.',
        '',
        '## Next Experiments',
        '',
        '- Rerun OVD2/OVD4 without `--max-images-full 32` for full-data AP50 once the subset conclusions are stable.',
        '- Extend OVD2/OVD4 from the 6-angle prompt-family sweep to a 12-angle prompt-family sweep if time allows.',
        '- Add actual feature/logit hooks for OVD5 to separate visual feature collapse from text-image alignment drift.',
        '- Repeat OVD3 on all 12 angles if the fusion head completes the 32-image subset cleanly.',
        '- Compare OpenCLIP ViT-L/14 prompt embeddings with the exact SkyCLIP encoder originally used to build OpenRSD support assets.',
        '',
        '## Important Logs',
        '',
        f'- command log: `{args.work_dir / "commands.jsonl"}`',
        f'- dryrun inventory json: `{args.work_dir / "exp_ovd0/openrsd_inventory.json"}`',
        f'- smoke json: `{args.work_dir / "preflight_ovd_smoke/smoke.json"}`',
        f'- multigpu json: `{args.work_dir / "preflight_ovd_multigpu_batchsize/multigpu_batchsize.json"}`',
    ])
    write_md(args.result_md_dir / 'summary_openrsd_ovd_rotation_20260508.md', lines)


def run_all_full_sequence(args: argparse.Namespace) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    reports['ovd0'] = inventory(args)
    reports['smoke'] = run_smoke(args)
    if reports['smoke'].get('status') != 'DONE':
        write_summary(args, reports)
        return reports
    reports['multigpu'] = run_multigpu_batchsize(args)
    for name, func in [('ovd1', run_ovd1), ('ovd2', run_ovd2), ('ovd3', run_ovd3), ('ovd4', run_ovd4), ('ovd5', run_ovd5)]:
        reports[name] = func(args)
        write_summary(args, reports)
    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run OpenRSD OVD rotation experiments')
    parser.add_argument('--repo-root', type=Path, default=Path('/data1/zcy/OpenRSD'))
    parser.add_argument('--result-md-dir', type=Path, default=Path('/data1/zcy/OpenRSD/resultmd'))
    parser.add_argument('--weights-dir', type=Path, default=Path('/data1/zcy/OpenRSD/results'))
    parser.add_argument('--work-dir', type=Path, default=Path('/data1/zcy/OpenRSD/work_dirs/openrsd_ovd_rotation_20260508'))
    parser.add_argument('--gpu-ids', default='4,5,6,7')
    parser.add_argument('--exp', default='all', choices=['all', 'ovd0', 'ovd1', 'ovd2', 'ovd3', 'ovd4', 'ovd5'])
    parser.add_argument('--mode', default='dryrun', choices=['dryrun', 'smoke', 'full', 'debug'])
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--batch-size-candidates', default='1,2,4,8,16')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--max-images-for-smoke', type=int, default=16)
    parser.add_argument('--max-images-full', type=int, default=None)
    parser.add_argument('--max-iters-for-smoke', type=int, default=None)
    parser.add_argument('--only-model', default=None)
    parser.add_argument('--only-dataset', default=None)
    parser.add_argument('--only-angle', default=None)
    parser.add_argument('--only-prompt-set', default=None)
    parser.add_argument('--only-head', default=None)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--resume', action='store_true')
    default_python = Path('/data/zcy/anaconda3/envs/openrsd/bin/python')
    if not default_python.exists():
        default_python = Path(sys.executable)
    parser.add_argument('--python-bin', type=Path, default=default_python)
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.result_md_dir = args.result_md_dir.resolve()
    args.weights_dir = args.weights_dir.resolve()
    args.work_dir = args.work_dir.resolve()
    if args.mode == 'smoke' and len(parse_csv_arg(args.gpu_ids)) == 1 and parse_csv_arg(args.gpu_ids)[0] != '4':
        log(f'WARNING smoke requested gpu_ids={args.gpu_ids}; user requirement expects physical GPU 4.')
    return args


def main() -> None:
    args = parse_args()
    args.result_md_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.exp == 'all' and args.mode == 'full':
        run_all_full_sequence(args)
        return
    reports: dict[str, Any] = {}
    if args.mode == 'dryrun' or args.exp == 'ovd0':
        reports['ovd0'] = inventory(args)
        write_summary(args, reports)
        return
    if args.mode == 'smoke' and args.exp == 'ovd1' and args.batch_size_candidates and not args.batch_size:
        # Batch-size preflight can run in distributed mode when multiple GPUs
        # are provided, or as a single-GPU fallback when distributed launch is
        # not stable in the current environment.
        reports['multigpu'] = run_multigpu_batchsize(args)
        write_summary(args, reports)
        return
    if args.mode == 'smoke':
        reports['smoke'] = run_smoke(args)
        write_summary(args, reports)
        return
    dispatch = {
        'ovd1': run_ovd1,
        'ovd2': run_ovd2,
        'ovd3': run_ovd3,
        'ovd4': run_ovd4,
        'ovd5': run_ovd5,
    }
    if args.exp == 'all':
        for key in ('ovd1', 'ovd2', 'ovd3', 'ovd4', 'ovd5'):
            reports[key] = dispatch[key](args)
            write_summary(args, reports)
    else:
        reports[args.exp] = dispatch[args.exp](args)
        write_summary(args, reports)


if __name__ == '__main__':
    main()
