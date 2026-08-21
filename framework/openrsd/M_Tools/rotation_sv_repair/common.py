#!/usr/bin/env python3
"""Shared context, paths, and utilities for rotation SV repair suite."""
from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / 'tools') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'tools'))

# Lazy-import torch/mm stack only when running GPU inference (see build_model_ctx).
verify = None
probe_base = None


def _ensure_verify():
    global verify, probe_base
    if verify is None:
        from tools.rotation_diagnostics import probe_rotated_stage_outputs as _probe
        from tools.verify_sv_attractor_gpu89 import common as _verify
        probe_base = _probe
        verify = _verify
    return verify, probe_base

def _dota_classes():
    _ensure_verify()
    return list(probe_base.DOTA1_CLASSES)


CLASSES = [
    'baseball-diamond', 'basketball-court', 'bridge', 'ground-track-field',
    'harbor', 'helicopter', 'large-vehicle', 'plane', 'roundabout', 'ship',
    'small-vehicle', 'soccer-ball-field', 'storage-tank', 'swimming-pool', 'tennis-court',
]
SMALL = CLASSES.index('small-vehicle')
TENNIS = CLASSES.index('tennis-court')
BASEBALL = CLASSES.index('baseball-diamond')
SOCCER = CLASSES.index('soccer-ball-field')
COURT_CLASSES = (TENNIS, BASEBALL, SOCCER)
COURT_NAMES = ('tennis-court', 'baseball-diamond', 'soccer-ball-field')
LOW_RISK_NAMES = ('harbor', 'bridge', 'roundabout', 'storage-tank')
LOW_RISK_IDX = tuple(CLASSES.index(n) for n in LOW_RISK_NAMES)
NUM_CLASSES = len(CLASSES)

P0148 = 'P0148__1024__651___0'
HIGHRISK_TILES = [
    'P0682__1024__553___0', P0148, 'P0297__1024__0___0',
    'P1461__1024__3296___4944', 'P1447__1024__0___1873', 'P1854__1024__7336___1048',
]
LOWRISK_TILES = [
    'P0132__1024__0___0', 'P0097__1024__0___208', 'P1442__1024__0___824',
    'P0312__1024__2383___1756', 'P0182__1024__824___824',
]

ANGLES_FULL = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
ANGLES_SMOKE = [0, 30, 60, 90]
ANGLE_STR_FULL = [f'{a:03d}' for a in ANGLES_FULL]
ANGLE_STR_SMOKE = [f'{a:03d}' for a in ANGLES_SMOKE]

DOTA_SWEEP = REPO_ROOT / 'data/DOTA1_1024_500/angle_sweep_val/realistic'
SS_TRAIN = REPO_ROOT / 'data/DOTA1_1024_500/ss_train'
DEFAULT_CONFIG = REPO_ROOT / 'M_configs/Step2_A10_Large_Pretrain_Stage3/A10_flex_rtm_v3_1_formal.py'
DEFAULT_CHECKPOINT = REPO_ROOT / 'results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth'
DEFAULT_SUPPORT = REPO_ROOT / 'data/DOTA1_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl'

POSTPROCESS = dict(score_thr=0.01, nms_iou=0.5, max_per_img=1000)

VLM_SEARCH_ROOTS = [
    REPO_ROOT, REPO_ROOT / 'results', REPO_ROOT / 'pretrained', REPO_ROOT / 'checkpoints',
    Path('/data1/zcy'), Path('/data/zcy'),
]


@dataclass
class SuiteContext:
    repo_root: Path
    work_dir: Path
    result_dir: Path
    gpu_ids: str
    mode: str
    exp: str
    teacher: str
    student: str
    pseudo_label_mode: str
    vocab_mode: str
    teacher_vlm: str
    only_model: str = ''
    only_checkpoint: str = ''
    only_config: str = ''
    only_angle: str = ''
    only_tile: str = ''
    max_images: int = 0
    force: bool = False
    resume: bool = True
    no_train: bool = False
    train_adapter: bool = False
    distill: bool = False
    debug_small: bool = False
    batch_size: int = 4
    num_workers: int = 2
    progress: Dict[str, Any] = field(default_factory=dict)

    @property
    def tables_dir(self) -> Path:
        d = self.result_dir / 'tables'
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def figures_dir(self) -> Path:
        d = self.result_dir / 'figures'
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def human_audit_dir(self) -> Path:
        d = self.result_dir / 'human_audit'
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def logs_dir(self) -> Path:
        d = self.work_dir / 'logs'
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def adapter_ckpt_dir(self) -> Path:
        d = self.work_dir / 'adapter_ckpts'
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def pseudo_label_dir(self) -> Path:
        d = self.work_dir / 'pseudo_labels'
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def angles(self) -> List[int]:
        if self.only_angle:
            return [int(a.strip()) % 360 for a in self.only_angle.split(',') if a.strip()]
        if self.mode in ('dryrun', 'debug'):
            return ANGLES_SMOKE[:2]
        if self.mode == 'smoke' or self.debug_small:
            return ANGLES_SMOKE
        return ANGLES_FULL

    def fres_path(self, name: str) -> Path:
        self.result_dir.mkdir(parents=True, exist_ok=True)
        return self.result_dir / name

    def load_progress(self):
        p = self.work_dir / 'progress.json'
        if p.exists():
            self.progress = json.loads(p.read_text(encoding='utf-8'))

    def save_progress(self):
        self.work_dir.mkdir(parents=True, exist_ok=True)
        (self.work_dir / 'progress.json').write_text(
            json.dumps(self.progress, indent=2, ensure_ascii=False), encoding='utf-8')

    def mark_step(self, step: str, status: str, **kw):
        self.progress.setdefault('steps', {})[step] = dict(
            status=status, updated_at=datetime.now().isoformat(timespec='seconds'), **kw)
        self.save_progress()

    def step_done(self, step: str) -> bool:
        if self.force:
            return False
        if not self.resume:
            return False
        return self.progress.get('steps', {}).get(step, {}).get('status') == 'DONE'

    def should_skip_output(self, path: Path) -> bool:
        return self.resume and not self.force and path.exists() and path.stat().st_size > 0


def setup_env(gpu_ids: str):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    os.environ.setdefault('PYTHONNOUSERSITE', '1')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/mplconfig')
    try:
        import torch
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def now_iso() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def write_csv(path: Path, rows: List[dict], fields: Optional[Sequence[str]] = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fields and rows:
        fields = list(rows[0].keys())
    fields = list(fields or [])
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})


def read_csv(path: Path) -> List[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def mean_key(rows: List[dict], key: str, default=float('nan')) -> float:
    vals = []
    for r in rows:
        try:
            vals.append(float(r[key]))
        except (TypeError, ValueError, KeyError):
            pass
    return float(np.mean(vals)) if vals else default


def git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return 'unknown'


def write_fres_header(f, title: str, ctx: SuiteContext, status: str = 'DONE'):
    f.write(f'# {title}\n\n')
    f.write(f'- **status:** `{status}`\n')
    f.write(f'- **generated:** {now_iso()}\n')
    f.write(f'- **mode:** `{ctx.mode}`\n')
    f.write(f'- **CUDA_VISIBLE_DEVICES:** `{os.environ.get("CUDA_VISIBLE_DEVICES", "")}`\n')
    f.write(f'- **batch_size:** `{ctx.batch_size}`\n')
    f.write(f'- **num_workers:** `{ctx.num_workers}`\n')
    f.write(f'- **work_dir:** `{ctx.work_dir}`\n')
    f.write(f'- **result_dir:** `{ctx.result_dir}`\n')
    f.write(f'- **small_vehicle index:** `{SMALL}` (`{CLASSES[SMALL]}`)\n')
    f.write(f'- **class_order:** {CLASSES}\n\n')


def write_failed_fres(ctx: SuiteContext, fres_name: str, title: str, exc: BaseException):
    path = ctx.fres_path(fres_name)
    tb = traceback.format_exc()
    with open(path, 'w', encoding='utf-8') as f:
        write_fres_header(f, title, ctx, status='FAILED')
        f.write('## Error\n\n```\n')
        f.write(tb[-8000:])
        f.write('\n```\n')
    log_path = ctx.logs_dir / f'failed_{fres_name.replace(".md", "")}.log'
    log_path.write_text(tb, encoding='utf-8')
    return path


def parse_class_histogram(h) -> Dict[str, int]:
    if isinstance(h, dict):
        return h
    if isinstance(h, str) and h.strip():
        try:
            return json.loads(h)
        except json.JSONDecodeError:
            return {}
    return {}


def discover_config_checkpoint(ctx: SuiteContext) -> tuple[Path, Path, Path]:
    config = Path(ctx.only_config) if ctx.only_config else DEFAULT_CONFIG
    if not config.is_absolute():
        config = ctx.repo_root / config
    ckpt = Path(ctx.only_checkpoint) if ctx.only_checkpoint else DEFAULT_CHECKPOINT
    if not ckpt.is_absolute():
        ckpt = ctx.repo_root / ckpt
    support = DEFAULT_SUPPORT
    if not support.exists():
        fb = list((ctx.repo_root / 'data').rglob('Step5_3_Prepare_Visual_Text_DINOv2_support.pkl'))
        if fb:
            support = fb[0]
    return config, ckpt, support


def list_angle_sweep_stems(angle: int = 0, limit: int = 0) -> List[str]:
    d = DOTA_SWEEP / f'angle_{angle:03d}' / 'images'
    if not d.is_dir():
        return []
    stems = sorted(p.stem for p in d.glob('*.png'))
    stems += sorted(p.stem for p in d.glob('*.jpg'))
    stems = sorted(set(stems))
    if limit > 0:
        return stems[:limit]
    return stems


def list_ss_train_stems(limit: int = 0) -> List[str]:
    stems = set()
    for sub in ('images', '.'):
        d = SS_TRAIN / sub if sub != '.' else SS_TRAIN
        if not d.is_dir():
            continue
        for p in d.glob('*.png'):
            stems.add(p.stem)
        for p in d.glob('*.jpg'):
            stems.add(p.stem)
    out = sorted(stems)
    if limit > 0:
        return out[:limit]
    return out


def tile_has_class(tile_id: str, class_names: Sequence[str]) -> bool:
    ann = SS_TRAIN / 'annfiles' / f'{tile_id}.txt'
    if not ann.exists():
        ann = ctx_ann_vis(tile_id)
    if ann is None or not ann.exists():
        return False
    text = ann.read_text(encoding='utf-8', errors='replace').lower()
    return any(c.replace('-', ' ').replace('_', ' ') in text or c in text for c in class_names)


def ctx_ann_vis(tile_id: str) -> Optional[Path]:
    p = REPO_ROOT / 'vis' / tile_id / 'dataset' / 'annfiles' / f'{tile_id}.txt'
    return p if p.exists() else None


def build_model_ctx(ctx: SuiteContext):
    _ensure_verify()
    config, ckpt, support = discover_config_checkpoint(ctx)
    bargs = SimpleNamespace(
        config=str(config),
        checkpoint=str(ckpt),
        image_dir=str(SS_TRAIN),
        out_dir=str(ctx.work_dir / '_tmp_infer'),
        angles=ctx.angles,
        angle_step=None,
        score_thr=POSTPROCESS['score_thr'],
        iou_thr=POSTPROCESS['nms_iou'],
        support_shot=8,
        support_type='visual',
        support_feat=str(support),
        normalized_class_dict='data/normalized_class_dict.pkl',
        batch_size=max(1, int(ctx.batch_size)),
        num_workers=max(0, int(ctx.num_workers)),
        seed=20260524,
        device='cuda:0',
        max_spatial_size=32,
        result_md_dir=str(ctx.result_dir),
        class_names=CLASSES,
    )
    probe_base.setup_reproducibility(20260524)
    _, model, cfg = probe_base.build_runner_model(bargs, str(support))
    import torch
    device = torch.device('cuda:0')
    model.to(device).eval()
    _, _, det_support, name2id, id2name = probe_base.prepare_support(bargs, device)
    from tools.sv_attractor_repair_gpu89.repair_forward import freeze_openrsd
    frozen = freeze_openrsd(model)
    return bargs, model, device, det_support, name2id, id2name, cfg, frozen, config, ckpt, support


def run_logged(ctx: SuiteContext, name: str, fn, *args, **kwargs) -> dict:
    log_path = ctx.logs_dir / f'{name}_{ctx.mode}.log'
    t0 = time.time()
    try:
        with open(log_path, 'w', encoding='utf-8') as logf:
            logf.write(f'# {name} started {now_iso()}\n')
            logf.flush()
            result = fn(ctx, *args, **kwargs)
        result.setdefault('log_path', str(log_path))
        result['duration_sec'] = round(time.time() - t0, 2)
        return result
    except Exception as exc:
        tb = traceback.format_exc()
        log_path.write_text(tb, encoding='utf-8')
        return dict(status='FAILED', error=str(exc), log_path=str(log_path),
                    traceback=tb[-4000:], duration_sec=round(time.time() - t0, 2))
