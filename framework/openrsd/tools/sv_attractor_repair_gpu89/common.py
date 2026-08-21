#!/usr/bin/env python
"""Shared utilities for SV attractor repair experiments."""
from __future__ import annotations

import csv
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.rotation_diagnostics import probe_rotated_stage_outputs as probe_base
from tools.verify_sv_attractor_gpu89 import common as verify

CLASSES = verify.CLASSES
SMALL = verify.SMALL
LARGE = verify.LARGE
SHIP = verify.SHIP
NUM_CLASSES = verify.NUM_CLASSES
DEFAULT_TILE = verify.DEFAULT_TILE
P0148_PREFIX = 'P0148__1024__651___0'

SS_TRAIN_ROOT = REPO_ROOT / 'data/DOTA1_1024_500/ss_train'
ANGLE_SWEEP_VAL = REPO_ROOT / 'data/DOTA1_1024_500/angle_sweep_val/realistic'

POSTPROCESS = dict(score_thr=0.01, nms_iou=0.5, max_per_img=1000)
SMOKE_ANGLES = [0, 90]
FULL_ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

R1_GRID_BIAS = [-1.0, -0.8, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0]
R1_GRID_BIAS_FULL = [-1.2, -1.0, -0.8, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0]
R3_FIXED_ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]


@dataclass
class RepairContext:
    repo_root: Path
    config: Path
    checkpoint: Path
    support_pkl: Path
    verify_work_dir: Path
    work_dir: Path
    result_md_dir: Path
    gpu: int
    mode: str
    angles: List[int]
    tiles: List[str]
    train_max_images: int
    heldout_max_images: int
    eval_max_images: int
    iters: int
    batch_size: int
    lr: float
    force: bool = False
    resume: bool = False
    progress: Dict[str, Any] = field(default_factory=dict)

    def tables_dir(self) -> Path:
        d = self.work_dir / 'tables'
        d.mkdir(parents=True, exist_ok=True)
        return d

    def ckpt_dir(self) -> Path:
        d = self.work_dir / 'checkpoints'
        d.mkdir(parents=True, exist_ok=True)
        return d

    def figures_dir(self) -> Path:
        d = self.work_dir / 'figures'
        d.mkdir(parents=True, exist_ok=True)
        return d

    def log_dir(self) -> Path:
        d = self.work_dir / 'logs'
        d.mkdir(parents=True, exist_ok=True)
        return d

    def splits_dir(self) -> Path:
        d = self.work_dir / 'splits'
        d.mkdir(parents=True, exist_ok=True)
        return d

    def fres(self, name: str) -> Path:
        self.result_md_dir.mkdir(parents=True, exist_ok=True)
        return self.result_md_dir / name

    def load_progress(self):
        p = self.work_dir / 'progress.json'
        if p.exists():
            self.progress = json.loads(p.read_text())

    def save_progress(self):
        (self.work_dir / 'progress.json').write_text(
            json.dumps(self.progress, indent=2, ensure_ascii=False))

    def mark(self, step: str, status: str, **kw):
        self.progress.setdefault('steps', {})[step] = dict(
            status=status, updated_at=time.strftime('%Y-%m-%d %H:%M:%S'), **kw)
        self.save_progress()

    def step_ok(self, step: str) -> bool:
        return self.progress.get('steps', {}).get(step, {}).get('status') == 'OK'

    def blocked(self) -> bool:
        return bool(self.progress.get('blocked'))

    def set_blocked(self, reason: str):
        self.progress['blocked'] = True
        self.progress['blocked_reason'] = reason
        self.save_progress()


def write_csv(path: Path, rows: List[dict], fields: Optional[Sequence[str]] = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fields and rows:
        fields = list(rows[0].keys())
    fields = fields or []
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> List[dict]:
    if not Path(path).exists():
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def mean_key(rows: List[dict], key: str, default=float('nan')) -> float:
    vals = []
    for r in rows:
        try:
            vals.append(float(r[key]))
        except Exception:
            pass
    return float(np.mean(vals)) if vals else default


def git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    except Exception:
        return 'unknown'


def package_versions() -> Dict[str, str]:
    out = {'python': sys.version.split()[0]}
    for name in ['torch', 'mmcv', 'mmengine', 'mmdet', 'mmrotate']:
        try:
            m = __import__(name)
            out[name] = getattr(m, '__version__', '?')
        except Exception:
            out[name] = 'N/A'
    out['cuda'] = torch.version.cuda if torch.cuda.is_available() else 'N/A'
    return out


def list_ss_train_stems(root: Path) -> List[str]:
    stems = set()
    for sub in ['images', '.']:
        d = root / sub if sub != '.' else root
        if not d.is_dir():
            continue
        for p in d.glob('*.png'):
            stems.add(p.stem)
        for p in d.glob('*.jpg'):
            stems.add(p.stem)
    return sorted(stems)


def build_splits(ctx: RepairContext) -> Dict[str, Any]:
    split_path = ctx.splits_dir() / 'calib_split.json'
    all_stems = [s for s in list_ss_train_stems(SS_TRAIN_ROOT)
                 if not s.startswith(P0148_PREFIX)]
    rng = random.Random(20260519)
    rng.shuffle(all_stems)
    n = len(all_stems)
    n_held = max(1, int(n * 0.2))
    heldout_pool = sorted(all_stems[:n_held])
    train_pool = sorted(all_stems[n_held:])

    if split_path.exists():
        data = json.loads(split_path.read_text())
        data.setdefault('train_pool', train_pool)
        data.setdefault('heldout_pool', heldout_pool)
    else:
        data = dict(train_pool=train_pool, heldout_pool=heldout_pool)

    n_train = ctx.train_max_images if ctx.train_max_images else len(train_pool)
    n_held = ctx.heldout_max_images if ctx.heldout_max_images else len(heldout_pool)
    data['train_calib'] = train_pool[:n_train]
    data['heldout_calib'] = heldout_pool[:n_held]
    data['p0148_excluded'] = P0148_PREFIX
    data['n_train'] = len(data['train_calib'])
    data['n_heldout'] = len(data['heldout_calib'])
    data['cross_tiles'] = discover_cross_tiles(ctx, 25)
    data['eval_angles'] = ctx.angles
    data['postprocess'] = POSTPROCESS
    split_path.write_text(json.dumps(data, indent=2))
    return data


def fres_for_mode(ctx: RepairContext, smoke_name: str, full_name: str) -> Path:
    return ctx.fres(full_name if ctx.mode == 'full' else smoke_name)


def discover_cross_tiles(ctx: RepairContext, n: int) -> List[str]:
    verify_csv = ctx.verify_work_dir / 'exp_06_tile_angle/ftable_tile_angle_visual.csv'
    if verify_csv.exists():
        rows = read_csv(verify_csv)
        tiles = sorted({r['tile_id'] for r in rows if r.get('tile_id') != DEFAULT_TILE})
        if tiles:
            return tiles[:n]
    return verify.discover_cross_tiles(ctx.repo_root, n)


def image_path_for_stem(stem: str) -> Optional[Path]:
    for ext in ['.png', '.jpg']:
        for base in [SS_TRAIN_ROOT / 'images', SS_TRAIN_ROOT]:
            p = base / f'{stem}{ext}'
            if p.exists():
                return p
    return None


def parse_class_histogram(h) -> Dict[str, int]:
    if isinstance(h, dict):
        return h
    if isinstance(h, str) and h.strip():
        return json.loads(h)
    return {}


def parse_dota_txt(ann_path: Path) -> List[dict]:
    boxes = []
    if not ann_path.exists():
        return boxes
    for line in ann_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        poly = list(map(float, parts[:8]))
        cls = parts[8]
        boxes.append(dict(class_name=cls, poly=poly))
    return boxes


def poly_to_xyxy(poly):
    pts = np.asarray(poly, dtype=float).reshape(-1, 2)
    return [float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 0].max())]


def build_model_ctx(ctx: RepairContext):
    bargs = SimpleNamespace(
        config=str(ctx.config),
        checkpoint=str(ctx.checkpoint),
        image_dir=str(SS_TRAIN_ROOT),
        out_dir=str(ctx.work_dir / '_tmp'),
        angles=ctx.angles,
        angle_step=None,
        score_thr=POSTPROCESS['score_thr'],
        iou_thr=POSTPROCESS['nms_iou'],
        support_shot=8,
        support_type='visual',
        support_feat=str(ctx.support_pkl),
        normalized_class_dict='data/normalized_class_dict.pkl',
        batch_size=1,
        num_workers=0,
        seed=20260519,
        device='cuda:0',
        max_spatial_size=32,
        result_md_dir=str(ctx.result_md_dir),
        class_names=CLASSES,
    )
    probe_base.setup_reproducibility(20260519)
    _, model, cfg = probe_base.build_runner_model(bargs, str(ctx.support_pkl))
    device = torch.device('cuda:0')
    model.to(device).eval()
    _, _, det_support, name2id, id2name = probe_base.prepare_support(bargs, device)
    from tools.sv_attractor_repair_gpu89.repair_forward import freeze_openrsd, count_trainable
    frozen = freeze_openrsd(model)
    return bargs, model, device, det_support, name2id, id2name, cfg, frozen


def write_fres_header(f, title: str, ctx: RepairContext, status: str = 'OK'):
    f.write(f'# {title}\n\n')
    f.write(f'- status: `{status}`\n')
    f.write(f'- CUDA_VISIBLE_DEVICES: `{os.environ.get("CUDA_VISIBLE_DEVICES", "")}`\n')
    f.write(f'- work_dir: `{ctx.work_dir}`\n')
    f.write(f'- mode: `{ctx.mode}`\n')
    f.write(f'- postprocess: score_thr={POSTPROCESS["score_thr"]}, '
            f'nms_iou={POSTPROCESS["nms_iou"]}, max_per_img={POSTPROCESS["max_per_img"]}\n')
    f.write(f'- support: `{ctx.support_pkl}`\n')
    f.write(f'- class_order: `{CLASSES}`\n\n')
