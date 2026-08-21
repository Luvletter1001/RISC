#!/usr/bin/env python
"""Shared utilities for small-vehicle attractor verification."""
from __future__ import annotations

import csv
import json
import os
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
from tools.rotation_overnight_gpu89 import common as overnight

CLASSES = probe_base.DOTA1_CLASSES
SMALL = CLASSES.index('small-vehicle')
LARGE = CLASSES.index('large-vehicle')
SHIP = CLASSES.index('ship')
NUM_CLASSES = len(CLASSES)
UNIFORM_BASELINE = 1.0 / NUM_CLASSES

DEFAULT_TILE = 'P0148__1024__651___0'
DEFAULT_CONFIG = (
    'M_configs/Step2_A10_Large_Pretrain_Stage3/A10_flex_rtm_v3_1_formal.py')
DEFAULT_CHECKPOINT = (
    'results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth')
DEFAULT_SUPPORT_FALLBACK = (
    'data/DOTA1_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl')

SMOKE_ANGLES = [0, 45, 90]
FULL_ANGLES_SPARSE = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
FULL_ANGLES_DENSE = list(range(0, 360, 5))

INTERVENTIONS_SMOKE = [
    'original', 'zero_sv', 'swap_sv_lv', 'random_sv_seed0', 'normalize_all']
INTERVENTIONS_FULL = [
    'original', 'zero_sv', 'random_sv_seed0', 'random_sv_seed1', 'random_sv_seed2',
    'swap_sv_lv', 'swap_sv_ship', 'norm_sv_to_class_mean', 'normalize_all',
    'class_centering_all', 'remove_sv_proj_0.25', 'remove_sv_proj_0.5',
    'remove_sv_proj_1.0']


def parse_angles(text: str, mode: str) -> List[int]:
    if text.strip():
        return sorted({int(a.strip()) % 360 for a in text.split(',') if a.strip()})
    if mode == 'full':
        return FULL_ANGLES_SPARSE
    return SMOKE_ANGLES


def parse_tiles(text: str, repo_root: Path) -> List[str]:
    if not text.strip():
        return [DEFAULT_TILE]
    p = Path(text)
    if p.is_file():
        return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    return [t.strip() for t in text.split(',') if t.strip()]


def tile_image_dir(repo_root: Path, tile_id: str) -> Path:
    cand = repo_root / 'vis' / tile_id / 'dataset' / 'images'
    if cand.is_dir():
        return cand
    raise FileNotFoundError(f'tile image dir not found: {cand}')


def tile_ann_dir(repo_root: Path, tile_id: str) -> Path:
    return repo_root / 'vis' / tile_id / 'dataset' / 'annfiles'


def resolve_support_from_config(config_path: Path) -> tuple[Optional[Path], str]:
    try:
        from mmengine.config import Config
        cfg = Config.fromfile(str(config_path))
        sdict = cfg.model.get('support_feat_dict', {})
        dota1 = sdict.get('Data1_DOTA1', '')
        p = (config_path.parent.parent.parent / dota1).resolve()
        if not p.is_absolute():
            p = (REPO_ROOT / dota1).resolve()
        if Path(dota1).is_absolute():
            p = Path(dota1)
        if p.exists():
            return p, 'config:Data1_DOTA1'
    except Exception as exc:
        return None, f'config_parse_failed:{exc}'
    fb = REPO_ROOT / DEFAULT_SUPPORT_FALLBACK
    if fb.exists():
        return fb, 'fallback'
    return None, 'missing'


def write_csv(path: Path, rows: List[dict], fields: Optional[Sequence[str]] = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = []
    if fields is None and rows:
        fields = list(rows[0].keys())
    elif fields is None:
        fields = []
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> List[dict]:
    path = Path(path)
    if not path.exists():
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


def bbox_row(pred, idx: int):
    """Extract one rotated box as numpy/list from mmrotate RotatedBoxes."""
    bboxes = pred.bboxes
    if hasattr(bboxes, 'tensor'):
        row = bboxes.tensor[idx].detach().cpu().numpy()
    else:
        row = bboxes[idx].detach().cpu().numpy()
    return row


def bbox_row_list(pred, idx: int) -> list:
    return bbox_row(pred, idx).tolist()


def class_entropy_from_counts(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            ent -= p * np.log(p)
    return float(ent)


@dataclass
class RunContext:
    repo_root: Path
    config: Path
    checkpoint: Path
    support_pkl: Optional[Path]
    work_dir: Path
    result_md_dir: Path
    gpu: int
    mode: str
    angles: List[int]
    tiles: List[str]
    score_thr: float = 0.01
    nms_iou: float = 0.5
    topk: int = 1000
    max_images: int = 0
    force: bool = False
    resume: bool = False
    progress: Dict[str, Any] = field(default_factory=dict)

    def exp_dir(self, name: str) -> Path:
        d = self.work_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def fres_path(self, name: str) -> Path:
        self.result_md_dir.mkdir(parents=True, exist_ok=True)
        return self.result_md_dir / name

    def log_dir(self) -> Path:
        d = self.work_dir / 'logs'
        d.mkdir(parents=True, exist_ok=True)
        return d

    def load_progress(self):
        p = self.work_dir / 'progress.json'
        if p.exists():
            self.progress = json.loads(p.read_text())

    def save_progress(self):
        p = self.work_dir / 'progress.json'
        p.write_text(json.dumps(self.progress, indent=2, ensure_ascii=False))

    def step_done(self, step: str) -> bool:
        return self.progress.get('steps', {}).get(step, {}).get('status') == 'OK'

    def mark_step(self, step: str, status: str, **extra):
        self.progress.setdefault('steps', {})[step] = dict(
            status=status, updated_at=time.strftime('%Y-%m-%d %H:%M:%S'), **extra)
        self.save_progress()

    def blocked(self) -> bool:
        return self.progress.get('blocked', False)

    def set_blocked(self, reason: str):
        self.progress['blocked'] = True
        self.progress['blocked_reason'] = reason
        self.save_progress()


def make_base_args(ctx: RunContext, image_dir: Path, out_dir: Path,
                   support_type: str = 'visual') -> SimpleNamespace:
    sp = str(ctx.support_pkl) if ctx.support_pkl else ''
    return SimpleNamespace(
        config=str(ctx.config),
        checkpoint=str(ctx.checkpoint),
        image_dir=str(image_dir),
        out_dir=str(out_dir),
        angles=ctx.angles,
        angle_step=None,
        score_thr=ctx.score_thr,
        iou_thr=ctx.nms_iou,
        support_shot=8,
        support_type=support_type,
        support_feat=sp,
        normalized_class_dict='data/normalized_class_dict.pkl',
        batch_size=1,
        num_workers=0,
        seed=20260519,
        device='cuda:0',
        max_spatial_size=32,
        result_md_dir=str(ctx.result_md_dir))


def build_model(ctx: RunContext, support_type: str = 'visual'):
    bargs = make_base_args(ctx, ctx.exp_dir('_tmp'), support_type)
    probe_base.setup_reproducibility(bargs.seed)
    sp = str(ctx.support_pkl) if ctx.support_pkl else ''
    if not sp:
        sp, _ = probe_base.resolve_support_path('')
    _, model, cfg = probe_base.build_runner_model(bargs, sp)
    device = torch.device('cuda:0')
    model.to(device).eval()
    _, _, det_support_data, name2id, id2name = probe_base.prepare_support(bargs, device)
    return bargs, model, device, det_support_data, name2id, id2name, cfg


def build_loader(ctx: RunContext, image_dir: Path, angles: Optional[List[int]] = None):
    bargs = make_base_args(ctx, image_dir, ctx.exp_dir('_tmp'))
    if angles is not None:
        bargs.angles = angles
    return probe_base.build_dataloader(bargs, bargs.angles)


def apply_intervention(
        support_feats: torch.Tensor,
        support_labels: torch.Tensor,
        intervention: str,
        device: torch.device) -> torch.Tensor:
    feats = support_feats.clone()
    labels = support_labels.clone()

    def small_mask():
        return labels == SMALL

    if intervention == 'original':
        pass
    elif intervention == 'zero_sv':
        feats[small_mask()] = 0
    elif intervention.startswith('random_sv_seed'):
        seed = int(intervention.split('seed')[-1]) if 'seed' in intervention else 0
        gen = torch.Generator(device=device)
        gen.manual_seed(20260519 + seed)
        m = small_mask()
        repl = torch.randn(feats[m].shape, generator=gen, device=device, dtype=feats.dtype)
        tn = feats[m].norm(dim=1, keepdim=True).mean().clamp_min(1e-12)
        repl = repl / repl.norm(dim=1, keepdim=True).clamp_min(1e-12) * tn
        feats[m] = repl
    elif intervention == 'swap_sv_lv':
        sm, lm = small_mask(), labels == LARGE
        tmp = feats[sm].clone()
        feats[sm] = feats[lm][:tmp.shape[0]]
        feats[lm] = tmp[:feats[lm].shape[0]]
    elif intervention == 'swap_sv_ship':
        sm, shm = small_mask(), labels == SHIP
        tmp = feats[sm].clone()
        feats[sm] = feats[shm][:tmp.shape[0]]
        feats[shm] = tmp[:feats[shm].shape[0]]
    elif intervention == 'norm_sv_to_class_mean':
        mn = feats.norm(dim=1).mean().clamp_min(1e-12)
        m = small_mask()
        feats[m] = feats[m] / feats[m].norm(dim=1, keepdim=True).clamp_min(1e-12) * mn
    elif intervention == 'normalize_all':
        mn = feats.norm(dim=1).mean().clamp_min(1e-12)
        feats = feats / feats.norm(dim=1, keepdim=True).clamp_min(1e-12) * mn
    elif intervention == 'class_centering_all':
        feats = feats - feats.mean(dim=0, keepdim=True)
    elif intervention.startswith('remove_sv_proj_'):
        alpha = float(intervention.split('_')[-1])
        sv = feats[small_mask()].mean(dim=0)
        sv = sv / sv.norm().clamp_min(1e-12)
        proj = (feats @ sv).unsqueeze(1) * sv.unsqueeze(0)
        feats = feats - alpha * proj
    else:
        raise ValueError(f'unknown intervention: {intervention}')
    return feats


def build_mapped_support(
        model, ctx: RunContext, det_support_data, name2id, device,
        support_type: str, intervention: str):
    embed_name = 'visual_embeds' if support_type == 'visual' else 'text_embeds'
    mapping = (model.visual_support_mapping if support_type == 'visual'
               else model.text_support_mapping)
    feats, labels = [], []
    for cls_name, info in det_support_data.items():
        cid = name2id[cls_name]
        arr = np.asarray(info[embed_name])[:8]
        feats.append(arr)
        labels.append(np.ones(len(arr)) * cid)
    raw = torch.tensor(np.concatenate(feats), device=device).float()
    lab = torch.tensor(np.concatenate(labels), device=device).long()
    order = torch.argsort(lab)
    mapped = mapping(raw[order])
    lab = lab[order]
    mapped = apply_intervention(mapped, lab, intervention, device)
    return mapped.unsqueeze(0), lab.unsqueeze(0)


def forward_dense(model, x, sf, sl, bargs):
    outs_all = model.bbox_head(
        x, sf, sl, sl,
        support_shot=bargs.support_shot,
        num_classes=NUM_CLASSES,
        num_in_classes=NUM_CLASSES,
        align_style='labelled',
        support_type=bargs.support_type,
        text_cls_scale=0.0)
    return outs_all[:-2]


decode_dense = overnight.decode_dense


def dense_stats_from_levels(levels) -> Dict[str, Any]:
    """Aggregate dense stats across FPN levels."""
    rows = []
    all_scores = []
    for lvl, logits, scores, boxes in levels:
        top1 = scores.argmax(dim=1)
        sv = scores[:, SMALL]
        lv = scores[:, LARGE]
        margin = scores.topk(2, dim=1).values
        m = (margin[:, 0] - margin[:, 1]) if margin.shape[1] > 1 else margin[:, 0]
        rows.append(dict(
            level=lvl,
            num_locations=int(scores.shape[0]),
            dense_top1_sv_ratio=float((top1 == SMALL).float().mean().item()),
            mean_sv_score=float(sv.mean().item()),
            mean_lv_score=float(lv.mean().item()),
            mean_margin=float(m.mean().item()),
        ))
        all_scores.append(scores)
    if not all_scores:
        return dict(rows=[], scores=None, boxes=None)
    cat = torch.cat(all_scores)
    top1 = cat.argmax(dim=1)
    return dict(
        rows=rows,
        scores=cat,
        dense_top1_sv_ratio=float((top1 == SMALL).float().mean().item()),
        dense_top1_lv_ratio=float((top1 == LARGE).float().mean().item()),
        dense_top1_ship_ratio=float((top1 == SHIP).float().mean().item()),
        mean_sv_score=float(cat[:, SMALL].mean().item()),
        mean_margin=float(
            (cat.topk(2, dim=1).values[:, 0] - cat.topk(2, dim=1).values[:, 1]).mean().item()),
    )


def final_detection_stats(head, outs, metas) -> dict:
    row = overnight.post_summary(0, 'original', head, outs, metas)
    hist = json.loads(row['class_histogram'])
    ship = hist.get('ship', 0)
    total = row['detection_total']
    return dict(
        detection_total=total,
        small_vehicle_count=row['small_vehicle_count'],
        small_vehicle_ratio=row['small_vehicle_ratio'],
        large_vehicle_ratio=row['large_vehicle_ratio'],
        ship_ratio=float(ship / total) if total else 0.0,
        mean_score=row['mean_score'],
        max_score=row['max_score'],
        top1_class=row['top1_class'],
        class_histogram=hist,
        class_entropy=class_entropy_from_counts(hist),
    )


def discover_cross_tiles(repo_root: Path, max_tiles: int = 20) -> List[str]:
    tiles = []
    vis = repo_root / 'vis'
    if not vis.is_dir():
        return tiles
    for dataset_dir in sorted(vis.glob('*/dataset')):
        tid = dataset_dir.parent.name
        if tid == DEFAULT_TILE:
            continue
        img = dataset_dir / 'images'
        ann = dataset_dir / 'annfiles'
        if img.is_dir() and ann.is_dir() and list(img.glob('*_rot000.jpg')):
            tiles.append(tid)
        if len(tiles) >= max_tiles:
            break
    return tiles


def write_fres_header(f, title: str, ctx: RunContext, status: str = 'OK'):
    f.write(f'# {title}\n\n')
    f.write(f'- status: `{status}`\n')
    f.write(f'- CUDA_VISIBLE_DEVICES: `{os.environ.get("CUDA_VISIBLE_DEVICES", "")}`\n')
    f.write(f'- work_dir: `{ctx.work_dir}`\n')
    f.write(f'- config: `{ctx.config}`\n')
    f.write(f'- checkpoint: `{ctx.checkpoint}`\n')
    f.write(f'- support_pkl: `{ctx.support_pkl}`\n')
    f.write(f'- mode: `{ctx.mode}`\n')
    f.write(f'- angles: `{ctx.angles}`\n')
    f.write(f'- tiles: `{ctx.tiles}`\n\n')


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=repo_root, text=True).strip()
    except Exception:
        return 'unknown'


def package_versions() -> Dict[str, str]:
    out = {}
    for name in ['torch', 'mmcv', 'mmengine', 'mmdet', 'mmrotate']:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, '__version__', 'unknown')
        except Exception:
            out[name] = 'not_installed'
    out['python'] = sys.version.split()[0]
    out['cuda'] = torch.version.cuda if torch.cuda.is_available() else 'N/A'
    return out
