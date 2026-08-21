#!/usr/bin/env python
"""Shared utilities for SV attractor mechanism study."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.rotation_diagnostics import probe_rotated_stage_outputs as probe_base
from tools.rotation_overnight_gpu89 import common as oc
from tools.sv_attractor_repair_gpu89.exp_combined import load_repair_stack
from tools.sv_attractor_repair_gpu89.exp_r3 import apply_fixed_alpha_support
from tools.sv_attractor_repair_gpu89.repair_forward import (
    build_support_tensors,
    dense_stats_from_logits,
    final_detection_from_outs,
    sv_support_direction,
)
from tools.sv_attractor_repair_gpu89.repair_logits import apply_repair_logits
from tools.sv_attractor_repair_gpu89.repair_modules import ClassWiseCalibration, RepairStack
from tools.verify_sv_attractor_gpu89 import common as verify

CLASSES = verify.CLASSES
SMALL = verify.SMALL
NUM_CLASSES = verify.NUM_CLASSES
DEFAULT_TILE = verify.DEFAULT_TILE
P0148_PREFIX = 'P0148__1024__651___0'

RESULT_MD = REPO_ROOT / 'resultmd/exp_mechanism_sv_attractor_gpu89'
WORK_ROOT = REPO_ROOT / 'work_dirs/exp_mechanism_sv_attractor_gpu89'
COUNTERFACTUAL_ROOT = WORK_ROOT / 'counterfactual_images'
ANGLE_SWEEP_CACHE = WORK_ROOT / 'angle_sweep_cache'

ATLAS_ANGLES = [0, 30, 60, 90, 180, 270]
ATLAS_ANGLES_EXT = list(range(0, 360, 30))
DECOMP_ANGLES = [0, 90, 180, 270]

REPAIR_WD = REPO_ROOT / 'work_dirs/sv_attractor_repair_gpu89_20260519_161938'
R1_CKPT = REPAIR_WD / 'checkpoints/R1C_train_bias_temperature_full_full_best.pth'
R3_CKPT = REPAIR_WD / 'checkpoints/R3B_levelwise_alpha_full_best.pth'

SCHEMA_ATLAS = [
    'run_id', 'tile_id', 'angle', 'method', 'det_count', 'small_vehicle_count',
    'final_sv_ratio', 'dense_top1_sv_ratio', 'dense_sv_mean_logit',
    'dense_sv_margin_vs_runnerup', 'dense_entropy', 'objectness_mean', 'objectness_p95',
    'mean_score', 'top1_class_histogram', 'gt_small_vehicle', 'gt_sv_count',
    'ap50', 'sv_ap50', 'background_to_sv_cosine_mean', 'background_to_sv_cosine_p95',
    'config_path', 'checkpoint_path', 'git_commit', 'seed', 'timestamp', 'notes',
]

log = logging.getLogger('mechanism_sv')


def setup_logging(work_dir: Path, name: str):
    (work_dir / 'logs').mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(work_dir / 'logs' / f'log_{name}.log'),
            logging.StreamHandler(),
        ],
    )


def append_csv(path: Path, row: dict, fields: Sequence[str]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction='ignore')
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, 'NA') for k in fields})


def read_csv(path: Path) -> List[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def row_key(tile_id: str, angle: int, method: str, extra: str = '') -> str:
    return f'{tile_id}|{angle}|{method}|{extra}'


def already_done(path: Path, tile_id: str, angle: int, method: str, extra: str = '') -> bool:
    if not path.exists():
        return False
    key = row_key(tile_id, angle, method, extra)
    for r in read_csv(path):
        if row_key(r.get('tile_id', ''), int(float(r.get('angle', -1))), r.get('method', ''),
                   r.get('condition', r.get('intervention', ''))) == key:
            try:
                float(r.get('dense_top1_sv_ratio', 'NA'))
                return True
            except (TypeError, ValueError):
                return False
    return False


@dataclass
class MechContext:
    repo_root: Path = REPO_ROOT
    config: Path = field(default_factory=lambda: REPO_ROOT / verify.DEFAULT_CONFIG)
    checkpoint: Path = field(default_factory=lambda: REPO_ROOT / verify.DEFAULT_CHECKPOINT)
    support_pkl: Optional[Path] = None
    work_dir: Path = WORK_ROOT
    result_md_dir: Path = RESULT_MD
    repair_work_dir: Path = REPAIR_WD
    gpu: int = 8
    seed: int = 20260520
    score_thr: float = 0.01
    nms_iou: float = 0.5
    progress: Dict[str, Any] = field(default_factory=dict)

    def load_progress(self):
        p = self.work_dir / 'progress.json'
        if p.exists():
            self.progress = json.loads(p.read_text())

    def save_progress(self):
        (self.work_dir / 'progress.json').write_text(
            json.dumps(self.progress, indent=2, ensure_ascii=False))

    def repair_progress(self) -> dict:
        p = self.repair_work_dir / 'progress.json'
        return json.loads(p.read_text()) if p.exists() else {}


def vis_tile_dir(repo: Path, tile_id: str) -> Optional[Path]:
    d = repo / 'vis' / tile_id / 'dataset' / 'images'
    return d if d.is_dir() else None


def angle_sweep_dir(repo: Path, angle: int) -> Path:
    return repo / 'data/DOTA1_1024_500/angle_sweep_val/realistic' / f'angle_{angle:03d}'


def resolve_tile_source(repo: Path, tile_id: str, angle: int) -> Tuple[Path, str]:
    """Return (dataset_root with images+annfiles), source_tag."""
    vd = vis_tile_dir(repo, tile_id)
    if vd is not None:
        img = vd / f'{tile_id}_rot{angle:03d}.jpg'
        if not img.exists():
            img = vd / f'{tile_id}_rot{angle:03d}.png'
        if img.exists():
            return vd.parent, 'vis_rotated'
    asv = angle_sweep_dir(repo, angle)
    for ext in ('.png', '.jpg'):
        if (asv / 'images' / f'{tile_id}{ext}').exists():
            return asv, 'angle_sweep'
    if vd is not None:
        return vd.parent, 'vis_partial'
    raise FileNotFoundError(f'no image for {tile_id} angle={angle}')


def txt_ann_to_pkl_dict(txt_path: Path) -> dict:
    from tools.sv_attractor_repair_gpu89.common import parse_dota_txt
    boxes = parse_dota_txt(txt_path)
    texts = [b['class_name'] for b in boxes]
    polys = [np.asarray(b['poly'], dtype=np.float32) for b in boxes]
    if polys:
        polys_arr = np.stack(polys, axis=0)
    else:
        polys_arr = np.zeros((0, 8), dtype=np.float32)
    return dict(texts=texts, polys=polys_arr, text_embeds=None, visual_embeds=None, cls_list=None)


def prepare_angle_sweep_mini_dataset(repo: Path, tile_id: str, angle: int) -> Path:
    """Single-tile dataset root: DOTADatasetOnline requires annfiles/*.pkl."""
    from commonlibs.common_tools import pklsave
    asv = angle_sweep_dir(repo, angle)
    img_src = None
    for ext in ('.png', '.jpg'):
        p = asv / 'images' / f'{tile_id}{ext}'
        if p.exists():
            img_src = p
            break
    if img_src is None:
        raise FileNotFoundError(f'angle_sweep image missing: {tile_id} @ {angle}')
    root = ANGLE_SWEEP_CACHE / tile_id / f'angle_{angle:03d}' / 'dataset'
    img_dir = root / 'images'
    ann_dir = root / 'annfiles'
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    dst_img = img_dir / f'{tile_id}.png'
    if dst_img.exists() or dst_img.is_symlink():
        dst_img.unlink(missing_ok=True)
    dst_img.symlink_to(img_src.resolve())
    ann_pkl = ann_dir / f'{tile_id}.pkl'
    txt_ann = asv / 'annfiles' / f'{tile_id}.txt'
    empty_ann = dict(
        texts=[], polys=np.zeros((0, 8), dtype=np.float32),
        text_embeds=None, visual_embeds=None, cls_list=None)
    if txt_ann.exists():
        pklsave(txt_ann_to_pkl_dict(txt_ann), str(ann_pkl))
    elif not ann_pkl.exists():
        pklsave(empty_ann, str(ann_pkl))
    return root


def prepare_angle_sweep_batch_dataset(
    repo: Path, tile_ids: Sequence[str], angle: int, batch_id: str,
) -> Path:
    """Multi-tile dataset root for batched step1_inference (symlinks + ann pkls)."""
    from commonlibs.common_tools import pklsave

    asv = angle_sweep_dir(repo, angle)
    root = ANGLE_SWEEP_CACHE / 'batch' / batch_id / f'angle_{angle:03d}' / 'dataset'
    img_dir = root / 'images'
    ann_dir = root / 'annfiles'
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    empty_ann = dict(
        texts=[], polys=np.zeros((0, 8), dtype=np.float32),
        text_embeds=None, visual_embeds=None, cls_list=None)
    for tile_id in tile_ids:
        img_src = None
        for ext in ('.png', '.jpg'):
            p = asv / 'images' / f'{tile_id}{ext}'
            if p.exists():
                img_src = p
                break
        if img_src is None:
            raise FileNotFoundError(f'angle_sweep image missing: {tile_id} @ {angle}')
        dst_img = img_dir / f'{tile_id}.png'
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink(missing_ok=True)
        dst_img.symlink_to(img_src.resolve())
        ann_pkl = ann_dir / f'{tile_id}.pkl'
        txt_ann = asv / 'annfiles' / f'{tile_id}.txt'
        if txt_ann.exists():
            pklsave(txt_ann_to_pkl_dict(txt_ann), str(ann_pkl))
        elif not ann_pkl.exists():
            pklsave(empty_ann, str(ann_pkl))
    return root


def build_mini_dataloader(bargs, dataset_root: Path, filter_empty_gt: bool = False):
    """Dataloader for a mini dataset (1 image + 1 pkl)."""
    from functools import partial
    from mmengine.registry import DATA_SAMPLERS, DATASETS, FUNCTIONS
    from torch.utils.data import DataLoader
    from M_AD.datasets.transforms.loading import LoadAnnotationsOnline  # noqa: F401
    from M_AD.datasets.transforms.formatting import PackDetInputsMM  # noqa: F401
    from M_AD.datasets.transforms.transforms import ConvertBoxTypeSafe  # noqa: F401

    dataset_root = Path(dataset_root)
    test_pipeline = [
        dict(type='mmdet.LoadImageFromFile', file_client_args=dict(backend='disk')),
        dict(type='mmrotate.LoadAnnotationsOnline', with_bbox=True, box_type='qbox'),
        dict(type='mmrotate.ConvertBoxTypeSafe', box_type_mapping=dict(gt_bboxes='rbox')),
        dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
        dict(type='mmdet.Pad', size=(1024, 1024), pad_val=dict(img=(114, 114, 114))),
        dict(type='mmrotate.PackDetInputsMM',
             meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
    ]
    test_dataset = dict(
        type='mmrotate.DOTADatasetOnline',
        data_root=str(dataset_root),
        ann_file='annfiles',
        data_prefix=dict(img_path='images'),
        img_shape=(1024, 1024),
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=filter_empty_gt),
        pipeline=test_pipeline)
    dataset = DATASETS.build(test_dataset)
    sampler = DATA_SAMPLERS.build(
        dict(type='DefaultSampler', shuffle=False),
        default_args=dict(dataset=dataset, seed=2024))
    collate_fn = FUNCTIONS.get('pseudo_collate')
    return DataLoader(
        dataset=dataset, sampler=sampler, batch_size=bargs.batch_size,
        num_workers=bargs.num_workers, persistent_workers=False,
        drop_last=False, collate_fn=partial(collate_fn))


def discover_cross_tiles(repo: Path) -> List[str]:
    rp = REPAIR_WD / 'splits/calib_split.json'
    if rp.exists():
        data = json.loads(rp.read_text())
        ct = [t for t in data.get('cross_tiles', []) if t != DEFAULT_TILE]
        if ct:
            return sorted(ct)
    tiles = []
    vis = repo / 'vis'
    for d in sorted(vis.glob('P*/dataset/images')):
        tid = d.parent.parent.name
        if tid != DEFAULT_TILE and list(d.glob('*_rot000.*')):
            tiles.append(tid)
    return tiles


def list_vis_tiles(repo: Path) -> List[str]:
    out = []
    for d in sorted((repo / 'vis').glob('P*')):
        if (d / 'dataset' / 'images').is_dir():
            out.append(d.name)
    return out


def heldout_candidates(repo: Path, n: int = 80) -> List[str]:
    split_path = REPAIR_WD / 'splits/calib_split.json'
    cross = set(discover_cross_tiles(repo)) | {DEFAULT_TILE}
    if not split_path.exists():
        return []
    held = json.loads(split_path.read_text()).get('heldout_calib', [])
    return [h for h in held if h not in cross][:n]


def stratify_tiles(probe_rows: List[dict], n_each: int = 10) -> Dict[str, List[str]]:
    """Split by baseline final_sv at angle 0."""
    by_tile = {}
    for r in probe_rows:
        if int(float(r.get('angle', -1))) != 0:
            continue
        tid = r['tile_id']
        try:
            v = float(r['final_sv_ratio'])
        except (TypeError, ValueError):
            continue
        if np.isnan(v):
            continue
        by_tile.setdefault(tid, []).append(v)
    means = [(t, float(np.mean(v))) for t, v in by_tile.items()]
    means.sort(key=lambda x: -x[1])
    n = len(means)
    if n < n_each * 3:
        hi = means[: max(1, n // 3)]
        mid = means[max(1, n // 3): max(2, 2 * n // 3)]
        lo = means[max(2, 2 * n // 3):]
    else:
        hi, mid, lo = means[:n_each], means[n_each:2 * n_each], means[-n_each:]
    return {
        'high': [t for t, _ in hi],
        'medium': [t for t, _ in mid],
        'low': [t for t, _ in lo],
    }


def build_model(ctx: MechContext, support_type: str = 'visual'):
    sp, note = verify.resolve_support_from_config(ctx.config)
    if ctx.support_pkl:
        sp = ctx.support_pkl
    elif sp is None:
        sp = REPO_ROOT / verify.DEFAULT_SUPPORT_FALLBACK
    ctx.support_pkl = Path(sp) if sp else None
    bargs = SimpleNamespace(
        config=str(ctx.config),
        checkpoint=str(ctx.checkpoint),
        image_dir=str(REPO_ROOT),
        out_dir=str(ctx.work_dir / '_tmp'),
        angles=ATLAS_ANGLES,
        angle_step=None,
        score_thr=ctx.score_thr,
        iou_thr=ctx.nms_iou,
        support_shot=8,
        support_type=support_type,
        support_feat=str(ctx.support_pkl),
        normalized_class_dict='data/normalized_class_dict.pkl',
        batch_size=1,
        num_workers=0,
        seed=ctx.seed,
        device='cuda:0',
        max_spatial_size=32,
        result_md_dir=str(ctx.result_md_dir),
        class_names=CLASSES,
    )
    probe_base.setup_reproducibility(ctx.seed)
    _, model, cfg = probe_base.build_runner_model(bargs, str(ctx.support_pkl))
    device = torch.device('cuda:0')
    model.to(device).eval()
    _, _, det_support, name2id, id2name = probe_base.prepare_support(bargs, device)
    return bargs, model, device, det_support, name2id, id2name, cfg


def apply_support_intervention(
        support_feats: torch.Tensor,
        support_labels: torch.Tensor,
        intervention: str,
        device: torch.device,
        det_support=None,
        name2id=None) -> torch.Tensor:
    if intervention in ('original', 'visual_support', 'baseline'):
        return support_feats
    if intervention == 'zero_sv_embedding':
        return verify.apply_intervention(support_feats, support_labels, 'zero_sv', device)
    if intervention == 'swap_sv_with_nearest_class_embedding':
        feats = support_feats.clone()
        lab = support_labels.squeeze(0) if support_labels.dim() > 1 else support_labels
        sv_mask = lab == SMALL
        sv_f = feats[0, sv_mask] if feats.dim() == 3 else feats[sv_mask]
        best_cls, best_sim = None, -2.0
        for cid in range(NUM_CLASSES):
            if cid == SMALL:
                continue
            mask = lab == cid
            if not mask.any():
                continue
            other = feats[0, mask].mean(0) if feats.dim() == 3 else feats[mask].mean(0)
            sim = F.cosine_similarity(sv_f.mean(0), other, dim=0).item()
            if sim > best_sim:
                best_sim, best_cls = sim, cid
        if best_cls is not None:
            mask = lab == SMALL
            repl = feats[0, lab == best_cls].mean(0, keepdim=True)
            if feats.dim() == 3:
                feats[0, mask] = repl.expand_as(feats[0, mask])
            else:
                feats[mask] = repl.expand_as(feats[mask])
        return feats
    if intervention == 'orthogonalize_sv_embedding_to_background_mean':
        feats = support_feats.clone()
        lab = support_labels.squeeze(0) if support_labels.dim() > 1 else support_labels
        non_sv = lab != SMALL
        if non_sv.any():
            bg_mean = feats[0, non_sv].mean(0) if feats.dim() == 3 else feats[non_sv].mean(0)
            bg_mean = F.normalize(bg_mean, dim=0)
            sv_mask = lab == SMALL
            sv = feats[0, sv_mask] if feats.dim() == 3 else feats[sv_mask]
            proj = (sv @ bg_mean).unsqueeze(1) * bg_mean.unsqueeze(0)
            if feats.dim() == 3:
                feats[0, sv_mask] = sv - proj
            else:
                feats[sv_mask] = sv - proj
        return feats
    if intervention == 'normalize_all_class_embeddings':
        return verify.apply_intervention(support_feats, support_labels, 'normalize_all', device)
    return verify.apply_intervention(support_feats, support_labels, intervention, device)


def build_repair_for_method(ctx: MechContext, device, method: str) -> Tuple[RepairStack, Optional[Any]]:
    """Return (repair_stack, support_modifier callable or None)."""
    rep = RepairStack()
    rep.sv_idx = SMALL
    prog = ctx.repair_progress()
    sf_mod = None

    if method in ('baseline', 'visual_support', 'text_support'):
        return rep, None
    if method == 'best_R1':
        if R1_CKPT.exists():
            calib = ClassWiseCalibration(NUM_CLASSES, True).to(device)
            calib.load_state_dict(torch.load(R1_CKPT, map_location=device))
            rep.calibration = calib
        return rep, None
    if method == 'best_C5':
        if R1_CKPT.exists():
            calib = ClassWiseCalibration(NUM_CLASSES, True).to(device)
            calib.load_state_dict(torch.load(R1_CKPT, map_location=device))
            rep.calibration = calib
        sf_mod = lambda sf, sl, sv: apply_fixed_alpha_support(sf, sl, sv, 0.5)
        return rep, sf_mod
    if method == 'bias_only_R1':
        if R1_CKPT.exists():
            sd = torch.load(R1_CKPT, map_location=device)
            calib = ClassWiseCalibration(NUM_CLASSES, False).to(device)
            with torch.no_grad():
                calib.bias.copy_(sd['bias'])
                calib.raw_tau.zero_()
            rep.calibration = calib
        return rep, None
    if method == 'temperature_only_R1':
        if R1_CKPT.exists():
            sd = torch.load(R1_CKPT, map_location=device)
            calib = ClassWiseCalibration(NUM_CLASSES, True).to(device)
            with torch.no_grad():
                calib.bias.zero_()
                calib.raw_tau.copy_(sd['raw_tau'])
            rep.calibration = calib
        return rep, None
    if method in ('bias_plus_temperature_R1', 'R1_bias+temperature'):
        if R1_CKPT.exists():
            calib = ClassWiseCalibration(NUM_CLASSES, True).to(device)
            calib.load_state_dict(torch.load(R1_CKPT, map_location=device))
            rep.calibration = calib
        return rep, None
    if method == 'oracle_remove_sv_bias':
        rep.grid_sv_bias = -3.0
        return rep, None
    if method.startswith('zero_sv') or 'embedding' in method:
        return rep, None
    return rep, None


def dense_metrics_extended(logits_list: List[torch.Tensor], sv_idx: int,
                           pred_embeds=None, sv_direction=None) -> Dict[str, float]:
    parts, embed_parts = [], []
    for i, logit in enumerate(logits_list):
        scores = logit.sigmoid()
        flat = scores.permute(0, 2, 3, 1).reshape(-1, scores.shape[1])
        parts.append(flat)
        if pred_embeds is not None and i < len(pred_embeds) and pred_embeds[i] is not None:
            pe = pred_embeds[i].permute(0, 2, 3, 1).reshape(-1, pred_embeds[i].shape[1])
            embed_parts.append(pe)
    cat = torch.cat(parts, dim=0)
    top1 = cat.argmax(dim=1)
    margin = cat.topk(2, dim=1).values
    sv_margin = (margin[:, 0] - margin[:, 1]) if margin.shape[1] > 1 else margin[:, 0]
    sv_scores = cat[:, sv_idx]
    probs = cat / cat.sum(dim=1, keepdim=True).clamp_min(1e-12)
    entropy = (-(probs * probs.log().clamp_min(-50))).sum(dim=1)
    obj = cat.max(dim=1).values

    out = dict(
        dense_top1_sv_ratio=float((top1 == sv_idx).float().mean().item()),
        dense_sv_mean_logit=float(sv_scores.mean().item()),
        dense_sv_margin_vs_runnerup=float(sv_margin.mean().item()),
        dense_sv_margin_p95=float(torch.quantile(sv_margin, 0.95).item()),
        dense_entropy=float(entropy.mean().item()),
        objectness_mean=float(obj.mean().item()),
        objectness_p95=float(torch.quantile(obj, 0.95).item()),
        num_dense_locations=int(cat.shape[0]),
    )
    if embed_parts and sv_direction is not None:
        emb = torch.cat(embed_parts, dim=0)
        sv_dir = F.normalize(sv_direction, dim=0)
        cos = F.cosine_similarity(F.normalize(emb, dim=-1), sv_dir.unsqueeze(0), dim=-1)
        out['background_to_sv_cosine_mean'] = float(cos.mean().item())
        out['background_to_sv_cosine_p95'] = float(torch.quantile(cos, 0.95).item())
    return out


def parse_gt_stats(repo: Path, tile_id: str, angle: int) -> Tuple[int, int]:
    """Returns (has_sv_gt, sv_count)."""
    ann_candidates = [
        repo / 'vis' / tile_id / 'dataset' / 'annfiles' / f'{tile_id}_rot{angle:03d}.pkl',
        angle_sweep_dir(repo, angle) / 'annfiles' / f'{tile_id}.pkl',
    ]
    for ap in ann_candidates:
        if not ap.exists():
            continue
        try:
            from commonlibs.common_tools import pklload
            ann = pklload(str(ap))
            texts = ann.get('texts', [])
            sv = sum(1 for t in texts if t == 'small-vehicle')
            return (1 if sv > 0 else 0, sv)
        except Exception as exc:
            log.warning('GT parse failed %s: %s', ap, exc)
    return 0, 0


def eval_one(
        ctx: MechContext,
        model,
        bargs,
        device,
        det_support,
        name2id,
        tile_id: str,
        angle: int,
        method: str,
        support_intervention: str = 'original',
        image_override: Optional[Path] = None,
        extra_tag: str = '',
) -> dict:
    """Run single tile×angle×method; returns metric row."""
    try:
        if image_override is not None:
            dataset_root = image_override.parent.parent  # .../dataset/images/file -> dataset
            source = 'counterfactual'
            cf_stem = image_override.stem
        else:
            dataset_root, source = resolve_tile_source(ctx.repo_root, tile_id, angle)
            cf_stem = None
    except FileNotFoundError as exc:
        return dict(tile_id=tile_id, angle=angle, method=method, notes=str(exc))

    lb = SimpleNamespace(**vars(bargs))
    if source == 'angle_sweep':
        dataset_root = prepare_angle_sweep_mini_dataset(ctx.repo_root, tile_id, angle)
        source = 'angle_sweep_mini'
        loader = build_mini_dataloader(lb, dataset_root, filter_empty_gt=False)
    else:
        lb.image_dir = str(dataset_root / 'images')
        loader = probe_base.build_dataloader(lb, [angle])

    st = support_intervention
    if method == 'text_support':
        bargs.support_type = 'text'
    elif method == 'visual_support':
        bargs.support_type = 'visual'
    if method == 'zero_sv_embedding':
        st = 'zero_sv'

    if bargs.support_type == 'text':
        sf, sl = verify.build_mapped_support(
            model, verify.RunContext(
                repo_root=ctx.repo_root, config=ctx.config, checkpoint=ctx.checkpoint,
                support_pkl=ctx.support_pkl, work_dir=ctx.work_dir,
                result_md_dir=ctx.result_md_dir, gpu=ctx.gpu, mode='mech',
                angles=[angle], tiles=[tile_id]),
            det_support, name2id, device, 'text', st)
    else:
        sf, sl = build_support_tensors(model, det_support, name2id, device, 'visual', 8)
        sf = apply_support_intervention(sf, sl, st, device)

    rep, sf_mod = build_repair_for_method(ctx, device, method)
    if sf_mod is not None:
        sv_dir = sv_support_direction(model, sf, sl, SMALL)
        sf = sf_mod(sf, sl, sv_dir)

    row_base = dict(
        tile_id=tile_id,
        angle=angle,
        method=method,
        condition=extra_tag or method,
        config_path=str(ctx.config),
        checkpoint_path=str(ctx.checkpoint),
        git_commit=verify.git_commit(ctx.repo_root),
        seed=ctx.seed,
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        notes=source,
    )

    with torch.no_grad():
        for data_info in loader:
            stem = Path(data_info['data_samples'][0].img_path).stem
            if cf_stem is not None:
                if stem != cf_stem:
                    continue
            elif tile_id not in stem and not stem.startswith(tile_id):
                continue
            data = model.data_preprocessor(data_info, False)
            data['inputs'] = data['inputs'].to(device)
            x = model.prompt_extract_feats(data['inputs'])
            metas = [s.metainfo for s in data['data_samples']]
            outs_all = model.bbox_head(
                x, sf, sl, sl, support_shot=8, num_classes=NUM_CLASSES,
                num_in_classes=NUM_CLASSES, align_style='labelled',
                support_type=bargs.support_type, text_cls_scale=0.0)
            pred_embeds = outs_all[3] if len(outs_all) > 3 else None
            if rep.calibration is not None or rep.grid_sv_bias is not None:
                cls_adj = apply_repair_logits(outs_all, rep)
            else:
                cls_adj = list(outs_all[0])
            outs = (tuple(cls_adj), outs_all[1], outs_all[2])
            sv_dir = sv_support_direction(model, sf, sl, SMALL)
            dm = dense_metrics_extended(
                cls_adj, SMALL, pred_embeds, sv_dir)
            fin = final_detection_from_outs(model, outs, metas,
                                            score_thr=ctx.score_thr)
            hist = fin.get('class_histogram', {})
            if isinstance(hist, str):
                hist = json.loads(hist)
            gt_sv, gt_cnt = parse_gt_stats(ctx.repo_root, tile_id, angle)
            row = {**row_base, **dm}
            row.update(
                det_count=fin['detection_total'],
                small_vehicle_count=int(fin['final_sv_ratio'] * max(fin['detection_total'], 1)),
                final_sv_ratio=fin['final_sv_ratio'],
                mean_score=fin.get('mean_score', float('nan')),
                top1_class_histogram=json.dumps(hist, ensure_ascii=True),
                gt_small_vehicle=gt_sv,
                gt_sv_count=gt_cnt,
                ap50='NA',
                sv_ap50='NA',
            )
            if 'small_vehicle_count' not in fin:
                labels_hist = hist.get('small-vehicle', 0)
                row['small_vehicle_count'] = labels_hist
            return row
    return {**row_base, 'notes': 'no_matching_batch'}


def file_sha256(path: Path, n: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(n):
            h.update(chunk)
    return h.hexdigest()[:16]
