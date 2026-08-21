#!/usr/bin/env python
"""openrsd_eval_metric + eval_rbbox_map: AP50 overall / per-class."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmengine import Config
from mmengine.evaluator import Evaluator
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmdet.registry import DATASETS
from mmdet.utils import register_all_modules as register_mmdet
from mmrotate.utils import register_all_modules as register_mmrotate
from mmrotate.evaluation.functional.mean_ap import eval_rbbox_map

from tools.analysis_tools.eval_metric import merge_predictions_with_ground_truth
from tools.sv_attractor_repair_gpu89 import common as C
from tools.sv_attractor_repair_gpu89.exp_combined import load_repair_stack
from tools.sv_attractor_repair_gpu89.exp_r3 import apply_fixed_alpha_support
from tools.sv_attractor_repair_gpu89.repair_forward import (
    build_support_tensors, final_detection_from_outs, sv_support_direction)
from tools.sv_attractor_repair_gpu89.repair_modules import RepairStack


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def metric_from_eval_results(results: dict) -> Dict[str, Any]:
    def gv(*keys):
        for k in keys:
            if k in results:
                return float(results[k])
        return float('nan')

    raw = {}
    for k, v in results.items():
        try:
            raw[str(k)] = float(v)
        except (TypeError, ValueError):
            raw[str(k)] = v
    return dict(
        ap50_overall=gv('dota/AP50', 'AP50'),
        map_overall=gv('dota/mAP', 'mAP'),
        raw=raw,
    )


def _as_rbox_tensor(bboxes) -> torch.Tensor:
    if hasattr(bboxes, 'tensor'):
        return bboxes.tensor.detach().cpu()
    if isinstance(bboxes, torch.Tensor):
        return bboxes.detach().cpu()
    return torch.as_tensor(bboxes)


def classwise_ap50(samples: list, classes: tuple, predict_box_type: str = 'rbox',
                   iou_thr: float = 0.5) -> Tuple[List[dict], float]:
    annotations = []
    det_results = []
    for sample in samples:
        gt = sample.get('gt_instances', {})
        ign = sample.get('ignored_instances', {})
        if not gt:
            ann = dict(
                labels=np.zeros(0, dtype=np.int64),
                bboxes=np.zeros((0, 5), dtype=np.float32),
                labels_ignore=np.zeros(0, dtype=np.int64),
                bboxes_ignore=np.zeros((0, 5), dtype=np.float32),
            )
        else:
            ann = dict(
                labels=_to_numpy(gt['labels']).astype(np.int64),
                bboxes=_as_rbox_tensor(gt['bboxes']).numpy().astype(np.float32),
                labels_ignore=_to_numpy(ign.get('labels', [])).astype(np.int64)
                if ign else np.zeros(0, dtype=np.int64),
                bboxes_ignore=(
                    _as_rbox_tensor(ign['bboxes']).numpy().astype(np.float32).reshape(-1, 5)
                    if ign and ign.get('bboxes') is not None and _as_rbox_tensor(ign['bboxes']).numel()
                    else np.zeros((0, 5), dtype=np.float32)),
            )
        annotations.append(ann)
        pred = sample['pred_instances']
        boxes = _as_rbox_tensor(pred['bboxes']).numpy().astype(np.float32)
        labels = _to_numpy(pred['labels']).astype(np.int64)
        scores = _to_numpy(pred['scores']).astype(np.float32)
        per_class = []
        for lid in range(len(classes)):
            idx = np.where(labels == lid)[0]
            if len(idx):
                per_class.append(np.hstack([boxes[idx], scores[idx].reshape(-1, 1)]))
            else:
                per_class.append(np.zeros((0, 6), dtype=np.float32))
        det_results.append(per_class)

    mean_ap, cls_results = eval_rbbox_map(
        det_results, annotations, iou_thr=iou_thr, use_07_metric=True,
        box_type=predict_box_type, dataset=classes, logger='silent', nproc=4)
    rows = []
    for name, cr in zip(classes, cls_results):
        ap = float(np.asarray(cr['ap']).reshape(-1)[0])
        rows.append(dict(class_name=name, ap50=ap, num_gts=int(cr['num_gts']),
                         num_dets=int(cr['num_dets'])))
    return rows, float(mean_ap)


def prepare_heldout_subset(ctx: C.RepairContext, max_images: int) -> Path:
    splits = C.build_splits(ctx)
    stems = splits['heldout_calib'][:max_images]
    root = ctx.work_dir / 'eval_subsets' / 'heldout_calib'
    img_dir = root / 'images'
    ann_dir = root / 'annfiles'
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    for p in list(img_dir.iterdir()):
        p.unlink(missing_ok=True)
    for p in list(ann_dir.iterdir()):
        p.unlink(missing_ok=True)
    for stem in stems:
        img = C.image_path_for_stem(stem)
        ann = C.SS_TRAIN_ROOT / 'annfiles' / f'{stem}.txt'
        if img and img.exists():
            (img_dir / img.name).symlink_to(img.resolve())
        if ann.exists():
            (ann_dir / ann.name).symlink_to(ann.resolve())
    (root / 'meta.json').write_text(json.dumps(dict(stems=stems, n=len(stems)), indent=2))
    return root


def build_eval_cfg(ctx: C.RepairContext, subset_root: Path) -> Config:
    register_mmdet(init_default_scope=False)
    register_mmrotate(init_default_scope=False)
    cfg = Config.fromfile(str(ctx.config))
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    parent = subset_root.parent
    rel = subset_root.name
    cfg.merge_from_dict({
        'val_dataloader.dataset.type': 'DOTADataset',
        'val_dataloader.dataset.data_root': str(parent),
        'val_dataloader.dataset.ann_file': f'{rel}/annfiles',
        'val_dataloader.dataset.data_prefix': dict(img_path=f'{rel}/images'),
        'val_dataloader.dataset.metainfo': dict(classes=C.CLASSES),
        'val_dataloader.dataset.filter_cfg': dict(filter_empty_gt=False),
        'val_dataloader.dataset.test_mode': True,
        'val_dataloader.batch_size': 1,
        'val_dataloader.num_workers': 0,
        'val_dataloader.persistent_workers': False,
        'test_dataloader.dataset.type': 'DOTADataset',
        'test_dataloader.dataset.data_root': str(parent),
        'test_dataloader.dataset.ann_file': f'{rel}/annfiles',
        'test_dataloader.dataset.data_prefix': dict(img_path=f'{rel}/images'),
        'test_dataloader.dataset.metainfo': dict(classes=C.CLASSES),
        'test_dataloader.dataset.filter_cfg': dict(filter_empty_gt=False),
        'test_dataloader.dataset.test_mode': True,
        'test_dataloader.batch_size': 1,
        'test_dataloader.num_workers': 0,
        'test_dataloader.persistent_workers': False,
    })
    return cfg


def rotate_image_tensor(img_tensor: torch.Tensor, angle: int) -> torch.Tensor:
    if angle % 360 == 0:
        return img_tensor
    mean = torch.tensor([103.53, 116.28, 123.675], device=img_tensor.device)
    std = torch.tensor([57.375, 57.12, 58.395], device=img_tensor.device)
    img = img_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    img_u8 = np.clip(img * std.cpu().numpy() + mean.cpu().numpy(), 0, 255).astype(np.uint8)
    if angle == 90:
        rot = cv2.rotate(img_u8, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        rot = cv2.rotate(img_u8, cv2.ROTATE_180)
    elif angle == 270:
        rot = cv2.rotate(img_u8, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        rot = img_u8
    t = torch.from_numpy(rot).permute(2, 0, 1).float().unsqueeze(0).to(img_tensor.device)
    return (t - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)


def poly8_to_rbox(poly8: List[float]) -> np.ndarray:
    pts = np.asarray(poly8, dtype=np.float32).reshape(4, 2)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), ang = rect
    if w < h:
        w, h = h, w
        ang += 90
    return np.array([cx, cy, w, h, np.deg2rad(ang)], dtype=np.float32)


def rotate_poly8(poly8: List[float], angle: int, w: int, h: int) -> List[float]:
    pts = np.asarray(poly8, dtype=np.float32).reshape(4, 2)
    if angle == 90:
        nr = np.stack([pts[:, 1], h - 1 - pts[:, 0]], axis=1)
    elif angle == 180:
        nr = np.stack([w - 1 - pts[:, 0], h - 1 - pts[:, 1]], axis=1)
    elif angle == 270:
        nr = np.stack([w - 1 - pts[:, 1], pts[:, 0]], axis=1)
    else:
        nr = pts
    return nr.reshape(-1).tolist()


def load_gt_sample(stem: str, angle: int, ann_path: Path, img_shape: Tuple[int, int]):
    w, h = img_shape
    labels, bboxes = [], []
    if ann_path.exists():
        for line in ann_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            poly = [float(x) for x in parts[:8]]
            if angle:
                poly = rotate_poly8(poly, angle, w, h)
            cls = parts[8]
            if cls not in C.CLASSES:
                continue
            labels.append(C.CLASSES.index(cls))
            bboxes.append(poly8_to_rbox(poly))
    return dict(
        img_id=f'{stem}_rot{angle:03d}' if angle else stem,
        gt_instances=dict(
            labels=torch.tensor(labels, dtype=torch.long),
            bboxes=torch.tensor(bboxes, dtype=torch.float32) if bboxes else
            torch.zeros((0, 5), dtype=torch.float32),
        ),
        ignored_instances=dict(
            labels=torch.zeros(0, dtype=torch.long),
            bboxes=torch.zeros((0, 5), dtype=torch.float32),
        ),
    )


def method_to_repair(ctx, device, method: str) -> Tuple[RepairStack, Optional[callable]]:
    if method == 'baseline':
        return RepairStack(), None
    tag_map = {
        'best_R1': 'R1',
        'best_R2': 'R2',
        'best_R3': 'R3',
        'best_combined': ctx.progress.get('best_combined_method', 'C5_R1+R3'),
    }
    tag = tag_map.get(method, method)
    rep = load_repair_stack(ctx, device, tag)
    sf_mod = None
    if method == 'best_R3' or 'R3' in tag:
        sf_mod = lambda sf, sl, sv: apply_fixed_alpha_support(sf, sl, sv, 0.5)
    return rep, sf_mod


def forward_with_repair(model, x, sf, sl, rep: RepairStack, sf_mod=None):
    if sf_mod is not None:
        sv = sv_support_direction(model, sf, sl, C.SMALL)
        sf = sf_mod(sf, sl, sv)
    if getattr(rep, 'rotation_adapter', None) is not None:
        try:
            from M_Tools.rotation_sv_repair.rotation_adapter import apply_rotation_adapter_to_feats
            x = apply_rotation_adapter_to_feats(x, rep.rotation_adapter)
        except ImportError:
            pass
    outs_all = model.bbox_head(
        x, sf, sl, sl, support_shot=8, num_classes=C.NUM_CLASSES,
        num_in_classes=C.NUM_CLASSES, align_style='labelled',
        support_type='visual', text_cls_scale=0.0)
    pred_embeds = outs_all[3] if len(outs_all) > 3 else None
    cls_adj = []
    for li, cs in enumerate(outs_all[0]):
        b, c, h, w = cs.shape
        flat = cs.permute(0, 2, 3, 1).reshape(-1, c)
        vis = None
        if pred_embeds is not None and rep.neg_bank is not None:
            pe = pred_embeds[li].permute(0, 2, 3, 1).reshape(-1, pred_embeds[li].shape[1])
            if pe.shape[1] == rep.neg_bank.prototypes.shape[1]:
                vis = pe
        flat = rep.apply_logits(flat, vis)
        cls_adj.append(flat.reshape(b, h, w, c).permute(0, 3, 1, 2))
    return (tuple(cls_adj), outs_all[1], outs_all[2])


def infer_heldout_method(
        ctx: C.RepairContext,
        model,
        device,
        det_support,
        name2id,
        subset_root: Path,
        method: str,
        angles: List[int],
        max_images: int,
        score_thr: float = 0.01,
) -> Dict[str, Any]:
    from mmengine.dataset import pseudo_collate
    from torch.utils.data import DataLoader

    rep, sf_mod = method_to_repair(ctx, device, method)
    sf_base, sl = build_support_tensors(model, det_support, name2id, device, 'visual', 8)
    cfg = build_eval_cfg(ctx, subset_root)
    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=pseudo_collate)

    ap_samples = []
    ratio_rows = []
    n_seen = 0

    with torch.no_grad():
        for batch in loader:
            if n_seen >= max_images:
                break
            n_seen += 1
            stem = Path(batch['data_samples'][0].img_path).stem
            for angle in angles:
                data = model.data_preprocessor(batch, False)
                data['inputs'] = data['inputs'].to(device)
                if angle:
                    data['inputs'] = rotate_image_tensor(data['inputs'], angle)
                x = model.prompt_extract_feats(data['inputs'])
                metas = [s.metainfo for s in data['data_samples']]
                outs = forward_with_repair(model, x, sf_base, sl, rep, sf_mod)
                pred = model.bbox_head.predict_by_feat(
                    *outs, batch_img_metas=metas, rescale=True, with_nms=True)[0]
                keep = pred.scores >= score_thr
                fin = final_detection_from_outs(model, outs, metas, score_thr=score_thr)
                ratio_rows.append(dict(
                    stem=stem, angle=angle,
                    final_sv_ratio=fin['final_sv_ratio'],
                    detection_total=fin['detection_total'],
                ))
                if angle != 0:
                    continue
                ds = batch['data_samples'][0]
                gt = ds.gt_instances
                ap_samples.append(dict(
                    img_id=stem,
                    gt_instances=dict(
                        labels=gt.labels.cpu(),
                        bboxes=_as_rbox_tensor(gt.bboxes),
                    ),
                    ignored_instances=dict(
                        labels=torch.zeros(0, dtype=torch.long),
                        bboxes=torch.zeros((0, 5), dtype=torch.float32),
                    ),
                    pred_instances=dict(
                        bboxes=_as_rbox_tensor(pred.bboxes[keep]),
                        scores=pred.scores[keep].detach().cpu(),
                        labels=pred.labels[keep].detach().cpu(),
                    ),
                ))

    cls_rows, map_ap = classwise_ap50(ap_samples, tuple(C.CLASSES))
    out = dict(
        method=method,
        ap50_overall=map_ap,
        detection_total=float(np.mean([r['detection_total'] for r in ratio_rows])) if ratio_rows else 0,
        final_sv_ratio=float(np.mean([r['final_sv_ratio'] for r in ratio_rows])) if ratio_rows else float('nan'),
        n_samples=len(ap_samples),
    )
    for row in cls_rows:
        if row['class_name'] == 'small-vehicle':
            out['sv_ap50'] = row['ap50']
        elif row['class_name'] == 'large-vehicle':
            out['lv_ap50'] = row['ap50']
        elif row['class_name'] == 'ship':
            out['ship_ap50'] = row['ap50']
    out['per_class'] = cls_rows
    return out


def run_openrsd_eval(
        ctx: C.RepairContext,
        predictions: List[dict],
        subset_root: Path,
        log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    out = dict(status='FAILED', error='', ap50_overall=float('nan'), per_class=[])
    try:
        cfg = build_eval_cfg(ctx, subset_root)
        samples = merge_predictions_with_ground_truth(cfg, predictions)
        evaluator = Evaluator(cfg.val_evaluator)
        dataset = DATASETS.build(cfg.test_dataloader.dataset)
        evaluator.dataset_meta = dataset.metainfo
        log_path = log_path or ctx.log_dir() / 'openrsd_eval_last.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w') as lf:
            import contextlib
            with contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
                results = evaluator.offline_evaluate(samples)
            lf.write(f'\n{results}\n')
        overall = metric_from_eval_results(results)
        cls_rows, _ = classwise_ap50(samples, tuple(C.CLASSES))
        ap50 = overall['ap50_overall']
        if np.isnan(ap50) and cls_rows:
            ap50 = float(np.mean([r['ap50'] for r in cls_rows]))
        out.update(status='OK', ap50_overall=ap50,
                   per_class=cls_rows, raw=overall['raw'], log_path=str(log_path))
        for row in cls_rows:
            if row['class_name'] == 'small-vehicle':
                out['sv_ap50'] = row['ap50']
            elif row['class_name'] == 'large-vehicle':
                out['lv_ap50'] = row['ap50']
            elif row['class_name'] == 'ship':
                out['ship_ap50'] = row['ap50']
    except Exception as exc:
        out['error'] = f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}'
    return out


def run_evaluator_smoke_test(ctx: C.RepairContext, n_images: int = 4) -> Dict[str, Any]:
    subset = prepare_heldout_subset(ctx, n_images)
    bargs, model, device, det_support, name2id, _, _, _ = C.build_model_ctx(ctx)
    # offline DOTAMetric path
    cfg = build_eval_cfg(ctx, subset)
    from mmengine.dataset import pseudo_collate
    from torch.utils.data import DataLoader
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=pseudo_collate)
    rep, sf_mod = RepairStack(), None
    sf, sl = build_support_tensors(model, det_support, name2id, device, 'visual', 8)
    preds = []
    with torch.no_grad():
        for batch in loader:
            data = model.data_preprocessor(batch, False)
            data['inputs'] = data['inputs'].to(device)
            x = model.prompt_extract_feats(data['inputs'])
            metas = [s.metainfo for s in data['data_samples']]
            outs = forward_with_repair(model, x, sf, sl, rep, sf_mod)
            pred = model.bbox_head.predict_by_feat(
                *outs, batch_img_metas=metas, rescale=True, with_nms=True)[0]
            stem = Path(batch['data_samples'][0].img_path).stem
            preds.append(dict(
                img_id=stem,
                pred_instances=dict(bboxes=pred.bboxes, scores=pred.scores, labels=pred.labels),
            ))
    log = ctx.log_dir() / 'evaluator_smoke_test.log'
    return run_openrsd_eval(ctx, preds, subset, log)


def run_subprocess_eval_metric(pkl_path: Path, config_path: Path, log_path: Path) -> Dict[str, Any]:
    py = os.environ.get('OPENRSD_PYTHON', '/data/zcy/anaconda3/envs/openrsd/bin/python')
    cmd = [py, str(REPO_ROOT / 'tools/openrsd_eval_metric.py'), str(config_path), str(pkl_path)]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    log_path.write_text((proc.stdout or '') + '\n' + (proc.stderr or ''))
    if proc.returncode != 0:
        return dict(status='FAILED', error=proc.stderr or proc.stdout, returncode=proc.returncode)
    ap50 = None
    m = re.search(r'dota/AP50[:\'"]?\s*[:=]?\s*([0-9.]+)', proc.stdout or '')
    if m:
        ap50 = float(m.group(1))
    return dict(status='OK' if ap50 is not None else 'PARTIAL',
                ap50_overall=ap50, stdout=(proc.stdout or '')[:3000])
