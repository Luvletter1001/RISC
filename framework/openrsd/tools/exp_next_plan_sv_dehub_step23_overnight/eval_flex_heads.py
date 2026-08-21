#!/usr/bin/env python
"""Flex alignment vs fusion eval via prompt_predict (fixes val_using_aux bypass)."""
from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.exp_mechanism_sv_attractor_gpu89 import common_attractor_utils as mech
from tools.exp_next_plan_sv_dehub_step23_overnight import common_overnight_utils as U
from tools.sv_attractor_repair_gpu89.repair_forward import build_support_tensors, dense_stats_from_logits


def _pred_summary(data_samples, id2name: dict, score_thr: float = 0.01, max_export: int = 400) -> dict:
    pred = data_samples[0].pred_instances
    keep = pred.scores >= score_thr
    labels = pred.labels[keep].detach().cpu().numpy()
    scores = pred.scores[keep].detach().cpu().numpy()
    boxes = pred.bboxes[keep]
    if hasattr(boxes, 'tensor'):
        boxes = boxes.tensor.detach().cpu().numpy()
    else:
        boxes = boxes.detach().cpu().numpy()
    hist = {}
    for lid in labels:
        name = id2name.get(int(lid), str(int(lid)))
        hist[name] = hist.get(name, 0) + 1
    det = int(keep.sum().item())
    sv = int((labels == mech.SMALL).sum())
    n = min(det, max_export)
    return dict(
        det_count=det,
        small_vehicle_count=sv,
        non_sv_det_count=max(0, det - sv),
        final_sv_ratio=float(sv / det) if det else 0.0,
        mean_score=float(scores.mean()) if len(scores) else float('nan'),
        top1_class_histogram=json.dumps(hist, ensure_ascii=True),
        pred_labels_json=json.dumps(labels[:n].astype(int).tolist()),
        pred_boxes_json=json.dumps(boxes[:n].tolist()),
        tennis_court_count=int(hist.get('tennis-court', 0)),
        large_vehicle_count=int(hist.get('large-vehicle', 0)),
        basketball_court_count=int(hist.get('basketball-court', 0)),
    )


def _dense_for_head(model, x, sf, sl, bargs, val_using_aux: bool) -> dict:
    kwargs = dict(
        support_shot=bargs.support_shot,
        num_classes=mech.NUM_CLASSES,
        num_in_classes=mech.NUM_CLASSES,
        align_style='labelled',
        support_type=bargs.support_type,
        text_cls_scale=0.0,
    )
    if val_using_aux and getattr(model, 'aux_bbox_head', None) is not None:
        x_in = [model.aux_convs[i](f) for i, f in enumerate(x)]
        outs = model.aux_bbox_head(x_in, sf, sl, sl, **kwargs)
        src = 'aux_bbox_head.forward'
    else:
        outs = model.bbox_head(x, sf, sl, sl, **kwargs)
        src = 'bbox_head.forward'
    cls_list = list(outs[0])
    dm = mech.dense_metrics_extended(cls_list, mech.SMALL)
    dm['logit_source'] = src
    return dm


def eval_one_flex_head(
        ctx: mech.MechContext,
        model,
        bargs,
        device,
        det_support,
        name2id,
        id2name,
        tile_id: str,
        angle: int,
        support_type: str = 'visual',
        val_using_aux: bool = False,
) -> dict:
    """Single tile eval: prompt_predict for final dets; dense from matching head."""
    bargs.support_type = support_type
    if support_type == 'text':
        from tools.verify_sv_attractor_gpu89 import common as verify
        sf, sl = verify.build_mapped_support(
            model, verify.RunContext(
                repo_root=ctx.repo_root, config=ctx.config, checkpoint=ctx.checkpoint,
                support_pkl=ctx.support_pkl, work_dir=ctx.work_dir,
                result_md_dir=ctx.result_md_dir, gpu=ctx.gpu, mode='mech',
                angles=[angle], tiles=[tile_id]),
            det_support, name2id, device, 'text', 'original')
    else:
        sf, sl = build_support_tensors(model, det_support, name2id, device, 'visual', 8)

    lb = SimpleNamespace(**vars(bargs))
    tennis_img = (ctx.result_md_dir / 'work' / 'tennis_rotations' / tile_id / 'dataset' / 'images'
                  / f'{tile_id}_rot{angle:03d}.png')
    if tennis_img.exists():
        dataset_root = tennis_img.parent.parent
        lb.image_dir = str(dataset_root / 'images')
        from tools.rotation_diagnostics import probe_rotated_stage_outputs as probe_base
        loader = probe_base.build_dataloader(lb, [angle])
        source = 'tennis_rotations'
    else:
        dataset_root, source = mech.resolve_tile_source(ctx.repo_root, tile_id, angle)
    if source == 'angle_sweep':
        dataset_root = mech.prepare_angle_sweep_mini_dataset(ctx.repo_root, tile_id, angle)
        loader = mech.build_mini_dataloader(lb, dataset_root, filter_empty_gt=False)
    else:
        lb.image_dir = str(dataset_root / 'images')
        from tools.rotation_diagnostics import probe_rotated_stage_outputs as probe_base
        loader = probe_base.build_dataloader(lb, [angle])

    model.val_using_aux = val_using_aux
    head_mode = 'fusion' if val_using_aux else 'alignment'

    with torch.no_grad():
        for data_info in loader:
            stem = Path(data_info['data_samples'][0].img_path).stem
            if tile_id not in stem and not stem.startswith(tile_id):
                continue
            data = model.data_preprocessor(data_info, False)
            data['inputs'] = data['inputs'].to(device)
            x = model.prompt_extract_feats(data['inputs'])
            dm = _dense_for_head(model, x, sf, sl, bargs, val_using_aux)
            samples = model.prompt_predict(
                deepcopy(x), data['inputs'], deepcopy(data['data_samples']),
                val_support_data=det_support,
                val_name2id=name2id,
                val_using_aux=val_using_aux,
                rescale=True,
                support_shot=bargs.support_shot,
                support_type=support_type,
            )
            fin = _pred_summary(samples, id2name, ctx.score_thr)
            dm['prompt_predict_source'] = 'prompt_predict'
            row = dict(
                tile_id=tile_id, angle=angle, head_mode=head_mode,
                val_using_aux=val_using_aux, support_mode=support_type,
                checkpoint_path=str(ctx.checkpoint), notes=source,
                **dm, **fin,
            )
            row['dense_top1_sv_ratio'] = dm.get('dense_top1_sv_ratio', float('nan'))
            return row
    return dict(tile_id=tile_id, angle=angle, head_mode=head_mode, notes='no_batch')


def smoke_head_switch(ctx: mech.MechContext, tile: str = 'P0148__1024__651___0', angle: int = 0) -> list:
    bargs, model, device, det_support, name2id, id2name, _ = mech.build_model(ctx, 'visual')
    rows = []
    for aux in (False, True):
        r = eval_one_flex_head(ctx, model, bargs, device, det_support, name2id, id2name,
                               tile, angle, 'visual', val_using_aux=aux)
        rows.append(r)
    return rows
