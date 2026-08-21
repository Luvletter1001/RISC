#!/usr/bin/env python
"""Heldout AP via prompt_predict (alignment vs fusion)."""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.sv_attractor_repair_gpu89 import common as C
from tools.sv_attractor_repair_gpu89.eval_ap import (
    _as_rbox_tensor, build_eval_cfg, classwise_ap50, prepare_heldout_subset, rotate_image_tensor)
from tools.sv_attractor_repair_gpu89.repair_forward import build_support_tensors


def actually_used_head(model, val_using_aux: bool) -> str:
    if val_using_aux and getattr(model, 'aux_bbox_head', None) is not None:
        return 'aux_bbox_head.predict'
    return 'bbox_head.predict'


def infer_heldout_flex_ap(
        ctx: C.RepairContext,
        model,
        device,
        det_support,
        name2id,
        id2name: dict,
        subset_root: Path,
        val_using_aux: bool = False,
        angles: List[int] = None,
        max_images: int = 500,
        score_thr: float = 0.01,
) -> Dict[str, Any]:
    from mmengine.dataset import pseudo_collate
    from mmengine.registry import init_default_scope
    from mmdet.registry import DATASETS
    from mmdet.utils import register_all_modules as register_mmdet
    from mmrotate.utils import register_all_modules as register_mmrotate
    from torch.utils.data import DataLoader

    if angles is None:
        angles = [0, 90]
    register_mmdet(init_default_scope=False)
    register_mmrotate(init_default_scope=False)
    cfg = build_eval_cfg(ctx, subset_root)
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=pseudo_collate)

    model.val_using_aux = val_using_aux
    head_used = actually_used_head(model, val_using_aux)
    ap_samples = []
    ratio_rows = []
    all_hists = []
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
                samples_out = model.prompt_predict(
                    deepcopy(x), data['inputs'], deepcopy(data['data_samples']),
                    val_support_data=det_support,
                    val_name2id=name2id,
                    val_using_aux=val_using_aux,
                    rescale=True,
                    support_shot=8,
                    support_type='visual',
                )
                pred = samples_out[0].pred_instances
                keep = pred.scores >= score_thr
                labels = pred.labels[keep].detach().cpu()
                scores = pred.scores[keep].detach().cpu()
                hist = {}
                sv = 0
                for lid, sc in zip(labels.tolist(), scores.tolist()):
                    name = id2name.get(int(lid), str(int(lid)))
                    hist[name] = hist.get(name, 0) + 1
                    if int(lid) == C.SMALL:
                        sv += 1
                det = int(keep.sum().item())
                all_hists.append(hist)
                ratio_rows.append(dict(
                    stem=stem, angle=angle,
                    final_sv_ratio=float(sv / det) if det else 0.0,
                    detection_total=det,
                    sv_det_count=sv,
                    non_sv_det_count=max(0, det - sv),
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
                        scores=scores,
                        labels=labels,
                    ),
                ))

    cls_rows, map_ap = classwise_ap50(ap_samples, tuple(C.CLASSES))
    sv_ap = float('nan')
    for row in cls_rows:
        if row['class_name'] == 'small-vehicle':
            sv_ap = row['ap50']
    merged_hist = {}
    for h in all_hists:
        for k, v in h.items():
            merged_hist[k] = merged_hist.get(k, 0) + v
    det_mean = float(np.mean([r['detection_total'] for r in ratio_rows])) if ratio_rows else 0
    fsv = float(np.mean([r['final_sv_ratio'] for r in ratio_rows])) if ratio_rows else float('nan')
    sv_det = float(np.mean([r['sv_det_count'] for r in ratio_rows])) if ratio_rows else 0
    non_sv = float(np.mean([r['non_sv_det_count'] for r in ratio_rows])) if ratio_rows else 0
    return dict(
        ap50_overall=map_ap,
        sv_ap50=sv_ap,
        per_class=cls_rows,
        detection_total=det_mean,
        final_sv_ratio=fsv,
        sv_det_count=sv_det,
        non_sv_det_count=non_sv,
        class_histogram=json.dumps(merged_hist, ensure_ascii=True),
        actually_used_head=head_used,
        val_using_aux=val_using_aux,
        n_samples=len(ap_samples),
    )
