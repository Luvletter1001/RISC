#!/usr/bin/env python
"""Forward helpers with frozen OpenRSD + optional RepairStack."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from tools.sv_attractor_repair_gpu89.repair_modules import RepairStack

try:
    from M_Tools.rotation_sv_repair.rotation_adapter import apply_rotation_adapter_to_feats
except ImportError:
    apply_rotation_adapter_to_feats = None  # type: ignore


def freeze_openrsd(model) -> int:
    n = 0
    for p in model.parameters():
        if p.requires_grad:
            p.requires_grad_(False)
            n += p.numel()
    return n


def count_trainable(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build_support_tensors(model, det_support, name2id, device, support_type, support_shot):
    embed_name = 'visual_embeds' if support_type == 'visual' else 'text_embeds'
    mapping = (model.visual_support_mapping if support_type == 'visual'
               else model.text_support_mapping)
    feats, labels = [], []
    for cls_name, info in det_support.items():
        cid = name2id[cls_name]
        import numpy as np
        arr = np.asarray(info[embed_name])[:support_shot]
        feats.append(arr)
        labels.append(np.ones(len(arr)) * cid)
    raw = torch.tensor(np.concatenate(feats), device=device).float()
    lab = torch.tensor(np.concatenate(labels), device=device).long()
    order = torch.argsort(lab)
    mapped = mapping(raw[order])
    return mapped.unsqueeze(0), lab[order].unsqueeze(0)


def sv_support_direction(model, support_feats, support_labels, sv_idx: int) -> torch.Tensor:
    mask = support_labels.squeeze(0) == sv_idx
    return support_feats.squeeze(0)[mask].mean(dim=0)


def forward_head_dense(
        model,
        feat_x,
        support_feats,
        support_labels,
        bargs,
        repair: Optional[RepairStack] = None,
        return_visual: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], Optional[List[torch.Tensor]]]:
    """Returns per-level cls logits (pre-sigmoid), bbox, angle; optional visual feats."""
    head = model.bbox_head
    kwargs = dict(
        support_shot=bargs.support_shot,
        num_classes=len(bargs.class_names) if hasattr(bargs, 'class_names') else 15,
        num_in_classes=len(bargs.class_names) if hasattr(bargs, 'class_names') else 15,
        align_style='labelled',
        support_type=bargs.support_type,
        text_cls_scale=0.0,
    )
    if hasattr(bargs, 'class_names'):
        kwargs['num_classes'] = len(bargs.class_names)
        kwargs['num_in_classes'] = len(bargs.class_names)

    if (repair is not None and getattr(repair, 'rotation_adapter', None) is not None
            and apply_rotation_adapter_to_feats is not None):
        feat_x = apply_rotation_adapter_to_feats(feat_x, repair.rotation_adapter)

    outs_all = head(feat_x, support_feats, support_labels, support_labels, **kwargs)
    cls_scores = list(outs_all[0])
    bbox_preds = outs_all[1]
    angle_preds = outs_all[2]

    if repair is None:
        logits_list = [c for c in cls_scores]
        return logits_list, list(bbox_preds), list(angle_preds), None

  # Re-forward with hooks is heavy; apply repair on logits after forward
    logits_list = []
    visual_list = [] if return_visual else None
    sv_dir = None
    if repair is not None and (repair.fixed_alpha is not None or repair.antihub_alpha is not None
                               or repair.antihub_gate is not None or repair.lowrank is not None):
        sv_dir = sv_support_direction(model, support_feats, support_labels, repair.sv_idx)

    # Need per-level pred_embed: call head internals via second partial forward
    # Simpler: apply repair only on cls_scores tensors (logits before sigmoid in head are cls_scores)
    for lvl, cls_logit in enumerate(cls_scores):
        logit = cls_logit
        if return_visual and visual_list is not None:
            # approximate visual from logit path - skip unless needed
            visual_list.append(None)
        if repair is not None:
            b, c, h, w = logit.shape
            flat = logit.permute(0, 2, 3, 1).reshape(-1, c)
            if repair.grid_sv_bias is not None and repair.calibration is None:
                bias = torch.zeros(c, device=flat.device, dtype=flat.dtype)
                bias[repair.sv_idx] = repair.grid_sv_bias
                flat = flat + bias
            if repair.calibration is not None:
                flat = repair.calibration(flat)
            if repair.neg_bank is not None:
                # use pred_embed proxy: inverse not available; use flat as pseudo visual
                flat = repair.neg_bank.adjust_logits(flat, flat[:, :repair.neg_bank.prototypes.shape[1]])
            logit = flat.reshape(b, h, w, c).permute(0, 3, 1, 2)
        logits_list.append(logit)

    return logits_list, list(bbox_preds), list(angle_preds), visual_list


def logits_to_scores(logits_list: List[torch.Tensor]) -> List[torch.Tensor]:
    return [l.sigmoid() for l in logits_list]


def dense_stats_from_logits(logits_list: List[torch.Tensor], sv_idx: int) -> Dict[str, float]:
    parts = []
    for logit in logits_list:
        scores = logit.sigmoid()
        flat = scores.permute(0, 2, 3, 1).reshape(-1, scores.shape[1])
        parts.append(flat)
    cat = torch.cat(parts, dim=0)
    top1 = cat.argmax(dim=1)
    margin = cat.topk(2, dim=1).values
    m = (margin[:, 0] - margin[:, 1]) if margin.shape[1] > 1 else margin[:, 0]
    return dict(
        dense_top1_sv_ratio=float((top1 == sv_idx).float().mean().item()),
        mean_sv_score=float(cat[:, sv_idx].mean().item()),
        mean_margin=float(m.mean().item()),
        num_locations=int(cat.shape[0]),
    )


def final_detection_from_outs(model, outs, metas, score_thr: float = 0.01):
    from tools.rotation_overnight_gpu89 import common as oc
    row = oc.post_summary(0, 'repair', model.bbox_head, outs, metas)
    import json
    from tools.sv_attractor_repair_gpu89.common import parse_class_histogram
    hist = parse_class_histogram(row['class_histogram'])
    return dict(
        detection_total=row['detection_total'],
        final_sv_ratio=row['small_vehicle_ratio'],
        final_lv_ratio=row['large_vehicle_ratio'],
        mean_score=row['mean_score'],
        top1_class=row['top1_class'],
        class_histogram=hist,
    )
