#!/usr/bin/env python3
"""Non-deployable oracle upper bounds for repair potential."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from M_Tools.rotation_sv_repair.common import CLASSES, SMALL


def oracle_best_view(predictions_by_angle: Dict[int, List[dict]], gt_class: int) -> dict:
    best_score = -1.0
    best = None
    for ang, plist in predictions_by_angle.items():
        for p in plist:
            if p.get('class_id') == gt_class:
                s = float(p.get('score', 0))
                if s > best_score:
                    best_score = s
                    best = dict(angle=ang, **p)
    return best or {}


def oracle_best_prompt(prompt_scores: Dict[str, float]) -> str:
    return max(prompt_scores, key=prompt_scores.get) if prompt_scores else ''


def oracle_sv_filter_with_gt(dets: List[dict], gt_boxes: List[dict], iou_thr: float = 0.5) -> List[dict]:
    """Keep SV only when matched to GT SV."""
    from M_Tools.rotation_sv_repair.orbit_distillation import iou_xyxy
    kept = []
    for d in dets:
        if d.get('class_id') != SMALL:
            kept.append(d)
            continue
        box = np.array(d['bbox'])
        match = False
        for g in gt_boxes:
            if g.get('class_id') != SMALL:
                continue
            if iou_xyxy(box, np.array(g['bbox'])) >= iou_thr:
                match = True
                break
        if match:
            kept.append(d)
    return kept
