#!/usr/bin/env python3
"""Postprocess-only SV repair baselines (not the final paper method)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from M_Tools.rotation_sv_repair.common import CLASSES, COURT_NAMES, LOW_RISK_NAMES, SMALL, SuiteContext


def apply_sv_score_penalty(scores: torch.Tensor, labels: torch.Tensor, penalty: float) -> torch.Tensor:
    out = scores.clone()
    mask = labels == SMALL
    out[mask] = out[mask] * penalty
    return out


def apply_sv_dynamic_threshold(scores: torch.Tensor, labels: torch.Tensor, thr: float) -> torch.Tensor:
    out = scores.clone()
    mask = (labels == SMALL) & (scores < thr)
    out[mask] = 0.0
    return out


def apply_sv_margin_filter(scores: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    top2 = scores.topk(2, dim=-1).values
    m = top2[:, 0] - top2[:, 1]
    out = scores.clone()
    sv_mask = labels == SMALL
    out[sv_mask & (m < margin)] = 0.0
    return out


def per_class_topk(scores: torch.Tensor, labels: torch.Tensor, k: int = 100) -> torch.Tensor:
    out = torch.zeros_like(scores)
    for c in range(scores.shape[1]):
        idx = labels == c
        if not idx.any():
            continue
        s = scores[idx, c]
        kk = min(k, s.numel())
        thr = s.topk(kk).values.min() if kk > 0 else 0
        keep = idx & (scores[:, c] >= thr)
        out[keep, c] = scores[keep, c]
    return out


def context_suppression_geometry(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
                               court_idx: Tuple[int, ...], sv_idx: int = SMALL) -> np.ndarray:
    """Downweight SV in large low-aspect court-like regions (statistical, not tile-specific)."""
    out = scores.copy()
    if boxes.shape[0] == 0:
        return out
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    wh = np.maximum(boxes[:, 2] - boxes[:, 0], 1e-3) / np.maximum(boxes[:, 3] - boxes[:, 1], 1e-3)
    large = areas > np.percentile(areas, 75) if len(areas) > 4 else areas > areas.mean()
    court_present = any((labels == c).any() for c in court_idx)
    for i in range(len(labels)):
        if labels[i] != sv_idx:
            continue
        if large[i] and wh[i] > 2.5 and court_present:
            out[i] *= 0.5
    return out


POSTPROCESS_METHODS = {
    'baseline': {},
    'per_class_topk_100': dict(topk=100),
    'sv_score_penalty_0.85': dict(penalty=0.85),
    'sv_score_penalty_0.70': dict(penalty=0.70),
    'sv_dynamic_threshold_0.4': dict(sv_thr=0.4),
    'sv_margin_filter_0.10': dict(margin=0.10),
    'context_suppression': dict(context=True),
    'low_risk_safe_mode': dict(safe_low_risk=True),
}


def summarize_postprocess_delta(before: dict, after: dict) -> dict:
    return dict(
        sv_ratio_drop=float(before.get('final_sv_ratio', 0)) - float(after.get('final_sv_ratio', 0)),
        dense_sv_drop=float(before.get('dense_top1_sv_ratio', 0)) - float(after.get('dense_top1_sv_ratio', 0)),
        final_sv_drop=float(before.get('final_sv_count', 0)) - float(after.get('final_sv_count', 0)),
        true_sv_preserve_rate=after.get('true_sv_preserve_rate', 'NA'),
        low_risk_drift=after.get('low_risk_drift', 'NA'),
    )
