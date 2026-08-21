#!/usr/bin/env python3
"""Embedding / prototype / logit-bias calibration for SV attractor repair."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from M_Tools.rotation_sv_repair.common import CLASSES, COURT_CLASSES, SMALL, SuiteContext
from tools.sv_attractor_repair_gpu89.repair_modules import ClassWiseCalibration, NegativePrototypeBank, RepairStack

COURT_IDX = COURT_CLASSES

EMBEDDING_METHODS = [
    'l2_normalize_all', 'sv_norm_to_mean', 'sv_temperature_scaling', 'class_temperature_vector',
    'tennis_sv_margin_boost', 'background_hard_negative_centroid', 'support_prototype_smoothing',
    'rotation_averaged_class_prototype', 'class_logit_bias_search', 'court_vs_sv_margin_filter',
]

BIAS_GRID_SV = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0]
BIAS_GRID_COURT = [0.0, 0.1, 0.2, 0.3, 0.4]
MARGIN_GRID = [0.05, 0.10, 0.15, 0.20]


def build_repair_stack(method: str, sv_idx: int = SMALL, dim: int = 256) -> RepairStack:
    stack = RepairStack()
    stack.sv_idx = sv_idx
    stack.embed_dim = dim
    if method == 'class_logit_bias_search':
        stack.calibration = ClassWiseCalibration(len(CLASSES), train_temperature=False)
        stack.calibration.set_sv_bias_only(sv_idx, -0.4)
    elif method == 'tennis_sv_margin_boost':
        stack.calibration = ClassWiseCalibration(len(CLASSES))
        with torch.no_grad():
            stack.calibration.bias[COURT_IDX[0]] = 0.2
            stack.calibration.bias[sv_idx] = -0.3
    elif method == 'background_hard_negative_centroid':
        stack.neg_bank = NegativePrototypeBank(dim, num_proto=8)
    elif method == 'sv_temperature_scaling':
        stack.calibration = ClassWiseCalibration(len(CLASSES), train_temperature=True)
        with torch.no_grad():
            stack.calibration.bias[sv_idx] = -0.2
    elif method == 'court_vs_sv_margin_filter':
        stack.grid_sv_bias = -0.2
    else:
        stack.grid_sv_bias = 0.0
    return stack


def cosine_margin_matrix(support_feats: torch.Tensor, labels: torch.Tensor, num_classes: int) -> np.ndarray:
    protos = []
    for c in range(num_classes):
        m = labels == c
        if m.any():
            protos.append(F.normalize(support_feats[m].mean(0), dim=0))
        else:
            protos.append(torch.zeros(support_feats.shape[-1], device=support_feats.device))
    P = torch.stack(protos)
    sim = (P @ P.t()).detach().cpu().numpy()
    return sim


def search_best_bias(cal_rows: List[dict]) -> dict:
    if not cal_rows:
        return {}
    best = min(cal_rows, key=lambda r: float(r.get('P0148_dense_sv', 1e9)))
    return best
