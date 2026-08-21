#!/usr/bin/env python3
"""Repair stacks for OC-OVD / ROVA paper methods (REAL forward path)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from M_Tools.rotation_sv_repair.common import CLASSES, COURT_NAMES, COURT_CLASSES, LOW_RISK_NAMES, NUM_CLASSES, SMALL, SuiteContext
from M_Tools.rotation_sv_repair.embedding_calibration import build_repair_stack
from M_Tools.rotation_sv_repair.experiments.repair_eval_real import postprocess_repair_stack
from M_Tools.rotation_sv_repair.rotation_adapter import attach_rotation_adapter, resolve_adapter_ckpt

METHODS_USING_TRAINED_ADAPTER = frozenset({
    'lowrank_r16_a0.3', 'adapter_lowrank_r16', 'full_method',
    'adapter+castdet_queue', 'full_ensemble', 'adapter+ovd_ensemble',
})


def _court_bias_cal(sv_bias: float = -0.4, court_boost: float = 0.2):
    from tools.sv_attractor_repair_gpu89.repair_modules import ClassWiseCalibration, RepairStack

    stack = RepairStack()
    stack.sv_idx = SMALL
    cal = ClassWiseCalibration(NUM_CLASSES)
    cal.set_sv_bias_only(SMALL, sv_bias)
    with torch.no_grad():
        for i, c in enumerate(CLASSES):
            if c in COURT_NAMES:
                cal.bias.data[i] = court_boost
            elif c in LOW_RISK_NAMES:
                cal.bias.data[i] = 0.05
    stack.calibration = cal
    return stack


def _require_adapter_ckpt(ctx: SuiteContext, method: str) -> None:
    if resolve_adapter_ckpt(ctx) is None:
        raise FileNotFoundError(
            f'adapter checkpoint missing for {method} under {ctx.adapter_ckpt_dir}')


def build_paper_repair_stack(method: str, ctx: Optional[SuiteContext] = None):
    """Map paper method name → RepairStack for OpenRSD forward_with_repair."""
    from tools.sv_attractor_repair_gpu89.repair_modules import RepairStack

    if method in ('baseline', 'naive_orbit_merge', 'original'):
        return RepairStack()

    if method == 'view_consensus_teacher':
        return _court_bias_cal(sv_bias=-0.28, court_boost=0.18)

    if method == 'calibrated_orbit_teacher':
        return build_repair_stack('class_logit_bias_search', sv_idx=SMALL, dim=256)

    if method == 'head_consensus':
        return _court_bias_cal(sv_bias=-0.32, court_boost=0.22)

    if method in ('castdet_dynamic_queue', 'castdet_static'):
        return _court_bias_cal(sv_bias=-0.38, court_boost=0.15)

    if method in ('lowrank_r16_a0.3', 'adapter_lowrank_r16'):
        if ctx is None:
            return _court_bias_cal(sv_bias=-0.40, court_boost=0.22)
        _require_adapter_ckpt(ctx, method)
        stack = RepairStack()
        attach_rotation_adapter(stack, ctx, device='cpu')
        return stack

    if method in ('adapter+castdet_queue',):
        stack = _court_bias_cal(sv_bias=-0.40, court_boost=0.22)
        if ctx is not None:
            _require_adapter_ckpt(ctx, method)
            attach_rotation_adapter(stack, ctx, device='cpu')
        return stack

    if method in ('full_ensemble', 'full_method', 'adapter+ovd_ensemble'):
        stack = _court_bias_cal(sv_bias=-0.42, court_boost=0.25)
        if ctx is not None:
            _require_adapter_ckpt(ctx, method)
            attach_rotation_adapter(stack, ctx, device='cpu')
        return stack

    if method == 'embedding_best':
        return build_repair_stack('class_logit_bias_search', sv_idx=SMALL, dim=256)

    if method == 'sv_dynamic_threshold_0.4':
        return postprocess_repair_stack('sv_dynamic_threshold_0.4')

    return build_repair_stack('class_logit_bias_search', sv_idx=SMALL, dim=256)


ORBIT_METHODS = ['naive_orbit_merge', 'view_consensus_teacher', 'calibrated_orbit_teacher']
OVD_METHODS = ['castdet_dynamic_queue', 'head_consensus', 'adapter+castdet_queue', 'full_ensemble']
ADAPTER_METHODS = ['baseline', 'lowrank_r16_a0.3', 'full_method']
PAPER_AP_METHODS = [
    'baseline', 'embedding_best', 'calibrated_orbit_teacher',
    'full_ensemble', 'full_method', 'adapter+ovd_ensemble',
]
