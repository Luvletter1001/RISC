#!/usr/bin/env python
"""Unified metrics for full repair experiments."""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from tools.rotation_diagnostics import probe_rotated_stage_outputs as probe_base
from tools.sv_attractor_repair_gpu89 import common as C
from tools.sv_attractor_repair_gpu89 import criteria as CRIT
from tools.sv_attractor_repair_gpu89.eval_ap import infer_heldout_method, prepare_heldout_subset
from tools.sv_attractor_repair_gpu89.exp_combined import load_repair_stack
from tools.sv_attractor_repair_gpu89.exp_r3 import apply_fixed_alpha_support
from tools.sv_attractor_repair_gpu89.repair_forward import (
    build_support_tensors, dense_stats_from_logits, final_detection_from_outs)
from tools.sv_attractor_repair_gpu89.repair_logits import apply_repair_logits as _apply_repair_logits
from tools.sv_attractor_repair_gpu89.repair_modules import RepairStack
from tools.verify_sv_attractor_gpu89 import common as verify


def class_entropy_from_hist(hist: dict) -> float:
    total = sum(hist.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in C.CLASSES:
        p = hist.get(c, 0) / total
        if p > 0:
            ent -= p * math.log(p + 1e-12)
    return float(ent)


def eval_p0148_rows(ctx, model, bargs, device, det_support, name2id,
                    method: str, angles: Optional[List[int]] = None) -> List[dict]:
    angles = angles or ctx.angles
    rep, sf_mod = _method_stack(ctx, device, method)
    image_dir = verify.tile_image_dir(ctx.repo_root, C.DEFAULT_TILE)
    lb = SimpleNamespace(**vars(bargs))
    lb.image_dir = str(image_dir)
    lb.out_dir = str(ctx.work_dir / '_tmp')
    loader = probe_base.build_dataloader(lb, angles)
    sf, sl = build_support_tensors(model, det_support, name2id, device, 'visual', 8)
    rows = []
    with torch.no_grad():
        for data_info in loader:
            angle = probe_base.angle_from_name(data_info['data_samples'][0].img_path)
            data = model.data_preprocessor(data_info, False)
            data['inputs'] = data['inputs'].to(device)
            x = model.prompt_extract_feats(data['inputs'])
            metas = [s.metainfo for s in data['data_samples']]
            sf_use = sf_mod(sf, sl, sv_support_direction(model, sf, sl, C.SMALL)) if sf_mod else sf
            outs_all = model.bbox_head(
                x, sf_use, sl, sl, support_shot=8, num_classes=C.NUM_CLASSES,
                num_in_classes=C.NUM_CLASSES, align_style='labelled',
                support_type='visual', text_cls_scale=0.0)
            cls_adj = _apply_repair_logits(outs_all, rep)
            outs = (tuple(cls_adj), outs_all[1], outs_all[2])
            dst = dense_stats_from_logits(cls_adj, C.SMALL)
            fin = final_detection_from_outs(model, outs, metas)
            hist = C.parse_class_histogram(fin['class_histogram'])
            rows.append(dict(
                method=method, split='P0148', tile_id=C.DEFAULT_TILE, angle=angle,
                dense_top1_sv_ratio=dst['dense_top1_sv_ratio'],
                final_sv_ratio=fin['final_sv_ratio'],
                final_lv_ratio=fin['final_lv_ratio'],
                final_ship_ratio=float(hist.get('ship', 0) / max(fin['detection_total'], 1)),
                detection_total=fin['detection_total'],
                class_entropy=class_entropy_from_hist(hist),
            ))
    return rows


def sv_support_direction(model, sf, sl, idx):
    from tools.sv_attractor_repair_gpu89.repair_forward import sv_support_direction as _sv
    return _sv(model, sf, sl, idx)


def _method_stack(ctx, device, method: str):
    if method in ('baseline', 'C0_baseline'):
        return RepairStack(), None
    tag = {
        'best_R1': 'R1', 'C1_best_R1': 'R1',
        'best_R2': 'R2', 'best_R2B': 'R2B', 'C2_best_R2B_visual': 'R2B',
        'best_R3': 'R3', 'C3_best_R3': 'R3',
        'best_combined': ctx.progress.get('best_combined_method', 'C5_R1+R3'),
    }.get(method, method)
    rep = load_repair_stack(ctx, device, tag)
    sf_mod = None
    if 'R3' in tag or method in ('best_R3', 'C3_best_R3'):
        sf_mod = lambda sf, sl, sv: apply_fixed_alpha_support(sf, sl, sv, 0.5)
    return rep, sf_mod


def eval_cross_tile_rows(ctx, model, bargs, device, det_support, name2id,
                         method: str, tiles: List[str],
                         angles: Optional[List[int]] = None) -> List[dict]:
    angles = angles or ([0, 90, 180, 270] if ctx.mode == 'full' else [0, 90])
    rows = []
    for tid in tiles:
        try:
            image_dir = verify.tile_image_dir(ctx.repo_root, tid)
        except FileNotFoundError:
            continue
        lb = SimpleNamespace(**vars(bargs))
        lb.image_dir = str(image_dir)
        rep, sf_mod = _method_stack(ctx, device, method)
        loader = probe_base.build_dataloader(lb, angles)
        sf, sl = build_support_tensors(model, det_support, name2id, device, 'visual', 8)
        with torch.no_grad():
            for data_info in loader:
                angle = probe_base.angle_from_name(data_info['data_samples'][0].img_path)
                data = model.data_preprocessor(data_info, False)
                data['inputs'] = data['inputs'].to(device)
                x = model.prompt_extract_feats(data['inputs'])
                metas = [s.metainfo for s in data['data_samples']]
                sf_use = sf_mod(sf, sl, sv_support_direction(model, sf, sl, C.SMALL)) if sf_mod else sf
                outs_all = model.bbox_head(
                    x, sf_use, sl, sl, support_shot=8, num_classes=C.NUM_CLASSES,
                    num_in_classes=C.NUM_CLASSES, align_style='labelled',
                    support_type='visual', text_cls_scale=0.0)
                cls_adj = _apply_repair_logits(outs_all, rep)
                outs = (tuple(cls_adj), outs_all[1], outs_all[2])
                dst = dense_stats_from_logits(cls_adj, C.SMALL)
                fin = final_detection_from_outs(model, outs, metas)
                rows.append(dict(
                    method=method, split='cross_tile', tile_id=tid, angle=angle,
                    dense_top1_sv_ratio=dst['dense_top1_sv_ratio'],
                    final_sv_ratio=fin['final_sv_ratio'],
                    final_lv_ratio=fin['final_lv_ratio'],
                    detection_total=fin['detection_total'],
                ))
    return rows


def eval_heldout_ap(ctx, model, device, det_support, name2id, method: str,
                    max_images: int) -> Dict[str, Any]:
    subset = prepare_heldout_subset(ctx, max_images)
    try:
        return infer_heldout_method(
            ctx, model, device, det_support, name2id, subset, method,
            angles=[0], max_images=max_images)
    except Exception as exc:
        return dict(method=method, error=str(exc), ap50_overall=float('nan'))


def full_method_report(ctx, model, bargs, device, det_support, name2id,
                       method: str, heldout_n: Optional[int] = None) -> dict:
    heldout_n = heldout_n or ctx.heldout_max_images
    p0148 = eval_p0148_rows(ctx, model, bargs, device, det_support, name2id, method)
    ap = eval_heldout_ap(ctx, model, device, det_support, name2id, method, heldout_n)
    row = dict(
        method=method,
        P0148_final_sv=C.mean_key(p0148, 'final_sv_ratio'),
        P0148_dense_sv=C.mean_key(p0148, 'dense_top1_sv_ratio'),
        P0148_det_total=C.mean_key(p0148, 'detection_total'),
        P0148_lv_ratio=C.mean_key(p0148, 'final_lv_ratio'),
        P0148_ship_ratio=C.mean_key(p0148, 'final_ship_ratio'),
        class_entropy=C.mean_key(p0148, 'class_entropy'),
        ap50_overall=ap.get('ap50_overall', float('nan')),
        sv_ap50=ap.get('sv_ap50', float('nan')),
        lv_ap50=ap.get('lv_ap50', float('nan')),
        ship_ap50=ap.get('ship_ap50', float('nan')),
        heldout_detection_total=ap.get('detection_total', float('nan')),
        per_class=ap.get('per_class', []),
    )
    if ap.get('error'):
        row['ap_error'] = ap['error']
    return row


def check_success_criteria(baseline: dict, candidate: dict) -> dict:
    b_sv = float(baseline.get('P0148_final_sv', 1))
    c_sv = float(candidate.get('P0148_final_sv', 1))
    b_ap = float(baseline.get('ap50_overall', 0))
    c_ap = float(candidate.get('ap50_overall', 0))
    b_sv_ap = float(baseline.get('sv_ap50', 0))
    c_sv_ap = float(candidate.get('sv_ap50', 0))
    b_det = float(baseline.get('P0148_det_total', 1))
    c_det = float(candidate.get('P0148_det_total', 0))
    lv = float(candidate.get('P0148_lv_ratio', 0))
    ship = float(candidate.get('P0148_ship_ratio', 0))
    sv_drop = (b_sv - c_sv) / max(b_sv, 1e-6)
    checks = dict(
        p0148_sv_drop_ok=sv_drop >= CRIT.FULL_SUCCESS_CRITERIA['p0148_final_sv_ratio_drop_min'],
        p0148_sv_drop_pct=sv_drop * 100,
        ap50_ok=(b_ap - c_ap) <= CRIT.FULL_SUCCESS_CRITERIA['overall_ap50_drop_max_points'],
        ap50_drop_pt=(b_ap - c_ap) * 100,
        sv_ap50_ok=(b_sv_ap - c_sv_ap) <= CRIT.FULL_SUCCESS_CRITERIA['small_vehicle_ap50_drop_max_points'],
        sv_ap50_drop_pt=(b_sv_ap - c_sv_ap) * 100,
        no_lv_flood=lv <= CRIT.FULL_SUCCESS_CRITERIA['large_vehicle_flood_max_ratio'],
        no_ship_flood=ship <= CRIT.FULL_SUCCESS_CRITERIA['ship_flood_max_ratio'],
        no_collapse=c_det >= b_det * CRIT.FULL_SUCCESS_CRITERIA['detection_total_collapse_min_ratio'],
    )
    checks['pass_all'] = all([
        checks['p0148_sv_drop_ok'], checks['ap50_ok'], checks['sv_ap50_ok'],
        checks['no_lv_flood'], checks['no_ship_flood'], checks['no_collapse'],
    ])
    return checks
