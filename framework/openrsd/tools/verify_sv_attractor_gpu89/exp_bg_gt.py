#!/usr/bin/env python
"""Step 4: background / GT stratification (P3, P6 partial). GPU 9."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from commonlibs.common_tools import pklload
from tools.rotation_diagnostics import probe_rotated_stage_outputs as probe_base
from tools.verify_sv_attractor_gpu89 import common as C


def poly_to_xyxy(poly):
    pts = np.asarray(poly, dtype=float).reshape(-1, 2)
    return [float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max())]


def point_in_box(px, py, box):
    return box[0] <= px <= box[2] and box[1] <= py <= box[3]


def iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return float(inter / (aa + bb - inter + 1e-12))


def run(ctx: C.RunContext) -> dict:
    if ctx.blocked():
        return dict(status='BLOCKED')

    out = ctx.exp_dir('exp_04_bg_gt')
    fres = ctx.fres_path('fres_04_background_gt_stratification.md')
    tile = C.DEFAULT_TILE
    image_dir = C.tile_image_dir(ctx.repo_root, tile)
    ann_dir = C.tile_ann_dir(ctx.repo_root, tile)
    angle = 0
    ann_path = ann_dir / f'{tile}_rot{angle:03d}.pkl'
    img_path = image_dir / f'{tile}_rot{angle:03d}.jpg'

    gt_boxes, gt_classes = [], []
    ann_note = ''
    if ann_path.exists():
        ann = pklload(str(ann_path))
        for text, poly in zip(ann['texts'], ann['polys']):
            gt_boxes.append(poly_to_xyxy(poly))
            gt_classes.append(text)
    else:
        ann_note = f'ann not found: {ann_path}'

    bargs, model, device, det_support, name2id, id2name, _ = C.build_model(ctx, 'visual')
    loader = C.build_loader(ctx, image_dir, [angle])
    region_stats = {k: [] for k in [
        'inside_any_gt', 'outside_gt', 'near_border', 'inside_sv_gt']}

    det_audit = []
    with torch.no_grad():
        sf, sl = C.build_mapped_support(
            model, ctx, det_support, name2id, device, 'visual', 'original')
        for data_info in loader:
            data = model.data_preprocessor(data_info, False)
            data['inputs'] = data['inputs'].to(device)
            h, w = data['data_samples'][0].img_shape
            x = model.prompt_extract_feats(data['inputs'])
            metas = [s.metainfo for s in data['data_samples']]
            outs = C.forward_dense(model, x, sf, sl, bargs)
            levels = C.decode_dense(
                model.bbox_head, outs[0], outs[1], outs[2], metas[0]['img_shape'])
            stats = C.dense_stats_from_levels(levels)
            scores = stats['scores']
            top1 = scores.argmax(dim=1)
            # grid centers approximate from index (coarse)
            n = scores.shape[0]
            side = int(np.sqrt(n))
            for idx in range(min(n, side * side)):
                r, c = idx // side, idx % side
                px = (c + 0.5) / side * w
                py = (r + 0.5) / side * h
                in_gt = any(point_in_box(px, py, b) for b in gt_boxes)
                in_sv = any(point_in_box(px, py, b) and cl == 'small-vehicle'
                            for b, cl in zip(gt_boxes, gt_classes))
                near_border = px < 32 or py < 32 or px > w - 32 or py > h - 32
                key = 'inside_sv_gt' if in_sv else (
                    'inside_any_gt' if in_gt else (
                        'near_border' if near_border else 'outside_gt'))
                region_stats[key].append(int(top1[idx] == C.SMALL))

            pred = model.bbox_head.predict_by_feat(
                *outs, batch_img_metas=metas, rescale=True, with_nms=True)[0]
            image = cv2.imread(str(img_path)) if img_path.exists() else None
            crop_dir = out / 'crops'
            crop_dir.mkdir(exist_ok=True)
            sv_indices = [i for i, l in enumerate(pred.labels.tolist()) if l == C.SMALL]
            for rank, i in enumerate(sv_indices):
                box = C.bbox_row(pred, i)
                x_c, y_c, bw, bh = box[:4]
                xyxy = [x_c - bw / 2, y_c - bh / 2, x_c + bw / 2, y_c + bh / 2]
                best_any, best_sv = 0.0, 0.0
                for b, cl in zip(gt_boxes, gt_classes):
                    v = iou_xyxy(xyxy, b)
                    best_any = max(best_any, v)
                    if cl == 'small-vehicle':
                        best_sv = max(best_sv, v)
                score = float(pred.scores[i])
                crop_path = ''
                if image is not None:
                    x1, y1, x2, y2 = [int(max(0, v)) for v in xyxy]
                    pad = 16
                    cy1, cy2 = max(0, y1 - pad), min(image.shape[0], y2 + pad)
                    cx1, cx2 = max(0, x1 - pad), min(image.shape[1], x2 + pad)
                    crop_path = str(crop_dir / (
                        f'sv_{rank:04d}_score_{score:.3f}_iou_sv_{best_sv:.2f}_'
                        f'iou_any_{best_any:.2f}.jpg'))
                    crop = image[cy1:cy2, cx1:cx2]
                    if crop.size:
                        cv2.imwrite(crop_path, crop)
                det_audit.append(dict(
                    image_id=tile, angle=angle, det_id=rank, score=score,
                    bbox=json.dumps(xyxy), iou_any_gt=best_any, iou_sv_gt=best_sv,
                    crop_path=crop_path, false_positive_type='',
                    is_true_vehicle='', notes=''))

    strat_rows = []
    for region, vals in region_stats.items():
        if not vals:
            continue
        strat_rows.append(dict(
            region=region, dense_sv_ratio=float(np.mean(vals)),
            mean_sv_score='', mean_margin='', sample_count=len(vals)))

    iou_rows = []
    for thr in [0.1, 0.3, 0.5]:
        matched_sv = sum(1 for r in det_audit if r['iou_sv_gt'] >= thr)
        matched_any = sum(1 for r in det_audit if r['iou_any_gt'] >= thr)
        iou_rows.append(dict(
            iou_thr=thr, sv_det_count=len(det_audit),
            matched_sv_gt_count=matched_sv, matched_any_gt_count=matched_any,
            unmatched_count=len(det_audit) - matched_any))

    C.write_csv(out / 'ftable_dense_gt_stratification.csv', strat_rows)
    C.write_csv(out / 'ftable_gt_match_summary.csv', iou_rows)
    C.write_csv(out / 'manual_audit_template.csv', det_audit[:300])

    outside_sv = next((r['dense_sv_ratio'] for r in strat_rows
                       if r['region'] == 'outside_gt'), float('nan'))
    p3 = 'SUPPORTED' if outside_sv > 0.5 and len(det_audit) > 0 else 'PARTIAL'
    p6 = 'PARTIAL'  # border vs outside compared in strat table

    with open(fres, 'w') as f:
        C.write_fres_header(f, 'Background / GT Stratification (P3, P6)', ctx)
        if ann_note:
            f.write(f'- ann note: {ann_note}\n\n')
        f.write(f'## P3 Verdict: **{p3}**\n\n')
        f.write(f'## P6 (border) Verdict: **{p6}** — compare outside_gt vs near_border\n\n')
        f.write('| region | dense_sv_ratio | sample_count |\n|---|---:|---:|\n')
        for r in strat_rows:
            f.write(f"| {r['region']} | {r['dense_sv_ratio']:.4f} | {r['sample_count']} |\n")
        f.write('\n| iou_thr | sv_dets | matched_sv | matched_any | unmatched |\n')
        f.write('|---:|---:|---:|---:|---:|\n')
        for r in iou_rows:
            f.write(
                f"| {r['iou_thr']} | {r['sv_det_count']} | {r['matched_sv_gt_count']} | "
                f"{r['matched_any_gt_count']} | {r['unmatched_count']} |\n")
        f.write('\n**Note**: automatic GT proxy is not final; manual_audit_template.csv required.\n')
        f.write(f'\nCrops exported: up to `{min(300, len(det_audit))}` under `{out / "crops"}`\n')

    ctx.mark_step('bg_gt', 'OK', p3=p3)
    return dict(status='OK', p3=p3, fres=str(fres))
