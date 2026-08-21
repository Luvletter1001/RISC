#!/usr/bin/env python
"""Step 2: dense logit attractor (P1). GPU 8."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tools.rotation_diagnostics import probe_rotated_stage_outputs as probe_base
from tools.verify_sv_attractor_gpu89 import common as C


def _pre_nms_rows(angle, scores, boxes, thresholds):
    flat = scores.reshape(-1)
    labels = torch.arange(scores.shape[1], device=scores.device).repeat(scores.shape[0])
    rows = []
    for thr in thresholds:
        keep = flat >= thr
        labs = labels[keep]
        total = int(keep.sum().item())
        small = int((labs == C.SMALL).sum().item())
        rows.append(dict(
            angle=angle, score_thr=thr, candidate_total=total,
            pre_nms_sv_ratio=float(small / total) if total else 0.0))
    return rows


def _topk_rows(angle, scores, ks):
    flat = scores.reshape(-1)
    labels = torch.arange(scores.shape[1], device=scores.device).repeat(scores.shape[0])
    rows = []
    for k in ks:
        kk = min(k, flat.numel())
        vals, idx = torch.topk(flat, kk)
        labs = labels[idx]
        small = int((labs == C.SMALL).sum().item())
        rows.append(dict(
            angle=angle, mode='no_nms_global_topk', topk=kk,
            no_nms_sv_ratio=float(small / kk) if kk else 0.0))
    return rows


def run(ctx: C.RunContext) -> dict:
    if ctx.blocked():
        return dict(status='BLOCKED')

    out = ctx.exp_dir('exp_02_dense')
    fres = ctx.fres_path('fres_02_dense_logit_attractor.md')
    image_dir = C.tile_image_dir(ctx.repo_root, C.DEFAULT_TILE)
    bargs, model, device, det_support, name2id, _, _ = C.build_model(ctx, 'visual')
    loader = C.build_loader(ctx, image_dir, ctx.angles)

    by_angle, by_level, hist_rows, topk_ex = [], [], [], []
    thresholds = [0.001, 0.01, 0.05, 0.1, 0.3]
    topks = [100, 300, 1000, 5000]

    with torch.no_grad():
        sf, sl = C.build_mapped_support(
            model, ctx, det_support, name2id, device, 'visual', 'original')
        for data_info in loader:
            img_path = data_info['data_samples'][0].img_path
            angle = probe_base.angle_from_name(img_path)
            if angle not in ctx.angles:
                continue
            data = model.data_preprocessor(data_info, False)
            data['inputs'] = data['inputs'].to(device)
            x = model.prompt_extract_feats(data['inputs'])
            metas = [s.metainfo for s in data['data_samples']]
            outs = C.forward_dense(model, x, sf, sl, bargs)
            levels = C.decode_dense(
                model.bbox_head, outs[0], outs[1], outs[2], metas[0]['img_shape'])
            stats = C.dense_stats_from_levels(levels)
            npz = out / 'npz' / f'rot{angle:03d}.npz'
            npz.parent.mkdir(parents=True, exist_ok=True)
            npz_data = {}
            for lvl, logits, scores, boxes in levels:
                npz_data[f'level{lvl}_scores'] = scores.cpu().numpy().astype(np.float16)
            np.savez_compressed(npz, **npz_data)

            pre_rows = _pre_nms_rows(angle, stats['scores'], None, thresholds)
            tk_rows = _topk_rows(angle, stats['scores'], topks)
            for pr in pre_rows:
                pr['dense_top1_sv_ratio'] = stats['dense_top1_sv_ratio']
            for tr in tk_rows:
                tr['dense_top1_sv_ratio'] = stats['dense_top1_sv_ratio']

            by_angle.append(dict(
                angle=angle,
                dense_top1_sv_ratio=stats['dense_top1_sv_ratio'],
                pre_nms_sv_ratio_001=next(
                    (r['pre_nms_sv_ratio'] for r in pre_rows if r['score_thr'] == 0.01), 0),
                no_nms_top1000_sv_ratio=next(
                    (r['no_nms_sv_ratio'] for r in tk_rows if r['topk'] == 1000), 0),
                mean_sv_score=stats['mean_sv_score'],
                mean_margin=stats['mean_margin'],
            ))
            for lr in stats['rows']:
                lr['angle'] = angle
                by_level.append(lr)
            hist = {}
            top1 = stats['scores'].argmax(dim=1).cpu().numpy()
            for c in range(C.NUM_CLASSES):
                hist[C.CLASSES[c]] = int((top1 == c).sum())
            hist_rows.append(dict(angle=angle, **{f'count_{k}': v for k, v in hist.items()}))

    C.write_csv(out / 'ftable_dense_by_angle.csv', by_angle,
                ['angle', 'dense_top1_sv_ratio', 'pre_nms_sv_ratio_001',
                 'no_nms_top1000_sv_ratio', 'mean_sv_score', 'mean_margin'])
    C.write_csv(out / 'ftable_dense_by_level.csv', by_level)
    C.write_csv(out / 'ftable_dense_class_hist.csv', hist_rows)

    mean_dense = C.mean_key(by_angle, 'dense_top1_sv_ratio')
    p1 = 'SUPPORTED' if mean_dense > C.UNIFORM_BASELINE * 3 else 'INCONCLUSIVE'

    with open(fres, 'w') as f:
        C.write_fres_header(f, 'Dense Logit Attractor (P1)', ctx)
        f.write(f'## P1 Verdict: **{p1}**\n\n')
        f.write(f'- uniform baseline 1/15 = `{C.UNIFORM_BASELINE:.4f}`\n')
        f.write(f'- mean dense_top1_sv_ratio = `{mean_dense:.4f}`\n\n')
        f.write('## Table: angle | level metrics (aggregated by angle)\n\n')
        f.write('| angle | dense_top1_sv | pre_nms@0.01 | no_nms_top1k | mean_sv | margin |\n')
        f.write('|---:|---:|---:|---:|---:|---:|\n')
        for r in by_angle:
            f.write(
                f"| {r['angle']} | {r['dense_top1_sv_ratio']:.4f} | "
                f"{r['pre_nms_sv_ratio_001']:.4f} | {r['no_nms_top1000_sv_ratio']:.4f} | "
                f"{r['mean_sv_score']:.4f} | {r['mean_margin']:.4f} |\n")
        f.write('\n## CSV\n\n')
        for name in ['ftable_dense_by_angle.csv', 'ftable_dense_by_level.csv',
                     'ftable_dense_class_hist.csv']:
            f.write(f'- `{out / name}`\n')

    ctx.mark_step('dense', 'OK', p1=p1, mean_dense=mean_dense)
    return dict(status='OK', p1=p1, fres=str(fres))
