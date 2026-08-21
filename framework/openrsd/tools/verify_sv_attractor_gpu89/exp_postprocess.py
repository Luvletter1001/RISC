#!/usr/bin/env python
"""Step 5: postprocess amplifier (P4). GPU 8 — offline on saved dense npz."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from mmcv.ops import nms_rotated

from tools.verify_sv_attractor_gpu89 import common as C


def run_offline_ablation(scores: np.ndarray, boxes: np.ndarray = None):
    """scores: (N, C) numpy."""
    tscores = torch.tensor(scores, dtype=torch.float32)
    n, c = tscores.shape
    flat = tscores.reshape(-1)
    labels = torch.arange(c).repeat(n)
    rows = []
    for k in [100, 300, 1000, 3000, 5000]:
        kk = min(k, flat.numel())
        _, idx = torch.topk(flat, kk)
        labs = labels[idx]
        small = int((labs == C.SMALL).sum())
        rows.append(dict(
            postprocess='no_nms_global_topk', score_thr='', nms_iou='',
            topk=kk, max_per_img='', sv_ratio=float(small / kk),
            detection_total=kk, class_entropy=''))
    for thr in [0.001, 0.01, 0.05, 0.1, 0.3, 0.5]:
        keep = flat >= thr
        labs = labels[keep]
        total = int(keep.sum())
        small = int((labs == C.SMALL).sum())
        rows.append(dict(
            postprocess='score_thr', score_thr=thr, nms_iou='', topk='',
            max_per_img='', sv_ratio=float(small / total) if total else 0,
            detection_total=total, class_entropy=''))
    return rows


def run(ctx: C.RunContext) -> dict:
    if ctx.blocked():
        return dict(status='BLOCKED')

    out = ctx.exp_dir('exp_05_postprocess')
    fres = ctx.fres_path('fres_05_postprocess_amplifier.md')
    dense_dir = ctx.exp_dir('exp_02_dense') / 'npz'
    all_rows = []
    if not dense_dir.is_dir():
        status = 'PARTIAL'
        with open(fres, 'w') as f:
            C.write_fres_header(f, 'Postprocess Amplifier (P4)', ctx, status)
            f.write('Dense npz missing; run `dense` on GPU8 first.\n')
        ctx.mark_step('postprocess', 'PARTIAL', reason='no dense npz')
        return dict(status=status, fres=str(fres))

    for npz_path in sorted(dense_dir.glob('rot*.npz')):
        angle = int(npz_path.stem.replace('rot', ''))
        data = np.load(npz_path)
        parts = []
        for k in sorted(data.files):
            parts.append(data[k].reshape(-1, data[k].shape[-1]))
        scores = np.concatenate(parts, axis=0).astype(np.float32)
        rows = run_offline_ablation(scores)
        for r in rows:
            r['angle'] = angle
        all_rows.extend(rows)

    C.write_csv(out / 'ftable_postprocess_ablation.csv', all_rows)

    no_nms = [r for r in all_rows if r['postprocess'] == 'no_nms_global_topk' and r['topk'] == 1000]
    mean_no_nms = C.mean_key(no_nms, 'sv_ratio')
    p4 = 'SUPPORTED' if mean_no_nms > C.UNIFORM_BASELINE * 2 else 'INCONCLUSIVE'

    with open(fres, 'w') as f:
        C.write_fres_header(f, 'Postprocess Amplifier (P4)', ctx)
        f.write(f'## P4 Verdict: **{p4}**\n\n')
        f.write(f'- mean no_nms top1000 sv_ratio: `{mean_no_nms:.4f}`\n')
        f.write('\n| postprocess | score_thr | topk | sv_ratio | detection_total |\n')
        f.write('|---|---|---:|---:|---:|\n')
        for r in all_rows[:40]:
            f.write(
                f"| {r['postprocess']} | {r['score_thr']} | {r['topk']} | "
                f"{r['sv_ratio']:.4f} | {r['detection_total']} |\n")
        f.write(f'\nFull table: `{out / "ftable_postprocess_ablation.csv"}`\n')

    ctx.mark_step('postprocess', 'OK', p4=p4)
    return dict(status='OK', p4=p4, fres=str(fres))
