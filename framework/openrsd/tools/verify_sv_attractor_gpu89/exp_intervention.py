#!/usr/bin/env python
"""Step 3: embedding causal intervention (P2). GPU 9."""
from __future__ import annotations

import json

import numpy as np
import torch

from tools.rotation_diagnostics import probe_rotated_stage_outputs as probe_base
from tools.verify_sv_attractor_gpu89 import common as C


def run(ctx: C.RunContext) -> dict:
    if ctx.blocked():
        return dict(status='BLOCKED')

    out = ctx.exp_dir('exp_03_intervention')
    fres = ctx.fres_path('fres_03_embedding_causal_intervention.md')
    interventions = (C.INTERVENTIONS_FULL if ctx.mode == 'full'
                     else C.INTERVENTIONS_SMOKE)
    angles = ctx.angles if ctx.mode == 'full' else [0, 45, 90]

    image_dir = C.tile_image_dir(ctx.repo_root, C.DEFAULT_TILE)
    bargs, model, device, det_support, name2id, _, _ = C.build_model(ctx, 'visual')
    loader = C.build_loader(ctx, image_dir, angles)

    dense_rows, final_rows = [], []
    with torch.no_grad():
        for data_info in loader:
            angle = probe_base.angle_from_name(data_info['data_samples'][0].img_path)
            if angle not in angles:
                continue
            data = model.data_preprocessor(data_info, False)
            data['inputs'] = data['inputs'].to(device)
            x = model.prompt_extract_feats(data['inputs'])
            metas = [s.metainfo for s in data['data_samples']]
            for intervention in interventions:
                sf, sl = C.build_mapped_support(
                    model, ctx, det_support, name2id, device, 'visual', intervention)
                outs = C.forward_dense(model, x, sf, sl, bargs)
                levels = C.decode_dense(
                    model.bbox_head, outs[0], outs[1], outs[2], metas[0]['img_shape'])
                dstats = C.dense_stats_from_levels(levels)
                fin = C.final_detection_stats(model.bbox_head, outs, metas)
                dense_rows.append(dict(
                    angle=angle, intervention=intervention,
                    dense_sv_ratio=dstats['dense_top1_sv_ratio'],
                    dense_lv_ratio=dstats['dense_top1_lv_ratio'],
                    dense_ship_ratio=dstats['dense_top1_ship_ratio'],
                    mean_sv_score=dstats['mean_sv_score'],
                    mean_sv_margin=dstats['mean_margin']))
                final_rows.append(dict(
                    angle=angle, intervention=intervention,
                    final_sv_ratio=fin['small_vehicle_ratio'],
                    final_lv_ratio=fin['large_vehicle_ratio'],
                    final_ship_ratio=fin['ship_ratio'],
                    detection_total=fin['detection_total'],
                    mean_score=fin['mean_score'],
                    top1_class=fin['top1_class']))

    C.write_csv(out / 'ftable_embedding_intervention_dense.csv', dense_rows)
    C.write_csv(out / 'ftable_embedding_intervention_final.csv', final_rows)

    by_int = {}
    for r in final_rows:
        by_int.setdefault(r['intervention'], []).append(r)
    means = {k: C.mean_key(v, 'final_sv_ratio') for k, v in by_int.items()}
    zero_sv = means.get('zero_sv', float('nan'))
    swap_lv = means.get('swap_sv_lv', float('nan'))
    orig = means.get('original', float('nan'))
    lv_swap = C.mean_key(by_int.get('swap_sv_lv', []), 'final_lv_ratio')

    if zero_sv < 0.05 and orig > 0.5:
        p2 = 'STRONGLY_SUPPORTED'
    elif zero_sv < orig * 0.5:
        p2 = 'SUPPORTED'
    else:
        p2 = 'INCONCLUSIVE'

    with open(fres, 'w') as f:
        C.write_fres_header(f, 'Embedding Causal Intervention (P2)', ctx)
        f.write(f'## P2 Verdict: **{p2}**\n\n')
        f.write('| intervention | dense_sv | final_sv | final_lv | final_ship | mean_score |\n')
        f.write('|---|---:|---:|---:|---:|---:|\n')
        for inter in interventions:
            d = [x for x in dense_rows if x['intervention'] == inter]
            fin = [x for x in final_rows if x['intervention'] == inter]
            f.write(
                f"| {inter} | {C.mean_key(d, 'dense_sv_ratio'):.4f} | "
                f"{C.mean_key(fin, 'final_sv_ratio'):.4f} | "
                f"{C.mean_key(fin, 'final_lv_ratio'):.4f} | "
                f"{C.mean_key(fin, 'final_ship_ratio'):.4f} | "
                f"{C.mean_key(fin, 'mean_score'):.4f} |\n")
        f.write(f'\n- original final_sv: `{orig:.4f}`\n')
        f.write(f'- zero_sv final_sv: `{zero_sv:.4f}`\n')
        f.write(f'- swap_sv_lv final_lv: `{lv_swap:.4f}`\n')
        f.write('\n## CSV\n\n')
        f.write(f'- `{out / "ftable_embedding_intervention_dense.csv"}`\n')
        f.write(f'- `{out / "ftable_embedding_intervention_final.csv"}`\n')

    ctx.mark_step('intervention', 'OK', p2=p2)
    return dict(status='OK', p2=p2, fres=str(fres))
