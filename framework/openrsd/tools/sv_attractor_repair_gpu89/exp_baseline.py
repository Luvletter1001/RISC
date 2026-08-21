#!/usr/bin/env python
"""Baseline reconfirm with unified postprocess."""
from __future__ import annotations

import json

from tools.sv_attractor_repair_gpu89 import common as C
from tools.sv_attractor_repair_gpu89.full_metrics import (
    eval_cross_tile_rows, eval_p0148_rows, full_method_report)
from tools.verify_sv_attractor_gpu89 import common as verify


def eval_tile_angles(ctx, model, bargs, device, det_support, name2id, tile_id, angles, tag):
    """Legacy row builder (smoke / csv export)."""
    from types import SimpleNamespace
    import torch
    from tools.rotation_diagnostics import probe_rotated_stage_outputs as probe_base
    from tools.sv_attractor_repair_gpu89.repair_forward import (
        build_support_tensors, dense_stats_from_logits, final_detection_from_outs)

    rows = []
    try:
        image_dir = verify.tile_image_dir(ctx.repo_root, tile_id)
    except FileNotFoundError:
        return rows
    lb = SimpleNamespace(**vars(bargs))
    lb.image_dir = str(image_dir)
    lb.out_dir = str(ctx.work_dir / '_tmp')
    loader = probe_base.build_dataloader(lb, angles)
    sf, sl = build_support_tensors(model, det_support, name2id, device, 'visual', bargs.support_shot)
    with torch.no_grad():
        for data_info in loader:
            angle = probe_base.angle_from_name(data_info['data_samples'][0].img_path)
            if angle not in angles:
                continue
            data = model.data_preprocessor(data_info, False)
            data['inputs'] = data['inputs'].to(device)
            x = model.prompt_extract_feats(data['inputs'])
            metas = [s.metainfo for s in data['data_samples']]
            outs_all = model.bbox_head(
                x, sf, sl, sl, support_shot=bargs.support_shot,
                num_classes=C.NUM_CLASSES, num_in_classes=C.NUM_CLASSES,
                align_style='labelled', support_type='visual', text_cls_scale=0.0)
            outs = outs_all[:-2]
            dst = dense_stats_from_logits(list(outs[0]), C.SMALL)
            fin = final_detection_from_outs(model, outs, metas)
            hist = C.parse_class_histogram(fin['class_histogram'])
            rows.append(dict(
                method='baseline', split=tag, tile_id=tile_id, angle=angle,
                dense_top1_sv_ratio=dst['dense_top1_sv_ratio'],
                final_sv_ratio=fin['final_sv_ratio'],
                final_lv_ratio=fin['final_lv_ratio'],
                final_ship_ratio=float(hist.get('ship', 0) / max(fin['detection_total'], 1)),
                detection_total=fin['detection_total'],
                mean_score=fin['mean_score'],
                ap50='', sv_ap50='', class_entropy=''))
    return rows


def run(ctx: C.RepairContext) -> dict:
    if ctx.blocked():
        return dict(status='BLOCKED')

    smoke_fres = 'fres_01_baseline_reconfirm.md'
    full_fres = 'fres_01_full_baseline_reconfirm.md'
    fres_path = C.fres_for_mode(ctx, smoke_fres, full_fres)

    splits = C.build_splits(ctx)
    bargs, model, device, det_support, name2id, _, _, _ = C.build_model_ctx(ctx)

    rows = []
    rows.extend(eval_tile_angles(ctx, model, bargs, device, det_support, name2id,
                                   C.DEFAULT_TILE, ctx.angles, 'P0148'))
    eval_angles_cross = [0, 90, 180, 270] if ctx.mode == 'full' else [0, 90]
    n_tiles = 25 if ctx.mode == 'full' else 5
    for tid in splits['cross_tiles'][:n_tiles]:
        rows.extend(eval_tile_angles(
            ctx, model, bargs, device, det_support, name2id, tid, eval_angles_cross, 'cross_tile'))

    out_csv = ctx.tables_dir() / (
        'ftable_full_baseline_reconfirm.csv' if ctx.mode == 'full'
        else 'ftable_baseline_reconfirm.csv')
    C.write_csv(out_csv, rows)

    p0148 = [r for r in rows if r['split'] == 'P0148']
    cross = [r for r in rows if r['split'] == 'cross_tile']
    smoke_p0148_sv = 0.7551
    full_report = None
    if ctx.mode == 'full':
        full_report = full_method_report(
            ctx, model, bargs, device, det_support, name2id, 'baseline', splits['n_heldout'])
        ctx.progress['full_baseline_metrics'] = full_report
        ctx.save_progress()

    with open(fres_path, 'w') as f:
        title = 'Full Baseline Reconfirm' if ctx.mode == 'full' else 'Baseline Reconfirm'
        C.write_fres_header(f, title, ctx)
        f.write(f'- config: `{ctx.config}`\n')
        f.write(f'- checkpoint: `{ctx.checkpoint}`\n')
        f.write(f'- support: `{ctx.support_pkl}`\n')
        f.write(f'- train_calib: {splits["n_train"]} (P0148 excluded: `{splits["p0148_excluded"]}`)\n')
        f.write(f'- heldout_calib: {splits["n_heldout"]}\n')
        f.write(f'- csv: `{out_csv}`\n\n')

        f.write('## P0148 12-angle ratios\n\n')
        f.write('| angle | dense_sv | final_sv | det_total | lv_ratio | ship_ratio |\n')
        f.write('|---:|---:|---:|---:|---:|---:|\n')
        for r in sorted(p0148, key=lambda x: int(x['angle'])):
            f.write(f"| {r['angle']} | {float(r['dense_top1_sv_ratio']):.4f} | "
                    f"{float(r['final_sv_ratio']):.4f} | {r['detection_total']} | "
                    f"{float(r.get('final_lv_ratio', 0)):.4f} | "
                    f"{float(r.get('final_ship_ratio', 0)):.4f} |\n")
        f.write(f'\n- mean P0148 final_sv: `{C.mean_key(p0148, "final_sv_ratio"):.4f}`\n')
        f.write(f'- mean P0148 dense_sv: `{C.mean_key(p0148, "dense_top1_sv_ratio"):.4f}`\n')

        f.write('\n## Cross-tile\n\n')
        f.write(f'- tiles evaluated: {len(set(r["tile_id"] for r in cross))}\n')
        f.write(f'- mean cross_tile final_sv: `{C.mean_key(cross, "final_sv_ratio"):.4f}`\n')
        if cross:
            worst = max(cross, key=lambda r: float(r['final_sv_ratio']))
            best = min(cross, key=lambda r: float(r['final_sv_ratio']))
            f.write(f'- worst: tile `{worst["tile_id"]}` angle {worst["angle"]} sv={float(worst["final_sv_ratio"]):.4f}\n')
            f.write(f'- best: tile `{best["tile_id"]}` angle {best["angle"]} sv={float(best["final_sv_ratio"]):.4f}\n')

        if ctx.mode == 'full' and full_report:
            f.write('\n## Heldout AP (eval_rbbox_map, angle=0)\n\n')
            f.write(f'- AP50 overall: `{full_report.get("ap50_overall", float("nan")):.4f}`\n')
            f.write(f'- sv_AP50: `{full_report.get("sv_ap50", float("nan")):.4f}`\n')
            f.write(f'- lv_AP50: `{full_report.get("lv_ap50", float("nan")):.4f}`\n')
            f.write(f'- ship_AP50: `{full_report.get("ship_ap50", float("nan")):.4f}`\n')
            if full_report.get('ap_error'):
                f.write(f'- AP error: `{full_report["ap_error"][:500]}`\n')
            f.write('\n## Smoke vs full baseline check\n\n')
            cur = C.mean_key(p0148, 'final_sv_ratio')
            f.write(f'- smoke reference P0148 mean final_sv: `{smoke_p0148_sv:.4f}`\n')
            f.write(f'- full P0148 mean final_sv: `{cur:.4f}`\n')
            delta = abs(cur - smoke_p0148_sv)
            f.write(f'- |delta|: `{delta:.4f}`\n')
            if delta > 0.15:
                f.write('\n**WARNING**: large delta vs smoke — check angles/postprocess/data path.\n')
                ctx.set_blocked(f'baseline full vs smoke delta {delta:.3f}')
            else:
                f.write('\nBaseline full consistent with smoke (within tolerance).\n')
        elif ctx.mode != 'full':
            f.write('\nAP50: see heldout_eval / full baseline step.\n')

    if ctx.blocked():
        return dict(status='BLOCKED', fres=str(fres_path))
    ctx.mark('baseline', 'OK')
    return dict(status='OK', fres=str(fres_path))
