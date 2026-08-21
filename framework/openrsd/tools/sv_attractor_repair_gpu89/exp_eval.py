#!/usr/bin/env python
"""Evaluation on angle / cross-tile / heldout."""
from __future__ import annotations

from pathlib import Path

from tools.sv_attractor_repair_gpu89 import common as C
from tools.sv_attractor_repair_gpu89.exp_baseline import eval_tile_angles
from tools.sv_attractor_repair_gpu89.full_metrics import eval_cross_tile_rows, eval_p0148_rows, full_method_report


def _eval_methods(ctx):
    ms = ['baseline', 'best_R1', 'best_R3', 'best_combined']
    if ctx.progress.get('best_r2_visual_ckpt'):
        ms.insert(2, 'best_R2B')
    return ms


def run(ctx: C.RepairContext) -> dict:
    if ctx.blocked():
        return dict(status='BLOCKED')
    fres = C.fres_for_mode(ctx, 'fres_06_eval_angle_cross_tile_ap.md',
                            'fres_06_full_eval_angle_cross_tile_ap.md')
    splits = C.build_splits(ctx)
    bargs, model, device, det_support, name2id, _, _, _ = C.build_model_ctx(ctx)

    p0148_all = []
    cross_all = []
    ap_summary = []
    per_class_rows = []

    if ctx.mode == 'full':
        for method in _eval_methods(ctx):
            p0148_all.extend(eval_p0148_rows(
                ctx, model, bargs, device, det_support, name2id, method, ctx.angles))
            cross_all.extend(eval_cross_tile_rows(
                ctx, model, bargs, device, det_support, name2id, method,
                splits['cross_tiles']))
            rep = full_method_report(ctx, model, bargs, device, det_support, name2id, method)
            ap_summary.append(rep)
            for pr in rep.get('per_class', []):
                per_class_rows.append(dict(method=method, **pr))
        C.write_csv(ctx.tables_dir() / 'ftable_full_eval_p0148_angles.csv', p0148_all)
        C.write_csv(ctx.tables_dir() / 'ftable_full_eval_cross_tile.csv', cross_all)
        C.write_csv(ctx.tables_dir() / 'ftable_full_eval_per_class_ap.csv', per_class_rows)
        C.write_csv(ctx.tables_dir() / 'ftable_full_eval_ap_summary.csv',
                    [{k: v for k, v in r.items() if k != 'per_class'} for r in ap_summary])
    else:
        rows = []
        rows.extend(eval_tile_angles(
            ctx, model, bargs, device, det_support, name2id,
            C.DEFAULT_TILE, ctx.angles, 'P0148_baseline'))
        cross_angles = [0, 90]
        for tid in splits['cross_tiles'][:5]:
            rows.extend(eval_tile_angles(
                ctx, model, bargs, device, det_support, name2id, tid, cross_angles, 'cross_tile'))
        C.write_csv(ctx.tables_dir() / 'ftable_eval_angle_ap.csv', rows)
        p0148_all = [r for r in rows if 'P0148' in r.get('split', '')]
        cross_all = [r for r in rows if r.get('split') == 'cross_tile']

    angle_sweep = C.ANGLE_SWEEP_VAL
    angle_sweep_note = 'NOT_AVAILABLE'
    if angle_sweep.exists():
        angle_sweep_note = f'path exists: {angle_sweep} (manual eval deferred)'

    with open(fres, 'w') as f:
        C.write_fres_header(f, 'Full Eval Angle / Cross-tile / AP' if ctx.mode == 'full' else 'Eval', ctx)
        p0148_base = [r for r in p0148_all if r.get('method') == 'baseline']
        f.write(f'- P0148 baseline mean final_sv: `{C.mean_key(p0148_base, "final_sv_ratio"):.4f}`\n')
        f.write(f'- cross-tile baseline mean final_sv: `{C.mean_key([r for r in cross_all if r.get("method")=="baseline"], "final_sv_ratio"):.4f}`\n')
        f.write(f'- angle_sweep_val: `{angle_sweep_note}`\n\n')
        if ap_summary:
            f.write('## AP summary (heldout angle=0)\n\n')
            f.write('| method | AP50 | sv_AP50 | lv_AP50 | ship_AP50 |\n')
            f.write('|---|---:|---:|---:|---:|\n')
            for r in ap_summary:
                f.write(f"| {r['method']} | {r.get('ap50_overall', float('nan')):.4f} | "
                        f"{r.get('sv_ap50', float('nan')):.4f} | "
                        f"{r.get('lv_ap50', float('nan')):.4f} | "
                        f"{r.get('ship_ap50', float('nan')):.4f} |\n")
                if r.get('ap_error'):
                    f.write(f"\nAP error ({r['method']}): `{r['ap_error'][:300]}`\n")

    ctx.mark('eval', 'OK')
    return dict(status='OK', fres=str(fres))
