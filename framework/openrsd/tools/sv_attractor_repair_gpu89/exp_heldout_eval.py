#!/usr/bin/env python
"""Heldout calib smoke eval: AP50 + sv_AP50 + detection_total + final_sv_ratio."""
from __future__ import annotations

from tools.sv_attractor_repair_gpu89 import common as C
from tools.sv_attractor_repair_gpu89.eval_ap import (
    infer_heldout_method, prepare_heldout_subset, run_evaluator_smoke_test)


def heldout_methods(ctx):
    ms = ['baseline', 'best_R1', 'best_R3', 'best_combined']
    if ctx.progress.get('best_r2_visual_ckpt'):
        ms.insert(2, 'best_R2B')
    elif ctx.mode != 'full':
        ms.insert(2, 'best_R2')
    if ctx.mode == 'full' and ctx.progress.get('best_r2b_K'):
        pass
    return ms


def run(ctx: C.RepairContext) -> dict:
    if ctx.blocked():
        return dict(status='BLOCKED')

    fres = C.fres_for_mode(ctx, 'fres_01b_heldout_smoke_eval.md', 'fres_full_heldout_eval.md')
    angles = [0, 90] if ctx.mode == 'full' else [0, 90]
    max_n = ctx.eval_max_images
    subset = prepare_heldout_subset(ctx, max_n)

    bargs, model, device, det_support, name2id, _, _, _ = C.build_model_ctx(ctx)
    rows = []
    for method in heldout_methods(ctx):
        try:
            res = infer_heldout_method(
                ctx, model, device, det_support, name2id, subset,
                method, angles, max_n, score_thr=C.POSTPROCESS['score_thr'])
            rows.append(res)
        except Exception as exc:
            rows.append(dict(method=method, error=str(exc), ap50_overall=float('nan')))

    flat = []
    for r in rows:
        flat.append({k: v for k, v in r.items() if k != 'per_class'})
    csv_out = ('ftable_full_eval_heldout_ap.csv' if ctx.mode == 'full'
               else 'ftable_heldout_smoke_eval.csv')
    C.write_csv(ctx.tables_dir() / csv_out, flat)

    eval_smoke = run_evaluator_smoke_test(ctx, min(4, max_n))

    with open(fres, 'w') as f:
        C.write_fres_header(f, 'Heldout Calib Smoke Eval', ctx)
        f.write(f'- eval_max_images: `{max_n}`\n')
        f.write(f'- angles: `{angles}`\n')
        f.write(f'- subset: `{subset}`\n')
        f.write(f'- DOTAMetric smoke: `{eval_smoke.get("status")}` '
                f'ap50=`{eval_smoke.get("ap50_overall", "nan")}`\n')
        if eval_smoke.get('error'):
            f.write(f'- DOTAMetric error: `{eval_smoke["error"][:500]}`\n')
        f.write('\n## Results (mean over images × angles)\n\n')
        f.write('| method | AP50 | sv_AP50 | lv_AP50 | ship_AP50 | detection_total | final_sv_ratio |\n')
        f.write('|---|---:|---:|---:|---:|---:|---:|\n')
        for r in rows:
            f.write(
                f"| {r.get('method')} | {r.get('ap50_overall', float('nan')):.4f} | "
                f"{r.get('sv_ap50', float('nan')):.4f} | "
                f"{r.get('lv_ap50', float('nan')):.4f} | "
                f"{r.get('ship_ap50', float('nan')):.4f} | "
                f"{r.get('detection_total', 0):.1f} | "
                f"{r.get('final_sv_ratio', float('nan')):.4f} |\n")
        f.write('\n## Note\n\n')
        f.write('- AP50/sv_AP50: `eval_rbbox_map` at **angle=0** (GT from val pipeline); '
                'final_sv_ratio/detection_total: mean over angles 0+90.\n')
        f.write('- If detection_total collapses while final_sv_ratio drops, fix is likely suppressing boxes only.\n')

    ctx.mark('heldout_eval', 'OK', n_methods=len(rows))
    return dict(status='OK', fres=str(fres), rows=rows, eval_smoke=eval_smoke)
