#!/usr/bin/env python
"""Evaluate existing R2B visual prototypes on P0148 + heldout AP (no retrain)."""
from __future__ import annotations

from pathlib import Path

import torch

from tools.sv_attractor_repair_gpu89 import common as C
from tools.sv_attractor_repair_gpu89.full_metrics import (
    check_success_criteria, eval_p0148_rows, full_method_report)
from tools.sv_attractor_repair_gpu89.repair_modules import NegativePrototypeBank, RepairStack


def run(ctx: C.RepairContext) -> dict:
    if ctx.blocked():
        return dict(status='BLOCKED')

    fres = ctx.fres('fres_03b_r2b_visual_eval.md')
    ckpts = ctx.progress.get('r2_visual_ckpts', {})
    if not ckpts:
        with open(fres, 'w') as f:
            C.write_fres_header(f, 'R2B Visual Eval', ctx, 'FAILED')
            f.write('- No R2B checkpoints in progress.r2_visual_ckpts\n')
        return dict(status='FAILED')

    bargs, model, device, det_support, name2id, _, _, _ = C.build_model_ctx(ctx)
    baseline = ctx.progress.get('full_baseline_metrics', {})
    rows = []
    ap_rows = []

    for k, path in sorted(ckpts.items(), key=lambda x: int(x[0][1:])):
        if not Path(path).exists():
            continue
        sd = torch.load(path, map_location=device)
        dim = int(sd['prototypes'].shape[1])
        bank = NegativePrototypeBank(dim, int(sd['prototypes'].shape[0])).to(device)
        bank.load_state_dict(sd)
        rep = RepairStack()
        rep.neg_bank = bank
        rep.sv_idx = C.SMALL
        ctx.progress['best_r2_visual_ckpt'] = path

        tag = 'best_R2B'
        p0148 = eval_p0148_rows(ctx, model, bargs, device, det_support, name2id, tag)
        for r in p0148:
            r['variant'] = k
            r['lambda_neg'] = float(bank.lambda_neg.item())
        rows.extend(p0148)

        ap = full_method_report(ctx, model, bargs, device, det_support, name2id, tag)
        ap['K'] = k
        ap['ckpt'] = path
        if baseline:
            ap['criteria'] = check_success_criteria(baseline, ap)
        ap_rows.append(ap)

    C.write_csv(ctx.tables_dir() / 'ftable_r2b_visual_proto_full.csv', rows)
    C.write_csv(ctx.tables_dir() / 'ftable_r2b_heldout_ap.csv',
                [{k: v for k, v in r.items() if k != 'per_class'} for r in ap_rows])

    if ap_rows:
        best = min(ap_rows, key=lambda r: float(r.get('P0148_final_sv', 1e9)))
        ctx.progress['best_r2_visual_ckpt'] = best.get('ckpt', '')
        ctx.progress['best_r2b_K'] = best.get('K', '')
        ctx.save_progress()

    with open(fres, 'w') as f:
        C.write_fres_header(f, 'R2B Visual Prototype Eval (P0148 + heldout AP)', ctx)
        f.write('| K | P0148_final_sv | AP50 | sv_AP50 | det_P0148 | pass_criteria |\n')
        f.write('|---|---:|---:|---:|---:|---|\n')
        for r in ap_rows:
            crit = r.get('criteria', {})
            f.write(f"| {r.get('K')} | {float(r.get('P0148_final_sv', 0)):.4f} | "
                    f"{float(r.get('ap50_overall', 0)):.4f} | "
                    f"{float(r.get('sv_ap50', 0)):.4f} | "
                    f"{float(r.get('P0148_det_total', 0)):.0f} | "
                    f"`{crit.get('pass_all', False)}` |\n")
        if ap_rows:
            f.write(f'\n- best K: `{ctx.progress.get("best_r2b_K")}`\n')
            f.write(f'- best ckpt: `{ctx.progress.get("best_r2_visual_ckpt")}`\n')
        b_sv = float(baseline.get('P0148_final_sv', 0.75))
        f.write(f'\n- vs baseline P0148 sv drop: '
                f'`{(1 - float(ap_rows[0].get("P0148_final_sv", 1))/max(b_sv,1e-6))*100:.1f}%` (best row)\n')

    ctx.mark('r2b_eval', 'OK')
    return dict(status='OK', fres=str(fres), best_K=ctx.progress.get('best_r2b_K'))
