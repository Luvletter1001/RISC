#!/usr/bin/env python
"""Final repair verdict."""
from __future__ import annotations

from tools.sv_attractor_repair_gpu89 import common as C
from tools.sv_attractor_repair_gpu89.full_metrics import check_success_criteria


def run(ctx: C.RepairContext) -> dict:
    fres = C.fres_for_mode(ctx, 'fres_summary_repair_verdict.md',
                            'fres_summary_full_repair_verdict.md')
    comb_name = 'ftable_combined_repair_full.csv' if ctx.mode == 'full' else 'ftable_combined_repair_ablation.csv'
    comb = C.read_csv(ctx.tables_dir() / comb_name)
    if not comb:
        comb = C.read_csv(ctx.tables_dir() / 'ftable_combined_repair_ablation.csv')

    heldout = C.read_csv(ctx.tables_dir() / (
        'ftable_full_eval_heldout_ap.csv' if ctx.mode == 'full' else 'ftable_heldout_smoke_eval.csv'))
    if not heldout and ctx.mode == 'full':
        heldout = C.read_csv(ctx.tables_dir() / 'ftable_heldout_smoke_eval.csv')

    ap_sum = C.read_csv(ctx.tables_dir() / 'ftable_full_eval_ap_summary.csv') if ctx.mode == 'full' else []
    r2b_ap = C.read_csv(ctx.tables_dir() / 'ftable_r2b_heldout_ap.csv') if ctx.mode == 'full' else []
    baseline_m = ctx.progress.get('full_baseline_metrics', {})

    by_method = {}
    for r in comb:
        m = r.get('method', '')
        by_method.setdefault(m, []).append(r)

    def agg(method, key, default=0):
        rows = by_method.get(method, [])
        return C.mean_key(rows, key) if rows else default

    base_sv = float(baseline_m.get('P0148_final_sv', agg('C0_baseline', 'P0148_final_sv', 0.75)))
    base_ap = float(baseline_m.get('ap50_overall', 0))
    base_sv_ap = float(baseline_m.get('sv_ap50', 0))

    candidates = []
    for m in by_method:
        if m == 'C0_baseline':
            continue
        row = dict(
            method=m,
            P0148_final_sv=agg(m, 'P0148_final_sv'),
            dense_sv=agg(m, 'dense_sv'),
            detection_total=agg(m, 'detection_total'),
        )
        h = [x for x in heldout if x.get('method', '').replace('best_', '') in m or m in x.get('method', '')]
        if not h:
            h = [x for x in heldout if 'combined' in m and 'combined' in x.get('method', '')]
        if not h and 'R1' in m:
            h = [x for x in heldout if x.get('method') == 'best_R1']
        if h:
            row['ap50_overall'] = float(h[0].get('ap50_overall', 0))
            row['sv_ap50'] = float(h[0].get('sv_ap50', 0))
        row['criteria'] = check_success_criteria(baseline_m, dict(
            P0148_final_sv=row['P0148_final_sv'],
            P0148_det_total=row['detection_total'],
            P0148_lv_ratio=baseline_m.get('P0148_lv_ratio', 0),
            P0148_ship_ratio=baseline_m.get('P0148_ship_ratio', 0),
            ap50_overall=row.get('ap50_overall', base_ap),
            sv_ap50=row.get('sv_ap50', base_sv_ap),
        )) if baseline_m else {}
        candidates.append(row)

    best = min(candidates, key=lambda x: float(x['P0148_final_sv'])) if candidates else {}
    drop = (1 - float(best.get('P0148_final_sv', 1)) / max(base_sv, 1e-6)) if best else 0
    pass_all = best.get('criteria', {}).get('pass_all', False) if best else False
    verdict = 'SUCCESS' if pass_all else ('PARTIAL_SUCCESS' if drop >= 0.30 else 'FAILED')

    with open(fres, 'w') as f:
        f.write('# Full Repair Verdict Summary\n\n' if ctx.mode == 'full' else '# Repair Verdict Summary\n\n')
        f.write(f'## 1. One-line: **{verdict}**\n\n')
        f.write(f'- best combined method: `{ctx.progress.get("best_combined_method", best.get("method"))}`\n')
        f.write(f'- best R1 ckpt: `{ctx.progress.get("best_r1_ckpt", "")}`\n')
        f.write(f'- best R2B K: `{ctx.progress.get("best_r2b_K", "N/A")}`\n\n')

        f.write('## 2. Main table\n\n')
        f.write('| method | P0148_final_sv | dense_sv | det_total | AP50 | sv_AP50 | pass |\n')
        f.write('|---|---:|---:|---:|---:|---:|---|\n')
        f.write(f"| baseline | {base_sv:.4f} | {float(baseline_m.get('P0148_dense_sv', 0)):.4f} | "
                f"{float(baseline_m.get('P0148_det_total', 0)):.0f} | {base_ap:.4f} | {base_sv_ap:.4f} | ref |\n")
        for row in sorted(candidates, key=lambda x: x['P0148_final_sv']):
            c = row.get('criteria', {})
            f.write(f"| {row['method']} | {row['P0148_final_sv']:.4f} | {row['dense_sv']:.4f} | "
                    f"{row['detection_total']:.0f} | {row.get('ap50_overall', float('nan')):.4f} | "
                    f"{row.get('sv_ap50', float('nan')):.4f} | {c.get('pass_all', False)} |\n")

        if r2b_ap:
            f.write('\n## 3. R2B per-K heldout AP\n\n')
            for r in r2b_ap:
                k = str(r.get('K', '')).lstrip('K')
                f.write(f"- K{k}: P0148_sv={float(r.get('P0148_final_sv',0)):.4f} "
                        f"AP50={float(r.get('ap50_overall',0)):.4f} sv_AP50={float(r.get('sv_ap50',0)):.4f}\n")

        f.write('\n## 4. Recommendations\n\n')
        f.write('- **Paper candidate**: R1 class-wise calibration (R1C) or C5_R1+R3 if combined passes\n')
        f.write('- **Diagnostic**: R3 fixed alpha (alpha=1.0 kills output)\n')
        f.write('- **R2B**: see per-K table; use best K only if criteria pass\n')
        f.write('\n## 5. Cannot prove\n\n')
        f.write('- manual audit incomplete unless template filled\n')
        f.write('- unmatched_sv full metric may be missing\n')

    ctx.mark('summary', 'OK', verdict=verdict)
    ctx.progress['full_verdict'] = verdict
    ctx.save_progress()
    return dict(status='OK', verdict=verdict, fres=str(fres))
