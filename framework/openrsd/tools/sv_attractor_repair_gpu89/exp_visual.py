#!/usr/bin/env python
"""Visual evidence pack."""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tools.sv_attractor_repair_gpu89 import common as C


def run(ctx: C.RepairContext) -> dict:
    fres = C.fres_for_mode(ctx, 'fres_07_visual_failure_success_pack.md',
                            'fres_07_full_visual_failure_success_pack.md')
    fig_dir = ctx.figures_dir()
    figures = []

    base_csv = ctx.tables_dir() / (
        'ftable_full_baseline_reconfirm.csv' if ctx.mode == 'full' else 'ftable_baseline_reconfirm.csv')
    comb_csv = ctx.tables_dir() / (
        'ftable_combined_repair_full.csv' if ctx.mode == 'full' else 'ftable_combined_repair_ablation.csv')
    r3_csv = ctx.tables_dir() / (
        'ftable_r3_full_fixed_alpha.csv' if ctx.mode == 'full' else 'ftable_r3_fixed_alpha.csv')

    if base_csv.exists():
        rows = C.read_csv(base_csv)
        p0148 = [r for r in rows if r.get('split') == 'P0148']
        if p0148:
            angles = [int(float(r['angle'])) for r in p0148]
            vals = [float(r['final_sv_ratio']) for r in p0148]
            p = fig_dir / 'baseline_angle_vs_final_sv.png'
            plt.figure(figsize=(8, 4))
            plt.plot(angles, vals, 'o-')
            plt.xlabel('angle'); plt.ylabel('final_sv_ratio')
            plt.title('Baseline P0148'); plt.grid(alpha=0.3)
            plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
            figures.append(('baseline angle curve', p))

    if r3_csv.exists():
        rows = C.read_csv(r3_csv)
        variants = sorted(set(r['variant'] for r in rows))
        means = [C.mean_key([r for r in rows if r['variant'] == v], 'final_sv_ratio') for v in variants]
        p = fig_dir / 'r3_alpha_sweep.png'
        plt.figure(figsize=(9, 4))
        plt.bar(range(len(variants)), means)
        plt.xticks(range(len(variants)), variants, rotation=45, ha='right')
        plt.ylabel('final_sv_ratio'); plt.title('R3 fixed alpha')
        plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
        figures.append(('R3 alpha sweep', p))

    if comb_csv.exists():
        rows = C.read_csv(comb_csv)
        methods = sorted(set(r['method'] for r in rows))
        means = [C.mean_key([r for r in rows if r['method'] == m], 'P0148_final_sv') for m in methods]
        p = fig_dir / 'combined_ablation.png'
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(methods)), means)
        plt.xticks(range(len(methods)), methods, rotation=30, ha='right')
        plt.ylabel('P0148 final_sv'); plt.title('Combined ablation')
        plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
        figures.append(('combined ablation', p))

    with open(fres, 'w') as f:
        C.write_fres_header(f, 'Visual Failure/Success Pack', ctx)
        for title, path in figures:
            f.write(f'### {title}\n\n![{title}]({path})\n\n- `{path}`\n')
        if not figures:
            f.write('- PARTIAL: no upstream CSV for plots\n')
        crop_dir = ctx.verify_work_dir / 'exp_04_bg_gt/crops'
        if crop_dir.is_dir():
            f.write(f'\n- reuse verify crops: `{crop_dir}`\n')

    ctx.mark('visual', 'OK' if figures else 'PARTIAL')
    return dict(status='OK', fres=str(fres), n=len(figures))
