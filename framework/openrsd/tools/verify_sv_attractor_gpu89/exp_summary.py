#!/usr/bin/env python
"""Final summary verdict."""
from __future__ import annotations

from tools.verify_sv_attractor_gpu89 import common as C


def _read_step(ctx, step, key, default='N/A'):
    return ctx.progress.get('steps', {}).get(step, {}).get(key, default)


def run(ctx: C.RunContext) -> dict:
    fres = ctx.fres_path('fres_summary_sv_attractor_verdict.md')

    if ctx.blocked():
        verdict = 'BLOCKED'
    else:
        p0 = _read_step(ctx, 'mapping', 'p0', 'N/A')
        p1 = _read_step(ctx, 'dense', 'p1', 'N/A')
        p2 = _read_step(ctx, 'intervention', 'p2', 'N/A')
        p3 = _read_step(ctx, 'bg_gt', 'p3', 'N/A')
        p4 = _read_step(ctx, 'postprocess', 'p4', 'N/A')
        p5 = _read_step(ctx, 'tile_angle', 'p5', 'N/A')
        supported = sum(1 for p in [p1, p2, p3, p4, p5]
                        if p and 'SUPPORTED' in str(p))
        if p0 == 'REJECTED':
            verdict = 'BLOCKED'
        elif supported >= 4 and p0 == 'SUPPORTED':
            verdict = 'CONFIRMED' if 'STRONGLY' in str(p2) else 'PARTIALLY_CONFIRMED'
        elif supported >= 2:
            verdict = 'PARTIALLY_CONFIRMED'
        else:
            verdict = 'INCONCLUSIVE'

    dense_mean = _read_step(ctx, 'dense', 'mean_dense', 'N/A')
    inter_zero = ''
    inter_csv = ctx.exp_dir('exp_03_intervention') / 'ftable_embedding_intervention_final.csv'
    if inter_csv.exists():
        rows = C.read_csv(inter_csv)
        inter_zero = f"{C.mean_key([r for r in rows if r['intervention']=='zero_sv'], 'final_sv_ratio'):.4f}"

    with open(fres, 'w') as f:
        f.write('# Small-Vehicle Attractor Verification Verdict\n\n')
        f.write(f'## 1. One-line conclusion: **{verdict}**\n\n')
        f.write('## 2. Proposition table\n\n')
        f.write('| 命题 | 结论 | 关键证据 | 反证风险 |\n')
        f.write('|---|---|---|---|\n')
        props = [
            ('P0 mapping sanity', 'mapping', 'p0', 'class order / shuffle'),
            ('P1 dense logits bias', 'dense', 'p1', 'dense dump failed'),
            ('P2 embedding causal', 'intervention', 'p2', 'intervention no effect'),
            ('P3 background/GT', 'bg_gt', 'p3', 'GT ann missing; manual audit pending'),
            ('P4 postprocess amplifier', 'postprocess', 'p4', 'no dense npz'),
            ('P5 tile-angle generalization', 'tile_angle', 'p5', 'only P0148 probed'),
            ('P6 artifact exclusion', 'bg_gt', 'p3', 'coarse grid stratification'),
        ]
        for label, step, key, risk in props:
            val = _read_step(ctx, step, key, 'N/A')
            f.write(f'| {label} | {val} | step `{step}` | {risk} |\n')

        f.write('\n## 3. Key numbers\n\n')
        f.write(f'- mean dense_top1_sv_ratio: `{dense_mean}`\n')
        f.write(f'- zero_sv final_sv_ratio (mean): `{inter_zero}`\n')
        f.write(f'- work_dir: `{ctx.work_dir}`\n')

        f.write('\n## 4. Causal chain\n\n')
        f.write('```\n')
        f.write('tile/rotation → dense feature–embedding alignment → small-vehicle dense hub\n')
        f.write('→ embedding intervention transfers dominance → NMS/top-k amplifies → visual flood\n')
        f.write('```\n')

        f.write('\n## 5. Cannot prove\n\n')
        f.write('- Cannot generalize from P0148 alone to all DOTA tiles.\n')
        f.write('- Cannot claim all detections are FP without manual crop audit.\n')
        f.write('- If P0 fails, embedding attractor claim is BLOCKED.\n')
        f.write('- Final detections alone cannot separate cls bias vs postprocess.\n')

        f.write('\n## 6. Next steps\n\n')
        f.write('- class-wise temperature calibration\n')
        f.write('- anti-hub embedding regularization\n')
        f.write('- background negative prompts\n')
        f.write('- rotation-orbit consistency / view-calibrated TTA\n')
        f.write('- manual audit ≥200 crops from manual_audit_template.csv\n')

    ctx.mark_step('summary', 'OK', verdict=verdict)
    return dict(status='OK', verdict=verdict, fres=str(fres))
