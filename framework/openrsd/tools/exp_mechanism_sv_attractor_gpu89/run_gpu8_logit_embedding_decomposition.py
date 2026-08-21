#!/usr/bin/env python
"""GPU8: logit / embedding / objectness decomposition."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.exp_mechanism_sv_attractor_gpu89 import common_attractor_utils as U

CSV = U.RESULT_MD / 'ftable_02_logit_embedding_decomposition.csv'
SCHEMA = [
    'tile_id', 'risk_group', 'angle', 'intervention', 'dense_top1_sv_ratio',
    'dense_sv_mean_logit', 'dense_sv_margin_vs_runnerup', 'dense_sv_margin_p95',
    'dense_entropy', 'objectness_mean', 'objectness_p95', 'final_sv_ratio',
    'det_count', 'ap50', 'notes', 'timestamp',
]

INTERVENTIONS = [
    'baseline',
    'zero_sv_embedding',
    'swap_sv_with_nearest_class_embedding',
    'orthogonalize_sv_embedding_to_background_mean',
    'normalize_all_class_embeddings',
    'bias_only_R1',
    'temperature_only_R1',
    'bias_plus_temperature_R1',
]


def risk_group(tile_id: str, hi: set, lo: set) -> str:
    if tile_id in hi:
        return 'high'
    if tile_id in lo:
        return 'low'
    return 'medium'


def _mean_key(sub, k):
    import statistics
    vals = []
    for r in sub:
        try:
            vals.append(float(r[k]))
        except (TypeError, ValueError):
            pass
    return statistics.mean(vals) if vals else float('nan')


def write_report(rows: list, out: Path):
    csv_path = U.RESULT_MD / 'ftable_02_logit_embedding_decomposition.csv'
    rows = U.read_csv(csv_path) if csv_path.exists() else rows
    rows = [r for r in rows if str(r.get('dense_top1_sv_ratio', 'NA')) != 'NA']
    if not rows:
        out.write_text('# Logit decomposition\n\nNo valid rows.\n')
        return
    mean_key = _mean_key
    lines = [
        '# Logit / Embedding / Objectness Decomposition (Exp2)',
        '',
        'objectness 代理：dense 各 location 的 max class score（模型 `with_objectness=False`）。',
        '',
        '| group | intervention | dense_top1_sv | sv_margin | objectness | final_sv | interpretation |',
        '|---|---|---:|---:|---:|---:|---|',
    ]
    interp = {
        'zero_sv_embedding': 'H1 embedding hub',
        'bias_only_R1': 'H4 class prior (bias)',
        'temperature_only_R1': 'H4 temperature only',
        'bias_plus_temperature_R1': 'H4 full R1',
    }
    for grp in ['high', 'medium', 'low']:
        for iv in INTERVENTIONS:
            sub = [r for r in rows if r.get('risk_group') == grp and r.get('intervention') == iv]
            if not sub:
                continue
            lines.append(
                f"| {grp} | {iv} | {mean_key(sub, 'dense_top1_sv_ratio'):.3f} | "
                f"{mean_key(sub, 'dense_sv_margin_vs_runnerup'):.3f} | "
                f"{mean_key(sub, 'objectness_mean'):.3f} | {mean_key(sub, 'final_sv_ratio'):.3f} | "
                f"{interp.get(iv, '—')} |")

    z = [r for r in rows if r.get('intervention') == 'zero_sv_embedding']
    b = [r for r in rows if r.get('intervention') == 'baseline']
    if z and b:
        d_dense = mean_key(z, 'dense_top1_sv_ratio') - mean_key(b, 'dense_top1_sv_ratio')
        d_obj = mean_key(z, 'objectness_mean') - mean_key(b, 'objectness_mean')
        lines += [
            '',
            '## H1 / H4 判定',
            '',
            f'- zero_sv dense_top1_sv delta: **{d_dense:.3f}**',
            f'- zero_sv objectness delta: **{d_obj:.3f}**',
        ]
        if d_dense < -0.3 and abs(d_obj) < 0.05:
            lines.append('- **H1 PARTIALLY_SUPPORTED**：embedding 干预主要改 class margin，objectness 几乎不变。')
        else:
            lines.append('- H1 需结合 per-tile 表进一步判断。')

    bo = [r for r in rows if r.get('intervention') == 'bias_only_R1']
    bt = [r for r in rows if r.get('intervention') == 'bias_plus_temperature_R1']
    if bo and bt and b:
        lines.append(
            f"- bias-only final_sv drop: **{mean_key(bo, 'final_sv_ratio') - mean_key(b, 'final_sv_ratio'):.3f}**; "
            f"full R1 drop: **{mean_key(bt, 'final_sv_ratio') - mean_key(b, 'final_sv_ratio'):.3f}**")
        if abs(mean_key(bo, 'final_sv_ratio') - mean_key(bt, 'final_sv_ratio')) < 0.05:
            lines.append('- **H4**：bias-only 与 full R1 接近 → bias 是主因，temperature 辅助。')
        else:
            lines.append('- **H4**：temperature 有显著附加作用。')

    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def pick_tiles(ctx: U.MechContext) -> tuple:
    summ = U.read_csv(U.RESULT_MD / 'ftable_01_attractor_atlas_tile_summary.csv')
    if summ:
        ordered = sorted(summ, key=lambda r: -float(r.get('baseline_final_sv', 0)))
        ids = [r['tile_id'] for r in ordered]
        hi = set(ids[:4])
        lo = set(ids[-4:])
        mid = set(ids[4:8])
        return [U.DEFAULT_TILE], hi, mid, lo
    return [U.DEFAULT_TILE], {U.DEFAULT_TILE}, set(), set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=8)
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    ctx = U.MechContext(gpu=args.gpu)
    ctx.work_dir = U.WORK_ROOT / 'gpu8_decomp'
    U.setup_logging(ctx.work_dir, 'decomp')

    p0148, hi, mid, lo = pick_tiles(ctx)
    tiles = list(p0148) + list(hi) + list(mid) + list(lo)
    tiles = list(dict.fromkeys(tiles))[:13]

    bargs, model, device, det_support, name2id, _, _ = U.build_model(ctx)
    for tid in tiles:
        grp = risk_group(tid, hi, lo)
        for angle in U.DECOMP_ANGLES:
            for iv in INTERVENTIONS:
                if U.already_done(CSV, tid, angle, iv):
                    continue
                if iv in ('baseline', 'bias_only_R1', 'temperature_only_R1', 'bias_plus_temperature_R1'):
                    method, st = iv, 'original'
                else:
                    method, st = 'baseline', iv
                try:
                    row = U.eval_one(ctx, model, bargs, device, det_support, name2id,
                                     tid, angle, method, support_intervention=st)
                    row['risk_group'] = grp
                    row['intervention'] = iv
                    U.append_csv(CSV, row, SCHEMA)
                except Exception as exc:
                    U.log.exception('%s %s %s', tid, angle, iv)
                    U.append_csv(CSV, dict(
                        tile_id=tid, angle=angle, intervention=iv,
                        risk_group=grp, notes=str(exc)), SCHEMA)

    rows = U.read_csv(CSV)
    write_report(rows, U.RESULT_MD / 'fres_02_logit_embedding_decomposition.md')
    print('done', CSV)


if __name__ == '__main__':
    main()
