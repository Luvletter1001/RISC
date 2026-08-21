#!/usr/bin/env python
"""Exp6: minimal repair mechanism (R1 components, C5, oracle)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.exp_mechanism_sv_attractor_gpu89 import common_attractor_utils as U

CSV = U.RESULT_MD / 'ftable_06_minimal_repair_mechanism.csv'
SCHEMA = [
    'tile_id', 'risk_group', 'angle', 'method', 'dense_sv_margin_vs_runnerup',
    'final_sv_ratio', 'det_count', 'objectness_mean', 'ap50', 'sv_ap50', 'notes',
]

METHODS = [
    'baseline', 'bias_only_R1', 'temperature_only_R1', 'bias_plus_temperature_R1',
    'best_C5', 'oracle_remove_sv_bias',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=9)
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    ctx = U.MechContext(gpu=args.gpu)
    ctx.work_dir = U.WORK_ROOT / 'gpu9_repair_mech'
    U.setup_logging(ctx.work_dir, 'repair_mech')

    tiles = [U.DEFAULT_TILE]
    hi_csv = U.read_csv(U.RESULT_MD / 'ftable_01_attractor_highrisk_tiles.csv')
    summ = U.read_csv(U.RESULT_MD / 'ftable_01_attractor_atlas_tile_summary.csv')
    hi, lo = [], []
    if summ:
        ordered = sorted(summ, key=lambda r: -float(r.get('baseline_final_sv', 0)))
        hi = [r['tile_id'] for r in ordered[:5]]
        lo = [r['tile_id'] for r in ordered[-5:]]
        tiles += hi + lo
    tiles = list(dict.fromkeys(tiles))

    bargs, model, device, det_support, name2id, _, _ = U.build_model(ctx)
    for tid in tiles:
        grp = 'high' if tid in hi else ('low' if tid in lo else 'reference')
        for angle in [0, 90]:
            for method in METHODS:
                if U.already_done(CSV, tid, angle, method):
                    continue
                try:
                    row = U.eval_one(ctx, model, bargs, device, det_support, name2id,
                                     tid, angle, method)
                    row['risk_group'] = grp
                    U.append_csv(CSV, row, SCHEMA)
                except Exception as exc:
                    U.append_csv(CSV, dict(
                        tile_id=tid, angle=angle, method=method,
                        risk_group=grp, notes=str(exc)), SCHEMA)

    write_report(U.read_csv(CSV))


def write_report(rows: list):
    import statistics
    lines = ['# Minimal Repair Mechanism (Exp6)', '']

    def mean_meth(m, k):
        vals = []
        for r in rows:
            if r.get('method') != m:
                continue
            try:
                vals.append(float(r[k]))
            except (TypeError, ValueError):
                pass
        return statistics.mean(vals) if vals else float('nan')

    for m in METHODS:
        fv, mar = mean_meth(m, 'final_sv_ratio'), mean_meth(m, 'dense_sv_margin_vs_runnerup')
        if fv == fv:
            lines.append(f'- **{m}**: mean final_sv={fv:.3f}, margin={mar:.3f}')
    b = mean_meth('baseline', 'final_sv_ratio')
    bo = mean_meth('bias_only_R1', 'final_sv_ratio')
    bt = mean_meth('bias_plus_temperature_R1', 'final_sv_ratio')
    lines += ['', '## 结论', '']
    if abs(bo - bt) < 0.05 * max(b, 0.01):
        lines.append('- bias-only ≈ full R1 → **bias 足够**')
    else:
        lines.append('- temperature 有不可忽略增益')
    lo = [r for r in rows if r.get('risk_group') == 'low']
    if lo:
        r1l = mean_meth('bias_plus_temperature_R1', 'final_sv_ratio')
        bl = mean_meth('baseline', 'final_sv_ratio')
        r1l = statistics.mean([float(r['final_sv_ratio']) for r in lo
                               if r.get('method') == 'bias_plus_temperature_R1' and r.get('final_sv_ratio') != 'NA'])
        bl = statistics.mean([float(r['final_sv_ratio']) for r in lo
                              if r.get('method') == 'baseline' and r.get('final_sv_ratio') != 'NA'])
        if r1l < bl - 0.1:
            lines.append('- low-risk tile 上 R1 **可能过度抑制** SV')
        else:
            lines.append('- low-risk tile 副作用有限')
    (U.RESULT_MD / 'fres_06_minimal_repair_mechanism.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
