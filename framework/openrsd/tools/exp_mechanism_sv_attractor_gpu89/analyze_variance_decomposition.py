#!/usr/bin/env python
"""Exp4: variance decomposition from atlas CSV (stdlib + csv)."""
from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.exp_mechanism_sv_attractor_gpu89 import common_attractor_utils as U

OUT_CSV = U.RESULT_MD / 'ftable_04_variance_decomposition.csv'
OUT_MD = U.RESULT_MD / 'fres_04_rotation_tile_variance_decomposition.md'


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def anova_share(values, labels):
    vals = [_f(v) for v in values]
    labs = list(labels)
    pairs = [(a, b) for a, b in zip(vals, labs) if a == a]
    if len(pairs) < 2:
        return 0.0
    grand = statistics.mean([p[0] for p in pairs])
    total = sum((a - grand) ** 2 for a, _ in pairs)
    if total < 1e-12:
        return 0.0
    between = 0.0
    for lab in set(labs):
        sub = [a for a, l in pairs if l == lab]
        if not sub:
            continue
        m = statistics.mean(sub)
        between += len(sub) * (m - grand) ** 2
    return between / total


def main():
    raw = U.RESULT_MD / 'ftable_01_attractor_atlas_raw.csv'
    rows = [r for r in U.read_csv(raw) if r.get('method') == 'baseline']
    if not rows:
        OUT_MD.write_text('# Variance decomposition\n\nNo atlas data.\n')
        return

    summ_rows = []
    for metric in ['final_sv_ratio', 'dense_top1_sv_ratio', 'dense_sv_margin_vs_runnerup']:
        tile_s = anova_share([r[metric] for r in rows], [r['tile_id'] for r in rows])
        ang_s = anova_share([r[metric] for r in rows], [r['angle'] for r in rows])
        summ_rows.append(dict(
            metric=metric,
            tile_variance_share=round(tile_s, 4),
            angle_variance_share=round(ang_s, 4),
            interaction_share=round(max(0.0, 1.0 - tile_s - ang_s), 4),
        ))

    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summ_rows[0].keys()))
        w.writeheader()
        w.writerows(summ_rows)

    fs = summ_rows[0]
    lines = [
        '# Rotation vs Tile Variance Decomposition (Exp4)',
        '',
        '| metric | tile_variance_share | angle_variance_share | interaction_share | conclusion |',
        '|---|---:|---:|---:|---|',
    ]
    for r in summ_rows:
        if r['tile_variance_share'] > r['angle_variance_share'] * 1.5:
            concl = 'tile/background dominated'
        elif r['angle_variance_share'] > r['tile_variance_share'] * 1.5:
            concl = 'rotation trigger dominated'
        else:
            concl = 'mixed tile×angle'
        lines.append(
            f"| {r['metric']} | {r['tile_variance_share']:.3f} | "
            f"{r['angle_variance_share']:.3f} | {r['interaction_share']:.3f} | {concl} |")

    tile_avg = defaultdict(list)
    for r in rows:
        tile_avg[r['tile_id']].append(_f(r['final_sv_ratio']))
    p0148 = statistics.mean(tile_avg[U.DEFAULT_TILE]) if U.DEFAULT_TILE in tile_avg else float('nan')
    others = [statistics.mean(v) for t, v in tile_avg.items() if t != U.DEFAULT_TILE]
    omed = statistics.median(others) if others else float('nan')

    lines += [
        '',
        '## P0148 特异性',
        '',
        f'- P0148 mean final_sv: **{p0148:.3f}**',
        f'- other tiles median: **{omed:.3f}**',
        '',
        '## 论文问题定义建议',
        '',
    ]
    if fs['tile_variance_share'] > 0.5:
        lines.append('- 主问题：**场景/背景相关的 OVD 分类 hub**；旋转是调制因子。')
    else:
        lines.append('- 主问题保留 **旋转–分类头非等变性** 与场景交互。')

    OUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('wrote', OUT_MD)


if __name__ == '__main__':
    main()
