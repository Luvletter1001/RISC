#!/usr/bin/env python
"""Exp5: risk predictor (csv-only, no pandas DataFrame from dicts)."""
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

FEAT_CSV = U.RESULT_MD / 'ftable_05_risk_predictor_features.csv'
RES_CSV = U.RESULT_MD / 'ftable_05_risk_predictor_results.csv'
OUT_MD = U.RESULT_MD / 'fres_05_risk_predictor.md'


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def label_risk(v: float) -> str:
    if v >= 0.5:
        return 'high'
    if v >= 0.2:
        return 'medium'
    return 'low'


def pearson(xs, ys):
    pairs = [(a, b) for a, b in zip(xs, ys) if a == a and b == b]
    if len(pairs) < 3:
        return float('nan')
    mx = statistics.mean([p[0] for p in pairs])
    my = statistics.mean([p[1] for p in pairs])
    num = sum((a - mx) * (b - my) for a, b in pairs)
    den = (sum((a - mx) ** 2 for a, _ in pairs) * sum((b - my) ** 2 for _, b in pairs)) ** 0.5
    return num / den if den else float('nan')


def main():
    atlas = U.read_csv(U.RESULT_MD / 'ftable_01_attractor_atlas_raw.csv')
    if not atlas:
        OUT_MD.write_text('# Risk Predictor\n\nNo data.\n')
        return

    by_tile = defaultdict(lambda: defaultdict(list))
    for r in atlas:
        if r.get('method') != 'baseline':
            continue
        tid = r['tile_id']
        for k in ('final_sv_ratio', 'dense_top1_sv_ratio', 'dense_sv_margin_vs_runnerup',
                  'dense_entropy', 'objectness_p95', 'background_to_sv_cosine_mean', 'gt_sv_count'):
            by_tile[tid][k].append(_f(r.get(k)))

    features = []
    for tid, d in by_tile.items():
        fs = _f(statistics.mean(d['final_sv_ratio']))
        features.append(dict(
            tile_id=tid,
            final_sv_ratio=fs,
            dense_top1_sv_ratio=statistics.mean(d['dense_top1_sv_ratio']),
            dense_sv_margin_vs_runnerup=statistics.mean(d['dense_sv_margin_vs_runnerup']),
            dense_entropy=statistics.mean(d['dense_entropy']),
            objectness_p95=statistics.mean(d['objectness_p95']),
            background_to_sv_cosine_mean=statistics.mean(d['background_to_sv_cosine_mean']),
            gt_sv_count=max(d['gt_sv_count']) if d['gt_sv_count'] else 0,
            risk_label=label_risk(fs),
        ))

    with open(FEAT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(features[0].keys()))
        w.writeheader()
        w.writerows(features)

    y = [1.0 if ft['risk_label'] == 'high' else 0.0 for ft in features]
    preds = ['dense_sv_margin_vs_runnerup', 'dense_top1_sv_ratio', 'objectness_p95',
             'background_to_sv_cosine_mean', 'dense_entropy', 'gt_sv_count']
    results = []
    for p in preds:
        x = [_f(ft[p]) for ft in features]
        results.append(dict(predictor=p, pearson_with_high=round(pearson(x, y), 4)))

    best = max(results, key=lambda r: abs(r['pearson_with_high']) if r['pearson_with_high'] == r['pearson_with_high'] else 0)

    with open(RES_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['predictor', 'pearson_with_high'])
        w.writeheader()
        w.writerows(results)

    lines = [
        '# Risk Predictor (Exp5)',
        '',
        f'- strongest feature: **{best["predictor"]}** (r={best["pearson_with_high"]})',
        '',
        '## 机制指向',
        '',
    ]
    if 'margin' in best['predictor'] or 'dense_top1' in best['predictor']:
        lines.append('- 支持 **H1 embedding/margin hub** 作为 tile 级 diagnostic。')
    if 'cosine' in best['predictor']:
        lines.append('- 支持 **H2 background–SV 特征相似度**。')
    lines.append('- 可在 angle=0 仅跑 dense 统计作 **轻量预警**（无需 full detector 网格）。')

    OUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('wrote', OUT_MD)


if __name__ == '__main__':
    main()
