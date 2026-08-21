#!/usr/bin/env python
"""Post-hoc analysis tables for SV-DeHub-Lite v1 eval and training curves."""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.exp_sv_dehub_lite_train_gpu89 import common_dehub_utils as U
from tools.exp_sv_dehub_lite_train_gpu89 import patch_config_sv_dehub_lite as C


def mean_metric(rows, ckpt: str, group: str, field: str) -> float:
    vals = []
    for r in rows:
        if r.get('checkpoint') != ckpt or r.get('group') != group:
            continue
        try:
            vals.append(float(r.get(field, 0) or 0))
        except (TypeError, ValueError):
            pass
    return statistics.mean(vals) if vals else float('nan')


def compute_verdict(rows, ckpt: str, ap_rows: list) -> str:
    bl_h = mean_metric(rows, 'baseline_epoch24', 'high', 'final_sv_ratio')
    bl_d = mean_metric(rows, 'baseline_epoch24', 'high', 'dense_top1_sv_ratio')
    hf = mean_metric(rows, ckpt, 'high', 'final_sv_ratio')
    hd = mean_metric(rows, ckpt, 'high', 'dense_top1_sv_ratio')
    if bl_h != bl_h or hf != hf:
        return 'FAILED_DEBUG'
    rel_f = hf / bl_h - 1 if bl_h else 0
    rel_d = hd / bl_d - 1 if bl_d else 0

    ap_row = next((r for r in ap_rows if r.get('checkpoint') == ckpt), None)
    ap_note = (ap_row or {}).get('notes', '')
    if rel_f <= -0.30 and rel_d <= -0.30:
        base = 'PROMISING'
    elif rel_d <= -0.30 and rel_f > -0.15:
        base = 'DENSE_ONLY_SUCCESS'
    elif rel_f <= -0.10:
        base = 'AP_TRADEOFF'
    elif abs(rel_f) < 0.05:
        base = 'INEFFECTIVE'
    else:
        base = 'PARTIAL'

    if 'proxy' in str(ap_note).lower() and base == 'PROMISING':
        return base
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result-dir', type=Path, default=C.RESULT_DIR)
    args = ap.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)

    rows = U.read_csv(args.result_dir / C.EVAL_RAW.name)
    curve = U.read_csv(C.CURVE_CSV)
    ap_rows = U.read_csv(C.EVAL_AP)

    bl_h = mean_metric(rows, 'baseline_epoch24', 'high', 'final_sv_ratio')
    bl_l = mean_metric(rows, 'baseline_epoch24', 'low', 'final_sv_ratio')
    bl_d = mean_metric(rows, 'baseline_epoch24', 'high', 'dense_top1_sv_ratio')

    detail_rows = []
    for ck in sorted({r['checkpoint'] for r in rows}):
        if ck == 'baseline_epoch24':
            continue
        hf = mean_metric(rows, ck, 'high', 'final_sv_ratio')
        hl = mean_metric(rows, ck, 'low', 'final_sv_ratio')
        hd = mean_metric(rows, ck, 'high', 'dense_top1_sv_ratio')
        det = mean_metric(rows, ck, 'high', 'det_count')
        delta = (hf / bl_h - 1) * 100 if bl_h == bl_h else 0
        detail_rows.append(dict(
            checkpoint=ck,
            group='high',
            dense_top1_sv=f'{hd:.4f}',
            final_sv=f'{hf:.4f}',
            det_count=f'{det:.1f}',
            delta_final_sv_pct=f'{delta:+.1f}',
            verdict=compute_verdict(rows, ck, ap_rows),
            low_final_sv=f'{hl:.4f}',
        ))

    out_detail = args.result_dir / 'ftable_analyze_checkpoint_verdict.csv'
    if detail_rows:
        with open(out_detail, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            w.writeheader()
            w.writerows(detail_rows)

    meta = {
        'analyzed_at': datetime.now().isoformat(),
        'eval_rows': len(rows),
        'curve_rows': len(curve),
        'baseline_high_final_sv': bl_h,
        'baseline_low_final_sv': bl_l,
        'baseline_high_dense_sv': bl_d,
        'best_verdict_ckpt': max(detail_rows, key=lambda x: -float(x['delta_final_sv_pct'].replace('%', '').replace('+', '')))['checkpoint'] if detail_rows else 'none',
        'verdicts': {r['checkpoint']: r['verdict'] for r in detail_rows},
    }
    import json
    (args.result_dir / 'fmeta_analyze_sv_dehub.json').write_text(
        json.dumps(meta, indent=2), encoding='utf-8')

    print('analyze: baseline high final_sv', f'{bl_h:.3f}')
    for r in detail_rows:
        print(f"  {r['checkpoint']}: delta {r['delta_final_sv_pct']}% -> {r['verdict']}")
    print('wrote', out_detail)


if __name__ == '__main__':
    main()
