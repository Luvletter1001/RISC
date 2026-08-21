#!/usr/bin/env python
"""Experiment C: low-risk det_count / class histogram side-effect audit."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.exp_sv_dehub_lite_train_gpu89 import common_dehub_utils as du
from tools.exp_sv_dehub_lite_train_gpu89.plan3h import common_plan3h_utils as P
from tools.sv_attractor_repair_gpu89.common import parse_class_histogram


def read_csv(path: Path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def audit_row(rows, ckpt: str) -> dict:
    low = [r for r in rows if r.get('checkpoint') == ckpt and r.get('group') == 'low']
    sv_total, non_sv_total = 0, 0
    scores = []
    per_class = {}
    for r in low:
        hist = parse_class_histogram(r.get('top1_class_histogram', '{}'))
        det = int(float(r.get('det_count', 0) or 0))
        sv = int(hist.get('small-vehicle', 0))
        sv_total += sv
        non_sv_total += max(0, det - sv)
        for k, v in hist.items():
            per_class[k] = per_class.get(k, 0) + int(v)
        try:
            scores.append(float(r.get('mean_score', 0)))
        except (TypeError, ValueError):
            pass
    main_cls = max(per_class, key=per_class.get) if per_class else 'NA'
    fs = [float(r['final_sv_ratio']) for r in low]
    return dict(
        checkpoint=ckpt,
        n_angles=len(low),
        low_final_sv=statistics.mean(fs) if fs else float('nan'),
        low_det_count=statistics.mean([float(r['det_count']) for r in low]) if low else float('nan'),
        sv_det_total=sv_total,
        non_sv_det_total=non_sv_total,
        sv_fraction=sv_total / max(1, sv_total + non_sv_total),
        mean_score=statistics.mean(scores) if scores else float('nan'),
        main_changed_class=main_cls,
        top3_classes=json.dumps(
            sorted(per_class.items(), key=lambda x: -x[1])[:3], ensure_ascii=False),
        interpretation='',
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-csv', type=Path, default=P.OVERNIGHT_EVAL_RAW)
    args = ap.parse_args()
    P.ensure_dirs()
    rows = read_csv(args.eval_csv)
    out_csv = P.VERIFY_DIR / 'ftable_03_lowrisk_side_effect_audit.csv'
    audit_rows = []
    for ckpt, _ in P.CHECKPOINTS:
        ar = audit_row(rows, ckpt)
        bl_sv_frac = None
        audit_rows.append(ar)

    bl = audit_rows[0]
    for ar in audit_rows[1:]:
        det_ratio = ar['low_det_count'] / bl['low_det_count'] if bl['low_det_count'] else float('nan')
        if det_ratio > 1.5 and ar['non_sv_det_total'] > ar['sv_det_total']:
            ar['interpretation'] = 'non-SV detections increased (possible score_thr noise)'
        elif det_ratio > 1.5:
            ar['interpretation'] = 'det_count up; SV still dominant'
        elif ar['low_final_sv'] < bl['low_final_sv'] * 0.7:
            ar['interpretation'] = 'final_sv dropped on low-risk (hub suppression)'
        else:
            ar['interpretation'] = 'moderate change'

    fields = list(audit_rows[0].keys())
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(audit_rows)

    md = [
        '# Low-risk Side Effect Audit',
        '',
        f'- generated: {datetime.now().isoformat(timespec="seconds")}',
        f'- source: `{args.eval_csv}`',
        '',
        '## Table',
        '',
        '| checkpoint | low_final_sv | low_det_count | sv_det | non_sv_det | mean_score | main_class | interpretation |',
        '|---|---:|---:|---:|---:|---:|---|---|',
    ]
    for ar in audit_rows:
        md.append(
            f"| {ar['checkpoint']} | {ar['low_final_sv']:.3f} | {ar['low_det_count']:.1f} | "
            f"{ar['sv_det_total']} | {ar['non_sv_det_total']} | {ar['mean_score']:.3f} | "
            f"{ar['main_changed_class']} | {ar['interpretation']} |"
        )

    md += [
        '',
        '## Answers',
        '',
        '1. **det_count rise source:** compare sv_det_total vs non_sv_det_total per checkpoint.',
        '2. **Confidence:** mean_score column — drop suggests low-conf noise; rise suggests high-conf new dets.',
        '3. **AP impact:** see full AP eval in `fres_full_ap_eval.md`.',
        '4. **Safer checkpoint:** iter_1000 typically lower det_count than iter_2000 if side effect present.',
    ]
    (P.VERIFY_DIR / 'fres_03_lowrisk_side_effect_audit.md').write_text('\n'.join(md), encoding='utf-8')
    print('wrote', out_csv)


if __name__ == '__main__':
    main()
