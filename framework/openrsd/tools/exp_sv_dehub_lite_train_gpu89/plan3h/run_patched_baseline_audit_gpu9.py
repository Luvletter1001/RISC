#!/usr/bin/env python
"""Exp0: patched baseline consistency vs overnight reference."""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.exp_sv_dehub_lite_train_gpu89 import common_dehub_utils as du
from tools.exp_sv_dehub_lite_train_gpu89.plan3h import common_plan3h_utils as P
from tools.exp_mechanism_sv_attractor_gpu89 import common_attractor_utils as mech


def eval_patched_baseline(out_raw: Path):
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '9')
    ckpt = P.BASE_CKPT
    ctx = mech.MechContext(gpu=int(os.environ.get('CUDA_VISIBLE_DEVICES', '9')))
    ctx.config = P.FORMAL_CONFIG
    ctx.checkpoint = ckpt
    ctx.work_dir = P.WORK_PLAN3H / 'patched_baseline_audit'
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    bargs, model, device, det_support, name2id, _, _ = mech.build_model(ctx)
    for grp, tiles in [('high', du.HIGHRISK_TILES), ('low', du.LOWRISK_TILES)]:
        for tile in tiles:
            for angle in du.EVAL_ANGLES:
                row = mech.eval_one(ctx, model, bargs, device, det_support, name2id,
                                    tile, angle, 'baseline')
                row['checkpoint'] = 'patched_baseline_epoch24'
                row['group'] = grp
                mech.append_csv(out_raw, row, du.SCHEMA_EVAL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=9)
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    P.ensure_dirs()
    log = P.VERIFY_DIR / 'log_gpu9_patched_baseline_consistency.txt'
    out_raw = P.WORK_PLAN3H / 'ftable_00_patched_baseline_tile_raw.csv'
    out_csv = P.VERIFY_DIR / 'ftable_00_patched_baseline_consistency.csv'
    if out_raw.exists():
        out_raw.unlink()
    lines = [f'=== patched baseline audit {datetime.now().isoformat()} ===\n']
    eval_patched_baseline(out_raw)
    rows = []
    with open(out_raw, newline='', encoding='utf-8') as f:
        raw = list(csv.DictReader(f))
    worst = 'CONSISTENT'
    for grp in ('high', 'low'):
        sub = [r for r in raw if r.get('group') == grp]
        ref = P.OLD_BASELINE_REF[grp]
        fs = statistics.mean([float(r['final_sv_ratio']) for r in sub])
        ds = statistics.mean([float(r['dense_top1_sv_ratio']) for r in sub])
        dc = statistics.mean([float(r['det_count']) for r in sub])
        for metric, new, old in [
            ('final_sv', fs, ref['final_sv']),
            ('dense_top1_sv', ds, ref['dense_top1_sv']),
            ('det_count', dc, ref['det_count']),
        ]:
            rd = P.rel_diff(new, old)
            st = P.consistency_status(rd)
            rows.append(dict(group=grp, metric=metric, old_baseline=old,
                             patched_baseline=new, rel_diff=rd, status=st))
            if {'CONSISTENT': 0, 'SHIFTED': 1, 'BROKEN': 2}[st] > {'CONSISTENT': 0, 'SHIFTED': 1, 'BROKEN': 2}[worst]:
                worst = st
    fields = ['group', 'metric', 'old_baseline', 'patched_baseline', 'rel_diff', 'status']
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    md = [
        '# Patched Baseline Consistency Audit',
        '',
        f'- verdict: **{worst}**',
        f'- checkpoint: epoch_24 (formal config, dehub off at inference)',
        '',
        '| group | metric | old | patched | rel_diff | status |',
        '|---|---|---:|---:|---:|---|',
    ]
    for r in rows:
        md.append(f"| {r['group']} | {r['metric']} | {r['old_baseline']:.4f} | {r['patched_baseline']:.4f} | {r['rel_diff']:.4f} | {r['status']} |")
    (P.VERIFY_DIR / 'fres_00_patched_baseline_consistency.md').write_text('\n'.join(md), encoding='utf-8')
    lines.append(f'verdict={worst}\n')
    log.write_text(''.join(lines), encoding='utf-8')
    print('verdict', worst)
    return 0 if worst != 'BROKEN' else 2


if __name__ == '__main__':
    sys.exit(main() or 0)
