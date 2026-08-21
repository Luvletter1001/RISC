#!/usr/bin/env python
"""Retry atlas rows with NA / division-by-zero (angle_sweep pkl fix)."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.exp_mechanism_sv_attractor_gpu89 import common_attractor_utils as U
from tools.exp_mechanism_sv_attractor_gpu89.run_gpu9_attractor_atlas import (
    ATLAS_CSV, METHODS, summarize_atlas, TILE_LIST_JSON)

RESULT = U.RESULT_MD


def failed_jobs(csv_path: Path):
    rows = U.read_csv(csv_path)
    jobs = []
    for r in rows:
        try:
            float(r.get('dense_top1_sv_ratio', 'NA'))
            continue
        except (TypeError, ValueError):
            pass
        jobs.append((r['tile_id'], int(float(r['angle'])), r['method']))
    return jobs


def prune_failed_rows(csv_path: Path, jobs: set):
    """Remove NA rows that will be re-appended (optional clean)."""
    if not csv_path.exists() or not jobs:
        return
    rows = U.read_csv(csv_path)
    keyset = {(t, a, m) for t, a, m in jobs}
    kept = [r for r in rows
            if (r['tile_id'], int(float(r['angle'])), r['method']) not in keyset]
    if len(kept) == len(rows):
        return
    fields = list(rows[0].keys()) if rows else U.SCHEMA_ATLAS
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=9)
    ap.add_argument('--prune', action='store_true', help='Remove stale NA rows before retry')
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    ctx = U.MechContext(gpu=args.gpu)
    ctx.work_dir = U.WORK_ROOT / 'gpu9_resume'
    U.setup_logging(ctx.work_dir, 'resume')

    jobs = failed_jobs(ATLAS_CSV)
    print(f'failed jobs to retry: {len(jobs)}')
    if not jobs:
        print('nothing to retry')
        return

    if args.prune:
        prune_failed_rows(ATLAS_CSV, set(jobs))

    bargs, model, device, det_support, name2id, _, _ = U.build_model(ctx)
    ok, fail = 0, 0
    for tile_id, angle, method in jobs:
        st = 'zero_sv' if method == 'zero_sv_embedding' else 'original'
        try:
            row = U.eval_one(ctx, model, bargs, device, det_support, name2id,
                             tile_id, angle, method, support_intervention=st)
            row['run_id'] = U.row_key(tile_id, angle, method)
            U.append_csv(ATLAS_CSV, row, U.SCHEMA_ATLAS)
            if row.get('dense_top1_sv_ratio', 'NA') not in ('NA', ''):
                ok += 1
            else:
                fail += 1
                print('still NA', tile_id, angle, method, row.get('notes'))
        except Exception as exc:
            fail += 1
            U.log.exception('retry %s %s %s', tile_id, angle, method)
            U.append_csv(ATLAS_CSV, dict(
                tile_id=tile_id, angle=angle, method=method, notes=str(exc)),
                U.SCHEMA_ATLAS)

    summarize_atlas(ATLAS_CSV, RESULT / 'fres_01_attractor_atlas.md', TILE_LIST_JSON)
    from tools.exp_mechanism_sv_attractor_gpu89 import analyze_variance_decomposition as avd
    from tools.exp_mechanism_sv_attractor_gpu89 import analyze_risk_predictor as arp
    avd.main()
    arp.main()
    from tools.exp_mechanism_sv_attractor_gpu89 import build_final_mechanism_report as brep
    brep.main()
    print(f'retry done ok={ok} fail={fail}')


if __name__ == '__main__':
    main()
