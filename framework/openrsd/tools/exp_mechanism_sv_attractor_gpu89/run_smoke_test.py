#!/usr/bin/env python
"""Quick smoke: one forward + R1 bias-only on P0148 angle 0."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.exp_mechanism_sv_attractor_gpu89 import common_attractor_utils as U


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '8')
    ctx = U.MechContext()
    ctx.work_dir = U.WORK_ROOT / 'smoke'
    (ctx.work_dir / 'logs').mkdir(parents=True, exist_ok=True)
    bargs, model, device, det_support, name2id, _, _ = U.build_model(ctx)
    tid = U.DEFAULT_TILE
    tests = [
        ('baseline', 'original'),
        ('bias_only_R1', 'original'),
        ('zero_sv_embedding', 'zero_sv'),
    ]
    ok = True
    for method, st in tests:
        try:
            row = U.eval_one(ctx, model, bargs, device, det_support, name2id,
                             tid, 0, method, support_intervention=st)
            assert row.get('dense_top1_sv_ratio', 'NA') != 'NA', row.get('notes')
            print(f'OK {method}: dense_sv={row["dense_top1_sv_ratio"]} final_sv={row["final_sv_ratio"]}')
        except Exception as exc:
            ok = False
            print(f'FAIL {method}: {exc}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
