#!/usr/bin/env python
"""Heldout AP + mechanism eval for overnight dehub checkpoints (GPU9)."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.exp_mechanism_sv_attractor_gpu89 import common_attractor_utils as mech
from tools.exp_next_plan_sv_dehub_step23_overnight import common_overnight_utils as U
from tools.exp_sv_dehub_lite_train_gpu89 import common_dehub_utils as du
from tools.exp_sv_dehub_lite_train_gpu89.plan3h import common_plan3h_utils as P
from tools.exp_sv_dehub_lite_train_gpu89.plan3h.run_full_ap_eval_gpu9 import (
    eval_tiles_fresh, read_csv, append_csv,
)
from tools.sv_attractor_repair_gpu89.eval_ap import infer_heldout_method, prepare_heldout_subset
from tools.sv_attractor_repair_gpu89 import common as RC


def discover_branch_ckpts(branch: str) -> list:
    wd = U.BRANCHES[branch]['work']
    out = []
    for p in sorted(wd.glob('iter_*.pth')):
        out.append((f'{branch}_{p.stem}', p))
    for p in sorted((wd / 'checkpoints').glob('*.pth')) if (wd / 'checkpoints').exists() else []:
        out.append((f'{branch}_{p.stem}', p))
    return out


def run_ap(ctx: RC.RepairContext, held_n: int, raw_path: Path, summ_fields: list):
    held = U.load_heldout_stems(held_n)
    if len(held) < 10:
        return dict(notes='heldout_empty')
    subset = prepare_heldout_subset(ctx, len(held))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(ctx.gpu)
    bargs, model, device, det_support, name2id, _, _, _ = RC.build_model_ctx(ctx)
    res = infer_heldout_method(
        ctx, model, device, det_support, name2id, subset,
        'baseline', [0, 90], len(held), score_thr=RC.POSTPROCESS['score_thr'])
    row = dict(
        checkpoint=str(ctx.checkpoint),
        heldout_n=len(held),
        ap50=res.get('ap50_overall', 'NA'),
        sv_ap50=res.get('sv_ap50', 'NA'),
        detection_total=res.get('detection_total', 'NA'),
        final_sv_ratio=res.get('final_sv_ratio', 'NA'),
        notes=res.get('error', ''),
        timestamp=datetime.now().isoformat(timespec='seconds'),
    )
    U.append_csv(raw_path, row, summ_fields)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--heldout-n', type=int, default=500)
    ap.add_argument('--gpu', type=int, default=9)
    ap.add_argument('--baseline-only', action='store_true')
    args = ap.parse_args()
    U.ensure_dirs()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    summ500 = U.RESULT_DIR / 'ftable_24_heldout500_ap_summary.csv'
    raw500 = U.RESULT_DIR / 'ftable_24_heldout500_ap_raw.csv'
    summ200 = U.RESULT_DIR / 'ftable_22_heldout200_ap_summary.csv'
    fields = ['checkpoint', 'heldout_n', 'ap50', 'sv_ap50', 'detection_total',
              'final_sv_ratio', 'notes', 'timestamp']

    jobs = [('baseline_epoch24', U.INIT_CKPT)]
    if not args.baseline_only:
        for br in ('A', 'B', 'C'):
            jobs.extend(discover_branch_ckpts(br))

    for name, ckpt in jobs:
        if not ckpt.exists():
            continue
        ctx = RC.RepairContext(
            repo_root=REPO, config=U.FORMAL_CONFIG, checkpoint=ckpt,
            support_pkl=P.SUPPORT_PKL, verify_work_dir=U.REPAIR_WD,
            work_dir=U.WORK_DIR / 'heldout_ap' / name,
            result_md_dir=U.RESULT_DIR, gpu=args.gpu, mode='overnight',
            angles=[0, 90], tiles=[], train_max_images=0,
            heldout_max_images=args.heldout_n, eval_max_images=args.heldout_n,
            iters=0, batch_size=1, lr=0.0,
        )
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        dest = summ500 if args.heldout_n >= 400 else summ200
        try:
            run_ap(ctx, args.heldout_n, raw500, fields)
            print('AP ok', name, args.heldout_n)
        except Exception as exc:
            U.append_csv(raw500, dict(checkpoint=str(ckpt), notes=str(exc)), fields)
            print('AP fail', name, exc)

    if not args.baseline_only:
        tile_raw = U.RESULT_DIR / 'ftable_20_dehub_mechanism_eval_raw.csv'
        for name, ckpt in jobs:
            if not ckpt.exists():
                continue
            eval_tiles_fresh(name, ckpt, tile_raw)

    print('heldout eval done', args.heldout_n)


if __name__ == '__main__':
    sys.exit(main() or 0)
