#!/usr/bin/env python
"""Task B: heldout AP50 / sv_AP50 per official stage."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.exp_official_step123_sv_attractor_eval.common_official_utils import (
    REPAIR_WD, RESULT_DIR, STAGES, SUPPORT_FALLBACK, WORK_DIR, append_csv, ensure_dirs,
    head_label, load_heldout_stems,
)
from tools.sv_attractor_repair_gpu89 import common as RC
from tools.sv_attractor_repair_gpu89.eval_ap import infer_heldout_method, prepare_heldout_subset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=list(STAGES.keys()), required=True)
    ap.add_argument('--heldout-n', type=int, default=200)
    ap.add_argument('--gpu', type=int, default=9)
    ap.add_argument('--fusion', action='store_true', help='Also eval fusion head')
    args = ap.parse_args()
    ensure_dirs()
    spec = STAGES[args.stage]
    if not spec.primary_ckpt.exists():
        print('missing ckpt', spec.primary_ckpt)
        return 1
    held = load_heldout_stems(args.heldout_n)
    if len(held) < 10:
        print('heldout too small', len(held))
        return 1

    raw_path = RESULT_DIR / 'ftable_20_heldout_ap_raw.csv'
    summ_path = RESULT_DIR / 'ftable_21_heldout_ap_summary.csv'
    per_class = RESULT_DIR / 'ftable_22_heldout_per_class_ap.csv'
    fields = [
        'stage', 'checkpoint', 'support_mode', 'head_mode', 'val_using_aux', 'heldout_n',
        'ap50', 'sv_ap50', 'lv_ap50', 'ship_ap50', 'detection_total', 'final_sv_ratio',
        'notes', 'timestamp',
    ]

    for aux in ([False, True] if args.fusion and spec.dense_hook_ok else [False]):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        ctx = RC.RepairContext(
            repo_root=REPO,
            config=spec.config,
            checkpoint=spec.primary_ckpt,
            support_pkl=SUPPORT_FALLBACK,
            verify_work_dir=REPAIR_WD,
            work_dir=WORK_DIR / 'heldout_ap' / args.stage / head_label(aux),
            result_md_dir=RESULT_DIR,
            gpu=args.gpu,
            mode='official_step123',
            angles=[0, 90],
            tiles=[],
            train_max_images=0,
            heldout_max_images=len(held),
            eval_max_images=len(held),
            iters=0,
            batch_size=1,
            lr=0.0,
        )
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        try:
            bargs, model, device, det_support, name2id, _, _, _ = RC.build_model_ctx(ctx)
            if hasattr(model, 'val_using_aux'):
                model.val_using_aux = aux
            subset = prepare_heldout_subset(ctx, len(held))
            res = infer_heldout_method(
                ctx, model, device, det_support, name2id, subset,
                'baseline', [0, 90], len(held), score_thr=RC.POSTPROCESS['score_thr'])
            err = res.get('error', '')
        except Exception as exc:
            res = dict(ap50_overall=float('nan'), sv_ap50=float('nan'), error=str(exc))
            err = str(exc)

        row = dict(
            stage=args.stage,
            checkpoint=str(spec.primary_ckpt),
            support_mode='visual',
            head_mode=head_label(aux),
            val_using_aux=aux,
            heldout_n=len(held),
            ap50=res.get('ap50_overall', 'NA'),
            sv_ap50=res.get('sv_ap50', 'NA'),
            lv_ap50=res.get('lv_ap50', 'NA'),
            ship_ap50=res.get('ship_ap50', 'NA'),
            detection_total=res.get('detection_total', 'NA'),
            final_sv_ratio=res.get('final_sv_ratio', 'NA'),
            notes=err,
            timestamp=datetime.now().isoformat(timespec='seconds'),
        )
        append_csv(raw_path, row, fields)
        for pc in res.get('per_class', []) or []:
            append_csv(per_class, dict(stage=args.stage, head_mode=head_label(aux), **pc),
                       ['stage', 'head_mode', 'class_name', 'ap50', 'num_gts', 'num_dets'])

    # rewrite summary from raw
    import csv
    rows = []
    if raw_path.exists():
        with open(raw_path, encoding='utf-8') as f:
            rows = [r for r in csv.DictReader(f) if r.get('stage') == args.stage]
    with open(summ_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        if f.tell() == 0:
            w.writeheader()
        w.writerows(rows)
    print('AP done', args.stage, 'heldout', len(held))
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
