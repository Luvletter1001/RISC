#!/usr/bin/env python
"""Eval ablation checkpoints on high/low tiles + heldout AP smoke."""
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
from tools.exp_sv_dehub_lite_train_gpu89.plan3h.run_full_ap_eval_gpu9 import (
    append_csv, tile_means_from_overnight,
)
from tools.exp_mechanism_sv_attractor_gpu89 import common_attractor_utils as mech
from tools.sv_attractor_repair_gpu89 import common as RC
from tools.sv_attractor_repair_gpu89.eval_ap import infer_heldout_method, prepare_heldout_subset

ABLATIONS = [
    ('dehub_only', P.WORK_PLAN3H / 'dehub_only' / 'iter_1000.pth'),
    ('hard_negative_only', P.WORK_PLAN3H / 'paste_only' / 'iter_1000.pth'),
    ('head_finetune_only', P.WORK_PLAN3H / 'head_finetune_only' / 'iter_1000.pth'),
]


def eval_tiles(ckpt_name: str, ckpt_path: Path, out_raw: Path):
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '9')
    ctx = mech.MechContext(gpu=int(os.environ.get('CUDA_VISIBLE_DEVICES', '9')))
    ctx.checkpoint = ckpt_path
    ctx.work_dir = P.WORK_PLAN3H / 'ablation_tile' / ckpt_name
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    bargs, model, device, det_support, name2id, _, _ = mech.build_model(ctx)
    for grp, tiles in [('high', du.HIGHRISK_TILES), ('low', du.LOWRISK_TILES)]:
        for tile in tiles:
            for angle in du.EVAL_ANGLES:
                row = mech.eval_one(ctx, model, bargs, device, det_support, name2id,
                                    tile, angle, 'baseline')
                row['checkpoint'] = ckpt_name
                row['group'] = grp
                mech.append_csv(out_raw, row, du.SCHEMA_EVAL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--heldout-n', type=int, default=100)
    args = ap.parse_args()
    P.ensure_dirs()
    summ = P.VERIFY_DIR / 'ftable_02_ablation_eval_summary.csv'
    fields = [
        'variant', 'checkpoint', 'high_final_sv', 'high_dense_sv', 'low_final_sv',
        'low_det_count', 'ap50', 'sv_ap50', 'heldout_n', 'dehub_loss_mean', 'notes',
    ]
    rows_out = []
    held = P.load_heldout_stems(args.heldout_n)

    loss_rows = {}
    loss_path = P.VERIFY_DIR / 'ftable_ablation_train_loss.csv'
    if loss_path.exists():
        with open(loss_path, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                v = r.get('variant', '')
                try:
                    loss_rows.setdefault(v, []).append(float(r.get('loss_dehub', 0) or 0))
                except ValueError:
                    pass

    tile_tmp = P.VERIFY_DIR / 'ftable_ablation_tile_raw.csv'
    if tile_tmp.exists():
        tile_tmp.unlink()

    for variant, ckpt_path in ABLATIONS:
        if not ckpt_path.exists():
            rows_out.append(dict(variant=variant, checkpoint=str(ckpt_path),
                                notes='missing_ckpt'))
            continue
        eval_tiles(variant, ckpt_path, tile_tmp)
        trows = []
        with open(tile_tmp, encoding='utf-8') as f:
            trows = list(csv.DictReader(f))
        def mean_f(grp, col):
            vals = [float(r[col]) for r in trows if r.get('group') == grp]
            return statistics.mean(vals) if vals else float('nan')

        ap50, sv_ap = float('nan'), float('nan')
        notes = ''
        if held:
            ctx = RC.RepairContext(
                repo_root=REPO, config=P.FORMAL_CONFIG, checkpoint=ckpt_path,
                support_pkl=P.SUPPORT_PKL, verify_work_dir=P.REPAIR_WD,
                work_dir=P.WORK_PLAN3H / 'ablation_ap' / variant,
                result_md_dir=P.VERIFY_DIR, gpu=9, mode='full',
                angles=[0, 90], tiles=[], train_max_images=0,
                heldout_max_images=len(held), eval_max_images=len(held),
                iters=0, batch_size=1, lr=0.0,
            )
            try:
                _, model, device, det_support, name2id, _, _, _ = RC.build_model_ctx(ctx)
                subset = prepare_heldout_subset(ctx, len(held))
                res = infer_heldout_method(
                    ctx, model, device, det_support, name2id, subset,
                    'baseline', [0, 90], len(held))
                ap50 = res.get('ap50_overall', float('nan'))
                sv_ap = res.get('sv_ap50', float('nan'))
            except Exception as exc:
                notes = str(exc)

        dh = loss_rows.get(variant.replace('_only', '_only'), loss_rows.get(variant, []))
        if variant == 'v1_combined':
            dh = []
        rows_out.append(dict(
            variant=variant,
            checkpoint=ckpt_path.name,
            high_final_sv=mean_f('high', 'final_sv_ratio'),
            high_dense_sv=mean_f('high', 'dense_top1_sv_ratio'),
            low_final_sv=mean_f('low', 'final_sv_ratio'),
            low_det_count=mean_f('low', 'det_count'),
            ap50=ap50,
            sv_ap50=sv_ap,
            heldout_n=len(held),
            dehub_loss_mean=statistics.mean(dh) if dh else 0,
            notes=notes,
        ))

    with open(summ, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows_out)

    bl_h = statistics.mean([float(r['final_sv_ratio']) for r in list(csv.DictReader(open(P.OVERNIGHT_EVAL_RAW))) 
                            if r['checkpoint']=='baseline_epoch24' and r['group']=='high'])
    lines = [
        '# Minimal Ablation',
        '',
        f'- date: {datetime.now().isoformat(timespec="seconds")}',
        f'- heldout AP n: {len(held)}',
        '',
        '| variant | high_final_sv | high_dense_sv | AP50 | sv_AP50 | low_det | dehub_mean |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for r in rows_out:
        lines.append(
            f"| {r['variant']} | {r['high_final_sv']:.3f} | {r['high_dense_sv']:.3f} | "
            f"{r['ap50']} | {r['sv_ap50']} | {r['low_det_count']:.1f} | {r['dehub_loss_mean']} |"
        )
    lines += ['', f'- baseline high final_sv (overnight): {bl_h:.3f}', '']
    (P.VERIFY_DIR / 'fres_ablation_minimal.md').write_text('\n'.join(lines), encoding='utf-8')
    print('wrote', summ)


if __name__ == '__main__':
    main()
