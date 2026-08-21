#!/usr/bin/env python
"""Experiment A: real heldout AP50 / sv_AP50 for dehub checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.exp_sv_dehub_lite_train_gpu89 import common_dehub_utils as du
from tools.exp_sv_dehub_lite_train_gpu89.verify import common_verify_utils as V
from tools.sv_attractor_repair_gpu89 import common as RC
from tools.sv_attractor_repair_gpu89.eval_ap import infer_heldout_method, prepare_heldout_subset
from tools.exp_mechanism_sv_attractor_gpu89 import common_attractor_utils as mech


def read_csv(path: Path):
    if not path.exists():
        return []
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def append_csv(path: Path, row: dict, fields: list):
    new = not path.exists() or path.stat().st_size == 0
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, 'NA') for k in fields})


def tile_means_from_overnight(ckpt: str) -> dict:
    rows = read_csv(V.OVERNIGHT_EVAL_RAW)
    out = {}
    for grp in ('high', 'low'):
        fs = [float(r['final_sv_ratio']) for r in rows
              if r.get('checkpoint') == ckpt and r.get('group') == grp]
        ds = [float(r['dense_top1_sv_ratio']) for r in rows
              if r.get('checkpoint') == ckpt and r.get('group') == grp]
        dets = [float(r['det_count']) for r in rows
                if r.get('checkpoint') == ckpt and r.get('group') == grp]
        out[f'{grp}_final_sv'] = statistics.mean(fs) if fs else float('nan')
        out[f'{grp}_dense_sv'] = statistics.mean(ds) if ds else float('nan')
        out[f'{grp}_det_count'] = statistics.mean(dets) if dets else float('nan')
    return out


def eval_tiles_fresh(ckpt_name: str, ckpt_path: Path, out_raw: Path):
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '9')
    ctx = mech.MechContext(gpu=int(os.environ.get('CUDA_VISIBLE_DEVICES', '9')))
    ctx.checkpoint = ckpt_path
    ctx.work_dir = V.WORK_VERIFY / 'tile_eval' / ckpt_name
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    bargs, model, device, det_support, name2id, _, _ = mech.build_model(ctx)
    for grp, tiles in [('high', du.HIGHRISK_TILES), ('low', du.LOWRISK_TILES)]:
        for tile in tiles:
            for angle in du.EVAL_ANGLES:
                try:
                    row = mech.eval_one(ctx, model, bargs, device, det_support, name2id,
                                        tile, angle, 'baseline')
                    row['checkpoint'] = ckpt_name
                    row['group'] = grp
                    mech.append_csv(out_raw, row, du.SCHEMA_EVAL)
                except Exception as exc:
                    mech.append_csv(out_raw, dict(
                        checkpoint=ckpt_name, tile_id=tile, angle=angle, group=grp,
                        notes=str(exc)), du.SCHEMA_EVAL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--heldout-n', type=int, default=100,
                    help='Heldout images (100/200/500); documented in report')
    ap.add_argument('--refresh-tiles', action='store_true',
                    help='Re-run high/low tile eval (default: use overnight CSV)')
    args = ap.parse_args()
    V.ensure_dirs()
    log_path = V.VERIFY_DIR / 'log_gpu9_full_ap_eval.txt'
    raw_ap = V.VERIFY_DIR / 'ftable_full_ap_eval_raw.csv'
    summ_ap = V.VERIFY_DIR / 'ftable_full_ap_eval_summary.csv'
    per_class = V.VERIFY_DIR / 'ftable_per_class_ap.csv'
    tile_raw = V.VERIFY_DIR / 'ftable_tile_eval_verify.csv'

    held = V.load_heldout_stems(args.heldout_n)
    if len(held) < 10:
        log_path.write_text(f'FAIL: heldout stems={len(held)}\n', encoding='utf-8')
        print('heldout pool too small')
        return 1

    ap_fields = [
        'checkpoint', 'heldout_n', 'ap50', 'map_overall', 'sv_ap50', 'lv_ap50', 'ship_ap50',
        'detection_total', 'final_sv_ratio', 'high_final_sv', 'high_dense_sv', 'high_det_count',
        'low_final_sv', 'low_dense_sv', 'low_det_count', 'notes', 'timestamp',
    ]
    cls_fields = ['checkpoint', 'class_name', 'ap50', 'num_gts', 'num_dets']

    lines = [f'=== full AP eval start {datetime.now().isoformat()} heldout_n={len(held)} ===\n']
    rows_summary = []

    for ckpt_name, ckpt_path in V.CHECKPOINTS:
        if not ckpt_path.exists():
            lines.append(f'SKIP missing {ckpt_name} {ckpt_path}\n')
            continue
        lines.append(f'--- {ckpt_name} ---\n')
        ctx = RC.RepairContext(
            repo_root=REPO,
            config=V.FORMAL_CONFIG,
            checkpoint=ckpt_path,
            support_pkl=V.SUPPORT_PKL,
            verify_work_dir=V.REPAIR_WD,
            work_dir=V.WORK_VERIFY / 'heldout_ap' / ckpt_name,
            result_md_dir=V.VERIFY_DIR,
            gpu=int(os.environ.get('CUDA_VISIBLE_DEVICES', '9')),
            mode='full',
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
            subset = prepare_heldout_subset(ctx, len(held))
            res = infer_heldout_method(
                ctx, model, device, det_support, name2id, subset,
                'baseline', [0, 90], len(held), score_thr=RC.POSTPROCESS['score_thr'])
            lines.append(json.dumps({k: v for k, v in res.items() if k != 'per_class'}, default=str) + '\n')
        except Exception as exc:
            import traceback
            lines.append(traceback.format_exc() + '\n')
            res = dict(ap50_overall=float('nan'), sv_ap50=float('nan'), error=str(exc))

        tiles = tile_means_from_overnight(ckpt_name)
        row = dict(
            checkpoint=ckpt_name,
            heldout_n=len(held),
            ap50=res.get('ap50_overall', 'NA'),
            map_overall=res.get('map_overall', float('nan')),
            sv_ap50=res.get('sv_ap50', 'NA'),
            lv_ap50=res.get('lv_ap50', 'NA'),
            ship_ap50=res.get('ship_ap50', 'NA'),
            detection_total=res.get('detection_total', 'NA'),
            final_sv_ratio=res.get('final_sv_ratio', 'NA'),
            high_final_sv=tiles.get('high_final_sv', 'NA'),
            high_dense_sv=tiles.get('high_dense_sv', 'NA'),
            high_det_count=tiles.get('high_det_count', 'NA'),
            low_final_sv=tiles.get('low_final_sv', 'NA'),
            low_dense_sv=tiles.get('low_dense_sv', 'NA'),
            low_det_count=tiles.get('low_det_count', 'NA'),
            notes=res.get('error', ''),
            timestamp=datetime.now().isoformat(timespec='seconds'),
        )
        append_csv(raw_ap, row, ap_fields)
        rows_summary.append(row)
        for pc in res.get('per_class', []) or []:
            append_csv(per_class, dict(checkpoint=ckpt_name, **pc), cls_fields)

    with open(summ_ap, 'w', newline='', encoding='utf-8') as f:
        if rows_summary:
            w = csv.DictWriter(f, fieldnames=ap_fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows_summary)

    bl = next((r for r in rows_summary if r['checkpoint'] == 'baseline_epoch24'), None)
    fres_lines = [
        '# Full AP Eval — SV-DeHub-Lite v1',
        '',
        f'- date: {datetime.now().isoformat(timespec="seconds")}',
        f'- heldout_n: **{len(held)}** (calib_split heldout_calib)',
        f'- angles for AP: 0° only (`eval_rbbox_map`)',
        f'- angles for final_sv/det: 0°+90° mean',
        '',
        '## Summary',
        '',
        '| checkpoint | AP50 | sv_AP50 | high_final_sv | low_final_sv | det_total |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for r in rows_summary:
        fres_lines.append(
            f"| {r['checkpoint']} | {r['ap50']} | {r['sv_ap50']} | "
            f"{r['high_final_sv']} | {r['low_final_sv']} | {r['detection_total']} |"
        )
    if bl:
        fres_lines += ['', '## Delta vs baseline', '']
        try:
            b_ap = float(bl['ap50'])
            b_sv = float(bl['sv_ap50'])
            for r in rows_summary:
                if r['checkpoint'] == 'baseline_epoch24':
                    continue
                da = (float(r['ap50']) - b_ap) * 100 if r['ap50'] != 'NA' else float('nan')
                ds = (float(r['sv_ap50']) - b_sv) * 100 if r['sv_ap50'] != 'NA' else float('nan')
                fres_lines.append(
                    f"- **{r['checkpoint']}**: ΔAP50={da:+.2f} pt, Δsv_AP50={ds:+.2f} pt"
                )
        except (TypeError, ValueError):
            fres_lines.append('- delta computation failed (NA values)')

    (V.VERIFY_DIR / 'fres_full_ap_eval.md').write_text('\n'.join(fres_lines), encoding='utf-8')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.writelines(lines)
    print('wrote', summ_ap, 'rows', len(rows_summary))
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
