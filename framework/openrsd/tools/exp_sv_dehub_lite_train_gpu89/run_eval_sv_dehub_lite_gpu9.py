#!/usr/bin/env python
"""Evaluate baseline + dehub checkpoints on high/low-risk tiles and AP smoke."""
from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.exp_sv_dehub_lite_train_gpu89 import common_dehub_utils as U
from tools.exp_mechanism_sv_attractor_gpu89 import common_attractor_utils as mech

log = logging.getLogger('sv_dehub_eval')


def setup_logging(result_dir: Path):
    log_path = result_dir / 'log_gpu9_eval_sv_dehub_lite_v1.txt'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    log.addHandler(fh)
    log.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) for h in log.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(message)s'))
        log.addHandler(sh)


def eval_checkpoint(ckpt_name: str, ckpt_path: Path, tiles: list, group: str, out_csv: Path):
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '9')
    ctx = mech.MechContext(gpu=int(os.environ.get('CUDA_VISIBLE_DEVICES', '9')))
    ctx.checkpoint = ckpt_path
    ctx.work_dir = U.WORK_ROOT / 'eval' / ckpt_name
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    bargs, model, device, det_support, name2id, _, _ = mech.build_model(ctx)
    for tile in tiles:
        for angle in U.EVAL_ANGLES:
            key_done = any(
                r.get('tile_id') == tile and int(float(r.get('angle', -1))) == angle
                and r.get('checkpoint') == ckpt_name
                for r in U.read_csv(out_csv))
            if key_done:
                continue
            try:
                row = mech.eval_one(ctx, model, bargs, device, det_support, name2id,
                                    tile, angle, 'baseline', support_intervention='original')
                row['checkpoint'] = ckpt_name
                row['group'] = group
                row['method'] = 'baseline'
                row['dense_sv_margin'] = row.get('dense_sv_margin_vs_runnerup', 'NA')
                row['objectness_proxy'] = row.get('objectness_mean', 'NA')
                row['ap50'] = row.get('ap50', 'NA')
                row['sv_ap50'] = row.get('sv_ap50', 'NA')
                mech.append_csv(out_csv, row, U.SCHEMA_EVAL)
                log.info('eval ok %s %s %s', ckpt_name, tile[:20], angle)
            except Exception as exc:
                log.exception('eval fail %s %s %s: %s', ckpt_name, tile, angle, exc)
                mech.append_csv(out_csv, dict(
                    checkpoint=ckpt_name, tile_id=tile, angle=angle, group=group,
                    notes=str(exc)), U.SCHEMA_EVAL)


def _valid_heldout_stems(repo: Path, held: list, angle: int = 0) -> list:
    valid = []
    try:
        asv = mech.angle_sweep_dir(repo, angle)
    except Exception:
        return []
    for h in held:
        for ext in ('.png', '.jpg'):
            if (asv / 'images' / f'{h}{ext}').exists():
                valid.append(h)
                break
    return valid


def eval_ap_smoke(ckpt_name: str, ckpt_path: Path, out_csv: Path, n_images: int = 120):
    """Heldout detection proxy: mean final_sv and det_count on angle_sweep heldout stems."""
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '9')
    held = _valid_heldout_stems(REPO, mech.heldout_candidates(REPO, 200))
    ap_note_extra = ''
    if not held:
        held = list(U.LOWRISK_TILES)
        ap_note_extra = 'heldout_angle_sweep_missing; using low-risk control tiles as proxy'
    if not held:
        mech.append_csv(out_csv, dict(
            checkpoint=ckpt_name, ap50='NA', sv_ap50='NA',
            notes='heldout_candidates empty'), [
            'checkpoint', 'ap50', 'sv_ap50', 'det_count_proxy', 'final_sv_proxy',
            'n_images', 'notes',
        ])
        return

    random.seed(42)
    sample = held[:min(n_images, len(held))]
    ctx = mech.MechContext(gpu=int(os.environ.get('CUDA_VISIBLE_DEVICES', '9')))
    ctx.checkpoint = ckpt_path
    ctx.work_dir = U.WORK_ROOT / 'eval' / f'{ckpt_name}_ap_smoke'
    ctx.work_dir.mkdir(parents=True, exist_ok=True)

    done = {r.get('checkpoint') for r in U.read_csv(out_csv)}
    if ckpt_name in done:
        return

    try:
        bargs, model, device, det_support, name2id, _, _ = mech.build_model(ctx)
    except Exception as exc:
        mech.append_csv(out_csv, dict(
            checkpoint=ckpt_name, ap50='NA', sv_ap50='NA', notes=f'build_model: {exc}'), [
            'checkpoint', 'ap50', 'sv_ap50', 'det_count_proxy', 'final_sv_proxy',
            'n_images', 'notes',
        ])
        return

    final_svs, dets = [], []
    for stem in sample:
        try:
            row = mech.eval_one(ctx, model, bargs, device, det_support, name2id,
                                stem, 0, 'baseline', support_intervention='original')
            if row.get('notes') in ('no_matching_batch',) or 'no image' in str(row.get('notes', '')):
                continue
            if row.get('final_sv_ratio') not in (None, 'NA', ''):
                final_svs.append(float(row['final_sv_ratio']))
            if row.get('det_count') not in (None, 'NA', ''):
                dets.append(float(row['det_count']))
        except Exception as exc:
            log.warning('ap_smoke tile %s: %s', stem, exc)

    n_ok = len(final_svs)
    fs_mean = sum(final_svs) / n_ok if n_ok else float('nan')
    det_mean = sum(dets) / len(dets) if dets else float('nan')
    mech.append_csv(out_csv, dict(
        checkpoint=ckpt_name,
        ap50='NA',
        sv_ap50='NA',
        det_count_proxy=f'{det_mean:.2f}',
        final_sv_proxy=f'{fs_mean:.4f}',
        n_images=n_ok,
        notes=f'detection_proxy_only; full COCO AP not run. {ap_note_extra}'.strip(),
    ), [
        'checkpoint', 'ap50', 'sv_ap50', 'det_count_proxy', 'final_sv_proxy',
        'n_images', 'notes',
    ])
    log.info('ap_smoke %s n=%d proxy_final_sv=%.4f', ckpt_name, n_ok, fs_mean)


def write_summaries(result_dir: Path, out_raw: Path):
    rows = U.read_csv(out_raw)
    summ = {}
    for r in rows:
        if r.get('group') not in ('high', 'low'):
            continue
        key = (r['checkpoint'], r.get('group', ''))
        try:
            summ.setdefault(key, []).append(float(r.get('final_sv_ratio', 0) or 0))
        except (TypeError, ValueError):
            pass
    sum_rows = []
    for (ck, grp), vals in summ.items():
        dense_vals = []
        for r in rows:
            if r['checkpoint'] == ck and r.get('group') == grp:
                try:
                    dense_vals.append(float(r.get('dense_top1_sv_ratio', 0) or 0))
                except (TypeError, ValueError):
                    pass
        sum_rows.append(dict(
            checkpoint=ck, group=grp, n=len(vals),
            mean_final_sv=sum(vals) / len(vals) if vals else 0,
            mean_dense_sv=sum(dense_vals) / len(dense_vals) if dense_vals else 0,
        ))
    sum_path = result_dir / 'ftable_eval_sv_dehub_lite_summary.csv'
    if sum_rows:
        with open(sum_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(sum_rows[0].keys()))
            w.writeheader()
            w.writerows(sum_rows)

    compare = []
    bl_h = [float(r['final_sv_ratio']) for r in rows
            if r['checkpoint'] == 'baseline_epoch24' and r.get('group') == 'high']
    bl_mean = sum(bl_h) / len(bl_h) if bl_h else 1.0
    for ck in sorted({r['checkpoint'] for r in rows}):
        if ck == 'baseline_epoch24':
            continue
        h = [float(r['final_sv_ratio']) for r in rows if r['checkpoint'] == ck and r.get('group') == 'high']
        l = [float(r['final_sv_ratio']) for r in rows if r['checkpoint'] == ck and r.get('group') == 'low']
        if h:
            compare.append(dict(
                checkpoint=ck,
                high_mean=sum(h) / len(h),
                low_mean=sum(l) / len(l) if l else 0,
                delta_high_vs_baseline=(sum(h) / len(h)) / bl_mean - 1 if bl_mean else 0,
            ))
    cmp_path = result_dir / 'ftable_eval_highrisk_lowrisk_compare.csv'
    if compare:
        with open(cmp_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(compare[0].keys()))
            w.writeheader()
            w.writerows(compare)


def collect_checkpoints(work_dir: Path) -> list[tuple[str, Path]]:
    checkpoints = [('baseline_epoch24', U.BASE_CKPT)]
    for p in U.list_checkpoints(work_dir):
        name = p.stem
        if 'iter' in name or name.isdigit():
            checkpoints.append((name, p))
    return checkpoints


def run_eval_pass(work_dir: Path, result_dir: Path, do_ap: bool, ap_n: int = 40):
    out_raw = result_dir / 'ftable_eval_sv_dehub_lite_raw.csv'
    ap_csv = result_dir / 'ftable_eval_ap_smoke.csv'
    for name, path in collect_checkpoints(work_dir):
        if not path.exists():
            log.warning('skip missing %s', path)
            continue
        log.info('=== checkpoint %s ===', name)
        eval_checkpoint(name, path, U.HIGHRISK_TILES, 'high', out_raw)
        eval_checkpoint(name, path, U.LOWRISK_TILES, 'low', out_raw)
        if do_ap:
            eval_ap_smoke(name, path, ap_csv, ap_n)
    write_summaries(result_dir, out_raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-dir', type=Path, default=REPO)
    ap.add_argument('--result-dir', type=Path, default=U.RESULT_MD)
    ap.add_argument('--work-dir', type=Path, default=U.WORK_ROOT / 'train_v1')
    ap.add_argument('--watch-checkpoints', action='store_true')
    ap.add_argument('--poll-sec', type=int, default=300)
    ap.add_argument('--eval-highrisk', action='store_true')
    ap.add_argument('--eval-lowrisk', action='store_true')
    ap.add_argument('--eval-ap-smoke', action='store_true')
    ap.add_argument('--ap-n-images', type=int, default=40,
                    help='Heldout stems for AP smoke proxy')
    ap.add_argument('--ap-only', action='store_true', help='Only run AP smoke pass')
    args = ap.parse_args()
    U.ensure_dirs()
    setup_logging(args.result_dir)

    do_tiles = args.eval_highrisk or args.eval_lowrisk or not (
        args.watch_checkpoints and not args.eval_ap_smoke)
    do_ap = args.eval_ap_smoke

    if args.ap_only:
        ap_path = args.result_dir / 'ftable_eval_ap_smoke.csv'
        ap_path.write_text(
            'checkpoint,ap50,sv_ap50,det_count_proxy,final_sv_proxy,n_images,notes\n',
            encoding='utf-8',
        )
        for name, path in collect_checkpoints(args.work_dir):
            if path.exists():
                eval_ap_smoke(name, path, ap_path, args.ap_n_images)
        (args.result_dir / 'fres_eval_sv_dehub_lite_v1.md').write_text(
            f'# AP smoke only\n- n_images={args.ap_n_images}\n', encoding='utf-8')
        return

    if args.watch_checkpoints:
        log.info('watch mode poll=%ds', args.poll_sec)
        seen = set()
        while True:
            for name, path in collect_checkpoints(args.work_dir):
                key = (name, path.stat().st_mtime)
                if key in seen:
                    continue
                seen.add(key)
                out_raw = args.result_dir / 'ftable_eval_sv_dehub_lite_raw.csv'
                if do_tiles or True:
                    eval_checkpoint(name, path, U.HIGHRISK_TILES, 'high', out_raw)
                    eval_checkpoint(name, path, U.LOWRISK_TILES, 'low', out_raw)
                if do_ap:
                    eval_ap_smoke(name, path, args.result_dir / 'ftable_eval_ap_smoke.csv', args.ap_n_images)
                write_summaries(args.result_dir, out_raw)
            ckpts = collect_checkpoints(args.work_dir)
            raw = U.read_csv(args.result_dir / 'ftable_eval_sv_dehub_lite_raw.csv')
            names = {n for n, _ in ckpts}
            covered = {r['checkpoint'] for r in raw}
            if names and names.issubset(covered):
                log.info('all checkpoints evaluated; exiting watch')
                break
            time.sleep(args.poll_sec)
    else:
        run_eval_pass(args.work_dir, args.result_dir, do_ap, args.ap_n_images)

    rows = U.read_csv(args.result_dir / 'ftable_eval_sv_dehub_lite_raw.csv')
    (args.result_dir / 'fres_eval_sv_dehub_lite_v1.md').write_text(
        f'# Eval SV-DeHub-Lite v1\n\n'
        f'- generated: {datetime.now().isoformat()}\n'
        f'- checkpoints: {len(collect_checkpoints(args.work_dir))}\n'
        f'- raw rows: {len(rows)}\n'
        f'- ap_smoke: {do_ap}\n',
        encoding='utf-8',
    )
    log.info('wrote eval artifacts under %s', args.result_dir)


if __name__ == '__main__':
    main()
