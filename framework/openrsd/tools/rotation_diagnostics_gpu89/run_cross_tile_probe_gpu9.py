#!/usr/bin/env python
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


CONFIG = 'M_configs/Step2_A10_Large_Pretrain_Stage3/A10_flex_rtm_v3_1_formal.py'
CHECKPOINT = 'results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--report-dir', required=True)
    parser.add_argument('--angles', nargs='+', type=int, required=True)
    parser.add_argument('--max-tiles', type=int, default=12)
    parser.add_argument('--support-types', nargs='+', default=['visual', 'text'])
    return parser.parse_args()


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def discover_tiles(project_root, max_tiles):
    vis_root = project_root / 'vis'
    candidates = []
    for dataset_dir in sorted(vis_root.glob('*/dataset')):
        image_dir = dataset_dir / 'images'
        ann_dir = dataset_dir / 'annfiles'
        if not image_dir.is_dir() or not ann_dir.is_dir():
            continue
        rot000 = list(image_dir.glob('*_rot000.jpg'))
        if not rot000:
            continue
        tile_id = dataset_dir.parent.name
        if tile_id.startswith('P0148__1024__651___0'):
            continue
        angles = {
            int(m.group(1))
            for p in image_dir.glob('*_rot*.jpg')
            for m in [re.search(r'_rot(\d{3})', p.name)]
            if m
        }
        candidates.append(dict(tile_id=tile_id, image_dir=image_dir,
                               angle_count=len(angles)))
    return candidates[:max_tiles]


def run_probe(project_root, image_dir, angles, out_dir, support_type, report_dir):
    cmd = [
        sys.executable,
        'tools/rotation_diagnostics/probe_rotated_stage_outputs.py',
        '--config', CONFIG,
        '--checkpoint', CHECKPOINT,
        '--image-dir', str(image_dir),
        '--support-type', support_type,
        '--angles', *[str(a) for a in angles],
        '--out-dir', str(out_dir),
        '--result-md-dir', str(report_dir),
    ]
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_dir.parent / f'{out_dir.name}.log'
    with open(log_path, 'w') as log:
        proc = subprocess.run(
            cmd, cwd=project_root, stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode, log_path


def summarize_tile(tile_id, image_dir, support_type, probe_dir):
    rows = []
    det_path = probe_dir / 'detections_summary.csv'
    if not det_path.exists():
        return rows
    for row in read_csv(det_path):
        hist = json.loads(row.get('class_histogram') or '{}')
        top3 = Counter(hist).most_common(3)
        top1 = top3[0][0] if top3 else ''
        rows.append(dict(
            tile_id=tile_id,
            source_path=str(image_dir),
            angle=int(float(row['angle'])),
            support_type=support_type,
            detection_total=int(float(row['detection_total'])),
            small_vehicle_count=int(float(row['small_vehicle_count'])),
            small_vehicle_ratio=float(row['small_vehicle_ratio']),
            top1_class=top1,
            top3_class_counts=json.dumps(top3, ensure_ascii=True),
            mean_score=float(row['mean_score']),
            max_score=float(row['max_score']),
        ))
    return rows


def classify_scene(rows):
    if not rows:
        return 'unknown'
    rot0 = [r for r in rows if r['angle'] == 0]
    base = rot0[0] if rot0 else rows[0]
    top = base['top1_class']
    ratio = base['small_vehicle_ratio']
    total = base['detection_total']
    if total < 20:
        return 'sparse_or_background'
    if ratio > 0.5:
        return 'vehicle_dense_or_small_vehicle_biased'
    if top in ['ship', 'harbor']:
        return 'harbor_ship'
    if top == 'plane':
        return 'airport_plane'
    if top in ['tennis-court', 'baseball-diamond', 'ground-track-field']:
        return 'sports_field'
    return 'mixed'


def write_report(report_dir, out_dir, rows, failures):
    report = report_dir / 'fres_cross_tile_generalization.md'
    by_tile = defaultdict(list)
    for row in rows:
        by_tile[(row['tile_id'], row['support_type'])].append(row)
    agg_rows = []
    for (tile_id, support_type), items in sorted(by_tile.items()):
        rot0 = next((r for r in items if r['angle'] == 0), items[0])
        ratios = [r['small_vehicle_ratio'] for r in items]
        agg_rows.append(dict(
            tile_id=tile_id,
            support_type=support_type,
            scene=classify_scene(items),
            rot000_small_vehicle_ratio=rot0['small_vehicle_ratio'],
            angle_mean_small_vehicle_ratio=float(np.mean(ratios)),
            max_small_vehicle_ratio=float(np.max(ratios)),
            high_at_rot000=rot0['small_vehicle_ratio'] >= 0.8,
            high_after_rotation=(
                rot0['small_vehicle_ratio'] < 0.8 and np.max(ratios) >= 0.8),
        ))
    agg_csv = out_dir / 'ftable_cross_tile_aggregate_summary.csv'
    write_csv(agg_csv, agg_rows, list(agg_rows[0].keys()) if agg_rows else [
        'tile_id', 'support_type', 'scene', 'rot000_small_vehicle_ratio',
        'angle_mean_small_vehicle_ratio', 'max_small_vehicle_ratio',
        'high_at_rot000', 'high_after_rotation'])
    visual = [r for r in agg_rows if r['support_type'] == 'visual']
    text = [r for r in agg_rows if r['support_type'] == 'text']
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    visual_mean = np.mean([r['angle_mean_small_vehicle_ratio']
                           for r in visual]) if visual else np.nan
    text_mean = np.mean([r['angle_mean_small_vehicle_ratio']
                         for r in text]) if text else np.nan
    visual_high0 = sum(1 for r in visual if r['high_at_rot000'])
    text_high0 = sum(1 for r in text if r['high_at_rot000'])
    with open(report, 'w') as f:
        f.write('# Cross Tile Generalization\n\n')
        f.write(f'- CUDA_VISIBLE_DEVICES: `{cuda_visible}`\n')
        f.write(f'- out_dir: `{out_dir}`\n')
        f.write(f'- aggregate table: `{agg_csv}`\n')
        f.write(f'- failed tile/support runs: `{len(failures)}`\n\n')
        f.write('## Findings\n\n')
        if visual:
            f.write(f'- visual support tiles: `{len(visual)}`; mean angle small_vehicle_ratio `{visual_mean:.6f}`\n')
            f.write(f'- visual high at rot000 count: `{visual_high0}`\n')
        if text:
            f.write(f'- text support tiles: `{len(text)}`; mean angle small_vehicle_ratio `{text_mean:.6f}`\n')
            f.write(f'- text high at rot000 count: `{text_high0}`\n')
        if failures:
            f.write('\n## Failures\n\n')
            for item in failures:
                f.write(f"- `{item}`\n")
    return report, agg_csv


def main():
    args = parse_args()
    project_root = Path(args.project_root)
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    tiles = discover_tiles(project_root, args.max_tiles)
    rows = []
    failures = []
    for tile in tiles:
        for support_type in args.support_types:
            probe_dir = out_dir / 'probes' / f'{tile["tile_id"]}_{support_type}'
            code, log = run_probe(project_root, tile['image_dir'], args.angles,
                                  probe_dir, support_type, report_dir)
            if code != 0:
                failures.append(f'{tile["tile_id"]}:{support_type}:exit={code}:log={log}')
                continue
            rows.extend(summarize_tile(tile['tile_id'], tile['image_dir'],
                                       support_type, probe_dir))
    fields = [
        'tile_id', 'source_path', 'angle', 'support_type', 'detection_total',
        'small_vehicle_count', 'small_vehicle_ratio', 'top1_class',
        'top3_class_counts', 'mean_score', 'max_score',
    ]
    all_csv = out_dir / 'ftable_cross_tile_all_results.csv'
    write_csv(all_csv, rows, fields)
    visual_csv = out_dir / 'ftable_cross_tile_visual_support_summary.csv'
    text_csv = out_dir / 'ftable_cross_tile_text_support_summary.csv'
    compare_csv = out_dir / 'ftable_cross_tile_visual_vs_text_compare.csv'
    write_csv(visual_csv, [r for r in rows if r['support_type'] == 'visual'], fields)
    write_csv(text_csv, [r for r in rows if r['support_type'] == 'text'], fields)
    compare_rows = []
    lookup = {(r['tile_id'], r['angle'], r['support_type']): r for r in rows}
    for key in sorted({(r['tile_id'], r['angle']) for r in rows}):
        v = lookup.get((key[0], key[1], 'visual'))
        t = lookup.get((key[0], key[1], 'text'))
        if not v or not t:
            continue
        compare_rows.append(dict(
            tile_id=key[0],
            angle=key[1],
            visual_small_vehicle_ratio=v['small_vehicle_ratio'],
            text_small_vehicle_ratio=t['small_vehicle_ratio'],
            delta_text_minus_visual=t['small_vehicle_ratio'] - v['small_vehicle_ratio'],
            visual_top1=v['top1_class'],
            text_top1=t['top1_class'],
        ))
    if compare_rows:
        write_csv(compare_csv, compare_rows, list(compare_rows[0].keys()))
    else:
        write_csv(compare_csv, [], [
            'tile_id', 'angle', 'visual_small_vehicle_ratio',
            'text_small_vehicle_ratio', 'delta_text_minus_visual',
            'visual_top1', 'text_top1'])
    report, agg_csv = write_report(report_dir, out_dir, rows, failures)
    summary = dict(
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        selected_tiles=[t['tile_id'] for t in tiles],
        out_dir=str(out_dir),
        all_csv=str(all_csv),
        visual_csv=str(visual_csv),
        text_csv=str(text_csv),
        compare_csv=str(compare_csv),
        aggregate_csv=str(agg_csv),
        report=str(report),
        failures=failures)
    with open(out_dir / 'cross_tile_gpu9_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    if failures and not rows:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
