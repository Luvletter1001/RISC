#!/usr/bin/env python
import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.rotation_overnight_gpu89 import common as C


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--project-root', default='/data1/zcy/OpenRSD')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--report-dir', required=True)
    p.add_argument('--angles', nargs='+', type=int, default=[0, 45, 90, 180, 270])
    p.add_argument('--max-tiles', type=int, default=30)
    p.add_argument('--support-types', nargs='+', default=['visual', 'text'])
    return p.parse_args()


def classify_scene(row):
    top = row.get('top1_class', '')
    ratio = float(row.get('small_vehicle_ratio', 0))
    total = int(float(row.get('detection_total', 0)))
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


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        'tools/rotation_diagnostics_gpu89/run_cross_tile_probe_gpu9.py',
        '--project-root', args.project_root,
        '--out-dir', str(out_dir / 'raw_probe'),
        '--report-dir', str(report_dir),
        '--angles', *[str(x) for x in args.angles],
        '--max-tiles', str(args.max_tiles),
        '--support-types', *args.support_types,
    ]
    log = out_dir / 'cross_tile_raw_probe.log'
    with open(log, 'w') as f:
        code = subprocess.run(cmd, cwd=args.project_root, stdout=f, stderr=subprocess.STDOUT).returncode
    raw = out_dir / 'raw_probe'
    all_rows = C.read_csv(raw / 'ftable_cross_tile_all_results.csv')
    summary_csv = out_dir / 'ftable_cross_tile_30_summary.csv'
    C.write_csv(summary_csv, all_rows, list(all_rows[0].keys()) if all_rows else [
        'tile_id', 'source_path', 'angle', 'support_type', 'detection_total',
        'small_vehicle_count', 'small_vehicle_ratio', 'top1_class',
        'top3_class_counts', 'mean_score', 'max_score'])
    scene_rows = []
    by_scene = defaultdict(list)
    for row in all_rows:
        if int(float(row['angle'])) == 0:
            scene = classify_scene(row)
            by_scene[(scene, row['support_type'])].append(float(row['small_vehicle_ratio']))
    for (scene, support_type), vals in sorted(by_scene.items()):
        scene_rows.append(dict(scene=scene, support_type=support_type,
                               tile_count=len(vals),
                               rot000_small_vehicle_ratio_mean=float(np.mean(vals)),
                               rot000_small_vehicle_ratio_max=float(np.max(vals))))
    scene_csv = out_dir / 'ftable_cross_tile_scene_group_summary.csv'
    C.write_csv(scene_csv, scene_rows, list(scene_rows[0].keys()) if scene_rows else [
        'scene', 'support_type', 'tile_count', 'rot000_small_vehicle_ratio_mean',
        'rot000_small_vehicle_ratio_max'])
    compare_rows = C.read_csv(raw / 'ftable_cross_tile_visual_vs_text_compare.csv')
    delta_csv = out_dir / 'ftable_cross_tile_visual_text_delta.csv'
    C.write_csv(delta_csv, compare_rows, list(compare_rows[0].keys()) if compare_rows else [
        'tile_id', 'angle', 'visual_small_vehicle_ratio',
        'text_small_vehicle_ratio', 'delta_text_minus_visual',
        'visual_top1', 'text_top1'])
    visual = [float(r['small_vehicle_ratio']) for r in all_rows if r['support_type'] == 'visual']
    text = [float(r['small_vehicle_ratio']) for r in all_rows if r['support_type'] == 'text']
    sparse = [r for r in scene_rows if r['scene'] == 'sparse_or_background']
    report = report_dir / 'fres_cross_tile_extended.md'
    with open(report, 'w') as f:
        f.write('# Cross-Tile Extended Overnight Probe\n\n')
        f.write(f'- CUDA_VISIBLE_DEVICES: `{C.cuda_visible()}`\n')
        f.write(f'- raw probe exit code: `{code}`\n')
        f.write(f'- out_dir: `{out_dir}`\n')
        f.write(f'- raw log: `{log}`\n\n')
        f.write('## Tables\n\n')
        f.write(f'- summary: `{summary_csv}`\n')
        f.write(f'- scene groups: `{scene_csv}`\n')
        f.write(f'- visual/text delta: `{delta_csv}`\n\n')
        f.write('## Key Numbers\n\n')
        f.write(f'- visual mean small_vehicle_ratio: `{np.mean(visual) if visual else np.nan:.6f}`\n')
        f.write(f'- text mean small_vehicle_ratio: `{np.mean(text) if text else np.nan:.6f}`\n')
        if compare_rows:
            deltas = [float(r['delta_text_minus_visual']) for r in compare_rows]
            f.write(f'- mean text-minus-visual delta: `{np.mean(deltas):.6f}`\n')
        f.write(f'- sparse/background scene groups: `{len(sparse)}`\n')
    with open(out_dir / 'gpu9_cross_tile_extended_summary.json', 'w') as f:
        json.dump(dict(report=str(report), raw_exit_code=code,
                       summary_csv=str(summary_csv), scene_csv=str(scene_csv),
                       delta_csv=str(delta_csv), raw_log=str(log)), f, indent=2)
    print(report)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
