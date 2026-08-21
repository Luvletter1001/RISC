#!/usr/bin/env python
"""Step 6: tile-angle generalization (P5). GPU8 visual / GPU9 text."""
from __future__ import annotations

import json
import subprocess
import sys

from tools.verify_sv_attractor_gpu89 import common as C


def run(ctx: C.RunContext, support_type: str) -> dict:
    if ctx.blocked():
        return dict(status='BLOCKED')

    out = ctx.exp_dir('exp_06_tile_angle') / support_type
    out.mkdir(parents=True, exist_ok=True)
    max_tiles = 20 if ctx.mode == 'full' else 5
    tiles = [C.DEFAULT_TILE] + C.discover_cross_tiles(ctx.repo_root, max_tiles - 1)
    tiles = tiles[:max_tiles]

    rows = []
    for tile in tiles:
        image_dir = C.tile_image_dir(ctx.repo_root, tile)
        probe_out = out / tile
        if not (probe_out / 'detections_summary.csv').exists() or ctx.force:
            cmd = [
                sys.executable,
                str(ctx.repo_root / 'tools/rotation_diagnostics/probe_rotated_stage_outputs.py'),
                '--config', str(ctx.config),
                '--checkpoint', str(ctx.checkpoint),
                '--image-dir', str(image_dir),
                '--support-type', support_type,
                '--angles', *[str(a) for a in ctx.angles],
                '--out-dir', str(probe_out),
                '--result-md-dir', str(ctx.result_md_dir),
            ]
            log = ctx.log_dir() / f'tile_angle_{support_type}_{tile}.log'
            with open(log, 'w') as lf:
                proc = subprocess.run(cmd, cwd=ctx.repo_root, stdout=lf, stderr=subprocess.STDOUT)
            if proc.returncode != 0:
                rows.append(dict(tile_id=tile, support_type=support_type,
                                 status='FAILED', return_code=proc.returncode))
                continue
        det = C.read_csv(probe_out / 'detections_summary.csv')
        for r in det:
            hist = json.loads(r.get('class_histogram') or '{}')
            rows.append(dict(
                tile_id=tile, support_type=support_type, angle=int(float(r['angle'])),
                official_post_nms_sv_ratio=float(r.get('small_vehicle_ratio', 0)),
                detection_total=int(float(r.get('detection_total', 0))),
                mean_score=float(r.get('mean_score', 0)),
                top1_class=max(hist, key=hist.get) if hist else '',
                status='OK'))

    csv_path = ctx.exp_dir('exp_06_tile_angle') / f'ftable_tile_angle_{support_type}.csv'
    C.write_csv(csv_path, rows)

    p0148 = [r for r in rows if r.get('tile_id') == C.DEFAULT_TILE and r.get('status') == 'OK']
    cross = [r for r in rows if r.get('tile_id') != C.DEFAULT_TILE and r.get('status') == 'OK']
    p0148_mean = C.mean_key(p0148, 'official_post_nms_sv_ratio')
    cross_mean = C.mean_key(cross, 'official_post_nms_sv_ratio')
    return dict(
        support_type=support_type, rows=rows, p0148_mean=p0148_mean,
        cross_mean=cross_mean, csv=str(csv_path))


def run_and_write_md(ctx: C.RunContext, gpu: int) -> dict:
    fres = ctx.fres_path('fres_06_tile_angle_generalization.md')
    results = []
    if gpu == 8:
        results.append(run(ctx, 'visual'))
    if gpu == 9:
        results.append(run(ctx, 'text'))
    # merge prior CSV if resume on other GPU
    for st in ['visual', 'text']:
        if any(r.get('support_type') == st for r in results):
            continue
        prior = ctx.exp_dir('exp_06_tile_angle') / f'ftable_tile_angle_{st}.csv'
        if prior.exists():
            rows = C.read_csv(prior)
            if rows:
                results.append(dict(
                    support_type=st, rows=rows,
                    p0148_mean=C.mean_key(
                        [r for r in rows if r.get('tile_id') == C.DEFAULT_TILE],
                        'official_post_nms_sv_ratio'),
                    cross_mean=C.mean_key(
                        [r for r in rows if r.get('tile_id') != C.DEFAULT_TILE],
                        'official_post_nms_sv_ratio'),
                    csv=str(prior)))

    p0148_v = next((r['p0148_mean'] for r in results if r.get('support_type') == 'visual'), float('nan'))
    cross_v = next((r['cross_mean'] for r in results if r.get('support_type') == 'visual'), float('nan'))
    p5 = 'SUPPORTED' if p0148_v > cross_v + 0.1 else 'PARTIAL'

    with open(fres, 'w') as f:
        C.write_fres_header(f, 'Tile-Angle Generalization (P5)', ctx)
        f.write(f'## P5 Verdict: **{p5}**\n\n')
        f.write(f'- GPU role: `{gpu}` support runs\n')
        for r in results:
            f.write(f"- {r['support_type']}: P0148 mean `{r['p0148_mean']:.4f}`, "
                    f"cross-tile mean `{r['cross_mean']:.4f}`\n")
        f.write('\n| tile_group | support | mean_final_sv |\n|---|---|---:|\n')
        for r in results:
            st = r['support_type']
            f.write(f'| P0148 | {st} | {r["p0148_mean"]:.4f} |\n')
            f.write(f'| cross-tile | {st} | {r["cross_mean"]:.4f} |\n')
        f.write('\n## CSV\n\n')
        for r in results:
            f.write(f'- `{r["csv"]}`\n')

    ctx.mark_step('tile_angle', 'OK', p5=p5, gpu=gpu)
    return dict(status='OK', p5=p5, fres=str(fres))
