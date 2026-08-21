#!/usr/bin/env python
"""GPU9: tile × angle × method attractor atlas."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.exp_mechanism_sv_attractor_gpu89 import common_attractor_utils as U

METHODS = [
    'baseline', 'best_R1', 'best_C5', 'zero_sv_embedding',
    'text_support', 'visual_support',
]
QUICK_METHODS = ['baseline']
ATLAS_CSV = U.RESULT_MD / 'ftable_01_attractor_atlas_raw.csv'
PROBE_CSV = U.RESULT_MD / 'ftable_01_quick_probe_0deg.csv'
TILE_LIST_JSON = U.RESULT_MD / 'fmeta_01_tile_list.json'


def run_quick_probe(ctx: U.MechContext, model, bargs, device, det_support, name2id, tiles):
    U.setup_logging(ctx.work_dir, 'atlas_quick_probe')
    for tid in tiles:
        if U.already_done(PROBE_CSV, tid, 0, 'baseline'):
            continue
        row = U.eval_one(ctx, model, bargs, device, det_support, name2id, tid, 0, 'baseline')
        row['run_id'] = U.row_key(tid, 0, 'baseline')
        U.append_csv(PROBE_CSV, row, U.SCHEMA_ATLAS)


def _f(v, default=float('nan')):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def summarize_atlas(raw_csv: Path, out_md: Path, tile_json: Path):
    rows = U.read_csv(raw_csv)
    rows = [r for r in rows if r.get('method') == 'baseline' and _f(r.get('final_sv_ratio')) == _f(r.get('final_sv_ratio'))]
    if not rows:
        out_md.write_text('# Attractor Atlas\n\nNo data.\n')
        return

    from collections import defaultdict
    import statistics

    p0148_vals = [_f(r['final_sv_ratio']) for r in rows if r['tile_id'] == U.DEFAULT_TILE]
    p0148_mean = statistics.mean(p0148_vals) if p0148_vals else float('nan')

    tile_means = defaultdict(list)
    ang_means = defaultdict(list)
    for r in rows:
        tile_means[r['tile_id']].append(_f(r['final_sv_ratio']))
        ang_means[str(r['angle'])].append(_f(r['final_sv_ratio']))
    tile_avg = {t: statistics.mean(v) for t, v in tile_means.items()}
    sorted_tiles = sorted(tile_avg.items(), key=lambda x: -x[1])
    rank_idx = [t for t, _ in sorted_tiles].index(U.DEFAULT_TILE) if U.DEFAULT_TILE in tile_avg else -1
    p0148_pct = 1.0 - rank_idx / max(len(sorted_tiles) - 1, 1) if rank_idx >= 0 else float('nan')

    tile_var = statistics.pvariance(list(tile_avg.values())) if len(tile_avg) > 1 else 0.0
    ang_avg = {a: statistics.mean(v) for a, v in ang_means.items()}
    ang_var = statistics.pvariance(list(ang_avg.values())) if len(ang_avg) > 1 else 0.0

    lines = [
        '# Attractor Atlas (Exp1)',
        '',
        f'- rows: {len(U.read_csv(raw_csv))}',
        f'- P0148 baseline mean final_sv: **{p0148_mean:.3f}**',
        f'- P0148 percentile among tiles (high=1): **{p0148_pct:.2f}**',
        f'- tile mean variance: {tile_var:.4f}, angle mean variance: {ang_var:.4f}',
        '',
        '## 1. P0148 是否极端点',
        '',
    ]
    if p0148_pct >= 0.9:
        lines.append('- P0148 处于 top 10% high-risk，是 **high-risk 群体极端点**。')
    elif p0148_pct >= 0.7:
        lines.append('- P0148 偏高但 **仍有同量级 high-risk tiles**。')
    else:
        lines.append('- P0148 **不是**唯一极端点；tile 异质性主导。')

    lines += ['', '## 2. tile vs angle effect', '']
    if tile_var > ang_var * 1.5:
        lines.append('- **tile effect 更强**（背景/场景假说 H2 一致）。')
    elif ang_var > tile_var * 1.5:
        lines.append('- **angle effect 更强**（旋转 equivariance H3 一致）。')
    else:
        lines.append('- tile 与 angle 方差同量级，**交互**可能重要。')

    all_rows = U.read_csv(raw_csv)
    r1_rows = [r for r in all_rows if r.get('method') == 'best_R1']
    if r1_rows:
        deltas = []
        for r in r1_rows:
            b = next((x for x in rows if x['tile_id'] == r['tile_id'] and str(x['angle']) == str(r['angle'])), None)
            if b:
                deltas.append(_f(r['final_sv_ratio']) - _f(b['final_sv_ratio']))
        if deltas:
            lines += [
                '',
                '## 3. R1 是否普遍降 SV',
                '',
                f"- mean delta final_sv (R1-baseline): **{statistics.mean(deltas):.3f}**",
                f"- pairs with delta<-0.1: **{sum(1 for d in deltas if d < -0.1)}** / {len(deltas)}",
            ]

    def corr(xs, ys):
        pairs = [(a, b) for a, b in zip(xs, ys) if a == a and b == b]
        if len(pairs) < 3:
            return float('nan')
        mx = statistics.mean([p[0] for p in pairs])
        my = statistics.mean([p[1] for p in pairs])
        num = sum((a - mx) * (b - my) for a, b in pairs)
        den = (sum((a - mx) ** 2 for a, _ in pairs) * sum((b - my) ** 2 for _, b in pairs)) ** 0.5
        return num / den if den else float('nan')

    dsv = [_f(r['dense_top1_sv_ratio']) for r in rows]
    fsv = [_f(r['final_sv_ratio']) for r in rows]
    om = [_f(r['objectness_mean']) for r in rows]
    mar = [_f(r['dense_sv_margin_vs_runnerup']) for r in rows]
    lines += [
        '',
        '## 4. dense vs final',
        '',
        f"- corr(dense_top1_sv, final_sv): **{corr(dsv, fsv):.3f}**",
        f"- corr(objectness, final_sv): **{corr(om, fsv):.3f}**",
        f"- corr(sv_margin, final_sv): **{corr(mar, fsv):.3f}**",
    ]

    summ_path = U.RESULT_MD / 'ftable_01_attractor_atlas_tile_summary.csv'
    hi_path = U.RESULT_MD / 'ftable_01_attractor_highrisk_tiles.csv'
    with open(summ_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['tile_id', 'baseline_final_sv', 'baseline_dense_sv', 'n_angles'])
        w.writeheader()
        for tid, vals in tile_means.items():
            dense = [_f(r['dense_top1_sv_ratio']) for r in rows if r['tile_id'] == tid]
            w.writerow(dict(
                tile_id=tid,
                baseline_final_sv=statistics.mean(vals),
                baseline_dense_sv=statistics.mean(dense) if dense else float('nan'),
                n_angles=len(vals),
            ))
    hi_sorted = sorted(tile_avg.items(), key=lambda x: -x[1])[:10]
    with open(hi_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['tile_id', 'baseline_final_sv'])
        w.writeheader()
        for tid, v in hi_sorted:
            w.writerow(dict(tile_id=tid, baseline_final_sv=v))

    lines += ['', '## High-risk tiles (top 10)', '', '| tile_id | final_sv |', '|---|---:|']
    for tid, v in hi_sorted:
        lines.append(f"| {tid} | {v:.3f} |")

    out_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def extra_heldout_tiles(repo: Path, selected: set, n: int) -> list:
    import random
    asv0 = repo / 'data/DOTA1_1024_500/angle_sweep_val/realistic/angle_000/images'
    if not asv0.is_dir() or n <= 0:
        return []
    pool = sorted({p.stem for p in asv0.glob('*.png')} - selected)
    rng = random.Random(U.MechContext.seed + 7)
    rng.shuffle(pool)
    out = []
    for stem in pool:
        if all((repo / f'data/DOTA1_1024_500/angle_sweep_val/realistic/angle_{a:03d}/images/{stem}.png').exists()
               for a in U.ATLAS_ANGLES):
            out.append(stem)
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=9)
    ap.add_argument('--skip-quick-probe', action='store_true')
    ap.add_argument('--extend-angles', action='store_true',
                    help='12 angles for high-risk tiles')
    ap.add_argument('--max-tiles', type=int, default=0)
    ap.add_argument('--extra-heldout', type=int, default=0,
                    help='Append N angle_sweep heldout tiles not in vis/cross')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    ctx = U.MechContext(gpu=args.gpu)
    ctx.work_dir = U.WORK_ROOT / 'gpu9_atlas'
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    U.setup_logging(ctx.work_dir, 'atlas')
    ctx.load_progress()

    cross = U.discover_cross_tiles(ctx.repo_root)
    bargs, model, device, det_support, name2id, _, _ = U.build_model(ctx)
    ctx.progress['_model_cache'] = True

    held_cand = []
    asv0 = ctx.repo_root / 'data/DOTA1_1024_500/angle_sweep_val/realistic/angle_000/images'
    if asv0.is_dir():
        asv_stems = {p.stem for p in asv0.glob('*.png')}
        held_cand = [h for h in U.heldout_candidates(ctx.repo_root, 200) if h in asv_stems][:80]
    if not args.skip_quick_probe:
        for tid in list(cross) + [U.DEFAULT_TILE] + held_cand:
            if U.already_done(PROBE_CSV, tid, 0, 'baseline'):
                continue
            try:
                row = U.eval_one(ctx, model, bargs, device, det_support, name2id,
                                 tid, 0, 'baseline')
                U.append_csv(PROBE_CSV, row, U.SCHEMA_ATLAS)
            except Exception as exc:
                U.log.exception('quick probe %s: %s', tid, exc)

    strat = U.stratify_tiles(U.read_csv(PROBE_CSV), 10)
    tiles = [U.DEFAULT_TILE] + cross
    for grp in ('high', 'medium', 'low'):
        for t in strat.get(grp, []):
            if t not in tiles:
                tiles.append(t)
    for t in held_cand:
        if len(tiles) >= 50 and args.max_tiles <= 0:
            break
        if t not in tiles:
            tiles.append(t)
    if args.extra_heldout > 0:
        extra = extra_heldout_tiles(ctx.repo_root, set(tiles), args.extra_heldout)
        for t in extra:
            if t not in tiles:
                tiles.append(t)
        U.log.info('extra heldout tiles: %s', extra)
    if args.max_tiles > 0:
        tiles = tiles[:args.max_tiles]
    TILE_LIST_JSON.write_text(json.dumps(
        dict(cross=cross, stratify=strat, selected=tiles, extra_heldout=args.extra_heldout),
        indent=2))

    high_risk = [t for t, _ in sorted(
        [(t, float(np.mean([float(r['final_sv_ratio']) for r in U.read_csv(PROBE_CSV)
                            if r['tile_id'] == t]))) for t in tiles if t != U.DEFAULT_TILE],
        key=lambda x: -x[1])[:10]]

    for tid in tiles:
        angles = U.ATLAS_ANGLES_EXT if args.extend_angles and tid in high_risk else U.ATLAS_ANGLES
        for angle in angles:
            for method in METHODS:
                if U.already_done(ATLAS_CSV, tid, angle, method):
                    continue
                st = 'zero_sv' if method == 'zero_sv_embedding' else 'original'
                try:
                    row = U.eval_one(
                        ctx, model, bargs, device, det_support, name2id,
                        tid, angle, method, support_intervention=st)
                    row['run_id'] = U.row_key(tid, angle, method)
                    U.append_csv(ATLAS_CSV, row, U.SCHEMA_ATLAS)
                except Exception as exc:
                    U.log.exception('atlas %s %s %s: %s', tid, angle, method, exc)
                    U.append_csv(ATLAS_CSV, dict(
                        tile_id=tid, angle=angle, method=method, notes=str(exc)),
                        U.SCHEMA_ATLAS)

    try:
        summarize_atlas(ATLAS_CSV, U.RESULT_MD / 'fres_01_attractor_atlas.md', TILE_LIST_JSON)
    except ImportError:
        U.log.warning('pandas missing; skip atlas md summary')
    ctx.progress['atlas_done'] = True
    ctx.save_progress()
    print('atlas complete:', ATLAS_CSV)


if __name__ == '__main__':
    main()
