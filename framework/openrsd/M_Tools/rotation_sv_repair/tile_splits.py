#!/usr/bin/env python3
"""Explicit tile splits: diagnostic / calibration / train / val / final_test."""
from __future__ import annotations

import random
from pathlib import Path
from typing import List

from M_Tools.rotation_sv_repair.common import (
    HIGHRISK_TILES, LOWRISK_TILES, P0148, REPO_ROOT, SuiteContext,
    list_angle_sweep_stems, list_ss_train_stems, tile_has_class, write_csv, write_fres_header,
)

COURT_TAGS = (
    ('tennis-court', 'contains_tennis'),
    ('baseball-diamond', 'contains_baseball'),
    ('soccer-ball-field', 'contains_soccer'),
    ('small-vehicle', 'contains_true_sv'),
)


def _flags(tile_id: str) -> dict:
    return dict(
        contains_tennis='yes' if tile_has_class(tile_id, ['tennis-court', 'tennis court']) else 'no',
        contains_baseball='yes' if tile_has_class(tile_id, ['baseball-diamond']) else 'no',
        contains_soccer='yes' if tile_has_class(tile_id, ['soccer-ball-field', 'soccer']) else 'no',
        contains_true_sv='yes' if tile_has_class(tile_id, ['small-vehicle', 'small vehicle']) else 'no',
        is_hub='yes' if tile_id in HIGHRISK_TILES else 'no',
        is_low_risk='yes' if tile_id in LOWRISK_TILES else 'no',
        is_stress_case='yes' if tile_id == P0148 or tile_id in HIGHRISK_TILES else 'no',
    )


def build_tile_splits(ctx: SuiteContext) -> List[dict]:
    sweep_cap = 200 if ctx.mode in ('dryrun', 'smoke', 'debug') else 2000
    all_stems = list(dict.fromkeys(
        list_ss_train_stems(500) + list_angle_sweep_stems(0, sweep_cap)
        + HIGHRISK_TILES + LOWRISK_TILES + [P0148]))
    pool = [s for s in all_stems if s != P0148 and not s.startswith('P0148__')]
    rng = random.Random(20260524)
    rng.shuffle(pool)
    n = len(pool)
    n_final = max(20, int(n * 0.15))
    n_val = max(15, int(n * 0.10))
    n_cal = max(15, int(n * 0.10))
    n_adapter = max(25, int(n * 0.20))

    final_test = sorted(pool[:n_final])
    val = sorted(pool[n_final:n_final + n_val])
    calib = sorted(pool[n_final + n_val:n_final + n_val + n_cal])
    adapter_train = sorted(pool[n_final + n_val + n_cal:n_final + n_val + n_cal + n_adapter])
    rest = sorted(pool[n_final + n_val + n_cal + n_adapter:])

    rows: List[dict] = []
    for tid in HIGHRISK_TILES + [P0148]:
        fl = _flags(tid)
        rows.append(dict(tile_id=tid, split='diagnostic_stress', reason='known_hub_or_P0148', **fl))
    for tid in final_test:
        fl = _flags(tid)
        rows.append(dict(tile_id=tid, split='final_test', reason='heldout_unseen', **fl))
    for tid in val:
        fl = _flags(tid)
        rows.append(dict(tile_id=tid, split='validation', reason='early_stop_model_select', **fl))
    for tid in calib:
        fl = _flags(tid)
        rows.append(dict(tile_id=tid, split='calibration', reason='threshold_bias_prompt_search', **fl))
    for tid in adapter_train:
        fl = _flags(tid)
        rows.append(dict(tile_id=tid, split='adapter_train', reason='adapter_distill_train', **fl))
    for tid in rest[: max(0, 50 - len(rows))]:
        if any(r['tile_id'] == tid for r in rows):
            continue
        fl = _flags(tid)
        rows.append(dict(tile_id=tid, split='adapter_train', reason='overflow_train_pool', **fl))

    if ctx.only_tile:
        only = [t.strip() for t in ctx.only_tile.split(',') if t.strip()]
        rows = [r for r in rows if r['tile_id'] in only] or [
            dict(tile_id=t, split='diagnostic_stress', reason='only_tile_override', **_flags(t))
            for t in only
        ]
    if ctx.max_images > 0:
        rows = rows[:ctx.max_images]
    return rows


def run_splits(ctx: SuiteContext) -> dict:
    fres = 'fres_001_tile_splits.md'
    csv_path = ctx.tables_dir / 'ftable_001_tile_splits.csv'
    if ctx.should_skip_output(csv_path) and ctx.should_skip_output(ctx.fres_path(fres)):
        return dict(status='DONE', skipped=True)
    rows = build_tile_splits(ctx)
    fields = ['tile_id', 'split', 'reason', 'contains_tennis', 'contains_baseball',
              'contains_soccer', 'contains_true_sv', 'is_hub', 'is_low_risk', 'is_stress_case']
    write_csv(csv_path, rows, fields)
    ctx.progress['tile_splits'] = rows
    ctx.save_progress()

    counts = {}
    for r in rows:
        counts[r['split']] = counts.get(r['split'], 0) + 1
    low_n = counts.get('final_test', 0) < 20
    status = 'PARTIAL' if low_n else 'DONE'
    with open(ctx.fres_path(fres), 'w', encoding='utf-8') as f:
        write_fres_header(f, 'Tile Splits', ctx, status)
        if low_n:
            f.write('\n> **LOW_SAMPLE_WARNING:** final_test tiles < 20; interpret heldout metrics cautiously.\n\n')
        f.write(f'- P0148 in final_test: **no** (excluded from heldout primary metrics)\n')
        f.write(f'- csv: `{csv_path}`\n\n')
        f.write('| split | count |\n|:---|---:|\n')
        for k, v in sorted(counts.items()):
            f.write(f'| {k} | {v} |\n')
        f.write('\n## Rules\n\n')
        f.write('- diagnostic_stress: P0148 + hub tiles — diagnosis/visualization only.\n')
        f.write('- calibration: threshold / logit bias / prompt search.\n')
        f.write('- adapter_train: lightweight adapter + distillation.\n')
        f.write('- validation: early stopping.\n')
        f.write('- final_test: primary verdict split.\n')
    ctx.mark_step('splits', status)
    return dict(status=status, csv=str(csv_path), counts=counts, low_sample_warning=low_n)
