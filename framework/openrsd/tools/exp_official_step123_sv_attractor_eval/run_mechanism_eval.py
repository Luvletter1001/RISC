#!/usr/bin/env python
"""Task A/C: high-low tiles and P0148 12-angle mechanism eval."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.exp_mechanism_sv_attractor_gpu89 import common_attractor_utils as mech
from tools.exp_official_step123_sv_attractor_eval.common_official_utils import (
    MECH_ANGLES, P0148_ANGLES, P0148_TILE, RESULT_DIR, SCHEMA_MECH_RAW, STAGES,
    WORK_DIR, append_csv, ensure_dirs, eval_combos_for_stage, head_label,
    load_highrisk_tiles, load_lowrisk_tiles,
)
from tools.verify_sv_attractor_gpu89 import common as verify


def parse_histogram(hist_str) -> dict:
    if isinstance(hist_str, dict):
        return hist_str
    try:
        return json.loads(hist_str) if hist_str else {}
    except json.JSONDecodeError:
        return {}


def enrich_row(row: dict, stage: str, ckpt: Path, cfg: Path, support: str, aux: bool, group: str):
    hist = parse_histogram(row.get('top1_class_histogram', '{}'))
    det = int(float(row.get('det_count', 0) or 0))
    sv = int(hist.get('small-vehicle', row.get('small_vehicle_count', 0)) or 0)
    row.update(
        stage=stage,
        checkpoint=str(ckpt),
        config_path=str(cfg),
        support_mode=support,
        head_mode=head_label(aux),
        val_using_aux=aux,
        group=group,
        non_sv_det_count=max(0, det - sv),
        tennis_court_count=int(hist.get('tennis-court', 0)),
        large_vehicle_count=int(hist.get('large-vehicle', 0)),
        basketball_court_count=int(hist.get('basketball-court', 0)),
        harbor_count=int(hist.get('harbor', 0)),
        timestamp=datetime.now().isoformat(timespec='seconds'),
    )
    return row


def run_combo(ctx, tiles, angles, group, out_raw: Path, full_matrix: bool):
    spec = STAGES[ctx.stage_name]
    if not spec.primary_ckpt.exists():
        return
    for support, aux in eval_combos_for_stage(spec, full_matrix):
        if not spec.dense_hook_ok:
            continue
        os.environ['CUDA_VISIBLE_DEVICES'] = str(ctx.gpu)
        mctx = mech.MechContext(
            gpu=ctx.gpu,
            config=spec.config,
            checkpoint=spec.primary_ckpt,
            work_dir=WORK_DIR / ctx.stage_name / f'{support}_{head_label(aux)}',
            result_md_dir=RESULT_DIR,
        )
        mctx.work_dir.mkdir(parents=True, exist_ok=True)
        try:
            bargs, model, device, det_support, name2id, _, _ = mech.build_model(mctx, support_type=support)
        except Exception as exc:
            append_csv(out_raw, dict(stage=ctx.stage_name, notes=f'BUILD_FAIL:{exc}'), SCHEMA_MECH_RAW)
            continue
        if hasattr(model, 'val_using_aux'):
            model.val_using_aux = aux
        method = 'visual_support' if support == 'visual' else 'text_support'
        for tile in tiles:
            for angle in angles:
                try:
                    row = mech.eval_one(
                        mctx, model, bargs, device, det_support, name2id,
                        tile, angle, method, support_intervention='original')
                    row = enrich_row(row, ctx.stage_name, spec.primary_ckpt, spec.config,
                                     support, aux, group)
                    append_csv(out_raw, row, SCHEMA_MECH_RAW)
                except Exception as exc:
                    append_csv(out_raw, dict(
                        stage=ctx.stage_name, tile_id=tile, angle=angle, group=group,
                        notes=f'EVAL_ERROR:{exc}'), SCHEMA_MECH_RAW)


class Ctx:
    def __init__(self, stage_name: str, gpu: int):
        self.stage_name = stage_name
        self.gpu = gpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=list(STAGES.keys()), required=True)
    ap.add_argument('--task', choices=['highlow', 'p0148'], default='highlow')
    ap.add_argument('--gpu', type=int, default=8)
    ap.add_argument('--full-matrix', action='store_true')
    args = ap.parse_args()
    ensure_dirs()
    ctx = Ctx(args.stage, args.gpu)
    if args.task == 'highlow':
        out = RESULT_DIR / 'ftable_10_highlow_mechanism_raw.csv'
        run_combo(ctx, load_highrisk_tiles(), MECH_ANGLES, 'high', out, args.full_matrix)
        run_combo(ctx, load_lowrisk_tiles(), MECH_ANGLES, 'low', out, args.full_matrix)
    else:
        out = RESULT_DIR / 'ftable_30_p0148_12angle_raw.csv'
        run_combo(ctx, [P0148_TILE], P0148_ANGLES, 'p0148', out, args.full_matrix)
    print('done', args.stage, args.task)


if __name__ == '__main__':
    main()
