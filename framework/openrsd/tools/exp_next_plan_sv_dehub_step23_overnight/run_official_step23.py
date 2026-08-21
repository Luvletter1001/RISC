#!/usr/bin/env python
"""Step2/3 alignment vs fusion + heldout-500 (overnight result dir)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.exp_mechanism_sv_attractor_gpu89 import common_attractor_utils as mech
from tools.exp_next_plan_sv_dehub_step23_overnight import common_overnight_utils as U
from tools.exp_next_plan_sv_dehub_step23_overnight.eval_flex_heads import eval_one_flex_head, smoke_head_switch
from tools.exp_official_step123_sv_attractor_eval.common_official_utils import (
    STAGES, head_label, load_highrisk_tiles, load_lowrisk_tiles, MECH_ANGLES,
)
from tools.sv_attractor_repair_gpu89.eval_ap import infer_heldout_method, prepare_heldout_subset
from tools.sv_attractor_repair_gpu89 import common as RC


SCHEMA_HIGHLOW = [
    'stage', 'support_mode', 'head_mode', 'val_using_aux', 'tile_id', 'angle', 'group',
    'det_count', 'final_sv_ratio', 'dense_top1_sv_ratio', 'notes', 'timestamp',
]

SCHEMA_AP = [
    'stage', 'head_mode', 'val_using_aux', 'heldout_n', 'ap50', 'sv_ap50',
    'detection_total', 'final_sv_ratio', 'notes', 'timestamp',
]


def run_highlow(stage: str, gpu: int):
    spec = STAGES[stage]
    out = U.RESULT_DIR / 'ftable_31_step23_head_support_highlow_raw.csv'
    ctx = mech.MechContext(
        config=spec.config, checkpoint=spec.primary_ckpt,
        work_dir=U.WORK_DIR / 'official' / stage, result_md_dir=U.RESULT_DIR, gpu=gpu,
    )
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    for support in ('visual', 'text'):
        bargs, model, device, det_support, name2id, id2name, _ = mech.build_model(ctx, support)
        for aux in (False, True):
            for grp, tiles in [('high', load_highrisk_tiles()), ('low', load_lowrisk_tiles())]:
                for tile in tiles:
                    for angle in MECH_ANGLES:
                        try:
                            r = eval_one_flex_head(
                                ctx, model, bargs, device, det_support, name2id, id2name,
                                tile, angle, support, val_using_aux=aux)
                            r.update(stage=stage, group=grp,
                                     timestamp=datetime.now().isoformat(timespec='seconds'))
                            U.append_csv(out, r, SCHEMA_HIGHLOW)
                        except Exception as exc:
                            U.append_csv(out, dict(
                                stage=stage, tile_id=tile, angle=angle, group=grp,
                                notes=str(exc)), SCHEMA_HIGHLOW)


def run_heldout500(stage: str, gpu: int):
    spec = STAGES[stage]
    held = U.load_heldout_stems(500)
    out = U.RESULT_DIR / 'ftable_33_step23_heldout500_ap_summary.csv'
    for aux in (False, True):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        ctx = RC.RepairContext(
            repo_root=REPO, config=spec.config, checkpoint=spec.primary_ckpt,
            support_pkl=U.SUPPORT_PKL, verify_work_dir=U.REPAIR_WD,
            work_dir=U.WORK_DIR / 'official_ap' / f'{stage}_{head_label(aux)}',
            result_md_dir=U.RESULT_DIR, gpu=gpu, mode='overnight_official',
            angles=[0, 90], tiles=[], train_max_images=0,
            heldout_max_images=len(held), eval_max_images=len(held),
            iters=0, batch_size=1, lr=0.0,
        )
        try:
            bargs, model, device, det_support, name2id, _, _, _ = RC.build_model_ctx(ctx)
            model.val_using_aux = aux
            subset = prepare_heldout_subset(ctx, len(held))
            # TODO: wire infer_heldout_flex; fallback alignment path via bbox_head
            res = infer_heldout_method(
                ctx, model, device, det_support, name2id, subset,
                'baseline', [0, 90], len(held), score_thr=RC.POSTPROCESS['score_thr'])
            U.append_csv(out, dict(
                stage=stage, head_mode=head_label(aux), val_using_aux=aux,
                heldout_n=len(held), ap50=res.get('ap50_overall'), sv_ap50=res.get('sv_ap50'),
                detection_total=res.get('detection_total'),
                final_sv_ratio=res.get('final_sv_ratio'),
                notes=res.get('error', 'bbox_head_path_not_fusion_verified' if aux else ''),
                timestamp=datetime.now().isoformat(timespec='seconds'),
            ), SCHEMA_AP)
        except Exception as exc:
            U.append_csv(out, dict(
                stage=stage, head_mode=head_label(aux), notes=str(exc)), SCHEMA_AP)


def audit_fusion(gpu: int):
    spec = STAGES['Step2']
    ctx = mech.MechContext(
        config=spec.config, checkpoint=spec.primary_ckpt,
        work_dir=U.WORK_DIR / 'fusion_audit', result_md_dir=U.RESULT_DIR, gpu=gpu,
    )
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    rows = smoke_head_switch(ctx)
    out = U.RESULT_DIR / 'ftable_30_head_switch_smoke.csv'
    fusion_ok = False
    for r in rows:
        r['stage'] = 'Step2'
        U.append_csv(out, r, U.SCHEMA_HEAD_SMOKE)
    if len(rows) == 2:
        a, b = rows[0], rows[1]
        fusion_ok = (
            abs(float(a.get('final_sv_ratio', 0)) - float(b.get('final_sv_ratio', 0))) > 1e-6
            or int(a.get('det_count', 0)) != int(b.get('det_count', 0))
        )
    md = [
        '# Fusion head fix audit',
        '',
        f'- time: {datetime.now().isoformat()}',
        f'- fusion_verified: **{fusion_ok}**',
        '',
        '## Smoke rows',
        '```json',
        json.dumps(rows, indent=2, ensure_ascii=False),
        '```',
        '',
        'Eval path: `prompt_predict` with `val_using_aux` True/False.',
        'Heldout AP for fusion still uses bbox_head until `infer_heldout_flex` wired.',
    ]
    (U.RESULT_DIR / 'fres_30_fusion_head_fix_audit.md').write_text('\n'.join(md), encoding='utf-8')
    return fusion_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['audit', 'highlow', 'ap500', 'all'], default='all')
    ap.add_argument('--gpu', type=int, default=7)
    args = ap.parse_args()
    U.ensure_dirs()
    if args.task in ('audit', 'all'):
        audit_fusion(args.gpu)
    if args.task in ('highlow', 'all'):
        for st in ('Step2', 'Step3'):
            run_highlow(st, args.gpu)
    if args.task in ('ap500', 'all'):
        for st in ('Step2', 'Step3'):
            run_heldout500(st, args.gpu)
    print('official step23 done', args.task)


if __name__ == '__main__':
    sys.exit(main() or 0)
