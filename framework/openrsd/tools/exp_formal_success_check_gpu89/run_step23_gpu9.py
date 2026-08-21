#!/usr/bin/env python
"""Task B+C: fusion smoke + Step2/3 heldout-500 AP on GPU9."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.exp_formal_success_check_gpu89 import common_formal_utils as U
from tools.exp_formal_success_check_gpu89.infer_heldout_flex import infer_heldout_flex_ap
from tools.exp_mechanism_sv_attractor_gpu89 import common_attractor_utils as mech
from tools.exp_next_plan_sv_dehub_step23_overnight.eval_flex_heads import smoke_head_switch
from tools.sv_attractor_repair_gpu89 import common as RC
from tools.sv_attractor_repair_gpu89.eval_ap import prepare_heldout_subset


def checksum_scores(row: dict) -> str:
    s = f"{row.get('det_count')}_{row.get('final_sv_ratio')}_{row.get('mean_score')}"
    return hashlib.md5(s.encode()).hexdigest()[:12]


def run_fusion_smoke(gpu: int) -> str:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    ctx = mech.MechContext(config=U.STEP2_CONFIG, checkpoint=U.STEP2_CKPT,
                           work_dir=U.WORK_DIR / 'fusion_smoke', result_md_dir=U.RESULT_DIR, gpu=gpu)
    rows = smoke_head_switch(ctx)
    out = U.RESULT_DIR / 'ftable_30_fusion_head_switch_smoke.csv'
    if out.exists():
        out.unlink()
    fusion_ok = False
    for r in rows:
        r['stage'] = 'Step2'
        r['actually_used_head'] = r.get('logit_source', 'NA')
        r['score_checksum'] = checksum_scores(r)
        preds = []
        hist = U.parse_hist(r.get('top1_class_histogram', '{}'))
        for cls, cnt in sorted(hist.items(), key=lambda x: -x[1])[:10]:
            preds.append(f'{cls}:{cnt}')
        r['first10_det_classes'] = ';'.join(preds)
        U.append_csv(out, r, [
            'stage', 'tile_id', 'angle', 'head_mode', 'val_using_aux', 'det_count',
            'final_sv_ratio', 'mean_score', 'actually_used_head', 'logit_source',
            'score_checksum', 'first10_det_classes', 'notes'])
    if len(rows) == 2:
        a, b = rows[0], rows[1]
        fusion_ok = (
            abs(float(a.get('final_sv_ratio', 0)) - float(b.get('final_sv_ratio', 0))) > 1e-6
            or int(a.get('det_count', 0)) != int(b.get('det_count', 0))
            or a.get('score_checksum') != b.get('score_checksum')
        )
    status = 'FUSION_VERIFIED' if fusion_ok else 'FUSION_NOT_VERIFIED'
    md = [
        '# Fusion head switch audit (formal check)',
        '',
        f'- time: {datetime.now().isoformat()}',
        f'- status: **{status}**',
        '',
        'Path: `prompt_predict` with `val_using_aux` False/True.',
        'Heldout AP uses `infer_heldout_flex_ap` (same path).',
        '',
        '```json',
        json.dumps(rows, indent=2, ensure_ascii=False),
        '```',
    ]
    (U.RESULT_DIR / 'fres_30_fusion_head_switch_audit.md').write_text('\n'.join(md), encoding='utf-8')
    return status


def run_stage_ap(stage: str, cfg: Path, ckpt: Path, gpu: int, held_n: int,
                 step2_align_hist: dict = None) -> list:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    held = U.load_heldout_stems(held_n)
    rows = []
    for aux in (False, True):
        head = 'fusion' if aux else 'alignment'
        ctx = RC.RepairContext(
            repo_root=REPO, config=cfg, checkpoint=ckpt,
            support_pkl=U.SUPPORT_PKL, verify_work_dir=U.REPAIR_WD,
            work_dir=U.WORK_DIR / 'step23_ap' / f'{stage}_{head}',
            result_md_dir=U.RESULT_DIR, gpu=gpu, mode='formal_check',
            angles=[0, 90], tiles=[], train_max_images=0,
            heldout_max_images=len(held), eval_max_images=len(held),
            iters=0, batch_size=1, lr=0.0,
        )
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        try:
            bargs, model, device, det_support, name2id, id2name, _, _ = RC.build_model_ctx(ctx)
            subset = prepare_heldout_subset(ctx, len(held))
            res = infer_heldout_flex_ap(
                ctx, model, device, det_support, name2id, id2name, subset,
                val_using_aux=aux, max_images=len(held))
            hist = U.parse_hist(res.get('class_histogram', '{}'))
            kl = 'NA'
            if step2_align_hist and head == 'fusion':
                p = U.hist_to_probs(hist, list(RC.CLASSES))
                q = U.hist_to_probs(step2_align_hist, list(RC.CLASSES))
                kl = U.kl_div(p, q)
            row = dict(
                stage=stage, name=f'{stage}_{head}', head_mode=head, val_using_aux=aux,
                heldout_n=len(held), ap50=res.get('ap50_overall'), sv_ap50=res.get('sv_ap50'),
                detection_total=res.get('detection_total'),
                final_sv_ratio=res.get('final_sv_ratio'),
                sv_det_count=res.get('sv_det_count'),
                non_sv_det_count=res.get('non_sv_det_count'),
                class_histogram=res.get('class_histogram'),
                class_hist_kl_vs_step2_align=kl,
                actually_used_head=res.get('actually_used_head'),
                notes='', timestamp=datetime.now().isoformat(timespec='seconds'),
            )
            rows.append(row)
            U.append_csv(U.RESULT_DIR / 'ftable_33_step23_heldout500_ap_summary.csv', row, U.SCHEMA_AP)
            for pc in res.get('per_class', []) or []:
                U.append_csv(U.RESULT_DIR / 'ftable_34_step23_heldout500_per_class_ap.csv',
                             dict(stage=stage, head_mode=head, **pc),
                             ['stage', 'head_mode', 'class_name', 'ap50', 'num_gts', 'num_dets'])
            print('AP ok', stage, head, res.get('ap50_overall'))
        except Exception as exc:
            U.append_csv(U.RESULT_DIR / 'ftable_33_step23_heldout500_ap_summary.csv',
                         dict(stage=stage, head_mode=head, notes=str(exc), ap50='AP_EVAL_FAILED'),
                         U.SCHEMA_AP)
            print('AP fail', stage, head, exc)
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['smoke', 'ap', 'all'], default='all')
    ap.add_argument('--gpu', type=int, default=9)
    ap.add_argument('--heldout-n', type=int, default=500)
    args = ap.parse_args()
    U.ensure_dirs()
    fusion_status = 'SKIPPED'
    if args.task in ('smoke', 'all'):
        fusion_status = run_fusion_smoke(args.gpu)
    if args.task in ('ap', 'all'):
        s2 = run_stage_ap('Step2', U.STEP2_CONFIG, U.STEP2_CKPT, args.gpu, args.heldout_n)
        align_hist = U.parse_hist(
            next((r.get('class_histogram', '{}') for r in s2 if r.get('head_mode') == 'alignment'), '{}'))
        run_stage_ap('Step3', U.STEP3_CONFIG, U.STEP3_CKPT, args.gpu, args.heldout_n, align_hist)
        lines = ['# Step2/Step3 heldout-500 official AP', '', f'- fusion_preflight: {fusion_status}', '']
        for r in U.read_csv(U.RESULT_DIR / 'ftable_33_step23_heldout500_ap_summary.csv'):
            lines.append(
                f"- {r.get('stage')} {r.get('head_mode')}: AP50={r.get('ap50')} sv_AP50={r.get('sv_ap50')} "
                f"head={r.get('actually_used_head')}")
        (U.RESULT_DIR / 'fres_33_step23_heldout500_official.md').write_text('\n'.join(lines), encoding='utf-8')
    print('step23 gpu9 done', fusion_status)
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
