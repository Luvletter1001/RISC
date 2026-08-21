#!/usr/bin/env python
"""Task A: unified class drift on GPU8."""
from __future__ import annotations

import json
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.exp_formal_success_check_gpu89 import common_formal_utils as U
from tools.exp_mechanism_sv_attractor_gpu89 import common_attractor_utils as mech
from tools.exp_next_plan_sv_dehub_step23_overnight.eval_flex_heads import eval_one_flex_head
from tools.verify_sv_attractor_gpu89 import common as verify

CLASSES = verify.CLASSES


def score_percentiles(scores: list) -> dict:
    if not scores:
        return dict(p50='NA', p75='NA', p90='NA', p95='NA')
    a = np.asarray(scores, dtype=float)
    return dict(
        score_p50=float(np.quantile(a, 0.5)),
        score_p75=float(np.quantile(a, 0.75)),
        score_p90=float(np.quantile(a, 0.9)),
        score_p95=float(np.quantile(a, 0.95)),
    )


def top3_from_hist(hist: dict) -> str:
    items = sorted(hist.items(), key=lambda x: -x[1])[:3]
    return json.dumps(items, ensure_ascii=True)


def drift_verdict_row(row: dict, base_low_det: float, base_low_non_sv: float) -> str:
    grp = row.get('group', '')
    if grp != 'low':
        return 'N/A_group_high'
    det = float(row.get('det_count', 0))
    infl = float(row.get('non_sv_inflation_vs_baseline', 0) or 0)
    kl = float(row.get('class_hist_kl_vs_baseline', 0) or 0)
    tennis = int(float(row.get('tennis_court_count', 0) or 0))
    lv = int(float(row.get('large_vehicle_count', 0) or 0))
    det_ratio = (det - base_low_det) / max(base_low_det, 1)
    if det_ratio > 1.0 or infl > 1.0 or tennis > 80 or lv > 120:
        return 'RISK'
    if det_ratio > 0.5 or infl > 0.5 or kl > 0.15:
        return 'WATCH'
    return 'SAFE'


def run_checkpoint(name: str, ckpt: Path, cfg: Path, gpu: int, raw_path: Path) -> list:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    ctx = mech.MechContext(config=cfg, checkpoint=ckpt,
                           work_dir=U.WORK_DIR / 'drift' / name, result_md_dir=U.RESULT_DIR, gpu=gpu)
    bargs, model, device, det_support, name2id, id2name, _ = mech.build_model(ctx, 'visual')
    rows = []
    for grp, tiles in [('high', U.HIGHRISK), ('low', U.LOWRISK)]:
        for tile in tiles:
            for angle in U.ANGLES:
                try:
                    r = eval_one_flex_head(ctx, model, bargs, device, det_support, name2id, id2name,
                                           tile, angle, 'visual', val_using_aux=False)
                    hist = U.parse_hist(r.get('top1_class_histogram', '{}'))
                    top1 = max(hist.items(), key=lambda x: x[1])[0] if hist else 'NA'
                    row = dict(
                        source_type='dehub' if 'iter' in name or name == 'baseline_epoch24' else 'official',
                        name=name, checkpoint=str(ckpt), config_path=str(cfg),
                        support='visual', head='alignment', group=grp,
                        tile_id=tile, angle=angle,
                        final_sv=r.get('final_sv_ratio'), dense_sv=r.get('dense_top1_sv_ratio'),
                        det_count=r.get('det_count'), sv_det_count=r.get('small_vehicle_count'),
                        non_sv_det_count=r.get('non_sv_det_count'),
                        top1_class_histogram=r.get('top1_class_histogram', '{}'),
                        top1_class=top1, top1_class_count=hist.get(top1, 0),
                        top3_classes=top3_from_hist(hist),
                        tennis_court_count=r.get('tennis_court_count'),
                        large_vehicle_count=r.get('large_vehicle_count'),
                        basketball_court_count=r.get('basketball_court_count'),
                        harbor_count=hist.get('harbor', 0), plane_count=hist.get('plane', 0),
                        mean_score=r.get('mean_score'),
                        notes=r.get('notes', ''),
                        timestamp=datetime.now().isoformat(timespec='seconds'),
                    )
                    row.update(score_percentiles([]))
                    U.append_csv(raw_path, row, U.SCHEMA_DRIFT)
                    rows.append(row)
                except Exception as exc:
                    U.append_csv(raw_path, dict(name=name, group=grp, tile_id=tile, angle=angle,
                                                notes=str(exc)), U.SCHEMA_DRIFT)
    return rows


def enrich_vs_baseline(all_rows: list) -> list:
    base = {}
    for r in all_rows:
        if r.get('name') != 'baseline_epoch24':
            continue
        key = (r.get('group'), r.get('tile_id', ''), r.get('angle', ''))
        if 'tile_id' not in r:
            key = (r.get('group'),)
        base.setdefault(r.get('group'), []).append(r)

    base_grp = {}
    for g in ('high', 'low'):
        sub = [r for r in all_rows if r.get('name') == 'baseline_epoch24' and r.get('group') == g]
        if sub:
            base_grp[g] = dict(
                final=statistics.mean([float(r['final_sv']) for r in sub]),
                dense=statistics.mean([float(r['dense_sv']) for r in sub if r.get('dense_sv') != 'NA']),
                det=statistics.mean([float(r['det_count']) for r in sub]),
                non_sv=statistics.mean([float(r['non_sv_det_count']) for r in sub]),
                hist=sum((U.parse_hist(r.get('top3_classes', '[]')) for r in sub), []),
            )
    # aggregate hist from top1_class counts per row
    base_hist = {}
    for r in all_rows:
        if r.get('name') != 'baseline_epoch24':
            continue
        for cls, cnt in U.parse_hist(r.get('top1_class_histogram', '{}')).items() if False else []:
            pass
    base_hist_rows = [r for r in all_rows if r.get('name') == 'baseline_epoch24']
    base_hist_agg = {}
    for r in base_hist_rows:
        for cls, cnt in U.parse_hist(
                json.dumps({r.get('top1_class', 'NA'): r.get('top1_class_count', 0)})).items():
            base_hist_agg[cls] = base_hist_agg.get(cls, 0) + int(cnt)
    for r in base_hist_rows:
        h = {}
        tc = r.get('top1_class')
        if tc and tc != 'NA':
            h[tc] = int(float(r.get('top1_class_count', 0)))
        for cls in ('tennis-court', 'large-vehicle', 'basketball-court', 'small-vehicle'):
            if cls in str(r.get('tennis_court_count', '')):
                pass
        hist = dict(
            **{k: int(r.get(f'{k.replace("-","_")}_count', 0) or 0) for k in []},
        )
        hist = U.parse_hist(r.get('top1_class_histogram', '{}')) if 'top1_class_histogram' in r else {}
        if not hist and r.get('top1_class', 'NA') != 'NA':
            hist = {r['top1_class']: int(float(r.get('top1_class_count', 0)))}
        for cls, cnt in hist.items():
            base_hist_agg[cls] = base_hist_agg.get(cls, 0) + int(cnt)

    base_probs = U.hist_to_probs(base_hist_agg, CLASSES)
    out = []
    for r in all_rows:
        hist = U.parse_hist(r.get('top1_class_histogram', '{}'))
        probs = U.hist_to_probs(hist, CLASSES)
        r['class_hist_kl_vs_baseline'] = U.kl_div(probs, base_probs)
        r['class_hist_l1_vs_baseline'] = U.l1_dist(probs, base_probs)
        g = r.get('group')
        if g in base_grp:
            r['non_sv_inflation_vs_baseline'] = (
                (float(r['non_sv_det_count']) - base_grp[g]['non_sv']) / max(base_grp[g]['non_sv'], 1))
            if g == 'high':
                r['high_final_sv_delta'] = float(r['final_sv']) - base_grp[g]['final']
                r['high_dense_sv_delta'] = float(r.get('dense_sv', 0)) - base_grp[g]['dense']
            if g == 'low':
                r['low_det_count_delta'] = float(r['det_count']) - base_grp[g]['det']
                r['low_non_sv_delta'] = float(r['non_sv_det_count']) - base_grp[g]['non_sv']
        r['drift_verdict'] = drift_verdict_row(r, base_grp.get('low', {}).get('det', 54),
                                                base_grp.get('low', {}).get('non_sv', 40))
        out.append(r)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=8)
    args = ap.parse_args()
    U.ensure_dirs()
    raw = U.RESULT_DIR / 'ftable_40_class_drift_all.csv'
    if raw.exists():
        raw.unlink()

    inv = {r['name']: r for r in U.checkpoint_inventory() if r['exists']}
    all_rows = []
    for name in ['baseline_epoch24', 'B_iter8000', 'B_iter5000', 'A_iter8000',
                 'Step2_official_epoch24', 'Step3_official_epoch24']:
        if name not in inv:
            continue
        r = inv[name]
        rows = run_checkpoint(name, Path(r['checkpoint_path']), Path(r['config_path']),
                              args.gpu, raw)
        all_rows.extend(rows)
        print('drift done', name, len(rows))

    enriched = enrich_vs_baseline(all_rows)
    summ_path = U.RESULT_DIR / 'ftable_41_class_drift_summary.csv'
    if summ_path.exists():
        summ_path.unlink()
    fields = ['name', 'group', 'final_sv', 'dense_sv', 'det_count', 'non_sv_det_count',
              'non_sv_inflation_vs_baseline', 'tennis_court_count', 'large_vehicle_count',
              'class_hist_kl_vs_baseline', 'drift_verdict']
    import csv
    agg = {}
    for r in enriched:
        key = (r['name'], r['group'])
        agg.setdefault(key, []).append(r)
    with open(summ_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for (name, grp), rs in sorted(agg.items()):
            w.writerow(dict(
                name=name, group=grp,
                final_sv=statistics.mean([float(x['final_sv']) for x in rs]),
                dense_sv=statistics.mean([float(x['dense_sv']) for x in rs if x.get('dense_sv') != 'NA']),
                det_count=statistics.mean([float(x['det_count']) for x in rs]),
                non_sv_det_count=statistics.mean([float(x['non_sv_det_count']) for x in rs]),
                non_sv_inflation_vs_baseline=statistics.mean(
                    [float(x.get('non_sv_inflation_vs_baseline', 0) or 0) for x in rs]),
                tennis_court_count=statistics.mean([float(x.get('tennis_court_count', 0)) for x in rs]),
                large_vehicle_count=statistics.mean([float(x.get('large_vehicle_count', 0)) for x in rs]),
                class_hist_kl_vs_baseline=statistics.mean(
                    [float(x.get('class_hist_kl_vs_baseline', 0) or 0) for x in rs]),
                drift_verdict=rs[0].get('drift_verdict', 'NA'),
            ))

    md = ['# Class drift audit', '', f'- rows: {len(enriched)}', '']
    (U.RESULT_DIR / 'fres_40_class_drift_all.md').write_text('\n'.join(md), encoding='utf-8')
    print('class drift complete', len(enriched))
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
