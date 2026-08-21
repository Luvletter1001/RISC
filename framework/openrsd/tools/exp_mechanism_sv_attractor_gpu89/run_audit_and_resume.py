#!/usr/bin/env python
"""Audit mechanism experiment outputs and write resume plan."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.exp_mechanism_sv_attractor_gpu89 import common_attractor_utils as U

RESULT = U.RESULT_MD
ATLAS = RESULT / 'ftable_01_attractor_atlas_raw.csv'
METHODS = ['baseline', 'best_R1', 'best_C5', 'zero_sv_embedding', 'text_support', 'visual_support']
ANGLES = U.ATLAS_ANGLES


def read_rows(p: Path):
    if not p.exists():
        return []
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def atlas_audit(rows):
    by = defaultdict(set)
    err = 0
    for r in rows:
        if str(r.get('dense_top1_sv_ratio', '')) in ('NA', '') or 'no image' in str(r.get('notes', '')):
            err += 1
        by[(r['tile_id'], r['method'])].add(int(float(r['angle'])))
    tiles = sorted({r['tile_id'] for r in rows})
    incomplete = [
        dict(tile=t, method=m, n_angles=len(by[(t, m)]), need=len(ANGLES))
        for t in tiles for m in METHODS
        if len(by[(t, m)]) < len(ANGLES)
    ]
    return dict(
        rows=len(rows), tiles=len(tiles), methods=sorted({r['method'] for r in rows}),
        errors=err, incomplete=incomplete,
        complete=len(incomplete) == 0 and err == 0,
    )


def combo_audit(path: Path, tile_key, group_key, groups, angles):
    rows = read_rows(path)
    by = defaultdict(set)
    for r in rows:
        by[(r[tile_key], r[group_key])].add(int(float(r['angle'])))
    tiles = sorted({r[tile_key] for r in rows})
    missing = []
    for t in tiles:
        for g in groups:
            have = by.get((t, g), set())
            need = set(angles)
            if have != need:
                missing.append(dict(tile=t, group=g, have=sorted(have), missing=sorted(need - have)))
    return dict(rows=len(rows), tiles=len(tiles), missing=missing, complete=len(missing) == 0)


def extra_heldout_plan(n: int = 10):
    meta_path = RESULT / 'fmeta_01_tile_list.json'
    selected = set()
    if meta_path.exists():
        selected = set(json.loads(meta_path.read_text()).get('selected', []))
    asv0 = REPO / 'data/DOTA1_1024_500/angle_sweep_val/realistic/angle_000/images'
    if not asv0.is_dir():
        return []
    import random
    pool = sorted({p.stem for p in asv0.glob('*.png')} - selected)
    random.Random(U.MechContext.seed).shuffle(pool)
    out = []
    for stem in pool:
        if all((REPO / f'data/DOTA1_1024_500/angle_sweep_val/realistic/angle_{a:03d}/images/{stem}.png').exists()
               for a in ANGLES):
            out.append(stem)
        if len(out) >= n:
            break
    return out


def main():
    RESULT.mkdir(parents=True, exist_ok=True)
    atlas_rows = read_rows(ATLAS)
    audit = {
        'atlas': atlas_audit(atlas_rows),
        'decomp': combo_audit(
            RESULT / 'ftable_02_logit_embedding_decomposition.csv',
            'tile_id', 'intervention',
            ['baseline', 'zero_sv_embedding', 'swap_sv_with_nearest_class_embedding',
             'orthogonalize_sv_embedding_to_background_mean', 'normalize_all_class_embeddings',
             'bias_only_R1', 'temperature_only_R1', 'bias_plus_temperature_R1'],
            U.DECOMP_ANGLES),
        'background': combo_audit(
            RESULT / 'ftable_03_background_counterfactual.csv',
            'tile_id', 'condition',
            ['A_original', 'B_gt_object_masked', 'C_background_only', 'D_object_only',
             'E_highrisk_bg_paste', 'F_lowrisk_bg_paste', 'G_random_patch_shuffle'],
            U.DECOMP_ANGLES),
        'repair': combo_audit(
            RESULT / 'ftable_06_minimal_repair_mechanism.csv',
            'tile_id', 'method',
            ['baseline', 'bias_only_R1', 'temperature_only_R1', 'bias_plus_temperature_R1',
             'best_C5', 'oracle_remove_sv_bias'],
            [0, 90]),
        'reports_present': {p.name: (RESULT / p.name).exists() for p in [
            Path('fres_00_preflight.md'), Path('fres_01_attractor_atlas.md'),
            Path('fres_02_logit_embedding_decomposition.md'),
            Path('fres_03_background_counterfactual.md'),
            Path('fres_04_rotation_tile_variance_decomposition.md'),
            Path('fres_05_risk_predictor.md'), Path('fres_06_minimal_repair_mechanism.md'),
            Path('fres_mechanism_sv_attractor_summary.md')]},
        'known_fixed_bugs': [
            'setup_logging mkdir logs/',
            'summarize_atlas no pandas',
            'R1 bias copy torch.no_grad()',
            'heldout quick probe uses angle_sweep stems only',
            'angle_sweep heldout: txt→pkl mini dataset (prepare_angle_sweep_mini_dataset)',
        ],
        'original_gap': 'Design +30 heldout beyond vis/cross; atlas has +10 heldout (30 tiles total)',
        'extra_heldout_candidates': extra_heldout_plan(10),
    }
    audit['resume_needed'] = {
        'atlas_extra_heldout': len(audit['extra_heldout_candidates']) > 0,
        'atlas_incomplete': not audit['atlas']['complete'],
        'any_subexp_incomplete': any(
            not audit[k]['complete'] for k in ('decomp', 'background', 'repair')),
    }

    (RESULT / 'fmeta_audit_status.json').write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')

    lines = [
        '# Audit & Resume Status',
        '',
        f"- atlas: **{audit['atlas']['rows']}** rows, **{audit['atlas']['tiles']}** tiles, "
        f"complete={audit['atlas']['complete']}",
        f"- decomp complete={audit['decomp']['complete']} ({audit['decomp']['rows']} rows)",
        f"- background complete={audit['background']['complete']} ({audit['background']['rows']} rows)",
        f"- repair complete={audit['repair']['complete']} ({audit['repair']['rows']} rows)",
        '',
        '## 原任务缺口',
        '',
        f"- Atlas：**{audit['atlas']['tiles']}** tiles（P0148 + 19 cross + 10 heldout），"
        f"{'**完整**' if audit['atlas']['complete'] else '**不完整**'}。",
    ]
    if audit['resume_needed']['atlas_extra_heldout']:
        lines += [
            f"- 可选扩展：angle_sweep 池内仍有候选 heldout，例如 "
            f"`{audit['extra_heldout_candidates'][:3]}...`（`--extra-heldout N`）。",
        ]
    lines += [
        '',
        '## 历史失败（已修复）',
        '',
    ]
    for b in audit['known_fixed_bugs']:
        lines.append(f'- {b}')
    lines += ['', '## Resume 命令', '', '```bash', '# heldout 扩展 (GPU9)', 
              'CUDA_VISIBLE_DEVICES=9 python tools/exp_mechanism_sv_attractor_gpu89/run_gpu9_attractor_atlas.py --extra-heldout 10',
              '# 仅重算报告', 'python tools/exp_mechanism_sv_attractor_gpu89/run_audit_and_resume.py',
              'python tools/exp_mechanism_sv_attractor_gpu89/analyze_variance_decomposition.py',
              'python tools/exp_mechanism_sv_attractor_gpu89/analyze_risk_predictor.py',
              'python tools/exp_mechanism_sv_attractor_gpu89/build_final_mechanism_report.py',
              '```']
    (RESULT / 'fres_audit_resume.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(audit['resume_needed'], indent=2))
    print('wrote', RESULT / 'fres_audit_resume.md')


if __name__ == '__main__':
    main()
