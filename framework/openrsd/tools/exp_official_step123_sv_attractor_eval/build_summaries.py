#!/usr/bin/env python
"""Aggregate CSVs into summary MD reports."""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.exp_official_step123_sv_attractor_eval.common_official_utils import (
    RESULT_DIR, STAGES, ensure_dirs, read_csv,
)


def mean_field(rows, key):
    vals = []
    for r in rows:
        try:
            v = float(r[key])
            if v == v:
                vals.append(v)
        except (TypeError, ValueError, KeyError):
            pass
    return statistics.mean(vals) if vals else float('nan')


def build_highlow():
    raw = read_csv(RESULT_DIR / 'ftable_10_highlow_mechanism_raw.csv')
    summ_rows = []
    for stage in STAGES:
        for support in ('visual', 'text'):
            for head in ('alignment', 'fusion'):
                sub = [r for r in raw if r.get('stage') == stage
                       and r.get('support_mode') == support and r.get('head_mode') == head]
                if not sub:
                    continue
                hi = [r for r in sub if r.get('group') == 'high']
                lo = [r for r in sub if r.get('group') == 'low']
                hf = mean_field(hi, 'final_sv_ratio')
                lf = mean_field(lo, 'final_sv_ratio')
                hd = mean_field(hi, 'dense_top1_sv_ratio')
                ld = mean_field(lo, 'dense_top1_sv_ratio')
                det = mean_field(sub, 'det_count')
                verdict = 'STRONG_ATTRACTOR' if hf > 0.5 else ('MILD_ATTRACTOR' if hf > 0.35 else 'CLEAN')
                summ_rows.append(dict(
                    stage=stage, support=support, head=head,
                    high_final_sv=hf, low_final_sv=lf,
                    high_dense_sv=hd, low_dense_sv=ld,
                    high_low_gap=hf - lf if hf == hf and lf == lf else 'NA',
                    det_count=det, verdict=verdict,
                ))
    import csv
    path = RESULT_DIR / 'ftable_11_highlow_mechanism_summary.csv'
    if summ_rows:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(summ_rows[0].keys()))
            w.writeheader()
            w.writerows(summ_rows)
    md = ['# High/Low Mechanism Response', '', '| stage | support | head | high_final_sv | low_final_sv | high_dense_sv | low_dense_sv | gap | verdict |',
          '|---|---|---|---:|---:|---:|---:|---:|---|']
    for r in summ_rows:
        md.append(f"| {r['stage']} | {r['support']} | {r['head']} | {r['high_final_sv']:.4f} | {r['low_final_sv']:.4f} | {r['high_dense_sv']:.4f} | {r['low_dense_sv']:.4f} | {r['high_low_gap']:.4f} | {r['verdict']} |")
    (RESULT_DIR / 'fres_10_highlow_mechanism_response.md').write_text('\n'.join(md), encoding='utf-8')


def build_p0148():
    raw = read_csv(RESULT_DIR / 'ftable_30_p0148_12angle_raw.csv')
    summ = []
    for stage in STAGES:
        sub = [r for r in raw if r.get('stage') == stage]
        if not sub:
            continue
        fs = [float(r['final_sv_ratio']) for r in sub if r.get('final_sv_ratio', 'NA') != 'NA']
        ds = [float(r['dense_top1_sv_ratio']) for r in sub if r.get('dense_top1_sv_ratio', 'NA') != 'NA']
        worst = max(sub, key=lambda r: float(r.get('final_sv_ratio', 0) or 0), default={})
        summ.append(dict(
            stage=stage,
            mean_final_sv=statistics.mean(fs) if fs else 'NA',
            worst_final_sv=max(fs) if fs else 'NA',
            mean_dense_sv=statistics.mean(ds) if ds else 'NA',
            angle_range='0-330 step30',
            worst_angle=worst.get('angle', 'NA'),
        ))
    import csv
    sp = RESULT_DIR / 'ftable_31_p0148_12angle_summary.csv'
    if summ:
        with open(sp, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
            w.writeheader()
            w.writerows(summ)
    md = ['# P0148 12-angle Stage Response', '',
          '| stage | mean_final_sv | worst_final_sv | mean_dense_sv | worst_angle |',
          '|---|---:|---:|---:|---:|']
    for r in summ:
        md.append(f"| {r['stage']} | {r['mean_final_sv']} | {r['worst_final_sv']} | {r['mean_dense_sv']} | {r['worst_angle']} |")
    (RESULT_DIR / 'fres_30_p0148_12angle_stage_response.md').write_text('\n'.join(md), encoding='utf-8')


def build_ap_md():
    rows = read_csv(RESULT_DIR / 'ftable_21_heldout_ap_summary.csv')
    md = ['# Heldout AP / sv_AP50', '',
          '| stage | head | AP50 | sv_AP50 | final_sv | det_total | notes |',
          '|---|---|---:|---:|---:|---:|---|']
    for r in rows:
        md.append(f"| {r.get('stage','')} | {r.get('head_mode','')} | {r.get('ap50','')} | {r.get('sv_ap50','')} | {r.get('final_sv_ratio','')} | {r.get('detection_total','')} | {r.get('notes','')} |")
    (RESULT_DIR / 'fres_20_heldout_ap_sv_ap.md').write_text('\n'.join(md), encoding='utf-8')


def build_evolution():
    hi = read_csv(RESULT_DIR / 'ftable_11_highlow_mechanism_summary.csv')
    ap = read_csv(RESULT_DIR / 'ftable_21_heldout_ap_summary.csv')
    p14 = read_csv(RESULT_DIR / 'ftable_31_p0148_12angle_summary.csv')
    rows = []
    for stage in STAGES:
        hsub = [r for r in hi if r.get('stage') == stage and r.get('support') == 'visual'
                and r.get('head') == 'alignment']
        asub = [r for r in ap if r.get('stage') == stage and r.get('head_mode') == 'alignment']
        psub = [r for r in p14 if r.get('stage') == stage]
        rows.append(dict(
            stage=stage,
            high_final_sv=hsub[0].get('high_final_sv', 'NA') if hsub else 'NA',
            low_final_sv=hsub[0].get('low_final_sv', 'NA') if hsub else 'NA',
            ap50=asub[0].get('ap50', 'NA') if asub else 'NA',
            sv_ap50=asub[0].get('sv_ap50', 'NA') if asub else 'NA',
            p0148_mean_final=psub[0].get('mean_final_sv', 'NA') if psub else 'NA',
        ))
    import csv
    ep = RESULT_DIR / 'ftable_40_stage_evolution_summary.csv'
    if rows:
        with open(ep, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    md = [
        '# Stage Evolution Analysis',
        '',
        '## Step1 → Step2 → Step3 (visual + alignment head)',
        '',
        '| stage | high_final_sv | low_final_sv | AP50 | sv_AP50 | P0148 mean final_sv |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        md.append(f"| {r['stage']} | {r['high_final_sv']} | {r['low_final_sv']} | {r['ap50']} | {r['sv_ap50']} | {r['p0148_mean_final']} |")
    md += [
        '',
        '## Answers (draft — refine after data)',
        '',
        '1. SV attractor in Step1: see Step1 row; dense hook N/A for E_Rtmdet_v2.',
        '2. Highest high_final_sv: compare Step1/2/3 columns.',
        '3. Highest dense_top1_sv: from highlow summary fusion/align rows.',
        '4. Best sv_AP50: max sv_AP50 column.',
        '5. AP up + attractor worse: check Step3 vs Step2.',
        '6. Step3 self-train: compare Step3 vs Step2 low_det and final_sv.',
        '7. visual vs text: compare rows in ftable_11.',
        '8. alignment vs fusion: compare head columns.',
    ]
    (RESULT_DIR / 'fres_40_stage_evolution_analysis.md').write_text('\n'.join(md), encoding='utf-8')


def build_final():
    hi = read_csv(RESULT_DIR / 'ftable_11_highlow_mechanism_summary.csv')
    ap = read_csv(RESULT_DIR / 'ftable_21_heldout_ap_summary.csv')
    evo = read_csv(RESULT_DIR / 'ftable_40_stage_evolution_summary.csv')

    def stage_verdict(stage):
        h = [r for r in hi if r.get('stage') == stage and r.get('head') == 'alignment'
             and r.get('support') == 'visual']
        if not h:
            return 'EVAL_INCOMPLETE'
        try:
            hf = float(h[0]['high_final_sv'])
            if hf > 0.50:
                return 'STRONG_ATTRACTOR'
            if hf > 0.35:
                return 'MILD_ATTRACTOR'
            return 'CLEAN'
        except (TypeError, ValueError):
            return 'EVAL_INCOMPLETE'

    v1, v2, v3 = stage_verdict('Step1'), stage_verdict('Step2'), stage_verdict('Step3')
    evo_verdict = 'INCONCLUSIVE'
    if v2 == 'STRONG_ATTRACTOR' and v1 == 'EVAL_INCOMPLETE':
        evo_verdict = 'ATTRACTOR_PRESENT_FROM_STEP2'
    elif v2 == 'STRONG_ATTRACTOR' and v1 != 'STRONG_ATTRACTOR':
        evo_verdict = 'ATTRACTOR_EMERGES_IN_STEP2'
    elif v1 == 'STRONG_ATTRACTOR':
        evo_verdict = 'ATTRACTOR_PRESENT_FROM_STEP1'
    elif v3 in ('CLEAN', 'MILD_ATTRACTOR') and v2 == 'STRONG_ATTRACTOR':
        evo_verdict = 'ATTRACTOR_MITIGATED_BY_STEP3'
    elif v2 == 'STRONG_ATTRACTOR' and v3 == 'STRONG_ATTRACTOR':
        evo_verdict = 'ATTRACTOR_PRESENT_FROM_STEP2'

    lines = [
        '# Official OpenRSD Step1/Step2/Step3 Weight Response to SV Attractor',
        '',
        f'- result_dir: `{RESULT_DIR}`',
        '',
        '## 1. Experiment goal',
        'Eval-only: official training-stage weights on SV attractor (no training, no weight edits).',
        '',
        '## 2–5. See fres_00 repo mapping, weight inventory, support audit, preflight, smoke.',
        '',
        '## 6. High-risk / low-risk',
        'See `fres_10_highlow_mechanism_response.md`.',
        '',
        '## 7. Heldout AP',
        'See `fres_20_heldout_ap_sv_ap.md`.',
        '',
        '## 8. P0148 12-angle',
        'See `fres_30_p0148_12angle_stage_response.md`.',
        '',
        '## 9. Stage evolution',
        'See `fres_40_stage_evolution_analysis.md`.',
        '',
        '## 12. Final verdict',
        '',
        f'- Step1: **{v1}** (dense hook unavailable on E_Rtmdet_v2)',
        f'- Step2: **{v2}**',
        f'- Step3: **{v3}**',
        f'- Evolution: **{evo_verdict}**',
        '',
        '## 13. Next action',
    ]
    if evo_verdict == 'ATTRACTOR_EMERGES_IN_STEP2':
        lines.append('A10/Step2 is the attractor formation point; add dehub regularization at Step2 finetune.')
    elif evo_verdict == 'ATTRACTOR_MITIGATED_BY_STEP3':
        lines.append('Study Step3 self-train recipe; SV-DeHub may target Step3 if attractor returns.')
    elif evo_verdict == 'ATTRACTOR_PRESENT_FROM_STEP1':
        lines.append('Issue likely from large-scale pretrain/support embedding; fix earlier than A10 finetune.')
    else:
        lines.append('Complete Step1 detection path + heldout-500 AP before new method work.')

    (RESULT_DIR / 'fres_official_step123_sv_attractor_summary.md').write_text('\n'.join(lines), encoding='utf-8')

    # terminal block
    best_sv = max(ap, key=lambda r: float(r.get('sv_ap50', 0) or 0), default={})
    print('[Official Step123 SV Attractor Eval Final]')
    print('RESULT_DIR:', RESULT_DIR)
    print('tested_stages: Step1,Step2,Step3')
    print('step1_verdict:', v1)
    print('step2_verdict:', v2)
    print('step3_verdict:', v3)
    print('stage_evolution_verdict:', evo_verdict)
    print('final_report_path:', RESULT_DIR / 'fres_official_step123_sv_attractor_summary.md')


def main():
    ensure_dirs()
    build_highlow()
    build_p0148()
    build_ap_md()
    build_evolution()
    build_final()


if __name__ == '__main__':
    main()
