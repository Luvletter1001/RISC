#!/usr/bin/env python
"""Build final 3h verification summary."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tools.exp_sv_dehub_lite_train_gpu89.plan3h import common_plan3h_utils as P


def read_csv(path: Path):
    if not path.exists():
        return []
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verdict', default='')
    args = ap.parse_args()
    P.ensure_dirs()
    out = P.VERIFY_DIR / 'fres_verify_ap_ablation_plan3h_summary.md'
    bl_rows = read_csv(P.VERIFY_DIR / 'ftable_00_patched_baseline_consistency.csv')
    ap_rows = read_csv(P.VERIFY_DIR / 'ftable_01_full_ap_eval_summary.csv')
    ab_rows = read_csv(P.VERIFY_DIR / 'ftable_02_ablation_eval_summary.csv')
    lr_rows = read_csv(P.VERIFY_DIR / 'ftable_03_lowrisk_side_effect_audit.csv')
    mon_rows = read_csv(P.VERIFY_DIR / 'ftable_plan3h_monitor.csv')

    pb_verdict = 'NA'
    fres0 = P.VERIFY_DIR / 'fres_00_patched_baseline_consistency.md'
    if fres0.exists():
        for line in fres0.read_text().splitlines():
            if 'verdict' in line.lower() and '**' in line:
                pb_verdict = line.split('**')[1]
                break

    verdict = args.verdict
    if not verdict:
        if pb_verdict == 'BROKEN':
            verdict = 'SUPPORT_FIX_CONFOUNDED'
        elif not ap_rows or all(str(r.get('ap50', 'NA')) in ('NA', 'nan', '') for r in ap_rows):
            verdict = 'PROMISING_BUT_AP_UNKNOWN'
        elif ap_rows:
            bl = next((r for r in ap_rows if r['checkpoint'] == 'patched_baseline_epoch24'), ap_rows[0] if ap_rows else None)
            best = max(ap_rows, key=lambda r: float(r['ap50']) if str(r.get('ap50', '')).replace('NA', '').strip() else -1)
            try:
                b_ap = float(bl['ap50']) if bl else 0
                if float(best['ap50']) >= b_ap - 0.015:
                    verdict = 'VERIFIED_PROMISING'
                else:
                    verdict = 'AP_TRADEOFF'
            except (TypeError, ValueError):
                verdict = 'PROMISING_BUT_AP_UNKNOWN'
        else:
            verdict = 'FAILED'

    best_safe = 'iter_1000'
    best_agg = 'iter_2000'
    rec = 'iter_2000'
    if ap_rows:
        for r in ap_rows:
            try:
                if float(r.get('high_final_sv', 99)) < 0.35 and float(r.get('ap50', 0)) >= 0.73:
                    best_agg = r['checkpoint']
                if float(r.get('low_det_count', 999)) < 100 and float(r.get('ap50', 0)) >= 0.72:
                    best_safe = r['checkpoint']
            except (TypeError, ValueError):
                pass
        rec = best_safe if verdict == 'VERIFIED_PROMISING' else best_agg

    lines = [
        '# SV-DeHub-Lite v1 Three-Hour Verification Summary',
        '',
        f'- generated: {datetime.now().isoformat(timespec="seconds")}',
        f'- verify_dir: `{P.VERIFY_DIR}`',
        '',
        '## 1. Three-hour timeline',
        '',
        '| time | action | status | notes |',
        '|---|---|---|---|',
    ]
    for r in mon_rows[-8:]:
        lines.append(f"| {r.get('check_time','')} | monitor | {r.get('action_taken','')} | AP rows={r.get('ap_eval_rows','')} |")
    if not mon_rows:
        lines.append('| T0 | pipeline start | ok | |')

    lines += ['', '## 2. Patched baseline consistency', '', f'**Overall: {pb_verdict}**', '',
              '| group | metric | old_baseline | patched_baseline | rel_diff | status |',
              '|---|---|---:|---:|---:|---|']
    for r in bl_rows:
        lines.append(f"| {r.get('group','')} | {r.get('metric','')} | {r.get('old_baseline','')} | {r.get('patched_baseline','')} | {r.get('rel_diff','')} | {r.get('status','')} |")

    lines += ['', '## 3. Full AP verification', '',
              '| checkpoint | AP50 | sv_AP50 | high_final_sv | low_final_sv | det_count | verdict |',
              '|---|---:|---:|---:|---:|---:|---|']
    for r in ap_rows:
        v = 'ok'
        try:
            if float(r['ap50']) < 0.65:
                v = 'AP_TRADEOFF'
        except (TypeError, ValueError):
            v = 'AP_EVAL_FAILED'
        lines.append(f"| {r.get('checkpoint','')} | {r.get('ap50','')} | {r.get('sv_ap50','')} | {r.get('high_final_sv','')} | {r.get('low_final_sv','')} | {r.get('detection_total','')} | {v} |")

    lines += ['', '## 4. Minimal ablation', '',
              '| variant | high_final_sv | dense_top1_sv | AP50 | sv_AP50 | interpretation |',
              '|---|---:|---:|---:|---:|---|']
    for r in ab_rows:
        lines.append(f"| {r.get('variant','')} | {r.get('high_final_sv','')} | {r.get('high_dense_sv','')} | {r.get('ap50','')} | {r.get('sv_ap50','')} | |")

    lines += ['', '## 5. Low-risk side effect audit', '',
              '| checkpoint | low_final_sv | low_det_count | main_changed_class | mean_score | interpretation |',
              '|---|---:|---:|---|---:|---|']
    for r in lr_rows:
        lines.append(f"| {r.get('checkpoint','')} | {r.get('low_final_sv','')} | {r.get('low_det_count','')} | {r.get('main_changed_class','')} | {r.get('mean_score','')} | {r.get('interpretation','')} |")

    lines += [
        '', '## 6. Best checkpoint selection', '',
        f'- **best_safe_checkpoint**: {best_safe}',
        f'- **best_aggressive_checkpoint**: {best_agg}',
        f'- **recommended_checkpoint_for_next_train**: {rec}',
        '', '## 7. Final verdict', '', f'**{verdict}**', '',
        '## 8. Next action', '',
    ]
    next_map = {
        'VERIFIED_PROMISING': f'Resume 8k from {rec}; heldout 500 AP at end.',
        'PROMISING_BUT_AP_UNKNOWN': 'Fix AP eval pipeline before more training.',
        'AP_TRADEOFF': 'Lower dehub weight or restrict dehub to paste regions.',
        'ABLATION_INCONCLUSIVE': 'Re-run cleaner loss-only / paste-only checks.',
        'SUPPORT_FIX_CONFOUNDED': 'Re-run overnight smoke with patched baseline as reference.',
        'FAILED': 'Pause train direction; return to R1/C5.',
    }
    lines.append(next_map.get(verdict, 'Review partial logs in VERIFY_DIR.'))

    out.write_text('\n'.join(lines), encoding='utf-8')
    meta = dict(verdict=verdict, best_safe=best_safe, best_agg=best_agg, recommended=rec)
    (P.VERIFY_DIR / 'fmeta_plan3h_final.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    bl_ap = next((r for r in ap_rows if r.get('checkpoint') == 'patched_baseline_epoch24'), {})
    i2 = next((r for r in ap_rows if r.get('checkpoint') == 'iter_2000'), {})
    print('[SV-DeHub Plan3H Final]')
    print('VERIFY_DIR:', P.VERIFY_DIR)
    print('patched_baseline_status:', pb_verdict)
    print('full_AP_status:', 'ok' if ap_rows else 'missing')
    print('ablation_status:', 'ok' if ab_rows else 'partial')
    print('lowrisk_side_effect_status:', 'ok' if lr_rows else 'partial')
    print('best_safe_checkpoint:', best_safe)
    print('best_aggressive_checkpoint:', best_agg)
    print('recommended_checkpoint_for_next_train:', rec)
    try:
        print('AP50_delta:', float(i2.get('ap50', 0)) - float(bl_ap.get('ap50', 0)))
        print('sv_AP50_delta:', float(i2.get('sv_ap50', 0)) - float(bl_ap.get('sv_ap50', 0)))
        print('highrisk_final_sv_delta:', float(i2.get('high_final_sv', 0)) - float(bl_ap.get('high_final_sv', 0)))
        print('lowrisk_det_count_change:', float(i2.get('low_det_count', 0)) - float(bl_ap.get('low_det_count', 0)))
    except (TypeError, ValueError):
        pass
    print('debug_events:', 'see fres_debug_* if present')
    print('final_verdict:', verdict)
    print('final_report_path:', out)
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
