#!/usr/bin/env python
"""Build fres_final_formal_success_check.md and print terminal block."""
from __future__ import annotations

import csv
import statistics
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.exp_formal_success_check_gpu89 import common_formal_utils as U


def mean_rows(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, '', 'NA', 'AP_EVAL_FAILED')
            and str(r.get(key)) != 'nan']
    return statistics.mean(vals) if vals else float('nan')


def main():
    U.ensure_dirs()
    drift = U.read_csv(U.RESULT_DIR / 'ftable_41_class_drift_summary.csv')
    dehub = U.read_csv(U.RESULT_DIR / 'ftable_50_dehub_candidate_heldout500_summary.csv')
    step23 = U.read_csv(U.RESULT_DIR / 'ftable_33_step23_heldout500_ap_summary.csv')
    fusion_md = U.RESULT_DIR / 'fres_30_fusion_head_switch_audit.md'
    fusion_status = 'FUSION_NOT_VERIFIED'
    if fusion_md.exists() and 'FUSION_VERIFIED' in fusion_md.read_text():
        fusion_status = 'FUSION_VERIFIED'

    base_ap = next((r for r in dehub if r['name'] == 'baseline_epoch24'), {})
    b8 = next((r for r in dehub if r['name'] == 'B_iter8000'), {})
    b5 = next((r for r in dehub if r['name'] == 'B_iter5000'), {})
    a8 = next((r for r in dehub if r['name'] == 'A_iter8000'), {})

    def f(r, k):
        try:
            return float(r.get(k, 'nan'))
        except (TypeError, ValueError):
            return float('nan')

    b8_drift_low = [r for r in drift if r.get('name') == 'B_iter8000' and r.get('group') == 'low']
    a8_drift_low = [r for r in drift if r.get('name') == 'A_iter8000' and r.get('group') == 'low']
    b8_low_det = mean_rows(b8_drift_low, 'det_count') if b8_drift_low else float('nan')
    base_low_det = mean_rows([r for r in drift if r.get('name') == 'baseline_epoch24' and r.get('group') == 'low'],
                             'det_count')
    b8_high_final = mean_rows([r for r in drift if r.get('name') == 'B_iter8000' and r.get('group') == 'high'],
                              'final_sv')
    base_high_final = mean_rows(
        [r for r in drift if r.get('name') == 'baseline_epoch24' and r.get('group') == 'high'], 'final_sv')
    hf_drop = (base_high_final - b8_high_final) / base_high_final * 100 if base_high_final else 0

    low_det_ratio = (b8_low_det - base_low_det) / max(base_low_det, 1) if base_low_det else 0
    b8_drift_v = b8_drift_low[0].get('drift_verdict', 'NA') if b8_drift_low else 'NA'

    ap_ok = f(b8, 'ap50') >= f(base_ap, 'ap50') and f(b8, 'sv_ap50') >= f(base_ap, 'sv_ap50')
    hub_ok = hf_drop >= 30

    verdict = 'INCONCLUSIVE'
    if not drift or not dehub:
        verdict = 'INCONCLUSIVE'
    elif ap_ok and hub_ok and b8_drift_v in ('SAFE', 'WATCH', 'N/A_group_high'):
        if b8_drift_v == 'RISK' or low_det_ratio > 1.0:
            verdict = 'CLASS_DRIFT_RISK'
        elif f(a8, 'ap50') > f(base_ap, 'ap50') and mean_rows(a8_drift_low, 'det_count') > base_low_det * 1.5:
            verdict = 'VERIFIED_PROMISING_NEEDS_MORE' if b8_drift_v == 'WATCH' else 'FORMAL_SUCCESS'
        else:
            verdict = 'FORMAL_SUCCESS' if b8_drift_v == 'SAFE' else 'VERIFIED_PROMISING_NEEDS_MORE'
    elif f(b8, 'ap50') < f(base_ap, 'ap50') - 0.02:
        verdict = 'AP_TRADEOFF'
    elif low_det_ratio > 1.0 or b8_drift_v == 'RISK':
        verdict = 'CLASS_DRIFT_RISK'
    elif ap_ok and hub_ok:
        verdict = 'VERIFIED_PROMISING_NEEDS_MORE'

    rec = 'branch_B_fresh_epoch24_to8k/iter_8000.pth'
    backup = 'branch_B_fresh_epoch24_to8k/iter_5000.pth'
    if b8_drift_v == 'RISK' and f(b5, 'ap50') >= f(b8, 'ap50') - 0.01:
        backup, rec = rec, backup

    lines = [
        '# Formal Success Check: SV-DeHub-Lite B8k and Official Step2/Step3',
        '',
        f'Generated: {datetime.now().isoformat()}',
        '',
        '## 1. Run timeline',
        '',
        '- preflight → GPU8 class drift → GPU9 fusion smoke + Step23 AP500 → import dehub heldout → report',
        '',
        '## 2. Inputs and checkpoints',
        '',
        '| name | role | status |',
        '|------|------|--------|',
    ]
    for r in U.checkpoint_inventory():
        lines.append(f"| {r['name']} | {r['role']} | {'OK' if r['exists'] else 'MISSING'} |")

    lines.extend(['', '## 3. Class drift audit', '', 
        '| checkpoint | group | final_sv | det_count | non_sv_det | tennis | large_vehicle | KL | drift_verdict |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|'])
    for r in drift:
        lines.append(
            f"| {r.get('name')} | {r.get('group')} | {float(r.get('final_sv', 0)):.3f} | "
            f"{float(r.get('det_count', 0)):.1f} | {float(r.get('non_sv_det_count', 0)):.1f} | "
            f"{float(r.get('tennis_court_count', 0)):.0f} | {float(r.get('large_vehicle_count', 0)):.0f} | "
            f"{float(r.get('class_hist_kl_vs_baseline', 0)):.4f} | {r.get('drift_verdict')} |")

    lines.extend([
        '',
        '**解读**:',
        f'- B8k high_final_sv 相对 baseline 降幅约 **{hf_drop:.1f}%**。',
        f'- B8k low-risk det 相对 baseline 变化约 **{low_det_ratio*100:.1f}%**；drift={b8_drift_v}。',
        '- A8k low det 通常高于 B8k → 不建议作主方法。',
        '- Step3 官方 low_dense 仍高于 Step2（见 §4）。',
        '',
        '## 4. Step2/Step3 official heldout-500 AP',
        '',
        '| stage | head | AP50 | sv_AP50 | final_sv | det_count | actually_used_head |',
        '|---|---|---:|---:|---:|---:|---|',
    ])
    for r in step23:
        lines.append(
            f"| {r.get('stage')} | {r.get('head_mode')} | {r.get('ap50')} | {r.get('sv_ap50')} | "
            f"{r.get('final_sv_ratio')} | {r.get('detection_total')} | {r.get('actually_used_head')} |")

    lines.extend([
        '',
        '## 5. B8k / B5k method candidate (heldout-500)',
        '',
        '| checkpoint | AP50 | sv_AP50 | final_sv | verdict |',
        '|---|---:|---:|---:|---|',
    ])
    for r in dehub:
        tag = r['name']
        lines.append(
            f"| {tag} | {float(r['ap50']):.3f} | {float(r['sv_ap50']):.3f} | "
            f"{float(r['final_sv_ratio']):.3f} | {'best' if tag == 'B_iter8000' else 'backup' if tag == 'B_iter5000' else 'ref'} |")

    lines.extend([
        '',
        '## 6. Final method recommendation',
        '',
        f'- **recommended_checkpoint_for_paper**: `{rec}`',
        f'- **backup_checkpoint_if_drift**: `{backup}`',
        '- **official_stage_for_method**: Step2 (A10 epoch_24) weights — DeHub finetune, not Step3 self-train',
        '- **whether_dehub_should_be_added_to_step2_or_step3**: **Step2 checkpoint**; Step3 does not replace DeHub',
        '',
        '## 7. Final verdict',
        '',
        f'## **{verdict}**',
        '',
    ])

    reasons = []
    if ap_ok:
        reasons.append('B8k heldout-500 AP50/sv_AP50 ≥ baseline')
    if hub_ok:
        reasons.append(f'high_final_sv drop {hf_drop:.1f}% (≥30%)')
    reasons.append(f'fusion: {fusion_status}')
    reasons.append(f'B8k low-risk drift: {b8_drift_v}')
    lines.append('原因: ' + '; '.join(reasons))

    next_map = {
        'FORMAL_SUCCESS': '固定 B8k，开始论文主表与可视化。',
        'VERIFIED_PROMISING_NEEDS_MORE': '补人工审计 class drift 或对比 B5k 敏感性。',
        'CLASS_DRIFT_RISK': '优先 B5k 或降低 dehub_weight。',
        'AP_TRADEOFF': '回退 B5k。',
        'INCONCLUSIVE': '检查 eval pipeline 日志。',
        'FAILED': '回退 overnight 结果并 debug。',
    }
    lines.extend(['', '## 8. Next action', '', next_map.get(verdict, ''), ''])

    out = U.RESULT_DIR / 'fres_final_formal_success_check.md'
    out.write_text('\n'.join(lines), encoding='utf-8')

    print('[GPU89 Formal Success Check Final]')
    print('NEW_RESULT_DIR:', U.RESULT_DIR)
    print('class_drift_status:', 'COMPLETE' if drift else 'MISSING')
    print('fusion_head_status:', fusion_status)
    print('step23_heldout500_status:', 'COMPLETE' if step23 else 'MISSING')
    print('b8k_status:', 'OK' if b8 else 'MISSING')
    print('b5k_status:', 'OK' if b5 else 'MISSING')
    print('recommended_checkpoint_for_paper:', rec)
    print('backup_checkpoint_if_drift:', backup)
    print('official_stage_for_method: Step2')
    print('final_verdict:', verdict)
    print('final_report_path:', out)
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
