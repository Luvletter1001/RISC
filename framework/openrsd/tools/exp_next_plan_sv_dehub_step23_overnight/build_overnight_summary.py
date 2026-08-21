#!/usr/bin/env python
"""Build fres_final_overnight_summary.md from CSV artifacts."""
from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.exp_next_plan_sv_dehub_step23_overnight import common_overnight_utils as U


def read_csv(path: Path):
    if not path.exists():
        return []
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    U.ensure_dirs()
    ap500 = read_csv(U.RESULT_DIR / 'ftable_24_heldout500_ap_summary.csv')
    step31 = read_csv(U.RESULT_DIR / 'ftable_31_step23_head_support_highlow_raw.csv')
    smoke = read_csv(U.RESULT_DIR / 'ftable_30_head_switch_smoke.csv')

    fusion_ok = False
    if len(smoke) >= 2:
        a, b = smoke[0], smoke[1]
        try:
            fusion_ok = abs(float(a['final_sv_ratio']) - float(b['final_sv_ratio'])) > 1e-6
        except (KeyError, ValueError):
            pass

    verdict = 'INCONCLUSIVE'
    if ap500:
        for r in ap500:
            try:
                ap = float(r.get('ap50', 'nan'))
                if ap > 0.7:
                    verdict = 'VERIFIED_PROMISING_NEEDS_MORE'
            except ValueError:
                pass

    lines = [
        '# Next Plan Overnight Summary: SV-DeHub + Official Step2/3',
        '',
        f'Generated: {datetime.now().isoformat()}',
        '',
        '## 1. Run timeline',
        'See `ftable_00_monitor.csv`, `log_pipeline_overnight.txt`.',
        '',
        '## 2. Main conclusions',
        f'- fusion_head_smoke_verified: {fusion_ok}',
        f'- heldout500_rows: {len(ap500)}',
        f'- step23_highlow_rows: {len(step31)}',
        '',
        '## 3. SV-DeHub heldout-500',
        '',
        '| checkpoint | AP50 | sv_AP50 |',
        '|---|---:|---:|',
    ]
    for r in ap500[:20]:
        lines.append(f"| {r.get('checkpoint','?')} | {r.get('ap50','NA')} | {r.get('sv_ap50','NA')} |")

    lines.extend([
        '',
        '## 7. Final verdict',
        '',
        f'**{verdict}**',
        '',
        '## 8. Next action',
        'Review heldout-500 and class drift tables; confirm A/B 8k checkpoints.',
    ])
    out = U.RESULT_DIR / 'fres_final_overnight_summary.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print('[Plan Overnight Final]')
    print('NEW_RESULT_DIR:', U.RESULT_DIR)
    print('final_verdict:', verdict)
    print('final_report_path:', out)
    print('fusion_head_status:', 'VERIFIED' if fusion_ok else 'FUSION_NOT_VERIFIED')
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
