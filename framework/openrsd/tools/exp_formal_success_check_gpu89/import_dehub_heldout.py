#!/usr/bin/env python
"""Task D: import overnight heldout-500 for dehub candidates."""
from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.exp_formal_success_check_gpu89 import common_formal_utils as U


def main():
    U.ensure_dirs()
    prev = U.PREV_RESULT / 'ftable_24_heldout500_ap_raw.csv'
    out = U.RESULT_DIR / 'ftable_50_dehub_candidate_heldout500_summary.csv'
    if not prev.exists():
        print('missing prev heldout csv')
        return 1
    rows = [r for r in csv.DictReader(prev.open()) if r.get('heldout_n') == '500']
    seen = {}
    for r in rows:
        ck = r['checkpoint']
        if 'branch_A' in ck and '8000' in ck:
            tag = 'A_iter8000'
        elif 'branch_B' in ck and '8000' in ck:
            tag = 'B_iter8000'
        elif 'branch_B' in ck and '5000' in ck:
            tag = 'B_iter5000'
        elif 'epoch_24' in ck:
            tag = 'baseline_epoch24'
        else:
            tag = Path(ck).name
        seen[tag] = r
    fields = ['name', 'checkpoint', 'heldout_n', 'ap50', 'sv_ap50', 'detection_total',
              'final_sv_ratio', 'source', 'imported_at']
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for tag, r in sorted(seen.items()):
            w.writerow(dict(
                name=tag, checkpoint=r['checkpoint'], heldout_n=r['heldout_n'],
                ap50=r['ap50'], sv_ap50=r['sv_ap50'],
                detection_total=r['detection_total'], final_sv_ratio=r['final_sv_ratio'],
                source='overnight_import', imported_at=datetime.now().isoformat(),
            ))
    md = ['# DeHub candidate heldout-500 (imported)', '']
    for tag, r in sorted(seen.items()):
        md.append(f"- **{tag}**: AP50={float(r['ap50']):.3f} sv_AP50={float(r['sv_ap50']):.3f} "
                  f"final_sv={float(r['final_sv_ratio']):.3f}")
    (U.RESULT_DIR / 'fres_50_dehub_candidate_heldout500.md').write_text('\n'.join(md), encoding='utf-8')
    print('imported', len(seen), 'rows')
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
