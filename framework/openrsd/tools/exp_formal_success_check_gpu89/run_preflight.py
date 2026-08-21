#!/usr/bin/env python
"""Preflight for formal success check."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.exp_formal_success_check_gpu89 import common_formal_utils as U


def main():
    U.ensure_dirs()
    checks = {}
    checks['prev_result'] = U.PREV_RESULT.exists()
    checks['prev_work'] = U.PREV_WORK.exists()
    checks['prev_summary'] = (U.PREV_RESULT / 'fres_final_overnight_summary.md').exists()
    inv = U.checkpoint_inventory()
    for r in inv:
        checks[f"ckpt_{r['name']}"] = r['exists']
    checks['heldout_split'] = (U.REPAIR_WD / 'splits/calib_split.json').exists()
    checks['heldout_n'] = len(U.load_heldout_stems(500))
    checks['highrisk_n'] = len(U.HIGHRISK)
    checks['lowrisk_n'] = len(U.LOWRISK)
    checks['support_pkl'] = U.SUPPORT_PKL.exists()

    flex = REPO / 'M_AD/models/detectors/Flex_Rtmdet_v3_1_formal.py'
    checks['flex_detector'] = flex.exists()
    if flex.exists():
        txt = flex.read_text(encoding='utf-8')
        checks['prompt_predict'] = 'def prompt_predict' in txt
        checks['val_using_aux_branch'] = 'if val_using_aux' in txt

    U.git_patch_out(U.RESULT_DIR / 'code_diff_formal_check_start.patch')

    import csv
    with open(U.RESULT_DIR / 'ftable_02_checkpoint_selection.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=U.SCHEMA_CKPT, extrasaction='ignore')
        w.writeheader()
        w.writerows(inv)

    meta = dict(checks=checks, inventory=inv, time=datetime.now().isoformat())
    (U.RESULT_DIR / 'fmeta_01_preflight.json').write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')

    lines = ['# Preflight — FORMAL_SUCCESS check', '', f'- time: {meta["time"]}', '']
    for k, v in checks.items():
        lines.append(f'- {k}: **{v}**')
    lines.extend(['', '## Checkpoints', ''])
    for r in inv:
        st = 'OK' if r['exists'] else 'MISSING'
        lines.append(f"- {r['name']}: {st} — `{r['checkpoint_path']}`")
    (U.RESULT_DIR / 'fres_01_preflight.md').write_text('\n'.join(lines), encoding='utf-8')
    print('preflight done', checks)
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
