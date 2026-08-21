#!/usr/bin/env python
"""Scan WEIGHT_DIR and build inventory + test matrix."""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.exp_official_step123_sv_attractor_eval.common_official_utils import (
    RESULT_DIR, STAGES, SUPPORT_FALLBACK, WEIGHT_DIR, append_csv, ensure_dirs, file_md5,
    git_commit,
)


def infer_stage(path: Path) -> str:
    s = str(path).lower()
    if 'a08' in s or 'step1' in s or 'rtm_v2_base' in s:
        return 'Step1'
    if 'a12' in s or 'step3' in s or 'self_train' in s or 'dota2only' in s:
        return 'Step3'
    if 'a10' in s or 'step2' in s or 'stage3' in s or 'flex_rtm_v3' in s:
        return 'Step2'
    return 'Unknown'


def rank_candidate(path: Path, stage: str) -> int:
    name = path.name.lower()
    score = 0
    if 'weights_only' in name:
        score += 100
    m = re.search(r'epoch[_]?(\d+)', name)
    if m:
        ep = int(m.group(1))
        if stage == 'Step2' and ep == 24:
            score += 50
        if stage == 'Step3' and ep == 24:
            score += 50
        score += ep
    if 'latest' in name:
        score += 30
    if 'best' in name:
        score += 25
    if name.startswith('._'):
        score -= 1000
    return score


def main():
    ensure_dirs()
    rows = []
    for root, _, files in os.walk(WEIGHT_DIR):
        for fn in files:
            if not fn.endswith(('.pth', '.pt', '.ckpt')):
                continue
            if fn.startswith('._'):
                continue
            p = Path(root) / fn
            stage = infer_stage(p)
            ep = ''
            m = re.search(r'epoch[_]?(\d+)', fn)
            if m:
                ep = m.group(1)
            rows.append(dict(
                weight_path=str(p),
                file_name=fn,
                size_bytes=p.stat().st_size,
                modified_time=datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
                md5=file_md5(p),
                inferred_stage=stage,
                inferred_config='',
                epoch=ep,
                is_weights_only='weights_only' in fn.lower(),
                has_optimizer='weights_only' not in fn.lower(),
                candidate_rank=rank_candidate(p, stage),
                reason=f'stage={stage}',
            ))
    rows.sort(key=lambda r: (-int(r['candidate_rank']), r['weight_path']))
    fields = list(rows[0].keys()) if rows else []
    inv = RESULT_DIR / 'ftable_00_weight_inventory.csv'
    with open(inv, 'w', newline='', encoding='utf-8') as f:
        import csv
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    matrix = []
    for key, spec in STAGES.items():
        for role, ckpt in [('primary', spec.primary_ckpt), ('secondary', spec.secondary_ckpt)]:
            if ckpt is None:
                continue
            status = 'READY' if ckpt.exists() else 'CHECKPOINT_MISSING'
            matrix.append(dict(
                stage=key, config_path=str(spec.config), checkpoint_path=str(ckpt),
                support_path=str(SUPPORT_FALLBACK), head_mode='alignment+fusion_if_supported',
                support_mode='visual+text_if_supported', dataset='DOTA1_mechanism+heldout_calib',
                split='highlow+P0148+heldout', status=status,
                note=f'{role}; {spec.note}',
            ))
    mpath = RESULT_DIR / 'ftable_01_official_step_test_matrix.csv'
    mfields = list(matrix[0].keys())
    with open(mpath, 'w', newline='', encoding='utf-8') as f:
        import csv
        w = csv.DictWriter(f, fieldnames=mfields)
        w.writeheader()
        w.writerows(matrix)

    lines = [
        '# Weight Inventory',
        '',
        f'- scanned: `{WEIGHT_DIR}`',
        f'- files: **{len(rows)}**',
        f'- git: {git_commit()}',
        '',
        '## Primary checkpoints (test matrix)',
        '',
        '| stage | role | checkpoint | status |',
        '|---|---|---|---|',
    ]
    for m in matrix:
        lines.append(f"| {m['stage']} | {m['note'].split(';')[0]} | `{Path(m['checkpoint_path']).name}` | {m['status']} |")
    (RESULT_DIR / 'fres_00_weight_inventory.md').write_text('\n'.join(lines), encoding='utf-8')
    print('wrote', inv, mpath)


if __name__ == '__main__':
    main()
