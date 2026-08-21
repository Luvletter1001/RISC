#!/usr/bin/env python
"""Preflight: paths, sv index, checkpoint, config."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.exp_sv_dehub_lite_train_gpu89 import common_dehub_utils as U
from tools.rotation_diagnostics import probe_rotated_stage_outputs as probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-dir', type=Path, default=REPO)
    ap.add_argument('--result-dir', type=Path, default=U.RESULT_MD)
    args = ap.parse_args()
    U.ensure_dirs()
    sv_idx = probe.DOTA1_CLASSES.index('small-vehicle')
    lines = [
        '# Preflight SV-DeHub-Lite v1',
        '',
        f'- PROJECT: `{args.project_dir}`',
        f'- RESULT_DIR: `{args.result_dir}`',
        f'- BASE_CONFIG exists: {U.BASE_CONFIG.exists()}',
        f'- DEBUB_CONFIG exists: {U.DEBUB_CONFIG.exists()}',
        f'- BASE_CKPT exists: {U.BASE_CKPT.exists()} ({U.BASE_CKPT})',
        f'- small-vehicle DOTA1 index: **{sv_idx}** (name=`small-vehicle`)',
        f'- HIGHRISK tiles: {len(U.HIGHRISK_TILES)}',
        f'- PATCH_BANK dir: `{U.PATCH_BANK}`',
        '',
        '## Next steps',
        '1. build_hard_negative_patch_bank.py',
        '2. run_gradient_audit (after config load)',
        '3. run_train_sv_dehub_lite_gpu8.py',
        '4. run_eval_sv_dehub_lite_gpu9.py',
    ]
    out = args.result_dir / 'fres_00_preflight_sv_dehub_lite.md'
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    meta = dict(
        sv_class_index=sv_idx,
        base_config=str(U.BASE_CONFIG),
        dehub_config=str(U.DEBUB_CONFIG),
        base_checkpoint=str(U.BASE_CKPT),
        highrisk_tiles=U.HIGHRISK_TILES,
    )
    (args.result_dir / 'fmeta_train_config_sv_dehub_lite_v1.json').write_text(
        __import__('json').dumps(meta, indent=2), encoding='utf-8')
    print('wrote', out)


if __name__ == '__main__':
    main()
