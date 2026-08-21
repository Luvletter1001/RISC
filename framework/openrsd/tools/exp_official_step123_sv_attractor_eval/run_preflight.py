#!/usr/bin/env python
"""Preflight checks for official step123 eval."""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.exp_official_step123_sv_attractor_eval.common_official_utils import (
    RESULT_DIR, STAGES, SUPPORT_FALLBACK, ensure_dirs, git_commit, load_highrisk_tiles,
    load_heldout_stems, load_lowrisk_tiles,
)
from tools.exp_mechanism_sv_attractor_gpu89 import common_attractor_utils as mech


def main():
    ensure_dirs()
    lines = [
        '# Preflight — Official Step1/2/3 SV Attractor Eval',
        '',
        f'- time: {datetime.now().isoformat()}',
        f'- git: {git_commit()}',
        '',
        '## Environment',
        '',
    ]
    try:
        import torch
        lines.append(f'- torch: {torch.__version__}')
        lines.append(f'- cuda available: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            lines.append(f'- gpu count: {torch.cuda.device_count()}')
    except Exception as exc:
        lines.append(f'- torch import fail: {exc}')

    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                       '--format=csv,noheader'], text=True)
        lines.append('- nvidia-smi:\n```\n' + out.strip() + '\n```')
    except Exception as exc:
        lines.append(f'- nvidia-smi fail: {exc}')

    lines += ['', '## Checkpoints', '']
    for k, spec in STAGES.items():
        ok = spec.primary_ckpt.exists()
        lines.append(f'- **{k}** primary `{spec.primary_ckpt.name}`: {"OK" if ok else "MISSING"}')
        if spec.secondary_ckpt:
            ok2 = spec.secondary_ckpt.exists()
            lines.append(f'  - secondary `{spec.secondary_ckpt.name}`: {"OK" if ok2 else "MISSING"}')

    lines += ['', '## Support pkl', '']
    lines.append(f'- fallback: `{SUPPORT_FALLBACK}` exists={SUPPORT_FALLBACK.exists()}')

    lines += ['', '## Tiles', '']
    for t in load_highrisk_tiles()[:3]:
        d = mech.vis_tile_dir(REPO, t)
        lines.append(f'- high `{t}`: {d is not None}')
    for t in load_lowrisk_tiles()[:2]:
        d = mech.vis_tile_dir(REPO, t)
        lines.append(f'- low `{t}`: {d is not None}')

    held = load_heldout_stems(500)
    lines.append(f'- heldout_calib pool: **{len(held)}** stems')

    lines += ['', '## Config load smoke (Step2 only)', '']
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = '9'
        spec = STAGES['Step2']
        ctx = mech.MechContext(config=spec.config, checkpoint=spec.primary_ckpt, gpu=9)
        mech.build_model(ctx)
        lines.append('- Step2 build_model: **OK**')
    except Exception as exc:
        lines.append(f'- Step2 build_model: **FAIL** {exc}')

    lines += ['', '## Step1 dense hook', '']
    lines.append('- Step1 (E_Rtmdet_v2): **no prompt_extract_feats** → dense metrics NA; detection/AP only if load succeeds')

    (RESULT_DIR / 'fres_02_preflight.md').write_text('\n'.join(lines), encoding='utf-8')
    print('wrote fres_02_preflight.md')


if __name__ == '__main__':
    main()
