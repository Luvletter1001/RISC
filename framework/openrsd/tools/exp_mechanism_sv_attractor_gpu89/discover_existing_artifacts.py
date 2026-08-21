#!/usr/bin/env python
"""Discover paths and write preflight + evidence graph markdown."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.exp_mechanism_sv_attractor_gpu89 import common_attractor_utils as U
from tools.verify_sv_attractor_gpu89 import common as verify

EVIDENCE_DIRS = [
    'resultmd/exp_full_experiment_report/fres_full_experiment_report.md',
    'resultmd/exp_rotation_stage_probe_P0148',
    'resultmd/exp_rotation_causal_probe_P0148_gpu89',
    'resultmd/exp_rotation_overnight_gpu89',
    'resultmd/exp_rotation_final_causal_proof_P0148_gpu89',
    'resultmd/exp_verify_sv_attractor_gpu89',
    'resultmd/exp_sv_attractor_repair_full_20260519',
    'resultmd/exp_sv_attractor_repair_gpu89',
]

WORK_DIRS = [
    'work_dirs/rotation_stage_probe_P0148_full',
    'work_dirs/rotation_stage_probe_P0148_full_physgpu8_9',
    'work_dirs/rotation_causal_probe_P0148_gpu89_20260518_164044',
    'work_dirs/rotation_final_causal_proof_P0148_gpu89',
    'work_dirs/verify_sv_attractor_gpu89_20260519_113039',
    'work_dirs/sv_attractor_repair_gpu89_20260519_161938',
]


def gpu_check(indices=(8, 9)):
    lines = []
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,name,memory.free,utilization.gpu',
             '--format=csv,noheader'], text=True)
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if int(parts[0]) in indices:
                lines.append(line)
    except Exception as exc:
        lines.append(f'nvidia-smi failed: {exc}')
    return lines


def find_work_dirs(pattern: str) -> list:
    wd = REPO / 'work_dirs'
    return sorted(str(p) for p in wd.glob(pattern) if p.is_dir())


def write_preflight(out: Path):
    cfg = REPO / verify.DEFAULT_CONFIG
    ckpt = REPO / verify.DEFAULT_CHECKPOINT
    sp, sp_note = verify.resolve_support_from_config(cfg)
    asv = REPO / 'data/DOTA1_1024_500/angle_sweep_val/realistic/angle_000'
    ss = REPO / 'data/DOTA1_1024_500/ss_train'
    r1 = U.R1_CKPT
    r3 = U.R3_CKPT
    vis_n = len(U.list_vis_tiles(REPO))
    lines = [
        '# Preflight — exp_mechanism_sv_attractor_gpu89',
        '',
        f'- generated: preflight scan',
        f'- git_commit: `{verify.git_commit(REPO)}`',
        f'- python: `{sys.executable}`',
        f'- package_versions: `{json.dumps(verify.package_versions())}`',
        f'- CUDA_VISIBLE_DEVICES: `{os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")}`',
        '',
        '## GPU 8 / 9',
        '',
    ]
    for g in gpu_check():
        lines.append(f'- `{g}`')
    lines += [
        '',
        '## Paths',
        '',
        f'| item | path | exists |',
        f'|---|---|:---:|',
        f'| A10 config | `{cfg}` | {"✅" if cfg.exists() else "❌"} |',
        f'| epoch_24 ckpt | `{ckpt}` | {"✅" if ckpt.exists() else "❌"} |',
        f'| support pkl | `{sp}` ({sp_note}) | {"✅" if sp and Path(sp).exists() else "❌"} |',
        f'| DOTA1 ss_train | `{ss}` | {"✅" if ss.is_dir() else "❌"} |',
        f'| angle_sweep_val | `{asv}` | {"✅" if asv.is_dir() else "❌"} |',
        f'| vis tiles (count) | `{REPO}/vis` | {vis_n} tiles |',
        f'| best_R1 ckpt | `{r1}` | {"✅" if r1.exists() else "❌"} |',
        f'| best_R3 ckpt | `{r3}` | {"✅" if r3.exists() else "❌"} |',
        f'| repair progress | `{U.REPAIR_WD}/progress.json` | '
        f'{"✅" if (U.REPAIR_WD/"progress.json").exists() else "❌"} |',
        '',
        '## Reusable scripts',
        '',
        '- `tools/verify_sv_attractor_gpu89/` — dense, intervention, bg_gt',
        '- `tools/sv_attractor_repair_gpu89/` — R1/R3 repair forward',
        '- `tools/rotation_overnight_gpu89/` — decode_dense, post_summary',
        '- `tools/rotation_diagnostics/probe_rotated_stage_outputs.py` — dataloader',
        '',
        '## work_dirs (discovered)',
        '',
    ]
    for pat in ['rotation_stage_probe*', 'rotation_causal*', 'rotation_final*',
                'verify_sv*', 'sv_attractor_repair*']:
        found = find_work_dirs(pat)
        for p in found[:5]:
            lines.append(f'- `{p}`')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('wrote', out)


def write_evidence_graph(out: Path):
    body = """# Existing Evidence Graph — SV Attractor Mechanism Study

## 1. 已经确认的结论

- P0148 存在 **dense-logit 级** small-vehicle attractor（dense_top1_sv 常 >0.85，verify mean ~0.79–0.87）。
- **Embedding 干预**（zero_sv / swap）可强烈改变 final_sv（zero → ~0）。
- **NMS/postprocess 是放大器**：dense 已有偏置，NMS 后 final_sv 仍高但可低于 dense（final proof: dense ~0.89 → NMS ~0.69）。
- **R1 class-wise calibration** 在 P0148 与 heldout 上降低 final_sv（~0.75→~0.44）且 AP50 基本保持。
- **Cross-tile**：P0148 mean final_sv ~0.97 vs cross-tile ~0.34（overnight）；P0148 不是唯一高 ratio tile，但属极端尾部。
- **R2B visual negative prototype** 当前版本 **失败**；**R3 α=1.0** 导致检测塌缩。

## 2. 已经削弱/否定的假设

- **H5 padding/border 主因**：border-mask proxy 几乎不改变 ratio（causal probe）。
- **H7「P0148 GT 多导致观感洪水」**：auto GT proxy 在 zero embedding 后 ratio→0，非纯 GT 数量主因。
- **text vs visual support 分叉**：两者 ratio 几乎相同（Δ~1e-4），非 support 模态问题。
- **R2B 负原型**：四组 K 均未降 P0148 sv。

## 3. 仍未解释的问题

- attractor 是 **tile-specific**、**angle-specific** 还是 **embedding/support-global**？
- dense SV 偏置来自 **objectness** 还是 **class margin / hubness**？
- high-risk tile 是否共享 **背景纹理 shortcut**？
- R1 是 **全局类先验校正** 还是 **掩盖 background→SV shortcut**？
- 是否存在可预警 high-risk 的 **diagnostic 指标**？

## 4. 本轮实验只针对的未解释问题

| 问题 | 实验 |
|---|---|
| Q1 tile vs angle vs embedding | Exp1 atlas + Exp4 variance |
| Q2 margin vs objectness | Exp2 decomposition |
| Q3 P0148 是否异常 | Exp1 atlas 排名 |
| Q4 R1 机理 | Exp2 + Exp6 minimal repair |
| Q5 risk predictor | Exp5 |
| H2 background | Exp3 counterfactual |

## 5. 本轮明确不重复的旧工作

- ❌ P0148 baseline vs R1 单点修复评估（已有 full repair）
- ❌ 72° stage probe 全量重跑
- ❌ R2B / R3 α 网格扫描
- ❌ heldout 500 张 full AP 重训
- ✅ 允许：新 tile×angle atlas、embedding/background 反事实、方差分解、R1 分量消融

## 6. 源报告索引

"""
    for rel in EVIDENCE_DIRS:
        p = REPO / rel
        tag = '✅' if p.exists() else '❌'
        body += f'- {tag} `{rel}`\n'
    body += '\n## 7. work_dirs 索引\n\n'
    for rel in WORK_DIRS:
        p = REPO / rel
        if p.exists():
            body += f'- ✅ `{rel}`\n'
        else:
            alts = find_work_dirs(Path(rel).name.split('_')[0] + '*')
            for a in alts[:2]:
                body += f'- ⚠️ `{rel}` missing; alt `{a}`\n'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding='utf-8')
    print('wrote', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight-only', action='store_true')
    ap.add_argument('--evidence-only', action='store_true')
    args = ap.parse_args()
    U.RESULT_MD.mkdir(parents=True, exist_ok=True)
    if not args.evidence_only:
        write_preflight(U.RESULT_MD / 'fres_00_preflight.md')
    if not args.preflight_only:
        write_evidence_graph(U.RESULT_MD / 'fres_00_existing_evidence_graph.md')


if __name__ == '__main__':
    main()
