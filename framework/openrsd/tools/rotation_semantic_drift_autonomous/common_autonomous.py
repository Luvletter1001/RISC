#!/usr/bin/env python
"""Shared paths, models, GPU selection for autonomous rotation drift overnight."""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_DEFAULT = Path('/data1/zcy/OpenRSD')
PYTHON_DEFAULT = Path('/data/zcy/anaconda3/envs/openrsd/bin/python')

PREV_EXP = REPO_DEFAULT / 'resultmd/exp_rotation_semantic_drift_overnight_gpu89_20260521'
PREV_WORK = REPO_DEFAULT / 'work_dirs/exp_next_plan_sv_dehub_step23_overnight'

STEP2_CONFIG = REPO_DEFAULT / 'M_configs/Step2_A10_Large_Pretrain_Stage3/A10_flex_rtm_v3_1_formal.py'
STEP2_CKPT = REPO_DEFAULT / 'results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth'
STEP3_CONFIG = REPO_DEFAULT / 'M_configs/Step3_A12_SelfTrain/A12_flex_rtm_v3_1_self_training_Labelver5.py'
STEP3_CKPT = REPO_DEFAULT / 'results/MMR_AD_A12_flex_rtm_v3_1_self_training_Labelver5/epoch_24.pth'
SUPPORT_PKL = REPO_DEFAULT / 'data/DOTA1_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl'
ANN_DIR = REPO_DEFAULT / 'data/DOTA1_1024_500/ss_train/annfiles'
VIS_ROOT = REPO_DEFAULT / 'vis'
ANGLE_SWEEP = REPO_DEFAULT / 'data/DOTA1_1024_500/angle_sweep_val/realistic'

ORBIT_ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
RENDER_ANGLES = [0, 30, 60, 90, 120, 150]
P0148 = 'P0148__1024__651___0'
HIGHRISK = [
    'P0682__1024__553___0', P0148, 'P0297__1024__0___0',
    'P1461__1024__3296___4944', 'P1447__1024__0___1873', 'P1854__1024__7336___1048',
]
LOWRISK = [
    'P0132__1024__0___0', 'P0097__1024__0___208', 'P1442__1024__0___824',
    'P0312__1024__2383___1756', 'P0182__1024__824___824',
]

HYPOTHESES = [
    ('H1', 'class/support order or mapping error'),
    ('H2', 'small-vehicle embedding / dense margin attractor'),
    ('H3', 'rendering / interpolation / padding artifact'),
    ('H4', 'oblique angle geometric failure'),
    ('H5', 'support embedding intervention fix'),
    ('H6', 'background/context FP'),
    ('H7', 'object-level tennis-to-SV transition'),
    ('H8', 'crop/border/canvas artifact'),
    ('H9', 'scene prior / local-instance prior collapse'),
    ('H10', 'naive Step3 self-training amplifies mechanism hub'),
    ('H11', 'annotation / coordinate / GT matching artifact'),
    ('H12', 'postprocess threshold/NMS/top-k amplification'),
    ('H13', 'B-branch DeHub / B8k suppresses hub'),
]

VERDICTS = [
    'STRONGLY_SUPPORTED', 'PARTIALLY_SUPPORTED', 'WEAKLY_SUPPORTED',
    'INCONCLUSIVE', 'WEAKLY_REJECTED', 'REJECTED', 'BLOCKED',
]

ENV_PREFIX = (
    'env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mplconfig '
    f'PYTHONPATH={REPO_DEFAULT}:{REPO_DEFAULT}/tools'
)


class AutonomousCtx:
    def __init__(
        self,
        repo_root: Path,
        work_dir: Path,
        result_md_dir: Path,
        python: Path = PYTHON_DEFAULT,
        mode: str = 'full',
        smoke: bool = False,
    ):
        self.repo_root = Path(repo_root)
        self.work_dir = Path(work_dir)
        self.result_md_dir = Path(result_md_dir)
        self.python = Path(python)
        self.mode = mode
        self.smoke = smoke or mode == 'smoke'
        self.controller_dir = self.work_dir / 'exp_00_controller'
        self.state_path = self.controller_dir / 'fstate_controller.json'
        self.live_status_path = self.result_md_dir / 'fres_live_status.md'
        self.hypothesis_board_path = self.result_md_dir / 'ftable_hypothesis_board.csv'
        self.log_dir = self.work_dir / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))

    def exp_dir(self, name: str) -> Path:
        d = self.result_md_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def unique_path(self, directory: Path, prefix: str, ext: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        base = directory / f'{prefix}{ext}'
        if not base.exists():
            return base
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        alt = directory / f'{prefix}_{ts}{ext}'
        if not alt.exists():
            return alt
        for i in range(1, 1000):
            p = directory / f'{prefix}_{ts}_{i}{ext}'
            if not p.exists():
                return p
        return alt


def discover_models(repo: Path) -> List[dict]:
    prev_work = repo / 'work_dirs/exp_next_plan_sv_dehub_step23_overnight'
    b5 = prev_work / 'branch_B_fresh_epoch24_to8k/iter_5000.pth'
    if not b5.exists():
        for alt in (4000, 6000):
            p = prev_work / f'branch_B_fresh_epoch24_to8k/iter_{alt}.pth'
            if p.exists():
                b5 = p
                break
    return [
        dict(name='baseline_epoch24', ckpt=STEP2_CKPT, cfg=STEP2_CONFIG, head='alignment'),
        dict(name='B_iter8000', ckpt=prev_work / 'branch_B_fresh_epoch24_to8k/iter_8000.pth',
             cfg=STEP2_CONFIG, head='alignment'),
        dict(name='B_iter5000', ckpt=b5, cfg=STEP2_CONFIG, head='alignment'),
        dict(name='Step3_alignment', ckpt=STEP3_CKPT, cfg=STEP3_CONFIG, head='alignment'),
    ]


def select_gpus(gpu_ids: str) -> List[int]:
    if gpu_ids == 'auto':
        try:
            out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                           '--format=csv,noheader,nounits'], text=True)
            candidates = []
            for line in out.strip().splitlines():
                idx, mem, util = [x.strip() for x in line.split(',')]
                i, m, u = int(idx), float(mem), float(util)
                if m < 5000 and u < 10:
                    candidates.append(i)
            for preferred in (8, 9):
                if preferred in candidates:
                    return [preferred]
            return candidates[:2] if candidates else [8]
        except Exception:
            return [8, 9]
    return [int(x) for x in gpu_ids.split(',')]


def compute_stop_time(run_until: str, min_hours: float) -> Tuple[datetime, str]:
    now = datetime.now()
    try:
        parts = run_until.split(':')
        hour, minute = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        min_stop = now + timedelta(hours=min_hours)
        if target < min_stop:
            target = min_stop
            note = f'run_until extended to min_hours={min_hours}'
        else:
            note = 'run_until ok'
        return target, note
    except Exception as exc:
        target = now + timedelta(hours=min_hours)
        return target, f'time_parse_error:{exc}'


def append_csv(path: Path, row: dict, fields: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, 'NA') for k in fields})


def write_csv(path: Path, rows: List[dict], fields: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, 'NA') for k in fields})


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def parse_hist(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if raw in (None, '', 'NA', '{}'):
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def parse_dota_ann(ann_path: Path) -> List[dict]:
    objs = []
    if not ann_path.exists():
        return objs
    for i, line in enumerate(ann_path.read_text().splitlines()):
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        poly = [float(x) for x in parts[:8]]
        cls = parts[8]
        cx = sum(poly[0::2]) / 4
        cy = sum(poly[1::2]) / 4
        objs.append(dict(
            gt_object_id=f'{ann_path.stem}_{i}_{cls}',
            gt_class=cls, gt_bbox=poly, gt_cx=cx, gt_cy=cy))
    return objs


def image_path(tile: str, angle: int, repo: Path = REPO_DEFAULT) -> Optional[Path]:
    vd = repo / 'vis' / tile / 'dataset' / 'images'
    for ext in ('.jpg', '.png'):
        p = vd / f'{tile}_rot{angle:03d}{ext}'
        if p.exists():
            return p
    asv = repo / 'data/DOTA1_1024_500/angle_sweep_val/realistic' / f'angle_{angle:03d}' / 'images'
    for ext in ('.png', '.jpg'):
        p = asv / f'{tile}{ext}'
        if p.exists():
            return p
    return None


def tiles_with_rotated_images(tile_ids: List[str], repo: Path = REPO_DEFAULT) -> List[str]:
    return [t for t in tile_ids if image_path(t, 0, repo) is not None]


def find_tennis_ann_tiles(repo: Path, max_n: int = 20) -> List[dict]:
    rows = []
    if not ANN_DIR.exists():
        return rows
    for ann in sorted(ANN_DIR.glob('P*.txt')):
        objs = parse_dota_ann(ann)
        tennis = [o for o in objs if o['gt_class'] == 'tennis-court']
        if not tennis:
            continue
        tile = ann.stem
        has_rot = image_path(tile, 0, repo) is not None
        proxy = not has_rot
        rows.append(dict(
            tile_id=tile,
            ann_path=str(ann),
            tennis_gt_count=len(tennis),
            has_rotated_images=has_rot,
            co_located=has_rot,
            proxy=proxy,
            selected_reason='tennis_ann_with_rotation' if has_rot else 'tennis_ann_proxy_only',
        ))
        if len(rows) >= max_n:
            break
    rows.sort(key=lambda r: (not r['co_located'], -r['tennis_gt_count']))
    return rows[:max_n]


def init_hypothesis_board(ctx: AutonomousCtx):
    if ctx.hypothesis_board_path.exists():
        return
    rows = []
    defaults = {
        'H10': 'STRONGLY_SUPPORTED', 'H13': 'STRONGLY_SUPPORTED',
        'H2': 'PARTIALLY_SUPPORTED', 'H5': 'PARTIALLY_SUPPORTED',
        'H6': 'PARTIALLY_SUPPORTED', 'H9': 'PARTIALLY_SUPPORTED',
        'H3': 'WEAKLY_SUPPORTED', 'H8': 'WEAKLY_SUPPORTED',
        'H4': 'INCONCLUSIVE', 'H7': 'INCONCLUSIVE',
        'H1': 'INCONCLUSIVE', 'H11': 'INCONCLUSIVE', 'H12': 'PARTIALLY_SUPPORTED',
    }
    for hid, desc in HYPOTHESES:
        rows.append(dict(
            hypothesis_id=hid, description=desc,
            verdict=defaults.get(hid, 'INCONCLUSIVE'),
            evidence='prior_overnight_summary',
            last_updated=datetime.now().isoformat(),
            next_needed='autonomous_queue',
        ))
    write_csv(ctx.hypothesis_board_path, rows,
              ['hypothesis_id', 'description', 'verdict', 'evidence', 'last_updated', 'next_needed'])


def load_state(ctx: AutonomousCtx) -> dict:
    if ctx.state_path.exists():
        return json.loads(ctx.state_path.read_text(encoding='utf-8'))
    return {}


def save_state(ctx: AutonomousCtx, state: dict):
    ctx.controller_dir.mkdir(parents=True, exist_ok=True)
    state['current_time'] = datetime.now().isoformat()
    ctx.state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')


def snapshot_env(ctx: AutonomousCtx, gpu: int) -> dict:
    snap = dict(gpu=gpu, time=datetime.now().isoformat())
    try:
        snap['nvidia_smi'] = subprocess.check_output(['nvidia-smi'], text=True, stderr=subprocess.STDOUT)[:4000]
    except Exception as exc:
        snap['nvidia_smi'] = str(exc)
    try:
        snap['git_head'] = subprocess.check_output(
            ['git', '-C', str(ctx.repo_root), 'rev-parse', 'HEAD'], text=True).strip()
        snap['git_status'] = subprocess.check_output(
            ['git', '-C', str(ctx.repo_root), 'status', '--short'], text=True)[:2000]
    except Exception as exc:
        snap['git_head'] = str(exc)
    try:
        snap['hostname'] = subprocess.check_output(['hostname'], text=True).strip()
    except Exception:
        snap['hostname'] = 'unknown'
    snap['python'] = str(ctx.python)
    snap['conda'] = os.environ.get('CONDA_DEFAULT_ENV', 'NA')
    path = ctx.controller_dir / f"fjson_env_snapshot_{datetime.now().strftime('%H%M%S')}.json"
    path.write_text(json.dumps(snap, indent=2), encoding='utf-8')
    return snap


def git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return 'unknown'
