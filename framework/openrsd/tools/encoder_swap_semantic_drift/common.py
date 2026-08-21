#!/usr/bin/env python3
"""Shared context for encoder-swap overnight experiments."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_DEFAULT = Path('/data1/zcy/OpenRSD')
PYTHON_DEFAULT = Path('/data/zcy/anaconda3/envs/openrsd/bin/python')
PRETRAINED_DEFAULT = REPO_DEFAULT / 'pretrained'
WORK_DEFAULT = REPO_DEFAULT / 'work_dirs/exp_encoder_swap_semantic_drift_20260525_overnight'
RESULT_DEFAULT = REPO_DEFAULT / 'resultmd/exp_encoder_swap_semantic_drift_20260525_overnight'

DOTA_CLASSES = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
    'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',
    'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter',
]
# OpenRSD detector order (hyphenated)
OPENRSD_CLASSES = [
    'baseball-diamond', 'basketball-court', 'bridge', 'ground-track-field',
    'harbor', 'helicopter', 'large-vehicle', 'plane', 'roundabout', 'ship',
    'small-vehicle', 'soccer-ball-field', 'storage-tank', 'swimming-pool', 'tennis-court',
]
SMALL = OPENRSD_CLASSES.index('small-vehicle')
LV = OPENRSD_CLASSES.index('large-vehicle')
TENNIS = OPENRSD_CLASSES.index('tennis-court')

P0148 = 'P0148__1024__651___0'
HIGHRISK = [
    'P0682__1024__553___0', P0148, 'P0297__1024__0___0',
    'P1461__1024__3296___4944', 'P1447__1024__0___1873', 'P1854__1024__7336___1048',
]
LOWRISK = [
    'P0132__1024__0___0', 'P0097__1024__0___208', 'P1442__1024__0___824',
    'P0312__1024__2383___1756', 'P0182__1024__824___824',
]
ANGLES_FULL = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
ANGLES_SMOKE = [0, 90]
ANGLES_HEATMAP = [0, 30, 90, 150]

STEP2_CONFIG = REPO_DEFAULT / 'M_configs/Step2_A10_Large_Pretrain_Stage3/A10_flex_rtm_v3_1_formal.py'
STEP2_CKPT = REPO_DEFAULT / 'results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth'
STEP3_CONFIG = REPO_DEFAULT / 'M_configs/Step3_A12_SelfTrain/A12_flex_rtm_v3_1_self_training_Labelver5.py'
STEP3_CKPT = REPO_DEFAULT / 'results/MMR_AD_A12_flex_rtm_v3_1_self_training_Labelver5/epoch_24.pth'
B8K_CKPT = REPO_DEFAULT / 'work_dirs/exp_next_plan_sv_dehub_step23_overnight/branch_B_fresh_epoch24_to8k/iter_8000.pth'
ORIG_SUPPORT = REPO_DEFAULT / 'data/DOTA1_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl'
TEXT_TARGET_DIM = 768
VIS_TARGET_DIM = 1024
OPENRSD_EMBED_DIM = 256

MODEL_SPECS = {
    'baseline': dict(config=STEP2_CONFIG, checkpoint=STEP2_CKPT, label='A10 epoch24'),
    'B8k': dict(config=STEP2_CONFIG, checkpoint=B8K_CKPT, label='SV-DeHub B8k'),
    'Step3': dict(config=STEP3_CONFIG, checkpoint=STEP3_CKPT, label='A12 self-train epoch24'),
}

ENV_PREFIX = (
    'env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mplconfig '
    f'PYTHONPATH={REPO_DEFAULT}:{REPO_DEFAULT}/tools'
)


@dataclass
class ExpResult:
    status: str
    exp_id: str
    work_subdir: str
    rows: List[dict] = field(default_factory=list)
    outputs: Dict[str, str] = field(default_factory=dict)
    interpretation: str = ''
    error: str = ''
    next_guesses: List[str] = field(default_factory=list)
    hypothesis_updates: Dict[str, dict] = field(default_factory=dict)


@dataclass
class EncoderSwapCtx:
    repo_root: Path
    work_dir: Path
    result_md_dir: Path
    pretrained_root: Path
    python: Path = PYTHON_DEFAULT
    mode: str = 'full'
    force: bool = False
    smoke: bool = False
    batch_size: int = 12
    ap_batch_size: int = 8
    gpu_ids: List[str] = field(default_factory=lambda: ['8'])

    def __post_init__(self):
        self.repo_root = Path(self.repo_root)
        self.work_dir = Path(self.work_dir)
        self.result_md_dir = Path(self.result_md_dir)
        self.pretrained_root = Path(self.pretrained_root)
        self.controller_dir = self.work_dir / 'exp_00_controller'
        self.cache_dir = self.work_dir / 'cache_embeddings'
        self.support_swap_dir = self.work_dir / 'support_swaps'
        self.log_dir = self.work_dir / 'logs'
        for d in (self.controller_dir, self.cache_dir, self.support_swap_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.state_path = self.controller_dir / 'fstate_controller.json'
        self.live_status_path = self.result_md_dir / 'fres_live_status.md'
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))
        if str(self.repo_root / 'tools') not in sys.path:
            sys.path.insert(0, str(self.repo_root / 'tools'))

    @property
    def angles(self) -> List[int]:
        return ANGLES_SMOKE if self.smoke else ANGLES_FULL

    def exp_work(self, name: str) -> Path:
        d = self.work_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def exp_result(self, name: str) -> Path:
        d = self.result_md_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d


def bind_gpu(gpu: str) -> str:
    """Pin one physical GPU and return the visible torch device name."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda:0'
    except Exception:
        pass
    return 'cpu'


def select_gpus(gpu_ids: str) -> List[str]:
    if gpu_ids.strip().lower() != 'auto':
        return [g.strip() for g in gpu_ids.split(',') if g.strip()]
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
             '--format=csv,noheader,nounits'],
            text=True, stderr=subprocess.DEVNULL,
        )
        stats = []
        for line in out.strip().splitlines():
            idx, mem, util = [x.strip() for x in line.split(',')]
            stats.append((int(idx), int(mem), int(util)))
        preferred = [s for s in stats if s[0] in (8, 9)]
        rest = [s for s in stats if s[0] not in (8, 9)]
        pool = sorted(preferred, key=lambda x: (x[2], x[1])) + sorted(rest, key=lambda x: (x[2], x[1]))
        pick = [str(pool[0][0])]
        if len(pool) > 1 and pool[0][1] < 20000:
            pick.append(str(pool[1][0]))
        return pick[:2]
    except Exception:
        return ['8']


def compute_stop_time(run_until: str, min_hours: float) -> Tuple[datetime, str]:
    now = datetime.now()
    parts = run_until.split(':')
    hour, minute = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    min_stop = now + timedelta(hours=min_hours)
    if min_stop > target:
        target = min_stop
        note = f'min_hours={min_hours} extends past {run_until}'
    else:
        note = f'run_until={run_until}'
    return target, note


def git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return 'unknown'


def snapshot_env(repo: Path, log_path: Path) -> dict:
    lines = [
        f'date: {datetime.now().isoformat()}',
        f'hostname: {os.uname().nodename}',
        f'git: {git_commit(repo)}',
        f'python: {sys.executable}',
    ]
    try:
        lines.append(subprocess.check_output(['nvidia-smi'], text=True, stderr=subprocess.DEVNULL)[:4000])
    except Exception:
        pass
    log_path.write_text('\n'.join(lines), encoding='utf-8')
    return dict(hostname=os.uname().nodename, git=git_commit(repo))


def write_csv(path: Path, rows: List[dict], fields: Optional[Sequence[str]] = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fields and rows:
        fields = list(rows[0].keys())
    fields = list(fields or [])
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})


def read_csv(path: Path) -> List[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_state(ctx: EncoderSwapCtx) -> dict:
    if ctx.state_path.exists():
        return json.loads(ctx.state_path.read_text(encoding='utf-8'))
    return {}


def save_state(ctx: EncoderSwapCtx, state: dict):
    state['updated_at'] = datetime.now().isoformat()
    ctx.state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')


def cosine_matrix(emb: np.ndarray) -> np.ndarray:
    x = emb.astype(np.float64)
    x = x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
    return x @ x.T


def hubness_score(emb: np.ndarray, idx: int) -> float:
    cm = cosine_matrix(emb)
    others = [i for i in range(len(emb)) if i != idx]
    return float(np.mean([cm[idx, j] for j in others]))


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)
