#!/usr/bin/env python
"""Shared paths and stage definitions for official Step1/2/3 SV attractor eval."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path('/data1/zcy/OpenRSD')
RESULT_DIR = REPO / 'resultmd/exp_official_step123_sv_attractor_eval'
WORK_DIR = REPO / 'work_dirs/exp_official_step123_sv_attractor_eval'
WEIGHT_DIR = REPO / 'results'
PYTHON = Path('/data/zcy/anaconda3/envs/openrsd/bin/python')
MECH_HIGHRISK = REPO / 'resultmd/exp_mechanism_sv_attractor_gpu89/ftable_01_attractor_highrisk_tiles.csv'
REPAIR_WD = REPO / 'work_dirs/sv_attractor_repair_gpu89_20260519_161938'
SUPPORT_FALLBACK = REPO / 'data/DOTA1_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl'

HIGHRISK_DEFAULT = [
    'P0682__1024__553___0', 'P0148__1024__651___0', 'P0297__1024__0___0',
    'P1461__1024__3296___4944', 'P1447__1024__0___1873', 'P1854__1024__7336___1048',
]
LOWRISK_DEFAULT = [
    'P0132__1024__0___0', 'P0097__1024__0___208', 'P1442__1024__0___824',
    'P0312__1024__2383___1756', 'P0182__1024__824___824',
]
MECH_ANGLES = [0, 90, 180, 270]
P0148_ANGLES = list(range(0, 360, 30))
P0148_TILE = 'P0148__1024__651___0'

SCHEMA_MECH_RAW = [
    'stage', 'checkpoint', 'config_path', 'support_mode', 'head_mode', 'val_using_aux',
    'tile_id', 'angle', 'group', 'det_count', 'small_vehicle_count', 'final_sv_ratio',
    'dense_top1_sv_ratio', 'dense_sv_mean_logit', 'dense_sv_margin_vs_runnerup',
    'dense_entropy', 'objectness_mean', 'objectness_p95', 'mean_score',
    'score_p50', 'score_p75', 'score_p90', 'score_p95', 'top1_class_histogram',
    'non_sv_det_count', 'tennis_court_count', 'large_vehicle_count',
    'basketball_court_count', 'harbor_count', 'notes', 'timestamp',
]

SCHEMA_CMD = [
    'task', 'stage', 'command', 'gpu', 'start_time', 'end_time', 'return_code',
    'log_path', 'output_path', 'notes',
]


@dataclass
class StageSpec:
    stage: str
    config: Path
    primary_ckpt: Path
    secondary_ckpt: Optional[Path]
    detector: str
    dense_hook_ok: bool
    note: str = ''


STAGES: Dict[str, StageSpec] = {
    'Step1': StageSpec(
        'Step1',
        REPO / 'M_configs/Step1_A08_Large_Pretrain/A08_e_rtm_v2_base.py',
        REPO / 'results/MMR_AD_A08_e_rtm_v2_base_recheck/epoch_36.pth',
        None,
        'E_Rtmdet_v2',
        False,
        'Large pretrain A08; OpenRotated head v2; no prompt_extract_feats dense hook',
    ),
    'Step2': StageSpec(
        'Step2',
        REPO / 'M_configs/Step2_A10_Large_Pretrain_Stage3/A10_flex_rtm_v3_1_formal.py',
        REPO / 'results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth',
        REPO / 'results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24.pth',
        'Flex_Rtmdet_v3_1_formal',
        True,
        'Stage3 large pretrain; val_using_aux=False=Align, True=Fusion',
    ),
    'Step3': StageSpec(
        'Step3',
        REPO / 'M_configs/Step3_A12_SelfTrain/A12_flex_rtm_v3_1_self_training_Labelver5.py',
        REPO / 'results/MMR_AD_A12_flex_rtm_v3_1_self_training_Labelver5/epoch_24.pth',
        REPO / 'results/MMR_AD_A12_flex_rtm_v3_1_DOTA2only_ss_train/epoch_12.pth',
        'Flex_Rtmdet_v3_1_formal',
        True,
        'Self-train Labelver5 primary; DOTA2only_ss_train epoch_12 secondary',
    ),
}


def ensure_dirs():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ['git', '-C', str(REPO), 'rev-parse', '--short', 'HEAD'], text=True).strip()
    except Exception:
        return 'unknown'


def file_md5(path: Path, max_bytes: int = 64 * 1024 * 1024) -> str:
    if not path.exists():
        return ''
    h = hashlib.md5()
    with open(path, 'rb') as f:
        h.update(f.read(max_bytes))
    return h.hexdigest()


def append_csv(path: Path, row: dict, fields: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, 'NA') for k in fields})


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_highrisk_tiles() -> List[str]:
    if MECH_HIGHRISK.exists():
        rows = read_csv(MECH_HIGHRISK)
        tiles = [r['tile_id'] for r in rows if r.get('tile_id')]
        if tiles:
            return tiles[:10]
    return HIGHRISK_DEFAULT


def load_lowrisk_tiles() -> List[str]:
    return LOWRISK_DEFAULT


def load_heldout_stems(n: int) -> List[str]:
    split_path = REPAIR_WD / 'splits/calib_split.json'
    if split_path.exists():
        data = json.loads(split_path.read_text(encoding='utf-8'))
        held = data.get('heldout_calib', [])
        if held:
            return held[:n]
    return []


def head_label(val_using_aux: bool) -> str:
    return 'fusion' if val_using_aux else 'alignment'


def eval_combos_for_stage(spec: StageSpec, full_matrix: bool = False) -> List[Tuple[str, bool]]:
    """Return list of (support_mode, val_using_aux)."""
    combos = [('visual', False)]
    if full_matrix and spec.dense_hook_ok:
        combos += [
            ('visual', True),
            ('text', False),
            ('text', True),
        ]
    elif spec.dense_hook_ok:
        combos += [('text', False)]
    return combos
