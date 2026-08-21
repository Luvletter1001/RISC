#!/usr/bin/env python
"""Shared paths and helpers for SV-DeHub-Lite train experiment."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RESULT_MD = REPO / 'resultmd/exp_sv_dehub_lite_train_gpu89'
WORK_ROOT = REPO / 'work_dirs/exp_sv_dehub_lite_train_gpu89'
MECH_RESULT = REPO / 'resultmd/exp_mechanism_sv_attractor_gpu89'
BASE_CONFIG = REPO / 'M_configs/Step2_A10_Large_Pretrain_Stage3/A10_flex_rtm_v3_1_formal.py'
DEBUB_CONFIG = REPO / 'M_configs/experiments/sv_dehub_lite_v1.py'
BASE_CKPT = REPO / 'results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth'
PATCH_BANK = WORK_ROOT / 'hard_negative_bank'
PYTHON = Path('/data/zcy/anaconda3/envs/openrsd/bin/python')

HIGHRISK_TILES = [
    'P0682__1024__553___0',
    'P0148__1024__651___0',
    'P0297__1024__0___0',
    'P1461__1024__3296___4944',
    'P1447__1024__0___1873',
    'P1854__1024__7336___1048',
]
EVAL_ANGLES = [0, 90, 180, 270]
LOWRISK_TILES = [
    'P0132__1024__0___0',
    'P0097__1024__0___208',
    'P1442__1024__0___824',
    'P0312__1024__2383___1756',
    'P0182__1024__824___824',
]

SCHEMA_EVAL = [
    'checkpoint', 'tile_id', 'angle', 'group', 'dense_top1_sv_ratio', 'final_sv_ratio',
    'det_count', 'small_vehicle_count', 'dense_sv_margin', 'dense_entropy',
    'objectness_proxy', 'mean_score', 'top1_class_histogram', 'ap50', 'sv_ap50', 'notes',
]


def ensure_dirs():
    RESULT_MD.mkdir(parents=True, exist_ok=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    PATCH_BANK.mkdir(parents=True, exist_ok=True)


def append_csv(path: Path, row: dict, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, 'NA') for k in fields})


def read_csv(path: Path):
    if not path.exists():
        return []
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def list_checkpoints(work_dir: Path):
    ckpts = []
    for p in sorted(work_dir.glob('*.pth')):
        ckpts.append(p)
    for p in sorted((work_dir / 'checkpoints').glob('*.pth')) if (work_dir / 'checkpoints').exists() else []:
        ckpts.append(p)
    return sorted(set(ckpts), key=lambda x: x.stat().st_mtime)
