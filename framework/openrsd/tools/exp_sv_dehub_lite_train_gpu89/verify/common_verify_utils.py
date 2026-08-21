#!/usr/bin/env python
"""Shared paths for SV-DeHub verification / ablation."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path('/data1/zcy/OpenRSD')
RESULT_PARENT = REPO / 'resultmd/exp_sv_dehub_lite_train_gpu89'
VERIFY_DIR = RESULT_PARENT / 'verify_ap_ablation_20260521'
WORK_VERIFY = REPO / 'work_dirs/exp_sv_dehub_lite_train_gpu89/verify_ablation'
TRAIN_V1 = REPO / 'work_dirs/exp_sv_dehub_lite_train_gpu89/train_v1'
BASE_CKPT = REPO / 'results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth'
FORMAL_CONFIG = REPO / 'M_configs/Step2_A10_Large_Pretrain_Stage3/A10_flex_rtm_v3_1_formal.py'
DEBUB_CONFIG = REPO / 'M_configs/experiments/sv_dehub_lite_v1.py'
REPAIR_WD = REPO / 'work_dirs/sv_attractor_repair_gpu89_20260519_161938'
SUPPORT_PKL = REPO / 'data/DOTA1_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl'
PYTHON = Path('/data/zcy/anaconda3/envs/openrsd/bin/python')

CHECKPOINTS = [
    ('baseline_epoch24', BASE_CKPT),
    ('iter_1000', TRAIN_V1 / 'iter_1000.pth'),
    ('iter_2000', TRAIN_V1 / 'iter_2000.pth'),
    ('iter_3000', TRAIN_V1 / 'iter_3000.pth'),
]

OVERNIGHT_EVAL_RAW = RESULT_PARENT / 'ftable_eval_sv_dehub_lite_raw.csv'


def ensure_dirs():
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    WORK_VERIFY.mkdir(parents=True, exist_ok=True)


def load_heldout_stems(n: int = 200) -> list:
    split_path = REPAIR_WD / 'splits/calib_split.json'
    if split_path.exists():
        data = json.loads(split_path.read_text(encoding='utf-8'))
        held = data.get('heldout_calib', [])
        if held:
            return held[:n]
    return []
