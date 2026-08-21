#!/usr/bin/env python
"""Shared paths for SV-DeHub 3h verification (plan3h)."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path('/data1/zcy/OpenRSD')
RESULT_PARENT = REPO / 'resultmd/exp_sv_dehub_lite_train_gpu89'
VERIFY_DIR = RESULT_PARENT / 'verify_ap_ablation_plan3h_20260521'
WORK_PLAN3H = REPO / 'work_dirs/exp_sv_dehub_lite_train_gpu89/verify_plan3h'
TRAIN_V1 = REPO / 'work_dirs/exp_sv_dehub_lite_train_gpu89/train_v1'
BASE_CKPT = REPO / 'results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth'
FORMAL_CONFIG = REPO / 'M_configs/Step2_A10_Large_Pretrain_Stage3/A10_flex_rtm_v3_1_formal.py'
REPAIR_WD = REPO / 'work_dirs/sv_attractor_repair_gpu89_20260519_161938'
SUPPORT_PKL = REPO / 'data/DOTA1_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl'
PYTHON = Path('/data/zcy/anaconda3/envs/openrsd/bin/python')
OVERNIGHT_EVAL_RAW = RESULT_PARENT / 'ftable_eval_sv_dehub_lite_raw.csv'
PRIOR_VERIFY = RESULT_PARENT / 'verify_ap_ablation_20260521'

OLD_BASELINE_REF = {
    'high': {'dense_top1_sv': 0.531, 'final_sv': 0.654, 'det_count': 534.5},
    'low': {'dense_top1_sv': 0.255, 'final_sv': 0.157, 'det_count': 54.5},
}

CHECKPOINTS = [
    ('patched_baseline_epoch24', BASE_CKPT),
    ('iter_1000', TRAIN_V1 / 'iter_1000.pth'),
    ('iter_2000', TRAIN_V1 / 'iter_2000.pth'),
    ('iter_3000', TRAIN_V1 / 'iter_3000.pth'),
]

ABLATION_VARIANTS = [
    ('dehub_only', REPO / 'M_configs/experiments/sv_dehub_ablation_dehub_only_v1.py',
     WORK_PLAN3H / 'dehub_only'),
    ('hard_negative_only', REPO / 'M_configs/experiments/sv_dehub_ablation_paste_only_v1.py',
     WORK_PLAN3H / 'paste_only'),
    ('head_finetune_only', REPO / 'M_configs/experiments/sv_dehub_ablation_head_finetune_only_v1.py',
     WORK_PLAN3H / 'head_finetune_only'),
    ('lower_weight_combined', REPO / 'M_configs/experiments/sv_dehub_ablation_lower_weight_v1.py',
     WORK_PLAN3H / 'lower_weight_combined'),
]


def ensure_dirs():
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    WORK_PLAN3H.mkdir(parents=True, exist_ok=True)


def load_heldout_stems(n: int = 200) -> list:
    split_path = REPAIR_WD / 'splits/calib_split.json'
    if split_path.exists():
        data = json.loads(split_path.read_text(encoding='utf-8'))
        held = data.get('heldout_calib', [])
        if held:
            return held[:n]
    return []


def rel_diff(new: float, old: float) -> float:
    if old == 0:
        return float('inf') if new != 0 else 0.0
    return abs(new - old) / abs(old)


def consistency_status(rel: float) -> str:
    if rel < 0.05:
        return 'CONSISTENT'
    if rel < 0.15:
        return 'SHIFTED'
    return 'BROKEN'


def overall_baseline_verdict(rows: list) -> str:
    order = {'CONSISTENT': 0, 'SHIFTED': 1, 'BROKEN': 2}
    worst = 'CONSISTENT'
    for r in rows:
        s = r.get('status', 'BROKEN')
        if order.get(s, 2) > order.get(worst, 0):
            worst = s
    return worst
