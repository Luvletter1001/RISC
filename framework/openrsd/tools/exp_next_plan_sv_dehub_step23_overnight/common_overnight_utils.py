#!/usr/bin/env python
"""Paths and schemas for overnight SV-DeHub + Step2/3 plan."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

REPO = Path('/data1/zcy/OpenRSD')
RESULT_DIR = REPO / 'resultmd/exp_next_plan_sv_dehub_step23_overnight'
WORK_DIR = REPO / 'work_dirs/exp_next_plan_sv_dehub_step23_overnight'
TOOL_DIR = REPO / 'tools/exp_next_plan_sv_dehub_step23_overnight'
PYTHON = Path('/data/zcy/anaconda3/envs/openrsd/bin/python')

INIT_CKPT = REPO / 'results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth'
DEBUB_CONFIG = REPO / 'M_configs/experiments/sv_dehub_lite_v1.py'
LOWER_WEIGHT_CONFIG = REPO / 'M_configs/experiments/sv_dehub_ablation_lower_weight_v1.py'
FORMAL_CONFIG = REPO / 'M_configs/Step2_A10_Large_Pretrain_Stage3/A10_flex_rtm_v3_1_formal.py'
STEP3_CONFIG = REPO / 'M_configs/Step3_A12_SelfTrain/A12_flex_rtm_v3_1_self_training_Labelver5.py'
STEP3_CKPT = REPO / 'results/MMR_AD_A12_flex_rtm_v3_1_self_training_Labelver5/epoch_24.pth'
STEP1_CONFIG = REPO / 'M_configs/Step1_A08_Large_Pretrain/A08_e_rtm_v2_base.py'
STEP1_CKPT = REPO / 'results/MMR_AD_A08_e_rtm_v2_base_recheck/epoch_36.pth'

OLD_TRAIN_V1 = REPO / 'work_dirs/exp_sv_dehub_lite_train_gpu89/train_v1'
ITER_1000 = OLD_TRAIN_V1 / 'iter_1000.pth'
REPAIR_WD = REPO / 'work_dirs/sv_attractor_repair_gpu89_20260519_161938'
SUPPORT_PKL = REPO / 'data/DOTA1_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl'

BRANCHES = {
    'A': dict(work=WORK_DIR / 'branch_A_iter1000_to8k', resume=ITER_1000, max_iters=8000, gpu=8),
    'B': dict(work=WORK_DIR / 'branch_B_fresh_epoch24_to8k', resume=None, max_iters=8000, gpu=6),
    'C': dict(work=WORK_DIR / 'branch_C_lowerw_fresh', resume=None, max_iters=8000, gpu=6,
              config=LOWER_WEIGHT_CONFIG),
}

SCHEMA_MONITOR = [
    'check_time', 'gpu_status', 'running_tasks', 'completed_tasks', 'failed_tasks',
    'latest_train_iter_A', 'latest_train_iter_B', 'latest_train_iter_C',
    'latest_ckpt_A', 'latest_ckpt_B', 'latest_ckpt_C',
    'eval_queue_size', 'latest_eval_result', 'detected_error', 'action_taken', 'next_action',
]

SCHEMA_TRAIN_CURVE = [
    'branch', 'iter', 'global_iter', 'fresh_or_resume', 'init_checkpoint',
    'total_loss', 'cls_loss', 'bbox_loss', 'dehub_loss', 'dehub_ratio', 'paste_count',
    'lr', 'grad_norm_cls', 'checkpoint_path', 'status',
]

SCHEMA_CKPT_INV = [
    'branch', 'checkpoint_name', 'checkpoint_path', 'source_init', 'train_iter',
    'global_iter', 'exists', 'size_bytes', 'modified_time', 'md5',
]

SCHEMA_HEAD_SMOKE = [
    'stage', 'tile_id', 'angle', 'head_mode', 'val_using_aux', 'det_count',
    'final_sv_ratio', 'mean_score', 'top1_class', 'logit_source', 'notes',
]


def ensure_dirs():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    TOOL_DIR.mkdir(parents=True, exist_ok=True)


def append_csv(path: Path, row: dict, fields: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, 'NA') for k in fields})


def load_heldout_stems(n: int = 500) -> list:
    sp = REPAIR_WD / 'splits/calib_split.json'
    if sp.exists():
        data = json.loads(sp.read_text(encoding='utf-8'))
        held = data.get('heldout_calib', [])
        if held:
            return held[:n]
    return []


def file_md5(path: Path) -> str:
    if not path.exists():
        return ''
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()
