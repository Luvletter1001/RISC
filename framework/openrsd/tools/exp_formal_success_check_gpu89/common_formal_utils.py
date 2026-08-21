#!/usr/bin/env python
"""Paths and helpers for FORMAL_SUCCESS check (GPU8/9 only)."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path('/data1/zcy/OpenRSD')
RESULT_DIR = REPO / 'resultmd/exp_formal_success_check_gpu89'
WORK_DIR = REPO / 'work_dirs/exp_formal_success_check_gpu89'
TOOL_DIR = REPO / 'tools/exp_formal_success_check_gpu89'
PREV_RESULT = REPO / 'resultmd/exp_next_plan_sv_dehub_step23_overnight'
PREV_WORK = REPO / 'work_dirs/exp_next_plan_sv_dehub_step23_overnight'

PYTHON = Path('/data/zcy/anaconda3/envs/openrsd/bin/python')
REPAIR_WD = REPO / 'work_dirs/sv_attractor_repair_gpu89_20260519_161938'
SUPPORT_PKL = REPO / 'data/DOTA1_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl'

BASE_CKPT = REPO / 'results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth'
FORMAL_CONFIG = REPO / 'M_configs/Step2_A10_Large_Pretrain_Stage3/A10_flex_rtm_v3_1_formal.py'
STEP2_CONFIG = FORMAL_CONFIG
STEP2_CKPT = BASE_CKPT
STEP3_CONFIG = REPO / 'M_configs/Step3_A12_SelfTrain/A12_flex_rtm_v3_1_self_training_Labelver5.py'
STEP3_CKPT = REPO / 'results/MMR_AD_A12_flex_rtm_v3_1_self_training_Labelver5/epoch_24.pth'

A8K = PREV_WORK / 'branch_A_iter1000_to8k/iter_8000.pth'
B8K = PREV_WORK / 'branch_B_fresh_epoch24_to8k/iter_8000.pth'
B5K = PREV_WORK / 'branch_B_fresh_epoch24_to8k/iter_5000.pth'

HIGHRISK = [
    'P0682__1024__553___0', 'P0148__1024__651___0', 'P0297__1024__0___0',
    'P1461__1024__3296___4944', 'P1447__1024__0___1873', 'P1854__1024__7336___1048',
]
LOWRISK = [
    'P0132__1024__0___0', 'P0097__1024__0___208', 'P1442__1024__0___824',
    'P0312__1024__2383___1756', 'P0182__1024__824___824',
]
ANGLES = [0, 90, 180, 270]

SCHEMA_CKPT = ['name', 'source', 'checkpoint_path', 'config_path', 'role', 'exists',
               'substitute_for', 'reason']
SCHEMA_DRIFT = [
    'source_type', 'name', 'checkpoint', 'config_path', 'support', 'head', 'group',
    'final_sv', 'dense_sv', 'det_count', 'sv_det_count', 'non_sv_det_count',
    'non_sv_inflation_vs_baseline', 'top1_class', 'top1_class_count', 'top3_classes',
    'tennis_court_count', 'large_vehicle_count', 'basketball_court_count',
    'harbor_count', 'plane_count', 'class_hist_kl_vs_baseline', 'class_hist_l1_vs_baseline',
    'mean_score', 'score_p50', 'score_p75', 'score_p90', 'score_p95',
    'high_final_sv_delta', 'high_dense_sv_delta', 'low_det_count_delta', 'low_non_sv_delta',
    'drift_verdict', 'notes', 'timestamp',
]
SCHEMA_AP = [
    'stage', 'name', 'head_mode', 'val_using_aux', 'heldout_n', 'ap50', 'sv_ap50',
    'detection_total', 'final_sv_ratio', 'sv_det_count', 'non_sv_det_count',
    'class_histogram', 'class_hist_kl_vs_step2_align', 'actually_used_head', 'notes', 'timestamp',
]


def ensure_dirs():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)


def append_csv(path: Path, row: dict, fields: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, 'NA') for k in fields})


def read_csv(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_heldout_stems(n: int = 500) -> list:
    sp = REPAIR_WD / 'splits/calib_split.json'
    if sp.exists():
        data = json.loads(sp.read_text(encoding='utf-8'))
        held = data.get('heldout_calib', [])
        if held:
            return held[:n]
    return []


def git_patch_out(path: Path):
    try:
        diff = subprocess.check_output(
            ['git', '-C', str(REPO), 'diff', 'tools/exp_formal_success_check_gpu89'],
            text=True, stderr=subprocess.DEVNULL)
        path.write_text(diff or '# no diff\n', encoding='utf-8')
    except Exception:
        path.write_text('# git diff failed\n', encoding='utf-8')


def parse_hist(hist_str) -> dict:
    if isinstance(hist_str, dict):
        return hist_str
    try:
        return json.loads(hist_str) if hist_str else {}
    except json.JSONDecodeError:
        return {}


def hist_to_probs(hist: dict, classes: list) -> list:
    total = sum(hist.values()) or 1
    return [hist.get(c, 0) / total for c in classes]


def kl_div(p: list, q: list, eps: float = 1e-12) -> float:
    import math
    s = 0.0
    for pi, qi in zip(p, q):
        pi, qi = max(pi, eps), max(qi, eps)
        s += pi * math.log(pi / qi)
    return s


def l1_dist(p: list, q: list) -> float:
    return sum(abs(a - b) for a, b in zip(p, q))


def checkpoint_inventory() -> list:
    items = [
        ('baseline_epoch24', 'official', BASE_CKPT, FORMAL_CONFIG, 'baseline'),
        ('A_iter8000', 'overnight_A', A8K, FORMAL_CONFIG, 'dehub_resume_8k'),
        ('B_iter8000', 'overnight_B', B8K, FORMAL_CONFIG, 'dehub_fresh_8k_paper'),
        ('B_iter5000', 'overnight_B', B5K, FORMAL_CONFIG, 'dehub_fresh_5k_backup'),
        ('Step2_official_epoch24', 'official', STEP2_CKPT, STEP2_CONFIG, 'official_step2'),
        ('Step3_official_epoch24', 'official', STEP3_CKPT, STEP3_CONFIG, 'official_step3'),
    ]
    rows = []
    for name, source, ckpt, cfg, role in items:
        ex = ckpt.exists()
        rows.append(dict(
            name=name, source=source, checkpoint_path=str(ckpt), config_path=str(cfg),
            role=role, exists=ex, substitute_for='', reason='' if ex else 'MISSING',
        ))
    return rows
