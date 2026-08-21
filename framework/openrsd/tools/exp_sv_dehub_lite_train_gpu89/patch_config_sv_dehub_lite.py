#!/usr/bin/env python
"""Central paths and hyperparameters for SV-DeHub-Lite v1 experiment."""
from __future__ import annotations

from pathlib import Path

REPO = Path('/data1/zcy/OpenRSD')
RESULT_DIR = REPO / 'resultmd/exp_sv_dehub_lite_train_gpu89'
TOOL_DIR = REPO / 'tools/exp_sv_dehub_lite_train_gpu89'
WORK_DIR = REPO / 'work_dirs/exp_sv_dehub_lite_train_gpu89'
TRAIN_WORK_DIR = WORK_DIR / 'train_v1'
MECH_DIR = REPO / 'resultmd/exp_mechanism_sv_attractor_gpu89'

PYTHON = Path('/data/zcy/anaconda3/envs/openrsd/bin/python')
DEBUB_CONFIG = REPO / 'M_configs/experiments/sv_dehub_lite_v1.py'
BASE_CKPT = REPO / 'results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth'
PATCH_BANK_DIR = WORK_DIR / 'hard_negative_bank'
PATCH_BANK_CSV = RESULT_DIR / 'ftable_hard_negative_patch_bank.csv'
MECH_HIGHRISK_CSV = MECH_DIR / 'ftable_01_attractor_highrisk_tiles.csv'

# Training
MAX_ITERS_DEFAULT = 3000
SAVE_EVERY_DEFAULT = 1000
DEHUB_MARGIN = 0.05
DEHUB_LOSS_WEIGHT = 0.05
HARD_NEGATIVE_PASTE_PROB = 0.3
TRAIN_GPU = 8
EVAL_GPU = 9

# Eval tiles
HIGHRISK_TILES = [
    'P0682__1024__553___0',
    'P0148__1024__651___0',
    'P0297__1024__0___0',
    'P1461__1024__3296___4944',
    'P1447__1024__0___1873',
    'P1854__1024__7336___1048',
]
LOWRISK_TILES = [
    'P0132__1024__0___0',
    'P0097__1024__0___208',
    'P1442__1024__0___824',
    'P0312__1024__2383___1756',
    'P0182__1024__824___824',
]
EVAL_ANGLES = [0, 90, 180, 270]

# Logs / tables
LOG_TRAIN = RESULT_DIR / 'log_gpu8_train_sv_dehub_lite_v1.txt'
LOG_EVAL = RESULT_DIR / 'log_gpu9_eval_sv_dehub_lite_v1.txt'
LOG_HOURLY_LOOP = RESULT_DIR / 'log_hourly_monitor_loop.txt'
CURVE_CSV = RESULT_DIR / 'ftable_train_loss_curve.csv'
HOURLY_CSV = RESULT_DIR / 'ftable_hourly_monitor.csv'
HOURLY_MD = RESULT_DIR / 'fres_hourly_monitor.md'
EVAL_RAW = RESULT_DIR / 'ftable_eval_sv_dehub_lite_raw.csv'
EVAL_SUMMARY = RESULT_DIR / 'ftable_eval_sv_dehub_lite_summary.csv'
EVAL_COMPARE = RESULT_DIR / 'ftable_eval_highrisk_lowrisk_compare.csv'
EVAL_AP = RESULT_DIR / 'ftable_eval_ap_smoke.csv'
FINAL_REPORT = RESULT_DIR / 'fres_sv_dehub_lite_train_summary.md'

ERROR_TAGS = [
    'PATH_MISSING', 'CHECKPOINT_MISSING', 'CONFIG_ERROR', 'DATASET_ERROR',
    'ANNOTATION_ERROR', 'IMPORT_ERROR', 'CUDA_OOM', 'CUDA_ILLEGAL_MEMORY',
    'NAN_LOSS', 'EMPTY_DETECTION', 'CSV_SCHEMA_ERROR', 'LOSS_NOT_BACKPROP',
    'HARD_NEGATIVE_AUG_BUG', 'EVAL_BUG', 'LOGIC_BUG', 'UNKNOWN',
]

VERDICTS = [
    'PROMISING', 'DENSE_ONLY_SUCCESS', 'AP_TRADEOFF', 'INEFFECTIVE', 'FAILED_DEBUG',
]

HOURLY_FIELDS = [
    'check_time', 'gpu8_train_process_alive', 'gpu9_eval_process_alive',
    'gpu8_memory_used', 'gpu9_memory_used', 'latest_train_iter', 'latest_checkpoint',
    'train_log_updated', 'eval_log_updated', 'train_csv_rows', 'eval_csv_rows',
    'latest_total_loss', 'latest_cls_loss', 'latest_dehub_loss', 'latest_dehub_ratio',
    'latest_eval_checkpoint', 'latest_highrisk_final_sv', 'latest_lowrisk_final_sv',
    'detected_error', 'action_taken', 'next_action',
]
