#!/usr/bin/env python3
"""Ensemble OVD teacher-student: orbit + OpenRSD heads + CastDet queue + prompts."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from M_Tools.rotation_sv_repair.castdet_style_teacher import build_pseudo_label, update_queue_from_regions
from M_Tools.rotation_sv_repair.common import SuiteContext
from M_Tools.rotation_sv_repair.dynamic_pseudo_label_queue import DynamicPseudoLabelQueue
from M_Tools.rotation_sv_repair.openrsd_head_teacher import head_consensus_label
from M_Tools.rotation_sv_repair.orbit_distillation import calibrated_orbit_teacher


def fuse_teachers(region: dict, heads: dict, orbit_label: str, prompt_label: str) -> dict:
    hc = head_consensus_label(
        heads.get('alignment', ''), heads.get('fusion', ''),
        heads.get('text', ''), heads.get('image', ''), heads.get('final', ''),
    )
    pl = build_pseudo_label(
        region,
        heads.get('final', 'small-vehicle'),
        prompt_label or heads.get('text', ''),
        orbit_label or heads.get('final', ''),
        sv_penalty=0.2 if hc['is_unstable_sv'] else 0.0,
    )
    if hc['repair_action'] == 'court_pseudo_override':
        pl['class_name'] = hc['consensus_label']
    teachers = ['orbit', 'openrsd_heads', 'castdet_style', 'lae_prompt']
    return dict(**pl, **hc, teacher_components=';'.join(teachers))


def ensemble_verdict(rows: List[dict]) -> str:
    if not rows:
        return 'NO_DATA'
    sv_drop = sum(float(r.get('sv_ratio_drop', 0) or 0) for r in rows) / len(rows)
    preserve = sum(float(r.get('true_sv_preserve', 1) or 1) for r in rows) / len(rows)
    if sv_drop > 0.05 and preserve > 0.9:
        return 'GOOD'
    if sv_drop > 0.02:
        return 'PROMISING'
    return 'BAD'
