#!/usr/bin/env python3
"""Sample per-class GT crops from DOTA ss_train for image encoder prototypes."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

from tools.encoder_swap_semantic_drift.common import OPENRSD_CLASSES, REPO_DEFAULT

SS_IMAGES = REPO_DEFAULT / 'data/DOTA1_1024_500/ss_train/images'
SS_ANN = REPO_DEFAULT / 'data/DOTA1_1024_500/ss_train/annfiles'


def _parse_dota_ann_line(line: str):
    parts = line.strip().split()
    if len(parts) < 9:
        return None
    coords = list(map(float, parts[:8]))
    name = parts[8].lower()
    return name, coords


def sample_class_crops(repo: Path, per_class: int = 20, smoke: bool = False) -> Dict[str, List[Path]]:
    n = 3 if smoke else per_class
    out: Dict[str, List[Path]] = {c: [] for c in OPENRSD_CLASSES}
    if not SS_ANN.is_dir():
        return out
    ann_files = sorted(SS_ANN.glob('*.txt'))
    random.shuffle(ann_files)
    for ann_path in ann_files:
        stem = ann_path.stem
        img_path = SS_IMAGES / f'{stem}.png'
        if not img_path.exists():
            img_path = SS_IMAGES / f'{stem}.jpg'
        if not img_path.exists():
            continue
        try:
            lines = ann_path.read_text(encoding='utf-8', errors='ignore').splitlines()
        except Exception:
            continue
        for line in lines:
            parsed = _parse_dota_ann_line(line)
            if not parsed:
                continue
            cls, _ = parsed
            oc = cls.replace(' ', '-')
            if oc not in out:
                continue
            if len(out[oc]) < n:
                out[oc].append(img_path)
        if all(len(v) >= n for v in out.values()):
            break
    return out
