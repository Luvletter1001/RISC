#!/usr/bin/env python3
"""LAE-DINO-style dynamic vocabulary and visual-guided prompt families."""
from __future__ import annotations

from typing import Dict, List, Tuple

PROMPT_FAMILIES: Dict[str, Dict[str, List[str]]] = {
    'F0_raw_dota': {
        'tennis-court': ['tennis court'],
        'baseball-diamond': ['baseball diamond'],
        'soccer-ball-field': ['soccer ball field'],
        'small-vehicle': ['small vehicle'],
    },
    'F1_aerial_nouns': {
        'tennis-court': ['aerial image of a tennis court'],
        'baseball-diamond': ['aerial image of a baseball diamond'],
        'soccer-ball-field': ['aerial image of a soccer ball field'],
        'small-vehicle': ['aerial image of a small vehicle'],
    },
    'F2_rotation_aware': {
        'tennis-court': ['rotated tennis court in an aerial image'],
        'baseball-diamond': ['rotated baseball diamond in an aerial image'],
        'soccer-ball-field': ['rotated soccer ball field in an aerial image'],
        'small-vehicle': ['rotated small vehicle in an aerial image'],
    },
    'F3_context_disambiguated': {
        'tennis-court': ['sports court with white lines in aerial imagery'],
        'baseball-diamond': ['baseball field with diamond shape in aerial imagery'],
        'soccer-ball-field': ['soccer field with rectangular field markings in aerial imagery'],
        'small-vehicle': ['small vehicle on road or parking lot in aerial imagery'],
    },
    'F4_negative_aware': {
        'tennis-court': ['tennis court, not a small vehicle'],
        'baseball-diamond': ['baseball diamond, not a small vehicle'],
        'soccer-ball-field': ['soccer field, not a small vehicle'],
        'small-vehicle': ['small vehicle, not sports field markings'],
    },
}

DYNAMIC_VOCAB_AUX = [
    'sports court', 'court marking', 'field line', 'road marking', 'parking lot',
    'rooftop texture', 'background texture',
    'hard negative small vehicle false positive',
]

SV_POSITIVE_PROMPTS = [
    'small vehicle on road', 'small vehicle in parking area',
]
SV_NEGATIVE_PROMPTS = [
    'not sports court lines', 'not field markings', 'not rooftop background',
]


def all_prompt_rows() -> List[dict]:
    rows = []
    for fam, classes in PROMPT_FAMILIES.items():
        for cls, prompts in classes.items():
            for p in prompts:
                rows.append(dict(class_name=cls, prompt=p, prompt_family=fam))
    for w in DYNAMIC_VOCAB_AUX:
        rows.append(dict(class_name='_aux_', prompt=w, prompt_family='dynamic_vocab'))
    return rows


def select_prompt_by_margin(scores: Dict[str, float]) -> str:
    return max(scores, key=scores.get) if scores else ''



def prompt_ensemble_scores(scores: Dict[str, List[float]], mode: str = 'robust_mean') -> Dict[str, float]:
    """Aggregate per-class prompt scores.

    Singleton prompt families are exactly equivalent to the original score; this
    is the required sanity check for prompt ensembles that only enable the base
    prompt.
    """
    out: Dict[str, float] = {}
    for cls, vals in scores.items():
        clean = [float(v) for v in vals]
        if not clean:
            out[cls] = 0.0
        elif mode == 'max':
            out[cls] = max(clean)
        elif mode == 'mean':
            out[cls] = sum(clean) / len(clean)
        elif mode == 'robust_mean':
            if len(clean) <= 2:
                out[cls] = sum(clean) / len(clean)
            else:
                ordered = sorted(clean)
                trimmed = ordered[1:-1]
                out[cls] = sum(trimmed) / len(trimmed)
        else:
            raise ValueError(f'unknown prompt ensemble mode: {mode}')
    return out
