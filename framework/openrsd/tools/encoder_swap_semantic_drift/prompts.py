#!/usr/bin/env python3
"""Prompt families for DOTA / remote sensing text encoders."""
from __future__ import annotations

from typing import Dict, List

from tools.encoder_swap_semantic_drift.common import DOTA_CLASSES, OPENRSD_CLASSES

PROMPT_FAMILIES = {
    'F0_raw': lambda c: c.replace('-', ' '),
    'F1_aerial': lambda c: f'aerial image of a {c.replace("-", " ")}',
    'F2_remote_sensing': lambda c: f'remote sensing image of a {c.replace("-", " ")}',
    'F3_satellite': lambda c: f'satellite image of a {c.replace("-", " ")}',
    'F4_overhead': lambda c: f'overhead view of a {c.replace("-", " ")}',
    'F5_rotated': lambda c: f'rotated {c.replace("-", " ")} in an aerial image',
}

# Short names for tables
FAMILY_SHORT = {
    'F0_raw': 'raw',
    'F1_aerial': 'aerial',
    'F2_remote_sensing': 'remote_sensing',
    'F3_satellite': 'satellite',
    'F4_overhead': 'overhead',
    'F5_rotated': 'rotated',
}


def build_prompt_bank(family_key: str) -> Dict[str, List[str]]:
    fn = PROMPT_FAMILIES[family_key]
    bank: Dict[str, List[str]] = {}
    for oc in OPENRSD_CLASSES:
        dota_name = oc.replace('-', ' ')
        if dota_name == 'baseball diamond':
            dota_name = 'baseball-diamond'
        bank[oc] = [fn(oc)]
    return bank


def all_prompt_texts(family_key: str) -> List[str]:
    return [t for texts in build_prompt_bank(family_key).values() for t in texts]
