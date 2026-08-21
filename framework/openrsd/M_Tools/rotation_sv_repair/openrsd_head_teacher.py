#!/usr/bin/env python3
"""OpenRSD alignment / fusion / prompt head consensus teacher."""
from __future__ import annotations

from typing import Dict, List, Optional

from M_Tools.rotation_sv_repair.common import CLASSES, COURT_NAMES, SMALL


def head_consensus_label(alignment_top1: str, fusion_top1: str, text_top1: str,
                         image_top1: str, final_top1: str) -> dict:
    court_votes = sum(1 for h in (alignment_top1, fusion_top1, text_top1, image_top1)
                      if h in COURT_NAMES)
    sv_votes = sum(1 for h in (alignment_top1, fusion_top1, text_top1, image_top1)
                   if h == 'small-vehicle')
    consensus = final_top1
    repair_action = 'none'
    unstable_sv = False
    if court_votes >= 2 and final_top1 == 'small-vehicle':
        for h in (alignment_top1, fusion_top1, text_top1, image_top1):
            if h in COURT_NAMES:
                consensus = h
                repair_action = 'court_pseudo_override'
                break
    elif sv_votes == 1 and final_top1 == 'small-vehicle':
        repair_action = 'downweight_sv'
        unstable_sv = True
    elif text_top1 != image_top1 and final_top1 == 'small-vehicle':
        unstable_sv = True
        repair_action = 'mark_unstable_sv'
    return dict(
        alignment_top1=alignment_top1,
        fusion_top1=fusion_top1,
        text_top1=text_top1,
        image_top1=image_top1,
        final_top1=final_top1,
        consensus_label=consensus,
        is_unstable_sv=unstable_sv,
        repair_action=repair_action,
    )


def ensemble_head_row(tile: str, angle: int, heads: Dict[str, str]) -> dict:
    return dict(
        tile=tile, angle=angle,
        alignment_top1=heads.get('alignment', ''),
        fusion_top1=heads.get('fusion', ''),
        text_top1=heads.get('text', ''),
        image_top1=heads.get('image', ''),
        final_top1=heads.get('final', ''),
        **{k: v for k, v in head_consensus_label(
            heads.get('alignment', ''), heads.get('fusion', ''),
            heads.get('text', ''), heads.get('image', ''), heads.get('final', ''),
        ).items() if k not in heads},
    )
