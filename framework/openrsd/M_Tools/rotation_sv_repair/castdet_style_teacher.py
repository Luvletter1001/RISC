#!/usr/bin/env python3
"""CastDet-style CLIP-activated teacher scoring for pseudo labels."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from M_Tools.rotation_sv_repair.common import CLASSES, SMALL
from M_Tools.rotation_sv_repair.dynamic_pseudo_label_queue import DynamicPseudoLabelQueue


def teacher_score(det_score: float, vlm_score: float, rotation_consensus: float,
                  sv_penalty: float = 0.0) -> float:
    return det_score + 0.5 * vlm_score + 0.3 * rotation_consensus - sv_penalty


def build_pseudo_label(region: dict, detector_top1: str, vlm_top1: str,
                       orbit_top1: str, sv_penalty: float) -> dict:
    scores = {
        'detector': CLASSES.index(detector_top1) if detector_top1 in CLASSES else -1,
        'vlm': CLASSES.index(vlm_top1) if vlm_top1 in CLASSES else -1,
        'orbit': CLASSES.index(orbit_top1) if orbit_top1 in CLASSES else -1,
    }
    final_cls = detector_top1
    override_sv = False
    if detector_top1 == 'small-vehicle' and vlm_top1 in ('tennis-court', 'baseball-diamond', 'soccer-ball-field'):
        final_cls = vlm_top1
        override_sv = True
    conf = teacher_score(
        float(region.get('det_score', 0)),
        float(region.get('vlm_score', 0)),
        float(region.get('orbit_consensus', 0)),
        sv_penalty,
    )
    return dict(
        class_name=final_cls,
        confidence=conf,
        detector_top1=detector_top1,
        vlm_top1=vlm_top1,
        orbit_top1=orbit_top1,
        override_small_vehicle=override_sv,
        unstable_sv=(detector_top1 == 'small-vehicle' and vlm_top1 != orbit_top1),
        is_sv_hard_negative=region.get('is_hard_negative_sv_fp', False),
    )


def update_queue_from_regions(queue: DynamicPseudoLabelQueue, regions: List[dict]):
    for r in regions:
        pl = build_pseudo_label(
            r,
            r.get('detector_top1', 'small-vehicle'),
            r.get('vlm_top1', r.get('detector_top1', '')),
            r.get('orbit_top1', r.get('detector_top1', '')),
            float(r.get('sv_penalty', 0)),
        )
        queue.add(pl)
