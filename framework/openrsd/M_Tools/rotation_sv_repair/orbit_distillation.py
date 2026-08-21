#!/usr/bin/env python3
"""Rotation-orbit consensus teachers and pseudo-label generation."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from M_Tools.rotation_sv_repair.common import ANGLES_FULL, SMALL


def map_box_to_reference(box: np.ndarray, from_angle: int, to_angle: int, img_size: int = 1024) -> np.ndarray:
    """Map axis-aligned xyxy box between rotation orbits (0/90/180/270 exact; others approximate)."""
    b = box.copy().astype(float)
    da = (to_angle - from_angle) % 360
    if da == 0:
        return b
    cx, cy = img_size / 2, img_size / 2
    x1, y1, x2, y2 = b
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    rad = np.deg2rad(da)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s], [s, c]])
    rot = []
    for p in corners:
        v = p - [cx, cy]
        pr = R @ v + [cx, cy]
        rot.append(pr)
    rot = np.array(rot)
    return np.array([rot[:, 0].min(), rot[:, 1].min(), rot[:, 0].max(), rot[:, 1].max()])


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter + 1e-6
    return inter / union


def naive_orbit_merge(predictions: List[List[dict]]) -> List[dict]:
    out = []
    for plist in predictions:
        out.extend(plist)
    return out


def view_consensus_teacher(predictions_by_angle: Dict[int, List[dict]],
                          iou_thr: float = 0.3, min_views: int = 2) -> List[dict]:
    ref_angle = 0
    ref_boxes = predictions_by_angle.get(ref_angle, [])
    if not ref_boxes:
        for a in sorted(predictions_by_angle):
            if predictions_by_angle[a]:
                ref_angle = a
                ref_boxes = predictions_by_angle[a]
                break
    consensus = []
    for rb in ref_boxes:
        box = np.array(rb['bbox'])
        cls = rb.get('class_id', rb.get('label', 0))
        matches = 1
        scores = [float(rb.get('score', 0))]
        for ang, plist in predictions_by_angle.items():
            if ang == ref_angle:
                continue
            for pb in plist:
                pb_box = map_box_to_reference(np.array(pb['bbox']), ang, ref_angle)
                if iou_xyxy(box, pb_box) >= iou_thr and pb.get('class_id', pb.get('label')) == cls:
                    matches += 1
                    scores.append(float(pb.get('score', 0)))
                    break
        if matches >= min_views:
            consensus.append(dict(
                bbox=box.tolist(), class_id=cls, score=float(np.mean(scores)),
                consensus_views=matches, teacher='view_consensus'))
    return consensus


def calibrated_orbit_teacher(predictions_by_angle: Dict[int, List[dict]],
                           sv_idx: int = SMALL) -> List[dict]:
    base = view_consensus_teacher(predictions_by_angle, iou_thr=0.3, min_views=2)
    out = []
    for b in base:
        if b.get('class_id') == sv_idx:
            runner = b.get('runner_up_margin', 0)
            if runner < 0.1:
                continue
        out.append({**b, 'teacher': 'calibrated_orbit'})
    return out
