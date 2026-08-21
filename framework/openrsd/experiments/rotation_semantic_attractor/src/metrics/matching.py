from __future__ import annotations

from typing import Dict, List, Sequence

from ..geometry import hbb_iou, polygon_to_hbb


def best_gt_match(pred_polygon: Sequence[Sequence[float]], gt_records: Sequence[Dict]) -> Dict:
    pred_hbb = polygon_to_hbb(pred_polygon)
    best = {"iou": 0.0, "gt": None, "fallback": True}
    for gt in gt_records:
        if "polygon" not in gt:
            continue
        iou = hbb_iou(pred_hbb, polygon_to_hbb(gt["polygon"]))
        if iou > best["iou"]:
            best = {"iou": iou, "gt": gt, "fallback": True}
    return best


def greedy_match(predictions: Sequence[Dict], gt_records: Sequence[Dict], iou_thr: float, gt_class: str | None = None) -> List[tuple[int, int, float]]:
    candidates = []
    for pi, pred in enumerate(predictions):
        for gi, gt in enumerate(gt_records):
            if gt_class is not None and gt.get("class_name") != gt_class:
                continue
            iou = hbb_iou(polygon_to_hbb(pred["polygon"]), polygon_to_hbb(gt["polygon"]))
            if iou >= iou_thr:
                candidates.append((iou, pi, gi))
    candidates.sort(reverse=True)
    used_p, used_g, matches = set(), set(), []
    for iou, pi, gi in candidates:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches.append((pi, gi, iou))
    return matches

