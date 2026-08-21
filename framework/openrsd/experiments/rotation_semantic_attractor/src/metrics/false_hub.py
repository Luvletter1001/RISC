from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

from ..constants import LARGE_VEHICLE, SMALL_VEHICLE
from .matching import best_gt_match, greedy_match


def compute_false_hub_rows(
    model_name: str,
    tile_id: str,
    angle: int,
    predictions: Sequence[Dict],
    gt_records: Sequence[Dict],
    iou_thr: float = 0.30,
    region_mode: str = "all_region",
) -> tuple[Dict, List[Dict]]:
    final_predictions = [p for p in predictions if p.get("stage") == "final"]
    sv_predictions = [p for p in final_predictions if p.get("class_name") == SMALL_VEHICLE]
    gt_sv = [g for g in gt_records if g.get("class_name") == SMALL_VEHICLE]
    gt_total = [g for g in gt_records if "polygon" in g]

    sv_matches = greedy_match(sv_predictions, gt_sv, iou_thr=iou_thr, gt_class=SMALL_VEHICLE)
    matched_sv_pred = {pi for pi, _, _ in sv_matches}
    matched_sv_gt = {gi for _, gi, _ in sv_matches}

    bg_false = 0
    object_flip = 0
    absorption = Counter()
    for idx, pred in enumerate(sv_predictions):
        best = best_gt_match(pred["polygon"], gt_total)
        gt = best["gt"]
        if best["iou"] < iou_thr or gt is None:
            bg_false += 1
        else:
            gt_class = gt.get("class_name", "")
            absorption[gt_class] += 1
            if gt_class not in {SMALL_VEHICLE, LARGE_VEHICLE}:
                object_flip += 1

    total_pred = len(final_predictions)
    num_sv = len(sv_predictions)
    false_sv = max(0, num_sv - len(matched_sv_pred))
    summary = {
        "model_name": model_name,
        "tile_id": tile_id,
        "angle": int(angle),
        "region_mode": region_mode,
        "total_pred": total_pred,
        "num_sv_pred": num_sv,
        "fr_sv": num_sv / total_pred if total_pred else 0.0,
        "false_sv_ratio": false_sv / num_sv if num_sv else 0.0,
        "bg_fsv_ratio": bg_false / num_sv if num_sv else 0.0,
        "object_flip_sv": object_flip,
        "true_sv_recall": len(matched_sv_gt) / len(gt_sv) if gt_sv else 0.0,
        "sv_tp_ratio": len(matched_sv_pred) / num_sv if num_sv else 0.0,
        "num_gt_sv": len(gt_sv),
        "num_gt_total": len(gt_total),
        "det_per_img": total_pred,
        "iou_thr": iou_thr,
        "iou_fallback": True,
    }
    per_class = [
        {
            "model_name": model_name,
            "tile_id": tile_id,
            "angle": int(angle),
            "region_mode": region_mode,
            "gt_class": cls,
            "sv_overlap_count": count,
        }
        for cls, count in sorted(absorption.items())
    ]
    return summary, per_class

