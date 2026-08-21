#!/usr/bin/env python
"""P3A AP-constrained Gaussian semantic-support score projection.

This is an oracle validation tool for the P3A idea.  It uses validation GT only
to define AP-critical positives that must not be demoted.  Gaussian support is
computed over class-conditioned log area only; it is not a bbox-IoU, GWD, KLD,
or NWD surrogate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from evaluate_sise_score_calibration import (  # noqa: E402
    box_area_rbox,
    box_iou_rotated,
    canonical_img_id,
    get_img_id,
    get_pred_arrays,
    label_name,
    load_class_names,
    load_priors,
    read_gt,
    scale_features,
)


DEFAULT_SIBLING_PAIRS = (
    "small-vehicle->large-vehicle,large-vehicle->small-vehicle")


def _parse_class_names(class_names: str | Sequence[str] | None,
                       config: str | None) -> tuple[str, ...]:
    if class_names:
        if isinstance(class_names, str):
            return tuple(
                item.strip() for item in class_names.split(",")
                if item.strip())
        return tuple(str(item) for item in class_names)
    return tuple(load_class_names(config))


def _as_mutable_pred_instances(pred_instances: Any) -> dict[str, Any]:
    if isinstance(pred_instances, dict):
        return dict(pred_instances)
    return {
        "bboxes": getattr(pred_instances, "bboxes"),
        "scores": getattr(pred_instances, "scores"),
        "labels": getattr(pred_instances, "labels"),
    }


def _clone_scores(scores: Any) -> Any:
    if hasattr(scores, "clone"):
        return scores.clone()
    return np.array(scores, copy=True)


def _score_value(scores: Any, index: int) -> float:
    value = scores[index]
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _assign_score(scores: Any, index: int, value: float) -> None:
    if hasattr(scores, "dtype") and hasattr(scores, "device"):
        scores[index] = torch.as_tensor(
            value, dtype=scores.dtype, device=scores.device)
    else:
        scores[index] = value


def _class_counts(labels: Sequence[int], mask: Sequence[bool],
                  class_names: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for label, selected in zip(labels, mask):
        if not selected:
            continue
        counts[label_name(int(label), class_names)] += 1
    return dict(sorted(counts.items()))


def _ann_path_for_image(ann_dir: Path, img_id: str) -> Path:
    path = ann_dir / f"{img_id}.txt"
    if path.exists():
        return path
    return ann_dir / f"{canonical_img_id(img_id)}.txt"


def _safe_factor(log_area_z: float, z_thr: float, lambda_: float,
                 min_factor: float) -> float:
    violation = max(0.0, float(log_area_z) - float(z_thr))
    return max(float(min_factor), math.exp(-float(lambda_) * violation))


def apply_p3a_projection(input_pkl: str | Path,
                         output_pkl: str | Path,
                         ann_dir: str | Path,
                         class_area_priors_csv: str | Path,
                         class_names: str | Sequence[str] | None = None,
                         config: str | None = None,
                         iou_thr: float = 0.5,
                         diff_thr: int = 100,
                         z_thr: float = 4.0,
                         scale_ratio_thr: float = 4.0,
                         lambda_: float = 0.5,
                         min_factor: float = 0.01,
                         min_score: float = 0.3,
                         exclude_sibling_pairs: str = DEFAULT_SIBLING_PAIRS,
                         max_images: int = 0) -> dict[str, Any]:
    """Apply AP-constrained score projection and write a new predictions.pkl."""
    input_pkl = Path(input_pkl)
    output_pkl = Path(output_pkl)
    ann_dir = Path(ann_dir)
    class_names_tuple = _parse_class_names(class_names, config)
    class_to_label = {
        class_name: idx for idx, class_name in enumerate(class_names_tuple)
    }
    priors = load_priors(class_area_priors_csv)
    sibling_pairs = {
        item.strip() for item in str(exclude_sibling_pairs).split(",")
        if item.strip()
    }

    with input_pkl.open("rb") as f:
        predictions = pickle.load(f)

    output = []
    changed_by_class: dict[str, int] = defaultdict(int)
    protected_by_class: dict[str, int] = defaultdict(int)
    candidate_by_class: dict[str, int] = defaultdict(int)
    summary = {
        "input_pkl": str(input_pkl),
        "output_pkl": str(output_pkl),
        "ann_dir": str(ann_dir),
        "class_area_priors_csv": str(class_area_priors_csv),
        "num_samples": len(predictions),
        "images_scanned": 0,
        "images_missing_ann": 0,
        "detections_loaded": 0,
        "localized_total": 0,
        "protected_correct": 0,
        "changed_correct": 0,
        "localized_wrong_excl_sibling": 0,
        "low_support_wrong_candidates": 0,
        "scores_changed": 0,
        "mean_score_delta": 0.0,
        "max_score_delta": 0.0,
        "iou_thr": float(iou_thr),
        "z_thr": float(z_thr),
        "scale_ratio_thr": float(scale_ratio_thr),
        "lambda": float(lambda_),
        "min_factor": float(min_factor),
        "min_score": float(min_score),
        "projection_type": "oracle_ap_constrained_semantic_log_area_support",
        "forbidden_bbox_gaussian_route": False,
    }
    score_delta_sum = 0.0

    for image_idx, sample in enumerate(predictions):
        if max_images > 0 and image_idx >= max_images:
            output.append(sample)
            continue
        summary["images_scanned"] += 1
        img_id = get_img_id(sample)
        ann_file = _ann_path_for_image(ann_dir, img_id)
        if not ann_file.exists():
            summary["images_missing_ann"] += 1
        gt_rows = read_gt(ann_file, class_to_label, diff_thr)

        updated_sample = dict(sample)
        pred_instances = _as_mutable_pred_instances(sample["pred_instances"])
        scores = _clone_scores(pred_instances["scores"])
        pred_instances["scores"] = scores
        updated_sample["pred_instances"] = pred_instances
        output.append(updated_sample)

        boxes, labels, original_scores = get_pred_arrays(sample, 0)
        summary["detections_loaded"] += len(original_scores)
        if len(original_scores) == 0 or not gt_rows:
            continue

        gt_boxes = np.asarray([row["rbox"] for row in gt_rows],
                              dtype=np.float32)
        ious = box_iou_rotated(
            torch.from_numpy(boxes.astype(np.float32)),
            torch.from_numpy(gt_boxes.astype(np.float32))).cpu().numpy()
        best_gt = ious.argmax(axis=1)
        best_iou = ious[np.arange(ious.shape[0]), best_gt]

        changed_mask = [False] * len(labels)
        protected_mask = [False] * len(labels)
        candidate_mask = [False] * len(labels)
        for det_idx, gt_idx in enumerate(best_gt):
            if float(best_iou[det_idx]) <= float(iou_thr):
                continue
            summary["localized_total"] += 1
            gt = gt_rows[int(gt_idx)]
            pred_label = int(labels[det_idx])
            gt_label = int(gt["label"])
            pred_class = label_name(pred_label, class_names_tuple)
            gt_class = str(gt["class_name"])
            pair = f"{pred_class}->{gt_class}"
            score = float(original_scores[det_idx])
            area = box_area_rbox(boxes[det_idx])
            feats = scale_features(
                area, pred_class, priors, z_thr, scale_ratio_thr)
            correct = pred_label == gt_label
            if correct:
                if score >= min_score:
                    summary["protected_correct"] += 1
                    protected_mask[det_idx] = True
                    protected_by_class[pred_class] += 1
                continue

            wrong_excl_sibling = pair not in sibling_pairs
            if wrong_excl_sibling:
                summary["localized_wrong_excl_sibling"] += 1
            if (wrong_excl_sibling and score >= min_score
                    and feats["logz_implausible"]):
                summary["low_support_wrong_candidates"] += 1
                candidate_mask[det_idx] = True
                candidate_by_class[pred_class] += 1
                factor = _safe_factor(
                    feats["log_area_z"], z_thr, lambda_, min_factor)
                new_score = score * factor
                if new_score < score - 1e-12:
                    _assign_score(scores, det_idx, new_score)
                    changed_mask[det_idx] = True
                    changed_by_class[pred_class] += 1
                    summary["scores_changed"] += 1
                    delta = score - new_score
                    score_delta_sum += delta
                    summary["max_score_delta"] = max(
                        summary["max_score_delta"], delta)

        summary["changed_correct"] += sum(
            1 for changed, protected in zip(changed_mask, protected_mask)
            if changed and protected)

    if summary["scores_changed"]:
        summary["mean_score_delta"] = (
            score_delta_sum / float(summary["scores_changed"]))
    summary["changed_by_class"] = dict(sorted(changed_by_class.items()))
    summary["protected_by_class"] = dict(sorted(protected_by_class.items()))
    summary["candidate_by_class"] = dict(sorted(candidate_by_class.items()))
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
    return summary


def _first_metric(row: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if row.get(key) is not None:
            return row[key]
    return None


def assess_p3a_gate(row: dict[str, Any],
                    strict_control: dict[str, Any]) -> str:
    """Strict AP-constrained ranking gate against no-G3 control metrics."""
    row_map = _first_metric(row, ("mAP", "AP50"))
    base_map = _first_metric(strict_control, ("mAP", "AP50"))
    row_precision = _first_metric(
        row, ("top5000_precision", "topk_precision_top5000"))
    base_precision = _first_metric(
        strict_control, ("top5000_precision", "topk_precision_top5000"))
    row_sise = _first_metric(
        row, ("top5000_sise_logz", "sise_logz_topk_top5000"))
    base_sise = _first_metric(
        strict_control, ("top5000_sise_logz", "sise_logz_topk_top5000"))
    row_topk_risk = _first_metric(
        row, ("top5000_risk_weighted_logz",
              "risk_weighted_logz_false_alarm_top5000"))
    base_topk_risk = _first_metric(
        strict_control, ("top5000_risk_weighted_logz",
                         "risk_weighted_logz_false_alarm_top5000"))
    row_all_risk = _first_metric(
        row, ("risk_weighted_logz", "risk_weighted_logz_false_alarm"))
    base_all_risk = _first_metric(
        strict_control, ("risk_weighted_logz",
                         "risk_weighted_logz_false_alarm"))

    if any(value is None for value in (
            row_map, base_map, row_precision, base_precision)):
        return "waiting/missing"
    if float(row.get("changed_correct", 0) or 0) > 0:
        return "fail"

    ap_safe = float(row_map) >= float(base_map) - 1e-12
    precision_safe = float(row_precision) >= float(base_precision) - 1e-12
    topk_sise_improved = (
        row_sise is not None and base_sise is not None
        and float(row_sise) < float(base_sise))
    topk_risk_improved = (
        row_topk_risk is not None and base_topk_risk is not None
        and float(row_topk_risk) < float(base_topk_risk))
    all_risk_improved = (
        row_all_risk is not None and base_all_risk is not None
        and float(row_all_risk) < float(base_all_risk))
    strict = (
        ap_safe and precision_safe
        and (topk_sise_improved or topk_risk_improved
             or all_risk_improved))
    return "strict pass" if strict else "fail"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply P3A AP-constrained semantic-support projection.")
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--output-pkl", required=True)
    parser.add_argument("--ann-dir", required=True)
    parser.add_argument("--class-area-priors-csv", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--class-names", default=None)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--diff-thr", type=int, default=100)
    parser.add_argument("--z-thr", type=float, default=4.0)
    parser.add_argument("--scale-ratio-thr", type=float, default=4.0)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.5)
    parser.add_argument("--min-factor", type=float, default=0.01)
    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument(
        "--exclude-sibling-pairs", default=DEFAULT_SIBLING_PAIRS)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--summary-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = apply_p3a_projection(
        input_pkl=args.input_pkl,
        output_pkl=args.output_pkl,
        ann_dir=args.ann_dir,
        class_area_priors_csv=args.class_area_priors_csv,
        class_names=args.class_names,
        config=args.config,
        iou_thr=args.iou_thr,
        diff_thr=args.diff_thr,
        z_thr=args.z_thr,
        scale_ratio_thr=args.scale_ratio_thr,
        lambda_=args.lambda_,
        min_factor=args.min_factor,
        min_score=args.min_score,
        exclude_sibling_pairs=args.exclude_sibling_pairs,
        max_images=args.max_images,
    )
    summary_json = args.summary_json
    if summary_json is None:
        summary_json = str(Path(args.output_pkl).with_suffix(".summary.json"))
    Path(summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_json).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + os.linesep,
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
