#!/usr/bin/env python
"""Apply class scale priors directly to dumped detection predictions."""

import argparse
import csv
import json
import math
import os
import pickle
from collections import defaultdict
from pathlib import Path

import torch


DOTAV2_CLASS_NAMES = [
    "airport",
    "baseball-diamond",
    "basketball-court",
    "bridge",
    "container-crane",
    "ground-track-field",
    "harbor",
    "helicopter",
    "helipad",
    "large-vehicle",
    "plane",
    "roundabout",
    "ship",
    "small-vehicle",
    "soccer-ball-field",
    "storage-tank",
    "swimming-pool",
    "tennis-court",
]


STRATEGIES = {
    "prior_outlier_reject",
    "scale_ratio_reject",
    "prior_outlier_downweight",
    "scale_ratio_downweight",
    "log_area_calibrate",
}


def load_class_area_priors(path):
    priors = {}
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"class", "p01_area", "median_area", "p99_area"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing prior columns: {sorted(missing)}")
        for row in reader:
            cls = row["class"]
            priors[cls] = {
                "p01_area": float(row["p01_area"]),
                "median_area": float(row["median_area"]),
                "p99_area": float(row["p99_area"]),
                "log_area_mean": float(
                    row.get("log_area_mean") or math.log(float(row["median_area"]))),
                "log_area_std": float(row.get("log_area_std") or 0.0),
            }
    return priors


def compute_box_areas(boxes):
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)
    if boxes.ndim != 2:
        raise ValueError(f"Expected 2D boxes, got shape {tuple(boxes.shape)}")

    dims = boxes.shape[1]
    if dims == 5:
        return boxes[:, 2].abs() * boxes[:, 3].abs()
    if dims == 8:
        xs = boxes[:, 0::2]
        ys = boxes[:, 1::2]
        shifted_xs = torch.roll(xs, shifts=-1, dims=1)
        shifted_ys = torch.roll(ys, shifts=-1, dims=1)
        return (xs * shifted_ys - shifted_xs * ys).sum(dim=1).abs() * 0.5
    if dims == 4:
        return (boxes[:, 2] - boxes[:, 0]).abs() * (
            boxes[:, 3] - boxes[:, 1]).abs()
    raise ValueError(f"Unsupported box shape {tuple(boxes.shape)}")


def _parse_class_names(class_names):
    if isinstance(class_names, str):
        return [item.strip() for item in class_names.split(",") if item.strip()]
    return list(class_names)


def _prior_arrays(labels, class_names, priors, device):
    p01 = torch.zeros_like(labels, dtype=torch.float32, device=device)
    median = torch.zeros_like(labels, dtype=torch.float32, device=device)
    p99 = torch.zeros_like(labels, dtype=torch.float32, device=device)
    log_mean = torch.zeros_like(labels, dtype=torch.float32, device=device)
    log_std = torch.zeros_like(labels, dtype=torch.float32, device=device)
    has_prior = torch.zeros_like(labels, dtype=torch.bool, device=device)

    for label_id in labels.unique().tolist():
        label_id = int(label_id)
        if label_id < 0 or label_id >= len(class_names):
            continue
        cls = class_names[label_id]
        prior = priors.get(cls)
        if not prior:
            continue
        mask = labels == label_id
        p01[mask] = prior["p01_area"]
        median[mask] = prior["median_area"]
        p99[mask] = prior["p99_area"]
        log_mean[mask] = prior["log_area_mean"]
        log_std[mask] = prior["log_area_std"]
        has_prior[mask] = True
    return p01, median, p99, log_mean, log_std, has_prior


def _scale_ratio(areas, median, has_prior):
    ratio = torch.ones_like(areas, dtype=torch.float32)
    valid = has_prior & (areas > 0) & (median > 0)
    if valid.any():
        large = areas[valid] / median[valid]
        small = median[valid] / areas[valid]
        ratio[valid] = torch.maximum(large, small)
    return ratio


def _flag_predictions(pred_instances, class_names, priors, scale_ratio_thr,
                      log_area_z_margin):
    bboxes = pred_instances["bboxes"]
    labels = pred_instances["labels"].long()
    device = labels.device
    areas = compute_box_areas(bboxes).to(device)
    p01, median, p99, log_mean, log_std, has_prior = _prior_arrays(
        labels, class_names, priors, device)

    prior_outlier = has_prior & ((areas < p01) | (areas > p99))
    scale_ratio = _scale_ratio(areas, median, has_prior)
    scale_ratio_flag = has_prior & (scale_ratio >= scale_ratio_thr)
    valid_log = has_prior & (areas > 0) & (log_std > 0)
    log_area_z = torch.zeros_like(areas, dtype=torch.float32)
    if valid_log.any():
        log_area_z[valid_log] = (
            torch.log(areas[valid_log]) - log_mean[valid_log]).abs() / log_std[valid_log]
    log_area_violation = torch.clamp(log_area_z - log_area_z_margin, min=0.0)
    log_area_flag = valid_log & (log_area_violation > 0)
    return {
        "areas": areas,
        "prior_outlier": prior_outlier,
        "scale_ratio": scale_ratio,
        "scale_ratio_flag": scale_ratio_flag,
        "log_area_z": log_area_z,
        "log_area_violation": log_area_violation,
        "log_area_flag": log_area_flag,
        "has_prior": has_prior,
    }


def _index_pred_instances(pred_instances, keep_mask):
    return {
        key: value[keep_mask] if hasattr(value, "__getitem__") else value
        for key, value in pred_instances.items()
    }


def _class_counts(labels, mask, class_names):
    counts = defaultdict(int)
    for label_id in labels[mask].detach().cpu().tolist():
        label_id = int(label_id)
        cls = class_names[label_id] if 0 <= label_id < len(class_names) else str(label_id)
        counts[cls] += 1
    return dict(sorted(counts.items()))


def _downweight_scores(pred_instances, flags, flag_key, downweight_lambda):
    scores = pred_instances["scores"]
    flag = flags[flag_key]
    if not flag.any():
        return pred_instances, 0

    areas = flags["areas"]
    ratio = flags["scale_ratio"].clamp(min=1.0)
    factor = torch.ones_like(scores, dtype=torch.float32, device=scores.device)
    factor[flag] = torch.exp(-downweight_lambda * torch.log(ratio[flag]))
    updated = dict(pred_instances)
    updated["scores"] = scores * factor.to(scores.dtype)
    return updated, int(flag.sum().item())


def _calibrate_scores_by_log_area(pred_instances, flags, downweight_lambda,
                                  calibration_min_score=None):
    scores = pred_instances["scores"]
    flag = flags["log_area_flag"]
    if calibration_min_score is not None:
        flag = flag & (scores >= calibration_min_score)
    if not flag.any():
        return pred_instances, 0

    violation = flags["log_area_violation"]
    factor = torch.ones_like(scores, dtype=torch.float32, device=scores.device)
    factor[flag] = torch.exp(-downweight_lambda * violation[flag])
    updated = dict(pred_instances)
    updated["scores"] = scores * factor.to(scores.dtype)
    return updated, int(flag.sum().item())


def _apply_to_sample(sample, class_names, priors, strategy, scale_ratio_thr,
                     downweight_lambda, log_area_z_margin,
                     calibration_min_score=None):
    pred_instances = sample.get("pred_instances", {})
    if "bboxes" not in pred_instances or len(pred_instances["bboxes"]) == 0:
        return sample, {
            "before": 0,
            "after": 0,
            "rejected": 0,
            "downweighted": 0,
            "calibrated": 0,
            "prior_outlier": 0,
            "scale_ratio": 0,
            "has_prior": 0,
            "rejected_by_class": {},
            "downweighted_by_class": {},
            "calibrated_by_class": {},
        }

    labels = pred_instances["labels"].long()
    flags = _flag_predictions(pred_instances, class_names, priors,
                              scale_ratio_thr, log_area_z_margin)
    before = int(labels.numel())
    prior_count = int(flags["prior_outlier"].sum().item())
    scale_count = int(flags["scale_ratio_flag"].sum().item())
    has_prior_count = int(flags["has_prior"].sum().item())

    if strategy == "prior_outlier_reject":
        reject = flags["prior_outlier"]
        keep = ~reject
        updated_pred = _index_pred_instances(pred_instances, keep)
        rejected = int(reject.sum().item())
        downweighted = 0
        calibrated = 0
        rejected_by_class = _class_counts(labels, reject, class_names)
        downweighted_by_class = {}
        calibrated_by_class = {}
    elif strategy == "scale_ratio_reject":
        reject = flags["scale_ratio_flag"]
        keep = ~reject
        updated_pred = _index_pred_instances(pred_instances, keep)
        rejected = int(reject.sum().item())
        downweighted = 0
        calibrated = 0
        rejected_by_class = _class_counts(labels, reject, class_names)
        downweighted_by_class = {}
        calibrated_by_class = {}
    elif strategy == "prior_outlier_downweight":
        updated_pred, downweighted = _downweight_scores(
            pred_instances, flags, "prior_outlier", downweight_lambda)
        rejected = 0
        calibrated = 0
        rejected_by_class = {}
        downweighted_by_class = _class_counts(
            labels, flags["prior_outlier"], class_names)
        calibrated_by_class = {}
    elif strategy == "scale_ratio_downweight":
        updated_pred, downweighted = _downweight_scores(
            pred_instances, flags, "scale_ratio_flag", downweight_lambda)
        rejected = 0
        calibrated = 0
        rejected_by_class = {}
        downweighted_by_class = _class_counts(
            labels, flags["scale_ratio_flag"], class_names)
        calibrated_by_class = {}
    elif strategy == "log_area_calibrate":
        updated_pred, calibrated = _calibrate_scores_by_log_area(
            pred_instances, flags, downweight_lambda, calibration_min_score)
        rejected = 0
        downweighted = 0
        rejected_by_class = {}
        downweighted_by_class = {}
        calibrated_mask = flags["log_area_flag"]
        if calibration_min_score is not None:
            calibrated_mask = calibrated_mask & (
                pred_instances["scores"] >= calibration_min_score)
        calibrated_by_class = _class_counts(labels, calibrated_mask, class_names)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    updated_sample = dict(sample)
    updated_sample["pred_instances"] = updated_pred
    return updated_sample, {
        "before": before,
        "after": int(len(updated_pred["bboxes"])),
        "rejected": rejected,
        "downweighted": downweighted,
        "calibrated": calibrated,
        "prior_outlier": prior_count,
        "scale_ratio": scale_count,
        "has_prior": has_prior_count,
        "rejected_by_class": rejected_by_class,
        "downweighted_by_class": downweighted_by_class,
        "calibrated_by_class": calibrated_by_class,
    }


def _merge_counts(dst, src):
    for key, value in src.items():
        dst[key] += int(value)


def apply_scale_prior_to_predictions(input_pkl, output_pkl,
                                     class_area_priors_csv, class_names,
                                     strategy, scale_ratio_thr=4.0,
                                     downweight_lambda=1.0,
                                     log_area_z_margin=1.0,
                                     calibration_min_score=None):
    if strategy not in STRATEGIES:
        raise ValueError(f"Unsupported strategy: {strategy}")

    class_names = _parse_class_names(class_names)
    priors = load_class_area_priors(class_area_priors_csv)

    with Path(input_pkl).open("rb") as f:
        predictions = pickle.load(f)

    output = []
    rejected_by_class = defaultdict(int)
    downweighted_by_class = defaultdict(int)
    calibrated_by_class = defaultdict(int)
    summary = {
        "input_pkl": str(input_pkl),
        "output_pkl": str(output_pkl),
        "class_area_priors_csv": str(class_area_priors_csv),
        "strategy": strategy,
        "scale_ratio_thr": float(scale_ratio_thr),
        "downweight_lambda": float(downweight_lambda),
        "log_area_z_margin": float(log_area_z_margin),
        "calibration_min_score": (
            None if calibration_min_score is None else float(calibration_min_score)),
        "num_samples": len(predictions),
        "detections_before": 0,
        "detections_after": 0,
        "detections_rejected": 0,
        "detections_downweighted": 0,
        "detections_calibrated": 0,
        "prior_outlier_flags": 0,
        "scale_ratio_flags": 0,
        "detections_with_prior": 0,
    }

    for sample in predictions:
        updated_sample, stats = _apply_to_sample(
            sample, class_names, priors, strategy, scale_ratio_thr,
            downweight_lambda, log_area_z_margin, calibration_min_score)
        output.append(updated_sample)
        summary["detections_before"] += stats["before"]
        summary["detections_after"] += stats["after"]
        summary["detections_rejected"] += stats["rejected"]
        summary["detections_downweighted"] += stats["downweighted"]
        summary["detections_calibrated"] += stats["calibrated"]
        summary["prior_outlier_flags"] += stats["prior_outlier"]
        summary["scale_ratio_flags"] += stats["scale_ratio"]
        summary["detections_with_prior"] += stats["has_prior"]
        _merge_counts(rejected_by_class, stats["rejected_by_class"])
        _merge_counts(downweighted_by_class, stats["downweighted_by_class"])
        _merge_counts(calibrated_by_class, stats["calibrated_by_class"])

    summary["rejected_by_class"] = dict(sorted(rejected_by_class.items()))
    summary["downweighted_by_class"] = dict(sorted(downweighted_by_class.items()))
    summary["calibrated_by_class"] = dict(sorted(calibrated_by_class.items()))
    before = summary["detections_before"]
    summary["reject_rate"] = (
        summary["detections_rejected"] / before if before else 0.0)
    summary["downweight_rate"] = (
        summary["detections_downweighted"] / before if before else 0.0)
    summary["calibrate_rate"] = (
        summary["detections_calibrated"] / before if before else 0.0)

    output_pkl = Path(output_pkl)
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply class scale-prior postprocessing to predictions.pkl")
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--output-pkl", required=True)
    parser.add_argument("--class-area-priors-csv", required=True)
    parser.add_argument(
        "--class-names",
        default=",".join(DOTAV2_CLASS_NAMES),
        help="Comma-separated class names in label-id order.")
    parser.add_argument("--strategy", required=True, choices=sorted(STRATEGIES))
    parser.add_argument("--scale-ratio-thr", type=float, default=4.0)
    parser.add_argument("--downweight-lambda", type=float, default=1.0)
    parser.add_argument("--log-area-z-margin", type=float, default=1.0)
    parser.add_argument(
        "--calibration-min-score",
        type=float,
        default=None,
        help="Only calibrate log-area outliers whose original score is at least this value.")
    parser.add_argument("--summary-json", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = apply_scale_prior_to_predictions(
        input_pkl=args.input_pkl,
        output_pkl=args.output_pkl,
        class_area_priors_csv=args.class_area_priors_csv,
        class_names=args.class_names,
        strategy=args.strategy,
        scale_ratio_thr=args.scale_ratio_thr,
        downweight_lambda=args.downweight_lambda,
        log_area_z_margin=args.log_area_z_margin,
        calibration_min_score=args.calibration_min_score,
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
