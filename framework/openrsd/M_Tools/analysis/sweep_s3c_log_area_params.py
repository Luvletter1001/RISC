#!/usr/bin/env python
"""Calibration-split sweep for S3C log-area score calibration."""

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from evaluate_sise_score_calibration import (
    box_area_rbox,
    boxes_to_rboxes,
    canonical_img_id,
    field,
    get_img_id,
    label_name,
    load_class_names,
    load_priors,
    qboxes_to_rboxes,
    read_gt,
    tensor_to_numpy,
)
from mmcv.ops import box_iou_rotated


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ann-dir", required=True)
    parser.add_argument("--class-area-priors-csv", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--z-margins", default="3.0,3.5,4.0,4.5,5.0")
    parser.add_argument("--lambdas", default="0.05,0.1,0.15,0.2,0.25")
    parser.add_argument("--metric-z-thr", type=float, default=4.0)
    parser.add_argument("--score-thr", type=float, default=0.999)
    parser.add_argument("--iou-thr", type=float, default=0.7)
    parser.add_argument("--diff-thr", type=int, default=100)
    parser.add_argument("--calibration-frac", type=float, default=0.5)
    parser.add_argument("--max-correct-drop-rate", type=float, default=0.002)
    parser.add_argument("--max-changed-correct-rate", type=float, default=0.005)
    parser.add_argument(
        "--exclude-sibling-pairs",
        default="small-vehicle->large-vehicle,large-vehicle->small-vehicle")
    parser.add_argument(
        "--ece-bins", default="0,0.5,0.7,0.9,0.99,0.999,1.000001")
    return parser.parse_args()


def split_for_image(img_id, calibration_frac):
    digest = hashlib.md5(canonical_img_id(img_id).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / float(0xFFFFFFFF)
    return "calibration" if bucket < calibration_frac else "holdout"


def scale_features(area, cls, priors):
    prior = priors.get(cls)
    if not prior or area <= 0:
        return 0.0, 0.0, False
    log_area = math.log(area)
    logz = abs((log_area - prior["log_area_mean"]) / prior["log_area_std"])
    med = prior["median_area"]
    ratio = max(area / med, med / area) if med > 0 else 0.0
    p0199 = area < prior["p01_area"] or area > prior["p99_area"]
    return logz, ratio, p0199


def collect_localized_records(args, class_names, priors, sibling_pairs):
    class_to_label = {name: idx for idx, name in enumerate(class_names)}
    ann_dir = Path(args.ann_dir)
    with Path(args.predictions).open("rb") as f:
        predictions = pickle.load(f)

    records = []
    image_counts = {"calibration": 0, "holdout": 0}
    for sample in predictions:
        img_id = get_img_id(sample)
        split = split_for_image(img_id, args.calibration_frac)
        image_counts[split] += 1
        ann_file = ann_dir / f"{img_id}.txt"
        if not ann_file.exists():
            ann_file = ann_dir / f"{canonical_img_id(img_id)}.txt"
        gt_rows = read_gt(ann_file, class_to_label, args.diff_thr)
        pred_instances = field(sample, "pred_instances")
        boxes = boxes_to_rboxes(tensor_to_numpy(field(pred_instances, "bboxes")))
        labels = tensor_to_numpy(field(pred_instances, "labels")).astype(np.int64)
        scores = tensor_to_numpy(field(pred_instances, "scores")).astype(np.float32)
        if len(scores) == 0 or len(gt_rows) == 0:
            continue
        gt_boxes = np.asarray([row["rbox"] for row in gt_rows], dtype=np.float32)
        ious = box_iou_rotated(
            torch.from_numpy(boxes.astype(np.float32)),
            torch.from_numpy(gt_boxes)).cpu().numpy()
        best_gt = ious.argmax(axis=1)
        best_iou = ious[np.arange(ious.shape[0]), best_gt]
        for det_idx, gt_idx in enumerate(best_gt):
            if best_iou[det_idx] <= args.iou_thr:
                continue
            pred_label = int(labels[det_idx])
            gt = gt_rows[int(gt_idx)]
            gt_label = int(gt["label"])
            pred_class = label_name(pred_label, class_names)
            gt_class = gt["class_name"]
            pair = f"{pred_class}->{gt_class}"
            area = box_area_rbox(boxes[det_idx])
            logz, ratio, p0199 = scale_features(area, pred_class, priors)
            records.append({
                "img_id": img_id,
                "det_idx": int(det_idx),
                "split": split,
                "score": float(scores[det_idx]),
                "correct": pred_label == gt_label,
                "pair": pair,
                "is_sibling": pair in sibling_pairs,
                "log_area_z": float(logz),
                "scale_ratio": float(ratio),
                "p0199_outlier": bool(p0199),
            })
    return records, image_counts


def ece(scores, correct, bins):
    if len(scores) == 0:
        return 0.0
    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    total = len(scores)
    value = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (scores >= lo) & (scores < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        avg_score = float(scores[mask].mean())
        acc = float(correct[mask].mean())
        value += count / total * abs(avg_score - acc)
    return value


def summarize(records, split, z_margin, lam, metric_z_thr, score_thr, bins):
    rows = [row for row in records if row["split"] == split]
    if not rows:
        return {}
    score = np.asarray([row["score"] for row in rows], dtype=np.float64)
    correct = np.asarray([row["correct"] for row in rows], dtype=bool)
    sibling = np.asarray([row["is_sibling"] for row in rows], dtype=bool)
    logz = np.asarray([row["log_area_z"] for row in rows], dtype=np.float64)
    violation = np.maximum(logz - z_margin, 0.0)
    calibrated = score * np.exp(-lam * violation)
    wrong = ~correct
    wrong_excl = wrong & ~sibling
    metric_sise = wrong_excl & (logz >= metric_z_thr)
    correct_logz = correct & (logz >= metric_z_thr)
    score_changed = np.abs(calibrated - score) > 1e-6
    correct_before_high = correct & (score >= score_thr)
    correct_after_high = correct & (calibrated >= score_thr)
    return {
        "split": split,
        "z_margin": z_margin,
        "lambda": lam,
        "localized_total": int(len(rows)),
        "localized_correct": int(correct.sum()),
        "localized_wrong": int(wrong.sum()),
        "wrong_excl_sibling": int(wrong_excl.sum()),
        "wrong_excl_sibling_before_high": int((wrong_excl & (score >= score_thr)).sum()),
        "wrong_excl_sibling_after_high": int((wrong_excl & (calibrated >= score_thr)).sum()),
        "sise_logz_before_high": int((metric_sise & (score >= score_thr)).sum()),
        "sise_logz_after_high": int((metric_sise & (calibrated >= score_thr)).sum()),
        "sise_logz_reduction": int((metric_sise & (score >= score_thr)).sum()) -
        int((metric_sise & (calibrated >= score_thr)).sum()),
        "sise_logz_reduction_rate": (
            (int((metric_sise & (score >= score_thr)).sum()) -
             int((metric_sise & (calibrated >= score_thr)).sum())) /
            int((metric_sise & (score >= score_thr)).sum())
            if int((metric_sise & (score >= score_thr)).sum()) else 0.0),
        "correct_before_high": int(correct_before_high.sum()),
        "correct_after_high": int(correct_after_high.sum()),
        "correct_drop_high": int((correct_before_high & ~correct_after_high).sum()),
        "correct_drop_high_rate": (
            int((correct_before_high & ~correct_after_high).sum()) /
            int(correct_before_high.sum()) if int(correct_before_high.sum()) else 0.0),
        "changed_correct": int((correct & score_changed).sum()),
        "changed_correct_rate": (
            int((correct & score_changed).sum()) / int(correct.sum())
            if int(correct.sum()) else 0.0),
        "changed_wrong": int((wrong & score_changed).sum()),
        "correct_logz_after_high": int((correct_logz & (calibrated >= score_thr)).sum()),
        "si_ece_all": ece(calibrated, correct, bins),
        "si_ece_logz": ece(calibrated[logz >= metric_z_thr],
                           correct[logz >= metric_z_thr], bins),
    }


def write_csv(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_candidate(rows, args):
    calibration = [row for row in rows if row["split"] == "calibration"]
    feasible = [
        row for row in calibration
        if row["correct_drop_high_rate"] <= args.max_correct_drop_rate
        and row["changed_correct_rate"] <= args.max_changed_correct_rate
    ]
    pool = feasible if feasible else calibration
    return sorted(
        pool,
        key=lambda row: (
            row["sise_logz_after_high"],
            row["correct_drop_high_rate"],
            row["changed_correct_rate"],
            row["si_ece_logz"],
        ))[0]


def fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows, cols):
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    z_margins = [float(item) for item in args.z_margins.split(",") if item]
    lambdas = [float(item) for item in args.lambdas.split(",") if item]
    bins = [float(item) for item in args.ece_bins.split(",") if item]
    sibling_pairs = {
        item for item in args.exclude_sibling_pairs.split(",") if item.strip()
    }
    class_names = load_class_names(args.config)
    priors = load_priors(args.class_area_priors_csv)
    records, image_counts = collect_localized_records(
        args, class_names, priors, sibling_pairs)

    rows = []
    for z_margin in z_margins:
        for lam in lambdas:
            for split in ("calibration", "holdout"):
                rows.append(summarize(
                    records, split, z_margin, lam, args.metric_z_thr,
                    args.score_thr, bins))

    selected = choose_candidate(rows, args)
    selected_holdout = next(
        row for row in rows
        if row["split"] == "holdout"
        and row["z_margin"] == selected["z_margin"]
        and row["lambda"] == selected["lambda"])
    write_csv(out_dir / "s3c_log_area_sweep.csv", rows)
    write_csv(out_dir / "localized_records.csv", records)
    payload = {
        "predictions": args.predictions,
        "ann_dir": args.ann_dir,
        "class_area_priors_csv": args.class_area_priors_csv,
        "config": args.config,
        "image_counts": image_counts,
        "localized_records": len(records),
        "score_thr": args.score_thr,
        "metric_z_thr": args.metric_z_thr,
        "max_correct_drop_rate": args.max_correct_drop_rate,
        "max_changed_correct_rate": args.max_changed_correct_rate,
        "selected_on_calibration": selected,
        "selected_holdout": selected_holdout,
        "outputs": {
            "sweep_csv": str(out_dir / "s3c_log_area_sweep.csv"),
            "localized_records_csv": str(out_dir / "localized_records.csv"),
            "report_md": str(out_dir / "s3c_log_area_sweep_report.md"),
        },
    }
    (out_dir / "s3c_log_area_sweep_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + os.linesep,
        encoding="utf-8")

    selected_rows = [selected, selected_holdout]
    best_holdout = sorted(
        [row for row in rows if row["split"] == "holdout"],
        key=lambda row: (
            row["sise_logz_after_high"],
            row["correct_drop_high_rate"],
            row["changed_correct_rate"],
            row["si_ece_logz"],
        ))[:8]
    cols = [
        "split", "z_margin", "lambda", "sise_logz_before_high",
        "sise_logz_after_high", "sise_logz_reduction_rate",
        "wrong_excl_sibling_after_high", "correct_after_high",
        "correct_drop_high_rate", "changed_correct_rate", "si_ece_logz",
    ]
    lines = [
        "# S3C Log-Area Calibration Split Sweep",
        "",
        f"score_thr={args.score_thr}, metric_z_thr={args.metric_z_thr}, "
        f"calibration_frac={args.calibration_frac}.",
        "",
        "## Selected by Calibration Split",
        "",
        markdown_table(selected_rows, cols),
        "",
        "## Top Holdout Candidates",
        "",
        markdown_table(best_holdout, cols),
    ]
    (out_dir / "s3c_log_area_sweep_report.md").write_text(
        "\n".join(lines) + os.linesep, encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
