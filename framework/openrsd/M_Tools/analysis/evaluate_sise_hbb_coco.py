#!/usr/bin/env python
"""Evaluate HBB semantic-scale errors on COCO-style datasets."""

import argparse
import csv
import json
import math
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_variant(raw):
    if ":" not in raw:
        raise ValueError(f"--variant must be name:path, got {raw}")
    name, path = raw.split(":", 1)
    if not name:
        raise ValueError(f"empty variant name in {raw}")
    return name, Path(path)


def parse_thresholds(raw):
    return [float(item) for item in raw.split(",") if item.strip()]


def threshold_tag(thr):
    return str(thr).replace(".", "p")


def tensor_to_numpy(value):
    if hasattr(value, "tensor"):
        value = value.tensor
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def field(obj, key):
    if isinstance(obj, dict):
        return obj[key]
    return getattr(obj, key)


def optional_field(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def sample_img_id(sample):
    img_id = optional_field(sample, "img_id")
    if img_id is not None:
        return int(img_id)
    metainfo = optional_field(sample, "metainfo", {}) or {}
    return int(metainfo["img_id"])


def load_coco_gt(ann_json):
    data = json.loads(Path(ann_json).read_text(encoding="utf-8"))
    categories = sorted(data["categories"], key=lambda item: item["id"])
    cat_id_to_label = {int(cat["id"]): idx for idx, cat in enumerate(categories)}
    label_to_name = {idx: cat["name"] for idx, cat in enumerate(categories)}
    gt_by_img = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        cat_id = int(ann["category_id"])
        if cat_id not in cat_id_to_label:
            continue
        x, y, w, h = [float(v) for v in ann["bbox"]]
        if w <= 0.0 or h <= 0.0:
            continue
        gt_by_img[int(ann["image_id"])].append({
            "label": cat_id_to_label[cat_id],
            "class_name": label_to_name[cat_id_to_label[cat_id]],
            "bbox": np.asarray([x, y, x + w, y + h], dtype=np.float32),
            "area": float(ann.get("area", w * h)),
        })
    return {
        "categories": categories,
        "cat_id_to_label": cat_id_to_label,
        "label_to_name": label_to_name,
        "gt_by_img": gt_by_img,
        "image_count": len(data.get("images", [])),
        "annotation_count": len(data.get("annotations", [])),
    }


def load_priors(path):
    priors = {}
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            cls = row.get("class", "")
            if not cls:
                continue
            priors[cls] = {
                "count": int(float(row.get("count", 0) or 0)),
                "p01_area": float(row.get("p01_area", 0) or 0),
                "p99_area": float(row.get("p99_area", 0) or 0),
                "median_area": float(row.get("median_area", 0) or 0),
                "log_area_mean": float(row.get("log_area_mean", 0) or 0),
                "log_area_std": max(float(row.get("log_area_std", 0) or 0), 1e-6),
            }
    return priors


def hbb_iou_matrix(boxes, gt_boxes):
    if len(boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(boxes), len(gt_boxes)), dtype=np.float32)
    boxes = boxes.astype(np.float32)
    gt_boxes = gt_boxes.astype(np.float32)
    lt = np.maximum(boxes[:, None, :2], gt_boxes[None, :, :2])
    rb = np.minimum(boxes[:, None, 2:], gt_boxes[None, :, 2:])
    wh = np.maximum(rb - lt, 0.0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(
        boxes[:, 3] - boxes[:, 1], 0.0)
    area2 = np.maximum(gt_boxes[:, 2] - gt_boxes[:, 0], 0.0) * np.maximum(
        gt_boxes[:, 3] - gt_boxes[:, 1], 0.0)
    union = area1[:, None] + area2[None, :] - inter
    return np.divide(inter, np.maximum(union, 1e-6), dtype=np.float32)


def pred_arrays(sample, max_dets_per_img):
    pred_instances = field(sample, "pred_instances")
    boxes = tensor_to_numpy(field(pred_instances, "bboxes")).reshape(-1, 4).astype(np.float32)
    labels = tensor_to_numpy(field(pred_instances, "labels")).astype(np.int64)
    scores = tensor_to_numpy(field(pred_instances, "scores")).astype(np.float32)
    if max_dets_per_img > 0 and len(scores) > max_dets_per_img:
        order = np.argsort(-scores)[:max_dets_per_img]
        boxes = boxes[order]
        labels = labels[order]
        scores = scores[order]
    return boxes, labels, scores


def box_area(box):
    return max(float(box[2] - box[0]), 0.0) * max(float(box[3] - box[1]), 0.0)


def scale_features(area, cls, priors, z_thr, ratio_thr):
    prior = priors.get(cls)
    if not prior or area <= 0:
        return {
            "log_area_z": 0.0,
            "scale_ratio": 0.0,
            "logz_implausible": False,
            "ratio_implausible": False,
            "p0199_outlier": False,
        }
    log_area = math.log(area)
    logz = abs((log_area - prior["log_area_mean"]) / prior["log_area_std"])
    median = prior["median_area"]
    ratio = max(area / median, median / area) if median > 0 else 0.0
    return {
        "log_area_z": logz,
        "scale_ratio": ratio,
        "logz_implausible": logz >= z_thr,
        "ratio_implausible": ratio >= ratio_thr,
        "p0199_outlier": area < prior["p01_area"] or area > prior["p99_area"],
    }


def label_name(label, label_to_name):
    label = int(label)
    return label_to_name.get(label, f"label_{label}")


def parse_sibling_pairs(raw):
    pairs = set()
    if not raw:
        return pairs
    for item in raw.split(","):
        item = item.strip()
        if item:
            pairs.add(item)
    return pairs


def add_threshold_counts(summary, prefix, scores, score_thrs):
    for thr in score_thrs:
        summary[f"{prefix}_score_ge_{threshold_tag(thr)}"] = int(
            sum(float(score) >= thr for score in scores))


def summarize_variant(name, pred_path, gt, priors, args, score_thrs, sibling_pairs):
    with pred_path.open("rb") as f:
        predictions = pickle.load(f)
    stats = Counter()
    wrong_scores = []
    wrong_excl_sibling_scores = []
    sise_logz_scores = []
    sise_p0199_scores = []
    correct_scores = []
    pair_stats = defaultdict(lambda: {
        "count": 0,
        "scores": [],
        "logz": [],
        "ratio": [],
        "logz_count": 0,
        "p0199_count": 0,
    })

    for image_idx, sample in enumerate(predictions):
        if args.max_images > 0 and image_idx >= args.max_images:
            break
        img_id = sample_img_id(sample)
        gt_rows = gt["gt_by_img"].get(img_id, [])
        boxes, labels, scores = pred_arrays(sample, args.max_dets_per_img)
        stats["images_scanned"] += 1
        stats["detections_checked"] += len(scores)
        stats["gt_objects_checked"] += len(gt_rows)
        if len(scores) == 0 or not gt_rows:
            continue

        gt_boxes = np.asarray([row["bbox"] for row in gt_rows], dtype=np.float32)
        ious = hbb_iou_matrix(boxes, gt_boxes)
        best_gt = ious.argmax(axis=1)
        best_iou = ious[np.arange(ious.shape[0]), best_gt]

        for det_idx, gt_idx in enumerate(best_gt):
            if best_iou[det_idx] < args.iou_thr:
                continue
            gt_row = gt_rows[int(gt_idx)]
            pred_label = int(labels[det_idx])
            gt_label = int(gt_row["label"])
            pred_class = label_name(pred_label, gt["label_to_name"])
            gt_class = gt_row["class_name"]
            score = float(scores[det_idx])
            feats = scale_features(
                box_area(boxes[det_idx]), pred_class, priors,
                args.z_thr, args.scale_ratio_thr)
            pair = f"{pred_class}->{gt_class}"
            is_sibling = pair in sibling_pairs
            correct = pred_label == gt_label
            stats["localized_total"] += 1
            if correct:
                stats["localized_correct"] += 1
                correct_scores.append(score)
                continue
            stats["localized_wrong"] += 1
            wrong_scores.append(score)
            if not is_sibling:
                stats["localized_wrong_excl_sibling"] += 1
                wrong_excl_sibling_scores.append(score)
            if feats["logz_implausible"] and not is_sibling:
                stats["sise_logz_wrong_excl_sibling"] += 1
                sise_logz_scores.append(score)
            if feats["p0199_outlier"] and not is_sibling:
                stats["sise_p0199_wrong_excl_sibling"] += 1
                sise_p0199_scores.append(score)
            pair_row = pair_stats[pair]
            pair_row["count"] += 1
            pair_row["scores"].append(score)
            pair_row["logz"].append(feats["log_area_z"])
            pair_row["ratio"].append(feats["scale_ratio"])
            if feats["logz_implausible"]:
                pair_row["logz_count"] += 1
            if feats["p0199_outlier"]:
                pair_row["p0199_count"] += 1

    summary = {
        "variant": name,
        "predictions": str(pred_path),
        "images_scanned": int(stats["images_scanned"]),
        "detections_checked": int(stats["detections_checked"]),
        "gt_objects_checked": int(stats["gt_objects_checked"]),
        "localized_total": int(stats["localized_total"]),
        "localized_correct": int(stats["localized_correct"]),
        "localized_wrong": int(stats["localized_wrong"]),
        "localized_wrong_excl_sibling": int(stats["localized_wrong_excl_sibling"]),
        "sise_logz_wrong_excl_sibling": int(stats["sise_logz_wrong_excl_sibling"]),
        "sise_p0199_wrong_excl_sibling": int(stats["sise_p0199_wrong_excl_sibling"]),
        "wrong_score_mean": float(np.mean(wrong_scores)) if wrong_scores else 0.0,
        "wrong_excl_sibling_score_mean": (
            float(np.mean(wrong_excl_sibling_scores))
            if wrong_excl_sibling_scores else 0.0),
        "sise_logz_score_mean": (
            float(np.mean(sise_logz_scores)) if sise_logz_scores else 0.0),
        "sise_p0199_score_mean": (
            float(np.mean(sise_p0199_scores)) if sise_p0199_scores else 0.0),
        "correct_score_mean": (
            float(np.mean(correct_scores)) if correct_scores else 0.0),
    }
    add_threshold_counts(summary, "wrong", wrong_scores, score_thrs)
    add_threshold_counts(summary, "wrong_excl_sibling", wrong_excl_sibling_scores, score_thrs)
    add_threshold_counts(summary, "sise_logz", sise_logz_scores, score_thrs)
    add_threshold_counts(summary, "sise_p0199", sise_p0199_scores, score_thrs)
    summary["top_pairs"] = []
    for pair, row in sorted(pair_stats.items(), key=lambda item: -item[1]["count"])[:30]:
        summary["top_pairs"].append({
            "pair": pair,
            "count": int(row["count"]),
            "score_mean": float(np.mean(row["scores"])) if row["scores"] else 0.0,
            "logz_mean": float(np.mean(row["logz"])) if row["logz"] else 0.0,
            "ratio_mean": float(np.mean(row["ratio"])) if row["ratio"] else 0.0,
            "logz_count": int(row["logz_count"]),
            "p0199_count": int(row["p0199_count"]),
        })
    return summary


def build_delta(reference, current):
    delta = {
        "variant": current["variant"],
        "reference": reference["variant"],
    }
    keys = [
        "localized_wrong",
        "localized_wrong_excl_sibling",
        "sise_logz_wrong_excl_sibling",
        "sise_p0199_wrong_excl_sibling",
    ]
    for key in keys:
        base = reference.get(key, 0)
        value = current.get(key, 0)
        delta[f"{key}_delta"] = int(value - base)
        delta[f"{key}_reduction_rate"] = (
            (base - value) / base if base else 0.0)
    for key, value in current.items():
        if key.endswith("_score_ge_0p5") or key.endswith("_score_ge_0p9"):
            base = reference.get(key, 0)
            delta[f"{key}_delta"] = int(value - base)
            delta[f"{key}_reduction_rate"] = (
                (base - value) / base if base else 0.0)
    return delta


def write_markdown(path, payload):
    lines = [
        "# HBB COCO SISE Audit",
        "",
        f"- ann_json: `{payload['ann_json']}`",
        f"- priors: `{payload['class_area_priors_csv']}`",
        f"- iou_thr: `{payload['iou_thr']}`",
        f"- z_thr: `{payload['z_thr']}`",
        "",
        "| variant | localized_wrong | SISE logz | SISE p01/p99 | wrong@0.5 | SISE logz@0.5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["variants"]:
        lines.append(
            f"| {row['variant']} | {row['localized_wrong_excl_sibling']} | "
            f"{row['sise_logz_wrong_excl_sibling']} | "
            f"{row['sise_p0199_wrong_excl_sibling']} | "
            f"{row.get('wrong_excl_sibling_score_ge_0p5', 0)} | "
            f"{row.get('sise_logz_score_ge_0p5', 0)} |")
    if payload["deltas"]:
        lines.extend(["", "## Deltas", ""])
        for row in payload["deltas"]:
            lines.append(
                f"- {row['variant']} vs {row['reference']}: "
                f"wrong_excl_sibling_delta={row['localized_wrong_excl_sibling_delta']}, "
                f"sise_logz_delta={row['sise_logz_wrong_excl_sibling_delta']}, "
                f"sise_p0199_delta={row['sise_p0199_wrong_excl_sibling_delta']}")
    Path(path).write_text("\n".join(lines) + os.linesep, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--ann-json", required=True)
    parser.add_argument("--class-area-priors-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--z-thr", type=float, default=4.0)
    parser.add_argument("--scale-ratio-thr", type=float, default=4.0)
    parser.add_argument("--score-thrs", default="0.5,0.7,0.9,0.99,0.999")
    parser.add_argument("--exclude-sibling-pairs", default="")
    parser.add_argument("--max-dets-per-img", type=int, default=100)
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_thrs = parse_thresholds(args.score_thrs)
    sibling_pairs = parse_sibling_pairs(args.exclude_sibling_pairs)
    gt = load_coco_gt(args.ann_json)
    priors = load_priors(args.class_area_priors_csv)
    variants = [
        summarize_variant(name, path, gt, priors, args, score_thrs, sibling_pairs)
        for name, path in (parse_variant(raw) for raw in args.variant)
    ]
    reference = variants[0]
    deltas = [build_delta(reference, row) for row in variants[1:]]
    payload = {
        "ann_json": args.ann_json,
        "class_area_priors_csv": args.class_area_priors_csv,
        "iou_thr": float(args.iou_thr),
        "z_thr": float(args.z_thr),
        "scale_ratio_thr": float(args.scale_ratio_thr),
        "max_dets_per_img": int(args.max_dets_per_img),
        "max_images": int(args.max_images),
        "image_count": gt["image_count"],
        "annotation_count": gt["annotation_count"],
        "category_names": [cat["name"] for cat in gt["categories"]],
        "variants": variants,
        "deltas": deltas,
    }
    summary_path = out_dir / "sise_hbb_coco_summary.json"
    md_path = out_dir / "sise_hbb_coco_summary.md"
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + os.linesep,
        encoding="utf-8")
    write_markdown(md_path, payload)
    print(json.dumps({
        "summary_json": str(summary_path),
        "variants": [
            {
                "variant": row["variant"],
                "wrong": row["localized_wrong_excl_sibling"],
                "sise_logz": row["sise_logz_wrong_excl_sibling"],
                "sise_p0199": row["sise_p0199_wrong_excl_sibling"],
            }
            for row in variants
        ],
        "deltas": deltas,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
