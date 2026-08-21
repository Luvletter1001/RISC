#!/usr/bin/env python
"""Evaluate scale-inconsistent semantic errors from saved detections."""

import argparse
import ast
import csv
import json
import math
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from openrsd_env import preload_installed_mmengine  # noqa: E402

preload_installed_mmengine()

from mmcv.ops import box_iou_rotated  # noqa: E402
from mmrotate.structures.bbox import qbox2rbox  # noqa: E402


DEFAULT_CLASSES = (
    "plane", "baseball-diamond", "bridge", "ground-track-field",
    "small-vehicle", "large-vehicle", "ship", "tennis-court",
    "basketball-court", "storage-tank", "soccer-ball-field", "roundabout",
    "harbor", "swimming-pool", "helicopter", "container-crane", "airport",
    "helipad",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute SISE@score and score calibration metrics.")
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="Variant in name:path format. The first variant is reference.")
    parser.add_argument("--ann-dir", required=True)
    parser.add_argument("--class-area-priors-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--iou-thr", type=float, default=0.7)
    parser.add_argument("--diff-thr", type=int, default=100)
    parser.add_argument("--z-thr", type=float, default=4.0)
    parser.add_argument("--scale-ratio-thr", type=float, default=4.0)
    parser.add_argument("--score-thrs", default="0.5,0.7,0.9,0.99,0.999")
    parser.add_argument(
        "--ece-bins", default="0,0.5,0.7,0.9,0.99,0.999,1.000001")
    parser.add_argument(
        "--exclude-sibling-pairs",
        default="small-vehicle->large-vehicle,large-vehicle->small-vehicle")
    parser.add_argument("--score-change-eps", type=float, default=1e-6)
    parser.add_argument("--max-dets-per-img", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


def load_class_names(config_path):
    if not config_path:
        return DEFAULT_CLASSES
    try:
        from mmengine.config import Config
        cfg = Config.fromfile(config_path)
        dataset_cfg = cfg.get("test_dataloader", {}).get("dataset", {})
        metainfo = dataset_cfg.get("metainfo") or cfg.get("metainfo") or {}
        classes = metainfo.get("classes") or cfg.get("class_name")
        if classes:
            return tuple(classes)
    except Exception as exc:  # noqa: BLE001
        print(
            f"Config import failed while reading classes; falling back to text parse: {exc}",
            file=sys.stderr)
    return tuple(parse_classes_from_config_text(Path(config_path)))


def parse_classes_from_config_text(config_path):
    text = config_path.read_text(encoding="utf-8")
    start = text.find("test_dataloader = dict")
    if start < 0:
        start = 0
    cls_pos = text.find("classes=[", start)
    if cls_pos < 0:
        cls_pos = text.find("classes = [", start)
    if cls_pos < 0:
        return DEFAULT_CLASSES
    open_pos = text.find("[", cls_pos)
    depth = 0
    close_pos = -1
    for idx in range(open_pos, len(text)):
        char = text[idx]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                close_pos = idx
                break
    if close_pos < 0:
        return DEFAULT_CLASSES
    try:
        classes = ast.literal_eval(text[open_pos:close_pos + 1])
    except (SyntaxError, ValueError):
        return DEFAULT_CLASSES
    return tuple(classes) if classes else DEFAULT_CLASSES


def parse_variant(raw):
    if ":" not in raw:
        raise ValueError(f"--variant must be name:path, got {raw}")
    name, path = raw.split(":", 1)
    if not name:
        raise ValueError(f"empty variant name in {raw}")
    return name, Path(path)


def canonical_img_id(img_id):
    img_id = str(img_id)
    if img_id.startswith("angle_") and "__" in img_id:
        return img_id.split("__", 1)[1]
    return img_id


def qboxes_to_rboxes(qboxes):
    if len(qboxes) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    qbox_tensor = torch.from_numpy(np.asarray(qboxes, dtype=np.float32))
    return qbox2rbox(qbox_tensor).numpy().astype(np.float32)


def boxes_to_rboxes(boxes):
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    boxes = boxes.reshape(boxes.shape[0], -1)
    if boxes.shape[1] == 5:
        return boxes.astype(np.float32)
    if boxes.shape[1] == 8:
        return qboxes_to_rboxes(boxes)
    raise ValueError(f"Unsupported bbox shape: {boxes.shape}")


def box_area_rbox(box):
    return max(float(box[2]), 0.0) * max(float(box[3]), 0.0)


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


def read_gt(ann_file, class_to_label, diff_thr):
    rows = []
    if not ann_file.exists():
        return rows
    for gt_index, raw in enumerate(ann_file.read_text().splitlines()):
        parts = raw.split()
        if len(parts) < 9:
            continue
        class_name = parts[8]
        if class_name not in class_to_label:
            continue
        difficulty = int(parts[9]) if len(parts) > 9 else 0
        if difficulty > diff_thr:
            continue
        qbox = np.asarray([float(x) for x in parts[:8]], dtype=np.float32)
        rows.append({
            "gt_index": gt_index,
            "label": class_to_label[class_name],
            "class_name": class_name,
            "rbox": qboxes_to_rboxes(qbox.reshape(1, 8))[0],
        })
    return rows


def get_img_id(sample):
    img_id = optional_field(sample, "img_id")
    if img_id is not None:
        return str(img_id)
    metainfo = optional_field(sample, "metainfo", {}) or {}
    return str(metainfo.get("img_id"))


def get_pred_arrays(sample, max_dets_per_img):
    pred_instances = field(sample, "pred_instances")
    boxes = boxes_to_rboxes(tensor_to_numpy(field(pred_instances, "bboxes")))
    labels = tensor_to_numpy(field(pred_instances, "labels")).astype(np.int64)
    scores = tensor_to_numpy(field(pred_instances, "scores")).astype(np.float32)
    if max_dets_per_img > 0 and len(scores) > max_dets_per_img:
        order = np.argsort(-scores)[:max_dets_per_img]
        boxes = boxes[order]
        labels = labels[order]
        scores = scores[order]
    return boxes, labels, scores


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
    med = prior["median_area"]
    ratio = max(area / med, med / area) if med > 0 else 0.0
    return {
        "log_area_z": logz,
        "scale_ratio": ratio,
        "logz_implausible": logz >= z_thr,
        "ratio_implausible": ratio >= ratio_thr,
        "p0199_outlier": area < prior["p01_area"] or area > prior["p99_area"],
    }


def pct(values, q):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


class EceAccumulator:
    def __init__(self, bins):
        self.bins = bins
        self.stats = [
            {"count": 0, "score_sum": 0.0, "correct_sum": 0.0}
            for _ in range(len(bins) - 1)
        ]

    def add(self, score, correct):
        for idx, (lo, hi) in enumerate(zip(self.bins[:-1], self.bins[1:])):
            if lo <= score < hi:
                self.stats[idx]["count"] += 1
                self.stats[idx]["score_sum"] += float(score)
                self.stats[idx]["correct_sum"] += float(correct)
                return

    def rows(self, variant, subset):
        total = sum(item["count"] for item in self.stats)
        rows = []
        ece = 0.0
        for idx, item in enumerate(self.stats):
            count = item["count"]
            lo = self.bins[idx]
            hi = self.bins[idx + 1]
            avg_score = item["score_sum"] / count if count else 0.0
            accuracy = item["correct_sum"] / count if count else 0.0
            gap = abs(avg_score - accuracy) if count else 0.0
            weight = count / total if total else 0.0
            ece += weight * gap
            rows.append({
                "variant": variant,
                "subset": subset,
                "bin": f"[{lo},{hi})",
                "count": count,
                "avg_score": avg_score,
                "accuracy": accuracy,
                "abs_gap": gap,
                "weight": weight,
            })
        return ece, rows


def label_name(label, class_names):
    label = int(label)
    if 0 <= label < len(class_names):
        return class_names[label]
    return f"label_{label}"


def init_threshold_counter(score_thrs):
    out = {}
    for thr in score_thrs:
        tag = threshold_tag(thr)
        out[f"score_ge_{tag}"] = 0
    return out


def threshold_tag(thr):
    return str(thr).replace(".", "p")


def add_threshold_counts(prefix, summary, scores, score_thrs):
    for thr in score_thrs:
        summary[f"{prefix}_score_ge_{threshold_tag(thr)}"] = sum(
            score >= thr for score in scores)


def summarize_variant(name, pred_path, ann_dir, priors, class_names, args,
                      score_thrs, ece_bins, sibling_pairs, reference_records):
    with pred_path.open("rb") as f:
        predictions = pickle.load(f)

    class_to_label = {name: idx for idx, name in enumerate(class_names)}
    stats = Counter()
    wrong_scores = []
    wrong_excl_sibling_scores = []
    sise_logz_scores = []
    sise_p0199_scores = []
    correct_scores = []
    correct_logz_scores = []
    pair_stats = defaultdict(lambda: {
        "count": 0,
        "scores": [],
        "logz": [],
        "ratio": [],
        "logz_count": 0,
        "p0199_count": 0,
    })
    ece_all = EceAccumulator(ece_bins)
    ece_scale = EceAccumulator(ece_bins)
    ece_logz = EceAccumulator(ece_bins)
    variant_records = {}

    for image_idx, sample in enumerate(predictions):
        if args.max_images > 0 and image_idx >= args.max_images:
            break
        stats["images_scanned"] += 1
        img_id = get_img_id(sample)
        ann_file = ann_dir / f"{img_id}.txt"
        if not ann_file.exists():
            ann_file = ann_dir / f"{canonical_img_id(img_id)}.txt"
        gt_rows = read_gt(ann_file, class_to_label, args.diff_thr)
        if not ann_file.exists():
            stats["images_missing_ann"] += 1
        pred_boxes, pred_labels, pred_scores = get_pred_arrays(
            sample, args.max_dets_per_img)
        stats["detections_checked"] += len(pred_scores)
        stats["gt_objects_checked"] += len(gt_rows)
        if len(pred_scores) == 0 or len(gt_rows) == 0:
            continue

        gt_boxes = np.asarray([row["rbox"] for row in gt_rows], dtype=np.float32)
        ious = box_iou_rotated(
            torch.from_numpy(pred_boxes.astype(np.float32)),
            torch.from_numpy(gt_boxes)).cpu().numpy()
        best_gt = ious.argmax(axis=1)
        best_iou = ious[np.arange(ious.shape[0]), best_gt]

        for det_idx, gt_idx in enumerate(best_gt):
            if best_iou[det_idx] <= args.iou_thr:
                continue
            gt = gt_rows[int(gt_idx)]
            pred_label = int(pred_labels[det_idx])
            gt_label = int(gt["label"])
            pred_class = label_name(pred_label, class_names)
            gt_class = gt["class_name"]
            score = float(pred_scores[det_idx])
            area = box_area_rbox(pred_boxes[det_idx])
            feats = scale_features(
                area, pred_class, priors, args.z_thr, args.scale_ratio_thr)
            correct = pred_label == gt_label
            pair = f"{pred_class}->{gt_class}"
            is_sibling = pair in sibling_pairs
            stats["localized_total"] += 1
            ece_all.add(score, correct)
            if feats["p0199_outlier"] or feats["logz_implausible"]:
                ece_scale.add(score, correct)
            if feats["logz_implausible"]:
                ece_logz.add(score, correct)

            record = {
                "score": score,
                "correct": correct,
                "pair": pair,
                "pred_class": pred_class,
                "gt_class": gt_class,
                "log_area_z": feats["log_area_z"],
                "scale_ratio": feats["scale_ratio"],
                "logz_implausible": feats["logz_implausible"],
                "p0199_outlier": feats["p0199_outlier"],
                "is_sibling": is_sibling,
            }
            variant_records[(img_id, int(det_idx))] = record

            if correct:
                stats["localized_correct"] += 1
                correct_scores.append(score)
                if feats["logz_implausible"]:
                    correct_logz_scores.append(score)
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
            pstats = pair_stats[pair]
            pstats["count"] += 1
            pstats["scores"].append(score)
            pstats["logz"].append(feats["log_area_z"])
            pstats["ratio"].append(feats["scale_ratio"])
            if feats["logz_implausible"]:
                pstats["logz_count"] += 1
            if feats["p0199_outlier"]:
                pstats["p0199_count"] += 1

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
        "correct_logz_implausible": len(correct_logz_scores),
        "localized_accuracy": (
            stats["localized_correct"] / stats["localized_total"]
            if stats["localized_total"] else 0.0),
    }
    add_threshold_counts("wrong", summary, wrong_scores, score_thrs)
    add_threshold_counts(
        "wrong_excl_sibling", summary, wrong_excl_sibling_scores, score_thrs)
    add_threshold_counts("sise_logz", summary, sise_logz_scores, score_thrs)
    add_threshold_counts("sise_p0199", summary, sise_p0199_scores, score_thrs)
    add_threshold_counts("correct", summary, correct_scores, score_thrs)
    add_threshold_counts("correct_logz", summary, correct_logz_scores, score_thrs)

    all_ece, ece_rows = ece_all.rows(name, "localized_all")
    scale_ece, scale_rows = ece_scale.rows(name, "scale_implausible")
    logz_ece, logz_rows = ece_logz.rows(name, "logz_implausible")
    summary["si_ece_localized_all"] = all_ece
    summary["si_ece_scale_implausible"] = scale_ece
    summary["si_ece_logz_implausible"] = logz_ece
    ece_rows.extend(scale_rows)
    ece_rows.extend(logz_rows)

    pair_rows = []
    for pair, pstats in pair_stats.items():
        row = {
            "variant": name,
            "pair": pair,
            "pred_class": pair.split("->", 1)[0],
            "gt_class": pair.split("->", 1)[1],
            "count": pstats["count"],
            "logz_sise_count": pstats["logz_count"],
            "p0199_sise_count": pstats["p0199_count"],
            "median_score": pct(pstats["scores"], 50),
            "median_log_area_z": pct(pstats["logz"], 50),
            "median_scale_ratio": pct(pstats["ratio"], 50),
        }
        for thr in score_thrs:
            row[f"score_ge_{threshold_tag(thr)}"] = sum(
                score >= thr for score in pstats["scores"])
        pair_rows.append(row)
    pair_rows.sort(key=lambda r: (-r["count"], r["pair"]))

    if reference_records is not None:
        changed_correct = 0
        changed_wrong = 0
        missing = 0
        threshold_drops = defaultdict(int)
        for key, ref in reference_records.items():
            cur = variant_records.get(key)
            if cur is None:
                missing += 1
                continue
            if abs(ref["score"] - cur["score"]) <= args.score_change_eps:
                continue
            if ref["correct"]:
                changed_correct += 1
            else:
                changed_wrong += 1
            for thr in score_thrs:
                if ref["score"] >= thr and cur["score"] < thr:
                    threshold_drops[threshold_tag(thr)] += 1
        summary["paired_reference_records"] = len(reference_records)
        summary["paired_missing_records"] = missing
        summary["paired_changed_correct"] = changed_correct
        summary["paired_changed_wrong"] = changed_wrong
        for thr in score_thrs:
            summary[f"paired_score_drop_below_{threshold_tag(thr)}"] = (
                threshold_drops[threshold_tag(thr)])

    return summary, pair_rows, ece_rows, variant_records


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows, columns):
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_report(path, summaries, pair_rows, args, score_thrs):
    main_thr = threshold_tag(max(score_thrs))
    cols = [
        "variant", "localized_wrong", "localized_wrong_excl_sibling",
        f"sise_logz_score_ge_{main_thr}",
        f"sise_p0199_score_ge_{main_thr}",
        "correct_logz_implausible", "si_ece_logz_implausible",
        "paired_changed_correct",
    ]
    pair_cols = [
        "variant", "pair", "count", f"score_ge_{main_thr}",
        "logz_sise_count", "p0199_sise_count", "median_log_area_z",
        "median_scale_ratio",
    ]
    top_pairs = sorted(
        pair_rows,
        key=lambda r: (-r.get(f"score_ge_{main_thr}", 0), r["variant"], r["pair"]))[:30]
    lines = [
        "# SISE Score Calibration Evaluation",
        "",
        f"`iou_thr={args.iou_thr}`, `z_thr={args.z_thr}`, "
        f"`scale_ratio_thr={args.scale_ratio_thr}`, "
        f"`max_dets_per_img={args.max_dets_per_img}`, "
        f"`max_images={args.max_images}`。",
        "",
        "## Variant Summary",
        "",
        markdown_table(summaries, cols),
        "",
        "## Top High-Confidence Wrong Pairs",
        "",
        markdown_table(top_pairs, pair_cols),
        "",
        "说明：`sise_logz_*` 使用预测类别的 source-excluded log-area z 分数定义，"
        "`small-vehicle<->large-vehicle` sibling pair 默认从 SISE 主指标中排除。",
    ]
    Path(path).write_text("\n".join(lines) + os.linesep, encoding="utf-8")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = [parse_variant(raw) for raw in args.variant]
    score_thrs = [float(item) for item in args.score_thrs.split(",") if item]
    ece_bins = [float(item) for item in args.ece_bins.split(",") if item]
    sibling_pairs = {
        item for item in args.exclude_sibling_pairs.split(",") if item.strip()
    }
    class_names = load_class_names(args.config)
    priors = load_priors(args.class_area_priors_csv)

    summaries = []
    all_pair_rows = []
    all_ece_rows = []
    reference_records = None
    for index, (name, pred_path) in enumerate(variants):
        summary, pair_rows, ece_rows, records = summarize_variant(
            name, pred_path, Path(args.ann_dir), priors, class_names, args,
            score_thrs, ece_bins, sibling_pairs,
            None if index == 0 else reference_records)
        summaries.append(summary)
        all_pair_rows.extend(pair_rows)
        all_ece_rows.extend(ece_rows)
        if index == 0:
            reference_records = records

    summary_json = out_dir / "sise_score_calibration_summary.json"
    pair_csv = out_dir / "sise_pair_summary.csv"
    ece_csv = out_dir / "sise_ece_bins.csv"
    variant_csv = out_dir / "sise_variant_summary.csv"
    report_md = out_dir / "sise_score_calibration_report.md"

    write_csv(variant_csv, summaries)
    write_csv(pair_csv, all_pair_rows)
    write_csv(ece_csv, all_ece_rows)
    summary_payload = {
        "ann_dir": args.ann_dir,
        "class_area_priors_csv": args.class_area_priors_csv,
        "variants": [{"name": name, "path": str(path)} for name, path in variants],
        "max_dets_per_img": args.max_dets_per_img,
        "max_images": args.max_images,
        "score_thrs": score_thrs,
        "ece_bins": ece_bins,
        "sibling_pairs_excluded_from_sise": sorted(sibling_pairs),
        "outputs": {
            "variant_summary_csv": str(variant_csv),
            "pair_summary_csv": str(pair_csv),
            "ece_bins_csv": str(ece_csv),
            "report_md": str(report_md),
        },
        "summaries": summaries,
    }
    summary_json.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False) + os.linesep,
        encoding="utf-8")
    write_report(report_md, summaries, all_pair_rows, args, score_thrs)
    print(json.dumps(summary_payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
