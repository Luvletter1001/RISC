#!/usr/bin/env python
"""Compute deployment-facing SISE risk metrics from saved detections.

This complements ``evaluate_sise_score_calibration.py``.  The older evaluator
answers whether semantic-scale inconsistent localized errors exist; this script
answers whether they affect high-confidence deployment surfaces such as fixed
score thresholds and top-K review queues.
"""

import argparse
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

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from evaluate_sise_score_calibration import (  # noqa: E402
    DEFAULT_CLASSES,
    box_iou_rotated,
    box_area_rbox,
    canonical_img_id,
    get_img_id,
    get_pred_arrays,
    label_name,
    load_class_names,
    load_priors,
    markdown_table,
    parse_variant,
    read_gt,
    scale_features,
    threshold_tag,
    write_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute deployment-facing risk metrics for SISE.")
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="Variant in name:path format. The first variant is reference.")
    parser.add_argument("--ann-dir", required=True)
    parser.add_argument("--class-area-priors-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--diff-thr", type=int, default=100)
    parser.add_argument("--z-thr", type=float, default=4.0)
    parser.add_argument("--scale-ratio-thr", type=float, default=4.0)
    parser.add_argument("--score-thrs", default="0.5,0.7,0.9,0.99,0.999")
    parser.add_argument("--topk", default="100,500,1000,5000")
    parser.add_argument(
        "--exclude-sibling-pairs",
        default="small-vehicle->large-vehicle,large-vehicle->small-vehicle")
    parser.add_argument("--max-dets-per-img", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument(
        "--preselect-score-thr",
        type=float,
        default=None,
        help=(
            "Optional pre-IoU score filter. Use only for deployment-surface "
            "audits where lower-score detections are intentionally out of "
            "scope. Default keeps all detections."))
    parser.add_argument(
        "--preselect-topk-per-img",
        type=int,
        default=0,
        help=(
            "Optional pre-IoU per-image top-K filter after score filtering. "
            "Default 0 keeps all remaining detections."))
    parser.add_argument(
        "--export-localized-records",
        action="store_true",
        help="Write localized per-detection records for each variant.")
    parser.add_argument(
        "--rank-tail-audit",
        action="store_true",
        help="Write AP-sensitive rank/tail audit CSVs for the first two variants.")
    parser.add_argument(
        "--pairing-mode",
        default="aligned_det_idx",
        choices=["aligned_det_idx"],
        help="Detection pairing mode for rank-tail audit.")
    parser.add_argument("--rank-drop-tol", type=int, default=50)
    parser.add_argument("--score-floor", type=float, default=0.3)
    parser.add_argument("--tail-z-thr", type=float, default=4.0)
    parser.add_argument("--missing-penalty", type=float, default=1.0)
    return parser.parse_args()


def load_predictions(pred_path):
    with Path(pred_path).open("rb") as f:
        return pickle.load(f)


def preselect_prediction_arrays(boxes, labels, scores, args):
    original_indices = np.arange(len(scores), dtype=np.int64)
    if len(scores) == 0:
        return boxes, labels, scores, original_indices
    selected = np.ones(len(scores), dtype=bool)
    if args.preselect_score_thr is not None:
        selected &= scores >= args.preselect_score_thr
    selected_indices = original_indices[selected]
    if (args.preselect_topk_per_img > 0
            and len(selected_indices) > args.preselect_topk_per_img):
        order = np.argsort(-scores[selected_indices])[:args.preselect_topk_per_img]
        selected_indices = selected_indices[order]
    return (
        boxes[selected_indices],
        labels[selected_indices],
        scores[selected_indices],
        selected_indices,
    )


def collect_records(name, pred_path, ann_dir, priors, class_names, args,
                    sibling_pairs):
    predictions = load_predictions(pred_path)
    class_to_label = {class_name: idx for idx, class_name in enumerate(class_names)}
    stats = Counter()
    records = []
    pair_risk = defaultdict(lambda: {
        "count": 0,
        "risk_logz": 0.0,
        "risk_p0199": 0.0,
        "score_sum": 0.0,
        "logz_sum": 0.0,
    })

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
        stats["detections_loaded"] += len(pred_scores)
        pred_boxes, pred_labels, pred_scores, original_det_indices = (
            preselect_prediction_arrays(pred_boxes, pred_labels, pred_scores, args))
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
            pair = f"{pred_class}->{gt_class}"
            is_sibling = pair in sibling_pairs
            score = float(pred_scores[det_idx])
            area = box_area_rbox(pred_boxes[det_idx])
            feats = scale_features(
                area, pred_class, priors, args.z_thr, args.scale_ratio_thr)
            correct = pred_label == gt_label
            wrong_excl_sibling = (not correct) and (not is_sibling)
            logz_surprise = max(0.0, feats["log_area_z"] - args.z_thr)
            risk_logz = score * logz_surprise if wrong_excl_sibling else 0.0
            risk_p0199 = score if wrong_excl_sibling and feats["p0199_outlier"] else 0.0
            record = {
                "variant": name,
                "img_id": img_id,
                "det_idx": int(original_det_indices[det_idx]),
                "score": score,
                "correct": bool(correct),
                "wrong": bool(not correct),
                "wrong_excl_sibling": bool(wrong_excl_sibling),
                "pred_class": pred_class,
                "gt_class": gt_class,
                "pair": pair,
                "is_sibling": bool(is_sibling),
                "log_area_z": float(feats["log_area_z"]),
                "p0199_outlier": bool(feats["p0199_outlier"]),
                "logz_implausible": bool(feats["logz_implausible"]),
                "risk_logz": float(risk_logz),
                "risk_p0199": float(risk_p0199),
            }
            records.append(record)
            stats["localized_total"] += 1
            stats["localized_correct"] += int(correct)
            stats["localized_wrong"] += int(not correct)
            stats["localized_wrong_excl_sibling"] += int(wrong_excl_sibling)
            stats["sise_logz_wrong_excl_sibling"] += int(
                wrong_excl_sibling and feats["logz_implausible"])
            stats["sise_p0199_wrong_excl_sibling"] += int(
                wrong_excl_sibling and feats["p0199_outlier"])
            if wrong_excl_sibling:
                pair_item = pair_risk[pair]
                pair_item["count"] += 1
                pair_item["score_sum"] += score
                pair_item["logz_sum"] += float(feats["log_area_z"])
                pair_item["risk_logz"] += risk_logz
                pair_item["risk_p0199"] += risk_p0199

    return stats, records, pair_risk


def count_where(records, predicate):
    return sum(1 for record in records if predicate(record))


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def rank_weight(rank):
    return 1.0 / math.log2(2.0 + max(float(rank), 1.0))


def add_rank_fields(records):
    ranked_records = [dict(record) for record in records]

    def sort_key(record):
        return (
            -float(record.get("score", 0.0)),
            str(record.get("img_id", "")),
            int(record.get("det_idx", 0)),
            str(record.get("pred_class", "")),
        )

    for rank, record in enumerate(sorted(ranked_records, key=sort_key), start=1):
        record["global_review_rank"] = rank

    by_class = defaultdict(list)
    by_image = defaultdict(list)
    for record in ranked_records:
        by_class[record.get("pred_class", "")].append(record)
        by_image[record.get("img_id", "")].append(record)

    for class_records in by_class.values():
        for rank, record in enumerate(sorted(class_records, key=sort_key), start=1):
            record["class_rank"] = rank

    for image_records in by_image.values():
        for rank, record in enumerate(sorted(image_records, key=sort_key), start=1):
            record["image_rank"] = rank

    return ranked_records


def pairing_key(record):
    return (
        str(record.get("img_id", "")),
        int(record.get("det_idx", 0)),
        str(record.get("pred_class", "")),
    )


def pair_records_by_aligned_det_idx(base_records, method_records):
    method_by_key = {pairing_key(record): record for record in method_records}
    used_method_keys = set()
    pairs = []
    for base in base_records:
        key = pairing_key(base)
        method = method_by_key.get(key)
        if method is not None:
            used_method_keys.add(key)
        pair = {
            "img_id": base.get("img_id", ""),
            "det_idx": base.get("det_idx", -1),
            "pred_class": base.get("pred_class", ""),
            "gt_class": base.get("gt_class", ""),
            "pair": base.get("pair", ""),
            "base_variant": base.get("variant", ""),
            "method_variant": method.get("variant", "") if method else "",
        }
        for prefix, record in (("base", base), ("method", method)):
            for field in [
                    "score", "correct", "wrong_excl_sibling", "log_area_z",
                    "logz_implausible", "p0199_outlier", "risk_logz",
                    "risk_p0199", "class_rank", "global_review_rank",
                    "image_rank"
            ]:
                pair[f"{prefix}_{field}"] = (
                    record.get(field, "") if record is not None else "")
        pair["method_missing"] = method is None
        pairs.append(pair)

    stats = {
        "base_count": len(base_records),
        "method_count": len(method_records),
        "paired_count": len(used_method_keys),
        "base_unmatched_count": len(base_records) - len(used_method_keys),
        "method_unmatched_count": len(method_records) - len(used_method_keys),
    }
    return pairs, stats


def method_removed_or_demoted(pair, score_floor, rank_drop_tol):
    if pair.get("method_missing"):
        return True
    method_score = pair.get("method_score")
    if method_score == "" or float(method_score) < score_floor:
        return True
    base_rank = pair.get("base_class_rank")
    method_rank = pair.get("method_class_rank")
    if base_rank == "" or method_rank == "":
        return False
    return int(method_rank) - int(base_rank) > rank_drop_tol


def positive_rank_loss(pair, score_floor, rank_drop_tol, missing_penalty):
    base_rank = int(pair.get("base_class_rank") or 1)
    weight = rank_weight(base_rank)
    if pair.get("method_missing"):
        return missing_penalty * weight
    method_score = pair.get("method_score")
    if method_score == "" or float(method_score) < score_floor:
        return missing_penalty * weight
    method_rank = int(pair.get("method_class_rank") or base_rank)
    rank_delta = method_rank - base_rank
    if rank_delta > rank_drop_tol:
        return weight * safe_div(rank_delta, max(rank_drop_tol, 1))
    return 0.0


def positive_demote_gain(pair, score_floor, rank_drop_tol, missing_penalty):
    base_rank = int(pair.get("base_class_rank") or 1)
    weight = rank_weight(base_rank)
    if pair.get("method_missing"):
        return missing_penalty * weight
    method_score = pair.get("method_score")
    if method_score == "" or float(method_score) < score_floor:
        return missing_penalty * weight
    method_rank = int(pair.get("method_class_rank") or base_rank)
    rank_delta = method_rank - base_rank
    if rank_delta > rank_drop_tol:
        return weight * safe_div(rank_delta, max(rank_drop_tol, 1))
    return 0.0


def summarize_rank_tail_audit(
        base_name,
        method_name,
        paired_records,
        score_floor=0.3,
        rank_drop_tol=50,
        tail_z_thr=4.0,
        missing_penalty=1.0,
        pairing_stats=None):
    pairing_stats = pairing_stats or {}
    ap_sensitive_wrong = [
        pair for pair in paired_records
        if pair.get("base_wrong_excl_sibling") is True
        and float(pair.get("base_score") or 0.0) >= score_floor
    ]
    ap_sensitive_sise = [
        pair for pair in ap_sensitive_wrong
        if pair.get("base_logz_implausible") is True
    ]
    ap_removed = [
        pair for pair in ap_sensitive_sise
        if method_removed_or_demoted(pair, score_floor, rank_drop_tol)
    ]
    correct_pairs = [
        pair for pair in paired_records
        if pair.get("base_correct") is True
        and float(pair.get("base_score") or 0.0) >= score_floor
    ]
    correct_drops = [
        pair for pair in correct_pairs
        if method_removed_or_demoted(pair, score_floor, rank_drop_tol)
    ]
    tail_valid = [
        pair for pair in paired_records
        if pair.get("base_correct") is True
        and abs(float(pair.get("base_log_area_z") or 0.0)) >= tail_z_thr
    ]
    lost_tail = [
        pair for pair in tail_valid
        if method_removed_or_demoted(pair, score_floor, rank_drop_tol)
    ]
    tp_harm = sum(
        positive_rank_loss(pair, score_floor, rank_drop_tol, missing_penalty)
        for pair in correct_drops)
    fp_benefit = sum(
        positive_demote_gain(pair, score_floor, rank_drop_tol, missing_penalty)
        for pair in ap_removed)
    tail_loss = safe_div(len(lost_tail), len(tail_valid))
    return {
        "base_variant": base_name,
        "method_variant": method_name,
        "pairing_mode": "aligned_det_idx",
        "paired_count": int(pairing_stats.get("paired_count", len(paired_records))),
        "base_unmatched_count": int(pairing_stats.get("base_unmatched_count", 0)),
        "method_unmatched_count": int(pairing_stats.get("method_unmatched_count", 0)),
        "score_floor": score_floor,
        "rank_drop_tol": rank_drop_tol,
        "tail_z_thr": tail_z_thr,
        "ap_sensitive_wrong_excl_sibling_count": len(ap_sensitive_wrong),
        "ap_sensitive_sise_count": len(ap_sensitive_sise),
        "ap_sensitive_sise_rate": safe_div(
            len(ap_sensitive_sise), len(ap_sensitive_wrong)),
        "ap_contributing_fp_removed": len(ap_removed),
        "correct_rank_drop_count": len(correct_drops),
        "tp_rank_harm": tp_harm,
        "sise_fp_rank_benefit": fp_benefit,
        "rank_disruptive_delta": tp_harm - fp_benefit,
        "tail_valid_tp": len(tail_valid),
        "lost_tail_valid_tp": len(lost_tail),
        "tail_valid_object_loss": tail_loss,
        "tail_safety_status": (
            "TAIL_SAFETY_UNDERPOWERED" if not tail_valid else
            ("PASS" if tail_loss <= 0.001 else "FAIL")),
    }


def safe_output_name(name):
    return "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(name))


def summarize_thresholds(name, records, images_scanned, score_thrs):
    rows = []
    for thr in score_thrs:
        selected = [record for record in records if record["score"] >= thr]
        total = len(selected)
        correct = count_where(selected, lambda r: r["correct"])
        wrong = count_where(selected, lambda r: r["wrong"])
        wrong_excl = count_where(selected, lambda r: r["wrong_excl_sibling"])
        sise_logz = count_where(
            selected,
            lambda r: r["wrong_excl_sibling"] and r["logz_implausible"])
        sise_p0199 = count_where(
            selected,
            lambda r: r["wrong_excl_sibling"] and r["p0199_outlier"])
        risk_logz = sum(record["risk_logz"] for record in selected)
        risk_p0199 = sum(record["risk_p0199"] for record in selected)
        rows.append({
            "variant": name,
            "score_thr": thr,
            "localized_ge_thr": total,
            "correct_ge_thr": correct,
            "wrong_ge_thr": wrong,
            "wrong_excl_sibling_ge_thr": wrong_excl,
            "sise_logz_ge_thr": sise_logz,
            "sise_p0199_ge_thr": sise_p0199,
            "fixed_score_precision": safe_div(correct, correct + wrong),
            "hc_far_per_image": safe_div(wrong_excl, images_scanned),
            "sise_logz_far_per_image": safe_div(sise_logz, images_scanned),
            "risk_weighted_logz_false_alarm": risk_logz,
            "risk_weighted_p0199_false_alarm": risk_p0199,
        })
    return rows


def summarize_topk(name, records, topks):
    rows = []
    ranked = sorted(records, key=lambda record: -record["score"])
    for k in topks:
        selected = ranked[:k]
        total = len(selected)
        correct = count_where(selected, lambda r: r["correct"])
        wrong_excl = count_where(selected, lambda r: r["wrong_excl_sibling"])
        sise_logz = count_where(
            selected,
            lambda r: r["wrong_excl_sibling"] and r["logz_implausible"])
        sise_p0199 = count_where(
            selected,
            lambda r: r["wrong_excl_sibling"] and r["p0199_outlier"])
        risk_logz = sum(record["risk_logz"] for record in selected)
        risk_p0199 = sum(record["risk_p0199"] for record in selected)
        rows.append({
            "variant": name,
            "topk": k,
            "effective_k": total,
            "min_score": selected[-1]["score"] if selected else 0.0,
            "correct_topk": correct,
            "wrong_excl_sibling_topk": wrong_excl,
            "sise_logz_topk": sise_logz,
            "sise_p0199_topk": sise_p0199,
            "topk_precision": safe_div(correct, total),
            "topk_sise_logz_rate": safe_div(sise_logz, total),
            "topk_sise_p0199_rate": safe_div(sise_p0199, total),
            "risk_weighted_logz_false_alarm": risk_logz,
            "risk_weighted_p0199_false_alarm": risk_p0199,
        })
    return rows


def summarize_fixed_recall(variant_records, variant_stats, score_thrs):
    """Measure precision at reference localized-recall operating points.

    The first variant is the reference. For each reference score threshold, use
    the reference localized-correct count as the target. Every candidate variant
    is swept by descending score until it reaches the same correct count. This
    is not COCO/DOTA AP recall; it is a deployment operating-point diagnostic
    over localized detections, meant to test whether SISE reduction improves
    precision when the useful correct-detection budget is held fixed.
    """
    if not variant_records:
        return []

    ref_name = next(iter(variant_records))
    ref_records = variant_records[ref_name]
    ref_gt_total = int(variant_stats[ref_name]["gt_objects_checked"])
    targets = []
    for thr in score_thrs:
        selected = [record for record in ref_records if record["score"] >= thr]
        target_correct = count_where(selected, lambda r: r["correct"])
        target_total = len(selected)
        target_sise_logz = count_where(
            selected,
            lambda r: r["wrong_excl_sibling"] and r["logz_implausible"])
        target_risk_logz = sum(record["risk_logz"] for record in selected)
        targets.append({
            "reference_score_thr": thr,
            "target_correct": target_correct,
            "target_total": target_total,
            "target_precision": safe_div(target_correct, target_total),
            "target_localized_recall": safe_div(target_correct, ref_gt_total),
            "target_sise_logz": target_sise_logz,
            "target_risk_logz": target_risk_logz,
        })

    rows = []
    for target in targets:
        for name, records in variant_records.items():
            if name == ref_name:
                selected = [
                    record for record in records
                    if record["score"] >= target["reference_score_thr"]
                ]
                selected_total = len(selected)
                selected_correct = target["target_correct"]
                reached_target = True
                operating_score = target["reference_score_thr"]
            else:
                selected = []
                selected_correct = 0
                reached_target = target["target_correct"] == 0
                if target["target_correct"] > 0:
                    ranked = sorted(records, key=lambda record: -record["score"])
                    for record in ranked:
                        selected.append(record)
                        selected_correct += int(record["correct"])
                        if selected_correct >= target["target_correct"]:
                            reached_target = True
                            break
                selected_total = len(selected)
                operating_score = selected[-1]["score"] if selected else 0.0

            selected_wrong_excl = count_where(
                selected, lambda r: r["wrong_excl_sibling"])
            selected_sise_logz = count_where(
                selected,
                lambda r: r["wrong_excl_sibling"] and r["logz_implausible"])
            selected_risk_logz = sum(record["risk_logz"] for record in selected)
            precision = safe_div(selected_correct, selected_total)
            rows.append({
                "variant": name,
                "reference_variant": ref_name,
                "reference_score_thr": target["reference_score_thr"],
                "target_correct": target["target_correct"],
                "target_total": target["target_total"],
                "target_precision": target["target_precision"],
                "target_localized_recall": target["target_localized_recall"],
                "reached_target_correct": reached_target,
                "operating_score_thr": operating_score,
                "selected_total": selected_total,
                "selected_correct": selected_correct,
                "selected_wrong_excl_sibling": selected_wrong_excl,
                "selected_sise_logz": selected_sise_logz,
                "fixed_recall_precision": precision,
                "fixed_recall_precision_delta_vs_ref": (
                    precision - target["target_precision"]),
                "fixed_recall_sise_logz_delta_vs_ref": (
                    selected_sise_logz - target["target_sise_logz"]),
                "risk_weighted_logz_false_alarm": selected_risk_logz,
                "risk_weighted_logz_delta_vs_ref": (
                    selected_risk_logz - target["target_risk_logz"]),
            })
    return rows


def summarize_variant(name, pred_path, stats, records, pair_risk, score_thrs,
                      topks):
    images_scanned = int(stats["images_scanned"])
    summary = {
        "variant": name,
        "predictions": str(pred_path),
        "images_scanned": images_scanned,
        "images_missing_ann": int(stats["images_missing_ann"]),
        "detections_loaded": int(stats["detections_loaded"]),
        "detections_checked": int(stats["detections_checked"]),
        "gt_objects_checked": int(stats["gt_objects_checked"]),
        "localized_total": int(stats["localized_total"]),
        "localized_correct": int(stats["localized_correct"]),
        "localized_wrong": int(stats["localized_wrong"]),
        "localized_wrong_excl_sibling": int(stats["localized_wrong_excl_sibling"]),
        "sise_logz_wrong_excl_sibling": int(
            stats["sise_logz_wrong_excl_sibling"]),
        "sise_p0199_wrong_excl_sibling": int(
            stats["sise_p0199_wrong_excl_sibling"]),
        "localized_precision": safe_div(
            stats["localized_correct"], stats["localized_total"]),
        "risk_weighted_logz_false_alarm": sum(
            record["risk_logz"] for record in records),
        "risk_weighted_p0199_false_alarm": sum(
            record["risk_p0199"] for record in records),
    }
    threshold_rows = summarize_thresholds(
        name, records, images_scanned, score_thrs)
    topk_rows = summarize_topk(name, records, topks)
    for row in threshold_rows:
        tag = threshold_tag(row["score_thr"])
        for key, value in row.items():
            if key in {"variant", "score_thr"}:
                continue
            summary[f"{key}_score_ge_{tag}"] = value
    for row in topk_rows:
        k = row["topk"]
        for key, value in row.items():
            if key in {"variant", "topk"}:
                continue
            summary[f"{key}_top{k}"] = value

    pair_rows = []
    for pair, item in pair_risk.items():
        count = item["count"]
        pair_rows.append({
            "variant": name,
            "pair": pair,
            "count": count,
            "avg_score": safe_div(item["score_sum"], count),
            "avg_log_area_z": safe_div(item["logz_sum"], count),
            "risk_weighted_logz_false_alarm": item["risk_logz"],
            "risk_weighted_p0199_false_alarm": item["risk_p0199"],
        })
    pair_rows.sort(
        key=lambda row: (-row["risk_weighted_logz_false_alarm"], -row["count"]))
    return summary, threshold_rows, topk_rows, pair_rows


def add_reference_deltas(rows, key_fields, metric_fields):
    ref_by_key = {}
    for row in rows:
        if not ref_by_key:
            pass
        key = tuple(row[field] for field in key_fields)
        if key not in ref_by_key:
            ref_by_key[key] = row
            continue
        ref = ref_by_key[key]
        for field in metric_fields:
            row[f"{field}_delta_vs_ref"] = row.get(field, 0) - ref.get(field, 0)
            if ref.get(field, 0):
                row[f"{field}_rate_vs_ref"] = safe_div(row.get(field, 0), ref[field])
    return rows


def write_report(path, summaries, threshold_rows, topk_rows, fixed_recall_rows,
                 pair_rows, args):
    threshold_cols = [
        "variant", "score_thr", "fixed_score_precision", "hc_far_per_image",
        "sise_logz_far_per_image", "wrong_excl_sibling_ge_thr",
        "sise_logz_ge_thr", "risk_weighted_logz_false_alarm",
    ]
    topk_cols = [
        "variant", "topk", "topk_precision", "topk_sise_logz_rate",
        "correct_topk", "wrong_excl_sibling_topk", "sise_logz_topk",
        "risk_weighted_logz_false_alarm",
    ]
    fixed_recall_cols = [
        "variant", "reference_score_thr", "target_correct",
        "target_localized_recall", "operating_score_thr",
        "fixed_recall_precision", "fixed_recall_precision_delta_vs_ref",
        "selected_sise_logz", "fixed_recall_sise_logz_delta_vs_ref",
    ]
    summary_cols = [
        "variant", "localized_precision", "localized_wrong_excl_sibling",
        "sise_logz_wrong_excl_sibling", "sise_p0199_wrong_excl_sibling",
        "risk_weighted_logz_false_alarm",
    ]
    pair_cols = [
        "variant", "pair", "count", "avg_score", "avg_log_area_z",
        "risk_weighted_logz_false_alarm",
    ]
    top_pairs = sorted(
        pair_rows,
        key=lambda row: -row["risk_weighted_logz_false_alarm"])[:30]
    lines = [
        "# SISE Deployment Risk Evaluation",
        "",
        f"`iou_thr={args.iou_thr}`, `z_thr={args.z_thr}`, "
        f"`max_dets_per_img={args.max_dets_per_img}`, "
        f"`max_images={args.max_images}`, "
        f"`preselect_score_thr={args.preselect_score_thr}`, "
        f"`preselect_topk_per_img={args.preselect_topk_per_img}`。",
        "",
        "## Variant Summary",
        "",
        markdown_table(summaries, summary_cols),
        "",
        "## Fixed Score Deployment Metrics",
        "",
        markdown_table(threshold_rows, threshold_cols),
        "",
        "## Top-K Review Queue Metrics",
        "",
        markdown_table(topk_rows, topk_cols),
        "",
        "## Fixed-Recall Precision Metrics",
        "",
        markdown_table(fixed_recall_rows, fixed_recall_cols),
        "",
        "## Top Risk Pairs",
        "",
        markdown_table(top_pairs, pair_cols),
        "",
        "说明：这些指标只统计 `IoU > iou_thr` 的 localized detections；"
        "`HC-FAR` 是 fixed score 阈值下每图 localized wrong excl sibling 数量，"
        "`TopK-SISE` 反映最高置信审阅队列中的 semantic-scale 风险；"
        "`Fixed-Recall Precision` 使用 reference variant 在固定 score 阈值下"
        "的 localized correct count 作为目标，比较候选方法达到同等正确数时"
        "需要保留多少预测以及对应 precision。",
    ]
    Path(path).write_text("\n".join(lines) + os.linesep, encoding="utf-8")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = [parse_variant(raw) for raw in args.variant]
    score_thrs = [float(item) for item in args.score_thrs.split(",") if item]
    topks = [int(item) for item in args.topk.split(",") if item]
    sibling_pairs = {
        item for item in args.exclude_sibling_pairs.split(",") if item.strip()
    }
    class_names = load_class_names(args.config) if args.config else DEFAULT_CLASSES
    priors = load_priors(args.class_area_priors_csv)

    summaries = []
    all_threshold_rows = []
    all_topk_rows = []
    variant_records = {}
    variant_stats = {}
    all_pair_rows = []
    for name, pred_path in variants:
        stats, records, pair_risk = collect_records(
            name, pred_path, Path(args.ann_dir), priors, class_names, args,
            sibling_pairs)
        records = add_rank_fields(records)
        summary, threshold_rows, topk_rows, pair_rows = summarize_variant(
            name, pred_path, stats, records, pair_risk, score_thrs, topks)
        summaries.append(summary)
        all_threshold_rows.extend(threshold_rows)
        all_topk_rows.extend(topk_rows)
        variant_records[name] = records
        variant_stats[name] = stats
        all_pair_rows.extend(pair_rows)

    threshold_metric_fields = [
        "fixed_score_precision", "hc_far_per_image", "sise_logz_far_per_image",
        "wrong_excl_sibling_ge_thr", "sise_logz_ge_thr",
        "risk_weighted_logz_false_alarm",
    ]
    topk_metric_fields = [
        "topk_precision", "topk_sise_logz_rate", "correct_topk",
        "wrong_excl_sibling_topk", "sise_logz_topk",
        "risk_weighted_logz_false_alarm",
    ]
    add_reference_deltas(all_threshold_rows, ["score_thr"], threshold_metric_fields)
    add_reference_deltas(all_topk_rows, ["topk"], topk_metric_fields)
    fixed_recall_rows = summarize_fixed_recall(
        variant_records, variant_stats, score_thrs)

    outputs = {
        "summary_json": str(out_dir / "deployment_risk_summary.json"),
        "variant_summary_csv": str(out_dir / "deployment_risk_variant_summary.csv"),
        "threshold_metrics_csv": str(out_dir / "deployment_risk_thresholds.csv"),
        "topk_metrics_csv": str(out_dir / "deployment_risk_topk.csv"),
        "fixed_recall_metrics_csv": str(
            out_dir / "deployment_risk_fixed_recall.csv"),
        "pair_risk_csv": str(out_dir / "deployment_risk_pairs.csv"),
        "report_md": str(out_dir / "deployment_risk_report.md"),
    }
    localized_outputs = {}
    if args.export_localized_records:
        for name, records in variant_records.items():
            path = out_dir / f"localized_records_{safe_output_name(name)}.csv"
            write_csv(path, records)
            localized_outputs[name] = str(path)
        outputs["localized_records_csv"] = localized_outputs

    rank_tail_outputs = {}
    rank_tail_summary_rows = []
    if args.rank_tail_audit:
        if len(variants) < 2:
            raise ValueError("--rank-tail-audit requires at least two variants")
        base_name = variants[0][0]
        method_name = variants[1][0]
        if args.pairing_mode != "aligned_det_idx":
            raise ValueError(f"Unsupported pairing mode: {args.pairing_mode}")
        paired_records, pairing_stats = pair_records_by_aligned_det_idx(
            variant_records[base_name], variant_records[method_name])
        rank_tail_summary_rows = [
            summarize_rank_tail_audit(
                base_name,
                method_name,
                paired_records,
                score_floor=args.score_floor,
                rank_drop_tol=args.rank_drop_tol,
                tail_z_thr=args.tail_z_thr,
                missing_penalty=args.missing_penalty,
                pairing_stats=pairing_stats,
            )
        ]
        rank_tail_outputs = {
            "paired_records_csv": str(out_dir / "rank_tail_paired_records.csv"),
            "summary_csv": str(out_dir / "rank_tail_summary.csv"),
        }
        write_csv(rank_tail_outputs["paired_records_csv"], paired_records)
        write_csv(rank_tail_outputs["summary_csv"], rank_tail_summary_rows)
        outputs["rank_tail_audit"] = rank_tail_outputs

    write_csv(outputs["variant_summary_csv"], summaries)
    write_csv(outputs["threshold_metrics_csv"], all_threshold_rows)
    write_csv(outputs["topk_metrics_csv"], all_topk_rows)
    write_csv(outputs["fixed_recall_metrics_csv"], fixed_recall_rows)
    write_csv(outputs["pair_risk_csv"], all_pair_rows)
    write_report(
        outputs["report_md"], summaries, all_threshold_rows, all_topk_rows,
        fixed_recall_rows, all_pair_rows, args)
    payload = {
        "ann_dir": args.ann_dir,
        "class_area_priors_csv": args.class_area_priors_csv,
        "variants": [{"name": name, "path": str(path)} for name, path in variants],
        "score_thrs": score_thrs,
        "topk": topks,
        "preselect_score_thr": args.preselect_score_thr,
        "preselect_topk_per_img": args.preselect_topk_per_img,
        "rank_tail_audit": args.rank_tail_audit,
        "pairing_mode": args.pairing_mode,
        "score_floor": args.score_floor,
        "rank_drop_tol": args.rank_drop_tol,
        "tail_z_thr": args.tail_z_thr,
        "sibling_pairs_excluded_from_sise": sorted(sibling_pairs),
        "outputs": outputs,
        "summaries": summaries,
        "rank_tail_summary": rank_tail_summary_rows,
    }
    Path(outputs["summary_json"]).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + os.linesep,
        encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
