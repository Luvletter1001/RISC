#!/usr/bin/env python
"""Build score-only calibration baselines for the semantic-scale gate.

The goal is deliberately narrow: test whether ordinary detector score
calibration can explain the deployment-facing SISE gains.  The calibrators are
only allowed to remap detector confidence scores using correctness labels from
a held-out calibration split.  They do not use log-area, class-scale priors, or
Gaussian semantic-scale features.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_OUT_DIR = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "calibration_baselines")
DEFAULT_RESULT_MD = Path(
    "resultmd/exp_p4_scale_semantic_validation/"
    "fres_20260620_score_only_calibration_baseline_gate.md")

DATASETS = [
    {
        "dataset": "HRRSD closed-set",
        "domain": "closed_set",
        "baseline_csv": Path(
            "work_dirs/gs3c_sise_rank_tail_audit_20260619/"
            "hrrsd_g1_z4/localized_records_baseline.csv"),
        "method_csv": Path(
            "work_dirs/gs3c_sise_rank_tail_audit_20260619/"
            "hrrsd_g1_z4/localized_records_g1_z4.csv"),
        "method": "g1_z4",
        "topk": 5000,
    },
    {
        "dataset": "DIOR-R closed-set",
        "domain": "closed_set",
        "baseline_csv": Path(
            "work_dirs/gs3c_sise_rank_tail_audit_20260619/"
            "dior_g1_z5p5/localized_records_baseline.csv"),
        "method_csv": Path(
            "work_dirs/gs3c_sise_rank_tail_audit_20260619/"
            "dior_g1_z5p5/localized_records_g1_z5p5.csv"),
        "method": "g1_z5p5",
        "topk": 10000,
    },
    {
        "dataset": "SHIPRS closed-set",
        "domain": "closed_set",
        "baseline_csv": Path(
            "work_dirs/gs3c_sise_rank_tail_audit_20260619/"
            "shiprs_g2_fitted/localized_records_baseline.csv"),
        "method_csv": Path(
            "work_dirs/gs3c_sise_rank_tail_audit_20260619/"
            "shiprs_g2_fitted/localized_records_g2_fitted.csv"),
        "method": "g2_fitted",
        "topk": 5000,
    },
]

SCORE_THRS = (0.5, 0.7, 0.9, 0.99, 0.999)
TOPKS = (100, 500, 1000, 5000, 10000)
ECE_BINS = (0.0, 0.5, 0.7, 0.9, 0.99, 0.999, 1.000001)


def to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def to_float(value, default=0.0):
    try:
        if value in {None, ""}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp_prob(value, eps=1e-6):
    return min(max(float(value), eps), 1.0 - eps)


def logit(value):
    value = clamp_prob(value)
    return math.log(value / (1.0 - value))


def sigmoid(value):
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def split_key(img_id):
    digest = hashlib.md5(str(img_id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def load_records(path):
    records = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append({
                "variant": row.get("variant", ""),
                "img_id": row.get("img_id", ""),
                "det_idx": int(to_float(row.get("det_idx"), -1)),
                "score": clamp_prob(to_float(row.get("score"))),
                "correct": to_bool(row.get("correct")),
                "wrong_excl_sibling": to_bool(row.get("wrong_excl_sibling")),
                "pred_class": row.get("pred_class", ""),
                "gt_class": row.get("gt_class", ""),
                "pair": row.get("pair", ""),
                "log_area_z": to_float(row.get("log_area_z")),
                "logz_implausible": to_bool(row.get("logz_implausible")),
                "p0199_outlier": to_bool(row.get("p0199_outlier")),
            })
    return records


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


class IdentityCalibrator:
    name = "identity"

    def fit(self, records):
        return self

    def predict_one(self, record):
        return record["score"]


class PlattCalibrator:
    name = "global_platt"

    def __init__(self, epochs=700, lr=0.05, l2=1e-4):
        self.epochs = epochs
        self.lr = lr
        self.l2 = l2
        self.a = 1.0
        self.b = 0.0

    def fit(self, records):
        if not records:
            return self
        xs = [logit(record["score"]) for record in records]
        ys = [1.0 if record["correct"] else 0.0 for record in records]
        n = float(len(xs))
        for _ in range(self.epochs):
            grad_a = 0.0
            grad_b = 0.0
            for x, y in zip(xs, ys):
                p = sigmoid(self.a * x + self.b)
                grad_a += (p - y) * x
                grad_b += (p - y)
            grad_a = grad_a / n + self.l2 * (self.a - 1.0)
            grad_b = grad_b / n + self.l2 * self.b
            self.a -= self.lr * grad_a
            self.b -= self.lr * grad_b
            if self.a < 0.0:
                self.a = 0.0
            self.a = min(self.a, 20.0)
            self.b = min(max(self.b, -20.0), 20.0)
        return self

    def predict_one(self, record):
        return clamp_prob(sigmoid(self.a * logit(record["score"]) + self.b))


class ClasswisePlattCalibrator:
    name = "classwise_platt"

    def __init__(self, min_class_records=50):
        self.min_class_records = min_class_records
        self.global_model = PlattCalibrator()
        self.models = {}

    def fit(self, records):
        self.global_model.fit(records)
        by_class = defaultdict(list)
        for record in records:
            by_class[record["pred_class"]].append(record)
        for cls, cls_records in by_class.items():
            positives = sum(1 for record in cls_records if record["correct"])
            negatives = len(cls_records) - positives
            if (len(cls_records) >= self.min_class_records
                    and positives > 0 and negatives > 0):
                self.models[cls] = PlattCalibrator().fit(cls_records)
        return self

    def predict_one(self, record):
        model = self.models.get(record["pred_class"], self.global_model)
        return model.predict_one(record)


class IsotonicCalibrator:
    name = "global_isotonic"

    def __init__(self):
        self.thresholds = []
        self.values = []
        self.fallback = 0.5

    def fit(self, records):
        if not records:
            return self
        pairs = sorted(
            (record["score"], 1.0 if record["correct"] else 0.0)
            for record in records)
        self.fallback = sum(y for _, y in pairs) / float(len(pairs))
        blocks = []
        for score, y in pairs:
            blocks.append({
                "score_max": score,
                "sum": y,
                "count": 1.0,
            })
            while len(blocks) >= 2:
                prev = blocks[-2]
                cur = blocks[-1]
                if prev["sum"] / prev["count"] <= cur["sum"] / cur["count"]:
                    break
                prev["score_max"] = cur["score_max"]
                prev["sum"] += cur["sum"]
                prev["count"] += cur["count"]
                blocks.pop()
        self.thresholds = [block["score_max"] for block in blocks]
        self.values = [
            clamp_prob(block["sum"] / block["count"]) for block in blocks
        ]
        return self

    def predict_one(self, record):
        if not self.thresholds:
            return clamp_prob(self.fallback)
        score = record["score"]
        lo = 0
        hi = len(self.thresholds) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if score <= self.thresholds[mid]:
                hi = mid
            else:
                lo = mid + 1
        return self.values[lo]


def with_calibrated_scores(records, calibrator, variant_name):
    out = []
    for record in records:
        item = dict(record)
        item["variant"] = variant_name
        item["calibrated_score"] = calibrator.predict_one(record)
        out.append(item)
    return out


def sort_key(record, score_key):
    return (
        -float(record.get(score_key, 0.0)),
        str(record.get("img_id", "")),
        int(record.get("det_idx", 0)),
        str(record.get("pred_class", "")),
    )


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def summarize_records(records, variant, score_key="calibrated_score"):
    scores = [float(record.get(score_key, record["score"])) for record in records]
    correct_total = sum(1 for record in records if record["correct"])
    wrong_total = sum(1 for record in records if record["wrong_excl_sibling"])
    sise_total = sum(
        1 for record in records
        if record["wrong_excl_sibling"] and record["logz_implausible"])
    row = {
        "variant": variant,
        "localized_total": len(records),
        "localized_correct": correct_total,
        "wrong_excl_sibling": wrong_total,
        "sise_logz_wrong_excl_sibling": sise_total,
        "localized_precision": safe_div(correct_total, len(records)),
        "ece": ece(records, score_key),
        "avg_score": safe_div(sum(scores), len(scores)),
    }
    for thr in SCORE_THRS:
        tag = threshold_tag(thr)
        selected = [
            record for record, score in zip(records, scores)
            if score >= thr
        ]
        correct = sum(1 for record in selected if record["correct"])
        wrong = sum(1 for record in selected if record["wrong_excl_sibling"])
        sise = sum(
            1 for record in selected
            if record["wrong_excl_sibling"] and record["logz_implausible"])
        row[f"total_score_ge_{tag}"] = len(selected)
        row[f"correct_score_ge_{tag}"] = correct
        row[f"wrong_excl_sibling_score_ge_{tag}"] = wrong
        row[f"sise_logz_score_ge_{tag}"] = sise
        row[f"precision_score_ge_{tag}"] = safe_div(correct, len(selected))
    ranked = sorted(records, key=lambda record: sort_key(record, score_key))
    for topk in TOPKS:
        selected = ranked[:min(topk, len(ranked))]
        correct = sum(1 for record in selected if record["correct"])
        wrong = sum(1 for record in selected if record["wrong_excl_sibling"])
        sise = sum(
            1 for record in selected
            if record["wrong_excl_sibling"] and record["logz_implausible"])
        row[f"correct_top{topk}"] = correct
        row[f"wrong_excl_sibling_top{topk}"] = wrong
        row[f"sise_logz_top{topk}"] = sise
        row[f"precision_top{topk}"] = safe_div(correct, len(selected))
        row[f"sise_logz_rate_top{topk}"] = safe_div(sise, len(selected))
        row[f"min_score_top{topk}"] = (
            float(selected[-1].get(score_key, selected[-1]["score"]))
            if selected else 0.0)
    return row


def fixed_correct_metrics(records, target_correct, score_key):
    ranked = sorted(records, key=lambda record: sort_key(record, score_key))
    selected = []
    correct = 0
    for record in ranked:
        selected.append(record)
        correct += 1 if record["correct"] else 0
        if correct >= target_correct:
            break
    wrong = sum(1 for record in selected if record["wrong_excl_sibling"])
    sise = sum(
        1 for record in selected
        if record["wrong_excl_sibling"] and record["logz_implausible"])
    return {
        "fixed_correct_target": target_correct,
        "fixed_correct_reached": correct >= target_correct,
        "fixed_correct_selected_total": len(selected),
        "fixed_correct_selected_correct": correct,
        "fixed_correct_wrong_excl_sibling": wrong,
        "fixed_correct_sise_logz": sise,
        "fixed_correct_precision": safe_div(correct, len(selected)),
    }


def ece(records, score_key):
    if not records:
        return 0.0
    total = len(records)
    score_sum = [0.0 for _ in range(len(ECE_BINS) - 1)]
    correct_sum = [0.0 for _ in range(len(ECE_BINS) - 1)]
    count = [0 for _ in range(len(ECE_BINS) - 1)]
    for record in records:
        score = float(record.get(score_key, record["score"]))
        for idx, (lo, hi) in enumerate(zip(ECE_BINS[:-1], ECE_BINS[1:])):
            if lo <= score < hi:
                score_sum[idx] += score
                correct_sum[idx] += 1.0 if record["correct"] else 0.0
                count[idx] += 1
                break
    value = 0.0
    for idx, n in enumerate(count):
        if n == 0:
            continue
        value += (n / total) * abs(score_sum[idx] / n - correct_sum[idx] / n)
    return value


def threshold_tag(thr):
    return str(thr).replace(".", "p")


def delta(row, base, key):
    return float(row.get(key, 0.0)) - float(base.get(key, 0.0))


def split_records(records, train_frac):
    train = []
    eval_records = []
    for record in records:
        if split_key(record["img_id"]) < train_frac:
            train.append(record)
        else:
            eval_records.append(record)
    return train, eval_records


def summarize_dataset(spec, train_frac):
    baseline_path = spec["baseline_csv"]
    method_path = spec["method_csv"]
    if not baseline_path.exists() or not method_path.exists():
        return {
            "dataset": spec["dataset"],
            "status": "missing_csv",
            "baseline_csv": str(baseline_path),
            "method_csv": str(method_path),
        }, []

    baseline_records = load_records(baseline_path)
    method_records = load_records(method_path)
    train_records, eval_records = split_records(baseline_records, train_frac)
    _, method_eval_records = split_records(method_records, train_frac)

    calibrators = [
        IdentityCalibrator(),
        PlattCalibrator(),
        ClasswisePlattCalibrator(),
        IsotonicCalibrator(),
    ]

    rows = []
    variant_records = {}
    for calibrator in calibrators:
        calibrator.fit(train_records)
        calibrated_records = with_calibrated_scores(
            eval_records, calibrator, calibrator.name)
        variant_records[calibrator.name] = calibrated_records
        summary = summarize_records(
            calibrated_records, calibrator.name, "calibrated_score")
        summary.update({
            "dataset": spec["dataset"],
            "domain": spec["domain"],
            "baseline_csv": str(baseline_path),
            "method_csv": str(method_path),
            "train_records": len(train_records),
            "eval_records": len(eval_records),
            "calibrator_uses_scale_features": False,
        })
        rows.append(summary)

    method_eval = with_calibrated_scores(
        method_eval_records, IdentityCalibrator(), spec["method"])
    variant_records[spec["method"]] = method_eval
    method_summary = summarize_records(
        method_eval, spec["method"], "calibrated_score")
    method_summary.update({
        "dataset": spec["dataset"],
        "domain": spec["domain"],
        "baseline_csv": str(baseline_path),
        "method_csv": str(method_path),
        "train_records": len(train_records),
        "eval_records": len(method_eval_records),
        "calibrator_uses_scale_features": True,
    })
    rows.append(method_summary)

    identity = next(row for row in rows if row["variant"] == "identity")
    topk = spec["topk"]
    topk_key = f"sise_logz_top{topk}"
    precision_key = f"precision_top{topk}"
    correct_key = f"correct_top{topk}"
    ece_key = "ece"
    fixed_correct_target = int(float(identity.get(correct_key, 0.0)))
    for row in rows:
        row.update(fixed_correct_metrics(
            variant_records[row["variant"]], fixed_correct_target,
            "calibrated_score"))

    score_only_rows = [
        row for row in rows
        if row["variant"] in {"global_platt", "classwise_platt",
                              "global_isotonic"}
    ]
    best_score_only = min(
        score_only_rows,
        key=lambda row: (
            float(row.get("fixed_correct_sise_logz", 0.0)),
            -float(row.get(precision_key, 0.0)),
            float(row.get(ece_key, 0.0)),
        ))
    method = method_summary
    topk_sise_identity = float(identity.get(topk_key, 0.0))
    topk_sise_best_calib = float(best_score_only.get(topk_key, 0.0))
    topk_sise_method = float(method.get(topk_key, 0.0))
    method_delta = topk_sise_method - topk_sise_identity
    best_calib_delta = topk_sise_best_calib - topk_sise_identity
    precision_identity = float(identity.get(precision_key, 0.0))
    method_precision_delta = float(method.get(precision_key, 0.0)) - precision_identity
    best_calib_precision_delta = (
        float(best_score_only.get(precision_key, 0.0)) - precision_identity)
    fixed_sise_identity = float(identity.get("fixed_correct_sise_logz", 0.0))
    fixed_sise_best_calib = float(
        best_score_only.get("fixed_correct_sise_logz", 0.0))
    fixed_sise_method = float(method.get("fixed_correct_sise_logz", 0.0))
    fixed_method_delta = fixed_sise_method - fixed_sise_identity
    fixed_best_calib_delta = fixed_sise_best_calib - fixed_sise_identity
    fixed_method_precision_delta = (
        float(method.get("fixed_correct_precision", 0.0))
        - float(identity.get("fixed_correct_precision", 0.0)))
    fixed_best_calib_precision_delta = (
        float(best_score_only.get("fixed_correct_precision", 0.0))
        - float(identity.get("fixed_correct_precision", 0.0)))

    dataset_row = {
        "dataset": spec["dataset"],
        "domain": spec["domain"],
        "status": "done",
        "method": spec["method"],
        "topk": topk,
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "method_eval_records": len(method_eval_records),
        "identity_sise_topk": topk_sise_identity,
        "best_score_only_variant": best_score_only["variant"],
        "best_score_only_sise_topk": topk_sise_best_calib,
        "method_sise_topk": topk_sise_method,
        "best_score_only_sise_delta": best_calib_delta,
        "method_sise_delta": method_delta,
        "fixed_correct_target": fixed_correct_target,
        "identity_fixed_correct_sise": fixed_sise_identity,
        "best_score_only_fixed_correct_sise": fixed_sise_best_calib,
        "method_fixed_correct_sise": fixed_sise_method,
        "best_score_only_fixed_correct_sise_delta": fixed_best_calib_delta,
        "method_fixed_correct_sise_delta": fixed_method_delta,
        "identity_fixed_correct_precision": float(
            identity.get("fixed_correct_precision", 0.0)),
        "best_score_only_fixed_correct_precision_delta": (
            fixed_best_calib_precision_delta),
        "method_fixed_correct_precision_delta": fixed_method_precision_delta,
        "method_fixed_correct_reached": bool(
            method.get("fixed_correct_reached")),
        "identity_precision_topk": precision_identity,
        "best_score_only_precision_delta": best_calib_precision_delta,
        "method_precision_delta": method_precision_delta,
        "identity_correct_topk": float(identity.get(correct_key, 0.0)),
        "best_score_only_correct_delta": delta(
            best_score_only, identity, correct_key),
        "method_correct_delta": delta(method, identity, correct_key),
        "identity_ece": float(identity.get(ece_key, 0.0)),
        "best_score_only_ece": float(best_score_only.get(ece_key, 0.0)),
        "method_ece": float(method.get(ece_key, 0.0)),
        "best_score_only_ece_delta": delta(best_score_only, identity, ece_key),
        "method_beats_score_only_topk_sise": (
            method_delta < best_calib_delta),
        "method_beats_score_only_fixed_correct_sise": (
            fixed_method_delta < fixed_best_calib_delta),
        "method_precision_non_regression": method_precision_delta >= -0.002,
        "method_fixed_correct_precision_non_regression": (
            fixed_method_precision_delta >= -0.002),
        "score_only_uses_scale_features": False,
        "method_uses_semantic_scale_features": True,
        "baseline_csv": str(baseline_path),
        "method_csv": str(method_path),
    }
    dataset_row["dataset_gate_pass"] = (
        dataset_row["method_beats_score_only_fixed_correct_sise"]
        and dataset_row["method_fixed_correct_precision_non_regression"]
        and dataset_row["method_fixed_correct_reached"]
        and fixed_method_delta <= 0)
    return dataset_row, rows


def fmt(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def md_table(rows, columns):
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [fmt(row.get(col, "")).replace("\n", " ") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_markdown(path, payload):
    dataset_rows = payload["dataset_rows"]
    variant_rows = payload["variant_rows"]
    pass_count = payload["method_beats_score_only_count"]
    lines = [
        "# Score-Only Calibration Baseline Gate - 2026-06-20",
        "",
        "## 结论",
        "",
        (
            f"- `score_only_calibration_gate_pass`: "
            f"`{str(payload['score_only_calibration_gate_pass']).lower()}`"
        ),
        f"- `datasets_done`: `{payload['datasets_done']}`",
        f"- `method_beats_score_only_count`: `{pass_count}`",
        "",
        "这个 gate 只检验一个边界问题：普通 detector score calibration 是否可以",
        "替代 G-S3C 的 semantic-scale 排序/降权收益。Platt 与 Isotonic 只读取",
        "`score -> correct`，不读取 `log_area_z`、类别尺度先验或 Gaussian 分布特征。",
        "",
        "## Dataset-Level Gate",
        "",
        md_table(dataset_rows, [
            "dataset", "method", "topk", "identity_sise_topk",
            "best_score_only_variant", "best_score_only_sise_delta",
            "method_sise_delta", "best_score_only_precision_delta",
            "method_precision_delta", "best_score_only_ece_delta",
            "fixed_correct_target",
            "best_score_only_fixed_correct_sise_delta",
            "method_fixed_correct_sise_delta",
            "method_fixed_correct_precision_delta",
            "method_fixed_correct_reached", "dataset_gate_pass",
        ]),
        "",
        "## Variant Summary",
        "",
        md_table(variant_rows, [
            "dataset", "variant", "localized_total", "localized_precision",
            "ece", "sise_logz_top100", "sise_logz_top500",
            "sise_logz_top1000", "sise_logz_top5000",
            "sise_logz_top10000", "fixed_correct_sise_logz",
            "fixed_correct_precision",
        ]),
        "",
        "## 审稿含义",
        "",
        "- 如果本 gate 通过：可以写成普通校准能够改善 confidence calibration，",
        "但不能充分替代语义-尺度路径上的 topK/high-risk 排序修正。",
        "- 如果本 gate 未通过：说明 G-S3C 的收益仍可能被 score-only calibration",
        "解释，需要把方法新颖性和有效性评分维持在 9.5 以下。",
        "- 无论通过与否，本 gate 不替代 `E-P2-clean-v2` 因果证据，也不替代",
        "`G3-v2/BASS` matched-control 训练证据。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--result-md", default=str(DEFAULT_RESULT_MD))
    parser.add_argument("--train-frac", type=float, default=0.5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    result_md = Path(args.result_md)

    dataset_rows = []
    variant_rows = []
    for spec in DATASETS:
        dataset_row, rows = summarize_dataset(spec, args.train_frac)
        dataset_rows.append(dataset_row)
        variant_rows.extend(rows)

    done_rows = [row for row in dataset_rows if row.get("status") == "done"]
    method_beats_count = sum(
        1 for row in done_rows
        if row.get("dataset_gate_pass") is True)
    score_only_calibration_gate_pass = (
        len(done_rows) >= 3 and method_beats_count >= 2)

    payload = {
        "score_only_calibration_gate_pass": score_only_calibration_gate_pass,
        "datasets_done": len(done_rows),
        "datasets_total": len(dataset_rows),
        "method_beats_score_only_count": method_beats_count,
        "train_frac": args.train_frac,
        "score_thrs": SCORE_THRS,
        "topks": TOPKS,
        "ece_bins": ECE_BINS,
        "score_only_baselines": [
            "global_platt", "classwise_platt", "global_isotonic"],
        "dataset_rows": dataset_rows,
        "variant_rows": variant_rows,
        "result_md": str(result_md),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "calibration_baseline_review.json", payload)
    write_csv(out_dir / "calibration_baseline_dataset_summary.csv", dataset_rows)
    write_csv(out_dir / "calibration_baseline_variant_summary.csv", variant_rows)
    build_markdown(result_md, payload)
    print(json.dumps({
        "score_only_calibration_gate_pass": score_only_calibration_gate_pass,
        "datasets_done": len(done_rows),
        "method_beats_score_only_count": method_beats_count,
        "json": str(out_dir / "calibration_baseline_review.json"),
        "dataset_csv": str(out_dir / "calibration_baseline_dataset_summary.csv"),
        "variant_csv": str(out_dir / "calibration_baseline_variant_summary.csv"),
        "md": str(result_md),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
