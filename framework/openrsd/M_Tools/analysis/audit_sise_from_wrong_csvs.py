#!/usr/bin/env python
"""Audit fixed SISE metrics from existing wrong-class CSV files."""

import argparse
import ast
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_sise_score_calibration import load_priors


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-summary-csv", required=True)
    parser.add_argument("--class-area-priors-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--metric-z-thr", type=float, default=4.0)
    parser.add_argument("--score-thrs", default="0.5,0.9,0.99,0.999")
    parser.add_argument(
        "--exclude-sibling-pairs",
        default="small-vehicle->large-vehicle,large-vehicle->small-vehicle")
    return parser.parse_args()


def parse_box(raw):
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    return [float(x) for x in ast.literal_eval(raw)]


def area_from_rbox(raw):
    box = parse_box(raw)
    if len(box) < 4:
        return 0.0
    return abs(box[2]) * abs(box[3])


def logz(area, cls, priors):
    prior = priors.get(cls)
    if not prior or area <= 0:
        return 0.0
    return abs((math.log(area) - prior["log_area_mean"]) / prior["log_area_std"])


def threshold_tag(thr):
    return str(thr).replace(".", "p")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def audit_model(row, priors, score_thrs, sibling_pairs, metric_z_thr):
    wrong_rows = read_csv(row["csv"])
    pair_counts = Counter()
    pair_sise = Counter()
    scores = []
    sise_scores = []
    nonsibling_scores = []
    for item in wrong_rows:
        pred = item["pred_class"]
        gt = item["gt_class"]
        pair = f"{pred}->{gt}"
        score = float(item["pred_score"])
        z = logz(area_from_rbox(item["pred_bbox_rbox"]), pred, priors)
        is_sibling = pair in sibling_pairs
        pair_counts[pair] += 1
        scores.append(score)
        if not is_sibling:
            nonsibling_scores.append(score)
        if (not is_sibling) and z >= metric_z_thr:
            sise_scores.append(score)
            pair_sise[pair] += 1

    out = {
        "model": row["model"],
        "mAP": float(row.get("mAP") or 0.0),
        "AP50": float(row.get("AP50") or 0.0),
        "wrong_total": len(wrong_rows),
        "wrong_excl_sibling": len(nonsibling_scores),
        "sise_logz_total": len(sise_scores),
        "sise_logz_rate_on_wrong_excl_sibling": (
            len(sise_scores) / len(nonsibling_scores)
            if nonsibling_scores else 0.0),
        "top_wrong_pairs": ";".join(
            f"{pair}:{count}" for pair, count in pair_counts.most_common(8)),
        "top_sise_pairs": ";".join(
            f"{pair}:{count}" for pair, count in pair_sise.most_common(8)),
    }
    for thr in score_thrs:
        tag = threshold_tag(thr)
        out[f"wrong_score_ge_{tag}"] = sum(score >= thr for score in scores)
        out[f"wrong_excl_sibling_score_ge_{tag}"] = sum(
            score >= thr for score in nonsibling_scores)
        out[f"sise_logz_score_ge_{tag}"] = sum(
            score >= thr for score in sise_scores)
    return out


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


def markdown_table(rows, cols):
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        cells = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    priors = load_priors(args.class_area_priors_csv)
    score_thrs = [float(item) for item in args.score_thrs.split(",") if item]
    sibling_pairs = {
        item for item in args.exclude_sibling_pairs.split(",") if item.strip()
    }
    rows = [
        audit_model(row, priors, score_thrs, sibling_pairs, args.metric_z_thr)
        for row in read_csv(args.run_summary_csv)
    ]
    out_csv = out_dir / "cross_detector_sise_logz_audit.csv"
    out_json = out_dir / "cross_detector_sise_logz_audit.json"
    out_md = out_dir / "cross_detector_sise_logz_audit.md"
    write_csv(out_csv, rows)
    payload = {
        "run_summary_csv": args.run_summary_csv,
        "class_area_priors_csv": args.class_area_priors_csv,
        "metric_z_thr": args.metric_z_thr,
        "score_thrs": score_thrs,
        "sibling_pairs_excluded": sorted(sibling_pairs),
        "rows": rows,
        "outputs": {"csv": str(out_csv), "md": str(out_md)},
    }
    out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + os.linesep,
        encoding="utf-8")
    cols = [
        "model", "mAP", "wrong_total", "wrong_excl_sibling",
        "sise_logz_total", "sise_logz_score_ge_0p999",
        "wrong_excl_sibling_score_ge_0p999", "top_sise_pairs",
    ]
    out_md.write_text(
        "# Cross-Detector Fixed SISE Audit\n\n"
        f"`metric_z_thr={args.metric_z_thr}`; sibling pairs excluded from SISE.\n\n"
        + markdown_table(rows, cols) + os.linesep,
        encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
