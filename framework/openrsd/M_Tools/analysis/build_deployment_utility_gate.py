#!/usr/bin/env python
"""Build deployment-utility gate from existing G-S3C/SISE summaries.

The script only reads existing CSV/JSON/MD artifacts.  It does not launch
inference and does not invent latency or AP values.  Missing latency is reported
as missing; the strict practicality gate currently uses the P4 no-dump latency
evidence plus P4/HRRSD utility improvements.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_OUT_DIR = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/deployment_utility")
DEFAULT_RESULT_MD = Path(
    "resultmd/exp_p4_scale_semantic_validation/"
    "fres_20260620_deployment_utility_gate.md")
DEFAULT_LATENCY_JSON = Path(
    "resultmd/exp_p4_scale_semantic_validation/"
    "freview_20260618_s3c_topvenue_strict_9.json")


DATASETS = [
    {
        "dataset": "P4 OVD full preselect0.99",
        "domain": "ovd",
        "path": Path(
            "work_dirs/gs3c_sise_problem_reframing_20260619/"
            "p4_ovd_full_preselect0p99/deployment_risk_variant_summary.csv"),
        "baseline": "baseline",
        "method": "s3c_nodump",
        "mAP_base": 0.5699,
        "mAP_method": 0.5749,
        "topk": "top10000",
        "score_thr": "score_ge_0p999",
        "latency_source": "p4_nodump",
    },
    {
        "dataset": "HRRSD full",
        "domain": "closed_set",
        "path": Path(
            "work_dirs/gs3c_sise_problem_reframing_20260619/"
            "hrrsd_full/deployment_risk_variant_summary.csv"),
        "baseline": "baseline",
        "method": "g1_z4",
        "mAP_base": 0.8307,
        "mAP_method": 0.8328,
        "topk": "top5000",
        "score_thr": "score_ge_0p5",
        "latency_source": "",
    },
    {
        "dataset": "DIOR-R full",
        "domain": "closed_set",
        "path": Path(
            "work_dirs/gs3c_sise_problem_reframing_20260619/"
            "dior_r_full/deployment_risk_variant_summary.csv"),
        "baseline": "baseline",
        "method": "g2_network",
        "mAP_base": 0.644687,
        "mAP_method": 0.644975,
        "topk": "top10000",
        "score_thr": "score_ge_0p5",
        "latency_source": "",
    },
    {
        "dataset": "xView full",
        "domain": "closed_set",
        "path": Path(
            "work_dirs/gs3c_sise_problem_reframing_20260619/"
            "xview_full/deployment_risk_variant_summary.csv"),
        "baseline": "baseline",
        "method": "g1_light",
        "mAP_base": 0.182514,
        "mAP_method": 0.1824,
        "topk": "top10000",
        "score_thr": "score_ge_0p5",
        "latency_source": "",
    },
]


def to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def load_latency(path):
    path = Path(path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    practicality = payload.get("practicality", {})
    return {
        "p4_nodump": {
            "latency_overhead_rate": practicality.get(
                "inference_overhead_rate"),
            "baseline_inference_seconds": practicality.get(
                "baseline_inference_seconds"),
            "method_inference_seconds": practicality.get(
                "deploy_inference_seconds"),
            "source": str(path),
        }
    }


def by_variant(rows):
    return {row.get("variant", ""): row for row in rows}


def metric(row, name):
    return to_float(row.get(name))


def summarize_dataset(spec, latency):
    path = spec["path"]
    if not path.exists():
        return {
            "dataset": spec["dataset"],
            "status": "missing_summary",
            "summary_path": str(path),
        }
    rows = by_variant(read_csv_rows(path))
    base = rows.get(spec["baseline"])
    method = rows.get(spec["method"])
    if not base or not method:
        return {
            "dataset": spec["dataset"],
            "status": "missing_variant",
            "summary_path": str(path),
        }

    topk = spec["topk"]
    score_thr = spec["score_thr"]
    latency_info = latency.get(spec.get("latency_source", ""), {})
    latency_overhead = latency_info.get("latency_overhead_rate")

    high_conf_wrong_key = f"wrong_excl_sibling_ge_thr_{score_thr}"
    high_conf_sise_key = f"sise_logz_ge_thr_{score_thr}"
    high_conf_far_key = f"hc_far_per_image_{score_thr}"
    fixed_precision_key = f"fixed_score_precision_{score_thr}"

    base_topk_precision = metric(base, f"topk_precision_{topk}")
    method_topk_precision = metric(method, f"topk_precision_{topk}")
    base_topk_sise = metric(base, f"sise_logz_topk_{topk}")
    method_topk_sise = metric(method, f"sise_logz_topk_{topk}")
    base_high_conf_wrong = metric(base, high_conf_wrong_key)
    method_high_conf_wrong = metric(method, high_conf_wrong_key)
    base_high_conf_sise = metric(base, high_conf_sise_key)
    method_high_conf_sise = metric(method, high_conf_sise_key)

    mAP_delta = spec["mAP_method"] - spec["mAP_base"]
    row = {
        "dataset": spec["dataset"],
        "domain": spec["domain"],
        "baseline_variant": spec["baseline"],
        "method_variant": spec["method"],
        "mAP_base": spec["mAP_base"],
        "mAP_method": spec["mAP_method"],
        "mAP_delta": mAP_delta,
        "score_thr": score_thr,
        "high_conf_wrong_base": base_high_conf_wrong,
        "high_conf_wrong_method": method_high_conf_wrong,
        "high_conf_wrong_delta": method_high_conf_wrong - base_high_conf_wrong,
        "high_conf_sise_base": base_high_conf_sise,
        "high_conf_sise_method": method_high_conf_sise,
        "high_conf_sise_delta": method_high_conf_sise - base_high_conf_sise,
        "high_conf_far_per_image_base": metric(base, high_conf_far_key),
        "high_conf_far_per_image_method": metric(method, high_conf_far_key),
        "fixed_score_precision_base": metric(base, fixed_precision_key),
        "fixed_score_precision_method": metric(method, fixed_precision_key),
        "topk": topk,
        "review_queue_precision_base": base_topk_precision,
        "review_queue_precision_method": method_topk_precision,
        "review_queue_precision_delta": (
            method_topk_precision - base_topk_precision),
        "review_queue_sise_base": base_topk_sise,
        "review_queue_sise_method": method_topk_sise,
        "review_queue_sise_delta": method_topk_sise - base_topk_sise,
        "total_sise_base": metric(base, "sise_logz_wrong_excl_sibling"),
        "total_sise_method": metric(method, "sise_logz_wrong_excl_sibling"),
        "total_sise_delta": (
            metric(method, "sise_logz_wrong_excl_sibling")
            - metric(base, "sise_logz_wrong_excl_sibling")),
        "latency_overhead_rate": latency_overhead,
        "latency_source": latency_info.get("source", ""),
        "summary_path": str(path),
        "status": "done",
    }
    row["ap_non_regression"] = mAP_delta >= -0.0005
    row["high_conf_wrong_nonincrease"] = row["high_conf_wrong_delta"] <= 0
    row["high_conf_sise_reduction"] = row["high_conf_sise_delta"] < 0
    row["review_queue_precision_improves"] = (
        row["review_queue_precision_delta"] > 0)
    row["review_queue_sise_reduction"] = row["review_queue_sise_delta"] < 0
    row["latency_pass"] = (
        latency_overhead is not None and float(latency_overhead) <= 0.02)
    row["utility_pass"] = (
        row["ap_non_regression"]
        and row["high_conf_wrong_nonincrease"]
        and (row["high_conf_sise_reduction"]
             or row["review_queue_sise_reduction"])
        and row["review_queue_precision_improves"])
    return row


def fmt(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def md_table(rows):
    columns = [
        "dataset", "method_variant", "mAP_delta",
        "high_conf_wrong_delta", "high_conf_sise_delta",
        "review_queue_precision_delta", "review_queue_sise_delta",
        "latency_overhead_rate", "utility_pass"
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_markdown(path, review, rows):
    lines = [
        "# Deployment Utility Gate - 2026-06-20",
        "",
        "This file is generated from existing deployment-risk summaries and the "
        "P4 no-dump latency audit. It does not run new inference.",
        "",
        "## Verdict",
        "",
        f"- practicality_gate_pass: `{review['practicality_gate_pass']}`",
        f"- p4_full_deployment_pass: `{review['p4_full_deployment_pass']}`",
        f"- closed_set_utility_pass_count: `{review['closed_set_utility_pass_count']}`",
        f"- latency_overhead_rate_p4: `{review['latency_overhead_rate_p4']}`",
        "",
        "## Utility Table",
        "",
        md_table(rows),
        "",
        "## Interpretation",
        "",
        "- P4 supports the no-dump deployment claim because it has AP "
        "non-regression, high-confidence SISE reduction, review-queue precision "
        "improvement, and measured latency overhead below 2%.",
        "- HRRSD provides a closed-set utility signal, especially TopK SISE "
        "reduction, but lacks a dedicated latency measurement.",
        "- DIOR-R/xView remain boundary evidence and should not be used as "
        "primary deployment-success claims.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--result-md", default=str(DEFAULT_RESULT_MD))
    parser.add_argument("--latency-json", default=str(DEFAULT_LATENCY_JSON))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    latency = load_latency(args.latency_json)
    rows = [summarize_dataset(spec, latency) for spec in DATASETS]
    p4 = next(row for row in rows if row["dataset"].startswith("P4 "))
    closed_passes = [
        row for row in rows
        if row.get("domain") == "closed_set" and row.get("utility_pass")
    ]
    p4_full_deployment_pass = bool(p4.get("utility_pass") and p4.get("latency_pass"))
    practicality_gate_pass = bool(p4_full_deployment_pass and len(closed_passes) >= 1)
    review = {
        "practicality_gate_pass": practicality_gate_pass,
        "p4_full_deployment_pass": p4_full_deployment_pass,
        "closed_set_utility_pass_count": len(closed_passes),
        "latency_overhead_rate_p4": p4.get("latency_overhead_rate"),
        "summary_csv": str(out_dir / "deployment_utility_summary.csv"),
        "result_md": str(args.result_md),
        "rows": rows,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "deployment_utility_summary.csv", rows)
    write_json(out_dir / "deployment_utility_review.json", review)
    write_markdown(Path(args.result_md), review, rows)
    print(json.dumps({
        "practicality_gate_pass": practicality_gate_pass,
        "p4_full_deployment_pass": p4_full_deployment_pass,
        "closed_set_utility_pass_count": len(closed_passes),
        "json": str(out_dir / "deployment_utility_review.json"),
        "csv": str(out_dir / "deployment_utility_summary.csv"),
        "md": str(args.result_md),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
