#!/usr/bin/env python
"""Evaluate E-P2-clean semantic-scale path intervention probes.

The old E9 scale-swap probe judged causality from final post-NMS class flips.
That is too brittle: context and other-class attractors can dominate the final
label.  This auditor instead expects a cleaner path-probe CSV where the same
object and context are evaluated along a log-area path, and it tests whether
the hard-negative logit margin rises with Gaussian semantic-scale support.

Expected input columns are intentionally simple and model-agnostic:

``case_id``, ``control_type``, ``context_id``, ``object_id``,
``gt_class``, ``hardneg_class``, ``path_step``, ``abs_z_gt``,
``abs_z_hardneg``, ``logit_gt``, ``logit_hardneg``.

Optional columns:
``support_advantage`` and ``hardneg_margin`` can be supplied directly.  When
missing, they are computed as ``abs_z_gt - abs_z_hardneg`` and
``logit_hardneg - logit_gt``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_OUT_DIR = (
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "ep2_clean_path_intervention"
)


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


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def pearson(xs, ys):
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom_x = sum(item * item for item in dx) ** 0.5
    denom_y = sum(item * item for item in dy) ** 0.5
    if denom_x <= 0 or denom_y <= 0:
        return 0.0
    return sum(x * y for x, y in zip(dx, dy)) / (denom_x * denom_y)


def spearman(xs, ys):
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    return pearson(average_ranks(xs), average_ranks(ys))


def support_advantage(row):
    if row.get("support_advantage") not in {None, ""}:
        return to_float(row.get("support_advantage"))
    abs_z_gt = row.get("abs_z_gt")
    abs_z_hardneg = row.get("abs_z_hardneg")
    if abs_z_gt in {None, ""}:
        abs_z_gt = abs(to_float(row.get("z_gt")))
    else:
        abs_z_gt = to_float(abs_z_gt)
    if abs_z_hardneg in {None, ""}:
        abs_z_hardneg = abs(to_float(row.get("z_hardneg")))
    else:
        abs_z_hardneg = to_float(abs_z_hardneg)
    return abs_z_gt - abs_z_hardneg


def hardneg_margin(row):
    if row.get("hardneg_margin") not in {None, ""}:
        return to_float(row.get("hardneg_margin"))
    return to_float(row.get("logit_hardneg")) - to_float(row.get("logit_gt"))


def unique_nonempty(rows, key):
    values = {str(row.get(key, "")).strip() for row in rows
              if str(row.get(key, "")).strip()}
    return values


def optional_fixed(rows, keys):
    observed = set()
    for key in keys:
        observed.update(unique_nonempty(rows, key))
    return len(observed) <= 1


def summarize_case(rows,
                   min_points=3,
                   min_rho=0.6,
                   min_margin_delta=0.2,
                   min_support_delta=0.5):
    rows = sorted(
        list(rows),
        key=lambda row: (
            to_float(row.get("path_step"), 0.0),
            to_float(row.get("log_area"), 0.0),
        ),
    )
    sample = rows[0] if rows else {}
    case_id = sample.get("case_id", "")
    control_type = sample.get("control_type", "clean") or "clean"
    fixed_context = optional_fixed(rows, ("context_id", "tile_img_id"))
    fixed_object = optional_fixed(rows, ("object_id", "source_crop_id"))
    supports = [support_advantage(row) for row in rows]
    margins = [hardneg_margin(row) for row in rows]
    rho = spearman(supports, margins)

    if supports:
        low_idx = min(range(len(supports)), key=lambda idx: supports[idx])
        high_idx = max(range(len(supports)), key=lambda idx: supports[idx])
        support_delta = supports[high_idx] - supports[low_idx]
        margin_delta = margins[high_idx] - margins[low_idx]
    else:
        support_delta = 0.0
        margin_delta = 0.0

    status = "fail"
    if len(rows) < int(min_points):
        status = "insufficient_points"
    elif not fixed_context or not fixed_object:
        status = "confounded_context_or_object"
    elif (rho >= float(min_rho)
          and margin_delta >= float(min_margin_delta)
          and support_delta >= float(min_support_delta)):
        status = "pass"

    return {
        "case_id": case_id,
        "pair": sample.get("pair", ""),
        "gt_class": sample.get("gt_class", ""),
        "hardneg_class": sample.get("hardneg_class", ""),
        "control_type": control_type,
        "num_points": len(rows),
        "fixed_context": fixed_context,
        "fixed_object": fixed_object,
        "rho_support_margin": rho,
        "support_delta": support_delta,
        "margin_delta": margin_delta,
        "pass_gate": status == "pass",
        "status": status,
    }


def summarize_control(case_rows, control_type):
    selected = [row for row in case_rows if row["control_type"] == control_type]
    total = len(selected)
    pass_cases = sum(1 for row in selected if row["pass_gate"])
    return {
        "control_type": control_type,
        "cases": total,
        "pass_cases": pass_cases,
        "pass_rate": pass_cases / total if total else 0.0,
    }


def group_cases(rows):
    grouped = defaultdict(list)
    for idx, row in enumerate(rows):
        case_id = row.get("case_id") or row.get("tile_img_id") or str(idx)
        control_type = row.get("control_type") or "clean"
        row = dict(row)
        row["case_id"] = case_id
        row["control_type"] = control_type
        grouped[(case_id, control_type)].append(row)
    return grouped


def build_report(review, case_csv, control_csv):
    clean = review["clean"]
    lines = [
        "# E-P2-clean Path Intervention Audit",
        "",
        "本报告只审计 path-probe CSV，不启动 GPU 推理。通过条件是：在固定"
        " context/object 的同一对象尺度路径上，hard-negative semantic-scale "
        "support advantage 与 hard-negative logit margin 同步上升，并且该现象"
        "明显强于 shuffled/random controls。",
        "",
        "## Gate",
        "",
        f"- gate_pass: `{review['gate_pass']}`",
        f"- clean pass rate: `{clean['pass_rate']:.6f}` "
        f"({clean['pass_cases']}/{clean['cases']})",
        f"- max control pass rate: `{review['max_control_pass_rate']:.6f}`",
        f"- specificity_gap: `{review['specificity_gap']:.6f}`",
        "",
        "## Outputs",
        "",
        f"- case summary: `{case_csv}`",
        f"- control summary: `{control_csv}`",
        "",
        "## Interpretation",
        "",
    ]
    if review["gate_pass"]:
        lines.append(
            "当前 path-probe 满足 E-P2-clean 因果 gate，可作为比旧 E9 "
            "scale-swap 更干净的机制证据。")
    else:
        lines.append(
            "当前 path-probe 尚未满足 E-P2-clean 因果 gate。不能把旧 E9 "
            "scale-swap 失败改写成强因果证据，只能作为 boundary evidence。")
    return "\n".join(lines) + "\n"


def audit_ep2_clean_path_intervention(input_csv,
                                      out_dir=DEFAULT_OUT_DIR,
                                      min_points=3,
                                      min_rho=0.6,
                                      min_margin_delta=0.2,
                                      min_support_delta=0.5,
                                      min_clean_pass_rate=0.5,
                                      min_specificity_gap=0.5):
    rows = read_csv_rows(input_csv)
    grouped = group_cases(rows)
    case_rows = [
        summarize_case(
            items,
            min_points=min_points,
            min_rho=min_rho,
            min_margin_delta=min_margin_delta,
            min_support_delta=min_support_delta)
        for _, items in sorted(grouped.items())
    ]
    control_types = sorted({row["control_type"] for row in case_rows})
    controls = {
        control_type: summarize_control(case_rows, control_type)
        for control_type in control_types
    }
    clean = controls.get("clean", {
        "control_type": "clean",
        "cases": 0,
        "pass_cases": 0,
        "pass_rate": 0.0,
    })
    non_clean = [
        row for key, row in controls.items()
        if key not in {"clean", "real", "main"}
    ]
    max_control_pass_rate = max(
        [row["pass_rate"] for row in non_clean] or [0.0])
    specificity_gap = clean["pass_rate"] - max_control_pass_rate
    gate_pass = (
        clean["cases"] > 0
        and clean["pass_rate"] >= float(min_clean_pass_rate)
        and non_clean
        and specificity_gap >= float(min_specificity_gap))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case_csv = out_dir / "ep2_clean_case_summary.csv"
    control_csv = out_dir / "ep2_clean_control_summary.csv"
    review_json = out_dir / "ep2_clean_review.json"
    report_md = out_dir / "ep2_clean_report.md"
    write_csv(case_csv, case_rows)
    write_csv(control_csv, controls.values())
    review = {
        "input_csv": str(input_csv),
        "case_summary_csv": str(case_csv),
        "control_summary_csv": str(control_csv),
        "report_md": str(report_md),
        "num_rows": len(rows),
        "num_case_controls": len(case_rows),
        "clean": clean,
        "controls": controls,
        "max_control_pass_rate": max_control_pass_rate,
        "specificity_gap": specificity_gap,
        "gate_pass": bool(gate_pass),
        "thresholds": {
            "min_points": int(min_points),
            "min_rho": float(min_rho),
            "min_margin_delta": float(min_margin_delta),
            "min_support_delta": float(min_support_delta),
            "min_clean_pass_rate": float(min_clean_pass_rate),
            "min_specificity_gap": float(min_specificity_gap),
        },
    }
    review_json.write_text(
        json.dumps(review, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    report_md.write_text(
        build_report(review, case_csv, control_csv), encoding="utf-8")
    return review


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate E-P2-clean semantic-scale path intervention CSV.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-points", type=int, default=3)
    parser.add_argument("--min-rho", type=float, default=0.6)
    parser.add_argument("--min-margin-delta", type=float, default=0.2)
    parser.add_argument("--min-support-delta", type=float, default=0.5)
    parser.add_argument("--min-clean-pass-rate", type=float, default=0.5)
    parser.add_argument("--min-specificity-gap", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    review = audit_ep2_clean_path_intervention(
        input_csv=args.input_csv,
        out_dir=args.out_dir,
        min_points=args.min_points,
        min_rho=args.min_rho,
        min_margin_delta=args.min_margin_delta,
        min_support_delta=args.min_support_delta,
        min_clean_pass_rate=args.min_clean_pass_rate,
        min_specificity_gap=args.min_specificity_gap,
    )
    print(json.dumps({
        "gate_pass": review["gate_pass"],
        "clean_pass_rate": review["clean"]["pass_rate"],
        "specificity_gap": review["specificity_gap"],
        "report_md": review["report_md"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
