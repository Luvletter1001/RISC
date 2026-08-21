#!/usr/bin/env python3
"""Calibrate FOCUS orientation confidence by audit category."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from focus_p0_common import (
    DEFAULT_ORIENTATION_CSV,
    DEFAULT_ORIENTATION_JSON,
    DEFAULT_RAW_LABEL_CSV,
    append_manifest,
    category_confidence_rows,
    ensure_exp_tree,
    labeled_rows,
    markdown_table,
    orientation_lookup,
    write_csv_rows,
    write_json,
    write_simple_figure,
)


FIELDS = [
    "audit_category", "human_label_group", "n",
    "theta_error_mean_if_gt_available", "confidence_mean", "confidence_std",
    "confidence_p10", "confidence_p50", "confidence_p90",
    "low_confidence_rate", "high_confidence_rate",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--human-label-csv", type=Path, default=DEFAULT_RAW_LABEL_CSV)
    parser.add_argument("--orientation-probe", type=Path, default=DEFAULT_ORIENTATION_JSON)
    parser.add_argument("--orientation-csv", type=Path, default=DEFAULT_ORIENTATION_CSV)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    exp_dir = args.output_dir.parent
    ensure_exp_tree(exp_dir)
    rows = labeled_rows(args.human_label_csv)
    lookup = orientation_lookup(args.orientation_csv)
    out_rows = category_confidence_rows(rows, lookup)

    out_csv = args.output_dir / "orientation_confidence_by_audit_category.csv"
    write_csv_rows(out_csv, out_rows, FIELDS)
    degenerate_high = any(
        row["audit_category"] == "degenerate_large_sv_box"
        and float(row["high_confidence_rate"]) > 0.20
        for row in out_rows)
    padding_high = any(
        row["audit_category"] == "padding_artifact"
        and float(row["high_confidence_rate"]) > 0.20
        for row in out_rows)
    sufficient_gate = not (degenerate_high or padding_high)
    verdict = (
        "orientation confidence is not a sufficient safety gate"
        if not sufficient_gate else
        "orientation confidence can be used only as a weak diagnostic gate"
    )

    labels = [f"{r['audit_category']}:{r['human_label_group']}" for r in out_rows]
    conf = [float(r["confidence_mean"]) for r in out_rows]
    err = [float(r["theta_error_mean_if_gt_available"]) for r in out_rows]
    write_simple_figure(
        exp_dir / "figures/orientation_confidence_by_category.png",
        exp_dir / "figures/orientation_confidence_by_category.pdf",
        "Orientation Confidence by Audit Category",
        labels,
        conf,
    )
    write_simple_figure(
        exp_dir / "figures/orientation_error_vs_confidence.png",
        exp_dir / "figures/orientation_error_vs_confidence.pdf",
        "Orientation Error Proxy by Audit Category",
        labels,
        err,
    )
    summary = {
        "status": "WRITTEN",
        "orientation_probe": str(args.orientation_probe),
        "orientation_csv": str(args.orientation_csv),
        "rows": len(out_rows),
        "padding_high_confidence": padding_high,
        "degenerate_high_confidence": degenerate_high,
        "sufficient_safety_gate": sufficient_gate,
        "verdict": verdict,
    }
    write_json(args.output_dir / "orientation_confidence_calibration.json", summary)
    md = [
        "# FOCUS-OVD Orientation Confidence Calibration",
        "",
        f"- status: `WRITTEN`",
        f"- verdict: `{verdict}`",
        f"- padding_high_confidence: `{padding_high}`",
        f"- degenerate_high_confidence: `{degenerate_high}`",
        "- Note: crop-level GT orientation is unavailable; theta error is the synthetic orientation-probe angle proxy.",
        "",
    ]
    md.extend(markdown_table(out_rows, FIELDS))
    (args.output_dir / "orientation_confidence_calibration.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    append_manifest(
        exp_dir, args.repo_root.resolve(),
        {"stage": "orientation_confidence_calibration", "status": "WRITTEN", "verdict": verdict})
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
