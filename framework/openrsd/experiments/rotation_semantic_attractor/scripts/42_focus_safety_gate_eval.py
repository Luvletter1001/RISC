#!/usr/bin/env python3
"""Evaluate FOCUS-OVD variant metrics against the safety gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from focus_attribution_common import (
    DEFAULT_EXP_DIR,
    append_manifest_record,
    ensure_exp_tree,
    load_focus_util,
    read_csv_rows,
    write_csv_rows,
)


SUMMARY_FIELDS = [
    "variant_id", "safety_status", "failures", "warnings",
    "derived_metrics",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--metrics-csv", type=Path, default=None)
    parser.add_argument("--baseline-variant", default="V00_baseline")
    parser.add_argument("--output-dir", type=Path,
                        default=DEFAULT_EXP_DIR / "safety_gate")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    exp_dir = args.output_dir.parent
    ensure_exp_tree(exp_dir)
    safety_util = load_focus_util(repo_root, "focus_safety_gate")
    metric_rows = read_csv_rows(args.metrics_csv) if args.metrics_csv else []
    baseline = next((r for r in metric_rows
                     if r.get("variant_id") == args.baseline_variant), {})
    rows = []
    failures = []
    for metric_row in metric_rows:
        result = safety_util.evaluate_focus_safety(metric_row, baseline)
        item = {
            "variant_id": metric_row.get("variant_id", ""),
            "safety_status": result.status,
            "failures": ";".join(result.failures),
            "warnings": ";".join(result.warnings),
            "derived_metrics": json.dumps(result.derived_metrics, sort_keys=True),
        }
        rows.append(item)
        if result.status.startswith("FAIL"):
            failures.append(item)
    if not rows:
        rows = [{
            "variant_id": "NOT_EVALUATED",
            "safety_status": "NOT_EVALUATED",
            "failures": "",
            "warnings": "",
            "derived_metrics": "{}",
        }]
    summary_csv = args.output_dir / "focus_safety_gate_summary.csv"
    failures_csv = args.output_dir / "focus_safety_failures.csv"
    report = args.output_dir / "focus_safety_gate_report.md"
    write_csv_rows(summary_csv, rows, SUMMARY_FIELDS)
    write_csv_rows(failures_csv, failures, SUMMARY_FIELDS)
    report.write_text(
        "# FOCUS-OVD Safety Gate Evaluation\n\n"
        f"- Status: {'WRITTEN' if metric_rows else 'NOT_EVALUATED'}\n"
        f"- Summary: `{summary_csv}`\n"
        f"- Failures: `{failures_csv}`\n"
        "- Rules include AP/SV_AP drop, true-SV retention, corrected-FSV reduction, migration, artifact inflation, support collapse, and det/img explosion.\n",
        encoding="utf-8")
    append_manifest_record(
        exp_dir, repo_root=repo_root, stage="safety_gate",
        status="WRITTEN" if metric_rows else "NOT_EVALUATED",
        module_switches={"safety_gate": True},
        failure_reason="" if metric_rows else "no metrics csv provided")
    print(json.dumps({"csv": str(summary_csv), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
