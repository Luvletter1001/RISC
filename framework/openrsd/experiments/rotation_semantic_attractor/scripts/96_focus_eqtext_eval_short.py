#!/usr/bin/env python3
"""Evaluate the five FOCUS-EQText short-run variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from focus_eqtext_common import (  # noqa: E402
    EVAL_VARIANTS,
    EXP_DIR,
    baseline_metrics,
    ensure_exp_tree,
    eval_metrics_for_variant,
    md_table,
    read_json,
    resolve,
    variant_defs,
    write_csv,
    write_json,
)


FIELDS = [
    "variant_id", "eval_status", "metric_source", "actual_detector_train",
    "p1a_realbatch_status", "corrected_FSV", "corrected_FSV_delta",
    "corrected_FSV_reduction", "dense_sv_ratio",
    "true_SV_positive_control_retention",
    "annotation_missing_true_vehicle_retention", "det/img",
    "migration_mass_ratio", "degenerate_large_sv_ratio",
    "visual_delta_norm", "text_delta_norm", "visual_support_cos_max",
    "text_support_cos_max", "dual_text_weight", "AP", "mAP",
]


def write_variant_markdown(path: Path, row: dict[str, Any]) -> None:
    lines = [f"# {row['variant_id']} EQText Short Eval", ""]
    lines.extend(md_table([row], FIELDS))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--exp-dir", type=Path, default=EXP_DIR)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    exp_dir = resolve(repo_root, args.exp_dir)
    ensure_exp_tree(exp_dir)
    variants = variant_defs()
    baseline = baseline_metrics(repo_root)
    train_records = {}
    for variant_id in EVAL_VARIANTS:
        train_records[variant_id] = read_json(
            exp_dir / "train" / variant_id / "realbatch_smoke_report.json",
            {})

    rows: list[dict[str, Any]] = []
    for variant_id in EVAL_VARIANTS:
        variant = variants[variant_id]
        train_record = train_records.get(variant_id, {})
        row = eval_metrics_for_variant(
            variant,
            baseline,
            train_record=train_record if train_record else None)
        if variant_id == "EQ_V00_baseline":
            row["metric_source"] = "baseline_verified_crop_label_proxy"
            row["p1a_realbatch_status"] = "NOT_TRAIN_REQUIRED"
        if variant_id == "EQ_V01_focus_zero_dual":
            preflight = read_json(
                exp_dir / "preflight/focus_eqtext_preflight.json", {})
            dual_eq = preflight.get("dual_zero_equivalence", {})
            row["metric_source"] = "dual_zero_baseline_equivalence_preflight"
            row["p1a_realbatch_status"] = dual_eq.get("status", "missing")
            row["baseline_equivalence_max_abs_diff"] = dual_eq.get(
                "max_abs_diff", "")
        out_dir = exp_dir / "eval" / variant_id
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "eval_summary.json", row)
        write_csv(out_dir / "eval_summary.csv", [row], FIELDS)
        write_variant_markdown(out_dir / "eval_summary.md", row)
        rows.append(row)

    write_csv(exp_dir / "eval/focus_eqtext_eval_all_variants.csv", rows, FIELDS)
    write_json(exp_dir / "eval/focus_eqtext_eval_all_variants.json", {
        "status": "PASS_FOCUS_EQTEXT_EVAL_SHORT",
        "ap_status": "AP_BLOCKED",
        "rows": rows,
    })
    print(json.dumps({
        "status": "PASS_FOCUS_EQTEXT_EVAL_SHORT",
        "variants": [row["variant_id"] for row in rows],
        "ap_status": "AP_BLOCKED",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
