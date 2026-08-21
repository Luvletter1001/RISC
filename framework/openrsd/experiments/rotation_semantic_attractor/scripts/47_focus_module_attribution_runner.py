#!/usr/bin/env python3
"""Run or summarize safe FOCUS-OVD module attribution variants."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from focus_attribution_common import (
    DEFAULT_EXP_DIR,
    append_manifest_record,
    ensure_exp_tree,
    markdown_table,
    read_csv_rows,
    write_csv_rows,
    write_json,
)


FIELDS = [
    "variant_id", "variant_status", "corrected_FSV", "dense_sv_ratio",
    "true_SV_recall_proxy", "mAP_AP_proxy", "SV_AP_proxy", "det/img",
    "migration_mass_ratio", "degenerate_large_sv_ratio",
    "support_geometry_status", "primary_metrics", "safety_metrics",
    "module_effect_verdict", "failure_reason",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def find_equivalence(exp_dir: Path) -> dict[str, Any]:
    for path in [
            exp_dir / "smoke/baseline_equivalence/equivalence_smoke.json",
            exp_dir / "baseline_equivalence/equivalence_smoke.json",
            Path("resultmd/exp_focus_ovd_20260608/baseline_equivalence/equivalence_smoke.json"),
    ]:
        data = read_json(path)
        if data:
            data["_path"] = str(path)
            return data
    return {}


def find_orientation(exp_dir: Path) -> dict[str, Any]:
    for path in [
            exp_dir / "orientation_probe/orientation_probe_summary.json",
            Path("resultmd/exp_focus_ovd_20260608/orientation_probe/orientation_probe_summary.json"),
    ]:
        data = read_json(path)
        if data:
            data["_path"] = str(path)
            return data
    return {}


def selected_variants(rows: list[dict[str, str]], priority: str | None,
                      variant: str | None) -> list[dict[str, str]]:
    if variant:
        return [row for row in rows if row.get("variant_id") == variant]
    if priority:
        return [row for row in rows if row.get("priority") == priority]
    return rows


def status_for_variant(row: dict[str, str], exp_dir: Path) -> dict[str, Any]:
    vid = row["variant_id"]
    train_required = row.get("train_required") == "true"
    eval_required = row.get("eval_required") == "true"
    equivalence = find_equivalence(exp_dir)
    orientation = find_orientation(exp_dir)
    base = {key: "NA" for key in FIELDS}
    base["variant_id"] = vid
    base["corrected_FSV"] = "NOT_EVALUATED"
    base["dense_sv_ratio"] = "NOT_EVALUATED"
    base["true_SV_recall_proxy"] = "NOT_EVALUATED"
    base["mAP_AP_proxy"] = "NOT_EVALUATED"
    base["SV_AP_proxy"] = "NOT_EVALUATED"
    base["det/img"] = "NOT_EVALUATED"
    base["migration_mass_ratio"] = "NOT_EVALUATED"
    base["degenerate_large_sv_ratio"] = "NOT_EVALUATED"
    base["support_geometry_status"] = "NOT_EVALUATED"
    base["primary_metrics"] = "{}"
    base["safety_metrics"] = "{}"

    if vid == "V00_baseline":
        base.update({
            "variant_status": "COMMAND_PLANNED_NOT_RUN",
            "module_effect_verdict": "BASELINE_REFERENCE_NOT_EVALUATED",
            "failure_reason": "full/small baseline eval was not run in this safe attribution pass",
        })
        return base
    if vid == "V01_focus_zero":
        if equivalence.get("status") == "PASS":
            base.update({
                "variant_status": "PASS_BASELINE_EQUIVALENCE_SMOKE",
                "support_geometry_status": "NO_DELTA_ZERO_RESIDUAL",
                "primary_metrics": json.dumps({
                    "max_abs_diff": equivalence.get("max_abs_diff"),
                    "max_delta_norm_ratio": equivalence.get("max_delta_norm_ratio"),
                    "artifact": equivalence.get("_path"),
                }, sort_keys=True),
                "module_effect_verdict": "BASELINE_EQUIVALENCE_PASS",
                "failure_reason": "",
            })
        else:
            base.update({
                "variant_status": "BLOCKED_BASELINE_EQUIVALENCE_NOT_PASS",
                "module_effect_verdict": "NOT_EVALUATED",
                "failure_reason": "baseline-equivalence smoke missing or not PASS",
            })
        return base
    if vid == "V10_orientation_probe_only":
        if orientation:
            base.update({
                "variant_status": "PROBE_WRITTEN",
                "primary_metrics": json.dumps({
                    "mean_periodic_error_degrees": orientation.get(
                        "mean_periodic_error_degrees"),
                    "min_confidence": orientation.get("min_confidence"),
                    "max_confidence": orientation.get("max_confidence"),
                    "artifact": orientation.get("_path"),
                }, sort_keys=True),
                "module_effect_verdict": "DIAGNOSTIC_ONLY",
                "failure_reason": "",
            })
        else:
            base.update({
                "variant_status": "NO_ORIENTATION_PROBE",
                "module_effect_verdict": "NOT_EVALUATED",
                "failure_reason": "orientation probe artifact missing",
            })
        return base
    if train_required:
        if equivalence.get("status") != "PASS":
            reason = "baseline-equivalence smoke has not passed in available artifacts"
            status = "BLOCKED_BASELINE_EQUIVALENCE_NOT_PASS"
        else:
            reason = "requires training or checkpoint eval; full training is forbidden in this pass"
            status = "NOT_EVALUATED_REQUIRES_TRAINING"
        base.update({
            "variant_status": status,
            "module_effect_verdict": "NOT_EVALUATED",
            "failure_reason": reason,
        })
        return base
    if eval_required:
        base.update({
            "variant_status": "NOT_EVALUATED_REQUIRES_EVAL",
            "module_effect_verdict": "NOT_EVALUATED",
            "failure_reason": "safe planner generated commands but did not run eval",
        })
        return base
    base.update({
        "variant_status": "COMMAND_PLANNED_NOT_RUN",
        "module_effect_verdict": "NOT_EVALUATED",
        "failure_reason": "variant not selected for smoke execution",
    })
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--matrix", type=Path,
                        default=DEFAULT_EXP_DIR / "ablation_plans/focus_module_attribution_matrix.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXP_DIR / "eval")
    parser.add_argument("--priority", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--angles", default="0,30,60,90,120,150,180,210,240,270,300,330")
    args = parser.parse_args()
    exp_dir = args.output_dir.parent
    ensure_exp_tree(exp_dir)
    matrix_rows = read_csv_rows(args.matrix)
    if not matrix_rows:
        raise FileNotFoundError(f"missing ablation matrix: {args.matrix}")
    rows = [status_for_variant(row, exp_dir)
            for row in selected_variants(matrix_rows, args.priority, args.variant)]
    csv_path = args.output_dir / "focus_module_attribution_eval.csv"
    json_path = args.output_dir / "focus_module_attribution_eval.json"
    md_path = args.output_dir / "focus_module_attribution_eval.md"
    write_csv_rows(csv_path, rows, FIELDS)
    write_json(json_path, {"rows": rows, "priority": args.priority,
                           "variant": args.variant, "angles": args.angles,
                           "limit": args.limit})
    md = [
        "# FOCUS-OVD Module Attribution Runner Output",
        "",
        f"- Matrix: `{args.matrix}`",
        f"- Priority filter: `{args.priority}`",
        f"- Variant filter: `{args.variant}`",
        f"- Angles: `{args.angles}`",
        "- No full benchmark or full model training is executed by this planner.",
        "",
    ]
    md.extend(markdown_table(rows, FIELDS))
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    append_manifest_record(
        exp_dir, repo_root=args.repo_root.resolve(), stage="module_attribution_runner",
        status="WRITTEN", module_switches={
            "priority": args.priority, "variant": args.variant,
            "rows": len(rows), "limit": args.limit},
        angles=args.angles)
    print(json.dumps({"csv": str(csv_path), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
