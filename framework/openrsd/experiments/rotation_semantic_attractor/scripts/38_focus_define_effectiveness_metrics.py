#!/usr/bin/env python3
"""Define FOCUS-OVD effectiveness metrics and decision rules."""

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
    markdown_table,
    write_csv_rows,
)


FIELDS = [
    "metric_name", "category", "definition", "higher_is_better",
    "required_for_decision", "decision_rule",
]


def metric(name: str, category: str, definition: str, higher: str,
           required: str, rule: str) -> dict[str, str]:
    return {
        "metric_name": name,
        "category": category,
        "definition": definition,
        "higher_is_better": higher,
        "required_for_decision": required,
        "decision_rule": rule,
    }


def build_metrics() -> list[dict[str, str]]:
    rows = []
    for name in ["mAP50", "AP50", "SV_AP50", "per-class AP"]:
        rows.append(metric(name, "core_detection", "standard detection metric",
                           "true", "true", "drop <= 1 point versus baseline"))
    for name in ["det/img", "SV pred/img", "LV pred/img"]:
        rows.append(metric(name, "core_detection", "prediction count per image",
                           "context", "true", "must not explode versus baseline"))
    for name in ["dense_sv_ratio", "query_sv_ratio", "preNMS_FR_SV",
                 "postNMS_FR_SV", "annotated_GT_FSV", "corrected_FSV",
                 "noSV_false_hub_rate", "strict_object_flip_count"]:
        rows.append(metric(name, "sv_burden", "small-vehicle burden indicator",
                           "false", "true" if name in {"dense_sv_ratio", "corrected_FSV"} else "false",
                           "effective modules reduce corrected_FSV >=20% and dense_sv_ratio"))
    for name in ["true_SV_recall", "true_SV_precision",
                 "true_SV_recall_retention", "migration_mass_ratio",
                 "top_migrated_classes", "class_JS", "class_KL",
                 "lowrisk_det_inflation", "degenerate_large_sv_ratio",
                 "padding_artifact_ratio"]:
        rows.append(metric(name, "safety", "safety preservation metric",
                           "context", "true", "true-SV retention >=90%, no migration or artifact inflation"))
    for name in ["orientation_error_vs_gt", "orientation_confidence_mean",
                 "orientation_confidence_by_category", "support_delta_norm_ratio",
                 "support_inter_class_cos_mean", "support_inter_class_cos_max",
                 "sv_logit_delta", "top1_shift_rate", "kappa_gate_active_rate"]:
        rows.append(metric(name, "focus_internal", "FOCUS internal diagnostic",
                           "context", "false", "support geometry must not collapse"))
    return rows


DECISION_RULES = {
    "EFFECTIVE": [
        "corrected_FSV decreases >= 20% on S3 or verified split",
        "dense_sv_ratio decreases",
        "true_SV_recall_retention >= 90%",
        "mAP50 drop <= 1 point",
        "SV_AP50 drop <= 1 point",
        "migration_mass_ratio not worse than baseline/DeHub reference",
        "degenerate_large_sv_ratio does not increase",
        "support inter-class cosine does not collapse",
    ],
    "DIAGNOSTIC_ONLY": [
        "explains risk but does not improve or alter output",
    ],
    "UNSAFE": [
        "AP drops > 1 point",
        "true-SV recall drops",
        "migration increases",
        "support geometry collapses",
        "det/img explodes",
    ],
    "NO_EFFECT": [
        "corrected_FSV, dense_sv_ratio, and SV pred/img barely change",
    ],
    "NEGATIVE_RESULT": [
        "confirms a bad route such as direct DeCLIP support swap or direction text prompt only",
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXP_DIR)
    args = parser.parse_args()
    exp_dir = args.output_dir
    ensure_exp_tree(exp_dir)
    rows = build_metrics()
    csv_path = exp_dir / "tables/focus_effectiveness_metric_definitions.csv"
    md_path = exp_dir / "reports/focus_effectiveness_metric_definitions.md"
    write_csv_rows(csv_path, rows, FIELDS)
    md = ["# FOCUS-OVD Effectiveness Metric Definitions", ""]
    md.extend(markdown_table(rows, FIELDS))
    md.extend(["", "## Decision Rules", ""])
    for label, rules in DECISION_RULES.items():
        md.append(f"### {label}")
        for rule in rules:
            md.append(f"- {rule}")
        md.append("")
    md_path.write_text("\n".join(md), encoding="utf-8")
    append_manifest_record(
        exp_dir, repo_root=args.repo_root.resolve(), stage="metric_definitions",
        status="WRITTEN", module_switches={"metric_count": len(rows)})
    print(json.dumps({"csv": str(csv_path), "metrics": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
