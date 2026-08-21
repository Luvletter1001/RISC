#!/usr/bin/env python3
"""Build FOCUS-OVD P0 safety gate and module verdict tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from focus_p0_common import (
    append_manifest,
    assign_verdict,
    ensure_exp_tree,
    markdown_table,
    read_csv_rows,
    write_csv_rows,
    write_json,
    write_simple_figure,
)


FIELDS = [
    "variant_id", "corrected_FSV", "corrected_FSV_delta",
    "dense_sv_ratio", "dense_sv_ratio_delta", "SV_pred_per_img",
    "det/img", "true_SV_positive_control_retention",
    "annotation_missing_true_vehicle_retention", "migration_mass_ratio",
    "degenerate_large_sv_ratio", "support_delta_norm_ratio",
    "support_inter_class_cos_mean", "support_inter_class_cos_max",
    "verdict", "reason",
]


def read_eval_rows(eval_dir: Path) -> list[dict[str, str]]:
    rows = []
    combined = eval_dir / "p0_eval_all_variants.csv"
    if combined.exists():
        return read_csv_rows(combined)
    for path in sorted(eval_dir.glob("*/eval_summary.csv")):
        rows.extend(read_csv_rows(path))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    exp_dir = args.output_dir.parent
    ensure_exp_tree(exp_dir)
    eval_rows = read_eval_rows(args.eval_dir)
    baseline = next((r for r in eval_rows if r.get("variant_id") == "V00_baseline"), {})
    verdict_rows = []
    failures = []
    for row in eval_rows:
        verdict = assign_verdict(row, baseline)
        item = {field: row.get(field, "") for field in FIELDS}
        item.update({"verdict": verdict.verdict, "reason": verdict.reason})
        verdict_rows.append(item)
        if verdict.verdict.startswith("UNSAFE") or verdict.verdict in {"NO_EFFECT"}:
            failures.append(item)

    write_csv_rows(args.output_dir / "p0_safety_gate_summary.csv", verdict_rows, FIELDS)
    write_csv_rows(exp_dir / "tables/p0_module_effectiveness_table.csv", verdict_rows, FIELDS)
    write_csv_rows(exp_dir / "tables/p0_module_failure_table.csv", failures, FIELDS)
    write_json(args.output_dir / "p0_safety_gate_summary.json", {
        "rows": verdict_rows,
        "effective": [r["variant_id"] for r in verdict_rows if r["verdict"] == "EFFECTIVE_CANDIDATE"],
        "weak": [r["variant_id"] for r in verdict_rows if r["verdict"] == "WEAK_EFFECT"],
        "unsafe": [r["variant_id"] for r in verdict_rows if r["verdict"].startswith("UNSAFE")],
        "no_effect": [r["variant_id"] for r in verdict_rows if r["verdict"] == "NO_EFFECT"],
    })

    labels = [r["variant_id"] for r in verdict_rows if r.get("corrected_FSV") not in {"", None}]
    fsv = [float(r["corrected_FSV"]) for r in verdict_rows if r.get("corrected_FSV") not in {"", None}]
    dense = [float(r["dense_sv_ratio"]) for r in verdict_rows if r.get("dense_sv_ratio") not in {"", None}]
    migration = [float(r["migration_mass_ratio"]) for r in verdict_rows if r.get("migration_mass_ratio") not in {"", None}]
    support = [float(r["support_inter_class_cos_max"]) for r in verdict_rows if r.get("support_inter_class_cos_max") not in {"", None}]
    write_simple_figure(
        exp_dir / "figures/p0_corrected_fsv_vs_true_sv_retention.png",
        exp_dir / "figures/p0_corrected_fsv_vs_true_sv_retention.pdf",
        "P0 Corrected FSV by Variant",
        labels,
        fsv,
    )
    write_simple_figure(
        exp_dir / "figures/p0_dense_sv_vs_migration.png",
        exp_dir / "figures/p0_dense_sv_vs_migration.pdf",
        "P0 Dense SV Ratio by Variant",
        labels,
        dense,
    )
    write_simple_figure(
        exp_dir / "figures/p0_support_geometry_by_variant.png",
        exp_dir / "figures/p0_support_geometry_by_variant.pdf",
        "P0 Support Cos Max by Variant",
        labels,
        support or migration,
    )
    md = [
        "# FOCUS-OVD P0 Safety Gate Report",
        "",
        f"- rows: `{len(verdict_rows)}`",
        f"- effective: `{[r['variant_id'] for r in verdict_rows if r['verdict'] == 'EFFECTIVE_CANDIDATE']}`",
        f"- unsafe: `{[r['variant_id'] for r in verdict_rows if r['verdict'].startswith('UNSAFE')]}`",
        "",
    ]
    md.extend(markdown_table(verdict_rows, FIELDS))
    (args.output_dir / "p0_safety_gate_report.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    (exp_dir / "reports/p0_module_verdict_report.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    append_manifest(
        exp_dir, args.repo_root.resolve(),
        {"stage": "p0_safety_and_verdict", "status": "WRITTEN", "rows": len(verdict_rows)})
    print(json.dumps({"rows": len(verdict_rows), "effective": [r["variant_id"] for r in verdict_rows if r["verdict"] == "EFFECTIVE_CANDIDATE"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
