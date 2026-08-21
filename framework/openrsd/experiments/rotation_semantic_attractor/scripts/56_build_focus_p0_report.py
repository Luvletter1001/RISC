#!/usr/bin/env python3
"""Build the final FOCUS-OVD P0 small train/eval report."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from focus_p0_common import (
    ensure_exp_tree,
    markdown_table,
    read_csv_rows,
    read_json,
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


def write_html(md_path: Path, html_path: Path) -> None:
    body = []
    for line in md_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            body.append(f"<li>{html.escape(line[2:])}</li>")
        elif line.startswith("|"):
            body.append(f"<pre>{html.escape(line)}</pre>")
        else:
            body.append(f"<p>{html.escape(line)}</p>")
    html_path.write_text("<html><body>\n" + "\n".join(body) + "\n</body></html>\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--exp-dir", type=Path, required=True)
    args = parser.parse_args()
    ensure_exp_tree(args.exp_dir)

    preflight = read_json(args.exp_dir / "preflight/focus_p0_preflight.json")
    calibration = read_json(args.exp_dir / "orientation_probe/orientation_confidence_calibration.json")
    safety_json = read_json(args.exp_dir / "safety/p0_safety_gate_summary.json")
    verdict_rows = read_csv_rows(args.exp_dir / "tables/p0_module_effectiveness_table.csv")
    train_variants = [
        path.parent.name
        for path in sorted((args.exp_dir / "train").glob("*/train_status.json"))
        if "DONE" in read_json(path).get("status", "")
    ]
    evaluated = [row["variant_id"] for row in verdict_rows]
    effective = safety_json.get("effective", [])
    weak = safety_json.get("weak", [])
    unsafe = safety_json.get("unsafe", [])
    no_effect = safety_json.get("no_effect", [])
    diagnostic = [row["variant_id"] for row in verdict_rows if row.get("verdict") == "DIAGNOSTIC_ONLY"]
    v24_p1 = "V24_full_focus_core" in effective

    md = [
        "# FOCUS-OVD P0 Train/Eval Report",
        "",
        "## Executive Verdict",
        "",
        f"- Effective candidates: `{effective}`",
        f"- Weak effects: `{weak}`",
        f"- Unsafe: `{unsafe}`",
        f"- No effect: `{no_effect}`",
        f"- Diagnostic only: `{diagnostic}`",
        f"- V24_full_focus_core can enter P1: `{v24_p1}`",
        "- Scope: verified-crop label proxy small-train/eval; no full benchmark, no full detector training, no DeCLIP support.",
        "",
        "## Preflight",
        "",
        f"- status: `{preflight.get('status')}`",
        f"- blocking_failed: `{preflight.get('blocking_failed')}`",
        "",
        "## Orientation Confidence Calibration",
        "",
        f"- verdict: `{calibration.get('verdict')}`",
        f"- sufficient_safety_gate: `{calibration.get('sufficient_safety_gate')}`",
        "",
        "## P0 Variants",
        "",
        f"- trained variants: `{train_variants}`",
        f"- evaluated variants: `{evaluated}`",
        "",
    ]
    md.extend(markdown_table(verdict_rows, FIELDS))
    md.extend([
        "",
        "## Safety",
        "",
        "- True vehicle preservation, migration, support geometry, degenerate boxes, and detection count are reported in the table above.",
        "- Orientation confidence is not promoted to an independent safety gate when padding or degenerate rows also receive high confidence.",
        "",
        "## Recommendation",
        "",
        "- Promote V24 to P1 only as a proxy-supported candidate if the table keeps `EFFECTIVE_CANDIDATE`.",
        "- Treat V23 as the simpler effective ablation when V24 passes.",
        "- Discard or tune unsafe variants before any P1 work.",
        "- Keep V10 diagnostic-only and V02/V03 as negative controls.",
        "",
        "## Next Commands",
        "",
        "- If V24 passes, generate a P1 command plan; do not run P1 in this pass.",
        "- If V24 fails but V21/V23 helps, tune loss weights and rerun P0.",
        "- If all fail, stop adding modules and inspect whether orientation residual reaches logits.",
    ])
    md_path = args.exp_dir / "reports/focus_p0_train_eval_report.md"
    html_path = args.exp_dir / "reports/focus_p0_train_eval_report.html"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    write_html(md_path, html_path)
    print(json.dumps({"report": str(md_path), "v24_can_enter_p1": v24_p1}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
