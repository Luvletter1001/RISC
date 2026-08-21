#!/usr/bin/env python3
"""Build the paper-grade interim FOCUS-OVD module attribution report."""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from collections import Counter
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
    write_simple_figure,
)


def verdict_counts(rows: list[dict[str, str]]) -> Counter[str]:
    return Counter(row.get("module_effect_verdict", "NOT_EVALUATED")
                   for row in rows)


def recommendation_rows(eval_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = [
        {"module": "Fourier orientation probe", "category": "Diagnostic",
         "recommendation": "keep as orientation quality evidence; not an output-changing module"},
        {"module": "SV-only residual support adapter", "category": "Core candidate",
         "recommendation": "requires P0 training/eval before main-method promotion"},
        {"module": "support distill / anti-attractor / preserve", "category": "Core candidate",
         "recommendation": "requires corrected-FSV P0 evidence and safety gate"},
        {"module": "spurious SV cluster mining", "category": "Diagnostic",
         "recommendation": "use for risk taxonomy and ablation tables only until stable"},
        {"module": "visual attribute gate", "category": "Auxiliary/safety",
         "recommendation": "do not gate outputs unless true-SV retention >= 90%"},
        {"module": "sensitivity channel mask", "category": "Ablation only",
         "recommendation": "keep max strength <= 0.05 and require AP/SV_AP report"},
        {"module": "safety gate", "category": "Safety layer",
         "recommendation": "apply to all FOCUS/SAGE/DeHub combinations as reporting gate"},
        {"module": "orbit teacher", "category": "Auxiliary pseudo-label source",
         "recommendation": "use only below human corrected-FSV labels"},
        {"module": "head consensus", "category": "Diagnostic",
         "recommendation": "mark unsupported if hooks are unavailable"},
        {"module": "negative-aware prompt", "category": "Negative/diagnostic",
         "recommendation": "auxiliary margin only; not a main prompt route"},
        {"module": "DINO-support CCL", "category": "Controlled variant",
         "recommendation": "separate from DeCLIP support replacement"},
        {"module": "DeCLIP direct support", "category": "Negative result",
         "recommendation": "do not enter main method"},
    ]
    if any(row.get("module_effect_verdict") == "BASELINE_EQUIVALENCE_PASS"
           for row in eval_rows):
        rows[0]["recommendation"] += "; baseline-equivalence prerequisite has smoke evidence"
    if any(row.get("variant_id") == "V11_orientation_adapter_sv_only"
           and row.get("module_effect_verdict", "").startswith("EFFECTIVE")
           for row in eval_rows):
        rows[1]["recommendation"] = (
            "AP-positive DOTA2-only candidate; require corrected-FSV, "
            "dense-SV, migration, and true-SV safety audit before main-method promotion")
    return rows


def write_html(md_path: Path, html_path: Path) -> None:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    body = []
    for line in lines:
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
    html_path.write_text(
        "<html><body>\n" + "\n".join(body) + "\n</body></html>\n",
        encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--exp-dir", type=Path, default=DEFAULT_EXP_DIR)
    args = parser.parse_args()
    exp_dir = args.exp_dir
    ensure_exp_tree(exp_dir)

    eval_rows = read_csv_rows(exp_dir / "eval/focus_module_attribution_eval.csv")
    inventory_rows = read_csv_rows(exp_dir / "module_inventory/focus_module_inventory.csv")
    matrix_rows = read_csv_rows(exp_dir / "ablation_plans/focus_module_attribution_matrix.csv")
    counts = verdict_counts(eval_rows)

    effectiveness_rows = []
    failure_rows = []
    for row in eval_rows:
        effectiveness_rows.append({
            "variant_id": row.get("variant_id", ""),
            "status": row.get("variant_status", ""),
            "verdict": row.get("module_effect_verdict", ""),
            "primary_metrics": row.get("primary_metrics", ""),
        })
        if row.get("failure_reason"):
            failure_rows.append({
                "variant_id": row.get("variant_id", ""),
                "status": row.get("variant_status", ""),
                "failure_reason": row.get("failure_reason", ""),
            })

    recommendation = recommendation_rows(eval_rows)
    write_csv_rows(
        exp_dir / "tables/focus_module_effectiveness_table.csv",
        effectiveness_rows,
        ["variant_id", "status", "verdict", "primary_metrics"])
    write_csv_rows(
        exp_dir / "tables/focus_module_failure_table.csv",
        failure_rows,
        ["variant_id", "status", "failure_reason"])
    write_csv_rows(
        exp_dir / "tables/focus_module_recommendation_table.csv",
        recommendation,
        ["module", "category", "recommendation"])

    figure_pairs = [
        ("focus_module_effectiveness_matrix", "Module Effectiveness Verdicts"),
        ("focus_safety_tradeoff", "Safety Tradeoff"),
        ("focus_corrected_fsv_vs_true_sv_recall", "Corrected FSV vs True-SV Recall"),
        ("focus_support_geometry_by_variant", "Support Geometry by Variant"),
        ("focus_migration_by_variant", "Migration by Variant"),
    ]
    for stem, title in figure_pairs:
        write_simple_figure(
            exp_dir / f"figures/{stem}.png",
            exp_dir / f"figures/{stem}.pdf",
            title,
            counts.keys(),
            counts.values())

    effective = [row["variant_id"] for row in eval_rows
                 if row.get("module_effect_verdict", "").startswith("EFFECTIVE")]
    unsafe = [row["variant_id"] for row in eval_rows
              if row.get("variant_status", "").startswith("FAIL")]
    risk = [row["variant_id"] for row in eval_rows
            if "RISK" in row.get("variant_status", "")
            or "RISK" in row.get("module_effect_verdict", "")]
    diagnostic = [row["variant_id"] for row in eval_rows
                  if row.get("module_effect_verdict") in {
                      "DIAGNOSTIC_ONLY", "BASELINE_EQUIVALENCE_PASS"}]
    support_collapse = any("SUPPORT_COLLAPSE" in row.get("variant_status", "")
                           for row in eval_rows)
    det_explosion = any("DET_EXPLOSION" in row.get("variant_status", "")
                        for row in eval_rows)
    true_sv_damage = any("TRUE_SV_DAMAGE" in row.get("variant_status", "")
                         for row in eval_rows)
    migration = any("MIGRATION" in row.get("variant_status", "")
                    for row in eval_rows)

    report_md = exp_dir / "reports/focus_module_attribution_report.md"
    report_html = exp_dir / "reports/focus_module_attribution_report.html"
    md = [
        "# FOCUS-OVD Module Attribution Report",
        "",
        "## Executive verdict",
        "",
        f"- Evaluated/planned rows in current runner output: {len(eval_rows)}",
        f"- Inventory rows: {len(inventory_rows)}",
        f"- Matrix variants: {len(matrix_rows)}",
        f"- Effective modules with metric support: {', '.join(effective) if effective else 'none in this interim pass'}",
        f"- Unsafe modules detected by metrics: {', '.join(unsafe) if unsafe else 'none evaluated as unsafe; most train/eval rows remain not evaluated'}",
        f"- Metric-supported risk flags: {', '.join(risk) if risk else 'none in evaluated rows'}",
        f"- Diagnostic-only evidence: {', '.join(diagnostic) if diagnostic else 'none'}",
        "- Main-method promotion: not allowed from AP-only evidence alone; corrected-FSV and safety metrics are required.",
        "",
        "## Core FOCUS results",
        "",
        "- `V01_focus_zero` can only support baseline-equivalence if the smoke artifact reports `PASS`.",
        "- `V10_orientation_probe_only` is diagnostic and cannot justify output changes.",
        "- `V11_orientation_adapter_sv_only` may contain DOTA2-only full AP evidence if ingested from the completed recovery run.",
        "- Loss-attribution variants remain blocked from claims until their own training/small eval is explicitly run.",
        "",
        "## Added modules",
        "",
        "- Cluster mining, attribute gate, channel mask, orbit teacher, head consensus, negative prompt, and DINO CCL are implemented as controlled audit/planning layers.",
        "- None of these modules directly replaces OpenRSD native support or DeCLIP support.",
        "- Head consensus explicitly reports `UNSUPPORTED_BY_CURRENT_CODE` when hooks are unavailable.",
        "",
        "## Negative controls",
        "",
        "- Direction text prompt only remains a negative-control route.",
        "- Direct DeCLIP support remains a negative-control route and is not recommended for the main method.",
        "- All-class residual is not default and is treated as a migration-risk stress test.",
        "",
        "## Safety analysis",
        "",
        f"- Support collapse observed: {support_collapse}",
        f"- Detection explosion observed: {det_explosion}",
        f"- True-SV damage observed: {true_sv_damage}",
        f"- Class migration observed: {migration}",
        "- These booleans are only meaningful for rows with actual metrics; `NOT_EVALUATED` rows do not clear safety.",
        "",
        "## Final recommendation",
        "",
        "A. Main-method modules: V11 is AP-positive only when the full DOTA2-only run is ingested; it still needs corrected-FSV and safety evidence before promotion.",
        "B. Auxiliary/safety modules: safety gate, orbit teacher, visual attribute score, and cluster mining are suitable for controlled P1 evidence generation.",
        "C. Negative/diagnostic modules: direction text prompt only, direct DeCLIP support, all-class residual, head consensus without hooks, and negative prompt vocabulary.",
        "",
        "## Effectiveness table",
        "",
    ]
    md.extend(markdown_table(effectiveness_rows, [
        "variant_id", "status", "verdict", "primary_metrics"]))
    md.extend(["", "## Recommendation table", ""])
    md.extend(markdown_table(recommendation, [
        "module", "category", "recommendation"]))
    report_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    write_html(report_md, report_html)
    append_manifest_record(
        exp_dir, repo_root=args.repo_root.resolve(), stage="attribution_report",
        status="WRITTEN", module_switches={"report_rows": len(eval_rows)})
    print(json.dumps({"report": str(report_md), "rows": len(eval_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
