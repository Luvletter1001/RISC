#!/usr/bin/env python3
"""Build FOCUS-EQText safety verdict and report."""

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
    EXP_DIR,
    ensure_exp_tree,
    md_table,
    read_json,
    resolve,
    write_csv,
    write_json,
)


def by_variant(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["variant_id"]): row for row in rows}


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def build_verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = by_variant(rows)
    baseline = lookup["EQ_V00_baseline"]
    zero = lookup["EQ_V01_focus_zero_dual"]
    head = lookup["EQ_V10_head_only"]
    text = lookup["EQ_V20_text_eq_only"]
    dual = lookup["EQ_V30_dual_eqtext"]
    base_fsv = as_float(baseline, "corrected_FSV")
    head_reduction = as_float(head, "corrected_FSV_reduction")
    dual_reduction = as_float(dual, "corrected_FSV_reduction")
    text_reduction = as_float(text, "corrected_FSV_reduction")
    train_evidence_ok = (
        str(head.get("p1a_realbatch_status", "")).startswith("PASS_")
        and str(text.get("p1a_realbatch_status", "")).startswith("PASS_")
        and str(dual.get("p1a_realbatch_status", "")).startswith("PASS_")
        and bool(head.get("actual_detector_train", False))
        and bool(text.get("actual_detector_train", False))
        and bool(dual.get("actual_detector_train", False)))
    baseline_equivalence = (
        as_float(zero, "corrected_FSV") == base_fsv
        and as_float(zero, "dense_sv_ratio") == as_float(baseline, "dense_sv_ratio")
        and as_float(zero, "dual_text_weight") == 0.0)
    checks = [
        {
            "criterion": "corrected_FSV reduction >= head_only",
            "ok": dual_reduction >= head_reduction,
            "detail": f"dual={dual_reduction}, head={head_reduction}",
        },
        {
            "criterion": "true retention >= 90%",
            "ok": as_float(dual, "true_SV_positive_control_retention") >= 0.90,
            "detail": dual["true_SV_positive_control_retention"],
        },
        {
            "criterion": "annotation-missing retention >= 90%",
            "ok": as_float(dual, "annotation_missing_true_vehicle_retention") >= 0.90,
            "detail": dual["annotation_missing_true_vehicle_retention"],
        },
        {
            "criterion": "text geometry does not collapse",
            "ok": (
                as_float(dual, "text_delta_norm") <= 0.05
                and as_float(dual, "text_support_cos_max") < 0.90),
            "detail": (
                f"text_delta={dual['text_delta_norm']}, "
                f"text_cos_max={dual['text_support_cos_max']}"),
        },
        {
            "criterion": "dual_text_weight <= 0.2",
            "ok": as_float(dual, "dual_text_weight") <= 0.200001,
            "detail": dual["dual_text_weight"],
        },
        {
            "criterion": "migration not worse",
            "ok": as_float(dual, "migration_mass_ratio") <= as_float(
                baseline, "migration_mass_ratio"),
            "detail": (
                f"dual={dual['migration_mass_ratio']}, "
                f"baseline={baseline['migration_mass_ratio']}"),
        },
        {
            "criterion": "degenerate_large_sv_ratio not increased",
            "ok": as_float(dual, "degenerate_large_sv_ratio") <= as_float(
                baseline, "degenerate_large_sv_ratio"),
            "detail": (
                f"dual={dual['degenerate_large_sv_ratio']}, "
                f"baseline={baseline['degenerate_large_sv_ratio']}"),
        },
        {
            "criterion": "focus_zero_dual baseline-equivalent",
            "ok": baseline_equivalence,
            "detail": (
                f"zero_corrected={zero['corrected_FSV']}, "
                f"baseline_corrected={baseline['corrected_FSV']}"),
        },
        {
            "criterion": "text side has independent signal",
            "ok": text_reduction > 0.0 and as_float(text, "text_delta_norm") > 0.0,
            "detail": (
                f"text_reduction={text_reduction}, "
                f"text_delta={text['text_delta_norm']}"),
        },
        {
            "criterion": "detector-level train evidence passed",
            "ok": train_evidence_ok,
            "detail": (
                f"head={head.get('p1a_realbatch_status')}, "
                f"text={text.get('p1a_realbatch_status')}, "
                f"dual={dual.get('p1a_realbatch_status')}"),
        },
    ]
    p2 = all(row["ok"] for row in checks)
    return {
        "status": "PASS_FOCUS_EQTEXT_SAFETY_REPORT",
        "recommend_p2": bool(p2),
        "recommendation": (
            "ENTER_P2" if p2 else "DO_NOT_ENTER_P2"),
        "ap_status": "AP_BLOCKED",
        "checks": checks,
        "answers": {
            "dual_zero_equivalent_baseline": baseline_equivalence,
            "text_side_independently_effective": checks[-2]["ok"],
            "dual_exceeds_head_only": dual_reduction >= head_reduction,
            "text_anchor_stable_geometry": checks[3]["ok"],
            "corrected_FSV_decreased": as_float(dual, "corrected_FSV") < base_fsv,
            "true_retention_preserved": checks[1]["ok"],
            "migration_worsened": not checks[5]["ok"],
            "AP_available": False,
            "P2_recommended": bool(p2),
        },
    }


def write_markdown(path: Path, verdict: dict[str, Any],
                   rows: list[dict[str, Any]]) -> None:
    metric_fields = [
        "variant_id", "corrected_FSV", "corrected_FSV_reduction",
        "dense_sv_ratio", "true_SV_positive_control_retention",
        "annotation_missing_true_vehicle_retention", "migration_mass_ratio",
        "degenerate_large_sv_ratio", "visual_delta_norm", "text_delta_norm",
        "text_support_cos_max", "dual_text_weight", "AP", "mAP",
    ]
    answer_rows = [
        {"question": key, "answer": value}
        for key, value in verdict["answers"].items()
    ]
    lines = [
        "# FOCUS-EQText DOTA Short Report",
        "",
        f"- status: `{verdict['status']}`",
        f"- recommendation: `{verdict['recommendation']}`",
        f"- AP status: `{verdict['ap_status']}`",
        "",
        "## Required Final Answers",
        "",
    ]
    lines.extend(md_table(answer_rows, ["question", "answer"]))
    lines.extend(["", "## Variant Metrics", ""])
    lines.extend(md_table(rows, metric_fields))
    lines.extend(["", "## P2 Gates", ""])
    lines.extend(md_table(verdict["checks"], ["criterion", "ok", "detail"]))
    lines.extend([
        "",
        "## Scope Notes",
        "",
        "- DeCLIP support was not enabled.",
        "- Native OpenRSD support bank was not replaced.",
        "- Text encoder is not trained by this experiment.",
        "- AP/mAP is `AP_BLOCKED` because no full DOTA detector AP evaluation was run in this short pass.",
        "- Corrected-FSV and safety rows are short safety metrics paired with P1A realbatch train evidence, not full benchmark AP evidence.",
        "- This short P1A realbatch adapter loop is single-process; subsequent training-mode runs must use a dual-card launcher on physical GPUs 6 and 9.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--exp-dir", type=Path, default=EXP_DIR)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    exp_dir = resolve(repo_root, args.exp_dir)
    ensure_exp_tree(exp_dir)
    eval_payload = read_json(
        exp_dir / "eval/focus_eqtext_eval_all_variants.json", {})
    rows = list(eval_payload.get("rows", []))
    verdict = build_verdict(rows)
    write_json(exp_dir / "reports/focus_eqtext_safety_report.json", verdict)
    write_csv(exp_dir / "reports/focus_eqtext_p2_gates.csv", verdict["checks"])
    report_md = exp_dir / "reports/focus_eqtext_safety_report.md"
    write_markdown(report_md, verdict, rows)
    print(json.dumps({
        "status": verdict["status"],
        "recommendation": verdict["recommendation"],
        "report": str(report_md),
    }, indent=2))
    return 0 if verdict["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
