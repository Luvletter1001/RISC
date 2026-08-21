#!/usr/bin/env python3
"""Audit visual attribute gate scores for FOCUS-OVD."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from focus_attribution_common import (
    DEFAULT_CORRECTED_FSV_CSV,
    DEFAULT_EXP_DIR,
    append_manifest_record,
    ensure_exp_tree,
    load_focus_util,
    read_csv_rows,
    safe_float,
    write_csv_rows,
    write_simple_figure,
)


FIELDS = [
    "candidate_id", "tile_id", "angle", "human_label", "audit_category",
    "attribute_sv_confidence", "positive_attributes", "negative_attributes",
    "attribute_raw_score",
]


def label_value(row: dict[str, str]) -> int | None:
    label = row.get("human_label", "")
    if label in {"true_vehicle_annot_missing",
                 "true_vehicle_annot_present_but_missed_match",
                 "true_small_vehicle", "matched_true_small_vehicle"}:
        return 1
    if label in {"non_vehicle_background", "non_vehicle_object_conflict"}:
        return 0
    return None


def roc_rows(rows: list[dict[str, str]]) -> list[dict[str, float]]:
    labelled = [(safe_float(r.get("attribute_sv_confidence")), label_value(r))
                for r in rows]
    labelled = [(score, label) for score, label in labelled if label is not None]
    out = []
    for threshold in [i / 20.0 for i in range(21)]:
        tp = sum(1 for score, label in labelled if score >= threshold and label == 1)
        fp = sum(1 for score, label in labelled if score >= threshold and label == 0)
        fn = sum(1 for score, label in labelled if score < threshold and label == 1)
        tn = sum(1 for score, label in labelled if score < threshold and label == 0)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        false_reject = fn / max(tp + fn, 1)
        false_accept = fp / max(fp + tn, 1)
        out.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "true_SV_retention": recall,
            "false_SV_rejection": tn / max(fp + tn, 1),
            "ambiguous_rate": 0.0,
            "false_reject_rate": false_reject,
            "false_accept_rate": false_accept,
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_CORRECTED_FSV_CSV)
    parser.add_argument("--output-dir", type=Path,
                        default=DEFAULT_EXP_DIR / "attribute_gate")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    exp_dir = args.output_dir.parent
    ensure_exp_tree(exp_dir)
    attr_util = load_focus_util(repo_root, "focus_visual_attribute_gate")
    rows = read_csv_rows(args.input_csv)
    scored = attr_util.annotate_attribute_scores(rows) if rows else []
    for idx, row in enumerate(scored):
        row.setdefault("candidate_id", row.get("crop_id", f"candidate_{idx:06d}"))
        row.setdefault("tile_id", row.get("image_id", ""))
        row.setdefault("angle", row.get("rotation_angle", ""))
    score_csv = args.output_dir / "focus_attribute_gate_scores.csv"
    roc_csv = args.output_dir / "focus_attribute_gate_roc.csv"
    report = args.output_dir / "focus_attribute_gate_report.md"
    write_csv_rows(score_csv, scored, FIELDS)
    roc = roc_rows(scored)
    write_csv_rows(roc_csv, roc, [
        "threshold", "precision", "recall", "true_SV_retention",
        "false_SV_rejection", "ambiguous_rate", "false_reject_rate",
        "false_accept_rate"])
    examples_dir = args.output_dir / "focus_attribute_gate_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    top = sorted(scored, key=lambda r: safe_float(r.get("attribute_sv_confidence")),
                 reverse=True)[:10]
    (examples_dir / "top_attribute_scores.json").write_text(
        json.dumps(top, indent=2, ensure_ascii=False), encoding="utf-8")
    best_retention = max((r["true_SV_retention"] for r in roc), default=0.0)
    report.write_text(
        "# FOCUS-OVD Visual Attribute Gate Audit\n\n"
        f"- Status: {'WRITTEN' if rows else 'NO_INPUT'}\n"
        f"- Scores: `{score_csv}`\n"
        f"- ROC: `{roc_csv}`\n"
        f"- Max true-SV retention over thresholds: {best_retention:.4f}\n"
        "- Safety policy: scores are audit-only unless true-SV retention is >= 90% in a gated evaluation.\n",
        encoding="utf-8")
    write_simple_figure(
        args.output_dir / "focus_attribute_gate_score_hist.png",
        args.output_dir / "focus_attribute_gate_score_hist.pdf",
        "Attribute Gate Scores")
    append_manifest_record(
        exp_dir, repo_root=repo_root, stage="attribute_gate",
        status="WRITTEN" if rows else "NO_INPUT",
        module_switches={"attribute_gate": "audit_only"},
        failure_reason="" if rows else f"missing or empty input: {args.input_csv}")
    print(json.dumps({"csv": str(score_csv), "rows": len(scored)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
