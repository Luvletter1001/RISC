#!/usr/bin/env python3
"""Generate rotation/orbit teacher labels for FOCUS-OVD."""

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
    safe_float,
    write_csv_rows,
    write_simple_figure,
)


FIELDS = [
    "candidate_id", "tile_id", "angle", "orbit_group_id",
    "sv_score_mean", "sv_score_std", "class_consistency", "box_consistency",
    "support_margin_consistency", "orientation_consistency",
    "orbit_stability_score", "teacher_label", "teacher_weight",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=DEFAULT_EXP_DIR / "orbit_teacher")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    exp_dir = args.output_dir.parent
    ensure_exp_tree(exp_dir)
    orbit_util = load_focus_util(repo_root, "focus_orbit_teacher")
    input_rows = read_csv_rows(args.input_csv) if args.input_csv else []
    rows = []
    for idx, row in enumerate(input_rows):
        label = orbit_util.assign_orbit_teacher_label(
            sv_score_mean=safe_float(row.get("sv_score_mean")),
            sv_score_std=safe_float(row.get("sv_score_std")),
            class_consistency=safe_float(row.get("class_consistency")),
            box_consistency=safe_float(row.get("box_consistency")),
            support_margin_consistency=safe_float(row.get("support_margin_consistency")),
            orientation_consistency=safe_float(row.get("orientation_consistency")),
        )
        out = {key: row.get(key, "") for key in FIELDS}
        out["candidate_id"] = out["candidate_id"] or row.get("crop_id", f"candidate_{idx:06d}")
        out["orbit_stability_score"] = label.orbit_stability_score
        out["teacher_label"] = label.teacher_label
        out["teacher_weight"] = label.teacher_weight
        rows.append(out)
    csv_path = args.output_dir / "focus_orbit_teacher_labels.csv"
    report = args.output_dir / "focus_orbit_teacher_report.md"
    write_csv_rows(csv_path, rows, FIELDS)
    report.write_text(
        "# FOCUS-OVD Orbit Consistency Teacher\n\n"
        f"- Status: {'WRITTEN' if rows else 'NO_INPUT'}\n"
        f"- Labels: `{csv_path}`\n"
        "- Human corrected-FSV labels take precedence; orbit labels are pseudo-label supplements only.\n",
        encoding="utf-8")
    write_simple_figure(
        args.output_dir / "focus_orbit_stability_by_category.png",
        args.output_dir / "focus_orbit_stability_by_category.pdf",
        "Orbit Stability by Category")
    append_manifest_record(
        exp_dir, repo_root=repo_root, stage="orbit_teacher",
        status="WRITTEN" if rows else "NO_INPUT",
        module_switches={"orbit_teacher": "labels_only"},
        failure_reason="" if rows else "no orbit consistency input provided")
    print(json.dumps({"csv": str(csv_path), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
