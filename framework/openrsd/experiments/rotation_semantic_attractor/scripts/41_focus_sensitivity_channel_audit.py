#!/usr/bin/env python3
"""Audit sensitivity-guided channel mask candidates."""

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
    "channel_id", "sensitivity_rotation", "sensitivity_context",
    "sensitivity_support_intervention", "true_sv_importance",
    "false_sv_importance", "mask_score", "mask_value",
    "recommended_action",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=DEFAULT_EXP_DIR / "channel_mask")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    exp_dir = args.output_dir.parent
    ensure_exp_tree(exp_dir)
    mask_util = load_focus_util(repo_root, "focus_sensitivity_channel_mask")
    input_rows = read_csv_rows(args.input_csv) if args.input_csv else []
    if input_rows:
        rotation = [safe_float(r.get("sensitivity_rotation")) for r in input_rows]
        context = [safe_float(r.get("sensitivity_context")) for r in input_rows]
        support = [safe_float(r.get("sensitivity_support_intervention")) for r in input_rows]
        true_imp = [safe_float(r.get("true_sv_importance")) for r in input_rows]
        false_imp = [safe_float(r.get("false_sv_importance")) for r in input_rows]
        rows = mask_util.summarize_sensitive_channels(
            rotation, context, support, true_imp, false_imp, max_strength=0.05)
    else:
        rows = []
    csv_path = args.output_dir / "focus_sensitive_channels.csv"
    report_path = args.output_dir / "focus_channel_mask_report.md"
    write_csv_rows(csv_path, rows, FIELDS)
    report_path.write_text(
        "# FOCUS-OVD Sensitivity-Guided Channel Mask Audit\n\n"
        f"- Status: {'WRITTEN' if rows else 'NO_INPUT'}\n"
        f"- Output: `{csv_path}`\n"
        "- First version is audit-only. Any enabled mask must use strength <= 0.05 and report AP/SV_AP/true-SV retention.\n",
        encoding="utf-8")
    write_simple_figure(
        args.output_dir / "focus_channel_sensitivity_heatmap.png",
        args.output_dir / "focus_channel_sensitivity_heatmap.pdf",
        "Channel Sensitivity",
        [str(r["channel_id"]) for r in rows[:20]],
        [float(r["mask_score"]) for r in rows[:20]])
    append_manifest_record(
        exp_dir, repo_root=repo_root, stage="channel_mask",
        status="WRITTEN" if rows else "NO_INPUT",
        module_switches={"channel_mask": "audit_only", "max_strength": 0.05},
        failure_reason="" if rows else "no channel sensitivity input provided")
    print(json.dumps({"csv": str(csv_path), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
