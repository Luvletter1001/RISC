#!/usr/bin/env python3
"""Plan DINO-support CCL controls without enabling DeCLIP as main support."""

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
    "control_id", "description", "support_source", "ccl_enabled",
    "negative_control", "status", "notes",
]


def build_rows() -> list[dict[str, str]]:
    return [
        {"control_id": "C0_baseline", "description": "Native baseline",
         "support_source": "native_dino_a10", "ccl_enabled": "false",
         "negative_control": "false", "status": "COMMAND_PLANNED",
         "notes": "reference only"},
        {"control_id": "C1_DINO_support_CCL", "description": "Native support plus CCL",
         "support_source": "native_dino_a10", "ccl_enabled": "true",
         "negative_control": "false", "status": "COMMAND_PLANNED",
         "notes": "separates CCL from DeCLIP support replacement"},
        {"control_id": "C2_DeCLIP_support_only_NEGATIVE_CONTROL",
         "description": "Direct DeCLIP support only",
         "support_source": "declip_direct", "ccl_enabled": "false",
         "negative_control": "true", "status": "NEGATIVE_CONTROL_PLANNED",
         "notes": "must not enter main method"},
        {"control_id": "C3_DeCLIP_support_CCL_NEGATIVE_CONTROL",
         "description": "Direct DeCLIP support plus CCL",
         "support_source": "declip_direct", "ccl_enabled": "true",
         "negative_control": "true", "status": "NEGATIVE_CONTROL_PLANNED",
         "notes": "must not enter main method"},
        {"control_id": "C4_DeCLIP_distilled_to_DINO_if_available",
         "description": "DeCLIP prototype distilled into native support space",
         "support_source": "declip_distilled_to_dino", "ccl_enabled": "true",
         "negative_control": "false", "status": "NOT_IMPLEMENTED",
         "notes": "allowed only as geometry-preserving adapter/distillation"},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path,
                        default=DEFAULT_EXP_DIR / "ccl_control")
    args = parser.parse_args()
    exp_dir = args.output_dir.parent
    ensure_exp_tree(exp_dir)
    rows = build_rows()
    csv_path = args.output_dir / "focus_ccl_control_summary.csv"
    report = args.output_dir / "focus_ccl_control_report.md"
    write_csv_rows(csv_path, rows, FIELDS)
    md = [
        "# FOCUS-OVD DINO-Support CCL Control",
        "",
        "- DINO support + CCL uses native DINO/A10 support.",
        "- DeCLIP variants are marked negative controls and are not main-method candidates.",
        "",
    ]
    md.extend(markdown_table(rows, FIELDS))
    report.write_text("\n".join(md) + "\n", encoding="utf-8")
    append_manifest_record(
        exp_dir, repo_root=args.repo_root.resolve(), stage="ccl_control",
        status="COMMANDS_PLANNED",
        module_switches={"dino_support_ccl": "controlled_variant_only"})
    print(json.dumps({"csv": str(csv_path), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
