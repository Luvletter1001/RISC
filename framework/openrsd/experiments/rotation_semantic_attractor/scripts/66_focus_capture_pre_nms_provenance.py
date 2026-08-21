#!/usr/bin/env python3
"""Diagnose availability of exact pre-NMS provenance for FOCUS P1B.

The existing verified crop metadata points to final-stage prediction records.
This script records that limitation explicitly and writes an empty provenance
candidate table instead of fabricating feature-level/grid provenance.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_EXP_DIR = Path(
    "resultmd/exp_focus_ovd_p1_unblock_loss_mapping_20260609")
DEFAULT_TARGETS = DEFAULT_EXP_DIR / "targets/focus_spatial_region_targets.csv"
FIELDS = [
    "prediction_id", "tile_id", "angle", "class_name", "score", "box",
    "feature_level", "grid_x", "grid_y", "anchor_id_or_point_id",
    "source_stage", "capture_status", "invalid_reason",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]],
              fields: list[str] = FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# FOCUS P1 Pre-NMS Provenance Capture",
        "",
        f"- status: `{payload['status']}`",
        f"- actual_pre_nms_capture: `{payload['actual_pre_nms_capture']}`",
        f"- provenance_candidates: `{payload['provenance_candidate_count']}`",
        f"- final_stage_source_rows: `{payload['source_stage_counts'].get('final', 0)}`",
        "",
        "## Interpretation",
        "",
        "Existing crop records are final-stage predictions. They do not carry "
        "`feature_level`, `grid_x`, `grid_y`, or anchor/point identifiers, so "
        "they cannot be used as exact pre-NMS detector loss targets.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--exp-dir", type=Path, default=DEFAULT_EXP_DIR)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    exp_dir = args.exp_dir if args.exp_dir.is_absolute() else repo / args.exp_dir
    targets = args.targets if args.targets.is_absolute() else repo / args.targets
    provenance_dir = exp_dir / "provenance"
    rows = read_csv(targets)
    source_stage_counts = Counter(row.get("source_stage", "") for row in rows)
    candidate_csv = provenance_dir / "focus_pre_nms_provenance_candidates.csv"
    write_csv(candidate_csv, [])
    payload = {
        "status": "PRE_NMS_PROVENANCE_NOT_AVAILABLE_EXISTING_FINAL_ONLY",
        "actual_pre_nms_capture": False,
        "target_csv": str(targets),
        "candidate_csv": str(candidate_csv),
        "input_target_rows": len(rows),
        "provenance_candidate_count": 0,
        "source_stage_counts": dict(source_stage_counts),
        "required_fields_missing": [
            "feature_level", "grid_x", "grid_y", "anchor_id_or_point_id",
        ],
        "evidence_level": "FINAL_STAGE_ONLY_DIAGNOSTIC",
    }
    write_json(provenance_dir / "focus_pre_nms_capture_summary.json", payload)
    write_markdown(provenance_dir / "focus_pre_nms_capture_summary.md", payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
