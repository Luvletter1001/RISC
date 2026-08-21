#!/usr/bin/env python3
"""Audit head consensus for FOCUS-OVD candidates."""

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
)


FIELDS = [
    "candidate_id", "tile_id", "angle", "bbox_head_sv_score",
    "alignment_head_sv_score", "fusion_head_sv_score", "support_margin",
    "head_agreement_score", "head_disagreement_type", "consensus_label",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=DEFAULT_EXP_DIR / "head_consensus")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    exp_dir = args.output_dir.parent
    ensure_exp_tree(exp_dir)
    consensus_util = load_focus_util(repo_root, "focus_head_consensus")
    input_rows = read_csv_rows(args.input_csv) if args.input_csv else []
    rows = []
    for idx, row in enumerate(input_rows):
        result = consensus_util.score_head_consensus(
            bbox_head_sv_score=safe_float(row.get("bbox_head_sv_score")),
            alignment_head_sv_score=safe_float(row.get("alignment_head_sv_score")),
            fusion_head_sv_score=safe_float(row.get("fusion_head_sv_score")),
            support_margin=safe_float(row.get("support_margin")))
        out = {key: row.get(key, "") for key in FIELDS}
        out["candidate_id"] = out["candidate_id"] or row.get("crop_id", f"candidate_{idx:06d}")
        out["head_agreement_score"] = result.head_agreement_score
        out["head_disagreement_type"] = result.head_disagreement_type
        out["consensus_label"] = result.consensus_label
        rows.append(out)
    status = "WRITTEN" if rows else "UNSUPPORTED_BY_CURRENT_CODE"
    csv_path = args.output_dir / "focus_head_consensus_scores.csv"
    report = args.output_dir / "focus_head_consensus_report.md"
    write_csv_rows(csv_path, rows, FIELDS)
    report.write_text(
        "# FOCUS-OVD Head Consensus Audit\n\n"
        f"- Status: {status}\n"
        f"- Scores: `{csv_path}`\n"
        "- If alignment/fusion/bbox hooks are unavailable, this module remains unsupported and no values are fabricated.\n",
        encoding="utf-8")
    append_manifest_record(
        exp_dir, repo_root=repo_root, stage="head_consensus",
        status=status, module_switches={"head_consensus": "audit_only"},
        failure_reason="" if rows else "no multi-head score input provided")
    print(json.dumps({"csv": str(csv_path), "rows": len(rows), "status": status}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
