#!/usr/bin/env python3
"""Mine diagnostic spurious-SV clusters for FOCUS-OVD."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
    markdown_table,
    read_csv_rows,
    write_csv_rows,
    write_simple_figure,
)


FIELDS = [
    "candidate_id", "tile_id", "angle", "model", "pred_score", "sv_logit",
    "sv_margin", "orientation_theta", "orientation_confidence",
    "orbit_stability", "support_similarity_sv", "support_similarity_lv",
    "nearest_negative_class", "human_label", "audit_category", "cluster_id",
    "cluster_name", "spurious_sv_score",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_CORRECTED_FSV_CSV)
    parser.add_argument("--output-dir", type=Path,
                        default=DEFAULT_EXP_DIR / "cluster_mining")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    exp_dir = args.output_dir.parent
    ensure_exp_tree(exp_dir)
    cluster_util = load_focus_util(repo_root, "focus_spurious_cluster")

    rows = read_csv_rows(args.input_csv)
    annotated = cluster_util.annotate_spurious_clusters(rows) if rows else []
    for idx, row in enumerate(annotated):
        row.setdefault("candidate_id", row.get("crop_id", f"candidate_{idx:06d}"))
        row.setdefault("tile_id", row.get("image_id", ""))
        row.setdefault("angle", row.get("rotation_angle", ""))
        row.setdefault("model", row.get("model", ""))
        for key in FIELDS:
            row.setdefault(key, "")

    csv_path = args.output_dir / "focus_spurious_sv_clusters.csv"
    md_path = args.output_dir / "focus_spurious_sv_cluster_summary.md"
    write_csv_rows(csv_path, annotated, FIELDS)
    counts = Counter(row.get("cluster_name", "NO_INPUT") for row in annotated)
    summary_rows = [{"cluster_name": key, "count": value}
                    for key, value in sorted(counts.items())]
    md = [
        "# FOCUS-OVD Spurious SV Cluster Mining",
        "",
        f"- Input CSV: `{args.input_csv}`",
        f"- Status: {'WRITTEN' if rows else 'NO_INPUT'}",
        "- Policy: diagnostic score only; no logits or detections are changed.",
        "",
    ]
    md.extend(markdown_table(summary_rows, ["cluster_name", "count"])
              if summary_rows else ["No candidate rows were available."])
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    write_simple_figure(
        args.output_dir / "focus_spurious_sv_score_hist.png",
        args.output_dir / "focus_spurious_sv_score_hist.pdf",
        "Spurious SV Score Histogram",
        [row["cluster_name"] for row in summary_rows],
        [row["count"] for row in summary_rows])
    write_simple_figure(
        args.output_dir / "focus_spurious_sv_umap.png",
        args.output_dir / "focus_spurious_sv_umap.pdf",
        "Spurious SV Cluster Projection")
    append_manifest_record(
        exp_dir, repo_root=repo_root, stage="cluster_mining",
        status="WRITTEN" if rows else "NO_INPUT",
        module_switches={"spurious_cluster": "diagnostic_only"},
        failure_reason="" if rows else f"missing or empty input: {args.input_csv}")
    print(json.dumps({"csv": str(csv_path), "rows": len(annotated)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
