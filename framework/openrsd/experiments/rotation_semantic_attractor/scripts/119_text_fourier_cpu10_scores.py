#!/usr/bin/env python3
"""Compute per-sample scores for ten CPU-only text/Fourier methods."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_fourier_cpu10_common import (
    BASELINE_SCORES_CSV,
    EXP_DIR,
    HUMAN_LABEL_CSV,
    ensure_tree,
    idea_fields,
    load_baseline_by_crop,
    read_csv,
    resolve,
    row_to_sample,
    sample_to_row,
    score_row_fields,
    write_csv,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--human-label-csv", type=Path, default=HUMAN_LABEL_CSV)
    parser.add_argument("--baseline-scores-csv", type=Path,
                        default=BASELINE_SCORES_CSV)
    parser.add_argument("--output-dir", type=Path, default=EXP_DIR)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    exp_dir = resolve(repo_root, args.output_dir)
    ensure_tree(exp_dir)
    human_csv = resolve(repo_root, args.human_label_csv)
    baseline_csv = resolve(repo_root, args.baseline_scores_csv)
    human_rows = read_csv(human_csv)
    if args.limit > 0:
        human_rows = human_rows[:args.limit]
    baseline_by_crop = load_baseline_by_crop(baseline_csv)

    out_rows = []
    missing_baseline = 0
    for row in human_rows:
        crop_id = row.get("crop_id", "")
        baseline_row = baseline_by_crop.get(crop_id)
        if baseline_row is None:
            missing_baseline += 1
        sample = row_to_sample(row, baseline_row)
        out_rows.append(sample_to_row(sample, row))

    scores_path = exp_dir / "tables" / "text_fourier_cpu10_sample_scores.csv"
    write_csv(scores_path, out_rows, score_row_fields())
    idea_scores_path = exp_dir / "tables" / "text_fourier_cpu100_idea_scores.csv"
    write_csv(
        idea_scores_path,
        out_rows,
        [
            "crop_id",
            "audit_category",
            "human_label",
            "binary_true_vehicle",
            "focus_sv_score",
            *idea_fields(),
        ])
    manifest = {
        "status": "PASS_SAMPLE_SCORES_WRITTEN" if out_rows else "NO_INPUT_ROWS",
        "human_label_csv": str(human_csv),
        "baseline_scores_csv": str(baseline_csv),
        "scores_csv": str(scores_path),
        "idea_scores_csv": str(idea_scores_path),
        "row_count": len(out_rows),
        "idea_count": len(idea_fields()),
        "missing_baseline_rows": missing_baseline,
        "limit": args.limit,
        "cpu_only": True,
        "detector_run": False,
        "training_run": False,
        "final_logits_modified": False,
    }
    write_json(exp_dir / "reports" / "text_fourier_cpu10_scores_manifest.json",
               manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if out_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
