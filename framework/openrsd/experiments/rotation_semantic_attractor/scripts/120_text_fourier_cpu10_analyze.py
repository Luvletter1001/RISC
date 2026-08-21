#!/usr/bin/env python3
"""Analyze and rank CPU-only text/Fourier methods."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_fourier_cpu10_common import (
    BASELINE_AUC,
    EXP_DIR,
    ensure_tree,
    read_csv,
    resolve,
    summarize_ideas,
    summarize_methods,
    write_csv,
    write_json,
)


SUMMARY_FIELDS = [
    "rank",
    "method_id",
    "name_zh",
    "auc",
    "auc_delta_vs_tsafe",
    "rank_score",
    "degenerate_top20_share",
    "padding_top20_share",
    "true_retention_proxy",
    "score_std",
    "collapse_flag",
    "one_way_down_only",
    "safe_candidate",
    "reject_reason",
    "description_zh",
]

IDEA_SUMMARY_FIELDS = [
    "rank",
    "idea_id",
    "title_zh",
    "source_ids",
    "auc",
    "auc_delta_vs_tsafe",
    "rank_score",
    "degenerate_top20_share",
    "padding_top20_share",
    "true_retention_proxy",
    "score_std",
    "collapse_flag",
    "one_way_down_only",
    "safe_candidate",
    "reject_reason",
    "sample_count",
    "labeled_count",
    "evidence_status",
    "verification_zh",
]


def write_rank_figure(summary: list[dict[str, object]], path_base: Path) -> dict[str, str]:
    paths = {
        "rank_png": str(path_base.with_suffix(".png")),
        "rank_pdf": str(path_base.with_suffix(".pdf")),
    }
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [str(row["method_id"]).split("_")[0] for row in summary]
        values = [float(row["rank_score"]) for row in summary]
        path_base.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.bar(labels, values)
        plt.ylabel("rank score")
        plt.xlabel("method")
        plt.title("CPU-only text/Fourier proxy ranking")
        plt.tight_layout()
        plt.savefig(paths["rank_png"])
        plt.savefig(paths["rank_pdf"])
        plt.close()
    except Exception as exc:
        for value in paths.values():
            path = Path(value)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"Figure unavailable: {exc}\n", encoding="utf-8")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--exp-dir", type=Path, default=EXP_DIR)
    parser.add_argument("--scores-csv", type=Path,
                        default=EXP_DIR / "tables" / "text_fourier_cpu10_sample_scores.csv")
    parser.add_argument("--baseline-auc", type=float, default=BASELINE_AUC)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    exp_dir = resolve(repo_root, args.exp_dir)
    ensure_tree(exp_dir)
    scores_csv = resolve(repo_root, args.scores_csv)
    rows = read_csv(scores_csv)
    summary = summarize_methods(rows, baseline_auc=args.baseline_auc)
    idea_summary = summarize_ideas(rows, baseline_auc=args.baseline_auc)
    summary_csv = exp_dir / "tables" / "text_fourier_cpu10_method_summary.csv"
    write_csv(summary_csv, summary, SUMMARY_FIELDS)
    idea_summary_csv = exp_dir / "tables" / "text_fourier_cpu100_idea_summary.csv"
    write_csv(idea_summary_csv, idea_summary, IDEA_SUMMARY_FIELDS)
    figures = write_rank_figure(
        summary,
        exp_dir / "figures" / "text_fourier_cpu10_rank_score")
    payload = {
        "status": "PASS_METHOD_ANALYSIS" if summary else "NO_METHOD_SUMMARY",
        "scores_csv": str(scores_csv),
        "summary_csv": str(summary_csv),
        "idea_summary_csv": str(idea_summary_csv),
        "baseline_auc": args.baseline_auc,
        "method_count": len(summary),
        "idea_count": len(idea_summary),
        "best_method": summary[0] if summary else {},
        "best_idea": idea_summary[0] if idea_summary else {},
        "safe_candidate_count": sum(bool(row["safe_candidate"]) for row in summary),
        "safe_idea_candidate_count": sum(bool(row["safe_candidate"]) for row in idea_summary),
        **figures,
    }
    write_json(exp_dir / "reports" / "text_fourier_cpu10_analysis.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if summary and idea_summary else 2


if __name__ == "__main__":
    raise SystemExit(main())
