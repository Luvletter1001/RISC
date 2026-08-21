#!/usr/bin/env python3
"""Audit negative-aware prompt vocabulary as auxiliary evidence only."""

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


FIELDS = ["candidate_id", "positive_max_sim", "negative_max_sim", "auxiliary_margin"]
VOCAB_FIELDS = ["prompt", "polarity"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--similarity-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=DEFAULT_EXP_DIR / "negative_prompt")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    exp_dir = args.output_dir.parent
    ensure_exp_tree(exp_dir)
    neg_util = load_focus_util(repo_root, "focus_negative_prompt")
    vocab_rows = neg_util.build_negative_prompt_rows()
    sim_rows = read_csv_rows(args.similarity_csv) if args.similarity_csv else []
    out_rows = []
    for idx, row in enumerate(sim_rows):
        pos = {"positive": safe_float(row.get("positive_max_sim"))}
        neg = {"negative": safe_float(row.get("negative_max_sim"))}
        margin = neg_util.auxiliary_margin_from_similarities(pos, neg)
        out_rows.append({
            "candidate_id": row.get("candidate_id", f"candidate_{idx:06d}"),
            "positive_max_sim": pos["positive"],
            "negative_max_sim": neg["negative"],
            "auxiliary_margin": margin,
        })
    scores_csv = args.output_dir / "focus_negative_prompt_scores.csv"
    vocab_csv = args.output_dir / "focus_negative_prompt_vocabulary.csv"
    report = args.output_dir / "focus_negative_prompt_report.md"
    write_csv_rows(scores_csv, out_rows, FIELDS)
    write_csv_rows(vocab_csv, vocab_rows, VOCAB_FIELDS)
    report.write_text(
        "# FOCUS-OVD Negative-Aware Prompt Audit\n\n"
        f"- Status: {'WRITTEN' if out_rows else 'VOCABULARY_ONLY'}\n"
        f"- Scores: `{scores_csv}`\n"
        f"- Vocabulary: `{vocab_csv}`\n"
        "- Negative-aware vocabulary is auxiliary only and does not replace native OpenRSD prompts or support logits.\n"
        "- Direction text prompt only is included only as a negative control.\n",
        encoding="utf-8")
    append_manifest_record(
        exp_dir, repo_root=repo_root, stage="negative_prompt",
        status="WRITTEN" if out_rows else "VOCABULARY_ONLY",
        module_switches={"negative_prompt": "auxiliary_only"})
    print(json.dumps({"scores": str(scores_csv), "rows": len(out_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
