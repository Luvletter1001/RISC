#!/usr/bin/env python3
"""Build deterministic FOCUS-OVD P0 train/eval/safety splits."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from focus_p0_common import (
    DEFAULT_RAW_LABEL_CSV,
    append_manifest,
    build_stratified_splits,
    ensure_exp_tree,
    focus_label,
    labeled_rows,
    markdown_table,
    write_json,
)


def summary_for(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        counts[focus_label(row)] += 1
        counts[f"category:{row.get('audit_category', '')}"] += 1
    return dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--human-label-csv", type=Path, default=DEFAULT_RAW_LABEL_CSV)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260609)
    args = parser.parse_args()

    exp_dir = args.output_dir.parent
    ensure_exp_tree(exp_dir)
    rows = labeled_rows(args.human_label_csv)
    splits = build_stratified_splits(rows, seed=args.seed, train_ratio=0.8)
    train_ids = {r["crop_id"] for r in splits["train"]}
    eval_ids = {r["crop_id"] for r in splits["eval"]}
    leakage = sorted(train_ids & eval_ids)
    status = "PASS" if not leakage else "FAIL_LEAKAGE"
    payloads = {
        "p0_train_split.json": {"status": status, "seed": args.seed, "rows": splits["train"]},
        "p0_eval_split.json": {"status": status, "seed": args.seed, "rows": splits["eval"]},
        "p0_safety_split.json": {"status": status, "seed": args.seed, "rows": splits["safety"]},
    }
    for name, payload in payloads.items():
        write_json(args.output_dir / name, payload)
    summary_rows = [
        {"split": "train", "count": len(splits["train"]), "summary": json.dumps(summary_for(splits["train"]), sort_keys=True)},
        {"split": "eval", "count": len(splits["eval"]), "summary": json.dumps(summary_for(splits["eval"]), sort_keys=True)},
        {"split": "safety", "count": len(splits["safety"]), "summary": json.dumps(summary_for(splits["safety"]), sort_keys=True)},
    ]
    md = [
        "# FOCUS-OVD P0 Split Summary",
        "",
        f"- status: `{status}`",
        f"- seed: `{args.seed}`",
        f"- leakage_count: `{len(leakage)}`",
        "",
    ]
    md.extend(markdown_table(summary_rows, ["split", "count", "summary"]))
    (exp_dir / "reports/p0_split_summary.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    append_manifest(
        exp_dir, args.repo_root.resolve(),
        {"stage": "p0_split_build", "status": status, "seed": args.seed, "leakage": leakage})
    print(json.dumps({"status": status, "train": len(splits["train"]), "eval": len(splits["eval"]), "safety": len(splits["safety"])}, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
