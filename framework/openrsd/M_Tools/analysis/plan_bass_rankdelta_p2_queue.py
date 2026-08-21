#!/usr/bin/env python
"""Decide whether the GPU6/GPU7 BASS-GSF P2 queue should launch.

P2 is deliberately gated behind the closure plan.  The queue launches only
after P0 and P1 are complete without a strict RankDelta pass, i.e. when the
closure planner emits ``RUN_P2_RANK_ASSIGNMENT``.  This avoids competing with
P0/P1 and keeps the paper from silently switching claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CLOSURE_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "bass_rankdelta_closure_plan.json")
OUT_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "bass_rankdelta_p2_queue_plan.json")
OUT_MD = Path(
    "resultmd/exp_p4_scale_semantic_validation/"
    "fstatus_20260620_bass_rankdelta_p2_queue_plan.md")
MARKER = Path(
    "work_dirs/train_queue_logs/"
    "bass_gsf_rankdelta_p2_launched_20260620.flag")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def plan_p2_queue(closure: dict[str, Any], marker: Path = MARKER
                  ) -> dict[str, Any]:
    closure_action = str(closure.get("action", "missing"))
    if closure_action == "STOP_P2_DONE_NO_STRICT":
        action = "STOP_P2_DONE_NO_STRICT"
        reason = (
            "P2 rows are complete and no strict pass exists; do not launch "
            "duplicate P2 jobs. Move to the Gaussian AP-constrained support "
            "projection route.")
    elif marker.exists():
        action = "WAIT_P2_EXISTING"
        reason = "A P2 launch marker exists; do not launch duplicate jobs."
    elif closure_action == "RUN_P2_RANK_ASSIGNMENT":
        action = "RUN_P2"
        reason = (
            "P0/P1 completed without strict pass; launch explicit "
            "rank/assignment BASS-GSF P2 variants on GPU6/GPU7.")
    elif closure_action == "RUN_CONFIRMATION_PACKAGE":
        action = "STOP_CONFIRMATION_ROUTE"
        reason = (
            "A strict RankDelta pass exists; run confirmation/ablation "
            "package instead of P2 rescue.")
    else:
        action = "WAIT_UPSTREAM"
        reason = (
            "P2 waits for closure action RUN_P2_RANK_ASSIGNMENT; current "
            f"closure action is {closure_action}.")

    return {
        "action": action,
        "reason": reason,
        "closure_action": closure_action,
        "marker": str(marker),
        "paper_position": closure.get("paper_position", ""),
        "queue": [
            {
                "gpu": 6,
                "variant": "assign_rank",
                "script": (
                    "M_Tools/experiments/"
                    "run_hrrsd_bass_gsf_p2_train_20260620.sh"),
                "args": "6 assign_rank 2",
                "claim_defended": (
                    "support-aware positive/hard-negative consistency plus "
                    "protected rank delta improves AP-risk Pareto status"),
            },
            {
                "gpu": 7,
                "variant": "posterior_rank",
                "script": (
                    "M_Tools/experiments/"
                    "run_hrrsd_bass_gsf_p2_train_20260620.sh"),
                "args": "7 posterior_rank 2",
                "claim_defended": (
                    "feature-conditioned posterior support with protected "
                    "rank delta preserves top-K ordering"),
            },
        ],
        "no_fabrication_rule": (
            "P2 queue planning launches no experiments by itself and writes no "
            "metrics. Paper rows may be filled only from eval JSON and "
            "deployment-risk summaries."),
    }


def write_markdown(path: Path, plan: dict[str, Any]) -> None:
    lines = [
        "# BASS-GSF P2 Queue Plan - 2026-06-20",
        "",
        "This file is generated from the RankDelta closure plan. It gates P2",
        "behind completed P0/P1 evidence and does not invent results.",
        "",
        "| field | value |",
        "|---|---|",
        f"| action | `{plan['action']}` |",
        f"| reason | {plan['reason']} |",
        f"| closure_action | `{plan['closure_action']}` |",
        f"| paper_position | {plan['paper_position']} |",
        f"| marker | `{plan['marker']}` |",
        "",
        "## Queue",
        "",
        "| GPU | variant | command args | claim defended |",
        "|---:|---|---|---|",
    ]
    for row in plan["queue"]:
        lines.append(
            f"| {row['gpu']} | `{row['variant']}` | "
            f"`{row['args']}` | {row['claim_defended']} |")
    lines += [
        "",
        "## No-Fabrication Rule",
        "",
        plan["no_fabrication_rule"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--closure-json", default=str(CLOSURE_JSON))
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument("--out-md", default=str(OUT_MD))
    parser.add_argument("--marker", default=str(MARKER))
    parser.add_argument("--action-only", action="store_true")
    args = parser.parse_args()

    plan = plan_p2_queue(read_json(Path(args.closure_json)), Path(args.marker))
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    write_markdown(Path(args.out_md), plan)

    if args.action_only:
        print(plan["action"])
    else:
        print(json.dumps(plan, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
