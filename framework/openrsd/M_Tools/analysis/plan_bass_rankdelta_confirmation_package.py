#!/usr/bin/env python
"""Plan the confirmation package after a strict RankDelta pass.

The confirmation route is the mirror of the P2 rescue route: it is enabled only
when the closure planner has found a strict pass.  It does not launch training
or edit metrics; it emits an auditable queue of repeat and ablation checks that
must pass before the paper can promote BASS-GSF to the main method.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CLOSURE_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "bass_rankdelta_closure_plan.json")
SUMMARY_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "bass_rankdelta_summary.json")
OUT_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "bass_rankdelta_confirmation_plan.json")
OUT_MD = Path(
    "resultmd/exp_p4_scale_semantic_validation/"
    "fstatus_20260620_bass_rankdelta_confirmation_plan.md")
MARKER = Path(
    "work_dirs/train_queue_logs/"
    "bass_gsf_rankdelta_confirmation_launched_20260620.flag")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _rows_by_method(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = summary.get("rows", [])
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("method")): row
        for row in rows
        if isinstance(row, dict) and row.get("method")
    }


def _selected_strict_methods(closure: dict[str, Any],
                             summary: dict[str, Any]) -> list[str]:
    methods = [
        str(method) for method in closure.get("strict_pass_methods", [])
        if method
    ]
    if methods:
        return methods
    rows = _rows_by_method(summary)
    return [
        method for method, row in rows.items()
        if row.get("gate") == "strict pass"
    ]


def _queue(strict_methods: list[str]) -> list[dict[str, str]]:
    source = ", ".join(strict_methods) if strict_methods else "strict-pass row"
    return [
        {
            "priority": "C0",
            "experiment": "repeat strict-pass configuration",
            "claim_defended": "strict pass is not a single-run artifact",
            "dependency": source,
            "paper_use": "main table only if repeat is stable",
        },
        {
            "priority": "C1",
            "experiment": "shuffled class-scale priors",
            "claim_defended": "benefit comes from semantic-scale support",
            "dependency": source,
            "paper_use": "main ablation",
        },
        {
            "priority": "C2",
            "experiment": "global Gaussian instead of class-conditioned prior",
            "claim_defended": "class conditioning is necessary",
            "dependency": source,
            "paper_use": "main ablation",
        },
        {
            "priority": "C3",
            "experiment": "no delta / no score gate / no GT keep",
            "claim_defended": "rank-aware support mechanism is necessary",
            "dependency": source,
            "paper_use": "main or appendix ablation",
        },
        {
            "priority": "C4",
            "experiment": "second dataset or detector-family transfer",
            "claim_defended": "method benefit generalizes beyond HRRSD",
            "dependency": source,
            "paper_use": "required for 9.5 applicability breadth",
        },
    ]


def plan_confirmation(closure: dict[str, Any],
                      summary: dict[str, Any],
                      marker: Path = MARKER) -> dict[str, Any]:
    closure_action = str(closure.get("action", "missing"))
    strict_methods = _selected_strict_methods(closure, summary)
    if marker.exists():
        action = "WAIT_CONFIRMATION_EXISTING"
        reason = (
            "A confirmation launch marker exists; do not launch duplicate "
            "repeat or ablation jobs.")
    elif closure_action == "RUN_CONFIRMATION_PACKAGE" and strict_methods:
        action = "RUN_CONFIRMATION"
        reason = (
            "At least one strict RankDelta row exists; run repeat, ablation, "
            "and transfer confirmation before promoting BASS-GSF.")
    elif closure_action == "RUN_P2_RANK_ASSIGNMENT":
        action = "STOP_P2_ROUTE"
        reason = (
            "No strict pass is available; P2 rescue route is the active path.")
    elif closure_action == "STOP_P2_DONE_NO_STRICT":
        action = "STOP_P2_DONE_NO_STRICT"
        reason = (
            "P2 is complete without a strict pass; no confirmation package "
            "exists. Redesign the method as a Gaussian AP-constrained support "
            "projection before confirmation planning.")
    else:
        action = "WAIT_UPSTREAM"
        reason = (
            "Confirmation waits for closure action RUN_CONFIRMATION_PACKAGE; "
            f"current closure action is {closure_action}.")
    return {
        "action": action,
        "reason": reason,
        "closure_action": closure_action,
        "strict_pass_methods": strict_methods,
        "marker": str(marker),
        "paper_position": closure.get("paper_position", ""),
        "queue": _queue(strict_methods),
        "no_fabrication_rule": (
            "This planner emits confirmation obligations only. Results may "
            "enter the paper only after eval JSON and deployment-risk summary "
            "JSON exist for each confirmation run."),
    }


def write_markdown(path: Path, plan: dict[str, Any]) -> None:
    lines = [
        "# BASS-GSF Confirmation Package Plan - 2026-06-20",
        "",
        "This file is generated from the RankDelta closure plan and summary.",
        "It does not launch experiments or infer metrics.",
        "",
        "| field | value |",
        "|---|---|",
        f"| action | `{plan['action']}` |",
        f"| reason | {plan['reason']} |",
        f"| closure_action | `{plan['closure_action']}` |",
        f"| strict_pass_methods | `{', '.join(plan['strict_pass_methods'])}` |",
        f"| paper_position | {plan['paper_position']} |",
        f"| marker | `{plan['marker']}` |",
        "",
        "## Queue",
        "",
        "| priority | experiment | claim defended | dependency | paper use |",
        "|---|---|---|---|---|",
    ]
    for row in plan["queue"]:
        lines.append(
            f"| {row['priority']} | {row['experiment']} | "
            f"{row['claim_defended']} | {row['dependency']} | "
            f"{row['paper_use']} |")
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
    parser.add_argument("--summary-json", default=str(SUMMARY_JSON))
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument("--out-md", default=str(OUT_MD))
    parser.add_argument("--marker", default=str(MARKER))
    parser.add_argument("--action-only", action="store_true")
    args = parser.parse_args()

    plan = plan_confirmation(
        read_json(Path(args.closure_json)),
        read_json(Path(args.summary_json)),
        Path(args.marker))
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
