#!/usr/bin/env python
"""Plan the next BASS-GSF closure step from real RankDelta summaries.

This script is deliberately read-only with respect to experiments: it does not
launch training, infer metrics, or mark missing rows as successes.  Its job is
to turn the current P0/P1 RankDelta state into an auditable next-step plan for
the paper and the GPU6/7 queue.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SUMMARY_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "bass_rankdelta_summary.json")
FOLLOWUP_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "bass_rankdelta_followup_decision.json")
OUT_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "bass_rankdelta_closure_plan.json")
OUT_MD = Path(
    "resultmd/exp_p4_scale_semantic_validation/"
    "fplan_20260620_bass_gsf_p2_closure_plan.md")

P0_METHODS = {
    "RankDelta min-score 0.30",
    "RankDelta min-score 0.50",
}
P1_METHODS = {
    "P1 gentle RankDelta",
    "P1 protected RankDelta",
}
P2_METHODS = {
    "P2 assign-rank BASS-GSF",
    "P2 posterior-rank BASS-GSF",
}

STRICT_GATE = {
    "mAP": ">= 0.862878 (no-G3 e2)",
    "top5000_precision": ">= 0.7168",
    "top5000_logz_sise": "<= 48",
    "localized_wrong": "< 6648",
    "risk_weighted_logz": "<= 14.2824",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def is_complete(row: dict[str, Any]) -> bool:
    return row.get("mAP") is not None and row.get("gate") != "waiting/missing"


def rows_by_method(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("method")): row
        for row in summary.get("rows", [])
        if isinstance(row, dict) and row.get("method")
    }


def _selected_rows(by_method: dict[str, dict[str, Any]],
                   methods: set[str]) -> list[dict[str, Any]]:
    return [by_method.get(name, {"method": name, "gate": "missing"})
            for name in sorted(methods)]


def _rankdelta_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row for row in summary.get("rows", [])
        if isinstance(row, dict)
        and str(row.get("kind", "")).startswith("rankdelta")
    ]


def _queue_for_confirmation(strict_methods: list[str]) -> list[dict[str, str]]:
    methods = ", ".join(strict_methods)
    return [
        {
            "priority": "C0",
            "experiment": "same-seed or adjacent-seed repeat",
            "claim_defended": "strict pass is not a single-run artifact",
            "dependency": f"strict pass from {methods}",
            "paper_use": "main table if repeat is stable",
        },
        {
            "priority": "C1",
            "experiment": "shuffled class-scale priors",
            "claim_defended": "benefit comes from semantic-scale support",
            "dependency": "strict-pass config",
            "paper_use": "main ablation",
        },
        {
            "priority": "C2",
            "experiment": "global Gaussian instead of class-conditioned prior",
            "claim_defended": "class conditioning is necessary",
            "dependency": "strict-pass config",
            "paper_use": "main ablation",
        },
        {
            "priority": "C3",
            "experiment": "no delta / no score gate / no GT keep",
            "claim_defended": "AP-risk mechanism is not accidental",
            "dependency": "strict-pass config",
            "paper_use": "main or appendix ablation",
        },
        {
            "priority": "C4",
            "experiment": "second dataset or detector-family transfer",
            "claim_defended": "method benefit generalizes beyond HRRSD",
            "dependency": "strict-pass config and available dataset roots",
            "paper_use": "required for 9.5 method breadth",
        },
    ]


def _queue_for_p2() -> list[dict[str, str]]:
    return [
        {
            "priority": "P2A",
            "experiment": "support-aware assignment or classification weight",
            "claim_defended": "Gaussian support changes assignment/training, not only suppression",
            "dependency": "P0/P1 no strict pass",
            "paper_use": "main only if AP-risk Pareto passes",
        },
        {
            "priority": "P2B",
            "experiment": "pairwise rank-preservation loss on high-quality positives",
            "claim_defended": "SISE reduction preserves top-K AP-sensitive ordering",
            "dependency": "P2A or current RankDelta head",
            "paper_use": "main ablation if it rescues top5000 precision",
        },
        {
            "priority": "P2C",
            "experiment": "feature-conditioned posterior density with virtual semantic-scale outliers",
            "claim_defended": "BASS-GSF is a Bayesian support-field method",
            "dependency": "density head and class/domain priors",
            "paper_use": "main method if matched controls pass",
        },
        {
            "priority": "P2D",
            "experiment": "second dataset smoke transfer after HRRSD pass",
            "claim_defended": "method is not HRRSD-only",
            "dependency": "one HRRSD P2 variant reaches strict or near-strict pass",
            "paper_use": "breadth gate",
        },
    ]


def _queue_for_gaussian_projection() -> list[dict[str, str]]:
    return [
        {
            "priority": "P3A",
            "experiment": "Gaussian AP-constrained support projection",
            "claim_defended": (
                "support likelihood-ratio changes the detector decision "
                "surface only inside AP-safe ordering constraints"),
            "dependency": "P0/P1/P2 complete without strict pass",
            "paper_use": "next main-method candidate, not a continuation of delta tuning",
        },
        {
            "priority": "P3B",
            "experiment": "pairwise AP-critical cone ablation",
            "claim_defended": (
                "positive-vs-hard-negative ordering constraints are necessary "
                "for preserving top-K precision"),
            "dependency": "P3A implementation",
            "paper_use": "main ablation if P3A reaches strict or near-strict pass",
        },
        {
            "priority": "P3C",
            "experiment": "shuffled/global/score-only Gaussian support controls",
            "claim_defended": (
                "benefit comes from class-conditioned semantic-scale support "
                "rather than generic calibration"),
            "dependency": "P3A positive HRRSD result",
            "paper_use": "required ablation before any 9.5 method claim",
        },
    ]


def _status_table(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    table = []
    for row in rows:
        table.append({
            "method": str(row.get("method", "")),
            "kind": str(row.get("kind", "")),
            "gate": str(row.get("gate", "missing")),
            "mAP": "TBD" if row.get("mAP") is None else f"{float(row['mAP']):.6f}",
            "top5000_precision": (
                "TBD" if row.get("top5000_precision") is None
                else f"{float(row['top5000_precision']):.4f}"),
        })
    return table


def plan_closure(summary: dict[str, Any],
                 followup: dict[str, Any] | None = None) -> dict[str, Any]:
    followup = followup or {}
    by_method = rows_by_method(summary)
    p0_rows = _selected_rows(by_method, P0_METHODS)
    p1_rows = _selected_rows(by_method, P1_METHODS)
    p2_rows = _selected_rows(by_method, P2_METHODS)
    rank_rows = _rankdelta_rows(summary)
    p0_done = bool(p0_rows) and all(is_complete(row) for row in p0_rows)
    p1_done = bool(p1_rows) and all(is_complete(row) for row in p1_rows)
    p2_done = bool(p2_rows) and all(is_complete(row) for row in p2_rows)
    strict_methods = [
        str(row.get("method")) for row in rank_rows
        if row.get("gate") == "strict pass"
    ]
    pilot_methods = [
        str(row.get("method")) for row in rank_rows
        if row.get("gate") == "pilot pass"
    ]

    if strict_methods:
        action = "RUN_CONFIRMATION_PACKAGE"
        reason = (
            "At least one RankDelta row reached strict pass; run repeat, "
            "ablation, and transfer checks before promoting BASS-GSF.")
        queue = _queue_for_confirmation(strict_methods)
        paper_position = "candidate main method after confirmation"
    elif not p0_done:
        action = "WAIT_P0"
        reason = "P0 RankDelta rows are still missing or incomplete."
        queue = [
            {
                "priority": "P0",
                "experiment": "GPU6/GPU7 RankDelta ms0.30/ms0.50",
                "claim_defended": "score-gated support pressure preserves ranking",
                "dependency": "GPU6/GPU7 idle under waiter thresholds",
                "paper_use": "TBD until eval JSON and risk JSON exist",
            }
        ]
        paper_position = "no new method result may be claimed"
    elif not p1_done:
        action = (
            "WAIT_P1_EXISTING"
            if followup.get("action") == "WAIT_P1_EXISTING" else "RUN_P1")
        reason = (
            "P0 is complete without strict pass; use the existing P1 "
            "gentle/protected follow-up path.")
        queue = [
            {
                "priority": "P1",
                "experiment": "gentle/protected RankDelta on GPU6/GPU7",
                "claim_defended": "weaker or GT-protected support pressure preserves AP",
                "dependency": "P0 summary complete and no strict pass",
                "paper_use": "appendix unless strict pass is reached",
            }
        ]
        paper_position = "pilot-only unless strict pass appears"
    elif p2_done:
        action = "STOP_P2_DONE_NO_STRICT"
        reason = (
            "P0, P1, and P2 are complete without strict pass; current "
            "RankDelta/P2 approximations are boundary evidence. Stop this "
            "queue and redesign BASS-GSF as an AP-constrained Gaussian "
            "support-field projection rather than stronger delta tuning.")
        queue = _queue_for_gaussian_projection()
        paper_position = (
            "P2 completed as boundary/failure; current RankDelta/P2 remains "
            "appendix evidence, and the next 9.5 route is Gaussian "
            "AP-constrained support projection")
    else:
        action = "RUN_P2_RANK_ASSIGNMENT"
        reason = (
            "P0 and P1 are complete without strict pass; pure delta tuning "
            "should stop and explicit rank/assignment BASS-GSF should start.")
        queue = _queue_for_p2()
        paper_position = (
            "current RankDelta remains appendix/boundary; P2 is required "
            "for a 9.5 trainable-method claim")

    return {
        "action": action,
        "reason": reason,
        "summary_status": summary.get("status", "missing"),
        "followup_action": followup.get("action", ""),
        "p0_done": p0_done,
        "p1_done": p1_done,
        "p2_done": p2_done,
        "strict_pass_methods": strict_methods,
        "pilot_pass_methods": pilot_methods,
        "paper_position": paper_position,
        "strict_gate": STRICT_GATE,
        "rankdelta_status": _status_table(rank_rows),
        "execution_priority": queue,
        "no_fabrication_rule": (
            "Only eval JSON plus deployment-risk summary JSON may fill result "
            "tables; checkpoints or logs alone keep rows as waiting/TBD."),
    }


def write_markdown(path: Path, plan: dict[str, Any]) -> None:
    lines = [
        "# BASS-GSF P2 Closure Plan - 2026-06-20",
        "",
        "This file is generated from real RankDelta summary JSON. It plans the",
        "next experiment step but does not launch training or invent results.",
        "",
        "| field | value |",
        "|---|---|",
        f"| action | `{plan['action']}` |",
        f"| reason | {plan['reason']} |",
        f"| summary_status | `{plan['summary_status']}` |",
        f"| followup_action | `{plan['followup_action']}` |",
        f"| p0_done | `{plan['p0_done']}` |",
        f"| p1_done | `{plan['p1_done']}` |",
        f"| paper_position | {plan['paper_position']} |",
        "",
        "## RankDelta Status",
        "",
        "| method | kind | gate | mAP | top5000_precision |",
        "|---|---|---|---:|---:|",
    ]
    for row in plan["rankdelta_status"]:
        lines.append(
            f"| {row['method']} | {row['kind']} | `{row['gate']}` | "
            f"{row['mAP']} | {row['top5000_precision']} |")
    lines += [
        "",
        "## Strict Gate",
        "",
        "| metric | threshold |",
        "|---|---|",
    ]
    for metric, threshold in plan["strict_gate"].items():
        lines.append(f"| {metric} | `{threshold}` |")
    lines += [
        "",
        "## Execution Priority",
        "",
        "| priority | experiment | claim defended | dependency | paper use |",
        "|---|---|---|---|---|",
    ]
    for row in plan["execution_priority"]:
        lines.append(
            f"| {row['priority']} | {row['experiment']} | "
            f"{row['claim_defended']} | {row['dependency']} | "
            f"{row['paper_use']} |")
    lines += [
        "",
        "## No-Fabrication Rule",
        "",
        plan["no_fabrication_rule"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", default=str(SUMMARY_JSON))
    parser.add_argument("--followup-json", default=str(FOLLOWUP_JSON))
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument("--out-md", default=str(OUT_MD))
    args = parser.parse_args()

    plan = plan_closure(
        read_json(Path(args.summary_json)),
        read_json(Path(args.followup_json)))
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    write_markdown(Path(args.out_md), plan)
    print(json.dumps({
        "action": plan["action"],
        "summary_status": plan["summary_status"],
        "out_json": str(out_json),
        "out_md": str(args.out_md),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
