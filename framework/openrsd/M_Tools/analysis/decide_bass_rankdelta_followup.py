#!/usr/bin/env python
"""Decide whether GPU6/7 BASS-GSF RankDelta needs P1 follow-up runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SUMMARY_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "bass_rankdelta_summary.json")
OUT_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "bass_rankdelta_followup_decision.json")
OUT_MD = Path(
    "resultmd/exp_p4_scale_semantic_validation/"
    "fstatus_20260620_bass_gsf_rankdelta_followup_decision.md")
MARKER = Path(
    "work_dirs/train_queue_logs/"
    "bass_gsf_rankdelta_followup_p1_launched_20260620.flag")

P0_METHODS = {
    "RankDelta min-score 0.30",
    "RankDelta min-score 0.50",
}
P1_METHODS = {
    "P1 gentle RankDelta",
    "P1 protected RankDelta",
}


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def is_done(row: dict) -> bool:
    return row.get("mAP") is not None and row.get("gate") != "waiting/missing"


def decide(summary: dict, marker: Path) -> dict:
    rows = summary.get("rows", [])
    by_method = {row.get("method"): row for row in rows}
    p0_rows = [by_method.get(name, {}) for name in sorted(P0_METHODS)]
    p1_rows = [by_method.get(name, {}) for name in sorted(P1_METHODS)]
    p0_done = bool(p0_rows) and all(is_done(row) for row in p0_rows)
    p1_done = bool(p1_rows) and all(is_done(row) for row in p1_rows)
    all_rank_rows = [row for row in rows
                     if str(row.get("kind", "")).startswith("rankdelta")]
    strict_passes = [
        row for row in all_rank_rows if row.get("gate") == "strict pass"
    ]
    pilot_passes = [
        row for row in all_rank_rows if row.get("gate") == "pilot pass"
    ]

    if strict_passes:
        action = "STOP_STRICT_PASS"
        reason = "At least one RankDelta row already passes the strict gate."
    elif not p0_done:
        action = "WAIT_P0"
        reason = "P0 RankDelta rows are still missing or incomplete."
    elif p1_done:
        action = "STOP_P1_DONE_NO_STRICT"
        reason = "P1 rows are complete and no strict pass is available."
    elif marker.exists():
        action = "WAIT_P1_EXISTING"
        reason = "A P1 launch marker exists, so do not launch duplicate jobs."
    else:
        action = "RUN_P1"
        reason = (
            "P0 is complete but no strict pass exists; run gentle/protected "
            "P1 variants on GPU6/GPU7.")

    return {
        "action": action,
        "reason": reason,
        "summary_status": summary.get("status", "missing"),
        "p0_done": p0_done,
        "p1_done": p1_done,
        "marker": str(marker),
        "strict_pass_methods": [row.get("method") for row in strict_passes],
        "pilot_pass_methods": [row.get("method") for row in pilot_passes],
        "p0_gates": {
            row.get("method", "missing"): row.get("gate", "missing")
            for row in p0_rows
        },
        "p1_gates": {
            row.get("method", "missing"): row.get("gate", "missing")
            for row in p1_rows
        },
    }


def write_markdown(path: Path, decision: dict) -> None:
    lines = [
        "# BASS-GSF RankDelta Follow-up Decision - 2026-06-20",
        "",
        "本文件由 `decide_bass_rankdelta_followup.py` 从真实 summary JSON 生成，",
        "只决定是否需要 GPU6/7 P1 follow-up，不生成任何实验结果。",
        "",
        "| field | value |",
        "|---|---|",
        f"| action | `{decision['action']}` |",
        f"| reason | {decision['reason']} |",
        f"| summary_status | `{decision['summary_status']}` |",
        f"| p0_done | `{decision['p0_done']}` |",
        f"| p1_done | `{decision['p1_done']}` |",
        f"| marker | `{decision['marker']}` |",
        "",
        "## P0 Gates",
        "",
        "| method | gate |",
        "|---|---|",
    ]
    for method, gate in decision["p0_gates"].items():
        lines.append(f"| {method} | `{gate}` |")
    lines += [
        "",
        "## P1 Gates",
        "",
        "| method | gate |",
        "|---|---|",
    ]
    for method, gate in decision["p1_gates"].items():
        lines.append(f"| {method} | `{gate}` |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", default=str(SUMMARY_JSON))
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument("--out-md", default=str(OUT_MD))
    parser.add_argument("--marker", default=str(MARKER))
    parser.add_argument("--action-only", action="store_true")
    args = parser.parse_args()

    decision = decide(read_json(Path(args.summary_json)), Path(args.marker))
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    write_markdown(Path(args.out_md), decision)

    if args.action_only:
        print(decision["action"])
    else:
        print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
