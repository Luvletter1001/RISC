#!/usr/bin/env python
"""Build the current -> 8/10 ICLR gate upgrade plan.

This script is deliberately non-experimental.  It reads the strict 9.5 gate
audit and writes the next execution target without changing any pass/fail
status.  A gate is promoted only when its upstream evidence JSON changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_GATE_JSON = Path(
    "work_dirs/iclr95_evidence_gate_20260620/iclr95_evidence_gate.json")
DEFAULT_OUT_JSON = Path(
    "work_dirs/iclr95_evidence_gate_20260620/"
    "iclr95_8of10_upgrade_plan.json")
DEFAULT_OUT_MD = Path(
    "resultmd/exp_p4_scale_semantic_validation/"
    "fplan_20260621_iclr95_gate_4to8_upgrade.md")

TARGET_SEQUENCE = [
    "Problem anatomy depth",
    "Literature breadth",
    "Practicality",
    "Evidence rigor",
    "Vision / conceptual height",
    "Current method novelty",
    "Method effectiveness",
    "Applicability breadth",
]

DEFERRED_AFTER_8 = [
    "ICLR paper readiness",
    "Overall AC score",
]

TARGET_ACTIONS = {
    "Problem anatomy depth": (
        "保持 predictive support-law + formal non-causal downgrade，不恢复因果声称。"),
    "Literature breadth": (
        "保持 8/8 literature clusters、closest-competitor matrix 和 baseline obligations。"),
    "Practicality": (
        "保留 no-dump latency、review-queue utility、source-disjoint priors 和 fallback。"),
    "Evidence rigor": (
        "继续把 E-P2/G3-v2/BASS-delta/P0/P1/P2 负结果作为边界证据公开。"),
    "Vision / conceptual height": (
        "把 Gaussian support field 写成 detector decision geometry，而不是 post-hoc score filter。"),
    "Current method novelty": (
        "P3A 必须让 Gaussian semantic support 进入 AP-constrained pre-topK/ranking/assignment projection。"),
    "Method effectiveness": (
        "P3A 必须同时 preserve/improve mAP 与 top-K precision，并降低 low-support semantic risk。"),
    "Applicability breadth": (
        "P3D 已用 DIOR-R 关闭第二数据集 AP-safe transfer；下一步转向 detector-family transfer。"),
    "ICLR paper readiness": (
        "8/10 后再补 final BASS figures、compiled PDF 和 submission checklist。"),
    "Overall AC score": (
        "只有 readiness 与整体叙事同时闭合后才更新，不能在 8/10 阶段提前抬分。"),
}

TARGET_ARTIFACTS = {
    "Vision / conceptual height": (
        "paper/semantic_scale_support_iclr/main.tex; "
        "paper/drafts/semantic_scale_support_mismatch_iclr_draft_v0_4_20260620.md"),
    "Current method novelty": (
        "work_dirs/semantic_scale_six_experiments_20260620/"
        "p3a_ap_projection_summary.json"),
    "Method effectiveness": (
        "work_dirs/semantic_scale_six_experiments_20260620/"
        "p3a_ap_projection_summary.json"),
    "Applicability breadth": (
        "resultmd/exp_p4_scale_semantic_validation/"
        "fres_20260621_p3d_dior_transfer_gate.md"),
}


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def _gate_by_dimension(gate_payload: dict) -> dict[str, dict]:
    rows = gate_payload.get("gates", [])
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("dimension")): row
        for row in rows
        if isinstance(row, dict) and row.get("dimension")
    }


def build_upgrade_plan(gate_payload: dict) -> dict:
    gates = _gate_by_dimension(gate_payload)
    current_passed = [
        name for name, row in gates.items() if row.get("gate_pass") is True
    ]
    target_rows = []
    for name in TARGET_SEQUENCE:
        row = gates.get(name, {})
        target_rows.append({
            "dimension": name,
            "current_gate_pass": bool(row.get("gate_pass", False)),
            "current_score": row.get("current_score", "missing"),
            "target_score": row.get("target_score", "9.50"),
            "role_in_8of10": (
                "already_counted" if row.get("gate_pass") is True
                else "must_close_next"),
            "pass_condition": TARGET_ACTIONS[name],
            "source_blocker": row.get("blocker", ""),
            "source_next_action": row.get("next_action", ""),
            "target_artifact": TARGET_ARTIFACTS.get(
                name, row.get("artifact", "")),
        })

    deferred_rows = []
    for name in DEFERRED_AFTER_8:
        row = gates.get(name, {})
        deferred_rows.append({
            "dimension": name,
            "current_gate_pass": bool(row.get("gate_pass", False)),
            "current_score": row.get("current_score", "missing"),
            "reason_deferred": TARGET_ACTIONS[name],
            "source_blocker": row.get("blocker", ""),
        })

    current_count = int(gate_payload.get("num_gates_passed",
                                         len(current_passed)))
    total = int(gate_payload.get("num_gates_total", len(gates) or 10))
    target_count = 8
    return {
        "current_gate": f"{current_count}/{total}",
        "target_gate": f"{target_count}/{total}",
        "current_num_passed": current_count,
        "target_num_passed": target_count,
        "additional_gates_needed": max(0, target_count - current_count),
        "target_sequence": TARGET_SEQUENCE,
        "deferred_after_8": DEFERRED_AFTER_8,
        "target_rows": target_rows,
        "deferred_rows": deferred_rows,
        "promotion_policy": (
            "Do not edit strict gate_pass fields by hand.  The 8/10 target is "
            "achieved only when build_iclr95_evidence_gate.py reads positive "
            "P3 evidence and naturally reports eight passed gates."),
        "next_experiment_route": {
            "P3A": (
                "Gaussian AP-constrained support projection; class/domain "
                "semantic-scale support affects pre-topK/ranking only inside "
                "AP-safe constraints."),
            "P3B": (
                "pairwise AP-critical cone ablation; proves rank protection is "
                "the missing ingredient after P2 failure."),
            "P3C": (
                "shuffled/global/score-only/no-posterior/no-virtual-outlier "
                "controls; proves the gain is semantic support, not generic "
                "calibration."),
            "P3D": (
                "DIOR-R second-dataset transfer; closes applicability "
                "breadth for 8/10 while leaving detector-family transfer as "
                "the next 9.5 step."),
        },
        "forbidden_route": (
            "Do not use Gaussian as bbox IoU, box similarity, GWD, KLD, or NWD. "
            "Those methods remain related-work boundaries only."),
    }


def _md_table(rows: list[dict], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [str(row.get(col, "")).replace("\n", " ") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown(path: Path, plan: dict) -> None:
    lines = [
        f"# ICLR Gate Upgrade Plan: {plan['current_gate']} -> "
        f"{plan['target_gate']} - 2026-06-21",
        "",
        "本文件是目标更新，不是结果抬分。当前严格机器门仍以真实证据为准；"
        "8/10 只在 P3 证据进入 `build_iclr95_evidence_gate.py` 后自动成立。",
        "",
        "## Gate Target",
        "",
        f"- current_gate: `{plan['current_gate']}`",
        f"- target_gate: `{plan['target_gate']}`",
        f"- additional_gates_needed: `{plan['additional_gates_needed']}`",
        f"- promotion_policy: {plan['promotion_policy']}",
        f"- forbidden_route: {plan['forbidden_route']}",
        "",
        "## 8/10 Target Rows",
        "",
        _md_table(plan["target_rows"], [
            "dimension", "current_gate_pass", "current_score",
            "role_in_8of10", "pass_condition", "target_artifact"
        ]),
        "",
        "## Deferred After 8/10",
        "",
        _md_table(plan["deferred_rows"], [
            "dimension", "current_gate_pass", "current_score",
            "reason_deferred"
        ]),
        "",
        "## P3 Route",
        "",
    ]
    for key, value in plan["next_experiment_route"].items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-json", default=str(DEFAULT_GATE_JSON))
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    parser.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    args = parser.parse_args()

    plan = build_upgrade_plan(read_json(Path(args.gate_json)))
    write_json(Path(args.out_json), plan)
    write_markdown(Path(args.out_md), plan)
    print(json.dumps({
        "current_gate": plan["current_gate"],
        "target_gate": plan["target_gate"],
        "additional_gates_needed": plan["additional_gates_needed"],
        "out_json": args.out_json,
        "out_md": args.out_md,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
