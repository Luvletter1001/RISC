#!/usr/bin/env python
"""Summarize P3A AP-constrained Gaussian support projection results.

This reader is non-experimental. It only consumes existing projection summary,
offline eval, and deployment-risk JSON artifacts generated for P3A.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from apply_p3a_ap_constrained_support_projection import assess_p3a_gate  # noqa: E402


EVAL_ROOT = Path("work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619")
RISK_ROOT = Path("work_dirs/gs3c_sise_problem_reframing_20260619")
OUT_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "p3a_ap_projection_summary.json")
OUT_MD = Path(
    "resultmd/exp_p4_scale_semantic_validation/"
    "fres_20260621_p3a_ap_constrained_support_projection.md")

ROWS = [
    {
        "method": "no-G3 e2",
        "kind": "control",
        "eval_dir": EVAL_ROOT / "eval_nog3_ctrl_e2",
        "risk_json": (
            RISK_ROOT
            / "hrrsd_bass_gsf_p3a_ap_projection_nog3_e2_z2_s01"
            / "deployment_risk_summary.json"),
        "risk_variant": "nog3_control",
        "gate": "strict AP control",
    },
    {
        "method": "P3A z4 s0.1",
        "kind": "p3a",
        "eval_dir": EVAL_ROOT / "eval_bass_gsf_p3a_ap_projection_nog3_e2_z4_s01",
        "projection_json": (
            EVAL_ROOT
            / "eval_bass_gsf_p3a_ap_projection_nog3_e2_z4_s01"
            / "p3a_projection_summary.json"),
        "risk_json": (
            RISK_ROOT
            / "hrrsd_bass_gsf_p3a_ap_projection_nog3_e2_z4_s01"
            / "deployment_risk_summary.json"),
        "risk_variant": "p3a_z4_s01",
    },
    {
        "method": "P3A z3 s0.1",
        "kind": "p3a",
        "eval_dir": EVAL_ROOT / "eval_bass_gsf_p3a_ap_projection_nog3_e2_z3_s01",
        "projection_json": (
            EVAL_ROOT
            / "eval_bass_gsf_p3a_ap_projection_nog3_e2_z3_s01"
            / "p3a_projection_summary.json"),
        "risk_json": (
            RISK_ROOT
            / "hrrsd_bass_gsf_p3a_ap_projection_nog3_e2_z3_s01"
            / "deployment_risk_summary.json"),
        "risk_variant": "p3a_z3_s01",
    },
    {
        "method": "P3A z2 s0.1",
        "kind": "p3a_main",
        "eval_dir": EVAL_ROOT / "eval_bass_gsf_p3a_ap_projection_nog3_e2_z2_s01",
        "projection_json": (
            EVAL_ROOT
            / "eval_bass_gsf_p3a_ap_projection_nog3_e2_z2_s01"
            / "p3a_projection_summary.json"),
        "risk_json": (
            RISK_ROOT
            / "hrrsd_bass_gsf_p3a_ap_projection_nog3_e2_z2_s01"
            / "deployment_risk_summary.json"),
        "risk_variant": "p3a_z2_s01",
    },
]


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_eval_metric(eval_dir: Path) -> dict[str, Any]:
    if not eval_dir.exists():
        return {"eval_status": "missing", "eval_json": "", "mAP": None, "AP50": None}
    candidates = [eval_dir / "offline_eval.json"]
    candidates.extend(sorted(eval_dir.glob("*/20*.json")))
    candidates.extend(sorted(eval_dir.glob("*.json")))
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        payload = read_json(path)
        if not isinstance(payload, dict):
            continue
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            payload = {**payload, **metrics}
        map_value = payload.get("dota/mAP", payload.get("mAP"))
        ap50_value = payload.get("dota/AP50", payload.get("AP50"))
        if map_value is not None:
            return {
                "eval_status": "done",
                "eval_json": str(path),
                "mAP": float(map_value),
                "AP50": float(ap50_value) if ap50_value is not None else None,
            }
    return {"eval_status": "missing", "eval_json": "", "mAP": None, "AP50": None}


def read_projection_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"projection_status": "control", "projection_json": ""}
    payload = read_json(path)
    if not isinstance(payload, dict):
        return {"projection_status": "missing", "projection_json": str(path)}
    return {
        "projection_status": "done",
        "projection_json": str(path),
        "z_thr": payload.get("z_thr"),
        "min_score": payload.get("min_score"),
        "protected_correct": payload.get("protected_correct"),
        "changed_correct": payload.get("changed_correct"),
        "low_support_wrong_candidates": payload.get("low_support_wrong_candidates"),
        "scores_changed": payload.get("scores_changed"),
        "forbidden_bbox_gaussian_route": payload.get(
            "forbidden_bbox_gaussian_route"),
    }


def _select_variant(payload: dict[str, Any], variant: str) -> dict[str, Any] | None:
    for row in payload.get("summaries", []) or []:
        if isinstance(row, dict) and row.get("variant") == variant:
            return row
    rows = payload.get("summaries", []) or []
    if rows and isinstance(rows[-1], dict):
        return rows[-1]
    return None


def _select_rank_tail(payload: dict[str, Any], variant: str) -> dict[str, Any]:
    for row in payload.get("rank_tail_summary", []) or []:
        if isinstance(row, dict) and row.get("method_variant") == variant:
            return row
    rows = payload.get("rank_tail_summary", []) or []
    if rows and isinstance(rows[-1], dict):
        return rows[-1]
    return {}


def read_risk_row(summary_path: Path, variant: str) -> dict[str, Any]:
    payload = read_json(summary_path)
    if not isinstance(payload, dict):
        return {"risk_status": "missing", "risk_json": str(summary_path)}
    selected = _select_variant(payload, variant)
    if selected is None:
        return {"risk_status": "missing", "risk_json": str(summary_path)}
    rank_tail = _select_rank_tail(payload, variant)
    return {
        "risk_status": "done",
        "risk_json": str(summary_path),
        "localized_correct": selected.get("localized_correct"),
        "localized_wrong": selected.get("localized_wrong"),
        "sise_logz": selected.get("sise_logz_wrong_excl_sibling"),
        "sise_p0199": selected.get("sise_p0199_wrong_excl_sibling"),
        "risk_weighted_logz": selected.get("risk_weighted_logz_false_alarm"),
        "risk_weighted_p0199": selected.get("risk_weighted_p0199_false_alarm"),
        "top5000_precision": selected.get("topk_precision_top5000"),
        "top5000_sise_logz": selected.get("sise_logz_topk_top5000"),
        "top5000_sise_p0199": selected.get("sise_p0199_topk_top5000"),
        "top5000_risk_weighted_logz": selected.get(
            "risk_weighted_logz_false_alarm_top5000"),
        "top5000_risk_weighted_p0199": selected.get(
            "risk_weighted_p0199_false_alarm_top5000"),
        "ap_contributing_fp_removed": rank_tail.get("ap_contributing_fp_removed"),
        "tp_rank_harm": rank_tail.get("tp_rank_harm"),
        "sise_fp_rank_benefit": rank_tail.get("sise_fp_rank_benefit"),
        "rank_disruptive_delta": rank_tail.get("rank_disruptive_delta"),
        "tail_safety_status": rank_tail.get("tail_safety_status"),
    }


def assess_gate(row: dict[str, Any], control: dict[str, Any]) -> str:
    if row.get("kind") == "control":
        return row.get("gate", "control")
    return assess_p3a_gate(row, control)


def build_rows() -> list[dict[str, Any]]:
    rows = []
    for spec in ROWS:
        row = {
            "method": spec["method"],
            "kind": spec["kind"],
            **read_eval_metric(Path(spec["eval_dir"])),
            **read_projection_summary(spec.get("projection_json")),
            **read_risk_row(Path(spec["risk_json"]), str(spec["risk_variant"])),
            "gate": spec.get("gate", "pending"),
        }
        rows.append(row)
    control = rows[0]
    for row in rows:
        row["gate"] = assess_gate(row, control)
    return rows


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "TBD"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def md_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        "method", "mAP", "top5000_precision", "top5000_sise_logz",
        "top5000_risk_weighted_logz", "scores_changed", "changed_correct",
        "ap_contributing_fp_removed", "tp_rank_harm", "gate",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [
            row.get("method", ""),
            fmt(row.get("mAP"), 6),
            fmt(row.get("top5000_precision"), 4),
            fmt(row.get("top5000_sise_logz"), 0),
            fmt(row.get("top5000_risk_weighted_logz"), 4),
            fmt(row.get("scores_changed"), 0),
            fmt(row.get("changed_correct"), 0),
            fmt(row.get("ap_contributing_fp_removed"), 0),
            fmt(row.get("tp_rank_harm"), 0),
            row.get("gate", ""),
        ]
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    main = next((row for row in rows if row.get("kind") == "p3a_main"), {})
    lines = [
        "# P3A AP-Constrained Gaussian Support Projection - 2026-06-21",
        "",
        "Status: generated from existing projection/eval/risk JSON. P3A is an "
        "oracle validation of the ranking principle: validation GT protects AP-"
        "critical correct positives, and Gaussian semantic support only lowers "
        "scores for localized wrong-class detections. It does not use bbox-IoU, "
        "GWD, KLD, or NWD as a Gaussian route.",
        "",
        "## Main Result",
        "",
        md_table(rows),
        "",
        "## Selected Row",
        "",
        f"- main variant: `{main.get('method', 'missing')}`",
        f"- gate: `{main.get('gate', 'missing')}`",
        f"- AP: `{fmt(main.get('mAP'), 6)}`",
        f"- top5000 precision: `{fmt(main.get('top5000_precision'), 4)}`",
        f"- top5000 log-z SISE: `{fmt(main.get('top5000_sise_logz'), 0)}`",
        f"- top5000 risk-weighted log-z false alarm: "
        f"`{fmt(main.get('top5000_risk_weighted_logz'), 4)}`",
        f"- protected correct positives: `{fmt(main.get('protected_correct'), 0)}`",
        f"- changed correct positives: `{fmt(main.get('changed_correct'), 0)}`",
        f"- TP rank harm: `{fmt(main.get('tp_rank_harm'), 0)}`",
        "",
        "## Gate Definition",
        "",
        "Strict pass for P3A means: mAP and top5000 precision are not lower than "
        "the no-G3 e2 control; no correct positive is changed; and at least one "
        "top-k semantic-risk measure improves.",
        "",
        "## Artifact Trace",
        "",
        "| method | eval_json | projection_json | risk_json |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('method')} | `{row.get('eval_json', '')}` | "
            f"`{row.get('projection_json', '')}` | "
            f"`{row.get('risk_json', '')}` |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument("--out-md", default=str(OUT_MD))
    args = parser.parse_args()
    rows = build_rows()
    strict_rows = [row for row in rows if row.get("gate") == "strict pass"]
    main_row = next((row for row in rows if row.get("kind") == "p3a_main"), {})
    payload = {
        "status": "done" if rows else "missing",
        "strict_pass": bool(strict_rows),
        "strict_pass_methods": [row.get("method") for row in strict_rows],
        "main_method": main_row.get("method"),
        "main_gate": main_row.get("gate"),
        "rows": rows,
        "result_md": args.out_md,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    write_markdown(Path(args.out_md), rows)
    print(json.dumps({
        "status": payload["status"],
        "main_gate": payload["main_gate"],
        "strict_pass_methods": payload["strict_pass_methods"],
        "out_json": str(out_json),
        "out_md": args.out_md,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
