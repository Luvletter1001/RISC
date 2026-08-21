#!/usr/bin/env python
"""Summarize HRRSD BASS-GSF rank-aware delta results.

This script reads existing eval JSON and deployment-risk summaries only.  It
does not launch training or inference, and it writes missing rows explicitly
when GPU jobs have not finished yet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EVAL_ROOT = Path("work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619")
RISK_ROOT = Path("work_dirs/gs3c_sise_problem_reframing_20260619")
OUT_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "bass_rankdelta_summary.json")
OUT_MD = Path(
    "resultmd/exp_p4_scale_semantic_validation/"
    "fres_20260620_bass_gsf_rankdelta_results.md")

CONTROL_RISK = {
    "epoch3 baseline": (
        RISK_ROOT
        / "hrrsd_bass_gsf_delta_e2_dw0p05_t0p25_d0p50_head"
        / "deployment_risk_summary.json",
        "baseline",
    ),
    "density w005 e2": (
        RISK_ROOT
        / "hrrsd_bass_gsf_delta_e2_dw0p05_t0p25_d0p50_head_vs_density_control"
        / "deployment_risk_summary.json",
        "density_control",
    ),
    "no-G3 e2": (
        RISK_ROOT
        / "hrrsd_bass_gsf_delta_e2_dw0p05_t0p25_d0p50_head_vs_nog3_control"
        / "deployment_risk_summary.json",
        "nog3_control",
    ),
    "BASS-GSF-Lite beta0.50": (
        RISK_ROOT
        / "hrrsd_bass_gsf_lite_e2_b0p50_thrm8p0_d0p50"
        / "deployment_risk_summary.json",
        "bass_gsf_lite_e2_b0p50_thrm8p0_d0p50",
    ),
}

ROWS = [
    {
        "method": "epoch3 baseline",
        "kind": "control",
        "eval_dir": EVAL_ROOT / "eval_epoch3_baseline_full",
        "risk": CONTROL_RISK["epoch3 baseline"],
        "gate": "reference",
    },
    {
        "method": "density w005 e2",
        "kind": "control",
        "eval_dir": EVAL_ROOT / "eval_epoch3_plus2_density_w005_full",
        "risk": CONTROL_RISK["density w005 e2"],
        "gate": "density control",
    },
    {
        "method": "no-G3 e2",
        "kind": "control",
        "eval_dir": EVAL_ROOT / "eval_nog3_ctrl_e2",
        "risk": CONTROL_RISK["no-G3 e2"],
        "gate": "strict AP control",
    },
    {
        "method": "BASS-GSF-Lite beta0.50",
        "kind": "pilot",
        "eval_dir": (
            EVAL_ROOT / "eval_bass_gsf_lite_e2_b0p50_thrm8p0_d0p50_full"),
        "risk": CONTROL_RISK["BASS-GSF-Lite beta0.50"],
        "gate": "best inference pilot",
    },
    {
        "method": "RankDelta min-score 0.30",
        "kind": "rankdelta_p0",
        "eval_dir": (
            EVAL_ROOT
            / "eval_bass_gsf_rankdelta_e2_dw0p05_ms0p30_t0p25_d0p50_head_full"),
        "risk": (
            RISK_ROOT
            / "hrrsd_bass_gsf_rankdelta_e2_dw0p05_ms0p30_t0p25_d0p50_head"
            / "deployment_risk_summary.json",
            "bass_gsf_rankdelta_e2_dw0p05_ms0p30_t0p25_d0p50",
        ),
        "gate": "pending",
    },
    {
        "method": "RankDelta min-score 0.50",
        "kind": "rankdelta_p0",
        "eval_dir": (
            EVAL_ROOT
            / "eval_bass_gsf_rankdelta_e2_dw0p05_ms0p50_t0p25_d0p50_head_full"),
        "risk": (
            RISK_ROOT
            / "hrrsd_bass_gsf_rankdelta_e2_dw0p05_ms0p50_t0p25_d0p50_head"
            / "deployment_risk_summary.json",
            "bass_gsf_rankdelta_e2_dw0p05_ms0p50_t0p25_d0p50",
        ),
        "gate": "pending",
    },
    {
        "method": "P1 gentle RankDelta",
        "kind": "rankdelta_p1",
        "eval_dir": (
            EVAL_ROOT
            / "eval_bass_gsf_rankdelta_e2_dw0p01_ms0p05_t0p10_d0p20_keep0p20_head_full"),
        "risk": (
            RISK_ROOT
            / "hrrsd_bass_gsf_rankdelta_e2_dw0p01_ms0p05_t0p10_d0p20_keep0p20_head"
            / "deployment_risk_summary.json",
            "bass_gsf_rankdelta_e2_dw0p01_ms0p05_t0p10_d0p20_keep0p20",
        ),
        "gate": "follow-up pending",
    },
    {
        "method": "P1 protected RankDelta",
        "kind": "rankdelta_p1",
        "eval_dir": (
            EVAL_ROOT
            / "eval_bass_gsf_rankdelta_e2_dw0p02_ms0p10_t0p15_d0p25_keep0p30_head_full"),
        "risk": (
            RISK_ROOT
            / "hrrsd_bass_gsf_rankdelta_e2_dw0p02_ms0p10_t0p15_d0p25_keep0p30_head"
            / "deployment_risk_summary.json",
            "bass_gsf_rankdelta_e2_dw0p02_ms0p10_t0p15_d0p25_keep0p30",
        ),
        "gate": "follow-up pending",
    },
    {
        "method": "P2 assign-rank BASS-GSF",
        "kind": "rankdelta_p2",
        "eval_dir": (
            EVAL_ROOT / "eval_bass_gsf_p2_assign_rank_e2_head_full"),
        "risk": (
            RISK_ROOT
            / "hrrsd_bass_gsf_p2_assign_rank_e2_head"
            / "deployment_risk_summary.json",
            "bass_gsf_p2_assign_rank_e2",
        ),
        "gate": "p2 pending",
    },
    {
        "method": "P2 posterior-rank BASS-GSF",
        "kind": "rankdelta_p2",
        "eval_dir": (
            EVAL_ROOT / "eval_bass_gsf_p2_posterior_rank_e2_head_full"),
        "risk": (
            RISK_ROOT
            / "hrrsd_bass_gsf_p2_posterior_rank_e2_head"
            / "deployment_risk_summary.json",
            "bass_gsf_p2_posterior_rank_e2",
        ),
        "gate": "p2 pending",
    },
]


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_eval_metric(eval_dir: Path) -> dict:
    if not eval_dir.exists():
        return {"status": "missing", "eval_json": "", "mAP": None, "AP50": None}
    candidates = sorted(eval_dir.glob("*/20*.json")) + sorted(
        eval_dir.glob("*.json"))
    for path in candidates:
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
                "status": "done",
                "eval_json": str(path),
                "mAP": float(map_value),
                "AP50": float(ap50_value) if ap50_value is not None else None,
            }
    return {"status": "missing", "eval_json": "", "mAP": None, "AP50": None}


def read_risk_row(summary_path: Path, variant: str) -> dict:
    payload = read_json(summary_path)
    if not isinstance(payload, dict):
        return {"risk_status": "missing", "risk_json": str(summary_path)}
    rows = payload.get("summaries", [])
    selected = None
    for row in rows:
        if row.get("variant") == variant:
            selected = row
            break
    if selected is None and rows:
        selected = rows[-1]
    if selected is None:
        return {"risk_status": "missing", "risk_json": str(summary_path)}
    return {
        "risk_status": "done",
        "risk_json": str(summary_path),
        "localized_correct": selected.get("localized_correct"),
        "localized_wrong": selected.get("localized_wrong"),
        "sise_logz": selected.get("sise_logz_wrong_excl_sibling"),
        "top5000_precision": selected.get("topk_precision_top5000"),
        "top5000_sise_logz": selected.get("sise_logz_topk_top5000"),
        "risk_weighted_logz": selected.get("risk_weighted_logz_false_alarm"),
    }


def fmt(value, digits=6):
    if value is None:
        return "TBD"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def assess_gate(row: dict, density: dict, nog3: dict) -> str:
    if not row["kind"].startswith("rankdelta"):
        return row.get("gate", "")
    required_keys = [
        "mAP", "top5000_precision", "top5000_sise_logz", "sise_logz",
        "localized_wrong", "risk_weighted_logz",
    ]
    if any(row.get(key) is None for key in required_keys):
        return "waiting/missing"
    pilot = (
        row["mAP"] >= density["mAP"]
        and row["top5000_precision"] >= density["top5000_precision"]
        and row["top5000_sise_logz"] <= density["top5000_sise_logz"]
        and row["sise_logz"] < density["sise_logz"])
    strict = (
        row["mAP"] >= nog3["mAP"]
        and row["top5000_precision"] >= nog3["top5000_precision"]
        and row["top5000_sise_logz"] <= nog3["top5000_sise_logz"]
        and row["localized_wrong"] < nog3["localized_wrong"])
    if strict:
        return "strict pass"
    if pilot:
        return "pilot pass"
    return "fail"


def build_rows() -> list[dict]:
    rows = []
    for spec in ROWS:
        metric = read_eval_metric(Path(spec["eval_dir"]))
        risk_path, variant = spec["risk"]
        risk = read_risk_row(Path(risk_path), variant)
        row = {
            "method": spec["method"],
            "kind": spec["kind"],
            **metric,
            **risk,
            "gate": spec["gate"],
        }
        rows.append(row)
    by_name = {row["method"]: row for row in rows}
    density = by_name["density w005 e2"]
    nog3 = by_name["no-G3 e2"]
    for row in rows:
        row["gate"] = assess_gate(row, density, nog3)
    return rows


def md_table(rows: list[dict]) -> str:
    columns = [
        "method", "mAP", "AP50", "localized_correct", "localized_wrong",
        "sise_logz", "top5000_precision", "top5000_sise_logz",
        "risk_weighted_logz", "gate",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join([
            str(row.get("method", "")),
            fmt(row.get("mAP"), 6),
            fmt(row.get("AP50"), 3),
            fmt(row.get("localized_correct"), 0),
            fmt(row.get("localized_wrong"), 0),
            fmt(row.get("sise_logz"), 0),
            fmt(row.get("top5000_precision"), 4),
            fmt(row.get("top5000_sise_logz"), 0),
            fmt(row.get("risk_weighted_logz"), 4),
            str(row.get("gate", "")),
        ]) + " |")
    return "\n".join(lines)


def write_markdown(path: Path, rows: list[dict]) -> None:
    p0_rows = [row for row in rows if row["kind"] == "rankdelta_p0"]
    p1_rows = [row for row in rows if row["kind"] == "rankdelta_p1"]
    p0_done = all(row.get("mAP") is not None for row in p0_rows)
    p1_done = all(row.get("mAP") is not None for row in p1_rows)
    if not p0_done:
        status = "waiting/missing"
    elif not p1_done:
        status = "p0_done/follow-up_missing"
    else:
        status = "done"
    lines = [
        "# BASS-GSF RankDelta GPU6/7 Results - 2026-06-20",
        "",
        f"Status: **{status}**.",
        "",
        "This file is generated from existing eval JSON and deployment-risk "
        "summaries. Missing rows mean the GPU6/7 jobs have not finished or the "
        "risk audit has not been produced yet.",
        "",
        "## Main Comparison",
        "",
        md_table(rows),
        "",
        "## Gate Definitions",
        "",
        "Strict pass:",
        "",
        "```text",
        "mAP >= no-G3 e2",
        "top5000_precision >= no-G3 e2",
        "top5000_logz_sise <= no-G3 e2",
        "localized_wrong < no-G3 e2",
        "```",
        "",
        "Pilot pass:",
        "",
        "```text",
        "mAP >= density w005 e2",
        "top5000_precision >= density w005 e2",
        "top5000_logz_sise <= density w005 e2",
        "global log-z SISE < density w005 e2",
        "```",
        "",
        "## Artifact Trace",
        "",
        "| method | eval_json | risk_json |",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | `{row.get('eval_json', '')}` | "
            f"`{row.get('risk_json', '')}` |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument("--out-md", default=str(OUT_MD))
    args = parser.parse_args()
    rows = build_rows()
    p0_rows = [row for row in rows if row["kind"] == "rankdelta_p0"]
    p1_rows = [row for row in rows if row["kind"] == "rankdelta_p1"]
    p0_done = all(row.get("mAP") is not None for row in p0_rows)
    p1_done = all(row.get("mAP") is not None for row in p1_rows)
    if not p0_done:
        status = "waiting/missing"
    elif not p1_done:
        status = "p0_done/follow-up_missing"
    else:
        status = "done"
    payload = {
        "status": status,
        "rows": rows,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    write_markdown(Path(args.out_md), rows)
    print(json.dumps({
        "status": payload["status"],
        "out_json": str(out_json),
        "out_md": args.out_md,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
