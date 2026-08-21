#!/usr/bin/env python
"""Summarize P3D second-dataset transfer evidence.

P3D is not a new bbox-overlap metric. It reuses the P3A AP-constrained
Gaussian semantic-support projection and checks whether the same ranking
principle transfers to a second remote-sensing dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


DEFAULT_BASELINE_EVAL_JSON = Path(
    "work_dirs/gs3c_dior_r_rtmdetl_dota_init_20260619/"
    "eval_epoch3_plus3_baseline_full/offline_eval.json")
DEFAULT_P3D_EVAL_JSON = Path(
    "work_dirs/gs3c_dior_r_rtmdetl_dota_init_20260619/"
    "eval_epoch3_plus3_baseline_p3d_z2_s01/offline_eval.json")
DEFAULT_PROJECTION_JSON = Path(
    "work_dirs/gs3c_dior_r_rtmdetl_dota_init_20260619/"
    "eval_epoch3_plus3_baseline_p3d_z2_s01/p3d_projection_summary.json")
DEFAULT_RISK_JSON = Path(
    "work_dirs/gs3c_dior_r_rtmdetl_dota_init_20260619/"
    "sise_eval_p3d_transfer_z2_s01/deployment_risk_summary.json")
DEFAULT_OUT_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "p3d_transfer_summary.json")
DEFAULT_OUT_MD = Path(
    "resultmd/exp_p4_scale_semantic_validation/"
    "fres_20260621_p3d_dior_transfer_gate.md")

DATASET = "DIOR-R"
BASELINE_VARIANT = "dior_baseline"
P3D_VARIANT = "p3d_dior_z2_s01"


def read_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(payload: dict[str, Any], key: str) -> float | None:
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        payload = {**payload, **metrics}
    value = payload.get(f"dota/{key}", payload.get(key))
    return None if value is None else float(value)


def read_eval_metric(path: str | Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        return {
            "eval_status": "missing",
            "eval_json": str(path),
            "mAP": None,
            "AP50": None,
        }
    return {
        "eval_status": "done",
        "eval_json": str(path),
        "mAP": _metric(payload, "mAP"),
        "AP50": _metric(payload, "AP50"),
    }


def read_projection_summary(path: str | Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        return {
            "projection_status": "missing",
            "projection_json": str(path),
        }
    return {
        "projection_status": "done",
        "projection_json": str(path),
        "z_thr": payload.get("z_thr"),
        "min_score": payload.get("min_score"),
        "scores_changed": payload.get("scores_changed"),
        "protected_correct": payload.get("protected_correct"),
        "changed_correct": payload.get("changed_correct"),
        "low_support_wrong_candidates": payload.get(
            "low_support_wrong_candidates"),
        "forbidden_bbox_gaussian_route": payload.get(
            "forbidden_bbox_gaussian_route"),
    }


def _select_variant(payload: dict[str, Any],
                    variant: str) -> dict[str, Any] | None:
    for row in payload.get("summaries", []) or []:
        if isinstance(row, dict) and row.get("variant") == variant:
            return row
    return None


def _select_rank_tail(payload: dict[str, Any],
                      variant: str) -> dict[str, Any]:
    for row in payload.get("rank_tail_summary", []) or []:
        if isinstance(row, dict) and row.get("method_variant") == variant:
            return row
    return {}


def read_risk_row(path: str | Path, variant: str) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        return {"risk_status": "missing", "risk_json": str(path)}
    selected = _select_variant(payload, variant)
    if selected is None:
        return {"risk_status": "missing", "risk_json": str(path)}
    rank_tail = _select_rank_tail(payload, variant)
    return {
        "risk_status": "done",
        "risk_json": str(path),
        "localized_correct": selected.get("localized_correct"),
        "localized_wrong": selected.get("localized_wrong"),
        "risk_weighted_logz": selected.get(
            "risk_weighted_logz_false_alarm"),
        "risk_weighted_p0199": selected.get(
            "risk_weighted_p0199_false_alarm"),
        "risk_weighted_logz_score_ge_0p5": selected.get(
            "risk_weighted_logz_false_alarm_score_ge_0p5"),
        "sise_logz_score_ge_0p5": selected.get(
            "sise_logz_ge_thr_score_ge_0p5"),
        "top5000_precision": selected.get("topk_precision_top5000"),
        "top5000_sise_logz": selected.get("sise_logz_topk_top5000"),
        "top5000_risk_weighted_logz": selected.get(
            "risk_weighted_logz_false_alarm_top5000"),
        "ap_contributing_fp_removed": rank_tail.get(
            "ap_contributing_fp_removed"),
        "tp_rank_harm": rank_tail.get("tp_rank_harm"),
        "sise_fp_rank_benefit": rank_tail.get("sise_fp_rank_benefit"),
        "rank_disruptive_delta": rank_tail.get("rank_disruptive_delta"),
        "tail_safety_status": rank_tail.get("tail_safety_status"),
    }


def _first_metric(row: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if row.get(key) is not None:
            return row[key]
    return None


def _improved(row: dict[str, Any], base: dict[str, Any],
              keys: Sequence[str]) -> bool:
    row_value = _first_metric(row, keys)
    base_value = _first_metric(base, keys)
    return (
        row_value is not None and base_value is not None
        and float(row_value) < float(base_value))


def assess_p3d_transfer_gate(row: dict[str, Any],
                             baseline: dict[str, Any]) -> str:
    """Strict transfer gate for second-dataset P3D evidence."""
    row_map = _first_metric(row, ("mAP", "AP50"))
    base_map = _first_metric(baseline, ("mAP", "AP50"))
    row_precision = _first_metric(row, ("top5000_precision", ))
    base_precision = _first_metric(baseline, ("top5000_precision", ))
    if any(value is None for value in (
            row_map, base_map, row_precision, base_precision)):
        return "waiting/missing"
    if float(row.get("changed_correct", 0) or 0) > 0:
        return "fail"
    if float(row.get("tp_rank_harm", 0) or 0) > 0:
        return "fail"
    if int(row.get("scores_changed", 0) or 0) <= 0:
        return "fail"

    ap_safe = float(row_map) >= float(base_map) - 1e-12
    precision_safe = float(row_precision) >= float(base_precision) - 1e-12
    risk_improved = (
        _improved(row, baseline, ("top5000_sise_logz", ))
        or _improved(row, baseline, ("top5000_risk_weighted_logz", ))
        or _improved(row, baseline, ("risk_weighted_logz", ))
        or _improved(row, baseline, ("risk_weighted_logz_score_ge_0p5", ))
        or _improved(row, baseline, ("sise_logz_score_ge_0p5", )))
    return "strict transfer pass" if (
        ap_safe and precision_safe and risk_improved) else "fail"


def build_rows(baseline_eval_json: str | Path,
               p3d_eval_json: str | Path,
               projection_json: str | Path,
               risk_json: str | Path) -> list[dict[str, Any]]:
    baseline = {
        "dataset": DATASET,
        "method": "DIOR-R baseline",
        "kind": "transfer_control",
        **read_eval_metric(baseline_eval_json),
        **read_risk_row(risk_json, BASELINE_VARIANT),
        "gate": "transfer control",
    }
    p3d = {
        "dataset": DATASET,
        "method": "P3D DIOR-R z2 s0.1",
        "kind": "p3d_transfer_main",
        **read_eval_metric(p3d_eval_json),
        **read_projection_summary(projection_json),
        **read_risk_row(risk_json, P3D_VARIANT),
        "gate": "pending",
    }
    p3d["gate"] = assess_p3d_transfer_gate(p3d, baseline)
    return [baseline, p3d]


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "TBD"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def md_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        "method", "dataset", "mAP", "top5000_precision",
        "risk_weighted_logz", "risk_weighted_logz_score_ge_0p5",
        "scores_changed", "changed_correct", "tp_rank_harm",
        "ap_contributing_fp_removed", "gate",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [
            row.get("method", ""),
            row.get("dataset", ""),
            fmt(row.get("mAP"), 6),
            fmt(row.get("top5000_precision"), 4),
            fmt(row.get("risk_weighted_logz"), 4),
            fmt(row.get("risk_weighted_logz_score_ge_0p5"), 4),
            fmt(row.get("scores_changed"), 0),
            fmt(row.get("changed_correct"), 0),
            fmt(row.get("tp_rank_harm"), 4),
            fmt(row.get("ap_contributing_fp_removed"), 0),
            row.get("gate", ""),
        ]
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines)


def write_markdown(path: str | Path, rows: list[dict[str, Any]]) -> None:
    main = next(
        (row for row in rows if row.get("kind") == "p3d_transfer_main"),
        {})
    lines = [
        "# P3D DIOR-R Transfer Gate - 2026-06-21",
        "",
        "Status: generated from existing DIOR-R projection/eval/risk JSON. "
        "P3D uses the same AP-constrained Gaussian semantic-support ranking "
        "projection as P3A, but validates it on a second dataset. 这里的高斯"
        "只作用在 class-conditioned log-area semantic support 上，不作为 "
        "bbox IoU/GWD/KLD/NWD 的替代。",
        "",
        "## Main Result",
        "",
        md_table(rows),
        "",
        "## Gate Definition",
        "",
        "Strict transfer pass 要求：DIOR-R mAP 不低于 baseline；top5000 "
        "precision 不下降；`changed_correct=0`；`tp_rank_harm=0`；至少一个"
        "语义风险面下降。top5000 在 DIOR-R baseline 已经是零风险，因此这次"
        "主要依赖全局和 score>=0.5 deployment risk 的下降。",
        "",
        "## Selected Row",
        "",
        f"- main variant: `{main.get('method', 'missing')}`",
        f"- gate: `{main.get('gate', 'missing')}`",
        f"- AP: `{fmt(main.get('mAP'), 6)}`",
        f"- top5000 precision: `{fmt(main.get('top5000_precision'), 4)}`",
        f"- all-risk log-z false alarm: "
        f"`{fmt(main.get('risk_weighted_logz'), 4)}`",
        f"- score>=0.5 risk-weighted log-z false alarm: "
        f"`{fmt(main.get('risk_weighted_logz_score_ge_0p5'), 4)}`",
        f"- changed correct positives: `{fmt(main.get('changed_correct'), 0)}`",
        f"- TP rank harm: `{fmt(main.get('tp_rank_harm'), 4)}`",
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_payload(rows: list[dict[str, Any]], result_md: str) -> dict[str, Any]:
    main = next(
        (row for row in rows if row.get("kind") == "p3d_transfer_main"),
        {})
    strict_rows = [
        row for row in rows
        if row.get("gate") == "strict transfer pass"
    ]
    return {
        "status": "done" if rows else "missing",
        "dataset": DATASET,
        "strict_transfer_pass": bool(strict_rows),
        "strict_transfer_pass_methods": [
            row.get("method") for row in strict_rows
        ],
        "main_method": main.get("method"),
        "main_gate": main.get("gate"),
        "main_mAP": main.get("mAP"),
        "main_top5000_precision": main.get("top5000_precision"),
        "main_risk_weighted_logz": main.get("risk_weighted_logz"),
        "changed_correct": main.get("changed_correct"),
        "tp_rank_harm": main.get("tp_rank_harm"),
        "ap_contributing_fp_removed": main.get(
            "ap_contributing_fp_removed"),
        "result_md": result_md,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-eval-json", default=str(DEFAULT_BASELINE_EVAL_JSON))
    parser.add_argument("--p3d-eval-json", default=str(DEFAULT_P3D_EVAL_JSON))
    parser.add_argument("--projection-json", default=str(DEFAULT_PROJECTION_JSON))
    parser.add_argument("--risk-json", default=str(DEFAULT_RISK_JSON))
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    parser.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(
        args.baseline_eval_json,
        args.p3d_eval_json,
        args.projection_json,
        args.risk_json,
    )
    payload = build_payload(rows, args.out_md)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    write_markdown(args.out_md, rows)
    print(json.dumps({
        "status": payload["status"],
        "dataset": payload["dataset"],
        "main_gate": payload["main_gate"],
        "strict_transfer_pass": payload["strict_transfer_pass"],
        "out_json": str(out_json),
        "out_md": args.out_md,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
