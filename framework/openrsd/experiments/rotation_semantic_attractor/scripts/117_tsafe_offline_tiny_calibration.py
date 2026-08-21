#!/usr/bin/env python3
"""Stage 3 offline one-way calibration for FOCUS-T-Safe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from focus_tsafe_common import (
    EXP_DIR,
    read_csv,
    read_json,
    resolve,
    safe_float,
    write_csv,
    write_json,
)


def apply_logit_downweight(
        focus_scores: torch.Tensor,
        eqtext_sv_similarity: torch.Tensor,
        max_negative_text_similarity: torch.Tensor,
        lambda_value: float) -> torch.Tensor:
    """Apply one-way downweight; output never exceeds focus_scores."""
    focus = focus_scores.float()
    violation = (max_negative_text_similarity.float()
                 - eqtext_sv_similarity.float()).clamp_min(0.0)
    adjusted = focus - float(lambda_value) * violation
    return torch.minimum(adjusted.clamp_min(0.0), focus)


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def evaluate_lambda(rows: list[dict[str, str]], lambda_value: float,
                    threshold: float = 0.5) -> dict[str, object]:
    focus = torch.tensor([safe_float(row.get("focus_sv_score")) for row in rows])
    eq = torch.tensor([safe_float(row.get("eqtext_sv_similarity")) for row in rows])
    neg = torch.tensor([safe_float(row.get("max_negative_text_similarity")) for row in rows])
    adjusted = apply_logit_downweight(focus, eq, neg, lambda_value)
    categories = [row.get("audit_category", "") for row in rows]
    baseline_keep = focus >= threshold
    adjusted_keep = adjusted >= threshold
    corrected_idx = [idx for idx, cat in enumerate(categories) if cat == "corrected_false_sv"]
    true_idx = [
        idx for idx, cat in enumerate(categories)
        if cat in {"annotation_missing_true_vehicle", "true_sv_positive_control"}
    ]
    deg_idx = [idx for idx, cat in enumerate(categories)
               if cat == "degenerate_large_sv_box"]
    ann_idx = [idx for idx, cat in enumerate(categories)
               if cat == "annotation_missing_true_vehicle"]

    baseline_true = int(baseline_keep[true_idx].sum()) if true_idx else 0
    adjusted_true = int(adjusted_keep[true_idx].sum()) if true_idx else 0
    true_retention = (
        adjusted_true / baseline_true if baseline_true > 0 else 1.0)
    baseline_ann = int(baseline_keep[ann_idx].sum()) if ann_idx else 0
    adjusted_ann = int(adjusted_keep[ann_idx].sum()) if ann_idx else 0
    annotation_retention = (
        adjusted_ann / baseline_ann if baseline_ann > 0 else 1.0)
    baseline_deg_rate = (
        float(baseline_keep[deg_idx].float().mean()) if deg_idx else 0.0)
    adjusted_deg_rate = (
        float(adjusted_keep[deg_idx].float().mean()) if deg_idx else 0.0)
    baseline_dets = int(baseline_keep.sum())
    adjusted_dets = int(adjusted_keep.sum())
    corrected_baseline = int(baseline_keep[corrected_idx].sum()) if corrected_idx else 0
    corrected_adjusted = int(adjusted_keep[corrected_idx].sum()) if corrected_idx else 0
    rejected = []
    if adjusted_dets > baseline_dets:
        rejected.append("small_vehicle_dets_increase")
    if true_retention < 0.95:
        rejected.append("true_retention_below_95")
    if adjusted_deg_rate > baseline_deg_rate + 1e-12:
        rejected.append("degenerate_score_increase")
    return {
        "lambda": lambda_value,
        "baseline_small_vehicle_dets": baseline_dets,
        "adjusted_small_vehicle_dets": adjusted_dets,
        "det_img_proxy": adjusted_dets,
        "corrected_FSV_baseline": corrected_baseline,
        "corrected_FSV_adjusted": corrected_adjusted,
        "corrected_FSV_delta": corrected_adjusted - corrected_baseline,
        "true_SV_retention": true_retention,
        "annotation_missing_retention": annotation_retention,
        "degenerate_large_sv_high_score_rate_baseline": baseline_deg_rate,
        "degenerate_large_sv_high_score_rate_adjusted": adjusted_deg_rate,
        "migration_proxy": _mean(
            [abs(float(a) - float(b)) for a, b in zip(adjusted.tolist(), focus.tolist())]),
        "one_way_down_only": bool(torch.all(adjusted <= focus)),
        "status": "PASS_OFFLINE_ONE_WAY" if not rejected else "REJECT_OFFLINE_CALIBRATION",
        "reject_reasons": "; ".join(rejected),
    }


def write_report(path: Path, payload: dict[str, object],
                 rows: list[dict[str, object]]) -> None:
    lines = [
        "# FOCUS-T-Safe Offline Tiny Calibration",
        "",
        f"- status: `{payload['status']}`",
        f"- selected_lambda: `{payload.get('selected_lambda', '')}`",
        f"- reason: `{payload.get('reason', '')}`",
        "",
        "This is an offline one-way sweep over existing FOCUS scores and shadow "
        "scores. It does not train and does not modify checkpoint weights.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path,
                        default=EXP_DIR / "shadow_scores" / "tsafe_shadow_scores.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=EXP_DIR / "offline_mixing")
    parser.add_argument("--analysis-json", type=Path,
                        default=EXP_DIR / "reports" / "tsafe_shadow_signal_report.json")
    parser.add_argument("--only-if-shadow-signal-pass", action="store_true")
    parser.add_argument("--lambdas", nargs="*", type=float,
                        default=[0.0, 0.001, 0.0025, 0.005])
    args = parser.parse_args()

    scores = args.scores if args.scores.is_absolute() else Path.cwd() / args.scores
    output_dir = args.output_dir if args.output_dir.is_absolute() else Path.cwd() / args.output_dir
    analysis_json = (
        args.analysis_json if args.analysis_json.is_absolute()
        else Path.cwd() / args.analysis_json)
    analysis = read_json(analysis_json, {})
    if args.only_if_shadow_signal_pass and analysis.get("status") != "SHADOW_SIGNAL_PASS":
        payload = {
            "status": "SKIPPED_SHADOW_SIGNAL_NOT_PASS",
            "reason": analysis.get("status", "missing_shadow_analysis"),
            "scores": str(scores),
            "analysis_json": str(analysis_json),
        }
        write_json(output_dir / "tsafe_tiny_calibration_report.json", payload)
        write_csv(output_dir / "tsafe_tiny_calibration_sweep.csv", [])
        write_report(output_dir / "tsafe_tiny_calibration_report.md", payload, [])
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    rows = read_csv(scores)
    sweep = [evaluate_lambda(rows, value) for value in args.lambdas]
    passing = [row for row in sweep if row["status"] == "PASS_OFFLINE_ONE_WAY"]
    selected = min(
        passing,
        key=lambda row: (row["corrected_FSV_adjusted"], row["lambda"]),
        default=None)
    payload = {
        "status": (
            "PASS_TINY_ONE_WAY_CALIBRATION" if selected is not None
            else "NO_SAFE_OFFLINE_CALIBRATION"),
        "selected_lambda": selected["lambda"] if selected else "",
        "selected_row": selected or {},
        "scores": str(scores),
        "analysis_json": str(analysis_json),
        "text_side_can_affect_logits": False,
        "checkpoint_modified": False,
        "trained": False,
    }
    write_csv(output_dir / "tsafe_tiny_calibration_sweep.csv", sweep)
    write_json(output_dir / "tsafe_tiny_calibration_report.json", payload)
    write_report(output_dir / "tsafe_tiny_calibration_report.md", payload, sweep)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
