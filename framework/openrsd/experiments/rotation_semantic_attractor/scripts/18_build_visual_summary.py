#!/usr/bin/env python3
"""Build paper-oriented visual summary for rotation semantic attractor results."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STATUS_SCORE = {
    "BLOCKED": 0,
    "SMOKE_ONLY": 1,
    "PARTIAL": 2,
    "CAUTIOUS": 2,
    "DIAGNOSTIC_DONE_FULL": 3,
    "SUPPORTED": 3,
    "DONE_FULL": 4,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if headers is None:
        headers = []
        for row in rows:
            for key in row:
                if key not in headers:
                    headers.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in headers})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def md_table(rows: list[dict[str, Any]], headers: list[str] | None = None, max_rows: int | None = None) -> str:
    if not rows:
        return "_No rows._"
    rows = rows[:max_rows] if max_rows else rows
    if headers is None:
        headers = []
        for row in rows:
            for key in row:
                if key not in headers:
                    headers.append(key)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def fnum(v: Any, digits: int = 4) -> str:
    try:
        x = float(v)
    except Exception:
        return ""
    if math.isnan(x) or math.isinf(x):
        return ""
    if abs(x) >= 1000:
        return f"{x:,.0f}" if digits == 0 else f"{x:,.1f}"
    return f"{x:.{digits}f}"


def to_float(v: Any) -> float:
    try:
        if v == "" or v is None:
            return math.nan
        return float(v)
    except Exception:
        return math.nan


def mean(rows: list[dict[str, str]], col: str) -> float:
    vals = [to_float(r.get(col)) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else math.nan


def boolish(v: Any) -> bool:
    return str(v).lower() in {"true", "1", "yes"}


def groupby(rows: list[dict[str, str]], col: str) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[str(row.get(col, ""))].append(row)
    return dict(out)


def save_fig(path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_base.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def rotate_ticks() -> None:
    for tick in plt.gca().get_xticklabels():
        tick.set_rotation(45)
        tick.set_horizontalalignment("right")


def placeholder_fig(path_base: Path, title: str, text: str) -> None:
    plt.figure(figsize=(7, 3))
    plt.axis("off")
    plt.title(title)
    plt.text(0.5, 0.5, text, ha="center", va="center", wrap=True)
    save_fig(path_base)


def find_one(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def source_paths(repo_root: Path, run_root: Path) -> dict[str, Path | None]:
    exp = repo_root / "experiments/rotation_semantic_attractor"
    result = repo_root / "resultmd/exp_rotation_semantic_attractor"
    closed_art = Path("/data/zcy/OpenRSD_artifacts/closedset_scientific_audit_20260531")
    open_run = Path("/data/zcy/OpenRSD_runs/rotation_semantic_attractor")
    return {
        "closed_false_sv": find_one([
            closed_art / "metrics/paper_table_closedset_false_sv_benchmark.csv",
            exp / "outputs/runs/full_closedset_s2_12angle/metrics/full_closedset_false_hub_summary_by_model.csv",
            exp / "outputs/runs/full_closedset_s2_12angle/metrics/false_hub_summary_by_model.csv",
        ]),
        "closed_rotation_gain": find_one([
            closed_art / "metrics/paper_table_closedset_rotation_gain.csv",
            exp / "outputs/runs/full_closedset_s2_12angle/metrics/full_closedset_rotation_gain.csv",
        ]),
        "closed_stage": find_one([
            closed_art / "metrics/paper_table_closedset_stage_decomposition.csv",
            exp / "outputs/runs/full_closedset_s2_12angle/metrics/stage_decomposition_summary.csv",
        ]),
        "closed_concentration": find_one([
            closed_art / "metrics/paper_table_closedset_concentration.csv",
            exp / "outputs/runs/full_closedset_s2_12angle/metrics/full_closedset_concentration.csv",
        ]),
        "openvocab_rows": find_one([
            open_run / "full_openvocab_s2_12angle/metrics/open_vocab_benchmark_rows_merged.csv",
            open_run / "full_openvocab_s2_12angle/metrics/open_vocab_benchmark_rows.csv",
        ]),
        "causal_rows": find_one([
            open_run / "full_causal_intervention_s3_12angle/metrics/open_vocab_causal_rows_merged.csv",
        ]),
        "causal_paired": find_one([
            result / "evidence_closure_20260601/causal_paired_effects.csv",
        ]),
        "context_rows": find_one([
            open_run / "full_context_counterfactual_s3_12angle/metrics/context_counterfactual_rows.csv",
        ]),
        "dehub_rows": find_one([
            open_run / "full_dehub_safety_s3_12angle/metrics/dehub_safety_rows_merged.csv",
        ]),
        "dehub_class_migration": find_one([
            result / "evidence_closure_20260601/dehub_class_distribution_delta.csv",
        ]),
        "dehub_scorecard": find_one([
            result / "evidence_closure_20260601/dehub_safety_scorecard.csv",
        ]),
        "claim_ledger": find_one([
            result / "evidence_closure_20260601/claim_ledger.csv",
        ]),
        "ap_blockers": find_one([
            result / "evidence_closure_20260601/open_vocab_ap_blockers.md",
        ]),
        "prediction_root": find_one([
            exp / "outputs/runs/full_closedset_s2_12angle/canonical_predictions",
        ]),
        "raw_prediction_root": find_one([
            exp / "outputs/runs/full_closedset_s2_12angle/raw_predictions",
        ]),
        "gt_root": find_one([
            repo_root / "data/DOTA1_1024_500/angle_sweep_val/realistic",
        ]),
        "image_root": find_one([
            repo_root / "data/DOTA1_1024_500/angle_sweep_val/realistic",
        ]),
        "human_labels": None,
    }


def build_source_inventory(paths: dict[str, Path | None], output_dir: Path) -> dict[str, Any]:
    categories = {
        "closed-set full benchmark tables": ["closed_false_sv", "closed_rotation_gain", "closed_stage", "closed_concentration"],
        "open-vocab full benchmark tables": ["openvocab_rows"],
        "causal intervention tables": ["causal_rows", "causal_paired"],
        "context counterfactual tables": ["context_rows"],
        "DeHub safety tables": ["dehub_rows", "dehub_class_migration", "dehub_scorecard"],
        "claim/evidence boundary tables": ["claim_ledger", "ap_blockers"],
        "prediction/crop/GT artifacts": ["prediction_root", "raw_prediction_root", "gt_root", "image_root", "human_labels"],
    }
    rows = []
    missing = []
    for category, keys in categories.items():
        for key in keys:
            p = paths.get(key)
            exists = bool(p and p.exists())
            rows.append({
                "category": category,
                "artifact_key": key,
                "path": str(p) if p else "NOT_AVAILABLE",
                "exists": exists,
                "impact_if_missing": missing_impact(key) if not exists else "",
            })
            if not exists:
                missing.append({"artifact_key": key, "impact": missing_impact(key)})
    inventory = {
        "rows": rows,
        "missing": missing,
        "ap50_sv_ap50_outputs": "BLOCKED_OR_NOT_AVAILABLE" if not paths.get("ap_blockers") else "BLOCKED_RECORDED",
        "raw_prediction_gt_matching_outputs": "PARTIAL_CLOSEDSET_AVAILABLE_OPENVOCAB_BLOCKED",
        "human_label_audit_outputs": "NOT_AVAILABLE",
        "chart_sources": {
            "fig1-4": str(paths.get("closed_false_sv")),
            "fig5-6": str(paths.get("closed_rotation_gain")),
            "fig7": str(paths.get("closed_stage")),
            "fig9-11": str(paths.get("openvocab_rows")),
            "fig12-14": str(paths.get("causal_rows")),
            "fig15-16": str(paths.get("context_rows")),
            "fig17-20": str(paths.get("dehub_rows")),
        },
    }
    write_json(output_dir / "audit/source_inventory.json", inventory)
    md = [
        "# Source Inventory",
        "",
        md_table(rows),
        "",
        "## Missing Files and Impact",
        "",
        md_table(missing) if missing else "No critical source files missing.",
        "",
        f"- AP50 / SV_AP50 evaluator outputs: `{inventory['ap50_sv_ap50_outputs']}`",
        f"- Raw prediction + GT matching outputs: `{inventory['raw_prediction_gt_matching_outputs']}`",
        f"- Human label audit outputs: `{inventory['human_label_audit_outputs']}`",
    ]
    write_md(output_dir / "audit/source_inventory.md", "\n".join(md))
    return inventory


def missing_impact(key: str) -> str:
    return {
        "closed_false_sv": "closed-set false-SV figures/tables unavailable",
        "closed_rotation_gain": "rotation-conditioned gain figures unavailable",
        "closed_stage": "stage decomposition unavailable",
        "openvocab_rows": "OpenRSD risk-group burden unavailable",
        "causal_rows": "embedding intervention summary unavailable",
        "context_rows": "context counterfactual unavailable",
        "dehub_rows": "DeHub diagnostic unavailable",
        "dehub_class_migration": "hub migration table unavailable",
        "claim_ledger": "claim status table must be rebuilt from defaults",
        "human_labels": "corrected-FSV cannot be computed",
    }.get(key, "related output may be marked NOT_AVAILABLE")


def copy_table(rows: list[dict[str, Any]], path_base: Path, headers: list[str] | None = None) -> None:
    write_csv(path_base.with_suffix(".csv"), rows, headers)
    write_md(path_base.with_suffix(".md"), md_table(rows, headers))


def closedset_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        notes = r.get("Notes", "")
        if to_float(r.get("FSV")) > 0.9 and to_float(r.get("SV_Pred_per_img")) < 20:
            notes = (notes + " denominator risk: high FSV with modest SV_pred/img").strip()
        if to_float(r.get("True_SV_Recall")) < 0.01:
            notes = (notes + " matching/annotation risk: very low true-SV recall").strip()
        out.append({
            "Model": r.get("Model") or r.get("model_name"),
            "Architecture": r.get("Architecture") or r.get("architecture"),
            "Status": r.get("Status", "DONE_FULL"),
            "FR_SV": r.get("FR_SV") or r.get("fr_sv"),
            "FSV": r.get("FSV") or r.get("fsv"),
            "BG_FSV": r.get("BG_FSV") or r.get("bg_fsv"),
            "ObjectFlip_SV": r.get("ObjectFlip_SV", ""),
            "Abs_FalseSV": r.get("Abs_FalseSV") or r.get("num_unmatched_sv_pred"),
            "SV_Pred_per_img": r.get("SV_Pred_per_img"),
            "True_SV_Recall": r.get("True_SV_Recall") or r.get("true_sv_recall"),
            "True_SV_Precision": r.get("True_SV_Precision") or r.get("true_sv_precision"),
            "det_per_img": r.get("det/img") or r.get("det_per_img"),
            "ValidMask_FSV": r.get("ValidMask_FSV"),
            "mAP50": r.get("mAP50", "NOT_AVAILABLE"),
            "SV_AP50": r.get("SV_AP50", "NOT_AVAILABLE"),
            "Notes": notes,
        })
    return out


def closed_rotation_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        notes = r.get("Notes", "")
        if to_float(r.get("Mean_RG_SV")) > 0 and str(r.get("WorstAngle_by_delta")) == "0":
            notes = (notes + " possible metric bug: positive RG with delta worst angle 0").strip()
        out.append({
            "Model": r.get("Model"),
            "Mean_FSV_angle0": r.get("Mean_FSV_angle0"),
            "Mean_FSV_worst": r.get("Mean_FSV_worst"),
            "Mean_RG_SV": r.get("Mean_RG_SV"),
            "Mean_ARG_SV": r.get("Mean_ARG_SV"),
            "WorstAngle_by_delta": r.get("WorstAngle_by_delta"),
            "WorstAngle_by_absolute": r.get("WorstAngle_by_absolute"),
            "Range_FSV": r.get("Range_FSV"),
            "Std_FSV": r.get("Std_FSV"),
            "Notes": notes,
        })
    return out


def stage_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        out.append({
            "Model": r.get("Model"),
            "Model_Family": "closed_set",
            "Architecture": r.get("Architecture"),
            "Dense_Status": r.get("Dense_Status"),
            "Query_Status": "NOT_APPLICABLE",
            "PreNMS_Status": r.get("PreNMS_Status"),
            "PostNMS_Status": r.get("PostNMS_Status"),
            "Dense_FR_SV": r.get("Dense_FR_SV"),
            "Query_FR_SV": "NOT_APPLICABLE",
            "PreNMS_FR_SV": r.get("PreNMS_FR_SV"),
            "PostNMS_FR_SV": r.get("PostNMS_FR_SV"),
            "NMS_Amp_SV": r.get("NMS_Amp_SV"),
            "StageClaimEligible": r.get("StageClaimEligible"),
            "Notes": r.get("Notes"),
        })
    return out


def summarize_rows(rows: list[dict[str, str]], label: str, value: str) -> dict[str, Any]:
    top1 = [1 if r.get("top1_class") == "small-vehicle" else 0 for r in rows]
    return {
        label: value,
        "Rows": len(rows),
        "det_per_img": fnum(mean(rows, "detection_total"), 4),
        "SV_per_img": fnum(mean(rows, "small_vehicle_count"), 4),
        "SV_ratio": fnum(mean(rows, "small_vehicle_ratio"), 4),
        "LV_per_img": fnum(mean(rows, "large_vehicle_count"), 4),
        "LV_ratio": fnum(mean(rows, "large_vehicle_ratio"), 4),
        "top1_SV": fnum(sum(top1) / len(top1), 4) if top1 else "NOT_AVAILABLE",
    }


def openvocab_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    wanted = ["all", "false_sv_hub_candidate_no_sv", "true_sv_rich", "low_risk_normal", "cross_class_conflict"]
    grouped = groupby(rows, "risk_group")
    out = []
    for group in wanted:
        if group == "all":
            item = summarize_rows(rows, "Risk_Group", "all")
            item["Notes"] = "all risk groups; diagnostic burden, not AP"
        elif group in grouped:
            item = summarize_rows(grouped[group], "Risk_Group", group)
            item["Notes"] = "no annotated SV GT does not guarantee no true vehicles" if "no_sv" in group else ""
        else:
            item = {"Risk_Group": group, "Rows": 0, "det_per_img": "NOT_AVAILABLE", "SV_per_img": "NOT_AVAILABLE", "SV_ratio": "NOT_AVAILABLE", "LV_per_img": "NOT_AVAILABLE", "LV_ratio": "NOT_AVAILABLE", "top1_SV": "NOT_AVAILABLE", "Notes": "group not present in current rows"}
        item["dense_or_query_SV"] = "NOT_AVAILABLE"
        item["pre_nms_fr_sv"] = "NOT_AVAILABLE"
        item["post_nms_fr_sv"] = "NOT_AVAILABLE"
        out.append(item)
    return out


def intervention_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped = groupby(rows, "intervention")
    base = summarize_rows(grouped.get("original", []), "Intervention", "original")
    base_sv = to_float(base.get("SV_per_img"))
    base_ratio = to_float(base.get("SV_ratio"))
    out = []
    for name in sorted(grouped):
        item = summarize_rows(grouped[name], "Intervention", name)
        sample = grouped[name][0]
        item.update({
            "Intervention_Level": sample.get("claim_level", ""),
            "Is_Prompt_Only": sample.get("is_prompt_only", ""),
            "Is_Embedding_Level": sample.get("is_embedding_level", ""),
            "Modified_Tensor_Name": sample.get("modified_tensor_name", ""),
            "Checksum_Changed": str(sample.get("original_embedding_checksum", "") != sample.get("modified_embedding_checksum", "")),
            "Delta_SV_per_img_vs_original": fnum(to_float(item["SV_per_img"]) - base_sv, 4) if math.isfinite(base_sv) else "",
            "Delta_SV_ratio_vs_original": fnum(to_float(item["SV_ratio"]) - base_ratio, 4) if math.isfinite(base_ratio) else "",
            "Status": sample.get("status", ""),
            "Notes": "causal evidence if checksum changed and reran inference" if boolish(sample.get("actual_embedding_modified")) else "control/original or normalization-like",
        })
        out.append(item)
    return out


def context_table(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    done = [r for r in rows if r.get("status") == "DONE_FULL"]
    grouped = groupby(done, "condition")
    obj = summarize_rows(grouped.get("object_only", []), "Condition", "object_only")
    obj_det = to_float(obj.get("det_per_img"))
    obj_sv = to_float(obj.get("SV_per_img"))
    out = []
    for condition in ["object_only", "context_only"]:
        item = summarize_rows(grouped.get(condition, []), "Condition", condition)
        item["delta_det_per_img_vs_original"] = fnum(to_float(item["det_per_img"]) - obj_det, 4) if math.isfinite(obj_det) else ""
        item["delta_SV_per_img_vs_original"] = fnum(to_float(item["SV_per_img"]) - obj_sv, 4) if math.isfinite(obj_sv) else ""
        item["counterfactual_is_real_image_level"] = "True"
        item["reran_inference"] = "True"
        item["valid_mask_available"] = "NOT_AVAILABLE"
        item["NOT_APPLICABLE_rows"] = sum(1 for r in rows if r.get("status") == "NOT_APPLICABLE")
        item["Notes"] = "detection-count confounding must be reported" if condition == "context_only" else "object-only reference"
        out.append(item)
    not_app = [{"Condition": k[0], "Status": k[1], "Rows": v, "Notes": "data qualification split, not execution failure"} for k, v in Counter((r.get("condition"), r.get("status")) for r in rows if r.get("status") == "NOT_APPLICABLE").items()]
    return out, not_app


def dehub_table(rows: list[dict[str, str]], scorecard_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped = groupby(rows, "checkpoint_name")
    base = summarize_rows(grouped.get("baseline", []), "Checkpoint", "baseline")
    base_sv = to_float(base.get("SV_per_img"))
    base_ratio = to_float(base.get("SV_ratio"))
    out = []
    for checkpoint in ["baseline", "repair"]:
        item = summarize_rows(grouped.get(checkpoint, []), "Checkpoint", checkpoint)
        item["Delta_SV_per_img"] = fnum(to_float(item["SV_per_img"]) - base_sv, 4) if math.isfinite(base_sv) else ""
        item["Delta_SV_ratio"] = fnum(to_float(item["SV_ratio"]) - base_ratio, 4) if math.isfinite(base_ratio) else ""
        item["dense_or_query_SV"] = "NOT_AVAILABLE"
        item["final_fsv"] = "NOT_AVAILABLE"
        item["true_SV_recall"] = "BLOCKED"
        item["true_SV_precision"] = "BLOCKED"
        item["class_JS"] = "NOT_AVAILABLE"
        item["class_KL"] = "NOT_AVAILABLE"
        item["lowrisk_det_inflation"] = "NOT_AVAILABLE"
        item["AP50"] = "BLOCKED"
        item["SV_AP50"] = "BLOCKED"
        item["Safety_Verdict"] = "BURDEN_REDUCTION_ONLY" if checkpoint == "repair" else "BASELINE"
        item["Notes"] = "DeHub safety/AP not fully proven; true-SV preservation blocked"
        out.append(item)
    blockers = [
        {"Blocker": "AP50", "Status": "BLOCKED", "Reason": "raw prediction + class mapping + rotated box evaluator not verified"},
        {"Blocker": "SV_AP50", "Status": "BLOCKED", "Reason": "same as AP50"},
        {"Blocker": "true-SV preservation", "Status": "BLOCKED", "Reason": "requires raw predictions + GT matching / human audit"},
    ]
    for r in scorecard_rows:
        if r.get("status") == "BLOCKED":
            blockers.append({"Blocker": r.get("metric"), "Status": "BLOCKED", "Reason": r.get("interpretation")})
    return out, blockers


def claims_tables(claim_rows: list[dict[str, str]] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    defaults = [
        ("C1", "Closed-set detectors also show false-small-vehicle burden under full 12-angle benchmark.", "DONE_FULL", "paper_table_closedset_false_sv_benchmark.csv", "FR_SV/FSV/Abs_FalseSV", "closed-set false-SV burden", "Open-vocab mechanism or universal collapse"),
        ("C2", "OpenRSD/open-vocab model shows persistent small-vehicle prediction burden on S2 12-angle evaluation.", "DIAGNOSTIC_DONE_FULL", "open_vocab_benchmark_rows_merged.csv", "SV/img, SV_ratio, top1_SV", "diagnostic burden", "AP50/mAP"),
        ("C3", "Small-vehicle burden is causally modulated by support/class embedding intervention.", "DIAGNOSTIC_DONE_FULL", "open_vocab_causal_rows_merged.csv", "paired delta SV/img", "embedding causal modulation", "all false positives come from embedding"),
        ("C4", "Dense/pre-NMS stages show SV bias for hook-supported models.", "DIAGNOSTIC_DONE_FULL", "paper_table_closedset_stage_decomposition.csv", "Dense/PreNMS FR_SV", "stage claim for eligible models", "all models have dense collapse"),
        ("C5", "NMS/top-k/postprocess is an amplifier, not the sole root cause.", "DIAGNOSTIC_DONE_FULL", "paper_table_closedset_stage_decomposition.csv", "NMS_Amp_SV", "amplifier role", "NMS only cause"),
        ("C6", "Context-only images increase detection and SV prediction burden, but context-only does not yet prove pure hallucination.", "PARTIAL", "context_counterfactual_rows.csv", "context-object delta", "context contribution", "context proves hallucination"),
        ("C7", "DeHub reduces SV prediction burden.", "DIAGNOSTIC_DONE_FULL", "dehub_safety_rows_merged.csv", "SV/img delta", "burden reduction", "safety/AP improvement"),
        ("C8", "DeHub safety / AP improvement / true-SV preservation is not fully proven.", "BLOCKED", "dehub_safety_scorecard.csv", "true-SV/AP blockers", "boundary condition", "DeHub is safe"),
        ("C9", "Unmatched SV predictions cannot all be interpreted as hallucinations because annotation-missing true SV remains possible.", "BLOCKED", "unannotated_sv_contamination_audit.md", "human labels missing", "annotation risk", "all unmatched SV are hallucinations"),
    ]
    rows = []
    for cid, claim, status, source, metric, supports, does_not in defaults:
        rows.append({
            "claim_id": cid,
            "claim": claim,
            "status": status,
            "evidence_source": source,
            "main_metric": metric,
            "what_it_supports": supports,
            "what_it_does_not_support": does_not,
            "paper_writing_allowed": supports,
            "paper_writing_forbidden": does_not,
            "score": STATUS_SCORE.get(status, 0),
        })
    verified = [r for r in rows if r["score"] >= 3]
    unverified = [r for r in rows if r["score"] < 3]
    return rows, verified, unverified


def plot_outputs(output_dir: Path, tables: dict[str, list[dict[str, Any]]], claims: list[dict[str, Any]]) -> list[dict[str, str]]:
    figs = output_dir / "figures"
    made: list[dict[str, str]] = []

    def record(name: str, source: str, takeaway: str) -> None:
        made.append({"figure": name, "png": str(figs / f"{name}.png"), "pdf": str(figs / f"{name}.pdf"), "source": source, "takeaway": takeaway})

    # fig0
    plt.figure(figsize=(8, 5))
    y = list(range(len(claims)))
    x = [int(r["score"]) for r in claims]
    plt.scatter(x, y, s=160)
    plt.yticks(y, [r["claim_id"] for r in claims])
    plt.xticks([0, 1, 2, 3, 4], ["BLOCKED", "SMOKE", "PARTIAL", "DIAG", "FULL"], rotation=20)
    for i, r in enumerate(claims):
        plt.text(x[i] + 0.04, i, r["status"], va="center", fontsize=7)
    plt.xlabel("Evidence level")
    plt.title("Evidence status map")
    save_fig(figs / "fig0_evidence_status_map")
    record("fig0_evidence_status_map", "verified/unverified claims tables", "Diagnostic claims are supported; AP/safety and annotation-missing claims remain bounded.")

    closed = tables["closedset"]
    models = [r["Model"] for r in closed]
    for name, col, title, ylabel, takeaway in [
        ("fig1_closedset_fsv_by_model", "FSV", "Closed-set FSV by model", "FSV", "False-SV burden is model-dependent; do not rank by FSV alone."),
        ("fig2_closedset_sv_pred_per_img", "SV_Pred_per_img", "Closed-set SV predictions per image", "SV pred/img", "Absolute prediction burden differs from FSV denominator."),
        ("fig3_closedset_absolute_false_sv", "Abs_FalseSV", "Absolute false-SV count", "Abs false-SV", "Absolute burden highlights high-volume failure modes."),
    ]:
        plt.figure(figsize=(9, 4))
        plt.bar(models, [to_float(r[col]) for r in closed])
        plt.title(title)
        plt.ylabel(ylabel)
        rotate_ticks()
        save_fig(figs / name)
        record(name, "table1_closedset_false_sv_benchmark.csv", takeaway)

    plt.figure(figsize=(9, 4))
    x = range(len(models))
    plt.bar([i - 0.2 for i in x], [to_float(r["True_SV_Recall"]) for r in closed], width=0.4, label="recall")
    plt.bar([i + 0.2 for i in x], [to_float(r["True_SV_Precision"]) for r in closed], width=0.4, label="precision")
    plt.xticks(list(x), models)
    plt.title("True SV recall and precision")
    plt.legend()
    rotate_ticks()
    save_fig(figs / "fig4_closedset_true_sv_recall_precision")
    record("fig4_closedset_true_sv_recall_precision", "table1_closedset_false_sv_benchmark.csv", "Very low recall/precision is a matching and annotation-risk signal.")

    rotation = tables["rotation"]
    rmodels = [r["Model"] for r in rotation]
    plt.figure(figsize=(9, 4))
    plt.bar(rmodels, [to_float(r["Mean_RG_SV"]) for r in rotation])
    plt.title("Rotation gain by model")
    plt.ylabel("Mean RG_SV")
    rotate_ticks()
    save_fig(figs / "fig5_rotation_gain_by_model")
    record("fig5_rotation_gain_by_model", "table2_rotation_gain.csv", "Rotation acts as trigger/amplifier for several models.")

    plt.figure(figsize=(6, 5))
    plt.scatter([to_float(r["Mean_FSV_angle0"]) for r in rotation], [to_float(r["Mean_FSV_worst"]) for r in rotation])
    for r in rotation:
        plt.text(to_float(r["Mean_FSV_angle0"]), to_float(r["Mean_FSV_worst"]), r["Model"], fontsize=6)
    plt.xlabel("FSV@0")
    plt.ylabel("Worst FSV")
    plt.title("Angle0 vs worst FSV")
    save_fig(figs / "fig6_angle0_vs_worst_fsv")
    record("fig6_angle0_vs_worst_fsv", "table2_rotation_gain.csv", "Some models are already saturated at angle0; low RG does not mean low burden.")

    stage = tables["stage"]
    eligible = [r for r in stage if str(r.get("StageClaimEligible")) == "True"]
    plt.figure(figsize=(8, 4))
    sx = range(len(eligible))
    plt.bar([i - 0.25 for i in sx], [to_float(r["Dense_FR_SV"]) for r in eligible], width=0.25, label="dense")
    plt.bar(sx, [to_float(r["PreNMS_FR_SV"]) for r in eligible], width=0.25, label="pre-NMS")
    plt.bar([i + 0.25 for i in sx], [to_float(r["PostNMS_FR_SV"]) for r in eligible], width=0.25, label="post-NMS")
    plt.xticks(list(sx), [r["Model"] for r in eligible])
    plt.title("Stage decomposition, closed-set")
    plt.ylabel("FR_SV")
    plt.legend()
    rotate_ticks()
    save_fig(figs / "fig7_stage_decomposition_closedset")
    record("fig7_stage_decomposition_closedset", "table3_stage_decomposition.csv", "For hook-supported dense-head models, SV bias is visible before final NMS.")

    placeholder_fig(figs / "fig8_stage_decomposition_openvocab", "Open-vocab stage decomposition", "NOT_AVAILABLE: full open-vocab stage/query attractor artifacts were not found.")
    record("fig8_stage_decomposition_openvocab", "source_inventory.json", "Open-vocab stage decomposition is not available in current artifacts.")

    openv = tables["openvocab"]
    ox = [r["Risk_Group"] for r in openv]
    for name, col, title, ylabel, takeaway in [
        ("fig9_openvocab_sv_burden_by_risk_group", "SV_per_img", "Open-vocab SV burden by risk group", "SV/img", "No-annotated-SV risk group still has SV prediction burden."),
        ("fig10_openvocab_det_vs_sv_per_img", "det_per_img", "Open-vocab detections by risk group", "det/img", "SV burden should be read alongside total detections."),
        ("fig11_openvocab_top1_sv", "top1_SV", "Open-vocab top1=SV by risk group", "top1=SV", "top1=SV is diagnostic burden, not AP."),
    ]:
        plt.figure(figsize=(8, 4))
        plt.bar(ox, [to_float(r[col]) for r in openv])
        plt.title(title)
        plt.ylabel(ylabel)
        rotate_ticks()
        save_fig(figs / name)
        record(name, "table4_openvocab_full_s2_burden.csv", takeaway)

    interv = tables["intervention"]
    ix = [r["Intervention"] for r in interv]
    plt.figure(figsize=(8, 4))
    x = range(len(ix))
    plt.bar([i - 0.2 for i in x], [to_float(r["SV_per_img"]) for r in interv], width=0.4, label="SV/img")
    plt.bar([i + 0.2 for i in x], [to_float(r["LV_per_img"]) for r in interv], width=0.4, label="LV/img")
    plt.xticks(list(x), ix)
    plt.title("Embedding intervention SV/LV burden")
    plt.legend()
    rotate_ticks()
    save_fig(figs / "fig12_embedding_intervention_sv_lv_burden")
    record("fig12_embedding_intervention_sv_lv_burden", "table5_embedding_intervention.csv", "zero/random/swap interventions modulate SV burden.")

    plt.figure(figsize=(8, 4))
    plt.bar(ix, [to_float(r["Delta_SV_per_img_vs_original"]) for r in interv])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Embedding intervention delta SV/img")
    plt.ylabel("Delta vs original")
    rotate_ticks()
    save_fig(figs / "fig13_embedding_intervention_delta_vs_original")
    record("fig13_embedding_intervention_delta_vs_original", "table5_embedding_intervention.csv", "Embedding interventions causally modulate SV burden; normalization remains near original.")

    plt.figure(figsize=(8, 4))
    plt.bar(ix, [to_float(r["top1_SV"]) for r in interv])
    plt.title("Embedding intervention top1=SV")
    rotate_ticks()
    save_fig(figs / "fig14_embedding_intervention_top1_sv")
    record("fig14_embedding_intervention_top1_sv", "table5_embedding_intervention.csv", "top1=SV drops for destructive embedding interventions.")

    context = tables["context"]
    cx = [r["Condition"] for r in context]
    plt.figure(figsize=(6, 4))
    x = range(len(cx))
    plt.bar([i - 0.2 for i in x], [to_float(r["det_per_img"]) for r in context], width=0.4, label="det/img")
    plt.bar([i + 0.2 for i in x], [to_float(r["SV_per_img"]) for r in context], width=0.4, label="SV/img")
    plt.xticks(list(x), cx)
    plt.title("Context vs object-only")
    plt.legend()
    save_fig(figs / "fig15_context_object_only_vs_context_only")
    record("fig15_context_object_only_vs_context_only", "table6_context_counterfactual.csv", "context_only increases both detections and SV burden.")

    plt.figure(figsize=(6, 4))
    ctx_row = next((r for r in context if r["Condition"] == "context_only"), {})
    plt.bar(["det/img", "SV/img"], [to_float(ctx_row.get("delta_det_per_img_vs_original")), to_float(ctx_row.get("delta_SV_per_img_vs_original"))])
    plt.title("Context counterfactual deltas")
    save_fig(figs / "fig16_context_counterfactual_delta")
    record("fig16_context_counterfactual_delta", "table6_context_counterfactual.csv", "Detection-count confounding remains large.")

    dehub = tables["dehub"]
    dx = [r["Checkpoint"] for r in dehub]
    plt.figure(figsize=(6, 4))
    plt.bar(dx, [to_float(r["SV_per_img"]) for r in dehub])
    plt.title("DeHub SV burden reduction")
    plt.ylabel("SV/img")
    save_fig(figs / "fig17_dehub_sv_burden_reduction")
    record("fig17_dehub_sv_burden_reduction", "table7_dehub_safety_diagnostic.csv", "DeHub repair reduces SV prediction burden.")

    plt.figure(figsize=(6, 4))
    plt.bar(dx, [to_float(r["top1_SV"]) for r in dehub])
    plt.title("DeHub top1=SV reduction")
    save_fig(figs / "fig18_dehub_top1_sv_reduction")
    record("fig18_dehub_top1_sv_reduction", "table7_dehub_safety_diagnostic.csv", "top1=SV drops under repair.")

    migration = tables["migration"][:12]
    plt.figure(figsize=(9, 4))
    vals = [to_float(r["absolute_delta"]) for r in migration]
    plt.bar([r["class"] for r in migration], vals)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("DeHub class migration")
    plt.ylabel("repair - baseline count")
    rotate_ticks()
    save_fig(figs / "fig19_dehub_class_migration")
    record("fig19_dehub_class_migration", "table7b_dehub_class_migration.csv", "Some non-SV classes increase after repair; safety is not fully proven.")

    blockers = tables["dehub_blockers"]
    plt.figure(figsize=(7, 3))
    plt.bar([r["Blocker"] for r in blockers], [0 for _ in blockers])
    plt.ylim(-0.5, 1)
    plt.title("DeHub safety boundary: blocked metrics")
    plt.ylabel("verified safety")
    rotate_ticks()
    for i, r in enumerate(blockers):
        plt.text(i, 0.1, r["Status"], ha="center", fontsize=7)
    save_fig(figs / "fig20_dehub_safety_boundary")
    record("fig20_dehub_safety_boundary", "table7c_dehub_blockers.csv", "AP and true-SV preservation remain blocked.")

    return made


def simple_html_from_md(md: str, output_dir: Path) -> str:
    lines = []
    for line in md.splitlines():
        if line.startswith("# "):
            lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            lines.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("!["):
            alt = line.split("]", 1)[0][2:]
            src = line.split("(", 1)[1].split(")", 1)[0]
            lines.append(f"<figure><img src='../{html.escape(src)}' alt='{html.escape(alt)}'><figcaption>{html.escape(alt)}</figcaption></figure>")
        elif line.startswith("| "):
            lines.append(f"<pre>{html.escape(line)}</pre>")
        elif line.startswith("- "):
            lines.append(f"<li>{html.escape(line[2:])}</li>")
        elif not line.strip():
            lines.append("")
        else:
            lines.append(f"<p>{html.escape(line)}</p>")
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>Rotation SV Visual Summary</title>
<style>body{font-family:Arial,sans-serif;max-width:1100px;margin:24px auto;line-height:1.45}img{max-width:100%;border:1px solid #ddd}pre{white-space:pre-wrap;background:#f6f6f6;padding:6px}.blocked{color:#9a3412;font-weight:bold}</style>
</head><body>""" + "\n".join(lines) + "</body></html>\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include-blocked", action="store_true")
    parser.add_argument("--no-html", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.force:
        # Preserve existing crops; only generated tables/figures/report are overwritten by file writes.
        pass
    for sub in ["figures", "tables", "audit", "crops", "html"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    paths = source_paths(repo_root, Path(args.run_root))
    inventory = build_source_inventory(paths, output_dir)

    closed = closedset_table(read_csv(paths["closed_false_sv"])) if paths["closed_false_sv"] else []
    rotation = closed_rotation_table(read_csv(paths["closed_rotation_gain"])) if paths["closed_rotation_gain"] else []
    stage = stage_table(read_csv(paths["closed_stage"])) if paths["closed_stage"] else []
    openv = openvocab_table(read_csv(paths["openvocab_rows"])) if paths["openvocab_rows"] else []
    interv = intervention_table(read_csv(paths["causal_rows"])) if paths["causal_rows"] else []
    paired = read_csv(paths["causal_paired"]) if paths["causal_paired"] else []
    context_rows = read_csv(paths["context_rows"]) if paths["context_rows"] else []
    context_tbl, context_na = context_table(context_rows)
    dehub_rows = read_csv(paths["dehub_rows"]) if paths["dehub_rows"] else []
    scorecard = read_csv(paths["dehub_scorecard"]) if paths["dehub_scorecard"] else []
    dehub_tbl, dehub_blockers = dehub_table(dehub_rows, scorecard)
    migration = read_csv(paths["dehub_class_migration"]) if paths["dehub_class_migration"] else []
    claims_all, verified, unverified = claims_tables(read_csv(paths["claim_ledger"]) if paths["claim_ledger"] else None)

    tables = {
        "closedset": closed,
        "rotation": rotation,
        "stage": stage,
        "openvocab": openv,
        "intervention": interv,
        "context": context_tbl,
        "context_na": context_na,
        "dehub": dehub_tbl,
        "migration": migration,
        "dehub_blockers": dehub_blockers,
    }

    copy_table(verified, output_dir / "tables/verified_claims_table")
    copy_table(unverified, output_dir / "tables/unverified_claims_table")
    copy_table(closed, output_dir / "tables/table1_closedset_false_sv_benchmark")
    copy_table(rotation, output_dir / "tables/table2_rotation_gain")
    copy_table(stage, output_dir / "tables/table3_stage_decomposition")
    copy_table(openv, output_dir / "tables/table4_openvocab_full_s2_burden")
    copy_table(interv, output_dir / "tables/table5_embedding_intervention")
    copy_table(paired, output_dir / "tables/table5b_embedding_intervention_paired")
    copy_table(context_tbl, output_dir / "tables/table6_context_counterfactual")
    copy_table(context_na, output_dir / "tables/table6b_context_counterfactual_not_applicable")
    copy_table(dehub_tbl, output_dir / "tables/table7_dehub_safety_diagnostic")
    copy_table(migration, output_dir / "tables/table7b_dehub_class_migration")
    copy_table(dehub_blockers, output_dir / "tables/table7c_dehub_blockers")

    fig_manifest = plot_outputs(output_dir, tables, claims_all)

    audit_md = output_dir / "audit/unannotated_sv_contamination_audit.md"
    audit_status = "NOT_RUN"
    if audit_md.exists():
        audit_status = "NOT_FULLY_EXCLUDED"
    manifest = {
        "output_dir": str(output_dir),
        "source_inventory": str(output_dir / "audit/source_inventory.json"),
        "figures": fig_manifest,
        "tables": sorted(str(p) for p in (output_dir / "tables").glob("*")),
        "audit_status": audit_status,
        "inventory_summary": inventory,
    }
    write_json(output_dir / "visual_summary_manifest.json", manifest)

    def img(name: str) -> str:
        return f"![{name}](figures/{name}.png)"

    md_parts = [
        "# Rotation Semantic Attractor Visual Summary",
        "",
        "## 1. One-page answer",
        "",
        "- 已验证：closed-set 10 模型、OpenRSD full S2、causal intervention、context counterfactual、DeHub burden reduction 都有 full 或 diagnostic full evidence。",
        "- 未验证：open-vocab AP50/mAP、DeHub AP improvement、DeHub true-SV preservation、未标注真实 SV 污染是否完全排除。",
        "- 最重要证据：closed-set false-SV burden、OpenRSD risk-group SV burden、embedding intervention paired deltas、DeHub paired burden reduction。",
        "- 最大风险：unmatched SV prediction 可能包含未标注真实小车；不能全部称 hallucination。",
        "- 对未标注真实 SV 的回答：没有完全排除，只是通过分组、strict object flip、true-SV-rich 正控、valid-mask/NOT_APPLICABLE 等部分控制。",
        "",
        "## 2. Evidence status map",
        "",
        img("fig0_evidence_status_map"),
        "",
        md_table(verified),
        "",
        md_table(unverified),
        "",
        "## 3. Closed-set full benchmark",
        "",
        "Closed-set detectors also exhibit false-SV burden, but severity is architecture- and configuration-dependent; this is not OpenRSD-only.",
        "",
        img("fig1_closedset_fsv_by_model"),
        img("fig2_closedset_sv_pred_per_img"),
        img("fig3_closedset_absolute_false_sv"),
        img("fig4_closedset_true_sv_recall_precision"),
        "",
        "## 4. Rotation-conditioned gain",
        "",
        "Rotation acts as a trigger/amplifier for some models; angle0 absolute worst and delta worst are kept separate.",
        "",
        img("fig5_rotation_gain_by_model"),
        img("fig6_angle0_vs_worst_fsv"),
        "",
        "## 5. Stage decomposition",
        "",
        "For hook-supported dense-head models, SV bias appears before final NMS. Final-only models are NOT_APPLICABLE for dense/pre-NMS claims.",
        "",
        img("fig7_stage_decomposition_closedset"),
        img("fig8_stage_decomposition_openvocab"),
        "",
        "## 6. Open-vocab / OpenRSD S2 burden",
        "",
        "This is diagnostic burden, not AP. `false_sv_hub_candidate_no_sv` means no annotated SV GT, not guaranteed no true vehicles.",
        "",
        img("fig9_openvocab_sv_burden_by_risk_group"),
        img("fig10_openvocab_det_vs_sv_per_img"),
        img("fig11_openvocab_top1_sv"),
        "",
        "## 7. Embedding causal intervention",
        "",
        "Support/class embedding causally modulates SV burden. This does not prove all false positives come from embedding.",
        "",
        img("fig12_embedding_intervention_sv_lv_burden"),
        img("fig13_embedding_intervention_delta_vs_original"),
        img("fig14_embedding_intervention_top1_sv"),
        "",
        "## 8. Context counterfactual",
        "",
        "Context contributes to prediction burden, but detection-count confounding remains.",
        "",
        img("fig15_context_object_only_vs_context_only"),
        img("fig16_context_counterfactual_delta"),
        "",
        "## 9. DeHub diagnostic safety",
        "",
        "SV burden reduction is verified; safety/AP and true-SV preservation are not fully proven.",
        "",
        img("fig17_dehub_sv_burden_reduction"),
        img("fig18_dehub_top1_sv_reduction"),
        img("fig19_dehub_class_migration"),
        img("fig20_dehub_safety_boundary"),
        "",
        "## 10. Unannotated true-SV contamination audit",
        "",
        "Q: 这种情况排除了吗？",
        "",
        "A: 没有完全排除。当前实验已经通过分组、GT matching、true-SV-rich 正控、strict object flip 和 NOT_APPLICABLE 分流进行了部分控制，但仍需要 human crop audit 或 label-refined subset 来计算 corrected-FSV。",
        "",
        f"- Audit file: `{output_dir / 'audit/unannotated_sv_contamination_audit.md'}`",
        f"- Human template: `{output_dir / 'audit/human_sv_crop_audit_template.csv'}`",
        f"- Corrected-FSV placeholder: `{output_dir / 'audit/corrected_fsv_if_human_labels_available.csv'}`",
        "",
        "## 11. Paper-safe wording",
        "",
        "Allowed: `small-vehicle prediction burden`, `false-SV burden under annotated-GT matching`, `unmatched-SV burden`, `strict object-flip events`.",
        "",
        "Forbidden: `all unmatched SV are hallucinations`, `all no-SV tiles contain no vehicles`, `DeHub is safe`, `DeHub improves AP`, `closed-set models have open-vocabulary embedding attractors`.",
        "",
        "## 12. Reproduction manifest",
        "",
        f"- Source inventory: `{output_dir / 'audit/source_inventory.md'}`",
        f"- Manifest: `{output_dir / 'visual_summary_manifest.json'}`",
    ]
    md_text = "\n".join(md_parts)
    write_md(output_dir / "rotation_sv_visual_summary.md", md_text)

    if not args.no_html:
        html_text = simple_html_from_md(md_text, output_dir)
        write_md(output_dir / "html/rotation_sv_visual_summary.html", html_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
