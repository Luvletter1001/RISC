#!/usr/bin/env python3
"""Ingest the completed FOCUS-OVD full DOTA2-only run into P0 effect tables."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from focus_attribution_common import (
    DEFAULT_EXP_DIR,
    append_manifest_record,
    ensure_exp_tree,
    markdown_table,
    read_csv_rows,
    write_csv_rows,
    write_json,
)


FIELDS = [
    "variant_id", "variant_status", "corrected_FSV", "dense_sv_ratio",
    "true_SV_recall_proxy", "mAP_AP_proxy", "SV_AP_proxy", "det/img",
    "migration_mass_ratio", "degenerate_large_sv_ratio",
    "support_geometry_status", "primary_metrics", "safety_metrics",
    "module_effect_verdict", "failure_reason",
]

DEFAULT_FOCUS_LOG = Path(
    "resultmd/exp_focus_ovd_20260608/train_full_gpu69_20260609/full_train_tmux.log")
DEFAULT_FOCUS_SUMMARY = Path(
    "resultmd/exp_focus_ovd_20260608/train_full_gpu69_20260609/"
    "fres_focus_ovd_dota2_recovery_summary.md")
DEFAULT_FOCUS_AUDIT = Path(
    "resultmd/exp_focus_ovd_20260608/train_full_gpu69_20260609/"
    "focus_trainable_audit_dota2_recovery.json")
DEFAULT_BASELINE_VAL = Path(
    "resultmd/exp_declip_ccl_bstage_20260607/"
    "analysis_declip_ccl_vs_baseline/tables/val_summary.csv")
DEFAULT_BASELINE_CLASS = Path(
    "resultmd/exp_declip_ccl_bstage_20260607/"
    "analysis_declip_ccl_vs_baseline/tables/per_class_metrics.csv")
DEFAULT_FOCUS_CONFIG = Path(
    "M_configs/experiments/focus_ovd/"
    "focus_ovd_a10_sv_only_dota2_recovery_full.py")
DEFAULT_FOCUS_CKPT = Path(
    "work_dirs/focus_ovd_a10_sv_only_dota2_recovery_full_gpu69_20260609/"
    "epoch_24.pth")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def read_baseline(val_csv: Path, class_csv: Path) -> dict[str, Any]:
    val_rows = []
    with val_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("run") == "baseline":
                row = dict(row)
                row["epoch"] = int(row["epoch"])
                row["mAP"] = float(row["mAP"])
                row["AP50"] = float(row["AP50"])
                val_rows.append(row)
    if not val_rows:
        raise ValueError(f"no baseline rows in {val_csv}")
    final = max(val_rows, key=lambda row: row["epoch"])
    best = max(val_rows, key=lambda row: row["mAP"])

    class_rows: dict[tuple[int, str], dict[str, Any]] = {}
    with class_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("run") != "baseline":
                continue
            key = (int(row["epoch"]), row["class"])
            class_rows[key] = {
                "ap": float(row["ap"]),
                "recall": float(row["recall"]),
                "num_dets": int(row["num_dets"]),
                "num_gts": int(row["num_gts"]),
            }

    final_sv = class_rows[(final["epoch"], "small-vehicle")]
    best_sv = class_rows[(best["epoch"], "small-vehicle")]
    final_lv = class_rows[(final["epoch"], "large-vehicle")]
    best_lv = class_rows[(best["epoch"], "large-vehicle")]
    final_ship = class_rows[(final["epoch"], "ship")]
    best_ship = class_rows[(best["epoch"], "ship")]
    return {
        "final": {**final, "small_vehicle": final_sv,
                  "large_vehicle": final_lv, "ship": final_ship},
        "best": {**best, "small_vehicle": best_sv,
                 "large_vehicle": best_lv, "ship": best_ship},
        "val_csv": str(val_csv),
        "per_class_csv": str(class_csv),
    }


def parse_focus_log(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        r"Epoch\(val\) \[(?P<epoch>\d+)\]\[(?P<done>\d+)/(?P<total>\d+)\]"
        r"\s+dota/mAP: (?P<map>[0-9.]+)\s+dota/AP50: (?P<ap50>[0-9.]+)"
        r"\s+dota/IoU_50_Detail: (?P<detail>\{.*?\})\s+data_time:",
        re.DOTALL)
    matches = list(pattern.finditer(text))
    if not matches:
        raise ValueError(f"no validation metrics found in {log_path}")
    match = matches[-1]
    detail = ast.literal_eval(match.group("detail"))
    return {
        "epoch": int(match.group("epoch")),
        "val_images": int(match.group("total")),
        "mAP": float(match.group("map")),
        "AP50": float(match.group("ap50")),
        "detail": detail,
        "log_path": str(log_path),
    }


def round4(value: float) -> float:
    return round(float(value), 4)


def sv_det_per_img(num_dets: int, val_images: int) -> float:
    return round(num_dets / val_images, 4)


def build_rows(existing_rows: list[dict[str, str]], focus: dict[str, Any],
               baseline: dict[str, Any], *, focus_config: Path,
               focus_ckpt: Path, focus_summary: Path,
               audit_path: Path) -> list[dict[str, str]]:
    audit = read_json(audit_path)
    rows_by_id = {row["variant_id"]: dict(row) for row in existing_rows}
    focus_sv = focus["detail"]["small-vehicle"]
    focus_lv = focus["detail"]["large-vehicle"]
    focus_ship = focus["detail"]["ship"]
    base_final = baseline["final"]
    base_best = baseline["best"]
    base_final_sv = base_final["small_vehicle"]
    base_best_sv = base_best["small_vehicle"]
    val_images = int(focus["val_images"])
    baseline_sv_det_per_image = sv_det_per_img(base_final_sv["num_dets"], val_images)
    focus_sv_det_per_image = sv_det_per_img(focus_sv["num_dets"], val_images)
    recall_delta = round4(focus_sv["recall"] - base_final_sv["recall"])
    det_delta = int(focus_sv["num_dets"] - base_final_sv["num_dets"])

    baseline_primary = {
        "baseline_final_epoch": base_final["epoch"],
        "baseline_final_mAP": base_final["mAP"],
        "baseline_final_AP50": base_final["AP50"],
        "baseline_final_SV_AP": base_final_sv["ap"],
        "baseline_final_SV_recall": base_final_sv["recall"],
        "baseline_final_SV_dets": base_final_sv["num_dets"],
        "baseline_best_epoch": base_best["epoch"],
        "baseline_best_mAP": base_best["mAP"],
        "baseline_best_AP50": base_best["AP50"],
        "baseline_best_SV_AP": base_best_sv["ap"],
        "baseline_best_SV_recall": base_best_sv["recall"],
        "baseline_best_SV_dets": base_best_sv["num_dets"],
        "source_val_csv": baseline["val_csv"],
        "source_per_class_csv": baseline["per_class_csv"],
    }
    v00 = {key: "NOT_EVALUATED" for key in FIELDS}
    v00.update({
        "variant_id": "V00_baseline",
        "variant_status": "DONE_EXISTING_DOTA2_BASELINE_REFERENCE",
        "true_SV_recall_proxy": str(base_final_sv["recall"]),
        "mAP_AP_proxy": str(base_final["mAP"]),
        "SV_AP_proxy": str(base_final_sv["ap"]),
        "det/img": f"sv={baseline_sv_det_per_image}",
        "support_geometry_status": "NATIVE_BASELINE_REFERENCE",
        "primary_metrics": json.dumps(baseline_primary, sort_keys=True),
        "safety_metrics": json.dumps({
            "baseline_final_SV_det_per_img": baseline_sv_det_per_image,
            "baseline_final_SV_dets": base_final_sv["num_dets"],
            "baseline_final_SV_recall": base_final_sv["recall"],
        }, sort_keys=True),
        "module_effect_verdict": "REFERENCE",
        "failure_reason": "historical DOTA2-only baseline reference; not rerun in this ingest pass",
    })

    primary = {
        "focus_epoch": focus["epoch"],
        "focus_mAP": focus["mAP"],
        "focus_AP50": focus["AP50"],
        "focus_SV_AP": focus_sv["ap"],
        "focus_SV_recall": focus_sv["recall"],
        "focus_SV_dets": focus_sv["num_dets"],
        "focus_large_vehicle_AP": focus_lv["ap"],
        "focus_ship_AP": focus_ship["ap"],
        "delta_mAP_vs_baseline_ep36": round4(focus["mAP"] - base_final["mAP"]),
        "delta_AP50_vs_baseline_ep36": round4(focus["AP50"] - base_final["AP50"]),
        "delta_SV_AP_vs_baseline_ep36": round4(focus_sv["ap"] - base_final_sv["ap"]),
        "delta_mAP_vs_baseline_best_ep12": round4(focus["mAP"] - base_best["mAP"]),
        "delta_AP50_vs_baseline_best_ep12": round4(focus["AP50"] - base_best["AP50"]),
        "delta_SV_AP_vs_baseline_best_ep12": round4(focus_sv["ap"] - base_best_sv["ap"]),
        "config": str(focus_config),
        "checkpoint": str(focus_ckpt),
        "summary": str(focus_summary),
        "log": focus["log_path"],
        "trainable_audit": str(audit_path),
        "trainable_audit_ok": bool(audit.get("ok")),
        "scope": "DOTA2-only recovery full; not A10 full-mix",
    }
    safety = {
        "corrected_FSV_status": "not_evaluated",
        "dense_sv_ratio_status": "not_evaluated",
        "migration_status": "not_evaluated",
        "focus_SV_recall": focus_sv["recall"],
        "baseline_ep36_SV_recall": base_final_sv["recall"],
        "delta_SV_recall_vs_baseline_ep36": recall_delta,
        "focus_SV_dets": focus_sv["num_dets"],
        "baseline_ep36_SV_dets": base_final_sv["num_dets"],
        "delta_SV_dets_vs_baseline_ep36": det_delta,
        "focus_SV_det_per_img": focus_sv_det_per_image,
        "baseline_ep36_SV_det_per_img": baseline_sv_det_per_image,
        "interpretation": "AP improves, but small-vehicle recall and detections drop; requires false-positive/false-negative safety audit.",
    }
    v11 = {key: "NOT_EVALUATED" for key in FIELDS}
    v11.update({
        "variant_id": "V11_orientation_adapter_sv_only",
        "variant_status": "DONE_FULL_DOTA2_ONLY_AP_POSITIVE_TRUE_SV_DAMAGE_RISK",
        "corrected_FSV": "NOT_EVALUATED",
        "dense_sv_ratio": "NOT_EVALUATED",
        "true_SV_recall_proxy": str(focus_sv["recall"]),
        "mAP_AP_proxy": str(focus["mAP"]),
        "SV_AP_proxy": str(focus_sv["ap"]),
        "det/img": f"sv={focus_sv_det_per_image}",
        "migration_mass_ratio": "NOT_EVALUATED",
        "degenerate_large_sv_ratio": "NOT_EVALUATED",
        "support_geometry_status": "TRAINABLE_AUDIT_PASS_ADAPTER_ONLY"
        if audit.get("ok") else "TRAINABLE_AUDIT_MISSING_OR_FAIL",
        "primary_metrics": json.dumps(primary, sort_keys=True),
        "safety_metrics": json.dumps(safety, sort_keys=True),
        "module_effect_verdict": "EFFECTIVE_AP_WITH_RECALL_RISK",
        "failure_reason": (
            "corrected-FSV, dense SV ratio, class migration, and support geometry "
            "safety metrics were not evaluated; run is DOTA2-only recovery, not A10 full-mix"
        ),
    })

    rows_by_id["V00_baseline"] = v00
    rows_by_id["V11_orientation_adapter_sv_only"] = v11
    ordered = []
    seen = set()
    for row in existing_rows:
        vid = row["variant_id"]
        ordered.append(rows_by_id[vid])
        seen.add(vid)
    for vid in ["V00_baseline", "V11_orientation_adapter_sv_only"]:
        if vid not in seen:
            ordered.append(rows_by_id[vid])
    return ordered


def write_full_report(exp_dir: Path, rows: list[dict[str, str]]) -> None:
    focus_row = next(row for row in rows
                     if row["variant_id"] == "V11_orientation_adapter_sv_only")
    baseline_row = next(row for row in rows if row["variant_id"] == "V00_baseline")
    baseline_metrics = json.loads(baseline_row["primary_metrics"])
    primary = json.loads(focus_row["primary_metrics"])
    safety = json.loads(focus_row["safety_metrics"])
    report_rows = [
        {
            "comparison": "FOCUS ep24 vs baseline ep36",
            "mAP_delta": primary["delta_mAP_vs_baseline_ep36"],
            "AP50_delta": primary["delta_AP50_vs_baseline_ep36"],
            "SV_AP_delta": primary["delta_SV_AP_vs_baseline_ep36"],
            "SV_recall_delta": safety["delta_SV_recall_vs_baseline_ep36"],
            "SV_det_delta": safety["delta_SV_dets_vs_baseline_ep36"],
        },
        {
            "comparison": "FOCUS ep24 vs baseline best ep12",
            "mAP_delta": primary["delta_mAP_vs_baseline_best_ep12"],
            "AP50_delta": primary["delta_AP50_vs_baseline_best_ep12"],
            "SV_AP_delta": primary["delta_SV_AP_vs_baseline_best_ep12"],
            "SV_recall_delta": "not_compared",
            "SV_det_delta": "not_compared",
        },
    ]
    out_dir = exp_dir / "eval_full"
    safety_input_rows = [
        {
            "variant_id": "V00_baseline",
            "mAP50": baseline_metrics["baseline_final_mAP"],
            "SV_AP50": baseline_metrics["baseline_final_SV_AP"],
            "true_SV_recall": baseline_metrics["baseline_final_SV_recall"],
            "corrected_FSV": "",
            "migration_mass_ratio": "",
            "det/img": safety["baseline_ep36_SV_det_per_img"],
            "degenerate_large_sv_ratio": "",
        },
        {
            "variant_id": "V11_orientation_adapter_sv_only",
            "mAP50": primary["focus_mAP"],
            "SV_AP50": primary["focus_SV_AP"],
            "true_SV_recall": primary["focus_SV_recall"],
            "corrected_FSV": "",
            "migration_mass_ratio": "",
            "det/img": safety["focus_SV_det_per_img"],
            "degenerate_large_sv_ratio": "",
        },
    ]
    write_csv_rows(out_dir / "focus_full_effect_safety_gate_input.csv",
                   safety_input_rows,
                   ["variant_id", "mAP50", "SV_AP50", "true_SV_recall",
                    "corrected_FSV", "migration_mass_ratio", "det/img",
                    "degenerate_large_sv_ratio"])
    write_csv_rows(out_dir / "focus_full_effect_judgment.csv", report_rows,
                   ["comparison", "mAP_delta", "AP50_delta", "SV_AP_delta",
                    "SV_recall_delta", "SV_det_delta"])
    write_json(out_dir / "focus_full_effect_judgment.json", {
        "baseline": json.loads(baseline_row["primary_metrics"]),
        "focus": primary,
        "safety": safety,
        "verdict": focus_row["module_effect_verdict"],
        "status": focus_row["variant_status"],
    })
    md = [
        "# FOCUS-OVD Full Effect Judgment",
        "",
        "## Verdict",
        "",
        "- V11 SV-only orientation adapter is AP-positive on the completed DOTA2-only full recovery run.",
        "- It is not yet a clean safety win because small-vehicle recall and detection count drop versus baseline ep36.",
        "- Corrected-FSV, dense SV ratio, class migration, and support-geometry safety are still not evaluated.",
        "- This is DOTA2-only recovery, not the original A10 full-mix setting.",
        "",
        "## Key Metrics",
        "",
    ]
    md.extend(markdown_table(report_rows, [
        "comparison", "mAP_delta", "AP50_delta", "SV_AP_delta",
        "SV_recall_delta", "SV_det_delta"]))
    md.extend([
        "",
        "## Artifact Links",
        "",
        f"- Config: `{primary['config']}`",
        f"- Checkpoint: `{primary['checkpoint']}`",
        f"- Train/eval log: `{primary['log']}`",
        f"- Source summary: `{primary['summary']}`",
    ])
    (exp_dir / "reports/focus_full_effect_judgment.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--exp-dir", type=Path, default=DEFAULT_EXP_DIR)
    parser.add_argument("--focus-log", type=Path, default=DEFAULT_FOCUS_LOG)
    parser.add_argument("--focus-summary", type=Path, default=DEFAULT_FOCUS_SUMMARY)
    parser.add_argument("--focus-audit", type=Path, default=DEFAULT_FOCUS_AUDIT)
    parser.add_argument("--baseline-val", type=Path, default=DEFAULT_BASELINE_VAL)
    parser.add_argument("--baseline-class", type=Path, default=DEFAULT_BASELINE_CLASS)
    parser.add_argument("--focus-config", type=Path, default=DEFAULT_FOCUS_CONFIG)
    parser.add_argument("--focus-checkpoint", type=Path, default=DEFAULT_FOCUS_CKPT)
    args = parser.parse_args()

    ensure_exp_tree(args.exp_dir)
    for required in [
            args.focus_log, args.focus_summary, args.focus_audit,
            args.baseline_val, args.baseline_class, args.focus_config,
            args.focus_checkpoint]:
        if not required.exists():
            raise FileNotFoundError(required)

    eval_csv = args.exp_dir / "eval/focus_module_attribution_eval.csv"
    existing_rows = read_csv_rows(eval_csv)
    if not existing_rows:
        raise FileNotFoundError(eval_csv)
    focus = parse_focus_log(args.focus_log)
    baseline = read_baseline(args.baseline_val, args.baseline_class)
    rows = build_rows(
        existing_rows, focus, baseline, focus_config=args.focus_config,
        focus_ckpt=args.focus_checkpoint, focus_summary=args.focus_summary,
        audit_path=args.focus_audit)

    write_csv_rows(eval_csv, rows, FIELDS)
    write_json(args.exp_dir / "eval/focus_module_attribution_eval.json", {
        "rows": rows,
        "ingested_full_effect": True,
        "focus_log": str(args.focus_log),
        "baseline_val": str(args.baseline_val),
        "baseline_class": str(args.baseline_class),
    })
    md = [
        "# FOCUS-OVD Module Attribution Runner Output",
        "",
        "- Full DOTA2-only recovery metrics have been ingested for V00 and V11.",
        "- V20-V24 remain unevaluated because their loss variants are not trained/evaluated in this run.",
        "",
    ]
    md.extend(markdown_table(rows, FIELDS))
    (args.exp_dir / "eval/focus_module_attribution_eval.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    write_full_report(args.exp_dir, rows)
    append_manifest_record(
        args.exp_dir,
        repo_root=args.repo_root.resolve(),
        stage="full_effect_ingest",
        status="WRITTEN",
        module_switches={
            "variant": "V11_orientation_adapter_sv_only",
            "scope": "DOTA2-only recovery full",
            "source": str(args.focus_log),
        },
        config_path=args.focus_config,
        checkpoint_path=args.focus_checkpoint,
        gpu="6,9")
    print(json.dumps({
        "eval_csv": str(eval_csv),
        "full_report": str(args.exp_dir / "reports/focus_full_effect_judgment.md"),
        "rows": len(rows),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
