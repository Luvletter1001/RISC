#!/usr/bin/env python3
"""Train the three FOCUS-EQText short-run variants on the P1A realbatch path."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from focus_eqtext_common import (  # noqa: E402
    BASE_CKPT,
    BASE_CONFIG,
    EXP_DIR,
    OPENRSD_PYTHON,
    P1A_DATASET,
    SPATIAL_TARGETS,
    TRAIN_VARIANTS,
    ensure_exp_tree,
    read_json,
    resolve,
    write_csv,
    write_json,
)


def train_variant(
        repo_root: Path,
        exp_dir: Path,
        variant_id: str,
        max_iters: int) -> dict[str, Any]:
    config_json = exp_dir / "configs" / f"{variant_id}.json"
    output_dir = exp_dir / "train" / variant_id
    log_path = exp_dir / "logs" / f"{variant_id}_train.log"
    cmd = [
        "rtk",
        "env",
        "CUDA_VISIBLE_DEVICES=6,9",
        "NCCL_P2P_DISABLE=1",
        "NCCL_IB_DISABLE=1",
        "PYTHONNOUSERSITE=1",
        str(resolve(repo_root, OPENRSD_PYTHON)),
        str(repo_root / "experiments/rotation_semantic_attractor/scripts/72_focus_p1a_realbatch_one_step_smoke.py"),
        "--repo-root", str(repo_root),
        "--spatial-targets", str(resolve(repo_root, SPATIAL_TARGETS)),
        "--smoke-dataset-json", str(resolve(repo_root, P1A_DATASET)),
        "--output-dir", str(output_dir),
        "--focus-config", str(resolve(repo_root, BASE_CONFIG)),
        "--checkpoint", str(resolve(repo_root, BASE_CKPT)),
        "--device", "auto",
        "--variant-config-json", str(config_json),
        "--max-iters", str(max_iters),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "6,9"
    env["NCCL_P2P_DISABLE"] = "1"
    env["NCCL_IB_DISABLE"] = "1"
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    report = read_json(output_dir / "realbatch_smoke_report.json", {})
    row = {
        "variant_id": variant_id,
        "returncode": proc.returncode,
        "status": report.get("status", "MISSING_REPORT"),
        "actual_detector_train": report.get("actual_detector_train", False),
        "max_iters": report.get("max_iters", max_iters),
        "loss_focus_anti": report.get("loss_focus_anti", ""),
        "loss_focus_text_anchor": report.get("loss_focus_text_anchor", ""),
        "loss_focus_preserve": report.get("loss_focus_preserve", ""),
        "loss_focus_total": report.get("loss_focus_total", ""),
        "adapter_grad_flow": report.get("adapter_grad_flow", ""),
        "frozen_grad_present": report.get("frozen_grad_present", ""),
        "visual_delta_norm": report.get("support_delta_norm_ratio", ""),
        "text_delta_norm": report.get("text_delta_norm_ratio", ""),
        "dual_text_weight": report.get("dual_text_weight", ""),
        "log": str(log_path),
        "report": str(output_dir / "realbatch_smoke_report.json"),
    }
    return row


def row_from_report(exp_dir: Path, variant_id: str,
                    max_iters: int) -> dict[str, Any]:
    output_dir = exp_dir / "train" / variant_id
    report = read_json(output_dir / "realbatch_smoke_report.json", {})
    return {
        "variant_id": variant_id,
        "returncode": 0 if str(report.get("status", "")).startswith("PASS_") else 1,
        "status": report.get("status", "MISSING_REPORT"),
        "actual_detector_train": report.get("actual_detector_train", False),
        "max_iters": report.get("max_iters", max_iters),
        "loss_focus_anti": report.get("loss_focus_anti", ""),
        "loss_focus_text_anchor": report.get("loss_focus_text_anchor", ""),
        "loss_focus_preserve": report.get("loss_focus_preserve", ""),
        "loss_focus_total": report.get("loss_focus_total", ""),
        "adapter_grad_flow": report.get("adapter_grad_flow", ""),
        "frozen_grad_present": report.get("frozen_grad_present", ""),
        "visual_delta_norm": report.get("support_delta_norm_ratio", ""),
        "text_delta_norm": report.get("text_delta_norm_ratio", ""),
        "dual_text_weight": report.get("dual_text_weight", ""),
        "log": str(exp_dir / "logs" / f"{variant_id}_train.log"),
        "report": str(output_dir / "realbatch_smoke_report.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--exp-dir", type=Path, default=EXP_DIR)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=TRAIN_VARIANTS,
        default=list(TRAIN_VARIANTS))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    exp_dir = resolve(repo_root, args.exp_dir)
    ensure_exp_tree(exp_dir)
    preflight = read_json(
        exp_dir / "preflight/focus_eqtext_preflight.json", {})
    dual_eq = preflight.get("dual_zero_equivalence", {})
    if not (str(preflight.get("status", "")).startswith("PASS_")
            and dual_eq.get("status") == "PASS"
            and float(dual_eq.get("max_abs_diff", 1.0)) == 0.0):
        payload = {
            "status": "BLOCKED_BASELINE_EQUIVALENCE_REQUIRED_BEFORE_TRAIN",
            "preflight_status": preflight.get("status", "missing"),
            "dual_zero_equivalence": dual_eq,
        }
        write_json(exp_dir / "train/focus_eqtext_train_summary.json", payload)
        print(json.dumps(payload, indent=2))
        return 2

    requested = set(args.variants)
    rows = []
    for variant_id in TRAIN_VARIANTS:
        if variant_id in requested:
            rows.append(train_variant(
                repo_root, exp_dir, variant_id, args.max_iters))
        else:
            rows.append(row_from_report(exp_dir, variant_id, args.max_iters))
    all_pass = all(
        row["returncode"] == 0
        and str(row["status"]).startswith("PASS_")
        and bool(row["actual_detector_train"])
        for row in rows)
    summary = {
        "status": (
            "PASS_FOCUS_EQTEXT_TRAIN_SHORT"
            if all_pass else "FAIL_FOCUS_EQTEXT_TRAIN_SHORT"),
        "gpu_policy": "CUDA_VISIBLE_DEVICES=6,9",
        "max_iters": int(args.max_iters),
        "variants": rows,
    }
    write_csv(exp_dir / "train/focus_eqtext_train_summary.csv", rows)
    write_json(exp_dir / "train/focus_eqtext_train_summary.json", summary)
    print(json.dumps({
        "status": summary["status"],
        "variants": [{r["variant_id"]: r["status"]} for r in rows],
    }, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
