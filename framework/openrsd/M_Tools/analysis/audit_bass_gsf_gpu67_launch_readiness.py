#!/usr/bin/env python
"""Audit whether the GPU6/GPU7 BASS-GSF queue can launch cleanly.

This script is non-experimental.  It checks scripts, configs, required inputs,
expected output slots, and the latest GPU wait audit, then writes a compact
status artifact for the paper workflow.  It never launches training, inference,
or risk evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


OUT_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "bass_gsf_gpu67_launch_readiness.json")
OUT_MD = Path(
    "resultmd/exp_p4_scale_semantic_validation/"
    "faudit_20260621_bass_gsf_gpu67_launch_readiness.md")

PYTHON_ENV = Path("/data/zcy/anaconda3/envs/openrsd/bin/python")
DATASET_ANN_DIR = Path(
    "/data1/zcy/datasets/HRRSD_800_0/internal_split_20260619/val/annfiles")

EVAL_ROOT = Path("work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619")
RISK_ROOT = Path("work_dirs/gs3c_sise_problem_reframing_20260619")
GPU_WAIT_AUDIT_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "gpu67_wait_blocker_audit.json")

SCRIPT_PATHS = [
    Path("M_Tools/experiments/run_hrrsd_bass_gsf_rankdelta_train_20260620.sh"),
    Path("M_Tools/experiments/run_hrrsd_bass_gsf_p2_train_20260620.sh"),
    Path(
        "M_Tools/experiments/"
        "wait_and_run_hrrsd_bass_gsf_rankdelta_gpu67_20260620.sh"),
    Path(
        "M_Tools/experiments/"
        "wait_and_run_hrrsd_bass_gsf_rankdelta_followup_gpu67_20260620.sh"),
    Path(
        "M_Tools/experiments/"
        "wait_and_run_hrrsd_bass_gsf_p2_gpu67_20260620.sh"),
]

CONFIG_PATHS = [
    Path(
        "M_configs/Diagnostics/"
        "hrrsd_rtmdet_l_dota_init_internal_epoch3_gpu0189.py"),
    Path(
        "M_configs/Diagnostics/"
        "hrrsd_rtmdet_l_dota_init_internal_eval_gpu0189.py"),
    Path(
        "M_configs/Diagnostics/"
        "hrrsd_rtmdet_l_dota_init_internal_bass_gsf_delta_train_gpu67.py"),
    Path(
        "M_configs/Diagnostics/"
        "hrrsd_rtmdet_l_dota_init_internal_bass_gsf_rankdelta_train_gpu67.py"),
    Path(
        "M_configs/Diagnostics/"
        "hrrsd_rtmdet_l_dota_init_internal_bass_gsf_delta_eval_gpu67.py"),
    Path(
        "M_configs/Diagnostics/"
        "hrrsd_rtmdet_l_dota_init_internal_bass_gsf_p2_assign_rank_train_gpu67.py"),
    Path(
        "M_configs/Diagnostics/"
        "hrrsd_rtmdet_l_dota_init_internal_bass_gsf_p2_posterior_rank_train_gpu67.py"),
]

REQUIRED_INPUTS = [
    {
        "name": "python_env",
        "path": PYTHON_ENV,
        "reason": "launcher Python executable",
    },
    {
        "name": "hrrsd_val_ann_dir",
        "path": DATASET_ANN_DIR,
        "reason": "deployment-risk annotation directory",
    },
    {
        "name": "epoch3_checkpoint",
        "path": EVAL_ROOT / "train_epoch3/epoch_3.pth",
        "reason": "base checkpoint for RankDelta/P2 training",
    },
    {
        "name": "class_area_priors",
        "path": (
            Path("work_dirs/gs3c_dataset_inventory_20260619")
            / "hrrsd_internal_train_area_priors.csv"),
        "reason": "source-disjoint Gaussian support prior",
    },
    {
        "name": "baseline_predictions",
        "path": EVAL_ROOT / "eval_epoch3_baseline_full/predictions.pkl",
        "reason": "baseline risk comparison input",
    },
    {
        "name": "density_control_predictions",
        "path": EVAL_ROOT / "eval_epoch3_plus2_density_w005_full/predictions.pkl",
        "reason": "density control risk comparison input",
    },
    {
        "name": "nog3_control_predictions",
        "path": EVAL_ROOT / "eval_nog3_ctrl_e2/predictions.pkl",
        "reason": "strict no-G3 AP control input",
    },
]

CONTROL_RISK_INPUTS = [
    {
        "name": "density_control_risk",
        "path": (
            RISK_ROOT
            / "hrrsd_bass_gsf_delta_e2_dw0p05_t0p25_d0p50_head_vs_density_control"
            / "deployment_risk_summary.json"),
    },
    {
        "name": "nog3_control_risk",
        "path": (
            RISK_ROOT
            / "hrrsd_bass_gsf_delta_e2_dw0p05_t0p25_d0p50_head_vs_nog3_control"
            / "deployment_risk_summary.json"),
    },
    {
        "name": "bass_gsf_lite_risk",
        "path": (
            RISK_ROOT
            / "hrrsd_bass_gsf_lite_e2_b0p50_thrm8p0_d0p50"
            / "deployment_risk_summary.json"),
    },
]

EXPECTED_OUTPUTS = [
    {
        "name": "P0 rankdelta min-score 0.30",
        "eval_json_dir": (
            EVAL_ROOT
            / "eval_bass_gsf_rankdelta_e2_dw0p05_ms0p30_t0p25_d0p50_head_full"),
        "risk_json": (
            RISK_ROOT
            / "hrrsd_bass_gsf_rankdelta_e2_dw0p05_ms0p30_t0p25_d0p50_head"
            / "deployment_risk_summary.json"),
    },
    {
        "name": "P0 rankdelta min-score 0.50",
        "eval_json_dir": (
            EVAL_ROOT
            / "eval_bass_gsf_rankdelta_e2_dw0p05_ms0p50_t0p25_d0p50_head_full"),
        "risk_json": (
            RISK_ROOT
            / "hrrsd_bass_gsf_rankdelta_e2_dw0p05_ms0p50_t0p25_d0p50_head"
            / "deployment_risk_summary.json"),
    },
    {
        "name": "P1 gentle RankDelta",
        "eval_json_dir": (
            EVAL_ROOT
            / "eval_bass_gsf_rankdelta_e2_dw0p01_ms0p05_t0p10_d0p20_keep0p20_head_full"),
        "risk_json": (
            RISK_ROOT
            / "hrrsd_bass_gsf_rankdelta_e2_dw0p01_ms0p05_t0p10_d0p20_keep0p20_head"
            / "deployment_risk_summary.json"),
    },
    {
        "name": "P1 protected RankDelta",
        "eval_json_dir": (
            EVAL_ROOT
            / "eval_bass_gsf_rankdelta_e2_dw0p02_ms0p10_t0p15_d0p25_keep0p30_head_full"),
        "risk_json": (
            RISK_ROOT
            / "hrrsd_bass_gsf_rankdelta_e2_dw0p02_ms0p10_t0p15_d0p25_keep0p30_head"
            / "deployment_risk_summary.json"),
    },
    {
        "name": "P2 assign-rank BASS-GSF",
        "eval_json_dir": EVAL_ROOT / "eval_bass_gsf_p2_assign_rank_e2_head_full",
        "risk_json": (
            RISK_ROOT / "hrrsd_bass_gsf_p2_assign_rank_e2_head"
            / "deployment_risk_summary.json"),
    },
    {
        "name": "P2 posterior-rank BASS-GSF",
        "eval_json_dir": EVAL_ROOT / "eval_bass_gsf_p2_posterior_rank_e2_head_full",
        "risk_json": (
            RISK_ROOT / "hrrsd_bass_gsf_p2_posterior_rank_e2_head"
            / "deployment_risk_summary.json"),
    },
]


def _exists(path: Path) -> bool:
    return path.exists()


def path_check(name: str, path: Path, reason: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "path": str(path),
        "exists": _exists(path),
        "reason": reason,
    }


def shell_syntax_check(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "syntax_ok": False}
    proc = subprocess.run(
        ["bash", "-n", str(path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return {
        "path": str(path),
        "exists": True,
        "syntax_ok": proc.returncode == 0,
        "stderr_tail": proc.stderr[-1000:],
    }


def detect_repo_local_data_refs(paths: list[Path]) -> list[dict[str, Any]]:
    repo_local_data = re.compile(r"(^|[\s'\"=:(])data/")
    hits: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not repo_local_data.search(line):
                continue
            if "/data1/zcy/datasets/" in line:
                continue
            hits.append({
                "path": str(path),
                "line": lineno,
                "text": line.strip(),
            })
    return hits


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def eval_json_status(eval_dir: Path) -> dict[str, Any]:
    if not eval_dir.exists():
        return {
            "eval_dir": str(eval_dir),
            "eval_status": "missing",
            "eval_json": "",
        }
    candidates = sorted(eval_dir.glob("*/20*.json")) + sorted(eval_dir.glob("*.json"))
    for path in candidates:
        payload = read_json(path)
        metrics = payload.get("metrics") if isinstance(payload, dict) else None
        if isinstance(metrics, dict):
            payload = {**payload, **metrics}
        if isinstance(payload, dict) and (
            "mAP" in payload or "dota/mAP" in payload):
            return {
                "eval_dir": str(eval_dir),
                "eval_status": "done",
                "eval_json": str(path),
            }
    return {
        "eval_dir": str(eval_dir),
        "eval_status": "missing_metric_json",
        "eval_json": "",
    }


def output_slot_status(spec: dict[str, Any]) -> dict[str, Any]:
    eval_status = eval_json_status(Path(spec["eval_json_dir"]))
    risk_path = Path(spec["risk_json"])
    risk_exists = risk_path.exists()
    if eval_status["eval_status"] == "done" and risk_exists:
        status = "complete"
    elif eval_status["eval_status"] == "done":
        status = "missing_risk"
    elif risk_exists:
        status = "missing_eval"
    else:
        status = "waiting"
    return {
        "name": spec["name"],
        **eval_status,
        "risk_json": str(risk_path),
        "risk_exists": risk_exists,
        "status": status,
    }


def build_audit() -> dict[str, Any]:
    script_checks = [shell_syntax_check(path) for path in SCRIPT_PATHS]
    config_checks = [
        path_check(path.name, path, "training/evaluation config")
        for path in CONFIG_PATHS
    ]
    input_checks = [
        path_check(row["name"], Path(row["path"]), row["reason"])
        for row in REQUIRED_INPUTS
    ]
    control_risk_checks = [
        path_check(row["name"], Path(row["path"]), "existing control risk summary")
        for row in CONTROL_RISK_INPUTS
    ]
    output_slots = [output_slot_status(row) for row in EXPECTED_OUTPUTS]
    data_ref_hits = detect_repo_local_data_refs(SCRIPT_PATHS + CONFIG_PATHS)
    gpu_wait = read_json(GPU_WAIT_AUDIT_JSON)

    script_ready = all(
        row.get("exists") and row.get("syntax_ok") for row in script_checks)
    config_ready = all(row["exists"] for row in config_checks)
    inputs_ready = all(row["exists"] for row in input_checks)
    controls_ready = all(row["exists"] for row in control_risk_checks)
    data_paths_ready = not data_ref_hits
    launch_assets_ready = (
        script_ready and config_ready and inputs_ready
        and controls_ready and data_paths_ready)
    gpu_action = gpu_wait.get("action", "missing")
    if not launch_assets_ready:
        action = "FIX_LAUNCH_BLOCKERS"
    elif gpu_action == "READY_TO_LAUNCH":
        action = "READY_TO_LAUNCH"
    elif gpu_action == "WAIT_GPU_BUSY":
        action = "READY_WHEN_GPU_FREE"
    else:
        action = "READY_WITH_UNVERIFIED_GPU_STATUS"

    blockers: list[str] = []
    if not script_ready:
        blockers.append("launcher script missing or bash syntax failed")
    if not config_ready:
        blockers.append("required config missing")
    if not inputs_ready:
        blockers.append("required checkpoint, predictions, priors, env, or dataset path missing")
    if not controls_ready:
        blockers.append("control deployment-risk summary missing")
    if not data_paths_ready:
        blockers.append("repo-local data/ reference detected in launch scripts/configs")
    if gpu_action == "WAIT_GPU_BUSY":
        blockers.append("GPU6/GPU7 are still busy; waiters should continue polling")
    elif gpu_action == "missing":
        blockers.append("GPU wait audit missing; run audit_gpu67_wait_blocker.py")

    return {
        "action": action,
        "launch_assets_ready": launch_assets_ready,
        "gpu_wait_action": gpu_action,
        "gpu_wait_json": str(GPU_WAIT_AUDIT_JSON),
        "blocker_count": len(blockers),
        "blockers": blockers,
        "script_checks": script_checks,
        "config_checks": config_checks,
        "input_checks": input_checks,
        "control_risk_checks": control_risk_checks,
        "repo_local_data_refs": data_ref_hits,
        "expected_output_slots": output_slots,
        "output_slots_complete": sum(
            1 for row in output_slots if row["status"] == "complete"),
        "output_slots_total": len(output_slots),
        "no_fabrication_rule": (
            "Launch readiness only validates prerequisites and expected output "
            "slots. RankDelta/P2 paper rows may be filled only when eval JSON "
            "and deployment-risk summary JSON both exist."),
    }


def write_markdown(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# BASS-GSF GPU6/7 Launch Readiness Audit - 2026-06-21",
        "",
        "This file is generated by "
        "`M_Tools/analysis/audit_bass_gsf_gpu67_launch_readiness.py`.",
        "It is non-experimental and does not launch training or inference.",
        "",
        "| field | value |",
        "|---|---|",
        f"| action | `{audit['action']}` |",
        f"| launch_assets_ready | `{audit['launch_assets_ready']}` |",
        f"| gpu_wait_action | `{audit['gpu_wait_action']}` |",
        f"| blocker_count | `{audit['blocker_count']}` |",
        f"| output_slots_complete | `{audit['output_slots_complete']}/{audit['output_slots_total']}` |",
        "",
        "## Blockers",
        "",
    ]
    if audit["blockers"]:
        lines.extend(f"- {item}" for item in audit["blockers"])
    else:
        lines.append("- none")
    lines += [
        "",
        "## Required Inputs",
        "",
        "| name | exists | path | reason |",
        "|---|---|---|---|",
    ]
    for row in audit["input_checks"]:
        lines.append(
            f"| {row['name']} | `{row['exists']}` | `{row['path']}` | {row['reason']} |")
    lines += [
        "",
        "## Expected Output Slots",
        "",
        "| name | status | eval_json | risk_json |",
        "|---|---|---|---|",
    ]
    for row in audit["expected_output_slots"]:
        lines.append(
            f"| {row['name']} | `{row['status']}` | "
            f"`{row['eval_json'] or row['eval_dir']}` | `{row['risk_json']}` |")
    lines += [
        "",
        "## Repo-Local Data References",
        "",
    ]
    if audit["repo_local_data_refs"]:
        lines += [
            "| file | line | text |",
            "|---|---:|---|",
        ]
        for hit in audit["repo_local_data_refs"]:
            text = str(hit["text"]).replace("|", "\\|")
            lines.append(f"| `{hit['path']}` | {hit['line']} | `{text}` |")
    else:
        lines.append("No forbidden repo-local `data/` references found in launch scripts/configs.")
    lines += [
        "",
        "## No-Fabrication Rule",
        "",
        audit["no_fabrication_rule"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument("--out-md", default=str(OUT_MD))
    args = parser.parse_args()

    audit = build_audit()
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    write_markdown(Path(args.out_md), audit)
    print(json.dumps({
        "action": audit["action"],
        "launch_assets_ready": audit["launch_assets_ready"],
        "gpu_wait_action": audit["gpu_wait_action"],
        "blocker_count": audit["blocker_count"],
        "output_slots_complete": audit["output_slots_complete"],
        "output_slots_total": audit["output_slots_total"],
        "out_json": args.out_json,
        "out_md": args.out_md,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
