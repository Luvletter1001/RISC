#!/usr/bin/env python
"""Refresh the BASS-GSF RankDelta paper evidence package.

This script is intentionally non-experimental: it never launches training or
inference.  It only runs the existing readers/gates in the order needed after a
RankDelta job finishes, then writes a compact manifest for the paper workflow.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


OUT_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "bass_rankdelta_paper_package_refresh.json")
OUT_MD = Path(
    "resultmd/exp_p4_scale_semantic_validation/"
    "fstatus_20260620_bass_rankdelta_paper_package_refresh.md")

SUMMARY_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "bass_rankdelta_summary.json")
DECISION_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "bass_rankdelta_followup_decision.json")
CLOSURE_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "bass_rankdelta_closure_plan.json")
CONFIRMATION_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "bass_rankdelta_confirmation_plan.json")
P2_QUEUE_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "bass_rankdelta_p2_queue_plan.json")
P3A_SUMMARY_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "p3a_ap_projection_summary.json")
P3D_SUMMARY_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "p3d_transfer_summary.json")
SNIPPET_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "bass_rankdelta_paper_snippets.json")
ICLR_GATE_JSON = Path(
    "work_dirs/iclr95_evidence_gate_20260620/iclr95_evidence_gate.json")
ICLR_8OF10_PLAN_JSON = Path(
    "work_dirs/iclr95_evidence_gate_20260620/"
    "iclr95_8of10_upgrade_plan.json")
PAPER_READINESS_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "semantic_scale_paper_readiness_audit.json")
SOURCE_PACKAGE_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "semantic_scale_source_package_audit.json")
GPU_WAIT_AUDIT_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "gpu67_wait_blocker_audit.json")
LAUNCH_READINESS_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "bass_gsf_gpu67_launch_readiness.json")
WAITER_LIVENESS_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "bass_gsf_gpu67_waiter_liveness.json")
GPU01_LIVE_QUEUE_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "bass_gsf_gpu01_live_queue.json")

REFRESH_STEPS = [
    {
        "name": "summarize_rankdelta",
        "script": "M_Tools/analysis/summarize_bass_rankdelta_results.py",
    },
    {
        "name": "decide_followup",
        "script": "M_Tools/analysis/decide_bass_rankdelta_followup.py",
    },
    {
        "name": "plan_closure",
        "script": "M_Tools/analysis/plan_bass_rankdelta_closure.py",
    },
    {
        "name": "plan_confirmation",
        "script": (
            "M_Tools/analysis/plan_bass_rankdelta_confirmation_package.py"),
    },
    {
        "name": "plan_p2_queue",
        "script": "M_Tools/analysis/plan_bass_rankdelta_p2_queue.py",
    },
    {
        "name": "build_paper_snippets",
        "script": "M_Tools/analysis/build_bass_rankdelta_paper_snippets.py",
    },
    {
        "name": "summarize_p3a_ap_projection",
        "script": "M_Tools/analysis/summarize_p3a_ap_projection_results.py",
    },
    {
        "name": "summarize_p3d_transfer",
        "script": "M_Tools/analysis/summarize_p3d_transfer_results.py",
    },
    {
        "name": "build_iclr95_gate",
        "script": "M_Tools/analysis/build_iclr95_evidence_gate.py",
    },
    {
        "name": "build_iclr95_8of10_upgrade_plan",
        "script": "M_Tools/analysis/build_iclr95_8of10_upgrade_plan.py",
    },
    {
        "name": "build_paper_figures",
        "script": "M_Tools/analysis/build_semantic_scale_paper_figures.py",
    },
    {
        "name": "audit_gpu67_wait",
        "script": "M_Tools/analysis/audit_gpu67_wait_blocker.py",
    },
    {
        "name": "audit_gpu67_launch_readiness",
        "script": "M_Tools/analysis/audit_bass_gsf_gpu67_launch_readiness.py",
    },
    {
        "name": "audit_gpu67_waiter_liveness",
        "script": "M_Tools/analysis/audit_bass_gsf_gpu67_waiter_liveness.py",
    },
    {
        "name": "audit_gpu01_live_queue",
        "script": "M_Tools/analysis/audit_bass_gsf_gpu01_live_queue.py",
    },
    {
        "name": "audit_source_package",
        "script": "M_Tools/analysis/audit_semantic_scale_source_package.py",
    },
    {
        "name": "audit_paper_readiness",
        "script": "M_Tools/analysis/audit_semantic_scale_paper_readiness.py",
    },
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_command_plan(python_executable: str) -> list[list[str]]:
    return [[python_executable, step["script"]] for step in REFRESH_STEPS]


def run_refresh_steps(python_executable: str) -> list[dict[str, Any]]:
    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("MPLCONFIGDIR", "/tmp/openrsd_mplconfig")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    results = []
    for step, command in zip(REFRESH_STEPS, build_command_plan(python_executable)):
        proc = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        results.append({
            "name": step["name"],
            "script": step["script"],
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        })
        if proc.returncode != 0:
            break
    return results


def _rankdelta_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = summary.get("rows", [])
    if not isinstance(rows, list):
        return []
    return [
        row for row in rows
        if isinstance(row, dict)
        and str(row.get("kind", "")).startswith("rankdelta")
    ]


def _gpu01_route_active(gpu01_live_queue: dict[str, Any]) -> bool:
    action = gpu01_live_queue.get("action")
    return action in {
        "P0_RUNNING_GPU01_WAITERS_ACTIVE",
        "P0_DONE_GPU01_WAITERS_ACTIVE",
        "GPU01_WAITERS_ACTIVE",
    }


def build_manifest(step_results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = read_json(SUMMARY_JSON)
    decision = read_json(DECISION_JSON)
    closure = read_json(CLOSURE_JSON)
    confirmation = read_json(CONFIRMATION_JSON)
    p2_queue = read_json(P2_QUEUE_JSON)
    p3a_summary = read_json(P3A_SUMMARY_JSON)
    p3d_summary = read_json(P3D_SUMMARY_JSON)
    snippets = read_json(SNIPPET_JSON)
    gate = read_json(ICLR_GATE_JSON)
    gate_8of10 = read_json(ICLR_8OF10_PLAN_JSON)
    readiness = read_json(PAPER_READINESS_JSON)
    source_package = read_json(SOURCE_PACKAGE_JSON)
    gpu_wait = read_json(GPU_WAIT_AUDIT_JSON)
    launch_readiness = read_json(LAUNCH_READINESS_JSON)
    waiter_liveness = read_json(WAITER_LIVENESS_JSON)
    gpu01_live_queue = read_json(GPU01_LIVE_QUEUE_JSON)
    rank_rows = _rankdelta_rows(summary)
    gpu01_active = _gpu01_route_active(gpu01_live_queue)
    gpu67_raw_wait_action = gpu_wait.get("action", "missing")
    gpu67_raw_launch_action = launch_readiness.get("action", "missing")
    gpu67_raw_waiters_action = waiter_liveness.get("action", "missing")
    gpu67_effective_action = (
        "HISTORICAL_INACTIVE_GPU01_ACTIVE"
        if gpu01_active else gpu67_raw_waiters_action)
    return {
        "ok": all(row["returncode"] == 0 for row in step_results),
        "steps": step_results,
        "active_route": "gpu01" if gpu01_active else "gpu67_or_unresolved",
        "summary_status": summary.get("status", "missing"),
        "followup_action": decision.get("action", "missing"),
        "closure_action": closure.get("action", "missing"),
        "confirmation_action": confirmation.get("action", "missing"),
        "p2_queue_action": p2_queue.get("action", "missing"),
        "p3a_status": p3a_summary.get("status", "missing"),
        "p3a_main_method": p3a_summary.get("main_method", "missing"),
        "p3a_main_gate": p3a_summary.get("main_gate", "missing"),
        "p3a_strict_pass": p3a_summary.get("strict_pass", False),
        "p3d_status": p3d_summary.get("status", "missing"),
        "p3d_dataset": p3d_summary.get("dataset", "missing"),
        "p3d_main_method": p3d_summary.get("main_method", "missing"),
        "p3d_main_gate": p3d_summary.get("main_gate", "missing"),
        "p3d_strict_transfer_pass": p3d_summary.get(
            "strict_transfer_pass", False),
        "paper_position": closure.get("paper_position", ""),
        "rankdelta_rows_total": len(rank_rows),
        "rankdelta_rows_filled": snippets.get("rankdelta_rows_filled", 0),
        "has_strict_pass": snippets.get("has_strict_pass", False),
        "iclr95_all_passed": gate.get("all_95_gates_passed", False),
        "iclr95_num_passed": gate.get("num_gates_passed"),
        "iclr95_num_total": gate.get("num_gates_total"),
        "iclr95_8of10_target": gate_8of10.get("target_gate", "missing"),
        "iclr95_8of10_current": gate_8of10.get("current_gate", "missing"),
        "iclr95_8of10_additional_needed": gate_8of10.get(
            "additional_gates_needed", "missing"),
        "iclr95_8of10_forbidden_route": gate_8of10.get(
            "forbidden_route", ""),
        "diagnostic_draft_ready": readiness.get(
            "diagnostic_draft_ready", False),
        "submission_ready": readiness.get("submission_ready", False),
        "source_package_ready": source_package.get(
            "source_package_ready", False),
        "pdf_build_ready": source_package.get("pdf_build_ready", False),
        "gpu67_wait_action": (
            "HISTORICAL_INACTIVE_GPU01_ACTIVE"
            if gpu01_active else gpu67_raw_wait_action),
        "gpu67_raw_wait_action": gpu67_raw_wait_action,
        "gpu67_external_or_unknown_process_count": gpu_wait.get(
            "external_or_unknown_process_count", "missing"),
        "gpu67_launch_action": (
            "HISTORICAL_INACTIVE_GPU01_ACTIVE"
            if gpu01_active else gpu67_raw_launch_action),
        "gpu67_raw_launch_action": gpu67_raw_launch_action,
        "gpu67_launch_assets_ready": launch_readiness.get(
            "launch_assets_ready", False),
        "gpu67_launch_blocker_count": launch_readiness.get(
            "blocker_count", "missing"),
        "gpu67_launch_output_slots_complete": launch_readiness.get(
            "output_slots_complete", "missing"),
        "gpu67_launch_output_slots_total": launch_readiness.get(
            "output_slots_total", "missing"),
        "gpu67_waiters_action": gpu67_effective_action,
        "gpu67_raw_waiters_action": gpu67_raw_waiters_action,
        "gpu67_active_waiter_count": waiter_liveness.get(
            "active_waiter_count", "missing"),
        "gpu67_waiter_count": waiter_liveness.get("waiter_count", "missing"),
        "gpu67_stale_waiter_count": waiter_liveness.get(
            "stale_waiter_count", "missing"),
        "gpu01_queue_action": gpu01_live_queue.get("action", "missing"),
        "gpu01_p0_running_count": gpu01_live_queue.get(
            "p0_running_count", "missing"),
        "gpu01_p0_variant_count": gpu01_live_queue.get(
            "p0_variant_count", "missing"),
        "gpu01_p0_outputs_complete": gpu01_live_queue.get(
            "p0_outputs_complete", "missing"),
        "gpu01_p0_outputs_total": gpu01_live_queue.get(
            "p0_outputs_total", "missing"),
        "gpu01_waiter_active_count": gpu01_live_queue.get(
            "waiter_active_count", "missing"),
        "gpu01_waiter_count": gpu01_live_queue.get("waiter_count", "missing"),
        "gpu01_stale_or_missing_count": gpu01_live_queue.get(
            "stale_or_missing_count", "missing"),
        "no_fabrication_rule": (
            "This refresh chain reads eval JSON and deployment-risk summaries "
            "only; paper rows are never inferred from checkpoints or logs."),
    }


def write_manifest(manifest: dict[str, Any], out_json: Path,
                   out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    lines = [
        "# BASS-GSF RankDelta Paper Package Refresh - 2026-06-20",
        "",
        "This file is generated by "
        "`M_Tools/analysis/refresh_bass_rankdelta_paper_package.py`.",
        "It does not launch experiments or infer missing metrics.",
        "",
        "| field | value |",
        "|---|---|",
        f"| ok | `{manifest['ok']}` |",
        f"| active_route | `{manifest['active_route']}` |",
        f"| summary_status | `{manifest['summary_status']}` |",
        f"| followup_action | `{manifest['followup_action']}` |",
        f"| closure_action | `{manifest['closure_action']}` |",
        f"| confirmation_action | `{manifest['confirmation_action']}` |",
        f"| p2_queue_action | `{manifest['p2_queue_action']}` |",
        f"| p3a_status | `{manifest['p3a_status']}` |",
        f"| p3a_main_method | `{manifest['p3a_main_method']}` |",
        f"| p3a_main_gate | `{manifest['p3a_main_gate']}` |",
        f"| p3a_strict_pass | `{manifest['p3a_strict_pass']}` |",
        f"| p3d_status | `{manifest['p3d_status']}` |",
        f"| p3d_dataset | `{manifest['p3d_dataset']}` |",
        f"| p3d_main_method | `{manifest['p3d_main_method']}` |",
        f"| p3d_main_gate | `{manifest['p3d_main_gate']}` |",
        f"| p3d_strict_transfer_pass | `{manifest['p3d_strict_transfer_pass']}` |",
        f"| paper_position | {manifest['paper_position']} |",
        f"| rankdelta_rows_filled | `{manifest['rankdelta_rows_filled']}/{manifest['rankdelta_rows_total']}` |",
        f"| has_strict_pass | `{manifest['has_strict_pass']}` |",
        f"| iclr95_gate | `{manifest['iclr95_num_passed']}/{manifest['iclr95_num_total']}` |",
        f"| iclr95_all_passed | `{manifest['iclr95_all_passed']}` |",
        f"| iclr95_8of10_current | `{manifest['iclr95_8of10_current']}` |",
        f"| iclr95_8of10_target | `{manifest['iclr95_8of10_target']}` |",
        f"| iclr95_8of10_additional_needed | `{manifest['iclr95_8of10_additional_needed']}` |",
        f"| diagnostic_draft_ready | `{manifest['diagnostic_draft_ready']}` |",
        f"| submission_ready | `{manifest['submission_ready']}` |",
        f"| source_package_ready | `{manifest['source_package_ready']}` |",
        f"| pdf_build_ready | `{manifest['pdf_build_ready']}` |",
        f"| gpu67_wait_action | `{manifest['gpu67_wait_action']}` |",
        f"| gpu67_raw_wait_action | `{manifest['gpu67_raw_wait_action']}` |",
        f"| gpu67_external_or_unknown_process_count | `{manifest['gpu67_external_or_unknown_process_count']}` |",
        f"| gpu67_launch_action | `{manifest['gpu67_launch_action']}` |",
        f"| gpu67_raw_launch_action | `{manifest['gpu67_raw_launch_action']}` |",
        f"| gpu67_launch_assets_ready | `{manifest['gpu67_launch_assets_ready']}` |",
        f"| gpu67_launch_blocker_count | `{manifest['gpu67_launch_blocker_count']}` |",
        f"| gpu67_launch_output_slots_complete | `{manifest['gpu67_launch_output_slots_complete']}/{manifest['gpu67_launch_output_slots_total']}` |",
        f"| gpu67_waiters_action | `{manifest['gpu67_waiters_action']}` |",
        f"| gpu67_raw_waiters_action | `{manifest['gpu67_raw_waiters_action']}` |",
        f"| gpu67_active_waiter_count | `{manifest['gpu67_active_waiter_count']}/{manifest['gpu67_waiter_count']}` |",
        f"| gpu67_stale_waiter_count | `{manifest['gpu67_stale_waiter_count']}` |",
        f"| gpu01_queue_action | `{manifest['gpu01_queue_action']}` |",
        f"| gpu01_p0_running_count | `{manifest['gpu01_p0_running_count']}/{manifest['gpu01_p0_variant_count']}` |",
        f"| gpu01_p0_outputs_complete | `{manifest['gpu01_p0_outputs_complete']}/{manifest['gpu01_p0_outputs_total']}` |",
        f"| gpu01_waiter_active_count | `{manifest['gpu01_waiter_active_count']}/{manifest['gpu01_waiter_count']}` |",
        f"| gpu01_stale_or_missing_count | `{manifest['gpu01_stale_or_missing_count']}` |",
        "",
        "## Step Status",
        "",
        "| step | script | returncode |",
        "|---|---|---:|",
    ]
    for step in manifest["steps"]:
        lines.append(
            f"| {step['name']} | `{step['script']}` | {step['returncode']} |")
    lines += [
        "",
        "## No-Fabrication Rule",
        "",
        manifest["no_fabrication_rule"],
        "",
        "## 8/10 Promotion Boundary",
        "",
        manifest["iclr95_8of10_forbidden_route"],
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument("--out-md", default=str(OUT_MD))
    args = parser.parse_args()

    step_results = run_refresh_steps(args.python)
    manifest = build_manifest(step_results)
    write_manifest(manifest, Path(args.out_json), Path(args.out_md))
    print(json.dumps({
        "ok": manifest["ok"],
        "active_route": manifest["active_route"],
        "summary_status": manifest["summary_status"],
        "followup_action": manifest["followup_action"],
        "closure_action": manifest["closure_action"],
        "confirmation_action": manifest["confirmation_action"],
        "p2_queue_action": manifest["p2_queue_action"],
        "p3a_status": manifest["p3a_status"],
        "p3a_main_method": manifest["p3a_main_method"],
        "p3a_main_gate": manifest["p3a_main_gate"],
        "p3a_strict_pass": manifest["p3a_strict_pass"],
        "p3d_status": manifest["p3d_status"],
        "p3d_dataset": manifest["p3d_dataset"],
        "p3d_main_method": manifest["p3d_main_method"],
        "p3d_main_gate": manifest["p3d_main_gate"],
        "p3d_strict_transfer_pass": manifest["p3d_strict_transfer_pass"],
        "rankdelta_rows_filled": manifest["rankdelta_rows_filled"],
        "rankdelta_rows_total": manifest["rankdelta_rows_total"],
        "iclr95_gate": (
            f"{manifest['iclr95_num_passed']}/"
            f"{manifest['iclr95_num_total']}"),
        "iclr95_8of10_current": manifest["iclr95_8of10_current"],
        "iclr95_8of10_target": manifest["iclr95_8of10_target"],
        "iclr95_8of10_additional_needed": (
            manifest["iclr95_8of10_additional_needed"]),
        "diagnostic_draft_ready": manifest["diagnostic_draft_ready"],
        "submission_ready": manifest["submission_ready"],
        "source_package_ready": manifest["source_package_ready"],
        "pdf_build_ready": manifest["pdf_build_ready"],
        "gpu67_wait_action": manifest["gpu67_wait_action"],
        "gpu67_raw_wait_action": manifest["gpu67_raw_wait_action"],
        "gpu67_launch_action": manifest["gpu67_launch_action"],
        "gpu67_raw_launch_action": manifest["gpu67_raw_launch_action"],
        "gpu67_launch_assets_ready": manifest["gpu67_launch_assets_ready"],
        "gpu67_launch_blocker_count": manifest["gpu67_launch_blocker_count"],
        "gpu67_waiters_action": manifest["gpu67_waiters_action"],
        "gpu67_raw_waiters_action": manifest["gpu67_raw_waiters_action"],
        "gpu67_active_waiter_count": manifest["gpu67_active_waiter_count"],
        "gpu67_waiter_count": manifest["gpu67_waiter_count"],
        "gpu01_queue_action": manifest["gpu01_queue_action"],
        "gpu01_p0_running_count": manifest["gpu01_p0_running_count"],
        "gpu01_p0_variant_count": manifest["gpu01_p0_variant_count"],
        "gpu01_waiter_active_count": manifest["gpu01_waiter_active_count"],
        "gpu01_waiter_count": manifest["gpu01_waiter_count"],
        "out_json": args.out_json,
        "out_md": args.out_md,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
