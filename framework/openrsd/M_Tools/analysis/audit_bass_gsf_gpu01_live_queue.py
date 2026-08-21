#!/usr/bin/env python
"""Audit the live GPU0/GPU1 BASS-GSF RankDelta queue.

This script is non-experimental. It checks the direct P0 sessions, the
GPU0/GPU1-rerouted follow-up waiters, and the P0 output slots. It never starts,
stops, or modifies tmux sessions.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


OUT_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "bass_gsf_gpu01_live_queue.json")
OUT_MD = Path(
    "resultmd/exp_p4_scale_semantic_validation/"
    "fstatus_20260621_bass_gsf_gpu01_live_queue.md")

DEFAULT_MAX_LOG_AGE_SECONDS = 600

P0_VARIANTS = [
    {
        "variant": "ms0p30",
        "gpu": 0,
        "exp_id": "bass_gsf_rankdelta_e2_dw0p05_ms0p30_t0p25_d0p50",
        "session": "bass_rankdelta_direct_p0_gpu0",
        "train_log": (
            "work_dirs/train_queue_logs/"
            "bass_gsf_rankdelta_e2_dw0p05_ms0p30_t0p25_d0p50_gpu0_train.log"),
    },
    {
        "variant": "ms0p50",
        "gpu": 1,
        "exp_id": "bass_gsf_rankdelta_e2_dw0p05_ms0p50_t0p25_d0p50",
        "session": "bass_rankdelta_direct_p0_gpu1",
        "train_log": (
            "work_dirs/train_queue_logs/"
            "bass_gsf_rankdelta_e2_dw0p05_ms0p50_t0p25_d0p50_gpu1_train.log"),
    },
]

WAITERS = [
    {
        "name": "P1 follow-up waiter",
        "session": "bass_rankdelta_followup_gpu01_wait",
        "log": "work_dirs/train_queue_logs/bass_rankdelta_followup_gpu01_stdout.log",
        "role": "launch P1 variants on GPU0/GPU1 if P0 lacks strict pass",
    },
    {
        "name": "P2 rank/assignment waiter",
        "session": "bass_rankdelta_p2_gpu01_wait",
        "log": "work_dirs/train_queue_logs/bass_rankdelta_p2_gpu01_stdout.log",
        "role": "launch P2 variants on GPU0/GPU1 if closure planner requires P2",
    },
]


def query_tmux_sessions() -> tuple[set[str], bool, str]:
    last_error = ""
    text = ""
    for command in (["rtk", "tmux", "list-sessions"], ["tmux", "list-sessions"]):
        proc = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode == 0:
            text = proc.stdout
            break
        last_error = proc.stderr[-1000:]
    else:
        return set(), False, last_error
    sessions = set()
    for line in text.splitlines():
        name = line.split(":", 1)[0].strip()
        if name:
            sessions.add(name)
    return sessions, True, ""


def file_status(path: Path, now: float,
                max_log_age_seconds: int | None = None) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "age_seconds": None,
            "fresh": False,
        }
    age = max(0.0, now - path.stat().st_mtime)
    return {
        "path": str(path),
        "exists": True,
        "age_seconds": round(age, 1),
        "fresh": (
            False if max_log_age_seconds is None
            else age <= max_log_age_seconds),
    }


def p0_output_paths(exp_id: str) -> dict[str, Path]:
    return {
        "checkpoint": Path(
            "work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/"
            f"train_epoch3_plus2_{exp_id}/epoch_2.pth"),
        "predictions": Path(
            "work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/"
            f"eval_{exp_id}_head_full/predictions.pkl"),
        "risk_summary": Path(
            "work_dirs/gs3c_sise_problem_reframing_20260619/"
            f"hrrsd_{exp_id}_head/deployment_risk_summary.json"),
    }


def build_p0_rows(sessions: set[str], tmux_query_ok: bool, now: float,
                  max_log_age_seconds: int) -> list[dict[str, Any]]:
    rows = []
    for spec in P0_VARIANTS:
        session_exists = spec["session"] in sessions
        train_log = file_status(
            Path(spec["train_log"]), now, max_log_age_seconds)
        outputs = {
            name: file_status(path, now)
            for name, path in p0_output_paths(spec["exp_id"]).items()
        }
        output_complete = all(row["exists"] for row in outputs.values())
        if output_complete:
            status = "outputs_complete"
        elif not tmux_query_ok and train_log["fresh"]:
            status = "log_active_tmux_unverified"
        elif session_exists and train_log["fresh"]:
            status = "running"
        elif session_exists:
            status = "session_active_log_stale_or_missing"
        elif train_log["fresh"]:
            status = "fresh_log_missing_session"
        else:
            status = "missing_or_stale_incomplete"
        rows.append({
            **spec,
            "session_exists": session_exists,
            "train_log": train_log,
            "outputs": outputs,
            "output_complete": output_complete,
            "status": status,
        })
    return rows


def build_waiter_rows(sessions: set[str], tmux_query_ok: bool, now: float,
                      max_log_age_seconds: int) -> list[dict[str, Any]]:
    rows = []
    for spec in WAITERS:
        session_exists = spec["session"] in sessions
        log = file_status(Path(spec["log"]), now, max_log_age_seconds)
        if not tmux_query_ok and log["fresh"]:
            status = "log_active_tmux_unverified"
        elif session_exists and log["fresh"]:
            status = "active"
        elif session_exists:
            status = "session_active_log_stale_or_missing"
        elif log["fresh"]:
            status = "fresh_log_missing_session"
        else:
            status = "missing_or_stale"
        rows.append({
            **spec,
            "session_exists": session_exists,
            "log": log,
            "status": status,
        })
    return rows


def build_audit(max_log_age_seconds: int = DEFAULT_MAX_LOG_AGE_SECONDS
                ) -> dict[str, Any]:
    now = time.time()
    sessions, tmux_query_ok, tmux_query_error = query_tmux_sessions()
    p0_rows = build_p0_rows(
        sessions, tmux_query_ok, now, max_log_age_seconds)
    waiter_rows = build_waiter_rows(
        sessions, tmux_query_ok, now, max_log_age_seconds)

    p0_running_count = sum(
        1 for row in p0_rows
        if row["status"] in {"running", "log_active_tmux_unverified"})
    p0_outputs_complete = sum(1 for row in p0_rows if row["output_complete"])
    waiter_active_count = sum(
        1 for row in waiter_rows
        if row["status"] in {"active", "log_active_tmux_unverified"})
    stale_or_missing_count = sum(
        1 for row in p0_rows + waiter_rows
        if row["status"] in {
            "session_active_log_stale_or_missing",
            "fresh_log_missing_session",
            "missing_or_stale",
            "missing_or_stale_incomplete",
        })

    if p0_outputs_complete == len(p0_rows) and waiter_active_count == len(waiter_rows):
        action = "P0_OUTPUTS_COMPLETE_WAITERS_ACTIVE"
    elif p0_running_count == len(p0_rows) and waiter_active_count == len(waiter_rows):
        action = "P0_RUNNING_GPU01_WAITERS_ACTIVE"
    elif not tmux_query_ok and p0_running_count == len(p0_rows):
        action = "P0_LOGS_ACTIVE_TMUX_UNVERIFIED"
    elif stale_or_missing_count:
        action = "CHECK_GPU01_QUEUE"
    else:
        action = "GPU01_QUEUE_PARTIAL"

    return {
        "action": action,
        "tmux_query_ok": tmux_query_ok,
        "tmux_query_error": tmux_query_error,
        "max_log_age_seconds": max_log_age_seconds,
        "p0_rows": p0_rows,
        "waiters": waiter_rows,
        "p0_running_count": p0_running_count,
        "p0_variant_count": len(p0_rows),
        "p0_outputs_complete": p0_outputs_complete,
        "p0_outputs_total": len(p0_rows),
        "waiter_active_count": waiter_active_count,
        "waiter_count": len(waiter_rows),
        "stale_or_missing_count": stale_or_missing_count,
        "no_experiment_rule": (
            "This live-queue audit never launches, kills, or modifies tmux "
            "sessions. It only reports the GPU0/GPU1-rerouted queue state."),
    }


def write_markdown(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# BASS-GSF GPU0/1 Live Queue Audit - 2026-06-21",
        "",
        "本文件由 `M_Tools/analysis/audit_bass_gsf_gpu01_live_queue.py` "
        "生成，只读检查 GPU0/GPU1 重定向队列，不启动或停止实验。",
        "",
        "| field | value |",
        "|---|---|",
        f"| action | `{audit['action']}` |",
        f"| tmux_query_ok | `{audit['tmux_query_ok']}` |",
        f"| p0_running_count | `{audit['p0_running_count']}/{audit['p0_variant_count']}` |",
        f"| p0_outputs_complete | `{audit['p0_outputs_complete']}/{audit['p0_outputs_total']}` |",
        f"| waiter_active_count | `{audit['waiter_active_count']}/{audit['waiter_count']}` |",
        f"| stale_or_missing_count | `{audit['stale_or_missing_count']}` |",
        f"| max_log_age_seconds | `{audit['max_log_age_seconds']}` |",
        "",
        "## P0 Direct Runs",
        "",
        "| variant | GPU | session | status | log fresh | outputs complete |",
        "|---|---:|---|---|---|---|",
    ]
    for row in audit["p0_rows"]:
        lines.append(
            f"| `{row['variant']}` | {row['gpu']} | `{row['session']}` | "
            f"`{row['status']}` | `{row['train_log']['fresh']}` | "
            f"`{row['output_complete']}` |")
    lines += [
        "",
        "## GPU0/1 Waiters",
        "",
        "| name | session | status | log fresh | role |",
        "|---|---|---|---|---|",
    ]
    for row in audit["waiters"]:
        lines.append(
            f"| {row['name']} | `{row['session']}` | `{row['status']}` | "
            f"`{row['log']['fresh']}` | {row['role']} |")
    lines += [
        "",
        "## No-Experiment Rule",
        "",
        audit["no_experiment_rule"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-log-age-seconds", type=int,
                        default=DEFAULT_MAX_LOG_AGE_SECONDS)
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument("--out-md", default=str(OUT_MD))
    args = parser.parse_args()

    audit = build_audit(args.max_log_age_seconds)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    write_markdown(Path(args.out_md), audit)
    print(json.dumps({
        "action": audit["action"],
        "p0_running_count": audit["p0_running_count"],
        "p0_variant_count": audit["p0_variant_count"],
        "p0_outputs_complete": audit["p0_outputs_complete"],
        "p0_outputs_total": audit["p0_outputs_total"],
        "waiter_active_count": audit["waiter_active_count"],
        "waiter_count": audit["waiter_count"],
        "stale_or_missing_count": audit["stale_or_missing_count"],
        "out_json": args.out_json,
        "out_md": args.out_md,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
