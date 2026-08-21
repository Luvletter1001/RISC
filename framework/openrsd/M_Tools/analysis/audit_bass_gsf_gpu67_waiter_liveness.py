#!/usr/bin/env python
"""Audit liveness of the GPU6/GPU7 BASS-GSF waiter sessions.

This script is non-experimental.  It checks whether the RankDelta, follow-up,
and P2 tmux waiters exist and whether their logs have been updated recently.
It never starts, stops, or sends input to a tmux session.
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
    "bass_gsf_gpu67_waiter_liveness.json")
OUT_MD = Path(
    "resultmd/exp_p4_scale_semantic_validation/"
    "fstatus_20260621_bass_gsf_gpu67_waiter_liveness.md")

DEFAULT_MAX_LOG_AGE_SECONDS = 600

WAITERS = [
    {
        "name": "P0 RankDelta waiter",
        "session": "bass_rankdelta_gpu67_wait",
        "log": "work_dirs/train_queue_logs/bass_gsf_rankdelta_gpu67_wait_20260620.log",
        "role": "launch P0 RankDelta variants when GPU6/7 are free",
    },
    {
        "name": "P1 follow-up waiter",
        "session": "bass_rankdelta_followup_gpu67_wait",
        "log": (
            "work_dirs/train_queue_logs/"
            "bass_gsf_rankdelta_followup_gpu67_wait_20260620.log"),
        "role": "launch P1 only after P0 completes without strict pass",
    },
    {
        "name": "P2 rank/assignment waiter",
        "session": "bass_rankdelta_p2_gpu67_wait",
        "log": "work_dirs/train_queue_logs/bass_gsf_p2_gpu67_wait_20260620.log",
        "role": "launch P2 only after closure planner emits RUN_P2",
    },
]


def run_text(command: list[str]) -> str:
    proc = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout


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


def log_status(path: Path, now: float, max_log_age_seconds: int) -> dict[str, Any]:
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
        "fresh": age <= max_log_age_seconds,
    }


def build_audit(max_log_age_seconds: int = DEFAULT_MAX_LOG_AGE_SECONDS
                ) -> dict[str, Any]:
    now = time.time()
    sessions, tmux_query_ok, tmux_query_error = query_tmux_sessions()
    rows = []
    for spec in WAITERS:
        session_exists = spec["session"] in sessions
        log = log_status(Path(spec["log"]), now, max_log_age_seconds)
        if not tmux_query_ok:
            if log["fresh"]:
                status = "log_active_tmux_unverified"
            elif log["exists"]:
                status = "stale_log_tmux_unverified"
            else:
                status = "missing_log_tmux_unverified"
        elif session_exists and log["fresh"]:
            status = "active"
        elif session_exists and log["exists"]:
            status = "stale_log"
        elif session_exists:
            status = "missing_log"
        else:
            status = "missing_session"
        rows.append({
            **spec,
            "session_exists": session_exists,
            "log": log,
            "status": status,
        })

    missing = [row for row in rows if row["status"] == "missing_session"]
    stale = [
        row for row in rows
        if row["status"] in {
            "stale_log",
            "missing_log",
            "stale_log_tmux_unverified",
            "missing_log_tmux_unverified",
        }
    ]
    log_active_unverified = (
        not tmux_query_ok
        and all(row["status"] == "log_active_tmux_unverified" for row in rows)
    )
    if log_active_unverified:
        action = "LOGS_ACTIVE_TMUX_UNVERIFIED"
    elif missing:
        action = "RESTART_MISSING_WAITERS"
    elif stale:
        action = "CHECK_STALE_WAITERS"
    else:
        action = "WAITERS_ACTIVE"

    return {
        "action": action,
        "tmux_query_ok": tmux_query_ok,
        "tmux_query_error": tmux_query_error,
        "max_log_age_seconds": max_log_age_seconds,
        "waiters": rows,
        "active_waiter_count": sum(
            1 for row in rows
            if row["status"] in {"active", "log_active_tmux_unverified"}),
        "waiter_count": len(rows),
        "missing_session_count": len(missing),
        "stale_waiter_count": len(stale),
        "no_experiment_rule": (
            "This liveness audit never launches, kills, or modifies tmux "
            "sessions. It only reports whether the existing GPU6/GPU7 waiters "
            "appear alive."),
    }


def write_markdown(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# BASS-GSF GPU6/7 Waiter Liveness Audit - 2026-06-21",
        "",
        "This file is generated by "
        "`M_Tools/analysis/audit_bass_gsf_gpu67_waiter_liveness.py`.",
        "It does not start, stop, or modify experiments.",
        "",
        "| field | value |",
        "|---|---|",
        f"| action | `{audit['action']}` |",
        f"| tmux_query_ok | `{audit['tmux_query_ok']}` |",
        f"| active_waiter_count | `{audit['active_waiter_count']}/{audit['waiter_count']}` |",
        f"| missing_session_count | `{audit['missing_session_count']}` |",
        f"| stale_waiter_count | `{audit['stale_waiter_count']}` |",
        f"| max_log_age_seconds | `{audit['max_log_age_seconds']}` |",
        "",
        "## Waiters",
        "",
        "| name | session | status | log fresh | log age seconds | role |",
        "|---|---|---|---|---:|---|",
    ]
    for row in audit["waiters"]:
        log = row["log"]
        lines.append(
            f"| {row['name']} | `{row['session']}` | `{row['status']}` | "
            f"`{log['fresh']}` | {log['age_seconds']} | {row['role']} |")
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
        "active_waiter_count": audit["active_waiter_count"],
        "waiter_count": audit["waiter_count"],
        "missing_session_count": audit["missing_session_count"],
        "stale_waiter_count": audit["stale_waiter_count"],
        "out_json": args.out_json,
        "out_md": args.out_md,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
