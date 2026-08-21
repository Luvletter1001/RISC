#!/usr/bin/env python3
"""Watch the FOCUS-EQText P2 EQ_V30 dual-GPU run on physical GPUs 6 and 9.

The watcher is intentionally conservative. It records evidence every poll and
only attempts a resume when the dual-rank GPU process is gone or a fatal/OOM
pattern is present while the run is not complete.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GPU6_UUID = "GPU-b30e9f80-7925-6816-9014-d228784c5a12"
GPU9_UUID = "GPU-ff760a2d-9a1b-38ee-06a0-8c284e002545"
TARGET_UUIDS = {GPU6_UUID, GPU9_UUID}
DEFAULT_SESSION = "focus_eqtext_p2_gpu69_v30"
DEFAULT_WATCH_SESSION = "focus_eqtext_p2_watch_v30"
DEFAULT_VARIANT = "EQ_V30_dual_eqtext"
DEFAULT_EXP_DIR = Path("resultmd/exp_focus_eqtext_dota_short_20260609")
DEFAULT_WORK_DIR = Path("work_dirs/focus_eqtext_dota_short_20260609") / DEFAULT_VARIANT
DEFAULT_LAUNCHER = (
    DEFAULT_EXP_DIR / "configs" / "run_focus_eqtext_p2_train_gpu69.sh")
DEFAULT_PYTHON = Path("/data/zcy/anaconda3/envs/openrsd/bin/python")

FATAL_PATTERNS = (
    re.compile(r"CUDA out of memory|out of memory", re.IGNORECASE),
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"RuntimeError:", re.IGNORECASE),
    re.compile(r"\bNCCL\b.*\b(error|failed|timeout|unhandled)\b", re.IGNORECASE),
    re.compile(r"\b(FATAL|Fatal)\b"),
    re.compile(r"(^|[^A-Za-z])nan([^A-Za-z]|$)", re.IGNORECASE),
)
EPOCH_RE = re.compile(r"Epoch\((train|val)\) \[(\d+)\]\[\s*(\d+)/(\d+)\]")
SAVE_RE = re.compile(r"Saving checkpoint at (\d+) epochs")
MAP_RE = re.compile(r"dota/mAP:\s*([0-9.]+)\s+dota/AP50:\s*([0-9.]+)")
SMALL_VEHICLE_RE = re.compile(
    r"'small-vehicle': \{'ap': ([0-9.]+), 'recall': ([0-9.]+), "
    r"'num_dets': ([0-9]+), 'num_gts': ([0-9]+)\}")


def now_payload() -> dict[str, str]:
    local = datetime.now().astimezone()
    utc = datetime.now(timezone.utc)
    return {
        "local": local.isoformat(timespec="seconds"),
        "utc": utc.isoformat(timespec="seconds"),
        "stamp": local.strftime("%Y%m%d_%H%M%S"),
    }


def run_cmd(cmd: list[str], cwd: Path, timeout: int = 30) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
        }
    except Exception as exc:  # pragma: no cover - evidence path
        return {
            "cmd": cmd,
            "returncode": -1,
            "stdout": f"{type(exc).__name__}: {exc}",
        }


def parse_gpu_rows(text: str) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            rows.append({
                "index": int(parts[0]),
                "uuid": parts[1],
                "memory_used_mb": int(parts[2]),
                "memory_total_mb": int(parts[3]),
                "utilization_gpu_pct": int(parts[4]),
            })
        except ValueError:
            continue
    return rows


def parse_compute_rows(text: str) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            rows.append({
                "pid": int(parts[0]),
                "gpu_uuid": parts[1],
                "used_memory_mb": int(parts[2]),
            })
        except ValueError:
            continue
    return rows


def tail_text(path: Path, max_lines: int) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:]) + ("\n" if lines else "")


def newest_log(work_dir: Path) -> Path | None:
    logs = sorted(work_dir.glob("*/**/*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def checkpoint_epochs(work_dir: Path) -> list[int]:
    epochs = []
    for path in work_dir.glob("epoch_*.pth"):
        match = re.fullmatch(r"epoch_(\d+)\.pth", path.name)
        if match:
            epochs.append(int(match.group(1)))
    return sorted(epochs)


def read_last_checkpoint(work_dir: Path) -> str:
    path = work_dir / "last_checkpoint"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def parse_log(log_path: Path | None) -> dict[str, Any]:
    if log_path is None or not log_path.exists():
        return {
            "log_path": None,
            "last_epoch": None,
            "last_phase": None,
            "last_iter": None,
            "last_total_iter": None,
            "last_saved_epoch": None,
            "latest_map": None,
            "latest_ap50": None,
            "small_vehicle": None,
            "fatal_matches": [],
        }
    text = log_path.read_text(encoding="utf-8", errors="replace")
    last_epoch: dict[str, Any] | None = None
    for match in EPOCH_RE.finditer(text):
        last_epoch = {
            "phase": match.group(1),
            "epoch": int(match.group(2)),
            "iter": int(match.group(3)),
            "total_iter": int(match.group(4)),
        }
    saved = [int(match.group(1)) for match in SAVE_RE.finditer(text)]
    map_match = None
    for match in MAP_RE.finditer(text):
        map_match = match
    small_match = None
    for match in SMALL_VEHICLE_RE.finditer(text):
        small_match = match
    fatal_matches = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if any(pattern.search(line) for pattern in FATAL_PATTERNS):
            fatal_matches.append({"line": line_no, "text": line[:500]})
    return {
        "log_path": str(log_path),
        "log_mtime": log_path.stat().st_mtime,
        "last_phase": None if last_epoch is None else last_epoch["phase"],
        "last_epoch": None if last_epoch is None else last_epoch["epoch"],
        "last_iter": None if last_epoch is None else last_epoch["iter"],
        "last_total_iter": None if last_epoch is None else last_epoch["total_iter"],
        "last_saved_epoch": max(saved) if saved else None,
        "latest_map": None if map_match is None else float(map_match.group(1)),
        "latest_ap50": None if map_match is None else float(map_match.group(2)),
        "small_vehicle": None if small_match is None else {
            "ap": float(small_match.group(1)),
            "recall": float(small_match.group(2)),
            "num_dets": int(small_match.group(3)),
            "num_gts": int(small_match.group(4)),
        },
        "fatal_matches": fatal_matches[-20:],
    }


def tmux_sessions(repo_root: Path) -> list[str]:
    proc = run_cmd(["rtk", "tmux", "list-sessions"], repo_root)
    if proc["returncode"] != 0:
        return []
    return [line.split(":", 1)[0] for line in proc["stdout"].splitlines() if line]


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"restart_count": 0, "restart_sessions": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"restart_count": 0, "restart_sessions": [], "state_error": "invalid_json"}


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def classify_status(snapshot: dict[str, Any], final_epoch: int) -> str:
    if snapshot.get("telemetry_errors"):
        return "TELEMETRY_BLOCKED"
    if not snapshot.get("gpu_rows") and not snapshot.get("compute_rows"):
        return "TELEMETRY_BLOCKED"
    target_processes = snapshot["target_compute_processes"]
    log = snapshot["log"]
    fatal = bool(log.get("fatal_matches"))
    max_epoch = snapshot.get("max_checkpoint_epoch") or 0
    completed = max_epoch >= final_epoch and not target_processes
    if completed:
        return "COMPLETED"
    if target_processes and len({p["gpu_uuid"] for p in target_processes}) == 2:
        if fatal:
            return "FATAL_PATTERN_BUT_RUNNING"
        if log.get("last_phase") == "val":
            return "VALIDATING"
        return "RUNNING"
    if fatal:
        return "FATAL_DETECTED"
    if max_epoch >= final_epoch:
        return "COMPLETED_OR_FINALIZING"
    return "NO_DUAL_GPU_PROCESS"


def should_restart(snapshot: dict[str, Any], state: dict[str, Any],
                   final_epoch: int, max_restarts: int) -> tuple[bool, str]:
    status = snapshot["status"]
    if state.get("restart_count", 0) >= max_restarts:
        return False, "restart_cap_reached"
    if status == "TELEMETRY_BLOCKED":
        return False, "telemetry_blocked"
    if status in {"FATAL_DETECTED", "NO_DUAL_GPU_PROCESS"}:
        max_epoch = snapshot.get("max_checkpoint_epoch") or 0
        if max_epoch >= final_epoch:
            return False, "final_epoch_present"
        return True, status
    return False, status


def launch_resume(args: argparse.Namespace, state: dict[str, Any],
                  stamp: str) -> dict[str, Any]:
    restart_no = int(state.get("restart_count", 0)) + 1
    session = f"{args.session}_resume_{stamp}"
    log_dir = args.watch_dir / "restart_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{session}.log"
    port = str(29700 + (int(time.time()) % 200))
    command = " ".join([
        "cd", shlex.quote(str(args.repo_root)),
        "&&", "env",
        "CUDA_VISIBLE_DEVICES=6,9",
        "NCCL_P2P_DISABLE=1",
        "NCCL_IB_DISABLE=1",
        "PYTHONNOUSERSITE=1",
        "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256",
        f"MASTER_PORT={shlex.quote(port)}",
        "RESUME=1",
        "bash", shlex.quote(str(args.launcher)),
        shlex.quote(args.variant),
        ">", shlex.quote(str(log_path)),
        "2>&1",
    ])
    proc = run_cmd([
        "rtk", "tmux", "new-session", "-d", "-s", session, command,
    ], args.repo_root)
    event = {
        "event": "restart_attempt",
        "session": session,
        "restart_no": restart_no,
        "command": command,
        "log": str(log_path),
        "result": proc,
    }
    if proc["returncode"] == 0:
        state["restart_count"] = restart_no
        state.setdefault("restart_sessions", []).append(session)
    return event


def collect_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    ts = now_payload()
    gpu_proc = run_cmd([
        "rtk", "nvidia-smi",
        "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], args.repo_root)
    compute_proc = run_cmd([
        "rtk", "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid,used_memory",
        "--format=csv,noheader,nounits",
    ], args.repo_root)
    gpu_rows = parse_gpu_rows(gpu_proc["stdout"])
    compute_rows = parse_compute_rows(compute_proc["stdout"])
    telemetry_errors = []
    if gpu_proc["returncode"] != 0:
        telemetry_errors.append({
            "name": "gpu_query",
            "returncode": gpu_proc["returncode"],
            "stdout": gpu_proc["stdout"][-1000:],
        })
    if compute_proc["returncode"] != 0:
        telemetry_errors.append({
            "name": "compute_query",
            "returncode": compute_proc["returncode"],
            "stdout": compute_proc["stdout"][-1000:],
        })
    target_processes = [
        row for row in compute_rows if row["gpu_uuid"] in TARGET_UUIDS]
    log_path = newest_log(args.work_dir)
    log_info = parse_log(log_path)
    epochs = checkpoint_epochs(args.work_dir)
    sessions = tmux_sessions(args.repo_root)
    snapshot = {
        "timestamp": ts,
        "variant": args.variant,
        "session": args.session,
        "session_present": args.session in sessions,
        "watch_session": args.watch_session,
        "watch_session_present": args.watch_session in sessions,
        "gpu_rows": gpu_rows,
        "compute_rows": compute_rows,
        "telemetry_errors": telemetry_errors,
        "target_compute_processes": target_processes,
        "target_gpu_process_count": len(target_processes),
        "target_gpu_uuids_present": sorted(
            {row["gpu_uuid"] for row in target_processes}),
        "last_checkpoint": read_last_checkpoint(args.work_dir),
        "checkpoint_epochs": epochs,
        "max_checkpoint_epoch": max(epochs) if epochs else None,
        "log": log_info,
    }
    snapshot["status"] = classify_status(snapshot, args.final_epoch)
    return snapshot


def write_snapshot(args: argparse.Namespace, snapshot: dict[str, Any]) -> None:
    stamp = snapshot["timestamp"]["stamp"]
    snapshot_path = args.watch_dir / "snapshots" / f"{stamp}.json"
    save_json(snapshot_path, snapshot)
    save_json(args.watch_dir / "latest_status.json", snapshot)
    append_jsonl(args.watch_dir / "watch_events.jsonl", {
        "event": "snapshot",
        "snapshot": str(snapshot_path),
        "timestamp": snapshot["timestamp"],
        "status": snapshot["status"],
        "max_checkpoint_epoch": snapshot.get("max_checkpoint_epoch"),
        "latest_map": snapshot["log"].get("latest_map"),
        "latest_ap50": snapshot["log"].get("latest_ap50"),
        "target_gpu_process_count": snapshot["target_gpu_process_count"],
    })
    log_path = Path(snapshot["log"]["log_path"]) if snapshot["log"].get("log_path") else None
    if log_path is not None:
        tail_path = args.watch_dir / "log_tails" / f"{stamp}_{log_path.name}.tail"
        tail_path.parent.mkdir(parents=True, exist_ok=True)
        tail_path.write_text(tail_text(log_path, args.tail_lines), encoding="utf-8")


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    args.repo_root = args.repo_root.resolve()
    args.exp_dir = resolve_path(args.repo_root, args.exp_dir)
    args.work_dir = resolve_path(args.repo_root, args.work_dir)
    args.launcher = resolve_path(args.repo_root, args.launcher)
    args.watch_dir = resolve_path(args.repo_root, args.watch_dir)
    args.watch_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.watch_dir / "watch_state.json"
    state = load_state(state_path)
    snapshot = collect_snapshot(args)
    write_snapshot(args, snapshot)
    do_restart, reason = should_restart(
        snapshot, state, args.final_epoch, args.max_restarts)
    event = {
        "event": "restart_decision",
        "timestamp": snapshot["timestamp"],
        "restart": do_restart,
        "reason": reason,
        "status": snapshot["status"],
    }
    if args.restart_on_fatal and do_restart:
        restart_event = launch_resume(args, state, snapshot["timestamp"]["stamp"])
        event["restart_event"] = restart_event
    save_json(state_path, state)
    append_jsonl(args.watch_dir / "watch_events.jsonl", event)
    return snapshot


def resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--exp-dir", type=Path, default=DEFAULT_EXP_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--launcher", type=Path, default=DEFAULT_LAUNCHER)
    parser.add_argument("--watch-dir", type=Path,
                        default=DEFAULT_EXP_DIR / "p2_watch")
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--watch-session", default=DEFAULT_WATCH_SESSION)
    parser.add_argument("--max-hours", type=float, default=10.0)
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--tail-lines", type=int, default=120)
    parser.add_argument("--final-epoch", type=int, default=24)
    parser.add_argument("--max-restarts", type=int, default=2)
    parser.add_argument("--restart-on-fatal", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    deadline = time.time() + args.max_hours * 3600.0
    while True:
        snapshot = run_once(args)
        print(json.dumps({
            "timestamp": snapshot["timestamp"],
            "status": snapshot["status"],
            "max_checkpoint_epoch": snapshot.get("max_checkpoint_epoch"),
            "latest_map": snapshot["log"].get("latest_map"),
            "latest_ap50": snapshot["log"].get("latest_ap50"),
            "target_gpu_process_count": snapshot["target_gpu_process_count"],
        }, indent=2), flush=True)
        if args.once or snapshot["status"] in {"COMPLETED", "COMPLETED_OR_FINALIZING"}:
            return 0
        if time.time() >= deadline:
            return 124
        time.sleep(max(args.poll_seconds, 1.0))


if __name__ == "__main__":
    raise SystemExit(main())
