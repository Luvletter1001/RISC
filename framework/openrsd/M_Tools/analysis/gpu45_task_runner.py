#!/usr/bin/env python3
"""Shared utilities for the GPU4/5 next-priority OpenRSD scheduler."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


GPU45 = "4,5"
RUN_TS_FMT = "%Y%m%d_%H%M%S"
DEFAULT_PYTHON = Path("/data/zcy/anaconda3/envs/openrsd/bin/python")


def now() -> str:
    return datetime.now().strftime("%F %T")


def run_ts() -> str:
    return datetime.now().strftime(RUN_TS_FMT)


def shell_join(argv: Sequence[Any]) -> str:
    return " ".join(shlex.quote(str(item)) for item in argv)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_quiet(argv: Sequence[Any], cwd: Path | None = None, timeout: int = 30) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            [str(item) for item in argv],
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as exc:  # noqa: BLE001
        return 1, "", repr(exc)


def nvidia_smi_text() -> str:
    rc, out, err = run_quiet(["rtk", "nvidia-smi"], timeout=20)
    return out if rc == 0 else err


def gpu45_query() -> dict[str, Any]:
    rc, out, err = run_quiet(
        [
            "rtk",
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=20,
    )
    gpus: dict[str, Any] = {}
    if rc == 0:
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5 or parts[0] not in {"4", "5"}:
                continue
            gpus[parts[0]] = {
                "name": parts[1],
                "memory_used_mb": int(float(parts[2])),
                "memory_total_mb": int(float(parts[3])),
                "utilization_gpu_pct": int(float(parts[4])),
            }
    rc2, out2, _ = run_quiet(
        [
            "rtk",
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout=20,
    )
    for gpu in gpus.values():
        gpu["processes"] = []
    if rc2 == 0 and out2.strip():
        for line in out2.splitlines():
            for gpu in gpus.values():
                # nvidia-smi compute-app query does not include index on this driver;
                # process ownership is still useful as a coarse signal.
                gpu["processes"].append(line.strip())
    if rc != 0:
        gpus["error"] = err
    return gpus


def base_env(repo_root: Path, gpu_ids: str = GPU45) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "MPLCONFIGDIR": "/tmp/mplconfig",
            "CUDA_VISIBLE_DEVICES": gpu_ids,
            "PYTHONPATH": f"{repo_root}:{repo_root / 'tools'}",
        }
    )
    return env


def command_string(argv: Sequence[Any], repo_root: Path, gpu_ids: str = GPU45) -> str:
    prefix = [
        "rtk",
        "env",
        "PYTHONNOUSERSITE=1",
        "MPLCONFIGDIR=/tmp/mplconfig",
        f"CUDA_VISIBLE_DEVICES={gpu_ids}",
        f"PYTHONPATH={repo_root}:{repo_root / 'tools'}",
    ]
    return shell_join([*prefix, *argv])


ERROR_PATTERNS = (
    ("CUDA out of memory", "CUDA_OOM"),
    ("out of memory", "OOM"),
    ("RuntimeError: CUDA error", "CUDA_RUNTIME"),
    ("FileNotFoundError", "FILE_NOT_FOUND"),
    ("No such file or directory", "FILE_NOT_FOUND"),
    ("KeyError: gt_instances", "GT_INSTANCES"),
    ("loss is nan", "LOSS_NAN"),
    ("loss is inf", "LOSS_INF"),
    ("Traceback (most recent call last)", "TRACEBACK"),
)


def detect_failure(stdout: str, stderr: str) -> str:
    text = f"{stdout}\n{stderr}".lower()
    for needle, kind in ERROR_PATTERNS:
        if needle.lower() in text:
            return kind
    return ""


def tail(path: Path, n: int = 100) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])


@dataclass
class ChildResult:
    task_name: str
    command: str
    start_time: str
    end_time: str
    return_code: int
    stdout_path: str
    stderr_path: str
    nvidia_smi_before: str
    nvidia_smi_after: str
    duration_sec: float
    gpu_ids: str
    failure_kind: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""


def run_child(
    *,
    task_name: str,
    argv: Sequence[Any],
    repo_root: Path,
    log_dir: Path,
    gpu_ids: str = GPU45,
    timeout_sec: int | None = None,
) -> ChildResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"
    before_path = log_dir / "nvidia_smi_before.txt"
    after_path = log_dir / "nvidia_smi_after.txt"
    command = command_string(argv, repo_root, gpu_ids)
    start_time = now()
    start = time.time()
    write_text(before_path, nvidia_smi_text())
    env_cmd = [
        "rtk",
        "env",
        "PYTHONNOUSERSITE=1",
        "MPLCONFIGDIR=/tmp/mplconfig",
        f"CUDA_VISIBLE_DEVICES={gpu_ids}",
        f"PYTHONPATH={repo_root}:{repo_root / 'tools'}",
        *map(str, argv),
    ]
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        out.write(f"[{start_time}] command={command}\n")
        out.write(f"cwd={repo_root}\n\n")
        out.flush()
        try:
            proc = subprocess.run(env_cmd, cwd=str(repo_root), stdout=out, stderr=err, text=True, timeout=timeout_sec, check=False)
            return_code = proc.returncode
        except subprocess.TimeoutExpired:
            return_code = 124
            err.write("\nTIMEOUT: command exceeded timeout_sec\n")
    write_text(after_path, nvidia_smi_text())
    stdout_tail = tail(stdout_path)
    stderr_tail = tail(stderr_path)
    failure_kind = detect_failure(stdout_tail, stderr_tail) if return_code else ""
    return ChildResult(
        task_name=task_name,
        command=command,
        start_time=start_time,
        end_time=now(),
        return_code=return_code,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        nvidia_smi_before=str(before_path),
        nvidia_smi_after=str(after_path),
        duration_sec=time.time() - start,
        gpu_ids=gpu_ids,
        failure_kind=failure_kind,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
    )


class ProgressWriter:
    def __init__(self, work_dir: Path, run_ts_value: str, time_budget_hours: float):
        self.work_dir = work_dir
        self.run_ts = run_ts_value
        self.time_budget_sec = time_budget_hours * 3600.0
        self.start_time = time.time()
        self.total = 12
        self.done: list[str] = []
        self.current_stage = "init"
        self.current_task = "init"
        self.current_task_progress = 0.0
        self.running_command = ""
        self.last_failure = ""
        self.next_task = "P0"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.write()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.write()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.write()
            self._stop.wait(60)

    def set(self, stage: str, task: str, progress: float = 0.0, command: str = "", next_task: str = "") -> None:
        self.current_stage = stage
        self.current_task = task
        self.current_task_progress = progress
        self.running_command = command
        if next_task:
            self.next_task = next_task
        self.write()

    def complete(self, task: str, next_task: str = "") -> None:
        if task not in self.done:
            self.done.append(task)
        if next_task:
            self.next_task = next_task
        self.write()

    def fail(self, message: str) -> None:
        self.last_failure = message
        self.write()

    def write(self) -> None:
        elapsed = time.time() - self.start_time
        remaining = max(self.time_budget_sec - elapsed, 0.0)
        progress = min(len(self.done) / max(self.total, 1), 1.0)
        filled = int(progress * 20)
        bar = "[" + "#" * filled + "-" * (20 - filled) + f"] {progress * 100:.1f}%"
        payload = {
            "run_ts": self.run_ts,
            "time": now(),
            "progress": progress,
            "current_stage": self.current_stage,
            "current_task": self.current_task,
            "current_task_progress": self.current_task_progress,
            "done_tasks": self.done,
            "running_command": self.running_command,
            "gpu_status": gpu45_query(),
            "last_heartbeat": now(),
            "last_failure": self.last_failure,
            "next_task": self.next_task,
            "elapsed_sec": elapsed,
            "remaining_to_24h_sec": remaining,
        }
        write_json(self.work_dir / "progress.json", payload)
        write_text(self.work_dir / "heartbeat.txt", now())
        append_jsonl(self.work_dir / "gpu_status.jsonl", {"time": now(), "gpu_status": payload["gpu_status"]})
        gpu_lines = []
        for idx in ("4", "5"):
            gpu = payload["gpu_status"].get(idx, {})
            gpu_lines.append(
                f"- GPU{idx}: `{gpu.get('memory_used_mb', 'NA')}/{gpu.get('memory_total_mb', 'NA')} MB`, "
                f"util `{gpu.get('utilization_gpu_pct', 'NA')}%`, processes `{len(gpu.get('processes', []))}`"
            )
        lines = [
            "# GPU45 Next Priority Progress",
            "",
            bar,
            "",
            f"- Current stage: `{self.current_stage}`",
            f"- Current subtask: `{self.current_task}`",
            f"- Current subtask progress: `{self.current_task_progress * 100:.1f}%`",
            f"- Running command: `{self.running_command}`",
            *gpu_lines,
            f"- Recent heartbeat time: `{now()}`",
            f"- Recent failure task: `{self.last_failure}`",
            f"- Next task: `{self.next_task}`",
            f"- Elapsed: `{elapsed / 3600:.2f} h`",
            f"- Remaining to 24h: `{remaining / 3600:.2f} h`",
        ]
        write_text(self.work_dir / "progress.md", "\n".join(lines))


def result_to_dict(result: ChildResult) -> dict[str, Any]:
    return asdict(result)
