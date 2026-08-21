#!/usr/bin/env python3
"""Strict GPU command runner for full-live OpenRSD v3 experiments."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


ERROR_PATTERNS = (
    ("CUDA out of memory", "CUDA_OOM"),
    ("RuntimeError: CUDA error", "CUDA_RUNTIME"),
    ("FileNotFoundError", "FILE_NOT_FOUND"),
    ("No such file or directory", "FILE_NOT_FOUND"),
    ("KeyError", "KEY_ERROR"),
    ("gt_instances", "GT_INSTANCES"),
    ("loss is nan", "LOSS_NAN"),
    ("loss is inf", "LOSS_INF"),
    ("Traceback (most recent call last)", "TRACEBACK"),
)


def now() -> str:
    return datetime.now().strftime("%F %T")


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def shell_join(argv: Sequence[Any]) -> str:
    import shlex

    return " ".join(shlex.quote(str(x)) for x in argv)


def read_text(path: Path, max_chars: int = 200000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    return text[-max_chars:] if len(text) > max_chars else text


def tail(path: Path, lines: int = 100) -> str:
    return "\n".join(read_text(path).splitlines()[-lines:])


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_quiet(argv: Sequence[Any], cwd: Path | None = None, timeout: int = 30) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            [str(x) for x in argv],
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


def nvidia_smi_text(gpu_ids: str = "4,5") -> str:
    rc, out, err = run_quiet(["rtk", "nvidia-smi", "-i", gpu_ids], timeout=20)
    return out if rc == 0 else err


def query_gpu_rows(gpu_ids: str = "4,5") -> list[dict[str, Any]]:
    rc, out, _ = run_quiet(
        [
            "rtk",
            "nvidia-smi",
            f"--id={gpu_ids}",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=20,
    )
    rows: list[dict[str, Any]] = []
    if rc != 0:
        return rows
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            rows.append(
                {
                    "gpu": parts[0],
                    "name": parts[1],
                    "mem_used_mb": int(float(parts[2])),
                    "mem_total_mb": int(float(parts[3])),
                    "util_pct": int(float(parts[4])),
                }
            )
        except ValueError:
            continue
    return rows


def query_gpu_processes(gpu_id: str) -> list[dict[str, Any]]:
    rc, out, _ = run_quiet(
        [
            "rtk",
            "nvidia-smi",
            "-i",
            gpu_id,
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout=20,
    )
    rows: list[dict[str, Any]] = []
    if rc != 0:
        return rows
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            rows.append({"pid": parts[0], "process": parts[1], "used_memory_mb": parts[2]})
    return rows


def detect_failure(stdout: str, stderr: str) -> str:
    text = f"{stdout}\n{stderr}".lower()
    for needle, kind in ERROR_PATTERNS:
        if needle.lower() in text:
            return kind
    return ""


@dataclass
class StrictResult:
    task_name: str
    command: str
    start_time: str
    end_time: str
    return_code: int
    stdout_path: str
    stderr_path: str
    nvidia_smi_before: str
    nvidia_smi_after: str
    nvidia_smi_during: str
    gpu_ids: str
    batch_size: int | None = None
    duration_sec: float = 0.0
    peak_mem_mb: dict[str, int] = field(default_factory=dict)
    process_seen: dict[str, bool] = field(default_factory=dict)
    failure_kind: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""


class GpuMonitor:
    def __init__(self, work_dir: Path, gpu_ids: str, task_name: str, interval: float = 5.0):
        self.work_dir = work_dir
        self.gpu_ids = [g.strip() for g in gpu_ids.split(",") if g.strip()]
        self.task_name = task_name
        self.interval = interval
        self.peak_mem = {g: 0 for g in self.gpu_ids}
        self.process_seen = {g: False for g in self.gpu_ids}
        self.during_log = work_dir / "gpu_status.jsonl"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _snapshot(self) -> None:
        rows = query_gpu_rows(",".join(self.gpu_ids))
        payload = {"time": iso_now(), "task": self.task_name, "gpus": rows, "processes": {}}
        for row in rows:
            gpu = str(row.get("gpu"))
            if gpu in self.peak_mem:
                self.peak_mem[gpu] = max(self.peak_mem[gpu], int(row.get("mem_used_mb") or 0))
        for gpu in self.gpu_ids:
            procs = query_gpu_processes(gpu)
            payload["processes"][gpu] = procs
            if procs:
                self.process_seen[gpu] = True
        append_jsonl(self.during_log, payload)

    def __enter__(self) -> "GpuMonitor":
        self._snapshot()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._snapshot()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self.interval)
            if not self._stop.is_set():
                self._snapshot()


class StrictGpuRunner:
    def __init__(self, repo_root: Path, work_dir: Path, gpu_ids: str = "4,5"):
        self.repo_root = repo_root
        self.work_dir = work_dir
        self.gpu_ids = gpu_ids
        self.commands_log = work_dir / "commands.jsonl"
        self.failures_log = work_dir / "failures.jsonl"
        work_dir.mkdir(parents=True, exist_ok=True)

    def env(self, gpu_ids: str | None = None) -> list[str]:
        gids = gpu_ids or self.gpu_ids
        return [
            "PYTHONNOUSERSITE=1",
            "MPLCONFIGDIR=/tmp/mplconfig",
            f"CUDA_VISIBLE_DEVICES={gids}",
            f"PYTHONPATH={self.repo_root}:{self.repo_root / 'tools'}",
        ]

    def command_string(self, argv: Sequence[Any], gpu_ids: str | None = None) -> str:
        return "rtk env " + shell_join([*self.env(gpu_ids), *argv])

    def run(
        self,
        task_name: str,
        argv: Sequence[Any],
        log_dir: Path,
        gpu_ids: str | None = None,
        batch_size: int | None = None,
        timeout_sec: int | None = None,
        monitor_gpu: bool = True,
    ) -> StrictResult:
        gids = gpu_ids or self.gpu_ids
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "stdout.log"
        stderr_path = log_dir / "stderr.log"
        before_path = log_dir / "nvidia_smi_before.txt"
        after_path = log_dir / "nvidia_smi_after.txt"
        command = self.command_string(argv, gids)
        start_time = iso_now()
        start = time.time()
        write_text(before_path, nvidia_smi_text(gids))
        env_cmd = ["rtk", "env", *self.env(gids), *map(str, argv)]
        monitor = GpuMonitor(self.work_dir, gids, task_name, interval=5.0) if monitor_gpu else None
        return_code = 1
        with stdout_path.open("w", encoding="utf-8") as out_f, stderr_path.open("w", encoding="utf-8") as err_f:
            out_f.write(f"[{start_time}] command={command}\n")
            out_f.write(f"cwd={self.repo_root}\n\n")
            out_f.flush()
            try:
                if monitor:
                    monitor.__enter__()
                proc = subprocess.run(
                    env_cmd,
                    cwd=str(self.repo_root),
                    stdout=out_f,
                    stderr=err_f,
                    text=True,
                    timeout=timeout_sec,
                    check=False,
                )
                return_code = proc.returncode
            except subprocess.TimeoutExpired:
                return_code = 124
                err_f.write("\nTIMEOUT: no completion before timeout_sec\n")
            finally:
                if monitor:
                    monitor.__exit__(None, None, None)
        write_text(after_path, nvidia_smi_text(gids))
        stdout_tail = tail(stdout_path, 100)
        stderr_tail = tail(stderr_path, 100)
        result = StrictResult(
            task_name=task_name,
            command=command,
            start_time=start_time,
            end_time=iso_now(),
            return_code=return_code,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            nvidia_smi_before=str(before_path),
            nvidia_smi_after=str(after_path),
            nvidia_smi_during=str(self.work_dir / "gpu_status.jsonl"),
            gpu_ids=gids,
            batch_size=batch_size,
            duration_sec=time.time() - start,
            peak_mem_mb=monitor.peak_mem if monitor else {},
            process_seen=monitor.process_seen if monitor else {},
            failure_kind=detect_failure(stdout_tail, stderr_tail),
            stdout_tail=stdout_tail if return_code else "",
            stderr_tail=stderr_tail if return_code else "",
        )
        payload = asdict(result)
        append_jsonl(self.commands_log, payload)
        if return_code != 0:
            append_jsonl(self.failures_log, payload)
        return result


class ProgressWriter:
    def __init__(self, work_dir: Path, gpu_ids: str = "4,5", total_tasks: int = 12):
        self.work_dir = work_dir
        self.gpu_ids = gpu_ids
        self.total_tasks = max(total_tasks, 1)
        self.completed = 0
        self.current_task = "initializing"
        self.current_subtask = ""
        self.current_angle = ""
        self.batch_size: int | None = None
        self.images_sec: float | None = None
        self.iter_sec: float | None = None
        self.last_failure = ""
        self.next_task = ""
        self.start_time = time.time()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def update(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.write_once()

    def start(self) -> None:
        self.write_once()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.write_once()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.write_once()
            self._stop.wait(60)

    def write_once(self) -> None:
        rows = query_gpu_rows(self.gpu_ids)
        processes = {g: query_gpu_processes(g) for g in [x.strip() for x in self.gpu_ids.split(",") if x.strip()]}
        percent = min(100.0, 100.0 * self.completed / self.total_tasks)
        filled = int(round(percent / 5.0))
        bar = "[" + "#" * filled + "-" * (20 - filled) + f"] {percent:.1f}%"
        idle = not any(processes.values())
        elapsed = time.time() - self.start_time
        payload = {
            "time": iso_now(),
            "progress_percent": percent,
            "bar": bar,
            "current_task": self.current_task,
            "current_subtask": self.current_subtask,
            "current_angle_prompt_head_variant": self.current_angle,
            "batch_size": self.batch_size,
            "images_sec": self.images_sec,
            "iter_sec": self.iter_sec,
            "gpu_status": rows,
            "gpu_processes": processes,
            "last_failure": self.last_failure,
            "next_task": self.next_task,
            "elapsed_sec": elapsed,
            "over_24h": elapsed >= 24 * 3600,
            "gpu_idle": idle,
            "gpu_idle_reason": "no CUDA compute processes visible" if idle else "",
        }
        write_json(self.work_dir / "progress.json", payload)
        write_text(self.work_dir / "heartbeat.txt", iso_now())
        lines = [
            f"# Full Live GPU v3 Progress",
            "",
            bar,
            "",
            f"- current_task: `{self.current_task}`",
            f"- current_subtask: `{self.current_subtask}`",
            f"- current angle / prompt / head / variant: `{self.current_angle}`",
            f"- batch_size: `{self.batch_size}`",
            f"- images/sec: `{self.images_sec}`",
            f"- iter/sec: `{self.iter_sec}`",
            f"- last_failure: `{self.last_failure}`",
            f"- next_task: `{self.next_task}`",
            f"- elapsed_hours: `{elapsed / 3600:.3f}`",
            f"- over_24h: `{elapsed >= 24 * 3600}`",
            f"- gpu_idle: `{idle}`",
            f"- gpu_idle_reason: `{payload['gpu_idle_reason']}`",
            "",
            "## GPU4/GPU5",
            "",
            "| gpu | name | mem_used_mb | mem_total_mb | util_pct | processes |",
            "|---:|---|---:|---:|---:|---|",
        ]
        for row in rows:
            gpu = str(row.get("gpu"))
            procs = processes.get(gpu, [])
            proc_text = "<br>".join(f"{p.get('pid')}:{Path(str(p.get('process'))).name}:{p.get('used_memory_mb')}MB" for p in procs) or "none"
            lines.append(
                f"| {gpu} | {row.get('name')} | {row.get('mem_used_mb')} | {row.get('mem_total_mb')} | {row.get('util_pct')} | {proc_text} |"
            )
        write_text(self.work_dir / "progress.md", "\n".join(lines))

