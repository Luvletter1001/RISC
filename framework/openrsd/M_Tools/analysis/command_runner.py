#!/usr/bin/env python3
"""Small command runner used by OpenRSD follow-up experiments."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


ERROR_PATTERNS = (
    ('CUDA out of memory', 'CUDA_OOM'),
    ('out of memory', 'OOM'),
    ('RuntimeError: CUDA error', 'CUDA_RUNTIME'),
    ('FileNotFoundError', 'FILE_NOT_FOUND'),
    ('No such file or directory', 'FILE_NOT_FOUND'),
    ('checkpoint missing', 'CHECKPOINT_MISSING'),
    ('Config File', 'CONFIG_MISSING'),
    ('prompt file missing', 'PROMPT_FILE_MISSING'),
    ('class mapping', 'CLASS_MAPPING_MISMATCH'),
    ('shape mismatch', 'TEXT_EMBEDDING_SHAPE_MISMATCH'),
    ('support embedding missing', 'SUPPORT_EMBEDDING_MISSING'),
    ('ann_file', 'ANNOTATION_PATH_MISSING'),
    ('img_path', 'IMAGE_PATH_MISSING'),
    ('predictions missing', 'PREDICTIONS_MISSING'),
    ('pkl length', 'PKL_LENGTH_MISMATCH'),
    ('img_id', 'IMG_ID_ALIGNMENT_FAILURE'),
    ('gt_instances', 'GT_INSTANCES'),
    ('loss is nan', 'LOSS_NAN'),
    ('loss is inf', 'LOSS_INF'),
    ('class mismatch', 'EVALUATION_CLASS_MISMATCH'),
    ('Traceback (most recent call last)', 'TRACEBACK'),
)


@dataclass
class CommandResult:
    task_name: str
    command: str
    start_time: str
    end_time: str
    return_code: int
    stdout_path: str
    stderr_path: str
    nvidia_smi_before: str
    nvidia_smi_after: str
    gpu_ids: str
    batch_size: int | None = None
    retry_count: int = 0
    duration_sec: float = 0.0
    failure_kind: str = ''
    stdout_tail: str = ''
    stderr_tail: str = ''
    peak_mem_mb: dict[str, int] = field(default_factory=dict)
    process_seen: dict[str, bool] = field(default_factory=dict)


def now() -> str:
    return datetime.now().strftime('%F %T')


def shell_join(argv: Sequence[Any]) -> str:
    import shlex

    return ' '.join(shlex.quote(str(item)) for item in argv)


def read_text(path: Path, max_chars: int = 200000) -> str:
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except FileNotFoundError:
        return ''
    return text[-max_chars:] if len(text) > max_chars else text


def tail(path: Path, count: int = 100) -> str:
    return '\n'.join(read_text(path).splitlines()[-count:])


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + '\n', encoding='utf-8')


def run_quiet(argv: Sequence[Any], timeout: int = 30, cwd: Path | None = None) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            [str(item) for item in argv],
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False)
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as exc:  # noqa: BLE001
        return 1, '', repr(exc)


def nvidia_smi_text() -> str:
    rc, out, err = run_quiet(['rtk', 'nvidia-smi'], timeout=20)
    return out if rc == 0 else err


def detect_failure(stdout: str, stderr: str) -> str:
    text = f'{stdout}\n{stderr}'.lower()
    for needle, kind in ERROR_PATTERNS:
        if needle.lower() in text:
            return kind
    return ''


class GpuMonitor:
    def __init__(self, gpu_ids: str, interval: float = 2.0):
        self.gpu_ids = [item.strip() for item in gpu_ids.split(',') if item.strip()]
        self.interval = interval
        self.peak_mem = {gpu: 0 for gpu in self.gpu_ids}
        self.process_seen = {gpu: False for gpu in self.gpu_ids}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _snapshot(self) -> None:
        rc, out, _ = run_quiet([
            'rtk', 'nvidia-smi', '--query-gpu=index,memory.used',
            '--format=csv,noheader,nounits'
        ], timeout=10)
        if rc == 0:
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2 and parts[0] in self.peak_mem:
                    try:
                        self.peak_mem[parts[0]] = max(self.peak_mem[parts[0]], int(float(parts[1])))
                    except ValueError:
                        pass
        rc, out, _ = run_quiet([
            'rtk', 'nvidia-smi',
            '--query-compute-apps=pid,process_name,used_memory',
            '--format=csv,noheader,nounits'
        ], timeout=10)
        if rc == 0 and out.strip():
            for gpu in self.gpu_ids:
                self.process_seen[gpu] = True

    def __enter__(self) -> 'GpuMonitor':
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
            self._snapshot()
            self._stop.wait(self.interval)


class CommandRunner:
    def __init__(self,
                 repo_root: Path,
                 work_dir: Path,
                 gpu_ids: str,
                 python_bin: Path,
                 command_log: Path | None = None,
                 failures_log: Path | None = None):
        self.repo_root = repo_root
        self.work_dir = work_dir
        self.gpu_ids = gpu_ids
        self.python_bin = python_bin
        self.command_log = command_log or work_dir / 'commands.jsonl'
        self.failures_log = failures_log or work_dir / 'failures.jsonl'
        self.command_log.parent.mkdir(parents=True, exist_ok=True)
        self.failures_log.parent.mkdir(parents=True, exist_ok=True)

    def env_assignments(self, gpu_ids: str | None = None) -> list[str]:
        gpu_ids = gpu_ids or self.gpu_ids
        return [
            'PYTHONNOUSERSITE=1',
            'MPLCONFIGDIR=/tmp/mplconfig',
            f'CUDA_VISIBLE_DEVICES={gpu_ids}',
            f'PYTHONPATH={self.repo_root}:{self.repo_root / "tools"}',
        ]

    def command_string(self, argv: Sequence[Any], gpu_ids: str | None = None) -> str:
        return 'rtk env ' + shell_join(self.env_assignments(gpu_ids)) + ' ' + shell_join(argv)

    def run(self,
            task_name: str,
            argv: Sequence[Any],
            log_dir: Path,
            gpu_ids: str | None = None,
            batch_size: int | None = None,
            retry_count: int = 0,
            monitor_gpu: bool = False,
            timeout_sec: int | None = None,
            skip_if: Path | None = None) -> CommandResult:
        gpu_ids = gpu_ids or self.gpu_ids
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / 'stdout.log'
        stderr_path = log_dir / 'stderr.log'
        before_path = log_dir / 'nvidia_smi_before.txt'
        after_path = log_dir / 'nvidia_smi_after.txt'
        command = self.command_string(argv, gpu_ids)
        start_time = now()
        start = time.time()
        write_text(before_path, nvidia_smi_text() or 'nvidia-smi unavailable')

        if skip_if and skip_if.exists():
            write_text(stdout_path, f'[{now()}] SKIPPED existing artifact: {skip_if}\n{command}')
            write_text(stderr_path, '')
            write_text(after_path, nvidia_smi_text() or 'nvidia-smi unavailable')
            result = CommandResult(task_name, command, start_time, now(), 0, str(stdout_path),
                                   str(stderr_path), str(before_path), str(after_path), gpu_ids,
                                   batch_size, retry_count, 0.0)
            self._record(result)
            return result

        env_cmd = ['rtk', 'env', *self.env_assignments(gpu_ids), *map(str, argv)]
        monitor = GpuMonitor(gpu_ids) if monitor_gpu else None
        with stdout_path.open('w', encoding='utf-8') as stdout_f, stderr_path.open('w', encoding='utf-8') as stderr_f:
            stdout_f.write(f'[{start_time}] command={command}\n')
            stdout_f.write(f'cwd={self.repo_root}\n\n')
            stdout_f.flush()
            if monitor:
                monitor.__enter__()
            try:
                proc = subprocess.run(
                    env_cmd,
                    cwd=str(self.repo_root),
                    stdout=stdout_f,
                    stderr=stderr_f,
                    text=True,
                    timeout=timeout_sec,
                    check=False)
                return_code = proc.returncode
            except subprocess.TimeoutExpired:
                return_code = 124
                stderr_f.write('\nTIMEOUT: command exceeded timeout_sec\n')
            finally:
                if monitor:
                    monitor.__exit__(None, None, None)

        write_text(after_path, nvidia_smi_text() or 'nvidia-smi unavailable')
        stdout_tail = tail(stdout_path, 100)
        stderr_tail = tail(stderr_path, 100)
        result = CommandResult(
            task_name=task_name,
            command=command,
            start_time=start_time,
            end_time=now(),
            return_code=return_code,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            nvidia_smi_before=str(before_path),
            nvidia_smi_after=str(after_path),
            gpu_ids=gpu_ids,
            batch_size=batch_size,
            retry_count=retry_count,
            duration_sec=time.time() - start,
            failure_kind=detect_failure(stdout_tail, stderr_tail),
            stdout_tail=stdout_tail if return_code else '',
            stderr_tail=stderr_tail if return_code else '',
            peak_mem_mb=monitor.peak_mem if monitor else {},
            process_seen=monitor.process_seen if monitor else {},
        )
        self._record(result)
        return result

    def _record(self, result: CommandResult) -> None:
        payload = asdict(result)
        with self.command_log.open('a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False) + '\n')
        if result.return_code != 0:
            with self.failures_log.open('a', encoding='utf-8') as f:
                f.write(json.dumps(payload, ensure_ascii=False) + '\n')
