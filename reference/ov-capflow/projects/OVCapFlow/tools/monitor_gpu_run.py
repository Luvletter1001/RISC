#!/usr/bin/env python3
"""Monitor a long-running multi-GPU training job and persist JSONL evidence."""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence


_GPU_QUERY_FIELDS = (
    'index,utilization.gpu,memory.used,memory.total,power.draw')
_FATAL_PATTERNS = (
    ('cuda_oom', re.compile(
        r'(?:cuda\s+out\s+of\s+memory|outofmemoryerror)', re.IGNORECASE)),
    ('nccl', re.compile(
        r'(?:distbackenderror|watchdog caught collective|'
        r'nccl\s+(?:error|failure|timeout)|'
        r'nccl[^\n]{0,120}communicator[^\n]{0,40}(?:abort|fail)|'
        r'nccl[^\n]{0,120}(?:operation|collective)'
        r'[^\n]{0,40}(?:timeout|abort|fail))', re.IGNORECASE)),
    ('dead_rank', re.compile(
        r'(?:childfailederror|rank\s*\d+.*(?:exit|fail|dead))',
        re.IGNORECASE)),
    ('non_finite', re.compile(
        r'(?<![A-Za-z])(?:nan|[+-]?inf)(?![A-Za-z])', re.IGNORECASE)),
    ('traceback', re.compile(r'traceback \(most recent call last\)',
                             re.IGNORECASE)),
    ('cuda_error', re.compile(r'(?:cuda error|cudaerror)', re.IGNORECASE)),
)


def _as_float(value: str, field: str) -> float:
    try:
        return float(value.strip())
    except ValueError as exc:
        raise ValueError('Invalid {} value: {!r}'.format(field, value)) from exc


def parse_nvidia_smi_csv(
        text: str, expected_indices: Sequence[int]) -> List[Dict[str, float]]:
    """Parse the no-header/nounits GPU query used by the monitor."""
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        columns = [item.strip() for item in line.split(',')]
        if len(columns) != 5:
            raise ValueError(
                'Expected five nvidia-smi columns, got {}: {!r}'.format(
                    len(columns), line))
        index = int(columns[0])
        rows.append({
            'index': index,
            'utilization_percent': _as_float(columns[1], 'utilization'),
            'memory_used_mib': _as_float(columns[2], 'memory.used'),
            'memory_total_mib': _as_float(columns[3], 'memory.total'),
            'power_draw_w': _as_float(columns[4], 'power.draw'),
        })

    actual = [int(row['index']) for row in rows]
    expected = [int(index) for index in expected_indices]
    if actual != expected:
        raise ValueError(
            'GPU indices do not match: expected {}, got {}'.format(
                expected, actual))
    return rows


def query_gpu_stats(gpu_indices: Sequence[int]) -> List[Dict[str, float]]:
    """Query physical GPU indices without depending on CUDA remapping."""
    command = [
        'nvidia-smi',
        '--id={}'.format(','.join(str(index) for index in gpu_indices)),
        '--query-gpu={}'.format(_GPU_QUERY_FIELDS),
        '--format=csv,noheader,nounits',
    ]
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=15)
    return parse_nvidia_smi_csv(completed.stdout, gpu_indices)


def detect_fatal_pattern(text: str) -> Optional[str]:
    """Return the stable label of the first fatal training-log signature."""
    for label, pattern in _FATAL_PATTERNS:
        if pattern.search(text):
            return label
    return None


def summarize_samples(samples: Iterable[dict]) -> dict:
    """Summarize utilization and memory with robust per-card medians."""
    materialized = list(samples)
    by_gpu = {}  # type: Dict[int, List[dict]]
    for sample in materialized:
        for gpu in sample.get('gpus', []):
            by_gpu.setdefault(int(gpu['index']), []).append(gpu)

    gpu_summary = {}
    for index in sorted(by_gpu):
        rows = by_gpu[index]
        gpu_summary[str(index)] = {
            'median_utilization_percent': statistics.median(
                row['utilization_percent'] for row in rows),
            'median_memory_used_mib': statistics.median(
                row['memory_used_mib'] for row in rows),
            'memory_total_mib': rows[-1]['memory_total_mib'],
            'median_power_draw_w': statistics.median(
                row['power_draw_w'] for row in rows),
        }
    return {'sample_count': len(materialized), 'gpus': gpu_summary}


class IncrementalLogReader:
    """Read only bytes appended since the previous monitor sample."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.offset = 0
        self.inode = None  # type: Optional[int]

    def __call__(self) -> str:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return ''
        if self.inode != stat.st_ino or stat.st_size < self.offset:
            self.offset = 0
            self.inode = stat.st_ino
        with self.path.open('r', encoding='utf-8', errors='replace') as stream:
            stream.seek(self.offset)
            text = stream.read()
            self.offset = stream.tell()
        return text


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _proc_parent_map() -> Dict[int, int]:
    parents = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / 'stat').read_text().split()
            parents[int(fields[0])] = int(fields[3])
        except (FileNotFoundError, IndexError, PermissionError, ValueError):
            continue
    return parents


class ProcessTreeGuard:
    """Remember launcher descendants so reparented ranks remain observable."""

    def __init__(self, root_pid: int):
        self.known_pids = {int(root_pid)}

    def __call__(self) -> bool:
        parents = _proc_parent_map()
        changed = True
        while changed:
            changed = False
            for pid, parent in parents.items():
                if parent in self.known_pids and pid not in self.known_pids:
                    self.known_pids.add(pid)
                    changed = True
        return any(_pid_exists(pid) for pid in self.known_pids)


def _append_record(stream, record: dict) -> None:
    stream.write(json.dumps(record, sort_keys=True) + '\n')
    stream.flush()


def _default_now() -> str:
    return datetime.now().astimezone().isoformat()


def run_monitor(
        pid: int,
        gpu_indices: Sequence[int],
        interval: int,
        train_log: Path,
        output: Path,
        max_samples: Optional[int] = None,
        query_fn: Callable[[Sequence[int]], List[dict]] = query_gpu_stats,
        alive_fn: Optional[Callable[[], bool]] = None,
        log_reader: Optional[Callable[[], str]] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], str] = _default_now) -> int:
    """Sample until interrupted, the training tree dies, or a fatal log appears."""
    if interval <= 0:
        raise ValueError('interval must be positive')
    if max_samples is not None and max_samples <= 0:
        raise ValueError('max_samples must be positive')
    gpu_indices = [int(index) for index in gpu_indices]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    alive_fn = alive_fn or ProcessTreeGuard(pid)
    log_reader = log_reader or IncrementalLogReader(Path(train_log))
    samples = []
    reason = 'interrupted'
    exit_code = 0

    with output.open('a', encoding='utf-8') as stream:
        try:
            while True:
                alive = bool(alive_fn())
                appended_log = log_reader()
                fatal_pattern = detect_fatal_pattern(appended_log)
                sample = {
                    'type': 'sample',
                    'timestamp': now_fn(),
                    'pid': int(pid),
                    'process_group_alive': alive,
                    'fatal_pattern': fatal_pattern,
                    'gpus': query_fn(gpu_indices),
                }
                samples.append(sample)
                _append_record(stream, sample)

                if fatal_pattern is not None:
                    reason = fatal_pattern
                    exit_code = 2
                    break
                if not alive:
                    reason = 'all_ranks_dead'
                    exit_code = 3
                    break
                if max_samples is not None and len(samples) >= max_samples:
                    reason = 'max_samples'
                    break
                sleep_fn(interval)
        except KeyboardInterrupt:
            reason = 'interrupted'

        summary = {'type': 'summary', 'timestamp': now_fn(), 'reason': reason}
        summary.update(summarize_samples(samples))
        _append_record(stream, summary)
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True,
                        help='torchrun launcher PID')
    parser.add_argument('--gpu-indices', type=int, nargs='+', required=True,
                        help='physical GPU indices to sample')
    parser.add_argument('--interval', type=int, default=30,
                        help='sampling interval in seconds (default: 30)')
    parser.add_argument('--train-log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-samples', type=int, default=None,
                        help=argparse.SUPPRESS)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run_monitor(
        pid=args.pid,
        gpu_indices=args.gpu_indices,
        interval=args.interval,
        train_log=args.train_log,
        output=args.output,
        max_samples=args.max_samples)


if __name__ == '__main__':
    sys.exit(main())
