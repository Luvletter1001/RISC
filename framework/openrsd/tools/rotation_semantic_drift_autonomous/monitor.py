#!/usr/bin/env python
"""Process/GPU monitoring for autonomous overnight."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple


def gpu_utilization(gpu_id: int) -> Tuple[float, float]:
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
             '--format=csv,noheader,nounits'], text=True)
        for line in out.strip().splitlines():
            idx, mem, util = [x.strip() for x in line.split(',')]
            if int(idx) == gpu_id:
                return float(mem), float(util)
    except Exception:
        pass
    return -1.0, -1.0


def tail_file(path: Path, n: int = 20) -> str:
    if not path.exists():
        return ''
    try:
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
        return '\n'.join(lines[-n:])
    except Exception as exc:
        return str(exc)


def check_hang(
    log_path: Path,
    output_path: Optional[Path],
    last_log_size: int,
    last_out_mtime: float,
    stall_minutes: int = 30,
    started_at: float = 0,
) -> Tuple[bool, str, int, float]:
    now = time.time()
    if started_at and (now - started_at) < stall_minutes * 60:
        return False, 'warming_up', last_log_size, last_out_mtime
    new_log = log_path.stat().st_size if log_path.exists() else 0
    new_out = output_path.stat().st_mtime if output_path and output_path.exists() else 0
    if new_log > last_log_size or new_out > last_out_mtime:
        return False, 'active', new_log, new_out
    return True, f'no growth for {stall_minutes}min', last_log_size, last_out_mtime
