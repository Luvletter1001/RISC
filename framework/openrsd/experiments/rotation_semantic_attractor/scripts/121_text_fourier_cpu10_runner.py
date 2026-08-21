#!/usr/bin/env python3
"""Single CPU-only runner for the text/Fourier 10-way offline study."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_fourier_cpu10_common import (
    EXP_DIR,
    assert_cpu_only_environment,
    configure_cpu_environment,
    ensure_tree,
    resolve,
    write_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def _run_stage(args: list[str]) -> dict[str, object]:
    start = time.time()
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False)
    return {
        "cmd": [sys.executable, *args],
        "returncode": proc.returncode,
        "duration_s": round(time.time() - start, 3),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def _lock(lock_path: Path) -> int:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    return os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--exp-dir", type=Path, default=EXP_DIR)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    configure_cpu_environment()
    assert_cpu_only_environment()
    repo_root = args.repo_root.resolve()
    exp_dir = resolve(repo_root, args.exp_dir)
    ensure_tree(exp_dir)
    mpl_config = exp_dir / "logs" / "mplconfig"
    xdg_cache = exp_dir / "logs" / "xdg_cache"
    mpl_config.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
    lock_path = exp_dir / "run.lock"
    manifest_path = exp_dir / "run_manifest.json"
    try:
        lock_fd = _lock(lock_path)
    except FileExistsError:
        payload = {
            "status": "BLOCKED_LOCK_EXISTS",
            "lock_path": str(lock_path),
            "note": "已有CPU-only text-Fourier runner锁文件，拒绝重复启动。",
        }
        write_json(manifest_path, payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 3
    stages = []
    status = "PASS_CPU10_RUN"
    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode("utf-8"))
        stage_args = [
            str(SCRIPT_DIR / "119_text_fourier_cpu10_scores.py"),
            "--repo-root", str(repo_root),
            "--output-dir", str(exp_dir),
        ]
        if args.limit > 0:
            stage_args.extend(["--limit", str(args.limit)])
        stages.append(_run_stage(stage_args))
        if stages[-1]["returncode"] != 0:
            status = "FAIL_SCORE_STAGE"
            return_code = int(stages[-1]["returncode"])
        else:
            stages.append(_run_stage([
                str(SCRIPT_DIR / "120_text_fourier_cpu10_analyze.py"),
                "--repo-root", str(repo_root),
                "--exp-dir", str(exp_dir),
            ]))
            if stages[-1]["returncode"] != 0:
                status = "FAIL_ANALYSIS_STAGE"
                return_code = int(stages[-1]["returncode"])
            else:
                stages.append(_run_stage([
                    str(SCRIPT_DIR / "122_build_text_fourier_cpu10_report.py"),
                    "--repo-root", str(repo_root),
                    "--exp-dir", str(exp_dir),
                ]))
                if stages[-1]["returncode"] != 0:
                    status = "FAIL_REPORT_STAGE"
                    return_code = int(stages[-1]["returncode"])
                else:
                    return_code = 0
        payload = {
            "status": status,
            "exp_dir": str(exp_dir),
            "limit": args.limit,
            "cpu_only": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", ""),
            "detector_run": False,
            "training_run": False,
            "final_logits_modified": False,
            "stages": stages,
        }
        write_json(manifest_path, payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return return_code
    finally:
        os.close(lock_fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
