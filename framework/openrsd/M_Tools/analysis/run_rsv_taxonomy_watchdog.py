#!/usr/bin/env python3
"""Strict queue + retry watchdog for RSV taxonomy pipeline (2500 cohort).

Runs r0 -> r1 -> r2 -> r3 -> r4 sequentially. Restarts failed steps.
Outer shell should re-invoke this script if the process dies.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path("/data1/zcy/OpenRSD")
WORK = REPO / "work_dirs/exp_rotation_gt_shift_taxonomy_20260527"
RESULT_MD = REPO / "resultmd/exp_rotation_gt_shift_taxonomy_20260527"
PYTHON = Path("/data/zcy/anaconda3/envs/openrsd/bin/python")
SUITE = REPO / "M_Tools/analysis/run_rsv_taxonomy_suite.py"

STEPS = ("r0", "r1", "r2", "r3", "r4")
MAX_ATTEMPTS = 8
RETRY_SLEEP_SEC = 180


def log(msg: str, log_path: Path) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def acquire_lock(lock_path: Path) -> int | None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except OSError:
        os.close(fd)
        return None


def cohort_ok(work: Path, n: int) -> bool:
    p = work / f"ftable_cohort_{n}.csv"
    if not p.exists():
        return False
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    return len(lines) >= n + 1  # header + n rows


def import_suite():
    sys.path.insert(0, str(REPO))
    from M_Tools.analysis import run_rsv_taxonomy_suite as suite
    return suite


def validate_step(step: str, work: Path, cohort_size: int) -> tuple[bool, str]:
    suite = import_suite()
    if step == "r0":
        if cohort_ok(work, cohort_size):
            return True, f"cohort_{cohort_size}.csv ok"
        return False, "cohort file missing or short"

    tiles = [r["tile_id"] for r in suite.read_csv(work / f"ftable_cohort_{cohort_size}.csv")]

    if step == "r1":
        ok, st = suite.validate_r1_complete(work, tiles)
        return ok, f"cache {st['total']}/{st['expected']}"

    if step == "r2":
        ok, st = suite.validate_r2_complete(work, tiles)
        if ok:
            return True, (
                f"tile_angle {st['n_rows']} rows "
                f"({st['n_tiles_done']}/{st['expected']} tiles)"
            )
        if st.get("n_tiles_done", 0) > 0:
            return False, (
                f"incomplete r2: {st['n_tiles_done']}/{st['expected']} tiles, "
                f"{st['n_rows']}/{st['expected_rows']} rows"
            )
        return False, "missing tile_angle summary"

    if step == "r3":
        need = work / "ftable_case_summary_tile.csv"
        if need.exists() and need.stat().st_size > 100:
            return True, str(need)
        return False, "missing tile summary"

    if step == "r4":
        need = RESULT_MD / "fres_010_rsv_taxonomy_summary.md"
        if need.exists() and need.stat().st_size > 50:
            return True, str(need)
        return False, "missing report md"

    return False, f"unknown step {step}"


def run_step(
    step: str,
    cohort_size: int,
    gpu_ids: str,
    batch_size: int,
    log_path: Path,
    *,
    r2_workers: int = 512,
    force: bool = False,
) -> int:
    cmd = [
        str(PYTHON), str(SUITE),
        "--repo-root", str(REPO),
        "--work-dir", str(WORK),
        "--result-md-dir", str(RESULT_MD),
        "--cohort-size", str(cohort_size),
        "--infer-gpu-ids", gpu_ids,
        "--infer-batch-size", str(batch_size),
        "--r2-workers", str(r2_workers),
        "--step", step,
        "--resume",
    ]
    if step == "r0":
        cmd.append("--extend-cohort")
    if force:
        cmd.append("--force")
    log(f"exec: {' '.join(cmd)}", log_path)
    proc = subprocess.run(cmd, cwd=str(REPO))
    return int(proc.returncode)


def step_marker(work: Path, step: str) -> Path:
    return work / f".done_{step}"


def pipeline_complete(work: Path) -> bool:
    return (work / ".done_pipeline").exists()


def run_pipeline(
    cohort_size: int,
    gpu_ids: str,
    batch_size: int,
    log_path: Path,
    *,
    r2_workers: int = 512,
) -> bool:
    for step in STEPS:
        marker = step_marker(WORK, step)
        force_step = False
        if marker.exists():
            ok, msg = validate_step(step, WORK, cohort_size)
            if ok:
                log(f"{step} skip (valid): {msg}", log_path)
                continue
            log(f"{step} marker stale, re-run: {msg}", log_path)
            marker.unlink(missing_ok=True)
            force_step = True

        for attempt in range(1, MAX_ATTEMPTS + 1):
            log(f"{step} attempt {attempt}/{MAX_ATTEMPTS}", log_path)
            rc = run_step(
                step,
                cohort_size,
                gpu_ids,
                batch_size,
                log_path,
                r2_workers=r2_workers,
                force=force_step,
            )
            ok, msg = validate_step(step, WORK, cohort_size)
            if rc == 0 and ok:
                meta = {"status": "OK", "validated": msg, "attempt": attempt}
                step_marker(WORK, step).write_text(
                    json.dumps(meta, indent=2), encoding="utf-8",
                )
                log(f"{step} OK: {msg}", log_path)
                break
            log(f"{step} FAIL rc={rc} valid={ok} msg={msg}", log_path)
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_SLEEP_SEC)
        else:
            log(f"{step} exhausted retries", log_path)
            return False

    (WORK / ".done_pipeline").write_text(
        json.dumps({"status": "OK", "cohort_size": cohort_size}, indent=2),
        encoding="utf-8",
    )
    log("pipeline complete", log_path)
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RSV taxonomy watchdog")
    p.add_argument("--cohort-size", type=int, default=2500)
    p.add_argument("--infer-gpu-ids", default="4,5,6,7")
    p.add_argument("--infer-batch-size", type=int, default=48)
    p.add_argument(
        "--r2-workers",
        type=int,
        default=512,
        help="Parallel workers for r2 (passed to suite, default 512)",
    )
    p.add_argument("--work-dir", type=Path, default=WORK)
    p.add_argument("--once", action="store_true",
                   help="Single pass (for outer shell loop)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    global WORK, RESULT_MD
    WORK = args.work_dir.resolve()
    RESULT_MD = (REPO / "resultmd/exp_rotation_gt_shift_taxonomy_20260527").resolve()
    log_path = WORK / "logs" / "watchdog.log"
    lock_path = WORK / "watchdog.lock"

    lock_fd = acquire_lock(lock_path)
    if lock_fd is None:
        log("another watchdog holds lock; exit 0", log_path)
        return 0

    try:
        if pipeline_complete(WORK):
            log("pipeline already done", log_path)
            return 0

        ok = run_pipeline(
            args.cohort_size,
            args.infer_gpu_ids,
            args.infer_batch_size,
            log_path,
            r2_workers=args.r2_workers,
        )
        return 0 if ok else 1
    except Exception:
        log(traceback.format_exc(), log_path)
        return 1
    finally:
        if lock_fd is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)


if __name__ == "__main__":
    raise SystemExit(main())
