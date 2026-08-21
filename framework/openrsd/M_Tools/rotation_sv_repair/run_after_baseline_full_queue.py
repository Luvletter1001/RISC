#!/usr/bin/env python3
"""Run full repair experiments after preserving the already completed baseline.

This queue intentionally excludes `baseline`. It protects fres_010/ftable_010
before each step and restores them after each step, so later repair/report
experiments cannot accidentally overwrite the full baseline evidence.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO = Path("/data1/zcy/OpenRSD")
PY = Path("/data/zcy/anaconda3/envs/openrsd/bin/python")
WORK = REPO / "work_dirs/exp_rotation_sv_repair_20260524"
RESULT = REPO / "resultmd/exp_rotation_sv_repair_20260524"
LOGS = WORK / "logs"
TABLES = RESULT / "tables"
LOCK = WORK / "baseline_lock"

EXPS = [
    "postprocess",
    "embedding",
    "prompt",
    "ovd_teacher_student",
    "orbit",
    "adapter",
    "official_ap",
    "true_sv",
    "oracle",
    "ablation",
    "stats",
    "audit",
]

TIMEOUTS = {
    "adapter": 60 * 60 * 4,
    "ovd_teacher_student": 60 * 60 * 3,
    "orbit": 60 * 60 * 3,
}
DEFAULT_TIMEOUT = 60 * 60 * 2


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run(cmd: list[str], log_path: Path, env: dict[str, str], timeout: int) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# started {now()}\n")
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(REPO),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
            f.write(f"\n# exit {proc.returncode} at {now()}\n")
            return int(proc.returncode)
        except subprocess.TimeoutExpired:
            f.write(f"\n# TIMEOUT after {timeout}s at {now()}\n")
            return 124


def query_gpus() -> list[tuple[int, int, int]]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return [(8, 0, 0), (9, 0, 0)]
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return rows


def pick_gpus() -> str:
    rows = {idx: (mem, util) for idx, mem, util in query_gpus()}
    for pair in ((8, 9), (6, 7), (4, 5)):
        if all(i in rows and rows[i][0] < 1000 and rows[i][1] < 50 for i in pair):
            return f"{pair[0]},{pair[1]}"
    for idx in (8, 9, 6, 7, 4, 5):
        if idx in rows and rows[idx][0] < 1000:
            return str(idx)
    # Last resort; still avoid 0-3 unless every preferred GPU is unavailable.
    best = sorted(rows, key=lambda i: rows[i][0])[0] if rows else 8
    return str(best)


def base_env(gpus: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "NCCL_P2P_DISABLE": "1",
            "NCCL_IB_DISABLE": "1",
            "PYTHONNOUSERSITE": "1",
            "MPLCONFIGDIR": "/tmp/mplconfig",
            "PYTHONPATH": f"{REPO}:{REPO / 'tools'}",
            "CUDA_VISIBLE_DEVICES": gpus,
        }
    )
    return env


def suite_cmd(mode: str, exp: str, gpus: str, extra: list[str] | None = None) -> list[str]:
    cmd = [
        str(PY),
        "M_Tools/rotation_sv_repair/run_rotation_sv_repair_suite.py",
        "--repo-root",
        str(REPO),
        "--work-dir",
        str(WORK),
        "--result-dir",
        str(RESULT),
        "--gpu-ids",
        gpus,
        "--mode",
        mode,
        "--exp",
        exp,
        "--resume",
        "--force",
    ]
    if exp == "adapter" or mode == "full":
        cmd.extend(["--train-adapter", "--distill"])
    if extra:
        cmd.extend(extra)
    return cmd


def preserve_baseline() -> dict[str, str]:
    LOCK.mkdir(parents=True, exist_ok=True)
    files = {
        "fres_010": RESULT / "fres_010_baseline_reconfirm.md",
        "ftable_010": TABLES / "ftable_010_baseline_reconfirm.csv",
    }
    locked = {}
    for key, src in files.items():
        dst = LOCK / src.name
        if src.exists():
            shutil.copy2(src, dst)
            locked[key] = str(dst)
    return locked


def restore_baseline() -> None:
    pairs = [
        (LOCK / "fres_010_baseline_reconfirm.md", RESULT / "fres_010_baseline_reconfirm.md"),
        (LOCK / "ftable_010_baseline_reconfirm.csv", TABLES / "ftable_010_baseline_reconfirm.csv"),
    ]
    for src, dst in pairs:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def write_summary(rows: list[dict], gpus: str) -> None:
    path = RESULT / "fres_099_after_baseline_full_queue.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# After-Baseline Full Queue\n\n")
        f.write(f"- **status:** `DONE`\n")
        f.write(f"- **generated:** {now()}\n")
        f.write(f"- **gpu_ids:** `{gpus}`\n")
        f.write(f"- **baseline_preserved_from:** `{LOCK}`\n")
        f.write(f"- **driver_log:** `{LOGS / 'after_baseline_full_queue_driver.log'}`\n\n")
        f.write("| exp | full_exit | debug_exit | full_log | debug_log |\n")
        f.write("|:---|---:|---:|:---|:---|\n")
        for r in rows:
            f.write(
                f"| {r['exp']} | {r['full_exit']} | {r.get('debug_exit', '')} | "
                f"`{r['full_log']}` | `{r.get('debug_log', '')}` |\n"
            )
    (WORK / "after_baseline_full_queue_status.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    RESULT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    driver = LOGS / "after_baseline_full_queue_driver.log"
    gpus = pick_gpus()
    env = base_env(gpus)
    rows: list[dict] = []
    with open(driver, "w", encoding="utf-8") as d:
        d.write(f"START {now()}\n")
        d.write(f"GPUS {gpus}\n")
        d.write(f"BASELINE_LOCK {preserve_baseline()}\n")
        d.flush()

    for exp in EXPS:
        restore_baseline()
        full_log = LOGS / f"after_baseline_full_{exp}.log"
        rc = run(suite_cmd("full", exp, gpus), full_log, env, TIMEOUTS.get(exp, DEFAULT_TIMEOUT))
        row = {"exp": exp, "full_exit": rc, "full_log": str(full_log)}
        if rc != 0:
            restore_baseline()
            debug_gpus = gpus.split(",")[0]
            debug_env = base_env(debug_gpus)
            debug_log = LOGS / f"after_baseline_debug_{exp}.log"
            debug_rc = run(
                suite_cmd(
                    "debug",
                    exp,
                    debug_gpus,
                    ["--max-images", "16", "--debug-small", "--no-train"],
                ),
                debug_log,
                debug_env,
                60 * 45,
            )
            row["debug_exit"] = debug_rc
            row["debug_log"] = str(debug_log)
        rows.append(row)
        restore_baseline()
        with open(driver, "a", encoding="utf-8") as d:
            d.write(json.dumps(row, ensure_ascii=False) + "\n")
            d.flush()

    restore_baseline()
    # Final verdict refresh uses full mode but only the verdict step.
    verdict_log = LOGS / "after_baseline_full_verdict_refresh.log"
    verdict_rc = run(suite_cmd("full", "verdict", gpus), verdict_log, env, 60 * 20)
    rows.append({"exp": "verdict", "full_exit": verdict_rc, "full_log": str(verdict_log)})
    restore_baseline()
    write_summary(rows, gpus)
    with open(driver, "a", encoding="utf-8") as d:
        d.write(f"DONE {now()}\n")
    return 0 if all(r["full_exit"] == 0 or r.get("debug_exit") == 0 for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
