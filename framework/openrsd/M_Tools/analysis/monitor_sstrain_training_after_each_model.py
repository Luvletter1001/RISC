#!/usr/bin/env python3
"""Low-token stage monitor for ss_train retraining.

The monitor is intentionally quiet. It does not tail logs while a process is
running. For external processes it polls at a coarse interval, then performs
one post-check after the process exits.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


MODEL_ORDER = ["redet", "retinanet_msrr", "retinanet_amp", "rtmdet_l", "h2rbox_v2", "h2rbox"]
BLOCKED_MODELS = {
    "h2rbox_v2": "training config missing; only eval adapter found",
    "h2rbox": "exact H2RBox 3xMS training config missing",
}
ERR_PAT = re.compile(
    r"Traceback|RuntimeError|CUDA out of memory|NCCL|loss is nan|\bNaN\b|killed|segmentation fault|FileNotFoundError|KeyError",
    re.IGNORECASE,
)


def now() -> str:
    return datetime.now().strftime("%F %T")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def run_quiet(cmd: list[str], cwd: Path, timeout: int = 30) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except Exception as exc:  # noqa: BLE001
        return 1, "", repr(exc)


def pgrep_training(repo_root: Path) -> list[dict[str, Any]]:
    rc, out, _ = run_quiet(["rtk", "pgrep", "-af", "run_sstrain_retrain_12angle_suite.py|tools/train.py"], repo_root)
    rows = []
    if rc not in (0, 1):
        return rows
    self_pid = os.getpid()
    for line in out.splitlines():
        if "monitor_sstrain_training_after_each_model.py" in line:
            continue
        parts = line.strip().split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == self_pid:
            continue
        cmd = parts[1] if len(parts) > 1 else ""
        if str(repo_root) not in cmd and "OpenRSD" not in cmd:
            continue
        rows.append({"pid": pid, "cmd": cmd, "model": infer_model(cmd)})
    return rows


def infer_model(cmd: str) -> str:
    for m in MODEL_ORDER:
        if m in cmd:
            return m
    return "unknown"


def proc_exists(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def newest_log(work_dir: Path, model: str) -> Path | None:
    candidates = []
    candidates.extend((work_dir / "logs" / f"train_{model}").glob("*/stdout.log"))
    candidates.extend((work_dir / "train" / model).glob("*.log"))
    candidates.extend((work_dir / "train" / model).glob("**/*.log"))
    candidates = [p for p in candidates if p.exists()]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def tail_lines(path: Path, n: int = 200) -> str:
    if not path or not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])


def generated_config_ok(path: Path, train_root: Path) -> tuple[bool, bool]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    train_only = str(train_root) in text and "trainval/annfiles" not in text and "angle_sweep_val" not in text.split("train_dataloader", 1)[-1].split("val_dataloader", 1)[0]
    forbidden_ckpt = any(x in text for x in ["load_from = '/data1/zcy/OpenRSD/weights", 'load_from = "/data1/zcy/OpenRSD/weights'])
    return train_only, forbidden_ckpt


def post_check(args: argparse.Namespace, model: str, return_code: int | None = None, pid: int | None = None, cmd: str = "") -> dict[str, Any]:
    if model in BLOCKED_MODELS:
        rec = base_record(model, "BLOCKED", pid, return_code, "", "", BLOCKED_MODELS[model], "next")
        return rec
    train_dir = args.work_dir / "train" / model
    save_ckpt = args.save_ckpt_dir / model / "latest.pth"
    latest = save_ckpt if save_ckpt.exists() else train_dir / "latest.pth"
    log = newest_log(args.work_dir, model)
    tail = tail_lines(log, 200) if log else ""
    err = ERR_PAT.search(tail)
    cfg = args.work_dir / "generated_configs" / f"{model}_ss_train.py"
    train_only, forbidden_ckpt = generated_config_ok(cfg, args.train_root)
    status = "FAILED"
    issue = ""
    if forbidden_ckpt or not train_only:
        status = "FAILED_FOR_LEAKAGE_RISK"
        issue = "config audit failed"
    elif latest.exists() and not err and (return_code in (0, None)):
        status = "TRAIN_DONE_EVAL_PENDING"
    elif latest.exists():
        status = "PARTIAL"
        issue = err.group(0) if err else "nonzero rc or unclear end"
    else:
        issue = err.group(0) if err else "checkpoint missing"
    return base_record(model, status, pid, return_code, str(latest) if latest.exists() else "", str(log or ""), issue, "next")


def base_record(model: str, status: str, pid: int | None, rc: int | None, ckpt: str, log: str, issue: str, next_action: str) -> dict[str, Any]:
    return {
        "model": model,
        "status": status,
        "pid": pid,
        "start_time": "",
        "end_time": now(),
        "return_code": rc,
        "checkpoint_path": ckpt,
        "log_path": log,
        "last_error_summary": issue,
        "next_action": next_action,
    }


def load_status(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"updated_at": now(), "records": []}


def upsert(status: dict[str, Any], rec: dict[str, Any]) -> None:
    records = [r for r in status.get("records", []) if r.get("model") != rec["model"]]
    records.append(rec)
    status["records"] = sorted(records, key=lambda r: MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 999)
    status["updated_at"] = now()


def write_md(args: argparse.Namespace, status: dict[str, Any]) -> None:
    lines = ["# ss_train Training Monitor", "", f"- updated_at: `{status.get('updated_at')}`", "", "| model | status | checkpoint | log | issue |", "|---|---|---|---|---|"]
    by_model = {r["model"]: r for r in status.get("records", [])}
    for m in MODEL_ORDER:
        r = by_model.get(m, {"status": "PENDING", "checkpoint_path": "", "log_path": "", "last_error_summary": ""})
        lines.append(f"| {m} | {r.get('status','')} | `{r.get('checkpoint_path','')}` | `{r.get('log_path','')}` | {r.get('last_error_summary','')} |")
    write_text(args.result_md_dir / "monitor_sstrain_training_status.md", "\n".join(lines))


def write_phase_summary(args: argparse.Namespace, status: dict[str, Any]) -> None:
    lines = ["# ss_train Training Phase Summary", "", f"- generated_at: `{now()}`", "", "| model | status | checkpoint | log | train_only_ss_train | forbidden_dota_ckpt | eval_ready | reason |", "|---|---|---|---|---:|---:|---:|---|"]
    by_model = {r["model"]: r for r in status.get("records", [])}
    for m in MODEL_ORDER:
        r = by_model.get(m, {"status": "PENDING", "checkpoint_path": "", "log_path": "", "last_error_summary": ""})
        cfg = args.work_dir / "generated_configs" / f"{m}_ss_train.py"
        train_only, forbidden = generated_config_ok(cfg, args.train_root) if cfg.exists() else (False, False)
        eval_ready = r.get("status") in {"DONE", "TRAIN_DONE_EVAL_PENDING"}
        lines.append(f"| {m} | {r.get('status','')} | `{r.get('checkpoint_path','')}` | `{r.get('log_path','')}` | {train_only} | {forbidden} | {eval_ready} | {r.get('last_error_summary','')} |")
    lines.extend(["", "## Next Eval Command", "", "```bash", "CUDA_VISIBLE_DEVICES=6,7 python M_Tools/analysis/run_sstrain_retrain_12angle_suite.py --repo-root /data1/zcy/OpenRSD --train-root /data1/zcy/OpenRSD/data/DOTA1_1024_500/ss_train --angle-root /data1/zcy/OpenRSD/data/DOTA1_1024_500/angle_sweep_val/realistic --work-dir /data1/zcy/OpenRSD/work_dirs/ss_train_retrain_12angle_no_tta --result-md-dir /data1/zcy/OpenRSD/resultmd --save-ckpt-dir /data1/zcy/OpenRSD/results/ss_train_retrain --gpu-ids 6,7 --mode eval --models all --eval-latest --no-tta", "```"])
    write_text(args.result_md_dir / "summary_sstrain_training_phase.md", "\n".join(lines))


def already_done(args: argparse.Namespace, model: str) -> bool:
    if model in BLOCKED_MODELS:
        return True
    return (args.save_ckpt_dir / model / "latest.pth").exists()


def launch_model(args: argparse.Namespace, model: str) -> tuple[int, int]:
    log_dir = args.work_dir / "monitor_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out = open(log_dir / f"{model}.stdout.log", "a", encoding="utf-8")
    err = open(log_dir / f"{model}.stderr.log", "a", encoding="utf-8")
    cmd = [
        "rtk", "env",
        "PYTHONNOUSERSITE=1",
        "MPLCONFIGDIR=/tmp/mplconfig",
        f"PYTHONPATH={args.repo_root}:{args.repo_root / 'tools'}",
        "NCCL_P2P_DISABLE=1",
        "NCCL_IB_DISABLE=1",
        "CUDA_VISIBLE_DEVICES=6,7",
        str(args.python_bin),
        str(args.repo_root / "M_Tools/analysis/run_sstrain_retrain_12angle_suite.py"),
        "--repo-root", str(args.repo_root),
        "--train-root", str(args.train_root),
        "--angle-root", str(args.angle_root),
        "--work-dir", str(args.work_dir),
        "--result-md-dir", str(args.result_md_dir),
        "--save-ckpt-dir", str(args.save_ckpt_dir),
        "--gpu-ids", "6,7",
        "--mode", "train",
        "--only-model", model,
        "--batch-size", "8",
        "--num-workers", "2",
        "--resume",
        "--dist-port-base", str(args.dist_port_base),
        "--distributed-launch", "torchrun",
    ]
    proc = subprocess.Popen(cmd, cwd=str(args.repo_root), stdout=out, stderr=err, text=True, start_new_session=True)
    return proc.pid, proc.wait()


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.train_root = args.train_root.resolve()
    args.angle_root = args.angle_root.resolve()
    args.work_dir = args.work_dir.resolve()
    args.result_md_dir = args.result_md_dir.resolve()
    args.save_ckpt_dir = args.save_ckpt_dir.resolve()
    status_path = args.work_dir / "results/training_status.json"
    status = load_status(status_path)

    running = pgrep_training(args.repo_root)
    top = next((r for r in running if "run_sstrain_retrain_12angle_suite.py" in r["cmd"]), None)
    if top:
        rec = base_record(top["model"], "RUNNING", top["pid"], None, "", "", "external process running", "wait")
        rec["command"] = top["cmd"]
        upsert(status, rec)
        write_json(status_path, status)
        write_md(args, status)
        while proc_exists(top["pid"]):
            time.sleep(args.poll_interval_sec)
        for m in MODEL_ORDER:
            if m in BLOCKED_MODELS or (args.work_dir / "train" / m).exists() or (args.save_ckpt_dir / m / "latest.pth").exists():
                upsert(status, post_check(args, m, None, top["pid"], top["cmd"]))
        write_json(status_path, status)
        write_md(args, status)

    for model in MODEL_ORDER:
        if model in BLOCKED_MODELS:
            upsert(status, post_check(args, model))
            continue
        recs = {r["model"]: r for r in status.get("records", [])}
        if already_done(args, model) or recs.get(model, {}).get("status") in {"DONE", "TRAIN_DONE_EVAL_PENDING"}:
            upsert(status, post_check(args, model))
            continue
        pid, rc = launch_model(args, model)
        upsert(status, base_record(model, "RUNNING", pid, None, "", "", "started by monitor", "wait"))
        write_json(status_path, status)
        write_md(args, status)
        upsert(status, post_check(args, model, rc, pid))
        write_json(status_path, status)
        write_md(args, status)

    write_phase_summary(args, status)
    print(f"monitor_done status_json={status_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path("/data1/zcy/OpenRSD"))
    p.add_argument("--train-root", type=Path, default=Path("/data1/zcy/OpenRSD/data/DOTA1_1024_500/ss_train"))
    p.add_argument("--angle-root", type=Path, default=Path("/data1/zcy/OpenRSD/data/DOTA1_1024_500/angle_sweep_val/realistic"))
    p.add_argument("--work-dir", type=Path, default=Path("/data1/zcy/OpenRSD/work_dirs/ss_train_retrain_12angle_no_tta"))
    p.add_argument("--result-md-dir", type=Path, default=Path("/data1/zcy/OpenRSD/resultmd"))
    p.add_argument("--save-ckpt-dir", type=Path, default=Path("/data1/zcy/OpenRSD/results/ss_train_retrain"))
    p.add_argument("--python-bin", type=Path, default=Path("/data/zcy/anaconda3/envs/openrsd/bin/python"))
    p.add_argument("--poll-interval-sec", type=int, default=1800)
    p.add_argument("--dist-port-base", type=int, default=35600)
    return p.parse_args()


if __name__ == "__main__":
    main()
