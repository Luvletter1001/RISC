#!/usr/bin/env python
"""Audit why the GPU6/GPU7 RankDelta queue is still waiting.

This script is non-experimental.  It only reads GPU occupancy and process
metadata, then writes a compact status artifact for the paper workflow.  It does
not launch, kill, or modify any process.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_OUT_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "gpu67_wait_blocker_audit.json")
DEFAULT_OUT_MD = Path(
    "resultmd/exp_p4_scale_semantic_validation/"
    "fstatus_20260621_gpu67_rankdelta_wait_blocker.md")


def run_text(command: list[str]) -> str:
    proc = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout


def _parse_csv_rows(text: str) -> list[list[str]]:
    rows = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if parts and any(parts):
            rows.append(parts)
    return rows


def query_gpu_rows() -> dict[str, dict[str, Any]]:
    text = run_text([
        "nvidia-smi",
        "--query-gpu=index,pci.bus_id,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    out: dict[str, dict[str, Any]] = {}
    for parts in _parse_csv_rows(text):
        if len(parts) < 5:
            continue
        idx, bus_id, mem_used, mem_total, util = parts[:5]
        out[idx] = {
            "index": int(idx),
            "bus_id": bus_id,
            "memory_used_mib": int(float(mem_used)),
            "memory_total_mib": int(float(mem_total)),
            "utilization_gpu_pct": int(float(util)),
            "processes": [],
        }
    return out


def query_compute_rows() -> list[dict[str, Any]]:
    text = run_text([
        "nvidia-smi",
        "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    rows = []
    for parts in _parse_csv_rows(text):
        if len(parts) < 4:
            continue
        bus_id, pid, process_name, used_memory = parts[:4]
        rows.append({
            "gpu_bus_id": bus_id,
            "pid": int(pid),
            "process_name": process_name,
            "used_memory_mib": int(float(used_memory)),
        })
    return rows


def query_process_rows(pids: list[int]) -> dict[int, dict[str, Any]]:
    if not pids:
        return {}
    text = run_text([
        "ps",
        "-p",
        ",".join(str(pid) for pid in sorted(set(pids))),
        "-o",
        "pid=,ppid=,user=,etimes=,cmd=",
    ])
    out: dict[int, dict[str, Any]] = {}
    for line in text.splitlines():
        parts = line.strip().split(maxsplit=4)
        if len(parts) < 5:
            continue
        pid, ppid, user, etimes, cmd = parts
        out[int(pid)] = {
            "pid": int(pid),
            "ppid": int(ppid),
            "user": user,
            "elapsed_seconds": int(etimes),
            "cmd": cmd,
        }
    return out


def classify_process(process: dict[str, Any],
                     expected_keywords: list[str]) -> str:
    cmd = str(process.get("cmd") or process.get("process_name") or "")
    lowered = cmd.lower()
    if expected_keywords and all(key.lower() in lowered
                                 for key in expected_keywords):
        return "expected_queue_process"
    if "hrrsd_bass_gsf" in lowered or "rankdelta" in lowered:
        return "related_bass_process"
    if not cmd:
        return "unknown_process"
    return "external_or_unrelated_process"


def build_snapshot(gpu_ids: list[int],
                   max_memory_mib: int,
                   max_util_pct: int,
                   expected_keywords: list[str]) -> dict[str, Any]:
    gpu_rows = query_gpu_rows()
    compute_rows = query_compute_rows()
    bus_to_index = {
        row["bus_id"]: str(row["index"])
        for row in gpu_rows.values()
    }
    pids = [row["pid"] for row in compute_rows]
    process_info = query_process_rows(pids)

    for compute in compute_rows:
        index = bus_to_index.get(compute["gpu_bus_id"])
        if index not in gpu_rows:
            continue
        proc = {**compute, **process_info.get(compute["pid"], {})}
        proc["classification"] = classify_process(proc, expected_keywords)
        gpu_rows[index]["processes"].append(proc)

    selected = []
    for gpu_id in gpu_ids:
        item = gpu_rows.get(str(gpu_id), {
            "index": gpu_id,
            "bus_id": "",
            "memory_used_mib": None,
            "memory_total_mib": None,
            "utilization_gpu_pct": None,
            "processes": [],
        })
        mem = item.get("memory_used_mib")
        util = item.get("utilization_gpu_pct")
        item["free_for_queue"] = (
            mem is not None and util is not None
            and mem <= max_memory_mib and util <= max_util_pct)
        selected.append(item)

    ready = all(item["free_for_queue"] for item in selected)
    action = "READY_TO_LAUNCH" if ready else "WAIT_GPU_BUSY"
    return {
        "action": action,
        "gpu_ids": gpu_ids,
        "max_memory_mib": max_memory_mib,
        "max_util_pct": max_util_pct,
        "gpus": selected,
        "external_or_unknown_process_count": sum(
            1 for gpu in selected
            for proc in gpu["processes"]
            if proc.get("classification")
            in {"external_or_unrelated_process", "unknown_process"}),
        "no_fabrication_rule": (
            "This audit only explains GPU wait state. It does not fill "
            "RankDelta result rows; eval JSON plus deployment-risk summary "
            "JSON are still required."),
    }


def format_log(snapshot: dict[str, Any]) -> str:
    lines = [
        (
            f"GPU wait audit action={snapshot['action']} "
            f"thresholds=mem<={snapshot['max_memory_mib']}MiB "
            f"util<={snapshot['max_util_pct']}%"
        )
    ]
    for gpu in snapshot["gpus"]:
        lines.append(
            f"GPU{gpu['index']} mem={gpu.get('memory_used_mib')}MiB "
            f"util={gpu.get('utilization_gpu_pct')}% "
            f"free={gpu.get('free_for_queue')}")
        for proc in gpu.get("processes", []):
            lines.append(
                f"  pid={proc.get('pid')} user={proc.get('user', 'NA')} "
                f"elapsed={proc.get('elapsed_seconds', 'NA')}s "
                f"mem={proc.get('used_memory_mib')}MiB "
                f"class={proc.get('classification')} "
                f"cmd={proc.get('cmd') or proc.get('process_name')}")
    return "\n".join(lines)


def write_markdown(path: Path, snapshot: dict[str, Any]) -> None:
    lines = [
        "# GPU6/7 RankDelta Wait Blocker Audit - 2026-06-21",
        "",
        "This file is generated by "
        "`M_Tools/analysis/audit_gpu67_wait_blocker.py`.",
        "It is a scheduler/status artifact only; it does not create results.",
        "",
        "| field | value |",
        "|---|---|",
        f"| action | `{snapshot['action']}` |",
        f"| gpu_ids | `{snapshot['gpu_ids']}` |",
        f"| max_memory_mib | `{snapshot['max_memory_mib']}` |",
        f"| max_util_pct | `{snapshot['max_util_pct']}` |",
        f"| external_or_unknown_process_count | `{snapshot['external_or_unknown_process_count']}` |",
        "",
        "## GPU Occupancy",
        "",
        "| gpu | mem used MiB | util pct | free for queue | processes |",
        "|---:|---:|---:|---|---|",
    ]
    for gpu in snapshot["gpus"]:
        proc_text = "<br>".join(
            (
                f"pid={proc.get('pid')} user={proc.get('user', 'NA')} "
                f"elapsed={proc.get('elapsed_seconds', 'NA')}s "
                f"mem={proc.get('used_memory_mib')}MiB "
                f"class={proc.get('classification')} "
                f"cmd={proc.get('cmd') or proc.get('process_name')}"
            )
            for proc in gpu.get("processes", [])
        ) or "none"
        lines.append(
            f"| {gpu['index']} | {gpu.get('memory_used_mib')} | "
            f"{gpu.get('utilization_gpu_pct')} | "
            f"`{gpu.get('free_for_queue')}` | {proc_text} |")
    lines += [
        "",
        "## No-Fabrication Rule",
        "",
        snapshot["no_fabrication_rule"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_gpus(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="6,7")
    parser.add_argument("--max-memory-mib", type=int, default=1024)
    parser.add_argument("--max-util-pct", type=int, default=20)
    parser.add_argument(
        "--expected-keyword",
        action="append",
        default=[],
        help="Keyword that must appear in cmd for expected_queue_process.")
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    parser.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    parser.add_argument(
        "--stdout-only",
        action="store_true",
        help="Print compact log text and do not write artifacts.")
    args = parser.parse_args()

    snapshot = build_snapshot(
        gpu_ids=parse_gpus(args.gpus),
        max_memory_mib=args.max_memory_mib,
        max_util_pct=args.max_util_pct,
        expected_keywords=args.expected_keyword,
    )
    if args.stdout_only:
        print(format_log(snapshot))
        return

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    write_markdown(Path(args.out_md), snapshot)
    print(json.dumps({
        "action": snapshot["action"],
        "external_or_unknown_process_count": (
            snapshot["external_or_unknown_process_count"]),
        "out_json": args.out_json,
        "out_md": args.out_md,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
