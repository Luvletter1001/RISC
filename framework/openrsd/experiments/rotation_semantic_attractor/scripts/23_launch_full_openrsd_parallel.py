#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = Path("/data/zcy/OpenRSD_runs/rotation_semantic_attractor")
DEFAULT_PYTHON = Path("/data/zcy/anaconda3/envs/openrsd/bin/python")


@dataclass(frozen=True)
class RunSpec:
    key: str
    run_dir: str
    session_prefix: str
    split: str
    run_family: str
    chunk_size: int
    rows_csv: str


RUN_SPECS = {
    "openvocab": RunSpec(
        key="openvocab",
        run_dir="full_openvocab_s2_12angle",
        session_prefix="rsa_p_openvocab_s2",
        split="experiments/rotation_semantic_attractor/outputs/splits/S2_final_test.json",
        run_family="open_vocab_benchmark",
        chunk_size=32,
        rows_csv="metrics/open_vocab_benchmark_rows.csv",
    ),
    "causal": RunSpec(
        key="causal",
        run_dir="full_causal_intervention_s3_12angle",
        session_prefix="rsa_p_causal_s3",
        split="experiments/rotation_semantic_attractor/outputs/splits/S3_safety_stratified.json",
        run_family="open_vocab_causal",
        chunk_size=24,
        rows_csv="metrics/open_vocab_causal_rows.csv",
    ),
    "dehub": RunSpec(
        key="dehub",
        run_dir="full_dehub_safety_s3_12angle",
        session_prefix="rsa_p_dehub_s3",
        split="experiments/rotation_semantic_attractor/outputs/splits/S3_safety_stratified.json",
        run_family="dehub_safety",
        chunk_size=24,
        rows_csv="metrics/dehub_safety_rows.csv",
    ),
}

DEFAULT_GPU_PLAN = {
    "openvocab": [0, 1, 2],
    "causal": [3, 4, 5, 6],
    "dehub": [7, 8, 9],
}


def _parse_gpu_plan(value: str) -> dict[str, list[int]]:
    if not value.strip():
        return dict(DEFAULT_GPU_PLAN)
    plan: dict[str, list[int]] = {}
    for part in value.replace(";", " ").split():
        if "=" not in part:
            raise ValueError(f"GPU plan part must be key=gpu,gpu: {part}")
        key, raw = part.split("=", 1)
        key = key.strip()
        if key not in RUN_SPECS:
            raise ValueError(f"unknown run key in GPU plan: {key}")
        gpus = [int(item) for item in raw.replace(",", " ").split() if item.strip()]
        if not gpus:
            raise ValueError(f"empty GPU list for {key}")
        plan[key] = gpus
    for key, gpus in DEFAULT_GPU_PLAN.items():
        plan.setdefault(key, list(gpus))
    return plan


def _selected_keys(value: str) -> list[str]:
    if not value.strip() or value.strip() == "all":
        return ["openvocab", "causal", "dehub"]
    keys = [item.strip() for item in value.replace(",", " ").split() if item.strip()]
    for key in keys:
        if key not in RUN_SPECS:
            raise ValueError(f"unknown run key: {key}")
    return keys


def _shell_join(parts: list[str | Path]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _build_job(spec: RunSpec, gpu: int, shard_index: int, shard_count: int, args) -> dict:
    root = Path(args.root)
    shard_name = f"shard_{shard_index:02d}_of_{shard_count:02d}"
    output_dir = root / spec.run_dir / "shards" / shard_name
    scratch_dir = root / "scratch_openrsd_streaming_parallel" / spec.run_dir / shard_name
    log_path = root / "logs" / f"{spec.run_dir}_{shard_name}.log"
    base_rows = root / spec.run_dir / spec.rows_csv
    session = f"{spec.session_prefix}_{shard_index:02d}"
    cmd_parts = [
        "cd",
        PROJECT_ROOT,
        "&&",
        "PYTHONNOUSERSITE=1",
        f"CUDA_VISIBLE_DEVICES={gpu}",
        "NCCL_P2P_DISABLE=1",
        "NCCL_IB_DISABLE=1",
        Path(args.python),
        "experiments/rotation_semantic_attractor/scripts/21_run_full_openrsd_streaming.py",
        "--split",
        spec.split,
        "--run-family",
        spec.run_family,
        "--chunk-size",
        str(spec.chunk_size),
        "--device",
        "cuda:0",
        "--output-dir",
        output_dir,
        "--scratch-dir",
        scratch_dir,
        "--task-shard-index",
        str(shard_index),
        "--task-shard-count",
        str(shard_count),
        "--extra-done-rows",
        base_rows,
        ">>",
        log_path,
        "2>&1",
    ]
    # This command is intentionally a shell command because tmux executes it.
    shell_cmd = " ".join(shlex.quote(str(part)) if part not in {"&&", ">>", "2>&1"} else str(part) for part in cmd_parts)
    return {
        "key": spec.key,
        "gpu": gpu,
        "session": session,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "output_dir": str(output_dir),
        "scratch_dir": str(scratch_dir),
        "log": str(log_path),
        "base_rows": str(base_rows),
        "command": shell_cmd,
        "tmux_command": _shell_join(["rtk", "tmux", "new-session", "-d", "-s", session, shell_cmd]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--only", default="all", help="Comma/space list: openvocab causal dehub")
    parser.add_argument(
        "--gpu-plan",
        default="",
        help="Optional plan like 'openvocab=0,1,2 causal=3,4,5,6 dehub=7,8,9'.",
    )
    parser.add_argument("--launch", action="store_true", help="Actually create tmux sessions. Default only prints commands.")
    parser.add_argument("--manifest", default="", help="Optional launch manifest path.")
    args = parser.parse_args()

    root = Path(args.root)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    plan = _parse_gpu_plan(args.gpu_plan)
    jobs = []
    for key in _selected_keys(args.only):
        spec = RUN_SPECS[key]
        gpus = plan[key]
        for shard_index, gpu in enumerate(gpus):
            jobs.append(_build_job(spec, gpu, shard_index, len(gpus), args))

    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": str(PROJECT_ROOT),
        "root": str(root),
        "jobs": jobs,
    }
    manifest_path = Path(args.manifest) if args.manifest else root / "parallel_launch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True))

    for job in jobs:
        print(job["tmux_command"])
        if args.launch:
            subprocess.run(["rtk", "tmux", "new-session", "-d", "-s", job["session"], job["command"]], check=True)
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
