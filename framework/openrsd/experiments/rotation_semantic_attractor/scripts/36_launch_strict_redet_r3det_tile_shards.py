#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from common import PROJECT_ROOT, exp_path, parse_angles
from experiments.rotation_semantic_attractor.src.utils.io import write_json


FULL_12_ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
DEFAULT_REGISTRY = (
    PROJECT_ROOT
    / "resultmd/exp_rotation_semantic_attractor/model_integrity_audit_20260602"
    / "redet_r3det_strict_rerun_model_registry.json"
)
DEFAULT_RUN_DIR = exp_path("outputs", "runs", "full_closedset_s2_12angle_strict_redet_r3det_20260603")


def _jobs(args: argparse.Namespace) -> list[dict]:
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    angles = parse_angles(args.angles, default=FULL_12_ANGLES)
    jobs = []
    job_index = 0
    for model in models:
        for shard_index in range(args.tile_shard_count):
            gpu = gpus[job_index % len(gpus)]
            shard_name = f"{model}_shard_{shard_index:02d}_of_{args.tile_shard_count:02d}"
            cmd = [
                args.python,
                str(exp_path("scripts", "03_run_angle_sweep.py")),
                "--split",
                str(args.split),
                "--models",
                model,
                "--model-registry",
                str(args.model_registry),
                "--angles",
                ",".join(str(a) for a in angles),
                "--output-dir",
                str(args.output_dir),
                "--seed",
                str(args.seed),
                "--asset-namespace",
                "per-model",
                "--manifest-name",
                f"manifests/{shard_name}.json",
                "--tile-shard-index",
                str(shard_index),
                "--tile-shard-count",
                str(args.tile_shard_count),
            ]
            jobs.append(
                {
                    "model": model,
                    "shard_index": shard_index,
                    "tile_shard_count": args.tile_shard_count,
                    "gpu": gpu,
                    "name": shard_name,
                    "command": cmd,
                    "log_path": str(args.output_dir / "logs" / "tile_shards" / f"{shard_name}.log"),
                }
            )
            job_index += 1
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="/data/zcy/anaconda3/envs/openrsd/bin/python")
    parser.add_argument("--model-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--models", default="redet,r3det_kfiou")
    parser.add_argument("--angles", default=",".join(str(a) for a in FULL_12_ANGLES))
    parser.add_argument("--split", type=Path, default=exp_path("outputs", "splits", "S2_final_test.json"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--tile-shard-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.tile_shard_count < 1:
        raise ValueError("--tile-shard-count must be >= 1")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs" / "tile_shards").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifests").mkdir(parents=True, exist_ok=True)
    jobs = _jobs(args)
    write_json(
        args.output_dir / "strict_redet_r3det_tile_shard_launch_manifest.json",
        {
            "command": "36_launch_strict_redet_r3det_tile_shards.py",
            "args": {key: str(value) for key, value in vars(args).items()},
            "jobs": jobs,
            "note": "Existing raw/canonical outputs are skipped by 03_run_angle_sweep.py. Do not run an unsharded worker for the same models in this output directory at the same time.",
        },
    )
    print(f"jobs={len(jobs)}")
    print(f"manifest={args.output_dir / 'strict_redet_r3det_tile_shard_launch_manifest.json'}")
    if args.dry_run:
        print("status=READY_TO_RUN")
        return

    active = []
    results = {}
    for job in jobs:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
        env["PYTHONNOUSERSITE"] = "1"
        env["MPLCONFIGDIR"] = "/tmp/mplconfig"
        log_path = Path(job["log_path"])
        log_fh = log_path.open("a")
        print(f"RUN name={job['name']} gpu={job['gpu']} log={log_path}", flush=True)
        proc = subprocess.Popen(job["command"], cwd=PROJECT_ROOT, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        active.append({"job": job, "proc": proc, "log_fh": log_fh})

    while active:
        time.sleep(args.poll_seconds)
        still_active = []
        for item in active:
            ret = item["proc"].poll()
            if ret is None:
                still_active.append(item)
                continue
            item["log_fh"].close()
            results[item["job"]["name"]] = ret
            print(f"DONE name={item['job']['name']} returncode={ret}", flush=True)
        active = still_active

    write_json(args.output_dir / "strict_redet_r3det_tile_shard_results.json", {"results": results})
    failed = {name: ret for name, ret in results.items() if ret != 0}
    if failed:
        raise SystemExit(f"failed_shards={failed}")
    print("status=DONE")


if __name__ == "__main__":
    main()
