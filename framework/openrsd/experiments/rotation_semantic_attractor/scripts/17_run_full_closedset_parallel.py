#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from common import PROJECT_ROOT, exp_path, parse_angles, read_jsonish
from experiments.rotation_semantic_attractor.src.utils.env import collect_env
from experiments.rotation_semantic_attractor.src.utils.io import read_json, write_json


FULL_12_ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]


def _count(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern))) if path.exists() else 0


def _is_model_done(run_dir: Path, model_key: str, expected: int) -> bool:
    raw_done = _count(run_dir / "raw_predictions" / model_key, "*/angle_*.json")
    canon_done = _count(run_dir / "canonical_predictions" / model_key, "*/angle_*.json")
    errors = _count(run_dir / "raw_predictions" / model_key, "*/*.error.json")
    return raw_done >= expected and canon_done >= expected and errors == 0


def _selected_models(args) -> list[str]:
    if args.models:
        return [item.strip() for item in args.models.split(",") if item.strip()]
    registry = read_jsonish(args.model_registry)["models"]
    ready = []
    for model_key, cfg in registry.items():
        if cfg.get("model_type") != "closed_set":
            continue
        if Path(cfg.get("config", "")).exists() and Path(cfg.get("checkpoint", "")).exists():
            ready.append(model_key)
    return ready[: args.minimum_models] if args.minimum_models else ready


def _write_central_manifest(args, run_dir: Path, split_path: Path, models: list[str], angles: list[int]) -> Path:
    manifest = {
        "command": "17_run_full_closedset_parallel.py",
        "args": vars(args),
        "split_path": str(split_path),
        "split_name": read_jsonish(split_path).get("split_name"),
        "angles": angles,
        "selected_models": models,
        "environment": collect_env(PROJECT_ROOT),
        "execution_mode": "model_parallel_shards",
        "model_status": {},
        "failures": [],
    }
    manifest_path = run_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def _model_command(args, run_dir: Path, split_path: Path, model_key: str, angles: list[int]) -> list[str]:
    cmd = [
        args.python,
        str(exp_path("scripts", "03_run_angle_sweep.py")),
        "--split",
        str(split_path),
        "--models",
        model_key,
        "--model-registry",
        str(args.model_registry),
        "--angles",
        ",".join(str(a) for a in angles),
        "--output-dir",
        str(run_dir),
        "--seed",
        str(args.seed),
        "--asset-namespace",
        "per-model",
        "--manifest-name",
        f"manifests/{model_key}.json",
    ]
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    return cmd


def _run_status_command(args, run_dir: Path, finalize: bool) -> None:
    cmd = [args.python, str(exp_path("scripts", "18_update_run_status.py")), "--run-dir", str(run_dir)]
    if finalize:
        cmd.append("--finalize")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _run_postprocess(args, run_dir: Path) -> None:
    commands = [
        [args.python, str(exp_path("scripts", "05_eval_false_hub_taxonomy.py")), "--run-dir", str(run_dir), "--iou-thr", str(args.iou_thr), "--seed", str(args.seed)],
        [args.python, str(exp_path("scripts", "06_stage_decomposition.py")), "--run-dir", str(run_dir), "--seed", str(args.seed)],
        [args.python, str(exp_path("scripts", "11_build_report.py")), "--run-dir", str(run_dir), "--output", str(exp_path("reports", "full_closedset_benchmark.md")), "--seed", str(args.seed)],
    ]
    for cmd in commands:
        print("POST " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="/data/zcy/anaconda3/envs/openrsd/bin/python")
    parser.add_argument("--model-registry", default=str(exp_path("configs", "model_registry.yaml")))
    parser.add_argument("--models", default="")
    parser.add_argument("--minimum-models", type=int, default=10)
    parser.add_argument("--angles", default=",".join(str(a) for a in FULL_12_ANGLES))
    parser.add_argument("--split", default="")
    parser.add_argument("--output-dir", default=str(exp_path("outputs", "runs", "full_closedset_s2_12angle")))
    parser.add_argument("--gpus", required=True, help="Comma-separated physical GPU ids for independent model shards.")
    parser.add_argument("--limit", type=int, default=0, help="Use only for smoke; nonzero prevents DONE_FULL gating.")
    parser.add_argument("--iou-thr", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--skip-postprocess", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "manifests").mkdir(parents=True, exist_ok=True)
    split_path = Path(args.split or exp_path("outputs", "splits", "S2_final_test.json"))
    models = _selected_models(args)
    angles = parse_angles(args.angles, default=FULL_12_ANGLES)
    split = read_jsonish(split_path)
    expected = len(split["tiles"][: args.limit or None]) * len(angles)
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")

    manifest_path = _write_central_manifest(args, run_dir, split_path, models, angles)
    commands = {model: _model_command(args, run_dir, split_path, model, angles) for model in models}
    write_json(
        run_dir / "full_closedset_parallel_plan.json",
        {
            "manifest": str(manifest_path),
            "gpus": gpus,
            "expected_tile_angle_per_model": expected,
            "commands": commands,
            "postprocess": not args.skip_postprocess,
        },
    )
    print(f"manifest={manifest_path}", flush=True)
    print(f"parallel_plan={run_dir / 'full_closedset_parallel_plan.json'}", flush=True)
    if args.dry_run:
        print("status=READY_TO_RUN")
        return

    pending = [model for model in models if not _is_model_done(run_dir, model, expected)]
    skipped = [model for model in models if model not in pending]
    for model in skipped:
        print(f"SKIP already_complete model={model}", flush=True)
    available_gpus = list(gpus)
    active = []
    results = {}
    while pending or active:
        while pending and available_gpus:
            model = pending.pop(0)
            gpu = available_gpus.pop(0)
            log_path = run_dir / "logs" / f"model_{model}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PYTHONNOUSERSITE"] = "1"
            log_fh = log_path.open("a")
            cmd = commands[model]
            print(f"RUN model={model} gpu={gpu} log={log_path}", flush=True)
            proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
            active.append({"model": model, "gpu": gpu, "proc": proc, "log_fh": log_fh, "log_path": log_path})

        time.sleep(args.poll_seconds)
        still_active = []
        for item in active:
            ret = item["proc"].poll()
            if ret is None:
                still_active.append(item)
                continue
            item["log_fh"].close()
            available_gpus.append(item["gpu"])
            results[item["model"]] = ret
            print(f"DONE model={item['model']} gpu={item['gpu']} returncode={ret}", flush=True)
            _run_status_command(args, run_dir, finalize=False)
        active = still_active

    _run_status_command(args, run_dir, finalize=True)
    if not args.skip_postprocess:
        _run_postprocess(args, run_dir)
    write_json(run_dir / "full_closedset_parallel_results.json", {"results": results, "skipped": skipped})
    print("status=DONE", flush=True)


if __name__ == "__main__":
    main()
