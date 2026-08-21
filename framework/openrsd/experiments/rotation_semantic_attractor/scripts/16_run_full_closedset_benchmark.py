#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from common import PROJECT_ROOT, exp_path, parse_angles, read_jsonish
from experiments.rotation_semantic_attractor.src.utils.io import write_json


FULL_12_ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]


def available_closedset_models(registry_path: Path, minimum: int) -> list[str]:
    registry = read_jsonish(registry_path)["models"]
    models = []
    for key, cfg in registry.items():
        if cfg.get("model_type") != "closed_set":
            continue
        if not Path(cfg.get("config", "")).exists():
            continue
        if not Path(cfg.get("checkpoint", "")).exists():
            continue
        models.append(key)
    return models[:minimum] if minimum else models


def command_plan(args) -> list[list[str]]:
    py = args.python
    registry = Path(args.model_registry)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    if not models:
        models = available_closedset_models(registry, args.minimum_models)
    angles = parse_angles(args.angles, default=FULL_12_ANGLES)
    run_dir = Path(args.output_dir or exp_path("outputs", "runs", "full_closedset_s2_12angle"))
    split_dir = exp_path("outputs", "splits")
    split_path = Path(args.split or split_dir / "S2_final_test.json")

    commands = []
    if not split_path.exists():
        commands.append([py, str(exp_path("scripts", "01_build_tile_split.py")), "--dataset", "DOTA", "--output-dir", str(split_dir), "--seed", str(args.seed)])
    commands.append(
        [
            py,
            str(exp_path("scripts", "03_run_angle_sweep.py")),
            "--split",
            str(split_path),
            "--models",
            ",".join(models),
            "--model-registry",
            str(registry),
            "--angles",
            ",".join(str(a) for a in angles),
            "--output-dir",
            str(run_dir),
            "--seed",
            str(args.seed),
        ]
        + (["--limit", str(args.limit)] if args.limit else [])
    )
    commands.extend(
        [
            [py, str(exp_path("scripts", "05_eval_false_hub_taxonomy.py")), "--run-dir", str(run_dir), "--iou-thr", str(args.iou_thr), "--seed", str(args.seed)],
            [py, str(exp_path("scripts", "06_stage_decomposition.py")), "--run-dir", str(run_dir), "--seed", str(args.seed)],
            [py, str(exp_path("scripts", "11_build_report.py")), "--run-dir", str(run_dir), "--output", str(exp_path("reports", "full_closedset_benchmark.md")), "--seed", str(args.seed)],
        ]
    )
    return commands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="/data/zcy/anaconda3/envs/openrsd/bin/python")
    parser.add_argument("--model-registry", default=str(exp_path("configs", "model_registry.yaml")))
    parser.add_argument("--models", default="", help="Comma-separated closed-set model keys. Empty selects checkpoint-backed models.")
    parser.add_argument("--minimum-models", type=int, default=10)
    parser.add_argument("--angles", default=",".join(str(a) for a in FULL_12_ANGLES))
    parser.add_argument("--split", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--limit", type=int, default=0, help="Use only for smoke; nonzero prevents DONE_FULL gating.")
    parser.add_argument("--intervention-limit", type=int, default=0, help="Deprecated for full closed-set benchmark; causal intervention runs use S3 separately.")
    parser.add_argument("--counterfactual-limit", type=int, default=0, help="Deprecated for full closed-set benchmark; context counterfactual runs use S3 separately.")
    parser.add_argument("--dehub-limit", type=int, default=0, help="Deprecated for full closed-set benchmark; DeHub safety runs use S3/S2 separately.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iou-thr", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    commands = command_plan(args)
    plan_path = Path(args.output_dir or exp_path("outputs", "runs", "full_closedset_s2_12angle")) / "full_closedset_command_plan.json"
    write_json(
        plan_path,
        {
            "execute": args.execute,
            "commands": commands,
            "done_full_requires": {
                "split": "S2_final_test",
                "angles": FULL_12_ANGLES,
                "limit": 0,
                "minimum_models": args.minimum_models,
            },
        },
    )
    print(f"command_plan={plan_path}")
    if not args.execute:
        print("status=READY_TO_RUN")
        return

    for cmd in commands:
        print("RUN " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    print("status=DONE")


if __name__ == "__main__":
    main()
