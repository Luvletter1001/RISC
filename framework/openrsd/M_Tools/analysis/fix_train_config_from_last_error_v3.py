#!/usr/bin/env python3
"""Generate fixed DOTA1 training configs for full-live v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from failure_autopsy_last_run_v3 import classify_training_error
from strict_gpu_runner_v3 import read_text, write_json, write_text
from write_openrsd_dota1_fix_configs import write_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("/data1/zcy/OpenRSD"))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, default=Path("/data1/zcy/OpenRSD/data/DOTA1_1024_500/ss_train"))
    parser.add_argument("--val-root", type=Path, default=Path("/data1/zcy/OpenRSD/data/DOTA1_1024_500/angle_sweep_val/realistic/angle_000"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/data1/zcy/OpenRSD/results/MMR_AD_A12_flex_rtm_v3_1_DOTA2only_ss_train_textcls00/epoch_12.pth"))
    parser.add_argument("--last-stderr", type=Path, default=Path("/data1/zcy/OpenRSD/work_dirs/openrsd_next_priority_gpu45_20260509_200920/P5_logs/P5_baseline_short_ft_train/stderr.log"))
    parser.add_argument("--max-iters", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_dir = args.work_dir / "G1_train_configs"
    train_work = args.work_dir / "G1_training"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    err_tail = "\n".join(read_text(args.last_stderr).splitlines()[-300:])
    err_class = classify_training_error(err_tail)
    outputs = {
        "baseline_short_ft": cfg_dir / "g1_baseline_short_ft.py",
        "rr_dense": cfg_dir / "g1_rr_dense.py",
        "consistency_cls": cfg_dir / "g1_consistency_cls.py",
    }
    write_config(
        outputs["baseline_short_ft"],
        args.repo_root,
        args.train_root,
        args.val_root,
        train_work / "baseline_short_ft",
        args.checkpoint,
        args.max_iters,
        args.batch_size,
        model_type="OpenRTMDet",
    )
    write_config(
        outputs["rr_dense"],
        args.repo_root,
        args.train_root,
        args.val_root,
        train_work / "rr_dense",
        args.checkpoint,
        args.max_iters,
        args.batch_size,
        model_type="OpenRTMDet",
    )
    write_config(
        outputs["consistency_cls"],
        args.repo_root,
        args.train_root,
        args.val_root,
        train_work / "consistency_cls",
        args.checkpoint,
        args.max_iters,
        args.batch_size,
        model_type="OpenRTMDetCrossViewConsistency",
    )
    payload = {
        "status": "DONE",
        "last_error_class": err_class,
        "train_root": str(args.train_root),
        "val_root": str(args.val_root),
        "checkpoint": str(args.checkpoint),
        "configs": {k: str(v) for k, v in outputs.items()},
        "fix_summary": "Generated DOTA1 configs with ss_train, disabled in-loop validation by setting val_interval=max_epochs+1, DOTA1 metainfo, and local support paths.",
    }
    write_json(args.out_json, payload)
    lines = ["# G1 Config Fix", "", f"- status: `{payload['status']}`", f"- last_error_class: `{err_class}`", ""]
    for name, path in outputs.items():
        lines.append(f"- {name}: `{path}`")
    write_text(args.work_dir / "G1_train_configs/config_fix.md", "\n".join(lines))


if __name__ == "__main__":
    main()

