#!/usr/bin/env python3
"""Write the FOCUS-OVD adapter training smoke plan without launching training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_OUT_DIR = Path("resultmd/exp_focus_ovd_20260608/train_smoke")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--emit-run-command", action="store_true")
    args = parser.parse_args()

    command = (
        "PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mplconfig "
        "/data/zcy/anaconda3/envs/openrsd/bin/python tools/train.py "
        "M_configs/experiments/focus_ovd/focus_ovd_a10_sv_only_smoke.py "
        "--work-dir work_dirs/focus_ovd_a10_sv_only_smoke"
    )
    plan = {
        "status": "SCAFFOLD_ONLY",
        "launches_training": False,
        "target": "small-vehicle only",
        "must_keep": [
            "native OpenRSD support bank",
            "zero-init residual support adapter",
            "box regression and NMS unchanged",
            "DeCLIP support excluded",
        ],
        "config_patch": {
            "bbox_head.use_focus_ovd": True,
            "bbox_head.focus_ovd.enable": True,
            "bbox_head.focus_ovd.adapter.apply_to_classes": ["small-vehicle"],
            "bbox_head.focus_ovd.adapter.alpha_init": 0.0,
            "bbox_head.focus_ovd.adapter.max_delta_norm_ratio": 0.05,
        },
        "suggested_command": command,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train_smoke_plan.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8")
    md = [
        "# FOCUS-OVD Train Smoke Plan",
        "",
        "This script does not launch training by default.",
        "",
        "```bash",
        command,
        "```",
        "",
        "- Scope: small-vehicle only.",
        "- Adapter: zero-init residual over native OpenRSD support.",
        "- Exclusions: DeCLIP support, box regression changes, NMS/postprocess changes.",
    ]
    (args.out_dir / "train_smoke_plan.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    if args.emit_run_command:
        print(command)
    else:
        print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
