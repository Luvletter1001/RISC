#!/usr/bin/env python3
"""Write the FOCUS-OVD evaluation smoke plan without running inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_OUT_DIR = Path("resultmd/exp_focus_ovd_20260608/eval_smoke")
DEFAULT_BASELINE = Path("results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE)
    args = parser.parse_args()

    plan = {
        "status": "SCAFFOLD_ONLY",
        "launches_inference": False,
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "baseline_checkpoint_exists": args.baseline_checkpoint.exists(),
        "required_checks": [
            "zero-residual logits match baseline before training",
            "corrected_false_sv score decreases after adapter training",
            "annotation_missing_true_vehicle recall is preserved",
            "DOTA AP for non-small-vehicle classes does not regress materially",
        ],
        "metrics": [
            "corrected-FSV rate on audited decidable false-SV rows",
            "preserve-positive small-vehicle recall",
            "class-score migration KL",
            "standard DOTA mAP/AP50",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "eval_smoke_plan.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8")
    md = [
        "# FOCUS-OVD Eval Smoke Plan",
        "",
        f"- Baseline checkpoint: `{args.baseline_checkpoint}`",
        f"- Exists: {args.baseline_checkpoint.exists()}",
        "- This script does not run inference.",
        "",
        "| check | purpose |",
        "| --- | --- |",
    ]
    for check in plan["required_checks"]:
        md.append(f"| {check} | safety gate |")
    (args.out_dir / "eval_smoke_plan.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
