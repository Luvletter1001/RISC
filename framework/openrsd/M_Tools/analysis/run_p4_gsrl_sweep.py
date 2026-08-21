#!/usr/bin/env python
"""Run a compact P4/GSRL candidate sweep with annfiles-only AP eval."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from M_Tools.analysis.apply_p4_gsrl_support_ranker import (  # noqa: E402
    apply_p4_gsrl_ranker,
)


DEFAULT_CANDIDATES = [
    {
        "name": "boost001_b000_rz10_pz30_ms005",
        "alpha": 0.01,
        "beta": 0.0,
        "reward_z": 1.0,
        "penalty_z": 3.0,
        "min_score": 0.05,
        "max_up_delta": 0.05,
        "max_down_delta": 0.0,
    },
    {
        "name": "boost002_b000_rz10_pz30_ms005",
        "alpha": 0.02,
        "beta": 0.0,
        "reward_z": 1.0,
        "penalty_z": 3.0,
        "min_score": 0.05,
        "max_up_delta": 0.10,
        "max_down_delta": 0.0,
    },
    {
        "name": "penalty000_b001_rz10_pz20_ms010",
        "alpha": 0.0,
        "beta": 0.01,
        "reward_z": 1.0,
        "penalty_z": 2.0,
        "min_score": 0.10,
        "max_up_delta": 0.0,
        "max_down_delta": 0.25,
    },
    {
        "name": "penalty000_b002_rz10_pz20_ms010",
        "alpha": 0.0,
        "beta": 0.02,
        "reward_z": 1.0,
        "penalty_z": 2.0,
        "min_score": 0.10,
        "max_up_delta": 0.0,
        "max_down_delta": 0.50,
    },
    {
        "name": "penalty000_b001_rz10_pz30_ms030",
        "alpha": 0.0,
        "beta": 0.01,
        "reward_z": 1.0,
        "penalty_z": 3.0,
        "min_score": 0.30,
        "max_up_delta": 0.0,
        "max_down_delta": 0.25,
    },
    {
        "name": "combo001_b001_rz10_pz20_ms010",
        "alpha": 0.01,
        "beta": 0.01,
        "reward_z": 1.0,
        "penalty_z": 2.0,
        "min_score": 0.10,
        "max_up_delta": 0.05,
        "max_down_delta": 0.25,
    },
    {
        "name": "combo002_b001_rz10_pz20_ms010",
        "alpha": 0.02,
        "beta": 0.01,
        "reward_z": 1.0,
        "penalty_z": 2.0,
        "min_score": 0.10,
        "max_up_delta": 0.10,
        "max_down_delta": 0.25,
    },
    {
        "name": "combo001_b002_rz15_pz30_ms010",
        "alpha": 0.01,
        "beta": 0.02,
        "reward_z": 1.5,
        "penalty_z": 3.0,
        "min_score": 0.10,
        "max_up_delta": 0.05,
        "max_down_delta": 0.50,
    },
    {
        "name": "combo002_b002_rz15_pz30_ms030",
        "alpha": 0.02,
        "beta": 0.02,
        "reward_z": 1.5,
        "penalty_z": 3.0,
        "min_score": 0.30,
        "max_up_delta": 0.10,
        "max_down_delta": 0.50,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compact P4/GSRL sweep over existing predictions.")
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--class-area-priors-csv", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ann-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--summary-json", default=None)
    return parser.parse_args()


def _metric(payload: dict[str, Any], key: str) -> float | None:
    metrics = payload.get("metrics", payload)
    value = metrics.get(key, metrics.get(f"dota/{key}"))
    return None if value is None else float(value)


def _eval_predictions(config: str,
                      predictions: Path,
                      ann_dir: str,
                      out_json: Path,
                      out_md: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "M_Tools/analysis/eval_predictions_annfiles_metric_json.py",
        "--config",
        config,
        "--predictions",
        str(predictions),
        "--ann-dir",
        ann_dir,
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("MPLCONFIGDIR", "/tmp/openrsd_mplconfig")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)
    return json.loads(out_json.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = DEFAULT_CANDIDATES
    if args.max_candidates > 0:
        candidates = candidates[:args.max_candidates]

    rows = []
    for candidate in candidates:
        variant_dir = out_dir / candidate["name"]
        variant_dir.mkdir(parents=True, exist_ok=True)
        predictions = variant_dir / "predictions.pkl"
        p4_json = variant_dir / "p4_summary.json"
        eval_json = variant_dir / "offline_eval.json"
        eval_md = variant_dir / "offline_eval.md"
        p4_summary = apply_p4_gsrl_ranker(
            input_pkl=args.input_pkl,
            output_pkl=predictions,
            class_area_priors_csv=args.class_area_priors_csv,
            config=args.config,
            alpha=candidate["alpha"],
            beta=candidate["beta"],
            reward_z=candidate["reward_z"],
            penalty_z=candidate["penalty_z"],
            min_score=candidate["min_score"],
            max_up_delta=candidate["max_up_delta"],
            max_down_delta=candidate["max_down_delta"],
        )
        p4_json.write_text(
            json.dumps(p4_summary, indent=2, ensure_ascii=False) + os.linesep,
            encoding="utf-8")
        eval_payload = _eval_predictions(
            args.config, predictions, args.ann_dir, eval_json, eval_md)
        row = {
            "name": candidate["name"],
            **candidate,
            "predictions": str(predictions),
            "p4_summary_json": str(p4_json),
            "eval_json": str(eval_json),
            "mAP": _metric(eval_payload, "mAP"),
            "AP50": _metric(eval_payload, "AP50"),
            "scores_changed": p4_summary["scores_changed"],
            "scores_boosted": p4_summary["scores_boosted"],
            "scores_penalized": p4_summary["scores_penalized"],
            "mean_logit_delta_changed": p4_summary[
                "mean_logit_delta_changed"],
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    rows = sorted(rows, key=lambda item: (
        -float(item["mAP"] if item["mAP"] is not None else -1.0),
        item["name"],
    ))
    summary_path = Path(args.summary_json) if args.summary_json else (
        out_dir / "sweep_summary.json")
    summary_path.write_text(
        json.dumps({
            "input_pkl": args.input_pkl,
            "config": args.config,
            "ann_dir": args.ann_dir,
            "class_area_priors_csv": args.class_area_priors_csv,
            "rows": rows,
        }, indent=2, ensure_ascii=False) + os.linesep,
        encoding="utf-8")
    print(json.dumps({"summary_json": str(summary_path), "best": rows[0]},
                     ensure_ascii=False),
          flush=True)


if __name__ == "__main__":
    main()
