#!/usr/bin/env python3
"""Run FOCUS-OVD P0 small proxy training for train-dependent variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from focus_p0_common import (
    append_manifest,
    ensure_exp_tree,
    parse_variants,
    read_csv_rows,
    read_json,
    train_proxy_rows,
    variant_config,
    write_csv_rows,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--max-iters", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--freeze-base-model", action="store_true")
    parser.add_argument("--seed", type=int, default=20260609)
    args = parser.parse_args()

    exp_dir = args.output_dir.parent
    ensure_exp_tree(exp_dir)
    index = {row["variant_id"]: row for row in read_csv_rows(args.config_index)}
    trained = []
    failed = []
    for variant_id in parse_variants(args.variants):
        cfg = variant_config(variant_id)
        out_dir = args.output_dir / variant_id
        out_dir.mkdir(parents=True, exist_ok=True)
        if not cfg["train_required"]:
            status = {
                "variant_id": variant_id,
                "status": "SKIPPED_NOT_TRAIN_REQUIRED",
                "reason": "non-train control",
            }
            write_json(out_dir / "train_status.json", status)
            continue
        if variant_id not in index:
            status = {
                "variant_id": variant_id,
                "status": "FAILED_CONFIG_MISSING",
                "reason": str(args.config_index),
            }
            write_json(out_dir / "train_status.json", status)
            failed.append(variant_id)
            continue
        rows = train_proxy_rows(variant_id, args.max_iters, args.seed)
        max_delta = max(float(r["delta_norm_ratio"]) for r in rows["delta"])
        has_nan = any(str(r["has_nan_or_inf"]) == "true" for r in rows["grad"])
        status_name = "DONE_SMALL_PROXY_TRAIN"
        failure_reason = ""
        if has_nan:
            status_name = "FAILED_NAN_INF"
            failure_reason = "NaN/Inf in proxy grad table"
        elif max_delta > 0.05:
            status_name = "FAILED_DELTA_NORM_BOUND"
            failure_reason = f"delta_norm_ratio={max_delta}"
        if status_name.startswith("FAILED"):
            failed.append(variant_id)
        else:
            trained.append(variant_id)

        write_csv_rows(out_dir / "loss_curve.csv", rows["loss"])
        write_csv_rows(out_dir / "grad_norm.csv", rows["grad"])
        write_csv_rows(out_dir / "delta_norm.csv", rows["delta"])
        write_csv_rows(out_dir / "support_geometry.csv", rows["geometry"])
        cfg_payload = read_json(Path(index[variant_id]["config_path"]))
        (out_dir / "config.py").write_text(
            "# FOCUS P0 proxy train config copy\n"
            f"config = {json.dumps(cfg_payload, indent=2, ensure_ascii=False)!r}\n",
            encoding="utf-8")
        checkpoint_payload = {
            "variant_id": variant_id,
            "checkpoint_type": "proxy_small_train_state",
            "actual_detector_checkpoint": False,
            "max_iters": args.max_iters,
            "last_delta_norm_ratio": rows["delta"][-1]["delta_norm_ratio"],
            "alpha": rows["delta"][-1]["alpha"],
        }
        write_json(out_dir / "checkpoint_last.pth", checkpoint_payload)
        write_json(out_dir / "checkpoint_best_proxy.pth", checkpoint_payload)
        train_manifest = {
            "variant_id": variant_id,
            "status": status_name,
            "failure_reason": failure_reason,
            "max_iters": args.max_iters,
            "eval_interval": args.eval_interval,
            "save_interval": args.save_interval,
            "freeze_base_model": bool(args.freeze_base_model),
            "actual_detector_train": False,
            "trainable_parameters": cfg_payload["trainable_parameters"],
            "frozen_parameters": cfg_payload["frozen_parameters"],
            "use_declip_support": False,
            "loss_weights": cfg_payload["loss_weights"],
        }
        write_json(out_dir / "train_manifest.json", train_manifest)
        write_json(out_dir / "train_status.json", train_manifest)
        (out_dir / "train.log").write_text(
            f"status={status_name}\n"
            "training_scope=verified_crop_label_proxy\n"
            "actual_detector_train=false\n"
            f"failure_reason={failure_reason}\n",
            encoding="utf-8")
    append_manifest(
        exp_dir, args.repo_root.resolve(),
        {"stage": "p0_training", "status": "WRITTEN", "trained": trained, "failed": failed})
    print(json.dumps({"trained": trained, "failed": failed}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
