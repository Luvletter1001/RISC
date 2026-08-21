#!/usr/bin/env python3
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from common import PROJECT_ROOT, add_common_args, read_jsonish
from experiments.rotation_semantic_attractor.src.metrics.context_metrics import write_context_counterfactual_images
from experiments.rotation_semantic_attractor.src.metrics.proxy_experiments import _fr_sv, context_proxy_rows
from experiments.rotation_semantic_attractor.src.model_adapters import MMDetAdapter
from experiments.rotation_semantic_attractor.src.utils.io import read_csv, read_json, write_csv, write_json
from experiments.rotation_semantic_attractor.src.utils.status import CLAIM_NONE, ExperimentStatus, status_metadata


def _all_region_rows(metrics_path: Path) -> list[dict]:
    if not metrics_path.exists():
        return []
    return [row for row in read_csv(metrics_path) if row.get("region_mode") == "all_region"]


def _real_counterfactual_rows(run_dir: Path, manifest: dict, args) -> list[dict]:
    registry = read_jsonish(manifest["args"]["model_registry"])["models"] if manifest.get("args", {}).get("model_registry") else {}
    metric_rows = _all_region_rows(run_dir / "metrics" / "false_hub_tile_angle.csv")
    if args.limit:
        metric_rows = metric_rows[: args.limit]
    rows = []
    adapters = {}
    for item in metric_rows:
        model = item["model_name"]
        cfg = registry.get(model, {})
        if model not in adapters:
            adapter = MMDetAdapter(model, cfg, PROJECT_ROOT)
            adapter.load(device=args.device)
            adapters[model] = adapter
        adapter = adapters[model]
        tile_id = item["tile_id"]
        angle = int(item["angle"])
        raw_path = run_dir / "raw_predictions" / model / tile_id / f"angle_{angle:03d}.json"
        if not raw_path.exists():
            continue
        raw = read_json(raw_path)
        image_path = raw.get("metadata", {}).get("rotated_image_path", "")
        ann_path = raw.get("metadata", {}).get("rotated_gt_path", "")
        if not image_path or not ann_path:
            continue
        cf_dir = run_dir / "counterfactual_images" / model / tile_id / f"angle_{angle:03d}"
        cf = write_context_counterfactual_images(image_path, ann_path, cf_dir)
        if not cf["conditions"]:
            meta = status_metadata(
                ExperimentStatus.UNSUPPORTED_BY_CURRENT_CODE,
                "no target small-vehicle GT polygon available for real image-level counterfactual",
                claim_level=CLAIM_NONE,
            )
            rows.append(
                {
                    "model_name": model,
                    "tile_id": tile_id,
                    "angle": angle,
                    "condition": "real_context_counterfactual",
                    "target_gt_class": cf["target_class"],
                    "counterfactual_is_real_image_level": False,
                    "modified_image_path": "",
                    "reran_inference": False,
                    "has_paired_comparison": False,
                    "completed_conditions": "",
                    "completed_condition_count": 0,
                    "proxy_only": False,
                    "mask_type": "target_gt_polygon",
                    **meta,
                    "notes": meta["status_reason"],
                }
            )
            continue
        completed = sorted(cf["conditions"])
        baseline_final = raw.get("final_predictions", [])
        baseline_fr = _fr_sv(baseline_final)
        for condition, modified_image_path in cf["conditions"].items():
            pred = adapter.infer(modified_image_path, tile_id=tile_id, angle=angle)
            pred_path = run_dir / "counterfactual_predictions" / model / tile_id / f"angle_{angle:03d}" / f"{condition}.json"
            write_json(pred_path, pred)
            final_predictions = pred.get("final_predictions", [])
            final_fr = _fr_sv(final_predictions)
            meta = status_metadata(
                ExperimentStatus.DONE_SMOKE,
                "real modified image generated and inference rerun on smoke image; not a full context causal benchmark",
                claim_level=CLAIM_NONE,
                is_scientific_result=False,
                include_in_main_table=False,
            )
            rows.append(
                {
                    "model_name": model,
                    "tile_id": tile_id,
                    "angle": angle,
                    "condition": condition,
                    "target_gt_class": cf["target_class"],
                    "counterfactual_is_real_image_level": True,
                    "modified_image_path": modified_image_path,
                    "prediction_path": str(pred_path),
                    "baseline_prediction_path": str(raw_path),
                    "mask_path": cf.get("mask_path", ""),
                    "mask_type": "target_gt_polygon",
                    "num_target_polygons": cf["num_target_polygons"],
                    "reran_inference": True,
                    "has_paired_comparison": True,
                    "completed_conditions": ";".join(completed),
                    "completed_condition_count": len(completed),
                    "proxy_only": False,
                    "is_smoke_limit": True,
                    "baseline_final_fr_sv": baseline_fr,
                    "final_fr_sv": final_fr,
                    "final_fsv": max(0.0, final_fr - baseline_fr),
                    "det_per_img": len(final_predictions),
                    **meta,
                    "notes": "",
                }
            )
    if not rows:
        for model in manifest.get("selected_models", []):
            meta = status_metadata(
                ExperimentStatus.UNSUPPORTED_BY_CURRENT_CODE,
                "false-hub rows or raw prediction metadata unavailable for real context counterfactual",
                claim_level=CLAIM_NONE,
            )
            rows.append(
                {
                    "model_name": model,
                    "tile_id": "",
                    "angle": "",
                    "condition": "real_context_counterfactual",
                    "target_gt_class": "small-vehicle",
                    "counterfactual_is_real_image_level": False,
                    "modified_image_path": "",
                    "reran_inference": False,
                    "has_paired_comparison": False,
                    "completed_conditions": "",
                    "completed_condition_count": 0,
                    "proxy_only": False,
                    "mask_type": "",
                    **meta,
                    "notes": meta["status_reason"],
                }
            )
    return rows


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--mode", choices=["auto", "real", "proxy"], default="auto")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    manifest = read_json(run_dir / "manifest.json")
    rows = []
    metrics_path = run_dir / "metrics" / "false_hub_tile_angle.csv"
    if args.mode in {"auto", "real"}:
        try:
            rows = _real_counterfactual_rows(run_dir, manifest, args)
        except Exception as exc:
            if args.mode == "real":
                raise
            meta = status_metadata(
                ExperimentStatus.FAILED,
                f"real context counterfactual failed: {type(exc).__name__}:{exc}",
                claim_level=CLAIM_NONE,
            )
            rows = [
                {
                    "model_name": ",".join(manifest.get("selected_models", [])),
                    "tile_id": "",
                    "angle": "",
                    "condition": "real_context_counterfactual",
                    "target_gt_class": "small-vehicle",
                    "counterfactual_is_real_image_level": False,
                    "modified_image_path": "",
                    "reran_inference": False,
                    "has_paired_comparison": False,
                    "proxy_only": False,
                    "mask_type": "",
                    **meta,
                    "notes": meta["status_reason"],
                    "traceback": traceback.format_exc(),
                }
            ]
    elif metrics_path.exists():
        for item in _all_region_rows(metrics_path):
            rows.extend(
                context_proxy_rows(
                    model_name=item["model_name"],
                    tile_id=item["tile_id"],
                    angle=int(item["angle"]),
                    false_hub_row=item,
                )
            )
    if not rows:
        for model in manifest.get("selected_models", []):
            meta = status_metadata(
                ExperimentStatus.UNSUPPORTED_BY_CURRENT_CODE,
                "false-hub metrics unavailable for context counterfactual proxy",
                claim_level=CLAIM_NONE,
            )
            rows.append(
                {
                    "model_name": model,
                    "tile_id": "",
                    "angle": "",
                    "condition": "context_proxy",
                    "target_gt_class": "",
                    "counterfactual_is_real_image_level": False,
                    "modified_image_path": "",
                    "reran_inference": False,
                    "proxy_only": True,
                    "mask_type": "",
                    **meta,
                    "notes": meta["status_reason"],
                }
            )
    out = Path(args.output_dir or run_dir / "metrics") / "context_counterfactual.csv"
    write_csv(out, rows)
    print(f"context_counterfactual={out}")


if __name__ == "__main__":
    main()
