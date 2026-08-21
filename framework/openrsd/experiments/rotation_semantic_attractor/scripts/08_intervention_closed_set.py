#!/usr/bin/env python3
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from common import PROJECT_ROOT, add_common_args, read_jsonish
from experiments.rotation_semantic_attractor.src.metrics.proxy_experiments import _fr_sv, closedset_intervention_rows
from experiments.rotation_semantic_attractor.src.model_adapters import MMDetAdapter
from experiments.rotation_semantic_attractor.src.model_adapters.closedset_intervention import (
    apply_closedset_classifier_intervention,
    find_classifier_layers,
    restore_classifier_layers,
    snapshot_classifier_layers,
)
from experiments.rotation_semantic_attractor.src.utils.io import read_json, write_csv, write_json
from experiments.rotation_semantic_attractor.src.utils.status import (
    CLAIM_CLOSED_SET_LOGIT,
    CLAIM_NONE,
    ExperimentStatus,
    status_metadata,
)


DEFAULT_REAL_INTERVENTIONS = [
    "suppress_small_vehicle_classifier_channel",
    "swap_small_large_vehicle_classifier_channel",
]


def _iter_raw_prediction_paths(run_dir: Path, model: str, limit: int):
    paths = sorted((run_dir / "raw_predictions" / model).glob("*/angle_*.json"))
    return paths[:limit] if limit else paths


def _real_intervention_rows(model: str, cfg: dict, run_dir: Path, args) -> list[dict]:
    rows = []
    adapter = MMDetAdapter(model, cfg, PROJECT_ROOT)
    adapter.load(device=args.device)
    layers = find_classifier_layers(adapter.model)
    raw_paths = _iter_raw_prediction_paths(run_dir, model, args.limit)
    if not layers:
        meta = status_metadata(
            ExperimentStatus.UNSUPPORTED_BY_CURRENT_CODE,
            "no classifier Conv2d layer with DOTA class channels was found for real closed-set intervention",
            claim_level=CLAIM_NONE,
        )
        return [
            {
                "model_name": model,
                "tile_id": "",
                "angle": "",
                "intervention_type": "",
                "actual_weight_or_logit_modified": False,
                "actual_weight_modified": False,
                "actual_logit_modified": False,
                "modified_layer_name": "",
                "modified_tensor_name": "",
                "modified_class_index": "",
                "target_class_name": "small-vehicle",
                "control_class_name": "",
                "proxy_only": False,
                "reran_inference": False,
                "has_paired_comparison": False,
                **meta,
                "unsupported_reason": meta["status_reason"],
            }
        ]
    if not raw_paths:
        meta = status_metadata(
            ExperimentStatus.UNSUPPORTED_BY_CURRENT_CODE,
            "no raw predictions/images available for real closed-set intervention",
            claim_level=CLAIM_NONE,
        )
        return [
            {
                "model_name": model,
                "tile_id": "",
                "angle": "",
                "intervention_type": "",
                "actual_weight_or_logit_modified": False,
                "actual_weight_modified": False,
                "actual_logit_modified": False,
                "modified_layer_name": ";".join(layer.name for layer in layers),
                "modified_tensor_name": "",
                "modified_class_index": "",
                "target_class_name": "small-vehicle",
                "control_class_name": "",
                "proxy_only": False,
                "reran_inference": False,
                "has_paired_comparison": False,
                **meta,
                "unsupported_reason": meta["status_reason"],
            }
        ]

    snapshots = snapshot_classifier_layers(layers)
    interventions = [item.strip() for item in args.interventions.split(",") if item.strip()]
    out_root = Path(args.output_dir or run_dir / "metrics").parent / "real_intervention_predictions" / model
    for intervention in interventions:
        restore_classifier_layers(layers, snapshots)
        evidence = apply_closedset_classifier_intervention(layers, intervention=intervention)
        meta = status_metadata(
            ExperimentStatus.DONE_SMOKE,
            "real closed-set classifier weight intervention reran inference on smoke images; not a full causal benchmark",
            claim_level=CLAIM_CLOSED_SET_LOGIT,
            is_scientific_result=False,
            include_in_main_table=False,
        )
        for raw_path in raw_paths:
            tile_id = raw_path.parts[-2]
            angle = int(raw_path.stem.split("_")[-1])
            raw_output = read_json(raw_path)
            image_path = raw_output.get("metadata", {}).get("rotated_image_path", "")
            baseline_final = raw_output.get("final_predictions", [])
            baseline_pre = raw_output.get("pre_nms_predictions", [])
            if not image_path:
                continue
            modified = adapter.infer(image_path, tile_id=tile_id, angle=angle)
            pred_path = out_root / intervention / tile_id / f"angle_{angle:03d}.json"
            write_json(pred_path, modified)
            final_predictions = modified.get("final_predictions", [])
            pre_nms_predictions = modified.get("pre_nms_predictions", [])
            final_fr = _fr_sv(final_predictions)
            baseline_fr = _fr_sv(baseline_final)
            pre_fr = _fr_sv(pre_nms_predictions)
            rows.append(
                {
                    "model_name": model,
                    "tile_id": tile_id,
                    "angle": int(angle),
                    "prediction_path": str(pred_path),
                    "baseline_prediction_path": str(raw_path),
                    "baseline_final_fr_sv": baseline_fr,
                    "baseline_pre_nms_fr_sv": _fr_sv(baseline_pre),
                    "final_fr_sv": final_fr,
                    "final_fsv": max(0.0, final_fr - baseline_fr),
                    "pre_nms_fr_sv": pre_fr,
                    "post_nms_fr_sv": final_fr,
                    "nms_amp_sv": final_fr - pre_fr if pre_nms_predictions else "",
                    "det_per_img": len(final_predictions),
                    "reran_inference": True,
                    "has_paired_comparison": True,
                    "proxy_only": False,
                    "is_smoke_limit": True,
                    **evidence,
                    **meta,
                    "unsupported_reason": "",
                }
            )
    restore_classifier_layers(layers, snapshots)
    return rows


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--mode", choices=["auto", "real", "proxy"], default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--interventions", default=",".join(DEFAULT_REAL_INTERVENTIONS))
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    manifest = read_json(run_dir / "manifest.json")
    rows = []
    registry = read_jsonish(manifest["args"]["model_registry"])["models"] if manifest.get("args", {}).get("model_registry") else {}
    for model in manifest.get("selected_models", []):
        cfg = registry.get(model, {})
        if args.mode in {"auto", "real"}:
            try:
                rows.extend(_real_intervention_rows(model, cfg, run_dir, args))
                continue
            except Exception as exc:
                if args.mode == "real":
                    raise
                meta = status_metadata(
                    ExperimentStatus.FAILED,
                    f"real closed-set intervention failed: {type(exc).__name__}:{exc}",
                    claim_level=CLAIM_NONE,
                )
                rows.append(
                    {
                        "model_name": model,
                        "tile_id": "",
                        "angle": "",
                        "intervention_type": "real_closedset_classifier_channel",
                        "actual_weight_or_logit_modified": False,
                        "actual_weight_modified": False,
                        "actual_logit_modified": False,
                        "modified_layer_name": "",
                        "modified_tensor_name": "",
                        "modified_class_index": "",
                        "target_class_name": "small-vehicle",
                        "control_class_name": "",
                        "proxy_only": False,
                        "reran_inference": False,
                        "has_paired_comparison": False,
                        **meta,
                        "unsupported_reason": meta["status_reason"],
                        "traceback": traceback.format_exc(),
                    }
                )
                continue
        model_rows = []
        for raw_path in sorted((run_dir / "raw_predictions" / model).glob("*/angle_*.json")):
            tile_id = raw_path.parts[-2]
            angle = int(raw_path.stem.split("_")[-1])
            raw_output = read_json(raw_path)
            model_rows.extend(
                closedset_intervention_rows(
                    model_name=model,
                    tile_id=tile_id,
                    angle=angle,
                    final_predictions=raw_output.get("final_predictions", []),
                    pre_nms_predictions=raw_output.get("pre_nms_predictions", []),
                )
            )
        if model_rows:
            rows.extend(model_rows)
        else:
            meta = status_metadata(
                ExperimentStatus.UNSUPPORTED_BY_CURRENT_CODE,
                "no raw predictions available for closed-set channel intervention proxy",
                claim_level=CLAIM_NONE,
            )
            rows.append(
                {
                    "model_name": model,
                    "tile_id": "",
                    "angle": "",
                    "intervention_type": "closedset_proxy",
                    "actual_weight_or_logit_modified": False,
                    "modified_layer_name": "",
                    "modified_class_index": "",
                    "target_class_name": "small-vehicle",
                    "control_class_name": "",
                    "proxy_only": True,
                    **meta,
                    "unsupported_reason": meta["status_reason"],
                }
            )
    out = Path(args.output_dir or run_dir / "metrics") / "closedset_channel_intervention.csv"
    write_csv(out, rows)
    print(f"closedset_intervention={out}")


if __name__ == "__main__":
    main()
