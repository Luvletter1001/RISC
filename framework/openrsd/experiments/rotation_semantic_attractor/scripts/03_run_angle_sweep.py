#!/usr/bin/env python3
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from PIL import Image

from common import PROJECT_ROOT, add_common_args, exp_path, parse_angles, read_jsonish
from experiments.rotation_semantic_attractor.src.dota_io import read_dota_txt, records_to_polygons, write_dota_txt
from experiments.rotation_semantic_attractor.src.model_adapters import MMDetAdapter
from experiments.rotation_semantic_attractor.src.rotation import apply_transform_to_polygon, rotate_image_and_polygons
from experiments.rotation_semantic_attractor.src.utils.env import collect_env
from experiments.rotation_semantic_attractor.src.utils.io import write_json
from experiments.rotation_semantic_attractor.src.utils.io import write_csv
from experiments.rotation_semantic_attractor.src.utils.seed import set_seed


def _device() -> tuple[str, str]:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0", ""
        return "cpu", "GPU_UNAVAILABLE_IN_TORCH"
    except Exception as exc:
        return "cpu", f"TORCH_IMPORT_ERROR:{type(exc).__name__}:{exc}"


def _adapter(model_key: str, cfg: dict):
    adapter_name = cfg.get("adapter", "mmdet")
    if adapter_name != "mmdet":
        raise ValueError(f"unsupported adapter:{adapter_name}")
    return MMDetAdapter(model_key, cfg, PROJECT_ROOT)


def _canonicalize_predictions(predictions: list[dict], inverse_matrix) -> list[dict]:
    out = []
    for pred in predictions:
        new_pred = dict(pred)
        polygon = apply_transform_to_polygon(pred["polygon"], inverse_matrix).tolist()
        new_pred["polygon"] = polygon
        new_pred["box"] = [coord for point in polygon for coord in point]
        new_pred["box_type"] = "polygon"
        out.append(new_pred)
    return out


def _write_rotated_assets(tile: dict, angle: int, run_dir: Path, model_key: str, asset_namespace: str):
    image = Image.open(tile["image_path"]).convert("RGB")
    records = read_dota_txt(tile["ann_path"])
    asset_parts = [model_key, tile["tile_id"]] if asset_namespace == "per-model" else [tile["tile_id"]]
    result = rotate_image_and_polygons(
        image,
        records_to_polygons(records),
        angle_deg=angle,
        expand=True,
        output_mask_path=run_dir / "valid_masks" / Path(*asset_parts) / f"angle_{angle:03d}.png",
    )
    image_dir = run_dir / "rotated_images" / Path(*asset_parts)
    gt_dir = run_dir / "rotated_gt" / Path(*asset_parts)
    image_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{tile['tile_id']}_angle_{angle:03d}.png"
    ann_path = gt_dir / f"{tile['tile_id']}_angle_{angle:03d}.txt"
    result.image.save(image_path)
    rotated_records = []
    for rec, poly in zip(records, result.polygons):
        new_rec = dict(rec)
        new_rec["polygon"] = poly
        rotated_records.append(new_rec)
    write_dota_txt(ann_path, rotated_records)
    return image_path, ann_path, result.transform


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--split", required=True)
    parser.add_argument("--model-registry", default=str(exp_path("configs", "model_registry.yaml")))
    parser.add_argument("--models", default="rotated_retinanet_msrr")
    parser.add_argument("--asset-namespace", choices=["shared", "per-model"], default="shared")
    parser.add_argument("--manifest-name", default="manifest.json")
    parser.add_argument("--tile-shard-index", type=int, default=0)
    parser.add_argument("--tile-shard-count", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    split_path = Path(args.split)
    split = read_jsonish(split_path)
    registry = read_jsonish(args.model_registry)["models"]
    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    angles = parse_angles(args.angles, default=[0, 90])
    run_dir = Path(args.output_dir or exp_path("outputs", "runs", "smoke"))
    if args.tile_shard_count < 1:
        raise ValueError("--tile-shard-count must be >= 1")
    if args.tile_shard_index < 0 or args.tile_shard_index >= args.tile_shard_count:
        raise ValueError("--tile-shard-index must be in [0, tile_shard_count)")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / args.manifest_name
    device, device_warning = _device()
    manifest = {
        "command": "03_run_angle_sweep.py",
        "args": vars(args),
        "split_path": str(split_path),
        "split_name": split.get("split_name"),
        "angles": angles,
        "selected_models": selected_models,
        "environment": collect_env(PROJECT_ROOT),
        "device": device,
        "device_warning": device_warning,
        "tile_shard_index": args.tile_shard_index,
        "tile_shard_count": args.tile_shard_count,
        "model_status": {},
        "failures": [],
    }
    write_json(manifest_path, manifest)

    if args.dry_run:
        print(f"dry_run_manifest={manifest_path}")
        return

    all_tiles = split["tiles"][: args.limit or None]
    tiles = [
        tile
        for tile_index, tile in enumerate(all_tiles)
        if args.tile_shard_count <= 1 or tile_index % args.tile_shard_count == args.tile_shard_index
    ]
    for model_key in selected_models:
        cfg = registry.get(model_key)
        if cfg is None:
            manifest["model_status"][model_key] = {"status": "NOT_RUN", "reason": "model_not_in_registry"}
            continue
        adapter = _adapter(model_key, cfg)
        ok, reason = adapter.available()
        if not ok:
            manifest["model_status"][model_key] = {"status": "NOT_RUN", "reason": reason}
            continue
        try:
            adapter.load(device=device)
            manifest["model_status"][model_key] = {"status": "LOADED", "device": device, "warning": device_warning}
        except Exception as exc:
            manifest["model_status"][model_key] = {"status": "NOT_RUN", "reason": f"load_failed:{type(exc).__name__}:{exc}"}
            manifest["failures"].append({"model": model_key, "stage": "load", "traceback": traceback.format_exc()})
            continue

        for tile in tiles:
            for angle in angles:
                raw_path = run_dir / "raw_predictions" / model_key / tile["tile_id"] / f"angle_{angle:03d}.json"
                canon_path = run_dir / "canonical_predictions" / model_key / tile["tile_id"] / f"angle_{angle:03d}.json"
                if raw_path.exists() and canon_path.exists():
                    continue
                try:
                    image_path, ann_path, transform = _write_rotated_assets(
                        tile,
                        angle,
                        run_dir,
                        model_key=model_key,
                        asset_namespace=args.asset_namespace,
                    )
                    raw_output = adapter.infer(image_path, tile_id=tile["tile_id"], angle=angle)
                    raw_output["metadata"]["rotated_image_path"] = str(image_path)
                    raw_output["metadata"]["rotated_gt_path"] = str(ann_path)
                    raw_output["metadata"]["transform_forward"] = transform.forward.tolist()
                    raw_output["metadata"]["transform_inverse"] = transform.inverse.tolist()
                    canonical = {
                        "final_predictions": _canonicalize_predictions(raw_output["final_predictions"], transform.inverse),
                        "pre_nms_predictions": _canonicalize_predictions(raw_output.get("pre_nms_predictions", []), transform.inverse),
                        "metadata": dict(raw_output["metadata"], coordinate_space="canonical"),
                    }
                    write_json(raw_path, raw_output)
                    write_json(canon_path, canonical)
                except Exception as exc:
                    failure = {
                        "model": model_key,
                        "tile_id": tile["tile_id"],
                        "angle": angle,
                        "reason": f"{type(exc).__name__}:{exc}",
                        "traceback": traceback.format_exc(),
                    }
                    manifest["failures"].append(failure)
                    write_json(raw_path.with_suffix(".error.json"), failure)
        expected = len(tiles) * len(angles)
        raw_done = len(list((run_dir / "raw_predictions" / model_key).glob("*/angle_*.json")))
        canon_done = len(list((run_dir / "canonical_predictions" / model_key).glob("*/angle_*.json")))
        model_errors = [f for f in manifest["failures"] if f.get("model") == model_key]
        prior = manifest["model_status"].get(model_key, {})
        if prior.get("status") == "LOADED":
            if raw_done >= expected and canon_done >= expected and not model_errors:
                prior["status"] = "DONE_FULL" if split.get("split_name") == "S2_final_test" and len(angles) >= 12 and not args.limit else "DONE_SMOKE"
            elif raw_done > 0:
                prior["status"] = "PARTIAL_FULL" if split.get("split_name") == "S2_final_test" and not args.limit else "PARTIAL_SMOKE"
            elif model_errors:
                prior["status"] = "FAILED"
        prior.update(
            {
                "expected_tile_angle": expected,
                "completed_tile_angle": min(raw_done, canon_done),
                "raw_completed_tile_angle": raw_done,
                "canonical_completed_tile_angle": canon_done,
                "failed_tile_angle": len(model_errors),
            }
        )
        manifest["model_status"][model_key] = prior
        status_name = f"{model_key}.json"
        if args.tile_shard_count > 1:
            status_name = f"{model_key}_shard_{args.tile_shard_index:02d}_of_{args.tile_shard_count:02d}.json"
        write_json(run_dir / "model_status" / status_name, {"model_name": model_key, **prior})
    write_json(manifest_path, manifest)
    run_rows = []
    for model_key, status in sorted(manifest["model_status"].items()):
        run_rows.append(
            {
                "model_name": model_key,
                "status": status.get("status", ""),
                "reason": status.get("reason", ""),
                "device": status.get("device", ""),
                "expected_tile_angle": status.get("expected_tile_angle", ""),
                "completed_tile_angle": status.get("completed_tile_angle", ""),
                "failed_tile_angle": status.get("failed_tile_angle", ""),
            }
        )
    write_csv(run_dir / "metrics" / "run_status_by_model.csv", run_rows)
    print(f"run_dir={run_dir}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
