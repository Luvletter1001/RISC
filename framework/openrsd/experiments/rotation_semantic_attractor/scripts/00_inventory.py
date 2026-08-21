#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import EXP_DIR, PROJECT_ROOT, add_common_args, exp_path, read_jsonish
from experiments.rotation_semantic_attractor.src.utils.env import collect_env
from experiments.rotation_semantic_attractor.src.utils.io import write_json


def _count_files(path: Path, suffixes: tuple[str, ...]) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for p in path.iterdir() if p.suffix.lower() in suffixes)


def _sample_names(path: Path, suffixes: tuple[str, ...], limit: int = 20) -> list[str]:
    if not path.is_dir():
        return []
    return [p.stem for p in sorted(path.iterdir()) if p.suffix.lower() in suffixes][:limit]


def _has_prefix(path: Path, prefix: str, suffixes: tuple[str, ...]) -> bool:
    if not path.is_dir():
        return False
    return any(p.suffix.lower() in suffixes for p in path.glob(f"{prefix}*"))


def inventory(args) -> dict:
    dataset_registry = read_jsonish(exp_path("configs", "dataset_registry.yaml"))["datasets"]
    model_registry = read_jsonish(exp_path("configs", "model_registry.yaml"))["models"]

    datasets = {}
    for key, item in dataset_registry.items():
        root = Path(item["data_root"])
        image_dir = root / item["image_dir"]
        ann_dir = root / item["ann_dir"]
        samples = _sample_names(image_dir, (".png", ".jpg", ".jpeg"), limit=50)
        datasets[key] = {
            **item,
            "data_root_exists": root.exists(),
            "image_dir_exists": image_dir.is_dir(),
            "ann_dir_exists": ann_dir.is_dir(),
            "num_images": _count_files(image_dir, (".png", ".jpg", ".jpeg")),
            "num_annotations": _count_files(ann_dir, (".txt",)),
            "stress_tiles_present": {
                tile: _has_prefix(image_dir, tile, (".png", ".jpg", ".jpeg"))
                for tile in ["P0148", "P1384", "P2197", "P2068", "P1478"]
            },
            "sample_tile_ids": samples[:20],
        }

    models = {}
    for key, item in model_registry.items():
        config = Path(item.get("config", "")) if item.get("config") else None
        checkpoint = Path(item.get("checkpoint", "")) if item.get("checkpoint") else None
        reason = ""
        status = "AVAILABLE"
        if config is None or not config.exists():
            status = "NOT_RUN"
            reason = f"missing_config:{item.get('config', '')}"
        elif checkpoint is None or not str(checkpoint):
            status = "NOT_RUN"
            reason = "missing_checkpoint"
        elif not checkpoint.exists():
            status = "NOT_RUN"
            reason = f"missing_checkpoint:{checkpoint}"
        models[key] = {
            **item,
            "config_exists": bool(config and config.exists()),
            "checkpoint_exists": bool(checkpoint and checkpoint.exists()),
            "status": status,
            "not_run_reason": reason,
        }

    cache_patterns = [
        "work_dirs/*angle12*/angle_000/predictions.pkl",
        "work_dirs/*angle12*/angle_090/predictions.pkl",
        "work_dirs/exp_rotation_gt_shift_taxonomy_20260527/pred_cache/*/angle_000_results.pkl",
    ]
    caches = []
    for pattern in cache_patterns:
        matches = sorted(PROJECT_ROOT.glob(pattern))
        caches.append({"pattern": pattern, "count": len(matches), "samples": [str(p) for p in matches[:10]]})

    return {
        "environment": collect_env(PROJECT_ROOT),
        "datasets": datasets,
        "models": models,
        "existing_caches": caches,
        "code_entries": {
            "openrsd_test": str(PROJECT_ROOT / "tools/openrsd_test.py"),
            "rotation_diagnostics": str(PROJECT_ROOT / "tools/rotation_diagnostics"),
            "mmrotate_configs": str(PROJECT_ROOT / "mmrotate_configs"),
        },
    }


def write_markdown(path: Path, data: dict) -> None:
    lines = ["# Asset Inventory", "", "## Environment", ""]
    env = data["environment"]
    lines.extend([f"- `{k}`: `{v}`" for k, v in env.items()])
    lines.extend(["", "## Datasets", ""])
    for key, item in data["datasets"].items():
        lines.append(f"### {key}")
        lines.append(f"- root: `{item['data_root']}`")
        lines.append(f"- images: `{item['image_dir']}` exists={item['image_dir_exists']} count={item['num_images']}")
        lines.append(f"- annotations: `{item['ann_dir']}` exists={item['ann_dir_exists']} count={item['num_annotations']}")
        lines.append(f"- stress tiles: `{item['stress_tiles_present']}`")
        lines.append("")
    lines.extend(["## Models", ""])
    for key, item in data["models"].items():
        lines.append(f"- `{key}`: {item['status']} {item.get('not_run_reason', '')}")
    lines.extend(["", "## Existing Caches", ""])
    for item in data["existing_caches"]:
        lines.append(f"- `{item['pattern']}`: {item['count']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()
    out_dir = Path(args.output_dir or exp_path("outputs", "inventory"))
    data = inventory(args)
    write_json(out_dir / "asset_inventory.json", data)
    write_markdown(out_dir / "asset_inventory.md", data)
    write_json(exp_path("reports", "asset_inventory.json"), data)
    write_markdown(exp_path("reports", "asset_inventory.md"), data)
    print(f"inventory_json={out_dir / 'asset_inventory.json'}")
    print(f"inventory_md={out_dir / 'asset_inventory.md'}")


if __name__ == "__main__":
    main()
