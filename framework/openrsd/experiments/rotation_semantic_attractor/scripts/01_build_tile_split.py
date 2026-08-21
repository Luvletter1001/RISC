#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

from common import add_common_args, exp_path, read_jsonish
from experiments.rotation_semantic_attractor.src.dota_io import read_dota_txt
from experiments.rotation_semantic_attractor.src.utils.io import write_json


def _dataset_key(name: str) -> str:
    lowered = name.lower()
    if lowered in {"dota", "dota1", "dota-v1.0"}:
        return "dota1_angle_sweep"
    if lowered in {"dota2", "dotav2", "dota-v2.0"}:
        return "dota2_angle_sweep"
    return name


def _tags(records: list[dict]) -> list[str]:
    classes = {r.get("class_name") for r in records}
    tags = []
    if "small-vehicle" in classes:
        tags.append("true_sv_rich")
    has_conflict = any(c in classes for c in ["ship", "tennis-court", "basketball-court", "baseball-diamond", "soccer-ball-field"])
    if has_conflict:
        tags.append("cross_class_conflict_candidate")
    if "small-vehicle" not in classes and has_conflict:
        tags.append("false_sv_hub_candidate")
    if "small-vehicle" not in classes and not has_conflict:
        tags.append("low_risk_normal")
    if not tags:
        tags.append("low_risk")
    return tags


def _tiles(dataset: dict, limit: int = 0) -> list[dict]:
    root = Path(dataset["data_root"])
    img_dir = root / dataset["image_dir"]
    ann_dir = root / dataset["ann_dir"]
    tiles = []
    for img in sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg")):
        ann = ann_dir / f"{img.stem}.txt"
        if not ann.exists():
            continue
        records = read_dota_txt(ann)
        tiles.append(
            {
                "tile_id": img.stem,
                "image_path": str(img),
                "ann_path": str(ann),
                "tags": _tags(records),
                "source": "inventory",
            }
        )
        if limit and len(tiles) >= limit:
            break
    return tiles


def _write_split(out_dir: Path, name: str, dataset_name: str, seed: int, tiles: list[dict]) -> Path:
    payload = {
        "split_name": name,
        "seed": seed,
        "dataset": dataset_name,
        "num_tiles": len(tiles),
        "tiles": tiles,
    }
    path = out_dir / f"{name}.json"
    write_json(path, payload)
    return path


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--dataset", default="DOTA")
    args = parser.parse_args()
    dataset_key = _dataset_key(args.dataset)
    registry = read_jsonish(exp_path("configs", "dataset_registry.yaml"))["datasets"]
    dataset = registry[dataset_key]
    random.seed(args.seed)

    all_tiles = _tiles(dataset, limit=0)
    stress_prefixes = ["P0148", "P1384", "P2197", "P2068", "P1478"]
    stress = [t for t in all_tiles if any(t["tile_id"].startswith(prefix) for prefix in stress_prefixes)]
    if not stress:
        stress = all_tiles[:2]
    if args.limit:
        stress = stress[: args.limit]

    shuffled = list(all_tiles)
    random.shuffle(shuffled)
    s1 = shuffled[: min(500, len(shuffled))]
    s2 = shuffled[: min(2500, len(shuffled))]
    s3 = [t for t in shuffled if "cross_class_conflict_candidate" in t["tags"] or "true_sv_rich" in t["tags"]][: min(500, len(shuffled))]

    out_dir = Path(args.output_dir or exp_path("outputs", "splits"))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        _write_split(out_dir, "S0_discovery_debug", dataset_key, args.seed, stress),
        _write_split(out_dir, "S1_calibration", dataset_key, args.seed, s1),
        _write_split(out_dir, "S2_final_test", dataset_key, args.seed, s2),
        _write_split(out_dir, "S3_safety_stratified", dataset_key, args.seed, s3),
        _write_split(out_dir, "S4_cross_dataset_transfer", "cross_dataset", args.seed, []),
    ]
    report = exp_path("reports", "split_summary.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Split Summary\n\n"
        + "\n".join(f"- `{p.name}`: {read_jsonish(p)['num_tiles']} tiles" for p in paths)
        + "\n"
    )
    print(f"split_dir={out_dir}")
    print(f"split_summary={report}")


if __name__ == "__main__":
    main()
