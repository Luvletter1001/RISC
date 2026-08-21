#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from pathlib import Path
import random

from common import exp_path, read_jsonish
from experiments.rotation_semantic_attractor.src.dota_io import read_dota_txt
from experiments.rotation_semantic_attractor.src.utils.io import write_json


REPORT = exp_path("reports", "full_split_freeze_report.md")
SPECIAL_PREFIXES = ["P0148", "P1384", "P2197", "P2068", "P1478"]
CONFLICT_CLASSES = {"ship", "tennis-court", "basketball-court", "baseball-diamond", "soccer-ball-field"}


def _tags(records: list[dict]) -> list[str]:
    classes = {r.get("class_name") for r in records}
    has_sv = "small-vehicle" in classes
    has_conflict = bool(classes & CONFLICT_CLASSES)
    tags = []
    if has_sv:
        tags.append("true_sv_rich")
    if has_conflict:
        tags.append("cross_class_conflict_candidate")
    if not has_sv and has_conflict:
        tags.append("false_sv_hub_candidate")
    if not has_sv and not has_conflict:
        tags.append("low_risk_normal")
    return tags or ["low_risk_normal"]


def _all_tiles() -> list[dict]:
    dataset = read_jsonish(exp_path("configs", "dataset_registry.yaml"))["datasets"]["dota1_angle_sweep"]
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
    return tiles


def _ensure_s3(seed: int = 20260530, target_total: int = 500) -> dict:
    path = exp_path("outputs", "splits", "S3_safety_stratified.json")
    data = read_jsonish(path) if path.exists() else {"tiles": []}
    current_tags = Counter(tag for tile in data.get("tiles", []) for tag in tile.get("tags", []))
    required = {"false_sv_hub_candidate", "true_sv_rich", "low_risk_normal", "cross_class_conflict_candidate"}
    if data.get("split_name") == "S3_safety_stratified" and required.issubset(current_tags):
        return {"changed": False, "reason": "existing S3 already contains required strata"}

    rng = random.Random(seed)
    all_tiles = _all_tiles()
    buckets = {
        "false_sv_hub_candidate": [t for t in all_tiles if "false_sv_hub_candidate" in t["tags"]],
        "true_sv_rich": [t for t in all_tiles if "true_sv_rich" in t["tags"]],
        "low_risk_normal": [t for t in all_tiles if "low_risk_normal" in t["tags"]],
        "cross_class_conflict_candidate": [t for t in all_tiles if "cross_class_conflict_candidate" in t["tags"]],
    }
    selected, seen = [], set()
    per_bucket = max(1, target_total // len(buckets))
    for key in ["false_sv_hub_candidate", "true_sv_rich", "low_risk_normal", "cross_class_conflict_candidate"]:
        candidates = list(buckets[key])
        rng.shuffle(candidates)
        for tile in candidates:
            if len([t for t in selected if key in t["tags"]]) >= per_bucket:
                break
            if tile["tile_id"] in seen:
                continue
            selected.append(tile)
            seen.add(tile["tile_id"])
    remaining = list(all_tiles)
    rng.shuffle(remaining)
    for tile in remaining:
        if len(selected) >= target_total:
            break
        if tile["tile_id"] not in seen:
            selected.append(tile)
            seen.add(tile["tile_id"])
    payload = {
        "split_name": "S3_safety_stratified",
        "seed": seed,
        "dataset": "dota1_angle_sweep",
        "num_tiles": len(selected),
        "selection_protocol": "fixed-seed stratified sample over false_sv_hub_candidate,true_sv_rich,low_risk_normal,cross_class_conflict_candidate",
        "tiles": selected,
    }
    write_json(path, payload)
    return {
        "changed": True,
        "reason": "rebuilt S3 because existing split lacked required strata",
        "candidate_counts": {key: len(value) for key, value in buckets.items()},
    }


def _split_summary(name: str, allow_scientific_table: bool) -> dict:
    path = exp_path("outputs", "splits", f"{name}.json")
    if not path.exists():
        return {"name": name, "exists": False, "path": str(path)}
    data = read_jsonish(path)
    tiles = data.get("tiles", [])
    tags = Counter(tag for tile in tiles for tag in tile.get("tags", []))
    prefixes = {
        prefix: any(str(tile.get("tile_id", "")).startswith(prefix) for tile in tiles)
        for prefix in SPECIAL_PREFIXES
    }
    sources = Counter(tile.get("source", "") for tile in tiles)
    return {
        "name": data.get("split_name", name),
        "exists": True,
        "path": str(path),
        "num_tiles": int(data.get("num_tiles", len(tiles))),
        "tile_sources": dict(sorted(sources.items())),
        "seed": data.get("seed", 20260530),
        "special_prefixes": prefixes,
        "strata": dict(sorted(tags.items())),
        "used_for_threshold_tuning": False,
        "allow_scientific_table": allow_scientific_table,
    }


def main() -> None:
    s3_action = _ensure_s3()
    summaries = [
        _split_summary("S2_final_test", True),
        _split_summary("S3_safety_stratified", True),
    ]
    lines = [
        "# Full Split Freeze Report",
        "",
        "- angle_protocol: `FULL_12 = 0,30,60,90,120,150,180,210,240,270,300,330`",
        "- seed: `20260530`",
        "- split policy: existing S2 is reused; S3 is rebuilt only if required strata are missing before any S3 full run.",
        f"- s3_action: `{s3_action}`",
        "",
    ]
    for item in summaries:
        lines.extend(
            [
                f"## {item['name']}",
                "",
                f"- exists: `{item.get('exists')}`",
                f"- path: `{item.get('path')}`",
                f"- tile_count: `{item.get('num_tiles', 0)}`",
                f"- tile_sources: `{item.get('tile_sources', {})}`",
                f"- seed: `{item.get('seed', '')}`",
                f"- contains_special_prefixes: `{item.get('special_prefixes', {})}`",
                f"- strata_counts: `{item.get('strata', {})}`",
                f"- used_for_threshold_tuning: `{item.get('used_for_threshold_tuning', False)}`",
                f"- allowed_in_scientific_table: `{item.get('allow_scientific_table', False)}`",
                "",
            ]
        )
    REPORT.write_text("\n".join(lines))
    print(f"split_freeze_report={REPORT}")


if __name__ == "__main__":
    main()
