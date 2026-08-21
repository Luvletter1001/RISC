#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import read_jsonish
from experiments.rotation_semantic_attractor.src.utils.io import read_json, write_csv, write_json


def _count_files(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern))) if path.exists() else 0


def _model_status(
    run_dir: Path,
    model_key: str,
    *,
    expected: int,
    split_name: str,
    angle_count: int,
    limit: int,
    finalize: bool,
) -> dict:
    raw_done = _count_files(run_dir / "raw_predictions" / model_key, "*/angle_*.json")
    canon_done = _count_files(run_dir / "canonical_predictions" / model_key, "*/angle_*.json")
    error_count = _count_files(run_dir / "raw_predictions" / model_key, "*/*.error.json")
    completed = min(raw_done, canon_done)

    status = "NOT_RUN"
    reason = ""
    if completed >= expected and error_count == 0:
        status = "DONE_FULL" if split_name == "S2_final_test" and angle_count >= 12 and limit == 0 else "DONE_SMOKE"
        reason = "all expected tile-angle predictions exist"
    elif error_count > 0 and completed == 0:
        status = "FAILED"
        reason = "all attempted tile-angle predictions failed"
    elif completed > 0:
        status = "FAILED" if finalize else "RUNNING"
        reason = "incomplete full model run" if finalize else "partial outputs present; run may still be active"

    return {
        "status": status,
        "reason": reason,
        "expected_tile_angle": expected,
        "completed_tile_angle": completed,
        "raw_completed_tile_angle": raw_done,
        "canonical_completed_tile_angle": canon_done,
        "failed_tile_angle": error_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest", default="manifest.json")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    manifest_path = run_dir / args.manifest
    manifest = read_json(manifest_path)
    split = read_jsonish(manifest["split_path"])
    limit = int(manifest.get("args", {}).get("limit") or 0)
    tiles = split["tiles"][: limit or None]
    angles = manifest.get("angles", [])
    expected = len(tiles) * len(angles)
    model_status = {}
    rows = []
    for model_key in manifest.get("selected_models", []):
        status = _model_status(
            run_dir,
            model_key,
            expected=expected,
            split_name=split.get("split_name", ""),
            angle_count=len(angles),
            limit=limit,
            finalize=args.finalize,
        )
        model_status[model_key] = status
        rows.append(
            {
                "model_name": model_key,
                "status": status["status"],
                "reason": status["reason"],
                "expected_tile_angle": status["expected_tile_angle"],
                "completed_tile_angle": status["completed_tile_angle"],
                "raw_completed_tile_angle": status["raw_completed_tile_angle"],
                "canonical_completed_tile_angle": status["canonical_completed_tile_angle"],
                "failed_tile_angle": status["failed_tile_angle"],
            }
        )
        write_json(run_dir / "model_status" / f"{model_key}.json", {"model_name": model_key, **status})

    manifest["model_status"] = model_status
    write_json(manifest_path, manifest)
    write_csv(run_dir / "metrics" / "run_status_by_model.csv", rows)
    print(f"manifest={manifest_path}")
    print(f"run_status={run_dir / 'metrics' / 'run_status_by_model.csv'}")


if __name__ == "__main__":
    main()
