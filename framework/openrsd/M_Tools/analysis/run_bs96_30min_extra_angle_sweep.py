#!/usr/bin/env python3
"""Run about 30 minutes of extra 12-angle inference at batch/GPU=96."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import sys

REPO = Path("/data1/zcy/OpenRSD")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from M_Tools.analysis.run_false_sv_hub_mining_20260526 import (  # noqa: E402
    ANGLE_SWEEP_REL,
    angle_sweep_pred_cache_path,
    read_csv,
    run_angle_sweep_inference_parallel,
    write_csv,
)
from M_Tools.analysis.run_rsv_taxonomy_suite import ANGLES_12  # noqa: E402


def load_or_select_tiles(repo: Path, work_dir: Path, target_tiles: int) -> list[str]:
    tile_csv = work_dir / "ftable_extra_tiles.csv"
    if tile_csv.exists() and tile_csv.stat().st_size > 20:
        return [r["tile_id"] for r in read_csv(tile_csv)]

    ann_dir = repo / ANGLE_SWEEP_REL / "angle_000" / "annfiles"
    all_tiles = sorted(p.stem for p in ann_dir.glob("*.txt"))
    existing = set()
    existing_csv = repo / "work_dirs/exp_rotation_gt_shift_taxonomy_20260527/ftable_cohort_2500.csv"
    if existing_csv.exists():
        existing.update(r["tile_id"] for r in read_csv(existing_csv))

    chosen = [t for t in all_tiles if t not in existing][:target_tiles]
    write_csv(
        tile_csv,
        [{"rank": i, "tile_id": t, "source": "angle_sweep_val_extra_non2500"} for i, t in enumerate(chosen, 1)],
        ["rank", "tile_id", "source"],
    )
    return chosen


def count_done(cache_dir: Path, tiles: list[str], angles: list[int]) -> int:
    return sum(
        1
        for tile in tiles
        for angle in angles
        if angle_sweep_pred_cache_path(cache_dir, tile, angle).exists()
        and angle_sweep_pred_cache_path(cache_dir, tile, angle).stat().st_size > 64
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=REPO)
    ap.add_argument("--work-dir", type=Path, default=REPO / "work_dirs/exp_bs96_30min_extra_angle_sweep_20260527")
    ap.add_argument("--target-tiles", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--gpu-ids", default="4,5,6,7")
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    work_dir = args.work_dir.resolve()
    cache_dir = work_dir / "pred_cache"
    log_dir = work_dir / "logs"
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    tiles = load_or_select_tiles(repo, work_dir, args.target_tiles)

    start = time.time()
    events: list[dict[str, Any]] = []
    log_path = log_dir / "run.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%F %T')}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(
        f"START target_tiles={len(tiles)} angles={len(ANGLES_12)} "
        f"batch_size={args.batch_size} gpus={gpu_ids}",
    )
    errors_total: list[str] = []
    for angle in ANGLES_12:
        angle_start = time.time()
        _, errors = run_angle_sweep_inference_parallel(
            repo=repo,
            tile_ids=tiles,
            angle=angle,
            gpu_ids=gpu_ids,
            cache_dir=cache_dir,
            gt_by_id={},
            log_fn=log,
            batch_size=args.batch_size,
            work_dir=work_dir,
        )
        elapsed = time.time() - angle_start
        done_angle = count_done(cache_dir, tiles, [angle])
        errors_total.extend(errors)
        row = {
            "angle": angle,
            "done_angle": done_angle,
            "target_tiles": len(tiles),
            "elapsed_sec": f"{elapsed:.3f}",
            "errors": len(errors),
            "first_error": errors[0][:300] if errors else "",
        }
        events.append(row)
        write_csv(work_dir / "ftable_angle_progress.csv", events)
        log(f"ANGLE_DONE {json.dumps(row, ensure_ascii=False)}")
        if errors:
            break

    total_elapsed = time.time() - start
    total_done = count_done(cache_dir, tiles, ANGLES_12)
    complete_tiles = sum(
        1
        for tile in tiles
        if all(
            angle_sweep_pred_cache_path(cache_dir, tile, angle).exists()
            and angle_sweep_pred_cache_path(cache_dir, tile, angle).stat().st_size > 64
            for angle in ANGLES_12
        )
    )
    summary = {
        "status": "OK" if not errors_total else "ERR",
        "target_tiles": len(tiles),
        "complete_12angle_tiles": complete_tiles,
        "done_image_angles": total_done,
        "expected_image_angles": len(tiles) * len(ANGLES_12),
        "elapsed_sec": f"{total_elapsed:.3f}",
        "image_angles_per_sec": f"{total_done / total_elapsed:.4f}" if total_elapsed > 0 else "0.0000",
        "batch_size_per_gpu": args.batch_size,
        "gpus": ",".join(str(x) for x in gpu_ids),
        "work_dir": str(work_dir),
        "cache_dir": str(cache_dir),
        "n_errors": len(errors_total),
        "first_error": errors_total[0][:500] if errors_total else "",
    }
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"SUMMARY {json.dumps(summary, ensure_ascii=False)}")
    return 0 if not errors_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
