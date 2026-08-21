#!/usr/bin/env python3
"""Benchmark angle-sweep inference batch sizes on GPUs 4,5,6,7."""
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
    angle_sweep_pred_cache_path,
    read_csv,
    run_angle_sweep_inference_parallel,
    write_csv,
)


def count_cache(cache_dir: Path, tiles: list[str], angle: int) -> int:
    return sum(
        1
        for tile in tiles
        if angle_sweep_pred_cache_path(cache_dir, tile, angle).exists()
        and angle_sweep_pred_cache_path(cache_dir, tile, angle).stat().st_size > 64
    )


def run_one(
    repo: Path,
    base_work: Path,
    tiles_all: list[str],
    angle: int,
    batch_size: int,
    gpu_ids: list[int],
) -> dict[str, Any]:
    n_images = min(len(tiles_all), batch_size * len(gpu_ids))
    tiles = tiles_all[:n_images]
    work_dir = base_work / f"bs{batch_size:04d}"
    cache_dir = work_dir / "pred_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "bench.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%F %T')}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    start = time.time()
    status = "OK"
    errors: list[str] = []
    try:
        _, errors = run_angle_sweep_inference_parallel(
            repo=repo,
            tile_ids=tiles,
            angle=angle,
            gpu_ids=gpu_ids,
            cache_dir=cache_dir,
            gt_by_id={},
            log_fn=log,
            batch_size=batch_size,
            work_dir=work_dir,
        )
        if errors:
            status = "ERR"
    except Exception as exc:  # noqa: BLE001 - benchmark records failures.
        status = "EXC"
        errors = [repr(exc)]
    elapsed = time.time() - start
    n_done = count_cache(cache_dir, tiles, angle)
    ips = n_done / elapsed if elapsed > 0 else 0.0
    row = {
        "batch_size_per_gpu": batch_size,
        "gpus": ",".join(str(x) for x in gpu_ids),
        "angle": angle,
        "requested_images": n_images,
        "done_images": n_done,
        "elapsed_sec": f"{elapsed:.3f}",
        "images_per_sec": f"{ips:.4f}",
        "images_per_30min": f"{ips * 1800:.1f}",
        "status": status,
        "n_errors": len(errors),
        "first_error": errors[0][:500] if errors else "",
        "work_dir": str(work_dir),
    }
    (work_dir / "bench_result.json").write_text(
        json.dumps(row, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=REPO)
    ap.add_argument(
        "--cohort-csv",
        type=Path,
        default=REPO / "work_dirs/exp_rotation_gt_shift_taxonomy_20260527/ftable_cohort_2500.csv",
    )
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=REPO / "work_dirs/exp_infer_bs_gpu4567_20260527",
    )
    ap.add_argument("--result-csv", type=Path, default=None)
    ap.add_argument("--angle", type=int, default=0)
    ap.add_argument("--gpu-ids", default="4,5,6,7")
    ap.add_argument("--batch-sizes", default="64,96,128,160,192")
    args = ap.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    result_csv = args.result_csv or (args.work_dir / "ftable_batch_benchmark.csv")
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    tiles = [r["tile_id"] for r in read_csv(args.cohort_csv)]
    if not tiles:
        raise RuntimeError(f"no tiles loaded from {args.cohort_csv}")

    rows: list[dict[str, Any]] = []
    for bs in batch_sizes:
        row = run_one(args.repo_root.resolve(), args.work_dir.resolve(), tiles, args.angle, bs, gpu_ids)
        rows.append(row)
        write_csv(result_csv, rows)
        print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)
        if row["status"] != "OK" or int(row["done_images"]) < int(row["requested_images"]):
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
