#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

from common import add_common_args
from experiments.rotation_semantic_attractor.src.metrics.stage_metrics import compute_stage_row
from experiments.rotation_semantic_attractor.src.utils.status import ExperimentStatus, can_be_done_full
from experiments.rotation_semantic_attractor.src.utils.io import read_json, write_csv


def _prediction_identity(raw_path: Path) -> tuple[str, str, int]:
    model = raw_path.parts[-3]
    tile_id = raw_path.parts[-2]
    angle = int(raw_path.stem.split("_")[-1])
    return model, tile_id, angle


def _compute_stage_path(raw_path_str: str) -> dict:
    raw_path = Path(raw_path_str)
    model, tile_id, angle = _prediction_identity(raw_path)
    return compute_stage_row(model, tile_id, angle, read_json(raw_path))


def _compute_stage_paths(raw_paths: list[Path], workers: int = 1) -> list[dict]:
    if workers <= 1 or len(raw_paths) <= 1:
        rows = [_compute_stage_path(str(path)) for path in raw_paths]
    else:
        chunksize = max(1, len(raw_paths) // (workers * 8))
        with Pool(processes=workers) as pool:
            rows = list(pool.imap_unordered(_compute_stage_path, [str(path) for path in raw_paths], chunksize=chunksize))
    rows.sort(key=lambda r: (r["model_name"], r["tile_id"], int(r["angle"])))
    return rows


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes for raw prediction JSON parsing.")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    manifest = read_json(run_dir / "manifest.json") if (run_dir / "manifest.json").exists() else {}
    split_name = manifest.get("split_name", "")
    angle_set = ",".join(str(a) for a in manifest.get("angles", []))
    is_full_protocol = (
        split_name == "S2_final_test"
        and len(manifest.get("angles", [])) >= 12
        and not bool(manifest.get("args", {}).get("limit", 0))
    )
    raw_paths = sorted((run_dir / "raw_predictions").glob("*/*/angle_*.json"))
    rows = _compute_stage_paths(raw_paths, workers=max(1, args.workers))
    metrics_dir = Path(args.output_dir or run_dir / "metrics")
    write_csv(metrics_dir / "stage_decomposition_tile_angle.csv", rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model_name"]].append(row)
    summary = []
    for model, items in grouped.items():
        post_vals = [float(r["post_nms_fr_sv"]) for r in items if r.get("post_nms_fr_sv") != ""]
        statuses = {r.get("status") for r in items}
        model_run_status = manifest.get("model_status", {}).get(model, {}).get("status", "")
        if ExperimentStatus.FAILED in statuses:
            status = ExperimentStatus.FAILED
        elif ExperimentStatus.UNSUPPORTED_BY_CURRENT_CODE in statuses:
            status = ExperimentStatus.UNSUPPORTED_BY_CURRENT_CODE
        else:
            status = ExperimentStatus.DONE_SMOKE
        dense_status = items[0].get("dense_logits_status", "")
        pre_nms_status = items[0].get("pre_nms_status", "")
        post_nms_status = items[0].get("post_nms_status", "")
        if is_full_protocol and model_run_status == ExperimentStatus.DONE_FULL:
            dense_status = ExperimentStatus.DONE_FULL if dense_status == ExperimentStatus.DONE_SMOKE else dense_status
            pre_nms_status = ExperimentStatus.DONE_FULL if pre_nms_status == ExperimentStatus.DONE_SMOKE else pre_nms_status
            post_nms_status = ExperimentStatus.DONE_FULL if post_nms_status == ExperimentStatus.DONE_SMOKE else post_nms_status
        row = {
            "model_name": model,
            "model_family": items[0].get("model_family", ""),
            "architecture_type": items[0].get("architecture_type", ""),
            "mean_post_nms_fr_sv": sum(post_vals) / len(post_vals) if post_vals else 0.0,
            "unsupported_reason": ";".join(sorted({r["unsupported_reason"] for r in items if r.get("unsupported_reason")})),
            "dense_logits_status": dense_status,
            "dense_logits_reason": items[0].get("dense_logits_reason", ""),
            "query_logits_status": items[0].get("query_logits_status", ""),
            "query_logits_reason": items[0].get("query_logits_reason", ""),
            "pre_nms_status": pre_nms_status,
            "pre_nms_reason": items[0].get("pre_nms_reason", ""),
            "post_nms_status": post_nms_status,
            "post_nms_reason": items[0].get("post_nms_reason", ""),
            "stage_metric_used": ",".join(sorted({r.get("stage_metric_used", "") for r in items if r.get("stage_metric_used")})),
            "experiment_family": "stage_decomposition",
            "split_name": split_name,
            "angle_set": angle_set,
            "actual_inference_run": True,
            "is_smoke_limit": bool(manifest.get("args", {}).get("limit", 0)),
            "status": status,
        }
        if is_full_protocol and model_run_status == ExperimentStatus.DONE_FULL:
            row["status"] = ExperimentStatus.DONE_FULL if can_be_done_full("stage_decomposition", {**row, "status": ExperimentStatus.DONE_FULL}) else status
        summary.append(row)
    write_csv(metrics_dir / "stage_decomposition_summary.csv", summary)
    print(f"stage_metrics={metrics_dir / 'stage_decomposition_summary.csv'}")


if __name__ == "__main__":
    main()
