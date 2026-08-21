#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

from common import add_common_args, read_jsonish
from experiments.rotation_semantic_attractor.src.dota_io import read_dota_txt
from experiments.rotation_semantic_attractor.src.metrics.false_hub import compute_false_hub_rows
from experiments.rotation_semantic_attractor.src.metrics.rotation_gain import oracle_best_view_rows, rotation_gain_rows
from experiments.rotation_semantic_attractor.src.utils.io import read_json, write_csv
from experiments.rotation_semantic_attractor.src.utils.status import CLAIM_CROSS_MODEL, ExperimentStatus, status_metadata

_WORKER_GT_BY_TILE = {}
_WORKER_IOU_THR = 0.30
_WORKER_ROW_META_BY_MODEL = {}


def _mean(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key) not in {"", None}]
    return sum(vals) / len(vals) if vals else 0.0


def _validmask_comparison_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["model_name"], row["tile_id"], row["angle"])][row.get("region_mode", "")] = row
    out = []
    for (model_name, tile_id, angle), modes in sorted(grouped.items()):
        all_row = modes.get("all_region")
        valid_row = modes.get("valid_mask_only")
        if not all_row or not valid_row:
            continue
        out.append(
            {
                "model_name": model_name,
                "tile_id": tile_id,
                "angle": angle,
                "all_region_fr_sv": all_row.get("fr_sv", ""),
                "valid_mask_fr_sv": valid_row.get("fr_sv", ""),
                "delta_fr_sv": float(valid_row.get("fr_sv", 0.0)) - float(all_row.get("fr_sv", 0.0)),
                "all_region_false_sv_ratio": all_row.get("false_sv_ratio", ""),
                "valid_mask_false_sv_ratio": valid_row.get("false_sv_ratio", ""),
                "delta_false_sv_ratio": float(valid_row.get("false_sv_ratio", 0.0)) - float(all_row.get("false_sv_ratio", 0.0)),
                "all_region_bg_fsv_ratio": all_row.get("bg_fsv_ratio", ""),
                "valid_mask_bg_fsv_ratio": valid_row.get("bg_fsv_ratio", ""),
                "delta_bg_fsv_ratio": float(valid_row.get("bg_fsv_ratio", 0.0)) - float(all_row.get("bg_fsv_ratio", 0.0)),
                "status": all_row.get("status", ""),
            }
        )
    return out


def _false_hub_status(split: dict, manifest: dict, model_name: str) -> dict:
    is_full = (
        split.get("split_name") == "S2_final_test"
        and len(manifest.get("angles", [])) >= 12
        and not bool(manifest.get("args", {}).get("limit", 0))
    )
    if is_full:
        model_status = manifest.get("model_status", {}).get(model_name, {})
        if manifest.get("model_status") and model_status.get("status") != ExperimentStatus.DONE_FULL:
            status = ExperimentStatus.FAILED if model_status.get("status") == ExperimentStatus.FAILED else ExperimentStatus.NOT_RUN
            return status_metadata(
                status,
                "false-hub taxonomy not promoted to DONE_FULL because model inference is incomplete or failed",
                claim_level=CLAIM_CROSS_MODEL,
                is_scientific_result=False,
                include_in_main_table=False,
            )
        return status_metadata(
            ExperimentStatus.DONE_FULL,
            "false-hub taxonomy computed on S2_final_test with full 12-angle inference",
            claim_level=CLAIM_CROSS_MODEL,
        )
    return status_metadata(
        ExperimentStatus.DONE_SMOKE,
        "false-hub taxonomy computed from smoke predictions",
        claim_level=CLAIM_CROSS_MODEL,
        is_scientific_result=False,
        include_in_main_table=False,
    )


def _prediction_identity(pred_path: Path) -> tuple[str, str, int]:
    model = pred_path.parts[-3]
    tile_id = pred_path.parts[-2]
    angle = int(pred_path.stem.split("_")[-1])
    return model, tile_id, angle


def _prediction_task(pred_path: Path, gt_records: list[dict], iou_thr: float, row_meta: dict) -> dict:
    return {
        "pred_path": str(pred_path),
        "gt_records": gt_records,
        "iou_thr": iou_thr,
        "row_meta": row_meta,
    }


def _compute_prediction_task(task: dict) -> tuple[list[dict], list[dict]]:
    pred_path = Path(task["pred_path"])
    model, tile_id, angle = _prediction_identity(pred_path)
    data = read_json(pred_path)
    preds = data.get("final_predictions", [])
    rows, absorption_rows = [], []
    base_row, base_absorption = compute_false_hub_rows(
        model,
        tile_id,
        angle,
        preds,
        task.get("gt_records", []),
        task.get("iou_thr", 0.30),
        "all_region",
    )
    for region_mode in ["all_region", "valid_mask_only"]:
        row = {**base_row, "region_mode": region_mode}
        row.update(task.get("row_meta", {}))
        rows.append(row)
        for absorption in base_absorption:
            absorption_rows.append({**absorption, "region_mode": region_mode})
    return rows, absorption_rows


def _collect_task_results(results) -> tuple[list[dict], list[dict]]:
    rows, absorption_rows = [], []
    for task_rows, task_absorption_rows in results:
        rows.extend(task_rows)
        absorption_rows.extend(task_absorption_rows)
    rows.sort(key=lambda r: (r["model_name"], r["tile_id"], int(r["angle"]), r.get("region_mode", "")))
    absorption_rows.sort(
        key=lambda r: (
            r["model_name"],
            r["tile_id"],
            int(r["angle"]),
            r.get("region_mode", ""),
            r.get("gt_class", ""),
        )
    )
    return rows, absorption_rows


def _compute_prediction_tasks(tasks: list[dict], workers: int = 1) -> tuple[list[dict], list[dict]]:
    if workers <= 1 or len(tasks) <= 1:
        return _collect_task_results(_compute_prediction_task(task) for task in tasks)
    chunksize = max(1, len(tasks) // (workers * 8))
    with Pool(processes=workers) as pool:
        return _collect_task_results(pool.imap_unordered(_compute_prediction_task, tasks, chunksize=chunksize))


def _init_path_worker(gt_by_tile: dict, iou_thr: float, row_meta_by_model: dict) -> None:
    global _WORKER_GT_BY_TILE, _WORKER_IOU_THR, _WORKER_ROW_META_BY_MODEL
    _WORKER_GT_BY_TILE = gt_by_tile
    _WORKER_IOU_THR = iou_thr
    _WORKER_ROW_META_BY_MODEL = row_meta_by_model


def _compute_prediction_path(pred_path_str: str) -> tuple[list[dict], list[dict]]:
    pred_path = Path(pred_path_str)
    model, tile_id, _ = _prediction_identity(pred_path)
    task = _prediction_task(
        pred_path,
        _WORKER_GT_BY_TILE.get(tile_id, []),
        _WORKER_IOU_THR,
        _WORKER_ROW_META_BY_MODEL.get(model, {}),
    )
    return _compute_prediction_task(task)


def _compute_prediction_paths(
    pred_paths: list[Path],
    gt_by_tile: dict,
    iou_thr: float,
    row_meta_by_model: dict,
    workers: int = 1,
) -> tuple[list[dict], list[dict]]:
    if workers <= 1 or len(pred_paths) <= 1:
        tasks = [
            _prediction_task(path, gt_by_tile.get(_prediction_identity(path)[1], []), iou_thr, row_meta_by_model.get(_prediction_identity(path)[0], {}))
            for path in pred_paths
        ]
        return _compute_prediction_tasks(tasks, workers=1)
    chunksize = max(1, len(pred_paths) // (workers * 8))
    with Pool(
        processes=workers,
        initializer=_init_path_worker,
        initargs=(gt_by_tile, iou_thr, row_meta_by_model),
    ) as pool:
        return _collect_task_results(pool.imap_unordered(_compute_prediction_path, [str(path) for path in pred_paths], chunksize=chunksize))


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--iou-thr", type=float, default=0.30)
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes for prediction JSON parsing.")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    manifest = read_json(run_dir / "manifest.json")
    split = read_jsonish(manifest["split_path"])
    gt_by_tile = {tile["tile_id"]: read_dota_txt(tile["ann_path"]) for tile in split["tiles"]}

    common_row_meta = {
        "experiment_family": "false_hub_benchmark",
        "split_name": split.get("split_name", ""),
        "angle_set": ",".join(str(a) for a in manifest.get("angles", [])),
        "actual_inference_run": True,
        "has_final_predictions": True,
        "has_gt_matching": True,
        "has_false_sv_taxonomy": True,
        "is_full_split": split.get("split_name") == "S2_final_test",
        "is_smoke_limit": bool(manifest.get("args", {}).get("limit", 0)),
    }
    selected_models = sorted({path.parts[-3] for path in (run_dir / "canonical_predictions").glob("*/*/angle_*.json")})
    row_meta_by_model = {model: {**_false_hub_status(split, manifest, model), **common_row_meta} for model in selected_models}
    pred_paths = sorted((run_dir / "canonical_predictions").glob("*/*/angle_*.json"))
    rows, absorption_rows = _compute_prediction_paths(pred_paths, gt_by_tile, args.iou_thr, row_meta_by_model, workers=max(1, args.workers))

    metrics_dir = Path(args.output_dir or run_dir / "metrics")
    write_csv(metrics_dir / "false_hub_tile_angle.csv", rows)
    write_csv(metrics_dir / "per_class_absorption.csv", absorption_rows)
    validmask_rows = _validmask_comparison_rows(rows)
    write_csv(metrics_dir / "validmask_comparison.csv", validmask_rows)

    by_model, by_angle = defaultdict(list), defaultdict(list)
    for row in rows:
        by_model[row["model_name"]].append(row)
        by_angle[(row["model_name"], row["angle"])].append(row)
    model_summary = []
    for model, items in sorted(by_model.items()):
        false_hub_meta = _false_hub_status(split, manifest, model)
        model_summary.append(
            {
                "model_name": model,
                "mean_final_fsv": _mean(items, "false_sv_ratio"),
                "mean_bg_fsv": _mean(items, "bg_fsv_ratio"),
                "mean_fr_sv": _mean(items, "fr_sv"),
                "mean_true_sv_recall": _mean(items, "true_sv_recall"),
                "num_rows": len(items),
                "experiment_family": "false_hub_benchmark",
                "split_name": split.get("split_name", ""),
                "angle_set": ",".join(str(a) for a in manifest.get("angles", [])),
                "actual_inference_run": True,
                "has_final_predictions": True,
                "has_gt_matching": True,
                "has_false_sv_taxonomy": True,
                "is_full_split": split.get("split_name") == "S2_final_test",
                "is_smoke_limit": bool(manifest.get("args", {}).get("limit", 0)),
                **false_hub_meta,
            }
        )
    angle_summary = [
        {
            "model_name": key[0],
            "angle": key[1],
            "mean_final_fsv": _mean(items, "false_sv_ratio"),
            "mean_bg_fsv": _mean(items, "bg_fsv_ratio"),
            "num_rows": len(items),
        }
        for key, items in sorted(by_angle.items())
    ]
    write_csv(metrics_dir / "false_hub_summary_by_model.csv", model_summary)
    write_csv(metrics_dir / "false_hub_summary_by_angle.csv", angle_summary)
    gain_rows, concentration = rotation_gain_rows(rows)
    write_csv(metrics_dir / "rotation_gain.csv", gain_rows)
    write_csv(metrics_dir / "concentration.csv", concentration)
    oracle_rows = oracle_best_view_rows(rows)
    for row in oracle_rows:
        row.update(
            status_metadata(
                ExperimentStatus.DONE_SMOKE,
                "non-deployable oracle best-view computed from actual per-angle rows",
                claim_level=CLAIM_CROSS_MODEL,
                is_scientific_result=False,
                include_in_main_table=False,
            )
        )
    write_csv(metrics_dir / "oracle_best_view.csv", oracle_rows)
    if any(row.get("status") == ExperimentStatus.DONE_FULL for row in model_summary):
        write_csv(metrics_dir / "full_closedset_false_hub_tile_angle.csv", rows)
        write_csv(metrics_dir / "full_closedset_false_hub_summary_by_model.csv", model_summary)
        write_csv(metrics_dir / "full_closedset_false_hub_summary_by_angle.csv", angle_summary)
        write_csv(metrics_dir / "full_closedset_rotation_gain.csv", gain_rows)
        write_csv(metrics_dir / "full_closedset_concentration.csv", concentration)
        write_csv(metrics_dir / "full_closedset_per_class_absorption.csv", absorption_rows)
        write_csv(metrics_dir / "full_closedset_validmask_comparison.csv", validmask_rows)
        write_csv(metrics_dir / "full_oracle_best_view.csv", oracle_rows)
    print(f"false_hub_metrics={metrics_dir / 'false_hub_tile_angle.csv'}")


if __name__ == "__main__":
    main()
