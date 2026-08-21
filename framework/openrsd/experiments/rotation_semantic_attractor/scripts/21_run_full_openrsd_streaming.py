#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import shutil
import time
import traceback
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from common import PROJECT_ROOT, add_common_args, exp_path, parse_angles
from experiments.rotation_semantic_attractor.src.dota_io import read_dota_txt, records_to_polygons
from experiments.rotation_semantic_attractor.src.metrics.context_metrics import write_context_counterfactual_images
from experiments.rotation_semantic_attractor.src.metrics.dehub_safety import (
    class_distribution_from_histograms,
    js_kl_divergence,
    mean_float,
)
from experiments.rotation_semantic_attractor.src.model_adapters.open_vocab_hooks import tensor_checksum
from experiments.rotation_semantic_attractor.src.model_adapters.openrsd_adapter import OpenRSDAdapter
from experiments.rotation_semantic_attractor.src.rotation import rotate_image_and_polygons
from experiments.rotation_semantic_attractor.src.utils.io import read_json, write_json
from experiments.rotation_semantic_attractor.src.utils.status import (
    CLAIM_DEHUB_REPAIR,
    CLAIM_NONE,
    CLAIM_OPEN_VOCAB_EMBEDDING,
    ExperimentStatus,
    status_metadata,
)


FULL_12_ANGLES = list(range(0, 360, 30))
DEFAULT_INTERVENTIONS = {
    "open_vocab_benchmark": ["original"],
    "open_vocab_causal": ["original", "zero_sv", "swap_sv_lv", "norm_sv_mean", "normalize_all", "random_sv"],
    "context_counterfactual": ["original"],
    "dehub_safety": ["original"],
}
DEFAULT_CONTEXT_CONDITIONS = ["object_only", "context_only"]
BASE_CONFIG = PROJECT_ROOT / "M_configs/Step2_A10_Large_Pretrain_Stage3/A10_flex_rtm_v3_1_formal.py"
BASE_CHECKPOINT = PROJECT_ROOT / "results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth"
REPAIR_CONFIG = PROJECT_ROOT / "M_configs/experiments/sv_dehub_lite_v1.py"
REPAIR_CANDIDATES = [
    PROJECT_ROOT / "work_dirs/exp_sv_dehub_lite_train_gpu89/train_v1/iter_3000.pth",
    PROJECT_ROOT / "work_dirs/exp_sv_dehub_lite_train_gpu89/train_v1/iter_2000.pth",
    PROJECT_ROOT / "work_dirs/exp_sv_dehub_lite_train_gpu89/train_v1/iter_1000.pth",
    PROJECT_ROOT / "work_dirs/exp_next_plan_sv_dehub_step23_overnight/branch_B_fresh_epoch24_to8k/iter_8000.pth",
]

ROW_FIELDS = [
    "run_family",
    "split_name",
    "checkpoint_name",
    "model_name",
    "model_family",
    "config",
    "checkpoint",
    "support_pkl",
    "tile_id",
    "angle",
    "condition",
    "intervention",
    "tags",
    "num_gt_total",
    "num_gt_sv",
    "risk_group",
    "image_stem",
    "source_image_path",
    "source_ann_path",
    "retained_image_path",
    "rotated_asset_retained",
    "counterfactual_is_real_image_level",
    "completed_conditions",
    "completed_condition_count",
    "target_gt_class",
    "num_target_polygons",
    "detection_total",
    "small_vehicle_count",
    "large_vehicle_count",
    "small_vehicle_ratio",
    "large_vehicle_ratio",
    "top1_class",
    "mean_score",
    "max_score",
    "class_histogram",
    "has_open_vocab_config",
    "has_checkpoint",
    "has_prompt_or_class_embedding_path",
    "is_prompt_only",
    "is_embedding_level",
    "is_visual_support_level",
    "actual_embedding_modified",
    "original_embedding_checksum",
    "modified_embedding_checksum",
    "modified_tensor_name",
    "reran_inference",
    "has_paired_comparison",
    "actual_inference_run",
    "same_split",
    "same_angles",
    "same_threshold",
    "same_evaluator",
    "has_false_hub_metrics",
    "has_true_sv_preservation",
    "has_det_per_img",
    "has_class_distribution_js_kl",
    "has_lowrisk_inflation",
    "has_ap50_or_proxy",
    "status",
    "status_reason",
    "is_proxy",
    "is_scientific_result",
    "include_in_main_table",
    "claim_level",
    "unsupported_reason",
    "traceback",
]

NORM_FIELDS = [
    "run_family",
    "split_name",
    "checkpoint_name",
    "intervention",
    "phase",
    "class_name",
    "mean_norm",
    "min_norm",
    "max_norm",
    "sv_lv_mean_cosine",
]

ASSET_FIELDS = [
    "run_family",
    "split_name",
    "checkpoint_name",
    "tile_id",
    "angle",
    "condition",
    "image_stem",
    "transient_image_path",
    "transient_ann_pkl_path",
    "transient_ann_txt_path",
    "source_image_path",
    "source_ann_path",
    "retained_after_chunk",
    "scratch_chunk",
]


def _default_repair_checkpoint() -> Path:
    for path in REPAIR_CANDIDATES:
        if path.exists():
            return path
    return REPAIR_CANDIDATES[0]


def _append_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _done_keys_from_file(rows_path: Path) -> set[tuple[str, str, int, str, str]]:
    keys = set()
    for row in _read_csv(rows_path):
        try:
            keys.add(
                (
                    row.get("checkpoint_name", ""),
                    row.get("tile_id", ""),
                    int(float(row.get("angle") or 0)),
                    row.get("condition", ""),
                    row.get("intervention", ""),
                )
            )
        except Exception:
            continue
    return keys


def _done_keys(rows_path: Path, extra_rows: list[Path] | None = None) -> set[tuple[str, str, int, str, str]]:
    keys = _done_keys_from_file(rows_path)
    for path in extra_rows or []:
        keys.update(_done_keys_from_file(path))
    return keys


def _parse_path_list(value: str) -> list[Path]:
    if not value.strip():
        return []
    return [Path(item.strip()) for item in value.replace(",", " ").split() if item.strip()]


def _safe_postprocess_cfg(head, args):
    cfg = copy.deepcopy(getattr(head, "test_cfg", None) or {})
    if args.postprocess_score_thr is not None:
        cfg["score_thr"] = float(args.postprocess_score_thr)
    if int(args.nms_pre) > 0:
        current = int(cfg.get("nms_pre", args.nms_pre) or args.nms_pre)
        cfg["nms_pre"] = min(current, int(args.nms_pre))
    if int(args.max_per_img) > 0:
        current = int(cfg.get("max_per_img", args.max_per_img) or args.max_per_img)
        cfg["max_per_img"] = min(current, int(args.max_per_img))
    return cfg


def _chunks(items: list[dict], size: int) -> Iterable[list[dict]]:
    size = max(1, int(size))
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _chunk_done(chunk: list[dict], checkpoint_name: str, args, done: set[tuple[str, str, int, str, str]]) -> bool:
    if args.run_family == "context_counterfactual":
        return False
    expected = {
        (checkpoint_name, task["tile"]["tile_id"], int(task["angle"]), "original", intervention)
        for task in chunk
        for intervention in args.interventions
    }
    return bool(expected) and expected.issubset(done)


def _apply_task_shard(tasks: list[dict], args) -> list[dict]:
    shard_count = int(args.task_shard_count)
    shard_index = int(args.task_shard_index)
    if shard_count <= 1:
        return tasks
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"task_shard_index must be in [0, {shard_count}), got {shard_index}")
    return [task for idx, task in enumerate(tasks) if idx % shard_count == shard_index]


def _first_openrsd_pair(inventory_path: Path) -> dict | None:
    inventory = read_json(inventory_path)
    for pair in inventory.get("runnable_pairs", []):
        if pair.get("candidate_model_family") == "openrsd" and pair.get("usable", True):
            return pair
    return None


def _support_from_pair_or_args(args) -> dict:
    pair = _first_openrsd_pair(Path(args.inventory)) if Path(args.inventory).exists() else None
    config = args.config or (pair or {}).get("config") or str(BASE_CONFIG)
    checkpoint = args.checkpoint or (pair or {}).get("checkpoint") or str(BASE_CHECKPOINT)
    support_pkl = args.support_pkl or (pair or {}).get("support_pkl") or str(
        PROJECT_ROOT / "data/DOTA1_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl"
    )
    return {
        "name": "OpenRSD full streaming",
        "model_type": "openrsd",
        "box_type": "obb",
        "adapter": "openrsd",
        "config": str(config),
        "checkpoint": str(checkpoint),
        "support_pkl": str(support_pkl),
        "prompt_protocol": str(exp_path("configs", "open_vocab_prompt_protocol.json")),
        "support_type": "visual",
        "support_shot": int(args.support_shot),
        "score_thr": float(args.score_thr),
        "iou_thr": float(args.iou_thr),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "result_md_dir": str(Path(args.output_dir) / "reports"),
    }


def _write_openrsd_ann_pkl(records: list[dict], pkl_path: Path) -> int:
    records = [record for record in records if "polygon" in record]
    texts = [record["class_name"] for record in records]
    polys = []
    for record in records:
        flat = []
        for point in record["polygon"]:
            flat.extend([float(point[0]), float(point[1])])
        polys.append(flat)
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open("wb") as f:
        pickle.dump(
            {
                "texts": texts,
                "polys": np.asarray(polys, dtype=np.float32),
                "text_embeds": None,
                "visual_embeds": None,
                "cls_list": texts,
                "ict_support_dict": None,
            },
            f,
        )
    return len(records)


def _write_dota_txt(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for record in records:
        if "polygon" not in record:
            continue
        coords = []
        for x, y in record["polygon"]:
            coords.extend([f"{float(x):.2f}", f"{float(y):.2f}"])
        lines.append(" ".join(coords + [record.get("class_name", ""), str(record.get("difficulty", 0))]))
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _gt_counts(records: list[dict]) -> tuple[int, int]:
    total = sum(1 for record in records if "polygon" in record)
    sv = sum(1 for record in records if record.get("class_name") == "small-vehicle" and "polygon" in record)
    return total, sv


def _risk_group(tile: dict, num_gt_sv: int) -> str:
    tags = set(tile.get("tags", []))
    if "low_risk_normal" in tags:
        return "low_risk_normal"
    if "true_sv_rich" in tags:
        return "true_sv_rich"
    if num_gt_sv > 0:
        return "sv_present"
    if "false_sv_hub_candidate" in tags:
        return "false_sv_hub_candidate_no_sv"
    return "no_sv"


def _make_status(args, full_reason: str, smoke_reason: str, claim_level: str = CLAIM_NONE) -> dict:
    status = ExperimentStatus.DONE_SMOKE if args.limit else ExperimentStatus.DONE_FULL
    return status_metadata(
        status,
        smoke_reason if args.limit else full_reason,
        claim_level=claim_level,
        is_scientific_result=(not args.limit),
        include_in_main_table=(not args.limit),
    )


def _base_row(args, tile: dict, angle: int, checkpoint_name: str, cfg: dict, *, condition: str, intervention: str) -> dict:
    return {
        "run_family": args.run_family,
        "split_name": args.split_name,
        "checkpoint_name": checkpoint_name,
        "model_name": checkpoint_name,
        "model_family": "openrsd",
        "config": cfg["config"],
        "checkpoint": cfg["checkpoint"],
        "support_pkl": cfg["support_pkl"],
        "tile_id": tile["tile_id"],
        "angle": int(angle),
        "condition": condition,
        "intervention": intervention,
        "tags": ";".join(tile.get("tags", [])),
        "source_image_path": tile.get("image_path", ""),
        "source_ann_path": tile.get("ann_path", ""),
        "retained_image_path": "",
        "rotated_asset_retained": False,
        "unsupported_reason": "",
        "traceback": "",
    }


def _prepare_rotated_original(task: dict, scratch: Path, args, checkpoint_name: str) -> tuple[dict, list[dict]]:
    tile = task["tile"]
    angle = int(task["angle"])
    image = Image.open(tile["image_path"]).convert("RGB")
    source_records = read_dota_txt(tile["ann_path"])
    result = rotate_image_and_polygons(image, records_to_polygons(source_records), angle_deg=angle, expand=True)
    rotated_records = []
    for record, polygon in zip(source_records, result.polygons):
        new_record = dict(record)
        new_record["polygon"] = polygon
        rotated_records.append(new_record)

    images_dir = scratch / "images"
    ann_dir = scratch / "annfiles"
    txt_dir = scratch / "ann_txt"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{tile['tile_id']}_rot{angle:03d}"
    image_path = images_dir / f"{stem}.png"
    ann_pkl = ann_dir / f"{stem}.pkl"
    ann_txt = txt_dir / f"{stem}.txt"
    result.image.save(image_path)
    _write_openrsd_ann_pkl(rotated_records, ann_pkl)
    _write_dota_txt(ann_txt, rotated_records)
    total, sv = _gt_counts(rotated_records)
    meta = {
        "tile": tile,
        "angle": angle,
        "condition": "original",
        "image_stem": stem,
        "image_path": str(image_path),
        "ann_pkl": str(ann_pkl),
        "ann_txt": str(ann_txt),
        "num_gt_total": total,
        "num_gt_sv": sv,
        "risk_group": _risk_group(tile, sv),
        "num_target_polygons": sv,
        "completed_conditions": "",
        "completed_condition_count": 0,
        "target_gt_class": "small-vehicle",
    }
    asset = {
        "run_family": args.run_family,
        "split_name": args.split_name,
        "checkpoint_name": checkpoint_name,
        "tile_id": tile["tile_id"],
        "angle": angle,
        "condition": "original",
        "image_stem": stem,
        "transient_image_path": str(image_path),
        "transient_ann_pkl_path": str(ann_pkl),
        "transient_ann_txt_path": str(ann_txt),
        "source_image_path": tile.get("image_path", ""),
        "source_ann_path": tile.get("ann_path", ""),
        "retained_after_chunk": False,
        "scratch_chunk": str(scratch),
    }
    return meta, [asset]


def _prepare_context_conditions(original: dict, scratch: Path, args, checkpoint_name: str) -> tuple[list[dict], list[dict], list[dict]]:
    tile = original["tile"]
    angle = original["angle"]
    if original["num_gt_sv"] <= 0:
        meta = status_metadata(
            ExperimentStatus.NOT_APPLICABLE,
            "no small-vehicle GT polygon available for image-level context counterfactual",
            claim_level=CLAIM_NONE,
            is_scientific_result=False,
            include_in_main_table=False,
        )
        row = {
            **_base_row(args, tile, angle, checkpoint_name, {"config": "", "checkpoint": "", "support_pkl": ""}, condition="no_target_sv", intervention="original"),
            "num_gt_total": original["num_gt_total"],
            "num_gt_sv": original["num_gt_sv"],
            "risk_group": original["risk_group"],
            "target_gt_class": "small-vehicle",
            "counterfactual_is_real_image_level": False,
            "completed_conditions": "",
            "completed_condition_count": 0,
            "num_target_polygons": 0,
            "reran_inference": False,
            "actual_inference_run": False,
            **meta,
            "unsupported_reason": meta["status_reason"],
        }
        return [], [], [row]

    cf = write_context_counterfactual_images(original["image_path"], original["ann_txt"], scratch / "context_work" / original["image_stem"])
    conditions = [c for c in args.context_conditions if c in cf.get("conditions", {})]
    completed = ";".join(sorted(conditions))
    metas: list[dict] = []
    assets: list[dict] = []
    images_dir = scratch / "images"
    ann_dir = scratch / "annfiles"
    for condition in conditions:
        stem = f"{original['image_stem']}__{condition}"
        image_path = images_dir / f"{stem}.png"
        ann_pkl = ann_dir / f"{stem}.pkl"
        shutil.copy2(cf["conditions"][condition], image_path)
        shutil.copy2(original["ann_pkl"], ann_pkl)
        meta = dict(original)
        meta.update(
            {
                "condition": condition,
                "image_stem": stem,
                "image_path": str(image_path),
                "ann_pkl": str(ann_pkl),
                "completed_conditions": completed,
                "completed_condition_count": len(conditions),
                "num_target_polygons": int(cf.get("num_target_polygons") or 0),
            }
        )
        metas.append(meta)
        assets.append(
            {
                "run_family": args.run_family,
                "split_name": args.split_name,
                "checkpoint_name": checkpoint_name,
                "tile_id": tile["tile_id"],
                "angle": angle,
                "condition": condition,
                "image_stem": stem,
                "transient_image_path": str(image_path),
                "transient_ann_pkl_path": str(ann_pkl),
                "transient_ann_txt_path": original["ann_txt"],
                "source_image_path": tile.get("image_path", ""),
                "source_ann_path": tile.get("ann_path", ""),
                "retained_after_chunk": False,
                "scratch_chunk": str(scratch),
            }
        )
    return metas, assets, []


def _prepare_chunk(chunk: list[dict], scratch: Path, args, checkpoint_name: str) -> tuple[dict, list[dict], list[dict]]:
    item_by_stem: dict[str, dict] = {}
    assets: list[dict] = []
    direct_rows: list[dict] = []
    for task in chunk:
        original, original_assets = _prepare_rotated_original(task, scratch, args, checkpoint_name)
        if args.run_family == "context_counterfactual":
            condition_items, condition_assets, skipped = _prepare_context_conditions(original, scratch, args, checkpoint_name)
            for item in condition_items:
                item_by_stem[item["image_stem"]] = item
            assets.extend(condition_assets)
            direct_rows.extend(skipped)
        else:
            item_by_stem[original["image_stem"]] = original
            assets.extend(original_assets)
    dataset = {
        "dataset_root": str(scratch),
        "images_dir": str(scratch / "images"),
        "ann_dir": str(scratch / "annfiles"),
        "items": list(item_by_stem.values()),
    }
    return {"dataset": dataset, "item_by_stem": item_by_stem}, assets, direct_rows


def _load_adapter(checkpoint_name: str, cfg: dict, args, angles: list[int]) -> OpenRSDAdapter:
    adapter = OpenRSDAdapter(checkpoint_name, cfg, PROJECT_ROOT)
    adapter.load(device=args.device, image_dir="", out_dir=Path(args.output_dir) / "adapter_work" / checkpoint_name, angles=angles)
    return adapter


def _run_checkpoint(checkpoint_name: str, cfg: dict, tasks: list[dict], args, angles: list[int]) -> dict:
    import torch
    from tools.rotation_diagnostics import probe_rotated_stage_outputs as base
    from tools.rotation_overnight_gpu89 import common as C

    out_dir = Path(args.output_dir)
    rows_path = out_dir / "metrics" / f"{args.run_family}_rows.csv"
    norm_path = out_dir / "metrics" / f"{args.run_family}_embedding_norms.csv"
    asset_path = out_dir / "indexes" / f"{args.run_family}_transient_asset_index.tsv"
    done = _done_keys(rows_path, _parse_path_list(args.extra_done_rows)) if args.resume else set()
    adapter = _load_adapter(checkpoint_name, cfg, args, angles)
    runtime = adapter.runtime
    assert runtime is not None
    postprocess_cfg = _safe_postprocess_cfg(runtime.model.bbox_head, args)

    base_support, _, _ = C.build_support(
        runtime.model,
        runtime.base_args,
        runtime.support_data,
        runtime.name2id,
        runtime.device,
        "original",
    )
    base_checksum = tensor_checksum(base_support)
    support_cache = {}
    norm_rows_once: list[dict] = []
    for intervention in args.interventions:
        support_feats, support_labels, norms = C.build_support(
            runtime.model,
            runtime.base_args,
            runtime.support_data,
            runtime.name2id,
            runtime.device,
            intervention,
        )
        checksum = tensor_checksum(support_feats)
        support_cache[intervention] = (support_feats, support_labels, checksum)
        for row in norms:
            norm_rows_once.append(
                {
                    "run_family": args.run_family,
                    "split_name": args.split_name,
                    "checkpoint_name": checkpoint_name,
                    **row,
                }
            )
    _append_csv(norm_path, norm_rows_once, NORM_FIELDS)

    completed_rows = 0
    failed_rows = 0
    for chunk_index, chunk in enumerate(_chunks(tasks, args.chunk_size)):
        if args.resume and _chunk_done(chunk, checkpoint_name, args, done):
            print(
                f"SKIP_DONE {args.run_family} checkpoint={checkpoint_name} chunk={chunk_index + 1} "
                f"done_keys={len(done)}",
                flush=True,
            )
            continue
        scratch = Path(args.scratch_dir) / args.run_family / checkpoint_name / f"chunk_{chunk_index:06d}"
        if scratch.exists():
            shutil.rmtree(scratch)
        try:
            prepared, assets, direct_rows = _prepare_chunk(chunk, scratch, args, checkpoint_name)
            fixed_direct_rows = []
            for row in direct_rows:
                row.update({"config": cfg["config"], "checkpoint": cfg["checkpoint"], "support_pkl": cfg["support_pkl"]})
                fixed_direct_rows.append(row)
            _append_csv(rows_path, fixed_direct_rows, ROW_FIELDS)
            _append_csv(asset_path, assets, ASSET_FIELDS)
            dataset = prepared["dataset"]
            item_by_stem = prepared["item_by_stem"]
            print(
                f"START {args.run_family} checkpoint={checkpoint_name} chunk={chunk_index + 1} "
                f"items={len(dataset['items'])} done_keys={len(done)} "
                f"post_score_thr={postprocess_cfg.get('score_thr')} "
                f"nms_pre={postprocess_cfg.get('nms_pre')} "
                f"max_per_img={postprocess_cfg.get('max_per_img')}",
                flush=True,
            )
            if not dataset["items"]:
                continue
            loader = adapter.build_loader(dataset["images_dir"], angles)
            with torch.no_grad():
                for data_info in loader:
                    img_path = data_info["data_samples"][0].img_path
                    angle = base.angle_from_name(img_path)
                    if angle not in angles:
                        continue
                    image_stem = Path(img_path).stem
                    item = item_by_stem.get(image_stem)
                    if item is None:
                        continue
                    tile = item["tile"]
                    data = runtime.model.data_preprocessor(data_info, False)
                    data["inputs"] = data["inputs"].to(runtime.device)
                    samples = data["data_samples"]
                    features = runtime.model.prompt_extract_feats(data["inputs"])
                    metas = [sample.metainfo for sample in samples]
                    out_rows: list[dict] = []
                    for intervention in args.interventions:
                        key = (checkpoint_name, tile["tile_id"], int(angle), item["condition"], intervention)
                        if key in done:
                            continue
                        print(
                            f"POSTPROCESS {args.run_family} checkpoint={checkpoint_name} "
                            f"chunk={chunk_index + 1} image={image_stem} angle={int(angle):03d} "
                            f"condition={item['condition']} intervention={intervention}",
                            flush=True,
                        )
                        support_feats, support_labels, checksum = support_cache[intervention]
                        outs = C.model_forward_dense(runtime.model, features, support_feats, support_labels, runtime.base_args)
                        post_start = time.time()
                        summary = C.post_summary(
                            angle,
                            intervention,
                            runtime.model.bbox_head,
                            outs,
                            metas,
                            cfg=postprocess_cfg,
                        )
                        post_elapsed = time.time() - post_start
                        if post_elapsed > 30:
                            print(
                                f"SLOW_POSTPROCESS {args.run_family} checkpoint={checkpoint_name} "
                                f"chunk={chunk_index + 1} image={image_stem} angle={int(angle):03d} "
                                f"intervention={intervention} seconds={post_elapsed:.1f}",
                                flush=True,
                            )
                        if args.run_family == "open_vocab_causal":
                            claim = CLAIM_OPEN_VOCAB_EMBEDDING
                            reason = "full S3 12-angle OpenRSD visual-support intervention with paired baseline/intervention forward passes"
                            smoke_reason = "limited OpenRSD visual-support intervention validation"
                        elif args.run_family == "dehub_safety":
                            claim = CLAIM_DEHUB_REPAIR
                            reason = "full S3 12-angle OpenRSD baseline/repair safety inference row"
                            smoke_reason = "limited DeHub safety validation"
                        else:
                            claim = CLAIM_NONE
                            reason = f"full {args.split_name} 12-angle OpenRSD streaming inference row"
                            smoke_reason = "limited OpenRSD streaming inference validation"
                        status = _make_status(args, reason, smoke_reason, claim)
                        row = {
                            **_base_row(args, tile, int(angle), checkpoint_name, cfg, condition=item["condition"], intervention=intervention),
                            **summary,
                            "num_gt_total": item["num_gt_total"],
                            "num_gt_sv": item["num_gt_sv"],
                            "risk_group": item["risk_group"],
                            "image_stem": image_stem,
                            "counterfactual_is_real_image_level": args.run_family == "context_counterfactual",
                            "completed_conditions": item.get("completed_conditions", ""),
                            "completed_condition_count": item.get("completed_condition_count", 0),
                            "target_gt_class": item.get("target_gt_class", "small-vehicle"),
                            "num_target_polygons": item.get("num_target_polygons", ""),
                            "has_open_vocab_config": True,
                            "has_checkpoint": True,
                            "has_prompt_or_class_embedding_path": True,
                            "is_prompt_only": False,
                            "is_embedding_level": True,
                            "is_visual_support_level": True,
                            "actual_embedding_modified": checksum != base_checksum,
                            "original_embedding_checksum": base_checksum,
                            "modified_embedding_checksum": checksum,
                            "modified_tensor_name": "visual_support_embeddings",
                            "reran_inference": True,
                            "has_paired_comparison": intervention != "original" or args.run_family in {"context_counterfactual", "dehub_safety"},
                            "actual_inference_run": True,
                            "same_split": True,
                            "same_angles": True,
                            "same_threshold": True,
                            "same_evaluator": True,
                            "has_false_hub_metrics": args.run_family == "dehub_safety",
                            "has_true_sv_preservation": args.run_family == "dehub_safety",
                            "has_det_per_img": True,
                            "has_class_distribution_js_kl": args.run_family == "dehub_safety",
                            "has_lowrisk_inflation": args.run_family == "dehub_safety",
                            "has_ap50_or_proxy": args.run_family == "dehub_safety",
                            **status,
                        }
                        out_rows.append(row)
                        done.add(key)
                    _append_csv(rows_path, out_rows, ROW_FIELDS)
                    completed_rows += len(out_rows)
            print(
                f"DONE {args.run_family} checkpoint={checkpoint_name} chunk={chunk_index + 1} "
                f"rows_added={completed_rows} failures={failed_rows}",
                flush=True,
            )
        except Exception as exc:
            failed_rows += len(chunk)
            meta = status_metadata(ExperimentStatus.FAILED, f"{type(exc).__name__}:{exc}", claim_level=CLAIM_NONE)
            failure_rows = []
            for task in chunk:
                tile = task["tile"]
                failure_rows.append(
                    {
                        **_base_row(args, tile, int(task["angle"]), checkpoint_name, cfg, condition="failed", intervention=""),
                        **meta,
                        "unsupported_reason": meta["status_reason"],
                        "traceback": traceback.format_exc(),
                    }
                )
            _append_csv(rows_path, failure_rows, ROW_FIELDS)
            print(f"FAILED {args.run_family} checkpoint={checkpoint_name} chunk={chunk_index + 1}: {exc}", flush=True)
        finally:
            if scratch.exists() and not args.keep_scratch:
                shutil.rmtree(scratch)

    del adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"checkpoint_name": checkpoint_name, "rows_added": completed_rows, "failed_chunks_or_tasks": failed_rows}


def _write_dehub_summary(args) -> None:
    out_dir = Path(args.output_dir)
    rows = [row for row in _read_csv(out_dir / "metrics" / f"{args.run_family}_rows.csv") if row.get("condition") == "original"]
    if not rows:
        return
    base = [row for row in rows if row.get("checkpoint_name") == "baseline"]
    repair = [row for row in rows if row.get("checkpoint_name") == "repair"]
    if not base or not repair:
        return
    base_dist = class_distribution_from_histograms(base)
    repair_dist = class_distribution_from_histograms(repair)
    class_js, class_kl = js_kl_divergence(base_dist, repair_dist)
    base_high = [row for row in base if int(float(row.get("num_gt_sv") or 0)) > 0]
    repair_high = [row for row in repair if int(float(row.get("num_gt_sv") or 0)) > 0]
    base_low = [row for row in base if int(float(row.get("num_gt_sv") or 0)) == 0 or row.get("risk_group") == "low_risk_normal"]
    repair_low = [row for row in repair if int(float(row.get("num_gt_sv") or 0)) == 0 or row.get("risk_group") == "low_risk_normal"]
    base_high_sv = mean_float(base_high, "small_vehicle_ratio")
    repair_high_sv = mean_float(repair_high, "small_vehicle_ratio")
    base_low_sv = mean_float(base_low, "small_vehicle_ratio")
    repair_low_sv = mean_float(repair_low, "small_vehicle_ratio")
    lowrisk_inflation = repair_low_sv - base_low_sv if base_low and repair_low else ""
    true_sv_preservation = repair_high_sv / base_high_sv if base_high and repair_high and base_high_sv else ""
    status = _make_status(
        args,
        "full S3 12-angle DeHub baseline/repair safety summary with paired evaluator",
        "limited DeHub safety summary validation",
        CLAIM_DEHUB_REPAIR,
    )
    summary = [
        {
            "method": "OpenRSD DeHub safety streaming full" if not args.limit else "OpenRSD DeHub safety streaming limited",
            "baseline_checkpoint": args.baseline_checkpoint,
            "repair_checkpoint": args.repair_checkpoint,
            "baseline_rows": len(base),
            "repair_rows": len(repair),
            "same_split": True,
            "same_angles": True,
            "same_threshold": True,
            "same_evaluator": True,
            "actual_inference_run": True,
            "has_false_hub_metrics": True,
            "has_true_sv_preservation": True,
            "has_det_per_img": True,
            "has_class_distribution_js_kl": True,
            "has_lowrisk_inflation": bool(base_low and repair_low),
            "has_ap50_or_proxy": True,
            "SVRatioProxy_Baseline_HighRisk": base_high_sv,
            "SVRatioProxy_Repair_HighRisk": repair_high_sv,
            "SVRatioProxy_Baseline_LowRisk": base_low_sv,
            "SVRatioProxy_Repair_LowRisk": repair_low_sv,
            "TrueSVPreservationProxy": true_sv_preservation,
            "LowRiskInflation": lowrisk_inflation,
            "Class_JS": class_js,
            "Class_KL": class_kl,
            **status,
            "unsupported_reason": "",
        }
    ]
    fields = list(summary[0].keys())
    for name in ["dehub_safety.csv", "class_drift.csv", "lowrisk_inflation.csv", "true_sv_preservation.csv"]:
        path = out_dir / "metrics" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary)


def _write_manifest(args, split: dict, angles: list[int], tasks: list[dict], checkpoint_results: list[dict], status: str, failures: list[dict]) -> None:
    out_dir = Path(args.output_dir)
    manifest = {
        "command": "21_run_full_openrsd_streaming.py",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "split_name": args.split_name,
        "split_path": str(args.split),
        "num_split_tiles": split.get("num_tiles", len(split.get("tiles", []))),
        "num_tasks": len(tasks),
        "task_shard_index": int(args.task_shard_index),
        "task_shard_count": int(args.task_shard_count),
        "extra_done_rows": str(args.extra_done_rows),
        "angles": angles,
        "status": status,
        "failures": failures,
        "checkpoint_results": checkpoint_results,
        "outputs": {
            "rows_csv": str(out_dir / "metrics" / f"{args.run_family}_rows.csv"),
            "embedding_norms_csv": str(out_dir / "metrics" / f"{args.run_family}_embedding_norms.csv"),
            "transient_asset_index": str(out_dir / "indexes" / f"{args.run_family}_transient_asset_index.tsv"),
        },
        "storage_policy": {
            "rotated_images_retained": False,
            "counterfactual_images_retained": bool(args.keep_scratch),
            "scratch_dir": str(args.scratch_dir),
        },
    }
    write_json(out_dir / "manifest.json", manifest)


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--run-family",
        choices=["open_vocab_benchmark", "open_vocab_causal", "context_counterfactual", "dehub_safety"],
        required=True,
    )
    parser.add_argument("--inventory", default=str(exp_path("outputs", "open_vocab_assets", "open_vocab_asset_inventory.json")))
    parser.add_argument("--config", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--support-pkl", default="")
    parser.add_argument("--baseline-config", default=str(BASE_CONFIG))
    parser.add_argument("--baseline-checkpoint", default=str(BASE_CHECKPOINT))
    parser.add_argument("--repair-config", default=str(REPAIR_CONFIG))
    parser.add_argument("--repair-checkpoint", default=str(_default_repair_checkpoint()))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=24)
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--postprocess-score-thr", type=float, default=None)
    parser.add_argument("--nms-pre", type=int, default=-1)
    parser.add_argument("--max-per-img", type=int, default=-1)
    parser.add_argument("--support-shot", type=int, default=8)
    parser.add_argument("--interventions", default="")
    parser.add_argument("--context-conditions", default=",".join(DEFAULT_CONTEXT_CONDITIONS))
    parser.add_argument("--tile-ids", default="", help="Optional comma/space separated tile_id filter for validation or targeted reruns.")
    parser.add_argument("--task-shard-index", type=int, default=0)
    parser.add_argument("--task-shard-count", type=int, default=1)
    parser.add_argument("--extra-done-rows", default="", help="Optional comma/space separated row CSVs to treat as already completed.")
    parser.add_argument("--scratch-dir", default="/data/zcy/OpenRSD_runs/rotation_semantic_attractor/scratch_openrsd_streaming")
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    split = read_json(Path(args.split))
    args.split_name = split.get("split_name", Path(args.split).stem)
    angles = parse_angles(args.angles, default=FULL_12_ANGLES)
    args.interventions = [
        item.strip()
        for item in (args.interventions or ",".join(DEFAULT_INTERVENTIONS[args.run_family])).replace(" ", ",").split(",")
        if item.strip()
    ]
    args.context_conditions = [item.strip() for item in args.context_conditions.split(",") if item.strip()]
    if not args.output_dir:
        args.output_dir = str(Path("/data/zcy/OpenRSD_runs/rotation_semantic_attractor") / f"full_{args.run_family}_{args.split_name}_12angle")

    tiles = split.get("tiles", [])
    if args.tile_ids.strip():
        wanted = {item.strip() for item in args.tile_ids.replace(",", " ").split() if item.strip()}
        tiles = [tile for tile in tiles if tile.get("tile_id") in wanted]
    tasks = [{"tile": tile, "angle": angle} for tile in tiles for angle in angles]
    if args.limit:
        tasks = tasks[: args.limit]
    tasks = _apply_task_shard(tasks, args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "metrics").mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "indexes").mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        _write_manifest(args, split, angles, tasks, [], ExperimentStatus.NOT_RUN, [])
        print(f"dry_run_manifest={Path(args.output_dir) / 'manifest.json'}")
        return

    failures: list[dict] = []
    checkpoint_results: list[dict] = []
    try:
        if args.run_family == "dehub_safety":
            base_cfg = _support_from_pair_or_args(args)
            base_cfg.update({"config": str(args.baseline_config), "checkpoint": str(args.baseline_checkpoint)})
            repair_cfg = dict(base_cfg)
            repair_cfg.update({"config": str(args.repair_config), "checkpoint": str(args.repair_checkpoint)})
            checkpoint_results.append(_run_checkpoint("baseline", base_cfg, tasks, args, angles))
            checkpoint_results.append(_run_checkpoint("repair", repair_cfg, tasks, args, angles))
            _write_dehub_summary(args)
        else:
            cfg = _support_from_pair_or_args(args)
            checkpoint_results.append(_run_checkpoint("openrsd_a10", cfg, tasks, args, angles))
        status = ExperimentStatus.DONE_SMOKE if args.limit else ExperimentStatus.DONE_FULL
    except Exception as exc:
        status = ExperimentStatus.FAILED
        failures.append({"reason": f"{type(exc).__name__}:{exc}", "traceback": traceback.format_exc()})
        raise
    finally:
        _write_manifest(args, split, angles, tasks, checkpoint_results, status, failures)
    print(f"status={status}")
    print(f"manifest={Path(args.output_dir) / 'manifest.json'}")


if __name__ == "__main__":
    main()
