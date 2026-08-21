#!/usr/bin/env python3
"""Run GT-based DOTA1 AP50 evaluation for the OpenRSD open-vocab A10 model.

This script reruns inference because the earlier full open-vocab diagnostic
tables retained only row-level counts, not raw boxes needed for AP.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmrotate.evaluation.functional.mean_ap import eval_rbbox_map
from mmrotate.structures.bbox import qbox2rbox

from experiments.rotation_semantic_attractor.src.dota_io import read_dota_txt
from tools.rotation_diagnostics import probe_rotated_stage_outputs as base
from tools.rotation_overnight_gpu89 import common as C


DOTA1_CLASSES = list(C.CLASSES)
DEFAULT_SPLIT = REPO_ROOT / "experiments/rotation_semantic_attractor/outputs/splits/S2_final_test.json"
DEFAULT_OUT = Path("/data/zcy/OpenRSD_runs/rotation_semantic_attractor/openvocab_dota1_ap_eval_20260603")
STREAMING_SCRIPT = REPO_ROOT / "experiments/rotation_semantic_attractor/scripts/21_run_full_openrsd_streaming.py"
DEFAULT_INVENTORY = REPO_ROOT / "experiments/rotation_semantic_attractor/outputs/open_vocab_assets/open_vocab_asset_inventory.json"
DEFAULT_CONFIG = REPO_ROOT / "M_configs/Step2_A10_Large_Pretrain_Stage3/A10_flex_rtm_v3_1_formal.py"
DEFAULT_CHECKPOINT = REPO_ROOT / "results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth"
DEFAULT_SUPPORT = REPO_ROOT / "data/DOTA1_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl"


def load_streaming_module():
    spec = importlib.util.spec_from_file_location("openrsd_streaming_runtime", STREAMING_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {STREAMING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    print(f"warning: skip malformed jsonl line {path}:{line_no}", file=sys.stderr, flush=True)


def image_id_for_task(task: dict[str, Any]) -> str:
    return f"{task['tile']['tile_id']}_rot{int(task['angle']):03d}"


def completed_image_ids_from_raw(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for row in iter_jsonl(path):
        image_id = row.get("image_id")
        if image_id:
            ids.add(str(image_id))
    return ids


def parse_angles(value: str) -> list[int]:
    if not value.strip():
        return list(range(0, 360, 30))
    return [int(x) for x in value.replace(",", " ").split()]


def as_rbox_numpy(bboxes: Any) -> np.ndarray:
    if hasattr(bboxes, "tensor"):
        arr = bboxes.tensor.detach().cpu().numpy()
    elif isinstance(bboxes, torch.Tensor):
        arr = bboxes.detach().cpu().numpy()
    else:
        arr = np.asarray(bboxes)
    arr = arr.astype(np.float32)
    if arr.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    return arr.reshape(-1, arr.shape[-1])[:, :5]


def ann_from_dota_txt(path: str | Path) -> dict[str, np.ndarray]:
    labels: list[int] = []
    boxes: list[np.ndarray] = []
    labels_ignore: list[int] = []
    boxes_ignore: list[np.ndarray] = []
    for record in read_dota_txt(path):
        if "polygon" not in record:
            continue
        cls = record.get("class_name", "")
        if cls not in DOTA1_CLASSES:
            continue
        qbox = torch.tensor(
            [[coord for point in record["polygon"] for coord in point]],
            dtype=torch.float32,
        )
        rbox = qbox2rbox(qbox).detach().cpu().numpy().astype(np.float32)[0]
        difficulty = str(record.get("difficulty", "0"))
        if difficulty == "0":
            labels.append(DOTA1_CLASSES.index(cls))
            boxes.append(rbox)
        else:
            labels_ignore.append(DOTA1_CLASSES.index(cls))
            boxes_ignore.append(rbox)
    return {
        "labels": np.asarray(labels, dtype=np.int64),
        "bboxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 5),
        "labels_ignore": np.asarray(labels_ignore, dtype=np.int64),
        "bboxes_ignore": np.asarray(boxes_ignore, dtype=np.float32).reshape(-1, 5),
    }


def pred_to_per_class_and_raw(pred: Any, image_id: str, tile_id: str, angle: int) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    boxes = as_rbox_numpy(pred.bboxes)
    labels = pred.labels.detach().cpu().numpy().astype(np.int64) if len(pred.labels) else np.zeros(0, dtype=np.int64)
    scores = pred.scores.detach().cpu().numpy().astype(np.float32) if len(pred.scores) else np.zeros(0, dtype=np.float32)
    per_class: list[np.ndarray] = []
    for cid in range(len(DOTA1_CLASSES)):
        idx = np.where(labels == cid)[0]
        if len(idx):
            per_class.append(np.hstack([boxes[idx], scores[idx].reshape(-1, 1)]).astype(np.float32))
        else:
            per_class.append(np.zeros((0, 6), dtype=np.float32))
    raw = [
        {
            "image_id": image_id,
            "tile_id": tile_id,
            "angle": int(angle),
            "class_id": int(label),
            "class_name": DOTA1_CLASSES[int(label)] if 0 <= int(label) < len(DOTA1_CLASSES) else str(label),
            "score": float(score),
            "rbox": [float(x) for x in box],
        }
        for box, label, score in zip(boxes, labels, scores)
    ]
    return per_class, raw


def raw_rows_to_per_class(rows: list[dict[str, Any]]) -> list[np.ndarray]:
    by_class: list[list[list[float]]] = [[] for _ in DOTA1_CLASSES]
    for row in rows:
        cid = int(row["class_id"])
        if not 0 <= cid < len(DOTA1_CLASSES):
            continue
        box = [float(x) for x in row["rbox"][:5]]
        by_class[cid].append(box + [float(row["score"])])
    return [
        np.asarray(items, dtype=np.float32).reshape(-1, 6) if items else np.zeros((0, 6), dtype=np.float32)
        for items in by_class
    ]


def eval_split(det_results: list[list[np.ndarray]], annotations: list[dict[str, np.ndarray]], nproc: int) -> tuple[float, list[dict[str, Any]]]:
    mean_ap, cls_results = eval_rbbox_map(
        det_results,
        annotations,
        iou_thr=0.5,
        use_07_metric=True,
        box_type="rbox",
        dataset=tuple(DOTA1_CLASSES),
        logger="silent",
        nproc=nproc,
    )
    rows = []
    for name, cr in zip(DOTA1_CLASSES, cls_results):
        rows.append(
            {
                "class_name": name,
                "ap50": float(np.asarray(cr["ap"]).reshape(-1)[0]),
                "num_gts": int(cr["num_gts"]),
                "num_dets": int(cr["num_dets"]),
            }
        )
    return float(mean_ap), rows


def write_eval_outputs(
    out_dir: Path,
    summary: dict[str, Any],
    overall_map: float,
    class_rows: list[dict[str, Any]],
    angle_rows: list[dict[str, Any]],
    angle_class_rows: list[dict[str, Any]],
) -> None:
    write_csv(out_dir / "metrics/openvocab_dota1_ap_by_class.csv", class_rows, list(class_rows[0].keys()))
    write_csv(out_dir / "metrics/openvocab_dota1_ap_by_angle.csv", angle_rows, list(angle_rows[0].keys()))
    write_csv(out_dir / "metrics/openvocab_dota1_ap_by_angle_class.csv", angle_class_rows, list(angle_class_rows[0].keys()))
    write_json(out_dir / "metrics/openvocab_dota1_ap_summary.json", summary)
    lines = [
        "# OpenVocab DOTA1 AP50 Evaluation",
        "",
        f"- status: `{summary['status']}`",
        f"- AP50 mAP: `{overall_map:.6f}`",
        f"- images: `{summary['num_images']}`",
        f"- angles: `{summary['angles']}`",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- raw predictions: `{summary['raw_predictions_jsonl']}`",
        "",
        "## Per-Angle AP50",
        "",
        "| angle | AP50 mAP | images |",
        "| --- | ---: | ---: |",
    ]
    for row in angle_rows:
        lines.append(f"| {row['angle']} | {float(row['ap50_map']):.6f} | {row['images']} |")
    lines.extend(["", "## Per-Class AP50", "", "| class | AP50 | GT | det |", "| --- | ---: | ---: | ---: |"])
    for row in class_rows:
        lines.append(f"| {row['class_name']} | {float(row['ap50']):.6f} | {row['num_gts']} | {row['num_dets']} |")
    (out_dir / "reports/openvocab_dota1_ap_eval_report.md").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "reports/openvocab_dota1_ap_eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_shards(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    if out_dir.exists() and args.force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = Path(args.scratch_dir) if args.scratch_dir else out_dir / "scratch_merge"
    raw_jsonl = out_dir / "predictions/openvocab_dota1_ap_raw_predictions.jsonl"
    raw_jsonl.unlink(missing_ok=True)

    input_dirs = [Path(x) for x in args.merge_input_dirs.replace(",", " ").split() if x.strip()]
    if not input_dirs:
        raise RuntimeError("--merge-input-dirs is required for --mode merge")
    raw_by_image: dict[str, list[dict[str, Any]]] = {}
    total_raw = 0
    for in_dir in input_dirs:
        path = in_dir / "predictions/openvocab_dota1_ap_raw_predictions.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        rows = []
        for row in iter_jsonl(path):
            raw_by_image.setdefault(row["image_id"], []).append(row)
            rows.append(row)
            total_raw += 1
        append_jsonl(raw_jsonl, rows)

    streaming = load_streaming_module()
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    args.run_family = "open_vocab_ap_eval"
    args.split_name = split.get("split_name", Path(args.split).stem)
    angles = parse_angles(args.angles)
    tiles = split.get("tiles", [])
    if args.limit:
        tiles = tiles[: args.limit]
    tasks = [{"tile": tile, "angle": angle} for tile in tiles for angle in angles]
    if not tasks:
        raise RuntimeError("no tasks selected")

    det_results: list[list[np.ndarray]] = []
    annotations: list[dict[str, np.ndarray]] = []
    meta_rows: list[dict[str, Any]] = []
    per_angle_dets: dict[int, list[list[np.ndarray]]] = {angle: [] for angle in angles}
    per_angle_anns: dict[int, list[dict[str, np.ndarray]]] = {angle: [] for angle in angles}
    started = time.time()
    for chunk_index in range(0, len(tasks), args.chunk_size):
        chunk = tasks[chunk_index : chunk_index + args.chunk_size]
        scratch = scratch_dir / f"chunk_{chunk_index // args.chunk_size:06d}"
        if scratch.exists():
            shutil.rmtree(scratch)
        prepared, _assets, direct_rows = streaming._prepare_chunk(chunk, scratch, args, "openrsd_a10")
        if direct_rows:
            raise RuntimeError(f"unexpected direct rows in AP eval merge: {len(direct_rows)}")
        for image_stem, item in prepared["item_by_stem"].items():
            angle = int(base.angle_from_name(image_stem))
            raw_rows = raw_by_image.get(image_stem, [])
            per_class = raw_rows_to_per_class(raw_rows)
            ann = ann_from_dota_txt(item["ann_txt"])
            det_results.append(per_class)
            annotations.append(ann)
            per_angle_dets[angle].append(per_class)
            per_angle_anns[angle].append(ann)
            meta_rows.append(
                {
                    "image_id": image_stem,
                    "tile_id": item["tile"]["tile_id"],
                    "angle": angle,
                    "num_gt": int(len(ann["labels"])),
                    "num_gt_ignore": int(len(ann["labels_ignore"])),
                    "num_pred": int(len(raw_rows)),
                    "ann_txt": item["ann_txt"],
                }
            )
        if scratch.exists() and not args.keep_scratch:
            shutil.rmtree(scratch)

    write_csv(out_dir / "metrics/openvocab_dota1_ap_eval_images.csv", meta_rows, list(meta_rows[0].keys()))
    overall_map, class_rows = eval_split(det_results, annotations, args.nproc_eval)
    angle_rows: list[dict[str, Any]] = []
    angle_class_rows: list[dict[str, Any]] = []
    for angle in angles:
        if not per_angle_dets[angle]:
            continue
        angle_map, rows = eval_split(per_angle_dets[angle], per_angle_anns[angle], args.nproc_eval)
        angle_rows.append({"angle": angle, "ap50_map": angle_map, "images": len(per_angle_dets[angle])})
        for row in rows:
            angle_class_rows.append({"angle": angle, **row})
    summary = {
        "status": "DONE",
        "mode": "merge",
        "ap50_map": overall_map,
        "num_images": len(meta_rows),
        "num_tasks": len(tasks),
        "num_raw_predictions": total_raw,
        "angles": angles,
        "split": str(args.split),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "support_pkl": str(args.support_pkl),
        "score_thr": args.score_thr,
        "raw_predictions_jsonl": str(raw_jsonl),
        "merge_input_dirs": [str(x) for x in input_dirs],
        "elapsed_sec": time.time() - started,
    }
    write_eval_outputs(out_dir, summary, overall_map, class_rows, angle_rows, angle_class_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["run", "merge"], default="run")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--support-pkl", default=str(DEFAULT_SUPPORT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--scratch-dir", default="")
    parser.add_argument("--angles", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--postprocess-score-thr", type=float, default=None)
    parser.add_argument("--nms-pre", type=int, default=-1)
    parser.add_argument("--max-per-img", type=int, default=-1)
    parser.add_argument("--support-shot", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--nproc-eval", type=int, default=4)
    parser.add_argument("--merge-input-dirs", default="")
    parser.add_argument("--task-shard-index", type=int, default=0)
    parser.add_argument("--task-shard-count", type=int, default=1)
    parser.add_argument("--skip-eval", action="store_true", help="Only save raw predictions and image metadata for shard inference.")
    parser.add_argument("--resume", action="store_true", help="Keep existing raw JSONL and skip image_ids already present in it.")
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.mode == "merge":
        return merge_shards(args)

    repo = Path(args.repo_root)
    out_dir = Path(args.output_dir)
    if out_dir.exists() and args.force and not args.resume:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = Path(args.scratch_dir) if args.scratch_dir else out_dir / "scratch"
    raw_jsonl = out_dir / "predictions/openvocab_dota1_ap_raw_predictions.jsonl"
    if not args.resume:
        raw_jsonl.unlink(missing_ok=True)

    streaming = load_streaming_module()
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    args.run_family = "open_vocab_ap_eval"
    args.split_name = split.get("split_name", Path(args.split).stem)
    angles = parse_angles(args.angles)
    tiles = split.get("tiles", [])
    if args.limit:
        tiles = tiles[: args.limit]
    tasks = [{"tile": tile, "angle": angle} for tile in tiles for angle in angles]
    if args.task_shard_count > 1:
        tasks = [task for i, task in enumerate(tasks) if i % args.task_shard_count == args.task_shard_index]
    resumed_image_ids = completed_image_ids_from_raw(raw_jsonl) if args.resume else set()
    if resumed_image_ids:
        tasks = [task for task in tasks if image_id_for_task(task) not in resumed_image_ids]
    if not tasks:
        raise RuntimeError("no tasks selected")

    cfg = streaming._support_from_pair_or_args(args)
    cfg["device"] = args.device
    cfg["score_thr"] = float(args.score_thr)
    cfg["batch_size"] = int(args.batch_size)
    cfg["num_workers"] = int(args.num_workers)
    cfg["support_shot"] = int(args.support_shot)
    checkpoint_name = "openrsd_a10"

    adapter = streaming._load_adapter(checkpoint_name, cfg, args, angles)
    runtime = adapter.runtime
    assert runtime is not None
    postprocess_cfg = streaming._safe_postprocess_cfg(runtime.model.bbox_head, args)
    support_feats, support_labels, _ = C.build_support(
        runtime.model,
        runtime.base_args,
        runtime.support_data,
        runtime.name2id,
        runtime.device,
        "original",
    )

    det_results: list[list[np.ndarray]] = []
    annotations: list[dict[str, np.ndarray]] = []
    meta_rows: list[dict[str, Any]] = []
    per_angle_dets: dict[int, list[list[np.ndarray]]] = {angle: [] for angle in angles}
    per_angle_anns: dict[int, list[dict[str, np.ndarray]]] = {angle: [] for angle in angles}

    started = time.time()
    for chunk_index in range(0, len(tasks), args.chunk_size):
        chunk = tasks[chunk_index : chunk_index + args.chunk_size]
        scratch = scratch_dir / f"chunk_{chunk_index // args.chunk_size:06d}"
        if scratch.exists():
            shutil.rmtree(scratch)
        prepared, _assets, direct_rows = streaming._prepare_chunk(chunk, scratch, args, checkpoint_name)
        if direct_rows:
            raise RuntimeError(f"unexpected direct rows in AP eval: {len(direct_rows)}")
        dataset = prepared["dataset"]
        item_by_stem = prepared["item_by_stem"]
        print(
            json.dumps(
                {
                    "stage": "chunk_start",
                    "chunk": chunk_index // args.chunk_size,
                    "items": len(dataset["items"]),
                    "processed": len(meta_rows),
                    "total": len(tasks),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        loader = adapter.build_loader(dataset["images_dir"], angles)
        chunk_raw_rows: list[dict[str, Any]] = []
        with torch.no_grad():
            for data_info in loader:
                img_path = data_info["data_samples"][0].img_path
                angle = int(base.angle_from_name(img_path))
                if angle not in angles:
                    continue
                image_stem = Path(img_path).stem
                item = item_by_stem.get(image_stem)
                if item is None:
                    continue
                data = runtime.model.data_preprocessor(data_info, False)
                data["inputs"] = data["inputs"].to(runtime.device)
                samples = data["data_samples"]
                features = runtime.model.prompt_extract_feats(data["inputs"])
                metas = [sample.metainfo for sample in samples]
                outs = C.model_forward_dense(runtime.model, features, support_feats, support_labels, runtime.base_args)
                preds = runtime.model.bbox_head.predict_by_feat(
                    *outs,
                    batch_img_metas=metas,
                    cfg=postprocess_cfg,
                    rescale=True,
                    with_nms=True,
                )
                for pred, sample in zip(preds, samples):
                    img_path = sample.img_path
                    angle = int(base.angle_from_name(img_path))
                    if angle not in angles:
                        continue
                    image_stem = Path(img_path).stem
                    item = item_by_stem.get(image_stem)
                    if item is None:
                        continue
                    per_class, raw_rows = pred_to_per_class_and_raw(pred, image_stem, item["tile"]["tile_id"], angle)
                    ann = None if args.skip_eval else ann_from_dota_txt(item["ann_txt"])
                    if not args.skip_eval:
                        assert ann is not None
                        det_results.append(per_class)
                        annotations.append(ann)
                        per_angle_dets[angle].append(per_class)
                        per_angle_anns[angle].append(ann)
                    chunk_raw_rows.extend(raw_rows)
                    meta_rows.append(
                        {
                            "image_id": image_stem,
                            "tile_id": item["tile"]["tile_id"],
                            "angle": angle,
                            "num_gt": "" if ann is None else int(len(ann["labels"])),
                            "num_gt_ignore": "" if ann is None else int(len(ann["labels_ignore"])),
                            "num_pred": int(len(raw_rows)),
                            "image_path": img_path,
                            "ann_txt": item["ann_txt"],
                        }
                    )
        append_jsonl(raw_jsonl, chunk_raw_rows)
        if scratch.exists() and not args.keep_scratch:
            shutil.rmtree(scratch)

    write_csv(out_dir / "metrics/openvocab_dota1_ap_eval_images.csv", meta_rows, list(meta_rows[0].keys()))
    if args.skip_eval:
        summary = {
            "status": "DONE_INFERENCE_ONLY",
            "num_images": len(meta_rows),
            "num_tasks": len(tasks),
            "angles": angles,
            "split": str(args.split),
            "checkpoint": cfg["checkpoint"],
            "config": cfg["config"],
            "support_pkl": cfg["support_pkl"],
            "score_thr": args.score_thr,
            "postprocess_cfg": dict(postprocess_cfg),
            "raw_predictions_jsonl": str(raw_jsonl),
            "elapsed_sec": time.time() - started,
            "batch_size": int(args.batch_size),
            "chunk_size": int(args.chunk_size),
            "resumed_existing_images": len(resumed_image_ids),
            "remaining_tasks_at_start": len(tasks),
        }
        write_json(out_dir / "metrics/openvocab_dota1_ap_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0

    overall_map, class_rows = eval_split(det_results, annotations, args.nproc_eval)
    angle_rows: list[dict[str, Any]] = []
    angle_class_rows: list[dict[str, Any]] = []
    for angle in angles:
        if not per_angle_dets[angle]:
            continue
        angle_map, rows = eval_split(per_angle_dets[angle], per_angle_anns[angle], args.nproc_eval)
        angle_rows.append({"angle": angle, "ap50_map": angle_map, "images": len(per_angle_dets[angle])})
        for row in rows:
            angle_class_rows.append({"angle": angle, **row})
    summary = {
        "status": "DONE",
        "ap50_map": overall_map,
        "num_images": len(meta_rows),
        "num_tasks": len(tasks),
        "angles": angles,
        "split": str(args.split),
        "checkpoint": cfg["checkpoint"],
        "config": cfg["config"],
        "support_pkl": cfg["support_pkl"],
        "score_thr": args.score_thr,
        "postprocess_cfg": dict(postprocess_cfg),
        "raw_predictions_jsonl": str(raw_jsonl),
        "elapsed_sec": time.time() - started,
    }
    write_eval_outputs(out_dir, summary, overall_map, class_rows, angle_rows, angle_class_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
