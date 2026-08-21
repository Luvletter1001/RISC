#!/usr/bin/env python
"""Evaluate HBB predictions.pkl files with pycocotools COCOeval."""

import argparse
import contextlib
import io
import json
import os
import pickle
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def tensor_to_numpy(value):
    if hasattr(value, "tensor"):
        value = value.tensor
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def field(obj, key):
    if isinstance(obj, dict):
        return obj[key]
    return getattr(obj, key)


def optional_field(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def sample_img_id(sample):
    img_id = optional_field(sample, "img_id")
    if img_id is not None:
        return int(img_id)
    metainfo = optional_field(sample, "metainfo", {}) or {}
    return int(metainfo["img_id"])


def predictions_to_coco_results(predictions, category_ids, max_dets_per_img=0):
    results = []
    for sample in predictions:
        img_id = sample_img_id(sample)
        pred_instances = field(sample, "pred_instances")
        boxes = tensor_to_numpy(field(pred_instances, "bboxes")).reshape(-1, 4)
        labels = tensor_to_numpy(field(pred_instances, "labels")).astype(np.int64)
        scores = tensor_to_numpy(field(pred_instances, "scores")).astype(np.float32)
        if max_dets_per_img > 0 and len(scores) > max_dets_per_img:
            order = np.argsort(-scores)[:max_dets_per_img]
            boxes = boxes[order]
            labels = labels[order]
            scores = scores[order]
        for box, label, score in zip(boxes, labels, scores):
            label = int(label)
            if label < 0 or label >= len(category_ids):
                continue
            x1, y1, x2, y2 = [float(x) for x in box]
            width = max(x2 - x1, 0.0)
            height = max(y2 - y1, 0.0)
            if width <= 0.0 or height <= 0.0:
                continue
            results.append({
                "image_id": img_id,
                "category_id": int(category_ids[label]),
                "bbox": [x1, y1, width, height],
                "score": float(score),
            })
    return results


def per_category_ap(coco_eval, categories):
    precision = coco_eval.eval["precision"]
    # precision dims: IoU, recall, category, area range, max det
    rows = []
    for idx, cat in enumerate(categories):
        values = precision[:, :, idx, 0, -1]
        values = values[values > -1]
        ap = float(np.mean(values)) if values.size else float("nan")
        values50 = precision[0, :, idx, 0, -1]
        values50 = values50[values50 > -1]
        ap50 = float(np.mean(values50)) if values50.size else float("nan")
        rows.append({
            "id": int(cat["id"]),
            "name": cat["name"],
            "AP": ap,
            "AP50": ap50,
        })
    return rows


def evaluate(ann_json, predictions_pkl, max_dets_per_img=0):
    coco_gt = COCO(ann_json)
    categories = sorted(coco_gt.loadCats(coco_gt.getCatIds()), key=lambda item: item["id"])
    category_ids = [cat["id"] for cat in categories]
    with Path(predictions_pkl).open("rb") as f:
        predictions = pickle.load(f)
    detections = predictions_to_coco_results(
        predictions, category_ids, max_dets_per_img=max_dets_per_img)
    if detections:
        coco_dt = coco_gt.loadRes(detections)
    else:
        coco_dt = coco_gt.loadRes([])
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    if max_dets_per_img > 0:
        coco_eval.params.maxDets = [1, 10, max_dets_per_img]
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    stats = [float(x) for x in coco_eval.stats]
    metrics = {
        "coco/bbox_mAP": stats[0],
        "coco/bbox_mAP_50": stats[1],
        "coco/bbox_mAP_75": stats[2],
        "coco/bbox_mAP_s": stats[3],
        "coco/bbox_mAP_m": stats[4],
        "coco/bbox_mAP_l": stats[5],
        "coco/bbox_AR_1": stats[6],
        "coco/bbox_AR_10": stats[7],
        "coco/bbox_AR_100": stats[8],
        "coco/bbox_AR_s": stats[9],
        "coco/bbox_AR_m": stats[10],
        "coco/bbox_AR_l": stats[11],
    }
    return {
        "ann_json": str(ann_json),
        "predictions": str(predictions_pkl),
        "num_prediction_samples": len(predictions),
        "num_coco_detections": len(detections),
        "max_dets_per_img_input": int(max_dets_per_img),
        "metrics": metrics,
        "per_category": per_category_ap(coco_eval, categories),
        "cocoeval_stdout": stream.getvalue(),
    }


def write_markdown(path, payload):
    metrics = payload["metrics"]
    lines = [
        "# HBB COCO Prediction Eval",
        "",
        f"- predictions: `{payload['predictions']}`",
        f"- ann_json: `{payload['ann_json']}`",
        f"- detections: `{payload['num_coco_detections']}`",
        f"- mAP: `{metrics['coco/bbox_mAP']}`",
        f"- AP50: `{metrics['coco/bbox_mAP_50']}`",
        "",
        "| class | AP | AP50 |",
        "|---|---:|---:|",
    ]
    for row in payload["per_category"]:
        lines.append(f"| {row['name']} | {row['AP']} | {row['AP50']} |")
    Path(path).write_text("\n".join(lines) + os.linesep, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-json", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--max-dets-per-img", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = evaluate(args.ann_json, args.predictions, args.max_dets_per_img)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + os.linesep,
        encoding="utf-8")
    if args.out_md:
        write_markdown(args.out_md, payload)
    print(json.dumps({
        "out_json": args.out_json,
        "mAP": payload["metrics"]["coco/bbox_mAP"],
        "AP50": payload["metrics"]["coco/bbox_mAP_50"],
        "detections": payload["num_coco_detections"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
