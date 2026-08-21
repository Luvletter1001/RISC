#!/usr/bin/env python
"""Evaluate dumped predictions and write structured metric JSON."""

import argparse
import json
import os
from pathlib import Path

import mmengine
from mmengine import Config
from mmengine.evaluator import Evaluator
from mmengine.registry import init_default_scope

from mmdet.registry import DATASETS
from tools.analysis_tools.eval_metric import merge_predictions_with_ground_truth


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    return value


def write_markdown(path, payload):
    metric = payload["metrics"]
    map_value = metric.get("mAP", metric.get("dota/mAP"))
    ap50_value = metric.get("AP50", metric.get("dota/AP50"))
    detail = metric.get("IoU_50_Detail", metric.get("dota/IoU_50_Detail", {}))
    rows = []
    for cls, info in detail.items():
        rows.append({
            "class": cls,
            "ap": info.get("ap"),
            "recall": info.get("recall"),
            "num_dets": info.get("num_dets"),
            "num_gts": info.get("num_gts"),
        })
    rows.sort(key=lambda row: row["class"])
    lines = [
        "# Prediction Metric JSON Eval",
        "",
        f"- predictions: `{payload['predictions']}`",
        f"- mAP: `{map_value}`",
        f"- AP50: `{ap50_value}`",
        f"- small-vehicle_AP50: `{detail.get('small-vehicle', {}).get('ap')}`",
        "",
        "| class | AP50 | recall | num_dets | num_gts |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['class']} | {row['ap']} | {row['recall']} | "
            f"{row['num_dets']} | {row['num_gts']} |")
    Path(path).write_text("\n".join(lines) + os.linesep, encoding="utf-8")


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get("default_scope", "mmdet"))
    cfg.test_dataloader.batch_size = args.batch_size
    cfg.test_dataloader.num_workers = args.num_workers
    cfg.test_dataloader.persistent_workers = False
    if "val_dataloader" in cfg:
        cfg.val_dataloader.batch_size = args.batch_size
        cfg.val_dataloader.num_workers = args.num_workers
        cfg.val_dataloader.persistent_workers = False

    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    predictions = mmengine.load(args.predictions)
    samples = merge_predictions_with_ground_truth(cfg, predictions)
    evaluator = Evaluator(cfg.val_evaluator)
    evaluator.dataset_meta = dataset.metainfo
    metrics = evaluator.offline_evaluate(samples)
    payload = {
        "config": args.config,
        "predictions": args.predictions,
        "num_predictions": len(predictions),
        "metrics": to_jsonable(metrics),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + os.linesep,
        encoding="utf-8")
    if args.out_md:
        write_markdown(args.out_md, payload)
    print(json.dumps({
        "out_json": args.out_json,
        "mAP": payload["metrics"].get("mAP", payload["metrics"].get("dota/mAP")),
        "AP50": payload["metrics"].get("AP50", payload["metrics"].get("dota/AP50")),
        "small_vehicle_AP50": payload["metrics"].get(
            "IoU_50_Detail", payload["metrics"].get("dota/IoU_50_Detail", {}))
        .get("small-vehicle", {}).get("ap"),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
