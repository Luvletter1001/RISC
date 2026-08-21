#!/usr/bin/env python
"""Evaluate dumped rotated detections against DOTA annfiles without images."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from openrsd_env import preload_installed_mmengine  # noqa: E402

preload_installed_mmengine()

import mmengine  # noqa: E402
from mmengine import Config  # noqa: E402
from mmengine.evaluator import Evaluator  # noqa: E402
from mmengine.registry import init_default_scope  # noqa: E402
from mmrotate.structures.bbox import qbox2rbox  # noqa: E402


DOTA1_CLASSES = (
    "plane", "baseball-diamond", "bridge", "ground-track-field",
    "small-vehicle", "large-vehicle", "ship", "tennis-court",
    "basketball-court", "storage-tank", "soccer-ball-field", "roundabout",
    "harbor", "swimming-pool", "helicopter",
)

DOTAV2_CLASSES = (
    "airport", "baseball-diamond", "basketball-court", "bridge",
    "container-crane", "ground-track-field", "harbor", "helicopter",
    "helipad", "large-vehicle", "plane", "roundabout", "ship",
    "small-vehicle", "soccer-ball-field", "storage-tank", "swimming-pool",
    "tennis-court",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline AP eval for predictions.pkl + DOTA annfiles.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ann-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--diff-thr", type=int, default=100)
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


def parse_classes_from_config_text(config_path: Path) -> tuple[str, ...]:
    text = config_path.read_text(encoding="utf-8")
    for match in re.finditer(r"(?<![A-Za-z0-9_])classes\s*=\s*([\[(])", text):
        open_pos = match.start(1)
        close_char = "]" if match.group(1) == "[" else ")"
        depth = 0
        close_pos = -1
        for idx in range(open_pos, len(text)):
            char = text[idx]
            if char == match.group(1):
                depth += 1
            elif char == close_char:
                depth -= 1
                if depth == 0:
                    close_pos = idx
                    break
        if close_pos < 0:
            continue
        try:
            classes = ast.literal_eval(text[open_pos:close_pos + 1])
        except (SyntaxError, ValueError):
            continue
        if classes and all(isinstance(item, str) for item in classes):
            return tuple(classes)
    return DOTA1_CLASSES


def load_class_names(config_path: str) -> tuple[str, ...]:
    cfg = Config.fromfile(config_path)
    dataset_cfg = cfg.get("test_dataloader", {}).get("dataset", {})
    metainfo = dataset_cfg.get("metainfo") or cfg.get("metainfo") or {}
    classes = metainfo.get("classes") or cfg.get("class_name")
    if classes:
        return tuple(classes)
    if dataset_cfg.get("type") == "DOTAv2Dataset":
        return DOTAV2_CLASSES
    return parse_classes_from_config_text(Path(config_path))


def qboxes_to_rboxes(qboxes: np.ndarray) -> torch.Tensor:
    if len(qboxes) == 0:
        return torch.zeros((0, 5), dtype=torch.float32)
    qbox_tensor = torch.from_numpy(np.asarray(qboxes, dtype=np.float32))
    return qbox2rbox(qbox_tensor).to(torch.float32)


def read_ann(path: Path, class_to_label: dict[str, int], diff_thr: int) -> tuple[dict, dict]:
    gt_qboxes = []
    gt_labels = []
    ignored_qboxes = []
    ignored_labels = []
    if path.exists():
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = raw.split()
            if len(parts) < 9:
                continue
            cls_name = parts[8]
            if cls_name not in class_to_label:
                continue
            difficulty = int(parts[9]) if len(parts) > 9 and parts[9].lstrip("-").isdigit() else 0
            qbox = [float(x) for x in parts[:8]]
            if difficulty > diff_thr:
                ignored_qboxes.append(qbox)
                ignored_labels.append(class_to_label[cls_name])
            else:
                gt_qboxes.append(qbox)
                gt_labels.append(class_to_label[cls_name])
    gt_instances = {
        "labels": torch.tensor(gt_labels, dtype=torch.long),
        "bboxes": qboxes_to_rboxes(np.asarray(gt_qboxes, dtype=np.float32)),
    }
    ignored_instances = {
        "labels": torch.tensor(ignored_labels, dtype=torch.long),
        "bboxes": qboxes_to_rboxes(np.asarray(ignored_qboxes, dtype=np.float32)),
    }
    return gt_instances, ignored_instances


def optional_field(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def get_img_id(sample) -> str:
    img_id = optional_field(sample, "img_id")
    if img_id is not None:
        return str(img_id)
    metainfo = optional_field(sample, "metainfo", {}) or {}
    return str(metainfo.get("img_id"))


def canonical_img_id(img_id: str) -> str:
    if img_id.startswith("angle_") and "__" in img_id:
        return img_id.split("__", 1)[1]
    return img_id


def build_samples(predictions, ann_dir: Path, class_names: tuple[str, ...], diff_thr: int, max_images: int):
    class_to_label = {name: idx for idx, name in enumerate(class_names)}
    samples = []
    missing_ann = 0
    for index, pred in enumerate(predictions):
        if max_images > 0 and index >= max_images:
            break
        img_id = get_img_id(pred)
        ann_path = ann_dir / f"{img_id}.txt"
        if not ann_path.exists():
            ann_path = ann_dir / f"{canonical_img_id(img_id)}.txt"
        if not ann_path.exists():
            missing_ann += 1
        gt_instances, ignored_instances = read_ann(ann_path, class_to_label, diff_thr)
        sample = {
            "img_id": img_id,
            "gt_instances": gt_instances,
            "ignored_instances": ignored_instances,
            "pred_instances": optional_field(pred, "pred_instances"),
        }
        samples.append(sample)
    return samples, missing_ann


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


def write_markdown(path: str, payload: dict) -> None:
    metric = payload["metrics"]
    map_value = metric_value(metric, "mAP")
    ap50_value = metric_value(metric, "AP50")
    lines = [
        "# Annfiles-Only Prediction Metric Eval",
        "",
        f"- predictions: `{payload['predictions']}`",
        f"- ann_dir: `{payload['ann_dir']}`",
        f"- samples: `{payload['num_samples']}`",
        f"- missing_ann: `{payload['missing_ann']}`",
        f"- mAP: `{map_value}`",
        f"- AP50: `{ap50_value}`",
    ]
    Path(path).write_text("\n".join(lines) + os.linesep, encoding="utf-8")


def metric_value(metrics: dict, key: str):
    return metrics.get(key, metrics.get(f"dota/{key}"))


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get("default_scope", "mmrotate"))
    evaluator_cfg = cfg.get("val_evaluator", dict(type="DOTAMetric", metric="mAP"))
    class_names = load_class_names(args.config)
    predictions = mmengine.load(args.predictions)
    samples, missing_ann = build_samples(
        predictions, Path(args.ann_dir), class_names, args.diff_thr, args.max_images)
    evaluator = Evaluator(evaluator_cfg)
    evaluator.dataset_meta = {"classes": class_names}
    metrics = evaluator.offline_evaluate(samples)
    payload = {
        "config": args.config,
        "predictions": args.predictions,
        "ann_dir": args.ann_dir,
        "class_names": class_names,
        "diff_thr": args.diff_thr,
        "num_predictions": len(predictions),
        "num_samples": len(samples),
        "missing_ann": missing_ann,
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
        "num_samples": len(samples),
        "missing_ann": missing_ann,
        "mAP": metric_value(payload["metrics"], "mAP"),
        "AP50": metric_value(payload["metrics"], "AP50"),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
