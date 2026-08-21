#!/usr/bin/env python3
"""Create oracle-ranked copies of fixed-query predictions for feasibility checks.

This is a diagnostic tool only. It uses ground truth to rewrite detection
scores and estimate whether fixed-query boxes contain enough usable candidates.
It does not modify boxes or labels, and it must not be used as an inference
method.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import torch
from mmcv.ops import box_iou_rotated

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mmrotate.structures.bbox import qbox2rbox  # noqa: E402


HRRSD_CLASSES = (
    "T",
    "airplane",
    "baseball",
    "basketball",
    "bridge",
    "crossroad",
    "ground",
    "harbor",
    "parking",
    "ship",
    "storage",
    "tennis",
    "vehicle",
)


def _tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
    if hasattr(value, "tensor"):
        value = value.tensor
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    out = torch.as_tensor(value)
    if dtype is not None:
        out = out.to(dtype=dtype)
    return out


def _load_gt(ann_file: Path, class_to_id: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    boxes: list[torch.Tensor] = []
    labels: list[int] = []
    if not ann_file.exists():
        return torch.empty(0, 5), torch.empty(0, dtype=torch.long)
    for line in ann_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 10 or parts[8] not in class_to_id:
            continue
        qbox = torch.tensor([[float(x) for x in parts[:8]]], dtype=torch.float32)
        boxes.append(qbox2rbox(qbox)[0])
        labels.append(class_to_id[parts[8]])
    if not boxes:
        return torch.empty(0, 5), torch.empty(0, dtype=torch.long)
    return torch.stack(boxes), torch.tensor(labels, dtype=torch.long)


def _mutable_pred_instances(pred_instances: Any) -> dict[str, Any]:
    if isinstance(pred_instances, dict):
        return dict(pred_instances)
    return {
        "bboxes": getattr(pred_instances, "bboxes"),
        "scores": getattr(pred_instances, "scores"),
        "labels": getattr(pred_instances, "labels"),
    }


def _same_class_iou_scores(
        bboxes: torch.Tensor,
        labels: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor) -> torch.Tensor:
    scores = torch.zeros(labels.shape[0], dtype=torch.float32)
    if bboxes.numel() == 0 or gt_boxes.numel() == 0:
        return scores
    for class_id in labels.unique().tolist():
        class_id = int(class_id)
        pred_idx = torch.where(labels == class_id)[0]
        gt_idx = torch.where(gt_labels == class_id)[0]
        if pred_idx.numel() == 0 or gt_idx.numel() == 0:
            continue
        ious = box_iou_rotated(bboxes[pred_idx].float(), gt_boxes[gt_idx].float())
        scores[pred_idx] = ious.max(dim=1).values
    return scores


def _one_to_one_iou_scores(
        bboxes: torch.Tensor,
        labels: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor) -> torch.Tensor:
    scores = torch.zeros(labels.shape[0], dtype=torch.float32)
    if bboxes.numel() == 0 or gt_boxes.numel() == 0:
        return scores
    for class_id in labels.unique().tolist():
        class_id = int(class_id)
        pred_idx = torch.where(labels == class_id)[0]
        gt_idx = torch.where(gt_labels == class_id)[0]
        if pred_idx.numel() == 0 or gt_idx.numel() == 0:
            continue
        ious = box_iou_rotated(bboxes[pred_idx].float(), gt_boxes[gt_idx].float())
        triples = []
        for pred_local in range(ious.shape[0]):
            for gt_local in range(ious.shape[1]):
                triples.append((
                    float(ious[pred_local, gt_local]),
                    int(pred_idx[pred_local]),
                    int(gt_idx[gt_local]),
                ))
        used_preds: set[int] = set()
        used_gts: set[int] = set()
        for iou, pred_abs, gt_abs in sorted(triples, reverse=True):
            if iou <= 0:
                break
            if pred_abs in used_preds or gt_abs in used_gts:
                continue
            used_preds.add(pred_abs)
            used_gts.add(gt_abs)
            scores[pred_abs] = iou
    return scores


def oracle_rank(
        input_pkl: Path,
        ann_dir: Path,
        output_pkl: Path,
        mode: str,
        classes: tuple[str, ...] = HRRSD_CLASSES) -> dict[str, Any]:
    class_to_id = {name: idx for idx, name in enumerate(classes)}
    with input_pkl.open("rb") as f:
        predictions = pickle.load(f)

    output = []
    total_dets = 0
    positive_scores = 0
    iou50_scores = 0
    score_sum = 0.0
    for sample in predictions:
        img_id = str(sample.get("img_id"))
        gt_boxes, gt_labels = _load_gt(ann_dir / f"{img_id}.txt", class_to_id)
        pred_instances = _mutable_pred_instances(sample.get("pred_instances", {}))
        bboxes = _tensor(pred_instances.get("bboxes", []), torch.float32)
        labels = _tensor(pred_instances.get("labels", []), torch.long)
        if mode == "same_class_iou":
            scores = _same_class_iou_scores(bboxes, labels, gt_boxes, gt_labels)
        elif mode == "one_to_one_iou":
            scores = _one_to_one_iou_scores(bboxes, labels, gt_boxes, gt_labels)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        pred_instances["scores"] = scores.to(
            dtype=_tensor(pred_instances.get("scores", scores)).dtype)
        total_dets += int(scores.numel())
        positive_scores += int((scores > 0).sum().item())
        iou50_scores += int((scores >= 0.5).sum().item())
        score_sum += float(scores.sum().item())
        updated = dict(sample)
        updated["pred_instances"] = pred_instances
        output.append(updated)

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as f:
        pickle.dump(output, f)
    return {
        "input_pkl": str(input_pkl),
        "output_pkl": str(output_pkl),
        "mode": mode,
        "num_images": len(output),
        "total_dets": total_dets,
        "positive_score_dets": positive_scores,
        "iou50_score_dets": iou50_scores,
        "mean_oracle_score": score_sum / max(total_dets, 1),
        "uses_gt": True,
        "diagnostic_only": True,
        "modifies_boxes": False,
        "modifies_labels": False,
        "uses_nms": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", type=Path, required=True)
    parser.add_argument("--ann-dir", type=Path, required=True)
    parser.add_argument("--output-pkl", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("same_class_iou", "one_to_one_iou"),
        required=True)
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = oracle_rank(args.input_pkl, args.ann_dir, args.output_pkl, args.mode)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
