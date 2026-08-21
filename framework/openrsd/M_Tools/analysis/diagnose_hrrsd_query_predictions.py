#!/usr/bin/env python3
"""Diagnose fixed-query HRRSD predictions against DOTA-style annfiles."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
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


def _quantiles(values: list[float], qs: tuple[float, ...]) -> list[float]:
    if not values:
        return [0.0 for _ in qs]
    return [float(x) for x in np.quantile(np.asarray(values), qs)]


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


def diagnose_variant(name: str, pred_path: Path, ann_dir: Path, classes: tuple[str, ...]) -> dict[str, Any]:
    with pred_path.open("rb") as f:
        predictions = pickle.load(f)

    class_to_id = {name: i for i, name in enumerate(classes)}
    scores: list[float] = []
    widths: list[float] = []
    heights: list[float] = []
    nearest_dist: list[float] = []
    nearest_rank: list[int] = []
    top10_close: list[bool] = []
    close50: list[bool] = []
    max_iou: list[float] = []
    any_iou50: list[bool] = []
    best_iou_rank: list[int] = []
    per_class = {i: {"gt": 0, "close50": 0, "top10_close": 0, "any_iou50": 0} for i in range(len(classes))}

    total_dets = 0
    for sample in predictions:
        img_id = str(sample.get("img_id"))
        gt_boxes, gt_labels = _load_gt(ann_dir / f"{img_id}.txt", class_to_id)
        pred = sample.get("pred_instances", {})
        bboxes = _tensor(pred.get("bboxes", []), torch.float32)
        pred_scores = _tensor(pred.get("scores", []), torch.float32)
        labels = _tensor(pred.get("labels", []), torch.long)
        if bboxes.numel() == 0 or pred_scores.numel() == 0:
            continue
        total_dets += int(pred_scores.numel())
        scores.extend(float(x) for x in pred_scores.tolist())
        widths.extend(float(x) for x in bboxes[:, 2].tolist())
        heights.extend(float(x) for x in bboxes[:, 3].tolist())

        order = torch.argsort(pred_scores, descending=True)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(order.numel(), dtype=order.dtype) + 1
        top = order[: min(10, int(order.numel()))]

        for gt_box, gt_label in zip(gt_boxes, gt_labels):
            cls_id = int(gt_label)
            per_class[cls_id]["gt"] += 1
            same = torch.where(labels == cls_id)[0]
            if same.numel() == 0:
                continue

            distances = torch.linalg.norm(bboxes[same, :2] - gt_box[:2], dim=1)
            nearest_local = int(torch.argmin(distances))
            nearest_idx = int(same[nearest_local])
            dist_value = float(distances[nearest_local])
            nearest_dist.append(dist_value)
            nearest_rank.append(int(rank[nearest_idx]))
            is_close = dist_value < 50.0
            close50.append(is_close)
            per_class[cls_id]["close50"] += int(is_close)

            top_same = top[labels[top] == cls_id]
            hit_top10 = False
            if top_same.numel() > 0:
                top_d = torch.linalg.norm(bboxes[top_same, :2] - gt_box[:2], dim=1)
                hit_top10 = bool((top_d < 50.0).any())
            top10_close.append(hit_top10)
            per_class[cls_id]["top10_close"] += int(hit_top10)

            ious = box_iou_rotated(bboxes[same], gt_box[None, :]).squeeze(1)
            best_local = int(torch.argmax(ious))
            best_idx = int(same[best_local])
            best_iou = float(ious[best_local])
            max_iou.append(best_iou)
            best_iou_rank.append(int(rank[best_idx]))
            has_iou50 = best_iou >= 0.5
            any_iou50.append(has_iou50)
            per_class[cls_id]["any_iou50"] += int(has_iou50)

    score_q50, score_q90, score_q99, score_max = _quantiles(scores, (0.5, 0.9, 0.99, 1.0))
    dist_med, dist_mean, dist_p90 = _quantiles(nearest_dist, (0.5, 0.5, 0.9))
    if nearest_dist:
        dist_mean = float(np.mean(np.asarray(nearest_dist)))
    iou_med, iou_mean, iou_p90 = _quantiles(max_iou, (0.5, 0.5, 0.9))
    if max_iou:
        iou_mean = float(np.mean(np.asarray(max_iou)))

    class_rows = []
    for cls_id, row in per_class.items():
        gt = row["gt"]
        if gt <= 0:
            continue
        class_rows.append({
            "class": classes[cls_id],
            "gt": gt,
            "close50": row["close50"] / gt,
            "top10_close": row["top10_close"] / gt,
            "any_iou50": row["any_iou50"] / gt,
        })
    class_rows.sort(key=lambda item: item["any_iou50"], reverse=True)

    return {
        "name": name,
        "pred_path": str(pred_path),
        "num_images": len(predictions),
        "total_dets": total_dets,
        "dets_per_image": total_dets / max(len(predictions), 1),
        "score_quantiles": {
            "q50": score_q50,
            "q90": score_q90,
            "q99": score_q99,
            "max": score_max,
        },
        "width_mean": float(np.mean(widths)) if widths else 0.0,
        "height_mean": float(np.mean(heights)) if heights else 0.0,
        "nearest_same_class": {
            "dist_median": dist_med,
            "dist_mean": dist_mean,
            "dist_p90": dist_p90,
            "close50_rate": float(np.mean(close50)) if close50 else 0.0,
            "rank_median": float(np.median(nearest_rank)) if nearest_rank else 0.0,
            "top10_close_rate": float(np.mean(top10_close)) if top10_close else 0.0,
        },
        "best_same_class_iou": {
            "max_iou_median": iou_med,
            "max_iou_mean": iou_mean,
            "max_iou_p90": iou_p90,
            "any_iou50_rate": float(np.mean(any_iou50)) if any_iou50 else 0.0,
            "best_iou_rank_median": float(np.median(best_iou_rank)) if best_iou_rank else 0.0,
        },
        "top_classes_by_any_iou50": class_rows[:8],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-dir", type=Path, required=True)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="NAME:PATH_TO_PREDICTIONS_PKL",
    )
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for item in args.variant:
        name, pred = item.split(":", 1)
        results.append(diagnose_variant(name, Path(pred), args.ann_dir, HRRSD_CLASSES))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for result in results:
        s = result["score_quantiles"]
        n = result["nearest_same_class"]
        i = result["best_same_class_iou"]
        print(result["name"])
        print(f"  dets/img {result['dets_per_image']:.1f} wh_mean {result['width_mean']:.2f}x{result['height_mean']:.2f}")
        print(f"  score q50/q90/q99/max {s['q50']:.6f} {s['q90']:.6f} {s['q99']:.6f} {s['max']:.6f}")
        print(
            "  nearest dist med/mean/p90 close rank top10 "
            f"{n['dist_median']:.2f} {n['dist_mean']:.2f} {n['dist_p90']:.2f} "
            f"{n['close50_rate']:.4f} {n['rank_median']:.0f} {n['top10_close_rate']:.4f}"
        )
        print(
            "  maxIoU med/mean/p90 any50 bestRank "
            f"{i['max_iou_median']:.3f} {i['max_iou_mean']:.3f} {i['max_iou_p90']:.3f} "
            f"{i['any_iou50_rate']:.4f} {i['best_iou_rank_median']:.0f}"
        )
        class_bits = []
        for row in result["top_classes_by_any_iou50"][:6]:
            class_bits.append(
                f"{row['class']}:iou50{row['any_iou50']:.2f}/close{row['close50']:.2f}/top10{row['top10_close']:.2f}/n{row['gt']}"
            )
        print("  best classes " + ", ".join(class_bits))


if __name__ == "__main__":
    main()
