#!/usr/bin/env python3
"""Measure AP50 headroom from an oracle background counter-support filter."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mmcv.ops import box_iou_rotated
from mmrotate.evaluation.functional.mean_ap import eval_rbbox_map
from mmrotate.structures.bbox import qbox2rbox

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from M_Tools.analysis.eval_per_bin_ap import DOTA_CLASSES


def _empty_prediction_record(image_id: str, tile_id: str = "") -> dict[str, Any]:
    return {
        "image_id": image_id,
        "tile_id": tile_id,
        "boxes": [],
        "scores": [],
        "labels": [],
    }


def load_angle_prediction_records(
        path: Path, angle: int) -> dict[str, dict[str, Any]]:
    """Stream one angle from the existing line-delimited prediction dump."""
    records: dict[str, dict[str, Any]] = {}
    class_to_label = {name: idx for idx, name in enumerate(DOTA_CLASSES)}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row["angle"]) != int(angle):
                continue
            image_id = str(row["image_id"])
            record = records.setdefault(
                image_id,
                _empty_prediction_record(image_id, str(row.get("tile_id", ""))))
            record["boxes"].append(row["rbox"])
            record["scores"].append(float(row["score"]))
            class_name = str(row.get("class_name", "")).lower().replace("_", "-")
            record["labels"].append(
                class_to_label.get(class_name, int(row["class_id"])))
    for record in records.values():
        record["boxes"] = np.asarray(record["boxes"], dtype=np.float32).reshape(-1, 5)
        record["scores"] = np.asarray(record["scores"], dtype=np.float32)
        record["labels"] = np.asarray(record["labels"], dtype=np.int64)
    return records


def parse_dota_annotation(
        path: Path,
        class_to_label: dict[str, int],
        diff_thr: int = 0) -> dict[str, np.ndarray]:
    """Parse one DOTA polygon annotation into regular and ignored rboxes."""
    qboxes = []
    labels = []
    qboxes_ignore = []
    labels_ignore = []
    for line in Path(path).read_text(
            encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) < 9 or parts[8] not in class_to_label:
            continue
        try:
            qbox = [float(value) for value in parts[:8]]
            difficulty = int(parts[9]) if len(parts) > 9 else 0
        except ValueError:
            continue
        if difficulty > diff_thr:
            qboxes_ignore.append(qbox)
            labels_ignore.append(class_to_label[parts[8]])
        else:
            qboxes.append(qbox)
            labels.append(class_to_label[parts[8]])

    def to_rboxes(values: list[list[float]]) -> np.ndarray:
        qbox_array = np.asarray(values, dtype=np.float32).reshape(-1, 8)
        if len(qbox_array) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        return qbox2rbox(torch.from_numpy(qbox_array)).numpy().astype(np.float32)

    return {
        "gt_boxes": to_rboxes(qboxes),
        "gt_labels": np.asarray(labels, dtype=np.int64),
        "gt_boxes_ignore": to_rboxes(qboxes_ignore),
        "gt_labels_ignore": np.asarray(labels_ignore, dtype=np.int64),
    }


def background_mask(
        pred_boxes: np.ndarray,
        gt_boxes: np.ndarray,
        iou_thr: float = 0.1) -> np.ndarray:
    """Return True for predictions with no overlap to any annotated object."""
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 5)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 5)
    if len(pred_boxes) == 0:
        return np.zeros((0,), dtype=bool)
    if len(gt_boxes) == 0:
        return np.ones((len(pred_boxes),), dtype=bool)
    ious = box_iou_rotated(
        torch.from_numpy(pred_boxes), torch.from_numpy(gt_boxes)).cpu().numpy()
    return ious.max(axis=1) < float(iou_thr)


def filter_sample(
        sample: dict[str, Any],
        mode: str,
        bg_iou_thr: float,
        small_vehicle_label: int = 4) -> dict[str, Any]:
    """Apply one oracle mode while preserving the original sample schema."""
    out = dict(sample)
    if mode == "baseline":
        return out
    if mode not in {"bg_oracle", "sv_bg_oracle"}:
        raise ValueError(f"unsupported mode: {mode}")
    boxes = np.asarray(sample["boxes"])
    scores = np.asarray(sample["scores"])
    labels = np.asarray(sample["labels"])
    is_background = background_mask(boxes, sample["gt_boxes"], bg_iou_thr)
    remove = is_background
    if mode == "sv_bg_oracle":
        remove = is_background & (labels == int(small_vehicle_label))
    keep = ~remove
    out["boxes"] = boxes[keep]
    out["scores"] = scores[keep]
    out["labels"] = labels[keep]
    out["removed_mask"] = remove
    return out


def evaluate_samples(
        samples: list[dict[str, Any]],
        class_names: tuple[str, ...] | list[str],
        nproc: int = 1) -> dict[str, Any]:
    """Evaluate in-memory rotated detections with the project DOTA metric."""
    annotations = []
    det_results = []
    for sample in samples:
        annotations.append({
            "bboxes": np.asarray(sample["gt_boxes"], dtype=np.float32).reshape(-1, 5),
            "labels": np.asarray(sample["gt_labels"], dtype=np.int64),
            "bboxes_ignore": np.asarray(
                sample.get("gt_boxes_ignore", []), dtype=np.float32).reshape(-1, 5),
            "labels_ignore": np.asarray(
                sample.get("gt_labels_ignore", []), dtype=np.int64),
        })
        boxes = np.asarray(sample["boxes"], dtype=np.float32).reshape(-1, 5)
        scores = np.asarray(sample["scores"], dtype=np.float32)
        labels = np.asarray(sample["labels"], dtype=np.int64)
        per_class = []
        for label in range(len(class_names)):
            selected = labels == label
            if selected.any():
                per_class.append(np.column_stack((boxes[selected], scores[selected])))
            else:
                per_class.append(np.zeros((0, 6), dtype=np.float32))
        det_results.append(per_class)

    mean_ap, class_results = eval_rbbox_map(
        det_results,
        annotations,
        iou_thr=0.5,
        use_07_metric=True,
        box_type="rbox",
        dataset=tuple(class_names),
        logger="silent",
        nproc=nproc,
    )
    per_class = []
    for class_name, result in zip(class_names, class_results):
        ap = float(np.asarray(result["ap"]).reshape(-1)[0])
        per_class.append({
            "class_name": class_name,
            "ap50": ap,
            "num_gts": int(np.asarray(result["num_gts"]).reshape(-1)[0]),
            "num_dets": int(result["num_dets"]),
        })
    return {"map50": float(mean_ap), "per_class": per_class}


def summarize_removed_predictions(
        base_samples: list[dict[str, Any]],
        filtered_samples: list[dict[str, Any]],
        tp_iou_thr: float = 0.5,
        ap_active_score_floor: float = 0.3) -> dict[str, int]:
    """Count removed predictions and verify that none are GT-matched TPs."""
    removed_total = 0
    removed_tp = 0
    ap_active_background = 0
    for base, filtered in zip(base_samples, filtered_samples):
        removed = np.asarray(
            filtered.get("removed_mask", np.zeros(len(base["scores"]), dtype=bool)),
            dtype=bool)
        removed_total += int(removed.sum())
        if not removed.any():
            continue
        boxes = np.asarray(base["boxes"], dtype=np.float32).reshape(-1, 5)
        scores = np.asarray(base["scores"], dtype=np.float32)
        labels = np.asarray(base["labels"], dtype=np.int64)
        gt_boxes = np.asarray(base["gt_boxes"], dtype=np.float32).reshape(-1, 5)
        gt_labels = np.asarray(base["gt_labels"], dtype=np.int64)
        is_background = background_mask(boxes, gt_boxes, iou_thr=0.1)
        ap_active_background += int(
            (removed & is_background & (scores >= ap_active_score_floor)).sum())
        if len(gt_boxes):
            removed_boxes = boxes[removed]
            removed_labels = labels[removed]
            ious = box_iou_rotated(
                torch.from_numpy(removed_boxes),
                torch.from_numpy(gt_boxes)).cpu().numpy()
            same_class = removed_labels[:, None] == gt_labels[None, :]
            same_class_iou = np.where(same_class, ious, -1.0)
            removed_tp += int((same_class_iou.max(axis=1) >= tp_iou_thr).sum())
    return {
        "removed_prediction_count": removed_total,
        "removed_tp_count": removed_tp,
        "ap_active_background_fp_removed": ap_active_background,
    }


def build_annotation_index(ann_dirs: list[Path]) -> dict[str, Path]:
    """Index existing DOTA tile annotations without copying the dataset."""
    index: dict[str, Path] = {}
    for ann_dir in ann_dirs:
        for path in sorted(Path(ann_dir).glob("*.txt")):
            index.setdefault(path.stem, path)
    return index


def load_image_rows(images_csv: Path, angle: int) -> list[dict[str, str]]:
    with Path(images_csv).open("r", encoding="utf-8", newline="") as handle:
        return [
            row for row in csv.DictReader(handle)
            if int(row["angle"]) == int(angle)
        ]


def assemble_samples(
        prediction_records: dict[str, dict[str, Any]],
        image_rows: list[dict[str, str]],
        annotation_index: dict[str, Path],
        class_names: tuple[str, ...] | list[str],
        diff_thr: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    class_to_label = {name: idx for idx, name in enumerate(class_names)}
    samples = []
    missing_tiles = []
    missing_reported_gt = 0
    for row in image_rows:
        image_id = row["image_id"]
        tile_id = row["tile_id"]
        pred = prediction_records.get(
            image_id, _empty_prediction_record(image_id, tile_id))
        pred = dict(pred)
        ann_path = annotation_index.get(tile_id)
        if ann_path is None:
            gt = {
                "gt_boxes": np.zeros((0, 5), dtype=np.float32),
                "gt_labels": np.zeros((0,), dtype=np.int64),
                "gt_boxes_ignore": np.zeros((0, 5), dtype=np.float32),
                "gt_labels_ignore": np.zeros((0,), dtype=np.int64),
            }
            missing_tiles.append(tile_id)
            missing_reported_gt += int(row.get("num_gt", 0))
        else:
            gt = parse_dota_annotation(ann_path, class_to_label, diff_thr)
        pred.update(gt)
        samples.append(pred)
    return samples, {
        "image_count": len(image_rows),
        "prediction_image_count": len(prediction_records),
        "missing_annotation_count": len(missing_tiles),
        "missing_reported_gt_count": missing_reported_gt,
        "missing_annotation_tiles": missing_tiles,
    }


def _per_class_rows(
        baseline: dict[str, Any],
        bg_oracle: dict[str, Any],
        sv_oracle: dict[str, Any]) -> list[dict[str, Any]]:
    bg_by_class = {row["class_name"]: row for row in bg_oracle["per_class"]}
    sv_by_class = {row["class_name"]: row for row in sv_oracle["per_class"]}
    rows = []
    for base in baseline["per_class"]:
        name = base["class_name"]
        bg = bg_by_class[name]
        sv = sv_by_class[name]
        rows.append({
            "class_name": name,
            "num_gts": base["num_gts"],
            "baseline_num_dets": base["num_dets"],
            "baseline_ap50": base["ap50"],
            "bg_oracle_num_dets": bg["num_dets"],
            "bg_oracle_ap50": bg["ap50"],
            "bg_oracle_delta": bg["ap50"] - base["ap50"],
            "sv_bg_oracle_num_dets": sv["num_dets"],
            "sv_bg_oracle_ap50": sv["ap50"],
            "sv_bg_oracle_delta": sv["ap50"] - base["ap50"],
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    checks = payload["checks"]
    status = payload["decision"]
    lines = [
        "# CSER Phase-0 Background Oracle",
        "",
        f"- status: `{status}`",
        f"- angle: `{payload['angle']}`",
        f"- bg_iou_thr: `{payload['bg_iou_thr']}`",
        f"- images: `{payload['input_status']['image_count']}`",
        f"- missing_annotation_count: `{payload['input_status']['missing_annotation_count']}`",
        "",
        "## 核心结果",
        "",
        "| variant | mAP50 | delta | removed_prediction_count | removed_tp_count |",
        "|---|---:|---:|---:|---:|",
        (
            f"| baseline | {metrics['baseline_map50']:.6f} | 0.000000 | 0 | 0 |"),
        (
            f"| bg_oracle_0p1 | {metrics['bg_oracle_map50']:.6f} | "
            f"{metrics['bg_oracle_delta']:+.6f} | "
            f"{payload['removal']['bg_oracle']['removed_prediction_count']} | "
            f"{payload['removal']['bg_oracle']['removed_tp_count']} |"),
        (
            f"| sv_bg_oracle_0p1 | {metrics['sv_bg_oracle_map50']:.6f} | "
            f"{metrics['sv_bg_oracle_delta']:+.6f} | "
            f"{payload['removal']['sv_bg_oracle']['removed_prediction_count']} | "
            f"{payload['removal']['sv_bg_oracle']['removed_tp_count']} |"),
        "",
        "## 五项检查",
        "",
        "| check | pass | value |",
        "|---|---|---|",
    ]
    for name, item in checks.items():
        lines.append(f"| {name} | {str(item['pass']).lower()} | {item['value']} |")
    lines.extend([
        "",
        "## 证据层级",
        "",
        (
            "- **Observed：** 本地 baseline 直接由已有 angle=0 预测重算；"
            f"相对历史值的误差为 {metrics['baseline_reference_abs_error']:.6f}。"),
        (
            "- **Oracle：** `bg_oracle_0p1` 与 `sv_bg_oracle_0p1` 使用验证 GT "
            "删除背景预测，只表示理论可恢复上界。"),
        (
            "- **Inference：** 只有 baseline 等价、TP 保留和 AP headroom gate "
            "同时通过，才允许进入 frozen-feature 最小拟合。"),
        "",
        "## 决策",
        "",
        payload["conclusion"],
        "",
        "本实验使用 GT 删除背景预测，只是 oracle 上界，不是可部署推理方法。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    class_names = tuple(DOTA_CLASSES)
    prediction_records = load_angle_prediction_records(
        args.predictions_jsonl, args.angle)
    image_rows = load_image_rows(args.images_csv, args.angle)
    annotation_index = build_annotation_index(args.ann_dir)
    base_samples, input_status = assemble_samples(
        prediction_records,
        image_rows,
        annotation_index,
        class_names,
        diff_thr=args.diff_thr,
    )
    baseline_samples = [
        filter_sample(sample, "baseline", args.bg_iou_thr)
        for sample in base_samples
    ]
    bg_samples = [
        filter_sample(sample, "bg_oracle", args.bg_iou_thr)
        for sample in base_samples
    ]
    sv_samples = [
        filter_sample(sample, "sv_bg_oracle", args.bg_iou_thr)
        for sample in base_samples
    ]
    baseline = evaluate_samples(baseline_samples, class_names, args.nproc)
    bg_oracle = evaluate_samples(bg_samples, class_names, args.nproc)
    sv_oracle = evaluate_samples(sv_samples, class_names, args.nproc)
    bg_removal = summarize_removed_predictions(base_samples, bg_samples)
    sv_removal = summarize_removed_predictions(base_samples, sv_samples)
    baseline_error = abs(baseline["map50"] - args.reference_map50)
    bg_delta = bg_oracle["map50"] - baseline["map50"]
    sv_delta = sv_oracle["map50"] - baseline["map50"]
    checks = {
        "baseline_equivalence": {
            "pass": baseline_error <= 0.002,
            "value": f"abs_error={baseline_error:.6f}",
        },
        "strict_output_scope": {"pass": True, "value": "offline_rbox_class_score_only"},
        "tp_preservation": {
            "pass": bg_removal["removed_tp_count"] == 0,
            "value": f"removed_tp_count={bg_removal['removed_tp_count']}",
        },
        "ap_active_background": {
            "pass": bg_removal["ap_active_background_fp_removed"] > 0,
            "value": (
                "ap_active_background_fp_removed="
                f"{bg_removal['ap_active_background_fp_removed']}"),
        },
        "map_headroom": {
            "pass": bg_delta >= 0.005,
            "value": f"delta={bg_delta:+.6f}",
        },
    }
    all_pass = all(item["pass"] for item in checks.values())
    if not checks["baseline_equivalence"]["pass"]:
        decision = "BLOCKED_BASELINE_MISMATCH"
        conclusion = "本地 AP50 未复现历史基线；在修复 GT/metric 对齐前，不解释 oracle 数值。"
    elif all_pass:
        decision = "PASS_TO_FROZEN_FEATURE_GATE"
        conclusion = "背景 false positive 具有足够 AP50 上界，允许进入 GPU 8/9 的 frozen-feature counter-support 最小拟合。"
    else:
        decision = "STOP_NO_MAP_HEADROOM"
        conclusion = "背景 false positive 未形成足够 AP50 上界，CSER 暂不进入训练。"
    payload = {
        "status": "DONE",
        "decision": decision,
        "angle": args.angle,
        "bg_iou_thr": args.bg_iou_thr,
        "reference_map50": args.reference_map50,
        "input_status": input_status,
        "metrics": {
            "baseline_map50": baseline["map50"],
            "baseline_reference_abs_error": baseline_error,
            "bg_oracle_map50": bg_oracle["map50"],
            "bg_oracle_delta": bg_delta,
            "sv_bg_oracle_map50": sv_oracle["map50"],
            "sv_bg_oracle_delta": sv_delta,
        },
        "removal": {
            "bg_oracle": bg_removal,
            "sv_bg_oracle": sv_removal,
        },
        "checks": checks,
        "conclusion": conclusion,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "cser_phase0_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        args.out_dir / "cser_phase0_per_class.csv",
        _per_class_rows(baseline, bg_oracle, sv_oracle),
    )
    _write_report(
        args.out_dir / "fres_cser_phase0_background_oracle.md", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--images-csv", type=Path, required=True)
    parser.add_argument("--ann-dir", type=Path, action="append", required=True)
    parser.add_argument("--angle", type=int, default=0)
    parser.add_argument("--bg-iou-thr", type=float, default=0.1)
    parser.add_argument("--diff-thr", type=int, default=0)
    parser.add_argument("--reference-map50", type=float, required=True)
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
