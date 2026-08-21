"""Utilities for dumping E-P2-clean pre-NMS path logits."""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import Tensor


FIELDNAMES = [
    "case_id",
    "control_type",
    "context_id",
    "object_id",
    "pair",
    "gt_class",
    "hardneg_class",
    "path_step",
    "log_area",
    "logit_gt",
    "logit_hardneg",
    "z_gt",
    "z_hardneg",
    "abs_z_gt",
    "abs_z_hardneg",
    "hardneg_margin",
    "support_advantage",
    "image_id",
    "img_path",
    "level_idx",
    "location_idx",
    "bbox_cx",
    "bbox_cy",
    "bbox_w",
    "bbox_h",
    "bbox_angle",
    "score_gt",
    "score_hardneg",
]

EP2_NAME_RE = re.compile(r"^ep2(?P<key>case|ctx|obj|ctrl|step)-(?P<value>.+)$")
EP2_KEY_MAP = {
    "case": "ep2_case_id",
    "ctx": "ep2_context_id",
    "obj": "ep2_object_id",
    "ctrl": "ep2_control_type",
    "step": "ep2_path_step",
}


def parse_ep2_metadata_from_path(path: str | os.PathLike | None) -> dict:
    """Parse E-P2 metadata encoded in a variant image filename."""
    if not path:
        return {}
    stem = Path(str(path)).stem
    parsed = {}
    for part in stem.split("__"):
        match = EP2_NAME_RE.match(part)
        if match is None:
            continue
        key = EP2_KEY_MAP.get(match.group("key"))
        if key:
            parsed[key] = match.group("value")
    return parsed


def _to_pair(pair) -> tuple[str, str]:
    if isinstance(pair, str):
        if "->" not in pair:
            raise ValueError(f"class pair must use 'gt->hardneg': {pair!r}")
        left, right = pair.split("->", 1)
        return left, right
    if isinstance(pair, Sequence) and len(pair) == 2:
        return str(pair[0]), str(pair[1])
    raise ValueError(f"invalid class pair: {pair!r}")


def _class_index(class_names: Sequence[str], class_name: str) -> int | None:
    try:
        return list(class_names).index(class_name)
    except ValueError:
        return None


def _meta_value(img_meta: dict | None, key: str, fallback=""):
    if not img_meta:
        return fallback
    if key in img_meta and img_meta[key] not in {None, ""}:
        return img_meta[key]
    parsed = parse_ep2_metadata_from_path(img_meta.get("img_path"))
    return parsed.get(key, fallback)


def _default_case_id(img_meta: dict | None, level_idx: int, location_idx: int,
                     pair: str) -> str:
    image_id = _meta_value(img_meta, "img_id", "")
    if not image_id:
        image_id = Path(str(_meta_value(img_meta, "img_path", ""))).stem
    return f"{image_id}:L{level_idx}:P{location_idx}:{pair}"


def build_ep2_path_probe_rows(
    *,
    cls_logits: Tensor,
    decoded_bboxes: Tensor,
    class_names: Sequence[str],
    class_pairs: Iterable[Sequence[str] | str],
    log_area_mean: Tensor,
    log_area_std: Tensor,
    valid_mask: Tensor,
    img_meta: dict | None = None,
    level_idx: int = -1,
    max_locations: int = 128,
) -> list[dict]:
    """Build canonical E-P2-clean rows from one dense feature level.

    ``cls_logits`` must be raw class logits before sigmoid/softmax. This
    function only records evidence; it never mutates logits or predictions.
    """
    if cls_logits.numel() == 0 or decoded_bboxes.numel() == 0:
        return []
    if decoded_bboxes.shape[-1] < 4:
        return []

    num_classes = int(cls_logits.shape[1])
    class_names = tuple(class_names)[:num_classes]
    means = log_area_mean.to(
        device=cls_logits.device, dtype=cls_logits.dtype)[:num_classes]
    stds = log_area_std.to(
        device=cls_logits.device, dtype=cls_logits.dtype)[:num_classes].clamp(
            min=1e-6)
    valid = valid_mask.to(device=cls_logits.device)[:num_classes].bool()

    area = (decoded_bboxes[:, 2].abs() * decoded_bboxes[:, 3].abs()).clamp(
        min=1e-6)
    log_area = torch.log(area).to(cls_logits.dtype)
    scores = torch.sigmoid(cls_logits)
    rows: list[dict] = []
    image_id = _meta_value(img_meta, "img_id", "")
    img_path = _meta_value(img_meta, "img_path", "")
    context_id = _meta_value(img_meta, "ep2_context_id", image_id)
    object_id = _meta_value(img_meta, "ep2_object_id", "")
    control_type = _meta_value(img_meta, "ep2_control_type", "clean")
    path_step = _meta_value(img_meta, "ep2_path_step", "")

    for raw_pair in class_pairs:
        gt_class, hardneg_class = _to_pair(raw_pair)
        gt_idx = _class_index(class_names, gt_class)
        hardneg_idx = _class_index(class_names, hardneg_class)
        if gt_idx is None or hardneg_idx is None:
            continue
        if not bool(valid[gt_idx]) or not bool(valid[hardneg_idx]):
            continue
        pair = f"{gt_class}->{hardneg_class}"
        pair_scores = cls_logits[:, hardneg_idx]
        keep = min(int(max_locations), int(pair_scores.numel()))
        if keep <= 0:
            continue
        _, loc_indices = torch.topk(pair_scores, keep)
        for loc_idx_tensor in loc_indices.detach().cpu():
            loc_idx = int(loc_idx_tensor.item())
            logit_gt = float(cls_logits[loc_idx, gt_idx].detach().cpu().item())
            logit_hardneg = float(
                cls_logits[loc_idx, hardneg_idx].detach().cpu().item())
            bbox = decoded_bboxes[loc_idx].detach().cpu().tolist()
            log_area_value = float(log_area[loc_idx].detach().cpu().item())
            z_gt = float(((log_area[loc_idx] - means[gt_idx]) / stds[gt_idx])
                         .detach().cpu().item())
            z_hardneg = float(
                ((log_area[loc_idx] - means[hardneg_idx])
                 / stds[hardneg_idx]).detach().cpu().item())
            case_id = _meta_value(
                img_meta,
                "ep2_case_id",
                _default_case_id(img_meta, level_idx, loc_idx, pair))
            row_object_id = object_id or f"{image_id}:L{level_idx}:P{loc_idx}"
            rows.append({
                "case_id": case_id,
                "control_type": control_type,
                "context_id": context_id,
                "object_id": row_object_id,
                "pair": pair,
                "gt_class": gt_class,
                "hardneg_class": hardneg_class,
                "path_step": path_step,
                "log_area": log_area_value,
                "logit_gt": logit_gt,
                "logit_hardneg": logit_hardneg,
                "z_gt": z_gt,
                "z_hardneg": z_hardneg,
                "abs_z_gt": abs(z_gt),
                "abs_z_hardneg": abs(z_hardneg),
                "hardneg_margin": logit_hardneg - logit_gt,
                "support_advantage": abs(z_gt) - abs(z_hardneg),
                "image_id": image_id,
                "img_path": img_path,
                "level_idx": int(level_idx),
                "location_idx": loc_idx,
                "bbox_cx": float(bbox[0]),
                "bbox_cy": float(bbox[1]),
                "bbox_w": float(bbox[2]),
                "bbox_h": float(bbox[3]),
                "bbox_angle": float(bbox[4]) if len(bbox) > 4 else 0.0,
                "score_gt": float(scores[loc_idx, gt_idx]
                                  .detach().cpu().item()),
                "score_hardneg": float(scores[loc_idx, hardneg_idx]
                                       .detach().cpu().item()),
            })
    return rows


def append_ep2_path_probe_rows(path: str | os.PathLike,
                               rows: Iterable[dict]) -> int:
    """Append rows to an E-P2-clean raw-logit CSV, writing header once."""
    rows = list(rows)
    if not rows:
        return 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})
    return len(rows)
