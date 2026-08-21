from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def polygon_to_hbb(poly: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    arr = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    return float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 0].max()), float(arr[:, 1].max())


def hbb_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def rbox_to_polygon(rbox: Sequence[float]) -> List[List[float]]:
    cx, cy, w, h, angle = [float(v) for v in rbox[:5]]
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    points = []
    for x, y in corners:
        points.append([cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a])
    return points


def flat_poly_to_points(values: Iterable[float]) -> List[List[float]]:
    nums = [float(v) for v in values]
    if len(nums) < 8 or len(nums) % 2 != 0:
        raise ValueError(f"expected even polygon coordinates >= 8, got {len(nums)}")
    return [[nums[i], nums[i + 1]] for i in range(0, len(nums), 2)]


def ensure_polygon(box: Sequence[float], box_type: str) -> List[List[float]]:
    if box_type == "polygon":
        if len(box) == 1 and isinstance(box[0], (list, tuple)):
            return [[float(x), float(y)] for x, y in box[0]]
        return flat_poly_to_points(box)
    if box_type == "hbb":
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    if box_type == "obb":
        return rbox_to_polygon(box)
    raise ValueError(f"unsupported box_type: {box_type}")

