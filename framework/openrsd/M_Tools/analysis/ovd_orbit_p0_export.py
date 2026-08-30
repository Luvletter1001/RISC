"""Normalization helpers for full OpenRSD semantic score fields.

The live hook integration belongs to a later no-GPU preflight.  This module
only converts already-aligned tensor fields into immutable P0 score records.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from M_Tools.analysis.ovd_orbit_p0 import OrbitP0Error, ScoreCarrier


def _to_cpu_float32(value: Any, *, name: str, trailing_shape: tuple[int, ...]) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - OpenRSD runtime requires torch
        raise OrbitP0Error('torch is required to normalize model tensors') from exc
    if not torch.is_tensor(value) or not torch.is_floating_point(value):
        raise OrbitP0Error(f'{name} must be a floating torch tensor')
    if value.ndim != len(trailing_shape) + 1 or tuple(value.shape[1:]) != trailing_shape:
        raise OrbitP0Error(f'{name} must have shape [N,{",".join(map(str, trailing_shape))}]')
    if not bool(torch.isfinite(value).all()):
        raise OrbitP0Error(f'{name} must be finite')
    return value.detach().to(device='cpu', dtype=torch.float32).contiguous().numpy().copy()


def normalize_level_outputs(
        *, view_id: str, scene_id: str, level: int, boxes: Any,
        scores: Any) -> tuple[ScoreCarrier, ...]:
    """Return exact `(level,row)` full-score records without score selection."""
    if isinstance(level, bool) or not isinstance(level, int) or level < 0:
        raise OrbitP0Error('level must be a nonnegative integer')
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - OpenRSD runtime requires torch
        raise OrbitP0Error('torch is required to normalize model tensors') from exc
    if (not torch.is_tensor(scores) or not torch.is_floating_point(scores)
            or scores.ndim != 2 or scores.shape[1] <= 0):
        raise OrbitP0Error('scores must be a floating [N,C] tensor with C > 0')
    boxes_np = _to_cpu_float32(boxes, name='boxes', trailing_shape=(5,))
    scores_np = _to_cpu_float32(
        scores, name='scores', trailing_shape=(int(scores.shape[1]),))
    if len(boxes_np) != len(scores_np):
        raise OrbitP0Error('boxes and scores must align')
    return tuple(
        ScoreCarrier(
            view_id=view_id,
            scene_id=scene_id,
            source=(level, row),
            box=boxes_np[row],
            scores=scores_np[row],
        )
        for row in range(len(boxes_np))
    )


__all__ = ['normalize_level_outputs']
