"""In-memory normalization helpers for full OpenRSD semantic score fields."""

from __future__ import annotations

from typing import Any

import numpy as np

from M_Tools.analysis.ovd_orbit_p0 import (
    OVD_ORBIT_P0_VIEW_IDS,
    OrbitP0Error,
    ScoreCarrier,
    collapse_carriers,
)


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
        scores: Any,
        calibrated_scores: Any | None = None) -> tuple[ScoreCarrier, ...]:
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
    calibrated_scores_np = None
    if calibrated_scores is not None:
        if (not torch.is_tensor(calibrated_scores)
                or not torch.is_floating_point(calibrated_scores)
                or calibrated_scores.ndim != 2
                or tuple(calibrated_scores.shape) != tuple(scores.shape)):
            raise OrbitP0Error(
                'calibrated_scores must be a floating [N,C] tensor aligned '
                'with scores')
        calibrated_scores_np = _to_cpu_float32(
            calibrated_scores,
            name='calibrated_scores',
            trailing_shape=(int(scores.shape[1]),))
    if len(boxes_np) != len(scores_np):
        raise OrbitP0Error('boxes and scores must align')
    return tuple(
        ScoreCarrier(
            view_id=view_id,
            scene_id=scene_id,
            source=(level, row),
            box=boxes_np[row],
            scores=scores_np[row],
            calibrated_scores=(
                None if calibrated_scores_np is None
                else calibrated_scores_np[row]),
        )
        for row in range(len(boxes_np))
    )


class OvdOrbitFullLogitSink:
    """Collect normalized complete score vectors for one image at a time."""

    def __init__(self) -> None:
        self._active: tuple[str, str] | None = None
        self._active_record_start: int | None = None
        self._last_committed_record_start: int | None = None
        self._records: list[ScoreCarrier] = []

    def begin_image(self, *, scene_id: str, view_id: str) -> None:
        if self._active is not None:
            raise OrbitP0Error('an image context is already active')
        if not isinstance(scene_id, str) or not scene_id:
            raise OrbitP0Error('scene_id must be a nonempty string')
        if (not isinstance(view_id, str)
                or view_id not in OVD_ORBIT_P0_VIEW_IDS):
            raise OrbitP0Error('view_id must belong to the frozen P0 views')
        self._last_committed_record_start = None
        self._active = (scene_id, view_id)
        self._active_record_start = len(self._records)

    def record_level(self,
                     *,
                     level: int,
                     boxes: Any,
                     scores: Any,
                     calibrated_scores: Any | None = None) -> None:
        if self._active is None:
            raise OrbitP0Error('an image context must be active before recording')
        scene_id, view_id = self._active
        self._records.extend(normalize_level_outputs(
            view_id=view_id,
            scene_id=scene_id,
            level=level,
            boxes=boxes,
            scores=scores,
            calibrated_scores=calibrated_scores,
        ))

    def record_level_with_calibrated_scores(
            self,
            *,
            level: int,
            boxes: Any,
            scores: Any,
            calibrated_scores: Any) -> None:
        self.record_level(
            level=level,
            boxes=boxes,
            scores=scores,
            calibrated_scores=calibrated_scores)

    def end_image(self) -> None:
        if self._active is None or self._active_record_start is None:
            raise OrbitP0Error('an image context must be active before ending')
        self._last_committed_record_start = self._active_record_start
        self._active = None
        self._active_record_start = None

    def abort_image(self) -> None:
        if self._active is not None:
            start = self._active_record_start
        else:
            start = self._last_committed_record_start
        if start is None:
            raise OrbitP0Error('an image context must be active before aborting')
        del self._records[start:]
        self._active = None
        self._active_record_start = None
        self._last_committed_record_start = None

    def snapshot(self) -> tuple[ScoreCarrier, ...]:
        if self._active is not None:
            raise OrbitP0Error('cannot snapshot while an image context is active')
        records = collapse_carriers(self._records)
        self._last_committed_record_start = None
        return records


__all__ = ['OvdOrbitFullLogitSink', 'normalize_level_outputs']
