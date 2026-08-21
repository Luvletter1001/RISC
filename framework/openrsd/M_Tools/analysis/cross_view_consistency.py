"""Prototype detector wrapper for cross-view consistency experiments.

This module is intentionally small: it reuses RTMDet's normal detection loss
and adds an auxiliary consistency loss between the original image view and a
rotated tensor view. It is a prototype diagnostic, not a full rotation-aware
box matching method.
"""

from __future__ import annotations

import random
from typing import Iterable

import torch
import torch.nn.functional as F

from mmdet.models.detectors.rtmdet import RTMDet
from mmdet.registry import MODELS as MMDET_MODELS
from mmrotate.registry import MODELS as MMROTATE_MODELS


def _rotate_batch(inputs: torch.Tensor, degrees: float) -> torch.Tensor:
    if abs(degrees) < 1e-6:
        return inputs
    theta = torch.tensor(
        [[
            [torch.cos(torch.deg2rad(torch.tensor(degrees, device=inputs.device))),
             -torch.sin(torch.deg2rad(torch.tensor(degrees, device=inputs.device))), 0.0],
            [torch.sin(torch.deg2rad(torch.tensor(degrees, device=inputs.device))),
             torch.cos(torch.deg2rad(torch.tensor(degrees, device=inputs.device))), 0.0],
        ]],
        dtype=inputs.dtype,
        device=inputs.device,
    )
    theta = theta.repeat(inputs.shape[0], 1, 1)
    grid = F.affine_grid(theta, inputs.size(), align_corners=False)
    return F.grid_sample(
        inputs, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


def _iter_tensors(outputs) -> Iterable[torch.Tensor]:
    if torch.is_tensor(outputs):
        yield outputs
    elif isinstance(outputs, (list, tuple)):
        for item in outputs:
            yield from _iter_tensors(item)
    elif isinstance(outputs, dict):
        for item in outputs.values():
            yield from _iter_tensors(item)


def _mean_consistency(student, teacher, loss_type: str = 'mse') -> torch.Tensor:
    losses = []
    for s, t in zip(_iter_tensors(student), _iter_tensors(teacher)):
        if s.shape != t.shape or not torch.is_floating_point(s):
            continue
        if loss_type == 'kl' and s.ndim >= 2:
            s_flat = s.flatten(2).transpose(1, 2).reshape(-1, s.shape[1])
            t_flat = t.flatten(2).transpose(1, 2).reshape(-1, t.shape[1])
            losses.append(
                F.kl_div(
                    F.log_softmax(s_flat, dim=-1),
                    F.softmax(t_flat.detach(), dim=-1),
                    reduction='batchmean'))
        else:
            losses.append(F.mse_loss(s, t.detach()))
    if losses:
        return torch.stack(losses).mean()
    for item in _iter_tensors(student):
        return torch.zeros((), device=item.device)
    return torch.zeros(())


@MMROTATE_MODELS.register_module(force=True)
@MMDET_MODELS.register_module(force=True)
class CrossViewConsistencyRTMDet(RTMDet):
    """RTMDet with a lightweight cross-view logits consistency term."""

    def __init__(self,
                 *args,
                 consistency_angles=(30, 60, 90, 120, 150),
                 lambda_cls: float = 0.1,
                 lambda_box: float = 0.0,
                 consistency_loss: str = 'mse',
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.consistency_angles = tuple(consistency_angles)
        self.lambda_cls = float(lambda_cls)
        self.lambda_box = float(lambda_box)
        self.consistency_loss = consistency_loss

    def loss(self, batch_inputs: torch.Tensor, batch_data_samples):
        losses = super().loss(batch_inputs, batch_data_samples)
        angle = float(random.choice(self.consistency_angles))
        with torch.no_grad():
            teacher_feats = self.extract_feat(batch_inputs)
            teacher_outs = self.bbox_head(teacher_feats)
        student_inputs = _rotate_batch(batch_inputs, angle)
        student_feats = self.extract_feat(student_inputs)
        student_outs = self.bbox_head(student_feats)

        # MMRotate RTMDet heads return nested outputs. For RTMDet this is
        # normally cls scores plus bbox/angle outputs. We keep the prototype
        # scoped to head-space consistency and record the exact limitation in
        # the experiment report.
        losses['loss_cv_cls'] = (
            _mean_consistency(student_outs[0], teacher_outs[0],
                              self.consistency_loss) * self.lambda_cls)
        if self.lambda_box > 0 and len(student_outs) > 1 and len(teacher_outs) > 1:
            losses['loss_cv_box'] = (
                _mean_consistency(student_outs[1:], teacher_outs[1:], 'mse') *
                self.lambda_box)
        return losses
