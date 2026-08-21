"""OpenRSD-compatible prototype cross-view consistency wrapper."""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from mmdet.registry import MODELS as MMDET_MODELS
from mmrotate.registry import MODELS as MMROTATE_MODELS

from M_AD.models.detectors.Flex_Rtmdet_v3_1_formal import OpenRTMDet


def _rotate_batch(inputs: torch.Tensor, degrees: float) -> torch.Tensor:
    if abs(degrees) < 1e-6:
        return inputs
    rad = torch.deg2rad(torch.tensor(float(degrees), device=inputs.device))
    cos_v = torch.cos(rad)
    sin_v = torch.sin(rad)
    zero = torch.zeros((), dtype=inputs.dtype, device=inputs.device)
    theta = torch.stack((
        torch.stack((cos_v.to(inputs.dtype), -sin_v.to(inputs.dtype), zero)),
        torch.stack((sin_v.to(inputs.dtype), cos_v.to(inputs.dtype), zero)),
    )).unsqueeze(0).repeat(inputs.shape[0], 1, 1)
    grid = F.affine_grid(theta, inputs.size(), align_corners=False)
    return F.grid_sample(inputs, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


def _global_pool_tuple(features) -> list[torch.Tensor]:
    out = []
    for feat in features if isinstance(features, (list, tuple)) else [features]:
        if torch.is_tensor(feat) and feat.ndim >= 4 and torch.is_floating_point(feat):
            out.append(feat.mean(dim=(2, 3)))
    return out


@MMROTATE_MODELS.register_module(force=True)
@MMDET_MODELS.register_module(force=True)
class OpenRTMDetCrossViewConsistency(OpenRTMDet):
    """Add a lightweight feature-space consistency term to OpenRTMDet.

    This is intentionally a prototype: it verifies that the training loop can
    carry a cross-view consistency loss without NaN/OOM, but it does not yet do
    rotation-aware box matching.
    """

    def __init__(self,
                 *args,
                 consistency_angles=(30, 60, 90, 120, 150),
                 lambda_cls: float = 0.1,
                 lambda_box: float = 0.0,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.consistency_angles = tuple(float(a) for a in consistency_angles)
        self.lambda_cls = float(lambda_cls)
        self.lambda_box = float(lambda_box)

    def loss(self, batch_inputs: torch.Tensor, batch_data_samples):
        losses = super().loss(batch_inputs, batch_data_samples)
        angle = random.choice(self.consistency_angles)
        with torch.no_grad():
            teacher_feats = _global_pool_tuple(self.extract_feat(batch_inputs))
        student_inputs = _rotate_batch(batch_inputs, angle)
        student_feats = _global_pool_tuple(self.extract_feat(student_inputs))
        terms = []
        for student, teacher in zip(student_feats, teacher_feats):
            if student.shape == teacher.shape:
                terms.append(F.mse_loss(student, teacher.detach()))
        if terms:
            losses['loss_cv_cls'] = torch.stack(terms).mean() * self.lambda_cls
        else:
            losses['loss_cv_cls'] = batch_inputs.sum() * 0.0
        if self.lambda_box > 0:
            losses['loss_cv_box'] = batch_inputs.sum() * 0.0
        return losses
