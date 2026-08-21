# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.roi_heads.bbox_heads import Shared2FCBBoxHead
from mmdet.registry import MODELS as MMDET_MODELS
from torch import Tensor

from mmrotate.registry import MODELS as MMROTATE_MODELS


class FAA(nn.Module):
    """Fourier angle alignment for 7x7 single-channel RoI features."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

        h, w = 7, 7
        h_idx = torch.arange(h)
        w_idx = torch.arange(w // 2 + 1)
        h_shift = torch.fft.fftshift(h_idx - h // 2, dim=0)
        w_shift = torch.cat(
            [w_idx[:w // 2], torch.tensor([-w // 2], dtype=w_idx.dtype)])
        h_grid, w_grid = torch.meshgrid(h_shift, w_shift, indexing='ij')

        rho = torch.sqrt(h_grid**2 + w_grid**2)
        theta = torch.atan2(h_grid, w_grid)
        theta = (theta + 2 * math.pi) % (2 * math.pi)
        mask = rho > self.eps

        self.register_buffer('valid_thetas', theta[mask])
        self.register_buffer('valid_rhos', rho[mask])
        self.register_buffer('mask_flat', mask.reshape(-1))
        self.m = len(self.valid_thetas)

        ys = torch.linspace(-1, 1, h)
        xs = torch.linspace(-1, 1, w)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        self.register_buffer('base_grid', torch.stack([grid_x, grid_y], -1))

    def _rotate_images(self, x: Tensor, theta: Tensor) -> Tensor:
        batch, _, h, w = x.shape
        cos_t = torch.cos(-theta)
        sin_t = torch.sin(-theta)
        rot_mat = torch.stack(
            [
                torch.stack([cos_t, -sin_t], dim=-1),
                torch.stack([sin_t, cos_t], dim=-1)
            ],
            dim=1)

        grid = self.base_grid.to(dtype=x.dtype).unsqueeze(0).expand(
            batch, -1, -1, -1)
        grid = grid.reshape(batch, h * w, 2)
        rotated_grid = torch.bmm(grid, rot_mat).reshape(batch, h, w, 2)
        return F.grid_sample(
            x,
            rotated_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, h, w = x.shape
        assert channels == 1 and h == 7 and w == 7, (
            f'Expected [N, 1, 7, 7], got {tuple(x.shape)}.')

        x_rfft = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        magnitude = x_rfft.abs() + self.eps
        mag_valid = magnitude.reshape(batch, -1)[:, self.mask_flat]

        weighted_energy = mag_valid * self.valid_rhos.to(mag_valid.device)
        max_idx = torch.argmax(weighted_energy, dim=1)
        theta = self.valid_thetas.to(mag_valid.device)[max_idx]
        return self._rotate_images(x, theta)


@MMDET_MODELS.register_module()
@MMROTATE_MODELS.register_module()
class FAAHead(Shared2FCBBoxHead):
    """Shared 2FC bbox head with Fourier-aligned RoI features.

    The aligned RoI feature is added to the original feature with a learnable
    small scale, matching the official Fourier-Angle-Alignment behavior while
    staying compatible with MMRotate 1.x box coders and predictors.
    """

    def __init__(self, *args, gamma_init: float = 1e-5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.faa = FAA()
        self.gamma = nn.Parameter(torch.ones(1) * gamma_init)

        if self.num_shared_fcs > 0:
            first_in_features = self.in_channels * self.roi_feat_area
            hidden_channels = self.fc_out_channels + self.in_channels
            rebuilt_fcs = nn.ModuleList()
            for i in range(self.num_shared_fcs):
                in_features = first_in_features if i == 0 else hidden_channels
                out_features = (
                    hidden_channels if i == 0 else self.fc_out_channels)
                rebuilt_fcs.append(nn.Linear(in_features, out_features))
            self.shared_fcs = rebuilt_fcs

    def forward(self, x: Tensor) -> tuple:
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.with_avg_pool:
            x = self.avg_pool(x)

        num_rois, channels, height, width = x.shape
        spatial_feat = x.flatten(1)

        if num_rois > 0:
            aligned_feat = self.faa(
                x.reshape(num_rois * channels, 1, height, width)).reshape(
                    num_rois, channels, height, width).flatten(1)
            x = spatial_feat + self.gamma * aligned_feat
        else:
            x = spatial_feat

        for fc in self.shared_fcs:
            x = self.relu(fc(x))

        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred
