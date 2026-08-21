# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.necks import FPN
from mmdet.registry import MODELS as MMDET_MODELS
from torch import Tensor

from mmrotate.registry import MODELS as MMROTATE_MODELS


class FAAFusion(nn.Module):
    """Fourier angle alignment module for one top-down FPN fusion step.

    This ports the FAAFusion idea from Fourier-Angle-Alignment to the
    MMRotate 1.x registry stack. It estimates dominant local directions from
    low-level and upsampled high-level features, rotates high-level local
    patches to the low-level direction, then fuses the aligned feature back.
    """

    def __init__(self,
                 channels: int = 256,
                 m: int = 7,
                 c_mid: int = 64,
                 eps: float = 1e-8,
                 layer_scale_init_value: float = 1e-5,
                 max_channels_per_chunk: int = 8) -> None:
        super().__init__()
        assert m % 2 == 1, 'Local window size m must be odd.'
        self.channels = channels
        self.m = m
        self.c_mid = c_mid
        self.eps = eps
        self.max_channels_per_chunk = max(1, max_channels_per_chunk)

        self.layer_scale = nn.Parameter(
            torch.full((1, 1, 1, 1), layer_scale_init_value))
        self.proj_low = nn.Conv2d(channels, c_mid, kernel_size=1, bias=False)
        self.proj_high = nn.Conv2d(channels, c_mid, kernel_size=1, bias=False)
        self.recon = nn.Conv2d(c_mid, channels, kernel_size=1, bias=False)

        self._init_freq_grids(m)

    def _init_freq_grids(self, m: int) -> None:
        freq = torch.fft.fftfreq(m, d=1.0) * m
        freq = torch.fft.fftshift(freq)
        h_grid, w_grid = torch.meshgrid(freq, freq, indexing='ij')

        rho = torch.sqrt(h_grid**2 + w_grid**2)
        theta = torch.atan2(h_grid, w_grid)
        theta = (theta + 2 * math.pi) % (2 * math.pi)

        mask = rho > self.eps
        self.register_buffer('valid_thetas', theta[mask])
        self.register_buffer('valid_rhos', rho[mask])
        self.register_buffer('mask_flat', mask.reshape(-1))

    def _estimate_main_direction(self, x_local: Tensor) -> Tensor:
        """Estimate dominant orientation from local magnitude spectra."""
        batch = x_local.shape[0]
        x_fft = torch.fft.fft2(x_local.squeeze(1), norm='ortho')
        mag = torch.fft.fftshift(x_fft, dim=(-2, -1)).abs() + self.eps

        mag_valid = mag.reshape(batch, -1)[:, self.mask_flat]
        weighted_energy = mag_valid * self.valid_rhos.to(mag_valid.device)
        max_idx = torch.argmax(weighted_energy, dim=1)
        return self.valid_thetas.to(mag_valid.device)[max_idx]

    def _rotate_spatial_patch(self, patch: Tensor, theta: Tensor) -> Tensor:
        batch, _, m, _ = patch.shape
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        center = (m - 1) / 2.0

        rot_mat = torch.zeros(batch, 2, 3, device=patch.device,
                              dtype=patch.dtype)
        rot_mat[:, 0, 0] = cos_t
        rot_mat[:, 0, 1] = -sin_t
        rot_mat[:, 1, 0] = sin_t
        rot_mat[:, 1, 1] = cos_t
        rot_mat[:, 0, 2] = center - cos_t * center + sin_t * center
        rot_mat[:, 1, 2] = center - sin_t * center - cos_t * center

        grid = F.affine_grid(rot_mat, patch.size(), align_corners=False)
        return F.grid_sample(
            patch,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)

    def forward(self, x_high: Tensor, x_low: Tensor) -> Tensor:
        batch, channels, h_low, w_low = x_low.shape
        assert channels == self.channels

        if x_high.shape[2:] != x_low.shape[2:]:
            x_high_up = F.interpolate(
                x_high, size=(h_low, w_low), mode='bilinear',
                align_corners=False)
        else:
            x_high_up = x_high

        x_low_mid = self.proj_low(x_low)
        x_high_mid = self.proj_high(x_high_up)

        patch_count = (h_low - self.m + 1) * (w_low - self.m + 1)
        x_high_aligned_mid = torch.zeros_like(x_high_mid)

        ones = x_low.new_ones(1, 1, h_low, w_low)
        ones_unfold = F.unfold(ones, kernel_size=self.m, stride=1, padding=0)
        ones_fold = F.fold(
            ones_unfold,
            output_size=(h_low, w_low),
            kernel_size=self.m,
            stride=1,
            padding=0)

        for channel_idx in range(0, self.c_mid, self.max_channels_per_chunk):
            end_channel = min(
                channel_idx + self.max_channels_per_chunk, self.c_mid)
            chunk_channels = end_channel - channel_idx
            x_low_c = x_low_mid[:, channel_idx:end_channel]
            x_high_c = x_high_mid[:, channel_idx:end_channel]

            low_unfold = F.unfold(
                x_low_c, kernel_size=self.m, stride=1, padding=0)
            high_unfold = F.unfold(
                x_high_c, kernel_size=self.m, stride=1, padding=0)

            low_patches = low_unfold.reshape(
                batch, chunk_channels, self.m * self.m,
                patch_count).permute(0, 1, 3, 2).reshape(
                    batch * chunk_channels * patch_count, 1, self.m, self.m)
            high_patches = high_unfold.reshape(
                batch, chunk_channels, self.m * self.m,
                patch_count).permute(0, 1, 3, 2).reshape(
                    batch * chunk_channels * patch_count, 1, self.m, self.m)

            theta_low = torch.remainder(
                self._estimate_main_direction(low_patches), math.pi)
            theta_high = torch.remainder(
                self._estimate_main_direction(high_patches), math.pi)
            theta_delta = theta_low - theta_high

            high_rotated = self._rotate_spatial_patch(
                high_patches, theta_delta)
            high_rotated_flat = high_rotated.reshape(
                batch, chunk_channels, patch_count,
                self.m * self.m).permute(0, 1, 3, 2).reshape(
                    batch, chunk_channels * self.m * self.m, patch_count)
            high_aligned = F.fold(
                high_rotated_flat,
                output_size=(h_low, w_low),
                kernel_size=self.m,
                stride=1,
                padding=0)
            x_high_aligned_mid[:, channel_idx:end_channel] = (
                high_aligned / (ones_fold + self.eps))

        x_high_aligned = self.recon(x_high_aligned_mid)
        return x_low + x_high_up + self.layer_scale * x_high_aligned


@MMDET_MODELS.register_module()
@MMROTATE_MODELS.register_module()
class FAAFusionFPN(FPN):
    """FPN variant with optional FAAFusion in top-down fusion steps."""

    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 num_outs: int,
                 fusion_modes: List[str],
                 start_level: int = 0,
                 end_level: int = -1,
                 add_extra_convs=False,
                 relu_before_extra_convs: bool = False,
                 no_norm_on_lateral: bool = False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier',
                     layer='Conv2d',
                     distribution='uniform'),
                 fam_cfg=dict(m=7, c_mid=64)) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            start_level=start_level,
            end_level=end_level,
            add_extra_convs=add_extra_convs,
            relu_before_extra_convs=relu_before_extra_convs,
            no_norm_on_lateral=no_norm_on_lateral,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            upsample_cfg=upsample_cfg,
            init_cfg=init_cfg)

        backbone_levels = self.backbone_end_level - self.start_level
        expected_steps = backbone_levels - 1
        assert len(fusion_modes) == expected_steps, (
            f'fusion_modes length ({len(fusion_modes)}) must be '
            f'{expected_steps}.')
        for mode in fusion_modes:
            assert mode in ('add', 'faa'), f'Invalid fusion mode: {mode}'

        self.fusion_modes = fusion_modes
        self.faa_modules = nn.ModuleList()
        for mode in fusion_modes:
            if mode == 'faa':
                self.faa_modules.append(
                    FAAFusion(channels=out_channels, **fam_cfg))
            else:
                self.faa_modules.append(nn.Identity())

    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        assert len(inputs) == len(self.in_channels)

        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            fusion_idx = used_backbone_levels - 1 - i
            mode = self.fusion_modes[fusion_idx]
            if mode == 'add':
                if 'scale_factor' in self.upsample_cfg:
                    upsampled = F.interpolate(laterals[i],
                                              **self.upsample_cfg)
                else:
                    prev_shape = laterals[i - 1].shape[2:]
                    upsampled = F.interpolate(
                        laterals[i], size=prev_shape, **self.upsample_cfg)
                laterals[i - 1] = laterals[i - 1] + upsampled
            else:
                laterals[i - 1] = self.faa_modules[fusion_idx](
                    laterals[i], laterals[i - 1])

        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for _ in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
