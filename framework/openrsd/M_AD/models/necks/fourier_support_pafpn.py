"""Fourier semantic-support adapters for CSPNeXt PAFPN necks.

These adapters inject frequency/orientation evidence into neck features.  They
do not model rotated boxes as Gaussians and do not compute IoU, GWD, or KLD.
The Gaussian option gates semantic support by Fourier reliability instead of
using Gaussian bbox distances.
"""

from __future__ import annotations

from typing import Sequence

import torch
from mmdet.models.necks.cspnext_pafpn import CSPNeXtPAFPN
from mmdet.registry import MODELS as MMDET_MODELS
from mmrotate.registry import MODELS as MMROTATE_MODELS
from torch import Tensor, nn
from torch.nn import functional as F

from M_AD.models.backbones.gaussian_support_cspnext import (
    GaussianSupportGlobalChannelAdapter,
)
from M_AD.models.necks.gaussian_support_pafpn import _load_support_priors


def _masked_mean(values: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    mask = mask.to(device=values.device, dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(0)
    denom = mask.sum(dim=(-2, -1)).clamp_min(eps)
    return (values * mask).sum(dim=(-2, -1)) / denom


class FourierSupportChannelAdapter(nn.Module):
    """Bounded channel residual driven by global Fourier descriptors."""

    descriptor_dim = 8
    mess_directional_dim = 4

    def __init__(
        self,
        in_channels: int,
        fft_size: int = 16,
        hidden_dim: int = 64,
        gamma_init: float = 0.0,
        residual_scale: float = 1.0,
        zero_init_adapter: bool = True,
        max_delta_norm_ratio: float = 0.0,
        include_mess_directional: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.fft_size = int(fft_size)
        self.hidden_dim = int(hidden_dim)
        self.residual_scale = float(residual_scale)
        self.max_delta_norm_ratio = float(max_delta_norm_ratio)
        self.include_mess_directional = bool(include_mess_directional)
        self.eps = float(eps)

        descriptor_dim = self.descriptor_dim
        if self.include_mess_directional:
            descriptor_dim += self.mess_directional_dim
        self.total_descriptor_dim = descriptor_dim

        self.norm = nn.GroupNorm(1, self.in_channels, affine=False)
        self.context_to_delta = nn.Sequential(
            nn.Linear(self.in_channels + descriptor_dim, self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, self.in_channels),
        )
        if zero_init_adapter:
            final = self.context_to_delta[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def _spatial_summary(self, x: Tensor) -> Tensor:
        summary = x.mean(dim=1, keepdim=True)
        if self.fft_size > 0 and summary.shape[-2:] != (
                self.fft_size, self.fft_size):
            summary = F.adaptive_avg_pool2d(
                summary, (self.fft_size, self.fft_size))
        summary = summary.float()
        summary = summary - summary.mean(dim=(-2, -1), keepdim=True)
        scale = summary.flatten(2).std(dim=-1, keepdim=True).view(
            summary.shape[0], 1, 1, 1).clamp_min(self.eps)
        return summary / scale

    def fourier_descriptor(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError('FourierSupportChannelAdapter expects BCHW')
        spatial = self._spatial_summary(x)
        _, _, h, w = spatial.shape
        spec = torch.fft.rfft2(spatial, norm='ortho')
        power = spec.abs().pow(2).squeeze(1)

        fy = torch.fft.fftfreq(
            h, device=x.device, dtype=spatial.dtype).view(h, 1)
        fx = torch.fft.rfftfreq(
            w, device=x.device, dtype=spatial.dtype).view(1, w // 2 + 1)
        radius = torch.sqrt(fx.pow(2) + fy.pow(2))
        valid = radius > 0
        low = (radius <= 0.18) & valid
        mid = (radius > 0.18) & (radius <= 0.35)
        high = radius > 0.35

        abs_x = fx.abs()
        abs_y = fy.abs()
        orient_denom = (abs_x + abs_y).clamp_min(self.eps)
        horizontal = abs_x / orient_denom
        vertical = abs_y / orient_denom
        diagonal = 1.0 - (abs_x - abs_y).abs() / orient_denom

        low_energy = _masked_mean(power, low, self.eps)
        mid_energy = _masked_mean(power, mid, self.eps)
        high_energy = _masked_mean(power, high, self.eps)
        horizontal_energy = _masked_mean(power * horizontal, valid, self.eps)
        vertical_energy = _masked_mean(power * vertical, valid, self.eps)
        diagonal_energy = _masked_mean(power * diagonal, valid, self.eps)
        orient_sum = (horizontal_energy + vertical_energy).clamp_min(self.eps)
        anisotropy = (horizontal_energy - vertical_energy).abs() / orient_sum
        band_sum = (low_energy + mid_energy + high_energy).clamp_min(self.eps)
        high_ratio = high_energy / band_sum

        descriptor = torch.stack(
            [
                torch.log1p(low_energy),
                torch.log1p(mid_energy),
                torch.log1p(high_energy),
                torch.log1p(horizontal_energy),
                torch.log1p(vertical_energy),
                torch.log1p(diagonal_energy),
                anisotropy,
                high_ratio,
            ],
            dim=-1,
        )
        return descriptor.to(dtype=x.dtype)

    def mess_directional_descriptor(self, x: Tensor) -> Tensor:
        spatial = self._spatial_summary(x)
        dx = (spatial[..., :, 1:] - spatial[..., :, :-1]).abs().mean(
            dim=(-3, -2, -1))
        dy = (spatial[..., 1:, :] - spatial[..., :-1, :]).abs().mean(
            dim=(-3, -2, -1))
        d1 = (spatial[..., 1:, 1:] - spatial[..., :-1, :-1]).abs().mean(
            dim=(-3, -2, -1))
        d2 = (spatial[..., 1:, :-1] - spatial[..., :-1, 1:]).abs().mean(
            dim=(-3, -2, -1))
        descriptor = torch.stack([dx, dy, d1, d2], dim=-1)
        return torch.log1p(descriptor).to(dtype=x.dtype)

    def support_descriptor(self, x: Tensor) -> Tensor:
        descriptor = self.fourier_descriptor(x)
        if self.include_mess_directional:
            descriptor = torch.cat(
                [descriptor, self.mess_directional_descriptor(x)], dim=-1)
        return descriptor

    def _bound_delta(self, x: Tensor, delta: Tensor) -> Tensor:
        if self.max_delta_norm_ratio <= 0:
            return delta
        delta_norm = delta.flatten(1).norm(dim=1).clamp_min(self.eps)
        max_norm = (
            self.max_delta_norm_ratio *
            x.detach().flatten(1).norm(dim=1).clamp_min(self.eps))
        ratio = torch.clamp(max_norm / delta_norm, max=1.0)
        return delta * ratio.view(-1, 1, 1, 1)

    def forward(self, x: Tensor, return_debug: bool = False):
        if x.ndim != 4:
            raise ValueError('FourierSupportChannelAdapter expects BCHW')
        descriptor = self.support_descriptor(x)
        pooled = x.mean(dim=(2, 3))
        context = torch.cat([pooled, descriptor], dim=-1)
        channel_delta = torch.tanh(self.context_to_delta(context)).reshape(
            x.shape[0], self.in_channels, 1, 1)
        residual = channel_delta * self.norm(x)
        delta = self.gamma * self.residual_scale * residual
        delta = self._bound_delta(x, delta)
        out = x + delta
        if return_debug:
            return out, {
                'fourier_descriptor': descriptor.detach(),
                'channel_delta': channel_delta.detach(),
                'gamma': self.gamma.detach(),
            }
        return out


class FourierGatedGaussianChannelAdapter(nn.Module):
    """Gaussian support residual gated by Fourier feature reliability."""

    def __init__(
        self,
        in_channels: int,
        gaussian_support_cfg: dict,
        fft_size: int = 16,
        hidden_dim: int = 64,
        confidence_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        gaussian_cfg = dict(gaussian_support_cfg)
        means, stds, valid = _load_support_priors(gaussian_cfg)
        self.gaussian = GaussianSupportGlobalChannelAdapter(
            in_channels=int(in_channels),
            support_means=means,
            support_stds=stds,
            valid_mask=valid,
            token_dim=int(gaussian_cfg.get('token_dim', 64)),
            temperature=float(gaussian_cfg.get('temperature', 1.0)),
            gamma_init=float(gaussian_cfg.get('gamma_init', 0.0)),
            residual_scale=float(gaussian_cfg.get('residual_scale', 1.0)),
            zero_init_adapter=bool(
                gaussian_cfg.get('zero_init_adapter', True)),
        )
        self.fourier = FourierSupportChannelAdapter(
            in_channels=int(in_channels),
            fft_size=fft_size,
            hidden_dim=hidden_dim,
            gamma_init=0.0,
            zero_init_adapter=True,
        )
        self.confidence_temperature = max(
            float(confidence_temperature), 1e-6)
        self.confidence_proj = nn.Sequential(
            nn.Linear(self.fourier.descriptor_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor, return_debug: bool = False):
        gaussian_out, gaussian_debug = self.gaussian(x, return_debug=True)
        descriptor = self.fourier.fourier_descriptor(x)
        confidence = torch.sigmoid(
            self.confidence_proj(descriptor) /
            self.confidence_temperature).reshape(x.shape[0], 1, 1, 1)
        out = x + confidence * (gaussian_out - x)
        if return_debug:
            return out, {
                'fourier_descriptor': descriptor.detach(),
                'fourier_confidence': confidence.detach(),
                'gaussian': gaussian_debug,
            }
        return out


class MessFourierDualChannelAdapter(FourierSupportChannelAdapter):
    """Fourier adapter extended with MessDet-style directional moments."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs['include_mess_directional'] = True
        super().__init__(*args, **kwargs)


@MMDET_MODELS.register_module()
@MMROTATE_MODELS.register_module()
class FourierSupportCSPNeXtPAFPN(CSPNeXtPAFPN):
    """CSPNeXtPAFPN with post-fusion Fourier semantic-support adapters."""

    def __init__(
        self,
        *args,
        fourier_support_fusion: dict | None = None,
        **kwargs,
    ) -> None:
        out_channels = kwargs.get('out_channels', args[1] if len(args) > 1
                                  else None)
        super().__init__(*args, **kwargs)
        self.fourier_support_fusion_cfg = fourier_support_fusion or {}
        self.fourier_support_fusion = None
        if not self.fourier_support_fusion_cfg.get('enable', True):
            return
        if out_channels is None:
            raise ValueError('out_channels is required for Fourier fusion')

        enabled_levels = self.fourier_support_fusion_cfg.get(
            'enabled_levels', None)
        if enabled_levels is None:
            self.fourier_support_enabled_levels = set(
                range(len(self.in_channels)))
        else:
            self.fourier_support_enabled_levels = {
                int(level) for level in enabled_levels
            }
            invalid = [
                level for level in self.fourier_support_enabled_levels
                if level < 0 or level >= len(self.in_channels)
            ]
            if invalid:
                raise ValueError(
                    f'Invalid fourier_support_fusion enabled_levels: '
                    f'{invalid}')

        mode = self.fourier_support_fusion_cfg.get(
            'mode', 'fourier_channel')
        self.fourier_support_fusion = nn.ModuleList([
            self._build_adapter(int(out_channels), mode)
            for _ in range(len(self.in_channels))
        ])

    def _fourier_kwargs(self) -> dict:
        cfg = self.fourier_support_fusion_cfg
        return dict(
            fft_size=int(cfg.get('fft_size', 16)),
            hidden_dim=int(cfg.get('hidden_dim', 64)),
            gamma_init=float(cfg.get('gamma_init', 0.0)),
            residual_scale=float(cfg.get('residual_scale', 1.0)),
            zero_init_adapter=bool(cfg.get('zero_init_adapter', True)),
            max_delta_norm_ratio=float(
                cfg.get('max_delta_norm_ratio', 0.0)),
        )

    def _build_adapter(self, out_channels: int, mode: str) -> nn.Module:
        if mode == 'fourier_channel':
            return FourierSupportChannelAdapter(
                in_channels=out_channels,
                **self._fourier_kwargs(),
            )
        if mode == 'mess_fourier_dual':
            return MessFourierDualChannelAdapter(
                in_channels=out_channels,
                **self._fourier_kwargs(),
            )
        if mode == 'fourier_gated_gaussian':
            gaussian_cfg = self.fourier_support_fusion_cfg.get(
                'gaussian_support_fusion')
            if gaussian_cfg is None:
                raise ValueError(
                    'fourier_gated_gaussian needs gaussian_support_fusion')
            return FourierGatedGaussianChannelAdapter(
                in_channels=out_channels,
                gaussian_support_cfg=gaussian_cfg,
                fft_size=int(self.fourier_support_fusion_cfg.get(
                    'fft_size', 16)),
                hidden_dim=int(self.fourier_support_fusion_cfg.get(
                    'hidden_dim', 64)),
                confidence_temperature=float(
                    self.fourier_support_fusion_cfg.get(
                        'confidence_temperature', 1.0)),
            )
        raise ValueError(f'Unsupported fourier_support_fusion mode: {mode}')

    def forward(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        outs = super().forward(inputs)
        fusion = self.fourier_support_fusion
        if fusion is None:
            return outs
        return tuple(
            adapter(out) if idx in self.fourier_support_enabled_levels else out
            for idx, (adapter, out) in enumerate(zip(fusion, outs)))
