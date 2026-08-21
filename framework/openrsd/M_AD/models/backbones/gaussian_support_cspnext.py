"""Gaussian semantic-support adapters for CSPNeXt backbones.

This module uses class-conditioned geometry priors as feature-routing tokens.
The Gaussian variables describe semantic scale/aspect support, not rotated box
overlap or Gaussian bbox regression distance.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from mmdet.models.backbones.cspnext import CSPNeXt
from mmdet.registry import MODELS as MMDET_MODELS
from mmrotate.registry import MODELS as MMROTATE_MODELS
from torch import Tensor, nn
from torch.nn import functional as F

from M_AD.models.utils.gaussian_semantic_scale import (
    load_class_geometry_priors,
)


def _as_float_tensor(value: Tensor | Sequence[Sequence[float]],
                     name: str) -> Tensor:
    tensor = value.detach().clone() if torch.is_tensor(value) else torch.tensor(
        value, dtype=torch.float32)
    tensor = tensor.to(dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[-1] != 2:
        raise ValueError(f'{name} must have shape (num_tokens, 2)')
    return tensor


class GaussianSupportTokenGate(nn.Module):
    """Feature-conditioned residual gate driven by Gaussian support tokens."""

    def __init__(
        self,
        in_channels: int,
        support_means: Tensor | Sequence[Sequence[float]],
        support_stds: Tensor | Sequence[Sequence[float]],
        valid_mask: Tensor | Sequence[bool] | None = None,
        token_dim: int = 64,
        temperature: float = 1.0,
        gamma_init: float = 0.0,
        residual_scale: float = 1.0,
        min_std: float = 1e-6,
    ) -> None:
        super().__init__()
        support_means = _as_float_tensor(support_means, 'support_means')
        support_stds = _as_float_tensor(support_stds, 'support_stds').clamp(
            min=float(min_std))
        if support_means.shape != support_stds.shape:
            raise ValueError('support_means and support_stds must match')
        if valid_mask is None:
            valid = torch.ones(support_means.shape[0], dtype=torch.bool)
        else:
            valid = valid_mask.detach().clone() if torch.is_tensor(
                valid_mask) else torch.tensor(valid_mask, dtype=torch.bool)
            valid = valid.to(dtype=torch.bool).reshape(-1)
            if valid.numel() != support_means.shape[0]:
                raise ValueError('valid_mask length must match support_means')

        self.in_channels = int(in_channels)
        self.token_dim = int(token_dim)
        self.temperature = max(float(temperature), 1e-6)
        self.residual_scale = float(residual_scale)
        self.register_buffer('support_means', support_means)
        self.register_buffer('support_stds', support_stds)
        self.register_buffer('support_valid_mask', valid)

        self.query_proj = nn.Conv2d(self.in_channels, self.token_dim, 1)
        self.prior_encoder = nn.Sequential(
            nn.Linear(4, self.token_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.context_to_gate = nn.Linear(self.token_dim, self.in_channels)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def _support_tokens(self) -> Tensor:
        prior = torch.cat(
            [self.support_means, self.support_stds.log()], dim=-1)
        tokens = self.prior_encoder(prior)
        return F.normalize(tokens, dim=-1)

    def forward(self, x: Tensor, return_debug: bool = False):
        if x.ndim != 4:
            raise ValueError('GaussianSupportTokenGate expects BCHW features')
        valid = self.support_valid_mask.to(device=x.device, dtype=torch.bool)
        if not torch.any(valid):
            if return_debug:
                b, _, h, w = x.shape
                empty = x.new_zeros((b, h * w, valid.numel()))
                return x, {'attention': empty, 'gamma': self.gamma.detach()}
            return x

        tokens = self._support_tokens().to(device=x.device, dtype=x.dtype)
        q = self.query_proj(x).flatten(2).transpose(1, 2)
        q = F.normalize(q, dim=-1)
        logits = q.matmul(tokens.transpose(0, 1))
        logits = logits / (math.sqrt(self.token_dim) * self.temperature)
        logits = logits.masked_fill(~valid[None, None, :], -torch.finfo(
            logits.dtype).max)
        attention = logits.softmax(dim=-1)
        context = attention.matmul(tokens)
        gate = torch.sigmoid(self.context_to_gate(context))
        gate = gate.transpose(1, 2).reshape_as(x)
        out = x + self.gamma * self.residual_scale * gate * x
        if return_debug:
            return out, {
                'attention': attention.detach(),
                'gamma': self.gamma.detach(),
            }
        return out


class GaussianSupportGlobalChannelAdapter(nn.Module):
    """Image-global Gaussian support adapter with normalized channel residuals.

    Unlike the P9A spatial token gate, this adapter pools each feature map before
    attending to support tokens. The resulting channel correction is applied to a
    normalized feature residual, keeping the initial behavior identity-safe and
    avoiding per-location score inflation.
    """

    def __init__(
        self,
        in_channels: int,
        support_means: Tensor | Sequence[Sequence[float]],
        support_stds: Tensor | Sequence[Sequence[float]],
        valid_mask: Tensor | Sequence[bool] | None = None,
        token_dim: int = 64,
        temperature: float = 1.0,
        gamma_init: float = 0.0,
        residual_scale: float = 1.0,
        min_std: float = 1e-6,
        zero_init_adapter: bool = True,
    ) -> None:
        super().__init__()
        support_means = _as_float_tensor(support_means, 'support_means')
        support_stds = _as_float_tensor(support_stds, 'support_stds').clamp(
            min=float(min_std))
        if support_means.shape != support_stds.shape:
            raise ValueError('support_means and support_stds must match')
        if valid_mask is None:
            valid = torch.ones(support_means.shape[0], dtype=torch.bool)
        else:
            valid = valid_mask.detach().clone() if torch.is_tensor(
                valid_mask) else torch.tensor(valid_mask, dtype=torch.bool)
            valid = valid.to(dtype=torch.bool).reshape(-1)
            if valid.numel() != support_means.shape[0]:
                raise ValueError('valid_mask length must match support_means')

        self.in_channels = int(in_channels)
        self.token_dim = int(token_dim)
        self.temperature = max(float(temperature), 1e-6)
        self.residual_scale = float(residual_scale)
        self.register_buffer('support_means', support_means)
        self.register_buffer('support_stds', support_stds)
        self.register_buffer('support_valid_mask', valid)

        self.query_proj = nn.Linear(self.in_channels, self.token_dim)
        self.prior_encoder = nn.Sequential(
            nn.Linear(4, self.token_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.norm = nn.GroupNorm(1, self.in_channels, affine=False)
        self.context_to_delta = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.token_dim, self.in_channels),
        )
        if zero_init_adapter:
            final = self.context_to_delta[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def _support_tokens(self) -> Tensor:
        prior = torch.cat(
            [self.support_means, self.support_stds.log()], dim=-1)
        tokens = self.prior_encoder(prior)
        return F.normalize(tokens, dim=-1)

    def forward(self, x: Tensor, return_debug: bool = False):
        if x.ndim != 4:
            raise ValueError(
                'GaussianSupportGlobalChannelAdapter expects BCHW features')
        valid = self.support_valid_mask.to(device=x.device, dtype=torch.bool)
        b = x.shape[0]
        if not torch.any(valid):
            if return_debug:
                empty = x.new_zeros((b, valid.numel()))
                delta = x.new_zeros((b, self.in_channels, 1, 1))
                return x, {
                    'attention': empty,
                    'channel_delta': delta,
                    'gamma': self.gamma.detach(),
                }
            return x

        tokens = self._support_tokens().to(device=x.device, dtype=x.dtype)
        pooled = x.mean(dim=(2, 3))
        q = F.normalize(self.query_proj(pooled), dim=-1)
        logits = q.matmul(tokens.transpose(0, 1))
        logits = logits / (math.sqrt(self.token_dim) * self.temperature)
        logits = logits.masked_fill(~valid[None, :], -torch.finfo(
            logits.dtype).max)
        attention = logits.softmax(dim=-1)
        context = attention.matmul(tokens)
        channel_delta = torch.tanh(self.context_to_delta(context))
        channel_delta = channel_delta.reshape(b, self.in_channels, 1, 1)
        residual = channel_delta * self.norm(x)
        out = x + self.gamma * self.residual_scale * residual
        if return_debug:
            return out, {
                'attention': attention.detach(),
                'channel_delta': channel_delta.detach(),
                'gamma': self.gamma.detach(),
            }
        return out


@MMDET_MODELS.register_module()
@MMROTATE_MODELS.register_module()
class GaussianSupportCSPNeXt(CSPNeXt):
    """CSPNeXt with Gaussian semantic-support token gates on output stages."""

    def __init__(
        self,
        *args,
        gaussian_support_adapter: dict | None = None,
        **kwargs,
    ) -> None:
        arch = kwargs.get('arch', args[0] if args else 'P5')
        widen_factor = kwargs.get('widen_factor', 1.0)
        arch_ovewrite = kwargs.get('arch_ovewrite', None)
        out_indices = kwargs.get('out_indices', (2, 3, 4))
        super().__init__(*args, **kwargs)
        self.gaussian_support_adapter_cfg = gaussian_support_adapter or {}
        self.gaussian_support_gates = nn.ModuleDict()
        if not self.gaussian_support_adapter_cfg.get('enable', True):
            return

        means, stds, valid = self._load_support_priors(
            self.gaussian_support_adapter_cfg)
        stage_channels = self._stage_channels(
            arch=arch,
            widen_factor=widen_factor,
            arch_ovewrite=arch_ovewrite,
        )
        stages = tuple(self.gaussian_support_adapter_cfg.get(
            'stages', out_indices))
        adapter_type = self.gaussian_support_adapter_cfg.get(
            'adapter_type', 'spatial_gate')
        for stage_idx in stages:
            if stage_idx not in out_indices:
                continue
            common_kwargs = dict(
                in_channels=stage_channels[int(stage_idx)],
                support_means=means,
                support_stds=stds,
                valid_mask=valid,
                token_dim=int(self.gaussian_support_adapter_cfg.get(
                    'token_dim', 64)),
                temperature=float(self.gaussian_support_adapter_cfg.get(
                    'temperature', 1.0)),
                gamma_init=float(self.gaussian_support_adapter_cfg.get(
                    'gamma_init', 0.0)),
                residual_scale=float(self.gaussian_support_adapter_cfg.get(
                    'residual_scale', 1.0)),
            )
            if adapter_type == 'spatial_gate':
                adapter = GaussianSupportTokenGate(**common_kwargs)
            elif adapter_type == 'global_channel':
                adapter = GaussianSupportGlobalChannelAdapter(
                    **common_kwargs,
                    zero_init_adapter=bool(
                        self.gaussian_support_adapter_cfg.get(
                            'zero_init_adapter', True)))
            else:
                raise ValueError(
                    f'Unsupported gaussian support adapter_type: '
                    f'{adapter_type}')
            self.gaussian_support_gates[str(stage_idx)] = adapter

    def _load_support_priors(self,
                             cfg: dict) -> tuple[Tensor, Tensor, Tensor]:
        if 'support_means' in cfg and 'support_stds' in cfg:
            means = _as_float_tensor(cfg['support_means'], 'support_means')
            stds = _as_float_tensor(cfg['support_stds'],
                                    'support_stds').clamp(min=1e-6)
            valid = cfg.get('valid_mask', None)
            if valid is None:
                valid = torch.ones(means.shape[0], dtype=torch.bool)
            else:
                valid = torch.tensor(valid, dtype=torch.bool).reshape(-1)
            return means, stds, valid

        priors_csv = cfg.get('class_geometry_priors_csv')
        class_names = cfg.get('class_names')
        num_classes = cfg.get('num_classes')
        if priors_csv is None or class_names is None or num_classes is None:
            raise ValueError(
                'gaussian_support_adapter needs either support_means/'
                'support_stds or class_geometry_priors_csv, class_names, '
                'and num_classes')
        means, stds, valid, _ = load_class_geometry_priors(
            priors_csv,
            class_names=class_names,
            num_classes=int(num_classes),
        )
        if cfg.get('shuffle_priors', False):
            perm = torch.arange(means.shape[0] - 1, -1, -1)
            means = means[perm]
            stds = stds[perm]
            valid = valid[perm]
        return means, stds, valid

    def _stage_channels(self, arch: str, widen_factor: float,
                        arch_ovewrite: dict | None) -> dict[int, int]:
        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        channels = {0: int(arch_setting[0][0] * widen_factor)}
        for i, (_, out_channels, _, _, _) in enumerate(arch_setting):
            channels[i + 1] = int(out_channels * widen_factor)
        return channels

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                if str(i) in self.gaussian_support_gates:
                    x = self.gaussian_support_gates[str(i)](x)
                outs.append(x)
        return tuple(outs)
