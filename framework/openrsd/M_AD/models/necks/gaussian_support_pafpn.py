"""Gaussian semantic-support fusion adapters for CSPNeXt PAFPN.

The Gaussian variables describe class-level semantic geometry support:
``[log(area), log(long_side / short_side)]``.  They are used to modulate
feature fusion channels, not to compute bbox overlap or GWD/KLD losses.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from mmdet.models.necks.cspnext_pafpn import CSPNeXtPAFPN
from mmdet.registry import MODELS as MMDET_MODELS
from mmrotate.registry import MODELS as MMROTATE_MODELS
from torch import Tensor, nn
from torch.nn import functional as F

from M_AD.models.backbones.gaussian_support_cspnext import (
    GaussianSupportGlobalChannelAdapter,
    _as_float_tensor,
)
from M_AD.models.utils.gaussian_semantic_scale import (
    load_class_geometry_priors,
)


def _load_support_priors(cfg: dict) -> tuple[Tensor, Tensor, Tensor]:
    if 'support_means' in cfg and 'support_stds' in cfg:
        means = _as_float_tensor(cfg['support_means'], 'support_means')
        stds = _as_float_tensor(cfg['support_stds'],
                                'support_stds').clamp(min=1e-6)
        valid = cfg.get('valid_mask')
        if valid is None:
            valid = torch.ones(means.shape[0], dtype=torch.bool)
        else:
            valid = torch.tensor(valid, dtype=torch.bool).reshape(-1)
            if valid.numel() != means.shape[0]:
                raise ValueError('valid_mask length must match support_means')
        return means, stds, valid

    priors_csv = cfg.get('class_geometry_priors_csv')
    class_names = cfg.get('class_names')
    num_classes = cfg.get('num_classes')
    if priors_csv is None or class_names is None or num_classes is None:
        raise ValueError(
            'gaussian_support_fusion needs either support_means/support_stds '
            'or class_geometry_priors_csv, class_names, and num_classes')
    means, stds, valid, _ = load_class_geometry_priors(
        priors_csv,
        class_names=class_names,
        num_classes=int(num_classes),
    )
    return means, stds, valid


class GaussianSupportCrossLevelChannelFusion(nn.Module):
    """Cross-level channel fusion driven by Gaussian support tokens.

    Each FPN output level contributes a pooled descriptor.  The query for one
    level is mixed with the global level summary before attending to fixed
    class-geometry Gaussian tokens.  The resulting channel residual is applied
    to normalized features, so zero-initialized variants are exact identity
    mappings.
    """

    def __init__(
        self,
        out_channels: int,
        num_levels: int,
        support_means: Tensor | Sequence[Sequence[float]],
        support_stds: Tensor | Sequence[Sequence[float]],
        valid_mask: Tensor | Sequence[bool] | None = None,
        token_dim: int = 64,
        temperature: float = 1.0,
        gamma_init: float = 0.0,
        residual_scale: float = 1.0,
        level_context_scale: float = 1.0,
        min_std: float = 1e-6,
        zero_init_adapter: bool = True,
    ) -> None:
        super().__init__()
        support_means = _as_float_tensor(support_means, 'support_means')
        support_stds = _as_float_tensor(support_stds,
                                        'support_stds').clamp(
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

        self.out_channels = int(out_channels)
        self.num_levels = int(num_levels)
        self.token_dim = int(token_dim)
        self.temperature = max(float(temperature), 1e-6)
        self.residual_scale = float(residual_scale)
        self.level_context_scale = float(level_context_scale)
        self.register_buffer('support_means', support_means)
        self.register_buffer('support_stds', support_stds)
        self.register_buffer('support_valid_mask', valid)

        self.query_proj = nn.Linear(self.out_channels, self.token_dim)
        self.prior_encoder = nn.Sequential(
            nn.Linear(4, self.token_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.context_to_delta = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.token_dim, self.out_channels),
        )
        self.norms = nn.ModuleList(
            nn.GroupNorm(1, self.out_channels, affine=False)
            for _ in range(self.num_levels))
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

    def forward(self,
                feats: Sequence[Tensor],
                return_debug: bool = False):
        if len(feats) != self.num_levels:
            raise ValueError(
                f'expected {self.num_levels} levels, got {len(feats)}')
        if any(feat.ndim != 4 for feat in feats):
            raise ValueError(
                'GaussianSupportCrossLevelChannelFusion expects BCHW levels')
        if any(feat.shape[1] != self.out_channels for feat in feats):
            raise ValueError('all levels must use out_channels')

        valid = self.support_valid_mask.to(
            device=feats[0].device, dtype=torch.bool)
        b = feats[0].shape[0]
        if not torch.any(valid):
            if return_debug:
                attention = feats[0].new_zeros(
                    (b, self.num_levels, valid.numel()))
                delta = feats[0].new_zeros(
                    (b, self.num_levels, self.out_channels, 1, 1))
                return tuple(feats), {
                    'attention': attention,
                    'channel_delta': delta,
                    'gamma': self.gamma.detach(),
                }
            return tuple(feats)

        pooled = torch.stack(
            [feat.mean(dim=(2, 3)) for feat in feats], dim=1)
        global_summary = pooled.mean(dim=1, keepdim=True)
        query_input = pooled + self.level_context_scale * (
            global_summary - pooled)

        tokens = self._support_tokens().to(
            device=feats[0].device, dtype=feats[0].dtype)
        q = self.query_proj(query_input.reshape(b * self.num_levels, -1))
        q = F.normalize(q, dim=-1).reshape(b, self.num_levels, -1)
        logits = q.matmul(tokens.transpose(0, 1))
        logits = logits / (math.sqrt(self.token_dim) * self.temperature)
        logits = logits.masked_fill(~valid[None, None, :], -torch.finfo(
            logits.dtype).max)
        attention = logits.softmax(dim=-1)
        context = attention.matmul(tokens)
        channel_delta = torch.tanh(
            self.context_to_delta(context.reshape(b * self.num_levels, -1)))
        channel_delta = channel_delta.reshape(
            b, self.num_levels, self.out_channels, 1, 1)

        outs = []
        for level, feat in enumerate(feats):
            residual = channel_delta[:, level] * self.norms[level](feat)
            outs.append(feat + self.gamma * self.residual_scale * residual)
        outs = tuple(outs)
        if return_debug:
            return outs, {
                'attention': attention.detach(),
                'channel_delta': channel_delta.detach(),
                'gamma': self.gamma.detach(),
            }
        return outs


@MMDET_MODELS.register_module()
@MMROTATE_MODELS.register_module()
class GaussianSupportCSPNeXtPAFPN(CSPNeXtPAFPN):
    """CSPNeXtPAFPN with post-fusion Gaussian semantic-support adapters."""

    def __init__(
        self,
        *args,
        gaussian_support_fusion: dict | None = None,
        **kwargs,
    ) -> None:
        out_channels = kwargs.get('out_channels', args[1] if len(args) > 1
                                  else None)
        super().__init__(*args, **kwargs)
        self.gaussian_support_fusion_cfg = gaussian_support_fusion or {}
        self.gaussian_support_fusion = None
        if not self.gaussian_support_fusion_cfg.get('enable', True):
            return
        if out_channels is None:
            raise ValueError('out_channels is required for Gaussian fusion')

        means, stds, valid = _load_support_priors(
            self.gaussian_support_fusion_cfg)
        mode = self.gaussian_support_fusion_cfg.get(
            'mode', 'per_level_channel')
        enabled_levels = self.gaussian_support_fusion_cfg.get(
            'enabled_levels', None)
        if enabled_levels is None:
            self.gaussian_support_enabled_levels = set(
                range(len(self.in_channels)))
        else:
            self.gaussian_support_enabled_levels = {
                int(level) for level in enabled_levels
            }
            invalid = [
                level for level in self.gaussian_support_enabled_levels
                if level < 0 or level >= len(self.in_channels)
            ]
            if invalid:
                raise ValueError(
                    f'Invalid gaussian_support_fusion enabled_levels: '
                    f'{invalid}')
        common_kwargs = dict(
            support_means=means,
            support_stds=stds,
            valid_mask=valid,
            token_dim=int(self.gaussian_support_fusion_cfg.get(
                'token_dim', 64)),
            temperature=float(self.gaussian_support_fusion_cfg.get(
                'temperature', 1.0)),
            gamma_init=float(self.gaussian_support_fusion_cfg.get(
                'gamma_init', 0.0)),
            residual_scale=float(self.gaussian_support_fusion_cfg.get(
                'residual_scale', 1.0)),
            zero_init_adapter=bool(self.gaussian_support_fusion_cfg.get(
                'zero_init_adapter', True)),
        )
        if mode == 'per_level_channel':
            self.gaussian_support_fusion = nn.ModuleList([
                GaussianSupportGlobalChannelAdapter(
                    in_channels=int(out_channels),
                    **common_kwargs,
                ) for _ in range(len(self.in_channels))
            ])
        elif mode == 'cross_level_channel':
            self.gaussian_support_fusion = GaussianSupportCrossLevelChannelFusion(
                out_channels=int(out_channels),
                num_levels=len(self.in_channels),
                level_context_scale=float(
                    self.gaussian_support_fusion_cfg.get(
                        'level_context_scale', 1.0)),
                **common_kwargs,
            )
        else:
            raise ValueError(
                f'Unsupported gaussian_support_fusion mode: {mode}')

    def forward(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        outs = super().forward(inputs)
        fusion = self.gaussian_support_fusion
        if fusion is None:
            return outs
        if isinstance(fusion, nn.ModuleList):
            return tuple(
                adapter(out) if idx in self.gaussian_support_enabled_levels
                else out
                for idx, (adapter, out) in enumerate(zip(fusion, outs)))
        return fusion(outs)
