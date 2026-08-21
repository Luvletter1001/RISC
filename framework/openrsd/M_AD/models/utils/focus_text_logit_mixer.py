from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from M_AD.models.utils.focus_text_anchor_calibration import (
    DEFAULT_DOTA2_CLASSES,
)


def _inverse_tanh_bounded(value: float, bound: float) -> float:
    if bound <= 0:
        return 0.0
    ratio = max(min(float(value) / float(bound), 0.999999), -0.999999)
    return math.atanh(ratio)


def _resolve_class_mask(
        num_classes: int,
        apply_to_classes: Sequence[int | str] | str | None,
        class_names: Sequence[str] | None) -> tuple[Tensor, tuple[str, ...]]:
    if apply_to_classes is None or apply_to_classes == "all":
        names = tuple(
            class_names if class_names is not None
            else DEFAULT_DOTA2_CLASSES[:num_classes])
        return torch.ones(num_classes, dtype=torch.bool), names

    requested = tuple(apply_to_classes)
    names_for_lookup = tuple(
        class_names if class_names is not None
        else DEFAULT_DOTA2_CLASSES[:num_classes])
    name_to_id = {name: idx for idx, name in enumerate(names_for_lookup)}
    mask = torch.zeros(num_classes, dtype=torch.bool)
    calibrated_names: list[str] = []
    for item in requested:
        if isinstance(item, str):
            if item not in name_to_id:
                raise ValueError(
                    f"Unknown text mixer class name {item!r}; available names "
                    f"are {list(names_for_lookup)!r}")
            class_id = name_to_id[item]
            class_name = item
        else:
            class_id = int(item)
            if class_id < 0 or class_id >= num_classes:
                raise ValueError(
                    f"Text mixer class id {class_id} is outside "
                    f"[0, {num_classes})")
            class_name = (
                names_for_lookup[class_id]
                if class_id < len(names_for_lookup) else str(class_id))
        mask[class_id] = True
        calibrated_names.append(class_name)
    return mask, tuple(calibrated_names)


class FocusTextLogitMixer(nn.Module):
    """Zero-init logit mixer driven by current text-support prototypes.

    The module never replaces the support bank. It computes a small class-logit
    residual from support prototype similarity, so the baseline path remains
    exact when gamma/beta are initialized to zero.
    """

    def __init__(
            self,
            num_classes: int,
            enable: bool = False,
            mode: str = "smooth",
            use_gamma: bool = True,
            use_beta: bool = False,
            gamma_init: float = 0.0,
            beta_init: float = 0.0,
            gamma_bound: float = 0.03,
            beta_bound: float = 0.05,
            apply_to_classes: Sequence[int | str] | str | None = None,
            class_names: Sequence[str] | None = None,
            sim_threshold: float = 0.0,
            sim_power: float = 1.0,
            topk: int = 0,
            anchor_weight: float = 0.01,
            eps: float = 1e-8) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.enable = bool(enable)
        self.mode = str(mode)
        self.use_gamma = bool(use_gamma)
        self.use_beta = bool(use_beta)
        self.gamma_bound = float(gamma_bound)
        self.beta_bound = float(beta_bound)
        self.sim_threshold = float(sim_threshold)
        self.sim_power = float(sim_power)
        self.topk = int(topk)
        self.anchor_weight = float(anchor_weight)
        self.eps = float(eps)

        mask, calibrated_names = _resolve_class_mask(
            self.num_classes, apply_to_classes, class_names)
        self.register_buffer("class_mask", mask)
        self.calibrated_class_list = calibrated_names

        gamma_raw_init = _inverse_tanh_bounded(
            gamma_init, self.gamma_bound)
        beta_raw_init = _inverse_tanh_bounded(beta_init, self.beta_bound)
        if self.use_gamma:
            self.raw_gamma = nn.Parameter(
                torch.full((self.num_classes,), gamma_raw_init))
        else:
            self.register_buffer(
                "raw_gamma", torch.full((self.num_classes,), gamma_raw_init))
        if self.use_beta:
            self.raw_beta = nn.Parameter(
                torch.full((self.num_classes,), beta_raw_init))
        else:
            self.register_buffer(
                "raw_beta", torch.full((self.num_classes,), beta_raw_init))
        self.last_debug: dict[str, Any] = {}

    def gamma_values(self) -> Tensor:
        if not self.use_gamma:
            return torch.zeros_like(self.raw_gamma)
        gamma = self.gamma_bound * torch.tanh(self.raw_gamma)
        return torch.where(self.class_mask, gamma, torch.zeros_like(gamma))

    def beta_values(self) -> Tensor:
        if not self.use_beta:
            return torch.zeros_like(self.raw_beta)
        beta = self.beta_bound * torch.tanh(self.raw_beta)
        return torch.where(self.class_mask, beta, torch.zeros_like(beta))

    def _flatten_logits(
            self,
            logits: Tensor) -> tuple[Tensor, str, tuple[int, ...], int]:
        if logits.shape[-1] == self.num_classes:
            shape = tuple(logits.shape)
            return logits.reshape(
                logits.shape[0], -1, self.num_classes), "last", shape, self.num_classes
        if logits.dim() >= 3:
            shape = tuple(logits.shape)
            num_channels = int(logits.shape[1])
            flat = logits.permute(0, *range(2, logits.dim()), 1).reshape(
                logits.shape[0], -1, num_channels)
            return flat, "channel", shape, num_channels
        raise ValueError(
            "Text mixer logits must be [B,C,...] or [...,num_classes]; "
            f"got shape {tuple(logits.shape)}")

    def _restore_logits(
            self,
            flat: Tensor,
            layout: str,
            shape: tuple[int, ...],
            num_channels: int) -> Tensor:
        if layout == "last":
            return flat.reshape(shape)
        batch = shape[0]
        spatial = shape[2:]
        channel_dim = len(spatial) + 1
        return flat.reshape(batch, *spatial, num_channels).permute(
            0, channel_dim, *range(1, len(spatial) + 1)).contiguous()

    def _prepare_support(
            self,
            support_feats: Tensor,
            support_labels: Tensor | None,
            batch: int,
            num_channels: int) -> tuple[Tensor, Tensor]:
        support = support_feats.detach()
        if support.dim() == 2:
            support = support[None].expand(batch, -1, -1)
        elif support.dim() == 3:
            if support.shape[0] == 1 and batch > 1:
                support = support.expand(batch, -1, -1)
            if support.shape[0] != batch:
                raise ValueError(
                    "Text mixer support batch does not match logits batch: "
                    f"{support.shape[0]} vs {batch}")
        else:
            raise ValueError(
                "Text mixer support must be [M,D] or [B,M,D], got "
                f"{tuple(support_feats.shape)}")

        if support_labels is None:
            labels = torch.arange(
                support.shape[1],
                device=support.device,
                dtype=torch.long).clamp(max=num_channels - 1)
            labels = labels[None].expand(batch, -1)
        else:
            labels = support_labels.detach().to(
                device=support.device, dtype=torch.long)
            if labels.dim() == 1:
                labels = labels[None].expand(batch, -1)
            elif labels.dim() == 2 and labels.shape[0] == 1 and batch > 1:
                labels = labels.expand(batch, -1)
            if labels.shape[:2] != support.shape[:2]:
                raise ValueError(
                    "Text mixer labels must match support [B,M], got "
                    f"{tuple(labels.shape)} for support {tuple(support.shape)}")
        return support, labels

    def _prototype_stats(
            self,
            support_feats: Tensor,
            support_labels: Tensor | None,
            batch: int,
            num_channels: int) -> tuple[Tensor, Tensor, Tensor]:
        support, labels = self._prepare_support(
            support_feats, support_labels, batch, num_channels)
        proto = support.new_zeros(batch, num_channels, support.shape[-1])
        valid = torch.zeros(
            batch, num_channels,
            dtype=torch.bool,
            device=support.device)
        for class_id in range(num_channels):
            mask = labels == class_id
            if not torch.any(mask):
                continue
            count = mask.sum(dim=1).clamp_min(1).to(support.dtype)
            summed = (support * mask[..., None].to(support.dtype)).sum(dim=1)
            proto[:, class_id] = summed / count[:, None]
            valid[:, class_id] = mask.any(dim=1)

        norm = proto.norm(dim=-1)
        normalized = F.normalize(proto, dim=-1)
        sim = torch.matmul(normalized, normalized.transpose(-1, -2))
        eye = torch.eye(
            num_channels,
            dtype=torch.bool,
            device=sim.device).unsqueeze(0)
        valid_pair = valid[:, :, None] & valid[:, None, :]
        sim = sim.masked_fill(eye | (~valid_pair), 0.0)
        sim = (sim - self.sim_threshold).clamp_min(0.0)
        if self.sim_power != 1.0:
            sim = sim.pow(self.sim_power)
        if self.topk > 0 and self.topk < num_channels:
            keep = torch.zeros_like(sim, dtype=torch.bool)
            _, idx = torch.topk(sim, k=self.topk, dim=-1)
            keep.scatter_(-1, idx, True)
            sim = sim.masked_fill(~keep, 0.0)
        sim = sim / sim.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return sim, norm, valid

    def _delta(
            self,
            flat_logits: Tensor,
            sim: Tensor,
            norm: Tensor,
            valid: Tensor) -> Tensor:
        if self.mode == "smooth":
            return torch.bmm(flat_logits, sim.transpose(1, 2))
        if self.mode == "suppress":
            return -torch.bmm(flat_logits, sim.transpose(1, 2))
        if self.mode == "sharpen":
            mixed = torch.bmm(flat_logits, sim.transpose(1, 2))
            return flat_logits - mixed
        if self.mode == "center":
            mean = flat_logits.mean(dim=-1, keepdim=True)
            return flat_logits - mean
        if self.mode == "norm_scale":
            masked_norm = norm.masked_fill(~valid, 0.0)
            denom = valid.sum(dim=-1, keepdim=True).clamp_min(1)
            mean = masked_norm.sum(dim=-1, keepdim=True) / denom
            stat = (masked_norm - mean) / mean.abs().clamp_min(self.eps)
            stat = stat.masked_fill(~valid, 0.0).clamp(-1.0, 1.0)
            return flat_logits * stat[:, None, :]
        raise ValueError(f"Unknown text mixer mode {self.mode!r}")

    def _episode_values(
            self,
            num_channels: int,
            device: torch.device,
            dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        gamma = self.gamma_values().to(device=device, dtype=dtype)
        beta = self.beta_values().to(device=device, dtype=dtype)
        if num_channels == self.num_classes:
            return gamma, beta
        mask = self.class_mask.to(device=device)
        if torch.any(mask):
            gamma_value = gamma[mask].mean()
            beta_value = beta[mask].mean()
        else:
            gamma_value = gamma.mean() * 0.0
            beta_value = beta.mean() * 0.0
        return (
            gamma_value.expand(num_channels),
            beta_value.expand(num_channels),
        )

    def forward(
            self,
            logits: Tensor,
            support_feats: Tensor,
            support_labels: Tensor | None = None) -> Tensor:
        if not self.enable:
            return logits
        flat, layout, shape, num_channels = self._flatten_logits(logits)
        with torch.no_grad():
            sim, norm, valid = self._prototype_stats(
                support_feats.to(device=logits.device, dtype=logits.dtype),
                support_labels,
                flat.shape[0],
                num_channels)
        gamma, beta = self._episode_values(
            num_channels,
            logits.device,
            logits.dtype)
        delta = self._delta(flat, sim, norm, valid)
        out = flat + delta * gamma.view(1, 1, num_channels)
        out = out + beta.view(1, 1, num_channels)
        self.last_debug = self.debug_state()
        self.last_debug.update({
            "mean_sim_mass": float(sim.sum(dim=-1).mean().detach().cpu().item()),
            "mode": self.mode,
            "episode_num_channels": int(num_channels),
        })
        return self._restore_logits(out, layout, shape, num_channels)

    def anchor_regularizer(self, ref: Tensor | None = None) -> Tensor:
        terms = []
        if self.use_gamma:
            terms.append(self.gamma_values().pow(2).mean())
        if self.use_beta:
            terms.append(self.beta_values().pow(2).mean())
        if terms:
            reg = torch.stack(terms).mean() * self.anchor_weight
        else:
            reg = self.raw_gamma.sum() * 0.0
        if ref is not None:
            reg = reg.to(device=ref.device, dtype=ref.dtype)
        if not self.enable:
            reg = reg * 0.0
        return reg

    def debug_state(self) -> dict[str, Any]:
        gamma = self.gamma_values().detach().cpu()
        beta = self.beta_values().detach().cpu()
        return {
            "enabled": self.enable,
            "mode": self.mode,
            "use_gamma": self.use_gamma,
            "use_beta": self.use_beta,
            "gamma_values": [float(v) for v in gamma.tolist()],
            "beta_values": [float(v) for v in beta.tolist()],
            "max_abs_gamma": float(gamma.abs().max().item())
            if gamma.numel() else 0.0,
            "max_abs_beta": float(beta.abs().max().item())
            if beta.numel() else 0.0,
            "calibrated_class_list": list(self.calibrated_class_list),
            "sim_threshold": self.sim_threshold,
            "sim_power": self.sim_power,
            "topk": self.topk,
        }
