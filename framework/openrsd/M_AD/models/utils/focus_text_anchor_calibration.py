from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import Tensor, nn


DEFAULT_DOTA2_CLASSES = (
    "airport",
    "baseball-diamond",
    "basketball-court",
    "bridge",
    "container-crane",
    "ground-track-field",
    "harbor",
    "helicopter",
    "helipad",
    "large-vehicle",
    "plane",
    "roundabout",
    "ship",
    "small-vehicle",
    "soccer-ball-field",
    "storage-tank",
    "swimming-pool",
    "tennis-court",
)


MONITORED_DOTA2_CLASSES = (
    "small-vehicle",
    "large-vehicle",
    "ship",
    "plane",
    "bridge",
    "helipad",
    "tennis-court",
    "basketball-court",
    "storage-tank",
    "roundabout",
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
        mask = torch.ones(num_classes, dtype=torch.bool)
        names = tuple(
            class_names if class_names is not None
            else DEFAULT_DOTA2_CLASSES[:num_classes])
        return mask, names

    requested = tuple(apply_to_classes)
    if len(requested) == 0:
        mask = torch.ones(num_classes, dtype=torch.bool)
        names = tuple(
            class_names if class_names is not None
            else DEFAULT_DOTA2_CLASSES[:num_classes])
        return mask, names

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
                    f"Unknown TAC class name {item!r}; available names are "
                    f"{list(names_for_lookup)!r}")
            class_id = name_to_id[item]
            class_name = item
        else:
            class_id = int(item)
            if class_id < 0 or class_id >= num_classes:
                raise ValueError(
                    f"TAC class id {class_id} is outside [0, {num_classes})")
            class_name = (
                names_for_lookup[class_id]
                if class_id < len(names_for_lookup) else str(class_id))
        mask[class_id] = True
        calibrated_names.append(class_name)
    return mask, tuple(calibrated_names)


class TextAnchorCalibration(nn.Module):
    """Tiny class-wise logit calibration anchored at exact identity."""

    def __init__(
            self,
            num_classes: int,
            enable: bool = True,
            use_alpha: bool = True,
            use_beta: bool = False,
            alpha_init: float = 0.0,
            beta_init: float = 0.0,
            alpha_bound: float = 0.05,
            beta_bound: float = 0.20,
            apply_to_classes: Sequence[int | str] | str | None = None,
            class_names: Sequence[str] | None = None,
            anchor_weight: float = 0.01,
            log_debug: bool = False) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.enable = bool(enable)
        self.use_alpha = bool(use_alpha)
        self.use_beta = bool(use_beta)
        self.alpha_bound = float(alpha_bound)
        self.beta_bound = float(beta_bound)
        self.anchor_weight = float(anchor_weight)
        self.log_debug = bool(log_debug)

        mask, calibrated_names = _resolve_class_mask(
            self.num_classes, apply_to_classes, class_names)
        self.register_buffer("class_mask", mask)
        self.calibrated_class_list = calibrated_names
        names_for_lookup = tuple(
            class_names if class_names is not None
            else DEFAULT_DOTA2_CLASSES[:self.num_classes])
        self.class_to_index = {
            name: idx for idx, name in enumerate(names_for_lookup)
        }

        alpha_raw_init = _inverse_tanh_bounded(alpha_init, self.alpha_bound)
        beta_raw_init = _inverse_tanh_bounded(beta_init, self.beta_bound)
        if self.use_alpha:
            self.raw_alpha = nn.Parameter(
                torch.full((self.num_classes,), alpha_raw_init))
        else:
            self.register_buffer(
                "raw_alpha", torch.full((self.num_classes,), alpha_raw_init))
        if self.use_beta:
            self.raw_beta = nn.Parameter(
                torch.full((self.num_classes,), beta_raw_init))
        else:
            self.register_buffer(
                "raw_beta", torch.full((self.num_classes,), beta_raw_init))
        self.last_debug: dict[str, Any] = {}

    def alpha_values(self) -> Tensor:
        if not self.use_alpha:
            return torch.zeros_like(self.raw_alpha)
        alpha = self.alpha_bound * torch.tanh(self.raw_alpha)
        return torch.where(self.class_mask, alpha, torch.zeros_like(alpha))

    def beta_values(self) -> Tensor:
        if not self.use_beta:
            return torch.zeros_like(self.raw_beta)
        beta = self.beta_bound * torch.tanh(self.raw_beta)
        return torch.where(self.class_mask, beta, torch.zeros_like(beta))

    def _view_for_logits(self, values: Tensor, logits: Tensor) -> Tensor:
        if logits.shape[-1] == self.num_classes:
            return values.view(*([1] * (logits.dim() - 1)), self.num_classes)
        if logits.dim() >= 2 and logits.shape[1] == self.num_classes:
            return values.view(1, self.num_classes, *([1] * (logits.dim() - 2)))
        raise ValueError(
            "TAC logits must have num_classes on channel dim 1 or last dim; "
            f"got shape {tuple(logits.shape)} for {self.num_classes} classes")

    def _has_full_class_dim(self, logits: Tensor) -> bool:
        return bool(
            logits.shape[-1] == self.num_classes
            or (logits.dim() >= 2 and logits.shape[1] == self.num_classes))

    def _channel_dim(self, logits: Tensor) -> int:
        if logits.shape[-1] == self.num_classes:
            return -1
        if logits.dim() >= 2:
            return 1
        raise ValueError(
            f"TAC logits must have a class dimension; got {tuple(logits.shape)}")

    def _episodic_values(
            self,
            logits: Tensor,
            class_indices: Sequence[int]) -> tuple[Tensor, Tensor]:
        channel_dim = self._channel_dim(logits)
        num_channels = int(logits.shape[channel_dim])
        alpha = logits.new_zeros(num_channels)
        beta = logits.new_zeros(num_channels)
        full_alpha = self.alpha_values().to(
            device=logits.device, dtype=logits.dtype)
        full_beta = self.beta_values().to(
            device=logits.device, dtype=logits.dtype)
        for channel_idx, class_idx in enumerate(class_indices[:num_channels]):
            class_idx = int(class_idx)
            if 0 <= class_idx < self.num_classes:
                alpha[channel_idx] = full_alpha[class_idx]
                beta[channel_idx] = full_beta[class_idx]
        return alpha, beta

    def _view_channel_values(self, values: Tensor, logits: Tensor) -> Tensor:
        channel_dim = self._channel_dim(logits)
        if channel_dim == -1:
            return values.view(*([1] * (logits.dim() - 1)), values.numel())
        return values.view(1, values.numel(), *([1] * (logits.dim() - 2)))

    def forward(self, logits: Tensor) -> Tensor:
        if not self.enable:
            return logits
        if not self._has_full_class_dim(logits):
            debug = self.debug_state()
            debug["skipped_incompatible_logits"] = True
            debug["incompatible_logits_shape"] = list(logits.shape)
            self.last_debug = debug
            return logits
        alpha = self.alpha_values().to(device=logits.device, dtype=logits.dtype)
        beta = self.beta_values().to(device=logits.device, dtype=logits.dtype)
        alpha_view = self._view_for_logits(alpha, logits)
        beta_view = self._view_for_logits(beta, logits)
        out = (1.0 + alpha_view) * logits + beta_view
        debug = self.debug_state()
        debug["skipped_incompatible_logits"] = False
        debug["incompatible_logits_shape"] = None
        self.last_debug = debug
        return out

    def forward_for_class_indices(
            self,
            logits: Tensor,
            class_indices: Sequence[int]) -> Tensor:
        if not self.enable:
            return logits
        alpha, beta = self._episodic_values(logits, class_indices)
        alpha_view = self._view_channel_values(alpha, logits)
        beta_view = self._view_channel_values(beta, logits)
        out = (1.0 + alpha_view) * logits + beta_view
        self.last_debug = self.debug_state()
        return out

    def forward_for_class_names(
            self,
            logits: Tensor,
            class_names: Sequence[str]) -> Tensor:
        class_indices = [
            self.class_to_index.get(str(class_name), -1)
            for class_name in class_names
        ]
        return self.forward_for_class_indices(logits, class_indices)

    def anchor_regularizer(self, ref: Tensor | None = None) -> Tensor:
        terms = []
        if self.use_alpha:
            terms.append(self.alpha_values().pow(2).mean())
        if self.use_beta:
            terms.append(self.beta_values().pow(2).mean())
        if terms:
            reg = torch.stack(terms).mean() * self.anchor_weight
        else:
            reg = self.raw_alpha.sum() * 0.0
        if ref is not None:
            reg = reg.to(device=ref.device, dtype=ref.dtype)
        if not self.enable:
            reg = reg * 0.0
        return reg

    def debug_state(self) -> dict[str, Any]:
        alpha = self.alpha_values().detach().cpu()
        beta = self.beta_values().detach().cpu()
        return {
            "enabled": self.enable,
            "use_alpha": self.use_alpha,
            "use_beta": self.use_beta,
            "alpha_values": [float(v) for v in alpha.tolist()],
            "beta_values": [float(v) for v in beta.tolist()],
            "max_abs_alpha": float(alpha.abs().max().item()) if alpha.numel() else 0.0,
            "max_abs_beta": float(beta.abs().max().item()) if beta.numel() else 0.0,
            "calibrated_class_list": list(self.calibrated_class_list),
        }
