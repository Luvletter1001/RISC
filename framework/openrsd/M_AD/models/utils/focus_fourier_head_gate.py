"""Fourier-conditioned logit adapter for FOCUS detector heads."""

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


def _inverse_tanh_bounded(value: float, bound: float) -> float:
    if bound <= 0.0:
        return 0.0
    ratio = max(min(float(value) / float(bound), 0.999999), -0.999999)
    return math.atanh(ratio)


def _resolve_class_mask(
        num_classes: int,
        apply_to_classes: Sequence[int | str] | str | None,
        class_names: Sequence[str] | None) -> tuple[Tensor, tuple[str, ...]]:
    names_for_lookup = tuple(
        class_names if class_names is not None
        else DEFAULT_DOTA2_CLASSES[:num_classes])
    if apply_to_classes is None or apply_to_classes == "all":
        return torch.ones(num_classes, dtype=torch.bool), names_for_lookup

    requested = tuple(apply_to_classes)
    if len(requested) == 0:
        return torch.ones(num_classes, dtype=torch.bool), names_for_lookup

    name_to_id = {name: idx for idx, name in enumerate(names_for_lookup)}
    mask = torch.zeros(num_classes, dtype=torch.bool)
    calibrated_names: list[str] = []
    for item in requested:
        if isinstance(item, str):
            normalized = str(item).lower().replace("_", "-")
            lookup = {
                str(name).lower().replace("_", "-"): idx
                for idx, name in enumerate(names_for_lookup)
            }
            if normalized not in lookup:
                raise ValueError(
                    f"Unknown Fourier head gate class {item!r}; available "
                    f"names are {list(names_for_lookup)!r}")
            class_id = lookup[normalized]
            class_name = names_for_lookup[class_id]
        else:
            class_id = int(item)
            if class_id < 0 or class_id >= num_classes:
                raise ValueError(
                    f"Fourier head gate class id {class_id} is outside "
                    f"[0, {num_classes})")
            class_name = (
                names_for_lookup[class_id]
                if class_id < len(names_for_lookup) else str(class_id))
        mask[class_id] = True
        calibrated_names.append(str(class_name))
    return mask, tuple(calibrated_names)


class FourierHeadLogitAdapter(nn.Module):
    """Tiny zero-initialized Fourier signal injection for class logits.

    The adapter consumes per-level orientation confidence/theta already
    produced by ``FourierOrientationLearner``. With alpha/beta initialized to
    zero, it is an exact identity map.
    """

    def __init__(
            self,
            num_classes: int,
            enable: bool = True,
            feature_mode: str = "confidence",
            harmonic_order: int = 2,
            use_alpha: bool = True,
            use_beta: bool = False,
            alpha_init: float = 0.0,
            beta_init: float = 0.0,
            alpha_bound: float = 0.03,
            beta_bound: float = 0.05,
            apply_to_classes: Sequence[int | str] | str | None = None,
            class_names: Sequence[str] | None = None,
            anchor_weight: float = 0.01) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.enable = bool(enable)
        self.feature_mode = str(feature_mode)
        self.harmonic_order = int(harmonic_order)
        self.use_alpha = bool(use_alpha)
        self.use_beta = bool(use_beta)
        self.alpha_bound = float(alpha_bound)
        self.beta_bound = float(beta_bound)
        self.anchor_weight = float(anchor_weight)

        mask, calibrated_names = _resolve_class_mask(
            self.num_classes, apply_to_classes, class_names)
        self.register_buffer("class_mask", mask)
        self.calibrated_class_list = calibrated_names

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

    def _signal(self, logits: Tensor, theta: Tensor | None,
                confidence: Tensor | None) -> Tensor | None:
        if confidence is None and theta is None:
            return None
        if confidence is None:
            confidence = logits.new_ones(theta.shape)
        confidence = confidence.to(device=logits.device, dtype=logits.dtype)
        if confidence.dim() == 4 and confidence.shape[1] == 1:
            confidence = confidence[:, 0]
        if confidence.dim() != 3:
            return None
        if confidence.shape[-2:] != logits.shape[-2:]:
            confidence = torch.nn.functional.interpolate(
                confidence[:, None],
                size=logits.shape[-2:],
                mode="bilinear",
                align_corners=False)[:, 0]
        confidence = confidence.clamp(0.0, 1.0)

        mode = self.feature_mode
        if mode == "confidence":
            signal = confidence
        elif mode == "low_confidence":
            signal = 1.0 - confidence
        elif mode in {"sin", "cos", "abs_cos", "abs_sin"}:
            if theta is None:
                return None
            theta = theta.to(device=logits.device, dtype=logits.dtype)
            if theta.dim() == 4 and theta.shape[1] == 1:
                theta = theta[:, 0]
            if theta.shape[-2:] != logits.shape[-2:]:
                theta = torch.nn.functional.interpolate(
                    theta[:, None],
                    size=logits.shape[-2:],
                    mode="bilinear",
                    align_corners=False)[:, 0]
            phase = float(self.harmonic_order) * theta
            if mode == "sin":
                signal = torch.sin(phase) * confidence
            elif mode == "cos":
                signal = torch.cos(phase) * confidence
            elif mode == "abs_sin":
                signal = torch.sin(phase).abs() * confidence
            else:
                signal = torch.cos(phase).abs() * confidence
        else:
            raise ValueError(f"Unsupported Fourier head feature_mode={mode!r}")
        return torch.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, logits: Tensor, theta: Tensor | None = None,
                confidence: Tensor | None = None) -> Tensor:
        if not self.enable:
            return logits
        if logits.dim() != 4 or logits.shape[1] != self.num_classes:
            self.last_debug = {
                **self.debug_state(),
                "skipped_incompatible_logits": True,
                "incompatible_logits_shape": list(logits.shape),
            }
            return logits
        signal = self._signal(logits, theta, confidence)
        if signal is None:
            self.last_debug = {
                **self.debug_state(),
                "skipped_missing_signal": True,
            }
            return logits
        alpha = self.alpha_values().to(device=logits.device, dtype=logits.dtype)
        beta = self.beta_values().to(device=logits.device, dtype=logits.dtype)
        alpha_view = alpha.view(1, self.num_classes, 1, 1)
        beta_view = beta.view(1, self.num_classes, 1, 1)
        signal_view = signal[:, None]
        out = (1.0 + alpha_view * signal_view) * logits + beta_view * signal_view
        self.last_debug = {
            **self.debug_state(),
            "skipped_incompatible_logits": False,
            "skipped_missing_signal": False,
            "signal_mean": float(signal.detach().mean().cpu().item()),
            "signal_max": float(signal.detach().max().cpu().item()),
        }
        return out

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
            "feature_mode": self.feature_mode,
            "harmonic_order": self.harmonic_order,
            "use_alpha": self.use_alpha,
            "use_beta": self.use_beta,
            "max_abs_alpha": float(alpha.abs().max().item()) if alpha.numel() else 0.0,
            "max_abs_beta": float(beta.abs().max().item()) if beta.numel() else 0.0,
            "calibrated_class_list": list(self.calibrated_class_list),
        }
