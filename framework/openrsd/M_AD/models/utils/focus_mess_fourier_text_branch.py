"""MessDet-inspired dual text branch for conservative FOCUS-OVD support fusion."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from M_AD.models.utils.focus_eqtext_adapter import FourierEquivariantTextAdapter


def _flatten_code(code: Tensor) -> Tensor:
    if code.dim() == 4:
        return code.reshape(code.shape[0], code.shape[1] * code.shape[2],
                            code.shape[3])
    if code.dim() == 3:
        return code
    raise ValueError(
        f"fourier cue must be [B,N,D] or [B,H,W,D], got {tuple(code.shape)}")


def _flatten_confidence(confidence: Optional[Tensor], code: Tensor) -> Tensor:
    if confidence is None:
        return code.new_ones(code.shape[:2])
    if confidence.dim() == 3:
        return confidence.reshape(confidence.shape[0], -1).to(code)
    if confidence.dim() == 2:
        return confidence.to(code)
    raise ValueError(
        f"confidence must be [B,N] or [B,H,W], got {tuple(confidence.shape)}")


def _expand_support(text_support: Tensor, batch: int,
                    positions: int) -> Tensor:
    if text_support.dim() == 2:
        return text_support[None, None].expand(
            batch, positions, text_support.shape[0], text_support.shape[1])
    if text_support.dim() == 3:
        if text_support.shape[0] == 1 and batch > 1:
            text_support = text_support.expand(batch, -1, -1)
        if text_support.shape[0] != batch:
            raise ValueError(
                f"text support batch {text_support.shape[0]} does not match "
                f"cue batch {batch}")
        return text_support[:, None].expand(
            batch, positions, text_support.shape[1], text_support.shape[2])
    if text_support.dim() == 4:
        if text_support.shape[0] != batch or text_support.shape[1] != positions:
            raise ValueError(
                "text support [B,N,M,D] must match cue batch/positions, "
                f"got {tuple(text_support.shape)} and {batch}/{positions}")
        return text_support
    raise ValueError(
        "text support must be [M,D], [B,M,D], or [B,N,M,D], "
        f"got {tuple(text_support.shape)}")


def _target_mask_from_names(
        support_count: int,
        device: torch.device,
        class_names: Optional[Sequence[str]],
        apply_to_classes: Iterable[str],
        allow_all_classes: bool) -> Tensor:
    if allow_all_classes:
        return torch.ones(support_count, dtype=torch.bool, device=device)
    if class_names is None:
        return torch.ones(support_count, dtype=torch.bool, device=device)
    targets = {str(name).lower().replace("_", "-") for name in apply_to_classes}
    mask = torch.zeros(support_count, dtype=torch.bool, device=device)
    for idx, name in enumerate(class_names[:support_count]):
        if str(name).lower().replace("_", "-") in targets:
            mask[idx] = True
    return mask


def _support_mask(
        support_count: int,
        batch: int,
        device: torch.device,
        support_labels: Optional[Tensor],
        apply_to_class_ids: Optional[Iterable[int]],
        class_names: Optional[Sequence[str]],
        apply_to_classes: Iterable[str],
        allow_all_classes: bool) -> Tensor:
    if allow_all_classes:
        return torch.ones(support_count, dtype=torch.bool, device=device)
    if support_labels is None:
        return _target_mask_from_names(
            support_count, device, class_names, apply_to_classes,
            allow_all_classes)

    labels = support_labels.to(device=device)
    if labels.dim() == 1:
        labels = labels[None, :].expand(batch, labels.shape[0])
    if labels.dim() != 2:
        raise ValueError(
            f"support_labels must be [M] or [B,M], got {tuple(labels.shape)}")
    if labels.shape[0] == 1 and batch > 1:
        labels = labels.expand(batch, labels.shape[1])
    if labels.shape[0] != batch or labels.shape[1] != support_count:
        raise ValueError(
            f"support_labels shape {tuple(labels.shape)} does not match "
            f"batch/support {batch}/{support_count}")

    if apply_to_class_ids is not None:
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for class_id in apply_to_class_ids:
            mask = mask | (labels == int(class_id))
        return mask & (labels >= 0)

    max_label = (
        int(labels[labels >= 0].max().item()) + 1
        if torch.any(labels >= 0) else 1)
    name_mask = _target_mask_from_names(
        max(max_label, len(class_names or []), support_count),
        device,
        class_names,
        apply_to_classes,
        allow_all_classes)
    safe = labels.long().clamp(min=0, max=name_mask.shape[0] - 1)
    valid = (labels >= 0) & (labels < name_mask.shape[0])
    return name_mask[safe] & valid


class MessDetTextDownsampleBranch(nn.Module):
    """C8 text support branch inspired by MessDet strict equivariant downsampling."""

    def __init__(
            self,
            support_dim: int,
            code_dim: int,
            group_order: int = 8,
            low_rank: int = 16,
            apply_to_class_ids: Optional[Iterable[int]] = None,
            class_names: Optional[Sequence[str]] = None,
            apply_to_classes: Iterable[str] = ("small-vehicle",),
            allow_all_classes: bool = False,
            alpha_m_init: float = 0.0,
            alpha_m_max: float = 0.02,
            max_delta_norm_ratio: float = 0.03,
            eps: float = 1e-8) -> None:
        super().__init__()
        if int(group_order) != 8:
            raise ValueError("MessDet text branch currently supports C8 only")
        if float(alpha_m_max) > 0.02:
            raise ValueError("alpha_m_max must be <= 0.02")
        self.support_dim = int(support_dim)
        self.code_dim = int(code_dim)
        self.group_order = int(group_order)
        self.apply_to_class_ids = (
            {int(v) for v in apply_to_class_ids}
            if apply_to_class_ids is not None else None)
        self.class_names = list(class_names) if class_names is not None else None
        self.apply_to_classes = tuple(str(v) for v in apply_to_classes)
        self.allow_all_classes = bool(allow_all_classes)
        self.alpha_m_max = float(alpha_m_max)
        self.max_delta_norm_ratio = float(max_delta_norm_ratio)
        self.eps = float(eps)

        hidden_dim = max(1, int(low_rank))
        self.hidden_dim = hidden_dim
        self.orbit_proj = nn.Linear(
            self.support_dim, self.group_order * hidden_dim)
        self.code_gate = nn.Sequential(
            nn.Linear(self.code_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_proj = nn.Linear(hidden_dim, self.support_dim)
        self.alpha_m = nn.Parameter(torch.tensor(float(alpha_m_init)))

    def _cyclic_mix(self, orbit: Tensor) -> Tensor:
        return (
            0.5 * orbit
            + 0.25 * torch.roll(orbit, shifts=1, dims=-2)
            + 0.25 * torch.roll(orbit, shifts=-1, dims=-2))

    def forward(
            self,
            text_support: Tensor,
            fourier_cue: Tensor,
            confidence: Optional[Tensor] = None,
            support_labels: Optional[Tensor] = None,
            class_names: Optional[Sequence[str]] = None
    ) -> Tuple[Tensor, Dict[str, Any]]:
        code = _flatten_code(fourier_cue)
        batch, positions, code_dim = code.shape
        if code_dim != self.code_dim:
            raise ValueError(f"expected code_dim={self.code_dim}, got {code_dim}")
        conf = _flatten_confidence(confidence, code).clamp(0.0, 1.0)
        base = _expand_support(
            text_support.to(device=code.device, dtype=code.dtype),
            batch,
            positions)
        support_count = base.shape[-2]
        class_mask = _support_mask(
            support_count,
            batch,
            code.device,
            support_labels,
            self.apply_to_class_ids,
            class_names if class_names is not None else self.class_names,
            self.apply_to_classes,
            self.allow_all_classes)

        if text_support.dim() == 4:
            support_core = base.mean(dim=1)
        elif text_support.dim() == 3:
            support_core = text_support.to(device=code.device, dtype=code.dtype)
            if support_core.shape[0] == 1 and batch > 1:
                support_core = support_core.expand(batch, -1, -1)
        else:
            support_core = text_support.to(
                device=code.device, dtype=code.dtype)[None].expand(
                    batch, -1, -1)

        orbit = self.orbit_proj(support_core).reshape(
            batch, support_count, self.group_order, self.hidden_dim)
        mixed_orbit = self._cyclic_mix(orbit)
        invariant = mixed_orbit.mean(dim=-2)
        gate = torch.sigmoid(self.code_gate(code))
        hidden_delta = invariant[:, None, :, :] * gate[:, :, None, :]
        raw_delta = self.out_proj(hidden_delta)
        if class_mask.dim() == 1:
            raw_delta = raw_delta * class_mask.view(
                1, 1, support_count, 1).to(code.dtype)
        else:
            raw_delta = raw_delta * class_mask[:, None, :, None].to(code.dtype)

        alpha = torch.clamp(
            self.alpha_m, min=-self.alpha_m_max, max=self.alpha_m_max)
        gated_delta = raw_delta * alpha * conf[:, :, None, None]
        support_norm = base.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        delta_norm = gated_delta.norm(dim=-1, keepdim=True)
        max_delta = support_norm * self.max_delta_norm_ratio
        scale = torch.minimum(
            torch.ones_like(delta_norm), max_delta / (delta_norm + self.eps))
        applied_delta = gated_delta * scale
        out = F.normalize(base + applied_delta, dim=-1)
        debug = {
            "mess_delta_norm": torch.nan_to_num(applied_delta.norm(dim=-1)),
            "mess_delta_norm_ratio": torch.nan_to_num(
                applied_delta.norm(dim=-1)
                / support_norm.squeeze(-1).clamp_min(self.eps)),
            "mess_equivariant_shape": tuple(mixed_orbit.shape),
            "mess_alpha": alpha.detach(),
            "class_mask": (
                class_mask[0] if class_mask.dim() == 2 else class_mask
            ).detach().cpu(),
        }
        return out, debug


class FocusMessFourierTextFusion(nn.Module):
    """Fuse MessDet-style and Fourier text branches before visual support fusion."""

    def __init__(
            self,
            mess_weight_init: float = 0.5,
            fourier_weight_init: float = 0.5,
            max_branch_weight: float = 0.5,
            eps: float = 1e-8) -> None:
        super().__init__()
        if max_branch_weight > 0.5:
            raise ValueError("max_branch_weight must be <= 0.5")
        self.mess_logit = nn.Parameter(
            torch.tensor(float(mess_weight_init)).clamp_min(eps).log())
        self.fourier_logit = nn.Parameter(
            torch.tensor(float(fourier_weight_init)).clamp_min(eps).log())
        self.max_branch_weight = float(max_branch_weight)
        self.eps = float(eps)

    def _weights(self, device: torch.device,
                 dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        raw = torch.stack([
            self.mess_logit.to(device=device, dtype=dtype).exp(),
            self.fourier_logit.to(device=device, dtype=dtype).exp(),
        ])
        weights = raw / raw.sum().clamp_min(self.eps)
        mess_w = torch.clamp(weights[0], min=0.0, max=self.max_branch_weight)
        fourier_w = 1.0 - mess_w
        return mess_w, fourier_w

    def forward(self, base: Tensor, mess_text: Tensor,
                fourier_text: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        if mess_text.shape != fourier_text.shape or mess_text.shape != base.shape:
            raise ValueError(
                "base, mess_text, and fourier_text must share shape, got "
                f"{tuple(base.shape)}, {tuple(mess_text.shape)}, "
                f"{tuple(fourier_text.shape)}")
        mess_w, fourier_w = self._weights(base.device, base.dtype)
        fused_delta = (
            (mess_text - base) * mess_w + (fourier_text - base) * fourier_w)
        return F.normalize(base + fused_delta, dim=-1), {
            "mess_branch_weight": float(mess_w.detach().cpu().item()),
            "fourier_branch_weight": float(fourier_w.detach().cpu().item()),
        }


class FocusMessFourierDualTextAdapter(nn.Module):
    """Adapter-compatible dual text branch returning bounded support residuals."""

    def __init__(
            self,
            support_dim: int,
            code_dim: int,
            apply_to_class_ids: Optional[Iterable[int]] = None,
            class_names: Optional[Sequence[str]] = None,
            apply_to_classes: Iterable[str] = ("small-vehicle",),
            allow_all_classes: bool = False,
            alpha_t_init: float = 0.0,
            alpha_t_max: float = 0.02,
            max_delta_norm_ratio: float = 0.03,
            mess_cfg: Optional[Dict[str, Any]] = None,
            fourier_cfg: Optional[Dict[str, Any]] = None,
            fusion_cfg: Optional[Dict[str, Any]] = None,
            eps: float = 1e-8) -> None:
        super().__init__()
        if alpha_t_max > 0.02:
            raise ValueError("alpha_t_max must be <= 0.02")
        self.support_dim = int(support_dim)
        self.code_dim = int(code_dim)
        self.alpha_t_max = float(alpha_t_max)
        self.max_delta_norm_ratio = float(max_delta_norm_ratio)
        self.eps = float(eps)

        mess_cfg = dict(mess_cfg or {})
        fourier_cfg = dict(fourier_cfg or {})
        fusion_cfg = dict(fusion_cfg or {})
        mess_cfg["alpha_m_max"] = min(
            float(mess_cfg.get("alpha_m_max", 0.02)), 0.02)
        mess_cfg["max_delta_norm_ratio"] = min(
            float(mess_cfg.get("max_delta_norm_ratio", max_delta_norm_ratio)),
            self.max_delta_norm_ratio)
        fourier_cfg["alpha_t_max"] = min(
            float(fourier_cfg.get("alpha_t_max", alpha_t_max)),
            self.alpha_t_max)
        fourier_cfg["max_delta_norm_ratio"] = min(
            float(fourier_cfg.get("max_delta_norm_ratio", max_delta_norm_ratio)),
            self.max_delta_norm_ratio)
        fusion_cfg["max_branch_weight"] = min(
            float(fusion_cfg.get("max_branch_weight", 0.5)), 0.5)
        self.mess_branch = MessDetTextDownsampleBranch(
            support_dim=support_dim,
            code_dim=code_dim,
            apply_to_class_ids=apply_to_class_ids,
            class_names=class_names,
            apply_to_classes=apply_to_classes,
            allow_all_classes=allow_all_classes,
            group_order=mess_cfg.get("group_order", 8),
            low_rank=mess_cfg.get("low_rank", 16),
            alpha_m_init=mess_cfg.get("alpha_m_init", 0.0),
            alpha_m_max=mess_cfg["alpha_m_max"],
            max_delta_norm_ratio=mess_cfg["max_delta_norm_ratio"])
        self.fourier_branch = FourierEquivariantTextAdapter(
            support_dim=support_dim,
            code_dim=code_dim,
            apply_to_class_ids=apply_to_class_ids,
            class_names=class_names,
            apply_to_classes=apply_to_classes,
            allow_all_classes=allow_all_classes,
            low_rank=fourier_cfg.get("low_rank", 16),
            alpha_t_init=fourier_cfg.get("alpha_t_init", alpha_t_init),
            alpha_t_max=fourier_cfg["alpha_t_max"],
            max_delta_norm_ratio=fourier_cfg["max_delta_norm_ratio"])
        self.text_branch_fusion = FocusMessFourierTextFusion(
            mess_weight_init=fusion_cfg.get("mess_weight_init", 0.5),
            fourier_weight_init=fusion_cfg.get("fourier_weight_init", 0.5),
            max_branch_weight=fusion_cfg["max_branch_weight"])

    @property
    def alpha_t(self) -> Tensor:
        return self.fourier_branch.alpha_t

    def _zero_param_connection(self, ref: Tensor) -> Tensor:
        zero = ref.new_zeros(())
        for param in self.parameters():
            zero = zero + param.sum() * ref.new_zeros(())
        return zero

    def forward(
            self,
            text_support: Tensor,
            fourier_cue: Tensor,
            confidence: Optional[Tensor] = None,
            class_ids: Optional[Tensor] = None,
            support_labels: Optional[Tensor] = None,
            class_names: Optional[Sequence[str]] = None
    ) -> Tuple[Tensor, Dict[str, Any]]:
        return self._forward_impl(
            text_support,
            fourier_cue,
            confidence,
            class_ids=class_ids,
            support_labels=support_labels,
            class_names=class_names)

    def _forward_impl(
            self,
            text_support: Tensor,
            fourier_cue: Tensor,
            confidence: Optional[Tensor] = None,
            class_ids: Optional[Tensor] = None,
            support_labels: Optional[Tensor] = None,
            class_names: Optional[Sequence[str]] = None
    ) -> Tuple[Tensor, Dict[str, Any]]:
        code = _flatten_code(fourier_cue)
        base = _expand_support(
            text_support.to(device=code.device, dtype=code.dtype),
            code.shape[0],
            code.shape[1])
        labels = class_ids if class_ids is not None else support_labels
        support_count = base.shape[-2]
        target_mask = _support_mask(
            support_count,
            code.shape[0],
            code.device,
            labels,
            self.mess_branch.apply_to_class_ids,
            class_names if class_names is not None else self.mess_branch.class_names,
            self.mess_branch.apply_to_classes,
            self.mess_branch.allow_all_classes)
        target_cols = (
            target_mask.any(dim=0) if target_mask.dim() == 2 else target_mask)
        target_indices = torch.nonzero(target_cols, as_tuple=False).flatten()
        if target_indices.numel() == 0:
            conditioned = (
                F.normalize(base, dim=-1) + self._zero_param_connection(base))
            zeros = base.new_zeros(base.shape[:-1])
            mess_w, fourier_w = self.text_branch_fusion._weights(
                base.device, base.dtype)
            return conditioned, {
                "text_delta_norm": zeros,
                "text_delta_norm_ratio": zeros,
                "alpha_t": torch.clamp(
                    self.alpha_t, min=-self.alpha_t_max,
                    max=self.alpha_t_max).detach(),
                "mess": {},
                "fourier": {},
                "text_branch_fusion": {
                    "mess_branch_weight": float(
                        mess_w.detach().cpu().item()),
                    "fourier_branch_weight": float(
                        fourier_w.detach().cpu().item()),
                },
            }

        full_support = target_indices.numel() == support_count
        branch_base = (
            base if full_support else base.index_select(-2, target_indices))
        branch_labels = labels
        if labels is not None and not full_support:
            label_tensor = labels.to(device=code.device)
            if label_tensor.dim() == 1:
                branch_labels = label_tensor.index_select(0, target_indices)
            elif label_tensor.dim() == 2:
                branch_labels = label_tensor.index_select(1, target_indices)
        branch_class_names = class_names
        if class_names is not None and not full_support:
            branch_class_names = [
                class_names[int(idx)]
                for idx in target_indices.detach().cpu().tolist()
                if int(idx) < len(class_names)
            ]

        mess_text, mess_debug = self.mess_branch(
            branch_base,
            code,
            confidence,
            support_labels=branch_labels,
            class_names=branch_class_names)
        mess_delta = mess_text - branch_base
        del mess_text
        fourier_text, fourier_debug = self.fourier_branch(
            branch_base,
            code,
            confidence,
            class_ids=branch_labels,
            support_labels=branch_labels,
            class_names=branch_class_names)
        mess_w, fourier_w = self.text_branch_fusion._weights(
            base.device, base.dtype)
        raw_delta = mess_delta * mess_w + (
            fourier_text - branch_base) * fourier_w
        del mess_delta, fourier_text
        fusion_debug = {
            "mess_branch_weight": float(mess_w.detach().cpu().item()),
            "fourier_branch_weight": float(fourier_w.detach().cpu().item()),
        }

        support_norm = branch_base.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        delta_norm = raw_delta.norm(dim=-1, keepdim=True)
        max_delta = support_norm * self.max_delta_norm_ratio
        scale = torch.minimum(
            torch.ones_like(delta_norm), max_delta / (delta_norm + self.eps))
        applied_delta = raw_delta * scale
        branch_conditioned = F.normalize(branch_base + applied_delta, dim=-1)
        if full_support:
            conditioned = branch_conditioned
            text_delta_norm = torch.nan_to_num(applied_delta.norm(dim=-1))
            text_delta_ratio = torch.nan_to_num(
                applied_delta.norm(dim=-1)
                / support_norm.squeeze(-1).clamp_min(self.eps))
        else:
            conditioned = F.normalize(base, dim=-1)
            conditioned = conditioned.index_copy(
                -2, target_indices, branch_conditioned)
            text_delta_norm = base.new_zeros(base.shape[:-1])
            text_delta_ratio = base.new_zeros(base.shape[:-1])
            text_delta_norm = text_delta_norm.index_copy(
                -1, target_indices,
                torch.nan_to_num(applied_delta.norm(dim=-1)))
            text_delta_ratio = text_delta_ratio.index_copy(
                -1, target_indices,
                torch.nan_to_num(
                    applied_delta.norm(dim=-1)
                    / support_norm.squeeze(-1).clamp_min(self.eps)))
        debug = {
            "text_delta_norm": text_delta_norm,
            "text_delta_norm_ratio": text_delta_ratio,
            "alpha_t": torch.clamp(
                self.alpha_t, min=-self.alpha_t_max,
                max=self.alpha_t_max).detach(),
            "mess": mess_debug,
            "fourier": fourier_debug,
            "text_branch_fusion": fusion_debug,
        }
        return conditioned, debug


def build_focus_text_adapter(
        eqtext_cfg: Dict[str, Any],
        support_dim: int,
        code_dim: int,
        class_names: Optional[Sequence[str]] = None) -> nn.Module:
    adapter_type = str(eqtext_cfg.get("type", "fourier")).lower()
    common = dict(
        support_dim=support_dim,
        code_dim=code_dim,
        class_names=class_names,
        apply_to_classes=eqtext_cfg.get(
            "apply_to_classes", ("small-vehicle",)),
        allow_all_classes=eqtext_cfg.get("allow_all_classes", False),
        alpha_t_init=eqtext_cfg.get("alpha_t_init", 0.0),
        alpha_t_max=eqtext_cfg.get("alpha_t_max", 0.05),
        max_delta_norm_ratio=eqtext_cfg.get("max_delta_norm_ratio", 0.05))
    if "apply_to_class_ids" in eqtext_cfg:
        common["apply_to_class_ids"] = eqtext_cfg["apply_to_class_ids"]
    if adapter_type in {"mess_fourier_dual", "messdet_fourier_dual"}:
        common["alpha_t_max"] = min(float(common["alpha_t_max"]), 0.02)
        common["max_delta_norm_ratio"] = min(
            float(common["max_delta_norm_ratio"]), 0.03)
        return FocusMessFourierDualTextAdapter(
            **common,
            mess_cfg=eqtext_cfg.get("mess_branch", {}),
            fourier_cfg=eqtext_cfg.get("fourier_branch", {}),
            fusion_cfg=eqtext_cfg.get("text_branch_fusion", {}))
    if adapter_type == "fourier":
        if "low_rank" in eqtext_cfg:
            common["low_rank"] = eqtext_cfg["low_rank"]
        return FourierEquivariantTextAdapter(**common)
    raise ValueError(f"unknown focus eqtext adapter type: {adapter_type}")
