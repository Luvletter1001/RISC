# FOCUS-OVD MessDet Dual Text Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a conservative MessDet-inspired dual text branch on top of FOCUS-OVD, with FOCUS-OVD kept as the baseline and failed direct-logit text modules excluded from the new main path.

**Architecture:** The new branch lives in support feature space. A wrapper text adapter accepts the same inputs as the current Fourier text adapter, builds one MessDet-style C8 text branch and one Fourier text branch, fuses them into a bounded text residual, and lets the existing FOCUS visual/text support fusion combine that text residual with the FOCUS-OVD support path. The image backbone, neck, bbox regression, NMS, and native support bank stay unchanged.

**Tech Stack:** PyTorch, OpenRSD/MMRotate head modules, existing `OrientationConditionedContrastiveEmbed`, existing `FourierOrientationLearner`, existing `FocusDualSupportFusion`, pytest, six-card torchrun with `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1`.

---

## File Structure

- Create `M_AD/models/utils/focus_mess_fourier_text_branch.py`
  - Owns the new text-side implementation.
  - Defines `MessDetTextDownsampleBranch`, `FocusMessFourierTextFusion`, `FocusMessFourierDualTextAdapter`, and `build_focus_text_adapter`.
  - Imports and reuses `FourierEquivariantTextAdapter` for the Fourier branch.
- Modify `M_AD/models/dense_heads/Flex_Rrtmdet_head_v3_1.py`
  - Replace direct `FourierEquivariantTextAdapter` construction with `build_focus_text_adapter`.
  - Keep existing default Fourier adapter behavior when `eqtext.type` is absent.
  - Do not add any direct class-logit hook.
- Create `tests/test_focus_mess_fourier_text_branch.py`
  - Unit tests for C8 cyclic mixing, zero-equivalence, class masking, gradient flow, and text branch fusion caps.
- Create `tests/test_focus_mess_fourier_embed.py`
  - Support-classifier integration tests for exact FOCUS-OVD fallback when text weight is zero.
- Create `tests/test_focus_mess_fourier_config_and_launcher.py`
  - Config and launcher safety tests for six-card training, trainability allowlist, and disabled failed modules.
- Create `M_configs/experiments/focus_ovd/focus_ovd_a10_mess_fourier_dual_text_6gpu_zero.py`
  - Zero-equivalence config with `max_text_weight=0.0`.
- Create `M_configs/experiments/focus_ovd/focus_ovd_a10_mess_fourier_dual_text_6gpu.py`
  - Nonzero conservative training config with `max_text_weight=0.02`.
- Create `M_Tools/experiments/run_focus_mess_fourier_dual_text_6gpu.sh`
  - Six-card launcher with NCCL P2P/IB disabled.

Use exact-path staging for every commit. The repository has many unrelated dirty files; do not run broad `git add .`, `git reset`, or checkout commands.

---

### Task 1: Failing Unit Tests For The New Text Branch

**Files:**
- Create: `tests/test_focus_mess_fourier_text_branch.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_focus_mess_fourier_text_branch.py` with this content:

```python
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _module():
    return _load_module(
        "focus_mess_fourier_text_branch",
        "M_AD/models/utils/focus_mess_fourier_text_branch.py")


def test_cyclic_mix_commutes_with_c8_roll():
    mod = _module()
    branch = mod.MessDetTextDownsampleBranch(
        support_dim=4,
        code_dim=6,
        group_order=8,
        low_rank=3,
        alpha_m_init=0.0,
        alpha_m_max=0.02)
    orbit = torch.randn(2, 5, 3, 8, 4)

    mixed_after_roll = branch._cyclic_mix(torch.roll(orbit, shifts=2, dims=-2))
    rolled_after_mix = torch.roll(branch._cyclic_mix(orbit), shifts=2, dims=-2)

    assert torch.allclose(mixed_after_roll, rolled_after_mix, atol=1e-6)


def test_messdet_text_branch_zero_alpha_keeps_support_and_reports_c8_shape():
    torch.manual_seed(3)
    mod = _module()
    branch = mod.MessDetTextDownsampleBranch(
        support_dim=4,
        code_dim=6,
        group_order=8,
        low_rank=3,
        alpha_m_init=0.0,
        alpha_m_max=0.02,
        apply_to_class_ids=(13,))
    base = F.normalize(torch.randn(1, 3, 4), dim=-1)
    cue = torch.randn(1, 2, 6)
    confidence = torch.ones(1, 2)
    labels = torch.tensor([[13, 2, 13]])

    out, debug = branch(
        base,
        cue,
        confidence,
        support_labels=labels)

    expected = base[:, None].expand(1, 2, 3, 4)
    assert torch.allclose(out, expected, atol=1e-6)
    assert tuple(debug["mess_equivariant_shape"]) == (1, 2, 3, 8, 4)
    assert float(debug["mess_delta_norm"].max()) == 0.0
    assert torch.equal(debug["class_mask"], torch.tensor([True, False, True]))


def test_messdet_text_branch_active_alpha_masks_non_target_supports():
    torch.manual_seed(5)
    mod = _module()
    branch = mod.MessDetTextDownsampleBranch(
        support_dim=4,
        code_dim=6,
        group_order=8,
        low_rank=3,
        alpha_m_init=0.0,
        alpha_m_max=0.02,
        apply_to_class_ids=(13,))
    with torch.no_grad():
        branch.alpha_m.fill_(0.02)
    base = F.normalize(torch.randn(1, 3, 4), dim=-1)
    cue = torch.randn(1, 2, 6)
    confidence = torch.ones(1, 2)
    labels = torch.tensor([[13, 2, 13]])

    _out, debug = branch(
        base,
        cue,
        confidence,
        support_labels=labels)

    assert float(debug["mess_delta_norm"][..., 1].max()) == 0.0
    assert float(debug["mess_delta_norm"][..., [0, 2]].max()) > 0.0


def test_dual_text_adapter_zero_alpha_keeps_support_and_backprops_to_alpha():
    torch.manual_seed(7)
    mod = _module()
    adapter = mod.FocusMessFourierDualTextAdapter(
        support_dim=4,
        code_dim=6,
        apply_to_class_ids=(13,),
        alpha_t_init=0.0,
        alpha_t_max=0.02,
        max_delta_norm_ratio=0.03,
        mess_cfg=dict(group_order=8, low_rank=3, alpha_m_max=0.02),
        fourier_cfg=dict(low_rank=3, alpha_t_max=0.02),
        fusion_cfg=dict(max_branch_weight=0.5))
    base = F.normalize(torch.randn(1, 3, 4), dim=-1)
    cue = torch.randn(1, 2, 6)
    confidence = torch.ones(1, 2)
    labels = torch.tensor([[13, 2, 13]])

    out, debug = adapter(
        base,
        cue,
        confidence,
        support_labels=labels)
    expected = base[:, None].expand(1, 2, 3, 4)
    assert torch.allclose(out, expected, atol=1e-6)
    assert float(debug["text_delta_norm"].max()) == 0.0
    assert 0.0 <= debug["text_branch_fusion"]["mess_branch_weight"] <= 0.5

    out[..., 0, :].sum().backward()
    assert adapter.alpha_t.grad is not None


def test_build_focus_text_adapter_preserves_default_fourier_type():
    mod = _module()
    adapter = mod.build_focus_text_adapter(
        eqtext_cfg=dict(enable=True, low_rank=3),
        support_dim=4,
        code_dim=6,
        class_names=("small-vehicle", "ship"))

    assert adapter.__class__.__name__ == "FourierEquivariantTextAdapter"
```

- [ ] **Step 2: Run the tests to verify they fail because the module is missing**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_focus_mess_fourier_text_branch.py -q
```

Expected: FAIL with `FileNotFoundError` for `M_AD/models/utils/focus_mess_fourier_text_branch.py`.

- [ ] **Step 3: Commit the failing tests**

Run:

```bash
rtk git add tests/test_focus_mess_fourier_text_branch.py
rtk git commit -m "test: add focus mess fourier text branch coverage"
```

Expected: commit succeeds and stages only `tests/test_focus_mess_fourier_text_branch.py`.

---

### Task 2: Implement The Text Branch Module

**Files:**
- Create: `M_AD/models/utils/focus_mess_fourier_text_branch.py`
- Test: `tests/test_focus_mess_fourier_text_branch.py`

- [ ] **Step 1: Add the implementation**

Create `M_AD/models/utils/focus_mess_fourier_text_branch.py` with the code below. This file is self-contained except for the existing Fourier text adapter import.

```python
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
        return code.reshape(code.shape[0], code.shape[1] * code.shape[2], code.shape[3])
    if code.dim() == 3:
        return code
    raise ValueError(f"fourier cue must be [B,N,D] or [B,H,W,D], got {tuple(code.shape)}")


def _flatten_confidence(confidence: Optional[Tensor], code: Tensor) -> Tensor:
    if confidence is None:
        return code.new_ones(code.shape[:2])
    if confidence.dim() == 3:
        return confidence.reshape(confidence.shape[0], -1).to(code)
    if confidence.dim() == 2:
        return confidence.to(code)
    raise ValueError(f"confidence must be [B,N] or [B,H,W], got {tuple(confidence.shape)}")


def _expand_support(text_support: Tensor, batch: int, positions: int) -> Tensor:
    if text_support.dim() == 2:
        return text_support[None, None].expand(batch, positions, text_support.shape[0], text_support.shape[1])
    if text_support.dim() == 3:
        if text_support.shape[0] == 1 and batch > 1:
            text_support = text_support.expand(batch, -1, -1)
        if text_support.shape[0] != batch:
            raise ValueError(f"text support batch {text_support.shape[0]} does not match cue batch {batch}")
        return text_support[:, None].expand(batch, positions, text_support.shape[1], text_support.shape[2])
    if text_support.dim() == 4:
        if text_support.shape[0] != batch or text_support.shape[1] != positions:
            raise ValueError(
                "text support [B,N,M,D] must match cue batch/positions, "
                f"got {tuple(text_support.shape)} and {batch}/{positions}")
        return text_support
    raise ValueError(f"text support must be [M,D], [B,M,D], or [B,N,M,D], got {tuple(text_support.shape)}")


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
            support_count, device, class_names, apply_to_classes, allow_all_classes)

    labels = support_labels.to(device=device)
    if labels.dim() == 1:
        labels = labels[None, :].expand(batch, labels.shape[0])
    if labels.dim() != 2:
        raise ValueError(f"support_labels must be [M] or [B,M], got {tuple(labels.shape)}")
    if labels.shape[0] == 1 and batch > 1:
        labels = labels.expand(batch, labels.shape[1])
    if labels.shape[0] != batch or labels.shape[1] != support_count:
        raise ValueError(
            f"support_labels shape {tuple(labels.shape)} does not match batch/support {batch}/{support_count}")

    if apply_to_class_ids is not None:
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for class_id in apply_to_class_ids:
            mask = mask | (labels == int(class_id))
        return mask & (labels >= 0)

    max_label = int(labels[labels >= 0].max().item()) + 1 if torch.any(labels >= 0) else 1
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
        self.orbit_proj = nn.Linear(self.support_dim, self.group_order * self.support_dim)
        self.code_gate = nn.Sequential(
            nn.Linear(self.code_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.support_dim),
        )
        self.out_proj = nn.Linear(self.support_dim, self.support_dim)
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
            class_names: Optional[Sequence[str]] = None) -> Tuple[Tensor, Dict[str, Any]]:
        code = _flatten_code(fourier_cue)
        batch, positions, code_dim = code.shape
        if code_dim != self.code_dim:
            raise ValueError(f"expected code_dim={self.code_dim}, got {code_dim}")
        conf = _flatten_confidence(confidence, code).clamp(0.0, 1.0)
        base = _expand_support(text_support.to(device=code.device, dtype=code.dtype), batch, positions)
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

        orbit_delta = self.orbit_proj(base).reshape(
            batch, positions, support_count, self.group_order, self.support_dim)
        orbit = base.unsqueeze(-2) + orbit_delta
        mixed_orbit = self._cyclic_mix(orbit)
        invariant = mixed_orbit.mean(dim=-2)
        gate = torch.sigmoid(self.code_gate(code)).unsqueeze(-2)
        raw_delta = self.out_proj((invariant - base) * gate)
        if class_mask.dim() == 1:
            raw_delta = raw_delta * class_mask.view(1, 1, support_count, 1).to(code.dtype)
        else:
            raw_delta = raw_delta * class_mask[:, None, :, None].to(code.dtype)

        alpha = torch.clamp(self.alpha_m, min=-self.alpha_m_max, max=self.alpha_m_max)
        gated_delta = raw_delta * alpha * conf[:, :, None, None]
        support_norm = base.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        delta_norm = gated_delta.norm(dim=-1, keepdim=True)
        max_delta = support_norm * self.max_delta_norm_ratio
        scale = torch.minimum(torch.ones_like(delta_norm), max_delta / (delta_norm + self.eps))
        applied_delta = gated_delta * scale
        out = F.normalize(base + applied_delta, dim=-1)
        debug = {
            "mess_delta_norm": torch.nan_to_num(applied_delta.norm(dim=-1)),
            "mess_delta_norm_ratio": torch.nan_to_num(
                applied_delta.norm(dim=-1) / support_norm.squeeze(-1).clamp_min(self.eps)),
            "mess_equivariant_shape": tuple(mixed_orbit.shape),
            "mess_alpha": alpha.detach(),
            "class_mask": (class_mask[0] if class_mask.dim() == 2 else class_mask).detach().cpu(),
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
        self.mess_logit = nn.Parameter(torch.tensor(float(mess_weight_init)).clamp_min(eps).log())
        self.fourier_logit = nn.Parameter(torch.tensor(float(fourier_weight_init)).clamp_min(eps).log())
        self.max_branch_weight = float(max_branch_weight)
        self.eps = float(eps)

    def _weights(self, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        raw = torch.stack([
            self.mess_logit.to(device=device, dtype=dtype).exp(),
            self.fourier_logit.to(device=device, dtype=dtype).exp(),
        ])
        weights = raw / raw.sum().clamp_min(self.eps)
        mess_w = torch.clamp(weights[0], min=0.0, max=self.max_branch_weight)
        fourier_w = 1.0 - mess_w
        return mess_w, fourier_w

    def forward(self, base: Tensor, mess_text: Tensor, fourier_text: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        if mess_text.shape != fourier_text.shape or mess_text.shape != base.shape:
            raise ValueError(
                "base, mess_text, and fourier_text must share shape, got "
                f"{tuple(base.shape)}, {tuple(mess_text.shape)}, {tuple(fourier_text.shape)}")
        mess_w, fourier_w = self._weights(base.device, base.dtype)
        fused_delta = (mess_text - base) * mess_w + (fourier_text - base) * fourier_w
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
            alpha_m_max=mess_cfg.get("alpha_m_max", 0.02),
            max_delta_norm_ratio=mess_cfg.get("max_delta_norm_ratio", max_delta_norm_ratio))
        self.fourier_branch = FourierEquivariantTextAdapter(
            support_dim=support_dim,
            code_dim=code_dim,
            apply_to_class_ids=apply_to_class_ids,
            class_names=class_names,
            apply_to_classes=apply_to_classes,
            allow_all_classes=allow_all_classes,
            low_rank=fourier_cfg.get("low_rank", 16),
            alpha_t_init=fourier_cfg.get("alpha_t_init", 0.0),
            alpha_t_max=fourier_cfg.get("alpha_t_max", 0.02),
            max_delta_norm_ratio=fourier_cfg.get("max_delta_norm_ratio", max_delta_norm_ratio))
        self.text_branch_fusion = FocusMessFourierTextFusion(
            mess_weight_init=fusion_cfg.get("mess_weight_init", 0.5),
            fourier_weight_init=fusion_cfg.get("fourier_weight_init", 0.5),
            max_branch_weight=fusion_cfg.get("max_branch_weight", 0.5))

    @property
    def alpha_t(self) -> Tensor:
        return self.fourier_branch.alpha_t

    def forward(
            self,
            text_support: Tensor,
            fourier_cue: Tensor,
            confidence: Optional[Tensor] = None,
            class_ids: Optional[Tensor] = None,
            support_labels: Optional[Tensor] = None,
            class_names: Optional[Sequence[str]] = None) -> Tuple[Tensor, Dict[str, Any]]:
        code = _flatten_code(fourier_cue)
        base = _expand_support(text_support.to(device=code.device, dtype=code.dtype), code.shape[0], code.shape[1])
        labels = class_ids if class_ids is not None else support_labels
        mess_text, mess_debug = self.mess_branch(
            base,
            code,
            confidence,
            support_labels=labels,
            class_names=class_names)
        fourier_text, fourier_debug = self.fourier_branch(
            base,
            code,
            confidence,
            class_ids=class_ids,
            support_labels=support_labels,
            class_names=class_names)
        fused_text, fusion_debug = self.text_branch_fusion(base, mess_text, fourier_text)

        raw_delta = fused_text - base
        support_norm = base.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        delta_norm = raw_delta.norm(dim=-1, keepdim=True)
        max_delta = support_norm * self.max_delta_norm_ratio
        scale = torch.minimum(torch.ones_like(delta_norm), max_delta / (delta_norm + self.eps))
        applied_delta = raw_delta * scale
        conditioned = F.normalize(base + applied_delta, dim=-1)
        debug = {
            "text_delta_norm": torch.nan_to_num(applied_delta.norm(dim=-1)),
            "text_delta_norm_ratio": torch.nan_to_num(
                applied_delta.norm(dim=-1) / support_norm.squeeze(-1).clamp_min(self.eps)),
            "alpha_t": torch.clamp(
                self.alpha_t,
                min=-self.alpha_t_max,
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
        apply_to_classes=eqtext_cfg.get("apply_to_classes", ("small-vehicle",)),
        allow_all_classes=eqtext_cfg.get("allow_all_classes", False),
        alpha_t_init=eqtext_cfg.get("alpha_t_init", 0.0),
        alpha_t_max=eqtext_cfg.get("alpha_t_max", 0.05),
        max_delta_norm_ratio=eqtext_cfg.get("max_delta_norm_ratio", 0.05))
    if "apply_to_class_ids" in eqtext_cfg:
        common["apply_to_class_ids"] = eqtext_cfg["apply_to_class_ids"]
    if adapter_type in {"mess_fourier_dual", "messdet_fourier_dual"}:
        common["alpha_t_max"] = min(float(common["alpha_t_max"]), 0.02)
        common["max_delta_norm_ratio"] = min(float(common["max_delta_norm_ratio"]), 0.03)
        return FocusMessFourierDualTextAdapter(
            **common,
            mess_cfg=eqtext_cfg.get("mess_branch", {}),
            fourier_cfg=eqtext_cfg.get("fourier_branch", {}),
            fusion_cfg=eqtext_cfg.get("text_branch_fusion", {}))
    if adapter_type == "fourier":
        return FourierEquivariantTextAdapter(**common)
    raise ValueError(f"unknown focus eqtext adapter type: {adapter_type}")
```

- [ ] **Step 2: Run the new unit tests**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_focus_mess_fourier_text_branch.py -q
```

Expected: PASS.

- [ ] **Step 3: Commit the implementation**

Run:

```bash
rtk git add M_AD/models/utils/focus_mess_fourier_text_branch.py tests/test_focus_mess_fourier_text_branch.py
rtk git commit -m "feat: add mess fourier dual text support adapter"
```

Expected: commit succeeds and stages only the new module and its tests.

---

### Task 3: Integrate The Adapter Factory Into The Dense Head

**Files:**
- Modify: `M_AD/models/dense_heads/Flex_Rrtmdet_head_v3_1.py`
- Test: `tests/test_focus_mess_fourier_text_branch.py`

- [ ] **Step 1: Replace the direct Fourier text adapter import**

In `M_AD/models/dense_heads/Flex_Rrtmdet_head_v3_1.py`, replace:

```python
from M_AD.models.utils.focus_eqtext_adapter import FourierEquivariantTextAdapter
```

with:

```python
from M_AD.models.utils.focus_mess_fourier_text_branch import (
    build_focus_text_adapter,
)
```

- [ ] **Step 2: Replace the `_init_focus_ovd_modules` eqtext construction**

In `_init_focus_ovd_modules`, replace the `if self.focus_eqtext_enabled:` block that constructs `FourierEquivariantTextAdapter` with:

```python
        if self.focus_eqtext_enabled:
            self.focus_text_adapter = build_focus_text_adapter(
                eqtext_cfg=eqtext_cfg,
                support_dim=self.embed_dims,
                code_dim=2 * len(harmonic_orders),
                class_names=None)
```

Keep the existing `self.focus_dual_fusion_enabled` block unchanged.

- [ ] **Step 3: Run the default adapter factory test**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_focus_mess_fourier_text_branch.py::test_build_focus_text_adapter_preserves_default_fourier_type -q
```

Expected: PASS, proving old `eqtext` configs without `type` still build the existing Fourier adapter.

- [ ] **Step 4: Run an import smoke for the dense head**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python - <<'PY'
from M_AD.models.dense_heads.Flex_Rrtmdet_head_v3_1 import OpenRotatedRTMDetSepBNHead
print(OpenRotatedRTMDetSepBNHead.__name__)
PY
```

Expected output contains:

```text
OpenRotatedRTMDetSepBNHead
```

- [ ] **Step 5: Commit the dense-head integration**

Run:

```bash
rtk git add M_AD/models/dense_heads/Flex_Rrtmdet_head_v3_1.py tests/test_focus_mess_fourier_text_branch.py
rtk git commit -m "feat: route focus eqtext through adapter factory"
```

Expected: commit succeeds and stages only the dense head plus the test file if pytest updated caches are not present in git status.

---

### Task 4: Add Support-Classifier Integration Coverage

**Files:**
- Create: `tests/test_focus_mess_fourier_embed.py`

- [ ] **Step 1: Write the integration tests**

Create `tests/test_focus_mess_fourier_embed.py` with this content:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from M_AD.models.utils.focus_contrastive_embed import (
    OrientationConditionedContrastiveEmbed,
)
from M_AD.models.utils.focus_dual_support_fusion import FocusDualSupportFusion
from M_AD.models.utils.focus_fourier_orientation import FourierOrientationLearner
from M_AD.models.utils.focus_mess_fourier_text_branch import (
    FocusMessFourierDualTextAdapter,
)
from M_AD.models.utils.focus_support_adapter import FourierSupportResidualAdapter


def _baseline_matching_scores(matching_scores, support_labels, support_shot, num_classes, num_in_classes):
    batch, positions, support_count = matching_scores.shape
    labels = support_labels[:, None, :].expand(batch, positions, support_count)
    scores = matching_scores.clone()
    scores[labels < 0] = -10
    in_len = num_in_classes * support_shot
    cls_scores = torch.full((batch, positions, num_classes), float("-inf"), device=scores.device)
    cls_scores[:, :, :num_in_classes] = scores[:, :, :in_len].reshape(
        batch, positions, num_in_classes, support_shot).max(dim=-1).values
    if support_count > in_len:
        cls_scores[:, :, -1] = scores[:, :, in_len:].max(dim=-1).values
    return cls_scores


def _baseline_logits(pred_embeds, support_feats, support_labels, log_scale, bias, support_shot, num_classes, num_in_classes):
    batch, dim, height, width = pred_embeds.shape
    x = pred_embeds.permute(0, 2, 3, 1).reshape(batch, height * width, dim)
    w = F.normalize(support_feats, dim=-1)
    scaled = (x @ w.transpose(-1, -2)) * log_scale.exp() + bias
    cls = _baseline_matching_scores(scaled, support_labels, support_shot, num_classes, num_in_classes)
    return cls.reshape(batch, height, width, cls.shape[-1]).permute(0, 3, 1, 2)


def test_dual_text_zero_weight_matches_focus_ovd_visual_baseline_exactly():
    torch.manual_seed(11)
    batch, dim, height, width = 1, 8, 3, 2
    support_shot = 2
    num_classes = 3
    pred_embeds = torch.randn(batch, dim, height, width)
    support_feats = torch.randn(batch, num_classes * support_shot, dim)
    text_feats = support_feats.clone()
    support_labels = torch.tensor([[0, 0, 1, 1, 2, 2]])
    identity = nn.Identity()
    log_scale = nn.Parameter(torch.tensor([-1.0]))
    bias = nn.Parameter(torch.tensor([-4.0]))
    orientation = FourierOrientationLearner(
        patch_size=3,
        num_angle_bins=18,
        harmonic_orders=(2, 4))
    visual_adapter = FourierSupportResidualAdapter(
        support_dim=dim,
        code_dim=4,
        apply_to_classes=("small-vehicle",),
        alpha_init=0.0)
    text_adapter = FocusMessFourierDualTextAdapter(
        support_dim=dim,
        code_dim=4,
        apply_to_classes=("small-vehicle",),
        alpha_t_init=0.0,
        alpha_t_max=0.02,
        max_delta_norm_ratio=0.03,
        mess_cfg=dict(low_rank=4),
        fourier_cfg=dict(low_rank=4, alpha_t_max=0.02),
        fusion_cfg=dict(max_branch_weight=0.5))
    fusion = FocusDualSupportFusion(
        visual_weight_init=0.98,
        text_weight_init=0.02,
        max_text_weight=0.0)
    focus = OrientationConditionedContrastiveEmbed()

    actual, debug = focus(
        pred_embeds,
        support_feats,
        support_labels,
        visual_fc=identity,
        text_fc=identity,
        log_scale=log_scale,
        bias=bias,
        orientation_learner=orientation,
        support_adapter=visual_adapter,
        text_support_adapter=text_adapter,
        dual_support_fusion=fusion,
        visual_support_feats=support_feats,
        text_support_feats=text_feats,
        eqtext_enabled=True,
        dual_fusion_enabled=True,
        head_residual_enabled=True,
        return_focus_debug=True,
        support_shot=support_shot,
        num_classes=num_classes,
        num_in_classes=num_classes,
        align_style="labelled",
        focus_class_names=("plane", "small-vehicle", "ship"))
    expected = _baseline_logits(
        pred_embeds,
        support_feats,
        support_labels,
        log_scale,
        bias,
        support_shot,
        num_classes,
        num_classes)

    assert torch.equal(actual, expected)
    assert debug["dual_fusion"]["dual_text_weight"] == 0.0
    assert debug["eqtext"]["text_branch_fusion"]["mess_branch_weight"] <= 0.5


def test_dual_text_nonzero_config_remains_bounded():
    fusion = FocusDualSupportFusion(
        visual_weight_init=0.98,
        text_weight_init=0.02,
        max_text_weight=0.02)
    visual = F.normalize(torch.randn(1, 6, 8), dim=-1)
    text = F.normalize(torch.randn(1, 6, 8), dim=-1)

    out, debug = fusion(
        visual,
        text,
        text_enabled=True,
        visual_alpha=torch.tensor(0.05),
        text_alpha=torch.tensor(0.02))

    assert out.shape == visual.shape
    assert 0.0 < debug["dual_text_weight"] <= 0.02
    assert torch.allclose(out.norm(dim=-1), torch.ones_like(out[..., 0]), atol=1e-5)
```

- [ ] **Step 2: Run the integration tests**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_focus_mess_fourier_embed.py -q
```

Expected: PASS.

- [ ] **Step 3: Run the existing FOCUS classifier regression test**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_focus_contrastive_embed.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit the integration tests**

Run:

```bash
rtk git add tests/test_focus_mess_fourier_embed.py
rtk git commit -m "test: cover focus mess fourier support fusion"
```

Expected: commit succeeds and stages only `tests/test_focus_mess_fourier_embed.py`.

---

### Task 5: Add Six-GPU Configs And Safety Tests

**Files:**
- Create: `M_configs/experiments/focus_ovd/focus_ovd_a10_mess_fourier_dual_text_6gpu_zero.py`
- Create: `M_configs/experiments/focus_ovd/focus_ovd_a10_mess_fourier_dual_text_6gpu.py`
- Create: `tests/test_focus_mess_fourier_config_and_launcher.py`

- [ ] **Step 1: Create the zero-equivalence six-GPU config**

Create `M_configs/experiments/focus_ovd/focus_ovd_a10_mess_fourier_dual_text_6gpu_zero.py`:

```python
_base_ = './focus_ovd_a10_sv_only_full.py'

work_dir = 'work_dirs/focus_ovd_a10_mess_fourier_dual_text_6gpu_zero_20260610'

num_gpus = 6
batch_size = 2

trainable_parameters = [
    'bbox_head.focus_support_adapter',
    'bbox_head.focus_text_adapter',
    'bbox_head.focus_dual_support_fusion',
]

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='mmdet.ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='FocusOVDTrainableAuditHook',
        trainable_substrings=trainable_parameters,
        log_path=(
            'resultmd/exp_focus_ovd_20260608/'
            'mess_fourier_dual_text_6gpu/focus_trainable_audit_zero.json'),
        fail_on_unexpected=True,
        priority='VERY_HIGH'),
]

model = dict(
    bbox_head=dict(
        use_focus_ovd=True,
        focus_text_anchor_calibration=dict(enable=False),
        focus_fourier_head_gate=dict(enable=False),
        focus_text_logit_mixer=dict(enable=False),
        focus_ovd=dict(
            enable=True,
            orientation=dict(
                patch_size=7,
                num_angle_bins=36,
                harmonic_orders=(2, 4, 6),
                detach_orientation=True),
            adapter=dict(
                apply_to_classes=('small-vehicle',),
                alpha_init=0.0,
                alpha_max=0.10,
                max_delta_norm_ratio=0.05,
                low_rank=16),
            eqtext=dict(
                enable=True,
                type='mess_fourier_dual',
                apply_to_classes=('small-vehicle',),
                allow_all_classes=False,
                alpha_t_init=0.0,
                alpha_t_max=0.02,
                max_delta_norm_ratio=0.03,
                mess_branch=dict(
                    group_order=8,
                    low_rank=16,
                    alpha_m_init=0.0,
                    alpha_m_max=0.02,
                    max_delta_norm_ratio=0.03),
                fourier_branch=dict(
                    low_rank=16,
                    alpha_t_init=0.0,
                    alpha_t_max=0.02,
                    max_delta_norm_ratio=0.03),
                text_branch_fusion=dict(
                    mess_weight_init=0.5,
                    fourier_weight_init=0.5,
                    max_branch_weight=0.5)),
            dual_fusion=dict(
                enable=True,
                visual_weight_init=0.98,
                text_weight_init=0.02,
                max_text_weight=0.0))))

train_dataloader = dict(
    batch_size=batch_size,
    sampler=dict(
        batch_size=batch_size,
        num_gpus=num_gpus))
```

- [ ] **Step 2: Create the conservative nonzero training config**

Create `M_configs/experiments/focus_ovd/focus_ovd_a10_mess_fourier_dual_text_6gpu.py`:

```python
_base_ = './focus_ovd_a10_mess_fourier_dual_text_6gpu_zero.py'

work_dir = 'work_dirs/focus_ovd_a10_mess_fourier_dual_text_6gpu_20260610'

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='mmdet.ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='FocusOVDTrainableAuditHook',
        trainable_substrings=[
            'bbox_head.focus_support_adapter',
            'bbox_head.focus_text_adapter',
            'bbox_head.focus_dual_support_fusion',
        ],
        log_path=(
            'resultmd/exp_focus_ovd_20260608/'
            'mess_fourier_dual_text_6gpu/focus_trainable_audit.json'),
        fail_on_unexpected=True,
        priority='VERY_HIGH'),
]

model = dict(
    bbox_head=dict(
        focus_ovd=dict(
            dual_fusion=dict(
                enable=True,
                visual_weight_init=0.98,
                text_weight_init=0.02,
                max_text_weight=0.02))))
```

- [ ] **Step 3: Write config safety tests**

Create `tests/test_focus_mess_fourier_config_and_launcher.py` with the initial config tests:

```python
from pathlib import Path

from mmengine.config import Config

from M_AD.engine.runner.meta_remove_runer import apply_trainable_parameter_filter


ZERO_CONFIG = Path("M_configs/experiments/focus_ovd/focus_ovd_a10_mess_fourier_dual_text_6gpu_zero.py")
TRAIN_CONFIG = Path("M_configs/experiments/focus_ovd/focus_ovd_a10_mess_fourier_dual_text_6gpu.py")


def test_dual_text_zero_config_is_six_gpu_and_disables_failed_logit_modules():
    cfg = Config.fromfile(ZERO_CONFIG)
    bbox_head = cfg.model["bbox_head"]
    focus = bbox_head["focus_ovd"]

    assert cfg.num_gpus == 6
    assert cfg.train_dataloader["sampler"]["num_gpus"] == 6
    assert cfg.trainable_parameters == [
        "bbox_head.focus_support_adapter",
        "bbox_head.focus_text_adapter",
        "bbox_head.focus_dual_support_fusion",
    ]
    assert bbox_head["focus_text_anchor_calibration"]["enable"] is False
    assert bbox_head["focus_fourier_head_gate"]["enable"] is False
    assert bbox_head["focus_text_logit_mixer"]["enable"] is False
    assert focus["eqtext"]["type"] == "mess_fourier_dual"
    assert focus["eqtext"]["mess_branch"]["group_order"] == 8
    assert focus["dual_fusion"]["max_text_weight"] == 0.0


def test_dual_text_train_config_keeps_text_weight_cap_at_two_percent():
    cfg = Config.fromfile(TRAIN_CONFIG)

    assert cfg.num_gpus == 6
    assert cfg.model["bbox_head"]["focus_ovd"]["dual_fusion"]["max_text_weight"] == 0.02


def test_dual_text_trainable_allowlist_keeps_only_focus_modules_trainable():
    import torch.nn as nn

    model = nn.Module()
    model.backbone = nn.Linear(2, 2)
    model.bbox_head = nn.Module()
    model.bbox_head.focus_support_adapter = nn.Linear(2, 2)
    model.bbox_head.focus_text_adapter = nn.Linear(2, 2)
    model.bbox_head.focus_dual_support_fusion = nn.Linear(2, 2)
    model.bbox_head.focus_text_logit_mixer = nn.Linear(2, 2)
    model.bbox_head.focus_fourier_head_gate = nn.Linear(2, 2)
    model.bbox_head.focus_text_anchor_calibration = nn.Linear(2, 2)

    summary = apply_trainable_parameter_filter(
        model,
        trainable_substrings=[
            "bbox_head.focus_support_adapter",
            "bbox_head.focus_text_adapter",
            "bbox_head.focus_dual_support_fusion",
        ])
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}

    assert trainable == {
        "bbox_head.focus_support_adapter.weight",
        "bbox_head.focus_support_adapter.bias",
        "bbox_head.focus_text_adapter.weight",
        "bbox_head.focus_text_adapter.bias",
        "bbox_head.focus_dual_support_fusion.weight",
        "bbox_head.focus_dual_support_fusion.bias",
    }
    assert all("focus_text_logit_mixer" not in name for name in summary["trainable"])
    assert all("focus_fourier_head_gate" not in name for name in summary["trainable"])
    assert all("focus_text_anchor_calibration" not in name for name in summary["trainable"])
```

- [ ] **Step 4: Run config safety tests**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_focus_mess_fourier_config_and_launcher.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit configs and config tests**

Run:

```bash
rtk git add M_configs/experiments/focus_ovd/focus_ovd_a10_mess_fourier_dual_text_6gpu_zero.py M_configs/experiments/focus_ovd/focus_ovd_a10_mess_fourier_dual_text_6gpu.py tests/test_focus_mess_fourier_config_and_launcher.py
rtk git commit -m "config: add focus mess fourier dual text six gpu runs"
```

Expected: commit succeeds and stages only the two configs plus config tests.

---

### Task 6: Add Six-Card Launcher And Launcher Tests

**Files:**
- Create: `M_Tools/experiments/run_focus_mess_fourier_dual_text_6gpu.sh`
- Modify: `tests/test_focus_mess_fourier_config_and_launcher.py`

- [ ] **Step 1: Create the six-card launcher**

Create `M_Tools/experiments/run_focus_mess_fourier_dual_text_6gpu.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-M_configs/experiments/focus_ovd/focus_ovd_a10_mess_fourier_dual_text_6gpu.py}
PYTHON=${PYTHON:-/data/zcy/anaconda3/envs/openrsd/bin/python}
GPUS=${GPUS:-0,1,2,3,4,5}
NPROC=${NPROC:-6}

export CUDA_VISIBLE_DEVICES="${GPUS}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONNOUSERSITE=1

exec "${PYTHON}" -m torch.distributed.run \
  --nproc_per_node="${NPROC}" \
  tools/train.py "${CONFIG}" \
  --launcher pytorch
```

- [ ] **Step 2: Make the launcher executable**

Run:

```bash
rtk chmod +x M_Tools/experiments/run_focus_mess_fourier_dual_text_6gpu.sh
```

Expected: command exits with code 0.

- [ ] **Step 3: Extend the launcher safety tests**

Append these tests to `tests/test_focus_mess_fourier_config_and_launcher.py`:

```python

LAUNCHER = Path("M_Tools/experiments/run_focus_mess_fourier_dual_text_6gpu.sh")


def test_dual_text_launcher_disables_unreliable_a40_nccl_transports():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "export NCCL_P2P_DISABLE=1" in text
    assert "export NCCL_IB_DISABLE=1" in text
    assert "--nproc_per_node=\"${NPROC}\"" in text
    assert "NPROC=${NPROC:-6}" in text
    assert "0,1,2,3,4,5" in text
    assert "focus_ovd_a10_mess_fourier_dual_text_6gpu.py" in text


def test_dual_text_launcher_is_executable():
    assert LAUNCHER.stat().st_mode & 0o111
```

- [ ] **Step 4: Run launcher tests**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_focus_mess_fourier_config_and_launcher.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the launcher**

Run:

```bash
rtk git add M_Tools/experiments/run_focus_mess_fourier_dual_text_6gpu.sh tests/test_focus_mess_fourier_config_and_launcher.py
rtk git commit -m "tools: add focus mess fourier dual text six gpu launcher"
```

Expected: commit succeeds and stages only the launcher plus launcher tests.

---

### Task 7: Full Local Verification Before Training

**Files:**
- Read: `M_AD/models/utils/focus_mess_fourier_text_branch.py`
- Read: `M_AD/models/dense_heads/Flex_Rrtmdet_head_v3_1.py`
- Read: `M_configs/experiments/focus_ovd/focus_ovd_a10_mess_fourier_dual_text_6gpu_zero.py`
- Read: `M_configs/experiments/focus_ovd/focus_ovd_a10_mess_fourier_dual_text_6gpu.py`

- [ ] **Step 1: Run focused unit and integration tests**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_focus_mess_fourier_text_branch.py tests/test_focus_mess_fourier_embed.py tests/test_focus_mess_fourier_config_and_launcher.py -q
```

Expected: PASS.

- [ ] **Step 2: Run FOCUS regression tests touched by the new path**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_focus_contrastive_embed.py tests/test_focus_eqtext_modules.py tests/test_focus_support_adapter.py tests/test_focus_ovd_freeze_and_launcher.py -q
```

Expected: PASS.

- [ ] **Step 3: Import both configs**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python - <<'PY'
from mmengine.config import Config
for path in [
    "M_configs/experiments/focus_ovd/focus_ovd_a10_mess_fourier_dual_text_6gpu_zero.py",
    "M_configs/experiments/focus_ovd/focus_ovd_a10_mess_fourier_dual_text_6gpu.py",
]:
    cfg = Config.fromfile(path)
    print(path, cfg.num_gpus, cfg.train_dataloader["sampler"]["num_gpus"])
PY
```

Expected output contains:

```text
M_configs/experiments/focus_ovd/focus_ovd_a10_mess_fourier_dual_text_6gpu_zero.py 6 6
M_configs/experiments/focus_ovd/focus_ovd_a10_mess_fourier_dual_text_6gpu.py 6 6
```

- [ ] **Step 4: Check that failed modules are not trainable in the new config**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python - <<'PY'
from mmengine.config import Config
cfg = Config.fromfile("M_configs/experiments/focus_ovd/focus_ovd_a10_mess_fourier_dual_text_6gpu.py")
print(cfg.trainable_parameters)
print(cfg.model["bbox_head"]["focus_text_anchor_calibration"]["enable"])
print(cfg.model["bbox_head"]["focus_fourier_head_gate"]["enable"])
print(cfg.model["bbox_head"]["focus_text_logit_mixer"]["enable"])
PY
```

Expected output contains:

```text
['bbox_head.focus_support_adapter', 'bbox_head.focus_text_adapter', 'bbox_head.focus_dual_support_fusion']
False
False
False
```

- [ ] **Step 5: Commit any verification-only corrections**

If a verification correction was made, stage only the files edited in that correction:

```bash
rtk git add M_AD/models/utils/focus_mess_fourier_text_branch.py M_AD/models/dense_heads/Flex_Rrtmdet_head_v3_1.py M_configs/experiments/focus_ovd/focus_ovd_a10_mess_fourier_dual_text_6gpu_zero.py M_configs/experiments/focus_ovd/focus_ovd_a10_mess_fourier_dual_text_6gpu.py tests/test_focus_mess_fourier_text_branch.py tests/test_focus_mess_fourier_embed.py tests/test_focus_mess_fourier_config_and_launcher.py M_Tools/experiments/run_focus_mess_fourier_dual_text_6gpu.sh
rtk git commit -m "fix: stabilize focus mess fourier dual text verification"
```

Expected: commit succeeds only if there were verification corrections. If there were no corrections, skip this step.

---

### Task 8: Training Smoke Commands

**Files:**
- Execute: `M_Tools/experiments/run_focus_mess_fourier_dual_text_6gpu.sh`
- Read output logs under the configured `work_dir`

- [ ] **Step 1: Run a config build smoke without launching a long job**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python tools/train.py M_configs/experiments/focus_ovd/focus_ovd_a10_mess_fourier_dual_text_6gpu_zero.py --cfg-options train_cfg.max_epochs=0
```

Expected: config builds and exits before long training. If this command fails because the runner does not accept `max_epochs=0`, run Step 2 directly and stop it after the first logged iteration.

- [ ] **Step 2: Launch the zero-equivalence six-card smoke**

Run:

```bash
rtk bash M_Tools/experiments/run_focus_mess_fourier_dual_text_6gpu.sh M_configs/experiments/focus_ovd/focus_ovd_a10_mess_fourier_dual_text_6gpu_zero.py
```

Expected:

- Command uses six visible cards from `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5`.
- Logs include the trainability audit JSON at `resultmd/exp_focus_ovd_20260608/mess_fourier_dual_text_6gpu/focus_trainable_audit_zero.json`.
- Audit JSON has no `unexpected_trainable` entries.
- The job reaches at least one training iteration without NCCL P2P/IB failure.

- [ ] **Step 3: Launch the conservative nonzero six-card run after zero smoke passes**

Run:

```bash
rtk bash M_Tools/experiments/run_focus_mess_fourier_dual_text_6gpu.sh M_configs/experiments/focus_ovd/focus_ovd_a10_mess_fourier_dual_text_6gpu.py
```

Expected:

- Command uses the same NCCL settings.
- `focus_trainable_audit.json` reports only `bbox_head.focus_support_adapter`, `bbox_head.focus_text_adapter`, and `bbox_head.focus_dual_support_fusion` as trainable.
- Early logs contain no direct-logit module trainability.
- Stop and inspect if small-vehicle detections explode in the first validation pass.

---

## Self-Review Notes

- Spec coverage: the plan covers the MessDet-style raw text branch, Fourier text branch reuse, text-branch fusion, visual support fusion, six-card configs, launcher, failed-module exclusion, zero-equivalence, and rollback-by-config.
- Scope: the plan does not replace the image detector and does not delete historical modules.
- Type consistency: the new wrapper matches the existing `text_support_adapter(text_support, fourier_cue, confidence, support_labels, class_names)` call pattern used by `OrientationConditionedContrastiveEmbed`.
