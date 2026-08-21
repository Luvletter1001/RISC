#!/usr/bin/env python3
"""Lightweight rotation-aware low-rank adapter on dense visual features."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

FeatX = Union[torch.Tensor, Sequence[torch.Tensor]]


class LowRankRotationAdapter(nn.Module):
    """y = x + alpha * B @ A @ LayerNorm(x)"""

    def __init__(self, dim: int, rank: int = 16, alpha: float = 0.3):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.alpha = alpha
        self.ln = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.alpha == 0:
            return x
        h = self.up(self.down(self.ln(x)))
        return x + self.alpha * h

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MLPRotationAdapter(nn.Module):
    def __init__(self, dim: int, hidden: int = 64, alpha: float = 0.3):
        super().__init__()
        self.alpha = alpha
        self.ln = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.alpha == 0:
            return x
        return x + self.alpha * self.net(self.ln(x))


def save_adapter(adapter: nn.Module, path: Path, meta: Optional[dict] = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state_dict=adapter.state_dict(), meta=meta or {})
    torch.save(payload, path)


def load_adapter(path: Path, factory, device='cpu') -> nn.Module:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    adapter = factory()
    adapter.load_state_dict(payload['state_dict'])
    adapter.to(device)
    return adapter


def apply_rotation_adapter_to_feats(feat_x: FeatX, adapter: nn.Module) -> FeatX:
    """Residual broadcast: spatial_feat += (adapter(pool(feat)) - pool(feat))."""
    if isinstance(feat_x, (list, tuple)):
        return type(feat_x)(apply_rotation_adapter_to_feats(t, adapter) for t in feat_x)
    if not isinstance(feat_x, torch.Tensor) or feat_x.dim() != 4:
        return feat_x
    b, c, h, w = feat_x.shape
    dim = getattr(adapter, 'dim', c)
    if c != dim:
        return feat_x
    pooled = F.adaptive_avg_pool2d(feat_x, 1).flatten(1)
    adapted = adapter(pooled)
    delta = (adapted - pooled).view(b, c, 1, 1)
    return feat_x + delta


def resolve_adapter_ckpt(ctx: Any, ckpt_name: str = 'adapter_best_sv_repair.pth') -> Optional[Path]:
    ckpt_dir = Path(getattr(ctx, 'adapter_ckpt_dir', Path(ctx.work_dir) / 'adapter_ckpts'))
    for name in (ckpt_name, 'adapter_latest.pth', 'adapter_best_no_drift.pth'):
        p = ckpt_dir / name
        if p.is_file():
            return p
    return None


def load_trained_rotation_adapter(
        ctx: Any,
        device: Union[str, torch.device] = 'cpu',
        ckpt_name: str = 'adapter_best_sv_repair.pth',
        dim: int = 256,
) -> Optional[LowRankRotationAdapter]:
    path = resolve_adapter_ckpt(ctx, ckpt_name)
    if path is None:
        return None
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    meta = payload.get('meta') or {}
    rank = int(meta.get('rank', 16))
    alpha = float(meta.get('alpha', 0.3))
    return load_adapter(
        path,
        lambda: LowRankRotationAdapter(dim, rank=rank, alpha=alpha),
        device=device,
    )


def attach_rotation_adapter(stack: Any, ctx: Any, device: Union[str, torch.device] = 'cpu') -> Any:
    """Attach trained adapter to RepairStack; returns stack unchanged if ckpt missing."""
    adapter = load_trained_rotation_adapter(ctx, device=device)
    if adapter is not None:
        stack.rotation_adapter = adapter
    return stack
