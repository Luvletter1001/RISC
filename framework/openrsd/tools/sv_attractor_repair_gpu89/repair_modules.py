#!/usr/bin/env python
"""Trainable repair modules (original OpenRSD weights stay frozen)."""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassWiseCalibration(nn.Module):
    """logit' = logit / tau_c + bias_c"""

    def __init__(self, num_classes: int, train_temperature: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.bias = nn.Parameter(torch.zeros(num_classes))
        if train_temperature:
            self.raw_tau = nn.Parameter(torch.zeros(num_classes))
        else:
            self.register_buffer('raw_tau', torch.zeros(num_classes))
        self.train_temperature = train_temperature

    def tau(self) -> torch.Tensor:
        return F.softplus(self.raw_tau) + 0.5

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: (N, C) or (B, C, H, W)
        if logits.dim() == 4:
            b, c, h, w = logits.shape
            x = logits.permute(0, 2, 3, 1).reshape(-1, c)
            x = x / self.tau().unsqueeze(0) + self.bias.unsqueeze(0)
            return x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return logits / self.tau().unsqueeze(0) + self.bias.unsqueeze(0)

    def set_sv_bias_only(self, sv_idx: int, bias_val: float):
        with torch.no_grad():
            self.bias.fill_(0)
            self.bias[sv_idx] = bias_val


class NegativePrototypeBank(nn.Module):
    """Background negative prototypes; subtract or filter at inference."""

    def __init__(self, dim: int, num_proto: int = 8):
        super().__init__()
        self.num_proto = num_proto
        self.prototypes = nn.Parameter(torch.randn(num_proto, dim) * 0.02)
        self.log_lambda = nn.Parameter(torch.tensor(0.0))
        self.delta = nn.Parameter(torch.tensor(0.0))

    @property
    def lambda_neg(self) -> torch.Tensor:
        return F.softplus(self.log_lambda)

    def neg_logits(self, visual_feat: torch.Tensor) -> torch.Tensor:
        # visual_feat: (N, D) normalized space
        proto = F.normalize(self.prototypes, dim=-1)
        feat = F.normalize(visual_feat, dim=-1)
        return (feat @ proto.t()).max(dim=-1).values

    def adjust_logits(self, logits: torch.Tensor, visual_feat: torch.Tensor) -> torch.Tensor:
        neg = self.neg_logits(visual_feat)
        return logits - self.lambda_neg * neg.unsqueeze(-1)

    def diversity_loss(self) -> torch.Tensor:
        p = F.normalize(self.prototypes, dim=-1)
        sim = p @ p.t()
        mask = ~torch.eye(self.num_proto, device=sim.device, dtype=torch.bool)
        return sim[mask].abs().mean()


class AntiHubLevelAlpha(nn.Module):
    """x' = x - alpha_l * proj_sv(x) per FPN level index."""

    def __init__(self, num_levels: int = 3):
        super().__init__()
        self.raw_alpha = nn.Parameter(torch.full((num_levels,), -2.0))

    def alpha(self, level: int) -> torch.Tensor:
        return torch.sigmoid(self.raw_alpha[level])

    def debias(self, x: torch.Tensor, sv_direction: torch.Tensor, level: int) -> torch.Tensor:
        sv = F.normalize(sv_direction, dim=-1)
        proj = (x * sv).sum(dim=-1, keepdim=True) * sv
        return x - self.alpha(level) * proj


class GatedAntiHub(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([1.0, 0.0]))
        self.b = nn.Parameter(torch.tensor(-1.0))

    def alpha_from_margin(self, sv_score: torch.Tensor, second_score: torch.Tensor) -> torch.Tensor:
        margin = sv_score - second_score
        return torch.sigmoid(self.w[0] * margin + self.w[1] * (sv_score - 0.5) + self.b)

    def debias(self, x: torch.Tensor, sv_direction: torch.Tensor,
               sv_score: torch.Tensor, second_score: torch.Tensor) -> torch.Tensor:
        a = self.alpha_from_margin(sv_score, second_score).unsqueeze(-1)
        sv = F.normalize(sv_direction, dim=-1)
        proj = (x * sv).sum(dim=-1, keepdim=True) * sv
        return x - a * proj


class LowRankAdapter(nn.Module):
    def __init__(self, dim: int, rank: int = 4):
        super().__init__()
        self.u = nn.Parameter(torch.zeros(dim, rank))
        self.v = nn.Parameter(torch.zeros(rank, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x @ self.v.t() @ self.u.t()


class RepairStack(nn.Module):
    """Composable repairs applied at logits and/or visual features."""

    def __init__(self):
        super().__init__()
        self.calibration: Optional[ClassWiseCalibration] = None
        self.neg_bank: Optional[NegativePrototypeBank] = None
        self.antihub_alpha: Optional[AntiHubLevelAlpha] = None
        self.antihub_gate: Optional[GatedAntiHub] = None
        self.lowrank: Optional[LowRankAdapter] = None
        self.fixed_alpha: Optional[float] = None
        self.grid_sv_bias: Optional[float] = None
        self.rotation_adapter: Optional[nn.Module] = None
        self.sv_idx: int = 10
        self.embed_dim: int = 256

    def apply_logits(self, logits: torch.Tensor, visual_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.grid_sv_bias is not None and self.calibration is None:
            bias = torch.zeros(logits.shape[-1], device=logits.device, dtype=logits.dtype)
            bias[self.sv_idx] = self.grid_sv_bias
            logits = logits + bias
        if self.calibration is not None:
            logits = self.calibration(logits)
        if self.neg_bank is not None and visual_feat is not None:
            logits = self.neg_bank.adjust_logits(logits, visual_feat)
        return logits

    def state_dict_trainable_only(self) -> Dict:
        return {k: v for k, v in self.state_dict().items() if v.numel() > 0}
