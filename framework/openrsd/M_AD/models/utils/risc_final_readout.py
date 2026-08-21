import math
from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class RISCFinalReadoutAdapter(nn.Module):
    """Class-shared bounded residual for the final semantic readout."""

    def __init__(
            self,
            embed_dims: int,
            rank: int = 8,
            enabled: bool = False,
            init_alpha: float = 0.0,
            max_alpha: float = 0.1,
            max_delta_norm_ratio: float = 0.05,
            init_seed: int = 20260822,
            eps: float = 1e-8) -> None:
        super().__init__()
        if type(embed_dims) is not int or embed_dims <= 0:
            raise ValueError('embed_dims must be a positive integer')
        if type(rank) is not int or rank <= 0 or rank > embed_dims:
            raise ValueError('rank must be in [1, embed_dims]')
        if type(enabled) is not bool:
            raise TypeError('enabled must be a bool')
        if not math.isfinite(float(max_alpha)) or float(max_alpha) <= 0:
            raise ValueError('max_alpha must be finite and positive')
        if (not math.isfinite(float(init_alpha))
                or abs(float(init_alpha)) >= float(max_alpha)):
            raise ValueError('init_alpha must satisfy abs(init_alpha) < max_alpha')
        if (not math.isfinite(float(max_delta_norm_ratio))
                or float(max_delta_norm_ratio) <= 0):
            raise ValueError(
                'max_delta_norm_ratio must be finite and positive')
        if type(init_seed) is not int:
            raise TypeError('init_seed must be an integer')
        if not math.isfinite(float(eps)) or float(eps) <= 0:
            raise ValueError('eps must be finite and positive')

        self.embed_dims = embed_dims
        self.rank = rank
        self.enabled = enabled
        self.max_alpha = float(max_alpha)
        self.max_delta_norm_ratio = float(max_delta_norm_ratio)
        self.eps = float(eps)
        self.norm = nn.LayerNorm(embed_dims, elementwise_affine=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed)
            self.down = nn.Linear(embed_dims, rank, bias=False)
            self.up = nn.Linear(rank, embed_dims, bias=False)
            nn.init.xavier_uniform_(self.down.weight)
            nn.init.xavier_uniform_(self.up.weight)

        ratio = float(init_alpha) / self.max_alpha
        self.raw_alpha = nn.Parameter(
            torch.tensor(math.atanh(ratio), dtype=torch.float32))
        self._debug: Dict[str, float | bool] = {
            'enabled': self.enabled,
            'alpha': float(init_alpha),
            'max_delta_norm_ratio': 0.0,
        }

    def _validate_input(self, value: Tensor) -> None:
        if not isinstance(value, Tensor):
            raise TypeError('RISC final readout input must be a Tensor')
        if value.dim() != 4:
            raise ValueError('RISC final readout input must be NCHW')
        if value.shape[1] != self.embed_dims:
            raise ValueError(
                'RISC final readout channel count must equal embed_dims')
        if not value.is_floating_point():
            raise TypeError('RISC final readout input must be floating point')

    def forward(self, value: Tensor) -> Tensor:
        self._validate_input(value)
        if not self.enabled:
            self._debug = {
                'enabled': False,
                'alpha': 0.0,
                'max_delta_norm_ratio': 0.0,
            }
            return value

        channels_last = value.permute(0, 2, 3, 1)
        nuisance = self.up(F.gelu(self.down(self.norm(channels_last))))
        alpha = self.max_alpha * torch.tanh(self.raw_alpha)
        raw_delta = alpha.to(dtype=nuisance.dtype) * nuisance

        base_norm = channels_last.norm(dim=-1, keepdim=True).clamp_min(
            self.eps)
        raw_norm = raw_delta.norm(dim=-1, keepdim=True)
        limit = base_norm * self.max_delta_norm_ratio
        clip_scale = torch.minimum(
            torch.ones_like(raw_norm), limit / (raw_norm + self.eps))
        delta = raw_delta * clip_scale
        output = channels_last - delta

        observed_ratio = delta.detach().norm(dim=-1) / base_norm.detach(
        ).squeeze(-1)
        self._debug = {
            'enabled': True,
            'alpha': float(alpha.detach().cpu().item()),
            'max_delta_norm_ratio': float(
                observed_ratio.max().cpu().item())
            if observed_ratio.numel() else 0.0,
        }
        return output.permute(0, 3, 1, 2).contiguous()

    def debug_state(self) -> Dict[str, float | bool]:
        return dict(self._debug)
