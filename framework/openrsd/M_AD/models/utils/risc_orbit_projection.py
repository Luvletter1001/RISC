from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _flatten_code(code: Tensor) -> Tensor:
    if code.dim() == 4:
        return code.reshape(code.shape[0], -1, code.shape[-1])
    if code.dim() == 3:
        return code
    raise ValueError(
        f"fourier_code must be [B,N,D] or [B,H,W,D], got {tuple(code.shape)}")


def _flatten_confidence(confidence: Tensor) -> Tensor:
    if confidence.dim() == 3:
        return confidence.reshape(confidence.shape[0], -1)
    if confidence.dim() == 2:
        return confidence
    raise ValueError(
        f"confidence must be [B,N] or [B,H,W], got {tuple(confidence.shape)}")


class RISCOrbitLowRankProjection(nn.Module):
    """Class-shared orientation-conditioned semantic nuisance projection.

    For an orthonormal basis ``B``, rank gate ``g`` and strength ``a``, the
    adapter applies ``t' = t - a * c * B diag(g) B^T t``.  The same symmetric
    operator can be interpreted as acting on the query in the semantic dot
    product, while leaving all localization features untouched.
    """

    def __init__(
            self,
            support_dim: int,
            code_dim: int,
            rank: int = 8,
            init_strength: float = 0.0,
            max_strength: float = 0.10,
            max_delta_norm_ratio: float = 0.05,
            normalize_after_projection: bool = True,
            basis_seed: int = 20260807,
            eps: float = 1e-8) -> None:
        super().__init__()
        self.support_dim = int(support_dim)
        self.code_dim = int(code_dim)
        self.rank = min(int(rank), self.support_dim)
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.max_strength = float(max_strength)
        self.max_delta_norm_ratio = float(max_delta_norm_ratio)
        self.normalize_after_projection = bool(normalize_after_projection)
        self.eps = float(eps)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(basis_seed))
        initial = torch.randn(
            self.support_dim, self.rank, generator=generator)
        initial = torch.linalg.qr(initial, mode="reduced").Q
        self.basis = nn.Parameter(initial)
        self.code_to_rank = nn.Linear(self.code_dim, self.rank)
        nn.init.zeros_(self.code_to_rank.weight)
        nn.init.zeros_(self.code_to_rank.bias)

        ratio = float(init_strength) / max(self.max_strength, self.eps)
        ratio = max(min(ratio, 1.0 - 1e-6), -1.0 + 1e-6)
        self.raw_strength = nn.Parameter(
            torch.tensor(float(torch.atanh(torch.tensor(ratio)))))

    def _expand_support(self, support: Tensor, batch: int,
                        positions: int) -> Tensor:
        if support.dim() == 2:
            return support[None, None].expand(
                batch, positions, support.shape[0], support.shape[1])
        if support.dim() == 3:
            if support.shape[0] not in (1, batch):
                raise ValueError(
                    f"support batch={support.shape[0]} does not match code batch={batch}")
            if support.shape[0] == 1 and batch != 1:
                support = support.expand(batch, support.shape[1], support.shape[2])
            return support[:, None].expand(
                batch, positions, support.shape[1], support.shape[2])
        raise ValueError(
            f"support_feats must be [M,D] or [B,M,D], got {tuple(support.shape)}")

    def _valid_support_mask(self, support_labels: Optional[Tensor],
                            batch: int, count: int,
                            device: torch.device) -> Tensor:
        if support_labels is None:
            return torch.ones(batch, count, dtype=torch.bool, device=device)
        labels = support_labels.to(device=device)
        if labels.dim() == 1:
            labels = labels[None, :]
        if labels.dim() != 2 or labels.shape[1] != count:
            raise ValueError(
                f"support_labels must be [M] or [B,M] with M={count}, got {tuple(labels.shape)}")
        if labels.shape[0] == 1 and batch != 1:
            labels = labels.expand(batch, count)
        if labels.shape[0] != batch:
            raise ValueError(
                f"support_labels batch={labels.shape[0]} does not match code batch={batch}")
        return labels >= 0

    def _inter_class_cosine(self, support: Tensor) -> Tensor:
        normalized = F.normalize(support, dim=-1)
        cosine = normalized @ normalized.transpose(-1, -2)
        count = cosine.shape[-1]
        if count <= 1:
            return cosine.new_zeros(cosine.shape[:-2])
        off_diagonal = ~torch.eye(
            count, dtype=torch.bool, device=cosine.device)
        return cosine[..., off_diagonal].reshape(
            *cosine.shape[:-2], -1).mean(dim=-1)

    def forward(
            self,
            support_feats: Tensor,
            fourier_code: Tensor,
            confidence: Tensor,
            support_labels: Optional[Tensor] = None,
            class_names: Optional[Sequence[str]] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        del class_names  # The projection is deliberately class-shared.
        code = _flatten_code(fourier_code)
        confidence_flat = _flatten_confidence(confidence)
        batch, positions, code_dim = code.shape
        if code_dim != self.code_dim:
            raise ValueError(f"expected code_dim={self.code_dim}, got {code_dim}")
        if confidence_flat.shape != (batch, positions):
            raise ValueError(
                "confidence positions do not match fourier_code: "
                f"{tuple(confidence_flat.shape)} vs {(batch, positions)}")

        base = self._expand_support(
            support_feats.to(device=code.device, dtype=code.dtype),
            batch, positions)
        if base.shape[-1] != self.support_dim:
            raise ValueError(
                f"expected support_dim={self.support_dim}, got {base.shape[-1]}")

        basis = torch.linalg.qr(self.basis, mode="reduced").Q
        rank_gate = torch.sigmoid(self.code_to_rank(code))
        coefficients = torch.matmul(base, basis)
        nuisance = torch.matmul(
            coefficients * rank_gate.unsqueeze(-2), basis.transpose(0, 1))

        strength = self.max_strength * torch.tanh(self.raw_strength)
        raw_delta = -strength * nuisance
        raw_delta = raw_delta * confidence_flat.clamp(0.0, 1.0)[..., None, None]

        support_count = base.shape[-2]
        valid_mask = self._valid_support_mask(
            support_labels, batch, support_count, code.device)
        raw_delta = raw_delta * valid_mask[:, None, :, None].to(code.dtype)

        base_norm = base.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        raw_norm = raw_delta.norm(dim=-1, keepdim=True)
        max_delta = base_norm * self.max_delta_norm_ratio
        clip_scale = torch.minimum(
            torch.ones_like(raw_norm), max_delta / (raw_norm + self.eps))
        delta = raw_delta * clip_scale
        delta_norm_ratio = delta.norm(dim=-1) / base_norm.squeeze(-1)

        conditioned = base + delta
        if (self.normalize_after_projection
                and bool(torch.any(delta.detach() != 0))):
            conditioned = F.normalize(conditioned, dim=-1)

        debug = {
            "projection_strength": strength.detach(),
            "rank_gate_mean": rank_gate.detach().mean(),
            "delta_norm": delta.detach().norm(dim=-1),
            "delta_norm_ratio": torch.nan_to_num(
                delta_norm_ratio.detach(), nan=0.0),
            "class_mask": valid_mask[0].detach().cpu(),
            "inter_class_cos_before": self._inter_class_cosine(base).detach(),
            "inter_class_cos_after": self._inter_class_cosine(conditioned).detach(),
        }
        return conditioned, debug
