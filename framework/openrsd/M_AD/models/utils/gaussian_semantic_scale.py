"""Gaussian semantic-scale logit utilities.

G-S3C models ``log(area) | class`` as a class-conditional Gaussian.  It is
intentionally separate from Gaussian-IoU/GWD/NWD style bbox similarity.
"""

from __future__ import annotations

import csv
import math
from os import PathLike
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def load_class_log_area_priors(
        priors_csv: str | PathLike,
        class_names: Sequence[str],
        num_classes: int,
) -> tuple[Tensor, Tensor, Tensor, list[str]]:
    """Load class-aligned Gaussian log-area priors from a CSV file."""
    class_names = tuple(str(name) for name in class_names)
    if len(class_names) < int(num_classes):
        class_names = class_names + tuple(
            str(idx) for idx in range(len(class_names), int(num_classes)))

    log_mean = torch.zeros(int(num_classes), dtype=torch.float32)
    log_std = torch.ones(int(num_classes), dtype=torch.float32)
    valid = torch.zeros(int(num_classes), dtype=torch.bool)
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}

    with open(priors_csv, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            cls_name = row.get('class')
            if cls_name not in name_to_idx:
                continue
            cls_idx = name_to_idx[cls_name]
            mean = float(row.get('log_area_mean') or 0.0)
            std = max(float(row.get('log_area_std') or 0.0), 1e-6)
            log_mean[cls_idx] = mean
            log_std[cls_idx] = std
            valid[cls_idx] = True
    return log_mean, log_std, valid, list(class_names)


def load_class_geometry_priors(
        priors_csv: str | PathLike,
        class_names: Sequence[str],
        num_classes: int,
) -> tuple[Tensor, Tensor, Tensor, list[str]]:
    """Load class-aligned diagonal Gaussian geometry priors.

    The geometry vector is ``[log(area), log(long_side / short_side)]``.  This
    is a semantic support prior, not a Gaussian box-overlap representation.
    """
    class_names = tuple(str(name) for name in class_names)
    if len(class_names) < int(num_classes):
        class_names = class_names + tuple(
            str(idx) for idx in range(len(class_names), int(num_classes)))

    means = torch.zeros((int(num_classes), 2), dtype=torch.float32)
    stds = torch.ones((int(num_classes), 2), dtype=torch.float32)
    valid = torch.zeros(int(num_classes), dtype=torch.bool)
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}

    with open(priors_csv, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            cls_name = row.get('class')
            if cls_name not in name_to_idx:
                continue
            cls_idx = name_to_idx[cls_name]
            log_area_mean = float(row.get('log_area_mean') or 0.0)
            log_area_std = max(float(row.get('log_area_std') or 0.0), 1e-6)
            log_aspect_mean = float(row.get('log_aspect_mean') or 0.0)
            log_aspect_std = max(
                float(row.get('log_aspect_std') or 1.0), 1e-6)
            means[cls_idx] = torch.tensor(
                [log_area_mean, log_aspect_mean], dtype=torch.float32)
            stds[cls_idx] = torch.tensor(
                [log_area_std, log_aspect_std], dtype=torch.float32)
            valid[cls_idx] = True
    return means, stds, valid, list(class_names)


def build_classwise_gaussian_parameter(
        scalar_value: float,
        classwise_value: dict[str, float] | Sequence[float] | None,
        class_names: Sequence[str],
        num_classes: int,
        name: str,
        min_value: float | None = None,
) -> Tensor:
    values = torch.full(
        (int(num_classes),), float(scalar_value), dtype=torch.float32)
    if classwise_value is None:
        return values
    if isinstance(classwise_value, dict):
        name_to_idx = {
            str(class_name): idx
            for idx, class_name in enumerate(class_names[:int(num_classes)])
        }
        for class_name, raw_value in classwise_value.items():
            if str(class_name) not in name_to_idx:
                raise ValueError(
                    f'{name} classwise key {class_name!r} is not in '
                    'class_names')
            value = float(raw_value)
            if min_value is not None and value < min_value:
                raise ValueError(
                    f'{name} classwise value for {class_name!r} must be '
                    f'>= {min_value}')
            values[name_to_idx[str(class_name)]] = value
        return values
    raw_values = [float(item) for item in classwise_value]
    if len(raw_values) != int(num_classes):
        raise ValueError(
            f'{name} classwise sequence length {len(raw_values)} does not '
            f'match num_classes {int(num_classes)}')
    tensor = torch.tensor(raw_values, dtype=torch.float32)
    if min_value is not None and torch.any(tensor < min_value):
        raise ValueError(f'{name} classwise values must be >= {min_value}')
    return tensor


def gaussian_log_area_stats(log_area: Tensor, means: Tensor, stds: Tensor,
                            valid_mask: Tensor) -> dict[str, Tensor]:
    """Return z-score and log-probability features for log-area priors."""
    stds = stds.clamp(min=1e-6)
    z = (log_area[:, None].to(means.dtype) - means[None, :]) / stds[None, :]
    log_std = torch.log(stds)[None, :].expand_as(z)
    log_prob = -0.5 * z.square() - log_std - 0.5 * math.log(2.0 * math.pi)
    valid = valid_mask[None, :].expand_as(z).to(dtype=torch.bool)
    return {
        'z': z,
        'abs_z': z.abs(),
        'log_prob': log_prob,
        'log_std': log_std,
        'valid': valid,
    }


def gaussian_geometry_support_stats(geometry: Tensor, means: Tensor,
                                    stds: Tensor,
                                    valid_mask: Tensor) -> dict[str, Tensor]:
    """Return diagonal Gaussian support features for semantic geometry.

    ``geometry`` is shaped ``(num_locations, num_dims)`` and currently uses two
    dimensions: log-area and log-aspect.  The returned log-probability sums over
    geometry dimensions, so it can replace scalar ``log_prob`` in support-based
    candidate selection.
    """
    if means.ndim != 2 or stds.ndim != 2:
        raise ValueError('geometry prior means/stds must be 2D tensors')
    if means.shape != stds.shape:
        raise ValueError(
            f'geometry prior means shape {tuple(means.shape)} does not match '
            f'stds shape {tuple(stds.shape)}')
    geometry = geometry.to(device=means.device, dtype=means.dtype)
    if geometry.ndim == 1:
        geometry = geometry[None, :]
    if geometry.shape[-1] != means.shape[-1]:
        raise ValueError(
            f'geometry has {geometry.shape[-1]} dims but priors have '
            f'{means.shape[-1]} dims')

    stds = stds.clamp(min=1e-6)
    z = (geometry[:, None, :] - means[None, :, :]) / stds[None, :, :]
    log_prob_dim = (
        -0.5 * z.square()
        - torch.log(stds)[None, :, :]
        - 0.5 * math.log(2.0 * math.pi))
    log_prob = log_prob_dim.sum(dim=-1)
    mahalanobis = z.square().sum(dim=-1)
    valid = valid_mask.to(device=means.device, dtype=torch.bool)
    valid_2d = valid[None, :].expand_as(log_prob)
    return {
        'z': z,
        'abs_z': z.abs(),
        'mahalanobis': mahalanobis,
        'log_prob_dim': log_prob_dim,
        'log_prob': torch.where(valid_2d, log_prob,
                                torch.zeros_like(log_prob)),
        'valid': valid_2d,
    }


def _as_classwise_tensor(value: float | Tensor,
                         like: Tensor,
                         name: str) -> Tensor:
    if torch.is_tensor(value):
        tensor = value.to(device=like.device, dtype=like.dtype)
        if tensor.ndim == 1:
            tensor = tensor[None, :]
        if tensor.shape[-1] != like.shape[-1]:
            raise ValueError(
                f'{name} last dimension {tensor.shape[-1]} does not match '
                f'z last dimension {like.shape[-1]}')
        return tensor.expand_as(like)
    return torch.full_like(like, float(value))


def continuous_gaussian_logit_energy_delta(
        z: Tensor,
        valid_mask: Tensor,
        z0: float | Tensor = 4.0,
        beta: float | Tensor = math.log(4.0),
) -> Tensor:
    """Compute non-positive continuous logit delta from Gaussian z-score."""
    valid = valid_mask.to(device=z.device, dtype=torch.bool)
    z0_tensor = _as_classwise_tensor(z0, z, 'z0')
    beta_tensor = _as_classwise_tensor(beta, z, 'beta').clamp(min=0.0)
    penalty = F.softplus(z.abs() - z0_tensor)
    delta = -beta_tensor * penalty
    return torch.where(valid, delta, torch.zeros_like(delta))


def gaussian_pos_hardneg_consistency_loss(
        cls_logits: Tensor,
        labels: Tensor,
        bbox_targets: Tensor,
        assign_metrics: Tensor,
        log_area_mean: Tensor,
        log_area_std: Tensor,
        valid_mask: Tensor,
        margin: float = 0.5,
        min_logprob_gap: float = 1.0,
        max_hardneg: int = 1,
        min_hardneg_logit: float | None = None,
        max_gt_abs_z: float | None = None,
) -> tuple[Tensor, dict[str, float | int]]:
    """Trainable G3 loss for semantic-scale positive/hard-negative ranking.

    For assigned positives, this loss finds classes whose Gaussian
    ``log(area)|class`` likelihood is much lower than the GT class likelihood.
    Those scale-incompatible classes are hard negatives. The loss then asks the
    GT class logit to exceed the strongest hard-negative logit by ``margin``.

    Gaussian likelihoods gate/select the hard negatives; gradients flow through
    class logits, not through the fixed dataset priors.
    """
    if cls_logits.numel() == 0:
        zero = cls_logits.sum() * 0.0
        return zero, {
            'num_pos': 0,
            'num_hardneg': 0,
            'num_candidate_pairs': 0,
            'num_pos_with_hardneg': 0,
            'num_ap_sensitive_pos': 0,
            'active_violation_count': 0,
            'mean_logprob_gap': 0.0,
            'hardneg_per_pos': 0.0,
            'active_violation_rate': 0.0,
        }

    num_classes = int(cls_logits.shape[-1])
    labels = labels.reshape(-1).to(device=cls_logits.device, dtype=torch.long)
    bbox_targets = bbox_targets.reshape(-1, bbox_targets.shape[-1]).to(
        device=cls_logits.device, dtype=cls_logits.dtype)
    assign_metrics = assign_metrics.reshape(-1).to(
        device=cls_logits.device, dtype=cls_logits.dtype)
    means = log_area_mean.to(
        device=cls_logits.device, dtype=cls_logits.dtype)[:num_classes]
    stds = log_area_std.to(
        device=cls_logits.device, dtype=cls_logits.dtype)[:num_classes]
    valid = valid_mask.to(device=cls_logits.device, dtype=torch.bool)[
        :num_classes]

    label_in_range = (labels >= 0) & (labels < num_classes)
    safe_labels = labels.clamp(min=0, max=max(num_classes - 1, 0))
    pos_mask = label_in_range & valid[safe_labels]
    if not torch.any(pos_mask):
        zero = cls_logits.sum() * 0.0
        return zero, {
            'num_pos': 0,
            'num_hardneg': 0,
            'num_candidate_pairs': 0,
            'num_pos_with_hardneg': 0,
            'num_ap_sensitive_pos': 0,
            'active_violation_count': 0,
            'mean_logprob_gap': 0.0,
            'hardneg_per_pos': 0.0,
            'active_violation_rate': 0.0,
        }

    pos_logits = cls_logits.reshape(-1, num_classes)[pos_mask]
    pos_labels = labels[pos_mask]
    pos_boxes = bbox_targets[pos_mask]
    pos_weights = assign_metrics[pos_mask].clamp(min=0.0)

    area = (pos_boxes[:, 2].abs() * pos_boxes[:, 3].abs()).clamp(min=1e-6)
    stats = gaussian_log_area_stats(torch.log(area), means, stds, valid)
    log_prob = stats['log_prob']
    pos_log_prob = log_prob.gather(1, pos_labels[:, None]).squeeze(1)
    pos_abs_z = stats['abs_z'].gather(1, pos_labels[:, None]).squeeze(1)
    gap = pos_log_prob[:, None] - log_prob
    candidate = (
        stats['valid']
        & (gap >= float(min_logprob_gap))
        & (torch.arange(num_classes, device=cls_logits.device)[None, :]
           != pos_labels[:, None]))
    if max_gt_abs_z is not None:
        candidate = candidate & (pos_abs_z[:, None] <= float(max_gt_abs_z))
    if min_hardneg_logit is not None:
        candidate = candidate & (pos_logits >= float(min_hardneg_logit))
    num_pos = int(pos_mask.sum().detach().cpu().item())
    num_candidate_pairs = int(candidate.sum().detach().cpu().item())
    num_pos_with_hardneg = int(
        candidate.any(dim=1).sum().detach().cpu().item())
    num_ap_sensitive_pos = num_pos_with_hardneg

    if not torch.any(candidate):
        zero = cls_logits.sum() * 0.0
        return zero, {
            'num_pos': num_pos,
            'num_hardneg': 0,
            'num_candidate_pairs': 0,
            'num_pos_with_hardneg': 0,
            'num_ap_sensitive_pos': 0,
            'active_violation_count': 0,
            'mean_logprob_gap': 0.0,
            'hardneg_per_pos': 0.0,
            'active_violation_rate': 0.0,
        }

    hardneg_logits = pos_logits.masked_fill(~candidate, -torch.inf)
    k = max(int(max_hardneg), 1)
    k = min(k, num_classes)
    top_values, top_indices = torch.topk(hardneg_logits, k=k, dim=1)
    top_valid = torch.isfinite(top_values)
    pos_values = pos_logits.gather(1, pos_labels[:, None]).expand_as(
        top_values)
    raw_loss = F.relu(float(margin) + top_values - pos_values)
    raw_loss = torch.where(top_valid, raw_loss, torch.zeros_like(raw_loss))
    active = top_valid & (raw_loss > 0)

    if pos_weights.numel() > 0:
        weight = pos_weights[:, None].expand_as(raw_loss)
        weighted = raw_loss * weight
        denom = (top_valid.to(cls_logits.dtype) * weight).sum().clamp(min=1.0)
        loss = weighted.sum() / denom
    else:
        loss = raw_loss[top_valid].mean()

    chosen_gap = gap.gather(1, top_indices.clamp(min=0))
    chosen_gap = chosen_gap[top_valid]
    num_hardneg = int(top_valid.sum().detach().cpu().item())
    active_violation_count = int(active.sum().detach().cpu().item())
    return loss, {
        'num_pos': num_pos,
        'num_hardneg': num_hardneg,
        'num_candidate_pairs': num_candidate_pairs,
        'num_pos_with_hardneg': num_pos_with_hardneg,
        'num_ap_sensitive_pos': num_ap_sensitive_pos,
        'active_violation_count': active_violation_count,
        'mean_logprob_gap': float(chosen_gap.mean().detach().cpu().item())
        if chosen_gap.numel() else 0.0,
        'hardneg_per_pos': float(num_hardneg) / max(float(num_pos), 1.0),
        'active_violation_rate': (
            float(active_violation_count) / max(float(num_hardneg), 1.0)),
    }


class GaussianScaleLogitAdapter(nn.Module):
    """Tiny per-class adapter for Gaussian semantic-scale logit deltas.

    The final layer is zero-initialized so the adapter starts as an identity
    correction.  With ``nonpositive_delta=True`` it can only suppress logits.
    """

    def __init__(self,
                 hidden: int = 64,
                 nonpositive_delta: bool = True) -> None:
        super().__init__()
        hidden = max(int(hidden), 1)
        self.nonpositive_delta = bool(nonpositive_delta)
        self.net = nn.Sequential(
            nn.Linear(5, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.register_buffer(
            '_softplus_zero',
            torch.tensor(math.log(2.0), dtype=torch.float32),
            persistent=False)

    def forward(self,
                abs_z: Tensor,
                signed_z: Tensor,
                gaussian_log_prob: Tensor,
                log_std: Tensor,
                valid_mask: Tensor) -> Tensor:
        valid = valid_mask.to(device=abs_z.device, dtype=torch.bool)
        features = torch.stack(
            [
                abs_z,
                signed_z,
                gaussian_log_prob,
                log_std,
                valid.to(dtype=abs_z.dtype),
            ],
            dim=-1,
        )
        raw_delta = self.net(features).squeeze(-1)
        if self.nonpositive_delta:
            zero = self._softplus_zero.to(
                device=raw_delta.device, dtype=raw_delta.dtype)
            delta = torch.clamp(-F.softplus(raw_delta) + zero, max=0.0)
        else:
            delta = raw_delta
        return torch.where(valid, delta, torch.zeros_like(delta))


class GaussianSemanticScaleDensityHead(nn.Module):
    """Predict feature-conditioned class-scale likelihood and safe logit delta.

    The head keeps dataset-level ``log(area)|class`` Gaussian priors as the
    anchor, then learns small per-location residuals and uncertainty. The
    final convolution is zero-initialized so the module starts as a prior-only
    density estimator with identity logit correction.
    """

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 hidden_channels: int = 64,
                 mean_residual_scale: float = 1.0,
                 log_std_delta_limit: float = 1.0,
                 max_delta_abs: float = 0.5,
                 nonpositive_delta: bool = True) -> None:
        super().__init__()
        hidden_channels = max(int(hidden_channels), 1)
        self.num_classes = int(num_classes)
        self.mean_residual_scale = float(mean_residual_scale)
        self.log_std_delta_limit = float(log_std_delta_limit)
        self.max_delta_abs = max(float(max_delta_abs), 0.0)
        self.nonpositive_delta = bool(nonpositive_delta)
        self.net = nn.Sequential(
            nn.Conv2d(int(in_channels), hidden_channels, kernel_size=3,
                      padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, self.num_classes * 3, kernel_size=1),
        )
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.register_buffer(
            '_softplus_zero',
            torch.tensor(math.log(2.0), dtype=torch.float32),
            persistent=False)

    def forward(self,
                features: Tensor,
                log_area: Tensor,
                prior_mean: Tensor,
                prior_std: Tensor,
                valid_mask: Tensor) -> dict[str, Tensor]:
        raw = self.net(features)
        raw = raw.permute(0, 2, 3, 1).reshape(-1, self.num_classes, 3)
        raw_mean = raw[..., 0]
        raw_log_std = raw[..., 1]
        raw_delta = raw[..., 2]

        means = prior_mean.to(
            device=features.device, dtype=features.dtype)[:self.num_classes]
        stds = prior_std.to(
            device=features.device, dtype=features.dtype)[
                :self.num_classes].clamp(min=1e-6)
        valid = valid_mask.to(device=features.device, dtype=torch.bool)[
            :self.num_classes]

        mean = means[None, :] + self.mean_residual_scale * torch.tanh(
            raw_mean)
        log_std_delta = self.log_std_delta_limit * torch.tanh(raw_log_std)
        std = stds[None, :] * torch.exp(log_std_delta)
        area = log_area.to(device=features.device, dtype=features.dtype)
        area = area.reshape(-1, 1)
        if area.shape[0] != mean.shape[0]:
            raise ValueError(
                f'log_area has {area.shape[0]} locations but features imply '
                f'{mean.shape[0]} locations')

        z = (area - mean) / std.clamp(min=1e-6)
        log_prob = (
            -0.5 * z.square()
            - torch.log(std.clamp(min=1e-6))
            - 0.5 * math.log(2.0 * math.pi))
        uncertainty = std
        if self.nonpositive_delta:
            zero = self._softplus_zero.to(
                device=features.device, dtype=features.dtype)
            delta = torch.clamp(-F.softplus(raw_delta) + zero, max=0.0)
        else:
            delta = raw_delta
        if self.max_delta_abs > 0:
            if self.nonpositive_delta:
                delta = torch.clamp(delta, min=-self.max_delta_abs)
            else:
                delta = torch.clamp(
                    delta, min=-self.max_delta_abs, max=self.max_delta_abs)

        valid_2d = valid[None, :].expand_as(log_prob)
        return {
            'mean': torch.where(valid_2d, mean, means[None, :]),
            'std': torch.where(valid_2d, std, stds[None, :]),
            'log_prob': torch.where(valid_2d, log_prob,
                                    torch.zeros_like(log_prob)),
            'uncertainty': torch.where(valid_2d, uncertainty,
                                       torch.zeros_like(uncertainty)),
            'logit_delta': torch.where(valid_2d, delta,
                                       torch.zeros_like(delta)),
            'valid': valid_2d,
        }


def semantic_scale_density_nll_loss(
        log_prob: Tensor,
        labels: Tensor,
        assign_metrics: Tensor,
        valid_mask: Tensor,
) -> tuple[Tensor, dict[str, float | int]]:
    """NLL loss for GT-class semantic-scale density on assigned positives."""
    if log_prob.numel() == 0:
        zero = log_prob.sum() * 0.0
        return zero, {'num_pos': 0, 'mean_gt_log_prob': 0.0}
    num_classes = int(log_prob.shape[-1])
    labels = labels.reshape(-1).to(device=log_prob.device, dtype=torch.long)
    assign_metrics = assign_metrics.reshape(-1).to(
        device=log_prob.device, dtype=log_prob.dtype)
    valid = valid_mask.to(device=log_prob.device, dtype=torch.bool)[
        :num_classes]
    label_in_range = (labels >= 0) & (labels < num_classes)
    safe_labels = labels.clamp(min=0, max=max(num_classes - 1, 0))
    pos_mask = label_in_range & valid[safe_labels]
    if not torch.any(pos_mask):
        zero = log_prob.sum() * 0.0
        return zero, {'num_pos': 0, 'mean_gt_log_prob': 0.0}

    gt_log_prob = log_prob.reshape(-1, num_classes)[pos_mask].gather(
        1, labels[pos_mask, None]).squeeze(1)
    weights = assign_metrics[pos_mask].clamp(min=0.0)
    if weights.numel() == 0 or float(weights.sum().detach().cpu()) <= 0.0:
        loss = -gt_log_prob.mean()
    else:
        loss = -(gt_log_prob * weights).sum() / weights.sum().clamp(min=1e-6)
    return loss, {
        'num_pos': int(pos_mask.sum().detach().cpu().item()),
        'mean_gt_log_prob': float(gt_log_prob.mean().detach().cpu().item()),
    }


def semantic_scale_density_delta_outlier_loss(
        logit_delta: Tensor,
        log_prob: Tensor,
        labels: Tensor,
        assign_metrics: Tensor,
        valid_mask: Tensor,
        cls_logits: Tensor | None = None,
        min_hardneg_score: float | None = None,
        min_hardneg_logit: float | None = None,
        min_logprob_gap: float = 1.0,
        target_negative_delta: float = 0.25,
        gt_keep_weight: float = 0.1,
        max_hardneg: int = 1,
) -> tuple[Tensor, dict[str, float | int]]:
    """Train density-head deltas on semantic-scale hard negatives.

    ``log_prob`` selects classes whose scale support is much lower than the
    assigned GT class.  Gradients then flow through ``logit_delta``: selected
    hard-negative classes are pushed toward a non-positive target while the GT
    class delta is weakly kept near zero.
    """
    if logit_delta.numel() == 0 or log_prob.numel() == 0:
        zero = logit_delta.sum() * 0.0
        return zero, {
            'num_pos': 0,
            'num_hardneg': 0,
            'num_candidate_pairs': 0,
            'num_pos_with_hardneg': 0,
            'mean_logprob_gap': 0.0,
            'mean_hardneg_delta': 0.0,
            'hardneg_raw_loss': 0.0,
            'gt_keep_raw_loss': 0.0,
            'num_score_gated_pairs': 0,
        }

    num_classes = int(log_prob.shape[-1])
    delta = logit_delta.reshape(-1, logit_delta.shape[-1])[:, :num_classes]
    log_prob = log_prob.reshape(-1, num_classes)
    labels = labels.reshape(-1).to(device=log_prob.device, dtype=torch.long)
    assign_metrics = assign_metrics.reshape(-1).to(
        device=log_prob.device, dtype=log_prob.dtype)
    valid = valid_mask.to(device=log_prob.device, dtype=torch.bool)[
        :num_classes]

    label_in_range = (labels >= 0) & (labels < num_classes)
    safe_labels = labels.clamp(min=0, max=max(num_classes - 1, 0))
    pos_mask = label_in_range & valid[safe_labels]
    num_pos = int(pos_mask.sum().detach().cpu().item())
    if not torch.any(pos_mask):
        zero = logit_delta.sum() * 0.0
        return zero, {
            'num_pos': 0,
            'num_hardneg': 0,
            'num_candidate_pairs': 0,
            'num_pos_with_hardneg': 0,
            'mean_logprob_gap': 0.0,
            'mean_hardneg_delta': 0.0,
            'hardneg_raw_loss': 0.0,
            'gt_keep_raw_loss': 0.0,
            'num_score_gated_pairs': 0,
        }

    pos_delta = delta[pos_mask]
    pos_log_prob = log_prob[pos_mask]
    pos_labels = labels[pos_mask]
    pos_weights = assign_metrics[pos_mask].clamp(min=0.0)
    gt_log_prob = pos_log_prob.gather(1, pos_labels[:, None]).squeeze(1)
    gap = gt_log_prob[:, None] - pos_log_prob
    class_idx = torch.arange(num_classes, device=log_prob.device)[None, :]
    candidate = (
        valid[None, :]
        & (class_idx != pos_labels[:, None])
        & (gap >= float(min_logprob_gap)))
    score_gate = torch.zeros_like(candidate)
    if cls_logits is not None and (
            min_hardneg_score is not None
            or min_hardneg_logit is not None):
        flat_logits = cls_logits.reshape(-1, cls_logits.shape[-1]).to(
            device=log_prob.device, dtype=log_prob.dtype)[:, :num_classes]
        pos_logits = flat_logits[pos_mask]
        if min_hardneg_logit is None:
            score = torch.sigmoid(pos_logits)
            score_gate = score < float(min_hardneg_score)
        else:
            score_gate = pos_logits < float(min_hardneg_logit)
        score_gate = score_gate & candidate
        candidate = candidate & ~score_gate

    num_candidate_pairs = int(candidate.sum().detach().cpu().item())
    num_score_gated_pairs = int(score_gate.sum().detach().cpu().item())
    num_pos_with_hardneg = int(
        candidate.any(dim=1).sum().detach().cpu().item())
    if not torch.any(candidate):
        zero = logit_delta.sum() * 0.0
        return zero, {
            'num_pos': num_pos,
            'num_hardneg': 0,
            'num_candidate_pairs': 0,
            'num_pos_with_hardneg': 0,
            'mean_logprob_gap': 0.0,
            'mean_hardneg_delta': 0.0,
            'hardneg_raw_loss': 0.0,
            'gt_keep_raw_loss': 0.0,
            'num_score_gated_pairs': num_score_gated_pairs,
        }

    k = min(max(int(max_hardneg), 1), num_classes)
    masked_gap = gap.masked_fill(~candidate, -torch.inf)
    _, top_idx = torch.topk(masked_gap, k=k, dim=1)
    top_valid = torch.isfinite(masked_gap.gather(1, top_idx))
    chosen_gap = gap.gather(1, top_idx)
    hardneg_delta = pos_delta.gather(1, top_idx)
    target = float(max(target_negative_delta, 0.0))
    hardneg_raw = F.relu(target + hardneg_delta)
    hardneg_raw = torch.where(
        top_valid, hardneg_raw, torch.zeros_like(hardneg_raw))

    weights = pos_weights[:, None].expand_as(hardneg_raw)
    hardneg_denom = (weights * top_valid.to(weights.dtype)).sum().clamp(
        min=1.0)
    hardneg_loss = (hardneg_raw * weights).sum() / hardneg_denom

    gt_delta = pos_delta.gather(1, pos_labels[:, None]).squeeze(1)
    if float(pos_weights.sum().detach().cpu()) > 0.0:
        gt_keep_loss = (gt_delta.square() * pos_weights).sum() / (
            pos_weights.sum().clamp(min=1e-6))
    else:
        gt_keep_loss = gt_delta.square().mean()
    loss = hardneg_loss + float(max(gt_keep_weight, 0.0)) * gt_keep_loss

    selected_delta = hardneg_delta[top_valid]
    selected_gap = chosen_gap[top_valid]
    return loss, {
        'num_pos': num_pos,
        'num_hardneg': int(top_valid.sum().detach().cpu().item()),
        'num_candidate_pairs': num_candidate_pairs,
        'num_pos_with_hardneg': num_pos_with_hardneg,
        'mean_logprob_gap': float(selected_gap.mean().detach().cpu().item())
        if selected_gap.numel() else 0.0,
        'mean_hardneg_delta': float(
            selected_delta.mean().detach().cpu().item())
        if selected_delta.numel() else 0.0,
        'hardneg_raw_loss': float(hardneg_loss.detach().cpu().item()),
        'gt_keep_raw_loss': float(gt_keep_loss.detach().cpu().item()),
        'num_score_gated_pairs': num_score_gated_pairs,
    }


def semantic_scale_density_positive_delta_loss(
        logit_delta: Tensor,
        log_prob: Tensor,
        labels: Tensor,
        assign_metrics: Tensor,
        valid_mask: Tensor,
        cls_logits: Tensor | None = None,
        min_gt_logprob: float | None = None,
        min_gt_score: float | None = None,
        target_positive_delta: float = 0.1,
) -> tuple[Tensor, dict[str, float | int]]:
    """Protect Gaussian-supported positives by pushing GT delta upward.

    This is an AP-sensitive ranking projection term: Gaussian likelihood
    selects assigned positives whose GT semantic scale is supported, while
    gradients flow only through the density head's GT-class logit delta.
    """
    if logit_delta.numel() == 0 or log_prob.numel() == 0:
        zero = logit_delta.sum() * 0.0
        return zero, {
            'num_pos': 0,
            'num_supported_pos': 0,
            'num_score_gated_pos': 0,
            'mean_gt_log_prob': 0.0,
            'mean_gt_delta': 0.0,
            'raw_loss': 0.0,
        }

    num_classes = int(log_prob.shape[-1])
    delta = logit_delta.reshape(-1, logit_delta.shape[-1])[:, :num_classes]
    log_prob = log_prob.reshape(-1, num_classes)
    labels = labels.reshape(-1).to(device=log_prob.device, dtype=torch.long)
    assign_metrics = assign_metrics.reshape(-1).to(
        device=log_prob.device, dtype=log_prob.dtype)
    valid = valid_mask.to(device=log_prob.device, dtype=torch.bool)[
        :num_classes]

    label_in_range = (labels >= 0) & (labels < num_classes)
    safe_labels = labels.clamp(min=0, max=max(num_classes - 1, 0))
    pos_mask = label_in_range & valid[safe_labels]
    num_pos = int(pos_mask.sum().detach().cpu().item())
    if not torch.any(pos_mask):
        zero = logit_delta.sum() * 0.0
        return zero, {
            'num_pos': 0,
            'num_supported_pos': 0,
            'num_score_gated_pos': 0,
            'mean_gt_log_prob': 0.0,
            'mean_gt_delta': 0.0,
            'raw_loss': 0.0,
        }

    pos_delta = delta[pos_mask]
    pos_log_prob = log_prob[pos_mask]
    pos_labels = labels[pos_mask]
    pos_weights = assign_metrics[pos_mask].clamp(min=0.0)
    gt_delta = pos_delta.gather(1, pos_labels[:, None]).squeeze(1)
    gt_log_prob = pos_log_prob.gather(1, pos_labels[:, None]).squeeze(1)

    supported = torch.ones_like(gt_log_prob, dtype=torch.bool)
    if min_gt_logprob is not None:
        supported = supported & (gt_log_prob >= float(min_gt_logprob))

    score_gated = torch.zeros_like(supported)
    if cls_logits is not None and min_gt_score is not None:
        flat_logits = cls_logits.reshape(-1, cls_logits.shape[-1]).to(
            device=log_prob.device, dtype=log_prob.dtype)[:, :num_classes]
        pos_logits = flat_logits[pos_mask]
        gt_score = torch.sigmoid(
            pos_logits.gather(1, pos_labels[:, None]).squeeze(1))
        score_gated = gt_score < float(min_gt_score)
        supported = supported & ~score_gated

    num_supported = int(supported.sum().detach().cpu().item())
    num_score_gated = int(score_gated.sum().detach().cpu().item())
    if not torch.any(supported):
        zero = logit_delta.sum() * 0.0
        return zero, {
            'num_pos': num_pos,
            'num_supported_pos': 0,
            'num_score_gated_pos': num_score_gated,
            'mean_gt_log_prob': float(
                gt_log_prob.mean().detach().cpu().item())
            if gt_log_prob.numel() else 0.0,
            'mean_gt_delta': float(gt_delta.mean().detach().cpu().item())
            if gt_delta.numel() else 0.0,
            'raw_loss': 0.0,
        }

    target = float(max(target_positive_delta, 0.0))
    raw = F.relu(target - gt_delta[supported])
    weights = pos_weights[supported]
    if weights.numel() == 0 or float(weights.sum().detach().cpu()) <= 0.0:
        loss = raw.mean()
    else:
        loss = (raw * weights).sum() / weights.sum().clamp(min=1e-6)

    return loss, {
        'num_pos': num_pos,
        'num_supported_pos': num_supported,
        'num_score_gated_pos': num_score_gated,
        'mean_gt_log_prob': float(
            gt_log_prob[supported].mean().detach().cpu().item())
        if gt_log_prob[supported].numel() else 0.0,
        'mean_gt_delta': float(
            gt_delta[supported].mean().detach().cpu().item())
        if gt_delta[supported].numel() else 0.0,
        'raw_loss': float(loss.detach().cpu().item()),
    }


def semantic_scale_density_pair_margin_loss(
        logit_delta: Tensor,
        log_prob: Tensor,
        labels: Tensor,
        assign_metrics: Tensor,
        valid_mask: Tensor,
        cls_logits: Tensor | None = None,
        min_hardneg_score: float | None = None,
        min_logprob_gap: float = 1.0,
        margin: float = 0.2,
        max_hardneg: int = 1,
) -> tuple[Tensor, dict[str, float | int]]:
    """Protect GT-vs-hard-negative ranking with Gaussian-selected pairs.

    Gaussian support selects semantically implausible hard-negative classes.
    The optimized object is the AP-sensitive relative ordering:
    ``gt_logit + gt_delta`` should exceed
    ``hardneg_logit + hardneg_delta`` by ``margin``.
    """
    if logit_delta.numel() == 0 or log_prob.numel() == 0:
        zero = logit_delta.sum() * 0.0
        return zero, {
            'num_pos': 0,
            'num_hardneg': 0,
            'num_candidate_pairs': 0,
            'num_pos_with_hardneg': 0,
            'num_score_gated_pairs': 0,
            'active_violation_count': 0,
            'mean_logprob_gap': 0.0,
            'mean_pair_margin': 0.0,
            'raw_loss': 0.0,
        }

    num_classes = int(log_prob.shape[-1])
    delta = logit_delta.reshape(-1, logit_delta.shape[-1])[:, :num_classes]
    log_prob = log_prob.reshape(-1, num_classes)
    labels = labels.reshape(-1).to(device=log_prob.device, dtype=torch.long)
    assign_metrics = assign_metrics.reshape(-1).to(
        device=log_prob.device, dtype=log_prob.dtype)
    valid = valid_mask.to(device=log_prob.device, dtype=torch.bool)[
        :num_classes]

    label_in_range = (labels >= 0) & (labels < num_classes)
    safe_labels = labels.clamp(min=0, max=max(num_classes - 1, 0))
    pos_mask = label_in_range & valid[safe_labels]
    num_pos = int(pos_mask.sum().detach().cpu().item())
    if not torch.any(pos_mask):
        zero = logit_delta.sum() * 0.0
        return zero, {
            'num_pos': 0,
            'num_hardneg': 0,
            'num_candidate_pairs': 0,
            'num_pos_with_hardneg': 0,
            'num_score_gated_pairs': 0,
            'active_violation_count': 0,
            'mean_logprob_gap': 0.0,
            'mean_pair_margin': 0.0,
            'raw_loss': 0.0,
        }

    pos_delta = delta[pos_mask]
    pos_log_prob = log_prob[pos_mask]
    pos_labels = labels[pos_mask]
    pos_weights = assign_metrics[pos_mask].clamp(min=0.0)
    gt_log_prob = pos_log_prob.gather(1, pos_labels[:, None]).squeeze(1)
    gap = gt_log_prob[:, None] - pos_log_prob
    class_idx = torch.arange(num_classes, device=log_prob.device)[None, :]
    candidate = (
        valid[None, :]
        & (class_idx != pos_labels[:, None])
        & (gap >= float(min_logprob_gap)))

    score_gated = torch.zeros_like(candidate)
    if cls_logits is None:
        pos_logits = torch.zeros_like(pos_delta)
    else:
        flat_logits = cls_logits.reshape(-1, cls_logits.shape[-1]).to(
            device=log_prob.device, dtype=log_prob.dtype)[:, :num_classes]
        pos_logits = flat_logits[pos_mask]
        if min_hardneg_score is not None:
            score = torch.sigmoid(pos_logits)
            score_gated = (score < float(min_hardneg_score)) & candidate
            candidate = candidate & ~score_gated

    num_candidate_pairs = int(candidate.sum().detach().cpu().item())
    num_score_gated_pairs = int(score_gated.sum().detach().cpu().item())
    num_pos_with_hardneg = int(
        candidate.any(dim=1).sum().detach().cpu().item())
    if not torch.any(candidate):
        zero = logit_delta.sum() * 0.0
        return zero, {
            'num_pos': num_pos,
            'num_hardneg': 0,
            'num_candidate_pairs': 0,
            'num_pos_with_hardneg': 0,
            'num_score_gated_pairs': num_score_gated_pairs,
            'active_violation_count': 0,
            'mean_logprob_gap': 0.0,
            'mean_pair_margin': 0.0,
            'raw_loss': 0.0,
        }

    k = min(max(int(max_hardneg), 1), num_classes)
    masked_gap = gap.masked_fill(~candidate, -torch.inf)
    _, top_idx = torch.topk(masked_gap, k=k, dim=1)
    top_valid = torch.isfinite(masked_gap.gather(1, top_idx))
    chosen_gap = gap.gather(1, top_idx)
    hardneg_delta = pos_delta.gather(1, top_idx)
    hardneg_logit = pos_logits.gather(1, top_idx)
    gt_delta = pos_delta.gather(1, pos_labels[:, None]).squeeze(1)
    gt_logit = pos_logits.gather(1, pos_labels[:, None]).squeeze(1)
    pair_margin = (
        gt_logit[:, None] + gt_delta[:, None]
        - hardneg_logit - hardneg_delta)
    raw = F.relu(float(margin) - pair_margin)
    raw = torch.where(top_valid, raw, torch.zeros_like(raw))

    weights = pos_weights[:, None].expand_as(raw)
    denom = (weights * top_valid.to(weights.dtype)).sum().clamp(min=1.0)
    loss = (raw * weights).sum() / denom
    selected_gap = chosen_gap[top_valid]
    selected_margin = pair_margin[top_valid]
    selected_raw = raw[top_valid]
    return loss, {
        'num_pos': num_pos,
        'num_hardneg': int(top_valid.sum().detach().cpu().item()),
        'num_candidate_pairs': num_candidate_pairs,
        'num_pos_with_hardneg': num_pos_with_hardneg,
        'num_score_gated_pairs': num_score_gated_pairs,
        'active_violation_count': int(
            (selected_raw > 0).sum().detach().cpu().item()),
        'mean_logprob_gap': float(selected_gap.mean().detach().cpu().item())
        if selected_gap.numel() else 0.0,
        'mean_pair_margin': float(
            selected_margin.mean().detach().cpu().item())
        if selected_margin.numel() else 0.0,
        'raw_loss': float(loss.detach().cpu().item()),
    }


def gaussian_positive_scale_consistency_loss(
        target_log_prob: Tensor,
        pred_log_prob: Tensor,
        labels: Tensor,
        assign_metrics: Tensor,
        valid_mask: Tensor,
        max_logprob_drop: float = 0.5,
        min_target_logprob: float | None = None,
) -> tuple[Tensor, dict[str, float | int]]:
    """Keep predicted positive boxes inside the GT class Gaussian support.

    The loss compares two support evaluations under the same semantic class:
    the assigned target box and the current predicted box.  It penalizes only
    when the predicted box's GT-class support drops more than
    ``max_logprob_drop`` below the target support.  This is not a bbox
    overlap/distance loss; it regularizes semantic scale support.
    """
    if target_log_prob.numel() == 0 or pred_log_prob.numel() == 0:
        zero = pred_log_prob.sum() * 0.0
        return zero, {
            'num_pos': 0,
            'num_supported_pos': 0,
            'active_violation_count': 0,
            'mean_target_log_prob': 0.0,
            'mean_pred_log_prob': 0.0,
            'mean_logprob_drop': 0.0,
            'raw_loss': 0.0,
        }

    num_classes = int(min(target_log_prob.shape[-1],
                          pred_log_prob.shape[-1]))
    target_support = target_log_prob.reshape(
        -1, target_log_prob.shape[-1])[:, :num_classes].to(
            device=pred_log_prob.device, dtype=pred_log_prob.dtype)
    pred_support = pred_log_prob.reshape(
        -1, pred_log_prob.shape[-1])[:, :num_classes]
    labels = labels.reshape(-1).to(device=pred_log_prob.device,
                                   dtype=torch.long)
    assign_metrics = assign_metrics.reshape(-1).to(
        device=pred_log_prob.device, dtype=pred_log_prob.dtype)
    valid = valid_mask.to(device=pred_log_prob.device, dtype=torch.bool)[
        :num_classes]

    label_in_range = (labels >= 0) & (labels < num_classes)
    safe_labels = labels.clamp(min=0, max=max(num_classes - 1, 0))
    pos_mask = label_in_range & valid[safe_labels]
    num_pos = int(pos_mask.sum().detach().cpu().item())
    if not torch.any(pos_mask):
        zero = pred_log_prob.sum() * 0.0
        return zero, {
            'num_pos': 0,
            'num_supported_pos': 0,
            'active_violation_count': 0,
            'mean_target_log_prob': 0.0,
            'mean_pred_log_prob': 0.0,
            'mean_logprob_drop': 0.0,
            'raw_loss': 0.0,
        }

    pos_labels = labels[pos_mask]
    pos_target = target_support[pos_mask].gather(
        1, pos_labels[:, None]).squeeze(1)
    pos_pred = pred_support[pos_mask].gather(
        1, pos_labels[:, None]).squeeze(1)
    pos_weights = assign_metrics[pos_mask].clamp(min=0.0)
    support_mask = torch.isfinite(pos_target) & torch.isfinite(pos_pred)
    if min_target_logprob is not None:
        support_mask = support_mask & (
            pos_target >= float(min_target_logprob))
    num_supported = int(support_mask.sum().detach().cpu().item())
    if not torch.any(support_mask):
        zero = pred_log_prob.sum() * 0.0
        return zero, {
            'num_pos': num_pos,
            'num_supported_pos': 0,
            'active_violation_count': 0,
            'mean_target_log_prob': 0.0,
            'mean_pred_log_prob': 0.0,
            'mean_logprob_drop': 0.0,
            'raw_loss': 0.0,
        }

    logprob_drop = pos_target.detach() - pos_pred
    raw = F.relu(logprob_drop - float(max_logprob_drop))
    raw = torch.where(support_mask, raw, torch.zeros_like(raw))
    weights = torch.where(
        support_mask, pos_weights, torch.zeros_like(pos_weights))
    denom = weights.sum().clamp(min=1.0)
    loss = (raw * weights).sum() / denom
    active = support_mask & (raw > 0)

    supported_target = pos_target[support_mask]
    supported_pred = pos_pred[support_mask]
    supported_drop = logprob_drop[support_mask]
    return loss, {
        'num_pos': num_pos,
        'num_supported_pos': num_supported,
        'active_violation_count': int(
            active.sum().detach().cpu().item()),
        'mean_target_log_prob': float(
            supported_target.mean().detach().cpu().item())
        if supported_target.numel() else 0.0,
        'mean_pred_log_prob': float(
            supported_pred.mean().detach().cpu().item())
        if supported_pred.numel() else 0.0,
        'mean_logprob_drop': float(
            supported_drop.mean().detach().cpu().item())
        if supported_drop.numel() else 0.0,
        'raw_loss': float(loss.detach().cpu().item()),
    }


def gaussian_support_negative_focal_loss(
        cls_logits: Tensor,
        labels: Tensor,
        log_prob: Tensor,
        valid_mask: Tensor,
        min_score: float | None = 0.05,
        min_logprob_gap: float = 1.0,
        gamma: float = 2.0,
        gap_scale: float = 8.0,
        max_extra_weight: float = 2.0,
        max_hardneg: int = 1,
) -> tuple[Tensor, dict[str, float | int]]:
    """Penalize low-support, high-score classes on negative locations.

    This is a train-only support-aware negative loss.  Unlike GWD/KLD/NWD, the
    Gaussian is not a bbox-overlap distance; it is a class-conditional support
    model over semantic geometry.  Positive locations are excluded so the loss
    cannot directly demote correct positives or disturb their AP ordering.
    """
    if cls_logits.numel() == 0 or log_prob.numel() == 0:
        zero = cls_logits.sum() * 0.0
        return zero, {
            'num_locations': 0,
            'num_positive_locations': 0,
            'num_negative_locations': 0,
            'num_candidate_pairs': 0,
            'num_score_gated_pairs': 0,
            'num_selected_pairs': 0,
            'mean_selected_score': 0.0,
            'mean_logprob_gap': 0.0,
            'mean_pair_weight': 0.0,
            'raw_loss': 0.0,
        }

    num_classes = int(min(cls_logits.shape[-1], log_prob.shape[-1]))
    logits = cls_logits.reshape(-1, cls_logits.shape[-1])[:, :num_classes]
    support = log_prob.reshape(-1, log_prob.shape[-1])[:, :num_classes].to(
        device=logits.device, dtype=logits.dtype)
    labels = labels.reshape(-1).to(device=logits.device, dtype=torch.long)
    valid = valid_mask.to(device=logits.device, dtype=torch.bool)[
        :num_classes]

    num_locations = int(logits.shape[0])
    label_in_range = (labels >= 0) & (labels < num_classes)
    negative_mask = labels >= num_classes
    num_positive = int(label_in_range.sum().detach().cpu().item())
    num_negative = int(negative_mask.sum().detach().cpu().item())
    if (not torch.any(negative_mask)) or (not torch.any(valid)):
        zero = cls_logits.sum() * 0.0
        return zero, {
            'num_locations': num_locations,
            'num_positive_locations': num_positive,
            'num_negative_locations': num_negative,
            'num_candidate_pairs': 0,
            'num_score_gated_pairs': 0,
            'num_selected_pairs': 0,
            'mean_selected_score': 0.0,
            'mean_logprob_gap': 0.0,
            'mean_pair_weight': 0.0,
            'raw_loss': 0.0,
        }

    neg_logits = logits[negative_mask]
    neg_support = support[negative_mask]
    valid_2d = valid[None, :].expand_as(neg_support)
    best_support = neg_support.masked_fill(~valid_2d, -torch.inf).max(
        dim=1).values
    finite_best = torch.isfinite(best_support)
    gap = best_support[:, None] - neg_support
    candidate = (
        finite_best[:, None]
        & valid_2d
        & (gap >= float(min_logprob_gap)))
    scores = torch.sigmoid(neg_logits)
    num_candidate_pairs = int(candidate.sum().detach().cpu().item())

    score_gated = torch.zeros_like(candidate)
    if min_score is not None:
        score_gated = (scores < float(min_score)) & candidate
        candidate = candidate & ~score_gated
    num_score_gated_pairs = int(score_gated.sum().detach().cpu().item())
    if not torch.any(candidate):
        zero = cls_logits.sum() * 0.0
        return zero, {
            'num_locations': num_locations,
            'num_positive_locations': num_positive,
            'num_negative_locations': num_negative,
            'num_candidate_pairs': num_candidate_pairs,
            'num_score_gated_pairs': num_score_gated_pairs,
            'num_selected_pairs': 0,
            'mean_selected_score': 0.0,
            'mean_logprob_gap': 0.0,
            'mean_pair_weight': 0.0,
            'raw_loss': 0.0,
        }

    k = min(max(int(max_hardneg), 1), num_classes)
    masked_score = scores.masked_fill(~candidate, -torch.inf)
    top_score, top_idx = torch.topk(masked_score, k=k, dim=1)
    top_valid = torch.isfinite(top_score)
    selected_logits = neg_logits.gather(1, top_idx)
    selected_gap = gap.gather(1, top_idx)
    selected_scores = scores.gather(1, top_idx)

    raw = F.softplus(selected_logits)
    score_weight = selected_scores.detach().clamp(min=0.0, max=1.0).pow(
        max(float(gamma), 0.0))
    gap_scale = max(float(gap_scale), 1e-6)
    extra = ((selected_gap.detach() - float(min_logprob_gap)) / gap_scale)
    extra = extra.clamp(min=0.0, max=max(float(max_extra_weight), 0.0))
    pair_weight = (1.0 + extra) * score_weight
    weighted = torch.where(
        top_valid, raw * pair_weight, torch.zeros_like(raw))
    denom = top_valid.to(dtype=logits.dtype).sum().clamp(min=1.0)
    loss = weighted.sum() / denom

    selected_scores = selected_scores[top_valid]
    selected_gap = selected_gap[top_valid]
    selected_weight = pair_weight[top_valid]
    return loss, {
        'num_locations': num_locations,
        'num_positive_locations': num_positive,
        'num_negative_locations': num_negative,
        'num_candidate_pairs': num_candidate_pairs,
        'num_score_gated_pairs': num_score_gated_pairs,
        'num_selected_pairs': int(top_valid.sum().detach().cpu().item()),
        'mean_selected_score': float(
            selected_scores.mean().detach().cpu().item())
        if selected_scores.numel() else 0.0,
        'mean_logprob_gap': float(selected_gap.mean().detach().cpu().item())
        if selected_gap.numel() else 0.0,
        'mean_pair_weight': float(
            selected_weight.mean().detach().cpu().item())
        if selected_weight.numel() else 0.0,
        'raw_loss': float(loss.detach().cpu().item()),
    }


def gaussian_ap_safe_support_projection_loss(
        cls_logits: Tensor,
        labels: Tensor,
        log_prob: Tensor,
        valid_mask: Tensor,
        assign_metrics: Tensor,
        min_logprob_gap: float = 1.5,
        protect_margin: float = 0.20,
        rank_margin: float = 0.05,
        margin_temperature: float = 0.25,
        support_temperature: float = 1.0,
        min_hardneg_score: float | None = 0.05,
        min_hardneg_logit: float | None = None,
        max_hardneg: int = 1,
        max_budget: float = 1.0,
        assign_metric_power: float = 1.0,
        detach_gt_logit: bool = True,
) -> tuple[Tensor, dict[str, float | int]]:
    """AP-safe positive-location projection for semantic support rivals.

    The Gaussian distribution is used as a semantic support model over class
    scale/geometry, not as a bbox overlap metric.  For each assigned positive
    location, the loss searches rival classes whose support is far below the
    GT class support.  A rival can be pushed down only when the current GT-vs-
    rival logit margin already exceeds ``protect_margin``; therefore AP-
    critical positives with small ranking margin receive no suppression budget.
    """
    if cls_logits.numel() == 0 or log_prob.numel() == 0:
        zero = cls_logits.sum() * 0.0
        return zero, {
            'num_pos': 0,
            'num_supported_pos': 0,
            'num_candidate_pairs': 0,
            'num_score_gated_pairs': 0,
            'num_hardneg': 0,
            'num_pos_with_hardneg': 0,
            'active_projection_count': 0,
            'mean_logprob_gap': 0.0,
            'mean_gt_margin': 0.0,
            'mean_safe_budget': 0.0,
            'mean_pair_weight': 0.0,
            'raw_loss': 0.0,
        }

    num_classes = int(min(cls_logits.shape[-1], log_prob.shape[-1]))
    logits = cls_logits.reshape(-1, cls_logits.shape[-1])[:, :num_classes]
    support = log_prob.reshape(-1, log_prob.shape[-1])[:, :num_classes].to(
        device=logits.device, dtype=logits.dtype)
    labels = labels.reshape(-1).to(device=logits.device, dtype=torch.long)
    assign_metrics = assign_metrics.reshape(-1).to(
        device=logits.device, dtype=logits.dtype)
    valid = valid_mask.to(device=logits.device, dtype=torch.bool)[
        :num_classes]

    label_in_range = (labels >= 0) & (labels < num_classes)
    safe_labels = labels.clamp(min=0, max=max(num_classes - 1, 0))
    pos_mask = label_in_range & valid[safe_labels]
    num_pos = int(pos_mask.sum().detach().cpu().item())
    if not torch.any(pos_mask) or not torch.any(valid):
        zero = cls_logits.sum() * 0.0
        return zero, {
            'num_pos': num_pos,
            'num_supported_pos': 0,
            'num_candidate_pairs': 0,
            'num_score_gated_pairs': 0,
            'num_hardneg': 0,
            'num_pos_with_hardneg': 0,
            'active_projection_count': 0,
            'mean_logprob_gap': 0.0,
            'mean_gt_margin': 0.0,
            'mean_safe_budget': 0.0,
            'mean_pair_weight': 0.0,
            'raw_loss': 0.0,
        }

    pos_logits = logits[pos_mask]
    pos_support = support[pos_mask]
    pos_labels = labels[pos_mask]
    pos_weights = assign_metrics[pos_mask].clamp(min=0.0)
    gt_support = pos_support.gather(1, pos_labels[:, None]).squeeze(1)
    finite_support = torch.isfinite(pos_support)
    finite_gt = torch.isfinite(gt_support)
    num_supported_pos = int(finite_gt.sum().detach().cpu().item())
    class_idx = torch.arange(num_classes, device=logits.device)
    valid_2d = valid[None, :].expand_as(pos_support)
    gap = gt_support[:, None] - pos_support
    candidate = (
        finite_gt[:, None]
        & finite_support
        & valid_2d
        & (class_idx[None, :] != pos_labels[:, None])
        & (gap >= float(min_logprob_gap)))
    num_candidate_pairs = int(candidate.sum().detach().cpu().item())

    scores = torch.sigmoid(pos_logits)
    score_gated = torch.zeros_like(candidate)
    if min_hardneg_score is not None:
        low_score = scores < float(min_hardneg_score)
        score_gated = score_gated | (candidate & low_score)
        candidate = candidate & ~low_score
    if min_hardneg_logit is not None:
        low_logit = pos_logits < float(min_hardneg_logit)
        score_gated = score_gated | (candidate & low_logit)
        candidate = candidate & ~low_logit
    num_score_gated_pairs = int(score_gated.sum().detach().cpu().item())
    if not torch.any(candidate):
        zero = cls_logits.sum() * 0.0
        return zero, {
            'num_pos': num_pos,
            'num_supported_pos': num_supported_pos,
            'num_candidate_pairs': num_candidate_pairs,
            'num_score_gated_pairs': num_score_gated_pairs,
            'num_hardneg': 0,
            'num_pos_with_hardneg': 0,
            'active_projection_count': 0,
            'mean_logprob_gap': 0.0,
            'mean_gt_margin': 0.0,
            'mean_safe_budget': 0.0,
            'mean_pair_weight': 0.0,
            'raw_loss': 0.0,
        }

    k = min(max(int(max_hardneg), 1), num_classes)
    masked_logits = pos_logits.masked_fill(~candidate, -torch.inf)
    top_logits, top_idx = torch.topk(masked_logits, k=k, dim=1)
    top_valid = torch.isfinite(top_logits)
    selected_logits = pos_logits.gather(1, top_idx)
    selected_gap = gap.gather(1, top_idx)
    selected_scores = scores.gather(1, top_idx)
    gt_logits = pos_logits.gather(1, pos_labels[:, None]).squeeze(1)
    gt_ref = gt_logits.detach() if detach_gt_logit else gt_logits

    current_margin = gt_logits.detach()[:, None] - selected_logits.detach()
    margin_temperature = max(float(margin_temperature), 1e-6)
    safe_budget = (
        (current_margin - float(protect_margin)) / margin_temperature)
    safe_budget = safe_budget.clamp(
        min=0.0, max=max(float(max_budget), 0.0))
    support_temperature = max(float(support_temperature), 1e-6)
    support_weight = torch.sigmoid(
        (selected_gap.detach() - float(min_logprob_gap))
        / support_temperature)
    metric_power = max(float(assign_metric_power), 0.0)
    assign_weight = pos_weights[:, None].pow(metric_power)
    pair_weight = assign_weight * safe_budget * support_weight
    pair_weight = torch.where(
        top_valid, pair_weight, torch.zeros_like(pair_weight))

    raw = F.softplus(selected_logits - gt_ref[:, None] + float(rank_margin))
    loss = (raw * pair_weight).sum() / pair_weight.sum().clamp(min=1.0)

    selected_valid_gap = selected_gap[top_valid]
    selected_valid_margin = current_margin[top_valid]
    selected_valid_budget = safe_budget[top_valid]
    selected_valid_weight = pair_weight[top_valid]
    active = top_valid & (pair_weight > 0)
    return loss, {
        'num_pos': num_pos,
        'num_supported_pos': num_supported_pos,
        'num_candidate_pairs': num_candidate_pairs,
        'num_score_gated_pairs': num_score_gated_pairs,
        'num_hardneg': int(top_valid.sum().detach().cpu().item()),
        'num_pos_with_hardneg': int(
            top_valid.any(dim=1).sum().detach().cpu().item()),
        'active_projection_count': int(
            active.sum().detach().cpu().item()),
        'mean_logprob_gap': float(
            selected_valid_gap.mean().detach().cpu().item())
        if selected_valid_gap.numel() else 0.0,
        'mean_gt_margin': float(
            selected_valid_margin.mean().detach().cpu().item())
        if selected_valid_margin.numel() else 0.0,
        'mean_safe_budget': float(
            selected_valid_budget.mean().detach().cpu().item())
        if selected_valid_budget.numel() else 0.0,
        'mean_pair_weight': float(
            selected_valid_weight.mean().detach().cpu().item())
        if selected_valid_weight.numel() else 0.0,
        'raw_loss': float(loss.detach().cpu().item()),
        'mean_selected_score': float(
            selected_scores[top_valid].mean().detach().cpu().item())
        if torch.any(top_valid) else 0.0,
    }


def gaussian_ap_constrained_support_ranking_projection_loss(
        cls_logits: Tensor,
        labels: Tensor,
        log_prob: Tensor,
        valid_mask: Tensor,
        assign_metrics: Tensor,
        min_logprob_gap: float = 1.5,
        protect_margin: float = 0.20,
        margin_temperature: float = 0.25,
        support_temperature: float = 1.0,
        min_hardneg_score: float | None = 0.05,
        min_hardneg_logit: float | None = None,
        max_hardneg: int = 1,
        max_budget: float = 1.0,
        assign_metric_power: float = 1.0,
) -> tuple[Tensor, Tensor, dict[str, float | int]]:
    """Project low-support rival logits only inside an AP-safe budget.

    This is a train-time semantic-scale projection.  It never modifies the
    assigned positive class logit.  Rival-class demotion is allowed only when
    the current GT-vs-rival logit margin already exceeds ``protect_margin``.
    Gaussian support is used over semantic scale/geometry, not as a box IoU,
    GWD, KLD, or NWD surrogate.
    """
    zero = cls_logits.sum() * 0.0
    projected = cls_logits.clone()
    if cls_logits.numel() == 0 or log_prob.numel() == 0:
        return zero, projected, {
            'num_pos': 0,
            'num_supported_pos': 0,
            'num_candidate_pairs': 0,
            'num_score_gated_pairs': 0,
            'num_hardneg': 0,
            'num_pos_with_hardneg': 0,
            'active_projection_count': 0,
            'protected_positive_count': 0,
            'mean_logprob_gap': 0.0,
            'mean_gt_margin': 0.0,
            'mean_safe_budget': 0.0,
            'mean_projection_delta': 0.0,
            'mean_pair_weight': 0.0,
            'mean_selected_score': 0.0,
            'raw_loss': 0.0,
        }

    original_shape = cls_logits.shape
    num_classes = int(min(cls_logits.shape[-1], log_prob.shape[-1]))
    logits = cls_logits.reshape(-1, cls_logits.shape[-1])[:, :num_classes]
    projected_flat = projected.reshape(-1, projected.shape[-1])
    support = log_prob.reshape(-1, log_prob.shape[-1])[:, :num_classes].to(
        device=logits.device, dtype=logits.dtype)
    labels = labels.reshape(-1).to(device=logits.device, dtype=torch.long)
    assign_metrics = assign_metrics.reshape(-1).to(
        device=logits.device, dtype=logits.dtype)
    valid = valid_mask.to(device=logits.device, dtype=torch.bool)[
        :num_classes]

    label_in_range = (labels >= 0) & (labels < num_classes)
    safe_labels = labels.clamp(min=0, max=max(num_classes - 1, 0))
    pos_mask = label_in_range & valid[safe_labels]
    num_pos = int(pos_mask.sum().detach().cpu().item())
    if not torch.any(pos_mask) or not torch.any(valid):
        return zero, projected.reshape(original_shape), {
            'num_pos': num_pos,
            'num_supported_pos': 0,
            'num_candidate_pairs': 0,
            'num_score_gated_pairs': 0,
            'num_hardneg': 0,
            'num_pos_with_hardneg': 0,
            'active_projection_count': 0,
            'protected_positive_count': num_pos,
            'mean_logprob_gap': 0.0,
            'mean_gt_margin': 0.0,
            'mean_safe_budget': 0.0,
            'mean_projection_delta': 0.0,
            'mean_pair_weight': 0.0,
            'mean_selected_score': 0.0,
            'raw_loss': 0.0,
        }

    pos_idx = torch.nonzero(pos_mask, as_tuple=False).view(-1)
    pos_logits = logits[pos_idx]
    pos_support = support[pos_idx]
    pos_labels = labels[pos_idx]
    pos_weights = assign_metrics[pos_idx].clamp(min=0.0)
    gt_support = pos_support.gather(1, pos_labels[:, None]).squeeze(1)
    finite_support = torch.isfinite(pos_support)
    finite_gt = torch.isfinite(gt_support)
    num_supported_pos = int(finite_gt.sum().detach().cpu().item())
    class_idx = torch.arange(num_classes, device=logits.device)
    valid_2d = valid[None, :].expand_as(pos_support)
    gap = gt_support[:, None] - pos_support
    candidate = (
        finite_gt[:, None]
        & finite_support
        & valid_2d
        & (class_idx[None, :] != pos_labels[:, None])
        & (gap >= float(min_logprob_gap)))
    num_candidate_pairs = int(candidate.sum().detach().cpu().item())

    scores = torch.sigmoid(pos_logits)
    score_gated = torch.zeros_like(candidate)
    if min_hardneg_score is not None:
        low_score = scores < float(min_hardneg_score)
        score_gated = score_gated | (candidate & low_score)
        candidate = candidate & ~low_score
    if min_hardneg_logit is not None:
        low_logit = pos_logits < float(min_hardneg_logit)
        score_gated = score_gated | (candidate & low_logit)
        candidate = candidate & ~low_logit
    num_score_gated_pairs = int(score_gated.sum().detach().cpu().item())
    if not torch.any(candidate):
        return zero, projected.reshape(original_shape), {
            'num_pos': num_pos,
            'num_supported_pos': num_supported_pos,
            'num_candidate_pairs': num_candidate_pairs,
            'num_score_gated_pairs': num_score_gated_pairs,
            'num_hardneg': 0,
            'num_pos_with_hardneg': 0,
            'active_projection_count': 0,
            'protected_positive_count': num_pos,
            'mean_logprob_gap': 0.0,
            'mean_gt_margin': 0.0,
            'mean_safe_budget': 0.0,
            'mean_projection_delta': 0.0,
            'mean_pair_weight': 0.0,
            'mean_selected_score': 0.0,
            'raw_loss': 0.0,
        }

    k = min(max(int(max_hardneg), 1), num_classes)
    masked_logits = pos_logits.masked_fill(~candidate, -torch.inf)
    top_logits, top_idx = torch.topk(masked_logits, k=k, dim=1)
    top_valid = torch.isfinite(top_logits)
    selected_logits = pos_logits.gather(1, top_idx)
    selected_gap = gap.gather(1, top_idx)
    selected_scores = scores.gather(1, top_idx)
    gt_logits = pos_logits.gather(1, pos_labels[:, None]).squeeze(1)
    current_margin = gt_logits.detach()[:, None] - selected_logits.detach()

    margin_temperature = max(float(margin_temperature), 1e-6)
    safe_budget = (
        (current_margin - float(protect_margin)) / margin_temperature)
    safe_budget = safe_budget.clamp(
        min=0.0, max=max(float(max_budget), 0.0))
    support_temperature = max(float(support_temperature), 1e-6)
    support_weight = torch.sigmoid(
        (selected_gap.detach() - float(min_logprob_gap))
        / support_temperature)
    metric_power = max(float(assign_metric_power), 0.0)
    pair_weight = pos_weights[:, None].pow(metric_power)
    projection_delta = safe_budget * support_weight * pair_weight
    projection_delta = torch.where(
        top_valid, projection_delta, torch.zeros_like(projection_delta))
    active = top_valid & (projection_delta > 0)

    projected_pos = pos_logits.clone()
    projected_selected = selected_logits - projection_delta
    projected_pos.scatter_(
        1,
        top_idx,
        torch.where(active, projected_selected, selected_logits))
    projected_flat[pos_idx, :num_classes] = projected_pos

    target = (selected_logits.detach() - projection_delta.detach())
    raw = 0.5 * (selected_logits - target).pow(2)
    active_weight = active.to(dtype=logits.dtype)
    loss = (raw * active_weight).sum() / active_weight.sum().clamp(min=1.0)

    selected_valid_gap = selected_gap[top_valid]
    selected_valid_margin = current_margin[top_valid]
    selected_valid_budget = safe_budget[top_valid]
    selected_valid_delta = projection_delta[top_valid]
    selected_valid_weight = pair_weight[top_valid]
    return loss, projected.reshape(original_shape), {
        'num_pos': num_pos,
        'num_supported_pos': num_supported_pos,
        'num_candidate_pairs': num_candidate_pairs,
        'num_score_gated_pairs': num_score_gated_pairs,
        'num_hardneg': int(top_valid.sum().detach().cpu().item()),
        'num_pos_with_hardneg': int(
            top_valid.any(dim=1).sum().detach().cpu().item()),
        'active_projection_count': int(
            active.sum().detach().cpu().item()),
        'protected_positive_count': num_pos,
        'mean_logprob_gap': float(
            selected_valid_gap.mean().detach().cpu().item())
        if selected_valid_gap.numel() else 0.0,
        'mean_gt_margin': float(
            selected_valid_margin.mean().detach().cpu().item())
        if selected_valid_margin.numel() else 0.0,
        'mean_safe_budget': float(
            selected_valid_budget.mean().detach().cpu().item())
        if selected_valid_budget.numel() else 0.0,
        'mean_projection_delta': float(
            selected_valid_delta.mean().detach().cpu().item())
        if selected_valid_delta.numel() else 0.0,
        'mean_pair_weight': float(
            selected_valid_weight.mean().detach().cpu().item())
        if selected_valid_weight.numel() else 0.0,
        'mean_selected_score': float(
            selected_scores[top_valid].mean().detach().cpu().item())
        if torch.any(top_valid) else 0.0,
        'raw_loss': float(loss.detach().cpu().item()),
    }
