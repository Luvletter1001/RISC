import math
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FourierOrientationLearner(nn.Module):
    """Estimate local dominant orientation from Fourier energy.

    The module is intentionally self-contained and has no trainable parameters.
    It consumes dense feature maps or RoI features and returns a periodic
    Fourier code that can condition a support adapter.
    """

    def __init__(
            self,
            patch_size: int = 7,
            num_angle_bins: int = 36,
            harmonic_orders: Iterable[int] = (2, 4, 6),
            confidence_type: str = "peak_entropy",
            min_confidence: float = 0.15,
            detach_orientation: bool = True,
            eps: float = 1e-8) -> None:
        super().__init__()
        if patch_size % 2 != 1:
            raise ValueError("patch_size must be odd")
        if num_angle_bins <= 1:
            raise ValueError("num_angle_bins must be greater than 1")
        self.patch_size = int(patch_size)
        self.num_angle_bins = int(num_angle_bins)
        self.harmonic_orders = tuple(int(v) for v in harmonic_orders)
        self.confidence_type = confidence_type
        self.min_confidence = float(min_confidence)
        self.detach_orientation = bool(detach_orientation)
        self.eps = float(eps)

        self._init_frequency_bins()

    def _init_frequency_bins(self) -> None:
        m = self.patch_size
        freq = torch.fft.fftfreq(m, d=1.0) * m
        fy, fx = torch.meshgrid(freq, freq, indexing="ij")
        rho = torch.sqrt(fx.pow(2) + fy.pow(2))
        theta = torch.remainder(torch.atan2(fy, fx), math.pi)
        mask = rho > self.eps

        bin_ids = torch.clamp(
            torch.floor(theta[mask] / math.pi * self.num_angle_bins).long(),
            max=self.num_angle_bins - 1)
        bin_centers = (
            torch.arange(self.num_angle_bins, dtype=torch.float32) + 0.5
        ) * math.pi / self.num_angle_bins

        self.register_buffer("valid_mask_flat", mask.reshape(-1))
        self.register_buffer("valid_rhos", rho[mask])
        self.register_buffer("valid_bin_ids", bin_ids)
        self.register_buffer("bin_centers", bin_centers)

    def _patch_energy(self, feat: Tensor) -> Tuple[Tensor, int, int]:
        if feat.dim() != 4:
            raise ValueError(f"expected [B,C,H,W] tensor, got {tuple(feat.shape)}")
        batch, channels, height, width = feat.shape
        pad = self.patch_size // 2
        padded = F.pad(feat, (pad, pad, pad, pad), mode="replicate")
        patches = F.unfold(
            padded,
            kernel_size=self.patch_size,
            padding=0,
            stride=1)
        patches = patches.transpose(1, 2).reshape(
            batch * height * width,
            channels,
            self.patch_size,
            self.patch_size)
        patch_energy = torch.linalg.vector_norm(patches, ord=2, dim=1)
        return patch_energy, height, width

    def _angle_histogram(self, patch_energy: Tensor) -> Tensor:
        spectrum = torch.fft.fft2(patch_energy, dim=(-2, -1), norm="ortho")
        magnitude = spectrum.abs().reshape(patch_energy.shape[0], -1)
        valid_energy = magnitude[:, self.valid_mask_flat]
        weighted = valid_energy * self.valid_rhos.to(valid_energy.device)

        hist = weighted.new_zeros(weighted.shape[0], self.num_angle_bins)
        bin_ids = self.valid_bin_ids.to(weighted.device)
        hist.scatter_add_(1, bin_ids.expand(weighted.shape[0], -1), weighted)
        hist = torch.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)
        return hist

    def _theta_from_histogram(self, hist: Tensor) -> Tuple[Tensor, Tensor]:
        energy_sum = hist.sum(dim=-1)
        probs = hist / (energy_sum.unsqueeze(-1) + self.eps)
        centers = self.bin_centers.to(hist.device, hist.dtype)

        sin2 = torch.sum(probs * torch.sin(2.0 * centers), dim=-1)
        cos2 = torch.sum(probs * torch.cos(2.0 * centers), dim=-1)
        theta = 0.5 * torch.atan2(sin2, cos2)
        theta = torch.remainder(theta, math.pi)

        max_energy = hist.max(dim=-1).values
        mean_energy = hist.mean(dim=-1)
        peakiness = (max_energy - mean_energy).clamp_min(0.0) / (
            max_energy + self.eps)
        texture_gate = energy_sum / (energy_sum + 1e-4)
        confidence = torch.nan_to_num(peakiness * texture_gate, nan=0.0)
        confidence = confidence.clamp(0.0, 1.0)
        return theta, confidence

    def fourier_code(self, theta: Tensor) -> Tensor:
        codes = []
        for order in self.harmonic_orders:
            codes.append(torch.sin(float(order) * theta))
            codes.append(torch.cos(float(order) * theta))
        return torch.stack(codes, dim=-1)

    def forward(self, feat: Tensor) -> Dict[str, Tensor]:
        patch_energy, height, width = self._patch_energy(feat.float())
        hist = self._angle_histogram(patch_energy)
        theta, confidence = self._theta_from_histogram(hist)

        batch = feat.shape[0]
        theta = theta.reshape(batch, height, width)
        confidence = confidence.reshape(batch, height, width)
        if self.detach_orientation:
            theta = theta.detach()
            confidence = confidence.detach()
        code = self.fourier_code(theta)
        hist = hist.reshape(batch, height, width, self.num_angle_bins)

        return {
            "theta": torch.nan_to_num(theta, nan=0.0),
            "confidence": torch.nan_to_num(confidence, nan=0.0).clamp(0.0, 1.0),
            "fourier_code": torch.nan_to_num(code, nan=0.0),
            "angle_energy": torch.nan_to_num(hist, nan=0.0),
        }
