"""Sensitivity-guided channel mask utilities for FOCUS-OVD attribution."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


def _normalize_abs(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0)
    values = np.abs(values)
    max_value = float(values.max()) if values.size else 0.0
    if max_value <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return values / max_value


def build_channel_mask(
        sensitivity: Iterable[float],
        true_sv_importance: Optional[Iterable[float]] = None,
        max_strength: float = 0.05) -> np.ndarray:
    """Build a multiplicative mask clipped to low-strength intervention."""
    strength = min(max(float(max_strength), 0.0), 0.05)
    sensitivity_score = _normalize_abs(np.asarray(list(sensitivity), dtype=np.float32))
    if true_sv_importance is None:
        protection = np.zeros_like(sensitivity_score)
    else:
        protection = _normalize_abs(
            np.asarray(list(true_sv_importance), dtype=np.float32))
        if protection.shape != sensitivity_score.shape:
            raise ValueError("true_sv_importance must match sensitivity length")
    mask_score = sensitivity_score * (1.0 - 0.5 * protection)
    mask_score = np.clip(mask_score, 0.0, 1.0)
    return np.clip(1.0 - strength * mask_score, 1.0 - strength, 1.0)


def summarize_sensitive_channels(
        rotation: Iterable[float],
        context: Iterable[float],
        support_intervention: Iterable[float],
        true_sv_importance: Iterable[float],
        false_sv_importance: Iterable[float],
        max_strength: float = 0.05) -> list[dict[str, float | int | str]]:
    rotation_arr = _normalize_abs(np.asarray(list(rotation), dtype=np.float32))
    context_arr = _normalize_abs(np.asarray(list(context), dtype=np.float32))
    support_arr = _normalize_abs(
        np.asarray(list(support_intervention), dtype=np.float32))
    true_arr = _normalize_abs(np.asarray(list(true_sv_importance), dtype=np.float32))
    false_arr = _normalize_abs(np.asarray(list(false_sv_importance), dtype=np.float32))
    n = min(len(rotation_arr), len(context_arr), len(support_arr),
            len(true_arr), len(false_arr))
    if n == 0:
        return []
    shortcut_score = (rotation_arr[:n] + context_arr[:n] +
                      support_arr[:n] + false_arr[:n]) / 4.0
    protected_score = true_arr[:n]
    mask_score = np.clip(shortcut_score * (1.0 - 0.5 * protected_score), 0.0, 1.0)
    mask = build_channel_mask(mask_score, protected_score, max_strength=max_strength)
    rows = []
    for idx in range(n):
        rows.append({
            "channel_id": idx,
            "sensitivity_rotation": float(rotation_arr[idx]),
            "sensitivity_context": float(context_arr[idx]),
            "sensitivity_support_intervention": float(support_arr[idx]),
            "true_sv_importance": float(true_arr[idx]),
            "false_sv_importance": float(false_arr[idx]),
            "mask_score": float(mask_score[idx]),
            "mask_value": float(mask[idx]),
            "recommended_action": (
                "audit_only" if mask_score[idx] < 0.5 else "candidate_low_strength_mask"),
        })
    return rows
