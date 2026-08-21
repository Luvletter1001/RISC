"""Rotation/orbit consistency teacher helpers for FOCUS-OVD."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OrbitTeacherLabel:
    orbit_stability_score: float
    teacher_label: str
    teacher_weight: float


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def assign_orbit_teacher_label(
        *,
        sv_score_mean: float,
        sv_score_std: float,
        class_consistency: float,
        box_consistency: float,
        support_margin_consistency: float,
        orientation_consistency: float) -> OrbitTeacherLabel:
    sv_mean = _clip(sv_score_mean)
    sv_std = max(0.0, float(sv_score_std))
    class_score = _clip(class_consistency)
    box_score = _clip(box_consistency)
    support_score = _clip(support_margin_consistency)
    orient_score = _clip(orientation_consistency)
    stability = (
        0.25 * class_score +
        0.25 * box_score +
        0.20 * support_score +
        0.15 * orient_score +
        0.15 * (1.0 - min(1.0, sv_std))
    )
    stability = _clip(stability)

    if sv_mean < 0.20 and stability < 0.50:
        label = "background_or_padding"
    elif stability >= 0.75 and sv_std <= 0.10 and sv_mean >= 0.50:
        label = "stable_true_sv_candidate"
    elif stability <= 0.45 or sv_std >= 0.25:
        label = "unstable_shortcut_candidate"
    else:
        label = "ambiguous"
    weight = stability if label == "stable_true_sv_candidate" else 1.0 - stability
    if label == "ambiguous":
        weight = min(weight, 0.35)
    return OrbitTeacherLabel(stability, label, _clip(weight))
