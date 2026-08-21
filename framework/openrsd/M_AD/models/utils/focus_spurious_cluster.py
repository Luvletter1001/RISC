"""Spurious small-vehicle cluster helpers for FOCUS-OVD attribution.

The first version is diagnostic-only: it assigns conservative cluster names and
scores from audit labels and available scalar evidence, but it never changes
model outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any, Iterable, Mapping


FALSE_SV_LABELS = {"non_vehicle_background", "non_vehicle_object_conflict"}
TRUE_VEHICLE_LABELS = {
    "true_vehicle_annot_missing",
    "true_vehicle_annot_present_but_missed_match",
}
TRUE_SV_LABELS = {"true_small_vehicle", "matched_true_small_vehicle"}
DEGENERATE_CATEGORIES = {"degenerate_large_sv_box"}
PADDING_CATEGORIES = {"padding_artifact"}


@dataclass(frozen=True)
class SpuriousClusterResult:
    cluster_id: int
    cluster_name: str
    spurious_sv_score: float
    use_as_hard_negative: bool
    reason: str


def _text(row: Mapping[str, Any], key: str) -> str:
    return str(row.get(key, "") or "").strip()


def _float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def compute_spurious_sv_score(row: Mapping[str, Any]) -> float:
    """Return a bounded shortcut-risk score from available scalar signals."""
    sv_margin = _float(row, "sv_margin", _float(row, "pred_score", 0.0))
    orientation_conf = _float(row, "orientation_confidence", 0.5)
    orbit_stability = _float(row, "orbit_stability", 0.5)
    sv_sim = _float(row, "support_similarity_sv", 0.0)
    lv_sim = _float(row, "support_similarity_lv", 0.0)
    pred_score = _float(row, "pred_score", 0.0)

    risk = 0.0
    risk += 0.35 * _sigmoid(3.0 * sv_margin)
    risk += 0.20 * _sigmoid(4.0 * (sv_sim - lv_sim))
    risk += 0.20 * (1.0 - max(0.0, min(1.0, orientation_conf)))
    risk += 0.15 * (1.0 - max(0.0, min(1.0, orbit_stability)))
    risk += 0.10 * max(0.0, min(1.0, pred_score))
    return max(0.0, min(1.0, risk))


def classify_spurious_sv_candidate(
        row: Mapping[str, Any]) -> SpuriousClusterResult:
    """Classify one candidate without treating uncertain rows as negatives."""
    category = _text(row, "audit_category")
    label = _text(row, "human_label")
    nearest_negative = _text(row, "nearest_negative_class")
    score = compute_spurious_sv_score(row)

    if category in PADDING_CATEGORIES:
        return SpuriousClusterResult(
            6, "padding_artifact_cluster", 0.0, False,
            "padding artifact is excluded from corrected-FSV negatives")
    if category in DEGENERATE_CATEGORIES:
        return SpuriousClusterResult(
            5, "degenerate_large_sv_cluster", 0.0, False,
            "degenerate large-SV boxes are excluded from corrected-FSV negatives")
    if label in TRUE_VEHICLE_LABELS:
        return SpuriousClusterResult(
            2, "annotation_missing_true_vehicle_cluster", 0.0, False,
            "annotation-missing true vehicles are preserve positives")
    if label in TRUE_SV_LABELS:
        return SpuriousClusterResult(
            1, "true_sv_cluster", 0.0, False,
            "verified true small vehicle")
    if label == "non_vehicle_object_conflict":
        return SpuriousClusterResult(
            4, "object_conflict_cluster", max(score, 0.5), True,
            "audited non-vehicle object conflict")
    if label in FALSE_SV_LABELS:
        return SpuriousClusterResult(
            3, "corrected_false_sv_context_cluster", max(score, 0.6), True,
            "audited corrected false small-vehicle")
    if nearest_negative and nearest_negative.lower() not in {
            "small-vehicle", "small_vehicle", "none", "nan"}:
        return SpuriousClusterResult(
            8, "class_migration_cluster", score, False,
            "nearest non-SV class suggests class migration audit")
    if category == "valid_unmatched_sv":
        return SpuriousClusterResult(
            7, "ambiguous_cluster", score, False,
            "valid unmatched SV without decisive human label")
    return SpuriousClusterResult(
        7, "ambiguous_cluster", score, False,
        "insufficient evidence for hard-negative use")


def annotate_spurious_clusters(
        rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        result = classify_spurious_sv_candidate(row)
        item = dict(row)
        item.setdefault("candidate_id", item.get("crop_id", f"candidate_{idx:06d}"))
        item["cluster_id"] = result.cluster_id
        item["cluster_name"] = result.cluster_name
        item["spurious_sv_score"] = result.spurious_sv_score
        item["use_as_hard_negative"] = str(result.use_as_hard_negative).lower()
        item["cluster_reason"] = result.reason
        out.append(item)
    return out
