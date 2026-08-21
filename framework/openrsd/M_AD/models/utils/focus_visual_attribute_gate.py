"""Visual attribute gate scoring for FOCUS-OVD attribution.

This module produces an audit score only. Low confidence is not a deletion
decision unless a later safety-gated experiment explicitly enables that policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any, Mapping


@dataclass(frozen=True)
class AttributeGateScore:
    attribute_sv_confidence: float
    positive_attributes: tuple[str, ...]
    negative_attributes: tuple[str, ...]
    raw_score: float


def _float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _text(row: Mapping[str, Any], key: str) -> str:
    return str(row.get(key, "") or "").strip().lower()


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def score_visual_attributes(row: Mapping[str, Any]) -> AttributeGateScore:
    width = _float(row, "box_w", _float(row, "w", 0.0))
    height = _float(row, "box_h", _float(row, "h", 0.0))
    area = width * height
    aspect = max(width, height) / max(min(width, height), 1e-6)
    orientation_conf = _float(row, "orientation_confidence", 0.0)
    orbit_stability = _float(row, "orbit_stability", 0.0)
    sv_sim = _float(row, "support_similarity_sv", 0.0)
    lv_sim = _float(row, "support_similarity_lv", 0.0)
    category = _text(row, "audit_category")
    context = " ".join(
        _text(row, key) for key in ("context", "scene_context", "notes"))

    positives: list[str] = []
    negatives: list[str] = []

    if 4.0 <= width <= 80.0 and 4.0 <= height <= 80.0 and aspect <= 4.0:
        positives.append("compact_object")
        positives.append("vehicle_like_extent")
    if sv_sim > lv_sim:
        positives.append("vehicle_like_texture")
    if "road" in context or "parking" in context:
        positives.append("road_or_parking_context")
    if orbit_stability >= 0.5:
        positives.append("stable_across_rotation")
    if orientation_conf >= 0.5:
        positives.append("high_orientation_confidence_if_directional")

    if any(token in context for token in ("court", "line")):
        negatives.append("court_line_like")
    if "roof" in context:
        negatives.append("roof_edge_like")
    if "ship" in context or "deck" in context:
        negatives.append("ship_deck_like")
    if "runway" in context:
        negatives.append("runway_texture_like")
    if "tank" in context:
        negatives.append("storage_tank_like")
    if area > 12000.0 or width > 160.0 or height > 160.0:
        negatives.append("large_region_box")
    if category == "padding_artifact" or "padding" in context or "border" in context:
        negatives.append("padding_or_border")
    if orientation_conf < 0.15 and sv_sim <= lv_sim:
        negatives.append("low_confidence_flat_texture")

    raw = 0.0
    raw += 0.45 * (sv_sim - lv_sim)
    raw += 0.25 * orientation_conf
    raw += 0.25 * orbit_stability
    raw += 0.18 * len(positives)
    raw -= 0.22 * len(negatives)
    confidence = _sigmoid(raw)
    return AttributeGateScore(
        attribute_sv_confidence=max(0.0, min(1.0, confidence)),
        positive_attributes=tuple(positives),
        negative_attributes=tuple(negatives),
        raw_score=raw,
    )


def annotate_attribute_scores(
        rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        score = score_visual_attributes(row)
        item = dict(row)
        item.setdefault("candidate_id", item.get("crop_id", f"candidate_{idx:06d}"))
        item["attribute_sv_confidence"] = score.attribute_sv_confidence
        item["positive_attributes"] = ";".join(score.positive_attributes)
        item["negative_attributes"] = ";".join(score.negative_attributes)
        item["attribute_raw_score"] = score.raw_score
        out.append(item)
    return out
