"""Negative-aware prompt vocabulary helpers for auxiliary FOCUS analysis."""

from __future__ import annotations

from typing import Mapping


DEFAULT_POSITIVE_PROMPTS = (
    "small vehicle",
    "vehicle seen from above",
    "compact road vehicle",
    "vehicle in parking area",
)

DEFAULT_NEGATIVE_PROMPTS = (
    "not a court line",
    "not a roof edge",
    "not a runway marking",
    "not a ship deck",
    "not a storage tank",
    "not a roundabout",
    "not background texture",
)

DIRECTION_TEXT_PROMPT_ONLY_NEGATIVE_CONTROL = (
    "small vehicle facing north",
    "small vehicle facing east",
    "small vehicle facing south",
    "small vehicle facing west",
)


def auxiliary_margin_from_similarities(
        positive_similarities: Mapping[str, float],
        negative_similarities: Mapping[str, float]) -> float:
    """Compute auxiliary margin; it is not a support-logit replacement."""
    if not positive_similarities or not negative_similarities:
        return 0.0
    pos = max(float(v) for v in positive_similarities.values())
    neg = max(float(v) for v in negative_similarities.values())
    return round(pos - neg, 6)


def build_negative_prompt_rows() -> list[dict[str, str]]:
    rows = []
    for prompt in DEFAULT_POSITIVE_PROMPTS:
        rows.append({"prompt": prompt, "polarity": "positive_auxiliary"})
    for prompt in DEFAULT_NEGATIVE_PROMPTS:
        rows.append({"prompt": prompt, "polarity": "negative_auxiliary"})
    for prompt in DIRECTION_TEXT_PROMPT_ONLY_NEGATIVE_CONTROL:
        rows.append({"prompt": prompt, "polarity": "direction_text_negative_control"})
    return rows
