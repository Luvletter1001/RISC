"""Head-consensus audit helpers for FOCUS-OVD."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeadConsensusResult:
    head_agreement_score: float
    head_disagreement_type: str
    consensus_label: str


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def score_head_consensus(
        *,
        bbox_head_sv_score: float,
        alignment_head_sv_score: float | None = None,
        fusion_head_sv_score: float | None = None,
        support_margin: float | None = None,
        human_false_sv: bool = False) -> HeadConsensusResult:
    scores = [_clip(bbox_head_sv_score)]
    if alignment_head_sv_score is not None:
        scores.append(_clip(alignment_head_sv_score))
    if fusion_head_sv_score is not None:
        scores.append(_clip(fusion_head_sv_score))
    agreement = 1.0 - (max(scores) - min(scores) if scores else 1.0)
    margin = 1.0 if support_margin is None else _clip(support_margin)

    bbox_high = _clip(bbox_head_sv_score) >= 0.75
    align_low = alignment_head_sv_score is not None and _clip(
        alignment_head_sv_score) < 0.40
    fusion_low = fusion_head_sv_score is not None and _clip(
        fusion_head_sv_score) < 0.40

    if bbox_high and align_low and fusion_low:
        return HeadConsensusResult(
            agreement, "bbox_high_alignment_fusion_low", "shortcut_candidate")
    if margin < 0.20:
        return HeadConsensusResult(
            agreement, "low_support_margin", "ambiguous")
    if agreement >= 0.80 and human_false_sv:
        return HeadConsensusResult(
            agreement, "heads_agree_but_human_false", "hard_negative_candidate")
    if agreement >= 0.70 and margin >= 0.40:
        return HeadConsensusResult(
            agreement, "heads_agree", "keep_candidate")
    return HeadConsensusResult(
        agreement, "heads_disagree", "ambiguous")
