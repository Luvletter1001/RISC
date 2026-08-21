#!/usr/bin/env python
"""P4 non-oracle Gaussian semantic-support ranker for dumped detections.

This is a score-only prediction transformer. It uses class-conditioned
log-area Gaussian priors to re-rank detections in logit space. It does not read
validation annotations, does not modify boxes, and does not use GWD/KLD/NWD or
any bbox-distance Gaussian route.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from M_Tools.analysis.apply_scale_prior_to_predictions import (  # noqa: E402
    DOTAV2_CLASS_NAMES,
    compute_box_areas,
    load_class_area_priors,
)


def _parse_class_names(class_names: str | Sequence[str] | None,
                       config: str | None) -> tuple[str, ...]:
    if class_names:
        if isinstance(class_names, str):
            return tuple(
                item.strip() for item in class_names.split(",")
                if item.strip())
        return tuple(str(item) for item in class_names)
    if config:
        from M_Tools.analysis.eval_predictions_annfiles_metric_json import (
            load_class_names,
        )
        return tuple(load_class_names(config))
    return tuple(DOTAV2_CLASS_NAMES)


def _as_mutable_pred_instances(pred_instances: Any) -> dict[str, Any]:
    if isinstance(pred_instances, dict):
        return dict(pred_instances)
    return {
        "bboxes": getattr(pred_instances, "bboxes"),
        "scores": getattr(pred_instances, "scores"),
        "labels": getattr(pred_instances, "labels"),
    }


def _scores_to_tensor(scores: Any) -> torch.Tensor:
    if isinstance(scores, torch.Tensor):
        return scores.detach().clone()
    return torch.as_tensor(scores, dtype=torch.float32).clone()


def _labels_to_tensor(labels: Any, device: torch.device) -> torch.Tensor:
    if isinstance(labels, torch.Tensor):
        return labels.detach().to(device=device, dtype=torch.long)
    return torch.as_tensor(labels, dtype=torch.long, device=device)


def _class_name(label_id: int, class_names: Sequence[str]) -> str:
    if 0 <= label_id < len(class_names):
        return str(class_names[label_id])
    return str(label_id)


def _prior_tensors(labels: torch.Tensor,
                   class_names: Sequence[str],
                   priors: dict[str, dict[str, float]],
                   device: torch.device) -> tuple[torch.Tensor, torch.Tensor,
                                                  torch.Tensor]:
    log_mean = torch.zeros_like(labels, dtype=torch.float32, device=device)
    log_std = torch.zeros_like(labels, dtype=torch.float32, device=device)
    has_prior = torch.zeros_like(labels, dtype=torch.bool, device=device)
    for label_id in labels.detach().cpu().unique().tolist():
        label_id = int(label_id)
        cls = _class_name(label_id, class_names)
        prior = priors.get(cls)
        if not prior:
            continue
        mask = labels == label_id
        log_mean[mask] = float(prior["log_area_mean"])
        log_std[mask] = max(float(prior["log_area_std"]), 1e-6)
        has_prior[mask] = True
    return log_mean, log_std, has_prior


def _class_counts(labels: torch.Tensor,
                  mask: torch.Tensor,
                  class_names: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for label_id in labels[mask].detach().cpu().tolist():
        counts[_class_name(int(label_id), class_names)] += 1
    return dict(sorted(counts.items()))


def _logit_update(scores: torch.Tensor,
                  delta: torch.Tensor,
                  eps: float = 1e-6) -> torch.Tensor:
    clipped = scores.float().clamp(min=eps, max=1.0 - eps)
    logits = torch.log(clipped / (1.0 - clipped))
    return torch.sigmoid(logits + delta)


def _apply_to_sample(sample: Any,
                     class_names: Sequence[str],
                     priors: dict[str, dict[str, float]],
                     alpha: float,
                     beta: float,
                     reward_z: float,
                     penalty_z: float,
                     min_score: float,
                     max_up_delta: float,
                     max_down_delta: float,
                     score_change_eps: float) -> tuple[dict[str, Any],
                                                       dict[str, Any]]:
    pred_instances = sample.get("pred_instances", {})
    if "bboxes" not in pred_instances or len(pred_instances["bboxes"]) == 0:
        return dict(sample), {
            "before": 0,
            "after": 0,
            "with_prior": 0,
            "valid": 0,
            "changed": 0,
            "boosted": 0,
            "penalized": 0,
            "delta_sum": 0.0,
            "abs_delta_sum": 0.0,
            "max_up": 0.0,
            "max_down": 0.0,
            "boosted_by_class": {},
            "penalized_by_class": {},
        }

    updated_sample = dict(sample)
    updated_pred = _as_mutable_pred_instances(pred_instances)
    original_scores = _scores_to_tensor(updated_pred["scores"])
    device = original_scores.device
    labels = _labels_to_tensor(updated_pred["labels"], device=device)
    areas = compute_box_areas(updated_pred["bboxes"]).to(device=device)
    log_mean, log_std, has_prior = _prior_tensors(
        labels, class_names, priors, device)
    valid = has_prior & (areas > 0) & (log_std > 0) & (
        original_scores.float() >= float(min_score))

    z = torch.zeros_like(original_scores, dtype=torch.float32, device=device)
    if valid.any():
        z[valid] = (
            torch.log(areas[valid]).float() - log_mean[valid]).abs() / log_std[valid]

    reward = torch.clamp(float(reward_z) - z, min=0.0)
    penalty = torch.clamp(z - float(penalty_z), min=0.0)
    delta = float(alpha) * reward - float(beta) * penalty
    delta = torch.clamp(
        delta, min=-float(max_down_delta), max=float(max_up_delta))
    delta = torch.where(valid, delta, torch.zeros_like(delta))

    updated_scores = original_scores.detach().clone()
    if valid.any():
        updated_scores[valid] = _logit_update(
            original_scores.float()[valid], delta[valid]).to(updated_scores.dtype)
    updated_pred["scores"] = updated_scores
    updated_sample["pred_instances"] = updated_pred

    changed = (updated_scores.float() - original_scores.float()).abs() > float(
        score_change_eps)
    boosted = changed & (delta > float(score_change_eps))
    penalized = changed & (delta < -float(score_change_eps))
    changed_delta = delta[changed]
    max_up = float(torch.clamp(delta, min=0.0).max().item()) if delta.numel() else 0.0
    max_down = float(torch.clamp(-delta, min=0.0).max().item()) if delta.numel() else 0.0
    return updated_sample, {
        "before": int(labels.numel()),
        "after": int(labels.numel()),
        "with_prior": int(has_prior.sum().item()),
        "valid": int(valid.sum().item()),
        "changed": int(changed.sum().item()),
        "boosted": int(boosted.sum().item()),
        "penalized": int(penalized.sum().item()),
        "delta_sum": float(changed_delta.sum().item()) if changed.any() else 0.0,
        "abs_delta_sum": (
            float(changed_delta.abs().sum().item()) if changed.any() else 0.0),
        "max_up": max_up,
        "max_down": max_down,
        "boosted_by_class": _class_counts(labels, boosted, class_names),
        "penalized_by_class": _class_counts(labels, penalized, class_names),
    }


def _merge_counts(dst: dict[str, int], src: dict[str, int]) -> None:
    for key, value in src.items():
        dst[key] += int(value)


def apply_p4_gsrl_ranker(input_pkl: str | Path,
                         output_pkl: str | Path,
                         class_area_priors_csv: str | Path,
                         class_names: str | Sequence[str] | None = None,
                         config: str | None = None,
                         alpha: float = 0.05,
                         beta: float = 0.10,
                         reward_z: float = 1.0,
                         penalty_z: float = 3.0,
                         min_score: float = 0.0,
                         max_up_delta: float = 0.25,
                         max_down_delta: float = 1.50,
                         score_change_eps: float = 1e-7) -> dict[str, Any]:
    """Apply non-oracle Gaussian semantic-support logit ranking."""
    input_pkl = Path(input_pkl)
    output_pkl = Path(output_pkl)
    class_names_tuple = _parse_class_names(class_names, config)
    priors = load_class_area_priors(class_area_priors_csv)
    with input_pkl.open("rb") as f:
        predictions = pickle.load(f)

    output = []
    boosted_by_class: dict[str, int] = defaultdict(int)
    penalized_by_class: dict[str, int] = defaultdict(int)
    summary = {
        "input_pkl": str(input_pkl),
        "output_pkl": str(output_pkl),
        "class_area_priors_csv": str(class_area_priors_csv),
        "config": config,
        "num_samples": len(predictions),
        "class_count": len(class_names_tuple),
        "alpha": float(alpha),
        "beta": float(beta),
        "reward_z": float(reward_z),
        "penalty_z": float(penalty_z),
        "min_score": float(min_score),
        "max_up_delta": float(max_up_delta),
        "max_down_delta": float(max_down_delta),
        "score_change_eps": float(score_change_eps),
        "detections_before": 0,
        "detections_after": 0,
        "detections_with_prior": 0,
        "detections_valid_for_update": 0,
        "scores_changed": 0,
        "scores_boosted": 0,
        "scores_penalized": 0,
        "mean_logit_delta_changed": 0.0,
        "mean_abs_logit_delta_changed": 0.0,
        "max_up_delta_observed": 0.0,
        "max_down_delta_observed": 0.0,
        "projection_type": "non_oracle_gaussian_semantic_support_logit_ranker",
        "uses_gt_at_inference": False,
        "forbidden_bbox_gaussian_route": False,
        "bbox_distance_gaussian_used": False,
        "score_update_space": "logit",
    }
    delta_sum = 0.0
    abs_delta_sum = 0.0

    for sample in predictions:
        updated_sample, stats = _apply_to_sample(
            sample=sample,
            class_names=class_names_tuple,
            priors=priors,
            alpha=alpha,
            beta=beta,
            reward_z=reward_z,
            penalty_z=penalty_z,
            min_score=min_score,
            max_up_delta=max_up_delta,
            max_down_delta=max_down_delta,
            score_change_eps=score_change_eps,
        )
        output.append(updated_sample)
        summary["detections_before"] += stats["before"]
        summary["detections_after"] += stats["after"]
        summary["detections_with_prior"] += stats["with_prior"]
        summary["detections_valid_for_update"] += stats["valid"]
        summary["scores_changed"] += stats["changed"]
        summary["scores_boosted"] += stats["boosted"]
        summary["scores_penalized"] += stats["penalized"]
        delta_sum += stats["delta_sum"]
        abs_delta_sum += stats["abs_delta_sum"]
        summary["max_up_delta_observed"] = max(
            summary["max_up_delta_observed"], stats["max_up"])
        summary["max_down_delta_observed"] = max(
            summary["max_down_delta_observed"], stats["max_down"])
        _merge_counts(boosted_by_class, stats["boosted_by_class"])
        _merge_counts(penalized_by_class, stats["penalized_by_class"])

    if summary["scores_changed"]:
        denom = float(summary["scores_changed"])
        summary["mean_logit_delta_changed"] = delta_sum / denom
        summary["mean_abs_logit_delta_changed"] = abs_delta_sum / denom
    summary["boosted_by_class"] = dict(sorted(boosted_by_class.items()))
    summary["penalized_by_class"] = dict(sorted(penalized_by_class.items()))

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply P4/GSRL non-oracle support ranker to predictions.pkl")
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--output-pkl", required=True)
    parser.add_argument("--class-area-priors-csv", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--class-names", default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.10)
    parser.add_argument("--reward-z", type=float, default=1.0)
    parser.add_argument("--penalty-z", type=float, default=3.0)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--max-up-delta", type=float, default=0.25)
    parser.add_argument("--max-down-delta", type=float, default=1.50)
    parser.add_argument("--score-change-eps", type=float, default=1e-7)
    parser.add_argument("--summary-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = apply_p4_gsrl_ranker(
        input_pkl=args.input_pkl,
        output_pkl=args.output_pkl,
        class_area_priors_csv=args.class_area_priors_csv,
        class_names=args.class_names,
        config=args.config,
        alpha=args.alpha,
        beta=args.beta,
        reward_z=args.reward_z,
        penalty_z=args.penalty_z,
        min_score=args.min_score,
        max_up_delta=args.max_up_delta,
        max_down_delta=args.max_down_delta,
        score_change_eps=args.score_change_eps,
    )
    summary_json = args.summary_json
    if summary_json is None:
        summary_json = str(Path(args.output_pkl).with_suffix(".summary.json"))
    Path(summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_json).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + os.linesep,
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
