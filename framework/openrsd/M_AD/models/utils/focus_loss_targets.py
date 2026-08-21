"""Utilities for FOCUS detector-loss target validation and configuration."""

from __future__ import annotations

from typing import Any, Mapping


FOCUS_LOSS_DEFAULTS: dict[str, Any] = {
    "enable": False,
    "target_mapping_mode": "none",
    "support_distill_weight": 0.0,
    "anti_attractor_weight": 0.0,
    "preserve_weight": 0.0,
    "migration_weight": 0.0,
    "anti_margin": 0.10,
    "preserve_margin": 0.10,
    "max_focus_points_per_image": 256,
    "require_target_mapping": True,
    "log_focus_debug": True,
}

SPATIAL_REGION_FIELDS = {
    "focus_target_id",
    "target_polygon",
    "target_center_x",
    "target_center_y",
    "target_role",
    "valid_for_loss",
}

EXACT_PROVENANCE_FIELDS = {
    "prediction_id",
    "feature_level",
    "grid_x",
    "grid_y",
    "target_role",
    "valid_for_exact_loss",
}


def normalize_focus_losses_config(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    out = dict(FOCUS_LOSS_DEFAULTS)
    if cfg:
        out.update(dict(cfg))
    out["enable"] = bool(out.get("enable", False))
    for key in (
            "support_distill_weight",
            "anti_attractor_weight",
            "preserve_weight",
            "migration_weight",
            "anti_margin",
            "preserve_margin"):
        out[key] = float(out.get(key, 0.0))
    out["max_focus_points_per_image"] = int(
        out.get("max_focus_points_per_image", 256))
    out["target_mapping_mode"] = str(out.get("target_mapping_mode", "none"))
    return out


def validate_focus_loss_targets_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    if not cfg.get("enable", False):
        return {"ok": True, "status": "DISABLED", "reason": "focus losses disabled"}
    needs_targets = (
        float(cfg.get("anti_attractor_weight", 0.0)) > 0.0
        or float(cfg.get("preserve_weight", 0.0)) > 0.0)
    if needs_targets and cfg.get("target_mapping_mode") == "none":
        return {
            "ok": False,
            "status": "BLOCKED_TARGET_MAPPING_REQUIRED",
            "reason": "anti/preserve loss needs spatial_region or pre_nms_provenance targets",
        }
    return {"ok": True, "status": "READY", "reason": "focus loss config valid"}


def has_spatial_region_fields(row: Mapping[str, Any]) -> bool:
    return SPATIAL_REGION_FIELDS.issubset(set(row.keys()))


def has_exact_provenance_fields(row: Mapping[str, Any]) -> bool:
    return EXACT_PROVENANCE_FIELDS.issubset(set(row.keys()))


def zero_like_from_scores(cls_scores: list[Any]) -> Any:
    if not cls_scores:
        return 0.0
    first = cls_scores[0]
    if hasattr(first, "sum"):
        return first.sum() * 0.0
    return 0.0
