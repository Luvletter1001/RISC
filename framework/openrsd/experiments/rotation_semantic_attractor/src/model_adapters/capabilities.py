from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..utils.status import ExperimentStatus


CLOSED_SET_QUERY_REASON = "closed-set dense/head detector does not use query logits"

TWO_STAGE_HINTS = ("roi_trans", "oriented_rcnn", "redet", "striprcnn")


def _architecture_type(model_key: str, cfg: Dict) -> str:
    key = model_key.lower()
    config = str(cfg.get("config", "")).lower()
    name = str(cfg.get("name", "")).lower()
    joined = " ".join([key, config, name])
    if any(hint in joined for hint in TWO_STAGE_HINTS):
        return "two_stage"
    if cfg.get("adapter") == "mmdet":
        return "dense_head"
    return "unknown"


def _asset_status(cfg: Dict) -> tuple[str, str]:
    config = cfg.get("config", "")
    checkpoint = cfg.get("checkpoint", "")
    if config and not Path(config).exists():
        return ExperimentStatus.NOT_AVAILABLE_ASSET, f"missing_config:{config}"
    if not checkpoint:
        return ExperimentStatus.NOT_AVAILABLE_ASSET, "missing_checkpoint"
    if checkpoint and not Path(checkpoint).exists():
        return ExperimentStatus.NOT_AVAILABLE_ASSET, f"missing_checkpoint:{checkpoint}"
    return ExperimentStatus.DONE_SMOKE, "asset available for smoke/full execution"


def build_capability_profile(model_key: str, cfg: Dict) -> Dict:
    adapter = cfg.get("adapter", "unknown")
    model_type = cfg.get("model_type", "unknown")
    if adapter == "mmdet" and model_type == "closed_set":
        model_family = "closed_set"
    elif model_type in {"open_vocab", "generic_ovd", "openrsd"}:
        model_family = model_type
    else:
        model_family = "unknown"

    architecture_type = _architecture_type(model_key, cfg)
    asset_status, asset_reason = _asset_status(cfg)

    supports_query_logits = bool(cfg.get("supports_query_logits", False))
    if model_family == "closed_set":
        query_status = ExperimentStatus.NOT_APPLICABLE
        query_reason = CLOSED_SET_QUERY_REASON
        supports_query_logits = False
        supports_text_prompts = False
        supports_text_embedding_intervention = False
        supports_visual_support_intervention = False
    else:
        supports_text_prompts = bool(cfg.get("supports_text_prompts", False))
        supports_text_embedding_intervention = bool(cfg.get("supports_text_embedding_intervention", False))
        supports_visual_support_intervention = bool(cfg.get("supports_visual_support_intervention", False))
        query_status = ExperimentStatus.DONE_SMOKE if supports_query_logits else ExperimentStatus.UNSUPPORTED_BY_CURRENT_CODE
        query_reason = "query logits hook available" if supports_query_logits else "query logits hook not implemented"

    return {
        "model_name": model_key,
        "display_name": cfg.get("name", model_key),
        "model_family": model_family,
        "architecture_type": architecture_type,
        "box_type": cfg.get("box_type", "unknown"),
        "asset_status": asset_status,
        "asset_status_reason": asset_reason,
        "supports_final_predictions": asset_status == ExperimentStatus.DONE_SMOKE,
        "supports_pre_nms_predictions": bool(cfg.get("supports_pre_nms", False)),
        "supports_dense_logits": bool(cfg.get("supports_dense_logits", False)),
        "supports_query_logits": supports_query_logits,
        "supports_text_prompts": supports_text_prompts,
        "supports_text_embedding_intervention": supports_text_embedding_intervention,
        "supports_visual_support_intervention": supports_visual_support_intervention,
        "supports_classifier_channel_intervention": bool(cfg.get("supports_classifier_channel_intervention", False)),
        "supports_weight_level_intervention": bool(cfg.get("supports_weight_level_intervention", False)),
        "supports_real_context_counterfactual": bool(cfg.get("supports_real_context_counterfactual", False)),
        "supports_dehub_checkpoint_comparison": bool(cfg.get("supports_dehub_checkpoint_comparison", False)),
        "query_logits_status": query_status,
        "query_logits_reason": query_reason,
    }

