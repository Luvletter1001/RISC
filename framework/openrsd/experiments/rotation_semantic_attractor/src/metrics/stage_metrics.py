from __future__ import annotations

from typing import Dict

from ..constants import SMALL_VEHICLE
from ..model_adapters.capabilities import CLOSED_SET_QUERY_REASON
from ..utils.status import ExperimentStatus


def _looks_like_closed_set_mmrotate(model_name: str, raw_output: Dict) -> bool:
    if raw_output.get("model_family") == "closed_set":
        return True
    if raw_output.get("adapter") == "mmdet":
        return True
    closed_set_prefixes = (
        "rotated_",
        "h2rbox",
        "redet",
        "oriented_",
        "r3det",
        "roi_trans",
        "s2anet",
        "striprcnn",
    )
    return model_name.startswith(closed_set_prefixes)


def compute_stage_row(model_name: str, tile_id: str, angle: int, raw_output: Dict) -> Dict:
    final_predictions = raw_output.get("final_predictions", [])
    pre_nms = raw_output.get("pre_nms_predictions", [])
    final_total = len(final_predictions)
    pre_total = len(pre_nms)
    final_sv = sum(1 for p in final_predictions if p.get("class_name") == SMALL_VEHICLE)
    pre_sv = sum(1 for p in pre_nms if p.get("class_name") == SMALL_VEHICLE)

    metadata = raw_output.get("metadata", {})
    capability = metadata.get("capability_profile", {})
    model_family = capability.get("model_family", "unknown")
    architecture_type = capability.get("architecture_type", "unknown")
    if model_family == "unknown" and _looks_like_closed_set_mmrotate(model_name, raw_output):
        model_family = "closed_set"
        architecture_type = "dense_head" if architecture_type == "unknown" else architecture_type

    pre_fr = pre_sv / pre_total if pre_total else None
    post_fr = final_sv / final_total if final_total else 0.0
    dense_logits = raw_output.get("dense_logits") or {}
    dense_or_query_sv = dense_logits.get("small_vehicle_topk_mean_score", "")
    if dense_or_query_sv == "":
        query_logits = raw_output.get("query_logits") or {}
        dense_or_query_sv = query_logits.get("small_vehicle_topk_mean_score", "")
    if dense_or_query_sv != "":
        dense_status = ExperimentStatus.DONE_SMOKE
        dense_reason = "dense small-vehicle score summary available"
    elif capability.get("supports_dense_logits"):
        dense_status = ExperimentStatus.UNSUPPORTED_BY_CURRENT_CODE
        dense_reason = "dense logits hook configured but no dense logits were emitted"
    elif architecture_type == "query_based":
        dense_status = ExperimentStatus.NOT_APPLICABLE
        dense_reason = "query-based detector does not use dense-head logits"
    else:
        dense_status = ExperimentStatus.NOT_APPLICABLE
        dense_reason = "dense logits are not applicable for this model configuration"

    if model_family == "closed_set":
        query_status = ExperimentStatus.NOT_APPLICABLE
        query_reason = CLOSED_SET_QUERY_REASON
    elif raw_output.get("query_logits"):
        query_status = ExperimentStatus.DONE_SMOKE
        query_reason = "query logits available"
    elif capability.get("supports_query_logits"):
        query_status = ExperimentStatus.UNSUPPORTED_BY_CURRENT_CODE
        query_reason = "query logits hook configured but no query logits were emitted"
    else:
        query_status = ExperimentStatus.UNSUPPORTED_BY_CURRENT_CODE
        query_reason = "query logits hook not implemented"

    if pre_fr is not None:
        pre_nms_status = ExperimentStatus.DONE_SMOKE
        pre_nms_reason = "pre-NMS predictions available"
    elif capability.get("supports_pre_nms_predictions"):
        pre_nms_status = ExperimentStatus.UNSUPPORTED_BY_CURRENT_CODE
        pre_nms_reason = "pre-NMS hook configured but no pre-NMS predictions were emitted"
    else:
        pre_nms_status = ExperimentStatus.NOT_APPLICABLE
        pre_nms_reason = "pre-NMS predictions are not available for this model configuration"

    post_nms_status = ExperimentStatus.DONE_SMOKE
    post_nms_reason = "final post-NMS predictions available"
    stage_metric_used = []
    if dense_or_query_sv != "":
        stage_metric_used.append("Dense_FR_SV" if architecture_type != "query_based" else "QAR_SV")
    if pre_fr is not None:
        stage_metric_used.append("PreNMS_FR_SV")
    stage_metric_used.append("PostNMS_FR_SV")
    if pre_fr is not None:
        stage_metric_used.append("NMS_Amp_SV")

    status = ExperimentStatus.DONE_SMOKE
    unsupported_reasons = []
    for stage_status, stage_name, stage_reason in [
        (dense_status, "dense", dense_reason),
        (pre_nms_status, "pre_nms", pre_nms_reason),
        (query_status, "query", query_reason),
    ]:
        if stage_status == ExperimentStatus.UNSUPPORTED_BY_CURRENT_CODE:
            unsupported_reasons.append(f"{stage_name}:{stage_reason}")
    if unsupported_reasons:
        status = ExperimentStatus.UNSUPPORTED_BY_CURRENT_CODE
    return {
        "model_name": model_name,
        "tile_id": tile_id,
        "angle": int(angle),
        "model_family": model_family,
        "architecture_type": architecture_type,
        "dense_or_query_sv": dense_or_query_sv,
        "dense_logits_status": dense_status,
        "dense_logits_reason": dense_reason,
        "query_logits_status": query_status,
        "query_logits_reason": query_reason,
        "pre_nms_status": pre_nms_status,
        "pre_nms_reason": pre_nms_reason,
        "post_nms_status": post_nms_status,
        "post_nms_reason": post_nms_reason,
        "stage_metric_used": ",".join(stage_metric_used),
        "pre_nms_fr_sv": "" if pre_fr is None else pre_fr,
        "post_nms_fr_sv": post_fr,
        "nms_amp_sv": "" if pre_fr is None else post_fr - pre_fr,
        "unsupported_reason": ";".join(unsupported_reasons),
        "status": status,
    }
