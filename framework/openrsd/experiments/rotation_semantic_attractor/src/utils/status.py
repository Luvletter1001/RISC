from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, Mapping


class ExperimentStatus:
    DONE_FULL = "DONE_FULL"
    DONE_SMOKE = "DONE_SMOKE"
    SMOKE_PROXY = "SMOKE_PROXY"
    SCHEMA_ONLY = "SCHEMA_ONLY"
    NOT_SELECTED_IN_THIS_SMOKE = "NOT_SELECTED_IN_THIS_SMOKE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNSUPPORTED_BY_CURRENT_CODE = "UNSUPPORTED_BY_CURRENT_CODE"
    NOT_AVAILABLE_ASSET = "NOT_AVAILABLE_ASSET"
    NOT_RUN = "NOT_RUN"
    FAILED = "FAILED"


CLAIM_OPEN_VOCAB_EMBEDDING = "open_vocab_embedding_mechanism"
CLAIM_CLOSED_SET_LOGIT = "closed_set_logit_or_false_hub"
CLAIM_CROSS_MODEL = "cross_model_phenomenon"
CLAIM_DEHUB_REPAIR = "dehub_repair"
CLAIM_ENGINEERING_SMOKE = "engineering_smoke"
CLAIM_SCHEMA_ONLY = "schema_only"
CLAIM_NONE = "none"


ALL_STATUSES = {
    ExperimentStatus.DONE_FULL,
    ExperimentStatus.DONE_SMOKE,
    ExperimentStatus.SMOKE_PROXY,
    ExperimentStatus.SCHEMA_ONLY,
    ExperimentStatus.NOT_SELECTED_IN_THIS_SMOKE,
    ExperimentStatus.NOT_APPLICABLE,
    ExperimentStatus.UNSUPPORTED_BY_CURRENT_CODE,
    ExperimentStatus.NOT_AVAILABLE_ASSET,
    ExperimentStatus.NOT_RUN,
    ExperimentStatus.FAILED,
}


def is_failure(status: str) -> bool:
    return status == ExperimentStatus.FAILED


def is_proxy_status(status: str) -> bool:
    return status in {ExperimentStatus.SMOKE_PROXY, ExperimentStatus.SCHEMA_ONLY}


def status_metadata(
    status: str,
    status_reason: str,
    *,
    claim_level: str = CLAIM_NONE,
    is_scientific_result: bool | None = None,
    include_in_main_table: bool | None = None,
) -> Dict:
    if status not in ALL_STATUSES:
        raise ValueError(f"unknown experiment status: {status}")
    is_proxy = is_proxy_status(status)
    if is_scientific_result is None:
        is_scientific_result = status == ExperimentStatus.DONE_FULL
    if include_in_main_table is None:
        include_in_main_table = status == ExperimentStatus.DONE_FULL
    if is_proxy:
        is_scientific_result = False
        include_in_main_table = False
    if status in {ExperimentStatus.NOT_APPLICABLE, ExperimentStatus.NOT_SELECTED_IN_THIS_SMOKE}:
        is_scientific_result = False
        include_in_main_table = False
    return {
        "status": status,
        "status_reason": status_reason,
        "is_proxy": is_proxy,
        "is_scientific_result": bool(is_scientific_result),
        "include_in_main_table": bool(include_in_main_table),
        "claim_level": claim_level,
    }


def count_statuses(rows: Iterable[Mapping]) -> Dict[str, int]:
    counter = Counter(str(row.get("status", "")) for row in rows)
    return {
        "failure_count": counter[ExperimentStatus.FAILED],
        "not_applicable_count": counter[ExperimentStatus.NOT_APPLICABLE],
        "not_selected_count": counter[ExperimentStatus.NOT_SELECTED_IN_THIS_SMOKE],
        "unsupported_count": counter[ExperimentStatus.UNSUPPORTED_BY_CURRENT_CODE],
        "proxy_count": counter[ExperimentStatus.SMOKE_PROXY] + counter[ExperimentStatus.SCHEMA_ONLY],
        "schema_only_count": counter[ExperimentStatus.SCHEMA_ONLY],
        "not_available_asset_count": counter[ExperimentStatus.NOT_AVAILABLE_ASSET],
        "not_run_count": counter[ExperimentStatus.NOT_RUN],
        "done_full_count": counter[ExperimentStatus.DONE_FULL],
        "done_smoke_count": counter[ExperimentStatus.DONE_SMOKE],
    }


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "done", "available"}


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _listish(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    text = str(value).strip()
    if not text:
        return []
    for sep in (";", "|", ","):
        if sep in text:
            return [part.strip() for part in text.split(sep) if part.strip()]
    return [text]


def _angle_count(row: Mapping) -> int:
    for key in ("angle_set", "angles", "used_angles"):
        values = _listish(row.get(key))
        if values:
            return len(values)
    return 0


def _is_full_split(row: Mapping) -> bool:
    split = str(row.get("split_name") or row.get("split") or "").lower()
    return split == "s2_final_test" or "full" in split or _truthy(row.get("is_full_split"))


def _hard_done_full_reject(row: Mapping) -> bool:
    status = str(row.get("status", "") or "")
    if status and status != ExperimentStatus.DONE_FULL:
        return True
    if status in {
        ExperimentStatus.DONE_SMOKE,
        ExperimentStatus.SMOKE_PROXY,
        ExperimentStatus.SCHEMA_ONLY,
        ExperimentStatus.NOT_SELECTED_IN_THIS_SMOKE,
        ExperimentStatus.NOT_APPLICABLE,
        ExperimentStatus.UNSUPPORTED_BY_CURRENT_CODE,
        ExperimentStatus.NOT_AVAILABLE_ASSET,
        ExperimentStatus.NOT_RUN,
        ExperimentStatus.FAILED,
    }:
        return True
    return any(
        _truthy(row.get(key))
        for key in ("is_proxy", "proxy_only", "is_schema_only", "is_smoke_limit", "smoke_limit")
    )


def _stage_is_real(status: Any) -> bool:
    return str(status) in {ExperimentStatus.DONE_FULL, ExperimentStatus.DONE_SMOKE} or _truthy(status)


def _stage_is_allowed_missing(status: Any) -> bool:
    return str(status) in {
        "",
        ExperimentStatus.NOT_APPLICABLE,
        ExperimentStatus.UNSUPPORTED_BY_CURRENT_CODE,
    }


def can_be_done_full(experiment_family: str, row_or_manifest: Mapping) -> bool:
    """Return whether a row has the minimum evidence required for DONE_FULL.

    This is intentionally conservative: smoke/proxy/schema rows are never
    eligible, and each experiment family must show real inference plus the
    family-specific evidence needed for scientific tables.
    """

    row = row_or_manifest
    family = str(experiment_family or row.get("experiment_family", "") or "").strip().lower()
    if _hard_done_full_reject(row):
        return False

    if family in {"generic", ""}:
        return str(row.get("status", "")) == ExperimentStatus.DONE_FULL

    if family == "false_hub_benchmark":
        has_angle_evidence = _angle_count(row) >= 12 or _truthy(row.get("has_manifest_angle_set"))
        return all(
            [
                _is_full_split(row),
                has_angle_evidence,
                _truthy(row.get("actual_inference_run")),
                _truthy(row.get("has_final_predictions")),
                _truthy(row.get("has_gt_matching")),
                _truthy(row.get("has_false_sv_taxonomy")),
            ]
        )

    if family == "stage_decomposition":
        stage_statuses = [
            row.get("dense_logits_status", ""),
            row.get("query_logits_status", ""),
            row.get("pre_nms_status", ""),
            row.get("post_nms_status", ""),
            row.get("final_status", ""),
        ]
        real_stage_count = sum(1 for status in stage_statuses if _stage_is_real(status))
        missing_ok = all(
            _stage_is_real(status) or _stage_is_allowed_missing(status)
            for status in stage_statuses
        )
        return _truthy(row.get("actual_inference_run")) and real_stage_count >= 2 and missing_ok

    if family == "open_vocab_embedding_intervention":
        original_checksum = str(row.get("original_embedding_checksum", "") or "")
        modified_checksum = str(row.get("modified_embedding_checksum", "") or "")
        return all(
            [
                _truthy(row.get("has_open_vocab_config")),
                _truthy(row.get("has_checkpoint")),
                _truthy(row.get("has_prompt_or_class_embedding_path")),
                _truthy(row.get("is_embedding_level")) or _truthy(row.get("is_visual_support_level")),
                _truthy(row.get("actual_embedding_modified")),
                _nonempty(original_checksum),
                _nonempty(modified_checksum),
                original_checksum != modified_checksum,
                _nonempty(row.get("modified_tensor_name")),
                _truthy(row.get("reran_inference")),
                _truthy(row.get("has_paired_comparison")),
                not _truthy(row.get("is_prompt_only")),
            ]
        )

    if family == "closedset_classifier_channel_intervention":
        changed = _truthy(row.get("actual_logit_modified")) or _truthy(row.get("actual_weight_modified"))
        return all(
            [
                changed,
                _truthy(row.get("reran_inference")),
                _truthy(row.get("has_paired_comparison")),
                _nonempty(row.get("modified_layer_name")),
                _nonempty(row.get("modified_tensor_name")),
                _nonempty(row.get("modified_class_index")),
            ]
        )

    if family == "context_counterfactual":
        conditions = set(str(item) for item in _listish(row.get("completed_conditions")))
        has_required_conditions = {"object_only", "context_only"}.issubset(conditions) or int(
            row.get("completed_condition_count") or 0
        ) >= 2
        return all(
            [
                _nonempty(row.get("modified_image_path")),
                _truthy(row.get("reran_inference")),
                _truthy(row.get("counterfactual_is_real_image_level")),
                _truthy(row.get("has_paired_comparison")),
                has_required_conditions,
            ]
        )

    if family == "dehub_safety":
        return all(
            [
                _nonempty(row.get("baseline_checkpoint")),
                _nonempty(row.get("repair_checkpoint")),
                _truthy(row.get("same_split")),
                _truthy(row.get("same_angles")) or _nonempty(row.get("angle_set")),
                _truthy(row.get("same_threshold")) or _truthy(row.get("same_evaluator")),
                _truthy(row.get("actual_inference_run")),
                _truthy(row.get("has_false_hub_metrics")),
                _truthy(row.get("has_true_sv_preservation")),
                _truthy(row.get("has_det_per_img")),
                _truthy(row.get("has_class_distribution_js_kl")) or (
                    _nonempty(row.get("class_js")) and _nonempty(row.get("class_kl"))
                ),
                _truthy(row.get("has_lowrisk_inflation")),
                _truthy(row.get("has_ap50_or_proxy")),
                not _truthy(row.get("is_schema_only")),
            ]
        )

    return False
