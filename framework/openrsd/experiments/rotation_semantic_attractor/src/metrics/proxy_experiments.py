from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence

from ..constants import SMALL_VEHICLE
from ..utils.status import CLAIM_NONE, ExperimentStatus, status_metadata


def _fr_sv(predictions: Sequence[Dict]) -> float:
    if not predictions:
        return 0.0
    return sum(1 for pred in predictions if pred.get("class_name") == SMALL_VEHICLE) / len(predictions)


def _det_per_img(predictions: Sequence[Dict]) -> int:
    return len(predictions)


def _drop_small_vehicle(predictions: Sequence[Dict]) -> List[Dict]:
    return [pred for pred in predictions if pred.get("class_name") != SMALL_VEHICLE]


def _score_scaled(predictions: Sequence[Dict], scale: float, threshold: float = 0.05) -> List[Dict]:
    rows = []
    for pred in predictions:
        new_pred = dict(pred)
        if new_pred.get("class_name") == SMALL_VEHICLE:
            new_pred["score"] = float(new_pred.get("score", 0.0)) * scale
        if float(new_pred.get("score", 0.0)) >= threshold:
            rows.append(new_pred)
    return rows


def _swap_small_vehicle(predictions: Sequence[Dict], target_class: str) -> List[Dict]:
    rows = []
    for pred in predictions:
        new_pred = dict(pred)
        if new_pred.get("class_name") == SMALL_VEHICLE:
            new_pred["class_name"] = target_class
        rows.append(new_pred)
    return rows


def closedset_intervention_rows(
    model_name: str,
    tile_id: str,
    angle: int,
    final_predictions: Sequence[Dict],
    pre_nms_predictions: Sequence[Dict],
) -> List[Dict]:
    baseline_pre = _fr_sv(pre_nms_predictions)
    baseline_post = _fr_sv(final_predictions)
    interventions = {
        "zero_small_vehicle_classifier_channel": _drop_small_vehicle(final_predictions),
        "subtract_small_vehicle_class_bias": _score_scaled(final_predictions, scale=0.5),
        "swap_small_vehicle_with_ship_channel": _swap_small_vehicle(final_predictions, "ship"),
        "swap_small_vehicle_with_tennis_court_channel": _swap_small_vehicle(final_predictions, "tennis-court"),
        "class_wise_temperature_scaling": _score_scaled(final_predictions, scale=0.75),
        "altered_nms_threshold": list(final_predictions),
    }
    rows = []
    for intervention, predictions in interventions.items():
        post = _fr_sv(predictions)
        meta = status_metadata(
            ExperimentStatus.SMOKE_PROXY,
            "schema proxy only; no weight-level or logit-level classifier-channel intervention was performed",
            claim_level=CLAIM_NONE,
        )
        rows.append(
            {
                "model_name": model_name,
                "tile_id": tile_id,
                "angle": int(angle),
                "intervention_type": intervention,
                "actual_weight_or_logit_modified": False,
                "modified_layer_name": "",
                "modified_class_index": 4,
                "target_class_name": SMALL_VEHICLE,
                "control_class_name": "ship" if "ship" in intervention else "tennis-court" if "tennis" in intervention else "",
                "proxy_only": True,
                "final_fr_sv": post,
                "final_fsv": max(0.0, post - baseline_post),
                "pre_nms_fr_sv": baseline_pre,
                "post_nms_fr_sv": baseline_post,
                "nms_amp_sv": baseline_post - baseline_pre if pre_nms_predictions else "",
                "det_per_img": _det_per_img(predictions),
                **meta,
                "unsupported_reason": meta["status_reason"],
            }
        )
    return rows


def context_proxy_rows(model_name: str, tile_id: str, angle: int, false_hub_row: Dict) -> List[Dict]:
    final_fr_sv = float(false_hub_row.get("fr_sv") or 0.0)
    final_fsv = float(false_hub_row.get("false_sv_ratio") or 0.0)
    bg_fsv = float(false_hub_row.get("bg_fsv_ratio") or 0.0)
    object_flip = float(false_hub_row.get("object_flip_sv") or 0.0)
    verdict = "CONTEXT_SENSITIVE_PROXY" if bg_fsv > 0.5 else "INCONCLUSIVE"
    meta = status_metadata(
        ExperimentStatus.SMOKE_PROXY,
        "proxy accounting only; no modified image was generated and no counterfactual inference rerun was performed",
        claim_level=CLAIM_NONE,
    )
    return [
        {
            "model_name": model_name,
            "tile_id": tile_id,
            "angle": int(angle),
            "condition": condition,
            "target_gt_class": SMALL_VEHICLE,
            "counterfactual_is_real_image_level": False,
            "modified_image_path": "",
            "reran_inference": False,
            "proxy_only": True,
            "mask_type": "",
            "final_fr_sv": final_fr_sv,
            "final_fsv": final_fsv,
            "bg_fsv_ratio": bg_fsv,
            "object_flip_sv": object_flip,
            "verdict": verdict,
            **meta,
            "notes": meta["status_reason"],
        }
        for condition in ["object_only_proxy", "context_only_proxy", "background_bank_proxy", "context_swap_proxy"]
    ]


def _float_or_blank(value):
    if value in {"", None}:
        return ""
    return float(value)


def dehub_safety_rows(method: str, false_hub_summary: Dict, stage_summary: Dict | None = None) -> List[Dict]:
    stage_summary = stage_summary or {}
    meta = status_metadata(
        ExperimentStatus.SCHEMA_ONLY,
        "requires both baseline and DeHub checkpoint on the same split; current row is schema/smoke accounting only",
        claim_level=CLAIM_NONE,
    )
    return [
        {
            "method": method,
            "baseline_checkpoint": "",
            "repair_checkpoint": "",
            "same_split": False,
            "actual_inference_run": False,
            "has_true_sv_preservation": bool(false_hub_summary.get("mean_true_sv_recall")),
            "has_class_drift": False,
            "has_lowrisk_inflation": False,
            "is_schema_only": True,
            "include_in_repair_table": False,
            "mAP50": "",
            "SV_AP50": "",
            "Dense_or_QAR_SV": _float_or_blank(stage_summary.get("mean_post_nms_fr_sv")),
            "Final_FSV": _float_or_blank(false_hub_summary.get("mean_final_fsv")),
            "ObjectFlip_SV": "",
            "BG_FSV": _float_or_blank(false_hub_summary.get("mean_bg_fsv")),
            "True_SV_Recall": _float_or_blank(false_hub_summary.get("mean_true_sv_recall")),
            "Class_JS": "",
            "LowRiskInflation": "",
            "Verdict": ExperimentStatus.SCHEMA_ONLY,
            **meta,
            "unsupported_reason": meta["status_reason"],
        }
    ]


def class_distribution(predictions: Iterable[Dict]) -> Dict[str, int]:
    return dict(Counter(pred.get("class_name", "") for pred in predictions if pred.get("class_name")))
