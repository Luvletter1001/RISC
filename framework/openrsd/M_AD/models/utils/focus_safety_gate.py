"""Safety gate for FOCUS-OVD attribution decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SafetyGateResult:
    status: str
    failures: tuple[str, ...]
    warnings: tuple[str, ...]
    derived_metrics: dict[str, float]


def _number(values: Mapping[str, Any], key: str) -> float | None:
    value = values.get(key)
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0.0:
        return None
    return num / den


def evaluate_focus_safety(
        metrics: Mapping[str, Any],
        baseline: Mapping[str, Any] | None = None,
        *,
        ap_drop_limit: float = 0.01,
        sv_ap_drop_limit: float = 0.01,
        true_sv_recall_retention_min: float = 0.90,
        corrected_fsv_reduction_min: float = 0.20,
        det_explosion_ratio: float = 1.50,
        support_cos_collapse_max: float = 0.95) -> SafetyGateResult:
    baseline = baseline or {}
    failures: list[str] = []
    warnings: list[str] = []
    derived: dict[str, float] = {}

    map50 = _number(metrics, "mAP50")
    base_map50 = _number(baseline, "mAP50")
    if map50 is not None and base_map50 is not None:
        derived["mAP50_drop"] = base_map50 - map50
        if derived["mAP50_drop"] > ap_drop_limit:
            failures.append("mAP50_drop")

    sv_ap = _number(metrics, "SV_AP50")
    base_sv_ap = _number(baseline, "SV_AP50")
    if sv_ap is not None and base_sv_ap is not None:
        derived["SV_AP50_drop"] = base_sv_ap - sv_ap
        if derived["SV_AP50_drop"] > sv_ap_drop_limit:
            failures.append("SV_AP50_drop")

    recall_retention = _ratio(
        _number(metrics, "true_SV_recall"),
        _number(baseline, "true_SV_recall"))
    if recall_retention is not None:
        derived["true_SV_recall_retention"] = recall_retention
        if recall_retention < true_sv_recall_retention_min:
            failures.append("true_SV_recall_retention")

    base_fsv = _number(baseline, "corrected_FSV")
    fsv = _number(metrics, "corrected_FSV")
    if base_fsv is not None and fsv is not None and base_fsv > 0:
        derived["corrected_FSV_reduction"] = (base_fsv - fsv) / base_fsv
        if derived["corrected_FSV_reduction"] < corrected_fsv_reduction_min:
            failures.append("corrected_FSV_reduction")

    migration = _number(metrics, "migration_mass_ratio")
    base_migration = _number(baseline, "migration_mass_ratio")
    if migration is not None and base_migration is not None:
        derived["migration_mass_delta"] = migration - base_migration
        if migration > base_migration + 1e-8:
            failures.append("migration_mass_ratio")

    det_ratio = _ratio(_number(metrics, "det/img"), _number(baseline, "det/img"))
    if det_ratio is not None:
        derived["det_per_img_ratio"] = det_ratio
        if det_ratio > det_explosion_ratio:
            failures.append("det/img_explosion")

    degenerate = _number(metrics, "degenerate_large_sv_ratio")
    base_degenerate = _number(baseline, "degenerate_large_sv_ratio")
    if degenerate is not None and base_degenerate is not None:
        derived["degenerate_large_sv_delta"] = degenerate - base_degenerate
        if degenerate > base_degenerate + 1e-8:
            failures.append("degenerate_large_sv_ratio")

    support_cos = _number(metrics, "support_inter_class_cos_max")
    if support_cos is not None:
        derived["support_inter_class_cos_max"] = support_cos
        if support_cos >= support_cos_collapse_max:
            failures.append("support_inter_class_cos_max")

    if not derived:
        return SafetyGateResult("NOT_EVALUATED", tuple(), tuple(), {})
    if "mAP50_drop" in failures or "SV_AP50_drop" in failures:
        status = "FAIL_AP_DROP"
    elif "true_SV_recall_retention" in failures:
        status = "FAIL_TRUE_SV_DAMAGE"
    elif "migration_mass_ratio" in failures:
        status = "FAIL_MIGRATION"
    elif "support_inter_class_cos_max" in failures:
        status = "FAIL_SUPPORT_COLLAPSE"
    elif "det/img_explosion" in failures:
        status = "FAIL_DET_EXPLOSION"
    elif "corrected_FSV_reduction" in failures:
        status = "FAIL_NO_FSV_REDUCTION"
    elif failures:
        status = "PASS_WITH_WARNINGS"
        warnings.extend(failures)
    else:
        status = "PASS"
    return SafetyGateResult(status, tuple(failures), tuple(warnings), derived)
