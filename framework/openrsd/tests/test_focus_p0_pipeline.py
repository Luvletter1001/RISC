import csv
import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON = (
    REPO_ROOT
    / "experiments/rotation_semantic_attractor/scripts/focus_p0_common.py"
)


def _load_common():
    spec = importlib.util.spec_from_file_location("focus_p0_common", COMMON)
    module = importlib.util.module_from_spec(spec)
    sys.modules["focus_p0_common"] = module
    spec.loader.exec_module(module)
    return module


def test_focus_p0_label_policy_keeps_true_vehicle_and_artifacts_out_of_negatives():
    common = _load_common()

    false_row = {
        "audit_category": "valid_unmatched_sv",
        "human_label": "non_vehicle_background",
        "valid_for_human_audit": "true",
    }
    true_row = {
        "audit_category": "valid_unmatched_sv",
        "human_label": "true_vehicle_annot_missing",
        "valid_for_human_audit": "true",
    }
    degenerate = {
        "audit_category": "degenerate_large_sv_box",
        "human_label": "actual_large_sv_degenerate_prediction",
        "valid_for_human_audit": "true",
    }

    assert common.focus_label(false_row) == "corrected_false_sv"
    assert common.use_as_hard_negative(false_row)
    assert common.focus_label(true_row) == "annotation_missing_true_vehicle"
    assert not common.use_as_hard_negative(true_row)
    assert common.focus_label(degenerate) == "excluded_failure_mode"
    assert not common.use_as_hard_negative(degenerate)


def test_focus_p0_split_is_deterministic_and_has_no_crop_leakage():
    common = _load_common()
    rows = []
    for idx in range(10):
        rows.append({
            "crop_id": f"neg_{idx}",
            "focus_label": "corrected_false_sv",
            "audit_category": "valid_unmatched_sv",
        })
        rows.append({
            "crop_id": f"pos_{idx}",
            "focus_label": "annotation_missing_true_vehicle",
            "audit_category": "valid_unmatched_sv",
        })

    split_a = common.build_stratified_splits(rows, seed=7, train_ratio=0.8)
    split_b = common.build_stratified_splits(rows, seed=7, train_ratio=0.8)

    assert split_a == split_b
    train_ids = {row["crop_id"] for row in split_a["train"]}
    eval_ids = {row["crop_id"] for row in split_a["eval"]}
    assert len(train_ids & eval_ids) == 0
    assert len([r for r in split_a["train"] if r["focus_label"] == "corrected_false_sv"]) == 8
    assert len([r for r in split_a["eval"] if r["focus_label"] == "corrected_false_sv"]) == 2


def test_focus_p0_verdict_requires_effect_and_safety():
    common = _load_common()
    baseline = {
        "corrected_FSV": 100,
        "dense_sv_ratio": 0.5,
        "det/img": 10,
        "migration_mass_ratio": 0.1,
        "true_SV_positive_control_retention": 1.0,
        "annotation_missing_true_vehicle_retention": 1.0,
        "degenerate_large_sv_ratio": 0.1,
        "support_inter_class_cos_max": 0.8,
        "support_delta_norm_ratio": 0.0,
    }
    effective = {
        **baseline,
        "variant_id": "V24_full_focus_core",
        "corrected_FSV": 70,
        "dense_sv_ratio": 0.4,
        "true_SV_positive_control_retention": 0.95,
        "annotation_missing_true_vehicle_retention": 0.94,
        "support_delta_norm_ratio": 0.03,
    }
    diagnostic = {
        **baseline,
        "variant_id": "V10_orientation_probe_only",
        "corrected_FSV": "",
        "dense_sv_ratio": "",
    }
    negative_control = {
        **baseline,
        "variant_id": "V02_focus_random_orientation",
        "corrected_FSV": 99,
        "dense_sv_ratio": 0.49,
        "migration_mass_ratio": 0.12,
        "degenerate_large_sv_ratio": 0.11,
    }

    assert common.assign_verdict(effective, baseline).verdict == "EFFECTIVE_CANDIDATE"
    assert common.assign_verdict(diagnostic, baseline).verdict == "DIAGNOSTIC_ONLY"
    assert common.assign_verdict(negative_control, baseline).verdict == "NEGATIVE_CONTROL_CONFIRMED"
