import numpy as np
import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_focus_util(name):
    path = REPO_ROOT / "M_AD" / "models" / "utils" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


focus_head_consensus = _load_focus_util("focus_head_consensus")
focus_negative_prompt = _load_focus_util("focus_negative_prompt")
focus_orbit_teacher = _load_focus_util("focus_orbit_teacher")
focus_safety_gate = _load_focus_util("focus_safety_gate")
focus_sensitivity_channel_mask = _load_focus_util("focus_sensitivity_channel_mask")
focus_spurious_cluster = _load_focus_util("focus_spurious_cluster")
focus_visual_attribute_gate = _load_focus_util("focus_visual_attribute_gate")

score_head_consensus = focus_head_consensus.score_head_consensus
DEFAULT_NEGATIVE_PROMPTS = focus_negative_prompt.DEFAULT_NEGATIVE_PROMPTS
DEFAULT_POSITIVE_PROMPTS = focus_negative_prompt.DEFAULT_POSITIVE_PROMPTS
auxiliary_margin_from_similarities = (
    focus_negative_prompt.auxiliary_margin_from_similarities)
assign_orbit_teacher_label = focus_orbit_teacher.assign_orbit_teacher_label
evaluate_focus_safety = focus_safety_gate.evaluate_focus_safety
build_channel_mask = focus_sensitivity_channel_mask.build_channel_mask
classify_spurious_sv_candidate = (
    focus_spurious_cluster.classify_spurious_sv_candidate)
score_visual_attributes = focus_visual_attribute_gate.score_visual_attributes


def test_spurious_cluster_keeps_true_vehicle_and_artifacts_out_of_hard_negative():
    true_vehicle = classify_spurious_sv_candidate(
        {"audit_category": "valid_unmatched_sv",
         "human_label": "true_vehicle_annot_missing"})
    degenerate = classify_spurious_sv_candidate(
        {"audit_category": "degenerate_large_sv_box",
         "human_label": "non_vehicle_background"})
    false_sv = classify_spurious_sv_candidate(
        {"audit_category": "valid_unmatched_sv",
         "human_label": "non_vehicle_background",
         "sv_margin": "0.8",
         "orientation_confidence": "0.2"})

    assert true_vehicle.cluster_name == "annotation_missing_true_vehicle_cluster"
    assert not true_vehicle.use_as_hard_negative
    assert degenerate.cluster_name == "degenerate_large_sv_cluster"
    assert not degenerate.use_as_hard_negative
    assert false_sv.cluster_name == "corrected_false_sv_context_cluster"
    assert false_sv.use_as_hard_negative
    assert 0.0 <= false_sv.spurious_sv_score <= 1.0


def test_attribute_gate_scores_true_vehicle_higher_than_padding_artifact():
    vehicle = score_visual_attributes({
        "box_w": 18,
        "box_h": 14,
        "orientation_confidence": 0.8,
        "orbit_stability": 0.7,
        "support_similarity_sv": 0.9,
        "support_similarity_lv": 0.2,
        "human_label": "true_vehicle_annot_missing",
    })
    padding = score_visual_attributes({
        "box_w": 220,
        "box_h": 200,
        "orientation_confidence": 0.05,
        "orbit_stability": 0.1,
        "support_similarity_sv": 0.4,
        "support_similarity_lv": 0.7,
        "audit_category": "padding_artifact",
    })

    assert vehicle.attribute_sv_confidence > padding.attribute_sv_confidence
    assert "padding_or_border" in padding.negative_attributes


def test_channel_mask_is_clipped_to_low_strength():
    sensitivity = np.array([0.0, 0.5, 2.0, -3.0], dtype=np.float32)
    true_importance = np.array([1.0, 0.8, 0.2, 0.1], dtype=np.float32)

    mask = build_channel_mask(
        sensitivity,
        true_sv_importance=true_importance,
        max_strength=0.05,
    )

    assert np.all(mask >= 0.95)
    assert np.all(mask <= 1.0)
    assert mask[2] < mask[1]


def test_safety_gate_blocks_ap_drop_and_true_sv_damage():
    baseline = {"mAP50": 0.62, "SV_AP50": 0.50, "true_SV_recall": 0.90,
                "corrected_FSV": 100, "migration_mass_ratio": 0.10,
                "det/img": 20, "support_inter_class_cos_max": 0.80}
    damaged = {"mAP50": 0.59, "SV_AP50": 0.49, "true_SV_recall": 0.70,
               "corrected_FSV": 70, "migration_mass_ratio": 0.10,
               "det/img": 20, "support_inter_class_cos_max": 0.80}

    result = evaluate_focus_safety(damaged, baseline)

    assert result.status == "FAIL_AP_DROP"
    assert "true_SV_recall_retention" in result.failures


def test_orbit_teacher_is_conservative_for_unstable_shortcuts():
    stable = assign_orbit_teacher_label(
        sv_score_mean=0.85,
        sv_score_std=0.03,
        class_consistency=0.95,
        box_consistency=0.90,
        support_margin_consistency=0.92,
        orientation_consistency=0.80,
    )
    unstable = assign_orbit_teacher_label(
        sv_score_mean=0.65,
        sv_score_std=0.35,
        class_consistency=0.30,
        box_consistency=0.20,
        support_margin_consistency=0.25,
        orientation_consistency=0.20,
    )

    assert stable.teacher_label == "stable_true_sv_candidate"
    assert unstable.teacher_label == "unstable_shortcut_candidate"
    assert unstable.teacher_weight < stable.teacher_weight


def test_head_consensus_marks_bbox_only_sv_as_shortcut_candidate():
    result = score_head_consensus(
        bbox_head_sv_score=0.95,
        alignment_head_sv_score=0.10,
        fusion_head_sv_score=0.20,
        support_margin=0.70,
    )

    assert result.consensus_label == "shortcut_candidate"
    assert result.head_disagreement_type == "bbox_high_alignment_fusion_low"


def test_negative_prompt_margin_is_auxiliary_and_uses_negative_vocabulary():
    margin = auxiliary_margin_from_similarities(
        positive_similarities={"small vehicle": 0.8,
                               "vehicle seen from above": 0.6},
        negative_similarities={"not a court line": 0.3,
                               "not a roof edge": 0.9},
    )

    assert margin == -0.1
    assert "small vehicle" in DEFAULT_POSITIVE_PROMPTS
    assert "not a court line" in DEFAULT_NEGATIVE_PROMPTS
