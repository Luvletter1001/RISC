from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _load_common():
    path = Path(
        "experiments/rotation_semantic_attractor/scripts/"
        "text_fourier_cpu10_common.py")
    spec = importlib.util.spec_from_file_location(
        "text_fourier_cpu10_common", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["text_fourier_cpu10_common"] = module
    spec.loader.exec_module(module)
    return module


def _sample(common):
    return common.NormalizedSample(
        crop_id="unit",
        audit_category="corrected_false_sv",
        human_label="non_vehicle_background",
        binary_true_vehicle=0,
        focus_sv_score=0.80,
        eqtext_sv_similarity=0.30,
        positive_text_similarity=0.40,
        max_negative_text_similarity=0.20,
        negative_text_margin=0.20,
        visual_text_consistency=0.70,
        orientation_theta=0.25,
        orientation_confidence=0.75,
        fourier_phase_norm=1.0,
        fft_low=0.20,
        fft_mid=0.50,
        fft_high=0.30,
        fft_entropy=0.35,
        padding_overlap_ratio=0.05,
        degenerate_risk=0.10,
    )


def test_cpu_guard_sets_required_environment(monkeypatch):
    common = _load_common()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    common.configure_cpu_environment()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert os.environ["OMP_NUM_THREADS"] == "4"
    assert os.environ["MKL_NUM_THREADS"] == "4"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "4"
    common.assert_cpu_only_environment()


def test_registry_has_exactly_ten_methods():
    common = _load_common()

    assert [method.method_id for method in common.METHOD_REGISTRY] == [
        "TF01_phase_conf_gate",
        "TF02_harmonic_prompt_router",
        "TF03_negative_phase_margin",
        "TF04_band_prompt_mixture",
        "TF05_entropy_safe_gate",
        "TF06_rotation_consistency",
        "TF07_phase_residual_shadow",
        "TF08_text_fourier_consensus",
        "TF09_padding_degenerate_risk",
        "TF10_one_way_downweight",
    ]


def test_auc_uses_pairwise_ranking_with_ties():
    common = _load_common()

    assert common.roc_auc([1, 0], [0.9, 0.1]) == 1.0
    assert common.roc_auc([1, 0], [0.1, 0.9]) == 0.0
    assert common.roc_auc([1, 0], [0.5, 0.5]) == 0.5


def test_method_scores_are_deterministic_and_tf10_is_one_way():
    common = _load_common()
    sample = _sample(common)

    first = common.compute_method_scores(sample)
    second = common.compute_method_scores(sample)

    assert first == second
    assert len(first) == 10
    assert all(0.0 <= value <= 1.0 for value in first.values())
    assert first["TF10_one_way_downweight"] <= sample.focus_sv_score


def test_method_summary_contains_rank_and_rejection_reason():
    common = _load_common()
    rows = [
        {
            "binary_true_vehicle": "1",
            "audit_category": "annotation_missing_true_vehicle",
            "TF01_phase_conf_gate": "0.9",
            "TF02_harmonic_prompt_router": "0.9",
            "TF03_negative_phase_margin": "0.9",
            "TF04_band_prompt_mixture": "0.9",
            "TF05_entropy_safe_gate": "0.9",
            "TF06_rotation_consistency": "0.9",
            "TF07_phase_residual_shadow": "0.9",
            "TF08_text_fourier_consensus": "0.9",
            "TF09_padding_degenerate_risk": "0.9",
            "TF10_one_way_downweight": "0.7",
            "focus_sv_score": "0.8",
        },
        {
            "binary_true_vehicle": "0",
            "audit_category": "corrected_false_sv",
            "TF01_phase_conf_gate": "0.1",
            "TF02_harmonic_prompt_router": "0.1",
            "TF03_negative_phase_margin": "0.1",
            "TF04_band_prompt_mixture": "0.1",
            "TF05_entropy_safe_gate": "0.1",
            "TF06_rotation_consistency": "0.1",
            "TF07_phase_residual_shadow": "0.1",
            "TF08_text_fourier_consensus": "0.1",
            "TF09_padding_degenerate_risk": "0.1",
            "TF10_one_way_downweight": "0.5",
            "focus_sv_score": "0.8",
        },
    ]

    summary = common.summarize_methods(rows, baseline_auc=0.5)

    assert len(summary) == 10
    assert summary[0]["rank"] == 1
    assert "safe_candidate" in summary[0]
    assert "reject_reason" in summary[0]


def test_idea_registry_has_exactly_one_hundred_numbered_ideas():
    common = _load_common()

    idea_ids = [idea.idea_id for idea in common.IDEA_REGISTRY]

    assert len(idea_ids) == 100
    assert idea_ids == [f"I{index:03d}" for index in range(1, 101)]


def test_idea_scores_are_deterministic_and_keep_first_ten_methods():
    common = _load_common()
    sample = _sample(common)

    method_scores = common.compute_method_scores(sample)
    first = common.compute_idea_scores(sample)
    second = common.compute_idea_scores(sample)

    assert first == second
    assert len(first) == 100
    assert all(0.0 <= value <= 1.0 for value in first.values())
    assert first["I001"] == method_scores["TF01_phase_conf_gate"]
    assert first["I010"] == method_scores["TF10_one_way_downweight"]


def test_idea_summary_contains_numeric_evidence_for_all_ideas():
    common = _load_common()
    true_sample = common.NormalizedSample(
        crop_id="true",
        audit_category="annotation_missing_true_vehicle",
        human_label="true_vehicle_annot_missing",
        binary_true_vehicle=1,
        focus_sv_score=0.85,
        eqtext_sv_similarity=0.60,
        positive_text_similarity=0.80,
        max_negative_text_similarity=-0.40,
        negative_text_margin=1.20,
        visual_text_consistency=0.80,
        orientation_theta=0.20,
        orientation_confidence=0.90,
        fourier_phase_norm=1.0,
        fft_low=0.20,
        fft_mid=0.55,
        fft_high=0.25,
        fft_entropy=0.30,
        padding_overlap_ratio=0.00,
        degenerate_risk=0.00,
    )
    false_sample = common.NormalizedSample(
        crop_id="false",
        audit_category="padding_artifact",
        human_label="non_vehicle_background",
        binary_true_vehicle=0,
        focus_sv_score=0.80,
        eqtext_sv_similarity=-0.20,
        positive_text_similarity=-0.30,
        max_negative_text_similarity=0.70,
        negative_text_margin=-1.00,
        visual_text_consistency=0.10,
        orientation_theta=1.10,
        orientation_confidence=0.20,
        fourier_phase_norm=0.20,
        fft_low=0.15,
        fft_mid=0.25,
        fft_high=0.60,
        fft_entropy=0.95,
        padding_overlap_ratio=0.75,
        degenerate_risk=0.40,
    )
    rows = [
        {
            "binary_true_vehicle": "1",
            "audit_category": true_sample.audit_category,
            "focus_sv_score": true_sample.focus_sv_score,
            **common.compute_idea_scores(true_sample),
        },
        {
            "binary_true_vehicle": "0",
            "audit_category": false_sample.audit_category,
            "focus_sv_score": false_sample.focus_sv_score,
            **common.compute_idea_scores(false_sample),
        },
    ]

    summary = common.summarize_ideas(rows, baseline_auc=0.5)

    assert len(summary) == 100
    assert summary[0]["rank"] == 1
    assert all(row["evidence_status"] == "PASS_NUMERIC_PROXY_EVIDENCE" for row in summary)
    assert all(row["sample_count"] == 2 for row in summary)
    assert all(row["labeled_count"] == 2 for row in summary)
