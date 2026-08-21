from pathlib import Path

import pytest

from M_Tools.analysis import build_s3c_reviewer_gate_audit as audit


def write_json(path: Path, payload: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def touch(path: Path, text: str = "ok") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_reviewer_gate_accepts_complete_gpu45_pre_nms_outputs(tmp_path,
                                                             monkeypatch):
    gpu67 = tmp_path / "p4_s3c_pre_nms_z4_l0p25_gpu67"
    gpu45 = tmp_path / "p4_s3c_pre_nms_z4_l0p25_gpu45"
    touch(gpu45 / "predictions.pkl")
    touch(gpu45 / "ap_eval.json", '{"metrics":{"mAP":0.5678,"AP50":0.568}}')
    touch(
        gpu45 / "sise_eval/sise_score_calibration_summary.json",
        '{"summaries":[{"variant":"base","sise_logz_score_ge_0p999":3894,'
        '"correct_score_ge_0p999":68538},'
        '{"variant":"pre_nms","sise_logz_score_ge_0p999":532,'
        '"correct_score_ge_0p999":68481}]}',
    )
    touch(
        gpu45 / "dense_topk_summary.json",
        '{"summary":{"flagged_rows_before_minus_after":68807,'
        '"focus_rows_before_minus_after":22395}}',
    )

    paths = {
        "calibration": write_json(
            tmp_path / "calibration.json",
            '{"selected_holdout":{"sise_logz_reduction_rate":0.9,'
            '"correct_drop_high_rate":0.01},'
            '"selected_on_calibration":{"z_margin":4.0,"lambda":0.25,'
            '"sise_logz_reduction_rate":0.91}}',
        ),
        "p4_sise": write_json(
            tmp_path / "p4_sise.json",
            '{"summaries":[{"sise_logz_score_ge_0p999":100,'
            '"si_ece_logz_implausible":0.9,"correct_score_ge_0p999":1000,'
            '"sise_logz_wrong_excl_sibling":200},'
            '{"sise_logz_score_ge_0p999":1,'
            '"si_ece_logz_implausible":0.6,"correct_score_ge_0p999":999}]}',
        ),
        "p4_ap": write_json(
            tmp_path / "p4_ap.json",
            '{"metrics":{"mAP":0.572,"AP50":0.572,'
            '"dota/IoU_50_Detail":{"small-vehicle":{"ap":0.0887}}}}',
        ),
        "cross_detector": write_json(
            tmp_path / "cross_detector.json",
            '{"rows":[{"model":"p4_focus_lowtext_eval_bundle_full9772",'
            '"sise_logz_score_ge_0p999":10},'
            '{"model":"redet","sise_logz_score_ge_0p999":0}]}',
        ),
        "dota1_sise": write_json(
            tmp_path / "dota1_sise.json",
            '{"summaries":[{"sise_logz_score_ge_0p9":2,'
            '"sise_logz_score_ge_0p999":0},'
            '{"sise_logz_score_ge_0p9":0,'
            '"sise_logz_score_ge_0p999":0}]}',
        ),
        "dota1_ap_base": write_json(
            tmp_path / "dota1_ap_base.json", '{"metrics":{"mAP":0.6891}}'),
        "dota1_ap_s3c": write_json(
            tmp_path / "dota1_ap_s3c.json", '{"metrics":{"mAP":0.6892}}'),
        "pre_nms_predictions": gpu67 / "predictions.pkl",
        "pre_nms_ap": gpu67 / "ap_eval.json",
        "pre_nms_sise": gpu67 / "sise_eval/sise_score_calibration_summary.json",
        "pre_nms_dense": gpu67 / "dense_topk_summary.json",
    }
    monkeypatch.setattr(audit, "PATHS", paths)
    monkeypatch.setattr(
        audit, "PRE_NMS_CANDIDATE_DIRS", [gpu67, gpu45], raising=False)

    payload = audit.build_payload()

    assert payload["evidence"]["pre_nms_complete"] is True
    assert payload["evidence"]["pre_nms_work_dir"] == str(gpu45)
    assert all(payload["evidence"]["pre_nms_outputs"].values())
    assert payload["problem_score"] >= 8.5
    assert payload["method_score"] >= 8.5
    assert payload["application_scope_score"] >= 8.0
    assert payload["domain_mode"] in {"ovd", "mixed"}
    assert payload["gaussian_energy_mode"] in {
        "s3c_guard", "continuous_logit_energy", "logit_adapter"}
    assert "application_scope" in payload["score_breakdown"]
    method_gate = next(
        row for row in payload["gates"]
        if row["gate"] == "method_level_pre_nms_integration")
    assert method_gate["status"] == "PASS"
    assert method_gate["current_score"] >= 8.5
    application_gate = next(
        row for row in payload["gates"]
        if row["gate"] == "application_scope_rs_generality")
    assert application_gate["current_score"] >= 8.0


def test_reviewer_gate_rejects_complete_but_weak_pre_nms_outputs(
        tmp_path, monkeypatch):
    gpu45 = tmp_path / "p4_s3c_pre_nms_z4_l0p25_gpu45"
    touch(gpu45 / "predictions.pkl")
    touch(gpu45 / "ap_eval.json", '{"metrics":{"mAP":0.55,"AP50":0.55}}')
    touch(
        gpu45 / "sise_eval/sise_score_calibration_summary.json",
        '{"summaries":[{"variant":"base","sise_logz_score_ge_0p999":100,'
        '"correct_score_ge_0p999":1000},'
        '{"variant":"pre_nms","sise_logz_score_ge_0p999":80,'
        '"correct_score_ge_0p999":980}]}',
    )
    touch(
        gpu45 / "dense_topk_summary.json",
        '{"summary":{"flagged_rows_before_minus_after":0,'
        '"focus_rows_before_minus_after":0}}',
    )

    paths = {
        "calibration": write_json(
            tmp_path / "calibration.json",
            '{"selected_holdout":{"sise_logz_reduction_rate":0.9,'
            '"correct_drop_high_rate":0.01},'
            '"selected_on_calibration":{"z_margin":4.0,"lambda":0.25,'
            '"sise_logz_reduction_rate":0.91}}',
        ),
        "p4_sise": write_json(
            tmp_path / "p4_sise.json",
            '{"summaries":[{"sise_logz_score_ge_0p999":100,'
            '"si_ece_logz_implausible":0.9,"correct_score_ge_0p999":1000,'
            '"sise_logz_wrong_excl_sibling":200},'
            '{"sise_logz_score_ge_0p999":1,'
            '"si_ece_logz_implausible":0.6,"correct_score_ge_0p999":999}]}',
        ),
        "p4_ap": write_json(
            tmp_path / "p4_ap.json",
            '{"metrics":{"mAP":0.572,"AP50":0.572}}',
        ),
        "cross_detector": write_json(
            tmp_path / "cross_detector.json",
            '{"rows":[{"model":"p4_focus_lowtext_eval_bundle_full9772",'
            '"sise_logz_score_ge_0p999":10}]}',
        ),
        "dota1_sise": write_json(
            tmp_path / "dota1_sise.json",
            '{"summaries":[{"sise_logz_score_ge_0p9":2,'
            '"sise_logz_score_ge_0p999":0},'
            '{"sise_logz_score_ge_0p9":0,'
            '"sise_logz_score_ge_0p999":0}]}',
        ),
        "dota1_ap_base": write_json(
            tmp_path / "dota1_ap_base.json", '{"metrics":{"mAP":0.6891}}'),
        "dota1_ap_s3c": write_json(
            tmp_path / "dota1_ap_s3c.json", '{"metrics":{"mAP":0.6892}}'),
        "pre_nms_predictions": tmp_path / "missing/predictions.pkl",
        "pre_nms_ap": tmp_path / "missing/ap_eval.json",
        "pre_nms_sise": tmp_path / "missing/sise_eval/sise_score_calibration_summary.json",
        "pre_nms_dense": tmp_path / "missing/dense_topk_summary.json",
    }
    monkeypatch.setattr(audit, "PATHS", paths)
    monkeypatch.setattr(
        audit, "PRE_NMS_CANDIDATE_DIRS", [gpu45], raising=False)

    payload = audit.build_payload()

    method_gate = next(
        row for row in payload["gates"]
        if row["gate"] == "method_level_pre_nms_integration")
    assert method_gate["status"] != "PASS"
    assert method_gate["current_score"] < 8.5
    assert payload["application_scope_score"] >= 8.0
    assert payload["all_core_scores_ge_8p5"] is False
