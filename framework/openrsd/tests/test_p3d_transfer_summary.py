import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path("M_Tools/analysis/summarize_p3d_transfer_results.py")
    spec = importlib.util.spec_from_file_location(
        "summarize_p3d_transfer_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_assess_p3d_transfer_gate_requires_ap_safety_and_risk_reduction():
    mod = _load_module()
    baseline = {
        "mAP": 0.644687,
        "top5000_precision": 1.0,
        "top5000_sise_logz": 0,
        "top5000_risk_weighted_logz": 0.0,
        "risk_weighted_logz": 2236.0,
        "risk_weighted_logz_score_ge_0p5": 6.1,
    }
    row = {
        "mAP": 0.645981,
        "top5000_precision": 1.0,
        "top5000_sise_logz": 0,
        "top5000_risk_weighted_logz": 0.0,
        "risk_weighted_logz": 1827.0,
        "risk_weighted_logz_score_ge_0p5": 0.01,
        "scores_changed": 4543,
        "changed_correct": 0,
        "tp_rank_harm": 0,
    }

    assert mod.assess_p3d_transfer_gate(
        row, baseline) == "strict transfer pass"

    row["mAP"] = 0.64
    assert mod.assess_p3d_transfer_gate(row, baseline) == "fail"

    row["mAP"] = 0.645981
    row["changed_correct"] = 1
    assert mod.assess_p3d_transfer_gate(row, baseline) == "fail"

    row["changed_correct"] = 0
    row["tp_rank_harm"] = 0.1
    assert mod.assess_p3d_transfer_gate(row, baseline) == "fail"


def test_build_rows_reads_projection_eval_and_risk(tmp_path):
    mod = _load_module()
    baseline_eval = tmp_path / "baseline_eval.json"
    p3d_eval = tmp_path / "p3d_eval.json"
    projection = tmp_path / "projection.json"
    risk = tmp_path / "risk.json"
    baseline_eval.write_text(json.dumps({
        "metrics": {"dota/mAP": 0.644687, "dota/AP50": 0.645}
    }), encoding="utf-8")
    p3d_eval.write_text(json.dumps({
        "metrics": {"dota/mAP": 0.645981, "dota/AP50": 0.646}
    }), encoding="utf-8")
    projection.write_text(json.dumps({
        "z_thr": 2.0,
        "min_score": 0.1,
        "scores_changed": 4543,
        "protected_correct": 100705,
        "changed_correct": 0,
        "low_support_wrong_candidates": 4543,
        "forbidden_bbox_gaussian_route": False,
    }), encoding="utf-8")
    risk.write_text(json.dumps({
        "summaries": [
            {
                "variant": "dior_baseline",
                "localized_correct": 103305,
                "localized_wrong": 155742,
                "risk_weighted_logz_false_alarm": 2236.0,
                "risk_weighted_logz_false_alarm_score_ge_0p5": 6.1,
                "sise_logz_ge_thr_score_ge_0p5": 9,
                "topk_precision_top5000": 1.0,
                "sise_logz_topk_top5000": 0,
                "risk_weighted_logz_false_alarm_top5000": 0.0,
            },
            {
                "variant": "p3d_dior_z2_s01",
                "localized_correct": 103305,
                "localized_wrong": 155742,
                "risk_weighted_logz_false_alarm": 1827.0,
                "risk_weighted_logz_false_alarm_score_ge_0p5": 0.01,
                "sise_logz_ge_thr_score_ge_0p5": 2,
                "topk_precision_top5000": 1.0,
                "sise_logz_topk_top5000": 0,
                "risk_weighted_logz_false_alarm_top5000": 0.0,
            },
        ],
        "rank_tail_summary": [{
            "method_variant": "p3d_dior_z2_s01",
            "ap_contributing_fp_removed": 54,
            "tp_rank_harm": 0,
            "sise_fp_rank_benefit": 10.18,
        }],
    }), encoding="utf-8")

    rows = mod.build_rows(baseline_eval, p3d_eval, projection, risk)
    payload = mod.build_payload(rows, "resultmd/p3d.md")

    assert rows[0]["gate"] == "transfer control"
    assert rows[1]["gate"] == "strict transfer pass"
    assert rows[1]["mAP"] == 0.645981
    assert rows[1]["scores_changed"] == 4543
    assert rows[1]["tp_rank_harm"] == 0
    assert payload["strict_transfer_pass"] is True
    assert payload["dataset"] == "DIOR-R"
