import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path("M_Tools/analysis/summarize_p3a_ap_projection_results.py")
    spec = importlib.util.spec_from_file_location(
        "summarize_p3a_ap_projection_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_assess_gate_accepts_ap_safe_topk_risk_projection():
    mod = _load_module()
    control = {
        "mAP": 0.862878,
        "top5000_precision": 0.7168,
        "top5000_sise_logz": 48,
        "top5000_risk_weighted_logz": 10.45,
        "risk_weighted_logz": 14.28,
    }
    row = {
        "kind": "p3a_main",
        "mAP": 0.862961,
        "top5000_precision": 0.7170,
        "top5000_sise_logz": 0,
        "top5000_risk_weighted_logz": 0.0,
        "risk_weighted_logz": 5.69,
        "changed_correct": 0,
    }

    assert mod.assess_gate(row, control) == "strict pass"


def test_read_projection_summary_records_forbidden_route(tmp_path):
    mod = _load_module()
    path = tmp_path / "projection.json"
    path.write_text(json.dumps({
        "z_thr": 2.0,
        "min_score": 0.1,
        "protected_correct": 3584,
        "changed_correct": 0,
        "low_support_wrong_candidates": 242,
        "scores_changed": 242,
        "forbidden_bbox_gaussian_route": False,
    }), encoding="utf-8")

    summary = mod.read_projection_summary(path)

    assert summary["projection_status"] == "done"
    assert summary["changed_correct"] == 0
    assert summary["forbidden_bbox_gaussian_route"] is False
