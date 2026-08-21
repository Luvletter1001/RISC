import importlib.util
from pathlib import Path


def _load_module():
    path = Path("M_Tools/analysis/build_bass_rankdelta_paper_snippets.py")
    spec = importlib.util.spec_from_file_location(
        "build_bass_rankdelta_paper_snippets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snippet_keeps_missing_rankdelta_as_tbd():
    mod = _load_module()
    payload = mod.build_payload({
        "status": "waiting/missing",
        "rows": [
            {"method": "density w005 e2", "kind": "control",
             "mAP": 0.860457, "gate": "density control"},
            {"method": "RankDelta min-score 0.30", "kind": "rankdelta_p0",
             "mAP": None, "gate": "waiting/missing"},
        ],
    }, {
        "action": "WAIT_P0",
        "paper_position": "no new method result may be claimed",
    })

    assert payload["rankdelta_rows_total"] == 1
    assert payload["rankdelta_rows_filled"] == 0
    assert payload["has_strict_pass"] is False
    assert "TBD" in payload["markdown_table"]
    assert "WAIT\\_P0" in payload["latex_table"]


def test_snippet_marks_strict_pass_without_inventing_extra_rows():
    mod = _load_module()
    payload = mod.build_payload({
        "status": "done",
        "rows": [
            {"method": "RankDelta min-score 0.50", "kind": "rankdelta_p0",
             "mAP": 0.863, "AP50": 0.864, "localized_wrong": 6200,
             "sise_logz": 40, "top5000_precision": 0.717,
             "top5000_sise_logz": 35, "risk_weighted_logz": 10.0,
             "gate": "strict pass"},
        ],
    }, {
        "action": "RUN_CONFIRMATION_PACKAGE",
        "paper_position": "candidate main method after confirmation",
    })

    assert payload["rankdelta_rows_total"] == 1
    assert payload["rankdelta_rows_filled"] == 1
    assert payload["has_strict_pass"] is True
    assert "0.863000" in payload["markdown_table"]
    assert "strict pass" in payload["latex_table"]
