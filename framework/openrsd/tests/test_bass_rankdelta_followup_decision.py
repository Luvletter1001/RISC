import importlib.util
from pathlib import Path


def _load_module():
    path = Path("M_Tools/analysis/decide_bass_rankdelta_followup.py")
    spec = importlib.util.spec_from_file_location(
        "decide_bass_rankdelta_followup", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(method, gate="waiting/missing", map_value=None, kind="rankdelta_p0"):
    return {
        "method": method,
        "kind": kind,
        "gate": gate,
        "mAP": map_value,
    }


def test_followup_decision_waits_until_p0_is_complete(tmp_path):
    mod = _load_module()
    summary = {
        "rows": [
            _row("RankDelta min-score 0.30"),
            _row("RankDelta min-score 0.50"),
            _row("P1 gentle RankDelta", kind="rankdelta_p1"),
            _row("P1 protected RankDelta", kind="rankdelta_p1"),
        ]
    }

    decision = mod.decide(summary, tmp_path / "missing.flag")

    assert decision["action"] == "WAIT_P0"
    assert decision["p0_done"] is False


def test_followup_decision_stops_on_strict_pass(tmp_path):
    mod = _load_module()
    summary = {
        "rows": [
            _row("RankDelta min-score 0.30", gate="strict pass", map_value=0.9),
            _row("RankDelta min-score 0.50", gate="fail", map_value=0.8),
            _row("P1 gentle RankDelta", kind="rankdelta_p1"),
            _row("P1 protected RankDelta", kind="rankdelta_p1"),
        ]
    }

    decision = mod.decide(summary, tmp_path / "missing.flag")

    assert decision["action"] == "STOP_STRICT_PASS"
    assert decision["strict_pass_methods"] == ["RankDelta min-score 0.30"]


def test_followup_decision_runs_p1_after_p0_fail(tmp_path):
    mod = _load_module()
    summary = {
        "rows": [
            _row("RankDelta min-score 0.30", gate="fail", map_value=0.8),
            _row("RankDelta min-score 0.50", gate="pilot pass", map_value=0.86),
            _row("P1 gentle RankDelta", kind="rankdelta_p1"),
            _row("P1 protected RankDelta", kind="rankdelta_p1"),
        ]
    }

    decision = mod.decide(summary, tmp_path / "missing.flag")

    assert decision["action"] == "RUN_P1"
    assert decision["p0_done"] is True
    assert decision["pilot_pass_methods"] == ["RankDelta min-score 0.50"]


def test_followup_decision_uses_marker_to_avoid_duplicate_p1(tmp_path):
    mod = _load_module()
    marker = tmp_path / "p1.flag"
    marker.write_text("launched\n", encoding="utf-8")
    summary = {
        "rows": [
            _row("RankDelta min-score 0.30", gate="fail", map_value=0.8),
            _row("RankDelta min-score 0.50", gate="fail", map_value=0.81),
            _row("P1 gentle RankDelta", kind="rankdelta_p1"),
            _row("P1 protected RankDelta", kind="rankdelta_p1"),
        ]
    }

    decision = mod.decide(summary, marker)

    assert decision["action"] == "WAIT_P1_EXISTING"
