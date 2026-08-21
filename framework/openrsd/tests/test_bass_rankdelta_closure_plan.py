import importlib.util
from pathlib import Path


def _load_module():
    path = Path("M_Tools/analysis/plan_bass_rankdelta_closure.py")
    spec = importlib.util.spec_from_file_location(
        "plan_bass_rankdelta_closure", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(method, gate="waiting/missing", map_value=None, kind="rankdelta_p0"):
    return {
        "method": method,
        "kind": kind,
        "gate": gate,
        "mAP": map_value,
        "top5000_precision": None,
    }


def _summary(*rows):
    return {"status": "synthetic", "rows": list(rows)}


def test_closure_plan_waits_for_p0_until_rankdelta_rows_exist():
    mod = _load_module()

    plan = mod.plan_closure(_summary(
        _row("RankDelta min-score 0.30"),
        _row("RankDelta min-score 0.50")))

    assert plan["action"] == "WAIT_P0"
    assert plan["p0_done"] is False
    assert plan["paper_position"] == "no new method result may be claimed"


def test_closure_plan_runs_confirmation_after_strict_pass():
    mod = _load_module()

    plan = mod.plan_closure(_summary(
        _row("RankDelta min-score 0.30", gate="strict pass", map_value=0.87),
        _row("RankDelta min-score 0.50", gate="fail", map_value=0.86)))

    assert plan["action"] == "RUN_CONFIRMATION_PACKAGE"
    assert plan["strict_pass_methods"] == ["RankDelta min-score 0.30"]
    assert any(row["priority"] == "C4" for row in plan["execution_priority"])


def test_closure_plan_routes_to_p1_after_p0_no_strict():
    mod = _load_module()

    plan = mod.plan_closure(
        _summary(
            _row("RankDelta min-score 0.30", gate="fail", map_value=0.85),
            _row("RankDelta min-score 0.50", gate="pilot pass", map_value=0.86),
            _row("P1 gentle RankDelta", kind="rankdelta_p1"),
            _row("P1 protected RankDelta", kind="rankdelta_p1")),
        {"action": "RUN_P1"})

    assert plan["action"] == "RUN_P1"
    assert plan["pilot_pass_methods"] == ["RankDelta min-score 0.50"]


def test_closure_plan_waits_for_existing_p1_marker():
    mod = _load_module()

    plan = mod.plan_closure(
        _summary(
            _row("RankDelta min-score 0.30", gate="fail", map_value=0.85),
            _row("RankDelta min-score 0.50", gate="fail", map_value=0.84),
            _row("P1 gentle RankDelta", kind="rankdelta_p1"),
            _row("P1 protected RankDelta", kind="rankdelta_p1")),
        {"action": "WAIT_P1_EXISTING"})

    assert plan["action"] == "WAIT_P1_EXISTING"


def test_closure_plan_stops_delta_tuning_after_p1_no_strict():
    mod = _load_module()

    plan = mod.plan_closure(_summary(
        _row("RankDelta min-score 0.30", gate="fail", map_value=0.85),
        _row("RankDelta min-score 0.50", gate="fail", map_value=0.84),
        _row("P1 gentle RankDelta", gate="fail", map_value=0.85,
             kind="rankdelta_p1"),
        _row("P1 protected RankDelta", gate="pilot pass", map_value=0.86,
             kind="rankdelta_p1")))

    assert plan["action"] == "RUN_P2_RANK_ASSIGNMENT"
    assert any(row["priority"] == "P2B" for row in plan["execution_priority"])


def test_closure_plan_stops_p2_after_complete_no_strict():
    mod = _load_module()

    plan = mod.plan_closure(_summary(
        _row("RankDelta min-score 0.30", gate="fail", map_value=0.85),
        _row("RankDelta min-score 0.50", gate="fail", map_value=0.84),
        _row("P1 gentle RankDelta", gate="pilot pass", map_value=0.862,
             kind="rankdelta_p1"),
        _row("P1 protected RankDelta", gate="fail", map_value=0.857,
             kind="rankdelta_p1"),
        _row("P2 assign-rank BASS-GSF", gate="fail", map_value=0.858,
             kind="rankdelta_p2"),
        _row("P2 posterior-rank BASS-GSF", gate="fail", map_value=0.861,
             kind="rankdelta_p2")))

    assert plan["action"] == "STOP_P2_DONE_NO_STRICT"
    assert plan["p2_done"] is True
    assert any(row["priority"] == "P3A" for row in plan["execution_priority"])
