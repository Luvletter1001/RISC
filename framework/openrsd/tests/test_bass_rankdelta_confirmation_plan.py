import importlib.util
from pathlib import Path


def _load_module():
    path = Path("M_Tools/analysis/plan_bass_rankdelta_confirmation_package.py")
    spec = importlib.util.spec_from_file_location(
        "plan_bass_rankdelta_confirmation_package", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_confirmation_waits_before_strict_pass(tmp_path):
    mod = _load_module()

    plan = mod.plan_confirmation(
        {"action": "WAIT_P0"},
        {"rows": []},
        tmp_path / "missing.flag")

    assert plan["action"] == "WAIT_UPSTREAM"
    assert plan["closure_action"] == "WAIT_P0"


def test_confirmation_runs_after_strict_closure_method(tmp_path):
    mod = _load_module()

    plan = mod.plan_confirmation(
        {"action": "RUN_CONFIRMATION_PACKAGE",
         "strict_pass_methods": ["RankDelta min-score 0.50"]},
        {"rows": []},
        tmp_path / "missing.flag")

    assert plan["action"] == "RUN_CONFIRMATION"
    assert plan["strict_pass_methods"] == ["RankDelta min-score 0.50"]
    assert [row["priority"] for row in plan["queue"]] == [
        "C0", "C1", "C2", "C3", "C4"]


def test_confirmation_finds_strict_method_from_summary(tmp_path):
    mod = _load_module()

    plan = mod.plan_confirmation(
        {"action": "RUN_CONFIRMATION_PACKAGE"},
        {"rows": [
            {"method": "P2 assign-rank BASS-GSF", "gate": "strict pass"},
        ]},
        tmp_path / "missing.flag")

    assert plan["action"] == "RUN_CONFIRMATION"
    assert plan["strict_pass_methods"] == ["P2 assign-rank BASS-GSF"]


def test_confirmation_does_not_duplicate_existing_marker(tmp_path):
    mod = _load_module()
    marker = tmp_path / "confirm.flag"
    marker.write_text("launched\n", encoding="utf-8")

    plan = mod.plan_confirmation(
        {"action": "RUN_CONFIRMATION_PACKAGE",
         "strict_pass_methods": ["RankDelta min-score 0.30"]},
        {"rows": []},
        marker)

    assert plan["action"] == "WAIT_CONFIRMATION_EXISTING"


def test_confirmation_stops_when_p2_route_is_active(tmp_path):
    mod = _load_module()

    plan = mod.plan_confirmation(
        {"action": "RUN_P2_RANK_ASSIGNMENT"},
        {"rows": []},
        tmp_path / "missing.flag")

    assert plan["action"] == "STOP_P2_ROUTE"


def test_confirmation_stops_after_p2_complete_no_strict(tmp_path):
    mod = _load_module()

    plan = mod.plan_confirmation(
        {"action": "STOP_P2_DONE_NO_STRICT"},
        {"rows": []},
        tmp_path / "missing.flag")

    assert plan["action"] == "STOP_P2_DONE_NO_STRICT"
