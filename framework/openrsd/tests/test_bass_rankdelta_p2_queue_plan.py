import importlib.util
from pathlib import Path


def _load_module():
    path = Path("M_Tools/analysis/plan_bass_rankdelta_p2_queue.py")
    spec = importlib.util.spec_from_file_location(
        "plan_bass_rankdelta_p2_queue", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p2_queue_waits_until_closure_routes_to_p2(tmp_path):
    mod = _load_module()

    plan = mod.plan_p2_queue({"action": "WAIT_P0"}, tmp_path / "missing.flag")

    assert plan["action"] == "WAIT_UPSTREAM"
    assert plan["closure_action"] == "WAIT_P0"


def test_p2_queue_runs_after_rank_assignment_action(tmp_path):
    mod = _load_module()

    plan = mod.plan_p2_queue(
        {"action": "RUN_P2_RANK_ASSIGNMENT"},
        tmp_path / "missing.flag")

    assert plan["action"] == "RUN_P2"
    assert [row["variant"] for row in plan["queue"]] == [
        "assign_rank",
        "posterior_rank",
    ]


def test_p2_queue_does_not_duplicate_existing_marker(tmp_path):
    mod = _load_module()
    marker = tmp_path / "p2.flag"
    marker.write_text("launched\n", encoding="utf-8")

    plan = mod.plan_p2_queue({"action": "RUN_P2_RANK_ASSIGNMENT"}, marker)

    assert plan["action"] == "WAIT_P2_EXISTING"


def test_p2_queue_stops_after_p2_complete_no_strict_even_with_marker(tmp_path):
    mod = _load_module()
    marker = tmp_path / "p2.flag"
    marker.write_text("launched\n", encoding="utf-8")

    plan = mod.plan_p2_queue(
        {"action": "STOP_P2_DONE_NO_STRICT"},
        marker)

    assert plan["action"] == "STOP_P2_DONE_NO_STRICT"


def test_p2_queue_stops_when_confirmation_route_exists(tmp_path):
    mod = _load_module()

    plan = mod.plan_p2_queue(
        {"action": "RUN_CONFIRMATION_PACKAGE"},
        tmp_path / "missing.flag")

    assert plan["action"] == "STOP_CONFIRMATION_ROUTE"
