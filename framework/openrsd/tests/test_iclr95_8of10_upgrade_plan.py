import importlib.util


def _load_module():
    path = "M_Tools/analysis/build_iclr95_8of10_upgrade_plan.py"
    spec = importlib.util.spec_from_file_location(
        "build_iclr95_8of10_upgrade_plan", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_upgrade_plan_targets_four_additional_gates_without_manual_promotion():
    mod = _load_module()
    gate_payload = {
        "num_gates_passed": 4,
        "num_gates_total": 10,
        "gates": [
            {"dimension": "Problem anatomy depth", "gate_pass": True,
             "current_score": "9.50"},
            {"dimension": "Literature breadth", "gate_pass": True,
             "current_score": "9.50"},
            {"dimension": "Practicality", "gate_pass": True,
             "current_score": "9.50"},
            {"dimension": "Evidence rigor", "gate_pass": True,
             "current_score": "9.50"},
            {"dimension": "Vision / conceptual height", "gate_pass": False,
             "current_score": "9.50", "blocker": "method gate missing"},
            {"dimension": "Current method novelty", "gate_pass": False,
             "current_score": "8.80"},
            {"dimension": "Method effectiveness", "gate_pass": False,
             "current_score": "8.60"},
            {"dimension": "Applicability breadth", "gate_pass": False,
             "current_score": "8.60"},
            {"dimension": "ICLR paper readiness", "gate_pass": False,
             "current_score": "8.80"},
            {"dimension": "Overall AC score", "gate_pass": False,
             "current_score": "8.60"},
        ],
    }

    plan = mod.build_upgrade_plan(gate_payload)

    assert plan["current_gate"] == "4/10"
    assert plan["target_gate"] == "8/10"
    assert plan["additional_gates_needed"] == 4
    must_close = [
        row["dimension"] for row in plan["target_rows"]
        if row["role_in_8of10"] == "must_close_next"
    ]
    assert must_close == [
        "Vision / conceptual height",
        "Current method novelty",
        "Method effectiveness",
        "Applicability breadth",
    ]
    assert "Do not edit strict gate_pass fields by hand" in (
        plan["promotion_policy"])
    assert "bbox IoU" in plan["forbidden_route"]
