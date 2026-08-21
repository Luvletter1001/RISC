import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path("M_Tools/analysis/refresh_bass_rankdelta_paper_package.py")
    spec = importlib.util.spec_from_file_location(
        "refresh_bass_rankdelta_paper_package", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_refresh_command_plan_preserves_dependency_order():
    mod = _load_module()

    plan = mod.build_command_plan("/env/python")

    assert [row[1] for row in plan] == [
        "M_Tools/analysis/summarize_bass_rankdelta_results.py",
        "M_Tools/analysis/decide_bass_rankdelta_followup.py",
        "M_Tools/analysis/plan_bass_rankdelta_closure.py",
        "M_Tools/analysis/plan_bass_rankdelta_confirmation_package.py",
        "M_Tools/analysis/plan_bass_rankdelta_p2_queue.py",
        "M_Tools/analysis/build_bass_rankdelta_paper_snippets.py",
        "M_Tools/analysis/summarize_p3a_ap_projection_results.py",
        "M_Tools/analysis/summarize_p3d_transfer_results.py",
        "M_Tools/analysis/build_iclr95_evidence_gate.py",
        "M_Tools/analysis/build_iclr95_8of10_upgrade_plan.py",
        "M_Tools/analysis/build_semantic_scale_paper_figures.py",
        "M_Tools/analysis/audit_gpu67_wait_blocker.py",
        "M_Tools/analysis/audit_bass_gsf_gpu67_launch_readiness.py",
        "M_Tools/analysis/audit_bass_gsf_gpu67_waiter_liveness.py",
        "M_Tools/analysis/audit_bass_gsf_gpu01_live_queue.py",
        "M_Tools/analysis/audit_semantic_scale_source_package.py",
        "M_Tools/analysis/audit_semantic_scale_paper_readiness.py",
    ]
    assert all(row[0] == "/env/python" for row in plan)


def test_refresh_manifest_reports_gate_and_missing_rows(tmp_path, monkeypatch):
    mod = _load_module()
    summary = tmp_path / "summary.json"
    decision = tmp_path / "decision.json"
    closure = tmp_path / "closure.json"
    confirmation = tmp_path / "confirmation.json"
    p2_queue = tmp_path / "p2_queue.json"
    p3a_summary = tmp_path / "p3a.json"
    p3d_summary = tmp_path / "p3d.json"
    snippets = tmp_path / "snippets.json"
    gate = tmp_path / "gate.json"
    gate_8of10 = tmp_path / "gate_8of10.json"
    source_package = tmp_path / "source_package.json"
    gpu_wait = tmp_path / "gpu_wait.json"
    launch_readiness = tmp_path / "launch_readiness.json"
    waiter_liveness = tmp_path / "waiter_liveness.json"
    gpu01_live_queue = tmp_path / "gpu01_live_queue.json"
    summary.write_text(json.dumps({
        "status": "waiting/missing",
        "rows": [
            {"method": "RankDelta min-score 0.30", "kind": "rankdelta_p0",
             "mAP": None, "gate": "waiting/missing"},
            {"method": "RankDelta min-score 0.50", "kind": "rankdelta_p0",
             "mAP": None, "gate": "waiting/missing"},
        ],
    }), encoding="utf-8")
    decision.write_text(json.dumps({"action": "WAIT_P0"}), encoding="utf-8")
    closure.write_text(json.dumps({
        "action": "WAIT_P0",
        "paper_position": "no new method result may be claimed",
    }), encoding="utf-8")
    confirmation.write_text(json.dumps({
        "action": "WAIT_UPSTREAM",
    }), encoding="utf-8")
    p2_queue.write_text(json.dumps({
        "action": "WAIT_UPSTREAM",
    }), encoding="utf-8")
    p3a_summary.write_text(json.dumps({
        "status": "done",
        "main_method": "P3A z2 s0.1",
        "main_gate": "strict pass",
        "strict_pass": True,
    }), encoding="utf-8")
    p3d_summary.write_text(json.dumps({
        "status": "done",
        "dataset": "DIOR-R",
        "main_method": "P3D DIOR-R z2 s0.1",
        "main_gate": "strict transfer pass",
        "strict_transfer_pass": True,
    }), encoding="utf-8")
    snippets.write_text(json.dumps({
        "rankdelta_rows_filled": 0,
        "has_strict_pass": False,
    }), encoding="utf-8")
    gate.write_text(json.dumps({
        "all_95_gates_passed": False,
        "num_gates_passed": 3,
        "num_gates_total": 10,
    }), encoding="utf-8")
    gate_8of10.write_text(json.dumps({
        "current_gate": "3/10",
        "target_gate": "8/10",
        "additional_gates_needed": 5,
        "forbidden_route": "Do not use Gaussian as bbox IoU.",
    }), encoding="utf-8")
    gpu_wait.write_text(json.dumps({
        "action": "WAIT_GPU_BUSY",
        "external_or_unknown_process_count": 2,
    }), encoding="utf-8")
    launch_readiness.write_text(json.dumps({
        "action": "READY_WHEN_GPU_FREE",
        "launch_assets_ready": True,
        "blocker_count": 1,
        "output_slots_complete": 0,
        "output_slots_total": 6,
    }), encoding="utf-8")
    waiter_liveness.write_text(json.dumps({
        "action": "RESTART_MISSING_WAITERS",
        "active_waiter_count": 0,
        "waiter_count": 3,
        "stale_waiter_count": 0,
    }), encoding="utf-8")
    gpu01_live_queue.write_text(json.dumps({
        "action": "P0_RUNNING_GPU01_WAITERS_ACTIVE",
        "p0_running_count": 2,
        "p0_variant_count": 2,
        "p0_outputs_complete": 0,
        "p0_outputs_total": 2,
        "waiter_active_count": 2,
        "waiter_count": 2,
        "stale_or_missing_count": 0,
    }), encoding="utf-8")
    source_package.write_text(json.dumps({
        "source_package_ready": True,
        "pdf_build_ready": False,
    }), encoding="utf-8")

    monkeypatch.setattr(mod, "SUMMARY_JSON", summary)
    monkeypatch.setattr(mod, "DECISION_JSON", decision)
    monkeypatch.setattr(mod, "CLOSURE_JSON", closure)
    monkeypatch.setattr(mod, "CONFIRMATION_JSON", confirmation)
    monkeypatch.setattr(mod, "P2_QUEUE_JSON", p2_queue)
    monkeypatch.setattr(mod, "P3A_SUMMARY_JSON", p3a_summary)
    monkeypatch.setattr(mod, "P3D_SUMMARY_JSON", p3d_summary)
    monkeypatch.setattr(mod, "SNIPPET_JSON", snippets)
    monkeypatch.setattr(mod, "ICLR_GATE_JSON", gate)
    monkeypatch.setattr(mod, "ICLR_8OF10_PLAN_JSON", gate_8of10)
    monkeypatch.setattr(mod, "SOURCE_PACKAGE_JSON", source_package)
    monkeypatch.setattr(mod, "GPU_WAIT_AUDIT_JSON", gpu_wait)
    monkeypatch.setattr(mod, "LAUNCH_READINESS_JSON", launch_readiness)
    monkeypatch.setattr(mod, "WAITER_LIVENESS_JSON", waiter_liveness)
    monkeypatch.setattr(mod, "GPU01_LIVE_QUEUE_JSON", gpu01_live_queue)

    manifest = mod.build_manifest([
        {"name": "summarize_rankdelta", "returncode": 0},
        {"name": "build_iclr95_gate", "returncode": 0},
    ])

    assert manifest["ok"] is True
    assert manifest["active_route"] == "gpu01"
    assert manifest["summary_status"] == "waiting/missing"
    assert manifest["followup_action"] == "WAIT_P0"
    assert manifest["closure_action"] == "WAIT_P0"
    assert manifest["confirmation_action"] == "WAIT_UPSTREAM"
    assert manifest["p2_queue_action"] == "WAIT_UPSTREAM"
    assert manifest["p3a_status"] == "done"
    assert manifest["p3a_main_method"] == "P3A z2 s0.1"
    assert manifest["p3a_main_gate"] == "strict pass"
    assert manifest["p3a_strict_pass"] is True
    assert manifest["p3d_status"] == "done"
    assert manifest["p3d_dataset"] == "DIOR-R"
    assert manifest["p3d_main_method"] == "P3D DIOR-R z2 s0.1"
    assert manifest["p3d_main_gate"] == "strict transfer pass"
    assert manifest["p3d_strict_transfer_pass"] is True
    assert manifest["rankdelta_rows_total"] == 2
    assert manifest["rankdelta_rows_filled"] == 0
    assert manifest["iclr95_num_passed"] == 3
    assert manifest["iclr95_8of10_current"] == "3/10"
    assert manifest["iclr95_8of10_target"] == "8/10"
    assert manifest["iclr95_8of10_additional_needed"] == 5
    assert manifest["iclr95_8of10_forbidden_route"] == (
        "Do not use Gaussian as bbox IoU.")
    assert manifest["source_package_ready"] is True
    assert manifest["pdf_build_ready"] is False
    assert manifest["gpu67_wait_action"] == "HISTORICAL_INACTIVE_GPU01_ACTIVE"
    assert manifest["gpu67_raw_wait_action"] == "WAIT_GPU_BUSY"
    assert manifest["gpu67_external_or_unknown_process_count"] == 2
    assert manifest["gpu67_launch_action"] == "HISTORICAL_INACTIVE_GPU01_ACTIVE"
    assert manifest["gpu67_raw_launch_action"] == "READY_WHEN_GPU_FREE"
    assert manifest["gpu67_launch_assets_ready"] is True
    assert manifest["gpu67_launch_blocker_count"] == 1
    assert manifest["gpu67_launch_output_slots_complete"] == 0
    assert manifest["gpu67_launch_output_slots_total"] == 6
    assert manifest["gpu67_waiters_action"] == "HISTORICAL_INACTIVE_GPU01_ACTIVE"
    assert manifest["gpu67_raw_waiters_action"] == "RESTART_MISSING_WAITERS"
    assert manifest["gpu67_active_waiter_count"] == 0
    assert manifest["gpu67_waiter_count"] == 3
    assert manifest["gpu67_stale_waiter_count"] == 0
    assert manifest["gpu01_queue_action"] == "P0_RUNNING_GPU01_WAITERS_ACTIVE"
    assert manifest["gpu01_p0_running_count"] == 2
    assert manifest["gpu01_p0_variant_count"] == 2
    assert manifest["gpu01_p0_outputs_complete"] == 0
    assert manifest["gpu01_p0_outputs_total"] == 2
    assert manifest["gpu01_waiter_active_count"] == 2
    assert manifest["gpu01_waiter_count"] == 2
    assert manifest["gpu01_stale_or_missing_count"] == 0


def test_refresh_manifest_preserves_gpu67_when_gpu01_not_active(tmp_path,
                                                                monkeypatch):
    mod = _load_module()
    summary = tmp_path / "summary.json"
    decision = tmp_path / "decision.json"
    closure = tmp_path / "closure.json"
    confirmation = tmp_path / "confirmation.json"
    p2_queue = tmp_path / "p2_queue.json"
    p3a_summary = tmp_path / "p3a.json"
    p3d_summary = tmp_path / "p3d.json"
    snippets = tmp_path / "snippets.json"
    gate = tmp_path / "gate.json"
    gate_8of10 = tmp_path / "gate_8of10.json"
    source_package = tmp_path / "source_package.json"
    gpu_wait = tmp_path / "gpu_wait.json"
    launch_readiness = tmp_path / "launch_readiness.json"
    waiter_liveness = tmp_path / "waiter_liveness.json"
    gpu01_live_queue = tmp_path / "gpu01_live_queue.json"
    summary.write_text(json.dumps({
        "status": "waiting/missing",
        "rows": [],
    }), encoding="utf-8")
    decision.write_text(json.dumps({"action": "WAIT_P0"}), encoding="utf-8")
    closure.write_text(json.dumps({
        "action": "WAIT_P0",
        "paper_position": "no new method result may be claimed",
    }), encoding="utf-8")
    confirmation.write_text(json.dumps({"action": "WAIT_UPSTREAM"}),
                            encoding="utf-8")
    p2_queue.write_text(json.dumps({"action": "WAIT_UPSTREAM"}),
                        encoding="utf-8")
    p3a_summary.write_text(json.dumps({
        "status": "missing",
        "main_gate": "missing",
        "strict_pass": False,
    }), encoding="utf-8")
    p3d_summary.write_text(json.dumps({
        "status": "missing",
        "main_gate": "missing",
        "strict_transfer_pass": False,
    }), encoding="utf-8")
    snippets.write_text(json.dumps({
        "rankdelta_rows_filled": 0,
        "has_strict_pass": False,
    }), encoding="utf-8")
    gate.write_text(json.dumps({
        "all_95_gates_passed": False,
        "num_gates_passed": 3,
        "num_gates_total": 10,
    }), encoding="utf-8")
    gate_8of10.write_text(json.dumps({
        "current_gate": "3/10",
        "target_gate": "8/10",
        "additional_gates_needed": 5,
        "forbidden_route": "Do not use Gaussian as bbox IoU.",
    }), encoding="utf-8")
    gpu_wait.write_text(json.dumps({
        "action": "WAIT_GPU_BUSY",
        "external_or_unknown_process_count": 2,
    }), encoding="utf-8")
    launch_readiness.write_text(json.dumps({
        "action": "READY_WHEN_GPU_FREE",
        "launch_assets_ready": True,
        "blocker_count": 1,
        "output_slots_complete": 0,
        "output_slots_total": 6,
    }), encoding="utf-8")
    waiter_liveness.write_text(json.dumps({
        "action": "RESTART_MISSING_WAITERS",
        "active_waiter_count": 0,
        "waiter_count": 3,
        "stale_waiter_count": 0,
    }), encoding="utf-8")
    gpu01_live_queue.write_text(json.dumps({
        "action": "WAIT_P0",
        "p0_running_count": 0,
        "p0_variant_count": 2,
        "p0_outputs_complete": 0,
        "p0_outputs_total": 2,
        "waiter_active_count": 0,
        "waiter_count": 2,
        "stale_or_missing_count": 2,
    }), encoding="utf-8")
    source_package.write_text(json.dumps({
        "source_package_ready": True,
        "pdf_build_ready": False,
    }), encoding="utf-8")

    monkeypatch.setattr(mod, "SUMMARY_JSON", summary)
    monkeypatch.setattr(mod, "DECISION_JSON", decision)
    monkeypatch.setattr(mod, "CLOSURE_JSON", closure)
    monkeypatch.setattr(mod, "CONFIRMATION_JSON", confirmation)
    monkeypatch.setattr(mod, "P2_QUEUE_JSON", p2_queue)
    monkeypatch.setattr(mod, "P3A_SUMMARY_JSON", p3a_summary)
    monkeypatch.setattr(mod, "P3D_SUMMARY_JSON", p3d_summary)
    monkeypatch.setattr(mod, "SNIPPET_JSON", snippets)
    monkeypatch.setattr(mod, "ICLR_GATE_JSON", gate)
    monkeypatch.setattr(mod, "ICLR_8OF10_PLAN_JSON", gate_8of10)
    monkeypatch.setattr(mod, "SOURCE_PACKAGE_JSON", source_package)
    monkeypatch.setattr(mod, "GPU_WAIT_AUDIT_JSON", gpu_wait)
    monkeypatch.setattr(mod, "LAUNCH_READINESS_JSON", launch_readiness)
    monkeypatch.setattr(mod, "WAITER_LIVENESS_JSON", waiter_liveness)
    monkeypatch.setattr(mod, "GPU01_LIVE_QUEUE_JSON", gpu01_live_queue)

    manifest = mod.build_manifest([
        {"name": "summarize_rankdelta", "returncode": 0},
    ])

    assert manifest["active_route"] == "gpu67_or_unresolved"
    assert manifest["gpu67_wait_action"] == "WAIT_GPU_BUSY"
    assert manifest["gpu67_raw_wait_action"] == "WAIT_GPU_BUSY"
    assert manifest["gpu67_launch_action"] == "READY_WHEN_GPU_FREE"
    assert manifest["gpu67_raw_launch_action"] == "READY_WHEN_GPU_FREE"
    assert manifest["gpu67_waiters_action"] == "RESTART_MISSING_WAITERS"
    assert manifest["gpu67_raw_waiters_action"] == "RESTART_MISSING_WAITERS"
