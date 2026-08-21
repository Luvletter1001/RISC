import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path("M_Tools/analysis/audit_bass_gsf_gpu67_launch_readiness.py")
    spec = importlib.util.spec_from_file_location(
        "audit_bass_gsf_gpu67_launch_readiness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_output_slot_status_requires_eval_and_risk(tmp_path):
    mod = _load_module()
    eval_dir = tmp_path / "eval"
    risk_json = tmp_path / "risk" / "deployment_risk_summary.json"
    eval_dir.mkdir()

    assert mod.output_slot_status({
        "name": "slot",
        "eval_json_dir": eval_dir,
        "risk_json": risk_json,
    })["status"] == "waiting"

    (eval_dir / "metrics.json").write_text(
        json.dumps({"mAP": 0.5}), encoding="utf-8")
    assert mod.output_slot_status({
        "name": "slot",
        "eval_json_dir": eval_dir,
        "risk_json": risk_json,
    })["status"] == "missing_risk"

    risk_json.parent.mkdir()
    risk_json.write_text(json.dumps({"summaries": []}), encoding="utf-8")
    assert mod.output_slot_status({
        "name": "slot",
        "eval_json_dir": eval_dir,
        "risk_json": risk_json,
    })["status"] == "complete"


def test_launch_readiness_waits_only_for_gpu_when_assets_ready(tmp_path,
                                                              monkeypatch):
    mod = _load_module()
    script = tmp_path / "run.sh"
    script.write_text("#!/usr/bin/env bash\ntrue\n", encoding="utf-8")
    config = tmp_path / "config.py"
    config.write_text("data_root = '/data1/zcy/datasets/HRRSD/'\n",
                      encoding="utf-8")
    gpu_wait = tmp_path / "gpu_wait.json"
    gpu_wait.write_text(json.dumps({"action": "WAIT_GPU_BUSY"}),
                        encoding="utf-8")
    eval_dir = tmp_path / "future_eval"
    risk_json = tmp_path / "future_risk" / "deployment_risk_summary.json"

    monkeypatch.setattr(mod, "SCRIPT_PATHS", [script])
    monkeypatch.setattr(mod, "CONFIG_PATHS", [config])
    monkeypatch.setattr(mod, "REQUIRED_INPUTS", [{
        "name": "input",
        "path": script,
        "reason": "test input",
    }])
    monkeypatch.setattr(mod, "CONTROL_RISK_INPUTS", [{
        "name": "control",
        "path": config,
    }])
    monkeypatch.setattr(mod, "EXPECTED_OUTPUTS", [{
        "name": "future slot",
        "eval_json_dir": eval_dir,
        "risk_json": risk_json,
    }])
    monkeypatch.setattr(mod, "GPU_WAIT_AUDIT_JSON", gpu_wait)

    audit = mod.build_audit()

    assert audit["action"] == "READY_WHEN_GPU_FREE"
    assert audit["launch_assets_ready"] is True
    assert audit["blocker_count"] == 1
    assert audit["blockers"] == [
        "GPU6/GPU7 are still busy; waiters should continue polling"]
    assert audit["expected_output_slots"][0]["status"] == "waiting"


def test_launch_readiness_flags_repo_local_data_reference(tmp_path,
                                                          monkeypatch):
    mod = _load_module()
    script = tmp_path / "run.sh"
    script.write_text("#!/usr/bin/env bash\ntrue\n", encoding="utf-8")
    config = tmp_path / "bad_config.py"
    config.write_text("data_root = 'data/HRRSD_800_0/'\n", encoding="utf-8")
    gpu_wait = tmp_path / "gpu_wait.json"
    gpu_wait.write_text(json.dumps({"action": "READY_TO_LAUNCH"}),
                        encoding="utf-8")

    monkeypatch.setattr(mod, "SCRIPT_PATHS", [script])
    monkeypatch.setattr(mod, "CONFIG_PATHS", [config])
    monkeypatch.setattr(mod, "REQUIRED_INPUTS", [{
        "name": "input",
        "path": script,
        "reason": "test input",
    }])
    monkeypatch.setattr(mod, "CONTROL_RISK_INPUTS", [{
        "name": "control",
        "path": script,
    }])
    monkeypatch.setattr(mod, "EXPECTED_OUTPUTS", [])
    monkeypatch.setattr(mod, "GPU_WAIT_AUDIT_JSON", gpu_wait)

    audit = mod.build_audit()

    assert audit["action"] == "FIX_LAUNCH_BLOCKERS"
    assert audit["launch_assets_ready"] is False
    assert audit["repo_local_data_refs"][0]["text"] == (
        "data_root = 'data/HRRSD_800_0/'")
