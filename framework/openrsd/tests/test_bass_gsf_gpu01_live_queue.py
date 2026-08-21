import importlib.util
from pathlib import Path


def _load_module():
    path = Path("M_Tools/analysis/audit_bass_gsf_gpu01_live_queue.py")
    spec = importlib.util.spec_from_file_location(
        "audit_bass_gsf_gpu01_live_queue", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpu01_live_queue_reports_running_p0_and_waiters(tmp_path, monkeypatch):
    mod = _load_module()
    log0 = tmp_path / "gpu0_train.log"
    log1 = tmp_path / "gpu1_train.log"
    waiter0 = tmp_path / "p1.log"
    waiter1 = tmp_path / "p2.log"
    for path in (log0, log1, waiter0, waiter1):
        path.write_text("tick\n", encoding="utf-8")

    monkeypatch.setattr(mod, "P0_VARIANTS", [
        {"variant": "a", "gpu": 0, "exp_id": "exp_a", "session": "s0",
         "train_log": str(log0)},
        {"variant": "b", "gpu": 1, "exp_id": "exp_b", "session": "s1",
         "train_log": str(log1)},
    ])
    monkeypatch.setattr(mod, "WAITERS", [
        {"name": "P1", "session": "w0", "log": str(waiter0), "role": "r0"},
        {"name": "P2", "session": "w1", "log": str(waiter1), "role": "r1"},
    ])
    monkeypatch.setattr(mod, "query_tmux_sessions",
                        lambda: ({"s0", "s1", "w0", "w1"}, True, ""))
    monkeypatch.setattr(mod.time, "time", lambda: log0.stat().st_mtime + 10)
    monkeypatch.setattr(mod, "p0_output_paths", lambda exp_id: {
        "checkpoint": tmp_path / exp_id / "epoch_2.pth",
        "predictions": tmp_path / exp_id / "predictions.pkl",
        "risk_summary": tmp_path / exp_id / "deployment_risk_summary.json",
    })

    audit = mod.build_audit(max_log_age_seconds=60)

    assert audit["action"] == "P0_RUNNING_GPU01_WAITERS_ACTIVE"
    assert audit["p0_running_count"] == 2
    assert audit["p0_outputs_complete"] == 0
    assert audit["waiter_active_count"] == 2
    assert audit["stale_or_missing_count"] == 0


def test_gpu01_live_queue_reports_complete_outputs(tmp_path, monkeypatch):
    mod = _load_module()
    log0 = tmp_path / "gpu0_train.log"
    log0.write_text("old\n", encoding="utf-8")
    waiter0 = tmp_path / "p1.log"
    waiter0.write_text("tick\n", encoding="utf-8")
    out_dir = tmp_path / "exp_a"
    out_dir.mkdir()
    for name in ("epoch_2.pth", "predictions.pkl",
                 "deployment_risk_summary.json"):
        (out_dir / name).write_text("x\n", encoding="utf-8")

    monkeypatch.setattr(mod, "P0_VARIANTS", [
        {"variant": "a", "gpu": 0, "exp_id": "exp_a", "session": "s0",
         "train_log": str(log0)},
    ])
    monkeypatch.setattr(mod, "WAITERS", [
        {"name": "P1", "session": "w0", "log": str(waiter0), "role": "r0"},
    ])
    monkeypatch.setattr(mod, "query_tmux_sessions",
                        lambda: ({"w0"}, True, ""))
    monkeypatch.setattr(mod.time, "time", lambda: waiter0.stat().st_mtime + 10)
    monkeypatch.setattr(mod, "p0_output_paths", lambda exp_id: {
        "checkpoint": out_dir / "epoch_2.pth",
        "predictions": out_dir / "predictions.pkl",
        "risk_summary": out_dir / "deployment_risk_summary.json",
    })

    audit = mod.build_audit(max_log_age_seconds=60)

    assert audit["action"] == "P0_OUTPUTS_COMPLETE_WAITERS_ACTIVE"
    assert audit["p0_outputs_complete"] == 1
    assert audit["p0_running_count"] == 0
    assert audit["waiter_active_count"] == 1


def test_gpu01_live_queue_flags_missing_or_stale_incomplete(tmp_path,
                                                            monkeypatch):
    mod = _load_module()
    log0 = tmp_path / "gpu0_train.log"
    log0.write_text("old\n", encoding="utf-8")

    monkeypatch.setattr(mod, "P0_VARIANTS", [
        {"variant": "a", "gpu": 0, "exp_id": "exp_a", "session": "s0",
         "train_log": str(log0)},
    ])
    monkeypatch.setattr(mod, "WAITERS", [])
    monkeypatch.setattr(mod, "query_tmux_sessions", lambda: (set(), True, ""))
    monkeypatch.setattr(mod.time, "time", lambda: log0.stat().st_mtime + 600)
    monkeypatch.setattr(mod, "p0_output_paths", lambda exp_id: {
        "checkpoint": tmp_path / exp_id / "epoch_2.pth",
        "predictions": tmp_path / exp_id / "predictions.pkl",
        "risk_summary": tmp_path / exp_id / "deployment_risk_summary.json",
    })

    audit = mod.build_audit(max_log_age_seconds=60)

    assert audit["action"] == "CHECK_GPU01_QUEUE"
    assert audit["p0_rows"][0]["status"] == "missing_or_stale_incomplete"
    assert audit["stale_or_missing_count"] == 1
