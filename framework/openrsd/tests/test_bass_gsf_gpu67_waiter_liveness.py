import importlib.util
from pathlib import Path


def _load_module():
    path = Path("M_Tools/analysis/audit_bass_gsf_gpu67_waiter_liveness.py")
    spec = importlib.util.spec_from_file_location(
        "audit_bass_gsf_gpu67_waiter_liveness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_waiter_liveness_reports_all_active(tmp_path, monkeypatch):
    mod = _load_module()
    log_a = tmp_path / "a.log"
    log_b = tmp_path / "b.log"
    log_a.write_text("tick\n", encoding="utf-8")
    log_b.write_text("tick\n", encoding="utf-8")

    monkeypatch.setattr(mod, "WAITERS", [
        {"name": "A", "session": "a", "log": str(log_a), "role": "role a"},
        {"name": "B", "session": "b", "log": str(log_b), "role": "role b"},
    ])
    monkeypatch.setattr(mod, "query_tmux_sessions", lambda: ({"a", "b"}, True, ""))
    monkeypatch.setattr(mod.time, "time", lambda: log_a.stat().st_mtime + 10)

    audit = mod.build_audit(max_log_age_seconds=60)

    assert audit["action"] == "WAITERS_ACTIVE"
    assert audit["active_waiter_count"] == 2
    assert audit["missing_session_count"] == 0
    assert audit["stale_waiter_count"] == 0


def test_waiter_liveness_reports_stale_log(tmp_path, monkeypatch):
    mod = _load_module()
    log_a = tmp_path / "a.log"
    log_a.write_text("old\n", encoding="utf-8")

    monkeypatch.setattr(mod, "WAITERS", [
        {"name": "A", "session": "a", "log": str(log_a), "role": "role a"},
    ])
    monkeypatch.setattr(mod, "query_tmux_sessions", lambda: ({"a"}, True, ""))
    monkeypatch.setattr(mod.time, "time", lambda: log_a.stat().st_mtime + 600)

    audit = mod.build_audit(max_log_age_seconds=60)

    assert audit["action"] == "CHECK_STALE_WAITERS"
    assert audit["waiters"][0]["status"] == "stale_log"
    assert audit["stale_waiter_count"] == 1


def test_waiter_liveness_reports_missing_session(tmp_path, monkeypatch):
    mod = _load_module()
    log_a = tmp_path / "a.log"
    log_a.write_text("tick\n", encoding="utf-8")

    monkeypatch.setattr(mod, "WAITERS", [
        {"name": "A", "session": "a", "log": str(log_a), "role": "role a"},
    ])
    monkeypatch.setattr(mod, "query_tmux_sessions", lambda: (set(), True, ""))

    audit = mod.build_audit(max_log_age_seconds=60)

    assert audit["action"] == "RESTART_MISSING_WAITERS"
    assert audit["waiters"][0]["status"] == "missing_session"
    assert audit["missing_session_count"] == 1


def test_waiter_liveness_uses_fresh_logs_when_tmux_unavailable(tmp_path,
                                                               monkeypatch):
    mod = _load_module()
    log_a = tmp_path / "a.log"
    log_a.write_text("tick\n", encoding="utf-8")

    monkeypatch.setattr(mod, "WAITERS", [
        {"name": "A", "session": "a", "log": str(log_a), "role": "role a"},
    ])
    monkeypatch.setattr(
        mod,
        "query_tmux_sessions",
        lambda: (set(), False, "operation not permitted"))
    monkeypatch.setattr(mod.time, "time", lambda: log_a.stat().st_mtime + 10)

    audit = mod.build_audit(max_log_age_seconds=60)

    assert audit["action"] == "LOGS_ACTIVE_TMUX_UNVERIFIED"
    assert audit["tmux_query_ok"] is False
    assert audit["active_waiter_count"] == 1
    assert audit["waiters"][0]["status"] == "log_active_tmux_unverified"
