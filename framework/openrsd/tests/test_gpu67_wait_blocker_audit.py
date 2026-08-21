import importlib.util
from pathlib import Path


def _load_module():
    path = Path("M_Tools/analysis/audit_gpu67_wait_blocker.py")
    spec = importlib.util.spec_from_file_location(
        "audit_gpu67_wait_blocker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpu_wait_blocker_reports_external_busy_process(monkeypatch):
    mod = _load_module()

    def fake_run_text(command):
        joined = " ".join(command)
        if "--query-gpu=index,pci.bus_id" in joined:
            return (
                "6, 00000000:3F:00.0, 13799, 46068, 100\n"
                "7, 00000000:40:00.0, 16, 46068, 0\n")
        if "--query-compute-apps=gpu_bus_id" in joined:
            return "00000000:3F:00.0, 88347, python, 13772\n"
        if command[:2] == ["ps", "-p"]:
            return "88347 87589 lwj 8000 python train_dino.py\n"
        return ""

    monkeypatch.setattr(mod, "run_text", fake_run_text)

    snapshot = mod.build_snapshot(
        gpu_ids=[6, 7],
        max_memory_mib=1024,
        max_util_pct=20,
        expected_keywords=["rankdelta"],
    )

    assert snapshot["action"] == "WAIT_GPU_BUSY"
    assert snapshot["external_or_unknown_process_count"] == 1
    assert snapshot["gpus"][0]["free_for_queue"] is False
    assert snapshot["gpus"][0]["processes"][0]["user"] == "lwj"
    assert snapshot["gpus"][0]["processes"][0]["classification"] == (
        "external_or_unrelated_process")


def test_gpu_wait_blocker_reports_ready_when_both_gpus_idle(monkeypatch):
    mod = _load_module()

    def fake_run_text(command):
        joined = " ".join(command)
        if "--query-gpu=index,pci.bus_id" in joined:
            return (
                "6, 00000000:3F:00.0, 16, 46068, 0\n"
                "7, 00000000:40:00.0, 16, 46068, 0\n")
        return ""

    monkeypatch.setattr(mod, "run_text", fake_run_text)

    snapshot = mod.build_snapshot(
        gpu_ids=[6, 7],
        max_memory_mib=1024,
        max_util_pct=20,
        expected_keywords=["rankdelta"],
    )

    assert snapshot["action"] == "READY_TO_LAUNCH"
    assert all(gpu["free_for_queue"] for gpu in snapshot["gpus"])
