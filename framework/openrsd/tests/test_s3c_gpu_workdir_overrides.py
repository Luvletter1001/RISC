import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_reviewer_gate_audit_uses_env_pre_nms_work_dir(monkeypatch):
    monkeypatch.setenv(
        "S3C_PRE_NMS_WORK_DIR",
        "work_dirs/semantic_ambiguity_study_20260617/p4_s3c_pre_nms_z4_l0p25_gpu45",
    )
    monkeypatch.setenv("S3C_PRE_NMS_LABEL", "GPU45")

    audit = load_module(
        "s3c_audit_env_test",
        "M_Tools/analysis/build_s3c_reviewer_gate_audit.py",
    )

    assert audit.PRE_NMS_LABEL == "GPU45"
    assert audit.PATHS["pre_nms_predictions"] == (
        audit.PRE_NMS_WORK_DIR / "predictions.pkl"
    )
    assert "gpu45" in str(audit.PATHS["pre_nms_dense"])


def test_gpu_finalizer_uses_env_pre_nms_work_dir_and_label(monkeypatch):
    monkeypatch.setenv(
        "S3C_PRE_NMS_WORK_DIR",
        "work_dirs/semantic_ambiguity_study_20260617/p4_s3c_pre_nms_z4_l0p25_gpu45",
    )
    monkeypatch.setenv("S3C_PRE_NMS_LABEL", "GPU45")

    finalizer = load_module(
        "s3c_finalizer_env_test",
        "M_Tools/experiments/finalize_p4_s3c_gpu67_audit.py",
    )

    assert finalizer.PRE_NMS_LABEL == "GPU45"
    assert finalizer.WORK_DIR.name == "p4_s3c_pre_nms_z4_l0p25_gpu45"
    assert "GPU45" in finalizer.START_MARKER
    assert finalizer.REQUIRED["dense_topk"] == (
        finalizer.WORK_DIR / "dense_topk_summary.json"
    )
