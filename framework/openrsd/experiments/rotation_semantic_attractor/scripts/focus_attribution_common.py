#!/usr/bin/env python3
"""Common helpers for FOCUS-OVD module attribution scripts."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_DATE = os.environ.get("FOCUS_ATTRIBUTION_DATE", date.today().strftime("%Y%m%d"))
DEFAULT_EXP_DIR = Path(
    f"resultmd/exp_focus_ovd_module_attribution_{DEFAULT_DATE}")

REQUIRED_SUBDIRS = (
    "preflight",
    "module_inventory",
    "configs",
    "ablation_plans",
    "smoke",
    "orientation_probe",
    "support_geometry",
    "cluster_mining",
    "attribute_gate",
    "channel_mask",
    "safety_gate",
    "orbit_teacher",
    "head_consensus",
    "negative_prompt",
    "ccl_control",
    "eval",
    "tables",
    "figures",
    "reports",
)

DEFAULT_CORRECTED_FSV_CSV = Path(
    "experiments/rotation_semantic_attractor/reports/visual_summary/audit/"
    "human_sv_crop_audit_labels_VERIFIED_EXPANDED.csv")
DEFAULT_SUPPORT_PKL = Path(
    "data/DOTA2_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl")
DEFAULT_CONFIG = Path(
    "M_configs/experiments/focus_ovd/focus_ovd_a10_sv_only_dota2_recovery_full.py")
DEFAULT_CHECKPOINT = Path(
    "results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth")


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def load_focus_util(repo_root: Path, name: str):
    path = repo_root / "M_AD" / "models" / "utils" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ensure_exp_tree(exp_dir: Path) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_SUBDIRS:
        (exp_dir / name).mkdir(parents=True, exist_ok=True)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]],
                   fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def markdown_table(rows: Sequence[Mapping[str, Any]],
                   fieldnames: Sequence[str]) -> list[str]:
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in fieldnames) + " |")
    return lines


def git_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        return result.stdout.strip()
    except Exception:
        return "UNKNOWN"


def load_manifest(exp_dir: Path) -> dict[str, Any]:
    path = exp_dir / "manifest.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"records": [], "manifest_error": "invalid_existing_json"}
    return {"records": []}


def append_manifest_record(
        exp_dir: Path,
        *,
        repo_root: Path,
        stage: str,
        status: str,
        module_switches: Mapping[str, Any] | None = None,
        failure_reason: str = "",
        config_path: Path | str | None = None,
        checkpoint_path: Path | str | None = None,
        support_pkl_path: Path | str | None = None,
        corrected_fsv_csv_path: Path | str | None = None,
        split: str = "S3_verified_or_smoke",
        angles: str = "0,30,60,90,120,150,180,210,240,270,300,330",
        gpu: str = "not_requested",
        random_seed: int = 20260609) -> None:
    ensure_exp_tree(exp_dir)
    manifest = load_manifest(exp_dir)
    manifest.update({
        "git_commit": git_commit(repo_root),
        "experiment_dir": str(exp_dir),
        "default_config_path": str(config_path or DEFAULT_CONFIG),
        "default_checkpoint_path": str(checkpoint_path or DEFAULT_CHECKPOINT),
        "default_support_pkl_path": str(support_pkl_path or DEFAULT_SUPPORT_PKL),
        "default_corrected_fsv_csv_path": str(
            corrected_fsv_csv_path or DEFAULT_CORRECTED_FSV_CSV),
        "records_schema": [
            "stage", "status", "config_path", "checkpoint_path",
            "support_pkl_path", "corrected_fsv_csv_path", "split", "angles",
            "gpu", "random_seed", "module_switches", "failure_reason",
        ],
    })
    record = {
        "stage": stage,
        "status": status,
        "config_path": str(config_path or DEFAULT_CONFIG),
        "checkpoint_path": str(checkpoint_path or DEFAULT_CHECKPOINT),
        "support_pkl_path": str(support_pkl_path or DEFAULT_SUPPORT_PKL),
        "corrected_fsv_csv_path": str(corrected_fsv_csv_path or DEFAULT_CORRECTED_FSV_CSV),
        "split": split,
        "angles": angles,
        "gpu": gpu,
        "random_seed": random_seed,
        "module_switches": dict(module_switches or {}),
        "failure_reason": failure_reason,
    }
    manifest.setdefault("records", []).append(record)
    write_json(exp_dir / "manifest.json", manifest)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def write_simple_figure(png_path: Path, pdf_path: Path, title: str,
                        labels: Iterable[str] = (), values: Iterable[float] = ()) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = list(labels)
        values = list(values)
        fig, ax = plt.subplots(figsize=(7, 4))
        if labels and values:
            ax.bar(labels, values, color="#3366aa")
            ax.tick_params(axis="x", labelrotation=30)
        else:
            ax.text(0.5, 0.5, "No evaluated data", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(png_path)
        fig.savefig(pdf_path)
        plt.close(fig)
    except Exception:
        png_path.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
            b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
            b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")
        pdf_path.write_text(
            "%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n",
            encoding="utf-8")
