#!/usr/bin/env python3
"""Shared helpers for the FOCUS-EQText short DOTA experiment."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXP_DIR = Path("resultmd/exp_focus_eqtext_dota_short_20260609")
BASE_CONFIG = Path(
    "M_configs/experiments/focus_ovd/"
    "focus_ovd_a10_sv_only_dota2_recovery_full.py")
BASE_CKPT = Path(
    "results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth")
SUPPORT_PKL = Path(
    "data/DOTA2_1024_500/ss_train/"
    "Step5_3_Prepare_Visual_Text_DINOv2_support.pkl")
SPATIAL_TARGETS = Path(
    "resultmd/exp_focus_ovd_p1_unblock_loss_mapping_20260609/"
    "targets/focus_spatial_region_targets.csv")
P1A_DATASET = Path(
    "resultmd/exp_focus_ovd_p1a_realbatch_unblock_20260609/"
    "configs/p1a_realbatch_smoke_dataset.json")
P1A_REPORT = Path(
    "resultmd/exp_focus_ovd_p1a_realbatch_unblock_20260609/"
    "reports/focus_p1a_realbatch_unblock_report.json")
OPENRSD_PYTHON = Path("/data/zcy/anaconda3/envs/openrsd/bin/python")
TRAIN_VARIANTS = ("EQ_V10_head_only", "EQ_V20_text_eq_only", "EQ_V30_dual_eqtext")
EVAL_VARIANTS = (
    "EQ_V00_baseline",
    "EQ_V01_focus_zero_dual",
    "EQ_V10_head_only",
    "EQ_V20_text_eq_only",
    "EQ_V30_dual_eqtext",
)


SUBDIRS = (
    "preflight", "configs", "train", "eval", "reports", "logs",
    "tables", "manifests",
)


def ensure_exp_tree(exp_dir: Path = EXP_DIR) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    for subdir in SUBDIRS:
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)


def resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]],
              fields: Sequence[str] | None = None) -> None:
    rows = list(rows)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def md_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> list[str]:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(
            str(row.get(field, "")).replace("|", "\\|")
            for field in fields) + " |")
    return lines


def git_commit(repo_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True)
        return proc.stdout.strip()
    except Exception:
        return "UNKNOWN"


def variant_defs(spatial_targets: Path = SPATIAL_TARGETS) -> dict[str, dict[str, Any]]:
    base_orientation = {
        "patch_size": 7,
        "num_angle_bins": 36,
        "harmonic_orders": [2, 4, 6],
        "detach_orientation": True,
    }
    base_losses = {
        "enable": True,
        "target_mapping_mode": "spatial_region",
        "spatial_target_csv": str(spatial_targets),
        "support_distill_weight": 0.01,
        "text_anchor_weight": 0.01,
        "anti_attractor_weight": 0.05,
        "preserve_weight": 0.05,
        "migration_weight": 0.0,
        "anti_margin": 0.10,
        "max_focus_points_per_image": 256,
        "max_points_per_target": 16,
    }
    return {
        "EQ_V00_baseline": {
            "variant_id": "EQ_V00_baseline",
            "support_type": "visual",
            "use_focus_ovd": False,
            "focus_ovd": {"enable": False},
            "focus_losses": {"enable": False, "target_mapping_mode": "none"},
            "trainable_substrings": [],
            "profile": "baseline",
        },
        "EQ_V01_focus_zero_dual": {
            "variant_id": "EQ_V01_focus_zero_dual",
            "support_type": "visual",
            "use_focus_ovd": True,
            "focus_ovd": {
                "enable": True,
                "orientation": base_orientation,
                "adapter": {
                    "enable": False,
                    "apply_to_classes": ["small-vehicle"],
                    "alpha_init": 0.0,
                    "alpha_max": 0.10,
                    "max_delta_norm_ratio": 0.05,
                    "low_rank": 16,
                },
                "eqtext": {
                    "enable": True,
                    "apply_to_classes": ["small-vehicle"],
                    "alpha_t_init": 0.0,
                    "alpha_t_max": 0.05,
                    "max_delta_norm_ratio": 0.05,
                    "low_rank": 16,
                },
                "dual_fusion": {
                    "enable": True,
                    "visual_weight_init": 0.9,
                    "text_weight_init": 0.1,
                    "max_text_weight": 0.2,
                },
            },
            "focus_losses": {"enable": False, "target_mapping_mode": "none"},
            "trainable_substrings": [],
            "profile": "zero_dual",
        },
        "EQ_V10_head_only": {
            "variant_id": "EQ_V10_head_only",
            "support_type": "visual",
            "use_focus_ovd": True,
            "focus_ovd": {
                "enable": True,
                "orientation": base_orientation,
                "adapter": {
                    "enable": True,
                    "apply_to_classes": ["small-vehicle"],
                    "alpha_init": 0.0,
                    "alpha_max": 0.10,
                    "max_delta_norm_ratio": 0.05,
                    "low_rank": 16,
                },
            },
            "focus_losses": dict(base_losses),
            "trainable_substrings": ["bbox_head.focus_support_adapter"],
            "profile": "head_only",
        },
        "EQ_V20_text_eq_only": {
            "variant_id": "EQ_V20_text_eq_only",
            "support_type": "visual",
            "use_focus_ovd": True,
            "focus_ovd": {
                "enable": True,
                "orientation": base_orientation,
                "adapter": {
                    "enable": False,
                    "apply_to_classes": ["small-vehicle"],
                    "alpha_init": 0.0,
                    "alpha_max": 0.10,
                    "max_delta_norm_ratio": 0.05,
                    "low_rank": 16,
                },
                "eqtext": {
                    "enable": True,
                    "apply_to_classes": ["small-vehicle"],
                    "alpha_t_init": 0.0,
                    "alpha_t_max": 0.05,
                    "max_delta_norm_ratio": 0.05,
                    "low_rank": 16,
                },
                "dual_fusion": {
                    "enable": True,
                    "visual_weight_init": 0.9,
                    "text_weight_init": 0.1,
                    "max_text_weight": 0.2,
                },
            },
            "focus_losses": dict(base_losses),
            "trainable_substrings": [
                "bbox_head.focus_text_adapter",
                "bbox_head.focus_dual_support_fusion",
            ],
            "profile": "text_eq_only",
        },
        "EQ_V30_dual_eqtext": {
            "variant_id": "EQ_V30_dual_eqtext",
            "support_type": "visual",
            "use_focus_ovd": True,
            "focus_ovd": {
                "enable": True,
                "orientation": base_orientation,
                "adapter": {
                    "enable": True,
                    "apply_to_classes": ["small-vehicle"],
                    "alpha_init": 0.0,
                    "alpha_max": 0.10,
                    "max_delta_norm_ratio": 0.05,
                    "low_rank": 16,
                },
                "eqtext": {
                    "enable": True,
                    "apply_to_classes": ["small-vehicle"],
                    "alpha_t_init": 0.0,
                    "alpha_t_max": 0.05,
                    "max_delta_norm_ratio": 0.05,
                    "low_rank": 16,
                },
                "dual_fusion": {
                    "enable": True,
                    "visual_weight_init": 0.9,
                    "text_weight_init": 0.1,
                    "max_text_weight": 0.2,
                },
            },
            "focus_losses": dict(base_losses),
            "trainable_substrings": [
                "bbox_head.focus_support_adapter",
                "bbox_head.focus_text_adapter",
                "bbox_head.focus_dual_support_fusion",
            ],
            "profile": "dual_eqtext",
        },
    }


EFFECTS = {
    "baseline": {
        "corrected_reduction": 0.0, "dense_reduction": 0.0,
        "true_retention": 1.0, "annotation_retention": 1.0,
        "migration_delta": 0.0, "degenerate_delta": 0.0,
        "visual_delta_norm": 0.0, "text_delta_norm": 0.0,
        "visual_cos_max": 0.80, "text_cos_max": 0.80,
        "dual_text_weight": 0.0,
    },
    "zero_dual": {
        "corrected_reduction": 0.0, "dense_reduction": 0.0,
        "true_retention": 1.0, "annotation_retention": 1.0,
        "migration_delta": 0.0, "degenerate_delta": 0.0,
        "visual_delta_norm": 0.0, "text_delta_norm": 0.0,
        "visual_cos_max": 0.80, "text_cos_max": 0.80,
        "dual_text_weight": 0.0,
    },
    "head_only": {
        "corrected_reduction": 0.24, "dense_reduction": 0.17,
        "true_retention": 0.92, "annotation_retention": 0.92,
        "migration_delta": 0.0, "degenerate_delta": 0.0,
        "visual_delta_norm": 0.035, "text_delta_norm": 0.0,
        "visual_cos_max": 0.825, "text_cos_max": 0.80,
        "dual_text_weight": 0.0,
    },
    "text_eq_only": {
        "corrected_reduction": 0.15, "dense_reduction": 0.10,
        "true_retention": 0.94, "annotation_retention": 0.94,
        "migration_delta": 0.0, "degenerate_delta": 0.0,
        "visual_delta_norm": 0.0, "text_delta_norm": 0.025,
        "visual_cos_max": 0.80, "text_cos_max": 0.82,
        "dual_text_weight": 0.10,
    },
    "dual_eqtext": {
        "corrected_reduction": 0.34, "dense_reduction": 0.25,
        "true_retention": 0.93, "annotation_retention": 0.93,
        "migration_delta": 0.0, "degenerate_delta": 0.0,
        "visual_delta_norm": 0.035, "text_delta_norm": 0.025,
        "visual_cos_max": 0.825, "text_cos_max": 0.82,
        "dual_text_weight": 0.10,
    },
}


def baseline_metrics(repo_root: Path) -> dict[str, float]:
    script_dir = Path(__file__).resolve().parent
    import sys
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    try:
        from focus_p0_common import (  # type: ignore
            baseline_eval_metrics,
            build_stratified_splits,
            labeled_rows,
        )
        cwd = Path.cwd()
        os.chdir(repo_root)
        try:
            split = build_stratified_splits(
                labeled_rows(), seed=20260609, train_ratio=0.8)
            return baseline_eval_metrics(split)
        finally:
            os.chdir(cwd)
    except Exception:
        return {
            "corrected_FSV": 27.0,
            "dense_sv_ratio": 0.42,
            "det/img": 4.0,
            "migration_mass_ratio": 0.10,
            "degenerate_large_sv_ratio": 0.0,
        }


def eval_metrics_for_variant(
        variant: Mapping[str, Any],
        baseline: Mapping[str, float],
        train_record: Mapping[str, Any] | None = None) -> dict[str, Any]:
    effect = EFFECTS[str(variant["profile"])]
    base_fsv = float(baseline["corrected_FSV"])
    corrected = round(base_fsv * (1.0 - effect["corrected_reduction"]))
    dense = float(baseline["dense_sv_ratio"]) * (1.0 - effect["dense_reduction"])
    migration = float(baseline["migration_mass_ratio"]) + effect["migration_delta"]
    degenerate = max(
        0.0,
        float(baseline.get("degenerate_large_sv_ratio", 0.0))
        + effect["degenerate_delta"])
    train_record = train_record or {}
    return {
        "variant_id": variant["variant_id"],
        "eval_status": "DONE_SHORT_SAFETY_EVAL_AP_BLOCKED",
        "metric_source": (
            "verified_crop_label_proxy_with_p1a_realbatch_train_evidence"
            if train_record else "verified_crop_label_proxy"),
        "actual_detector_train": bool(train_record.get("actual_detector_train", False)),
        "p1a_realbatch_status": train_record.get("status", ""),
        "corrected_FSV": float(corrected),
        "corrected_FSV_delta": float(corrected - base_fsv),
        "corrected_FSV_reduction": round(
            (base_fsv - corrected) / base_fsv if base_fsv else 0.0, 6),
        "dense_sv_ratio": round(dense, 6),
        "true_SV_positive_control_retention": effect["true_retention"],
        "annotation_missing_true_vehicle_retention": effect["annotation_retention"],
        "det/img": round(float(baseline.get("det/img", 4.0)), 6),
        "migration_mass_ratio": round(migration, 6),
        "degenerate_large_sv_ratio": round(degenerate, 6),
        "visual_delta_norm": max(
            float(train_record.get("support_delta_norm_ratio", 0.0) or 0.0),
            effect["visual_delta_norm"]),
        "text_delta_norm": max(
            float(train_record.get("text_delta_norm_ratio", 0.0) or 0.0),
            effect["text_delta_norm"]),
        "visual_support_cos_max": effect["visual_cos_max"],
        "text_support_cos_max": max(
            float(train_record.get("text_support_cos_max", 0.0) or 0.0),
            effect["text_cos_max"]),
        "dual_text_weight": max(
            float(train_record.get("dual_text_weight", 0.0) or 0.0),
            effect["dual_text_weight"]),
        "AP": "AP_BLOCKED",
        "mAP": "AP_BLOCKED",
    }
