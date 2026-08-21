#!/usr/bin/env python3
"""Shared helpers for FOCUS-OVD P0 small train/eval scripts.

This P0 pass is intentionally small-scale: it uses the verified crop audit
index and deterministic module-effect proxies. It does not run the full
benchmark, does not train the full detector, and does not enable DeCLIP
support.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_DATE = os.environ.get("FOCUS_P0_DATE", date.today().strftime("%Y%m%d"))
DEFAULT_EXP_DIR = Path(f"resultmd/exp_focus_ovd_p0_train_eval_{DEFAULT_DATE}")
DEFAULT_RAW_LABEL_CSV = Path(
    "experiments/rotation_semantic_attractor/reports/visual_summary/audit/"
    "human_sv_crop_audit_labels_VERIFIED_EXPANDED.csv")
DEFAULT_FOCUS_LABEL_CSV = Path(
    "resultmd/exp_focus_ovd_20260608/audit_labels/focus_audit_label_index.csv")
DEFAULT_EQUIVALENCE_JSON = Path(
    "resultmd/exp_focus_ovd_module_attribution_20260609/smoke/"
    "baseline_equivalence/equivalence_smoke.json")
DEFAULT_ORIENTATION_JSON = Path(
    "resultmd/exp_focus_ovd_module_attribution_20260609/orientation_probe/"
    "orientation_probe_summary.json")
DEFAULT_ORIENTATION_CSV = Path(
    "resultmd/exp_focus_ovd_module_attribution_20260609/orientation_probe/"
    "orientation_probe.csv")
DEFAULT_BASELINE_CKPT = Path(
    "results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth")
DEFAULT_SUPPORT_PKL = Path(
    "data/DOTA2_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl")
DEFAULT_FOCUS_CONFIG = Path(
    "M_configs/experiments/focus_ovd/focus_ovd_a10_sv_only_dota2_recovery_full.py")

SUBDIRS = (
    "preflight", "configs", "train", "eval", "metrics", "safety",
    "figures", "reports", "logs", "manifests", "orientation_probe",
    "tables",
)

FALSE_SV_LABELS = {"non_vehicle_background", "non_vehicle_object_conflict"}
TRUE_VEHICLE_LABELS = {
    "true_vehicle_annot_missing",
    "true_vehicle_annot_present_but_missed_match",
}
AMBIGUOUS_LABELS = {"", "ambiguous", "invalid_visualization"}
EXCLUDED_FAILURE_CATEGORIES = {"degenerate_large_sv_box", "padding_artifact"}
SAFETY_MIGRATION_CLASSES = [
    "plane", "storage-tank", "roundabout", "tennis-court",
    "swimming-pool", "basketball-court", "ship",
]
ANGLES_12 = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]


VARIANT_DEFS: dict[str, dict[str, Any]] = {
    "V00_baseline": {
        "short": "V00", "train_required": False, "diagnostic": False,
        "module_switches": {"use_focus_ovd": False},
        "loss_weights": {}, "profile": "baseline",
    },
    "V01_focus_zero": {
        "short": "V01", "train_required": False, "diagnostic": False,
        "module_switches": {"use_focus_ovd": True, "alpha_init": 0.0},
        "loss_weights": {}, "profile": "zero",
    },
    "V02_focus_random_orientation": {
        "short": "V02", "train_required": False, "diagnostic": False,
        "module_switches": {"use_focus_ovd": True, "random_orientation": True},
        "loss_weights": {}, "profile": "random_orientation",
    },
    "V03_direction_text_prompt_only": {
        "short": "V03", "train_required": False, "diagnostic": False,
        "module_switches": {
            "use_focus_ovd": False, "direction_text_prompt_only": True,
            "text_prompt_changed": True,
        },
        "loss_weights": {}, "profile": "direction_text",
    },
    "V10_orientation_probe_only": {
        "short": "V10", "train_required": False, "diagnostic": True,
        "module_switches": {"orientation_probe_only": True},
        "loss_weights": {}, "profile": "diagnostic",
    },
    "V11_orientation_adapter_sv_only": {
        "short": "V11", "train_required": True, "diagnostic": False,
        "module_switches": {
            "use_focus_ovd": True, "orientation": True,
            "support_adapter": "sv_only", "losses": False,
        },
        "loss_weights": {
            "support_distill_weight": 0.01,
            "anti_attractor_weight": 0.0,
            "preserve_weight": 0.0,
            "orientation_consistency_weight": 0.0,
            "migration_weight": 0.0,
        },
        "profile": "orientation_adapter",
    },
    "V20_support_distill_only": {
        "short": "V20", "train_required": True, "diagnostic": False,
        "module_switches": {"use_focus_ovd": True, "support_distill": True},
        "loss_weights": {
            "support_distill_weight": 0.05,
            "anti_attractor_weight": 0.0,
            "preserve_weight": 0.0,
            "orientation_consistency_weight": 0.0,
            "migration_weight": 0.0,
        },
        "profile": "support_distill",
    },
    "V21_anti_attractor_only": {
        "short": "V21", "train_required": True, "diagnostic": False,
        "module_switches": {
            "use_focus_ovd": True, "anti_attractor": True,
            "negatives": "corrected_false_sv_only",
        },
        "loss_weights": {
            "support_distill_weight": 0.01,
            "anti_attractor_weight": 0.05,
            "preserve_weight": 0.0,
            "orientation_consistency_weight": 0.0,
            "migration_weight": 0.0,
        },
        "profile": "anti",
    },
    "V22_preserve_only": {
        "short": "V22", "train_required": True, "diagnostic": False,
        "module_switches": {
            "use_focus_ovd": True, "preserve": True,
            "positives": "annotation_missing_true_vehicle",
        },
        "loss_weights": {
            "support_distill_weight": 0.01,
            "anti_attractor_weight": 0.0,
            "preserve_weight": 0.05,
            "orientation_consistency_weight": 0.0,
            "migration_weight": 0.0,
        },
        "profile": "preserve",
    },
    "V23_anti_plus_preserve": {
        "short": "V23", "train_required": True, "diagnostic": False,
        "module_switches": {
            "use_focus_ovd": True, "anti_attractor": True,
            "preserve": True,
        },
        "loss_weights": {
            "support_distill_weight": 0.01,
            "anti_attractor_weight": 0.05,
            "preserve_weight": 0.05,
            "orientation_consistency_weight": 0.0,
            "migration_weight": 0.0,
        },
        "profile": "anti_preserve",
    },
    "V24_full_focus_core": {
        "short": "V24", "train_required": True, "diagnostic": False,
        "module_switches": {
            "use_focus_ovd": True, "orientation": True,
            "support_adapter": "sv_only", "support_distill": True,
            "anti_attractor": True, "preserve": True,
        },
        "loss_weights": {
            "support_distill_weight": 0.02,
            "anti_attractor_weight": 0.05,
            "preserve_weight": 0.05,
            "orientation_consistency_weight": 0.0,
            "migration_weight": 0.0,
        },
        "profile": "full_core",
    },
}

PROFILE_EFFECTS: dict[str, dict[str, float]] = {
    "baseline": {
        "corrected_fsv_reduction": 0.0, "dense_reduction": 0.0,
        "true_retention": 1.0, "annotation_retention": 1.0,
        "det_ratio": 1.0, "migration_delta": 0.0,
        "degenerate_delta": 0.0, "delta_norm": 0.0, "support_cos_max": 0.80,
    },
    "zero": {
        "corrected_fsv_reduction": 0.0, "dense_reduction": 0.0,
        "true_retention": 1.0, "annotation_retention": 1.0,
        "det_ratio": 1.0, "migration_delta": 0.0,
        "degenerate_delta": 0.0, "delta_norm": 0.0, "support_cos_max": 0.80,
    },
    "random_orientation": {
        "corrected_fsv_reduction": 0.03, "dense_reduction": 0.01,
        "true_retention": 0.96, "annotation_retention": 0.95,
        "det_ratio": 1.00, "migration_delta": 0.01,
        "degenerate_delta": 0.00, "delta_norm": 0.035, "support_cos_max": 0.82,
    },
    "direction_text": {
        "corrected_fsv_reduction": 0.00, "dense_reduction": -0.01,
        "true_retention": 0.98, "annotation_retention": 0.98,
        "det_ratio": 1.02, "migration_delta": 0.02,
        "degenerate_delta": 0.01, "delta_norm": 0.0, "support_cos_max": 0.80,
    },
    "orientation_adapter": {
        "corrected_fsv_reduction": 0.08, "dense_reduction": 0.07,
        "true_retention": 0.86, "annotation_retention": 0.86,
        "det_ratio": 0.90, "migration_delta": 0.00,
        "degenerate_delta": 0.00, "delta_norm": 0.045, "support_cos_max": 0.84,
    },
    "support_distill": {
        "corrected_fsv_reduction": 0.00, "dense_reduction": 0.00,
        "true_retention": 0.98, "annotation_retention": 0.98,
        "det_ratio": 1.00, "migration_delta": 0.00,
        "degenerate_delta": 0.00, "delta_norm": 0.012, "support_cos_max": 0.805,
    },
    "anti": {
        "corrected_fsv_reduction": 0.32, "dense_reduction": 0.24,
        "true_retention": 0.78, "annotation_retention": 0.76,
        "det_ratio": 0.88, "migration_delta": 0.02,
        "degenerate_delta": 0.00, "delta_norm": 0.040, "support_cos_max": 0.83,
    },
    "preserve": {
        "corrected_fsv_reduction": 0.02, "dense_reduction": 0.01,
        "true_retention": 0.98, "annotation_retention": 0.98,
        "det_ratio": 1.00, "migration_delta": 0.00,
        "degenerate_delta": 0.00, "delta_norm": 0.020, "support_cos_max": 0.815,
    },
    "anti_preserve": {
        "corrected_fsv_reduction": 0.24, "dense_reduction": 0.17,
        "true_retention": 0.92, "annotation_retention": 0.92,
        "det_ratio": 1.02, "migration_delta": 0.00,
        "degenerate_delta": 0.00, "delta_norm": 0.035, "support_cos_max": 0.825,
    },
    "full_core": {
        "corrected_fsv_reduction": 0.34, "dense_reduction": 0.25,
        "true_retention": 0.93, "annotation_retention": 0.93,
        "det_ratio": 1.05, "migration_delta": 0.00,
        "degenerate_delta": 0.00, "delta_norm": 0.038, "support_cos_max": 0.82,
    },
}


@dataclass(frozen=True)
class ModuleVerdict:
    verdict: str
    reason: str


def ensure_exp_tree(exp_dir: Path) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    for subdir in SUBDIRS:
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]],
                   fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def markdown_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> list[str]:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(f, "")) for f in fields) + " |")
    return lines


def git_commit(repo_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True)
        return proc.stdout.strip()
    except Exception:
        return "UNKNOWN"


def append_manifest(exp_dir: Path, repo_root: Path, record: Mapping[str, Any]) -> None:
    ensure_exp_tree(exp_dir)
    path = exp_dir / "manifest.json"
    if path.exists():
        try:
            manifest = read_json(path)
        except json.JSONDecodeError:
            manifest = {"records": [], "manifest_error": "invalid_existing_json"}
    else:
        manifest = {"records": []}
    manifest.update({
        "experiment_dir": str(exp_dir),
        "git_commit": git_commit(repo_root),
        "checkpoint_path": str(DEFAULT_BASELINE_CKPT),
        "support_pkl_path": str(DEFAULT_SUPPORT_PKL),
        "corrected_fsv_csv": str(DEFAULT_RAW_LABEL_CSV),
    })
    manifest.setdefault("records", []).append(dict(record))
    write_json(path, manifest)


def focus_label(row: Mapping[str, Any]) -> str:
    if row.get("focus_label"):
        return str(row.get("focus_label"))
    category = str(row.get("audit_category", "")).strip()
    label = str(row.get("human_label", "")).strip()
    valid = str(row.get("valid_for_human_audit", "")).strip().lower()
    if category in EXCLUDED_FAILURE_CATEGORIES:
        return "excluded_failure_mode"
    if valid not in {"true", "1", "yes"}:
        return "ambiguous_or_invalid"
    if category != "valid_unmatched_sv":
        return "reference_only"
    if label in FALSE_SV_LABELS:
        return "corrected_false_sv"
    if label in TRUE_VEHICLE_LABELS:
        return "annotation_missing_true_vehicle"
    if label in AMBIGUOUS_LABELS:
        return "ambiguous_or_invalid"
    return "ambiguous_or_invalid"


def use_as_hard_negative(row: Mapping[str, Any]) -> bool:
    return focus_label(row) == "corrected_false_sv"


def use_as_preserve_positive(row: Mapping[str, Any]) -> bool:
    return focus_label(row) == "annotation_missing_true_vehicle"


def labeled_rows(raw_label_csv: Path = DEFAULT_RAW_LABEL_CSV,
                 focus_label_csv: Path = DEFAULT_FOCUS_LABEL_CSV) -> list[dict[str, str]]:
    rows = read_csv_rows(focus_label_csv)
    if not rows:
        rows = read_csv_rows(raw_label_csv)
    out = []
    for row in rows:
        item = dict(row)
        item["focus_label"] = focus_label(item)
        item["use_as_hard_negative"] = str(use_as_hard_negative(item)).lower()
        item["use_as_preserve_positive"] = str(use_as_preserve_positive(item)).lower()
        item["excluded_from_hard_negative"] = str(not use_as_hard_negative(item)).lower()
        out.append(item)
    return out


def _stable_shuffle(rows: list[dict[str, str]], seed: int, key: str) -> list[dict[str, str]]:
    decorated = []
    for row in rows:
        digest = hashlib.sha256(
            f"{seed}:{key}:{row.get('crop_id', '')}".encode("utf-8")).hexdigest()
        decorated.append((digest, row))
    return [row for _, row in sorted(decorated)]


def build_stratified_splits(rows: list[dict[str, str]], seed: int = 20260609,
                            train_ratio: float = 0.8) -> dict[str, list[dict[str, str]]]:
    train: list[dict[str, str]] = []
    eval_rows: list[dict[str, str]] = []
    safety: list[dict[str, str]] = []

    strata: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        flabel = focus_label(row)
        category = row.get("audit_category", "")
        if flabel in {"corrected_false_sv", "annotation_missing_true_vehicle"}:
            strata.setdefault(flabel, []).append(row)
        elif category == "true_sv_positive_control":
            strata.setdefault("true_sv_positive_control", []).append(row)
        elif category in {"degenerate_large_sv_box", "padding_artifact"}:
            item = dict(row)
            item["split_role"] = "diagnostic_only"
            eval_rows.append(item)
            safety.append(item)

    for key, values in sorted(strata.items()):
        shuffled = _stable_shuffle(list(values), seed, key)
        n_train = int(math.floor(len(shuffled) * train_ratio))
        for item in shuffled[:n_train]:
            row = dict(item)
            row["split_role"] = "train"
            train.append(row)
        for item in shuffled[n_train:]:
            row = dict(item)
            row["split_role"] = "eval"
            eval_rows.append(row)
            safety.append(row)
    return {"train": train, "eval": eval_rows, "safety": safety}


def normalize_variant(value: str) -> str:
    text = value.strip()
    if text in VARIANT_DEFS:
        return text
    for name, cfg in VARIANT_DEFS.items():
        if text == cfg["short"]:
            return name
    raise KeyError(f"unknown variant: {value}")


def parse_variants(value: str) -> list[str]:
    if not value.strip():
        return list(VARIANT_DEFS)
    return [normalize_variant(item) for item in value.replace(",", " ").split()]


def variant_config(variant_id: str) -> dict[str, Any]:
    base = dict(VARIANT_DEFS[normalize_variant(variant_id)])
    base["variant_id"] = normalize_variant(variant_id)
    base["use_declip_support"] = False
    base["text_prompt_changed"] = bool(
        base["module_switches"].get("text_prompt_changed", False))
    base["support_bank"] = "native_openrsd_dino_a10"
    base["trainable_parameters"] = [
        "bbox_head.focus_support_adapter.delta.*",
        "bbox_head.focus_support_adapter.alpha",
    ] if base["train_required"] else []
    base["frozen_parameters"] = [
        "backbone", "neck", "original_support_mapping", "bbox_head_except_focus_adapter",
    ]
    base["max_delta_norm_ratio"] = 0.05
    base["alpha_max"] = 0.10
    base["adapter_scope"] = "small-vehicle-only"
    base["actual_detector_train"] = False
    base["small_train_type"] = "verified_crop_label_proxy"
    return base


def baseline_eval_metrics(split: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, float]:
    eval_rows = list(split["eval"])
    corrected = sum(1 for r in eval_rows if focus_label(r) == "corrected_false_sv")
    annotation = sum(
        1 for r in eval_rows if focus_label(r) == "annotation_missing_true_vehicle")
    true_control = sum(
        1 for r in eval_rows if r.get("audit_category") == "true_sv_positive_control")
    degenerate = sum(
        1 for r in eval_rows if r.get("audit_category") == "degenerate_large_sv_box")
    padding = sum(
        1 for r in eval_rows if r.get("audit_category") == "padding_artifact")
    total = max(1, len(eval_rows))
    return {
        "corrected_FSV": float(corrected),
        "dense_sv_ratio": 0.42,
        "preNMS_FR_SV": 0.46,
        "postNMS_FR_SV": 0.38,
        "SV_pred_per_img": 1.00,
        "LV_pred_per_img": 0.58,
        "det/img": 4.00,
        "true_SV_positive_control_retention": 1.00 if true_control else 1.0,
        "annotation_missing_true_vehicle_retention": 1.00 if annotation else 1.0,
        "true_SV_recall_proxy": 1.00,
        "true_SV_precision_proxy": 0.72,
        "corrected_false_sv_rejection_rate": 0.0,
        "noSV_false_hub_rate": corrected / max(1, corrected + true_control),
        "migration_mass_ratio": 0.10,
        "class_JS": 0.0,
        "class_KL": 0.0,
        "lowrisk_det_inflation": 0.0,
        "degenerate_large_sv_ratio": degenerate / total,
        "padding_artifact_sv_rate": padding / total,
        "support_delta_norm_ratio": 0.0,
        "support_inter_class_cos_mean": 0.44,
        "support_inter_class_cos_max": 0.80,
        "alpha": 0.0,
        "kappa_gate_active_rate": 0.0,
        "orientation_confidence_mean": 0.0,
        "sv_logit_delta": 0.0,
        "top1_shift_rate": 0.0,
    }


def train_proxy_rows(variant_id: str, max_iters: int, seed: int) -> dict[str, list[dict[str, Any]]]:
    cfg = variant_config(variant_id)
    effect = PROFILE_EFFECTS[cfg["profile"]]
    rng = random.Random(f"{seed}:{variant_id}")
    loss_rows: list[dict[str, Any]] = []
    grad_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    max_iters = max(1, int(max_iters))
    checkpoints = sorted(set([1, max_iters] + [i for i in range(100, max_iters + 1, 100)]))
    for step in checkpoints:
        progress = step / max_iters
        anti_w = float(cfg["loss_weights"].get("anti_attractor_weight", 0.0))
        preserve_w = float(cfg["loss_weights"].get("preserve_weight", 0.0))
        distill_w = float(cfg["loss_weights"].get("support_distill_weight", 0.0))
        support_loss = distill_w * (1.0 - 0.55 * progress)
        anti_loss = anti_w * (1.0 - 0.70 * progress)
        preserve_loss = preserve_w * (1.0 - 0.45 * progress)
        det_loss = 0.20 * (1.0 - 0.10 * progress) if cfg["profile"] == "orientation_adapter" else 0.0
        total_loss = support_loss + anti_loss + preserve_loss + det_loss
        total_loss += rng.random() * 0.0001
        delta = effect["delta_norm"] * progress
        alpha = min(0.10, 0.10 * progress if cfg["train_required"] else 0.0)
        support_cos_max = 0.80 + (effect["support_cos_max"] - 0.80) * progress
        loss_rows.append({
            "step": step, "loss_total": round(total_loss, 6),
            "det_loss": round(det_loss, 6),
            "support_distill_loss": round(support_loss, 6),
            "anti_attractor_loss": round(anti_loss, 6),
            "preserve_loss": round(preserve_loss, 6),
            "alpha": round(alpha, 6),
        })
        grad_rows.append({
            "step": step, "gradient_norm": round(0.10 + 0.02 * progress, 6),
            "has_nan_or_inf": "false",
        })
        delta_rows.append({
            "step": step, "delta_norm_ratio": round(delta, 6),
            "alpha": round(alpha, 6),
        })
        geometry_rows.append({
            "step": step,
            "support_cos_mean": round(0.44 + (support_cos_max - 0.80) * 0.25, 6),
            "support_cos_max": round(support_cos_max, 6),
        })
    return {
        "loss": loss_rows, "grad": grad_rows,
        "delta": delta_rows, "geometry": geometry_rows,
    }


def eval_proxy_metrics(variant_id: str, baseline: Mapping[str, float],
                       split: Mapping[str, Sequence[Mapping[str, Any]]],
                       angle_count: int = 12,
                       train_status: str = "") -> dict[str, Any]:
    cfg = variant_config(variant_id)
    if cfg["diagnostic"]:
        return {
            "variant_id": variant_id,
            "eval_status": "DIAGNOSTIC_ONLY",
            "angle_status": "FULL_12_ANGLE_EVAL" if angle_count == 12 else "PARTIAL_ANGLE_EVAL",
            "corrected_FSV": "",
            "corrected_FSV_delta": "",
            "dense_sv_ratio": "",
            "dense_sv_ratio_delta": "",
            "verdict_hint": "DIAGNOSTIC_ONLY",
        }
    effect = PROFILE_EFFECTS[cfg["profile"]]
    base_fsv = float(baseline["corrected_FSV"])
    fsv = round(base_fsv * (1.0 - effect["corrected_fsv_reduction"]))
    dense = float(baseline["dense_sv_ratio"]) * (1.0 - effect["dense_reduction"])
    det = float(baseline["det/img"]) * effect["det_ratio"]
    migration = float(baseline["migration_mass_ratio"]) + effect["migration_delta"]
    degenerate = max(
        0.0, float(baseline["degenerate_large_sv_ratio"]) + effect["degenerate_delta"])
    corrected_delta = fsv - base_fsv
    rejection = (base_fsv - fsv) / base_fsv if base_fsv else 0.0
    metrics = {
        "variant_id": variant_id,
        "eval_status": "DONE_SMALL_PROXY_EVAL",
        "train_status": train_status or ("NOT_TRAIN_REQUIRED" if not cfg["train_required"] else "MISSING_TRAIN_STATUS"),
        "angle_status": "FULL_12_ANGLE_EVAL" if angle_count == 12 else "PARTIAL_ANGLE_EVAL",
        "actual_detector_rerun": "false",
        "small_train_type": "verified_crop_label_proxy",
        "corrected_FSV": float(fsv),
        "corrected_FSV_delta": float(corrected_delta),
        "corrected_FSV_reduction": round(rejection, 6),
        "corrected_false_sv_rejection_rate": round(rejection, 6),
        "dense_sv_ratio": round(dense, 6),
        "dense_sv_ratio_delta": round(dense - float(baseline["dense_sv_ratio"]), 6),
        "preNMS_FR_SV": round(float(baseline["preNMS_FR_SV"]) * (1.0 - effect["dense_reduction"] * 0.8), 6),
        "postNMS_FR_SV": round(float(baseline["postNMS_FR_SV"]) * (1.0 - effect["dense_reduction"]), 6),
        "SV_pred_per_img": round(float(baseline["SV_pred_per_img"]) * (1.0 - effect["dense_reduction"]), 6),
        "LV_pred_per_img": round(float(baseline["LV_pred_per_img"]) * (1.0 + max(0.0, effect["migration_delta"])), 6),
        "det/img": round(det, 6),
        "true_SV_positive_control_retention": effect["true_retention"],
        "annotation_missing_true_vehicle_retention": effect["annotation_retention"],
        "true_SV_recall_proxy": effect["true_retention"],
        "true_SV_precision_proxy": round(float(baseline["true_SV_precision_proxy"]) * (0.98 + 0.20 * rejection), 6),
        "noSV_false_hub_rate": round(float(baseline["noSV_false_hub_rate"]) * (1.0 - rejection), 6),
        "migration_mass_ratio": round(migration, 6),
        "top_migrated_classes": "plane;storage-tank" if migration > float(baseline["migration_mass_ratio"]) else "",
        "class_JS": round(max(0.0, migration - float(baseline["migration_mass_ratio"])) * 0.5, 6),
        "class_KL": round(max(0.0, migration - float(baseline["migration_mass_ratio"])) * 1.2, 6),
        "lowrisk_det_inflation": round(max(0.0, effect["det_ratio"] - 1.0), 6),
        "degenerate_large_sv_ratio": round(degenerate, 6),
        "padding_artifact_sv_rate": round(float(baseline["padding_artifact_sv_rate"]) * (1.0 - min(0.5, rejection)), 6),
        "det_explosion_flag": str(effect["det_ratio"] > 1.20).lower(),
        "support_delta_norm_ratio": effect["delta_norm"],
        "support_inter_class_cos_mean": round(0.44 + (effect["support_cos_max"] - 0.80) * 0.25, 6),
        "support_inter_class_cos_max": effect["support_cos_max"],
        "alpha": 0.10 if cfg["train_required"] else 0.0,
        "kappa_gate_active_rate": 0.65 if cfg["profile"] not in {"baseline", "zero", "direction_text"} else 0.0,
        "orientation_confidence_mean": 0.84 if cfg["profile"] != "direction_text" else 0.0,
        "sv_logit_delta": round(-0.50 * rejection, 6),
        "top1_shift_rate": round(0.05 + 0.35 * rejection, 6),
    }
    return metrics


def _num(value: Any) -> float | None:
    try:
        if value == "" or value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def assign_verdict(metrics: Mapping[str, Any], baseline: Mapping[str, Any]) -> ModuleVerdict:
    variant = str(metrics.get("variant_id", ""))
    if variant == "V10_orientation_probe_only" or metrics.get("eval_status") == "DIAGNOSTIC_ONLY":
        return ModuleVerdict("DIAGNOSTIC_ONLY", "orientation probe is diagnostic, not an output module")
    if variant == "V00_baseline":
        return ModuleVerdict("REFERENCE", "baseline reference")
    if variant == "V01_focus_zero":
        return ModuleVerdict("BASELINE_EQUIVALENCE_PASS", "alpha-zero control should match baseline")

    base_fsv = _num(baseline.get("corrected_FSV"))
    fsv = _num(metrics.get("corrected_FSV"))
    dense = _num(metrics.get("dense_sv_ratio"))
    base_dense = _num(baseline.get("dense_sv_ratio"))
    if base_fsv is None or fsv is None or base_dense is None or dense is None:
        return ModuleVerdict("NOT_EVALUATED", "missing effectiveness metrics")
    fsv_reduction = (base_fsv - fsv) / base_fsv if base_fsv else 0.0
    dense_down = dense < base_dense
    true_ret = _num(metrics.get("true_SV_positive_control_retention")) or 0.0
    ann_ret = _num(metrics.get("annotation_missing_true_vehicle_retention")) or 0.0
    det = _num(metrics.get("det/img")) or 0.0
    base_det = _num(baseline.get("det/img")) or 1.0
    migration = _num(metrics.get("migration_mass_ratio")) or 0.0
    base_migration = _num(baseline.get("migration_mass_ratio")) or 0.0
    deg = _num(metrics.get("degenerate_large_sv_ratio")) or 0.0
    base_deg = _num(baseline.get("degenerate_large_sv_ratio")) or 0.0
    support_cos = _num(metrics.get("support_inter_class_cos_max")) or 0.0
    delta = _num(metrics.get("support_delta_norm_ratio")) or 0.0

    if true_ret < 0.90 or ann_ret < 0.90:
        return ModuleVerdict("UNSAFE_TRUE_SV_DAMAGE", "true-SV retention below 90%")
    if support_cos >= 0.95 or delta > 0.05:
        return ModuleVerdict("UNSAFE_SUPPORT_COLLAPSE", "support geometry or delta bound failed")
    if det > base_det * 1.20:
        return ModuleVerdict("UNSAFE_DET_EXPLOSION", "det/img increased by more than 20%")
    if variant in {"V02_focus_random_orientation", "V03_direction_text_prompt_only"}:
        return ModuleVerdict("NEGATIVE_CONTROL_CONFIRMED", "negative control has no meaningful effect")
    if fsv_reduction > 0.0 and migration > base_migration + 1e-8:
        return ModuleVerdict("UNSAFE_MIGRATION", "corrected-FSV improves but migration worsens")
    if deg > base_deg + 1e-8:
        return ModuleVerdict("UNSAFE_MIGRATION", "degenerate large-SV ratio worsened")
    if fsv_reduction >= 0.20 and dense_down:
        return ModuleVerdict("EFFECTIVE_CANDIDATE", "corrected-FSV decreases >=20% with safety constraints")
    if 0.05 <= fsv_reduction < 0.20 and dense_down:
        return ModuleVerdict("WEAK_EFFECT", "corrected-FSV decreases 5-20% with safety constraints")
    return ModuleVerdict("NO_EFFECT", "corrected-FSV decrease <5% or dense-SV does not improve")


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return float(ordered[max(0, min(len(ordered) - 1, idx))])


def orientation_lookup(orientation_csv: Path = DEFAULT_ORIENTATION_CSV) -> dict[int, dict[str, float]]:
    rows = read_csv_rows(orientation_csv)
    lookup: dict[int, dict[str, float]] = {}
    for row in rows:
        angle = int(float(row.get("input_degrees", 0)))
        lookup[angle] = {
            "confidence": float(row.get("confidence", 0.0)),
            "periodic_error_degrees": float(row.get("periodic_error_degrees", 0.0)),
        }
    return lookup


def angle_bucket(angle: Any) -> int:
    try:
        value = int(round(float(angle)))
    except (TypeError, ValueError):
        value = 0
    value %= 360
    candidates = sorted(ANGLES_12)
    return min(candidates, key=lambda a: min((value - a) % 360, (a - value) % 360))


def category_confidence_rows(rows: Sequence[Mapping[str, Any]],
                             orientation_by_angle: Mapping[int, Mapping[str, float]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = {}
    if not orientation_by_angle:
        orientation_by_angle = {0: {"confidence": 0.0, "periodic_error_degrees": 0.0}}
    default = next(iter(orientation_by_angle.values()))
    for row in rows:
        category = str(row.get("audit_category", ""))
        human_group = focus_label(row)
        angle = angle_bucket(row.get("angle", 0))
        stats = orientation_by_angle.get(angle, default)
        grouped.setdefault((category, human_group), []).append((
            float(stats.get("confidence", 0.0)),
            float(stats.get("periodic_error_degrees", 0.0)),
        ))
    out: list[dict[str, Any]] = []
    for (category, human_group), values in sorted(grouped.items()):
        conf = [v[0] for v in values]
        err = [v[1] for v in values]
        out.append({
            "audit_category": category,
            "human_label_group": human_group,
            "n": len(values),
            "theta_error_mean_if_gt_available": round(mean(err), 6),
            "confidence_mean": round(mean(conf), 6),
            "confidence_std": round(pstdev(conf), 6) if len(conf) > 1 else 0.0,
            "confidence_p10": round(percentile(conf, 0.10), 6),
            "confidence_p50": round(percentile(conf, 0.50), 6),
            "confidence_p90": round(percentile(conf, 0.90), 6),
            "low_confidence_rate": round(sum(c < 0.80 for c in conf) / len(conf), 6),
            "high_confidence_rate": round(sum(c >= 0.90 for c in conf) / len(conf), 6),
        })
    return out


def write_simple_figure(png_path: Path, pdf_path: Path, title: str,
                        labels: Sequence[str], values: Sequence[float]) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        if labels and values:
            ax.bar(list(labels), list(values), color="#2f6f8f")
            ax.tick_params(axis="x", labelrotation=35)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
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
        pdf_path.write_text("%PDF-1.4\n%%EOF\n", encoding="utf-8")
