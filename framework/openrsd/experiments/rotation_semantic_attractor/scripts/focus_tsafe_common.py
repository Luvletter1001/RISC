#!/usr/bin/env python3
"""Shared helpers for FOCUS-T-Safe shadow diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


EXP_DIR = Path("resultmd/exp_focus_tsafe_20260610")
HUMAN_LABEL_CSV = Path(
    "experiments/rotation_semantic_attractor/reports/visual_summary/audit/"
    "human_sv_crop_audit_labels_VERIFIED_EXPANDED.csv")
FOCUS_CKPT = Path(
    "work_dirs/focus_ovd_a10_sv_only_dota2_recovery_full_gpu69_20260609/"
    "epoch_24.pth")
FOCUS_REF = {
    "mAP": 0.6775,
    "small_vehicle_AP": 0.5541,
    "method": "FOCUS-OVD",
}
SUBDIRS = (
    "preflight",
    "shadow_scores",
    "loss_only",
    "offline_mixing",
    "tables",
    "figures",
    "reports",
    "html",
)
FALSE_LABELS = {"non_vehicle_background", "non_vehicle_object_conflict"}
TRUE_LABELS = {
    "true_vehicle_annot_missing",
    "true_vehicle_annot_present_but_missed_match",
}
EXCLUDED_LABELS = {
    "actual_large_sv_degenerate_prediction",
    "valid_padding_artifact",
    "ambiguous",
    "invalid_visualization",
    "wrong_or_ambiguous",
    "uncertain",
    "",
}


def resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def ensure_tree(exp_dir: Path) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    for subdir in SUBDIRS:
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def md_table(rows: Sequence[Mapping[str, Any]],
             fields: Sequence[str]) -> list[str]:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(
            str(row.get(field, "")).replace("|", "\\|")
            for field in fields) + " |")
    return lines


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def canonical_category(row: Mapping[str, Any]) -> str:
    audit_category = str(row.get("audit_category", ""))
    label = str(row.get("human_label", ""))
    if audit_category in {"degenerate_large_sv_box", "padding_artifact",
                          "true_sv_positive_control"}:
        return audit_category
    if label in FALSE_LABELS:
        return "corrected_false_sv"
    if label in TRUE_LABELS:
        return "annotation_missing_true_vehicle"
    if audit_category == "true_sv_positive_control":
        return "true_sv_positive_control"
    return audit_category or "unknown"


def binary_true_vehicle(row: Mapping[str, Any]) -> int | None:
    category = canonical_category(row)
    label = str(row.get("human_label", ""))
    if category in {"annotation_missing_true_vehicle", "true_sv_positive_control"}:
        return 1
    if category == "corrected_false_sv" or label in FALSE_LABELS:
        return 0
    return None


def stable_unit_float(*parts: Any) -> float:
    digest = hashlib.sha256(
        "|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    pairs = [(int(label), float(score)) for label, score in zip(labels, scores)]
    positives = [score for label, score in pairs if label == 1]
    negatives = [score for label, score in pairs if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = len(positives) * len(negatives)
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total


def summarize(values: Sequence[float]) -> dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(vals),
        "mean": mean(vals),
        "min": min(vals),
        "max": max(vals),
    }


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)
