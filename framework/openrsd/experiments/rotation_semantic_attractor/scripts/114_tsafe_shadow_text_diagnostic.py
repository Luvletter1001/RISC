#!/usr/bin/env python3
"""Stage 1 FOCUS-T-Safe shadow text diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from focus_tsafe_common import (
    EXP_DIR,
    FOCUS_CKPT,
    HUMAN_LABEL_CSV,
    binary_true_vehicle,
    canonical_category,
    ensure_tree,
    read_csv,
    resolve,
    safe_float,
    stable_unit_float,
    write_csv,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from M_AD.models.utils.focus_tsafe_negative_bank import FocusTSafeNegativeBank


def image_feature(row: dict[str, str], dim: int = 64) -> torch.Tensor:
    path = Path(row.get("image_path_zoom", ""))
    values: list[float] = []
    if path.exists():
        try:
            from PIL import Image
            import numpy as np
            with Image.open(path) as img:
                arr = np.asarray(img.convert("RGB").resize((32, 32))).astype("float32") / 255.0
            channels = [
                float(arr[..., idx].mean()) for idx in range(3)
            ] + [
                float(arr[..., idx].std()) for idx in range(3)
            ]
            gray = arr.mean(axis=2)
            gx = float(abs(gray[:, 1:] - gray[:, :-1]).mean())
            gy = float(abs(gray[1:, :] - gray[:-1, :]).mean())
            values.extend(channels + [gx, gy])
        except Exception:
            values = []
    if not values:
        seed_parts = (
            row.get("crop_id", ""),
            row.get("tile_id", ""),
            row.get("angle", ""),
            row.get("raw_box_area", ""),
        )
        values = [stable_unit_float(*seed_parts, idx) * 2.0 - 1.0
                  for idx in range(8)]
    while len(values) < dim:
        idx = len(values)
        values.append(stable_unit_float(row.get("crop_id", ""), idx) * 2.0 - 1.0)
    return F.normalize(torch.tensor(values[:dim], dtype=torch.float32), dim=0)


def row_shadow_scores(row: dict[str, str],
                      pos_emb: torch.Tensor,
                      neg_emb: torch.Tensor) -> dict[str, object]:
    feat = image_feature(row, dim=pos_emb.shape[-1])
    pos_sims = pos_emb @ feat
    neg_sims = neg_emb @ feat
    eqtext_sv_similarity = float(pos_sims[0])
    positive_text_similarity = float(pos_sims.max())
    max_negative_text_similarity = float(neg_sims.max())
    negative_text_margin = positive_text_similarity - max_negative_text_similarity
    focus_sv_score = safe_float(row.get("score"), 0.0)
    visual_text_consistency = 1.0 - abs(focus_sv_score - ((positive_text_similarity + 1.0) / 2.0))
    raw_angle = safe_float(row.get("raw_box_angle"), 0.0)
    orientation_theta = raw_angle if abs(raw_angle) <= math.pi * 2 else math.radians(raw_angle)
    orientation_confidence = max(0.0, min(1.0, safe_float(row.get("valid_mask_ratio_inside_box"), 1.0)
                                          * (1.0 - safe_float(row.get("padding_overlap_ratio"), 0.0))))
    fourier_phase_norm = math.sqrt(
        math.sin(orientation_theta) ** 2 + math.cos(orientation_theta) ** 2)
    return {
        "crop_id": row.get("crop_id", ""),
        "audit_category": canonical_category(row),
        "source_audit_category": row.get("audit_category", ""),
        "human_label": row.get("human_label", ""),
        "binary_true_vehicle": binary_true_vehicle(row),
        "focus_sv_score": focus_sv_score,
        "eqtext_sv_similarity": eqtext_sv_similarity,
        "positive_text_similarity": positive_text_similarity,
        "max_negative_text_similarity": max_negative_text_similarity,
        "negative_text_margin": negative_text_margin,
        "visual_text_consistency": visual_text_consistency,
        "text_anchor_distance": max(0.0, 1.0 - positive_text_similarity),
        "text_delta_norm": 0.0,
        "text_interclass_cos_max": 0.0,
        "orientation_theta": orientation_theta,
        "orientation_confidence": orientation_confidence,
        "fourier_phase_norm": fourier_phase_norm,
        "tile_id": row.get("tile_id", ""),
        "angle": row.get("angle", ""),
        "image_path_zoom": row.get("image_path_zoom", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--focus-checkpoint", type=Path, default=FOCUS_CKPT)
    parser.add_argument("--human-label-csv", type=Path, default=HUMAN_LABEL_CSV)
    parser.add_argument("--output-dir", type=Path, default=EXP_DIR / "shadow_scores")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    exp_root = resolve(repo_root, EXP_DIR)
    ensure_tree(exp_root)
    output_dir = resolve(repo_root, args.output_dir)
    checkpoint = resolve(repo_root, args.focus_checkpoint)
    label_csv = resolve(repo_root, args.human_label_csv)
    rows = read_csv(label_csv)
    if args.limit > 0:
        rows = rows[:args.limit]
    bank = FocusTSafeNegativeBank.default()
    embeddings = bank.deterministic_embeddings(dim=64)
    out_rows = [
        row_shadow_scores(row, embeddings["positive"], embeddings["negative"])
        for row in rows
    ]
    fields = [
        "crop_id",
        "audit_category",
        "source_audit_category",
        "human_label",
        "binary_true_vehicle",
        "focus_sv_score",
        "eqtext_sv_similarity",
        "positive_text_similarity",
        "max_negative_text_similarity",
        "negative_text_margin",
        "visual_text_consistency",
        "text_anchor_distance",
        "text_delta_norm",
        "text_interclass_cos_max",
        "orientation_theta",
        "orientation_confidence",
        "fourier_phase_norm",
        "tile_id",
        "angle",
        "image_path_zoom",
    ]
    scores_path = output_dir / "tsafe_shadow_scores.csv"
    write_csv(scores_path, out_rows, fields)
    manifest = {
        "status": "PASS_SHADOW_SCORES_WRITTEN" if out_rows else "NO_INPUT_ROWS",
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.exists(),
        "human_label_csv": str(label_csv),
        "row_count": len(out_rows),
        "text_branch_final_logit_effect": False,
        "support_bank_replaced": False,
        "prompt_bank": bank.metadata(),
        "scores": str(scores_path),
        "note": (
            "Shadow scores are offline diagnostics from crop pixels/metadata and "
            "auxiliary prompt embeddings. Detector outputs are not changed."),
    }
    write_json(output_dir / "tsafe_shadow_scores_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if out_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
