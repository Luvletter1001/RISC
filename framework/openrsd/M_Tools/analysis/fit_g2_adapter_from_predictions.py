#!/usr/bin/env python
"""Fit a non-positive G-S3C G2 adapter from saved predictions.

This is an offline adapter-fitting path for evidence gathering.  It trains the
same GaussianScaleLogitAdapter used by the network head on localized prediction
records, then applies the learned non-positive score delta to a predictions.pkl.
It is not a substitute for full dense-head training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from M_AD.models.utils.gaussian_semantic_scale import (  # noqa: E402
    GaussianScaleLogitAdapter,
)
from M_Tools.analysis import evaluate_sise_score_calibration as sise  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit G-S3C G2 Gaussian scale adapter on predictions.pkl.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ann-dir", required=True)
    parser.add_argument("--class-area-priors-csv", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-pkl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--adapter-ckpt", required=True)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--diff-thr", type=int, default=100)
    parser.add_argument("--z-thr", type=float, default=4.0)
    parser.add_argument("--score-thrs", default="0.5,0.7,0.9,0.99,0.999")
    parser.add_argument(
        "--exclude-sibling-pairs",
        default="small-vehicle->large-vehicle,large-vehicle->small-vehicle")
    parser.add_argument(
        "--bad-mode",
        choices=("scale_wrong", "logz_wrong", "p0199_wrong", "wrong"),
        default="scale_wrong",
        help="Which localized wrong detections are fitted as suppress targets.")
    parser.add_argument("--train-image-frac", type=float, default=0.7)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--max-dets-per-img", type=int, default=0)
    parser.add_argument("--max-train-rows", type=int, default=200000)
    parser.add_argument("--adapter-hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bad-weight", type=float, default=3.0)
    parser.add_argument("--good-weight", type=float, default=1.0)
    parser.add_argument("--delta-l2", type=float, default=1e-3)
    parser.add_argument(
        "--rank-preserve-weight",
        type=float,
        default=0.0,
        help="Penalty weight that keeps AP-sensitive correct detections near zero delta.")
    parser.add_argument(
        "--rank-preserve-margin",
        type=float,
        default=0.05,
        help="Allowed negative delta magnitude for rank-protected good rows.")
    parser.add_argument(
        "--rank-protect-score-thr",
        type=float,
        default=0.5,
        help="Only correct rows above this score can be rank-protected.")
    parser.add_argument(
        "--rank-protect-image-rank",
        type=int,
        default=100,
        help="Protect correct rows with image-local score rank <= this value; 0 disables.")
    parser.add_argument(
        "--rank-protect-class-rank",
        type=int,
        default=0,
        help="Protect correct rows with class-local score rank <= this value; 0 disables.")
    parser.add_argument(
        "--rank-protect-global-rank",
        type=int,
        default=0,
        help="Protect correct rows with global score rank <= this value; 0 disables.")
    parser.add_argument(
        "--delta-target-weight",
        type=float,
        default=0.0,
        help="Optional MSE target: bad rows -> -max_delta_abs, good rows -> 0.")
    parser.add_argument("--max-delta-abs", type=float, default=1.0)
    parser.add_argument("--apply-min-abs-z", type=float, default=0.0)
    parser.add_argument("--apply-min-score", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_priors(path: Path) -> dict[str, dict[str, float]]:
    priors = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cls_name = row.get("class", "")
            if not cls_name:
                continue
            priors[cls_name] = {
                "p01_area": float(row.get("p01_area", 0) or 0),
                "p99_area": float(row.get("p99_area", 0) or 0),
                "log_area_mean": float(row.get("log_area_mean", 0) or 0),
                "log_area_std": max(float(row.get("log_area_std", 0) or 0), 1e-6),
            }
    return priors


def stable_fraction(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) / float(16**12)


def prior_features(area: float, cls_name: str, priors: dict[str, dict[str, float]]) -> dict:
    prior = priors.get(cls_name)
    if not prior or area <= 0:
        return {
            "abs_z": 0.0,
            "signed_z": 0.0,
            "gaussian_log_prob": 0.0,
            "log_std": 0.0,
            "valid": False,
            "p0199_outlier": False,
        }
    log_area = math.log(area)
    std = max(float(prior["log_area_std"]), 1e-6)
    signed_z = (log_area - float(prior["log_area_mean"])) / std
    log_std = math.log(std)
    gaussian_log_prob = -0.5 * signed_z * signed_z - log_std - 0.5 * math.log(2.0 * math.pi)
    return {
        "abs_z": abs(signed_z),
        "signed_z": signed_z,
        "gaussian_log_prob": gaussian_log_prob,
        "log_std": log_std,
        "valid": True,
        "p0199_outlier": (
            area < float(prior["p01_area"]) or area > float(prior["p99_area"])),
    }


def is_bad_record(args: argparse.Namespace, wrong: bool, is_sibling: bool,
                  feats: dict) -> bool:
    if not wrong or is_sibling:
        return False
    logz = feats["abs_z"] >= args.z_thr
    p0199 = bool(feats["p0199_outlier"])
    if args.bad_mode == "logz_wrong":
        return logz
    if args.bad_mode == "p0199_wrong":
        return p0199
    if args.bad_mode == "wrong":
        return True
    return logz or p0199


def add_fit_row_rank_fields(rows: list[dict]) -> list[dict]:
    """Assign score ranks used by AP-preserving G2 adapter training."""
    ranked_rows = [dict(row) for row in rows]
    global_order = sorted(
        range(len(ranked_rows)),
        key=lambda idx: (-float(ranked_rows[idx].get("score", 0.0)), idx))
    for rank, idx in enumerate(global_order, start=1):
        ranked_rows[idx]["global_rank"] = rank

    grouped: dict[tuple[str, str], list[int]] = {}
    for idx, row in enumerate(ranked_rows):
        img_id = str(row.get("img_id", ""))
        pred_label = str(row.get("pred_label", row.get("pred_class", "")))
        grouped.setdefault(("image", img_id), []).append(idx)
        grouped.setdefault(("class", pred_label), []).append(idx)

    for (scope, _), indices in grouped.items():
        ordered = sorted(
            indices,
            key=lambda idx: (-float(ranked_rows[idx].get("score", 0.0)), idx))
        key = "image_rank" if scope == "image" else "class_rank"
        for rank, idx in enumerate(ordered, start=1):
            ranked_rows[idx][key] = rank

    for row in ranked_rows:
        row.setdefault("global_rank", len(ranked_rows) + 1)
        row.setdefault("image_rank", len(ranked_rows) + 1)
        row.setdefault("class_rank", len(ranked_rows) + 1)
    return ranked_rows


def build_fit_rows(predictions: list, ann_dir: Path, class_names: tuple[str, ...],
                   priors: dict[str, dict[str, float]],
                   args: argparse.Namespace) -> tuple[list[dict], dict]:
    class_to_label = {name: idx for idx, name in enumerate(class_names)}
    sibling_pairs = {
        item for item in args.exclude_sibling_pairs.split(",") if item.strip()
    }
    rows = []
    stats = Counter()
    for image_idx, sample in enumerate(predictions):
        if args.max_images > 0 and image_idx >= args.max_images:
            break
        stats["images_scanned"] += 1
        img_id = sise.get_img_id(sample)
        ann_file = ann_dir / f"{img_id}.txt"
        if not ann_file.exists():
            ann_file = ann_dir / f"{sise.canonical_img_id(img_id)}.txt"
        if not ann_file.exists():
            stats["images_missing_ann"] += 1
        gt_rows = sise.read_gt(ann_file, class_to_label, args.diff_thr)
        pred_boxes, pred_labels, pred_scores = sise.get_pred_arrays(
            sample, args.max_dets_per_img)
        stats["detections_seen"] += len(pred_scores)
        stats["gt_seen"] += len(gt_rows)
        if len(pred_scores) == 0 or len(gt_rows) == 0:
            continue

        gt_boxes = np.asarray([row["rbox"] for row in gt_rows], dtype=np.float32)
        ious = sise.box_iou_rotated(
            torch.from_numpy(pred_boxes.astype(np.float32)),
            torch.from_numpy(gt_boxes)).cpu().numpy()
        best_gt = ious.argmax(axis=1)
        best_iou = ious[np.arange(ious.shape[0]), best_gt]
        train_image = stable_fraction(img_id) < args.train_image_frac

        for det_idx, gt_idx in enumerate(best_gt):
            if best_iou[det_idx] <= args.iou_thr:
                continue
            pred_label = int(pred_labels[det_idx])
            if pred_label < 0 or pred_label >= len(class_names):
                continue
            gt = gt_rows[int(gt_idx)]
            gt_label = int(gt["label"])
            pred_class = class_names[pred_label]
            pair = f"{pred_class}->{gt['class_name']}"
            feats = prior_features(
                sise.box_area_rbox(pred_boxes[det_idx]), pred_class, priors)
            if not feats["valid"]:
                stats["localized_no_prior"] += 1
                continue
            correct = pred_label == gt_label
            wrong = not correct
            is_sibling = pair in sibling_pairs
            bad = is_bad_record(args, wrong, is_sibling, feats)
            if not correct and not bad:
                stats["localized_ignored_wrong"] += 1
                continue
            rows.append({
                "img_id": img_id,
                "det_idx": int(det_idx),
                "pred_label": pred_label,
                "train_image": train_image,
                "target_good": 1.0 if correct else 0.0,
                "score": float(pred_scores[det_idx]),
                **feats,
            })
            if correct:
                stats["localized_good_rows"] += 1
            else:
                stats["localized_bad_rows"] += 1
    stats["fit_rows"] = len(rows)
    stats["fit_train_rows"] = sum(1 for row in rows if row["train_image"])
    stats["fit_holdout_rows"] = sum(1 for row in rows if not row["train_image"])
    return add_fit_row_rank_fields(rows), dict(stats)


def select_train_rows(rows: list[dict], max_rows: int, seed: int) -> list[dict]:
    train_rows = [row for row in rows if row["train_image"]]
    if max_rows <= 0 or len(train_rows) <= max_rows:
        return train_rows
    rng = np.random.default_rng(seed)
    bad = [idx for idx, row in enumerate(train_rows) if row["target_good"] < 0.5]
    good = [idx for idx, row in enumerate(train_rows) if row["target_good"] >= 0.5]
    bad_cap = min(len(bad), max_rows // 2)
    good_cap = max_rows - bad_cap
    chosen = []
    if bad_cap:
        chosen.extend(rng.choice(bad, size=bad_cap, replace=False).tolist())
    if good:
        good_cap = min(len(good), good_cap)
        chosen.extend(rng.choice(good, size=good_cap, replace=False).tolist())
    chosen = sorted(set(int(idx) for idx in chosen))
    return [train_rows[idx] for idx in chosen]


def rows_to_tensors(rows: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    def tensor(key: str) -> torch.Tensor:
        return torch.tensor(
            [row[key] for row in rows],
            dtype=torch.float32,
            device=device)

    def tensor_default(key: str, default: float) -> torch.Tensor:
        return torch.tensor(
            [row.get(key, default) for row in rows],
            dtype=torch.float32,
            device=device)

    score = tensor("score").clamp(min=1e-6, max=1 - 1e-6)
    target = tensor("target_good")
    return {
        "abs_z": tensor("abs_z"),
        "signed_z": tensor("signed_z"),
        "gaussian_log_prob": tensor("gaussian_log_prob"),
        "log_std": tensor("log_std"),
        "valid": torch.ones((len(rows),), dtype=torch.bool, device=device),
        "base_logit": torch.logit(score),
        "target_good": target,
        "score": score,
        "global_rank": tensor_default("global_rank", 1e9),
        "image_rank": tensor_default("image_rank", 1e9),
        "class_rank": tensor_default("class_rank", 1e9),
    }


def adapter_delta(adapter: GaussianScaleLogitAdapter, data: dict[str, torch.Tensor],
                  max_delta_abs: float) -> torch.Tensor:
    delta = adapter(
        data["abs_z"],
        data["signed_z"],
        data["gaussian_log_prob"],
        data["log_std"],
        data["valid"],
    )
    if max_delta_abs > 0:
        delta = delta.clamp(min=-float(max_delta_abs), max=0.0)
    return delta


def rank_protect_mask(data: dict[str, torch.Tensor],
                      args: argparse.Namespace) -> torch.Tensor:
    """Return correct high-rank rows whose scores should not be suppressed."""
    good = data["target_good"] >= 0.5
    score_keep = data["score"] >= float(
        getattr(args, "rank_protect_score_thr", 0.5))
    rank_terms = []
    image_rank = int(getattr(args, "rank_protect_image_rank", 0) or 0)
    class_rank = int(getattr(args, "rank_protect_class_rank", 0) or 0)
    global_rank = int(getattr(args, "rank_protect_global_rank", 0) or 0)
    if image_rank > 0:
        rank_terms.append(data["image_rank"] <= float(image_rank))
    if class_rank > 0:
        rank_terms.append(data["class_rank"] <= float(class_rank))
    if global_rank > 0:
        rank_terms.append(data["global_rank"] <= float(global_rank))
    if rank_terms:
        rank_keep = rank_terms[0]
        for term in rank_terms[1:]:
            rank_keep = torch.logical_or(rank_keep, term)
    else:
        rank_keep = torch.ones_like(good, dtype=torch.bool)
    return good & score_keep & rank_keep


def rank_preserve_loss(delta: torch.Tensor,
                       data: dict[str, torch.Tensor],
                       args: argparse.Namespace) -> torch.Tensor:
    """Penalize negative deltas on AP-sensitive correct detections."""
    weight = float(getattr(args, "rank_preserve_weight", 0.0) or 0.0)
    if weight <= 0:
        return delta.sum() * 0.0
    protected = rank_protect_mask(data, args)
    if not torch.any(protected):
        return delta.sum() * 0.0
    margin = max(float(getattr(args, "rank_preserve_margin", 0.0) or 0.0), 0.0)
    excess = torch.relu(-delta[protected] - margin)
    return excess.square().mean() * weight


def evaluate_fit(adapter: GaussianScaleLogitAdapter, rows: list[dict],
                 device: torch.device, args: argparse.Namespace) -> dict:
    if not rows:
        return {"rows": 0}
    data = rows_to_tensors(rows, device)
    with torch.no_grad():
        delta = adapter_delta(adapter, data, args.max_delta_abs)
        adjusted = torch.sigmoid(data["base_logit"] + delta)
        before = data["score"]
        target = data["target_good"]
    bad = target < 0.5
    good = target >= 0.5

    def mean_or_zero(values: torch.Tensor) -> float:
        return float(values.mean().item()) if values.numel() else 0.0

    return {
        "rows": len(rows),
        "good_rows": int(good.sum().item()),
        "bad_rows": int(bad.sum().item()),
        "mean_score_before": mean_or_zero(before),
        "mean_score_after": mean_or_zero(adjusted),
        "mean_delta": mean_or_zero(delta),
        "mean_good_delta": mean_or_zero(delta[good]),
        "mean_bad_delta": mean_or_zero(delta[bad]),
        "mean_good_score_before": mean_or_zero(before[good]),
        "mean_good_score_after": mean_or_zero(adjusted[good]),
        "mean_bad_score_before": mean_or_zero(before[bad]),
        "mean_bad_score_after": mean_or_zero(adjusted[bad]),
    }


def fit_adapter(rows: list[dict], args: argparse.Namespace,
                device: torch.device) -> tuple[GaussianScaleLogitAdapter, dict]:
    train_rows = select_train_rows(rows, args.max_train_rows, args.seed)
    if not train_rows:
        raise RuntimeError("No train rows available for G2 adapter fitting.")
    torch.manual_seed(args.seed)
    adapter = GaussianScaleLogitAdapter(
        hidden=args.adapter_hidden, nonpositive_delta=True).to(device)
    data = rows_to_tensors(train_rows, device)
    weights = torch.where(
        data["target_good"] >= 0.5,
        torch.full_like(data["target_good"], args.good_weight),
        torch.full_like(data["target_good"], args.bad_weight),
    )
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
    n = len(train_rows)
    batch_size = max(int(args.batch_size), 1)
    history = []
    for epoch in range(max(int(args.epochs), 1)):
        order = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            batch = {key: value[idx] for key, value in data.items()}
            batch_weights = weights[idx]
            delta = adapter_delta(adapter, batch, args.max_delta_abs)
            logits = batch["base_logit"] + delta
            loss = F.binary_cross_entropy_with_logits(
                logits, batch["target_good"], weight=batch_weights)
            if args.delta_target_weight > 0 and args.max_delta_abs > 0:
                target_delta = torch.where(
                    batch["target_good"] >= 0.5,
                    torch.zeros_like(delta),
                    torch.full_like(delta, -float(args.max_delta_abs)),
                )
                delta_loss = (delta - target_delta).square() * batch_weights
                loss = loss + float(args.delta_target_weight) * delta_loss.mean()
            if args.delta_l2 > 0:
                loss = loss + float(args.delta_l2) * delta.square().mean()
            loss = loss + rank_preserve_loss(delta, batch, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().item()) * int(idx.numel())
        if epoch == 0 or epoch == args.epochs - 1 or (epoch + 1) % 5 == 0:
            history.append({
                "epoch": epoch + 1,
                "loss": epoch_loss / max(n, 1),
            })
    fit_summary = {
        "selected_train_rows": n,
        "history": history,
        "train_eval": evaluate_fit(adapter, train_rows, device, args),
        "holdout_eval": evaluate_fit(
            adapter, [row for row in rows if not row["train_image"]], device, args),
    }
    return adapter, fit_summary


def tensor_to_device(value, device: torch.device) -> torch.Tensor:
    if hasattr(value, "detach"):
        return value.detach().to(device)
    return torch.as_tensor(value, device=device)


def get_pred_instances(sample):
    if isinstance(sample, dict):
        return sample.get("pred_instances", {})
    return getattr(sample, "pred_instances")


def set_pred_scores(sample, new_scores: torch.Tensor):
    updated = dict(sample) if isinstance(sample, dict) else sample
    pred_instances = get_pred_instances(sample)
    if isinstance(pred_instances, dict):
        pred_updated = dict(pred_instances)
        old_scores = pred_instances["scores"]
        pred_updated["scores"] = new_scores.to(
            device=old_scores.device, dtype=old_scores.dtype)
        updated["pred_instances"] = pred_updated
        return updated
    pred_instances.scores = new_scores.to(
        device=pred_instances.scores.device, dtype=pred_instances.scores.dtype)
    return updated


def apply_adapter(predictions: list, adapter: GaussianScaleLogitAdapter,
                  class_names: tuple[str, ...], priors: dict[str, dict[str, float]],
                  args: argparse.Namespace, device: torch.device) -> tuple[list, dict]:
    output = []
    stats = Counter()
    score_thrs = [float(item) for item in args.score_thrs.split(",") if item]
    delta_values = []
    adapter.eval()
    for sample in predictions:
        pred_instances = get_pred_instances(sample)
        if not pred_instances or len(pred_instances["scores"]) == 0:
            output.append(sample)
            continue
        boxes, labels, scores = sise.get_pred_arrays(sample, 0)
        features = []
        apply_mask = []
        for box, label, score in zip(boxes, labels, scores):
            cls = class_names[int(label)] if 0 <= int(label) < len(class_names) else ""
            feats = prior_features(sise.box_area_rbox(box), cls, priors)
            features.append(feats)
            apply_mask.append(
                feats["valid"]
                and feats["abs_z"] >= args.apply_min_abs_z
                and float(score) >= args.apply_min_score)
        if not features:
            output.append(sample)
            continue
        data = {
            "abs_z": torch.tensor([f["abs_z"] for f in features], dtype=torch.float32, device=device),
            "signed_z": torch.tensor([f["signed_z"] for f in features], dtype=torch.float32, device=device),
            "gaussian_log_prob": torch.tensor(
                [f["gaussian_log_prob"] for f in features], dtype=torch.float32, device=device),
            "log_std": torch.tensor([f["log_std"] for f in features], dtype=torch.float32, device=device),
            "valid": torch.tensor(apply_mask, dtype=torch.bool, device=device),
            "score": torch.tensor(scores, dtype=torch.float32, device=device).clamp(
                min=1e-6, max=1 - 1e-6),
        }
        data["base_logit"] = torch.logit(data["score"])
        with torch.no_grad():
            delta = adapter_delta(adapter, data, args.max_delta_abs)
            new_scores = torch.sigmoid(data["base_logit"] + delta)
        stats["detections_before"] += int(len(scores))
        changed = (delta < -1e-7).detach().cpu()
        stats["detections_changed"] += int(changed.sum().item())
        stats["detections_with_valid_prior"] += int(data["valid"].sum().item())
        for thr in score_thrs:
            tag = str(thr).replace(".", "p")
            before_count = int((data["score"] >= thr).sum().item())
            after_count = int((new_scores >= thr).sum().item())
            stats[f"score_ge_{tag}_before"] += before_count
            stats[f"score_ge_{tag}_after"] += after_count
            stats[f"score_ge_{tag}_dropped"] += before_count - after_count
        if changed.any():
            delta_values.extend(delta[changed.to(device)].detach().cpu().tolist())
        output.append(set_pred_scores(sample, new_scores.detach().cpu()))
    summary = dict(stats)
    summary["mean_changed_delta"] = (
        float(np.mean(delta_values)) if delta_values else 0.0)
    summary["min_changed_delta"] = (
        float(np.min(delta_values)) if delta_values else 0.0)
    return output, summary


def choose_device(raw: str) -> torch.device:
    if raw.startswith("cuda") and torch.cuda.is_available():
        return torch.device(raw)
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    class_names = sise.load_class_names(args.config)
    priors = load_priors(Path(args.class_area_priors_csv))
    with Path(args.predictions).open("rb") as f:
        predictions = pickle.load(f)
    rows, row_stats = build_fit_rows(
        predictions, Path(args.ann_dir), class_names, priors, args)
    adapter, fit_summary = fit_adapter(rows, args, device)
    output, apply_summary = apply_adapter(
        predictions, adapter, class_names, priors, args, device)

    output_pkl = Path(args.output_pkl)
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

    ckpt = {
        "state_dict": adapter.cpu().state_dict(),
        "meta": {
            "adapter_hidden": args.adapter_hidden,
            "nonpositive_delta": True,
            "class_names": list(class_names),
            "class_area_priors_csv": args.class_area_priors_csv,
            "bad_mode": args.bad_mode,
            "iou_thr": args.iou_thr,
            "z_thr": args.z_thr,
            "max_delta_abs": args.max_delta_abs,
            "apply_min_abs_z": args.apply_min_abs_z,
            "apply_min_score": args.apply_min_score,
        },
    }
    adapter_ckpt = Path(args.adapter_ckpt)
    adapter_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, adapter_ckpt)

    summary = {
        "predictions": args.predictions,
        "output_pkl": args.output_pkl,
        "adapter_ckpt": args.adapter_ckpt,
        "ann_dir": args.ann_dir,
        "config": args.config,
        "device": str(device),
        "args": vars(args),
        "row_stats": row_stats,
        "fit_summary": fit_summary,
        "apply_summary": apply_summary,
    }
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + os.linesep,
        encoding="utf-8")
    print(json.dumps({
        "output_pkl": args.output_pkl,
        "adapter_ckpt": args.adapter_ckpt,
        "fit_rows": row_stats.get("fit_rows", 0),
        "selected_train_rows": fit_summary.get("selected_train_rows", 0),
        "detections_changed": apply_summary.get("detections_changed", 0),
        "mean_changed_delta": apply_summary.get("mean_changed_delta", 0.0),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
