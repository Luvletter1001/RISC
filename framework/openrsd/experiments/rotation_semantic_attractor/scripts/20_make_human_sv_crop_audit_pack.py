#!/usr/bin/env python3
"""Create a lightweight human crop audit pack from existing prediction/image assets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


LABEL_COLUMNS = [
    "crop_id",
    "image_path",
    "tile_id",
    "angle",
    "model",
    "checkpoint",
    "pred_score",
    "pred_box",
    "best_gt_class",
    "best_gt_iou",
    "risk_group",
    "auto_label",
    "human_label",
    "human_confidence",
    "human_notes",
]


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if headers is None:
        headers = []
        for row in rows:
            for key in row:
                if key not in headers:
                    headers.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in headers})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def ann_has_sv(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    return "small-vehicle" in text


def polygon_bounds(pred: dict[str, Any]) -> tuple[float, float, float, float] | None:
    pts = pred.get("polygon")
    if not pts and pred.get("box"):
        box = pred.get("box")
        pts = [[box[i], box[i + 1]] for i in range(0, len(box), 2)] if len(box) >= 8 else None
    if not pts:
        return None
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def iter_prediction_files(prediction_root: Path, models: list[str]) -> list[Path]:
    files: list[Path] = []
    for model in models:
        root = prediction_root / model
        if not root.exists():
            continue
        files.extend(sorted(root.glob("*/*.json")))
    return files


def image_for(image_root: Path, tile_id: str, angle: int) -> Path:
    return image_root / f"angle_{angle:03d}" / "images" / f"{tile_id}.png"


def gt_for(gt_root: Path, tile_id: str, angle: int) -> Path:
    return gt_root / f"angle_{angle:03d}" / "annfiles" / f"{tile_id}.txt"


def draw_crop(image_path: Path, pred: dict[str, Any], out_path: Path, meta: dict[str, Any]) -> bool:
    bounds = polygon_bounds(pred)
    if bounds is None or not image_path.exists():
        return False
    img = plt.imread(str(image_path))
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bounds
    pad = max(32, 0.5 * max(x2 - x1, y2 - y1))
    cx1 = max(0, int(math.floor(x1 - pad)))
    cy1 = max(0, int(math.floor(y1 - pad)))
    cx2 = min(w, int(math.ceil(x2 + pad)))
    cy2 = min(h, int(math.ceil(y2 + pad)))
    if cx2 <= cx1 or cy2 <= cy1:
        return False
    crop = img[cy1:cy2, cx1:cx2]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(crop)
    rect = patches.Rectangle((x1 - cx1, y1 - cy1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2)
    ax.add_patch(rect)
    title = f"{meta['model']} {meta['tile_id']} angle={meta['angle']} score={meta['pred_score']:.3f}"
    ax.set_title(title, fontsize=7)
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--metrics-root", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-crops", type=int, default=500)
    parser.add_argument("--max-crops-per-tile", type=int, default=20)
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=20260601)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    crop_root = output_dir / "crops"
    audit_dir = output_dir / "audit"
    for sub in ["unmatched_sv", "strict_object_flip", "true_sv_positive_control", "dehub_baseline", "dehub_repair", "montage"]:
        (crop_root / sub).mkdir(parents=True, exist_ok=True)

    prediction_root = Path(args.prediction_root)
    image_root = Path(args.image_root)
    gt_root = Path(args.gt_root)
    if not prediction_root.exists() or not image_root.exists() or not gt_root.exists():
        status = {
            "status": "NOT_AVAILABLE_ASSET",
            "prediction_root_exists": prediction_root.exists(),
            "image_root_exists": image_root.exists(),
            "gt_root_exists": gt_root.exists(),
            "blocker": "prediction, image, or GT root missing",
        }
        write_json(audit_dir / "human_crop_audit_pack_status.json", status)
        write_csv(audit_dir / "human_sv_crop_audit_template.csv", [], LABEL_COLUMNS)
        write_csv(audit_dir / "unannotated_sv_sample_manifest.csv", [], LABEL_COLUMNS + ["sample_status", "blocker"])
        return 0

    models = [p.name for p in sorted(prediction_root.iterdir()) if p.is_dir()]
    files = iter_prediction_files(prediction_root, models)
    rng.shuffle(files)

    rows: list[dict[str, Any]] = []
    per_tile: Counter[str] = Counter()
    unmatched_count = 0
    positive_count = 0
    blockers = {
        "strict_object_flip": "NOT_AVAILABLE_AUTO_MATCH: crop script does not have pred-to-GT best-overlap labels",
        "dehub_baseline": "NOT_AVAILABLE_ASSET: DeHub row-level outputs do not include raw box crops in this artifact set",
        "dehub_repair": "NOT_AVAILABLE_ASSET: DeHub row-level outputs do not include raw box crops in this artifact set",
    }

    for pred_file in files:
        if len(rows) >= args.max_crops:
            break
        data = load_json(pred_file)
        preds = data.get("final_predictions", [])
        if not preds:
            continue
        model = pred_file.parts[-3]
        tile_id = pred_file.parts[-2]
        angle_match = pred_file.stem.replace("angle_", "")
        try:
            angle = int(angle_match)
        except ValueError:
            continue
        img = image_for(image_root, tile_id, angle)
        gt = gt_for(gt_root, tile_id, angle)
        has_sv_gt = ann_has_sv(gt)
        candidates = [
            p for p in preds
            if p.get("class_name") == "small-vehicle" and float(p.get("score", 0.0)) >= args.score_thr
        ]
        candidates.sort(key=lambda p: float(p.get("score", 0.0)), reverse=True)
        for pred in candidates[:5]:
            if len(rows) >= args.max_crops:
                break
            if per_tile[tile_id] >= args.max_crops_per_tile:
                continue
            if not has_sv_gt and unmatched_count <= positive_count + 5:
                bucket = "unmatched_sv"
                auto_label = "unmatched_sv_no_annotated_sv_gt"
                unmatched_count += 1
            elif has_sv_gt:
                bucket = "true_sv_positive_control"
                auto_label = "true_sv_rich_positive_control"
                positive_count += 1
            else:
                continue
            crop_id = f"{bucket}_{len(rows):05d}"
            out_path = crop_root / bucket / f"{crop_id}.png"
            score = float(pred.get("score", 0.0))
            meta = {"model": model, "tile_id": tile_id, "angle": angle, "pred_score": score}
            if not draw_crop(img, pred, out_path, meta):
                continue
            per_tile[tile_id] += 1
            rows.append({
                "crop_id": crop_id,
                "image_path": str(out_path),
                "tile_id": tile_id,
                "angle": angle,
                "model": model,
                "checkpoint": "closedset_full",
                "pred_score": f"{score:.6f}",
                "pred_box": json.dumps(pred.get("box", [])),
                "best_gt_class": "NOT_COMPUTED",
                "best_gt_iou": "NOT_COMPUTED",
                "risk_group": "no_annotated_sv_gt" if not has_sv_gt else "true_sv_rich",
                "auto_label": auto_label,
                "human_label": "",
                "human_confidence": "",
                "human_notes": "",
            })

    write_csv(audit_dir / "human_sv_crop_audit_template.csv", rows, LABEL_COLUMNS)
    manifest_headers = LABEL_COLUMNS + ["sample_status", "blocker"]
    manifest_rows = [{**r, "sample_status": "CROP_CREATED", "blocker": ""} for r in rows]
    for name, blocker in blockers.items():
        manifest_rows.append({
            "crop_id": name,
            "auto_label": name,
            "sample_status": "NOT_AVAILABLE_ASSET",
            "blocker": blocker,
        })
    write_csv(audit_dir / "unannotated_sv_sample_manifest.csv", manifest_rows, manifest_headers)
    write_json(audit_dir / "human_crop_audit_pack_status.json", {
        "status": "PARTIAL_CROP_PACK_CREATED" if rows else "NOT_AVAILABLE_ASSET",
        "created_crops": len(rows),
        "unmatched_sv_crops": unmatched_count,
        "true_sv_positive_control_crops": positive_count,
        "blockers": blockers,
        "template": str(audit_dir / "human_sv_crop_audit_template.csv"),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
