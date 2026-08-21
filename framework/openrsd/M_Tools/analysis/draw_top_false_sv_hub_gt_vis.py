#!/usr/bin/env python3
"""Draw angle-000 GT overlays for top-N false-SV-hub ranked tiles."""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/data1/zcy/OpenRSD")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from M_Tools.analysis.run_false_sv_hub_mining_20260526 import (  # noqa: E402
    ANGLE_SWEEP_REL,
    parse_dota_txt,
    to_float,
)

PALETTE = {
    "plane": (40, 190, 255),
    "small-vehicle": (80, 220, 60),
    "ship": (255, 140, 40),
    "large-vehicle": (230, 80, 180),
    "harbor": (90, 220, 220),
    "storage-tank": (120, 120, 255),
    "bridge": (180, 160, 40),
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def top_false_sv_tiles(work_dir: Path, n: int) -> list[dict]:
    rows = read_csv(work_dir / "fsv2_gt_match/ftable_sv_gt_matching.csv")
    box_sources = {"vis_pkl", "angle_sweep_infer"}
    base = [
        r for r in rows
        if r.get("method") == "baseline" and r.get("pred_source") in box_sources
    ]
    by_tile: dict[str, dict] = {}
    for r in base:
        by_tile.setdefault(r["tile_id"], r)
    ranked = []
    for r in by_tile.values():
        ranked.append({
            **r,
            "_ffr": to_float(r.get("false_sv_final_ratio_iou03")),
            "_hub": to_float(r.get("false_sv_hub_score_iou03")),
        })
    ranked.sort(key=lambda x: (-x["_ffr"], -x["_hub"]))
    out = []
    for i, r in enumerate(ranked[:n], 1):
        out.append({**r, "rank": i})
    return out


def gt_summary(texts: list[str]) -> str:
    c = Counter(texts)
    if not c:
        return "empty"
    parts = [f"{k}x{v}" for k, v in c.most_common(4)]
    return "_".join(parts).replace("-", "")


def draw_gt(
    img_path: Path, ann_path: Path, out_path: Path, rank: int, metrics: dict,
) -> dict:
    texts, polys = parse_dota_txt(ann_path)
    counts = Counter(texts)
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)
    h, w = img.shape[:2]
    for poly, cls in zip(polys, texts):
        pts = poly.reshape(-1, 2).astype(np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, max(w - 1, 0))
        pts[:, 1] = np.clip(pts[:, 1], 0, max(h - 1, 0))
        pts_i = np.round(pts).astype(np.int32)
        color = PALETTE.get(cls, (200, 200, 200))
        cv2.polylines(img, [pts_i], True, color, 2, lineType=cv2.LINE_AA)
        x, y = int(pts_i[:, 0].min()), int(pts_i[:, 1].min())
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), bl = cv2.getTextSize(cls, font, 0.4, 1)
        y = max(y - 4, th + bl + 4)
        x = min(max(x, 0), max(w - tw - 6, 0))
        cv2.rectangle(img, (x, y - th - bl - 4), (min(x + tw + 6, w - 1), y + bl), color, -1)
        cv2.putText(img, cls, (x + 3, y - 3), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    legend = (
        f"rank={rank:03d} | GT n={len(texts)} | "
        + ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        + f" | ffr={metrics.get('false_sv_final_ratio_iou03', '')} "
        f"fsr={metrics.get('final_sv_ratio', '')}"
    )
    cv2.rectangle(img, (6, 6), (min(w - 6, 8 + 9 * len(legend)), 38), (0, 0, 0), -1)
    cv2.putText(img, legend[:120], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return dict(
        rank=rank,
        n_gt=len(texts),
        gt_classes=";".join(f"{k}:{v}" for k, v in sorted(counts.items())),
        out_path=str(out_path),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path,
                    default=REPO / "work_dirs/exp_false_sv_hub_mining_20260526_dota12angle")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--angle", type=int, default=0)
    args = ap.parse_args()

    out_dir = args.out_dir or (
        REPO / "resultmd/exp_false_sv_hub_mining_20260526_dota12angle"
        / f"gt_vis_top{args.top_n}_false_sv_hub_angle{args.angle:03d}"
    )
    tiles = top_false_sv_tiles(args.work_dir, args.top_n)
    index_rows: list[dict] = []
    errors: list[str] = []

    img_root = REPO / ANGLE_SWEEP_REL / f"angle_{args.angle:03d}" / "images"
    ann_root = REPO / ANGLE_SWEEP_REL / f"angle_{args.angle:03d}" / "annfiles"

    for r in tiles:
        tile = r["tile_id"]
        rank = int(r["rank"])
        img_path = ann_path = None
        for ext in (".png", ".jpg"):
            p = img_root / f"{tile}{ext}"
            if p.exists():
                img_path = p
                break
        ann_path = ann_root / f"{tile}.txt"
        if img_path is None or not ann_path.exists():
            errors.append(f"rank{rank:03d} {tile}: missing image or ann")
            continue
        texts, _ = parse_dota_txt(ann_path)
        summ = gt_summary(texts)
        fname = f"rank{rank:03d}_{tile}_angle{args.angle:03d}_GT_{summ}.jpg"
        try:
            meta = draw_gt(img_path, ann_path, out_dir / fname, rank, r)
            index_rows.append({
                "rank": rank,
                "tile_id": tile,
                "n_gt": meta["n_gt"],
                "gt_classes": meta["gt_classes"],
                "false_sv_final_ratio_iou03": r.get("false_sv_final_ratio_iou03"),
                "final_sv_ratio": r.get("final_sv_ratio"),
                "gt_sv_count": r.get("gt_sv_count"),
                "sv_dets": r.get("sv_dets"),
                "tile_type_prelim": r.get("alignment_status", ""),
                "image_path": str(out_dir / fname),
            })
        except Exception as exc:
            errors.append(f"rank{rank:03d} {tile}: {exc}")

    index_path = out_dir / "ftable_gt_vis_index.csv"
    if index_rows:
        keys = list(index_rows[0].keys())
        with index_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(index_rows)

    print(f"saved {len(index_rows)} images -> {out_dir}")
    print(f"index -> {index_path}")
    if errors:
        print(f"errors ({len(errors)}):")
        for e in errors[:10]:
            print(" ", e)
    return 0 if len(index_rows) == len(tiles) else 1


if __name__ == "__main__":
    raise SystemExit(main())
