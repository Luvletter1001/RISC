#!/usr/bin/env python3
"""Render 1024x1024 court/field -> small-vehicle drift samples."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO = Path("/data1/zcy/OpenRSD")
WORK = REPO / "work_dirs/exp_rotation_gt_shift_taxonomy_20260527"
RESULT = REPO / "resultmd/exp_rotation_gt_shift_taxonomy_20260527"
AUDIT = WORK / "court_to_sv_drift_audit"
OUT = RESULT / "vis_court_to_sv_rotation_samples_1024"
OUT.mkdir(parents=True, exist_ok=True)

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from M_Tools.analysis.run_false_sv_hub_mining_20260526 import angle_sweep_image_path  # noqa: E402
from M_Tools.analysis.run_rsv_taxonomy_suite import load_gt, load_pred  # noqa: E402

TARGET_CLASSES = {
    "tennis-court",
    "basketball-court",
    "soccer-ball-field",
    "baseball-diamond",
    "ground-track-field",
}

PURPLE = (220, 40, 220)
MAGENTA = (255, 0, 255)
CYAN = (255, 220, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def norm_cls(x: str) -> str:
    return x.strip().lower().replace("_", "-")


def draw_poly(img: np.ndarray, poly: np.ndarray, color: tuple[int, int, int], thickness: int = 2, label: str = "") -> None:
    h, w = img.shape[:2]
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    pts_i = np.round(pts).astype(np.int32)
    cv2.polylines(img, [pts_i], True, color, thickness, lineType=cv2.LINE_AA)
    if label:
        x, y = int(pts_i[:, 0].min()), int(pts_i[:, 1].min())
        y = max(18, y - 4)
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def fill_poly_alpha(img: np.ndarray, poly: np.ndarray, color: tuple[int, int, int], alpha: float = 0.22) -> None:
    overlay = img.copy()
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] = np.clip(pts[:, 0], 0, img.shape[1] - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, img.shape[0] - 1)
    cv2.fillPoly(overlay, [np.round(pts).astype(np.int32)], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


def header(img: np.ndarray, lines: list[str]) -> None:
    cv2.rectangle(img, (0, 0), (1024, 86), BLACK, -1)
    for i, line in enumerate(lines[:3]):
        cv2.putText(img, line[:110], (12, 24 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, "GT target=purple | new rotation drift=magenta | persistent target-SV=cyan", (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1, cv2.LINE_AA)


def select_unique_scene(rows: list[dict[str, str]], n: int = 6) -> list[dict[str, str]]:
    # Keep only positive rows, then one tile per source image prefix Pxxxx.
    pos = [r for r in rows if int(float(r.get("max_sv_on_target", 0))) > 0]
    pos.sort(key=lambda r: (int(float(r.get("rot_gain_count", 0))), float(r.get("rot_gain_ratio", 0)), int(float(r.get("total_sv_on_target", 0)))), reverse=True)
    seen = set()
    out = []
    for r in pos:
        scene = r["tile_id"].split("__1024__")[0]
        if scene in seen:
            continue
        seen.add(scene)
        out.append(r)
        if len(out) >= n:
            break
    return out


def main() -> int:
    rank_rows = read_csv(AUDIT / "ftable_court_to_sv_tile_rank.csv")
    event_rows = read_csv(AUDIT / "ftable_court_to_sv_events.csv")
    events_by_tile_angle: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for e in event_rows:
        events_by_tile_angle[(e["tile_id"], int(float(e["angle"])))] .append(e)

    samples = select_unique_scene(rank_rows, n=6)
    index_rows = []
    for rank, row in enumerate(samples, 1):
        tile = row["tile_id"]
        peak = int(float(row["peak_angle"]))
        img_path = angle_sweep_image_path(REPO, tile, peak)
        if img_path is None:
            index_rows.append(dict(rank=rank, tile_id=tile, status="NO_IMAGE"))
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            index_rows.append(dict(rank=rank, tile_id=tile, status="IMREAD_FAIL", image_path=str(img_path)))
            continue
        if img.shape[0] != 1024 or img.shape[1] != 1024:
            img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)

        gt = load_gt(tile, peak, REPO)
        pred = load_pred(REPO, WORK / "pred_cache", tile, peak)
        target_gt_idx = {i for i, t in enumerate(gt["texts"]) if norm_cls(t) in TARGET_CLASSES}

        # Persistent if angle 0 had SV on the same target GT index/class.
        base_events = events_by_tile_angle.get((tile, 0), [])
        persistent_targets = {(int(float(e["gt_idx"])), norm_cls(e["gt_class"])) for e in base_events if e.get("gt_idx", "") != ""}
        peak_events = events_by_tile_angle.get((tile, peak), [])

        for gi in sorted(target_gt_idx):
            cls = norm_cls(gt["texts"][gi])
            fill_poly_alpha(img, gt["polys"][gi], PURPLE, 0.22)
            draw_poly(img, gt["polys"][gi], PURPLE, 2, cls)

        n_new = 0
        n_persist = 0
        for e in peak_events:
            pi = int(float(e["pred_idx"]))
            gi = int(float(e["gt_idx"])) if e.get("gt_idx", "") != "" else -1
            gcls = norm_cls(e.get("gt_class", ""))
            if pi >= len(pred["polys"]):
                continue
            is_persist = (gi, gcls) in persistent_targets
            color = CYAN if is_persist else MAGENTA
            n_persist += int(is_persist)
            n_new += int(not is_persist)
            score = float(e.get("score", 0) or 0)
            state = "persist" if is_persist else "NEW"
            label = f"SV {score:.2f} {state}"
            draw_poly(img, pred["polys"][pi], color, 3 if not is_persist else 2, label)

        header(img, [
            f"rank {rank:02d} scene={tile.split('__1024__')[0]} tile={tile}",
            f"angle 0 -> {peak:03d}; SV-on-target {row['angle0_sv_on_target']} -> {row['max_sv_on_target']} gain={row['rot_gain_count']}",
            f"target={row['dominant_target']} gt_target={row['n_gt_target']} gt_sv={row['n_gt_sv']} new={n_new} persistent={n_persist}",
        ])

        out_name = f"rank{rank:02d}_{tile}_angle{peak:03d}_court_to_sv.png"
        out_path = OUT / out_name
        cv2.imwrite(str(out_path), img)
        index_rows.append(dict(
            rank=rank,
            scene_id=tile.split("__1024__")[0],
            tile_id=tile,
            angle0=row["angle0_sv_on_target"],
            peak_angle=peak,
            peak_sv_on_target=row["max_sv_on_target"],
            rot_gain_count=row["rot_gain_count"],
            dominant_target=row["dominant_target"],
            n_gt_target=row["n_gt_target"],
            n_gt_sv=row["n_gt_sv"],
            new_highlighted=n_new,
            persistent_highlighted=n_persist,
            image_path=str(out_path),
            status="OK",
        ))

    write_csv(OUT / "ftable_vis_court_to_sv_samples.csv", index_rows)
    md = []
    md += ["# Court / Field -> Small-Vehicle Rotation Drift Samples", ""]
    md += ["- status: **DONE**", "- format: one 1024x1024 overlay per source scene prefix; overlapping sibling tiles are deduplicated.", "- color legend: purple target GT; magenta newly appeared rotation drift SV; cyan persistent target-SV already present at angle 0.", ""]
    md += ["| rank | scene_id | tile_id | angle0 | peak_angle | peak_sv_on_target | rot_gain_count | target | gt_sv | image |", "|---|---|---|---:|---:|---:|---:|---|---:|---|"]
    for r in index_rows:
        md.append(f"| {r.get('rank','')} | {r.get('scene_id','')} | {r.get('tile_id','')} | {r.get('angle0','')} | {r.get('peak_angle','')} | {r.get('peak_sv_on_target','')} | {r.get('rot_gain_count','')} | {r.get('dominant_target','')} | {r.get('n_gt_sv','')} | {r.get('image_path','')} |")
    md += ["", "## Notes", "", "These samples are selected by court/field-conditioned drift rather than whole-image small-vehicle ratio, so parking-lot-like true-SV scenes are not promoted unless SV boxes actually fall inside sports/field GT regions."]
    (RESULT / "fres_022_court_to_sv_visual_samples.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"status": "DONE", "n_samples": len(index_rows), "out_dir": str(OUT), "md": str(RESULT / "fres_022_court_to_sv_visual_samples.md")}, indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
