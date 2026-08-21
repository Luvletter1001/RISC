#!/usr/bin/env python3
"""Find rotation-induced court/field -> small-vehicle semantic drift cases."""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path("/data1/zcy/OpenRSD")
WORK = REPO / "work_dirs/exp_rotation_gt_shift_taxonomy_20260527"
RESULT = REPO / "resultmd/exp_rotation_gt_shift_taxonomy_20260527"
OUT = WORK / "court_to_sv_drift_audit"
OUT.mkdir(parents=True, exist_ok=True)
RESULT.mkdir(parents=True, exist_ok=True)

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from M_Tools.analysis.run_rsv_taxonomy_suite import ANGLES_12, load_gt, load_pred, read_csv, write_csv, is_sv_label  # noqa: E402

TARGET_CLASSES = {
    "tennis-court",
    "basketball-court",
    "soccer-ball-field",
    "baseball-diamond",
    "ground-track-field",
}
P0148 = "P0148__1024__651___0"


def norm_cls(x: str) -> str:
    return x.strip().lower().replace("_", "-")


def point_in_poly(x: float, y: float, poly: np.ndarray) -> bool:
    pts = np.asarray(poly, dtype=float).reshape(-1, 2)
    inside = False
    n = len(pts)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def centroid(poly: np.ndarray) -> tuple[float, float]:
    pts = np.asarray(poly, dtype=float).reshape(-1, 2)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def load_tiles() -> list[str]:
    p = WORK / "ftable_cohort_2500.csv"
    tiles = [r["tile_id"] for r in read_csv(p)]
    if P0148 not in tiles:
        # Include P0148 opportunistically if its prediction cache exists.
        cache = WORK / "pred_cache" / P0148
        if cache.exists():
            tiles.append(P0148)
    return tiles


def analyze_tile_angle(tile: str, angle: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gt = load_gt(tile, angle, REPO)
    pred = load_pred(REPO, WORK / "pred_cache", tile, angle)
    target_idx = [i for i, t in enumerate(gt["texts"]) if norm_cls(t) in TARGET_CLASSES]
    gt_sv_count = sum(1 for t in gt["texts"] if is_sv_label(t))
    pred_sv_idx = [i for i, t in enumerate(pred["texts"]) if is_sv_label(t)]

    events = []
    by_target = Counter()
    for pi in pred_sv_idx:
        cx, cy = centroid(pred["polys"][pi])
        hits = []
        for gi in target_idx:
            if point_in_poly(cx, cy, gt["polys"][gi]):
                hits.append((gi, gt["texts"][gi]))
        if not hits:
            continue
        gi, gcls = hits[0]
        by_target[norm_cls(gcls)] += 1
        events.append(dict(
            tile_id=tile,
            angle=angle,
            pred_idx=pi,
            pred_class=pred["texts"][pi],
            score=float(pred["scores"][pi]) if len(pred.get("scores", [])) > pi else "",
            pred_cx=f"{cx:.2f}",
            pred_cy=f"{cy:.2f}",
            gt_idx=gi,
            gt_class=norm_cls(gcls),
        ))

    row = dict(
        tile_id=tile,
        angle=angle,
        n_gt=len(gt["texts"]),
        n_gt_sv=gt_sv_count,
        n_gt_target=len(target_idx),
        n_pred=len(pred["texts"]),
        n_pred_sv=len(pred_sv_idx),
        n_sv_on_target=len(events),
        sv_on_target_ratio=(len(events) / len(pred_sv_idx)) if pred_sv_idx else 0.0,
    )
    for cls in sorted(TARGET_CLASSES):
        row[f"sv_on_{cls}"] = by_target[cls]
    return row, events


def main() -> int:
    tiles = load_tiles()
    tile_angle_rows = []
    event_rows = []
    for n, tile in enumerate(tiles, 1):
        for angle in ANGLES_12:
            row, events = analyze_tile_angle(tile, angle)
            tile_angle_rows.append(row)
            event_rows.extend(events)
        if n % 250 == 0:
            print(f"processed {n}/{len(tiles)}", flush=True)

    write_csv(OUT / "ftable_court_to_sv_tile_angle.csv", tile_angle_rows)
    write_csv(OUT / "ftable_court_to_sv_events.csv", event_rows)

    by_tile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in tile_angle_rows:
        by_tile[r["tile_id"]].append(r)

    tile_rows = []
    for tile, rows in by_tile.items():
        by_ang = {int(r["angle"]): r for r in rows}
        r0 = by_ang.get(0, {})
        c0 = int(r0.get("n_sv_on_target", 0))
        f0 = float(r0.get("sv_on_target_ratio", 0.0))
        max_row = max(rows, key=lambda r: (int(r["n_sv_on_target"]), float(r["sv_on_target_ratio"])))
        max_count = int(max_row["n_sv_on_target"])
        max_frac = float(max_row["sv_on_target_ratio"])
        total = sum(int(r["n_sv_on_target"]) for r in rows)
        target_gt = max(int(r["n_gt_target"]) for r in rows)
        gt_sv = max(int(r["n_gt_sv"]) for r in rows)
        class_counts = Counter()
        for r in rows:
            for cls in TARGET_CLASSES:
                v = int(r.get(f"sv_on_{cls}", 0))
                if v:
                    class_counts[cls] += v
        tile_rows.append(dict(
            tile_id=tile,
            total_sv_on_target=total,
            angle0_sv_on_target=c0,
            max_sv_on_target=max_count,
            rot_gain_count=max_count - c0,
            angle0_sv_on_target_ratio=f"{f0:.4f}",
            max_sv_on_target_ratio=f"{max_frac:.4f}",
            rot_gain_ratio=f"{(max_frac - f0):.4f}",
            peak_angle=int(max_row["angle"]),
            n_gt_target=target_gt,
            n_gt_sv=gt_sv,
            dominant_target=class_counts.most_common(1)[0][0] if class_counts else "",
            dominant_target_hits=class_counts.most_common(1)[0][1] if class_counts else 0,
            target_breakdown=";".join(f"{k}:{v}" for k, v in class_counts.most_common() if v),
            is_p0148=int(tile == P0148),
        ))

    tile_rows.sort(key=lambda r: (int(r["rot_gain_count"]), float(r["rot_gain_ratio"]), int(r["total_sv_on_target"])), reverse=True)
    write_csv(OUT / "ftable_court_to_sv_tile_rank.csv", tile_rows)

    non_p0148 = [r for r in tile_rows if r["tile_id"] != P0148]
    positive = [r for r in non_p0148 if int(r["max_sv_on_target"]) > 0]
    rot_positive = [r for r in non_p0148 if int(r["rot_gain_count"]) > 0]
    strong = [r for r in non_p0148 if int(r["rot_gain_count"]) >= 3 or float(r["rot_gain_ratio"]) >= 0.05]

    summary = dict(
        n_tiles=len(tiles),
        n_tiles_non_p0148=len(non_p0148),
        n_tile_angles=len(tile_angle_rows),
        n_events=len(event_rows),
        n_tiles_with_any_court_to_sv=len(positive),
        n_tiles_with_rotation_gain=len(rot_positive),
        n_strong_candidates=len(strong),
        p0148_included=P0148 in by_tile,
        top_non_p0148=positive[:20],
    )
    (OUT / "fjson_court_to_sv_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    def md_table(rows, fields, limit=None):
        if limit is not None:
            rows = rows[:limit]
        out = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
        for r in rows:
            out.append("| " + " | ".join(str(r.get(f, "")) for f in fields) + " |")
        return "\n".join(out)

    md = []
    md += ["# Court / Field -> Small-Vehicle Drift Audit", ""]
    md += ["- status: **DONE**", f"- work_dir: `{OUT}`", "- definition: SV prediction centroid inside sports/court/field GT polygon; ranked by rotation gain versus angle 0.", "- target GT classes: `baseball-diamond`, `basketball-court`, `ground-track-field`, `soccer-ball-field`, `tennis-court`", ""]
    md += ["## Why This Replaces Whole-Image SV Ratio", "", "Whole-image small-vehicle ratio is the wrong filter for this question because true parking lots naturally produce mostly small-vehicle detections. This audit conditions on non-SV court/field GT regions and asks whether SV detections appear inside those regions after rotation.", ""]
    md += ["## Summary", "", "| metric | value |", "|---|---:|"]
    for key in ["n_tiles", "n_tile_angles", "n_events", "n_tiles_with_any_court_to_sv", "n_tiles_with_rotation_gain", "n_strong_candidates", "p0148_included"]:
        md.append(f"| {key} | {summary[key]} |")
    md.append("")
    md += ["## Top Non-P0148 Candidates", "", md_table(positive, ["tile_id", "total_sv_on_target", "angle0_sv_on_target", "max_sv_on_target", "rot_gain_count", "angle0_sv_on_target_ratio", "max_sv_on_target_ratio", "rot_gain_ratio", "peak_angle", "n_gt_target", "n_gt_sv", "dominant_target", "target_breakdown"], 30), ""]
    md += ["## Interpretation", "", "- These are the images to inspect for the phenomenon you described: a sports/court/field object becomes small-vehicle after rotation.", "- The ranking intentionally ignores images whose only signal is high whole-image SV ratio.", "- `n_gt_sv` is shown only as a caveat: if a tile also has many true small vehicles, inspect the visual before using it as evidence.", "- Strong candidates are those with at least 3 additional SV-on-target detections after rotation, or a target-conditioned SV ratio gain >= 0.05.", ""]
    md += ["## Outputs", "", "- " + str(OUT / "ftable_court_to_sv_tile_rank.csv"), "- " + str(OUT / "ftable_court_to_sv_tile_angle.csv"), "- " + str(OUT / "ftable_court_to_sv_events.csv"), "- " + str(OUT / "fjson_court_to_sv_summary.json"), ""]
    (RESULT / "fres_021_court_field_to_sv_rotation_drift_audit.md").write_text("\n".join(md))
    print(json.dumps({"status": "DONE", "summary": summary, "md": str(RESULT / "fres_021_court_field_to_sv_rotation_drift_audit.md")}, indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
