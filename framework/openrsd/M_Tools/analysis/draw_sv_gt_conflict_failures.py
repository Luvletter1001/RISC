#!/usr/bin/env python3
"""Render clean small-vehicle predictions that conflict with non-SV GT."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path("/data1/zcy/OpenRSD")
WORK = REPO / "work_dirs/exp_rotation_gt_shift_taxonomy_20260527"
RESULT = REPO / "resultmd/exp_rotation_gt_shift_taxonomy_20260527"

cv2 = None
np = None

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SV_RED = (0, 0, 255)
GRAY = (155, 155, 155)
DARK_GRAY = (88, 88, 88)
LIGHT_GRAY = (210, 210, 210)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

EXCLUDED_GT_CLASSES = frozenset({
    "",
    "bg",
    "background",
    "small-vehicle",
    "small vehicle",
    "small_vehicle",
    "large-vehicle",
    "large vehicle",
    "large_vehicle",
})


def ensure_image_deps() -> None:
    """Import image dependencies only for rendering, not for filter tests."""
    global cv2, np
    if cv2 is not None and np is not None:
        return
    import cv2 as cv2_mod
    import numpy as np_mod

    cv2 = cv2_mod
    np = np_mod


def norm_cls(cls: str) -> str:
    return str(cls).strip().lower().replace("_", "-")


def is_sv_label_local(cls: str) -> bool:
    return norm_cls(cls) == "small-vehicle"


def is_eligible_conflict_gt(cls: str) -> bool:
    return norm_cls(cls) not in {norm_cls(c) for c in EXCLUDED_GT_CLASSES}


def is_sv_gt_conflict_event(row: dict[str, Any]) -> bool:
    """True for SV predictions on real non-SV, non-LV GT conflicts."""
    return row.get("event") == "sv_case1" and is_eligible_conflict_gt(row.get("gt_class", ""))


def is_strict_sv_misclassification(
    pred_class: str,
    gt_class: str,
    iou: float,
    min_iou: float,
) -> bool:
    """A predicted SV box strongly overlaps an annotated non-SV/non-LV GT."""
    return (
        is_sv_label_local(pred_class)
        and is_eligible_conflict_gt(gt_class)
        and float(iou) >= float(min_iou)
    )


def read_events(path: Path) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    return [r for r in rows if is_sv_gt_conflict_event(r)]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", errors="replace") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def draw_poly(
    img: Any,
    poly: Any,
    color: tuple[int, int, int],
    thickness: int = 1,
    label: str = "",
) -> None:
    ensure_image_deps()
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    h, w = img.shape[:2]
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    pts_i = np.round(pts).astype(np.int32)
    cv2.polylines(img, [pts_i], True, color, thickness, lineType=cv2.LINE_AA)
    if not label:
        return
    x, y = int(pts_i[:, 0].min()), int(pts_i[:, 1].min())
    y = max(18, y - 4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.34
    (tw, th), bl = cv2.getTextSize(label, font, scale, 1)
    x2 = min(w - 1, x + tw + 6)
    cv2.rectangle(img, (x, y - th - bl - 4), (x2, y + bl), color, -1)
    cv2.putText(img, label, (x + 3, y - 3), font, scale, WHITE, 1, cv2.LINE_AA)


def draw_header(img: Any, lines: list[str]) -> None:
    # Do not draw an in-image title bar. Some failure boxes sit at the top
    # image border; a header would hide exactly the evidence being audited.
    return


def class_breakdown(events: list[dict[str, str]]) -> str:
    counts = Counter(norm_cls(e.get("gt_class", "")) for e in events)
    return ";".join(f"{k}:{v}" for k, v in counts.most_common())


def collect_strict_misclassification_events(
    repo: Path,
    work_dir: Path,
    min_iou: float,
    tile_csv: Path | None = None,
) -> list[dict[str, Any]]:
    from M_Tools.analysis.run_rsv_taxonomy_suite import (
        ANGLES_12,
        iou_matrix_rotated,
        load_gt,
        load_pred,
    )

    if tile_csv is None:
        for candidate in (
            work_dir / "ftable_extra_tiles.csv",
            work_dir / "ftable_cohort_2500.csv",
            work_dir / "ftable_cohort_500.csv",
        ):
            if candidate.exists():
                tile_csv = candidate
                break
    if tile_csv is None or not tile_csv.exists():
        raise FileNotFoundError(
            f"no tile csv found under {work_dir}; pass --tile-csv explicitly",
        )
    tiles = [r["tile_id"] for r in read_csv_rows(tile_csv)]
    cache_dir = work_dir / "pred_cache"
    events: list[dict[str, Any]] = []
    for n, tile in enumerate(tiles, 1):
        for angle in ANGLES_12:
            gt = load_gt(tile, angle, repo)
            pred = load_pred(repo, cache_dir, tile, angle)
            sv_idx = [i for i, cls in enumerate(pred["texts"]) if is_sv_label_local(cls)]
            gt_idx = [i for i, cls in enumerate(gt["texts"]) if is_eligible_conflict_gt(cls)]
            if not sv_idx or not gt_idx:
                continue
            ious = iou_matrix_rotated(pred["polys"][sv_idx], gt["polys"][gt_idx])
            for si, pred_i in enumerate(sv_idx):
                best_j = int(ious[si].argmax()) if ious.shape[1] else -1
                if best_j < 0:
                    continue
                gt_i = gt_idx[best_j]
                best_iou = float(ious[si, best_j])
                pred_cls = pred["texts"][pred_i]
                gt_cls = gt["texts"][gt_i]
                if not is_strict_sv_misclassification(pred_cls, gt_cls, best_iou, min_iou):
                    continue
                score = float(pred["scores"][pred_i]) if pred_i < len(pred.get("scores", [])) else ""
                events.append(dict(
                    tile_id=tile,
                    angle=angle,
                    tile_csv=str(tile_csv),
                    event="strict_sv_miscls",
                    pred_idx=pred_i,
                    pred_class=pred_cls,
                    score=score,
                    gt_idx=gt_i,
                    gt_class=norm_cls(gt_cls),
                    iou=best_iou,
                    min_iou=min_iou,
                ))
        if n % 500 == 0:
            print(f"scanned {n}/{len(tiles)}", flush=True)
    return events


def render_one(
    repo: Path,
    work_dir: Path,
    out_dir: Path,
    rank: int,
    tile: str,
    angle: int,
    events: list[dict[str, str]],
) -> dict[str, Any]:
    ensure_image_deps()
    from M_Tools.analysis.run_false_sv_hub_mining_20260526 import angle_sweep_image_path
    from M_Tools.analysis.run_rsv_taxonomy_suite import load_gt, load_pred

    img_path = angle_sweep_image_path(repo, tile, angle)
    if img_path is None:
        return dict(status="ERR", rank=rank, tile_id=tile, angle=angle, error="no_image")
    img = cv2.imread(str(img_path))
    if img is None:
        return dict(status="ERR", rank=rank, tile_id=tile, angle=angle, error="imread_fail")
    if img.shape[:2] != (1024, 1024):
        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)

    cache_dir = work_dir / "pred_cache"
    pred = load_pred(repo, cache_dir, tile, angle)
    gt = load_gt(tile, angle, repo)

    conflict_pred_idx = {int(float(e["pred_idx"])) for e in events}
    conflict_gt_idx = {
        int(float(e["gt_idx"]))
        for e in events
        if str(e.get("gt_idx", "")).strip() not in ("", "-1")
    }

    # Draw conflicting GT first as gray context. All non-red boxes remain gray.
    for gi in sorted(conflict_gt_idx):
        if 0 <= gi < len(gt["polys"]):
            cls = norm_cls(gt["texts"][gi])
            draw_poly(img, gt["polys"][gi], DARK_GRAY, 2, f"GT {cls[:14]}")

    n_pred_sv = 0
    n_pred_non_sv = 0
    n_red = 0
    scores = pred.get("scores", [])
    deferred_red: list[tuple[int, str, Any, float, dict[str, str]]] = []
    for pi, (cls, poly) in enumerate(zip(pred["texts"], pred["polys"])):
        score = float(scores[pi]) if pi < len(scores) else 0.0
        if is_sv_label_local(cls):
            n_pred_sv += 1
            is_conflict = pi in conflict_pred_idx
            if is_conflict:
                n_red += 1
                hit = next(e for e in events if int(float(e["pred_idx"])) == pi)
                deferred_red.append((pi, cls, poly, score, hit))
            else:
                draw_poly(img, poly, GRAY, 1, f"SV {score:.2f}")
        else:
            n_pred_non_sv += 1
            draw_poly(img, poly, GRAY, 1, norm_cls(cls)[:12])

    # Draw conflict SV boxes last so gray context boxes cannot cover the red failure marker.
    for _pi, _cls, poly, _score, hit in deferred_red:
        gt_cls = norm_cls(hit.get("gt_class", ""))
        iou = float(hit.get("iou", 0.0) or 0.0)
        label = f"SV>{gt_cls[:10]} {iou:.2f}"
        draw_poly(img, poly, SV_RED, 5, label)

    breakdown = class_breakdown(events)
    draw_header(img, [
        f"Strict SV misclassification | rank={rank:03d} tile={tile} angle={angle:03d}",
        f"red=SV prediction strongly overlapping annotated non-SV GT; gray=all other predictions + GT context",
        f"red_sv={len(events)} pred_sv={n_pred_sv} pred_other={n_pred_non_sv} gt_conflict={breakdown}",
    ])

    fname = f"rank{rank:03d}_{tile}_angle{angle:03d}_strict_sv_miscls{len(events):02d}.jpg"
    out_path = out_dir / fname
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return dict(
        status="OK",
        rank=rank,
        tile_id=tile,
        angle=f"{angle:03d}",
        n_conflict_sv=len(events),
        n_red_conflict_sv=n_red,
        n_pred_sv=n_pred_sv,
        n_pred_non_sv=n_pred_non_sv,
        gt_conflict_breakdown=breakdown,
        image_path=str(out_path),
    )


def clean_output_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for pat in ("*.jpg", "*.png", "*.csv", "*.md", "*.json"):
        for p in out_dir.glob(pat):
            p.unlink()


def write_markdown(result_path: Path, out_dir: Path, image_rows: list[dict[str, Any]], event_rows: list[dict[str, Any]]) -> None:
    gt_counts = Counter(norm_cls(e.get("gt_class", "")) for e in event_rows)
    lines = [
        "# Clean SV-GT Conflict Failure Visualizations",
        "",
        "- status: **DONE**",
        "- selection: predicted class is `small-vehicle`, best overlapping GT is not small-vehicle, large-vehicle, or background, and rotated IoU passes the configured threshold.",
        "- rendering: only strict misclassified SV prediction boxes are red; all other prediction classes, including non-conflict SV boxes, are gray. Conflicting GT context is gray.",
        f"- image_dir: `{out_dir}`",
        "",
        "## Counts",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| strict_misclassification_events | {len(event_rows)} |",
        f"| rendered_tile_angles | {len(image_rows)} |",
        f"| conflict_tiles | {len({r['tile_id'] for r in image_rows})} |",
        "",
        "## GT Conflict Classes",
        "",
        "| gt_class | events |",
        "|---|---:|",
    ]
    for cls, n in gt_counts.most_common():
        lines.append(f"| {cls} | {n} |")
    lines += [
        "",
        "## Files",
        "",
        f"- `{out_dir / 'ftable_sv_gt_conflict_images.csv'}`",
        f"- `{out_dir / 'ftable_sv_gt_conflict_events.csv'}`",
        f"- `{out_dir / 'fjson_sv_gt_conflict_summary.json'}`",
    ]
    result_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=REPO)
    ap.add_argument("--work-dir", type=Path, default=WORK)
    ap.add_argument("--result-dir", type=Path, default=RESULT)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--tile-csv", type=Path, default=None)
    ap.add_argument("--min-iou", type=float, default=0.70)
    args = ap.parse_args()

    iou_tag = f"{int(round(args.min_iou * 100)):03d}"
    out_dir = args.out_dir or (args.result_dir / f"vis_sv_gt_miscls_iou{iou_tag}_clean")
    clean_output_dir(out_dir)

    events_path = args.work_dir / "ftable_case_sv_events.csv"
    events = collect_strict_misclassification_events(
        args.repo_root,
        args.work_dir,
        args.min_iou,
        args.tile_csv,
    )
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for e in events:
        grouped[(e["tile_id"], int(float(e["angle"])))].append(e)

    jobs = sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0][0], kv[0][1]))
    image_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for rank, ((tile, angle), evs) in enumerate(jobs, 1):
        row = render_one(args.repo_root, args.work_dir, out_dir, rank, tile, angle, evs)
        if row.get("status") == "OK":
            image_rows.append(row)
        else:
            errors.append(row)
        if rank % 25 == 0 or rank == len(jobs):
            print(f"rendered {rank}/{len(jobs)}", flush=True)

    event_rows: list[dict[str, Any]] = []
    image_by_key = {(r["tile_id"], int(r["angle"])): r["image_path"] for r in image_rows}
    for e in events:
        rr = dict(e)
        rr["gt_class"] = norm_cls(rr.get("gt_class", ""))
        rr["angle"] = f"{int(float(rr['angle'])):03d}"
        rr["image_path"] = image_by_key.get((rr["tile_id"], int(rr["angle"])), "")
        event_rows.append(rr)

    write_csv_rows(out_dir / "ftable_sv_gt_conflict_images.csv", image_rows)
    write_csv_rows(out_dir / "ftable_sv_gt_conflict_events.csv", event_rows)

    summary = {
        "status": "DONE" if not errors else "PARTIAL",
        "event_source": "computed_from_pred_cache_and_gt",
        "legacy_event_table": str(events_path),
        "tile_csv": str(args.tile_csv) if args.tile_csv else "auto",
        "min_iou": args.min_iou,
        "n_strict_misclassification_events": len(events),
        "n_rendered_tile_angles": len(image_rows),
        "n_errors": len(errors),
        "out_dir": str(out_dir),
        "gt_conflict_classes": dict(Counter(norm_cls(e.get("gt_class", "")) for e in events)),
        "errors": errors[:20],
    }
    (out_dir / "fjson_sv_gt_conflict_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path = args.result_dir / f"fres_024_strict_sv_misclassification_iou{iou_tag}_visualization.md"
    write_markdown(md_path, out_dir, image_rows, event_rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"markdown -> {md_path}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
