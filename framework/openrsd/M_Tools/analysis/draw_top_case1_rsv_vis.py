#!/usr/bin/env python3
"""Side-by-side GT (left) / Pred (right) vis for top-N case1_sv_fraction tiles."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/data1/zcy/OpenRSD")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from M_Tools.analysis.run_false_sv_hub_mining_20260526 import (  # noqa: E402
    angle_sweep_image_path,
    parse_dota_txt,
)
from M_Tools.analysis.run_rsv_taxonomy_suite import (  # noqa: E402
    ANGLES_12,
    IOU_WEAK,
    analyze_tile_angle,
    iou_matrix_rotated,
    is_sv_label,
    load_pred,
    read_csv,
    write_csv,
)

WRONG_BGR = (0, 0, 255)  # class mismatch vs eligible GT
# Only visualize GT / preds tied to annotated GT that is not background / large-vehicle.
GT_VIS_EXCLUDE = frozenset({"large-vehicle", "background", "bg"})

GT_PALETTE = {
    "plane": (40, 190, 255),
    "small-vehicle": (80, 220, 60),
    "ship": (255, 140, 40),
    "large-vehicle": (230, 80, 180),
    "harbor": (90, 220, 220),
    "storage-tank": (120, 120, 255),
    "bridge": (180, 160, 40),
    "swimming-pool": (255, 100, 200),
}

# Same palette for raw inference vis (class-colored boxes + labels).
PRED_PALETTE = dict(GT_PALETTE)

SV_COLORS = {
    "sv_case1": (0, 0, 255),
    "sv_bg": (200, 200, 200),
    "sv_residual": (0, 140, 255),
}


def _draw_poly(
    img: np.ndarray,
    poly: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    label: str | None = None,
) -> None:
    h, w = img.shape[:2]
    pts = poly.reshape(-1, 2).astype(np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, max(w - 1, 0))
    pts[:, 1] = np.clip(pts[:, 1], 0, max(h - 1, 0))
    pts_i = np.round(pts).astype(np.int32)
    cv2.polylines(img, [pts_i], True, color, thickness, lineType=cv2.LINE_AA)
    if not label:
        return
    x, y = int(pts_i[:, 0].min()), int(pts_i[:, 1].min())
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(label, font, 0.35, 1)
    y = max(y - 4, th + bl + 4)
    cv2.rectangle(img, (x, y - th - bl - 4), (x + tw + 6, y + bl), color, -1)
    cv2.putText(img, label, (x + 3, y - 3), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)


def _panel_header(img: np.ndarray, title: str, subtitle: str = "") -> None:
    h, w = img.shape[:2]
    bar_h = 52 if subtitle else 36
    cv2.rectangle(img, (0, 0), (w, bar_h), (0, 0, 0), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, title, (10, 24), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(img, subtitle, (10, 46), font, 0.42, (200, 220, 255), 1, cv2.LINE_AA)


def _merge_lr(left: np.ndarray, right: np.ndarray, gutter: int = 4) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + gutter + right.shape[1]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[: left.shape[0], : left.shape[1]] = left
    out[: right.shape[0], left.shape[1] + gutter :] = right
    if gutter > 0:
        out[:, left.shape[1] : left.shape[1] + gutter] = (48, 48, 48)
    return out


def _class_color(cls: str) -> tuple[int, int, int]:
    return PRED_PALETTE.get(cls, (180, 180, 180))


def _class_label(cls: str, score: float | None = None) -> str:
    short = cls[:12] if len(cls) > 12 else cls
    if score is None:
        return short
    return f"{short}:{score:.2f}"


def _norm_cls(cls: str) -> str:
    return cls.strip().lower().replace("_", "-")


def is_vis_eligible_gt(cls: str) -> bool:
    """GT with a real class label, excluding background and large-vehicle."""
    c = _norm_cls(cls)
    return bool(c) and c not in GT_VIS_EXCLUDE


def _eligible_gt_indices(gt_texts: list[str]) -> list[int]:
    return [i for i, t in enumerate(gt_texts) if is_vis_eligible_gt(t)]


def _pred_vs_eligible_gt(
    pred: dict,
    gt_texts: list[str],
    gt_polys: np.ndarray,
) -> list[tuple[bool, bool]]:
    """Per pred: (show, class_correct). show only if argmax GT is eligible and IoU>=IOU_WEAK."""
    n = len(pred["texts"])
    if n == 0:
        return []
    elig = _eligible_gt_indices(gt_texts)
    if not elig:
        return [(False, False)] * n
    elig_polys = gt_polys[elig]
    elig_texts = [gt_texts[i] for i in elig]
    ious = iou_matrix_rotated(pred["polys"], elig_polys)
    out: list[tuple[bool, bool]] = []
    for j in range(n):
        best_e = int(ious[j].argmax())
        best_iou = float(ious[j, best_e])
        if best_iou < IOU_WEAK:
            out.append((False, False))
            continue
        gcls = elig_texts[best_e]
        ok_cls = _norm_cls(pred["texts"][j]) == _norm_cls(gcls)
        out.append((True, ok_cls))
    return out


def render_tile_angle_raw(args_tuple: tuple) -> dict:
    """GT left; right = inference boxes: class palette if correct else red."""
    (
        repo_s,
        cache_s,
        out_s,
        rank,
        tile,
        angle,
        case1_frac,
        peak_angle,
    ) = args_tuple
    repo = Path(repo_s)
    cache_dir = Path(cache_s)
    out_dir = Path(out_s)

    img_path = angle_sweep_image_path(repo, tile, angle)
    if img_path is None:
        return dict(status="ERR", rank=rank, tile_id=tile, angle=angle, error="no_image")

    pred = load_pred(repo, cache_dir, tile, angle)
    base = cv2.imread(str(img_path))
    if base is None:
        return dict(status="ERR", rank=rank, tile_id=tile, angle=angle, error="imread_fail")

    gt_path = (
        repo / "data/DOTA1_1024_500/angle_sweep_val/realistic"
        / f"angle_{angle:03d}" / "annfiles" / f"{tile}.txt"
    )
    gt_texts, gt_polys = parse_dota_txt(gt_path)
    elig_idx = _eligible_gt_indices(gt_texts)

    img_gt = base.copy()
    for i in elig_idx:
        cls = gt_texts[i]
        _draw_poly(img_gt, gt_polys[i], GT_PALETTE.get(cls, (160, 160, 160)), 2, cls[:14])
    elig_counts = Counter(gt_texts[i] for i in elig_idx)
    gt_sub = (
        f"n_gt_elig={len(elig_idx)}/{len(gt_texts)} excl LV/bg | "
        + ", ".join(f"{k}:{v}" for k, v in elig_counts.most_common(4))
        if elig_counts else "no eligible GT"
    )
    _panel_header(img_gt, "GT (left)", gt_sub[:72])

    img_pred = base.copy()
    scores = pred.get("scores")
    pred_flags = _pred_vs_eligible_gt(pred, gt_texts, gt_polys)
    n_show = n_ok = n_wrong = 0
    for j, (cls, poly) in enumerate(zip(pred["texts"], pred["polys"])):
        show, ok_cls = pred_flags[j] if j < len(pred_flags) else (False, False)
        if not show:
            continue
        n_show += 1
        if ok_cls:
            n_ok += 1
        else:
            n_wrong += 1
        sc = float(scores[j]) if scores is not None and j < len(scores) else None
        color = _class_color(cls) if ok_cls else WRONG_BGR
        thick = 2 if not ok_cls else 1
        _draw_poly(img_pred, poly, color, thick, _class_label(cls, sc))
    pred_sub = (
        f"shown={n_show}/{len(pred['texts'])} ok={n_ok} wrong={n_wrong} | "
        "only preds on elig GT (no LV/bg) | palette=ok red=wrong"
    )
    _panel_header(img_pred, "Pred (right)", pred_sub[:72])

    img = _merge_lr(img_gt, img_pred)
    leg = (
        f"rank={rank:03d} | {tile} | ang={angle:03d} | "
        f"tile_case1={float(case1_frac):.3f} peak={int(float(peak_angle)):03d}"
    )
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        img, leg, (12, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )

    fname = (
        f"rank{rank:03d}_{tile}_angle{angle:03d}_"
        f"c1{float(case1_frac):.3f}_sh{n_show}_ok{n_ok}_wr{n_wrong}.jpg"
    )
    out_path = out_dir / fname
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return dict(
        status="OK",
        rank=rank,
        tile_id=tile,
        angle=angle,
        case1_sv_fraction=case1_frac,
        case1_peak_angle=peak_angle,
        n_pred_shown=n_show,
        n_pred_cls_ok=n_ok,
        n_pred_cls_wrong=n_wrong,
        n_gt_eligible=len(elig_idx),
        n_gt=len(gt_texts),
        image_path=str(out_path),
    )


def render_tile_angle(args_tuple: tuple) -> dict:
    (
        repo_s,
        cache_s,
        out_s,
        rank,
        tile,
        angle,
        case1_frac,
        peak_angle,
    ) = args_tuple
    repo = Path(repo_s)
    cache_dir = Path(cache_s)
    out_dir = Path(out_s)

    img_path = angle_sweep_image_path(repo, tile, angle)
    if img_path is None:
        return dict(status="ERR", rank=rank, tile_id=tile, angle=angle, error="no_image")

    pred0 = load_pred(repo, cache_dir, tile, 0) if angle != 0 else None
    events_gt, events_sv, summary = analyze_tile_angle(
        repo, cache_dir, tile, angle, pred0 if angle != 0 else None,
    )
    pred = load_pred(repo, cache_dir, tile, angle)

    sv_tags: dict[int, str] = {}
    for e in events_sv:
        if e["event"] in SV_COLORS:
            sv_tags[int(e["pred_idx"])] = e["event"]

    base = cv2.imread(str(img_path))
    if base is None:
        return dict(status="ERR", rank=rank, tile_id=tile, angle=angle, error="imread_fail")

    gt_path = (
        repo / "data/DOTA1_1024_500/angle_sweep_val/realistic"
        / f"angle_{angle:03d}" / "annfiles" / f"{tile}.txt"
    )
    gt_texts, gt_polys = parse_dota_txt(gt_path)

    n_case1 = int(summary["n_sv_case1"])
    n_bg = int(summary["n_sv_bg"])
    n_sv = int(summary["n_pred_sv"])
    n_pred_all = len(pred["texts"])

    # --- Left: GT only ---
    img_gt = base.copy()
    for poly, cls in zip(gt_polys, gt_texts):
        _draw_poly(img_gt, poly, GT_PALETTE.get(cls, (160, 160, 160)), 2, cls[:14])
    gt_counts = Counter(gt_texts)
    if gt_counts:
        gt_sub = f"n_gt={len(gt_texts)} | " + ", ".join(
            f"{k}:{v}" for k, v in gt_counts.most_common(4)
        )
    else:
        gt_sub = "empty"
    _panel_header(img_gt, "GT (left)", gt_sub[:70])

    # --- Right: predictions only ---
    img_pred = base.copy()
    for j, (cls, poly) in enumerate(zip(pred["texts"], pred["polys"])):
        if is_sv_label(cls):
            tag = sv_tags.get(j, "sv_bg")
            thick = 3 if tag == "sv_case1" else 1
            lbl = "SV" if tag == "sv_bg" else tag.replace("sv_", "")
            _draw_poly(
                img_pred, poly, SV_COLORS.get(tag, (128, 128, 128)), thick, lbl,
            )
        else:
            short = cls[:10] if len(cls) > 10 else cls
            _draw_poly(img_pred, poly, (120, 200, 120), 1, short)
    pred_sub = (
        f"n_pred={n_pred_all} sv={n_sv} | case1={n_case1} bg={n_bg} | "
        "red=case1 gray=bg orange=residual"
    )
    _panel_header(img_pred, "Pred (right)", pred_sub[:72])

    img = _merge_lr(img_gt, img_pred)
    leg = (
        f"rank={rank:03d} | {tile} | ang={angle:03d} | "
        f"tile_case1={float(case1_frac):.3f} peak={int(float(peak_angle)):03d}"
    )
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        img, leg, (12, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )

    fname = (
        f"rank{rank:03d}_{tile}_angle{angle:03d}_"
        f"c1{float(case1_frac):.3f}_sv{n_sv}_c1{n_case1}.jpg"
    )
    out_path = out_dir / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return dict(
        status="OK",
        rank=rank,
        tile_id=tile,
        angle=angle,
        case1_sv_fraction=case1_frac,
        case1_peak_angle=peak_angle,
        n_pred_sv=n_sv,
        n_sv_case1=n_case1,
        n_sv_bg=n_bg,
        n_gt=summary["n_gt"],
        image_path=str(out_path),
    )


def top_tiles_by_case1(tile_csv: Path, n: int) -> list[dict]:
    rows = read_csv(tile_csv)
    rows.sort(key=lambda r: (-float(r["case1_sv_fraction"]), -int(r["total_sv_case1"])))
    out = []
    for i, r in enumerate(rows[:n], 1):
        out.append({**r, "rank": i})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Vis top case1_sv_fraction tiles × 12 angles")
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=REPO / "work_dirs/exp_rotation_gt_shift_taxonomy_20260527",
    )
    ap.add_argument("--repo-root", type=Path, default=REPO)
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--workers",
        type=int,
        default=512,
        help="Parallel tile×angle workers (default 512)",
    )
    ap.add_argument("--angles", default="all", help="all or comma angles e.g. 0,30,90")
    ap.add_argument(
        "--pred-vis",
        choices=("taxonomy", "raw"),
        default="taxonomy",
        help="taxonomy: case1/bg/residual; raw: pred boxes, class color if correct else red",
    )
    args = ap.parse_args()

    tile_csv = args.work_dir / "ftable_case_summary_tile.csv"
    if not tile_csv.exists():
        raise FileNotFoundError(tile_csv)

    if args.out_dir is not None:
        out_dir = args.out_dir
    elif args.pred_vis == "raw":
        out_dir = (
            REPO / "resultmd/exp_rotation_gt_shift_taxonomy_20260527"
            / f"vis_top{args.top_n}_case1_sv_fraction_12angle_raw_pred"
        )
    else:
        out_dir = (
            REPO / "resultmd/exp_rotation_gt_shift_taxonomy_20260527"
            / f"vis_top{args.top_n}_case1_sv_fraction_12angle"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    tiles = top_tiles_by_case1(tile_csv, args.top_n)
    if args.angles.strip().lower() == "all":
        angles = ANGLES_12
    else:
        angles = [int(x.strip()) for x in args.angles.split(",")]

    cache_dir = args.work_dir / "pred_cache"
    repo_s, cache_s, out_s = str(args.repo_root.resolve()), str(cache_dir), str(out_dir)

    jobs = []
    for r in tiles:
        for ang in angles:
            jobs.append((
                repo_s,
                cache_s,
                out_s,
                int(r["rank"]),
                r["tile_id"],
                ang,
                r["case1_sv_fraction"],
                r["case1_peak_angle"],
            ))

    results: list[dict] = []
    errors: list[str] = []
    render_fn = render_tile_angle_raw if args.pred_vis == "raw" else render_tile_angle
    workers = max(1, min(args.workers, len(jobs)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(render_fn, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                row = fut.result()
            except Exception as exc:
                j = futs[fut]
                row = dict(status="ERR", tile_id=j[4], angle=j[5], error=str(exc))
            results.append(row)
            if row.get("status") != "OK":
                errors.append(f"{row.get('tile_id')} @{row.get('angle')}: {row.get('error')}")
            if done % 50 == 0 or done == len(jobs):
                print(f"progress {done}/{len(jobs)}", flush=True)

    ok_rows = [r for r in results if r.get("status") == "OK"]
    ok_rows.sort(key=lambda x: (int(x["rank"]), int(x["angle"])))
    index_path = out_dir / "ftable_vis_index.csv"
    if ok_rows:
        write_csv(index_path, ok_rows)

    tile_index = out_dir / "ftable_top_tiles.csv"
    write_csv(
        tile_index,
        [
            dict(
                rank=r["rank"],
                tile_id=r["tile_id"],
                case1_sv_fraction=r["case1_sv_fraction"],
                case1_peak_angle=r["case1_peak_angle"],
                total_sv_dets=r["total_sv_dets"],
                total_sv_case1=r["total_sv_case1"],
                total_case2_events=r["total_case2_events"],
            )
            for r in tiles
        ],
    )

    print(f"saved {len(ok_rows)}/{len(jobs)} -> {out_dir}")
    print(f"index -> {index_path}")
    if errors:
        print(f"errors ({len(errors)}):")
        for e in errors[:15]:
            print(" ", e)
    return 0 if len(ok_rows) == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
